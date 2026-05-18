"""Wav2Arkit inference: CPU-deployable port of NVIDIA Audio2Face-3D NIM.

In-tree port of the Audio2Face-3D pipeline (BVLS solver + skin animator + ONNX
regression network) plus the A2E (Audio2Emotion) classifier.

Bundle layout (from `nvidia/Audio2Face-3D-v2.3.1-James` on HuggingFace):
    <model_path>/
        network.onnx               # 169-D regression net (audio + 26-D emotion → PCA features)
        bs_skin.npz                # 52 ARKit skin blendshape deltas
        bs_skin_config.json
        bs_tongue.npz              # 16 tongue poses
        bs_tongue_config.json
        model_data.npz             # PCA bases, training-set means, eye/lip deltas
        a2f_ms_config.json         # face_params (skin_strength, eyelid_offset, etc.)
        implicit_emo_db.npz        # 16-D per-actor emotion embeddings (lookup via 6-class avg)

A2E (audio-to-emotion) lives in a SIBLING bundle from `nvidia/Audio2Emotion-v2.2`.
When `a2e_model_path` is provided, A2E produces a 6-D probability vector that fills
BOTH the implicit (slots 0-15 via per-class averages) and explicit (slots 16-25)
halves of the regression net's 26-D emotion input.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import onnxruntime as ort

from core.logger import get_logger

from .a2e import A2E_OUTPUT_CLASSES, Audio2Emotion
from .core import (
    ARKIT_NAMES as _SOLVER_ARKIT_NAMES,
    BUFFER_LEN,
    SkinAnimator,
    SolverState,
    load_assets,
    solve_one_frame,
)
from .utils import ARKitBlendShape

logger = get_logger(__name__)

NETWORK_SR = 16000

# Explicit emotion order in the 26-D network emotion input (last 10 of 26 slots).
EXPLICIT_EMOTION_ORDER = [
    "amazement", "anger", "cheekiness", "disgust", "fear",
    "grief", "joy", "outofbreath", "pain", "sadness",
]


class Wav2ArkitInference:
    """Audio → 52-channel ARKit blendshapes via NVIDIA Audio2Face-3D port.

    Args:
        model_path: Path to actor bundle directory (e.g. pretrained_models/a2f).
                    Must contain network.onnx, bs_skin/tongue.npz, *_config.json,
                    model_data.npz, a2f_ms_config.json.
        audio_sr: Network operating sample rate. Must be 16000.
        fps: Output blendshape frame rate (default 30).
        debug: Enable verbose logging during init.
        intra_op_num_threads: ONNX runtime thread count for the regression net.

    Public API (matches the previous Wav2ArkitInference exactly):
        infer_streaming(audio, sample_rate) → ({"code", "expression"}, timing)
        reset_context()
        get_blendshape_names() → list[str]
        weights_to_dict(frame_weights) → dict[str, float]
    """

    def __init__(
        self,
        model_path: str,
        audio_sr: int = NETWORK_SR,
        fps: float = 30.0,
        debug: bool = False,
        intra_op_num_threads: int = 4,
        a2e_model_path: str | None = None,
    ) -> None:
        if int(audio_sr) != NETWORK_SR:
            raise ValueError(
                f"audio_sr must be {NETWORK_SR} (network operates at 16k); got {audio_sr}"
            )
        self.audio_sr = NETWORK_SR
        self.fps = int(fps)
        self.debug = debug

        self._model_dir = Path(model_path).resolve()
        if not (self._model_dir / "network.onnx").exists():
            raise FileNotFoundError(
                f"{self._model_dir}/network.onnx not found — pass a NVIDIA A2F-3D actor bundle dir"
            )

        logger.info(f"Loading Audio2Face-3D bundle from {self._model_dir}...")

        # Heavy lifting: load NPZ/JSON assets + build BVLS regularization matrices once
        self._A = load_assets(self._model_dir, verbose=debug)
        self._sk_solver = SolverState(
            self._A["skin_basis"], self._A["skin_neutral"], self._A["bs_skin_cfg"],
            self._A["skin_scale_factor"], self._A["frontal_mask_xyz"],
            name="skin", verbose=debug,
        )
        self._ton_solver = SolverState(
            self._A["tongue_basis"], self._A["tongue_neutral"], self._A["bs_tongue_cfg"],
            self._A["tongue_scale_factor"], None,
            name="tongue", verbose=debug,
        )
        self._skin_animator = SkinAnimator(self._A, fps=float(self.fps), verbose=debug)

        # Regression net (audio + 26-D emotion → 169-D PCA features)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        so.intra_op_num_threads = int(intra_op_num_threads)
        self._sess = ort.InferenceSession(
            str(self._model_dir / "network.onnx"), so, providers=["CPUExecutionProvider"],
        )

        # tongueOut goes into output channel 51 (last ARKit slot)
        tongue_names = self._A["tongue_pose_names"]
        self._tongue_out_idx = tongue_names.index("tongueOut") if "tongueOut" in tongue_names else None

        # Remap from solver's internal bs_skin order to canonical ARKitBlendShape order
        self._port_to_canonical = np.array(
            [_SOLVER_ARKIT_NAMES.index(n) for n in ARKitBlendShape], dtype=np.int64,
        )

        # A2E (audio→emotion) — if a bundle path is supplied, run it per chunk to
        # generate the 10-D explicit half of the 26-D emotion vector. Otherwise
        # leave emotion at zeros (neutral).
        self._a2e: Audio2Emotion | None = None
        if a2e_model_path:
            self._a2e = Audio2Emotion(model_path=a2e_model_path)

        # Pre-compute per-A2E-class average implicit vectors from the actor's
        # implicit_emo_db. At inference, the implicit half of the 26-D emotion
        # input is then `sum_i probs6[i] * implicit_avg[A2E_class_i]`.
        self._implicit_avg_per_class: np.ndarray | None = None
        emo_db_path = self._model_dir / "implicit_emo_db.npz"
        if self._a2e is not None and emo_db_path.exists():
            self._implicit_avg_per_class = _build_implicit_class_averages(emo_db_path)
            logger.info(
                f"Implicit emotion DB loaded; per-class averages shape "
                f"{self._implicit_avg_per_class.shape}"
            )

        self._default_emotion = np.zeros(26, dtype=np.float32)

        # Warm up: first inference triggers ONNX graph optimization + librosa Numba JIT
        self._warmup()
        logger.info("Wav2ArkitInference ready")

    # ------------------------------------------------------------------------
    # Public API (matches the previous Wav2Arkit)
    # ------------------------------------------------------------------------

    def reset_context(self) -> None:
        """Reset per-turn solver state. Call at the start of each new speech turn."""
        self._sk_solver.prev_weights_active = np.zeros(self._sk_solver.N_active)
        self._ton_solver.prev_weights_active = np.zeros(self._ton_solver.N_active)
        self._skin_animator = SkinAnimator(self._A, fps=float(self.fps), verbose=False)
        if self._a2e is not None:
            self._a2e.reset()

    def get_blendshape_names(self) -> list[str]:
        """52 ARKit blendshape names in output channel order."""
        return ARKitBlendShape.copy()

    def weights_to_dict(self, weights: np.ndarray) -> dict[str, float]:
        """Convert a single (52,) frame to {name: float}."""
        if weights.shape[0] != 52:
            raise ValueError(f"Expected 52 weights, got {weights.shape[0]}")
        return {name: float(weights[i]) for i, name in enumerate(ARKitBlendShape)}

    def infer_streaming(
        self,
        audio: np.ndarray,
        sample_rate: float,
        emotion: np.ndarray | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Process an audio chunk → 52-channel blendshape frames.

        Args:
            audio: float32 mono audio in [-1, 1] (any sample rate, resampled internally).
            sample_rate: input sample rate (commonly 24000 from the OpenAI/Gemini pipeline).
            emotion: optional (26,) emotion vector (16 implicit + 10 explicit). If None,
                     uses self._default_emotion (zeros). Phase 2 wires A2E here.

        Returns:
            ({"code": 0, "expression": (T, 52) float32}, timing_dict) on success.
            ({"code": -1, "error": str, "expression": None}, {}) on failure.
        """
        try:
            t0 = time.perf_counter()
            audio_np = np.asarray(audio, dtype=np.float32).reshape(-1)
            if int(sample_rate) != NETWORK_SR:
                audio_np = librosa.resample(audio_np, orig_sr=int(sample_rate), target_sr=NETWORK_SR)
            audio_np = audio_np.astype(np.float32)

            # Build the 26-D emotion vector for the regression net.
            # Priority: caller-supplied `emotion` > A2E (if configured) > zeros.
            if emotion is not None:
                emo_vec = np.asarray(emotion, dtype=np.float32).reshape(26)
            elif self._a2e is not None:
                # One A2E pass produces 6-D probs; we use them for BOTH halves:
                #   - implicit (slots 0..15) = sum_i probs[i] * implicit_avg[i]
                #   - explicit (slots 16..25) = mapped + strength + smoothed
                probs6 = self._a2e.post_process(self._a2e.predict_raw(audio_np))
                emo_vec = np.zeros(26, dtype=np.float32)
                if self._implicit_avg_per_class is not None:
                    emo_vec[:16] = (probs6[:, None] * self._implicit_avg_per_class).sum(axis=0)
                # Compute explicit via the smoothed path so live_blend_coef is honored.
                # We re-run the strength + smoothing here so the explicit half mirrors
                # exactly what infer_explicit would have produced from the same audio.
                explicit10 = self._a2e.explicit_from_probs(probs6) * self._a2e.emotion_strength
                if self._a2e.enable_preferred_emotion:
                    explicit10 = (
                        (1.0 - self._a2e.preferred_emotion_strength) * explicit10
                        + self._a2e.preferred_emotion_strength * self._a2e.preferred_emotion
                    )
                if self._a2e._smoothed_explicit is None:
                    self._a2e._smoothed_explicit = explicit10
                else:
                    a = self._a2e.live_blend_coef
                    self._a2e._smoothed_explicit = (
                        a * explicit10 + (1.0 - a) * self._a2e._smoothed_explicit
                    )
                emo_vec[16:] = self._a2e._smoothed_explicit
            else:
                emo_vec = self._default_emotion

            feats_30 = _run_network_hires(audio_np, self._sess, emo_vec, self.fps)
            n_30 = feats_30.shape[0]
            if n_30 == 0:
                return ({"code": 0, "expression": np.zeros((0, 52), dtype=np.float32),
                         "headpose": None}, {})

            weights = np.zeros((n_30, 52), dtype=np.float32)
            for t in range(n_30):
                skin_pca = feats_30[t, :140]
                tongue_pca = feats_30[t, 140:150]

                skin_delta = skin_pca @ self._A["pca_skin_basis"]
                skin_target = self._skin_animator.step(skin_delta)
                w_skin = solve_one_frame(self._sk_solver, skin_target)

                tongue_delta = tongue_pca @ self._A["pca_tongue_basis"]
                tongue_target = self._A["tongue_neutral"] + tongue_delta
                w_tongue = solve_one_frame(self._ton_solver, tongue_target)

                weights[t, :] = w_skin.astype(np.float32)
                if self._tongue_out_idx is not None:
                    weights[t, 51] = w_tongue[self._tongue_out_idx]

            # Remap solver-internal bs_skin order → canonical ARKitBlendShape order
            expression = weights[:, self._port_to_canonical]
            timing = {"total_ms": (time.perf_counter() - t0) * 1000.0, "n_frames": int(n_30)}

            if self.debug:
                logger.debug(
                    f"infer_streaming: {audio_np.shape[0]} samples → {n_30} frames "
                    f"({timing['total_ms']:.0f}ms)"
                )

            return ({"code": 0, "expression": expression, "headpose": None}, timing)

        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            return ({"code": -1, "error": str(e), "expression": None, "headpose": None}, {})

    # ------------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------------

    def _warmup(self) -> None:
        """Trigger ONNX graph init + librosa Numba JIT via a dummy 1-second 24k pass."""
        try:
            dummy = np.zeros(NETWORK_SR, dtype=np.float32)
            result, _ = self.infer_streaming(dummy, sample_rate=NETWORK_SR)
            if result.get("code") == 0:
                frames = result.get("expression")
                logger.info(f"Wav2Arkit warmup ok ({0 if frames is None else len(frames)} frames)")
            else:
                logger.warning(f"Wav2Arkit warmup non-zero code: {result.get('error')}")
        except Exception as e:  # noqa: BLE001 — warmup failure is non-fatal
            logger.warning(f"Wav2Arkit warmup failed (non-critical): {e}")


def _build_implicit_class_averages(emo_db_path: Path) -> np.ndarray:
    """Read `implicit_emo_db.npz` and return a (6, 16) array of per-A2E-class mean vectors.

    The DB stores 163 named specs (~13 per emotion class, 29 for neutral, 17 for
    outofbreath) as a flat (47819, 16) array sliced via emo_spec_start/size. For
    each of the 6 A2E classes we average all matching specs into one 16-D vector
    that the regression net can consume as the implicit half of its 26-D input.

    A2E class → DB substring used to match spec names:
        angry   → 'anger'    happy  → 'joy'      sad     → 'sad'
        disgust → 'disgust'  fear   → 'fear'     neutral → 'neutral'
    """
    blob = np.load(emo_db_path)
    db = blob["emo_db"].astype(np.float32)          # (47819, 16)
    names = [n.decode() for n in blob["emo_spec_names"]]
    starts = blob["emo_spec_start"].astype(np.int64)
    sizes = blob["emo_spec_size"].astype(np.int64)

    # A2E class → which substring identifies matching specs in the DB
    a2e_to_db_substring = {
        "angry": "anger", "disgust": "disgust", "fear": "fear",
        "happy": "joy",  "neutral": "neutral", "sad": "sad",
    }

    avg_per_class = np.zeros((len(A2E_OUTPUT_CLASSES), 16), dtype=np.float32)
    for ci, a2e_cls in enumerate(A2E_OUTPUT_CLASSES):
        substr = a2e_to_db_substring[a2e_cls]
        matched_means: list[np.ndarray] = []
        for spec_i, spec_name in enumerate(names):
            tail = spec_name.split("_", 1)[1] if "_" in spec_name else spec_name
            if substr in tail.lower():
                s, n = int(starts[spec_i]), int(sizes[spec_i])
                matched_means.append(db[s : s + n].mean(axis=0))
        if matched_means:
            avg_per_class[ci] = np.mean(matched_means, axis=0)
    return avg_per_class


def _run_network_hires(
    audio_16k: np.ndarray,
    sess: ort.InferenceSession,
    emotion_vec: np.ndarray,
    fps: int,
) -> np.ndarray:
    """Run regression net at 30fps stride. Returns (n_30fps, 169) float64."""
    duration = len(audio_16k) / NETWORK_SR
    n_30 = int(round(duration * fps))
    if n_30 == 0:
        return np.zeros((0, 169), dtype=np.float64)

    pad = BUFFER_LEN // 2
    audio_pad = np.pad(audio_16k, (pad, BUFFER_LEN), mode="constant").astype(np.float32)
    emotion = emotion_vec.astype(np.float32).reshape(1, 1, 26)

    out = np.zeros((n_30, 169), dtype=np.float64)
    BATCH = 16
    chunks: list[np.ndarray] = []
    metas: list[int] = []
    for t in range(n_30):
        centre = pad + int(round(t * NETWORK_SR / fps))
        start = centre - BUFFER_LEN // 2
        chunks.append(audio_pad[start : start + BUFFER_LEN])
        metas.append(t)
        if len(chunks) == BATCH or t == n_30 - 1:
            batch_inp = np.stack(chunks).reshape(len(chunks), 1, BUFFER_LEN).astype(np.float32)
            batch_emo = np.tile(emotion, (len(chunks), 1, 1))
            o = sess.run(["result"], {"input": batch_inp, "emotion": batch_emo})[0]
            for k, idx in enumerate(metas):
                out[idx] = o[k, 0]
            chunks = []
            metas = []
    return out
