"""Audio2Emotion (A2E) wrapper.

Loads NVIDIA's Audio2Emotion ONNX (`nvidia/Audio2Emotion-v2.2` bundle), runs it
on a 16 kHz audio chunk, and maps the 6-class output to the 10-D explicit
emotion vector expected by the Audio2Face-3D regression network.

Model I/O (verified against the v2.2 bundle):
  input  : 'input_values'  shape=(batch_size, seq_len)  float32  (raw 16k audio)
  output : 'output'        shape=(batch_size, 6)        float32  (6 emotion logits)

Order of the 6 output classes (from network_info.json):
  [angry, disgust, fear, happy, neutral, sad]

Mapping into the 10-D explicit slot expected by the A2F regression net
(amazement, anger, cheekiness, disgust, fear, grief, joy, outofbreath, pain, sadness),
per model_config.json emotion_correspondence:
  angry   → slot 1 (anger)
  disgust → slot 3 (disgust)
  fear    → slot 4 (fear)
  happy   → slot 6 (joy)
  sad     → slot 9 (sadness)
  neutral → dropped (no A2F equivalent)
The four slots without an A2E source (amazement, cheekiness, grief,
outofbreath, pain) are left at zero unless the caller injects a
preferred_emotion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from core.logger import get_logger

logger = get_logger(__name__)


A2E_OUTPUT_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

# 10 explicit emotion slot names — the back half of the 26-D vector consumed
# by the A2F regression network. Must match scripts/config_identity.yml
# `emotion_with_timecode_list.*.emotions` keys.
EXPLICIT_EMOTION_ORDER = [
    "amazement", "anger", "cheekiness", "disgust", "fear",
    "grief", "joy", "outofbreath", "pain", "sadness",
]


class Audio2Emotion:
    """A2E inference: 16k audio → 10-D explicit emotion vector.

    The class also implements the temporal smoothing and emotion_strength
    scaling that NIM's a2e_config.yaml describes, so the output is directly
    usable as the explicit portion of the regression net's 26-D emotion input.

    Args:
        model_path: Path to the Audio2Emotion bundle directory (must contain
                    network.onnx, model_config.json, network_info.json).
        intra_op_num_threads: ONNX runtime thread count for CPU inference.

    The post-processing params (emotion_strength, live_blend_coef,
    emotion_correspondence, etc.) are loaded from model_config.json so
    behaviour matches NIM defaults out of the box.
    """

    def __init__(self, model_path: str | Path, intra_op_num_threads: int = 2) -> None:
        self._model_dir = Path(model_path).resolve()
        onnx_path = self._model_dir / "network.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"A2E network.onnx not found at {onnx_path} — pass a "
                f"nvidia/Audio2Emotion-v2.2 bundle directory"
            )

        # Read post-processing config from the bundle so we mirror NIM defaults
        cfg_path = self._model_dir / "model_config.json"
        ppc: dict = {}
        if cfg_path.exists():
            ppc = json.loads(cfg_path.read_text()).get("post_processing_config", {})
        self.emotion_strength: float = float(ppc.get("emotion_strength", 0.6))
        self.emotion_contrast: float = float(ppc.get("emotion_contrast", 1.0))
        self.live_blend_coef: float = float(ppc.get("live_blend_coef", 0.7))
        self.max_emotions: int = int(ppc.get("max_emotions", 6))
        self.emotion_correspondence: dict[str, int] = ppc.get(
            "emotion_correspondence",
            {"angry": 1, "disgust": 3, "fear": 4, "happy": 6, "sad": 9, "neutral": -1},
        )

        # Optional preferred emotion blend (defaults disabled, matches NIM)
        self.enable_preferred_emotion: bool = bool(ppc.get("enable_preferred_emotion", False))
        self.preferred_emotion_strength: float = float(ppc.get("preferred_emotion_strength", 0.5))
        self.preferred_emotion: np.ndarray = np.asarray(
            ppc.get("preferred_emotion", [0.0] * 10), dtype=np.float32
        )

        logger.info(
            f"Loading Audio2Emotion ONNX from {onnx_path} "
            f"(emotion_strength={self.emotion_strength}, blend_coef={self.live_blend_coef})"
        )
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        so.intra_op_num_threads = int(intra_op_num_threads)
        self._sess = ort.InferenceSession(
            str(onnx_path), so, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._sess.get_inputs()[0].name   # 'input_values'
        self._output_name = self._sess.get_outputs()[0].name  # 'output'

        # Smoothing state — last emitted 10-D explicit emotion (or None on init)
        self._smoothed_explicit: np.ndarray | None = None

    # ----- public API ----------------------------------------------------

    def reset(self) -> None:
        """Forget temporal-smoothing state. Call between turns."""
        self._smoothed_explicit = None

    def predict_raw(self, audio_16k: np.ndarray) -> np.ndarray:
        """Run A2E once on `audio_16k` (float32 in [-1, 1]).
        Returns the 6-class softmax probabilities."""
        audio = np.asarray(audio_16k, dtype=np.float32).reshape(1, -1)
        if audio.size == 0:
            return np.zeros(6, dtype=np.float32)
        logits = self._sess.run([self._output_name], {self._input_name: audio})[0][0]
        return _softmax(logits.astype(np.float32))

    def post_process(self, probs6: np.ndarray) -> np.ndarray:
        """Apply contrast + top-k filter to 6-class probs. Returns 6-D processed probs."""
        if self.emotion_contrast != 1.0:
            probs6 = np.power(probs6, self.emotion_contrast)
            s = probs6.sum()
            if s > 0:
                probs6 = probs6 / s
        if self.max_emotions and self.max_emotions < 6:
            top = np.argsort(-probs6)[: self.max_emotions]
            keep = np.zeros_like(probs6)
            keep[top] = probs6[top]
            probs6 = keep
        return probs6

    def explicit_from_probs(self, probs6: np.ndarray) -> np.ndarray:
        """Map 6-class probs to the 10-D explicit emotion vector (no smoothing/strength applied)."""
        explicit10 = np.zeros(10, dtype=np.float32)
        for i, cls in enumerate(A2E_OUTPUT_CLASSES):
            slot = int(self.emotion_correspondence.get(cls, -1))
            if 0 <= slot < 10:
                explicit10[slot] = probs6[i]
        return explicit10

    def infer_explicit(self, audio_16k: np.ndarray) -> np.ndarray:
        """End-to-end: audio chunk → 10-D explicit emotion vector ready to
        concatenate behind the 16-D implicit half of the A2F regression input.

        Pipeline: A2E ONNX → softmax → contrast → top-k → 6→10 mapping →
        emotion_strength → live_blend_coef smoothing → optional preferred-emotion blend.
        """
        probs6 = self.post_process(self.predict_raw(audio_16k))
        explicit10 = self.explicit_from_probs(probs6) * self.emotion_strength

        if self.enable_preferred_emotion:
            explicit10 = (
                (1.0 - self.preferred_emotion_strength) * explicit10
                + self.preferred_emotion_strength * self.preferred_emotion
            )

        if self._smoothed_explicit is None:
            self._smoothed_explicit = explicit10
        else:
            a = self.live_blend_coef
            self._smoothed_explicit = a * explicit10 + (1.0 - a) * self._smoothed_explicit

        return self._smoothed_explicit.astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()
