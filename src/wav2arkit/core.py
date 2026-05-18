"""Core: asset loading, BVLS solver, skin animator, single-frame solve.

Extracted from scripts/61_solver_port.py so the wav2arkit_port package is
self-contained. The original script now re-exports these symbols for backward
compatibility with the historical scripts (70_gen_port_hires.py et al.).

Pipeline per frame:
    audio  → network.onnx → 169-D PCA (140 skin + 10 tongue + 15 jaw + 4 eye)
        skin path:   140-D → vertex deltas via shapes_matrix_skin → SkinAnimator
                            → BVLS solve (L2 + L1 + symmetry + temporal reg) → 52-D weights
        tongue path: 10-D → vertex deltas → BVLS solve → 16-D → tongueOut → ARKit slot 51
        eye path:    4-D = [Rx_R, Ry_R, Rx_L, Ry_L] in radians (NOT solved; channels stay 0)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear


BUFFER_LEN = 8320
BUFFER_OFS = 4160
TARGET_SR = 16000

# 52 ARKit pose names in the order they appear in bs_skin.npz (= NIM's poseNames order).
ARKIT_NAMES = [
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft",
    "eyeSquintLeft", "eyeWideLeft",
    "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight",
    "eyeSquintRight", "eyeWideRight",
    "jawForward", "jawLeft", "jawRight", "jawOpen",
    "mouthClose", "mouthFunnel", "mouthPucker",
    "mouthLeft", "mouthRight",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight",
    "tongueOut",
]
assert len(ARKIT_NAMES) == 52


def load_assets(model_dir: Path | str, *, verbose: bool = True) -> dict:
    """Load a NIM actor bundle from `model_dir`.

    `model_dir` must contain: bs_skin.npz, bs_tongue.npz, model_data.npz,
    bs_skin_config.json, bs_tongue_config.json, a2f_ms_config.json.
    """
    model_dir = Path(model_dir)
    bs_skin = np.load(model_dir / "bs_skin.npz")
    bs_tongue = np.load(model_dir / "bs_tongue.npz")
    model_data = np.load(model_dir / "model_data.npz")
    bs_skin_cfg = json.loads((model_dir / "bs_skin_config.json").read_text())["blendshape_params"]
    bs_ton_cfg  = json.loads((model_dir / "bs_tongue_config.json").read_text())["blendshape_params"]
    a2f_cfg     = json.loads((model_dir / "a2f_ms_config.json").read_text())

    # ----- Skin (52 ARKit poses, 61520 verts × 3 = 184560 flat) -----
    skin_basis_3d = np.stack([bs_skin[n].astype(np.float64) for n in ARKIT_NAMES], axis=0)
    skin_basis = skin_basis_3d.reshape(52, -1)
    skin_neutral = bs_skin["neutral"].astype(np.float64).reshape(-1)
    frontal_mask_v = bs_skin["frontalMask"].astype(np.int64)
    frontal_mask_xyz = np.empty(frontal_mask_v.size * 3, dtype=np.int64)
    frontal_mask_xyz[0::3] = frontal_mask_v * 3
    frontal_mask_xyz[1::3] = frontal_mask_v * 3 + 1
    frontal_mask_xyz[2::3] = frontal_mask_v * 3 + 2

    # ----- Tongue (16 named poses + neutral, 5602 verts × 3) -----
    tongue_pose_names = [s.decode() for s in bs_tongue["poseNames"]]
    tongue_pose_names_noneutral = [n for n in tongue_pose_names if n != "neutral"]
    assert len(tongue_pose_names_noneutral) == 16, \
        f"expected 16 tongue poses, got {len(tongue_pose_names_noneutral)}"
    tongue_basis_3d = np.stack(
        [bs_tongue[n].astype(np.float64) for n in tongue_pose_names_noneutral], axis=0,
    )
    tongue_basis = tongue_basis_3d.reshape(16, -1)
    tongue_neutral = bs_tongue["neutral"].astype(np.float64).reshape(-1)

    # ----- PCA bases from model_data.npz -----
    pca_skin_basis = model_data["shapes_matrix_skin"].astype(np.float64).reshape(140, -1)
    pca_tongue_basis = model_data["shapes_matrix_tongue"].astype(np.float64).reshape(10, -1)
    shapes_mean_skin   = model_data["shapes_mean_skin"].astype(np.float64).reshape(-1)
    shapes_mean_tongue = model_data["shapes_mean_tongue"].astype(np.float64).reshape(-1)

    # Bounding-box scale factor for regularization
    skin_neutral_3d = skin_neutral.reshape(-1, 3)
    skin_bb = float(np.linalg.norm(skin_neutral_3d.max(axis=0) - skin_neutral_3d.min(axis=0)))
    skin_template_bb = float(bs_skin_cfg["templateBBSize"])
    skin_scale_factor = (skin_bb / skin_template_bb) ** 2

    tongue_neutral_3d = tongue_neutral.reshape(-1, 3)
    tongue_bb = float(np.linalg.norm(tongue_neutral_3d.max(axis=0) - tongue_neutral_3d.min(axis=0)))
    tongue_template_bb = float(bs_ton_cfg["templateBBSize"])
    tongue_scale_factor = (tongue_bb / tongue_template_bb) ** 2

    if verbose:
        print(f"  skin_bb={skin_bb:.3f}  templateBB={skin_template_bb:.3f}  "
              f"scaleFactor={skin_scale_factor:.5f}")
        print(f"  tongue_bb={tongue_bb:.3f}  templateBB={tongue_template_bb:.3f}  "
              f"scaleFactor={tongue_scale_factor:.5f}")

    # ----- AnimatorSkin pre-solver deltas -----
    eye_close_pose_delta = model_data["eye_close_pose_delta"].astype(np.float64).reshape(-1)
    lip_open_pose_delta  = model_data["lip_open_pose_delta"].astype(np.float64).reshape(-1)

    # ----- Face-mask (sigmoid over normalized Y of shapes_mean_skin) -----
    face_params = a2f_cfg.get("face_params", {})
    face_mask_level    = float(face_params.get("face_mask_level", 0.6))
    face_mask_softness = float(face_params.get("face_mask_softness", 0.0085))
    mean_skin_3d = shapes_mean_skin.reshape(-1, 3)
    y_vals = mean_skin_3d[:, 1]
    y_min, y_max = float(y_vals.min()), float(y_vals.max())
    norm_y = (y_vals - y_min) / max(1e-9, y_max - y_min)
    face_mask_lower = 1.0 / (1.0 + np.exp(-(face_mask_level - norm_y) / face_mask_softness))
    face_mask_lower_xyz = np.repeat(face_mask_lower, 3)

    return {
        "skin_basis": skin_basis,
        "skin_neutral": skin_neutral,
        "shapes_mean_skin": shapes_mean_skin,
        "shapes_mean_tongue": shapes_mean_tongue,
        "frontal_mask_xyz": frontal_mask_xyz,
        "tongue_basis": tongue_basis,
        "tongue_neutral": tongue_neutral,
        "tongue_pose_names": tongue_pose_names_noneutral,
        "pca_skin_basis": pca_skin_basis,
        "pca_tongue_basis": pca_tongue_basis,
        "skin_scale_factor": skin_scale_factor,
        "tongue_scale_factor": tongue_scale_factor,
        "bs_skin_cfg": bs_skin_cfg,
        "bs_tongue_cfg": bs_ton_cfg,
        "a2f_cfg": a2f_cfg,
        "eye_close_pose_delta": eye_close_pose_delta,
        "lip_open_pose_delta":  lip_open_pose_delta,
        "face_mask_lower_xyz":  face_mask_lower_xyz,
    }


class SolverState:
    """One per track (skin / tongue). Holds AMat and per-frame state."""

    def __init__(self, basis_flat: np.ndarray, neutral_flat: np.ndarray,
                 cfg: dict, scale_factor: float,
                 frontal_mask_xyz: np.ndarray | None = None,
                 name: str = "", *, verbose: bool = True):
        self.name = name
        self.N = int(cfg["numPoses"])
        active_poses = np.asarray(cfg["bsSolveActivePoses"], dtype=np.int32)
        cancel_poses = np.asarray(cfg["bsSolveCancelPoses"], dtype=np.int32)
        sym_poses    = np.asarray(cfg["bsSolveSymmetryPoses"], dtype=np.int32)
        self.multipliers = np.asarray(cfg["bsWeightMultipliers"], dtype=np.float64)
        self.offsets     = np.asarray(cfg["bsWeightOffsets"], dtype=np.float64)

        self.active_idx = np.where(active_poses == 1)[0]
        self.N_active = self.active_idx.size

        full_to_active = -np.ones(self.N, dtype=np.int64)
        full_to_active[self.active_idx] = np.arange(self.N_active)

        if frontal_mask_xyz is not None:
            self.A_full = basis_flat[self.active_idx][:, frontal_mask_xyz].T
            self.neutral_masked = neutral_flat[frontal_mask_xyz]
            self.mask = frontal_mask_xyz
        else:
            self.A_full = basis_flat[self.active_idx].T
            self.neutral_masked = neutral_flat
            self.mask = None

        L1Reg = float(cfg.get("strengthL1regularization", 0.0))
        L2Reg = float(cfg.get("strengthL2regularization", 0.0))
        TReg  = float(cfg.get("strengthTemporalSmoothing", 0.0))
        SReg  = float(cfg.get("strengthSymmetry", 0.0))
        L1s, L2s, Ts, Ss = 0.25 * scale_factor, 10.0 * scale_factor, 100.0 * scale_factor, 10.0 * scale_factor
        self.L1Reg, self.L2Reg, self.TReg, self.SReg = L1Reg, L2Reg, TReg, SReg
        self.L1s, self.L2s, self.Ts, self.Ss = L1s, L2s, Ts, Ss
        self.scale_factor = scale_factor

        # Symmetry groups
        groups: dict[int, list[int]] = {}
        for i in range(self.N):
            g = int(sym_poses[i])
            if g < 0:
                continue
            groups.setdefault(g, []).append(i)
        sym_pairs: list[tuple[int, int]] = []
        for members in groups.values():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    ai, bi = full_to_active[members[i]], full_to_active[members[j]]
                    if ai >= 0 and bi >= 0:
                        sym_pairs.append((int(ai), int(bi)))
        self.sym_pairs = sym_pairs
        if sym_pairs:
            S = np.zeros((len(sym_pairs), self.N_active), dtype=np.float64)
            for r, (ai, bi) in enumerate(sym_pairs):
                S[r, ai] = +1.0
                S[r, bi] = -1.0
            self.S = S
            StS = S.T @ S
        else:
            self.S = None
            StS = np.zeros((self.N_active, self.N_active), dtype=np.float64)

        AtA = self.A_full.T @ self.A_full
        self.AMat = (
            AtA
            + (L1Reg ** 2) * L1s * np.ones((self.N_active, self.N_active))
            + L2Reg * L2s * np.eye(self.N_active)
            + TReg  * Ts  * np.eye(self.N_active)
            + SReg  * Ss  * StS
        )

        cancel_pairs: list[tuple[int, int]] = []
        for i in range(self.N):
            j_or_neg = int(cancel_poses[i])
            if j_or_neg >= 0 and i < j_or_neg:
                ai, bi = full_to_active[i], full_to_active[j_or_neg]
                if ai >= 0 and bi >= 0:
                    cancel_pairs.append((int(ai), int(bi)))
        self.cancel_pairs = cancel_pairs

        self.prev_weights_active = np.zeros(self.N_active, dtype=np.float64)

        if verbose:
            print(f"  [{name}] N={self.N} N_active={self.N_active}  "
                  f"sym_pairs={len(sym_pairs)}  cancel_pairs={len(cancel_pairs)}  "
                  f"AMat={self.AMat.shape}  L1={L1Reg} L2={L2Reg} T={TReg} S={SReg}")


def solve_one_frame(state: SolverState, target_verts_flat: np.ndarray,
                    solver: str = "bvls") -> np.ndarray:
    """Run BVLS (or ADMM) for a single frame. Returns (N,) full-pose weights."""
    target_masked = target_verts_flat[state.mask] if state.mask is not None else target_verts_flat
    delta_target = target_masked - state.neutral_masked
    Atb = state.A_full.T @ delta_target
    b = Atb + state.TReg * state.scale_factor * state.prev_weights_active

    if solver == "bvls":
        res = lsq_linear(state.AMat, b, bounds=(0.0, 1.0), method="bvls", tol=1e-10)
        x = res.x.astype(np.float64)
        if state.cancel_pairs:
            l = np.zeros(state.N_active)
            u = np.ones(state.N_active)
            any_cancel = False
            for (i, j) in state.cancel_pairs:
                if x[i] >= x[j]:
                    u[j] = 1e-10; any_cancel = True
                else:
                    u[i] = 1e-10; any_cancel = True
            if any_cancel:
                res2 = lsq_linear(state.AMat, b, bounds=(l, u), method="bvls", tol=1e-10)
                x = res2.x.astype(np.float64)
    elif solver == "admm":
        if not hasattr(state, "_admm_cache"):
            from admm_solver import admm_init_cache
            state._admm_cache = admm_init_cache(state.AMat)
        from admm_solver import admm_solve
        l = np.zeros(state.N_active, dtype=np.float32)
        u = np.ones(state.N_active, dtype=np.float32)
        x = admm_solve(state._admm_cache, b, l, u, num_outer=2)
        if state.cancel_pairs:
            any_cancel = False
            for (i, j) in state.cancel_pairs:
                if x[i] >= x[j]:
                    u[j] = 1e-10; any_cancel = True
                else:
                    u[i] = 1e-10; any_cancel = True
            if any_cancel:
                x = admm_solve(state._admm_cache, b, l, u, num_outer=2)
    else:
        raise ValueError(f"Unknown solver: {solver!r}")

    state.prev_weights_active = x.copy()
    w_full = np.zeros(state.N, dtype=np.float64)
    w_full[state.active_idx] = x
    return w_full * state.multipliers + state.offsets


class SkinAnimator:
    """C++ AnimatorSkin pre-solver path:
        step1:   pose = skinStr*delta + eyeClose*(-eyelidOpen + blinkOff*blinkStr)
                       + lipOpen*lipOpenOffset
        smooth:  IIR degree-2 cascade on `pose`, separate lower + upper coefs
        step2:   target = neutral + smoothedUpper*upperStr*(1-mask)
                                  + smoothedLower*lowerStr*mask
    """

    def __init__(self, A: dict, fps: float = 30.0, *, verbose: bool = True):
        face = A["a2f_cfg"].get("face_params", {})
        self.skinStr      = float(face.get("skin_strength",      1.0))
        self.blinkStr     = float(face.get("blink_strength",     1.0))
        self.eyelidOff    = float(face.get("eyelid_offset",      0.0))
        self.blinkOff     = float(face.get("blink_offset",       0.0))
        self.lipOpenOff   = float(face.get("lip_close_offset",   0.0))
        self.lowerStr     = float(face.get("lower_face_strength", 1.0))
        self.upperStr     = float(face.get("upper_face_strength", 1.0))
        self.lowerSmooth  = float(face.get("lower_face_smoothing", 0.0))
        self.upperSmooth  = float(face.get("upper_face_smoothing", 0.0))

        self.neutral   = A["shapes_mean_skin"]
        self.eyeClose  = A["eye_close_pose_delta"]
        self.lipOpen   = A["lip_open_pose_delta"]
        self.mask      = A["face_mask_lower_xyz"]

        self.dt = 1.0 / fps
        self._init_lower = False
        self._init_upper = False
        n = self.neutral.size
        self._low_s1 = np.zeros(n, dtype=np.float64)
        self._low_s2 = np.zeros(n, dtype=np.float64)
        self._up_s1  = np.zeros(n, dtype=np.float64)
        self._up_s2  = np.zeros(n, dtype=np.float64)

        def alpha(s):
            return 1.0 if s <= 0.0 else float(1.0 - 0.5 ** (self.dt / s))
        self.aLow = alpha(self.lowerSmooth)
        self.aUp  = alpha(self.upperSmooth)

        if verbose:
            print(f"  [SkinAnimator] skinStr={self.skinStr} eyelidOff={self.eyelidOff} "
                  f"lipOff={self.lipOpenOff} lowerStr={self.lowerStr} upperStr={self.upperStr} "
                  f"aLow={self.aLow:.4f} aUp={self.aUp:.4f}")

    def step(self, delta_flat: np.ndarray) -> np.ndarray:
        pose = (self.skinStr * delta_flat
                + self.eyeClose * (-self.eyelidOff + self.blinkOff * self.blinkStr)
                + self.lipOpen * self.lipOpenOff)

        if not self._init_lower:
            self._low_s1[:] = pose
            self._low_s2[:] = pose
            self._init_lower = True
        else:
            self._low_s1 += (pose - self._low_s1) * self.aLow
            self._low_s2 += (self._low_s1 - self._low_s2) * self.aLow
        s_low = self._low_s2

        if not self._init_upper:
            self._up_s1[:] = pose
            self._up_s2[:] = pose
            self._init_upper = True
        else:
            self._up_s1 += (pose - self._up_s1) * self.aUp
            self._up_s2 += (self._up_s1 - self._up_s2) * self.aUp
        s_up = self._up_s2

        target = (self.neutral
                  + s_up  * self.upperStr * (1.0 - self.mask)
                  + s_low * self.lowerStr * self.mask)
        return target
