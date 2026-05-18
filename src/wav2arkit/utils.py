"""Backward-compat constants exported from the wav2arkit package.

The previous LAM-based implementation also lived here; that body has been
removed since the new Audio2Face-3D port (see ./core.py + ./inference.py)
owns the inference end-to-end. Only the public symbols still imported by
external code are kept.
"""

# Canonical 52-channel ARKit blendshape order — must match the order channels
# are emitted by Wav2ArkitInference.infer_streaming.
ARKitBlendShape = [
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
    "tongueOut",
]
assert len(ARKitBlendShape) == 52

# Retained for backward-compatibility import paths. Streaming context is now
# owned per-instance by Wav2ArkitInference (SolverState.prev_weights_active +
# SkinAnimator IIR memory). The old dict-based DEFAULT_CONTEXT is unused.
DEFAULT_CONTEXT: dict = {}
