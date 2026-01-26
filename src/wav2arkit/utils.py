"""
Utility functions and constants adapted from LAM_Audio2Expression.
https://github.com/aigc3d/LAM_Audio2Expression

Contains ARKit blendshape names, post-processing functions,
and context management for streaming inference.
"""

import json
import warnings

import numpy as np
from scipy.signal import savgol_filter

# ARKit 52 blendshape names in standard order
ARKitBlendShape = [
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
    "tongueOut",
]

# Left-right symmetric pairs for blendshape symmetrization
ARKitLeftRightPair = [
    ("jawLeft", "jawRight"),
    ("mouthLeft", "mouthRight"),
    ("mouthSmileLeft", "mouthSmileRight"),
    ("mouthFrownLeft", "mouthFrownRight"),
    ("mouthDimpleLeft", "mouthDimpleRight"),
    ("mouthStretchLeft", "mouthStretchRight"),
    ("mouthPressLeft", "mouthPressRight"),
    ("mouthLowerDownLeft", "mouthLowerDownRight"),
    ("mouthUpperUpLeft", "mouthUpperUpRight"),
    ("cheekSquintLeft", "cheekSquintRight"),
    ("noseSneerLeft", "noseSneerRight"),
    ("browDownLeft", "browDownRight"),
    ("browOuterUpLeft", "browOuterUpRight"),
    ("eyeBlinkLeft", "eyeBlinkRight"),
    ("eyeLookDownLeft", "eyeLookDownRight"),
    ("eyeLookInLeft", "eyeLookInRight"),
    ("eyeLookOutLeft", "eyeLookOutRight"),
    ("eyeLookUpLeft", "eyeLookUpRight"),
    ("eyeSquintLeft", "eyeSquintRight"),
    ("eyeWideLeft", "eyeWideRight"),
]

# Mouth-related blendshapes for silence detection
MOUTH_BLENDSHAPES = [
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "noseSneerLeft",
    "noseSneerRight",
    "cheekPuff",
]

# Default context for streaming inference
DEFAULT_CONTEXT = {
    "is_initial_input": True,
    "previous_audio": None,
    "previous_expression": None,
    "previous_volume": None,
    "previous_headpose": None,
}

# Return codes for inference
RETURN_CODE = {
    "SUCCESS": 0,
    "AUDIO_LENGTH_ERROR": 1,
    "CHECKPOINT_PATH_ERROR": 2,
    "MODEL_INFERENCE_ERROR": 3,
}

# Eye blink patterns for randomized blinks
BLINK_PATTERNS = [
    np.array([0.365, 0.950, 0.956, 0.917, 0.367, 0.119, 0.025]),
    np.array([0.235, 0.910, 0.945, 0.778, 0.191, 0.235, 0.089]),
    np.array([0.870, 0.950, 0.949, 0.696, 0.191, 0.073, 0.007]),
    np.array([0.000, 0.557, 0.953, 0.942, 0.426, 0.148, 0.018]),
]


def symmetrize_blendshapes(bs_params: np.ndarray, mode: str = "average", symmetric_pairs: list|None = None) -> np.ndarray:
    """
    Apply symmetrization to ARKit blendshape parameters.

    Args:
        bs_params: numpy array of shape (N, 52), batch of ARKit parameters
        mode: symmetrization mode ["average", "max", "min", "left_dominant", "right_dominant"]
        symmetric_pairs: list of left-right parameter pairs

    Returns:
        Symmetrized parameters with same shape (N, 52)
    """
    if symmetric_pairs is None:
        symmetric_pairs = ARKitLeftRightPair

    name_to_idx = {name: i for i, name in enumerate(ARKitBlendShape)}

    if bs_params.ndim != 2 or bs_params.shape[1] != 52:
        raise ValueError("Input must be of shape (N, 52)")

    symmetric_bs = bs_params.copy()

    valid_pairs = []
    for left, right in symmetric_pairs:
        left_idx = name_to_idx.get(left)
        right_idx = name_to_idx.get(right)
        if None not in (left_idx, right_idx):
            valid_pairs.append((left_idx, right_idx))

    for l_idx, r_idx in valid_pairs:
        left_col = symmetric_bs[:, l_idx]
        right_col = symmetric_bs[:, r_idx]

        if mode == "average":
            new_vals = (left_col + right_col) / 2
        elif mode == "max":
            new_vals = np.maximum(left_col, right_col)
        elif mode == "min":
            new_vals = np.minimum(left_col, right_col)
        elif mode == "left_dominant":
            new_vals = left_col
        elif mode == "right_dominant":
            new_vals = right_col
        else:
            raise ValueError(f"Invalid mode: {mode}")

        symmetric_bs[:, l_idx] = new_vals
        symmetric_bs[:, r_idx] = new_vals

    return symmetric_bs


def apply_savitzky_golay_smoothing(
    input_data: np.ndarray, window_length: int = 5, polyorder: int = 2, axis: int = 0
) -> tuple[np.ndarray, float | None]:
    """
    Apply Savitzky-Golay filter smoothing along specified axis.

    Args:
        input_data: 2D numpy array of shape (n_samples, n_features)
        window_length: Length of the filter window (must be odd and > polyorder)
        polyorder: Order of the polynomial fit
        axis: Axis along which to filter

    Returns:
        tuple: (smoothed_data, processing_time)
    """
    if input_data.ndim != 2:
        raise ValueError(f"Expected 2D input, got {input_data.ndim}D array")

    if window_length % 2 == 0 or window_length < 3:
        raise ValueError("Window length must be odd integer ≥ 3")

    if polyorder >= window_length:
        raise ValueError("Polynomial order must be < window length")

    # Ensure minimum length for filter
    if input_data.shape[0] < window_length:
        return input_data.copy(), None

    original_dtype = input_data.dtype
    working_data = input_data.astype(np.float64)

    try:
        smoothed_data = savgol_filter(
            working_data, window_length=window_length, polyorder=polyorder, axis=axis, mode="mirror"
        )
    except Exception as e:
        raise RuntimeError(f"Filtering failed: {e!s}") from e

    return np.clip(smoothed_data, 0.0, 1.0).astype(original_dtype), None


def find_low_value_regions(signal: np.ndarray, threshold: float, min_region_length: int = 5) -> list:
    """
    Identifies contiguous regions where values fall below threshold.

    Args:
        signal: Input 1D array
        threshold: Value threshold for identifying low regions
        min_region_length: Minimum consecutive samples required

    Returns:
        List of numpy arrays containing indices for qualifying regions
    """
    low_value_indices = np.where(signal < threshold)[0]
    contiguous_regions = []
    current_region_length = 0
    region_start_idx = 0

    for i in range(1, len(low_value_indices)):
        if low_value_indices[i] != low_value_indices[i - 1] + 1:
            if current_region_length >= min_region_length:
                contiguous_regions.append(low_value_indices[region_start_idx:i])
            region_start_idx = i
            current_region_length = 0
        current_region_length += 1

    if current_region_length >= min_region_length:
        contiguous_regions.append(low_value_indices[region_start_idx:])

    return contiguous_regions


def _blend_region_start(array: np.ndarray, region: np.ndarray, processed_boundary: int, blend_frames: int) -> None:
    """Applies linear blend between last active frame and silent region start."""
    blend_length = min(blend_frames, region[0] - processed_boundary)
    if blend_length <= 0:
        return

    pre_frame = array[region[0] - 1]
    for i in range(blend_length):
        weight = (i + 1) / (blend_length + 1)
        array[region[0] + i] = pre_frame * (1 - weight) + array[region[0] + i] * weight


def _blend_region_end(array: np.ndarray, region: np.ndarray, blend_frames: int) -> None:
    """Applies linear blend between silent region end and next active frame."""
    blend_length = min(blend_frames, array.shape[0] - region[-1] - 1)
    if blend_length <= 0:
        return

    post_frame = array[region[-1] + 1]
    for i in range(blend_length):
        weight = (i + 1) / (blend_length + 1)
        array[region[-1] - i] = post_frame * (1 - weight) + array[region[-1] - i] * weight


def smooth_mouth_movements(
    blend_shapes: np.ndarray,
    processed_frames: int,
    volume: np.ndarray | None = None,
    silence_threshold: float = 0.001,
    min_silence_duration: int = 7,
    blend_window: int = 3,
) -> np.ndarray:
    """
    Reduces jaw movement artifacts during silent periods.

    Args:
        blend_shapes: Array of facial blend shape weights [num_frames, num_blendshapes]
        processed_frames: Number of already processed frames
        volume: Audio volume array used to detect silent periods
        silence_threshold: Volume threshold for considering a frame silent
        min_silence_duration: Minimum consecutive silent frames to process
        blend_window: Number of frames to smooth at region boundaries

    Returns:
        Modified blend shape array with reduced mouth movements during silence
    """
    if volume is None:
        return blend_shapes

    silent_regions = find_low_value_regions(volume, threshold=silence_threshold, min_region_length=min_silence_duration)

    mouth_blend_indices = [ARKitBlendShape.index(name) for name in MOUTH_BLENDSHAPES if name in ARKitBlendShape]

    for region_indices in silent_regions:
        for region_indice in region_indices.tolist():
            if region_indice < blend_shapes.shape[0]:
                blend_shapes[region_indice, mouth_blend_indices] *= 0.1

        try:
            _blend_region_start(blend_shapes, region_indices, processed_frames, blend_window)
            _blend_region_end(blend_shapes, region_indices, blend_window)
        except IndexError as e:
            warnings.warn(f"Edge blending skipped at region {region_indices}: {e!s}")

    return blend_shapes


def _blend_animation_segment(
    array: np.ndarray, transition_start: int, blend_window: int, reference_frame: np.ndarray
) -> None:
    """Applies linear interpolation between reference frame and target frames."""
    actual_blend_length = min(blend_window, array.shape[0] - transition_start)

    for frame_offset in range(actual_blend_length):
        current_idx = transition_start + frame_offset
        blend_weight = (frame_offset + 1) / (actual_blend_length + 1)
        array[current_idx] = reference_frame * (1 - blend_weight) + array[current_idx] * blend_weight


def apply_frame_blending(
    blend_shapes: np.ndarray, processed_frames: int, initial_blend_window: int = 3, subsequent_blend_window: int = 5
) -> np.ndarray:
    """
    Smooths transitions between processed and unprocessed animation frames.

    Args:
        blend_shapes: Array of facial blend shape weights [num_frames, num_blendshapes]
        processed_frames: Number of already processed frames (0 means no previous processing)
        initial_blend_window: Max frames to blend at sequence start
        subsequent_blend_window: Max frames to blend between processed and new frames

    Returns:
        Modified blend shape array with smoothed transitions
    """
    if processed_frames > 0:
        _blend_animation_segment(
            blend_shapes,
            transition_start=processed_frames,
            blend_window=subsequent_blend_window,
            reference_frame=blend_shapes[processed_frames - 1],
        )
    else:
        _blend_animation_segment(
            blend_shapes,
            transition_start=0,
            blend_window=initial_blend_window,
            reference_frame=np.zeros_like(blend_shapes[0]),
        )
    return blend_shapes


def apply_random_eye_blinks_context(
    animation_params: np.ndarray, processed_frames: int = 0, intensity_range: tuple = (0.8, 1.0)
) -> np.ndarray:
    """
    Applies random eye blink patterns to facial animation parameters.

    Following the official LAM implementation, this function:
    1. Zeros out all eye blink values in unprocessed frames first
    2. Then applies random blink patterns at intervals

    Args:
        animation_params: Input facial animation parameters [num_frames, num_features]
        processed_frames: Number of already processed frames
        intensity_range: Tuple defining (min, max) scaling for blink intensity

    Returns:
        Modified animation parameters with random eye blinks
    """
    remaining_frames = animation_params.shape[0] - processed_frames

    if remaining_frames <= 7:
        return animation_params

    # CRITICAL: Zero out eye blink values in unprocessed frames first
    # This matches the official LAM apply_random_eye_blinks behavior
    # Eye blink indices are 8 (eyeBlinkLeft) and 9 (eyeBlinkRight)
    animation_params[processed_frames:, 8:10] = 0.0

    min_blink_interval = 40
    max_blink_interval = 100

    # Find last blink in previously processed frames
    previous_blink_indices = np.where(animation_params[:processed_frames, 8] > 0.5)[0]
    last_processed_blink = previous_blink_indices[-1] - 7 if previous_blink_indices.size > 0 else processed_frames

    blink_interval = np.random.randint(min_blink_interval, max_blink_interval)
    first_blink_start = max(0, blink_interval - last_processed_blink)

    if first_blink_start <= (remaining_frames - 7):
        blink_pattern = BLINK_PATTERNS[np.random.randint(0, 4)]
        intensity = np.random.uniform(*intensity_range)

        blink_start = processed_frames + first_blink_start
        blink_end = blink_start + 7

        # Apply pattern to both eyes (indices 8 and 9)
        animation_params[blink_start:blink_end, 8] = blink_pattern * intensity
        animation_params[blink_start:blink_end, 9] = blink_pattern * intensity

        remaining_after_blink = animation_params.shape[0] - blink_end
        if remaining_after_blink > min_blink_interval:
            second_intensity = np.random.uniform(*intensity_range)
            second_interval = np.random.randint(min_blink_interval, max_blink_interval)

            if (remaining_after_blink - 7) > second_interval:
                second_pattern = BLINK_PATTERNS[np.random.randint(0, 4)]
                second_blink_start = blink_end + second_interval
                second_blink_end = second_blink_start + 7

                animation_params[second_blink_start:second_blink_end, 8] = second_pattern * second_intensity
                animation_params[second_blink_start:second_blink_end, 9] = second_pattern * second_intensity

    return animation_params


def export_blendshape_animation(
    blendshape_weights: np.ndarray,
    output_path: str,
    blendshape_names: list[str],
    fps: float,
    rotation_data: np.ndarray | None = None,
) -> None:
    """
    Export blendshape animation data to JSON format compatible with ARKit.

    Args:
        blendshape_weights: 2D numpy array of shape (N, 52)
        output_path: Full path for output JSON file
        blendshape_names: Ordered list of 52 ARKit-standard blendshape names
        fps: Frame rate for timing calculations
        rotation_data: Optional 3D rotation data array of shape (N, 3)
    """
    if blendshape_weights.shape[1] != 52:
        raise ValueError(f"Expected 52 blendshapes, got {blendshape_weights.shape[1]}")
    if len(blendshape_names) != 52:
        raise ValueError(f"Requires 52 blendshape names, got {len(blendshape_names)}")

    animation_data = {
        "names": blendshape_names,
        "metadata": {"fps": fps, "frame_count": len(blendshape_weights), "blendshape_names": blendshape_names},
        "frames": [],
    }

    for frame_idx in range(blendshape_weights.shape[0]):
        frame_data = {
            "weights": blendshape_weights[frame_idx].tolist(),
            "time": frame_idx / fps,
            "rotation": rotation_data[frame_idx].tolist() if rotation_data is not None else [],
        }
        animation_data["frames"].append(frame_data)

    if not output_path.endswith(".json"):
        output_path += ".json"

    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(animation_data, json_file, indent=2, ensure_ascii=False)
