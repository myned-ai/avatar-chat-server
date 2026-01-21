"""
Wav2Arkit inference module.

Provides Wav2ArkitInference class for CPU-optimized inference
using the combined wav2arkit ONNX model.
"""

import numpy as np
import librosa
from typing import Dict, Optional, Tuple

from logger import get_logger
from .utils import ARKitBlendShape

logger = get_logger(__name__)


class Wav2ArkitInference:
    """
    Wav2Arkit audio to blendshape inference engine.

    Uses the combined ONNX model for efficient end-to-end
    CPU inference from raw audio to ARKit blendshapes.
    """

    def __init__(
            self,
            model_path: str,
            audio_sr: int = 16000,
            fps: float = 30.0,
            debug: bool = False
    ):
        """
        Initialize the Wav2Arkit inference engine.

        Args:
            model_path: Path to the ONNX model file (wav2arkit_cpu.onnx)
            audio_sr: Audio sample rate (default 16kHz)
            fps: Frame rate for blendshape output
            debug: Enable debug logging
        """
        import onnxruntime as ort

        self.audio_sr = audio_sr
        self.fps = fps
        self.debug = debug
        self.device = "cpu"

        logger.info(f"Loading Wav2Arkit model from {model_path}...")

        # Load ONNX model
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        if self.debug:
            logger.debug(f"Model input: {self.input_name}, output: {self.output_name}")

        # Streaming context
        self.context: Optional[Dict] = None

        # Warmup: trigger ONNX inference and Numba JIT compilation
        self._warmup()

        logger.info("Wav2Arkit model loaded successfully")

    def reset_context(self):
        """Reset the streaming context."""
        self.context = None

    def _warmup(self):
        """
        Warmup the model with realistic inference.

        This triggers:
        - ONNX Runtime optimization
        - Librosa resampling (Numba JIT compilation)
        - First-run overheads

        Prevents delay on first real inference.
        """
        try:
            logger.info("Running model warmup (triggering Numba JIT compilation)...")

            # Create realistic dummy audio at 24kHz (OpenAI sample rate)
            # This will trigger librosa resampling to 16kHz
            dummy_audio_24k = np.random.randn(24000).astype(np.float32)  # 1 second at 24kHz

            # Run full inference pipeline (including resampling)
            result, _ = self.infer_streaming(dummy_audio_24k, sample_rate=24000)

            if result.get("code") == 0:
                frames = result.get("expression")
                if frames is not None:
                    logger.info(f"Model warmup complete - generated {len(frames)} frames")
                else:
                    logger.warning("Model warmup returned no frames")
            else:
                logger.warning("Model warmup completed with non-zero code")

        except Exception as e:
            logger.warning(f"Model warmup failed (non-critical): {e}")

    def infer_streaming(
            self,
            audio: np.ndarray,
            sample_rate: float
    ) -> Tuple[Dict, Dict]:
        """
        Process a chunk of streaming audio and generate blendshapes.

        Args:
            audio: Audio samples as numpy array (mono, float32)
            sample_rate: Sample rate of the input audio

        Returns:
            Tuple of (result_dict, context)
            result_dict contains:
                - 'code': Return code (0 = success)
                - 'expression': Blendshape weights array (N, 52)
                - 'headpose': Head pose data (None)
        """
        try:
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=int(sample_rate), target_sr=16000)

            # Ensure audio is float32 and has batch dimension
            audio = audio.astype(np.float32)
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]  # Add batch dimension: (1, seq_len)

            # Run inference (audio -> blendshapes in one pass)
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: audio}
            )
            blendshapes = outputs[0]  # Shape: (1, seq_len, 52)

            # Remove batch dimension
            expression = blendshapes[0]  # Shape: (seq_len, 52)

            # Clip number of frames to match expected output (30 fps * audio duration)
            expected_frames = int(len(audio[0]) / 16000 * self.fps)
            if expression.shape[0] > expected_frames:
                expression = expression[:expected_frames]

            if self.debug:
                logger.debug(f"Inference: {len(audio[0])} samples -> {expression.shape[0]} frames")

            return {
                "code": 0,
                "expression": expression,
                "headpose": None
            }, {}

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                "code": -1,
                "error": str(e),
                "expression": None,
                "headpose": None
            }, {}

    def get_blendshape_names(self) -> list:
        """Return the list of ARKit blendshape names."""
        return ARKitBlendShape.copy()

    def weights_to_dict(self, weights: np.ndarray) -> Dict[str, float]:
        """
        Convert a single frame of weights to a dictionary.

        Args:
            weights: Array of 52 blendshape weights

        Returns:
            Dictionary mapping blendshape names to weights
        """
        if weights.shape[0] != 52:
            raise ValueError(f"Expected 52 weights, got {weights.shape[0]}")

        return {name: float(weights[i]) for i, name in enumerate(ARKitBlendShape)}
