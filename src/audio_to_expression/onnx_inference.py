"""
ONNX-based inference module for Audio2Expression.

Provides ONNXBlendshapeInference class for CPU-optimized inference
using a combined wav2vec2 + audio2exp ONNX model.
"""

import numpy as np
import librosa
from typing import Dict, Optional, Tuple

from logger import get_logger
from .utils import ARKitBlendShape

logger = get_logger(__name__)


class ONNXBlendshapeInference:
    """
    ONNX-based audio to blendshape inference engine.
    
    Uses a single combined ONNX model (wav2vec2 + audio2exp fused) for
    efficient end-to-end CPU inference from raw audio to blendshapes.
    """
    
    def __init__(
            self,
            model_path: str,
            audio_sr: int = 16000,
            fps: float = 30.0,
            debug: bool = False
    ):
        """
        Initialize the ONNX inference engine.
        
        Args:
            model_path: Path to the combined ONNX model file (wav2arkit_cpu.onnx)
            audio_sr: Audio sample rate (default 16kHz for Wav2Vec2)
            fps: Frame rate for blendshape output
            debug: Enable debug logging
        """
        import onnxruntime as ort
        
        self.audio_sr = audio_sr
        self.fps = fps
        self.debug = debug
        self.device = "cpu"  # ONNX CPU execution
        
        logger.info(f"Loading ONNX model from {model_path}...")
        
        # Load combined ONNX model (wav2vec2 + audio2exp fused)
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        if self.debug:
            logger.debug(f"ONNX model input: {self.input_name}, output: {self.output_name}")
        
        # Streaming context (for compatibility with PyTorch interface)
        self.context: Optional[Dict] = None

        # Warmup: trigger ONNX inference and Numba JIT compilation
        self._warmup()

        logger.info("ONNX model loaded successfully")
    
    def reset_context(self):
        """Reset the streaming context."""
        self.context = None

    def _warmup(self):
        """
        Warmup the ONNX model with realistic inference.

        This triggers:
        - ONNX Runtime optimization
        - Librosa resampling (Numba JIT compilation)
        - First-run overheads

        Prevents 1-2 second delay on first real inference.
        """
        try:
            logger.info("Running ONNX warmup (triggering Numba JIT compilation)...")

            # Create realistic dummy audio at 24kHz (OpenAI sample rate)
            # This will trigger librosa resampling to 16kHz
            dummy_audio_24k = np.random.randn(24000).astype(np.float32)  # 1 second at 24kHz

            # Run full inference pipeline (including resampling)
            result, _ = self.infer_streaming(dummy_audio_24k, sample_rate=24000)

            if result.get("code") == 0:
                frames = result.get("expression")
                if frames is not None:
                    logger.info(f"ONNX warmup complete - generated {len(frames)} frames")
                else:
                    logger.warning("ONNX warmup returned no frames")
            else:
                logger.warning("ONNX warmup completed with non-zero code")

        except Exception as e:
            logger.warning(f"ONNX warmup failed (non-critical): {e}")

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
                - 'headpose': Head pose data (None for ONNX)
        """
        try:
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=int(sample_rate), target_sr=16000)
            
            # Ensure audio is float32 and has batch dimension
            audio = audio.astype(np.float32)
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]  # Add batch dimension: (1, seq_len)
            
            # Run combined ONNX inference (audio -> blendshapes in one pass)
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
                logger.debug(f"ONNX inference: {len(audio[0])} samples -> {expression.shape[0]} frames")
            
            return {
                "code": 0,
                "expression": expression,
                "headpose": None
            }, {}
            
        except Exception as e:
            logger.error(f"ONNX inference error: {e}")
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
