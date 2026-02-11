"""
Wav2Arkit Service

Service layer for Wav2Arkit model inference.
Handles audio-to-blendshape conversion for facial animation.
Uses ONNX runtime for CPU-optimized inference.
"""

import base64
from pathlib import Path
from typing import Any

import numpy as np

from core.logger import get_logger
from core.settings import Settings, get_settings

logger = get_logger(__name__)


class Wav2ArkitService:
    """
    Service for Wav2Arkit blendshape inference.

    Uses ONNX runtime for CPU-optimized inference to convert
    audio to ARKit-compatible facial blendshapes.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the Wav2Arkit service.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._inference: Any | None = None
        self._available = False

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the ONNX model for CPU inference."""
        try:
            from wav2arkit import Wav2ArkitInference

            model_path = Path(self.settings.onnx_model_path)
            if not model_path.exists():
                logger.warning(f"Wav2Arkit model not found at: {model_path}")
                logger.warning("Wav2Arkit will be unavailable")
                return

            self._inference = Wav2ArkitInference(
                model_path=str(model_path),
                audio_sr=self.settings.wav2arkit_sample_rate,
                fps=self.settings.blendshape_fps,
                debug=self.settings.debug,
            )

            self._available = True
            logger.info("Wav2Arkit model loaded (CPU-optimized)")

            # Warmup pass
            self._warmup_model()

        except ImportError as e:
            logger.warning(f"ONNX Runtime not available: {e}")
            logger.warning("Install onnxruntime: pip install onnxruntime")
        except Exception as e:
            logger.warning(f"Failed to load Wav2Arkit model: {e}")

    def _warmup_model(self) -> None:
        """
        Run a warmup inference pass to initialize ONNX runtime.

        This eliminates the delay on the first real inference call.
        """
        if not self._inference:
            return

        try:
            logger.info("Running model warmup pass...")
            # Create dummy audio: 1 second of silence at model sample rate
            dummy_audio = np.zeros(self.settings.wav2arkit_sample_rate, dtype=np.float32)

            # Run inference (will trigger ONNX runtime initialization)
            result, _ = self.infer_streaming(
                dummy_audio,
                sample_rate=self.settings.wav2arkit_sample_rate,
            )

            if result.get("code") == 0:
                logger.info("Model warmup complete - ready for real-time inference")
            else:
                logger.warning("Model warmup completed with non-zero code")

        except Exception as e:
            logger.warning(f"Model warmup failed (non-critical): {e}")

    @property
    def is_available(self) -> bool:
        """Check if Wav2Arkit inference is available."""
        return self._available and self._inference is not None

    def reset_context(self) -> None:
        """Reset inference context for new speech session."""
        if self._inference:
            self._inference.reset_context()

    def infer_streaming(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
    ) -> tuple[dict, Any | None]:
        """
        Run streaming inference on audio chunk.

        Args:
            audio_data: Audio samples as float32 numpy array
            sample_rate: Sample rate of the audio

        Returns:
            Tuple of (result dict, timing info)
            Result contains 'code' (0=success) and 'expression' array
        """
        if not self.is_available:
            return {"code": -1, "error": "Wav2Arkit model not available"}, None

        if self._inference is None:
            return {"code": -1, "error": "Wav2Arkit inference not initialized"}, None
        
        return self._inference.infer_streaming(audio_data, sample_rate=sample_rate)

    def weights_to_dict(self, frame_weights: np.ndarray) -> dict[str, float]:
        """
        Convert frame weights array to named dictionary.

        Args:
            frame_weights: Array of 52 blendshape weights

        Returns:
            Dictionary mapping blendshape names to weights
        """
        if not self.is_available:
            return {}

        if self._inference is None:
            return {}
        
        return self._inference.weights_to_dict(frame_weights)

    def process_audio_chunk(
        self,
        audio_bytes: bytes,
        sample_rate: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Process audio chunk and return paired audio+blendshape frames.

        This is the main processing method that:
        1. Converts audio bytes to numpy array
        2. Runs Wav2Arkit inference to get blendshapes
        3. Pairs each blendshape frame with its audio slice

        Args:
            audio_bytes: PCM16 audio at OpenAI sample rate (24kHz)
            sample_rate: Optional override for input sample rate

        Returns:
            List of dicts with 'weights' (dict) and 'audio' (base64 string) for each frame
        """
        if not self.is_available:
            return []

        # Convert to numpy array (PCM16 to float32)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Run inference
        result, _ = self.infer_streaming(
            audio_float,
            sample_rate=sample_rate or self.settings.input_sample_rate,
        )

        expression = result.get("expression")
        
        # Handle inference failure or empty result by returning audio with neutral face
        # This ensures audio is never dropped even if model fails
        if result.get("code") != 0 or expression is None or len(expression) == 0:
            if result.get("code") != 0:
                logger.warning(f"Inference failed (code {result.get('code')}). Using fallback frames.")
            else:
                logger.warning("Inference returned no frames. Using fallback frames.")
            return self._create_fallback_frames(audio_bytes)

        # Pair each blendshape frame with its corresponding audio
        frames = []
        bytes_per_frame = self.settings.bytes_per_frame

        for i, frame_weights in enumerate(expression):
            weights_dict = self.weights_to_dict(frame_weights)

            # Extract audio slice for this frame
            audio_start = i * bytes_per_frame
            audio_end = min(audio_start + bytes_per_frame, len(audio_bytes))
            frame_audio = audio_bytes[audio_start:audio_end]

            # Pad with silence if needed
            if len(frame_audio) < bytes_per_frame:
                frame_audio = frame_audio + bytes(bytes_per_frame - len(frame_audio))

            # Pre-encode base64 here to avoid encoding in broadcast loop (30 FPS)
            frames.append(
                {
                    "weights": weights_dict,
                    "audio": base64.b64encode(frame_audio).decode("utf-8"),
                }
            )

        return frames

    def _create_fallback_frames(self, audio_bytes: bytes) -> list[dict[str, Any]]:
        """Create frames with audio but zero/neutral weights."""
        if not self._inference:
            return []
            
        frames = []
        bytes_per_frame = self.settings.bytes_per_frame
        
        # Calculate how many frames fit in this audio chunk
        # Use simple ceiling division logic or just process ensuring all audio is sent
        total_len = len(audio_bytes)
        num_frames = (total_len + bytes_per_frame - 1) // bytes_per_frame
        
        # Get neutral (zero) weights
        # We can cache this or invoke get_blendshape_names
        try:
            names = self._inference.get_blendshape_names()
            zero_weights = dict.fromkeys(names, 0.0)
        except Exception:
            zero_weights = {}

        for i in range(num_frames):
            audio_start = i * bytes_per_frame
            # Ensure we don't go past end
            audio_end = min(audio_start + bytes_per_frame, total_len)
            
            frame_audio = audio_bytes[audio_start:audio_end]
            
            if len(frame_audio) == 0:
                break
                
            # Pad final frame if needed to match protocol expectations
            if len(frame_audio) < bytes_per_frame:
                padding = bytes(bytes_per_frame - len(frame_audio))
                frame_audio = frame_audio + padding
            
            frames.append({
                "weights": zero_weights,
                "audio": base64.b64encode(frame_audio).decode("utf-8")
            })
            
        return frames


# Singleton instance (created on first import if needed)
_wav2arkit_service: Wav2ArkitService | None = None


def get_wav2arkit_service(
    settings: Settings | None = None,
) -> Wav2ArkitService:
    """
    Get or create the Wav2ArkitService singleton.

    Args:
        settings: Application settings (uses defaults if not provided)

    Returns:
        Wav2ArkitService instance
    """
    global _wav2arkit_service

    if _wav2arkit_service is None:
        settings = settings or get_settings()
        _wav2arkit_service = Wav2ArkitService(settings)

    return _wav2arkit_service
