"""
Audio2Expression Service

Service layer for Audio2Expression model inference.
Handles audio-to-blendshape conversion for facial animation.
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import base64
import numpy as np

from config import Settings, AudioConstants
from logger import get_logger

logger = get_logger(__name__)


class Audio2ExpService:
    """
    Service for Audio2Expression blendshape inference.
    
    Wraps the BlendshapeInference class to provide a clean service interface
    for converting audio to ARKit-compatible facial blendshapes.
    """
    
    def __init__(self, settings: Settings, audio_constants: AudioConstants):
        """
        Initialize the Audio2Expression service.
        
        Args:
            settings: Application settings
            audio_constants: Audio processing constants
        """
        self.settings = settings
        self.audio_constants = audio_constants
        self._inference: Optional[Any] = None
        self._available = False
        
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the model if available."""
        try:
            from audio_to_expression import BlendshapeInference
            
            model_path = Path(self.settings.model_path)
            if not model_path.exists():
                logger.warning(f"Model not found at: {model_path}")
                return
            
            device = "cuda" if self.settings.use_gpu else "cpu"
            
            self._inference = BlendshapeInference(
                model_path=str(model_path),
                device=device,
                identity_idx=self.settings.identity_idx,
                audio_sr=self.audio_constants.audio2exp_sample_rate,
                fps=self.audio_constants.blendshape_fps,
                debug=self.settings.debug,
            )
            
            self._available = True
            logger.info(f"Model loaded (identity: {self.settings.identity_idx})")

            # Warmup pass: Run inference once to initialize CUDA/GPU and compile kernels
            # This eliminates the ~1-2 second delay on first conversation
            self._warmup_model()

        except ImportError:
            logger.warning("Audio2Expression not available. Install torch, transformers, librosa.")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
    
    def _warmup_model(self) -> None:
        """
        Run a warmup inference pass to initialize GPU/CUDA and compile kernels.

        This eliminates the 1-2 second delay on the first real inference call.
        """
        if not self._inference:
            return

        try:
            logger.info("Running model warmup pass...")
            # Create dummy audio: 1 second of silence at model sample rate
            dummy_audio = np.zeros(self.audio_constants.audio2exp_sample_rate, dtype=np.float32)

            # Run inference (will trigger CUDA initialization, kernel compilation, etc.)
            result, _ = self.infer_streaming(
                dummy_audio,
                sample_rate=self.audio_constants.audio2exp_sample_rate,
            )

            if result.get("code") == 0:
                logger.info("Model warmup complete - ready for real-time inference")
            else:
                logger.warning("Model warmup completed with non-zero code")

        except Exception as e:
            logger.warning(f"Model warmup failed (non-critical): {e}")

    @property
    def is_available(self) -> bool:
        """Check if Audio2Expression inference is available."""
        return self._available and self._inference is not None

    def reset_context(self) -> None:
        """Reset inference context for new speech session."""
        if self._inference:
            self._inference.reset_context()
    
    def infer_streaming(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
    ) -> Tuple[Dict, Optional[Any]]:
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
            return {"code": -1, "error": "Audio2Expression model not available"}, None
        
        return self._inference.infer_streaming(audio_data, sample_rate=sample_rate)
    
    def weights_to_dict(self, frame_weights: np.ndarray) -> Dict[str, float]:
        """
        Convert frame weights array to named dictionary.
        
        Args:
            frame_weights: Array of 52 blendshape weights
            
        Returns:
            Dictionary mapping blendshape names to weights
        """
        if not self.is_available:
            return {}
        
        return self._inference.weights_to_dict(frame_weights)
    
    def process_audio_chunk(
        self,
        audio_bytes: bytes,
    ) -> List[Dict[str, Any]]:
        """
        Process audio chunk and return paired audio+blendshape frames.
        
        This is the main processing method that:
        1. Converts audio bytes to numpy array
        2. Runs Audio2Expression inference to get blendshapes
        3. Pairs each blendshape frame with its audio slice
        
        Args:
            audio_bytes: PCM16 audio at OpenAI sample rate (24kHz)

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
            sample_rate=self.audio_constants.openai_sample_rate,
        )
        
        if result.get("code") != 0:
            return []
        
        expression = result.get("expression")
        if expression is None or len(expression) == 0:
            return []
        
        # Pair each blendshape frame with its corresponding audio
        frames = []
        bytes_per_frame = self.audio_constants.bytes_per_frame

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
            frames.append({
                "weights": weights_dict,
                "audio": base64.b64encode(frame_audio).decode("utf-8"),
            })

        return frames


# Singleton instance (created on first import if needed)
_audio2exp_service: Optional[Audio2ExpService] = None


def get_audio2exp_service(
    settings: Optional[Settings] = None,
    audio_constants: Optional[AudioConstants] = None,
) -> Audio2ExpService:
    """
    Get or create the Audio2ExpService singleton.
    
    Args:
        settings: Application settings (uses defaults if not provided)
        audio_constants: Audio constants (uses defaults if not provided)
        
    Returns:
        Audio2ExpService instance
    """
    global _audio2exp_service
    
    if _audio2exp_service is None:
        from config import get_settings, get_audio_constants
        
        settings = settings or get_settings()
        audio_constants = audio_constants or get_audio_constants()
        _audio2exp_service = Audio2ExpService(settings, audio_constants)
    
    return _audio2exp_service
