"""
Real-time blendshape generation from audio.
This module provides streaming audio to ARKit blendshape inference,
"""

from .inference import BlendshapeInference
from .utils import ARKitBlendShape, DEFAULT_CONTEXT

__all__ = [
    'BlendshapeInference',
    'ARKitBlendShape',
    'DEFAULT_CONTEXT',
]
