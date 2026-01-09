"""
Streaming inference module adapted from LAM_Audio2Expression.
https://github.com/aigc3d/LAM_Audio2Expression

Provides BlendshapeInference class for real-time audio to blendshape
conversion with context management for continuous streaming.
"""

import os
import math
import numpy as np
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import librosa

from logger import get_logger
from .models.network import Audio2Expression

logger = get_logger(__name__)
from .utils import (
    ARKitBlendShape,
    DEFAULT_CONTEXT,
    RETURN_CODE,
    smooth_mouth_movements,
    apply_frame_blending,
    apply_savitzky_golay_smoothing,
    symmetrize_blendshapes,
    # NOTE: apply_random_eye_blinks_context removed - blinking now handled by frontend
)


class BlendshapeInference:
    """
    Real-time audio to blendshape inference engine.
    
    Handles model loading, streaming audio processing, and
    post-processing of blendshape coefficients.
    """
    
    def __init__(
            self,
            model_path: str,
            device: str = 'cuda',
            num_identity_classes: int = 12,
            identity_idx: int = 0,
            audio_sr: int = 16000,
            fps: float = 30.0,
            use_transformer: bool = False,
            debug: bool = False
    ):
        """
        Initialize the blendshape inference engine.
        
        Args:
            model_path: Path to the model checkpoint (.tar file or directory)
            device: Device to run inference on ('cuda' or 'cpu')
            num_identity_classes: Number of identity classes in the model
            identity_idx: Identity index to use (0-11 for streaming model)
            audio_sr: Audio sample rate expected by the model
            fps: Frame rate for blendshape output
            use_transformer: Whether to use transformer in identity encoder
            debug: Enable debug logging
        """
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.num_identity_classes = num_identity_classes
        self.identity_idx = identity_idx
        self.audio_sr = audio_sr
        self.fps = fps
        self.debug = debug
        
        # Model configuration matching streaming model
        self.model_config = {
            'pretrained_encoder_type': 'wav2vec',
            'pretrained_encoder_path': 'facebook/wav2vec2-base-960h',
            'num_identity_classes': num_identity_classes,
            'identity_feat_dim': 64,
            'hidden_dim': 512,
            'expression_dim': 52,
            'norm_type': 'ln',
            'decoder_depth': 3,
            'use_transformer': use_transformer,
            'num_attention_heads': 8,
            'num_transformer_layers': 6,
        }
        
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Streaming context
        self.context: Optional[Dict] = None
        self.max_frame_length = 64
        
        if self.debug:
            logger.debug(f"BlendshapeInference initialized on {self.device}")
            logger.debug(f"Identity classes: {num_identity_classes}, index: {identity_idx}")
            logger.debug(f"Audio sample rate: {audio_sr}, FPS: {fps}")
    
    def _load_model(self, model_path: str) -> Audio2Expression:
        """Load model from checkpoint."""
        logger.info(f"Loading blendshape model from {model_path}")
        
        # Build model
        model = Audio2Expression(**self.model_config)
        
        # Load checkpoint - handle different formats
        if os.path.isfile(model_path):
            # Direct file path (e.g., .tar, .pt, .pth)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        elif os.path.isdir(model_path):
            # Directory format - look for data.pkl or checkpoint file
            data_pkl = os.path.join(model_path, 'data.pkl')
            if os.path.exists(data_pkl):
                # PyTorch save format (directory with data.pkl)
                import pickle
                with open(data_pkl, 'rb') as f:
                    checkpoint = pickle.load(f)
            else:
                # Try loading directory as torch checkpoint
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Extract state dict
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove 'module.' and 'backbone.' prefixes if present (from DDP training)
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith('module.'):
                new_key = new_key[7:]
            if new_key.startswith('backbone.'):
                new_key = new_key[9:]
            new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict, strict=True)
        model = model.to(self.device)

        logger.info("Model loaded successfully")
        return model
    
    def reset_context(self):
        """Reset the streaming context."""
        self.context = None
    
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
            Tuple of (result_dict, updated_context)
            result_dict contains:
                - 'code': Return code (0 = success)
                - 'expression': Blendshape weights array (N, 52)
                - 'headpose': Head pose data (None for now)
        """
        if self.context is None:
            self.context = DEFAULT_CONTEXT.copy()
        
        context = self.context
        max_frame_length = self.max_frame_length
        
        frame_length = math.ceil(audio.shape[0] / sample_rate * 30)
        output_context = DEFAULT_CONTEXT.copy()
        
        # Compute volume for silence detection
        frame_len = min(int(1 / 30 * sample_rate), len(audio))
        hop_len = int(1 / 30 * sample_rate)
        if frame_len > 0 and hop_len > 0:
            volume = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=hop_len)[0]
            if volume.shape[0] > frame_length:
                volume = volume[:frame_length]
        else:
            volume = np.zeros(max(1, frame_length))
        
        # Resample audio if needed
        if sample_rate != self.audio_sr:
            in_audio = librosa.resample(
                audio.astype(np.float32),
                orig_sr=sample_rate,
                target_sr=self.audio_sr
            )
        else:
            in_audio = audio.copy()
        
        start_frame = int(max_frame_length - in_audio.shape[0] / self.audio_sr * 30)
        
        # Handle first input or no previous audio
        if context['is_initial_input'] or context['previous_audio'] is None:
            blank_audio_length = self.audio_sr * max_frame_length // 30 - in_audio.shape[0]
            blank_audio = np.zeros(blank_audio_length, dtype=np.float32)
            input_audio = np.concatenate([blank_audio, in_audio])
            output_context['previous_audio'] = input_audio
        else:
            clip_pre_audio_length = self.audio_sr * max_frame_length // 30 - in_audio.shape[0]
            clip_pre_audio = context['previous_audio'][-clip_pre_audio_length:]
            input_audio = np.concatenate([clip_pre_audio, in_audio])
            output_context['previous_audio'] = input_audio
        
        # Run inference
        with torch.no_grad():
            try:
                input_dict = {}
                input_dict['id_idx'] = F.one_hot(
                    torch.tensor(self.identity_idx),
                    self.num_identity_classes
                ).to(self.device, non_blocking=True)[None, ...]
                
                input_dict['input_audio_array'] = torch.FloatTensor(
                    input_audio
                ).to(self.device, non_blocking=True)[None, ...]
                
                output_dict = self.model(input_dict)
                out_exp = output_dict['pred_exp'].squeeze().cpu().numpy()[start_frame:, :]
                
                # Debug: Log raw model output before post-processing
                if self.debug and out_exp.shape[0] > 0:
                    sample_frame = out_exp[0]
                    logger.debug(
                        f"Model output - Shape: {out_exp.shape}, "
                        f"Range: [{out_exp.min():.4f}, {out_exp.max():.4f}], "
                        f"Frame[0] blink_L/R: {sample_frame[8]:.4f}/{sample_frame[9]:.4f}, "
                        f"jaw: {sample_frame[24]:.4f}, mouth_funnel/pucker: {sample_frame[31]:.4f}/{sample_frame[37]:.4f}"
                    )

            except Exception as e:
                if self.debug:
                    logger.error(f"Model inference error: {e}")
                return {
                    "code": RETURN_CODE['MODEL_INFERENCE_ERROR'],
                    "expression": None,
                    "headpose": None
                }, output_context
        
        # Post-processing
        if context['previous_expression'] is None:
            out_exp = self._apply_postprocessing(out_exp, audio_volume=volume)
        else:
            previous_length = context['previous_expression'].shape[0]
            combined_exp = np.concatenate([context['previous_expression'], out_exp], axis=0)
            combined_vol = np.concatenate([context['previous_volume'], volume], axis=0)
            out_exp = self._apply_postprocessing(
                combined_exp,
                audio_volume=combined_vol,
                processed_frames=previous_length
            )[previous_length:, :]
        
        # Debug: Log post-processed values
        if self.debug and out_exp.shape[0] > 0:
            sample_frame = out_exp[0]
            logger.debug(
                f"Post-processed - Range: [{out_exp.min():.4f}, {out_exp.max():.4f}], "
                f"Frame[0] blink_L/R: {sample_frame[8]:.4f}/{sample_frame[9]:.4f}, "
                f"jaw: {sample_frame[24]:.4f}, mouth_pucker: {sample_frame[37]:.4f}"
            )
        
        # Update context
        if context['previous_expression'] is not None:
            output_context['previous_expression'] = np.concatenate(
                [context['previous_expression'], out_exp], axis=0
            )[-max_frame_length:, :]
            output_context['previous_volume'] = np.concatenate(
                [context['previous_volume'], volume], axis=0
            )[-max_frame_length:]
        else:
            output_context['previous_expression'] = out_exp.copy()
            output_context['previous_volume'] = volume.copy()
        
        output_context['is_initial_input'] = False
        self.context = output_context
        
        return {
            "code": RETURN_CODE['SUCCESS'],
            "expression": out_exp,
            "headpose": None
        }, output_context
    
    def _apply_postprocessing(
            self,
            expression_params: np.ndarray,
            processed_frames: int = 0,
            audio_volume: np.ndarray = None
    ) -> np.ndarray:
        """
        Apply full post-processing pipeline to expression parameters.
        
        Matches the official LAM_Audio2Expression apply_expression_postprocessing method.
        The model output is already sigmoidized (0-1 range), so NO baseline subtraction needed.
        
        Args:
            expression_params: Raw output from model [num_frames, 52] - already in [0,1]
            processed_frames: Number of frames already processed
            audio_volume: Volume array for audio-visual sync
        
        Returns:
            Processed expression parameters
        """
        # Pipeline execution order matches official LAM_Audio2Expression
        # Note: Model output is already sigmoid'd (0-1), no baseline subtraction needed
        expression_params = smooth_mouth_movements(
            expression_params, processed_frames, audio_volume
        )
        expression_params = apply_frame_blending(
            expression_params, processed_frames
        )
        
        if expression_params.shape[0] >= 5:
            expression_params, _ = apply_savitzky_golay_smoothing(
                expression_params, window_length=5
            )
        
        expression_params = symmetrize_blendshapes(expression_params)
        
        # NOTE: Eye blinks are now handled entirely by the frontend widget
        # This reduces server computation and allows consistent blinking across all states
        # Frontend applies random blinks based on ChatState (Idle, Listening, Thinking, Responding)
        # Zero out blink values to ensure clean handoff to frontend
        expression_params[:, 8:10] = 0.0  # eyeBlinkLeft (8), eyeBlinkRight (9)
        
        return expression_params
    
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
