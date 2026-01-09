"""
Audio2Expression network adapted from LAM_Audio2Expression.
https://github.com/aigc3d/LAM_Audio2Expression
"""

import math
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder.wav2vec import Wav2Vec2Model
from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config


class Audio2Expression(nn.Module):
    def __init__(self,
                 device: torch.device = None,
                 pretrained_encoder_type: str = 'wav2vec',
                 pretrained_encoder_path: str = '',
                 wav2vec2_config_path: str = '',
                 num_identity_classes: int = 0,
                 identity_feat_dim: int = 64,
                 hidden_dim: int = 512,
                 expression_dim: int = 52,
                 norm_type: str = 'ln',
                 decoder_depth: int = 3,
                 use_transformer: bool = False,
                 num_attention_heads: int = 8,
                 num_transformer_layers: int = 6,
                 ):
        super().__init__()

        self.device = device

        # Initialize audio feature encoder
        if pretrained_encoder_type == 'wav2vec':
            if pretrained_encoder_path and os.path.exists(pretrained_encoder_path):
                self.audio_encoder = Wav2Vec2Model.from_pretrained(pretrained_encoder_path)
            elif wav2vec2_config_path and os.path.exists(wav2vec2_config_path):
                config = Wav2Vec2Config.from_pretrained(wav2vec2_config_path)
                self.audio_encoder = Wav2Vec2Model(config)
            else:
                # Load from HuggingFace
                self.audio_encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
            encoder_output_dim = 768
        else:
            raise NotImplementedError(f"Encoder type {pretrained_encoder_type} not supported")

        self.audio_encoder.feature_extractor._freeze_parameters()
        self.feature_projection = nn.Linear(encoder_output_dim, hidden_dim)

        self.identity_encoder = AudioIdentityEncoder(
            hidden_dim,
            num_identity_classes,
            identity_feat_dim,
            use_transformer,
            num_attention_heads,
            num_transformer_layers
        )

        self.decoder = nn.ModuleList([
            nn.Sequential(*[
                ConvNormRelu(hidden_dim, hidden_dim, norm=norm_type)
                for _ in range(decoder_depth)
            ])
        ])

        self.output_proj = nn.Linear(hidden_dim, expression_dim)

    def freeze_encoder_parameters(self, do_freeze=False):
        for name, param in self.audio_encoder.named_parameters():
            if 'feature_extractor' in name:
                param.requires_grad = False
            else:
                param.requires_grad = (not do_freeze)

    def forward(self, input_dict):
        if 'time_steps' not in input_dict:
            audio_length = input_dict['input_audio_array'].shape[1]
            time_steps = math.ceil(audio_length / 16000 * 30)
        else:
            time_steps = input_dict['time_steps']

        # Process audio through encoder
        audio_input = input_dict['input_audio_array'].flatten(start_dim=1)
        hidden_states = self.audio_encoder(audio_input, frame_num=time_steps).last_hidden_state

        # Project features to hidden dimension
        audio_features = self.feature_projection(hidden_states).transpose(1, 2)

        # Process identity-conditioned features
        audio_features = self.identity_encoder(audio_features, identity=input_dict['id_idx'])

        # Refine features through decoder
        audio_features = self.decoder[0](audio_features)

        # Generate output parameters
        audio_features = audio_features.permute(0, 2, 1)
        expression_params = self.output_proj(audio_features)

        return {'pred_exp': torch.sigmoid(expression_params)}


class AudioIdentityEncoder(nn.Module):
    def __init__(self,
                 hidden_dim,
                 num_identity_classes=0,
                 identity_feat_dim=64,
                 use_transformer=False,
                 num_attention_heads=8,
                 num_transformer_layers=6,
                 dropout_ratio=0.1,
                 ):
        super().__init__()

        in_dim = hidden_dim + identity_feat_dim
        self.id_mlp = nn.Conv1d(num_identity_classes, identity_feat_dim, 1, 1)
        self.first_net = SeqTranslator1D(in_dim, hidden_dim,
                                         min_layers_num=3,
                                         residual=True,
                                         norm='ln'
                                         )
        self.grus = nn.GRU(hidden_dim, hidden_dim, 1, batch_first=True)
        self.dropout = nn.Dropout(dropout_ratio)

        self.use_transformer = use_transformer
        if self.use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_attention_heads,
                dim_feedforward=2 * hidden_dim, batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

    def forward(self,
                audio_features: torch.Tensor,
                identity: torch.Tensor = None,
                time_steps: int = None) -> tuple:

        audio_features = self.dropout(audio_features)
        identity = identity.reshape(identity.shape[0], -1, 1).repeat(1, 1, audio_features.shape[2]).to(torch.float32)
        identity = self.id_mlp(identity)
        audio_features = torch.cat([audio_features, identity], dim=1)

        x = self.first_net(audio_features)

        if time_steps is not None:
            x = F.interpolate(x, size=time_steps, align_corners=False, mode='linear')

        if self.use_transformer:
            x = x.permute(0, 2, 1)
            x = self.transformer_encoder(x)
            x = x.permute(0, 2, 1)

        return x


class ConvNormRelu(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 type='1d',
                 leaky=False,
                 downsample=False,
                 kernel_size=None,
                 stride=None,
                 padding=None,
                 p=0,
                 groups=1,
                 residual=False,
                 norm='bn'):
        super(ConvNormRelu, self).__init__()
        self.residual = residual
        self.norm_type = norm

        if kernel_size is None and stride is None:
            if not downsample:
                kernel_size = 3
                stride = 1
            else:
                kernel_size = 4
                stride = 2

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(stride, tuple):
                padding = tuple(int((kernel_size - st) / 2) for st in stride)
            elif isinstance(kernel_size, tuple) and isinstance(stride, int):
                padding = tuple(int((ks - stride) / 2) for ks in kernel_size)
            elif isinstance(kernel_size, tuple) and isinstance(stride, tuple):
                padding = tuple(int((ks - st) / 2) for ks, st in zip(kernel_size, stride))
            else:
                padding = int((kernel_size - stride) / 2)

        if self.residual:
            if downsample:
                if type == '1d':
                    self.residual_layer = nn.Sequential(
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding
                        )
                    )
                elif type == '2d':
                    self.residual_layer = nn.Sequential(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding
                        )
                    )
            else:
                if in_channels == out_channels:
                    self.residual_layer = nn.Identity()
                else:
                    if type == '1d':
                        self.residual_layer = nn.Sequential(
                            nn.Conv1d(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding
                            )
                        )
                    elif type == '2d':
                        self.residual_layer = nn.Sequential(
                            nn.Conv2d(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding
                            )
                        )

        in_channels = in_channels * groups
        out_channels = out_channels * groups
        if type == '1d':
            self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding,
                                  groups=groups)
            self.norm = nn.BatchNorm1d(out_channels)
            self.dropout = nn.Dropout(p=p)
        elif type == '2d':
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding,
                                  groups=groups)
            self.norm = nn.BatchNorm2d(out_channels)
            self.dropout = nn.Dropout2d(p=p)
        if norm == 'gn':
            self.norm = nn.GroupNorm(2, out_channels)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(out_channels)
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = nn.ReLU()

    def forward(self, x, **kwargs):
        if self.norm_type == 'ln':
            out = self.dropout(self.conv(x))
            out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        else:
            out = self.norm(self.dropout(self.conv(x)))
        if self.residual:
            residual = self.residual_layer(x)
            out += residual
        return self.relu(out)


class SeqTranslator1D(nn.Module):
    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=None,
                 stride=None,
                 min_layers_num=None,
                 residual=True,
                 norm='bn'
                 ):
        super(SeqTranslator1D, self).__init__()

        conv_layers = nn.ModuleList([])
        conv_layers.append(ConvNormRelu(
            in_channels=C_in,
            out_channels=C_out,
            type='1d',
            kernel_size=kernel_size,
            stride=stride,
            residual=residual,
            norm=norm
        ))
        self.num_layers = 1
        if min_layers_num is not None and self.num_layers < min_layers_num:
            while self.num_layers < min_layers_num:
                conv_layers.append(ConvNormRelu(
                    in_channels=C_out,
                    out_channels=C_out,
                    type='1d',
                    kernel_size=kernel_size,
                    stride=stride,
                    residual=residual,
                    norm=norm
                ))
                self.num_layers += 1
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        return self.conv_layers(x)
