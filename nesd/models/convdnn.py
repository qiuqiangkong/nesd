from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchaudio
from einops import rearrange
import numpy as np
import math
from dataclasses import dataclass

from nesd.models.fourier import Fourier
from nesd.models.sinusoidal_pe import SinusoidalPE
from nesd.utils.torch import cart2sph


class ConvDNN(Fourier):
    def __init__(
        self, 
        audio_channels=4,
        n_fft=2048, 
        hop_length=480, 
        **kwargs
    ):
        super().__init__(
            n_fft=n_fft, 
            hop_length=hop_length, 
            return_complex=True, 
            normalized=True
        )

        self.ds_factor = 8  # Downsample factor

        self.angle_pe = SinusoidalPE(dim=256, scale=100.)

        # Encoder layers
        self.pre_layer = nn.Conv2d(10, 32, kernel_size=3, padding=1)  # 4 mag + 6 angles
        self.conv1a = ConvBlock(32, 32)
        self.conv1b = ConvBlock(32, 32)
        self.down1 = Downsample()

        self.conv2a = ConvBlock(32, 64)
        self.conv2b = ConvBlock(64, 64)
        self.down2 = Downsample()

        self.conv3a = ConvBlock(64, 128)
        self.conv3b = ConvBlock(128, 128)
        self.down3 = Downsample()

        self.mic_wav_fc = nn.Linear(128*128, 2048)
        self.lis_dir_fc = nn.Linear(512, 512)

        self.mlp = nn.Sequential(
            nn.Linear(2560, 2048), 
            nn.GELU(approximate='tanh'),
            nn.Linear(2048, 2048), 
            nn.GELU(approximate='tanh'),
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )


    def forward(self, audio: Tensor, lis_dir: Tensor) -> Tensor:
        r"""The model predicts whether lis_dir contains sources.

        b: batch_size
        c: audio_channels
        l: audio_samples
        t: frames_num
        f: freq_bins

        Args:
            audio: (b, c, l)

        Outputs:
            output: (b, c, l)
        """

        # Complex spectrum
        complex_sp = self.stft(audio)  # shape: (b, c, t, f)

        mag = complex_sp.abs()
        angle = complex_sp.angle()
        angle_diff = self.compute_diff_angle(angle)
        x = torch.cat([mag, angle_diff], dim=1)

        # pad stft
        T = x.shape[2]
        x = self.pad_tensor(x)  # x: (b, d, t, f)

        # Encode
        x0 = self.pre_layer(x)
        x1a = self.conv1a(x0)
        x1b = self.conv1b(x1a)
        x1c = self.down1(x1b)

        x2a = self.conv2a(x1c)
        x2b = self.conv2b(x2a)
        x2c = self.down2(x2b)

        x3a = self.conv3a(x2c)
        x3b = self.conv3b(x3a)
        x3c = self.down3(x3b)

        emb_mic = rearrange(x3c, 'b d t f -> b t (d f)')
        emb_mic = self.mic_wav_fc(emb_mic)  # (b, t', d)

        # Listener direction emb
        lis_dir = lis_dir[:, :, :, 0 :: self.ds_factor, :]  # (b, l, r, t', d)
        emb_lis_dir = self.get_direction_emb(lis_dir)  # (b, l, r, t', d)

        # Conbine mic emb and listener direction emb
        R = emb_lis_dir.shape[2]
        emb_mic = emb_mic[:, None, None, :, :].repeat(1, 1, R, 1, 1)  # (b, l, r, t', d)
        emb = torch.cat((emb_mic, emb_lis_dir), dim=-1)  # (b, l, r, t', d)
        
        out = self.mlp(emb)  # (b, l, r, t', d)
        out = F.interpolate(out, scale_factor=(1, self.ds_factor, 1), mode='nearest')  # (b, l, r, t, d)
        out = out[:, :, :, 0 : T, :]  # (b, l, r, t, d)

        return out

    def pad_tensor(self, x: Tensor) -> tuple[Tensor, int]:
        r"""Pad a spectrum that can be evenly divided by downsample_ratio.

        Args:
            x: E.g., (b, c, t=201, f=1025)
        
        Returns:
            out: E.g., (b, c, t=208, f=1024)
        """

        # Pad last frames, e.g., 201 -> 208
        T = x.shape[2]
        pad_t = math.ceil(T / self.ds_factor) * self.ds_factor - T
        x = F.pad(x, pad=(0, 0, 0, pad_t))

        # Remove last frequency bin, e.g., 1025 -> 1024
        out = x[:, :, :, 0 : -1]

        return out

    def compute_diff_angle(self, mic_phase: Tensor) -> Tensor:
        r"""Compute angle differences.

        Args:
            mic_phase: (b, c, t, f)

        Returns:
            diff_phases: (b, c*(c-1)/2, t, f)
        """

        mics_num = mic_phase.shape[1]

        diff_phases = []

        for i in range(1, mics_num):
            for j in range(0, i):
                diff_phase = (mic_phase[:, i, :, :] - mic_phase[:, j, :, :]) % (2 * math.pi)
                diff_phases.append(diff_phase)

        diff_phases = torch.stack(diff_phases, dim=1)

        return diff_phases

    def get_direction_emb(self, lis_dir: Tensor) -> Tensor:
        r"""Encode directions to embedding.

        Args:
            lis_dir: (b, l, r, t, 3)

        Returns:
            out: (b, l, r, t, d)
        """

        sph = cart2sph(lis_dir)  # (b, l, r, t, 3)
        theta, phi = sph[..., 1], sph[..., 2]
        theta_pe = self.angle_pe(theta)  # (b, l, r, t, d)
        phi_pe = self.angle_pe(phi)  # (b, l, r, t, d)
        out = torch.cat((theta_pe, phi_pe), dim=-1)  # (b, l, r, t, d)
        out = self.lis_dir_fc(out)  # (b, l, r, t, d)
        return out



class Downsample(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x: Tensor) -> Tensor:
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class ConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: tuple[int, int] = (3, 3)
    ):
        r"""Residual block."""
        super().__init__()

        padding = [kernel_size[0] // 2, kernel_size[1] // 2]

        self.norm1 = nn.GroupNorm(min(in_channels, 32), in_channels)
        self.norm2 = nn.GroupNorm(min(out_channels, 32), out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False)

        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.proj = None

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x: (b, d, t, f)

        Returns:
            output: (b, d, t, f)
        """     

        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))

        if self.proj:
            x = self.proj(x)
    
        return x + h
