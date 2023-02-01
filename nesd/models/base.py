from typing import List, NoReturn
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 3:
        (n_out, n_in, length) = layer.weight.size()
        n = n_in * length
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()
    
    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)
    
    if layer.bias is not None:
        layer.bias.data.fill_(0.)

def init_bn(bn):
    bn.weight.data.fill_(1.)


def init_gru(rnn):
    """Initialize a GRU layer. """
    
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
    
        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
        
    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
    
    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)



def act(x: torch.Tensor, activation: str) -> torch.Tensor:

    if activation == "relu":
        return F.relu_(x)

    elif activation == "leaky_relu":
        return F.leaky_relu_(x, negative_slope=0.01)

    elif activation == "swish":
        return x * torch.sigmoid(x)

    else:
        raise Exception("Incorrect activation!")


def cart2sph_torch(x, y, z):
    r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    azimuth = torch.atan2(y, x)
    azimuth %= math.pi * 2
    zenith = torch.acos(z / r)
    return r, azimuth, zenith

'''
def interpolate(x, ratio): 
    """Interpolate data in time domain. This is used to compensate the 
    resolution reduction in downsampling of a CNN.
    
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    # from IPython import embed; embed(using=False); os._exit(0)
    # x = x.transpose(1, 2)
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    # x = upsampled.transpose(1, 2)
    return x
'''
def interpolate(x, ratio): 
    """Interpolate data in time domain. This is used to compensate the 
    resolution reduction in downsampling of a CNN.
    
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    x = x[:, :, None, :].repeat(1, 1, ratio, 1)
    # (N, time_steps, ratio, features_num)

    x = rearrange(x, 'n t r c -> n (t r) c')
    # (N, time_steps * ratio, features_num)

    return x

class DFTBase(nn.Module):
    def __init__(self):
        r"""Base class for DFT and IDFT matrix.
        """
        super(DFTBase, self).__init__()

    def dft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(-2 * np.pi * 1j / n)
        W = np.power(omega, x * y)  # shape: (n, n)
        return W

    def idft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(2 * np.pi * 1j / n)
        W = np.power(omega, x * y)  # shape: (n, n)
        return W


class DFT(DFTBase):
    def __init__(self, n, norm):
        r"""Calculate discrete Fourier transform (DFT), inverse DFT (IDFT, 
        right DFT (RDFT) RDFT, and inverse RDFT (IRDFT.) 
        Args:
          n: fft window size
          norm: None | 'ortho'
        """
        super(DFT, self).__init__()

        self.W = self.dft_matrix(n)
        self.inv_W = self.idft_matrix(n)

        self.W_real = nn.Parameter(torch.Tensor(np.real(self.W)), requires_grad=False)
        self.W_imag = nn.Parameter(torch.Tensor(np.imag(self.W)), requires_grad=False)
        self.inv_W_real = nn.Parameter(torch.Tensor(np.real(self.inv_W)), requires_grad=False)
        self.inv_W_imag = nn.Parameter(torch.Tensor(np.imag(self.inv_W)), requires_grad=False)

        self.n = n
        self.norm = norm

    def dft(self, x_real, x_imag):
        r"""Calculate DFT of a signal.
        Args:
            x_real: (n,), real part of a signal
            x_imag: (n,), imag part of a signal
        Returns:
            z_real: (n,), real part of output
            z_imag: (n,), imag part of output
        """
        z_real = torch.matmul(x_real, self.W_real) - torch.matmul(x_imag, self.W_imag)
        z_imag = torch.matmul(x_imag, self.W_real) + torch.matmul(x_real, self.W_imag)
        # shape: (n,)

        if self.norm is None:
            pass
        elif self.norm == 'ortho':
            z_real /= math.sqrt(self.n)
            z_imag /= math.sqrt(self.n)

        return z_real, z_imag

    def idft(self, x_real, x_imag):
        r"""Calculate IDFT of a signal.
        Args:
            x_real: (n,), real part of a signal
            x_imag: (n,), imag part of a signal
        Returns:
            z_real: (n,), real part of output
            z_imag: (n,), imag part of output
        """
        z_real = torch.matmul(x_real, self.inv_W_real) - torch.matmul(x_imag, self.inv_W_imag)
        z_imag = torch.matmul(x_imag, self.inv_W_real) + torch.matmul(x_real, self.inv_W_imag)
        # shape: (n,)

        if self.norm is None:
            z_real /= self.n
        elif self.norm == 'ortho':
            z_real /= math.sqrt(n)
            z_imag /= math.sqrt(n)

        return z_real, z_imag

    def rdft(self, x_real):
        r"""Calculate right RDFT of signal.
        Args:
            x_real: (n,), real part of a signal
            x_imag: (n,), imag part of a signal
        Returns:
            z_real: (n // 2 + 1,), real part of output
            z_imag: (n // 2 + 1,), imag part of output
        """
        n_rfft = self.n // 2 + 1
        z_real = torch.matmul(x_real, self.W_real[..., 0 : n_rfft])
        z_imag = torch.matmul(x_real, self.W_imag[..., 0 : n_rfft])
        # shape: (n // 2 + 1,)

        if self.norm is None:
            pass
        elif self.norm == 'ortho':
            z_real /= math.sqrt(self.n)
            z_imag /= math.sqrt(self.n)

        return z_real, z_imag

    def irdft(self, x_real, x_imag):
        r"""Calculate IRDFT of signal.
        
        Args:
            x_real: (n // 2 + 1,), real part of a signal
            x_imag: (n // 2 + 1,), imag part of a signal
        Returns:
            z_real: (n,), real part of output
            z_imag: (n,), imag part of output
        """
        n_rfft = self.n // 2 + 1

        flip_x_real = torch.flip(x_real, dims=(-1,))
        flip_x_imag = torch.flip(x_imag, dims=(-1,))
        # shape: (n // 2 + 1,)

        x_real = torch.cat((x_real, flip_x_real[..., 1 : n_rfft - 1]), dim=-1)
        x_imag = torch.cat((x_imag, -1. * flip_x_imag[..., 1 : n_rfft - 1]), dim=-1)
        # shape: (n,)

        z_real = torch.matmul(x_real, self.inv_W_real) - torch.matmul(x_imag, self.inv_W_imag)
        # shape: (n,)

        if self.norm is None:
            z_real /= self.n
        elif self.norm == 'ortho':
            z_real /= math.sqrt(self.n)

        return z_real


def get_position_encodings(locts, N):

    x_pos = locts[:, :, :, 2]
    y_pos = locts[:, :, :, 3]

    pos_enc = []

    for l in range(5):
        angle = (2 ** l) * np.pi * (x_pos / N)
        pos_enc.append(angle)

        angle = (2 ** l) * np.pi * (y_pos / N)
        pos_enc.append(angle)

    a1 = torch.stack(pos_enc, dim=-1)
    a1 = torch.cat((torch.cos(a1), torch.sin(a1)), dim=-1)

    return a1


class PositionalEncoder:
    def __init__(self, factor):
        self.factor = factor
        self.room_scaler = 10 / math.pi

    def __call__(self, x, y, z):

        angles = []
        
        for i in range(self.factor):
            angles.append((2 ** i) * (x / self.room_scaler))
            angles.append((2 ** i) * (y / self.room_scaler))
            angles.append((2 ** i) * (z / self.room_scaler))

        angles = torch.stack(angles, dim=-1)

        positional_embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)

        return positional_embedding


class LookDirectionEncoder:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, azimuth, elevation):

        angles = []

        for i in range(self.factor):
            angles.append((2 ** i) * azimuth)
            angles.append((2 ** i) * elevation)

        angles = torch.stack(angles, dim=-1)

        positional_embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)

        return positional_embedding


class DepthEncoder:
    def __init__(self, factor):
        self.factor = factor
        self.room_scaler = 10 / math.pi

    def __call__(self, depth):

        angles = []
        
        for i in range(self.factor):
            angles.append((2 ** i) * (depth / self.room_scaler))

        angles = torch.stack(angles, dim=-1)

        depth_embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)

        return depth_embedding


# class PositionalEncoderRoomR:
#     def __init__(self, factor):
#         self.factor = factor
#         self.room_scaler = 10 / math.pi

#     def __call__(self, r):

#         angles = []
        
#         for i in range(self.factor):
#             angles.append((2 ** i) * (r / self.room_scaler))

#         angles = torch.stack(angles, dim=-1)

#         positional_embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)

#         return positional_embedding


class Base:
    def __init__(self):
        r"""Base function for extracting spectrogram, cos, and sin, etc."""
        pass

    def spectrogram(self, input: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
        r"""Calculate spectrogram.

        Args:
            input: (batch_size, segments_num)
            eps: float

        Returns:
            spectrogram: (batch_size, time_steps, freq_bins)
        """
        (real, imag) = self.stft(input)
        # from IPython import embed; embed(using=False); os._exit(0)
        return torch.clamp(real ** 2 + imag ** 2, eps, np.inf) ** 0.5

    def spectrogram_phase(
        self, input: torch.Tensor, eps: float = 0.0
    ) -> List[torch.Tensor]:
        r"""Calculate the magnitude, cos, and sin of the STFT of input.

        Args:
            input: (batch_size, segments_num)
            eps: float

        Returns:
            mag: (batch_size, time_steps, freq_bins)
            cos: (batch_size, time_steps, freq_bins)
            sin: (batch_size, time_steps, freq_bins)
        """
        (real, imag) = self.stft(input)
        mag = torch.clamp(real ** 2 + imag ** 2, eps, np.inf) ** 0.5
        cos = real / mag
        sin = imag / mag
        return mag, cos, sin

    def wav_to_spectrogram_phase(
        self, input: torch.Tensor, eps: float = 1e-10
    ) -> List[torch.Tensor]:
        r"""Convert waveforms to magnitude, cos, and sin of STFT.

        Args:
            input: (batch_size, channels_num, segment_samples)
            eps: float

        Outputs:
            mag: (batch_size, channels_num, time_steps, freq_bins)
            cos: (batch_size, channels_num, time_steps, freq_bins)
            sin: (batch_size, channels_num, time_steps, freq_bins)
        """
        batch_size, channels_num, segment_samples = input.shape

        # Reshape input with shapes of (n, segments_num) to meet the
        # requirements of the stft function.
        x = input.reshape(batch_size * channels_num, segment_samples)

        mag, cos, sin = self.spectrogram_phase(x, eps=eps)
        # mag, cos, sin: (batch_size * channels_num, 1, time_steps, freq_bins)

        _, _, time_steps, freq_bins = mag.shape
        mag = mag.reshape(batch_size, channels_num, time_steps, freq_bins)
        cos = cos.reshape(batch_size, channels_num, time_steps, freq_bins)
        sin = sin.reshape(batch_size, channels_num, time_steps, freq_bins)

        return mag, cos, sin

    def wav_to_spectrogram(
        self, input: torch.Tensor, eps: float = 1e-10
    ) -> List[torch.Tensor]:

        mag, cos, sin = self.wav_to_spectrogram_phase(input, eps)
        return mag
