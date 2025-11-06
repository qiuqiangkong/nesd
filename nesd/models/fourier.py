import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange


class Fourier(nn.Module):
    
    def __init__(self, 
        n_fft=2048, 
        hop_length=441, 
        return_complex=True, 
        normalized=True
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.return_complex = return_complex
        self.normalized = normalized

    def stft(self, waveform: Tensor) -> Tensor:
        r"""Calculate STFT of audio signals.

        b: batch_size
        c: channels_num
        l: audio_samples
        t: frames_num
        f: freq_bins

        Args:
            waveform: (b, c, l)

        Returns:
            complex_sp: (b, c, t, f)
        """

        B, C, T = waveform.shape

        x = rearrange(waveform, 'b c l -> (b c) l')  # (b*c, l)

        x = torch.stft(
            input=x, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(x.device),
            normalized=self.normalized,
            return_complex=self.return_complex
        )  # (b*c, f, t)

        complex_sp = rearrange(x, '(b c) f t -> b c t f', b=B, c=C)  # (b, c, t, f)

        return complex_sp

    def istft(self, complex_sp: Tensor) -> Tensor:
        r"""Convert ISTFT to audio.

        b: batch_size
        c: channels_num
        l: audio_samples
        t: frames_num
        f: freq_bins

        Args:
            complex_sp: (b, c, t, f)

        Returns:
            out: (b, c, l)
        """

        B, C, T, F = complex_sp.shape

        x = rearrange(complex_sp, 'b c t f -> (b c) f t')

        x = torch.istft(
            input=x, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(x.device),
            normalized=self.normalized,
        )  # (b*c, l)

        out = rearrange(x, '(b c) l -> b c l', b=B, c=C)  # (b, c, l)
        
        return x