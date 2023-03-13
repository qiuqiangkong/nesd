import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Dict, List, NoReturn, Tuple, Callable, Any

# from torchlibrosa.stft import ISTFT, STFT, magphase, Spectrogram, LogmelFilterBank
import torchlibrosa as tl

from nesd.models.base import init_layer, init_bn, init_gru, Base, cart2sph_torch, interpolate, get_position_encodings, PositionalEncoder, LookDirectionEncoder, DepthEncoder


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        
        super(ConvBlock, self).__init__()

        padding = [kernel_size[0] // 2, kernel_size[1] // 2]
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=kernel_size, stride=(1, 1),
                              padding=padding, bias=False)
                              
        # self.conv2 = nn.Conv2d(in_channels=out_channels, 
        #                       out_channels=out_channels,
        #                       kernel_size=kernel_size, stride=(1, 1),
        #                       padding=padding, bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        # self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        # init_layer(self.conv2)
        init_bn(self.bn1)
        # init_bn(self.bn2)

        
    def forward(self, input):
        
        x = input
        x = F.leaky_relu_(self.bn1(self.conv1(x)), negative_slope=0.01)
        # x = F.leaky_relu_(self.bn2(self.conv2(x)), negative_slope=0.01)

        x = F.avg_pool2d(x, kernel_size=(1, 2))

        return x


class SedModel01(nn.Module, Base):
    def __init__(self, 
        classes_num: int,
    ):
        super(SedModel01, self).__init__() 

        window_size = 512
        hop_size = 240
        center = True
        pad_mode = "reflect"
        window = "hann" 
        activation = "relu"
        momentum = 0.01
        sample_rate = 24000
        n_mels = 64

        self.window_size = window_size
        self.hop_size = hop_size
        self.pad_mode = pad_mode
        
        self.eps = 1e-10

        # Spectrogram
        self.spectrogram_extractor = tl.Spectrogram(n_fft=window_size, hop_length=hop_size)
        # sp = spectrogram_extractor.forward(batch_audio)   # (batch_size, 1, time_steps, freq_bins)

        # Log mel spectrogram
        self.logmel_extractor = tl.LogmelFilterBank(sr=sample_rate, n_fft=window_size, n_mels=n_mels)
        # logmel = logmel_extractor.forward(sp)   # (batch_size, 1, time_steps, mel_bins)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=32, kernel_size=(3, 3))
        self.conv_block2 = ConvBlock(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=128, kernel_size=(3, 3))
        self.conv_block4 = ConvBlock(in_channels=128, out_channels=256, kernel_size=(3, 3))

        self.gru = nn.GRU(input_size=1024, hidden_size=512, num_layers=2, bias=True, batch_first=True, dropout=0., bidirectional=True)

        self.fc = nn.Linear(1024, classes_num)

        self.init_weights()

    def init_weights(self):
        init_gru(self.gru)
        init_layer(self.fc)

    def forward(self, input_dict):

        waveform = input_dict['waveform']
        waveform = torch.mean(waveform, dim=1)

        x = self.spectrogram_extractor(waveform)
        x = self.logmel_extractor(x)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        x = rearrange(x, 'b c t f -> b t (c f)')
        x, _ = self.gru(x)
        x = torch.sigmoid(self.fc(x))

        output_dict = {'classwise_output': x}
        return output_dict