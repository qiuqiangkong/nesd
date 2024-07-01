import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import MelSpectrogram
from einops import rearrange
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=(3, 3), 
            padding=(1, 1),
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=(3, 3), 
            padding=(1, 1),
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, time_steps, freq_bins)

        Returns:
            output: (batch_size, out_channels, time_steps // 2, freq_bins // 2)
        """

        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x))) 

        output = F.avg_pool2d(x, kernel_size=(2, 2))
        
        return output


class ConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock2, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=(3, 3), 
            padding=(1, 1),
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=(3, 3), 
            padding=(1, 1),
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, time_steps, freq_bins)

        Returns:
            output: (batch_size, out_channels, time_steps // 2, freq_bins // 2)
        """

        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x))) 

        output = F.avg_pool2d(x, kernel_size=(1, 2))
        
        return output


class Cnn(nn.Module):
    def __init__(self, classes_num):
        super(Cnn, self).__init__()

        self.mel_extractor = MelSpectrogram(
            sample_rate=24000,
            n_fft=1024,
            hop_length=240,
            f_min=0.,
            f_max=12000,
            n_mels=64,
            power=2.0,
            normalized=True,
        )

        self.conv1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256)

        # self.gru = nn.GRU(
        #     input_size=2048, 
        #     hidden_size=512, 
        #     num_layers=3, 
        #     bias=True, 
        #     batch_first=True, 
        #     dropout=0.2, 
        #     bidirectional=True
        # )

        self.onset_fc = nn.Linear(2048, classes_num)

    def forward(self, audio):
        """Separation model.

        Args:
            mixture: (batch_size, channels_num, samples_num)

        Outputs:
            output: (batch_size, channels_num, samples_num)
        """
        x = self.mel_extractor(audio)
        # shape: (B, Freq, T)

        x = torch.log10(torch.clamp(x, 1e-8))

        x = rearrange(x, 'b f t -> b t f')
        x = x[:, None, :, :]
        # shape: (B, 1, T, Freq)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # shape: (B, C, T, Freq)

        x = rearrange(x, 'b c t f -> b t (c f)')

        x = torch.sigmoid(self.onset_fc(x))

        # x = x.repeat(1, 8, 1)
        x = x.repeat_interleave(repeats=8, dim=1)

        frame_roll = torch.cat((x, x[:, -1:, :]), dim=1)

        return frame_roll


class CRnn(nn.Module):
    def __init__(self, classes_num):
        super(CRnn, self).__init__()

        self.mel_extractor = MelSpectrogram(
            sample_rate=24000,
            n_fft=1024,
            hop_length=240,
            f_min=0.,
            f_max=12000,
            n_mels=64,
            power=2.0,
            normalized=True,
        )

        self.conv1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256)

        self.gru = nn.GRU(
            input_size=2048, 
            hidden_size=512, 
            num_layers=2, 
            bias=True, 
            batch_first=True, 
            dropout=0.2, 
            bidirectional=True
        )

        self.onset_fc = nn.Linear(1024, classes_num)

    def forward(self, audio):
        """Separation model.

        Args:
            mixture: (batch_size, channels_num, samples_num)

        Outputs:
            output: (batch_size, channels_num, samples_num)
        """
        x = self.mel_extractor(audio)
        # shape: (B, Freq, T)

        x = torch.log10(torch.clamp(x, 1e-8))

        x = rearrange(x, 'b f t -> b t f')
        x = x[:, None, :, :]
        # shape: (B, 1, T, Freq)

        x = self.conv1(x)
        # x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv2(x)
        # x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv3(x)
        # x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        # shape: (B, C, T, Freq)

        x = rearrange(x, 'b c t f -> b t (c f)')

        x, _ = self.gru(x)

        x = torch.sigmoid(self.onset_fc(x))

        # x = x.repeat(1, 8, 1)
        x = x.repeat_interleave(repeats=8, dim=1)

        frame_roll = torch.cat((x, x[:, -1:, :]), dim=1)

        return frame_roll


class CRnn1b(nn.Module):
    def __init__(self, classes_num):
        super(CRnn1b, self).__init__()

        self.mel_extractor = MelSpectrogram(
            sample_rate=24000,
            n_fft=1024,
            hop_length=240,
            f_min=0.,
            f_max=12000,
            n_mels=64,
            power=2.0,
            normalized=True,
        )

        self.conv1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256)

        self.gru = nn.GRU(
            input_size=2048, 
            hidden_size=512, 
            num_layers=2, 
            bias=True, 
            batch_first=True, 
            dropout=0.2, 
            bidirectional=True
        )

        self.onset_fc = nn.Linear(1024, classes_num)

        self.downsample_ratio = 8

    def forward(self, audio):
        """Separation model.

        Args:
            mixture: (batch_size, channels_num, samples_num)

        Outputs:
            output: (batch_size, channels_num, samples_num)
        """
        x = self.mel_extractor(audio)
        # shape: (B, Freq, T)

        x = torch.log10(torch.clamp(x, 1e-8))

        x = rearrange(x, 'b f t -> b t f')
        x = x[:, None, :, :]
        # shape: (B, 1, T, Freq)

        frames_num = x.shape[2]

        x = self.process_image(x)

        x = self.conv1(x)
        # x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv2(x)
        # x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv3(x)
        # x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        # shape: (B, C, T, Freq)

        x = rearrange(x, 'b c t f -> b t (c f)')

        x, _ = self.gru(x)

        x = torch.sigmoid(self.onset_fc(x))

        # x = x.repeat(1, 8, 1)
        x = x.repeat_interleave(repeats=self.downsample_ratio, dim=1)

        frame_roll = self.unprocess_image(x, frames_num)

        return frame_roll

    def process_image(self, x):
        """Cut a spectrum that can be evenly divided by downsample_ratio.

        Args:
            x: E.g., (B, C, 201, 1025)
        
        Outpus:
            output: E.g., (B, C, 208, 1024)
        """

        B, C, T, Freq = x.shape

        pad_len = (
            int(np.ceil(T / self.downsample_ratio)) * self.downsample_ratio
            - T
        )
        output = F.pad(x, pad=(0, 0, 0, pad_len))

        return output

    def unprocess_image(self, x, time_steps):
        """Patch a spectrum to the original shape. E.g.,
        
        Args:
            x: E.g., (B, C, 208, 1024)
        
        Outpus:
            output: E.g., (B, C, 201, 1025)
        """
        
        output = x[:, 0 : time_steps, :]

        return output


# No downsamplinng
class CRnn2(nn.Module):
    def __init__(self, classes_num):
        super(CRnn2, self).__init__()

        self.mel_extractor = MelSpectrogram(
            sample_rate=24000,
            n_fft=1024,
            hop_length=240,
            f_min=0.,
            f_max=12000,
            n_mels=64,
            power=2.0,
            normalized=True,
        )

        self.conv1 = ConvBlock2(in_channels=1, out_channels=64)
        self.conv2 = ConvBlock2(in_channels=64, out_channels=128)
        self.conv3 = ConvBlock2(in_channels=128, out_channels=256)

        self.gru = nn.GRU(
            input_size=2048, 
            hidden_size=512, 
            num_layers=2, 
            bias=True, 
            batch_first=True, 
            dropout=0.2, 
            bidirectional=True
        )

        self.onset_fc = nn.Linear(1024, classes_num)

    def forward(self, audio):
        """Separation model.

        Args:
            mixture: (batch_size, channels_num, samples_num)

        Outputs:
            output: (batch_size, channels_num, samples_num)
        """
        x = self.mel_extractor(audio)
        # shape: (B, Freq, T)

        x = torch.log10(torch.clamp(x, 1e-8))

        x = rearrange(x, 'b f t -> b t f')
        x = x[:, None, :, :]
        # shape: (B, 1, T, Freq)

        x = self.conv1(x)
        # x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv2(x)
        # x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv3(x)
        # x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        # shape: (B, C, T, Freq)

        x = rearrange(x, 'b c t f -> b t (c f)')

        x, _ = self.gru(x)

        frame_roll = torch.sigmoid(self.onset_fc(x))

        return frame_roll