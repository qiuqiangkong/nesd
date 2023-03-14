import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Dict, List, NoReturn, Tuple, Callable, Any

# from torchlibrosa.stft import ISTFT, STFT, magphase, Spectrogram, LogmelFilterBank
from torchlibrosa.stft import ISTFT, STFT, magphase, Spectrogram, LogmelFilterBank

from nesd.models.base import init_layer, init_bn, init_gru, Base, cart2sph_torch, interpolate, get_position_encodings, PositionalEncoder, LookDirectionEncoder, DepthEncoder
from nesd.models.models01 import Model01_Rnn_classwise


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


class ConvBlockCond(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, cond_emb_size):
        
        super(ConvBlockCond, self).__init__()

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

        self.cond_fc = nn.Linear(cond_emb_size, out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        # init_layer(self.conv2)
        init_bn(self.bn1)
        # init_bn(self.bn2)
        init_layer(self.cond_fc)

        
    def forward(self, input, cond_emb):

        z = self.cond_fc(cond_emb)
        z = rearrange(z, 'b t c -> b c t')[:, :, :, None]

        x = input
        x = F.leaky_relu_(self.bn1(self.conv1(x) + z), negative_slope=0.01)
        # x = F.leaky_relu_(self.bn2(self.conv2(x)), negative_slope=0.01)

        x = F.avg_pool2d(x, kernel_size=(1, 2))

        return x


class GruCond(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, cond_emb_size):

        super(GruCond, self).__init__()

        self.gru1 = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=2, bias=True, batch_first=True, dropout=0., bidirectional=True)
        # self.gru2 = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0., bidirectional=True)

        # self.cond_fc = self.cond_fc = nn.Linear(cond_emb_size, hidden_size * 2)

        self.init_weight()

    def init_weight(self):
        init_gru(self.gru1)
        # init_gru(self.gru2)
        # init_layer(self.cond_fc)

    def forward(self, x, cond_emb):

        # z = self.cond_fc(cond_emb)

        x, _ = self.gru1(x)
        # x += z

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
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size)
        # sp = spectrogram_extractor.forward(batch_audio)   # (batch_size, 1, time_steps, freq_bins)

        # Log mel spectrogram
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, n_mels=n_mels)
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


'''
class SedFzLocModel01(nn.Module, Base):
    def __init__(self, 
        microphones_num: int, 
        classes_num: int, 
        do_localization: bool, 
        do_sed: bool, 
        do_separation: bool,
    ):
        super(SedFzLocModel01, self).__init__() 

        self.fz_loc = Model01_Rnn_classwise(
            microphones_num=microphones_num,
            classes_num=classes_num,
            do_localization=do_localization,
            do_sed=do_sed,
            do_separation=do_separation,
        )

        for param in self.fz_loc.parameters():
            param.requires_grad = False

        #
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

    def init_weights(self):
        init_gru(self.gru)
        init_layer(self.fc)

    def forward(self, data_dict):

        agent_see_source_embedding = self.fz_loc(data_dict=data_dict)['agent_see_source_embedding']

        from IPython import embed; embed(using=False); os._exit(0)

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
'''

class SedFzLocModel01(nn.Module, Base):
    def __init__(self, 
        microphones_num: int, 
        classes_num: int,
    ):
        super(SedFzLocModel01, self).__init__() 

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
        positional_embedding_factor = 5
        cond_emb_size = 256
        
        self.stft = STFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.istft = ISTFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.look_direction_encoder = LookDirectionEncoder(
            factor=positional_embedding_factor
        )
        
        self.mic_look_direction_fc = nn.Linear(
            in_features=microphones_num * (positional_embedding_factor * 4), 
            out_features=128, 
            bias=True
        )

        self.agent_look_direction_fc = nn.Linear(
            in_features=(positional_embedding_factor * 4), 
            out_features=128, 
            bias=True
        )

        # Spectrogram
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size)
        # sp = spectrogram_extractor.forward(batch_audio)   # (batch_size, 1, time_steps, freq_bins)

        # Log mel spectrogram
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, n_mels=n_mels)
        # logmel = logmel_extractor.forward(sp)   # (batch_size, 1, time_steps, mel_bins)

        self.conv_block1 = ConvBlockCond(in_channels=1, out_channels=32, kernel_size=(3, 3), cond_emb_size=cond_emb_size)
        self.conv_block2 = ConvBlockCond(in_channels=32, out_channels=64, kernel_size=(3, 3), cond_emb_size=cond_emb_size)
        self.conv_block3 = ConvBlockCond(in_channels=64, out_channels=128, kernel_size=(3, 3), cond_emb_size=cond_emb_size)
        self.conv_block4 = ConvBlockCond(in_channels=128, out_channels=256, kernel_size=(3, 3), cond_emb_size=cond_emb_size)

        self.gru = GruCond(input_size=1024, hidden_size=512, num_layers=2, cond_emb_size=cond_emb_size)

        self.fc = nn.Linear(1024, classes_num)

    def init_weights(self):
        init_gru(self.gru)
        init_layer(self.fc)

    def convert_look_direction_to_embedding(self, look_direction):
        r"""Convert look direction to embedding.

        Args:
            look_direction: (batch_size, directions_num, frames_num, 3)

        Returns:
            look_direction_emb: (batch_size, directions_num, frames_num, emb_size)
        """
        _, look_direction_azimuth, look_direction_colatitude = cart2sph_torch(
            x=look_direction[:, :, :, 0], 
            y=look_direction[:, :, :, 1], 
            z=look_direction[:, :, :, 2],
        )

        look_direction_emb = self.look_direction_encoder(
            azimuth=look_direction_azimuth,
            elevation=look_direction_colatitude,
        )

        return look_direction_emb

    def forward(self, data_dict):

        mic_look_direction = data_dict['mic_look_direction']    # (bs, mics_num, frames_num, 3)
        agent_look_direction = data_dict['agent_look_direction']
        mic_waveform = data_dict['mic_waveform']    # (bs, mics_num, samples_num)
        agents_num = agent_look_direction.shape[1]
        batch_size = mic_waveform.shape[0]

        # ------ 1. Convert positions and look directions to embeddings ------
        mic_look_direction_emb = self.convert_look_direction_to_embedding(
            look_direction=mic_look_direction,
        )   # shape: (bs, mics_num, frames_num, emb_size)
        
        agent_look_direction_emb = self.convert_look_direction_to_embedding(
            look_direction=agent_look_direction,
        )   # shape: (bs, agents_num, frames_num, emb_size)

        #
        mic_look_direction_feature = rearrange(mic_look_direction_emb, 'b c t f -> b t (c f)')
        # shape: (bs, T, mics_num*emb_size=80)

        mic_look_direction_feature = F.leaky_relu_(
            self.mic_look_direction_fc(mic_look_direction_feature)
        )  # shape: (bs, T, 128)

        agent_look_direction_feature = F.leaky_relu_(
            self.agent_look_direction_fc(agent_look_direction_emb)
        )  # (bs, agents_num, T, 128)

        #
        mic_look_direction_feature = torch.tile(mic_look_direction_feature[:, None, :, :], (1, agents_num, 1, 1))

        cond_emb = torch.cat((mic_look_direction_feature, agent_look_direction_feature), dim=3)
        cond_emb = rearrange(cond_emb, 'b k t f -> (b k) t f')

        #
        waveform = torch.mean(mic_waveform, dim=1)
        waveform = torch.tile(waveform[:, None, :], (1, agents_num, 1))

        x = rearrange(waveform, 'b k t -> (b k) t')
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)

        x = self.conv_block1(x, cond_emb)
        x = self.conv_block2(x, cond_emb)
        x = self.conv_block3(x, cond_emb)
        x = self.conv_block4(x, cond_emb)

        x = rearrange(x, 'b c t f -> b t (c f)')
        x = self.gru(x, cond_emb)

        x = torch.sigmoid(self.fc(x))

        classwise_output = rearrange(x, '(b k) t c -> b k t c', b=batch_size)

        output_dict = {'classwise_output': classwise_output}
        return output_dict