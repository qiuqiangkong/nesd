import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Dict, List, NoReturn, Tuple, Callable, Any

from torchlibrosa.stft import ISTFT, STFT, magphase, Spectrogram, LogmelFilterBank

from nesd.models.base import init_layer, init_bn, init_gru, Base, cart2sph_torch, interpolate, get_position_encodings, PositionalEncoder, LookDirectionEncoder, LookDepthEncoder
from nesd.models.models01 import ConvBlockRes
from nesd.utils import PAD


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        
        super(ConvBlock, self).__init__()

        padding = [kernel_size[0] // 2, kernel_size[1] // 2]
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=kernel_size, stride=(1, 1),
                              padding=padding, bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=kernel_size, stride=(1, 1),
                              padding=padding, bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input):
        
        x = input
        x = F.leaky_relu_(self.bn1(self.conv1(x)), negative_slope=0.01)
        x = F.leaky_relu_(self.bn2(self.conv2(x)), negative_slope=0.01)

        return x


class EncoderBlockRes1B(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        downsample: Tuple,
        momentum: float,
    ):
        r"""Encoder block, contains 8 convolutional layers."""
        super(EncoderBlockRes1B, self).__init__()

        self.conv_block1 = ConvBlockRes(
            in_channels, out_channels, kernel_size, momentum
        )
        self.downsample = downsample

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

        Returns:
            encoder_pool: (batch_size, output_feature_maps, downsampled_time_steps, downsampled_freq_bins)
            encoder: (batch_size, output_feature_maps, time_steps, freq_bins)
        """
        encoder = self.conv_block1(input_tensor)
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder


class DecoderBlockRes1B(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        upsample: Tuple,
        momentum: float,
    ):
        r"""Decoder block, contains 1 transposed convolutional and 8 convolutional layers."""
        super(DecoderBlockRes1B, self).__init__()
        self.kernel_size = kernel_size
        self.stride = upsample

        self.conv1 = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.stride,
            stride=self.stride,
            padding=(0, 0),
            bias=False,
            dilation=(1, 1),
        )

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.conv_block2 = ConvBlockRes(
            out_channels * 2, out_channels, kernel_size, momentum
        )
        self.init_weights()

    def init_weights(self):
        r"""Initialize weights."""
        init_bn(self.bn1)
        init_layer(self.conv1)

    def forward(
        self, input_tensor: torch.Tensor, concat_tensor: torch.Tensor
    ) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, input_feature_maps, downsampled_time_steps, downsampled_freq_bins)
            concat_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

        Returns:
            output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)
        """
        x = self.conv1(F.leaky_relu_(self.bn1(input_tensor)))
        # (batch_size, input_feature_maps, time_steps, freq_bins)

        x = torch.cat((x, concat_tensor), dim=1)
        # (batch_size, input_feature_maps * 2, time_steps, freq_bins)

        x = self.conv_block2(x)
        # output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)

        return x


class Model02(nn.Module, Base):
    def __init__(self, 
        mics_num: int, 
        # classes_num: int, 
        # do_localization: bool, 
        # do_sed: bool, 
        # do_separation: bool,
    ):
        super(Model02, self).__init__() 

        window_size = 512
        hop_size = 240
        center = True
        pad_mode = "reflect"
        window = "hann" 
        activation = "relu"
        momentum = 0.01

        # self.window_size = window_size
        # self.hop_size = hop_size
        # self.pad_mode = pad_mode
        # self.do_localization = do_localization
        # self.do_sed = do_sed
        # self.do_separation = do_separation

        positional_embedding_factor = 5
        self.time_downsample_ratio = 2 ** 3
        self.eps = 1e-10

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

        
        self.position_encoder = PositionalEncoder(
            factor=positional_embedding_factor
        )

        self.look_direction_encoder = LookDirectionEncoder(
            factor=positional_embedding_factor
        )
        
        
        self.mag_fc = nn.Conv2d(
            in_channels=mics_num + 1,
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
            bias=True
        )

        self.phase_fc = nn.Conv2d(
            in_channels=mics_num * (mics_num - 1), 
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
            bias=True
        )

        self.mic_signal_encoder_block1 = EncoderBlockRes1B(
            in_channels=64, 
            out_channels=32, 
            kernel_size=(3, 3), 
            downsample=(2, 2), 
            momentum=momentum,
        )

        self.mic_signal_encoder_block2 = EncoderBlockRes1B(
            in_channels=32, 
            out_channels=64, 
            kernel_size=(3, 3), 
            downsample=(2, 2), 
            momentum=momentum,
        )

        self.mic_signal_encoder_block3 = EncoderBlockRes1B(
            in_channels=64, 
            out_channels=128, 
            kernel_size=(3, 3), 
            downsample=(2, 2), 
            momentum=momentum,
        )

        self.mic_signal_fc_reshape = nn.Linear(
            in_features=4096, 
            out_features=1024
        )

        self.mic_position_fc = nn.Linear(
            in_features=mics_num * (positional_embedding_factor * 6), 
            out_features=128, 
            bias=True
        )

        self.mic_look_direction_fc = nn.Linear(
            in_features=mics_num * (positional_embedding_factor * 4), 
            out_features=128, 
            bias=True
        )

        self.agent_position_fc = nn.Linear(
            in_features=(positional_embedding_factor * 6), 
            out_features=128, 
            bias=True
        )

        self.agent_look_direction_fc = nn.Linear(
            in_features=(positional_embedding_factor * 4), 
            out_features=128, 
            bias=True
        )

        # if self.do_localization:
        if True:
            self.loc_fc_block1 = ConvBlock(
                in_channels=1536, 
                out_channels=1024, 
                kernel_size=(1, 1)
            )

            self.loc_fc_block2 = ConvBlock(
                in_channels=1024, 
                out_channels=1024,
                kernel_size=(1, 1),
            )

            self.loc_fc_final = nn.Linear(1024, 1, bias=True)

        '''
        if self.do_separation:
            self.sep_fc_reshape = nn.Linear(
                in_features=1536, 
                out_features=4096, 
                bias=True,
            )

            self.sep_decoder_block1 = DecoderBlockRes1B(
                in_channels=128, 
                out_channels=128, 
                kernel_size=(3, 3), 
                upsample=(2, 2), 
                momentum=momentum,
            )

            self.sep_decoder_block2 = DecoderBlockRes1B(
                in_channels=128, 
                out_channels=64, 
                kernel_size=(3, 3), 
                upsample=(2, 2), 
                momentum=momentum,
            )

            self.sep_decoder_block3 = DecoderBlockRes1B(
                in_channels=64, 
                out_channels=32, 
                kernel_size=(3, 3), 
                upsample=(2, 2), 
                momentum=momentum,
            )

            self.sep_conv_final = nn.Conv2d(
                in_channels=32, 
                out_channels=3, 
                kernel_size=(1, 1), 
                stride=(1, 1), 
                padding=(0, 0), 
                bias=True
            )

        self.init_weights()
        '''

    def init_weights(self):

        init_layer(self.mag_fc)
        init_layer(self.phase_fc)
        init_layer(self.mic_signal_fc_reshape)

        init_layer(self.mic_position_fc)
        init_layer(self.mic_look_direction_fc)

        init_layer(self.agent_position_fc)
        init_layer(self.agent_look_direction_fc)

        if self.do_localization:
            init_layer(self.loc_fc_final)

        if self.do_separation:
            init_layer(self.sep_fc_reshape)
            init_layer(self.sep_conv_final)

    def feature_maps_to_wav(
        self,
        input_tensor: torch.Tensor,
        sp: torch.Tensor,
        sin_in: torch.Tensor,
        cos_in: torch.Tensor,
        audio_length: int,
    ) -> torch.Tensor:
        r"""Convert feature maps to waveform.

        Args:
            input_tensor: (N, 3, time_steps, freq_bins)
            sp: (N, 1, time_steps, freq_bins)
            sin_in: (N, 1, time_steps, freq_bins)
            cos_in: (N, 1, time_steps, freq_bins)

        Outputs:
            waveform: (N, segment_samples)
        """
        x = input_tensor
        mask_mag = torch.sigmoid(x[:, 0 : 1, :, :])
        _mask_real = torch.tanh(x[:, 1 : 2, :, :])
        _mask_imag = torch.tanh(x[:, 2 : 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        
        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in * mask_cos - sin_in * mask_sin
        )
        out_sin = (
            sin_in * mask_cos + cos_in * mask_sin
        )

        # Calculate |Y|.
        out_mag = F.relu_(sp * mask_mag)
        # (N, 1, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # (N, 1, time_steps, freq_bins)

        # ISTFT.
        waveform = self.istft(out_real, out_imag, audio_length)
        # (N, segment_samples)

        return waveform

    def convert_look_directions_to_embedding(self, look_direction):
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

    def convert_positions_to_embedding(self, position):
        r"""Convert look direction to embedding.

        Args:
            position: (batch_size, positions_num, frames_num, 3)

        Returns:
            position_emb: (batch_size, positions_num, frames_num, emb_size)
        """
        position_emb = self.position_encoder(
            x=position[:, :, :, 0],
            y=position[:, :, :, 1],
            z=position[:, :, :, 2],
        )
        return position_emb

    def repeat_conv_features(self, x, repeats_num):
        x = torch.tile(x[:, None, :, :, :], (1, repeats_num, 1, 1, 1))
        x = rearrange(x, 'b n c t f -> (b n) c t f')
        return x

    def forward(
        self, 
        data_dict: Dict, 
        do_separation=None,
    ):

        mic_positions = data_dict["mic_positions"]
        mic_look_directions = data_dict["mic_look_directions"]
        mic_signals = data_dict["mic_signals"]
        agent_positions = data_dict["agent_positions"]
        agent_look_directions = data_dict["agent_look_directions"]

        # ------ 1. Convert positions and look directions to embeddings ------
        mic_positions_emb = self.convert_positions_to_embedding(position=mic_positions)
        # shape: (bs, mics_num, frames_num, emb_size)

        mic_look_directions_emb = self.convert_look_directions_to_embedding(
            look_direction=mic_look_directions,
        )   # shape: (bs, mics_num, frames_num, emb_size)
        
        agent_positions_emb = self.convert_positions_to_embedding(position=agent_positions)
        # shape: (bs, agents_num, frames_num, emb_size)

        agent_look_directions_emb = self.convert_look_directions_to_embedding(
            look_direction=agent_look_directions,
        )   # shape: (bs, agents_num, frames_num, emb_size)

        # ------ 2. Extract features ------
        sum_signals = torch.sum(mic_signals, dim=1, keepdim=True)
        # (bs, 1, samples_num)

        total_mag, total_cos, total_sin = self.wav_to_spectrogram_phase(sum_signals, self.eps)
        # (bs, 1, frames_num, freq_bins)

        mic_mag, mic_cos, mic_sin = self.wav_to_spectrogram_phase(mic_signals, self.eps)
        # (bs, mics_num, frames_num, freq_bins)

        total_real = total_mag * total_cos  # (bs, 1, frames_num, freq_bins)
        total_imag = total_mag * total_sin  # (bs, 1, frames_num, freq_bins)

        delta_cos_list = []
        delta_sin_list = []

        for i in range(1, 4):
            for j in range(0, i):
                _delta_cos = mic_cos[:, i, :, :] * mic_cos[:, j, :, :] + \
                    mic_sin[:, i, :, :] * mic_sin[:, j, :, :]

                _delta_sin = mic_sin[:, i, :, :] * mic_cos[:, j, :, :] - \
                    mic_cos[:, i, :, :] * mic_sin[:, j, :, :]

                delta_cos_list.append(_delta_cos)
                delta_sin_list.append(_delta_sin)

        delta_cos = torch.stack(delta_cos_list, dim=1)
        delta_sin = torch.stack(delta_sin_list, dim=1)

        # delta_real = total_mag * delta_cos
        # delta_imag = total_mag * delta_sin

        mag_feature = torch.cat((mic_mag, total_mag), dim=1)
        # shape: (bs, mics_num + 1, frames_num, freq_bins)

        phase_feature = torch.cat((delta_cos, delta_sin), dim=1)
        # shape: (bs, mics_num * (mics_num - 1), frames_num, freq_bins)

        mag_feature = self.mag_fc(mag_feature)  # (bs, 32, frames_num, freq_bins)
        phase_feature = self.phase_fc(phase_feature)  # (bs, 32, frames_num, freq_bins)
        
        x = torch.cat((mag_feature, phase_feature), dim=1)
        # (bs, 64, frames_num, freq_bins)

        # Pad spectrogram to be evenly divided by downsample ratio.
        frames_num = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio))
            * self.time_downsample_ratio
            - frames_num
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        # x: (batch_size, input_channels * subbands_num, padded_time_steps, freq_bins)

        # Let frequency bins be evenly divided by 2, e.g., 257 -> 256
        x = x[..., 0 : x.shape[-1] - 1]  # (bs, input_channels, T, F)

        # Pad zero frames after the last frame.
        mic_positions_emb = F.pad(mic_positions_emb, pad=(0, 0, 0, pad_len))
        mic_look_directions_emb = F.pad(mic_look_directions_emb, pad=(0, 0, 0, pad_len))
        agent_positions_emb = F.pad(agent_positions_emb, pad=(0, 0, 0, pad_len))
        agent_look_directions_emb = F.pad(agent_look_directions_emb, pad=(0, 0, 0, pad_len))

        # ------ 3. Forward data to the network ------

        # 1) Transform mic signal.
        enc1_pool, enc1 = self.mic_signal_encoder_block1(x)
        enc2_pool, enc2 = self.mic_signal_encoder_block2(enc1_pool)
        enc3_pool, enc3 = self.mic_signal_encoder_block3(enc2_pool)
        # enc3_pool: (bs, 128, T=38, F=32)

        mic_signals_feature = rearrange(enc3_pool, 'b c t f -> b t (c f)')
        # (bs, T, C*F=4096)

        mic_signals_feature = F.leaky_relu_(
            self.mic_signal_fc_reshape(mic_signals_feature), negative_slope=0.01,
        )  # shape: (bs, T, 1024)
        
        # 2) Transform mic position and look direction embeddings.
        mic_positions_feature = rearrange(mic_positions_emb, 'b c t f -> b t (c f)')
        # shape: (bs, T, mics_num*emb_size=120)

        mic_positions_feature = F.leaky_relu_(
            self.mic_position_fc(mic_positions_feature)
        )  # shape: (bs, T, 128)

        mic_positions_feature = mic_positions_feature[:, 0 :: self.time_downsample_ratio, :]
        
        mic_look_directions_feature = rearrange(mic_look_directions_emb, 'b c t f -> b t (c f)')
        # shape: (bs, T, mics_num*emb_size=80)

        mic_look_directions_feature = F.leaky_relu_(
            self.mic_look_direction_fc(mic_look_directions_feature)
        )  # shape: (bs, T, 128)

        mic_look_directions_feature = mic_look_directions_feature[:, 0 :: self.time_downsample_ratio, :]

        # 3) Concatenate mic signal, position, and look direction features.
        mics_feature = torch.cat((
            mic_signals_feature, mic_positions_feature, mic_look_directions_feature
        ), dim=-1)  # (batch_size, T, 1280)

        agents_num = agent_positions.shape[1]

        mics_feature = mics_feature[:, None, :, :].expand(size=(-1, agents_num, -1, -1))
        # mics_feature = torch.tile(mics_feature[:, None, :, :], (1, agents_num, 1, 1))
        # shape: (bs, agents_num, T=38, 1280)

        
        # 4) Calculate agent position and look direction features.
        agent_positions_feature = F.leaky_relu_(self.agent_position_fc(agent_positions_emb))
        # shape: (bs, agents_num, T, 128)

        agent_positions_feature = agent_positions_feature[:, :, 0 :: self.time_downsample_ratio, :]
        # shape: (bs, agents_num, T=38, 128)

        agent_look_directions_feature = F.leaky_relu_(
            self.agent_look_direction_fc(agent_look_directions_emb)
        )  # (bs, agents_num, T, 128)

        agent_look_directions_feature = agent_look_directions_feature[:, :, 0 :: self.time_downsample_ratio, :]
        # shape: (bs, agents_num, T=38, 128)

        # Concatenate mic features and agent features.
        shared_feature = torch.cat((mics_feature, agent_positions_feature, 
            agent_look_directions_feature), dim=-1)
        # (bs, agents_num, T=38, 1536)

        output_dict = {}

        # if do_localization:
        if True:
            batch_size, agents_num, _T, _C = shared_feature.shape

            x = rearrange(shared_feature, 'b n t c -> (b n) c t')[:, :, :, None]
            # (bs * agents_num, C=1536, T, 1)

            x = self.loc_fc_block1(x)
            # (bs * agents_num, 1024, T, 1)

            x = self.loc_fc_block2(x)
            # (bs * agents_num, 1024, T, 1)

            x = rearrange(x.squeeze(), '(b n) c t -> (b n) t c', b=batch_size)
            # (bs * agents_num, T, 1024)

            x = torch.sigmoid(self.loc_fc_final(x))
            # (bs * agents_num, T=38, C=1)

            x = x.repeat_interleave(repeats=self.time_downsample_ratio, dim=1)[:, 0 : frames_num, :]
            # (bs * agents_num, T=301, C=1)

            agent_look_directions_has_source = rearrange(x.squeeze(), '(b n) t -> b n t', b=batch_size)

            output_dict['agent_look_directions_has_source'] = agent_look_directions_has_source

        # from IPython import embed; embed(using=False); os._exit(0)

        return output_dict
        '''
        if False:
        # if do_separation:

            max_agents_contain_waveform = data_dict['max_agents_contain_waveform']

            x = shared_feature[:, 0 : max_agents_contain_waveform, :, :]
            # (bs, n=max_agents_contain_waveform, T=38, C=1536)

            x = F.leaky_relu_(self.sep_fc_reshape(x))
            # (bs, n, T=38, 4096)

            x = rearrange(x, 'b n t (c f) -> (b n) c t f', c=128, f=32)
            # (bs * n, C=128, T=38, F=32)

            enc1 = self.repeat_conv_features(x=enc1, repeats_num=max_agents_contain_waveform)
            # (bs * n, C=32, T=304, F=256)

            enc2 = self.repeat_conv_features(x=enc2, repeats_num=max_agents_contain_waveform)
            # (bs * n, C=64, T=152, F=128)

            enc3 = self.repeat_conv_features(x=enc3, repeats_num=max_agents_contain_waveform)
            # (bs * n, C=128, T=76, F=64)

            x = self.sep_decoder_block1(x, enc3)
            x = self.sep_decoder_block2(x, enc2)
            x = self.sep_decoder_block3(x, enc1)
            # (bs * n, C=32, T=304, F=256)

            x = self.sep_conv_final(x)
            # (bs * n, C=3, T=304, F=256)

            # Recover shape
            x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257. 
            # (bs * n, C=3, T=304, F=257)

            x = x[:, :, 0:frames_num, :]
            # (bs * n, C=3, T=301, F=257)

            repeat_total_mag = self.repeat_conv_features(
                x=total_mag, repeats_num=max_agents_contain_waveform
            )  # (bs * n, C=1, T=301, F=257)

            repeat_total_sin = self.repeat_conv_features(
                x=total_sin, repeats_num=max_agents_contain_waveform
            )  # (bs * n, C=1, T=301, F=257)

            repeat_total_cos = self.repeat_conv_features(
                x=total_cos, repeats_num=max_agents_contain_waveform
            )  # (bs * n, C=1, T=301, F=257)

            audio_length = total_waveform.shape[-1]

            x = self.feature_maps_to_wav(
                input_tensor=x,
                sp=repeat_total_mag,
                sin_in=repeat_total_sin,
                cos_in=repeat_total_cos,
                audio_length=audio_length,
            )
            # (bs * n, segment_samples)

            agent_waveform = rearrange(x, '(b n) t -> b n t', b=batch_size)
            # (bs, n=max_agents_contain_waveform, segment_samples)

            output_dict['agent_waveform'] = agent_waveform

        return output_dict
        '''


class Model02_depth(nn.Module, Base):
    def __init__(self, 
        mics_num: int, 
        # classes_num: int, 
        # do_localization: bool, 
        # do_sed: bool, 
        # do_separation: bool,
    ):
        super(Model02_depth, self).__init__() 

        window_size = 512
        hop_size = 240
        center = True
        pad_mode = "reflect"
        window = "hann" 
        activation = "relu"
        momentum = 0.01

        # self.window_size = window_size
        # self.hop_size = hop_size
        # self.pad_mode = pad_mode
        # self.do_localization = do_localization
        # self.do_sed = do_sed
        # self.do_separation = do_separation

        positional_embedding_factor = 5
        self.time_downsample_ratio = 2 ** 3
        self.eps = 1e-10

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

        
        self.position_encoder = PositionalEncoder(
            factor=positional_embedding_factor
        )

        self.look_direction_encoder = LookDirectionEncoder(
            factor=positional_embedding_factor
        )
        
        self.look_depth_encoder = LookDepthEncoder(
            factor=positional_embedding_factor
        )
        
        self.mag_fc = nn.Conv2d(
            in_channels=mics_num + 1,
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
            bias=True
        )

        self.phase_fc = nn.Conv2d(
            in_channels=mics_num * (mics_num - 1), 
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
            bias=True
        )

        self.mic_signal_encoder_block1 = EncoderBlockRes1B(
            in_channels=64, 
            out_channels=32, 
            kernel_size=(3, 3), 
            downsample=(2, 2), 
            momentum=momentum,
        )

        self.mic_signal_encoder_block2 = EncoderBlockRes1B(
            in_channels=32, 
            out_channels=64, 
            kernel_size=(3, 3), 
            downsample=(2, 2), 
            momentum=momentum,
        )

        self.mic_signal_encoder_block3 = EncoderBlockRes1B(
            in_channels=64, 
            out_channels=128, 
            kernel_size=(3, 3), 
            downsample=(2, 2), 
            momentum=momentum,
        )

        self.mic_signal_fc_reshape = nn.Linear(
            in_features=4096, 
            out_features=1024
        )

        self.mic_position_fc = nn.Linear(
            in_features=mics_num * (positional_embedding_factor * 6), 
            out_features=128, 
            bias=True
        )

        self.mic_look_direction_fc = nn.Linear(
            in_features=mics_num * (positional_embedding_factor * 4), 
            out_features=128, 
            bias=True
        )

        self.agent_position_fc = nn.Linear(
            in_features=(positional_embedding_factor * 6), 
            out_features=128, 
            bias=True
        )

        self.agent_look_direction_fc = nn.Linear(
            in_features=(positional_embedding_factor * 4), 
            out_features=128, 
            bias=True
        )

        self.agent_look_depth_fc = nn.Linear(
            in_features=(positional_embedding_factor * 2), 
            out_features=128, 
            bias=True
        )
        
        # if self.do_localization:
        if True:
            self.loc_fc_block1 = ConvBlock(
                in_channels=1664, 
                out_channels=1024, 
                kernel_size=(1, 1)
            )

            self.loc_fc_block2 = ConvBlock(
                in_channels=1024, 
                out_channels=1024,
                kernel_size=(1, 1),
            )

            self.loc_fc_final = nn.Linear(1024, 1, bias=True)

        if True:
            self.depth_fc_block1 = ConvBlock(
                in_channels=1664, 
                out_channels=1024, 
                kernel_size=(1, 1)
            )

            self.depth_fc_block2 = ConvBlock(
                in_channels=1024, 
                out_channels=1024,
                kernel_size=(1, 1),
            )

            self.depth_fc_final = nn.Linear(1024, 1, bias=True)

        '''
        if self.do_separation:
            self.sep_fc_reshape = nn.Linear(
                in_features=1536, 
                out_features=4096, 
                bias=True,
            )

            self.sep_decoder_block1 = DecoderBlockRes1B(
                in_channels=128, 
                out_channels=128, 
                kernel_size=(3, 3), 
                upsample=(2, 2), 
                momentum=momentum,
            )

            self.sep_decoder_block2 = DecoderBlockRes1B(
                in_channels=128, 
                out_channels=64, 
                kernel_size=(3, 3), 
                upsample=(2, 2), 
                momentum=momentum,
            )

            self.sep_decoder_block3 = DecoderBlockRes1B(
                in_channels=64, 
                out_channels=32, 
                kernel_size=(3, 3), 
                upsample=(2, 2), 
                momentum=momentum,
            )

            self.sep_conv_final = nn.Conv2d(
                in_channels=32, 
                out_channels=3, 
                kernel_size=(1, 1), 
                stride=(1, 1), 
                padding=(0, 0), 
                bias=True
            )

        self.init_weights()
        '''

    def init_weights(self):

        init_layer(self.mag_fc)
        init_layer(self.phase_fc)
        init_layer(self.mic_signal_fc_reshape)

        init_layer(self.mic_position_fc)
        init_layer(self.mic_look_direction_fc)

        init_layer(self.agent_position_fc)
        init_layer(self.agent_look_direction_fc)

        # if self.do_localization:
        init_layer(self.loc_fc_final)
        init_layer(self.depth_fc_final)

        if self.do_separation:
            init_layer(self.sep_fc_reshape)
            init_layer(self.sep_conv_final)

    def feature_maps_to_wav(
        self,
        input_tensor: torch.Tensor,
        sp: torch.Tensor,
        sin_in: torch.Tensor,
        cos_in: torch.Tensor,
        audio_length: int,
    ) -> torch.Tensor:
        r"""Convert feature maps to waveform.

        Args:
            input_tensor: (N, 3, time_steps, freq_bins)
            sp: (N, 1, time_steps, freq_bins)
            sin_in: (N, 1, time_steps, freq_bins)
            cos_in: (N, 1, time_steps, freq_bins)

        Outputs:
            waveform: (N, segment_samples)
        """
        x = input_tensor
        mask_mag = torch.sigmoid(x[:, 0 : 1, :, :])
        _mask_real = torch.tanh(x[:, 1 : 2, :, :])
        _mask_imag = torch.tanh(x[:, 2 : 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        
        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in * mask_cos - sin_in * mask_sin
        )
        out_sin = (
            sin_in * mask_cos + cos_in * mask_sin
        )

        # Calculate |Y|.
        out_mag = F.relu_(sp * mask_mag)
        # (N, 1, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # (N, 1, time_steps, freq_bins)

        # ISTFT.
        waveform = self.istft(out_real, out_imag, audio_length)
        # (N, segment_samples)

        return waveform

    def convert_look_directions_to_embedding(self, look_direction):
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

    def convert_positions_to_embedding(self, position):
        r"""Convert look direction to embedding.

        Args:
            position: (batch_size, positions_num, frames_num, 3)

        Returns:
            position_emb: (batch_size, positions_num, frames_num, emb_size)
        """
        position_emb = self.position_encoder(
            x=position[:, :, :, 0],
            y=position[:, :, :, 1],
            z=position[:, :, :, 2],
        )
        return position_emb

    def convert_look_depths_to_embedding(self, look_depth):
        r"""Convert look direction to embedding.

        Args:
            position: (batch_size, positions_num, frames_num, 3)

        Returns:
            position_emb: (batch_size, positions_num, frames_num, emb_size)
        """
        look_depth_emb = self.look_depth_encoder(
            depth=look_depth[:, :, :, 0],
        )
        return look_depth_emb

    def repeat_conv_features(self, x, repeats_num):
        x = torch.tile(x[:, None, :, :, :], (1, repeats_num, 1, 1, 1))
        x = rearrange(x, 'b n c t f -> (b n) c t f')
        return x

    def forward(
        self, 
        data_dict: Dict, 
        do_separation=None,
    ):

        mic_positions = data_dict["mic_positions"]
        mic_look_directions = data_dict["mic_look_directions"]
        mic_signals = data_dict["mic_signals"]
        agent_positions = data_dict["agent_positions"]
        agent_look_directions = data_dict["agent_look_directions"]
        agent_look_depths = data_dict["agent_look_depths"]

        agent_look_depths_mask = (agent_look_depths != PAD) * 1.

        # ------ 1. Convert positions and look directions to embeddings ------
        mic_positions_emb = self.convert_positions_to_embedding(position=mic_positions)
        # shape: (bs, mics_num, frames_num, emb_size)

        mic_look_directions_emb = self.convert_look_directions_to_embedding(
            look_direction=mic_look_directions,
        )   # shape: (bs, mics_num, frames_num, emb_size)
        
        agent_positions_emb = self.convert_positions_to_embedding(position=agent_positions)
        # shape: (bs, agents_num, frames_num, emb_size)

        agent_look_directions_emb = self.convert_look_directions_to_embedding(
            look_direction=agent_look_directions,
        )   # shape: (bs, agents_num, frames_num, emb_size)

        agent_look_depths_emb = self.convert_look_depths_to_embedding(
            look_depth=agent_look_depths,
        )   # shape: (bs, agents_num, frames_num, emb_size)

        # ------ 2. Extract features ------
        sum_signals = torch.sum(mic_signals, dim=1, keepdim=True)
        # (bs, 1, samples_num)

        total_mag, total_cos, total_sin = self.wav_to_spectrogram_phase(sum_signals, self.eps)
        # (bs, 1, frames_num, freq_bins)

        mic_mag, mic_cos, mic_sin = self.wav_to_spectrogram_phase(mic_signals, self.eps)
        # (bs, mics_num, frames_num, freq_bins)

        total_real = total_mag * total_cos  # (bs, 1, frames_num, freq_bins)
        total_imag = total_mag * total_sin  # (bs, 1, frames_num, freq_bins)

        delta_cos_list = []
        delta_sin_list = []

        for i in range(1, 4):
            for j in range(0, i):
                _delta_cos = mic_cos[:, i, :, :] * mic_cos[:, j, :, :] + \
                    mic_sin[:, i, :, :] * mic_sin[:, j, :, :]

                _delta_sin = mic_sin[:, i, :, :] * mic_cos[:, j, :, :] - \
                    mic_cos[:, i, :, :] * mic_sin[:, j, :, :]

                delta_cos_list.append(_delta_cos)
                delta_sin_list.append(_delta_sin)

        delta_cos = torch.stack(delta_cos_list, dim=1)
        delta_sin = torch.stack(delta_sin_list, dim=1)

        # delta_real = total_mag * delta_cos
        # delta_imag = total_mag * delta_sin

        mag_feature = torch.cat((mic_mag, total_mag), dim=1)
        # shape: (bs, mics_num + 1, frames_num, freq_bins)

        phase_feature = torch.cat((delta_cos, delta_sin), dim=1)
        # shape: (bs, mics_num * (mics_num - 1), frames_num, freq_bins)

        mag_feature = self.mag_fc(mag_feature)  # (bs, 32, frames_num, freq_bins)
        phase_feature = self.phase_fc(phase_feature)  # (bs, 32, frames_num, freq_bins)
        
        x = torch.cat((mag_feature, phase_feature), dim=1)
        # (bs, 64, frames_num, freq_bins)

        # Pad spectrogram to be evenly divided by downsample ratio.
        frames_num = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio))
            * self.time_downsample_ratio
            - frames_num
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        # x: (batch_size, input_channels * subbands_num, padded_time_steps, freq_bins)

        # Let frequency bins be evenly divided by 2, e.g., 257 -> 256
        x = x[..., 0 : x.shape[-1] - 1]  # (bs, input_channels, T, F)

        # Pad zero frames after the last frame.
        mic_positions_emb = F.pad(mic_positions_emb, pad=(0, 0, 0, pad_len))
        mic_look_directions_emb = F.pad(mic_look_directions_emb, pad=(0, 0, 0, pad_len))
        agent_positions_emb = F.pad(agent_positions_emb, pad=(0, 0, 0, pad_len))
        agent_look_directions_emb = F.pad(agent_look_directions_emb, pad=(0, 0, 0, pad_len))
        agent_look_depths_emb = F.pad(agent_look_depths_emb, pad=(0, 0, 0, pad_len))
        agent_look_depths_mask = F.pad(agent_look_depths_mask, pad=(0, 0, 0, pad_len))

        # ------ 3. Forward data to the network ------

        # 1) Transform mic signal.
        enc1_pool, enc1 = self.mic_signal_encoder_block1(x)
        enc2_pool, enc2 = self.mic_signal_encoder_block2(enc1_pool)
        enc3_pool, enc3 = self.mic_signal_encoder_block3(enc2_pool)
        # enc3_pool: (bs, 128, T=38, F=32)

        mic_signals_feature = rearrange(enc3_pool, 'b c t f -> b t (c f)')
        # (bs, T, C*F=4096)

        mic_signals_feature = F.leaky_relu_(
            self.mic_signal_fc_reshape(mic_signals_feature), negative_slope=0.01,
        )  # shape: (bs, T, 1024)
        
        # 2) Transform mic position and look direction embeddings.
        mic_positions_feature = rearrange(mic_positions_emb, 'b c t f -> b t (c f)')
        # shape: (bs, T, mics_num*emb_size=120)

        mic_positions_feature = F.leaky_relu_(
            self.mic_position_fc(mic_positions_feature)
        )  # shape: (bs, T, 128)

        mic_positions_feature = mic_positions_feature[:, 0 :: self.time_downsample_ratio, :]
        
        mic_look_directions_feature = rearrange(mic_look_directions_emb, 'b c t f -> b t (c f)')
        # shape: (bs, T, mics_num*emb_size=80)

        mic_look_directions_feature = F.leaky_relu_(
            self.mic_look_direction_fc(mic_look_directions_feature)
        )  # shape: (bs, T, 128)

        mic_look_directions_feature = mic_look_directions_feature[:, 0 :: self.time_downsample_ratio, :]

        # 3) Concatenate mic signal, position, and look direction features.
        mics_feature = torch.cat((
            mic_signals_feature, mic_positions_feature, mic_look_directions_feature
        ), dim=-1)  # (batch_size, T, 1280)

        agents_num = agent_positions.shape[1]

        mics_feature = mics_feature[:, None, :, :].expand(size=(-1, agents_num, -1, -1))
        # mics_feature = torch.tile(mics_feature[:, None, :, :], (1, agents_num, 1, 1))
        # shape: (bs, agents_num, T=38, 1280)

        
        # 4) Calculate agent position and look direction features.
        # ---
        agent_positions_feature = F.leaky_relu_(self.agent_position_fc(agent_positions_emb))
        # shape: (bs, agents_num, T, 128)

        agent_positions_feature = agent_positions_feature[:, :, 0 :: self.time_downsample_ratio, :]
        # shape: (bs, agents_num, T=38, 128)

        # ---
        agent_look_directions_feature = F.leaky_relu_(
            self.agent_look_direction_fc(agent_look_directions_emb)
        )  # (bs, agents_num, T, 128)

        agent_look_directions_feature = agent_look_directions_feature[:, :, 0 :: self.time_downsample_ratio, :]
        # shape: (bs, agents_num, T=38, 128)

        # ---
        agent_look_depths_feature = F.leaky_relu_(
            self.agent_look_depth_fc(agent_look_depths_emb)
        )  # (bs, agents_num, T, 128)

        agent_look_depths_feature = agent_look_depths_feature * agent_look_depths_mask

        agent_look_depths_feature = agent_look_depths_feature[:, :, 0 :: self.time_downsample_ratio, :]
        # shape: (bs, agents_num, T=38, 128)

        # ---
        # Concatenate mic features and agent features.
        shared_feature = torch.cat((mics_feature, agent_positions_feature, 
            agent_look_directions_feature, agent_look_depths_feature), dim=-1)
        # (bs, agents_num, T=38, 1536)

        output_dict = {}

        # if do_localization:
        if True:
            batch_size, agents_num, _T, _C = shared_feature.shape

            x = rearrange(shared_feature, 'b n t c -> (b n) c t')[:, :, :, None]
            # (bs * agents_num, C=1536, T, 1)

            x = self.loc_fc_block1(x)
            # (bs * agents_num, 1024, T, 1)

            x = self.loc_fc_block2(x)
            # (bs * agents_num, 1024, T, 1)

            x = rearrange(x.squeeze(), '(b n) c t -> (b n) t c', b=batch_size)
            # (bs * agents_num, T, 1024)

            x = torch.sigmoid(self.loc_fc_final(x))
            # (bs * agents_num, T=38, C=1)

            x = x.repeat_interleave(repeats=self.time_downsample_ratio, dim=1)[:, 0 : frames_num, :]
            # (bs * agents_num, T=301, C=1)

            agent_look_directions_has_source = rearrange(x.squeeze(), '(b n) t -> b n t', b=batch_size)

            output_dict['agent_look_directions_has_source'] = agent_look_directions_has_source

        if True:
            batch_size, agents_num, _T, _C = shared_feature.shape

            x = rearrange(shared_feature, 'b n t c -> (b n) c t')[:, :, :, None]
            # (bs * agents_num, C=1536, T, 1)

            x = self.depth_fc_block1(x)
            # (bs * agents_num, 1024, T, 1)

            x = self.depth_fc_block2(x)
            # (bs * agents_num, 1024, T, 1)

            x = rearrange(x.squeeze(), '(b n) c t -> (b n) t c', b=batch_size)
            # (bs * agents_num, T, 1024)

            x = torch.sigmoid(self.depth_fc_final(x))
            # (bs * agents_num, T=38, C=1)

            x = x.repeat_interleave(repeats=self.time_downsample_ratio, dim=1)[:, 0 : frames_num, :]
            # (bs * agents_num, T=301, C=1)

            agent_look_directions_depth = rearrange(x.squeeze(), '(b n) t -> b n t', b=batch_size)

            output_dict['agent_look_depths_has_source'] = agent_look_directions_depth

        return output_dict
        '''
        if False:
        # if do_separation:

            max_agents_contain_waveform = data_dict['max_agents_contain_waveform']

            x = shared_feature[:, 0 : max_agents_contain_waveform, :, :]
            # (bs, n=max_agents_contain_waveform, T=38, C=1536)

            x = F.leaky_relu_(self.sep_fc_reshape(x))
            # (bs, n, T=38, 4096)

            x = rearrange(x, 'b n t (c f) -> (b n) c t f', c=128, f=32)
            # (bs * n, C=128, T=38, F=32)

            enc1 = self.repeat_conv_features(x=enc1, repeats_num=max_agents_contain_waveform)
            # (bs * n, C=32, T=304, F=256)

            enc2 = self.repeat_conv_features(x=enc2, repeats_num=max_agents_contain_waveform)
            # (bs * n, C=64, T=152, F=128)

            enc3 = self.repeat_conv_features(x=enc3, repeats_num=max_agents_contain_waveform)
            # (bs * n, C=128, T=76, F=64)

            x = self.sep_decoder_block1(x, enc3)
            x = self.sep_decoder_block2(x, enc2)
            x = self.sep_decoder_block3(x, enc1)
            # (bs * n, C=32, T=304, F=256)

            x = self.sep_conv_final(x)
            # (bs * n, C=3, T=304, F=256)

            # Recover shape
            x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257. 
            # (bs * n, C=3, T=304, F=257)

            x = x[:, :, 0:frames_num, :]
            # (bs * n, C=3, T=301, F=257)

            repeat_total_mag = self.repeat_conv_features(
                x=total_mag, repeats_num=max_agents_contain_waveform
            )  # (bs * n, C=1, T=301, F=257)

            repeat_total_sin = self.repeat_conv_features(
                x=total_sin, repeats_num=max_agents_contain_waveform
            )  # (bs * n, C=1, T=301, F=257)

            repeat_total_cos = self.repeat_conv_features(
                x=total_cos, repeats_num=max_agents_contain_waveform
            )  # (bs * n, C=1, T=301, F=257)

            audio_length = total_waveform.shape[-1]

            x = self.feature_maps_to_wav(
                input_tensor=x,
                sp=repeat_total_mag,
                sin_in=repeat_total_sin,
                cos_in=repeat_total_cos,
                audio_length=audio_length,
            )
            # (bs * n, segment_samples)

            agent_waveform = rearrange(x, '(b n) t -> b n t', b=batch_size)
            # (bs, n=max_agents_contain_waveform, segment_samples)

            output_dict['agent_waveform'] = agent_waveform

        return output_dict
        '''


class Model02_sep(nn.Module, Base):
    def __init__(self, 
        mics_num: int, 
        # classes_num: int, 
        # do_localization: bool, 
        # do_sed: bool, 
        # do_separation: bool,
    ):
        super(Model02_sep, self).__init__() 

        window_size = 512
        hop_size = 240
        center = True
        pad_mode = "reflect"
        window = "hann" 
        activation = "relu"
        momentum = 0.01

        # self.window_size = window_size
        # self.hop_size = hop_size
        # self.pad_mode = pad_mode
        # self.do_localization = do_localization
        # self.do_sed = do_sed
        # self.do_separation = do_separation

        positional_embedding_factor = 5
        self.time_downsample_ratio = 2 ** 3
        self.eps = 1e-10

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

        
        self.position_encoder = PositionalEncoder(
            factor=positional_embedding_factor
        )

        self.look_direction_encoder = LookDirectionEncoder(
            factor=positional_embedding_factor
        )
        
        self.look_depth_encoder = LookDepthEncoder(
            factor=positional_embedding_factor
        )
        
        self.mag_fc = nn.Conv2d(
            in_channels=mics_num + 1,
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
            bias=True
        )

        self.phase_fc = nn.Conv2d(
            in_channels=mics_num * (mics_num - 1), 
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
            bias=True
        )

        self.mic_signal_encoder_block1 = EncoderBlockRes1B(
            in_channels=64, 
            out_channels=32, 
            kernel_size=(3, 3), 
            downsample=(2, 2), 
            momentum=momentum,
        )

        self.mic_signal_encoder_block2 = EncoderBlockRes1B(
            in_channels=32, 
            out_channels=64, 
            kernel_size=(3, 3), 
            downsample=(2, 2), 
            momentum=momentum,
        )

        self.mic_signal_encoder_block3 = EncoderBlockRes1B(
            in_channels=64, 
            out_channels=128, 
            kernel_size=(3, 3), 
            downsample=(2, 2), 
            momentum=momentum,
        )

        self.mic_signal_fc_reshape = nn.Linear(
            in_features=4096, 
            out_features=1024
        )

        self.mic_position_fc = nn.Linear(
            in_features=mics_num * (positional_embedding_factor * 6), 
            out_features=128, 
            bias=True
        )

        self.mic_look_direction_fc = nn.Linear(
            in_features=mics_num * (positional_embedding_factor * 4), 
            out_features=128, 
            bias=True
        )

        self.agent_position_fc = nn.Linear(
            in_features=(positional_embedding_factor * 6), 
            out_features=128, 
            bias=True
        )

        self.agent_look_direction_fc = nn.Linear(
            in_features=(positional_embedding_factor * 4), 
            out_features=128, 
            bias=True
        )

        self.agent_look_depth_fc = nn.Linear(
            in_features=(positional_embedding_factor * 2), 
            out_features=128, 
            bias=True
        )
        
        # if self.do_localization:
        if True:
            self.loc_fc_block1 = ConvBlock(
                in_channels=1664, 
                out_channels=1024, 
                kernel_size=(1, 1)
            )

            self.loc_fc_block2 = ConvBlock(
                in_channels=1024, 
                out_channels=1024,
                kernel_size=(1, 1),
            )

            self.loc_fc_final = nn.Linear(1024, 1, bias=True)

        if True:
            self.depth_fc_block1 = ConvBlock(
                in_channels=1664, 
                out_channels=1024, 
                kernel_size=(1, 1)
            )

            self.depth_fc_block2 = ConvBlock(
                in_channels=1024, 
                out_channels=1024,
                kernel_size=(1, 1),
            )

            self.depth_fc_final = nn.Linear(1024, 1, bias=True)

        
        # if self.do_separation:
        if True:
            self.sep_fc_reshape = nn.Linear(
                in_features=1664, 
                out_features=4096, 
                bias=True,
            )

            self.sep_decoder_block1 = DecoderBlockRes1B(
                in_channels=128, 
                out_channels=128, 
                kernel_size=(3, 3), 
                upsample=(2, 2), 
                momentum=momentum,
            )

            self.sep_decoder_block2 = DecoderBlockRes1B(
                in_channels=128, 
                out_channels=64, 
                kernel_size=(3, 3), 
                upsample=(2, 2), 
                momentum=momentum,
            )

            self.sep_decoder_block3 = DecoderBlockRes1B(
                in_channels=64, 
                out_channels=32, 
                kernel_size=(3, 3), 
                upsample=(2, 2), 
                momentum=momentum,
            )

            self.sep_conv_final = nn.Conv2d(
                in_channels=32, 
                out_channels=3, 
                kernel_size=(1, 1), 
                stride=(1, 1), 
                padding=(0, 0), 
                bias=True
            )

        self.init_weights()
        

    def init_weights(self):

        init_layer(self.mag_fc)
        init_layer(self.phase_fc)
        init_layer(self.mic_signal_fc_reshape)

        init_layer(self.mic_position_fc)
        init_layer(self.mic_look_direction_fc)

        init_layer(self.agent_position_fc)
        init_layer(self.agent_look_direction_fc)

        # if self.do_localization:
        init_layer(self.loc_fc_final)
        init_layer(self.depth_fc_final)

        # if self.do_separation:
        if True:
            init_layer(self.sep_fc_reshape)
            init_layer(self.sep_conv_final)

    def feature_maps_to_wav(
        self,
        input_tensor: torch.Tensor,
        sp: torch.Tensor,
        sin_in: torch.Tensor,
        cos_in: torch.Tensor,
        audio_length: int,
    ) -> torch.Tensor:
        r"""Convert feature maps to waveform.

        Args:
            input_tensor: (N, 3, time_steps, freq_bins)
            sp: (N, 1, time_steps, freq_bins)
            sin_in: (N, 1, time_steps, freq_bins)
            cos_in: (N, 1, time_steps, freq_bins)

        Outputs:
            waveform: (N, segment_samples)
        """
        x = input_tensor
        mask_mag = torch.sigmoid(x[:, 0 : 1, :, :])
        _mask_real = torch.tanh(x[:, 1 : 2, :, :])
        _mask_imag = torch.tanh(x[:, 2 : 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        
        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in * mask_cos - sin_in * mask_sin
        )
        out_sin = (
            sin_in * mask_cos + cos_in * mask_sin
        )

        # Calculate |Y|.
        out_mag = F.relu_(sp * mask_mag)
        # (N, 1, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # (N, 1, time_steps, freq_bins)

        # ISTFT.
        waveform = self.istft(out_real, out_imag, audio_length)
        # (N, segment_samples)

        return waveform

    def convert_look_directions_to_embedding(self, look_direction):
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

    def convert_positions_to_embedding(self, position):
        r"""Convert look direction to embedding.

        Args:
            position: (batch_size, positions_num, frames_num, 3)

        Returns:
            position_emb: (batch_size, positions_num, frames_num, emb_size)
        """
        position_emb = self.position_encoder(
            x=position[:, :, :, 0],
            y=position[:, :, :, 1],
            z=position[:, :, :, 2],
        )
        return position_emb

    def convert_look_depths_to_embedding(self, look_depth):
        r"""Convert look direction to embedding.

        Args:
            position: (batch_size, positions_num, frames_num, 3)

        Returns:
            position_emb: (batch_size, positions_num, frames_num, emb_size)
        """
        look_depth_emb = self.look_depth_encoder(
            depth=look_depth[:, :, :, 0],
        )
        return look_depth_emb

    def repeat_conv_features(self, x, repeats_num):
        x = torch.tile(x[:, None, :, :, :], (1, repeats_num, 1, 1, 1))
        x = rearrange(x, 'b n c t f -> (b n) c t f')
        return x

    def forward(
        self, 
        data_dict: Dict, 
        do_separation=None,
        mode="train",
    ):

        mic_positions = data_dict["mic_positions"]
        mic_look_directions = data_dict["mic_look_directions"]
        mic_signals = data_dict["mic_signals"]
        agent_positions = data_dict["agent_positions"]
        agent_look_directions = data_dict["agent_look_directions"]
        agent_look_depths = data_dict["agent_look_depths"]

        # agent_signals = data_dict["agent_signals"]

        agent_look_depths_mask = (agent_look_depths != PAD) * 1.

        # ------ 1. Convert positions and look directions to embeddings ------
        mic_positions_emb = self.convert_positions_to_embedding(position=mic_positions)
        # shape: (bs, mics_num, frames_num, emb_size)

        mic_look_directions_emb = self.convert_look_directions_to_embedding(
            look_direction=mic_look_directions,
        )   # shape: (bs, mics_num, frames_num, emb_size)
        
        agent_positions_emb = self.convert_positions_to_embedding(position=agent_positions)
        # shape: (bs, agents_num, frames_num, emb_size)

        agent_look_directions_emb = self.convert_look_directions_to_embedding(
            look_direction=agent_look_directions,
        )   # shape: (bs, agents_num, frames_num, emb_size)

        agent_look_depths_emb = self.convert_look_depths_to_embedding(
            look_depth=agent_look_depths,
        )   # shape: (bs, agents_num, frames_num, emb_size)

        # ------ 2. Extract features ------
        sum_signals = torch.sum(mic_signals, dim=1, keepdim=True)
        # (bs, 1, samples_num)

        total_mag, total_cos, total_sin = self.wav_to_spectrogram_phase(sum_signals, self.eps)
        # (bs, 1, frames_num, freq_bins)

        mic_mag, mic_cos, mic_sin = self.wav_to_spectrogram_phase(mic_signals, self.eps)
        # (bs, mics_num, frames_num, freq_bins)

        total_real = total_mag * total_cos  # (bs, 1, frames_num, freq_bins)
        total_imag = total_mag * total_sin  # (bs, 1, frames_num, freq_bins)

        delta_cos_list = []
        delta_sin_list = []

        for i in range(1, 4):
            for j in range(0, i):
                _delta_cos = mic_cos[:, i, :, :] * mic_cos[:, j, :, :] + \
                    mic_sin[:, i, :, :] * mic_sin[:, j, :, :]

                _delta_sin = mic_sin[:, i, :, :] * mic_cos[:, j, :, :] - \
                    mic_cos[:, i, :, :] * mic_sin[:, j, :, :]

                delta_cos_list.append(_delta_cos)
                delta_sin_list.append(_delta_sin)

        delta_cos = torch.stack(delta_cos_list, dim=1)
        delta_sin = torch.stack(delta_sin_list, dim=1)

        # delta_real = total_mag * delta_cos
        # delta_imag = total_mag * delta_sin

        mag_feature = torch.cat((mic_mag, total_mag), dim=1)
        # shape: (bs, mics_num + 1, frames_num, freq_bins)

        phase_feature = torch.cat((delta_cos, delta_sin), dim=1)
        # shape: (bs, mics_num * (mics_num - 1), frames_num, freq_bins)

        mag_feature = self.mag_fc(mag_feature)  # (bs, 32, frames_num, freq_bins)
        phase_feature = self.phase_fc(phase_feature)  # (bs, 32, frames_num, freq_bins)
        
        x = torch.cat((mag_feature, phase_feature), dim=1)
        # (bs, 64, frames_num, freq_bins)

        # Pad spectrogram to be evenly divided by downsample ratio.
        frames_num = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio))
            * self.time_downsample_ratio
            - frames_num
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        # x: (batch_size, input_channels * subbands_num, padded_time_steps, freq_bins)

        # Let frequency bins be evenly divided by 2, e.g., 257 -> 256
        x = x[..., 0 : x.shape[-1] - 1]  # (bs, input_channels, T, F)

        # Pad zero frames after the last frame.
        mic_positions_emb = F.pad(mic_positions_emb, pad=(0, 0, 0, pad_len))
        mic_look_directions_emb = F.pad(mic_look_directions_emb, pad=(0, 0, 0, pad_len))
        agent_positions_emb = F.pad(agent_positions_emb, pad=(0, 0, 0, pad_len))
        agent_look_directions_emb = F.pad(agent_look_directions_emb, pad=(0, 0, 0, pad_len))
        agent_look_depths_emb = F.pad(agent_look_depths_emb, pad=(0, 0, 0, pad_len))
        agent_look_depths_mask = F.pad(agent_look_depths_mask, pad=(0, 0, 0, pad_len))

        # ------ 3. Forward data to the network ------

        # 1) Transform mic signal.
        enc1_pool, enc1 = self.mic_signal_encoder_block1(x)
        enc2_pool, enc2 = self.mic_signal_encoder_block2(enc1_pool)
        enc3_pool, enc3 = self.mic_signal_encoder_block3(enc2_pool)
        # enc3_pool: (bs, 128, T=38, F=32)

        mic_signals_feature = rearrange(enc3_pool, 'b c t f -> b t (c f)')
        # (bs, T, C*F=4096)

        mic_signals_feature = F.leaky_relu_(
            self.mic_signal_fc_reshape(mic_signals_feature), negative_slope=0.01,
        )  # shape: (bs, T, 1024)
        
        # 2) Transform mic position and look direction embeddings.
        mic_positions_feature = rearrange(mic_positions_emb, 'b c t f -> b t (c f)')
        # shape: (bs, T, mics_num*emb_size=120)

        mic_positions_feature = F.leaky_relu_(
            self.mic_position_fc(mic_positions_feature)
        )  # shape: (bs, T, 128)

        mic_positions_feature = mic_positions_feature[:, 0 :: self.time_downsample_ratio, :]
        
        mic_look_directions_feature = rearrange(mic_look_directions_emb, 'b c t f -> b t (c f)')
        # shape: (bs, T, mics_num*emb_size=80)

        mic_look_directions_feature = F.leaky_relu_(
            self.mic_look_direction_fc(mic_look_directions_feature)
        )  # shape: (bs, T, 128)

        mic_look_directions_feature = mic_look_directions_feature[:, 0 :: self.time_downsample_ratio, :]

        # 3) Concatenate mic signal, position, and look direction features.
        mics_feature = torch.cat((
            mic_signals_feature, mic_positions_feature, mic_look_directions_feature
        ), dim=-1)  # (batch_size, T, 1280)

        agents_num = agent_positions.shape[1]

        mics_feature = mics_feature[:, None, :, :].expand(size=(-1, agents_num, -1, -1))
        # mics_feature = torch.tile(mics_feature[:, None, :, :], (1, agents_num, 1, 1))
        # shape: (bs, agents_num, T=38, 1280)

        
        # 4) Calculate agent position and look direction features.
        # ---
        agent_positions_feature = F.leaky_relu_(self.agent_position_fc(agent_positions_emb))
        # shape: (bs, agents_num, T, 128)

        agent_positions_feature = agent_positions_feature[:, :, 0 :: self.time_downsample_ratio, :]
        # shape: (bs, agents_num, T=38, 128)

        # ---
        agent_look_directions_feature = F.leaky_relu_(
            self.agent_look_direction_fc(agent_look_directions_emb)
        )  # (bs, agents_num, T, 128)

        agent_look_directions_feature = agent_look_directions_feature[:, :, 0 :: self.time_downsample_ratio, :]
        # shape: (bs, agents_num, T=38, 128)

        # ---
        agent_look_depths_feature = F.leaky_relu_(
            self.agent_look_depth_fc(agent_look_depths_emb)
        )  # (bs, agents_num, T, 128)

        agent_look_depths_feature = agent_look_depths_feature * agent_look_depths_mask

        agent_look_depths_feature = agent_look_depths_feature[:, :, 0 :: self.time_downsample_ratio, :]
        # shape: (bs, agents_num, T=38, 128)

        # ---
        # Concatenate mic features and agent features.
        shared_feature = torch.cat((mics_feature, agent_positions_feature, 
            agent_look_directions_feature, agent_look_depths_feature), dim=-1)
        # (bs, agents_num, T=38, 1536)

        output_dict = {}

        # if do_localization:
        if True:
            batch_size, agents_num, _T, _C = shared_feature.shape

            x = rearrange(shared_feature, 'b n t c -> (b n) c t')[:, :, :, None]
            # (bs * agents_num, C=1536, T, 1)

            x = self.loc_fc_block1(x)
            # (bs * agents_num, 1024, T, 1)

            x = self.loc_fc_block2(x)
            # (bs * agents_num, 1024, T, 1)

            x = rearrange(x.squeeze(), '(b n) c t -> (b n) t c', b=batch_size)
            # (bs * agents_num, T, 1024)

            x = torch.sigmoid(self.loc_fc_final(x))
            # (bs * agents_num, T=38, C=1)

            x = x.repeat_interleave(repeats=self.time_downsample_ratio, dim=1)[:, 0 : frames_num, :]
            # (bs * agents_num, T=301, C=1)

            agent_look_directions_has_source = rearrange(x.squeeze(), '(b n) t -> b n t', b=batch_size)

            output_dict['agent_look_directions_has_source'] = agent_look_directions_has_source

        if True:
            batch_size, agents_num, _T, _C = shared_feature.shape

            x = rearrange(shared_feature, 'b n t c -> (b n) c t')[:, :, :, None]
            # (bs * agents_num, C=1536, T, 1)

            x = self.depth_fc_block1(x)
            # (bs * agents_num, 1024, T, 1)

            x = self.depth_fc_block2(x)
            # (bs * agents_num, 1024, T, 1)

            x = rearrange(x.squeeze(), '(b n) c t -> (b n) t c', b=batch_size)
            # (bs * agents_num, T, 1024)

            x = torch.sigmoid(self.depth_fc_final(x))
            # (bs * agents_num, T=38, C=1)

            x = x.repeat_interleave(repeats=self.time_downsample_ratio, dim=1)[:, 0 : frames_num, :]
            # (bs * agents_num, T=301, C=1)

            agent_look_directions_depth = rearrange(x.squeeze(), '(b n) t -> b n t', b=batch_size)

            output_dict['agent_look_depths_has_source'] = agent_look_directions_depth

        if True:

            if mode == "train":
                agent_active_indexes = data_dict["agent_active_indexes"]
                agent_active_indexes_mask = data_dict["agent_active_indexes_mask"]
                max_active_indexes = agent_active_indexes.shape[-1]

                batch_size = shared_feature.shape[0]
                tmp = []
                for i in range(batch_size):
                    tmp.append(shared_feature[i][agent_active_indexes[i]])
                x = torch.stack(tmp, dim=0)
                # (bs, n=max_active_indexes, T=38, C=1536)

            elif mode == "inference":
                x = shared_feature
                max_active_indexes = x.shape[1]
            else:
                raise NotImplementedError

            x = F.leaky_relu_(self.sep_fc_reshape(x))
            # (bs, n, T=38, 4096)

            x = rearrange(x, 'b n t (c f) -> (b n) c t f', c=128, f=32)
            # (bs * n, C=128, T=38, F=32)

            enc1 = self.repeat_conv_features(x=enc1, repeats_num=max_active_indexes)
            # (bs * n, C=32, T=304, F=256)

            enc2 = self.repeat_conv_features(x=enc2, repeats_num=max_active_indexes)
            # (bs * n, C=64, T=152, F=128)

            enc3 = self.repeat_conv_features(x=enc3, repeats_num=max_active_indexes)
            # (bs * n, C=128, T=76, F=64)

            x = self.sep_decoder_block1(x, enc3)
            x = self.sep_decoder_block2(x, enc2)
            x = self.sep_decoder_block3(x, enc1)
            # (bs * n, C=32, T=304, F=256)

            x = self.sep_conv_final(x)
            # (bs * n, C=3, T=304, F=256)

            # Recover shape
            x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257. 
            # (bs * n, C=3, T=304, F=257)

            x = x[:, :, 0:frames_num, :]
            # (bs * n, C=3, T=301, F=257)

            repeat_total_mag = self.repeat_conv_features(
                x=total_mag, repeats_num=max_active_indexes
            )  # (bs * n, C=1, T=301, F=257)

            repeat_total_sin = self.repeat_conv_features(
                x=total_sin, repeats_num=max_active_indexes
            )  # (bs * n, C=1, T=301, F=257)

            repeat_total_cos = self.repeat_conv_features(
                x=total_cos, repeats_num=max_active_indexes
            )  # (bs * n, C=1, T=301, F=257)

            audio_length = mic_signals.shape[-1]

            x = self.feature_maps_to_wav(
                input_tensor=x,
                sp=repeat_total_mag,
                sin_in=repeat_total_sin,
                cos_in=repeat_total_cos,
                audio_length=audio_length,
            )
            # (bs * n, segment_samples)

            agent_waveform = rearrange(x, '(b n) t -> b n t', b=batch_size)
            # (bs, n=max_agents_contain_waveform, segment_samples)

            if mode == "train":
                agent_waveform *= agent_active_indexes_mask[:, :, None]
            elif mode == "inference":
                pass
            else:
                raise NotImplementedError

            output_dict['agent_signals'] = agent_waveform

        return output_dict
        