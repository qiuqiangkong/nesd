import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Dict, List, NoReturn, Tuple, Callable, Any

from torchlibrosa.stft import ISTFT, STFT, magphase, Spectrogram, LogmelFilterBank

from nesd.models.base import init_layer, init_bn, init_gru, Base, cart2sph_torch, interpolate, get_position_encodings, PositionalEncoder, LookDirectionEncoder 


class ConvBlockRes(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        momentum: float,
    ):
        r"""Residual block."""
        super(ConvBlockRes, self).__init__()

        padding = [kernel_size[0] // 2, kernel_size[1] // 2]

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            )
            self.is_shortcut = True
        else:
            self.is_shortcut = False

        self.init_weights()

    def init_weights(self) -> NoReturn:
        r"""Initialize weights."""
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_layer(self.conv1)
        init_layer(self.conv2)

        if self.is_shortcut:
            init_layer(self.shortcut)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

        Returns:
            output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)
        """
        x = self.conv1(F.leaky_relu_(self.bn1(input_tensor), negative_slope=0.01))
        x = self.conv2(F.leaky_relu_(self.bn2(x), negative_slope=0.01))

        if self.is_shortcut:
            return self.shortcut(input_tensor) + x
        else:
            return input_tensor + x


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


class Model01(nn.Module, Base):
    def __init__(self, 
        microphones_num: int, 
        classes_num: int, 
        do_localization: bool, 
        do_sed: bool, 
        do_separation: bool,
    ):
        super(Model01, self).__init__() 

        window_size = 512
        hop_size = 240
        center = True
        pad_mode = "reflect"
        window = "hann" 
        activation = "relu"
        momentum = 0.01

        self.window_size = window_size
        self.hop_size = hop_size
        self.pad_mode = pad_mode
        self.do_localization = True
        self.do_sed = False
        self.do_separation = do_separation

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
            in_channels=microphones_num + 1,
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
            bias=True
        )

        self.phase_fc = nn.Conv2d(
            in_channels=microphones_num * (microphones_num - 1), 
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
            in_features=microphones_num * (positional_embedding_factor * 6), 
            out_features=128, 
            bias=True
        )

        self.mic_look_direction_fc = nn.Linear(
            in_features=microphones_num * (positional_embedding_factor * 4), 
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

        if self.do_localization:
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

    def convert_position_to_embedding(self, position):
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

    def forward(self, data_dict):

        mic_position = data_dict['mic_position']    # (bs, mics_num, frames_num, 3)
        mic_look_direction = data_dict['mic_look_direction']    # (bs, mics_num, frames_num, 3)
        mic_waveform = data_dict['mic_waveform']    # (bs, mics_num, samples_num)
        agent_position = data_dict['agent_position']    # (bs, agents_num, frames_num, 3)
        agent_look_direction = data_dict['agent_look_direction']    # (bs, agents_num, frames_num, 3)

        # ------ 1. Convert positions and look directions to embeddings ------
        mic_position_emb = self.convert_position_to_embedding(position=mic_position)
        # shape: (bs, mics_num, frames_num, emb_size)

        mic_look_direction_emb = self.convert_look_direction_to_embedding(
            look_direction=mic_look_direction,
        )   # shape: (bs, mics_num, frames_num, emb_size)
        
        agent_position_emb = self.convert_position_to_embedding(position=agent_position)
        # shape: (bs, agents_num, frames_num, emb_size)

        agent_look_direction_emb = self.convert_look_direction_to_embedding(
            look_direction=agent_look_direction,
        )   # shape: (bs, agents_num, frames_num, emb_size)

        # ------ 2. Extract features ------
        total_waveform = torch.sum(mic_waveform, dim=1, keepdim=True)
        # (bs, 1, samples_num)

        total_mag, total_cos, total_sin = self.wav_to_spectrogram_phase(total_waveform, self.eps)
        # (bs, 1, frames_num, freq_bins)

        mic_mag, mic_cos, mic_sin = self.wav_to_spectrogram_phase(mic_waveform, self.eps)
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
        mic_position_emb = F.pad(mic_position_emb, pad=(0, 0, 0, pad_len))
        mic_look_direction_emb = F.pad(mic_look_direction_emb, pad=(0, 0, 0, pad_len))
        agent_position_emb = F.pad(agent_position_emb, pad=(0, 0, 0, pad_len))
        agent_look_direction_emb = F.pad(agent_look_direction_emb, pad=(0, 0, 0, pad_len))

        # ------ 3. Forward data to the network ------

        # 1) Transform mic signal.
        enc1_pool, enc1 = self.mic_signal_encoder_block1(x)
        enc2_pool, enc2 = self.mic_signal_encoder_block2(enc1_pool)
        enc3_pool, enc3 = self.mic_signal_encoder_block3(enc2_pool)
        # enc3_pool: (bs, 128, T=38, F=32)

        mic_signal_feature = rearrange(enc3_pool, 'b c t f -> b t (c f)')
        # (bs, T, C*F=4096)

        mic_signal_feature = F.leaky_relu_(
            self.mic_signal_fc_reshape(mic_signal_feature), negative_slope=0.01,
        )  # shape: (bs, T, 1024)
        
        # 2) Transform mic position and look direction embeddings.
        mic_position_feature = rearrange(mic_position_emb, 'b c t f -> b t (c f)')
        # shape: (bs, T, mics_num*emb_size=120)

        mic_position_feature = F.leaky_relu_(
            self.mic_position_fc(mic_position_feature)
        )  # shape: (bs, T, 128)

        mic_position_feature = mic_position_feature[:, 0 :: self.time_downsample_ratio, :]
        
        mic_look_direction_feature = rearrange(mic_look_direction_emb, 'b c t f -> b t (c f)')
        # shape: (bs, T, mics_num*emb_size=80)

        mic_look_direction_feature = F.leaky_relu_(
            self.mic_look_direction_fc(mic_look_direction_feature)
        )  # shape: (bs, T, 128)

        mic_look_direction_feature = mic_look_direction_feature[:, 0 :: self.time_downsample_ratio, :]

        # 3) Concatenate mic signal, position, and look direction features.
        mic_feature = torch.cat((
            mic_signal_feature, mic_position_feature, mic_look_direction_feature
        ), dim=-1)  # (batch_size, T, 1280)

        agents_num = agent_position.shape[1]

        mic_feature = torch.tile(mic_feature[:, None, :, :], (1, agents_num, 1, 1))
        # shape: (bs, agents_num, T=38, 1280)

        # 4) Calculate agent position and look direction features.
        agent_position_feature = F.leaky_relu_(self.agent_position_fc(agent_position_emb))
        # shape: (bs, agents_num, T, 128)

        agent_position_feature = agent_position_feature[:, :, 0 :: self.time_downsample_ratio, :]
        # shape: (bs, agents_num, T=38, 128)

        agent_look_direction_feature = F.leaky_relu_(
            self.agent_look_direction_fc(agent_look_direction_emb)
        )  # (bs, agents_num, T, 128)

        agent_look_direction_feature = agent_look_direction_feature[:, :, 0 :: self.time_downsample_ratio, :]
        # shape: (bs, agents_num, T=38, 128)

        # Concatenate mic features and agent features.
        shared_feature = torch.cat((mic_feature, agent_position_feature, 
            agent_look_direction_feature), dim=-1)
        # (bs, agents_num, T=38, 1536)

        output_dict = {}

        if self.do_localization:
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

            agent_see_source = rearrange(x.squeeze(), '(b n) t -> b n t', b=batch_size)

            output_dict['agent_see_source'] = agent_see_source

        if self.do_separation:

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


import math
class PositionalEncoder2:
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


class PositionalEncoderRoomXYZ2:
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


class ConvBlockAfter(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlockAfter, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(1, 1), stride=(1, 1),
                              padding=(0, 0), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(1, 1), stride=(1, 1),
                              padding=(0, 0), bias=False)
                              
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
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))

        return x


class Model01_bak(nn.Module, Base):
    def __init__(self, microphones_num, classes_num, do_localization=None, do_sed=None, do_separation=None):
        super(Model01_bak, self).__init__() 

        window_size = 512
        hop_size = 240
        center = True
        pad_mode = "reflect"
        window = "hann" 
        activation = "relu"
        momentum = 0.01
        classes_num = 1

        self.window_size = window_size
        self.hop_size = hop_size
        self.pad_mode = pad_mode

        positional_embedding_factor = 5
        self.time_downsample_ratio = 2 ** 3

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

        self.positional_encoder = PositionalEncoder2(factor=positional_embedding_factor)
        self.positional_encoder_room_xyz = PositionalEncoderRoomXYZ2(factor=positional_embedding_factor)

        self.pre_value_cnn = nn.Conv2d(in_channels=microphones_num + 1, out_channels=32,
                              kernel_size=(1, 1), stride=(1, 1),
                              padding=(0, 0), bias=True)
        self.pre_angle_cnn = nn.Conv2d(in_channels=12, out_channels=32,
                              kernel_size=(1, 1), stride=(1, 1),
                              padding=(0, 0), bias=True)

        self.input_value_cnn1 = EncoderBlockRes1B(in_channels=64, out_channels=32, kernel_size=(3, 3), downsample=(2, 2), momentum=momentum)

        self.input_value_cnn2 = EncoderBlockRes1B(in_channels=32, out_channels=64, kernel_size=(3, 3), downsample=(2, 2), momentum=momentum)

        self.input_value_cnn3 = EncoderBlockRes1B(in_channels=64, out_channels=128, kernel_size=(3, 3), downsample=(2, 2), momentum=momentum)

        # self.input_value_cnn4 = EncoderBlockRes1B(in_channels=128, out_channels=256, kernel_size=(3, 3), downsample=(2, 2), momentum=momentum)

        self.input_value_fc1 = nn.Linear(in_features=32 * 128, out_features=1024)

        self.input_angle_fc = nn.Linear(in_features=(positional_embedding_factor * 4) * microphones_num, out_features=128, bias=True)
        self.input_pos_xyz_fc = nn.Linear(in_features=(positional_embedding_factor * 6) * microphones_num, out_features=128, bias=True)

        self.output_angle_fc = nn.Linear(in_features=(positional_embedding_factor * 4), out_features=128, bias=True)
        self.output_pos_xyz_fc = nn.Linear(in_features=(positional_embedding_factor * 6), out_features=128, bias=True)

        self.conv_block1 = ConvBlockAfter(in_channels=1536, out_channels=1024)
        self.conv_block2 = ConvBlockAfter(in_channels=1024, out_channels=1024)

        # self.fc_final_tracking = nn.Linear(1024, 1, bias=True)
        self.fc_final_cla = nn.Linear(1024, classes_num, bias=True)

        # Sep
        self.fc_inter = nn.Linear(in_features=1536, out_features=128 * 32, bias=True)

        self.dec_block1 = DecoderBlockRes1B(in_channels=128, out_channels=128, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        self.dec_block2 = DecoderBlockRes1B(in_channels=128, out_channels=64, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        self.dec_block3 = DecoderBlockRes1B(in_channels=64, out_channels=32, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        self.dec_final = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.init_weights()

        self.target_sources_num = 1
        self.output_channels = 1
        self.K = 3

    def init_weights(self):

        init_layer(self.pre_value_cnn)
        init_layer(self.pre_angle_cnn)
        init_layer(self.input_value_fc1)

        init_layer(self.input_angle_fc)
        init_layer(self.input_pos_xyz_fc)
        init_layer(self.output_angle_fc)
        init_layer(self.output_pos_xyz_fc)

        init_layer(self.fc_inter)
        # init_layer(self.fc_final_tracking)
        init_layer(self.fc_final_cla)

        init_layer(self.dec_final)

    def pad(self, x, window_size, hop_size):
        # To make show the center of fft window is the center of a frame
        x = F.pad(x, pad=(window_size // 2 - hop_size // 2, window_size // 2 - hop_size // 2), mode=self.pad_mode)
        return x

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
            input_tensor: (batch_size, target_sources_num * output_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, input_channels, time_steps, freq_bins)
            sin_in: (batch_size, input_channels, time_steps, freq_bins)
            cos_in: (batch_size, input_channels, time_steps, freq_bins)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (batch_size, target_sources_num * output_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.output_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, output_channels, self.K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        # linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # mask_cos, mask_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)
        # out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        # out_mag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size * target_sources_num * output_channels
        shape = (
            batch_size * self.target_sources_num * self.output_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * output_channels, segments_num)

        # Reshape.
        waveform = x.reshape(
            batch_size, self.target_sources_num * self.output_channels, audio_length
        )
        # (batch_size, target_sources_num * output_channels, segments_num)

        return waveform


    def forward(self, data_dict, do_sep=True):

        eps = 1e-10

        # mic_position = data_dict['mic_position']
        # mic_look_direction = data_dict['mic_look_direction']
        # mic_waveform = data_dict['mic_waveform']
        # ray_direction = data_dict['ray_direction']
        # ray_position = data_dict['ray_origin']

        mic_position = data_dict['mic_position']
        mic_look_direction = data_dict['mic_look_direction']
        mic_waveform = data_dict['mic_waveform']
        ray_direction = data_dict['agent_look_direction']
        ray_position = data_dict['agent_position']

        # Mic direction embedding
        _, mic_look_azimuth, mic_look_colatitude = cart2sph_torch(
            x=mic_look_direction[:, :, :, 0], 
            y=mic_look_direction[:, :, :, 1], 
            z=mic_look_direction[:, :, :, 2],
        )
        mic_direction_emb = self.positional_encoder(
            azimuth=mic_look_azimuth,
            elevation=mic_look_colatitude,
        )
        # (batch_size, mics_num, outputs_num)

        # ray direction embedding
        _, ray_di_azimuth, ray_di_zenith = cart2sph_torch(
            x=ray_direction[:, :, :, 0], 
            y=ray_direction[:, :, :, 1], 
            z=ray_direction[:, :, :, 2],
        )
        ray_direction_emb = self.positional_encoder(
            azimuth=ray_di_azimuth,
            elevation=ray_di_zenith,
        )
        # (batch_size, mics_num, outputs_num)

        mic_pos_xyz_emb = self.positional_encoder_room_xyz(
            x=mic_position[:, :, :, 0],
            y=mic_position[:, :, :, 1],
            z=mic_position[:, :, :, 2],
        )
        
        ray_pos_xyz_emb = self.positional_encoder_room_xyz(
            x=ray_position[:, :, :, 0],
            y=ray_position[:, :, :, 1],
            z=ray_position[:, :, :, 2],
        )

        omni_waveform = torch.sum(mic_waveform, dim=1, keepdim=True)
        omni_mag, omni_cos, omni_sin = self.wav_to_spectrogram_phase(omni_waveform, eps)
        mic_mag, mic_cos, mic_sin = self.wav_to_spectrogram_phase(mic_waveform, eps) # (batch_size, 1, L, F)
        # (batch_size, 1, L, F)

        omni_real = omni_mag * omni_cos
        omni_imag = omni_mag * omni_sin

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

        delta_real = omni_mag * delta_cos
        delta_imag = omni_mag * delta_sin

        x1 = torch.cat((omni_mag, mic_mag), dim=1)
        x2 = torch.cat((delta_cos, delta_sin), dim=1)

        x1 = self.pre_value_cnn(x1)
        x2 = self.pre_angle_cnn(x2)
        
        x = torch.cat((x1, x2), dim=1)

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio))
            * self.time_downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        # x: (batch_size, input_channels * subbands_num, padded_time_steps, freq_bins)
        # Let frequency bins be evenly divided by 2, e.g., 257 -> 256
        x = x[..., 0 : x.shape[-1] - 1]  # (bs, input_channels, T, F)

        mic_direction_emb = F.pad(mic_direction_emb, pad=(0, 0, 0, pad_len))
        mic_pos_xyz_emb = F.pad(mic_pos_xyz_emb, pad=(0, 0, 0, pad_len))
        ray_direction_emb = F.pad(ray_direction_emb, pad=(0, 0, 0, pad_len))
        ray_pos_xyz_emb = F.pad(ray_pos_xyz_emb, pad=(0, 0, 0, pad_len))

        # input value FC
        enc1_pool, enc1 = self.input_value_cnn1(x)
        enc2_pool, enc2 = self.input_value_cnn2(enc1_pool)
        enc3_pool, enc3 = self.input_value_cnn3(enc2_pool)
        # a1 = self.input_value_cnn4(a1)
        a1 = enc3_pool

        a1 = a1.permute(0, 2, 1, 3)
        a1 = a1.flatten(2)

        a1 = F.leaky_relu_(self.input_value_fc1(a1), negative_slope=0.01)

        # input angle FC
        a2 = mic_direction_emb.permute(0, 2, 1, 3).flatten(2)
        a2 = F.leaky_relu_(self.input_angle_fc(a2))    # (batch_size, 1, T', C)

        a2_xyz = mic_pos_xyz_emb.permute(0, 2, 1, 3).flatten(2)
        a2_xyz = F.leaky_relu_(self.input_pos_xyz_fc(a2_xyz))

        a2 = a2[:, 0 :: self.time_downsample_ratio, :]
        a2_xyz = a2_xyz[:, 0 :: self.time_downsample_ratio, :]

        x = torch.cat((a1, a2, a2_xyz), dim=-1) # (batch_size, T, C * 2)

        rays_num = ray_direction.shape[1]

        x = torch.tile(x[:, None, :, :], (1, rays_num, 1, 1))
        # (batch_size, outputs_num, T, C * 2)

        # output angle FC
        a3 = F.leaky_relu_(self.output_angle_fc(ray_direction_emb))
        a3 = a3[:, :, 0 :: self.time_downsample_ratio, :]
        # a3 = torch.tile(a3[:, :, None, :], (1, 1, frames_num, 1))
        a3_xyz = F.leaky_relu_(self.output_pos_xyz_fc(ray_pos_xyz_emb))
        a3_xyz = a3_xyz[:, :, 0 :: self.time_downsample_ratio, :]
        # a3_xyz = torch.tile(a3_xyz[:, :, None, :], (1, 1, frames_num, 1))
        # from IPython import embed; embed(using=False); os._exit(0)
        inter_emb = torch.cat((x, a3, a3_xyz), dim=-1)
        # (batch_size, outputs_num, T, C * 3)
        
        batch_size, rays_num, _T, _C = inter_emb.shape

        x = inter_emb.reshape(batch_size * rays_num, _T, _C)
        # (batch_size * outputs_num, T, C * 3)

        x = x.transpose(1, 2)[:, :, :, None]
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x[:, :, :, 0].transpose(1, 2)

        x = x.reshape(batch_size, rays_num, _T, x.shape[-1])
        # (batch_size, outputs_num, T, C * 2)

        output_dict = {}

        if True:
            # tracking_output = torch.sigmoid(self.fc_final_tracking(x)).flatten(2)
            # output_dict['ray_intersect_source'] = torch.mean(tracking_output, dim=-1)
            cla_output = torch.sigmoid(self.fc_final_cla(x))
            # cla_output = cla_output.repeat(1, 1, self.time_downsample_ratio, 1)
            # from IPython import embed; embed(using=False); os._exit(0)

            _bs, _rays_num, _T, _C = cla_output.shape
            tmp = cla_output.reshape(_bs * _rays_num, _T, _C)
            tmp = interpolate(tmp, self.time_downsample_ratio)
            cla_output = tmp.reshape(_bs, _rays_num, -1, _C)

            cla_output = cla_output[:, :, 0 : origin_len, :].flatten(2)
            output_dict['agent_see_source'] = cla_output

        # if do_sep:
        if False:
            max_rays_contain_waveform = data_dict['max_rays_contain_waveform']
            # from IPython import embed; embed(using=False); os._exit(0)

            x = F.leaky_relu_(self.fc_inter(inter_emb[:, 0 : max_rays_contain_waveform, :, :]))

            batch_size, sep_rays_num, _T, _C = x.shape
            x = x.reshape(batch_size * sep_rays_num, _T, 128, 64)
            x = x.permute(0, 2, 1, 3)

            enc1 = self.tile_omni(x=enc1, rays_num=max_rays_contain_waveform)
            enc2 = self.tile_omni(x=enc2, rays_num=max_rays_contain_waveform)
            enc3 = self.tile_omni(x=enc3, rays_num=max_rays_contain_waveform)

            # from IPython import embed; embed(using=False); os._exit(0)
            x = self.dec_block1(x, enc3)
            x = self.dec_block2(x, enc2)
            x = self.dec_block3(x, enc1)
            x = self.dec_final(x)

            # Recover shape
            x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257. 

            x = x[:, :, 0:origin_len, :]

            audio_length = omni_waveform.shape[2]

            # from IPython import embed; embed(using=False); os._exit(0)
            # import matplotlib.pyplot as plt
            # plt.matshow(np.log(omni_mag.data.cpu().numpy()[0, 0]).T, origin='lower', aspect='auto', cmap='jet')
            # plt.savefig('_zz.pdf')
            # import soundfile
            # soundfile.write(file='_zz.wav', data=omni_waveform.data.cpu().numpy()[0, 0], samplerate=24000)

            omni_mag0 = self.tile_omni(omni_mag, max_rays_contain_waveform)
            omni_cos0 = self.tile_omni(omni_cos, max_rays_contain_waveform)
            omni_sin0 = self.tile_omni(omni_sin, max_rays_contain_waveform)

            separated_audio = self.feature_maps_to_wav(
                input_tensor=x,
                sp=omni_mag0,
                sin_in=omni_sin0,
                cos_in=omni_cos0,
                audio_length=audio_length,
            )

            separated_audio = separated_audio.reshape(batch_size, sep_rays_num, -1)
            # from IPython import embed; embed(using=False); os._exit(0)

            output_dict['ray_waveform'] = separated_audio

        return output_dict

    def tile_omni(self, x, rays_num): 
        batch_size, _, _T, _F = x.shape
        return torch.tile(x[:, None, :, :, :], (1, rays_num, 1, 1, 1)).reshape(batch_size * rays_num, -1, _T, _F)


class Model01_bak2(nn.Module, Base):
    def __init__(self, microphones_num, classes_num, do_localization=None, do_sed=None, do_separation=None):
        super(Model01_bak2, self).__init__() 

        window_size = 512
        hop_size = 240
        center = True
        pad_mode = "reflect"
        window = "hann" 
        activation = "relu"
        momentum = 0.01
        classes_num = 1

        self.window_size = window_size
        self.hop_size = hop_size
        self.pad_mode = pad_mode

        positional_embedding_factor = 5
        self.time_downsample_ratio = 2 ** 3

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

        # self.pre_value_cnn = nn.Conv2d(in_channels=microphones_num + 1, out_channels=32,
        #                       kernel_size=(1, 1), stride=(1, 1),
        #                       padding=(0, 0), bias=True)
        # self.pre_angle_cnn = nn.Conv2d(in_channels=12, out_channels=32,
        #                       kernel_size=(1, 1), stride=(1, 1),
        #                       padding=(0, 0), bias=True)

        self.mag_fc = nn.Conv2d(
            in_channels=microphones_num + 1,
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
            bias=True
        )

        self.phase_fc = nn.Conv2d(
            in_channels=microphones_num * (microphones_num - 1), 
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
            bias=True
        )

        self.input_value_cnn1 = EncoderBlockRes1B(in_channels=64, out_channels=32, kernel_size=(3, 3), downsample=(2, 2), momentum=momentum)

        self.input_value_cnn2 = EncoderBlockRes1B(in_channels=32, out_channels=64, kernel_size=(3, 3), downsample=(2, 2), momentum=momentum)

        self.input_value_cnn3 = EncoderBlockRes1B(in_channels=64, out_channels=128, kernel_size=(3, 3), downsample=(2, 2), momentum=momentum)

        # self.input_value_cnn4 = EncoderBlockRes1B(in_channels=128, out_channels=256, kernel_size=(3, 3), downsample=(2, 2), momentum=momentum)

        self.input_value_fc1 = nn.Linear(in_features=32 * 128, out_features=1024)

        self.input_angle_fc = nn.Linear(in_features=(positional_embedding_factor * 4) * microphones_num, out_features=128, bias=True)
        self.input_pos_xyz_fc = nn.Linear(in_features=(positional_embedding_factor * 6) * microphones_num, out_features=128, bias=True)

        self.output_angle_fc = nn.Linear(in_features=(positional_embedding_factor * 4), out_features=128, bias=True)
        self.output_pos_xyz_fc = nn.Linear(in_features=(positional_embedding_factor * 6), out_features=128, bias=True)

        self.conv_block1 = ConvBlockAfter(in_channels=1536, out_channels=1024)
        self.conv_block2 = ConvBlockAfter(in_channels=1024, out_channels=1024)

        # self.fc_final_tracking = nn.Linear(1024, 1, bias=True)
        self.fc_final_cla = nn.Linear(1024, classes_num, bias=True)

        # Sep
        self.fc_inter = nn.Linear(in_features=1536, out_features=128 * 32, bias=True)

        self.dec_block1 = DecoderBlockRes1B(in_channels=128, out_channels=128, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        self.dec_block2 = DecoderBlockRes1B(in_channels=128, out_channels=64, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        self.dec_block3 = DecoderBlockRes1B(in_channels=64, out_channels=32, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        self.dec_final = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.init_weights()

        self.target_sources_num = 1
        self.output_channels = 1
        self.K = 3

    def init_weights(self):

        init_layer(self.mag_fc)
        init_layer(self.phase_fc)
        init_layer(self.input_value_fc1)

        init_layer(self.input_angle_fc)
        init_layer(self.input_pos_xyz_fc)
        init_layer(self.output_angle_fc)
        init_layer(self.output_pos_xyz_fc)

        init_layer(self.fc_inter)
        # init_layer(self.fc_final_tracking)
        init_layer(self.fc_final_cla)

        init_layer(self.dec_final)

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
            input_tensor: (batch_size, target_sources_num * output_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, input_channels, time_steps, freq_bins)
            sin_in: (batch_size, input_channels, time_steps, freq_bins)
            cos_in: (batch_size, input_channels, time_steps, freq_bins)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (batch_size, target_sources_num * output_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.output_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, output_channels, self.K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        # linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # mask_cos, mask_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)
        # out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        # out_mag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size * target_sources_num * output_channels
        shape = (
            batch_size * self.target_sources_num * self.output_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * output_channels, segments_num)

        # Reshape.
        waveform = x.reshape(
            batch_size, self.target_sources_num * self.output_channels, audio_length
        )
        # (batch_size, target_sources_num * output_channels, segments_num)

        return waveform

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

    def convert_position_to_embedding(self, position):
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

    def forward(self, data_dict, do_sep=True):

        self.eps = 1e-10

        mic_position = data_dict['mic_position']    # (bs, mics_num, frames_num, 3)
        mic_look_direction = data_dict['mic_look_direction']    # (bs, mics_num, frames_num, 3)
        mic_waveform = data_dict['mic_waveform']    # (bs, mics_num, samples_num)
        agent_position = data_dict['agent_position']    # (bs, agents_num, frames_num, 3)
        agent_look_direction = data_dict['agent_look_direction']    # (bs, agents_num, frames_num, 3)

        # ------ 1. Convert positions and look directions to embeddings ------
        mic_position_emb = self.convert_position_to_embedding(position=mic_position)
        # shape: (bs, mics_num, frames_num, emb_size)

        mic_look_direction_emb = self.convert_look_direction_to_embedding(
            look_direction=mic_look_direction,
        )   # shape: (bs, mics_num, frames_num, emb_size)
        
        agent_position_emb = self.convert_position_to_embedding(position=agent_position)
        # shape: (bs, agents_num, frames_num, emb_size)

        agent_look_direction_emb = self.convert_look_direction_to_embedding(
            look_direction=agent_look_direction,
        )   # shape: (bs, agents_num, frames_num, emb_size)

        # ------ 2. Extract features ------
        total_waveform = torch.sum(mic_waveform, dim=1, keepdim=True)
        # (bs, 1, samples_num)

        total_mag, total_cos, total_sin = self.wav_to_spectrogram_phase(total_waveform, self.eps)
        # (bs, 1, frames_num, freq_bins)

        mic_mag, mic_cos, mic_sin = self.wav_to_spectrogram_phase(mic_waveform, self.eps)
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
        mic_position_emb = F.pad(mic_position_emb, pad=(0, 0, 0, pad_len))
        mic_look_direction_emb = F.pad(mic_look_direction_emb, pad=(0, 0, 0, pad_len))
        agent_position_emb = F.pad(agent_position_emb, pad=(0, 0, 0, pad_len))
        agent_look_direction_emb = F.pad(agent_look_direction_emb, pad=(0, 0, 0, pad_len))

        mic_pos_xyz_emb = mic_position_emb
        mic_direction_emb = mic_look_direction_emb
        ray_pos_xyz_emb = agent_position_emb
        ray_direction_emb = agent_look_direction_emb

        ray_direction = agent_look_direction
        origin_len = frames_num
        
        # input value FC
        enc1_pool, enc1 = self.input_value_cnn1(x)
        enc2_pool, enc2 = self.input_value_cnn2(enc1_pool)
        enc3_pool, enc3 = self.input_value_cnn3(enc2_pool)
        # a1 = self.input_value_cnn4(a1)
        a1 = enc3_pool

        a1 = a1.permute(0, 2, 1, 3)
        a1 = a1.flatten(2)

        a1 = F.leaky_relu_(self.input_value_fc1(a1), negative_slope=0.01)

        # input angle FC
        a2 = mic_direction_emb.permute(0, 2, 1, 3).flatten(2)
        a2 = F.leaky_relu_(self.input_angle_fc(a2))    # (batch_size, 1, T', C)

        a2_xyz = mic_pos_xyz_emb.permute(0, 2, 1, 3).flatten(2)
        a2_xyz = F.leaky_relu_(self.input_pos_xyz_fc(a2_xyz))

        a2 = a2[:, 0 :: self.time_downsample_ratio, :]
        a2_xyz = a2_xyz[:, 0 :: self.time_downsample_ratio, :]

        x = torch.cat((a1, a2, a2_xyz), dim=-1) # (batch_size, T, C * 2)

        rays_num = ray_direction.shape[1]

        x = torch.tile(x[:, None, :, :], (1, rays_num, 1, 1))
        # (batch_size, outputs_num, T, C * 2)

        # output angle FC
        a3 = F.leaky_relu_(self.output_angle_fc(ray_direction_emb))
        a3 = a3[:, :, 0 :: self.time_downsample_ratio, :]
        # a3 = torch.tile(a3[:, :, None, :], (1, 1, frames_num, 1))
        a3_xyz = F.leaky_relu_(self.output_pos_xyz_fc(ray_pos_xyz_emb))
        a3_xyz = a3_xyz[:, :, 0 :: self.time_downsample_ratio, :]
        # a3_xyz = torch.tile(a3_xyz[:, :, None, :], (1, 1, frames_num, 1))
        # from IPython import embed; embed(using=False); os._exit(0)
        inter_emb = torch.cat((x, a3, a3_xyz), dim=-1)
        # (batch_size, outputs_num, T, C * 3)
        
        batch_size, rays_num, _T, _C = inter_emb.shape

        x = inter_emb.reshape(batch_size * rays_num, _T, _C)
        # (batch_size * outputs_num, T, C * 3)

        x = x.transpose(1, 2)[:, :, :, None]
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x[:, :, :, 0].transpose(1, 2)

        x = x.reshape(batch_size, rays_num, _T, x.shape[-1])
        # (batch_size, outputs_num, T, C * 2)

        output_dict = {}

        if True:
            # tracking_output = torch.sigmoid(self.fc_final_tracking(x)).flatten(2)
            # output_dict['ray_intersect_source'] = torch.mean(tracking_output, dim=-1)
            cla_output = torch.sigmoid(self.fc_final_cla(x))
            # cla_output = cla_output.repeat(1, 1, self.time_downsample_ratio, 1)
            # from IPython import embed; embed(using=False); os._exit(0)

            _bs, _rays_num, _T, _C = cla_output.shape
            tmp = cla_output.reshape(_bs * _rays_num, _T, _C)
            tmp = interpolate(tmp, self.time_downsample_ratio)
            cla_output = tmp.reshape(_bs, _rays_num, -1, _C)

            cla_output = cla_output[:, :, 0 : origin_len, :].flatten(2)
            output_dict['agent_see_source'] = cla_output

        # if do_sep:
        if False:
            max_rays_contain_waveform = data_dict['max_rays_contain_waveform']
            # from IPython import embed; embed(using=False); os._exit(0)

            x = F.leaky_relu_(self.fc_inter(inter_emb[:, 0 : max_rays_contain_waveform, :, :]))

            batch_size, sep_rays_num, _T, _C = x.shape
            x = x.reshape(batch_size * sep_rays_num, _T, 128, 64)
            x = x.permute(0, 2, 1, 3)

            enc1 = self.tile_omni(x=enc1, rays_num=max_rays_contain_waveform)
            enc2 = self.tile_omni(x=enc2, rays_num=max_rays_contain_waveform)
            enc3 = self.tile_omni(x=enc3, rays_num=max_rays_contain_waveform)

            # from IPython import embed; embed(using=False); os._exit(0)
            x = self.dec_block1(x, enc3)
            x = self.dec_block2(x, enc2)
            x = self.dec_block3(x, enc1)
            x = self.dec_final(x)

            # Recover shape
            x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257. 

            x = x[:, :, 0:origin_len, :]

            audio_length = omni_waveform.shape[2]

            # from IPython import embed; embed(using=False); os._exit(0)
            # import matplotlib.pyplot as plt
            # plt.matshow(np.log(omni_mag.data.cpu().numpy()[0, 0]).T, origin='lower', aspect='auto', cmap='jet')
            # plt.savefig('_zz.pdf')
            # import soundfile
            # soundfile.write(file='_zz.wav', data=omni_waveform.data.cpu().numpy()[0, 0], samplerate=24000)

            omni_mag0 = self.tile_omni(omni_mag, max_rays_contain_waveform)
            omni_cos0 = self.tile_omni(omni_cos, max_rays_contain_waveform)
            omni_sin0 = self.tile_omni(omni_sin, max_rays_contain_waveform)

            separated_audio = self.feature_maps_to_wav(
                input_tensor=x,
                sp=omni_mag0,
                sin_in=omni_sin0,
                cos_in=omni_cos0,
                audio_length=audio_length,
            )

            separated_audio = separated_audio.reshape(batch_size, sep_rays_num, -1)
            # from IPython import embed; embed(using=False); os._exit(0)

            output_dict['ray_waveform'] = separated_audio

        return output_dict

    def tile_omni(self, x, rays_num): 
        batch_size, _, _T, _F = x.shape
        return torch.tile(x[:, None, :, :, :], (1, rays_num, 1, 1, 1)).reshape(batch_size * rays_num, -1, _T, _F)


class Model01_bak3(nn.Module, Base):
    def __init__(self, microphones_num, classes_num, do_localization=None, do_sed=None, do_separation=None):
        super(Model01_bak3, self).__init__() 

        window_size = 512
        hop_size = 240
        center = True
        pad_mode = "reflect"
        window = "hann" 
        activation = "relu"
        momentum = 0.01
        classes_num = 1

        self.window_size = window_size
        self.hop_size = hop_size
        self.pad_mode = pad_mode

        positional_embedding_factor = 5
        self.time_downsample_ratio = 2 ** 3

        self.do_localization = True
        self.do_sed = False
        self.do_separation = False

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
            in_channels=microphones_num + 1,
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
            bias=True
        )

        self.phase_fc = nn.Conv2d(
            in_channels=microphones_num * (microphones_num - 1), 
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
            in_features=microphones_num * (positional_embedding_factor * 6), 
            out_features=128, 
            bias=True
        )

        self.mic_look_direction_fc = nn.Linear(
            in_features=microphones_num * (positional_embedding_factor * 4), 
            out_features=128, 
            bias=True
        )

        self.output_angle_fc = nn.Linear(in_features=(positional_embedding_factor * 4), out_features=128, bias=True)
        self.output_pos_xyz_fc = nn.Linear(in_features=(positional_embedding_factor * 6), out_features=128, bias=True)

        self.conv_block1 = ConvBlockAfter(in_channels=1536, out_channels=1024)
        self.conv_block2 = ConvBlockAfter(in_channels=1024, out_channels=1024)

        # self.fc_final_tracking = nn.Linear(1024, 1, bias=True)
        self.fc_final_cla = nn.Linear(1024, classes_num, bias=True)

        # Sep
        self.fc_inter = nn.Linear(in_features=1536, out_features=128 * 32, bias=True)

        self.dec_block1 = DecoderBlockRes1B(in_channels=128, out_channels=128, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        self.dec_block2 = DecoderBlockRes1B(in_channels=128, out_channels=64, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        self.dec_block3 = DecoderBlockRes1B(in_channels=64, out_channels=32, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        self.dec_final = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        # self.agent_position_fc = nn.Linear(
        #     in_features=(positional_embedding_factor * 6), 
        #     out_features=128, 
        #     bias=True
        # )

        # self.agent_look_direction_fc = nn.Linear(
        #     in_features=(positional_embedding_factor * 4), 
        #     out_features=128, 
        #     bias=True
        # )

        # if self.do_localization:
        #     self.loc_fc_block1 = ConvBlock(
        #         in_channels=1536, 
        #         out_channels=1024, 
        #         kernel_size=(1, 1)
        #     )

        #     self.loc_fc_block2 = ConvBlock(
        #         in_channels=1024, 
        #         out_channels=1024,
        #         kernel_size=(1, 1),
        #     )

        #     self.loc_fc_final = nn.Linear(1024, 1, bias=True)

        # if self.do_separation:
        #     self.sep_fc_reshape = nn.Linear(
        #         in_features=1536, 
        #         out_features=4096, 
        #         bias=True,
        #     )

        #     self.sep_decoder_block1 = DecoderBlockRes1B(
        #         in_channels=128, 
        #         out_channels=128, 
        #         kernel_size=(3, 3), 
        #         upsample=(2, 2), 
        #         momentum=momentum,
        #     )

        #     self.sep_decoder_block2 = DecoderBlockRes1B(
        #         in_channels=128, 
        #         out_channels=64, 
        #         kernel_size=(3, 3), 
        #         upsample=(2, 2), 
        #         momentum=momentum,
        #     )

        #     self.sep_decoder_block3 = DecoderBlockRes1B(
        #         in_channels=64, 
        #         out_channels=32, 
        #         kernel_size=(3, 3), 
        #         upsample=(2, 2), 
        #         momentum=momentum,
        #     )

        #     self.sep_conv_final = nn.Conv2d(
        #         in_channels=32, 
        #         out_channels=3, 
        #         kernel_size=(1, 1), 
        #         stride=(1, 1), 
        #         padding=(0, 0), 
        #         bias=True
        #     )

        self.init_weights()

    def init_weights(self):

        init_layer(self.mag_fc)
        init_layer(self.phase_fc)
        init_layer(self.mic_signal_fc_reshape)

        init_layer(self.mic_position_fc)
        init_layer(self.mic_look_direction_fc)

        init_layer(self.output_angle_fc)
        init_layer(self.output_pos_xyz_fc)

        init_layer(self.fc_inter)
        # init_layer(self.fc_final_tracking)
        init_layer(self.fc_final_cla)

        init_layer(self.dec_final)

        # init_layer(self.agent_position_fc)
        # init_layer(self.agent_look_direction_fc)

        # if self.do_localization:
        #     init_layer(self.loc_fc_final)

        # if self.do_separation:
        #     init_layer(self.sep_fc_reshape)
        #     init_layer(self.sep_conv_final)

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
            input_tensor: (batch_size, target_sources_num * output_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, input_channels, time_steps, freq_bins)
            sin_in: (batch_size, input_channels, time_steps, freq_bins)
            cos_in: (batch_size, input_channels, time_steps, freq_bins)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (batch_size, target_sources_num * output_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.output_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, output_channels, self.K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        # linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # mask_cos, mask_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)
        # out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        # out_mag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size * target_sources_num * output_channels
        shape = (
            batch_size * self.target_sources_num * self.output_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * output_channels, segments_num)

        # Reshape.
        waveform = x.reshape(
            batch_size, self.target_sources_num * self.output_channels, audio_length
        )
        # (batch_size, target_sources_num * output_channels, segments_num)

        return waveform

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

    def convert_position_to_embedding(self, position):
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

    def forward(self, data_dict, do_sep=True):

        self.eps = 1e-10

        mic_position = data_dict['mic_position']    # (bs, mics_num, frames_num, 3)
        mic_look_direction = data_dict['mic_look_direction']    # (bs, mics_num, frames_num, 3)
        mic_waveform = data_dict['mic_waveform']    # (bs, mics_num, samples_num)
        agent_position = data_dict['agent_position']    # (bs, agents_num, frames_num, 3)
        agent_look_direction = data_dict['agent_look_direction']    # (bs, agents_num, frames_num, 3)

        # ------ 1. Convert positions and look directions to embeddings ------
        mic_position_emb = self.convert_position_to_embedding(position=mic_position)
        # shape: (bs, mics_num, frames_num, emb_size)

        mic_look_direction_emb = self.convert_look_direction_to_embedding(
            look_direction=mic_look_direction,
        )   # shape: (bs, mics_num, frames_num, emb_size)
        
        agent_position_emb = self.convert_position_to_embedding(position=agent_position)
        # shape: (bs, agents_num, frames_num, emb_size)

        agent_look_direction_emb = self.convert_look_direction_to_embedding(
            look_direction=agent_look_direction,
        )   # shape: (bs, agents_num, frames_num, emb_size)

        # ------ 2. Extract features ------
        total_waveform = torch.sum(mic_waveform, dim=1, keepdim=True)
        # (bs, 1, samples_num)

        total_mag, total_cos, total_sin = self.wav_to_spectrogram_phase(total_waveform, self.eps)
        # (bs, 1, frames_num, freq_bins)

        mic_mag, mic_cos, mic_sin = self.wav_to_spectrogram_phase(mic_waveform, self.eps)
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
        mic_position_emb = F.pad(mic_position_emb, pad=(0, 0, 0, pad_len))
        mic_look_direction_emb = F.pad(mic_look_direction_emb, pad=(0, 0, 0, pad_len))
        agent_position_emb = F.pad(agent_position_emb, pad=(0, 0, 0, pad_len))
        agent_look_direction_emb = F.pad(agent_look_direction_emb, pad=(0, 0, 0, pad_len))

        # ------ 3. Forward data to the network ------

        # 1) Transform mic signal.
        enc1_pool, enc1 = self.mic_signal_encoder_block1(x)
        enc2_pool, enc2 = self.mic_signal_encoder_block2(enc1_pool)
        enc3_pool, enc3 = self.mic_signal_encoder_block3(enc2_pool)
        # enc3_pool: (bs, 128, T=38, F=32)

        mic_signal_feature = rearrange(enc3_pool, 'b c t f -> b t (c f)')
        # (bs, T, C*F=4096)

        mic_signal_feature = F.leaky_relu_(
            self.mic_signal_fc_reshape(mic_signal_feature), negative_slope=0.01,
        )  # shape: (bs, T, 1024)
        
        # 2) Transform mic position and look direction embeddings.
        mic_position_feature = rearrange(mic_position_emb, 'b c t f -> b t (c f)')
        # shape: (bs, T, mics_num*emb_size=120)

        mic_position_feature = F.leaky_relu_(
            self.mic_position_fc(mic_position_feature)
        )  # shape: (bs, T, 128)

        mic_position_feature = mic_position_feature[:, 0 :: self.time_downsample_ratio, :]
        
        mic_look_direction_feature = rearrange(mic_look_direction_emb, 'b c t f -> b t (c f)')
        # shape: (bs, T, mics_num*emb_size=80)

        mic_look_direction_feature = F.leaky_relu_(
            self.mic_look_direction_fc(mic_look_direction_feature)
        )  # shape: (bs, T, 128)

        mic_look_direction_feature = mic_look_direction_feature[:, 0 :: self.time_downsample_ratio, :]

        # 3) Concatenate mic signal, position, and look direction features.
        mic_feature = torch.cat((
            mic_signal_feature, mic_position_feature, mic_look_direction_feature
        ), dim=-1)  # (batch_size, T, 1280)

        agents_num = agent_position.shape[1]

        mic_feature = torch.tile(mic_feature[:, None, :, :], (1, agents_num, 1, 1))
        # shape: (bs, agents_num, T=38, 1280)

        x = mic_feature
        ray_direction_emb = agent_look_direction_emb
        ray_pos_xyz_emb = agent_position_emb
        origin_len = frames_num

        # output angle FC
        a3 = F.leaky_relu_(self.output_angle_fc(ray_direction_emb))
        a3 = a3[:, :, 0 :: self.time_downsample_ratio, :]
        # a3 = torch.tile(a3[:, :, None, :], (1, 1, frames_num, 1))
        a3_xyz = F.leaky_relu_(self.output_pos_xyz_fc(ray_pos_xyz_emb))
        a3_xyz = a3_xyz[:, :, 0 :: self.time_downsample_ratio, :]
        # a3_xyz = torch.tile(a3_xyz[:, :, None, :], (1, 1, frames_num, 1))
        # from IPython import embed; embed(using=False); os._exit(0)
        inter_emb = torch.cat((x, a3, a3_xyz), dim=-1)
        # (batch_size, outputs_num, T, C * 3)
        
        batch_size, rays_num, _T, _C = inter_emb.shape

        x = inter_emb.reshape(batch_size * rays_num, _T, _C)
        # (batch_size * outputs_num, T, C * 3)

        x = x.transpose(1, 2)[:, :, :, None]
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x[:, :, :, 0].transpose(1, 2)

        x = x.reshape(batch_size, rays_num, _T, x.shape[-1])
        # (batch_size, outputs_num, T, C * 2)

        output_dict = {}

        if True:
            # tracking_output = torch.sigmoid(self.fc_final_tracking(x)).flatten(2)
            # output_dict['ray_intersect_source'] = torch.mean(tracking_output, dim=-1)
            cla_output = torch.sigmoid(self.fc_final_cla(x))
            # cla_output = cla_output.repeat(1, 1, self.time_downsample_ratio, 1)
            # from IPython import embed; embed(using=False); os._exit(0)

            _bs, _rays_num, _T, _C = cla_output.shape
            tmp = cla_output.reshape(_bs * _rays_num, _T, _C)
            tmp = interpolate(tmp, self.time_downsample_ratio)
            cla_output = tmp.reshape(_bs, _rays_num, -1, _C)

            cla_output = cla_output[:, :, 0 : origin_len, :].flatten(2)
            output_dict['agent_see_source'] = cla_output

        # if do_sep:
        if False:
            max_rays_contain_waveform = data_dict['max_rays_contain_waveform']
            # from IPython import embed; embed(using=False); os._exit(0)

            x = F.leaky_relu_(self.fc_inter(inter_emb[:, 0 : max_rays_contain_waveform, :, :]))

            batch_size, sep_rays_num, _T, _C = x.shape
            x = x.reshape(batch_size * sep_rays_num, _T, 128, 64)
            x = x.permute(0, 2, 1, 3)

            enc1 = self.tile_omni(x=enc1, rays_num=max_rays_contain_waveform)
            enc2 = self.tile_omni(x=enc2, rays_num=max_rays_contain_waveform)
            enc3 = self.tile_omni(x=enc3, rays_num=max_rays_contain_waveform)

            # from IPython import embed; embed(using=False); os._exit(0)
            x = self.dec_block1(x, enc3)
            x = self.dec_block2(x, enc2)
            x = self.dec_block3(x, enc1)
            x = self.dec_final(x)

            # Recover shape
            x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257. 

            x = x[:, :, 0:origin_len, :]

            audio_length = omni_waveform.shape[2]

            # from IPython import embed; embed(using=False); os._exit(0)
            # import matplotlib.pyplot as plt
            # plt.matshow(np.log(omni_mag.data.cpu().numpy()[0, 0]).T, origin='lower', aspect='auto', cmap='jet')
            # plt.savefig('_zz.pdf')
            # import soundfile
            # soundfile.write(file='_zz.wav', data=omni_waveform.data.cpu().numpy()[0, 0], samplerate=24000)

            omni_mag0 = self.tile_omni(omni_mag, max_rays_contain_waveform)
            omni_cos0 = self.tile_omni(omni_cos, max_rays_contain_waveform)
            omni_sin0 = self.tile_omni(omni_sin, max_rays_contain_waveform)

            separated_audio = self.feature_maps_to_wav(
                input_tensor=x,
                sp=omni_mag0,
                sin_in=omni_sin0,
                cos_in=omni_cos0,
                audio_length=audio_length,
            )

            separated_audio = separated_audio.reshape(batch_size, sep_rays_num, -1)
            # from IPython import embed; embed(using=False); os._exit(0)

            output_dict['ray_waveform'] = separated_audio

        return output_dict

        # # 4) Calculate agent position and look direction features.
        # agent_position_feature = F.leaky_relu_(self.agent_position_fc(agent_position_emb))
        # # shape: (bs, agents_num, T, 128)

        # agent_position_feature = agent_position_feature[:, :, 0 :: self.time_downsample_ratio, :]
        # # shape: (bs, agents_num, T=38, 128)

        # agent_look_direction_feature = F.leaky_relu_(
        #     self.agent_look_direction_fc(agent_look_direction_emb)
        # )  # (bs, agents_num, T, 128)

        # agent_look_direction_feature = agent_look_direction_feature[:, :, 0 :: self.time_downsample_ratio, :]
        # # shape: (bs, agents_num, T=38, 128)

        # # Concatenate mic features and agent features.
        # shared_feature = torch.cat((mic_feature, agent_position_feature, 
        #     agent_look_direction_feature), dim=-1)
        # # (bs, agents_num, T=38, 1536)

        # output_dict = {}

        # if self.do_localization:
        #     batch_size, agents_num, _T, _C = shared_feature.shape

        #     x = rearrange(shared_feature, 'b n t c -> (b n) c t 1')
        #     # (bs * agents_num, C=1536, T, 1)

        #     x = self.loc_fc_block1(x)
        #     # (bs * agents_num, 1024, T, 1)

        #     x = self.loc_fc_block2(x)
        #     # (bs * agents_num, 1024, T, 1)

        #     x = rearrange(x, '(b n) c t 1 -> (b n) t c', b=batch_size)
        #     # (bs * agents_num, T, 1024)

        #     x = torch.sigmoid(self.loc_fc_final(x))
        #     # (bs * agents_num, T=38, C=1)

        #     x = interpolate(x, self.time_downsample_ratio)[:, 0 : frames_num, :]
        #     # (bs * agents_num, T=301, C=1)

        #     agent_see_source = rearrange(x, '(b n) t 1 -> b n t', b=batch_size)
        #     # (bs, agents_num, T=301)

        #     output_dict['agent_see_source'] = agent_see_source

        # if self.do_separation:

        #     max_agents_contain_waveform = data_dict['max_agents_contain_waveform']

        #     x = shared_feature[:, 0 : max_agents_contain_waveform, :, :]
        #     # (bs, n=max_agents_contain_waveform, T=38, C=1536)

        #     x = F.leaky_relu_(self.sep_fc_reshape(x))
        #     # (bs, n, T=38, 4096)

        #     x = rearrange(x, 'b n t (c f) -> (b n) c t f', c=128, f=32)
        #     # (bs * n, C=128, T=38, F=32)

        #     enc1 = self.repeat_conv_features(x=enc1, repeats_num=max_agents_contain_waveform)
        #     # (bs * n, C=32, T=304, F=256)

        #     enc2 = self.repeat_conv_features(x=enc2, repeats_num=max_agents_contain_waveform)
        #     # (bs * n, C=64, T=152, F=128)

        #     enc3 = self.repeat_conv_features(x=enc3, repeats_num=max_agents_contain_waveform)
        #     # (bs * n, C=128, T=76, F=64)

        #     x = self.sep_decoder_block1(x, enc3)
        #     x = self.sep_decoder_block2(x, enc2)
        #     x = self.sep_decoder_block3(x, enc1)
        #     # (bs * n, C=32, T=304, F=256)

        #     x = self.sep_conv_final(x)
        #     # (bs * n, C=3, T=304, F=256)

        #     # Recover shape
        #     x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257. 
        #     # (bs * n, C=3, T=304, F=257)

        #     x = x[:, :, 0:frames_num, :]
        #     # (bs * n, C=3, T=301, F=257)

        #     repeat_total_mag = self.repeat_conv_features(
        #         x=total_mag, repeats_num=max_agents_contain_waveform
        #     )  # (bs * n, C=1, T=301, F=257)

        #     repeat_total_sin = self.repeat_conv_features(
        #         x=total_sin, repeats_num=max_agents_contain_waveform
        #     )  # (bs * n, C=1, T=301, F=257)

        #     repeat_total_cos = self.repeat_conv_features(
        #         x=total_cos, repeats_num=max_agents_contain_waveform
        #     )  # (bs * n, C=1, T=301, F=257)

        #     audio_length = total_waveform.shape[-1]

        #     x = self.feature_maps_to_wav(
        #         input_tensor=x,
        #         sp=repeat_total_mag,
        #         sin_in=repeat_total_sin,
        #         cos_in=repeat_total_cos,
        #         audio_length=audio_length,
        #     )
        #     # (bs * n, segment_samples)

        #     agent_waveform = rearrange(x, '(b n) t -> b n t', b=batch_size)
        #     # (bs, n=max_agents_contain_waveform, segment_samples)

        #     output_dict['agent_waveform'] = agent_waveform

        # return output_dict


    def tile_omni(self, x, rays_num): 
        batch_size, _, _T, _F = x.shape
        return torch.tile(x[:, None, :, :, :], (1, rays_num, 1, 1, 1)).reshape(batch_size * rays_num, -1, _T, _F)


class Model01_bak4(nn.Module, Base):
    def __init__(self, microphones_num, classes_num, do_localization=None, do_sed=None, do_separation=None):
        super(Model01_bak4, self).__init__() 

        window_size = 512
        hop_size = 240
        center = True
        pad_mode = "reflect"
        window = "hann" 
        activation = "relu"
        momentum = 0.01
        classes_num = 1

        self.window_size = window_size
        self.hop_size = hop_size
        self.pad_mode = pad_mode

        positional_embedding_factor = 5
        self.time_downsample_ratio = 2 ** 3

        self.do_localization = True
        self.do_sed = False
        self.do_separation = False

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
            in_channels=microphones_num + 1,
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
            bias=True
        )

        self.phase_fc = nn.Conv2d(
            in_channels=microphones_num * (microphones_num - 1), 
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
            in_features=microphones_num * (positional_embedding_factor * 6), 
            out_features=128, 
            bias=True
        )

        self.mic_look_direction_fc = nn.Linear(
            in_features=microphones_num * (positional_embedding_factor * 4), 
            out_features=128, 
            bias=True
        )

        # self.output_angle_fc = nn.Linear(in_features=(positional_embedding_factor * 4), out_features=128, bias=True)
        # self.output_pos_xyz_fc = nn.Linear(in_features=(positional_embedding_factor * 6), out_features=128, bias=True)

        self.conv_block1 = ConvBlockAfter(in_channels=1536, out_channels=1024)
        self.conv_block2 = ConvBlockAfter(in_channels=1024, out_channels=1024)

        # # self.fc_final_tracking = nn.Linear(1024, 1, bias=True)
        self.fc_final_cla = nn.Linear(1024, classes_num, bias=True)

        # # Sep
        # self.fc_inter = nn.Linear(in_features=1536, out_features=128 * 32, bias=True)

        # self.dec_block1 = DecoderBlockRes1B(in_channels=128, out_channels=128, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        # self.dec_block2 = DecoderBlockRes1B(in_channels=128, out_channels=64, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        # self.dec_block3 = DecoderBlockRes1B(in_channels=64, out_channels=32, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        # self.dec_final = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

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

        if self.do_localization:
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

    def init_weights(self):

        init_layer(self.mag_fc)
        init_layer(self.phase_fc)
        init_layer(self.mic_signal_fc_reshape)

        init_layer(self.mic_position_fc)
        init_layer(self.mic_look_direction_fc)

        # init_layer(self.output_angle_fc)
        # init_layer(self.output_pos_xyz_fc)

        # init_layer(self.fc_inter)
        # # init_layer(self.fc_final_tracking)
        init_layer(self.fc_final_cla)

        # init_layer(self.dec_final)

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
            input_tensor: (batch_size, target_sources_num * output_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, input_channels, time_steps, freq_bins)
            sin_in: (batch_size, input_channels, time_steps, freq_bins)
            cos_in: (batch_size, input_channels, time_steps, freq_bins)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (batch_size, target_sources_num * output_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.output_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, output_channels, self.K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        # linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # mask_cos, mask_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)
        # out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        # out_mag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size * target_sources_num * output_channels
        shape = (
            batch_size * self.target_sources_num * self.output_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * output_channels, segments_num)

        # Reshape.
        waveform = x.reshape(
            batch_size, self.target_sources_num * self.output_channels, audio_length
        )
        # (batch_size, target_sources_num * output_channels, segments_num)

        return waveform

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

    def convert_position_to_embedding(self, position):
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

    def forward(self, data_dict, do_sep=True):

        self.eps = 1e-10

        mic_position = data_dict['mic_position']    # (bs, mics_num, frames_num, 3)
        mic_look_direction = data_dict['mic_look_direction']    # (bs, mics_num, frames_num, 3)
        mic_waveform = data_dict['mic_waveform']    # (bs, mics_num, samples_num)
        agent_position = data_dict['agent_position']    # (bs, agents_num, frames_num, 3)
        agent_look_direction = data_dict['agent_look_direction']    # (bs, agents_num, frames_num, 3)

        # ------ 1. Convert positions and look directions to embeddings ------
        mic_position_emb = self.convert_position_to_embedding(position=mic_position)
        # shape: (bs, mics_num, frames_num, emb_size)

        mic_look_direction_emb = self.convert_look_direction_to_embedding(
            look_direction=mic_look_direction,
        )   # shape: (bs, mics_num, frames_num, emb_size)
        
        agent_position_emb = self.convert_position_to_embedding(position=agent_position)
        # shape: (bs, agents_num, frames_num, emb_size)

        agent_look_direction_emb = self.convert_look_direction_to_embedding(
            look_direction=agent_look_direction,
        )   # shape: (bs, agents_num, frames_num, emb_size)

        # ------ 2. Extract features ------
        total_waveform = torch.sum(mic_waveform, dim=1, keepdim=True)
        # (bs, 1, samples_num)

        total_mag, total_cos, total_sin = self.wav_to_spectrogram_phase(total_waveform, self.eps)
        # (bs, 1, frames_num, freq_bins)

        mic_mag, mic_cos, mic_sin = self.wav_to_spectrogram_phase(mic_waveform, self.eps)
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
        mic_position_emb = F.pad(mic_position_emb, pad=(0, 0, 0, pad_len))
        mic_look_direction_emb = F.pad(mic_look_direction_emb, pad=(0, 0, 0, pad_len))
        agent_position_emb = F.pad(agent_position_emb, pad=(0, 0, 0, pad_len))
        agent_look_direction_emb = F.pad(agent_look_direction_emb, pad=(0, 0, 0, pad_len))

        # ------ 3. Forward data to the network ------

        # 1) Transform mic signal.
        enc1_pool, enc1 = self.mic_signal_encoder_block1(x)
        enc2_pool, enc2 = self.mic_signal_encoder_block2(enc1_pool)
        enc3_pool, enc3 = self.mic_signal_encoder_block3(enc2_pool)
        # enc3_pool: (bs, 128, T=38, F=32)

        mic_signal_feature = rearrange(enc3_pool, 'b c t f -> b t (c f)')
        # (bs, T, C*F=4096)

        mic_signal_feature = F.leaky_relu_(
            self.mic_signal_fc_reshape(mic_signal_feature), negative_slope=0.01,
        )  # shape: (bs, T, 1024)
        
        # 2) Transform mic position and look direction embeddings.
        mic_position_feature = rearrange(mic_position_emb, 'b c t f -> b t (c f)')
        # shape: (bs, T, mics_num*emb_size=120)

        mic_position_feature = F.leaky_relu_(
            self.mic_position_fc(mic_position_feature)
        )  # shape: (bs, T, 128)

        mic_position_feature = mic_position_feature[:, 0 :: self.time_downsample_ratio, :]
        
        mic_look_direction_feature = rearrange(mic_look_direction_emb, 'b c t f -> b t (c f)')
        # shape: (bs, T, mics_num*emb_size=80)

        mic_look_direction_feature = F.leaky_relu_(
            self.mic_look_direction_fc(mic_look_direction_feature)
        )  # shape: (bs, T, 128)

        mic_look_direction_feature = mic_look_direction_feature[:, 0 :: self.time_downsample_ratio, :]

        # 3) Concatenate mic signal, position, and look direction features.
        mic_feature = torch.cat((
            mic_signal_feature, mic_position_feature, mic_look_direction_feature
        ), dim=-1)  # (batch_size, T, 1280)

        agents_num = agent_position.shape[1]

        mic_feature = torch.tile(mic_feature[:, None, :, :], (1, agents_num, 1, 1))
        # shape: (bs, agents_num, T=38, 1280)

        # x = mic_feature
        # ray_direction_emb = agent_look_direction_emb
        # ray_pos_xyz_emb = agent_position_emb
        # origin_len = frames_num

        # # output angle FC
        # a3 = F.leaky_relu_(self.output_angle_fc(ray_direction_emb))
        # a3 = a3[:, :, 0 :: self.time_downsample_ratio, :]
        # # a3 = torch.tile(a3[:, :, None, :], (1, 1, frames_num, 1))
        # a3_xyz = F.leaky_relu_(self.output_pos_xyz_fc(ray_pos_xyz_emb))
        # a3_xyz = a3_xyz[:, :, 0 :: self.time_downsample_ratio, :]
        # # a3_xyz = torch.tile(a3_xyz[:, :, None, :], (1, 1, frames_num, 1))
        # # from IPython import embed; embed(using=False); os._exit(0)
        # inter_emb = torch.cat((x, a3, a3_xyz), dim=-1)
        # # (batch_size, outputs_num, T, C * 3)
        
        # batch_size, rays_num, _T, _C = inter_emb.shape

        # x = inter_emb.reshape(batch_size * rays_num, _T, _C)
        # # (batch_size * outputs_num, T, C * 3)

        # x = x.transpose(1, 2)[:, :, :, None]
        # x = self.conv_block1(x)
        # x = self.conv_block2(x)
        # x = x[:, :, :, 0].transpose(1, 2)

        # x = x.reshape(batch_size, rays_num, _T, x.shape[-1])
        # # (batch_size, outputs_num, T, C * 2)

        # output_dict = {}

        # if True:
        #     # tracking_output = torch.sigmoid(self.fc_final_tracking(x)).flatten(2)
        #     # output_dict['ray_intersect_source'] = torch.mean(tracking_output, dim=-1)
        #     cla_output = torch.sigmoid(self.fc_final_cla(x))
        #     # cla_output = cla_output.repeat(1, 1, self.time_downsample_ratio, 1)
        #     # from IPython import embed; embed(using=False); os._exit(0)

        #     _bs, _rays_num, _T, _C = cla_output.shape
        #     tmp = cla_output.reshape(_bs * _rays_num, _T, _C)
        #     tmp = interpolate(tmp, self.time_downsample_ratio)
        #     cla_output = tmp.reshape(_bs, _rays_num, -1, _C)

        #     cla_output = cla_output[:, :, 0 : origin_len, :].flatten(2)
        #     output_dict['agent_see_source'] = cla_output

        # # if do_sep:
        # if False:
        #     max_rays_contain_waveform = data_dict['max_rays_contain_waveform']
        #     # from IPython import embed; embed(using=False); os._exit(0)

        #     x = F.leaky_relu_(self.fc_inter(inter_emb[:, 0 : max_rays_contain_waveform, :, :]))

        #     batch_size, sep_rays_num, _T, _C = x.shape
        #     x = x.reshape(batch_size * sep_rays_num, _T, 128, 64)
        #     x = x.permute(0, 2, 1, 3)

        #     enc1 = self.tile_omni(x=enc1, rays_num=max_rays_contain_waveform)
        #     enc2 = self.tile_omni(x=enc2, rays_num=max_rays_contain_waveform)
        #     enc3 = self.tile_omni(x=enc3, rays_num=max_rays_contain_waveform)

        #     # from IPython import embed; embed(using=False); os._exit(0)
        #     x = self.dec_block1(x, enc3)
        #     x = self.dec_block2(x, enc2)
        #     x = self.dec_block3(x, enc1)
        #     x = self.dec_final(x)

        #     # Recover shape
        #     x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257. 

        #     x = x[:, :, 0:origin_len, :]

        #     audio_length = omni_waveform.shape[2]

        #     # from IPython import embed; embed(using=False); os._exit(0)
        #     # import matplotlib.pyplot as plt
        #     # plt.matshow(np.log(omni_mag.data.cpu().numpy()[0, 0]).T, origin='lower', aspect='auto', cmap='jet')
        #     # plt.savefig('_zz.pdf')
        #     # import soundfile
        #     # soundfile.write(file='_zz.wav', data=omni_waveform.data.cpu().numpy()[0, 0], samplerate=24000)

        #     omni_mag0 = self.tile_omni(omni_mag, max_rays_contain_waveform)
        #     omni_cos0 = self.tile_omni(omni_cos, max_rays_contain_waveform)
        #     omni_sin0 = self.tile_omni(omni_sin, max_rays_contain_waveform)

        #     separated_audio = self.feature_maps_to_wav(
        #         input_tensor=x,
        #         sp=omni_mag0,
        #         sin_in=omni_sin0,
        #         cos_in=omni_cos0,
        #         audio_length=audio_length,
        #     )

        #     separated_audio = separated_audio.reshape(batch_size, sep_rays_num, -1)
        #     # from IPython import embed; embed(using=False); os._exit(0)

        #     output_dict['ray_waveform'] = separated_audio

        # return output_dict

        # 4) Calculate agent position and look direction features.
        agent_position_feature = F.leaky_relu_(self.agent_position_fc(agent_position_emb))
        # shape: (bs, agents_num, T, 128)

        agent_position_feature = agent_position_feature[:, :, 0 :: self.time_downsample_ratio, :]
        # shape: (bs, agents_num, T=38, 128)

        agent_look_direction_feature = F.leaky_relu_(
            self.agent_look_direction_fc(agent_look_direction_emb)
        )  # (bs, agents_num, T, 128)

        agent_look_direction_feature = agent_look_direction_feature[:, :, 0 :: self.time_downsample_ratio, :]
        # shape: (bs, agents_num, T=38, 128)

        # Concatenate mic features and agent features.
        shared_feature = torch.cat((mic_feature, agent_position_feature, 
            agent_look_direction_feature), dim=-1)
        # (bs, agents_num, T=38, 1536)

        output_dict = {}

        inter_emb = shared_feature

        batch_size, rays_num, _T, _C = inter_emb.shape

        x = inter_emb.reshape(batch_size * rays_num, _T, _C)
        # (batch_size * outputs_num, T, C * 3)

        x = x.transpose(1, 2)[:, :, :, None]
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x[:, :, :, 0].transpose(1, 2)

        x = x.reshape(batch_size, rays_num, _T, x.shape[-1])
        # (batch_size, outputs_num, T, C * 2)

        output_dict = {}

        if True:
            origin_len = frames_num
            # tracking_output = torch.sigmoid(self.fc_final_tracking(x)).flatten(2)
            # output_dict['ray_intersect_source'] = torch.mean(tracking_output, dim=-1)
            cla_output = torch.sigmoid(self.fc_final_cla(x))
            # cla_output = cla_output.repeat(1, 1, self.time_downsample_ratio, 1)
            # from IPython import embed; embed(using=False); os._exit(0)

            _bs, _rays_num, _T, _C = cla_output.shape
            tmp = cla_output.reshape(_bs * _rays_num, _T, _C)
            tmp = interpolate(tmp, self.time_downsample_ratio)
            cla_output = tmp.reshape(_bs, _rays_num, -1, _C)

            cla_output = cla_output[:, :, 0 : origin_len, :].flatten(2)
            output_dict['agent_see_source'] = cla_output

        # if self.do_localization:
        #     batch_size, agents_num, _T, _C = shared_feature.shape

        #     x = rearrange(shared_feature, 'b n t c -> (b n) c t 1')
        #     # (bs * agents_num, C=1536, T, 1)

        #     x = self.loc_fc_block1(x)
        #     # (bs * agents_num, 1024, T, 1)

        #     x = self.loc_fc_block2(x)
        #     # (bs * agents_num, 1024, T, 1)

        #     x = rearrange(x, '(b n) c t 1 -> (b n) t c', b=batch_size)
        #     # (bs * agents_num, T, 1024)

        #     x = torch.sigmoid(self.loc_fc_final(x))
        #     # (bs * agents_num, T=38, C=1)

        #     x = interpolate(x, self.time_downsample_ratio)[:, 0 : frames_num, :]
        #     # (bs * agents_num, T=301, C=1)

        #     agent_see_source = rearrange(x, '(b n) t 1 -> b n t', b=batch_size)
        #     # (bs, agents_num, T=301)

        #     output_dict['agent_see_source'] = agent_see_source

        # if self.do_separation:

        #     max_agents_contain_waveform = data_dict['max_agents_contain_waveform']

        #     x = shared_feature[:, 0 : max_agents_contain_waveform, :, :]
        #     # (bs, n=max_agents_contain_waveform, T=38, C=1536)

        #     x = F.leaky_relu_(self.sep_fc_reshape(x))
        #     # (bs, n, T=38, 4096)

        #     x = rearrange(x, 'b n t (c f) -> (b n) c t f', c=128, f=32)
        #     # (bs * n, C=128, T=38, F=32)

        #     enc1 = self.repeat_conv_features(x=enc1, repeats_num=max_agents_contain_waveform)
        #     # (bs * n, C=32, T=304, F=256)

        #     enc2 = self.repeat_conv_features(x=enc2, repeats_num=max_agents_contain_waveform)
        #     # (bs * n, C=64, T=152, F=128)

        #     enc3 = self.repeat_conv_features(x=enc3, repeats_num=max_agents_contain_waveform)
        #     # (bs * n, C=128, T=76, F=64)

        #     x = self.sep_decoder_block1(x, enc3)
        #     x = self.sep_decoder_block2(x, enc2)
        #     x = self.sep_decoder_block3(x, enc1)
        #     # (bs * n, C=32, T=304, F=256)

        #     x = self.sep_conv_final(x)
        #     # (bs * n, C=3, T=304, F=256)

        #     # Recover shape
        #     x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257. 
        #     # (bs * n, C=3, T=304, F=257)

        #     x = x[:, :, 0:frames_num, :]
        #     # (bs * n, C=3, T=301, F=257)

        #     repeat_total_mag = self.repeat_conv_features(
        #         x=total_mag, repeats_num=max_agents_contain_waveform
        #     )  # (bs * n, C=1, T=301, F=257)

        #     repeat_total_sin = self.repeat_conv_features(
        #         x=total_sin, repeats_num=max_agents_contain_waveform
        #     )  # (bs * n, C=1, T=301, F=257)

        #     repeat_total_cos = self.repeat_conv_features(
        #         x=total_cos, repeats_num=max_agents_contain_waveform
        #     )  # (bs * n, C=1, T=301, F=257)

        #     audio_length = total_waveform.shape[-1]

        #     x = self.feature_maps_to_wav(
        #         input_tensor=x,
        #         sp=repeat_total_mag,
        #         sin_in=repeat_total_sin,
        #         cos_in=repeat_total_cos,
        #         audio_length=audio_length,
        #     )
        #     # (bs * n, segment_samples)

        #     agent_waveform = rearrange(x, '(b n) t -> b n t', b=batch_size)
        #     # (bs, n=max_agents_contain_waveform, segment_samples)

        #     output_dict['agent_waveform'] = agent_waveform

        return output_dict


    def tile_omni(self, x, rays_num): 
        batch_size, _, _T, _F = x.shape
        return torch.tile(x[:, None, :, :, :], (1, rays_num, 1, 1, 1)).reshape(batch_size * rays_num, -1, _T, _F)


class Model01_bak4p2(nn.Module, Base): 
    def __init__(self, microphones_num, classes_num, do_localization=None, do_sed=None, do_separation=None):
        super(Model01_bak4p2, self).__init__() 

        window_size = 512
        hop_size = 240
        center = True
        pad_mode = "reflect"
        window = "hann" 
        activation = "relu"
        momentum = 0.01
        classes_num = 1

        self.window_size = window_size
        self.hop_size = hop_size
        self.pad_mode = pad_mode

        positional_embedding_factor = 5
        self.time_downsample_ratio = 2 ** 3

        self.do_localization = True
        self.do_sed = False
        self.do_separation = False

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
            in_channels=microphones_num + 1,
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
            bias=True
        )

        self.phase_fc = nn.Conv2d(
            in_channels=microphones_num * (microphones_num - 1), 
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
            in_features=microphones_num * (positional_embedding_factor * 6), 
            out_features=128, 
            bias=True
        )

        self.mic_look_direction_fc = nn.Linear(
            in_features=microphones_num * (positional_embedding_factor * 4), 
            out_features=128, 
            bias=True
        )

        # self.output_angle_fc = nn.Linear(in_features=(positional_embedding_factor * 4), out_features=128, bias=True)
        # self.output_pos_xyz_fc = nn.Linear(in_features=(positional_embedding_factor * 6), out_features=128, bias=True)

        self.conv_block1 = ConvBlockAfter(in_channels=1536, out_channels=1024)
        self.conv_block2 = ConvBlockAfter(in_channels=1024, out_channels=1024)

        # # self.fc_final_tracking = nn.Linear(1024, 1, bias=True)
        self.fc_final_cla = nn.Linear(1024, classes_num, bias=True)

        # # Sep
        # self.fc_inter = nn.Linear(in_features=1536, out_features=128 * 32, bias=True)

        # self.dec_block1 = DecoderBlockRes1B(in_channels=128, out_channels=128, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        # self.dec_block2 = DecoderBlockRes1B(in_channels=128, out_channels=64, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        # self.dec_block3 = DecoderBlockRes1B(in_channels=64, out_channels=32, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        # self.dec_final = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

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

        if self.do_localization:
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

    def init_weights(self):

        init_layer(self.mag_fc)
        init_layer(self.phase_fc)
        init_layer(self.mic_signal_fc_reshape)

        init_layer(self.mic_position_fc)
        init_layer(self.mic_look_direction_fc)

        # init_layer(self.output_angle_fc)
        # init_layer(self.output_pos_xyz_fc)

        # init_layer(self.fc_inter)
        # # init_layer(self.fc_final_tracking)
        init_layer(self.fc_final_cla)

        # init_layer(self.dec_final)

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
            input_tensor: (batch_size, target_sources_num * output_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, input_channels, time_steps, freq_bins)
            sin_in: (batch_size, input_channels, time_steps, freq_bins)
            cos_in: (batch_size, input_channels, time_steps, freq_bins)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (batch_size, target_sources_num * output_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.output_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, output_channels, self.K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        # linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # mask_cos, mask_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)
        # out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        # out_mag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size * target_sources_num * output_channels
        shape = (
            batch_size * self.target_sources_num * self.output_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * output_channels, segments_num)

        # Reshape.
        waveform = x.reshape(
            batch_size, self.target_sources_num * self.output_channels, audio_length
        )
        # (batch_size, target_sources_num * output_channels, segments_num)

        return waveform

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

    def convert_position_to_embedding(self, position):
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

    def forward(self, data_dict, do_sep=True):

        self.eps = 1e-10

        mic_position = data_dict['mic_position']    # (bs, mics_num, frames_num, 3)
        mic_look_direction = data_dict['mic_look_direction']    # (bs, mics_num, frames_num, 3)
        mic_waveform = data_dict['mic_waveform']    # (bs, mics_num, samples_num)
        agent_position = data_dict['agent_position']    # (bs, agents_num, frames_num, 3)
        agent_look_direction = data_dict['agent_look_direction']    # (bs, agents_num, frames_num, 3)

        # ------ 1. Convert positions and look directions to embeddings ------
        mic_position_emb = self.convert_position_to_embedding(position=mic_position)
        # shape: (bs, mics_num, frames_num, emb_size)

        mic_look_direction_emb = self.convert_look_direction_to_embedding(
            look_direction=mic_look_direction,
        )   # shape: (bs, mics_num, frames_num, emb_size)
        
        agent_position_emb = self.convert_position_to_embedding(position=agent_position)
        # shape: (bs, agents_num, frames_num, emb_size)

        agent_look_direction_emb = self.convert_look_direction_to_embedding(
            look_direction=agent_look_direction,
        )   # shape: (bs, agents_num, frames_num, emb_size)

        # ------ 2. Extract features ------
        total_waveform = torch.sum(mic_waveform, dim=1, keepdim=True)
        # (bs, 1, samples_num)

        total_mag, total_cos, total_sin = self.wav_to_spectrogram_phase(total_waveform, self.eps)
        # (bs, 1, frames_num, freq_bins)

        mic_mag, mic_cos, mic_sin = self.wav_to_spectrogram_phase(mic_waveform, self.eps)
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
        mic_position_emb = F.pad(mic_position_emb, pad=(0, 0, 0, pad_len))
        mic_look_direction_emb = F.pad(mic_look_direction_emb, pad=(0, 0, 0, pad_len))
        agent_position_emb = F.pad(agent_position_emb, pad=(0, 0, 0, pad_len))
        agent_look_direction_emb = F.pad(agent_look_direction_emb, pad=(0, 0, 0, pad_len))

        # ------ 3. Forward data to the network ------

        # 1) Transform mic signal.
        enc1_pool, enc1 = self.mic_signal_encoder_block1(x)
        enc2_pool, enc2 = self.mic_signal_encoder_block2(enc1_pool)
        enc3_pool, enc3 = self.mic_signal_encoder_block3(enc2_pool)
        # enc3_pool: (bs, 128, T=38, F=32)

        mic_signal_feature = rearrange(enc3_pool, 'b c t f -> b t (c f)')
        # (bs, T, C*F=4096)

        mic_signal_feature = F.leaky_relu_(
            self.mic_signal_fc_reshape(mic_signal_feature), negative_slope=0.01,
        )  # shape: (bs, T, 1024)
        
        # 2) Transform mic position and look direction embeddings.
        mic_position_feature = rearrange(mic_position_emb, 'b c t f -> b t (c f)')
        # shape: (bs, T, mics_num*emb_size=120)

        mic_position_feature = F.leaky_relu_(
            self.mic_position_fc(mic_position_feature)
        )  # shape: (bs, T, 128)

        mic_position_feature = mic_position_feature[:, 0 :: self.time_downsample_ratio, :]
        
        mic_look_direction_feature = rearrange(mic_look_direction_emb, 'b c t f -> b t (c f)')
        # shape: (bs, T, mics_num*emb_size=80)

        mic_look_direction_feature = F.leaky_relu_(
            self.mic_look_direction_fc(mic_look_direction_feature)
        )  # shape: (bs, T, 128)

        mic_look_direction_feature = mic_look_direction_feature[:, 0 :: self.time_downsample_ratio, :]

        # 3) Concatenate mic signal, position, and look direction features.
        mic_feature = torch.cat((
            mic_signal_feature, mic_position_feature, mic_look_direction_feature
        ), dim=-1)  # (batch_size, T, 1280)

        agents_num = agent_position.shape[1]

        mic_feature = torch.tile(mic_feature[:, None, :, :], (1, agents_num, 1, 1))
        # shape: (bs, agents_num, T=38, 1280)

        # x = mic_feature
        # ray_direction_emb = agent_look_direction_emb
        # ray_pos_xyz_emb = agent_position_emb
        # origin_len = frames_num

        # # output angle FC
        # a3 = F.leaky_relu_(self.output_angle_fc(ray_direction_emb))
        # a3 = a3[:, :, 0 :: self.time_downsample_ratio, :]
        # # a3 = torch.tile(a3[:, :, None, :], (1, 1, frames_num, 1))
        # a3_xyz = F.leaky_relu_(self.output_pos_xyz_fc(ray_pos_xyz_emb))
        # a3_xyz = a3_xyz[:, :, 0 :: self.time_downsample_ratio, :]
        # # a3_xyz = torch.tile(a3_xyz[:, :, None, :], (1, 1, frames_num, 1))
        # # from IPython import embed; embed(using=False); os._exit(0)
        # inter_emb = torch.cat((x, a3, a3_xyz), dim=-1)
        # # (batch_size, outputs_num, T, C * 3)
        
        # batch_size, rays_num, _T, _C = inter_emb.shape

        # x = inter_emb.reshape(batch_size * rays_num, _T, _C)
        # # (batch_size * outputs_num, T, C * 3)

        # x = x.transpose(1, 2)[:, :, :, None]
        # x = self.conv_block1(x)
        # x = self.conv_block2(x)
        # x = x[:, :, :, 0].transpose(1, 2)

        # x = x.reshape(batch_size, rays_num, _T, x.shape[-1])
        # # (batch_size, outputs_num, T, C * 2)

        # output_dict = {}

        # if True:
        #     # tracking_output = torch.sigmoid(self.fc_final_tracking(x)).flatten(2)
        #     # output_dict['ray_intersect_source'] = torch.mean(tracking_output, dim=-1)
        #     cla_output = torch.sigmoid(self.fc_final_cla(x))
        #     # cla_output = cla_output.repeat(1, 1, self.time_downsample_ratio, 1)
        #     # from IPython import embed; embed(using=False); os._exit(0)

        #     _bs, _rays_num, _T, _C = cla_output.shape
        #     tmp = cla_output.reshape(_bs * _rays_num, _T, _C)
        #     tmp = interpolate(tmp, self.time_downsample_ratio)
        #     cla_output = tmp.reshape(_bs, _rays_num, -1, _C)

        #     cla_output = cla_output[:, :, 0 : origin_len, :].flatten(2)
        #     output_dict['agent_see_source'] = cla_output

        # # if do_sep:
        # if False:
        #     max_rays_contain_waveform = data_dict['max_rays_contain_waveform']
        #     # from IPython import embed; embed(using=False); os._exit(0)

        #     x = F.leaky_relu_(self.fc_inter(inter_emb[:, 0 : max_rays_contain_waveform, :, :]))

        #     batch_size, sep_rays_num, _T, _C = x.shape
        #     x = x.reshape(batch_size * sep_rays_num, _T, 128, 64)
        #     x = x.permute(0, 2, 1, 3)

        #     enc1 = self.tile_omni(x=enc1, rays_num=max_rays_contain_waveform)
        #     enc2 = self.tile_omni(x=enc2, rays_num=max_rays_contain_waveform)
        #     enc3 = self.tile_omni(x=enc3, rays_num=max_rays_contain_waveform)

        #     # from IPython import embed; embed(using=False); os._exit(0)
        #     x = self.dec_block1(x, enc3)
        #     x = self.dec_block2(x, enc2)
        #     x = self.dec_block3(x, enc1)
        #     x = self.dec_final(x)

        #     # Recover shape
        #     x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257. 

        #     x = x[:, :, 0:origin_len, :]

        #     audio_length = omni_waveform.shape[2]

        #     # from IPython import embed; embed(using=False); os._exit(0)
        #     # import matplotlib.pyplot as plt
        #     # plt.matshow(np.log(omni_mag.data.cpu().numpy()[0, 0]).T, origin='lower', aspect='auto', cmap='jet')
        #     # plt.savefig('_zz.pdf')
        #     # import soundfile
        #     # soundfile.write(file='_zz.wav', data=omni_waveform.data.cpu().numpy()[0, 0], samplerate=24000)

        #     omni_mag0 = self.tile_omni(omni_mag, max_rays_contain_waveform)
        #     omni_cos0 = self.tile_omni(omni_cos, max_rays_contain_waveform)
        #     omni_sin0 = self.tile_omni(omni_sin, max_rays_contain_waveform)

        #     separated_audio = self.feature_maps_to_wav(
        #         input_tensor=x,
        #         sp=omni_mag0,
        #         sin_in=omni_sin0,
        #         cos_in=omni_cos0,
        #         audio_length=audio_length,
        #     )

        #     separated_audio = separated_audio.reshape(batch_size, sep_rays_num, -1)
        #     # from IPython import embed; embed(using=False); os._exit(0)

        #     output_dict['ray_waveform'] = separated_audio

        # return output_dict

        # 4) Calculate agent position and look direction features.
        agent_position_feature = F.leaky_relu_(self.agent_position_fc(agent_position_emb))
        # shape: (bs, agents_num, T, 128)

        agent_position_feature = agent_position_feature[:, :, 0 :: self.time_downsample_ratio, :]
        # shape: (bs, agents_num, T=38, 128)

        agent_look_direction_feature = F.leaky_relu_(
            self.agent_look_direction_fc(agent_look_direction_emb)
        )  # (bs, agents_num, T, 128)

        agent_look_direction_feature = agent_look_direction_feature[:, :, 0 :: self.time_downsample_ratio, :]
        # shape: (bs, agents_num, T=38, 128)

        # Concatenate mic features and agent features.
        shared_feature = torch.cat((mic_feature, agent_position_feature, 
            agent_look_direction_feature), dim=-1)
        # (bs, agents_num, T=38, 1536)

        batch_size = shared_feature.shape[0]

        inter_emb = shared_feature

        batch_size, rays_num, _T, _C = inter_emb.shape

        x = inter_emb.reshape(batch_size * rays_num, _T, _C)
        # (batch_size * outputs_num, T, C * 3)

        x = x.transpose(1, 2)[:, :, :, None]
        x = self.loc_fc_block1(x)
        x = self.loc_fc_block2(x)
        x = x[:, :, :, 0].transpose(1, 2)

        x = x.reshape(batch_size, rays_num, _T, x.shape[-1])

        output_dict = {}

        if True:
            # x = rearrange(x, '(b n) t c -> b n t c', b=batch_size)
            origin_len = frames_num
            # tracking_output = torch.sigmoid(self.fc_final_tracking(x)).flatten(2)
            # output_dict['ray_intersect_source'] = torch.mean(tracking_output, dim=-1)
            cla_output = torch.sigmoid(self.fc_final_cla(x))
            # cla_output = cla_output.repeat(1, 1, self.time_downsample_ratio, 1)
            # from IPython import embed; embed(using=False); os._exit(0)

            _bs, _rays_num, _T, _C = cla_output.shape
            tmp = cla_output.reshape(_bs * _rays_num, _T, _C)
            tmp = interpolate(tmp, self.time_downsample_ratio)
            cla_output = tmp.reshape(_bs, _rays_num, -1, _C)

            cla_output = cla_output[:, :, 0 : origin_len, :].flatten(2)
            output_dict['agent_see_source'] = cla_output

        # if self.do_localization:
        #     batch_size, agents_num, _T, _C = shared_feature.shape

        #     x = rearrange(shared_feature, 'b n t c -> (b n) c t 1')
        #     # (bs * agents_num, C=1536, T, 1)

        #     x = self.loc_fc_block1(x)
        #     # (bs * agents_num, 1024, T, 1)

        #     x = self.loc_fc_block2(x)
        #     # (bs * agents_num, 1024, T, 1)

        #     x = rearrange(x, '(b n) c t 1 -> (b n) t c', b=batch_size)
        #     # (bs * agents_num, T, 1024)

        #     x = torch.sigmoid(self.loc_fc_final(x))
        #     # (bs * agents_num, T=38, C=1)

        #     x = interpolate(x, self.time_downsample_ratio)[:, 0 : frames_num, :]
        #     # (bs * agents_num, T=301, C=1)

        #     agent_see_source = rearrange(x, '(b n) t 1 -> b n t', b=batch_size)
        #     # (bs, agents_num, T=301)

        #     output_dict['agent_see_source'] = agent_see_source

        # if self.do_separation:

        #     max_agents_contain_waveform = data_dict['max_agents_contain_waveform']

        #     x = shared_feature[:, 0 : max_agents_contain_waveform, :, :]
        #     # (bs, n=max_agents_contain_waveform, T=38, C=1536)

        #     x = F.leaky_relu_(self.sep_fc_reshape(x))
        #     # (bs, n, T=38, 4096)

        #     x = rearrange(x, 'b n t (c f) -> (b n) c t f', c=128, f=32)
        #     # (bs * n, C=128, T=38, F=32)

        #     enc1 = self.repeat_conv_features(x=enc1, repeats_num=max_agents_contain_waveform)
        #     # (bs * n, C=32, T=304, F=256)

        #     enc2 = self.repeat_conv_features(x=enc2, repeats_num=max_agents_contain_waveform)
        #     # (bs * n, C=64, T=152, F=128)

        #     enc3 = self.repeat_conv_features(x=enc3, repeats_num=max_agents_contain_waveform)
        #     # (bs * n, C=128, T=76, F=64)

        #     x = self.sep_decoder_block1(x, enc3)
        #     x = self.sep_decoder_block2(x, enc2)
        #     x = self.sep_decoder_block3(x, enc1)
        #     # (bs * n, C=32, T=304, F=256)

        #     x = self.sep_conv_final(x)
        #     # (bs * n, C=3, T=304, F=256)

        #     # Recover shape
        #     x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257. 
        #     # (bs * n, C=3, T=304, F=257)

        #     x = x[:, :, 0:frames_num, :]
        #     # (bs * n, C=3, T=301, F=257)

        #     repeat_total_mag = self.repeat_conv_features(
        #         x=total_mag, repeats_num=max_agents_contain_waveform
        #     )  # (bs * n, C=1, T=301, F=257)

        #     repeat_total_sin = self.repeat_conv_features(
        #         x=total_sin, repeats_num=max_agents_contain_waveform
        #     )  # (bs * n, C=1, T=301, F=257)

        #     repeat_total_cos = self.repeat_conv_features(
        #         x=total_cos, repeats_num=max_agents_contain_waveform
        #     )  # (bs * n, C=1, T=301, F=257)

        #     audio_length = total_waveform.shape[-1]

        #     x = self.feature_maps_to_wav(
        #         input_tensor=x,
        #         sp=repeat_total_mag,
        #         sin_in=repeat_total_sin,
        #         cos_in=repeat_total_cos,
        #         audio_length=audio_length,
        #     )
        #     # (bs * n, segment_samples)

        #     agent_waveform = rearrange(x, '(b n) t -> b n t', b=batch_size)
        #     # (bs, n=max_agents_contain_waveform, segment_samples)

        #     output_dict['agent_waveform'] = agent_waveform

        return output_dict


    def tile_omni(self, x, rays_num): 
        batch_size, _, _T, _F = x.shape
        return torch.tile(x[:, None, :, :, :], (1, rays_num, 1, 1, 1)).reshape(batch_size * rays_num, -1, _T, _F)


class Model01_bak4p4(nn.Module, Base):
    def __init__(self, microphones_num, classes_num, do_localization=None, do_sed=None, do_separation=None):
        super(Model01_bak4p4, self).__init__() 

        window_size = 512
        hop_size = 240
        center = True
        pad_mode = "reflect"
        window = "hann" 
        activation = "relu"
        momentum = 0.01
        classes_num = 1

        self.window_size = window_size
        self.hop_size = hop_size
        self.pad_mode = pad_mode

        positional_embedding_factor = 5
        self.time_downsample_ratio = 2 ** 3

        self.do_localization = True
        self.do_sed = False
        self.do_separation = False

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
            in_channels=microphones_num + 1,
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
            bias=True
        )

        self.phase_fc = nn.Conv2d(
            in_channels=microphones_num * (microphones_num - 1), 
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
            in_features=microphones_num * (positional_embedding_factor * 6), 
            out_features=128, 
            bias=True
        )

        self.mic_look_direction_fc = nn.Linear(
            in_features=microphones_num * (positional_embedding_factor * 4), 
            out_features=128, 
            bias=True
        )

        # self.output_angle_fc = nn.Linear(in_features=(positional_embedding_factor * 4), out_features=128, bias=True)
        # self.output_pos_xyz_fc = nn.Linear(in_features=(positional_embedding_factor * 6), out_features=128, bias=True)

        self.conv_block1 = ConvBlockAfter(in_channels=1536, out_channels=1024)
        self.conv_block2 = ConvBlockAfter(in_channels=1024, out_channels=1024)

        # # self.fc_final_tracking = nn.Linear(1024, 1, bias=True)
        self.fc_final_cla = nn.Linear(1024, classes_num, bias=True)

        # # Sep
        # self.fc_inter = nn.Linear(in_features=1536, out_features=128 * 32, bias=True)

        # self.dec_block1 = DecoderBlockRes1B(in_channels=128, out_channels=128, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        # self.dec_block2 = DecoderBlockRes1B(in_channels=128, out_channels=64, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        # self.dec_block3 = DecoderBlockRes1B(in_channels=64, out_channels=32, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        # self.dec_final = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

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

        if self.do_localization:
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

    def init_weights(self):

        init_layer(self.mag_fc)
        init_layer(self.phase_fc)
        init_layer(self.mic_signal_fc_reshape)

        init_layer(self.mic_position_fc)
        init_layer(self.mic_look_direction_fc)

        # init_layer(self.output_angle_fc)
        # init_layer(self.output_pos_xyz_fc)

        # init_layer(self.fc_inter)
        # # init_layer(self.fc_final_tracking)
        init_layer(self.fc_final_cla)

        # init_layer(self.dec_final)

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
            input_tensor: (batch_size, target_sources_num * output_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, input_channels, time_steps, freq_bins)
            sin_in: (batch_size, input_channels, time_steps, freq_bins)
            cos_in: (batch_size, input_channels, time_steps, freq_bins)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (batch_size, target_sources_num * output_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.output_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, output_channels, self.K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        # linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # mask_cos, mask_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)
        # out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        # out_mag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size * target_sources_num * output_channels
        shape = (
            batch_size * self.target_sources_num * self.output_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * output_channels, segments_num)

        # Reshape.
        waveform = x.reshape(
            batch_size, self.target_sources_num * self.output_channels, audio_length
        )
        # (batch_size, target_sources_num * output_channels, segments_num)

        return waveform

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

    def convert_position_to_embedding(self, position):
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

    def forward(self, data_dict, do_sep=True):

        self.eps = 1e-10

        mic_position = data_dict['mic_position']    # (bs, mics_num, frames_num, 3)
        mic_look_direction = data_dict['mic_look_direction']    # (bs, mics_num, frames_num, 3)
        mic_waveform = data_dict['mic_waveform']    # (bs, mics_num, samples_num)
        agent_position = data_dict['agent_position']    # (bs, agents_num, frames_num, 3)
        agent_look_direction = data_dict['agent_look_direction']    # (bs, agents_num, frames_num, 3)

        # ------ 1. Convert positions and look directions to embeddings ------
        mic_position_emb = self.convert_position_to_embedding(position=mic_position)
        # shape: (bs, mics_num, frames_num, emb_size)

        mic_look_direction_emb = self.convert_look_direction_to_embedding(
            look_direction=mic_look_direction,
        )   # shape: (bs, mics_num, frames_num, emb_size)
        
        agent_position_emb = self.convert_position_to_embedding(position=agent_position)
        # shape: (bs, agents_num, frames_num, emb_size)

        agent_look_direction_emb = self.convert_look_direction_to_embedding(
            look_direction=agent_look_direction,
        )   # shape: (bs, agents_num, frames_num, emb_size)

        # ------ 2. Extract features ------
        total_waveform = torch.sum(mic_waveform, dim=1, keepdim=True)
        # (bs, 1, samples_num)

        total_mag, total_cos, total_sin = self.wav_to_spectrogram_phase(total_waveform, self.eps)
        # (bs, 1, frames_num, freq_bins)

        mic_mag, mic_cos, mic_sin = self.wav_to_spectrogram_phase(mic_waveform, self.eps)
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
        mic_position_emb = F.pad(mic_position_emb, pad=(0, 0, 0, pad_len))
        mic_look_direction_emb = F.pad(mic_look_direction_emb, pad=(0, 0, 0, pad_len))
        agent_position_emb = F.pad(agent_position_emb, pad=(0, 0, 0, pad_len))
        agent_look_direction_emb = F.pad(agent_look_direction_emb, pad=(0, 0, 0, pad_len))

        # ------ 3. Forward data to the network ------

        # 1) Transform mic signal.
        enc1_pool, enc1 = self.mic_signal_encoder_block1(x)
        enc2_pool, enc2 = self.mic_signal_encoder_block2(enc1_pool)
        enc3_pool, enc3 = self.mic_signal_encoder_block3(enc2_pool)
        # enc3_pool: (bs, 128, T=38, F=32)

        mic_signal_feature = rearrange(enc3_pool, 'b c t f -> b t (c f)')
        # (bs, T, C*F=4096)

        mic_signal_feature = F.leaky_relu_(
            self.mic_signal_fc_reshape(mic_signal_feature), negative_slope=0.01,
        )  # shape: (bs, T, 1024)
        
        # 2) Transform mic position and look direction embeddings.
        mic_position_feature = rearrange(mic_position_emb, 'b c t f -> b t (c f)')
        # shape: (bs, T, mics_num*emb_size=120)

        mic_position_feature = F.leaky_relu_(
            self.mic_position_fc(mic_position_feature)
        )  # shape: (bs, T, 128)

        mic_position_feature = mic_position_feature[:, 0 :: self.time_downsample_ratio, :]
        
        mic_look_direction_feature = rearrange(mic_look_direction_emb, 'b c t f -> b t (c f)')
        # shape: (bs, T, mics_num*emb_size=80)

        mic_look_direction_feature = F.leaky_relu_(
            self.mic_look_direction_fc(mic_look_direction_feature)
        )  # shape: (bs, T, 128)

        mic_look_direction_feature = mic_look_direction_feature[:, 0 :: self.time_downsample_ratio, :]

        # 3) Concatenate mic signal, position, and look direction features.
        mic_feature = torch.cat((
            mic_signal_feature, mic_position_feature, mic_look_direction_feature
        ), dim=-1)  # (batch_size, T, 1280)

        agents_num = agent_position.shape[1]

        mic_feature = torch.tile(mic_feature[:, None, :, :], (1, agents_num, 1, 1))
        # shape: (bs, agents_num, T=38, 1280)

        # x = mic_feature
        # ray_direction_emb = agent_look_direction_emb
        # ray_pos_xyz_emb = agent_position_emb
        # origin_len = frames_num

        # # output angle FC
        # a3 = F.leaky_relu_(self.output_angle_fc(ray_direction_emb))
        # a3 = a3[:, :, 0 :: self.time_downsample_ratio, :]
        # # a3 = torch.tile(a3[:, :, None, :], (1, 1, frames_num, 1))
        # a3_xyz = F.leaky_relu_(self.output_pos_xyz_fc(ray_pos_xyz_emb))
        # a3_xyz = a3_xyz[:, :, 0 :: self.time_downsample_ratio, :]
        # # a3_xyz = torch.tile(a3_xyz[:, :, None, :], (1, 1, frames_num, 1))
        # # from IPython import embed; embed(using=False); os._exit(0)
        # inter_emb = torch.cat((x, a3, a3_xyz), dim=-1)
        # # (batch_size, outputs_num, T, C * 3)
        
        # batch_size, rays_num, _T, _C = inter_emb.shape

        # x = inter_emb.reshape(batch_size * rays_num, _T, _C)
        # # (batch_size * outputs_num, T, C * 3)

        # x = x.transpose(1, 2)[:, :, :, None]
        # x = self.conv_block1(x)
        # x = self.conv_block2(x)
        # x = x[:, :, :, 0].transpose(1, 2)

        # x = x.reshape(batch_size, rays_num, _T, x.shape[-1])
        # # (batch_size, outputs_num, T, C * 2)

        # output_dict = {}

        # if True:
        #     # tracking_output = torch.sigmoid(self.fc_final_tracking(x)).flatten(2)
        #     # output_dict['ray_intersect_source'] = torch.mean(tracking_output, dim=-1)
        #     cla_output = torch.sigmoid(self.fc_final_cla(x))
        #     # cla_output = cla_output.repeat(1, 1, self.time_downsample_ratio, 1)
        #     # from IPython import embed; embed(using=False); os._exit(0)

        #     _bs, _rays_num, _T, _C = cla_output.shape
        #     tmp = cla_output.reshape(_bs * _rays_num, _T, _C)
        #     tmp = interpolate(tmp, self.time_downsample_ratio)
        #     cla_output = tmp.reshape(_bs, _rays_num, -1, _C)

        #     cla_output = cla_output[:, :, 0 : origin_len, :].flatten(2)
        #     output_dict['agent_see_source'] = cla_output

        # # if do_sep:
        # if False:
        #     max_rays_contain_waveform = data_dict['max_rays_contain_waveform']
        #     # from IPython import embed; embed(using=False); os._exit(0)

        #     x = F.leaky_relu_(self.fc_inter(inter_emb[:, 0 : max_rays_contain_waveform, :, :]))

        #     batch_size, sep_rays_num, _T, _C = x.shape
        #     x = x.reshape(batch_size * sep_rays_num, _T, 128, 64)
        #     x = x.permute(0, 2, 1, 3)

        #     enc1 = self.tile_omni(x=enc1, rays_num=max_rays_contain_waveform)
        #     enc2 = self.tile_omni(x=enc2, rays_num=max_rays_contain_waveform)
        #     enc3 = self.tile_omni(x=enc3, rays_num=max_rays_contain_waveform)

        #     # from IPython import embed; embed(using=False); os._exit(0)
        #     x = self.dec_block1(x, enc3)
        #     x = self.dec_block2(x, enc2)
        #     x = self.dec_block3(x, enc1)
        #     x = self.dec_final(x)

        #     # Recover shape
        #     x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257. 

        #     x = x[:, :, 0:origin_len, :]

        #     audio_length = omni_waveform.shape[2]

        #     # from IPython import embed; embed(using=False); os._exit(0)
        #     # import matplotlib.pyplot as plt
        #     # plt.matshow(np.log(omni_mag.data.cpu().numpy()[0, 0]).T, origin='lower', aspect='auto', cmap='jet')
        #     # plt.savefig('_zz.pdf')
        #     # import soundfile
        #     # soundfile.write(file='_zz.wav', data=omni_waveform.data.cpu().numpy()[0, 0], samplerate=24000)

        #     omni_mag0 = self.tile_omni(omni_mag, max_rays_contain_waveform)
        #     omni_cos0 = self.tile_omni(omni_cos, max_rays_contain_waveform)
        #     omni_sin0 = self.tile_omni(omni_sin, max_rays_contain_waveform)

        #     separated_audio = self.feature_maps_to_wav(
        #         input_tensor=x,
        #         sp=omni_mag0,
        #         sin_in=omni_sin0,
        #         cos_in=omni_cos0,
        #         audio_length=audio_length,
        #     )

        #     separated_audio = separated_audio.reshape(batch_size, sep_rays_num, -1)
        #     # from IPython import embed; embed(using=False); os._exit(0)

        #     output_dict['ray_waveform'] = separated_audio

        # return output_dict

        # 4) Calculate agent position and look direction features.
        agent_position_feature = F.leaky_relu_(self.agent_position_fc(agent_position_emb))
        # shape: (bs, agents_num, T, 128)

        agent_position_feature = agent_position_feature[:, :, 0 :: self.time_downsample_ratio, :]
        # shape: (bs, agents_num, T=38, 128)

        agent_look_direction_feature = F.leaky_relu_(
            self.agent_look_direction_fc(agent_look_direction_emb)
        )  # (bs, agents_num, T, 128)

        agent_look_direction_feature = agent_look_direction_feature[:, :, 0 :: self.time_downsample_ratio, :]
        # shape: (bs, agents_num, T=38, 128)

        # Concatenate mic features and agent features.
        shared_feature = torch.cat((mic_feature, agent_position_feature, 
            agent_look_direction_feature), dim=-1)
        # (bs, agents_num, T=38, 1536)

        batch_size = shared_feature.shape[0]

        x = shared_feature

        _B, _N, _T, _C = x.shape
        x = x.reshape(_B * _N, _T, _C).transpose(1, 2)[:, :, :, None]
        
        # x = rearrange(shared_feature, 'b n t c -> (b n) c t 1')

        # (bs * agents_num, C=1536, T, 1)

        x = self.loc_fc_block1(x)
        # (bs * agents_num, 1024, T, 1)

        x = self.loc_fc_block2(x)
        # (bs * agents_num, 1024, T, 1)

        x = x.reshape(_B, _N, -1, _T).transpose(2, 3)
        # x = rearrange(x, '(b n) c t 1 -> (b n) t c', b=batch_size)
        # (bs * agents_num, T, 1024)

        output_dict = {}

        if True:
            # x = rearrange(x, '(b n) t c -> b n t c', b=batch_size)
            origin_len = frames_num
            # tracking_output = torch.sigmoid(self.fc_final_tracking(x)).flatten(2)
            # output_dict['ray_intersect_source'] = torch.mean(tracking_output, dim=-1)
            cla_output = torch.sigmoid(self.fc_final_cla(x))
            # cla_output = cla_output.repeat(1, 1, self.time_downsample_ratio, 1)
            # from IPython import embed; embed(using=False); os._exit(0)

            _bs, _rays_num, _T, _C = cla_output.shape
            tmp = cla_output.reshape(_bs * _rays_num, _T, _C)
            tmp = interpolate(tmp, self.time_downsample_ratio)
            cla_output = tmp.reshape(_bs, _rays_num, -1, _C)

            cla_output = cla_output[:, :, 0 : origin_len, :].flatten(2)
            output_dict['agent_see_source'] = cla_output

        # if self.do_localization:
        #     batch_size, agents_num, _T, _C = shared_feature.shape

        #     x = rearrange(shared_feature, 'b n t c -> (b n) c t 1')
        #     # (bs * agents_num, C=1536, T, 1)

        #     x = self.loc_fc_block1(x)
        #     # (bs * agents_num, 1024, T, 1)

        #     x = self.loc_fc_block2(x)
        #     # (bs * agents_num, 1024, T, 1)

        #     x = rearrange(x, '(b n) c t 1 -> (b n) t c', b=batch_size)
        #     # (bs * agents_num, T, 1024)

        #     x = torch.sigmoid(self.loc_fc_final(x))
        #     # (bs * agents_num, T=38, C=1)

        #     x = interpolate(x, self.time_downsample_ratio)[:, 0 : frames_num, :]
        #     # (bs * agents_num, T=301, C=1)

        #     agent_see_source = rearrange(x, '(b n) t 1 -> b n t', b=batch_size)
        #     # (bs, agents_num, T=301)

        #     output_dict['agent_see_source'] = agent_see_source

        # if self.do_separation:

        #     max_agents_contain_waveform = data_dict['max_agents_contain_waveform']

        #     x = shared_feature[:, 0 : max_agents_contain_waveform, :, :]
        #     # (bs, n=max_agents_contain_waveform, T=38, C=1536)

        #     x = F.leaky_relu_(self.sep_fc_reshape(x))
        #     # (bs, n, T=38, 4096)

        #     x = rearrange(x, 'b n t (c f) -> (b n) c t f', c=128, f=32)
        #     # (bs * n, C=128, T=38, F=32)

        #     enc1 = self.repeat_conv_features(x=enc1, repeats_num=max_agents_contain_waveform)
        #     # (bs * n, C=32, T=304, F=256)

        #     enc2 = self.repeat_conv_features(x=enc2, repeats_num=max_agents_contain_waveform)
        #     # (bs * n, C=64, T=152, F=128)

        #     enc3 = self.repeat_conv_features(x=enc3, repeats_num=max_agents_contain_waveform)
        #     # (bs * n, C=128, T=76, F=64)

        #     x = self.sep_decoder_block1(x, enc3)
        #     x = self.sep_decoder_block2(x, enc2)
        #     x = self.sep_decoder_block3(x, enc1)
        #     # (bs * n, C=32, T=304, F=256)

        #     x = self.sep_conv_final(x)
        #     # (bs * n, C=3, T=304, F=256)

        #     # Recover shape
        #     x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257. 
        #     # (bs * n, C=3, T=304, F=257)

        #     x = x[:, :, 0:frames_num, :]
        #     # (bs * n, C=3, T=301, F=257)

        #     repeat_total_mag = self.repeat_conv_features(
        #         x=total_mag, repeats_num=max_agents_contain_waveform
        #     )  # (bs * n, C=1, T=301, F=257)

        #     repeat_total_sin = self.repeat_conv_features(
        #         x=total_sin, repeats_num=max_agents_contain_waveform
        #     )  # (bs * n, C=1, T=301, F=257)

        #     repeat_total_cos = self.repeat_conv_features(
        #         x=total_cos, repeats_num=max_agents_contain_waveform
        #     )  # (bs * n, C=1, T=301, F=257)

        #     audio_length = total_waveform.shape[-1]

        #     x = self.feature_maps_to_wav(
        #         input_tensor=x,
        #         sp=repeat_total_mag,
        #         sin_in=repeat_total_sin,
        #         cos_in=repeat_total_cos,
        #         audio_length=audio_length,
        #     )
        #     # (bs * n, segment_samples)

        #     agent_waveform = rearrange(x, '(b n) t -> b n t', b=batch_size)
        #     # (bs, n=max_agents_contain_waveform, segment_samples)

        #     output_dict['agent_waveform'] = agent_waveform

        return output_dict


    def tile_omni(self, x, rays_num): 
        batch_size, _, _T, _F = x.shape
        return torch.tile(x[:, None, :, :, :], (1, rays_num, 1, 1, 1)).reshape(batch_size * rays_num, -1, _T, _F)


class Model01_bak4p5(nn.Module, Base):
    def __init__(self, microphones_num, classes_num, do_localization=None, do_sed=None, do_separation=None):
        super(Model01_bak4p5, self).__init__() 

        window_size = 512
        hop_size = 240
        center = True
        pad_mode = "reflect"
        window = "hann" 
        activation = "relu"
        momentum = 0.01
        classes_num = 1

        self.window_size = window_size
        self.hop_size = hop_size
        self.pad_mode = pad_mode

        positional_embedding_factor = 5
        self.time_downsample_ratio = 2 ** 3

        self.do_localization = True
        self.do_sed = False
        self.do_separation = False

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
            in_channels=microphones_num + 1,
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
            bias=True
        )

        self.phase_fc = nn.Conv2d(
            in_channels=microphones_num * (microphones_num - 1), 
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
            in_features=microphones_num * (positional_embedding_factor * 6), 
            out_features=128, 
            bias=True
        )

        self.mic_look_direction_fc = nn.Linear(
            in_features=microphones_num * (positional_embedding_factor * 4), 
            out_features=128, 
            bias=True
        )

        # self.output_angle_fc = nn.Linear(in_features=(positional_embedding_factor * 4), out_features=128, bias=True)
        # self.output_pos_xyz_fc = nn.Linear(in_features=(positional_embedding_factor * 6), out_features=128, bias=True)

        self.conv_block1 = ConvBlockAfter(in_channels=1536, out_channels=1024)
        self.conv_block2 = ConvBlockAfter(in_channels=1024, out_channels=1024)

        # # self.fc_final_tracking = nn.Linear(1024, 1, bias=True)
        self.fc_final_cla = nn.Linear(1024, classes_num, bias=True)

        # # Sep
        # self.fc_inter = nn.Linear(in_features=1536, out_features=128 * 32, bias=True)

        # self.dec_block1 = DecoderBlockRes1B(in_channels=128, out_channels=128, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        # self.dec_block2 = DecoderBlockRes1B(in_channels=128, out_channels=64, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        # self.dec_block3 = DecoderBlockRes1B(in_channels=64, out_channels=32, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        # self.dec_final = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

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

        if self.do_localization:
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

    def init_weights(self):

        init_layer(self.mag_fc)
        init_layer(self.phase_fc)
        init_layer(self.mic_signal_fc_reshape)

        init_layer(self.mic_position_fc)
        init_layer(self.mic_look_direction_fc)

        # init_layer(self.output_angle_fc)
        # init_layer(self.output_pos_xyz_fc)

        # init_layer(self.fc_inter)
        # # init_layer(self.fc_final_tracking)
        init_layer(self.fc_final_cla)

        # init_layer(self.dec_final)

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
            input_tensor: (batch_size, target_sources_num * output_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, input_channels, time_steps, freq_bins)
            sin_in: (batch_size, input_channels, time_steps, freq_bins)
            cos_in: (batch_size, input_channels, time_steps, freq_bins)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (batch_size, target_sources_num * output_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.output_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, output_channels, self.K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        # linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # mask_cos, mask_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)
        # out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        # out_mag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size * target_sources_num * output_channels
        shape = (
            batch_size * self.target_sources_num * self.output_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * output_channels, segments_num)

        # Reshape.
        waveform = x.reshape(
            batch_size, self.target_sources_num * self.output_channels, audio_length
        )
        # (batch_size, target_sources_num * output_channels, segments_num)

        return waveform

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

    def convert_position_to_embedding(self, position):
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

    def forward(self, data_dict, do_sep=True):

        self.eps = 1e-10

        mic_position = data_dict['mic_position']    # (bs, mics_num, frames_num, 3)
        mic_look_direction = data_dict['mic_look_direction']    # (bs, mics_num, frames_num, 3)
        mic_waveform = data_dict['mic_waveform']    # (bs, mics_num, samples_num)
        agent_position = data_dict['agent_position']    # (bs, agents_num, frames_num, 3)
        agent_look_direction = data_dict['agent_look_direction']    # (bs, agents_num, frames_num, 3)

        # ------ 1. Convert positions and look directions to embeddings ------
        mic_position_emb = self.convert_position_to_embedding(position=mic_position)
        # shape: (bs, mics_num, frames_num, emb_size)

        mic_look_direction_emb = self.convert_look_direction_to_embedding(
            look_direction=mic_look_direction,
        )   # shape: (bs, mics_num, frames_num, emb_size)
        
        agent_position_emb = self.convert_position_to_embedding(position=agent_position)
        # shape: (bs, agents_num, frames_num, emb_size)

        agent_look_direction_emb = self.convert_look_direction_to_embedding(
            look_direction=agent_look_direction,
        )   # shape: (bs, agents_num, frames_num, emb_size)

        # ------ 2. Extract features ------
        total_waveform = torch.sum(mic_waveform, dim=1, keepdim=True)
        # (bs, 1, samples_num)

        total_mag, total_cos, total_sin = self.wav_to_spectrogram_phase(total_waveform, self.eps)
        # (bs, 1, frames_num, freq_bins)

        mic_mag, mic_cos, mic_sin = self.wav_to_spectrogram_phase(mic_waveform, self.eps)
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
        mic_position_emb = F.pad(mic_position_emb, pad=(0, 0, 0, pad_len))
        mic_look_direction_emb = F.pad(mic_look_direction_emb, pad=(0, 0, 0, pad_len))
        agent_position_emb = F.pad(agent_position_emb, pad=(0, 0, 0, pad_len))
        agent_look_direction_emb = F.pad(agent_look_direction_emb, pad=(0, 0, 0, pad_len))

        # ------ 3. Forward data to the network ------

        # 1) Transform mic signal.
        enc1_pool, enc1 = self.mic_signal_encoder_block1(x)
        enc2_pool, enc2 = self.mic_signal_encoder_block2(enc1_pool)
        enc3_pool, enc3 = self.mic_signal_encoder_block3(enc2_pool)
        # enc3_pool: (bs, 128, T=38, F=32)

        mic_signal_feature = rearrange(enc3_pool, 'b c t f -> b t (c f)')
        # (bs, T, C*F=4096)

        mic_signal_feature = F.leaky_relu_(
            self.mic_signal_fc_reshape(mic_signal_feature), negative_slope=0.01,
        )  # shape: (bs, T, 1024)
        
        # 2) Transform mic position and look direction embeddings.
        mic_position_feature = rearrange(mic_position_emb, 'b c t f -> b t (c f)')
        # shape: (bs, T, mics_num*emb_size=120)

        mic_position_feature = F.leaky_relu_(
            self.mic_position_fc(mic_position_feature)
        )  # shape: (bs, T, 128)

        mic_position_feature = mic_position_feature[:, 0 :: self.time_downsample_ratio, :]
        
        mic_look_direction_feature = rearrange(mic_look_direction_emb, 'b c t f -> b t (c f)')
        # shape: (bs, T, mics_num*emb_size=80)

        mic_look_direction_feature = F.leaky_relu_(
            self.mic_look_direction_fc(mic_look_direction_feature)
        )  # shape: (bs, T, 128)

        mic_look_direction_feature = mic_look_direction_feature[:, 0 :: self.time_downsample_ratio, :]

        # 3) Concatenate mic signal, position, and look direction features.
        mic_feature = torch.cat((
            mic_signal_feature, mic_position_feature, mic_look_direction_feature
        ), dim=-1)  # (batch_size, T, 1280)

        agents_num = agent_position.shape[1]

        mic_feature = torch.tile(mic_feature[:, None, :, :], (1, agents_num, 1, 1))
        # shape: (bs, agents_num, T=38, 1280)

        # x = mic_feature
        # ray_direction_emb = agent_look_direction_emb
        # ray_pos_xyz_emb = agent_position_emb
        # origin_len = frames_num

        # # output angle FC
        # a3 = F.leaky_relu_(self.output_angle_fc(ray_direction_emb))
        # a3 = a3[:, :, 0 :: self.time_downsample_ratio, :]
        # # a3 = torch.tile(a3[:, :, None, :], (1, 1, frames_num, 1))
        # a3_xyz = F.leaky_relu_(self.output_pos_xyz_fc(ray_pos_xyz_emb))
        # a3_xyz = a3_xyz[:, :, 0 :: self.time_downsample_ratio, :]
        # # a3_xyz = torch.tile(a3_xyz[:, :, None, :], (1, 1, frames_num, 1))
        # # from IPython import embed; embed(using=False); os._exit(0)
        # inter_emb = torch.cat((x, a3, a3_xyz), dim=-1)
        # # (batch_size, outputs_num, T, C * 3)
        
        # batch_size, rays_num, _T, _C = inter_emb.shape

        # x = inter_emb.reshape(batch_size * rays_num, _T, _C)
        # # (batch_size * outputs_num, T, C * 3)

        # x = x.transpose(1, 2)[:, :, :, None]
        # x = self.conv_block1(x)
        # x = self.conv_block2(x)
        # x = x[:, :, :, 0].transpose(1, 2)

        # x = x.reshape(batch_size, rays_num, _T, x.shape[-1])
        # # (batch_size, outputs_num, T, C * 2)

        # output_dict = {}

        # if True:
        #     # tracking_output = torch.sigmoid(self.fc_final_tracking(x)).flatten(2)
        #     # output_dict['ray_intersect_source'] = torch.mean(tracking_output, dim=-1)
        #     cla_output = torch.sigmoid(self.fc_final_cla(x))
        #     # cla_output = cla_output.repeat(1, 1, self.time_downsample_ratio, 1)
        #     # from IPython import embed; embed(using=False); os._exit(0)

        #     _bs, _rays_num, _T, _C = cla_output.shape
        #     tmp = cla_output.reshape(_bs * _rays_num, _T, _C)
        #     tmp = interpolate(tmp, self.time_downsample_ratio)
        #     cla_output = tmp.reshape(_bs, _rays_num, -1, _C)

        #     cla_output = cla_output[:, :, 0 : origin_len, :].flatten(2)
        #     output_dict['agent_see_source'] = cla_output

        # # if do_sep:
        # if False:
        #     max_rays_contain_waveform = data_dict['max_rays_contain_waveform']
        #     # from IPython import embed; embed(using=False); os._exit(0)

        #     x = F.leaky_relu_(self.fc_inter(inter_emb[:, 0 : max_rays_contain_waveform, :, :]))

        #     batch_size, sep_rays_num, _T, _C = x.shape
        #     x = x.reshape(batch_size * sep_rays_num, _T, 128, 64)
        #     x = x.permute(0, 2, 1, 3)

        #     enc1 = self.tile_omni(x=enc1, rays_num=max_rays_contain_waveform)
        #     enc2 = self.tile_omni(x=enc2, rays_num=max_rays_contain_waveform)
        #     enc3 = self.tile_omni(x=enc3, rays_num=max_rays_contain_waveform)

        #     # from IPython import embed; embed(using=False); os._exit(0)
        #     x = self.dec_block1(x, enc3)
        #     x = self.dec_block2(x, enc2)
        #     x = self.dec_block3(x, enc1)
        #     x = self.dec_final(x)

        #     # Recover shape
        #     x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257. 

        #     x = x[:, :, 0:origin_len, :]

        #     audio_length = omni_waveform.shape[2]

        #     # from IPython import embed; embed(using=False); os._exit(0)
        #     # import matplotlib.pyplot as plt
        #     # plt.matshow(np.log(omni_mag.data.cpu().numpy()[0, 0]).T, origin='lower', aspect='auto', cmap='jet')
        #     # plt.savefig('_zz.pdf')
        #     # import soundfile
        #     # soundfile.write(file='_zz.wav', data=omni_waveform.data.cpu().numpy()[0, 0], samplerate=24000)

        #     omni_mag0 = self.tile_omni(omni_mag, max_rays_contain_waveform)
        #     omni_cos0 = self.tile_omni(omni_cos, max_rays_contain_waveform)
        #     omni_sin0 = self.tile_omni(omni_sin, max_rays_contain_waveform)

        #     separated_audio = self.feature_maps_to_wav(
        #         input_tensor=x,
        #         sp=omni_mag0,
        #         sin_in=omni_sin0,
        #         cos_in=omni_cos0,
        #         audio_length=audio_length,
        #     )

        #     separated_audio = separated_audio.reshape(batch_size, sep_rays_num, -1)
        #     # from IPython import embed; embed(using=False); os._exit(0)

        #     output_dict['ray_waveform'] = separated_audio

        # return output_dict

        # 4) Calculate agent position and look direction features.
        agent_position_feature = F.leaky_relu_(self.agent_position_fc(agent_position_emb))
        # shape: (bs, agents_num, T, 128)

        agent_position_feature = agent_position_feature[:, :, 0 :: self.time_downsample_ratio, :]
        # shape: (bs, agents_num, T=38, 128)

        agent_look_direction_feature = F.leaky_relu_(
            self.agent_look_direction_fc(agent_look_direction_emb)
        )  # (bs, agents_num, T, 128)

        agent_look_direction_feature = agent_look_direction_feature[:, :, 0 :: self.time_downsample_ratio, :]
        # shape: (bs, agents_num, T=38, 128)

        # Concatenate mic features and agent features.
        shared_feature = torch.cat((mic_feature, agent_position_feature, 
            agent_look_direction_feature), dim=-1)
        # (bs, agents_num, T=38, 1536)

        batch_size = shared_feature.shape[0]

        x = rearrange(shared_feature, 'b n t c -> (b n) c t 1')
        # (bs * agents_num, C=1536, T, 1)

        x = self.loc_fc_block1(x)
        # (bs * agents_num, 1024, T, 1)

        x = self.loc_fc_block2(x)
        # (bs * agents_num, 1024, T, 1)

        x = rearrange(x, '(b n) c t 1 -> (b n) t c', b=batch_size)
        # (bs * agents_num, T, 1024)

        output_dict = {}

        if True:
            x = rearrange(x, '(b n) t c -> b n t c', b=batch_size)
            origin_len = frames_num
            # tracking_output = torch.sigmoid(self.fc_final_tracking(x)).flatten(2)
            # output_dict['ray_intersect_source'] = torch.mean(tracking_output, dim=-1)
            cla_output = torch.sigmoid(self.fc_final_cla(x))
            # cla_output = cla_output.repeat(1, 1, self.time_downsample_ratio, 1)
            # from IPython import embed; embed(using=False); os._exit(0)

            _bs, _rays_num, _T, _C = cla_output.shape
            tmp = cla_output.reshape(_bs * _rays_num, _T, _C)
            tmp = interpolate(tmp, self.time_downsample_ratio)
            cla_output = tmp.reshape(_bs, _rays_num, -1, _C)

            cla_output = cla_output[:, :, 0 : origin_len, :].flatten(2)
            output_dict['agent_see_source'] = cla_output

        # if self.do_localization:
        #     batch_size, agents_num, _T, _C = shared_feature.shape

        #     x = rearrange(shared_feature, 'b n t c -> (b n) c t 1')
        #     # (bs * agents_num, C=1536, T, 1)

        #     x = self.loc_fc_block1(x)
        #     # (bs * agents_num, 1024, T, 1)

        #     x = self.loc_fc_block2(x)
        #     # (bs * agents_num, 1024, T, 1)

        #     x = rearrange(x, '(b n) c t 1 -> (b n) t c', b=batch_size)
        #     # (bs * agents_num, T, 1024)

        #     x = torch.sigmoid(self.loc_fc_final(x))
        #     # (bs * agents_num, T=38, C=1)

        #     x = interpolate(x, self.time_downsample_ratio)[:, 0 : frames_num, :]
        #     # (bs * agents_num, T=301, C=1)

        #     agent_see_source = rearrange(x, '(b n) t 1 -> b n t', b=batch_size)
        #     # (bs, agents_num, T=301)

        #     output_dict['agent_see_source'] = agent_see_source

        # if self.do_separation:

        #     max_agents_contain_waveform = data_dict['max_agents_contain_waveform']

        #     x = shared_feature[:, 0 : max_agents_contain_waveform, :, :]
        #     # (bs, n=max_agents_contain_waveform, T=38, C=1536)

        #     x = F.leaky_relu_(self.sep_fc_reshape(x))
        #     # (bs, n, T=38, 4096)

        #     x = rearrange(x, 'b n t (c f) -> (b n) c t f', c=128, f=32)
        #     # (bs * n, C=128, T=38, F=32)

        #     enc1 = self.repeat_conv_features(x=enc1, repeats_num=max_agents_contain_waveform)
        #     # (bs * n, C=32, T=304, F=256)

        #     enc2 = self.repeat_conv_features(x=enc2, repeats_num=max_agents_contain_waveform)
        #     # (bs * n, C=64, T=152, F=128)

        #     enc3 = self.repeat_conv_features(x=enc3, repeats_num=max_agents_contain_waveform)
        #     # (bs * n, C=128, T=76, F=64)

        #     x = self.sep_decoder_block1(x, enc3)
        #     x = self.sep_decoder_block2(x, enc2)
        #     x = self.sep_decoder_block3(x, enc1)
        #     # (bs * n, C=32, T=304, F=256)

        #     x = self.sep_conv_final(x)
        #     # (bs * n, C=3, T=304, F=256)

        #     # Recover shape
        #     x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257. 
        #     # (bs * n, C=3, T=304, F=257)

        #     x = x[:, :, 0:frames_num, :]
        #     # (bs * n, C=3, T=301, F=257)

        #     repeat_total_mag = self.repeat_conv_features(
        #         x=total_mag, repeats_num=max_agents_contain_waveform
        #     )  # (bs * n, C=1, T=301, F=257)

        #     repeat_total_sin = self.repeat_conv_features(
        #         x=total_sin, repeats_num=max_agents_contain_waveform
        #     )  # (bs * n, C=1, T=301, F=257)

        #     repeat_total_cos = self.repeat_conv_features(
        #         x=total_cos, repeats_num=max_agents_contain_waveform
        #     )  # (bs * n, C=1, T=301, F=257)

        #     audio_length = total_waveform.shape[-1]

        #     x = self.feature_maps_to_wav(
        #         input_tensor=x,
        #         sp=repeat_total_mag,
        #         sin_in=repeat_total_sin,
        #         cos_in=repeat_total_cos,
        #         audio_length=audio_length,
        #     )
        #     # (bs * n, segment_samples)

        #     agent_waveform = rearrange(x, '(b n) t -> b n t', b=batch_size)
        #     # (bs, n=max_agents_contain_waveform, segment_samples)

        #     output_dict['agent_waveform'] = agent_waveform

        return output_dict


    def tile_omni(self, x, rays_num): 
        batch_size, _, _T, _F = x.shape
        return torch.tile(x[:, None, :, :, :], (1, rays_num, 1, 1, 1)).reshape(batch_size * rays_num, -1, _T, _F)


class Model01_bak4p7(nn.Module, Base):
    def __init__(self, microphones_num, classes_num, do_localization=None, do_sed=None, do_separation=None):
        super(Model01_bak4p7, self).__init__() 

        window_size = 512
        hop_size = 240
        center = True
        pad_mode = "reflect"
        window = "hann" 
        activation = "relu"
        momentum = 0.01
        classes_num = 1

        self.window_size = window_size
        self.hop_size = hop_size
        self.pad_mode = pad_mode

        positional_embedding_factor = 5
        self.time_downsample_ratio = 2 ** 3

        self.do_localization = True
        self.do_sed = False
        self.do_separation = False

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
            in_channels=microphones_num + 1,
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
            bias=True
        )

        self.phase_fc = nn.Conv2d(
            in_channels=microphones_num * (microphones_num - 1), 
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
            in_features=microphones_num * (positional_embedding_factor * 6), 
            out_features=128, 
            bias=True
        )

        self.mic_look_direction_fc = nn.Linear(
            in_features=microphones_num * (positional_embedding_factor * 4), 
            out_features=128, 
            bias=True
        )

        # self.output_angle_fc = nn.Linear(in_features=(positional_embedding_factor * 4), out_features=128, bias=True)
        # self.output_pos_xyz_fc = nn.Linear(in_features=(positional_embedding_factor * 6), out_features=128, bias=True)

        self.conv_block1 = ConvBlockAfter(in_channels=1536, out_channels=1024)
        self.conv_block2 = ConvBlockAfter(in_channels=1024, out_channels=1024)

        # # self.fc_final_tracking = nn.Linear(1024, 1, bias=True)
        self.fc_final_cla = nn.Linear(1024, classes_num, bias=True)

        # # Sep
        # self.fc_inter = nn.Linear(in_features=1536, out_features=128 * 32, bias=True)

        # self.dec_block1 = DecoderBlockRes1B(in_channels=128, out_channels=128, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        # self.dec_block2 = DecoderBlockRes1B(in_channels=128, out_channels=64, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        # self.dec_block3 = DecoderBlockRes1B(in_channels=64, out_channels=32, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        # self.dec_final = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

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

        if self.do_localization:
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

    def init_weights(self):

        init_layer(self.mag_fc)
        init_layer(self.phase_fc)
        init_layer(self.mic_signal_fc_reshape)

        init_layer(self.mic_position_fc)
        init_layer(self.mic_look_direction_fc)

        # init_layer(self.output_angle_fc)
        # init_layer(self.output_pos_xyz_fc)

        # init_layer(self.fc_inter)
        # # init_layer(self.fc_final_tracking)
        init_layer(self.fc_final_cla)

        # init_layer(self.dec_final)

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
            input_tensor: (batch_size, target_sources_num * output_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, input_channels, time_steps, freq_bins)
            sin_in: (batch_size, input_channels, time_steps, freq_bins)
            cos_in: (batch_size, input_channels, time_steps, freq_bins)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (batch_size, target_sources_num * output_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.output_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, output_channels, self.K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        # linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # mask_cos, mask_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)
        # out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        # out_mag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size * target_sources_num * output_channels
        shape = (
            batch_size * self.target_sources_num * self.output_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * output_channels, segments_num)

        # Reshape.
        waveform = x.reshape(
            batch_size, self.target_sources_num * self.output_channels, audio_length
        )
        # (batch_size, target_sources_num * output_channels, segments_num)

        return waveform

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

    def convert_position_to_embedding(self, position):
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

    def forward(self, data_dict, do_sep=True):

        self.eps = 1e-10

        mic_position = data_dict['mic_position']    # (bs, mics_num, frames_num, 3)
        mic_look_direction = data_dict['mic_look_direction']    # (bs, mics_num, frames_num, 3)
        mic_waveform = data_dict['mic_waveform']    # (bs, mics_num, samples_num)
        agent_position = data_dict['agent_position']    # (bs, agents_num, frames_num, 3)
        agent_look_direction = data_dict['agent_look_direction']    # (bs, agents_num, frames_num, 3)

        # ------ 1. Convert positions and look directions to embeddings ------
        mic_position_emb = self.convert_position_to_embedding(position=mic_position)
        # shape: (bs, mics_num, frames_num, emb_size)

        mic_look_direction_emb = self.convert_look_direction_to_embedding(
            look_direction=mic_look_direction,
        )   # shape: (bs, mics_num, frames_num, emb_size)
        
        agent_position_emb = self.convert_position_to_embedding(position=agent_position)
        # shape: (bs, agents_num, frames_num, emb_size)

        agent_look_direction_emb = self.convert_look_direction_to_embedding(
            look_direction=agent_look_direction,
        )   # shape: (bs, agents_num, frames_num, emb_size)

        # ------ 2. Extract features ------
        total_waveform = torch.sum(mic_waveform, dim=1, keepdim=True)
        # (bs, 1, samples_num)

        total_mag, total_cos, total_sin = self.wav_to_spectrogram_phase(total_waveform, self.eps)
        # (bs, 1, frames_num, freq_bins)

        mic_mag, mic_cos, mic_sin = self.wav_to_spectrogram_phase(mic_waveform, self.eps)
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
        mic_position_emb = F.pad(mic_position_emb, pad=(0, 0, 0, pad_len))
        mic_look_direction_emb = F.pad(mic_look_direction_emb, pad=(0, 0, 0, pad_len))
        agent_position_emb = F.pad(agent_position_emb, pad=(0, 0, 0, pad_len))
        agent_look_direction_emb = F.pad(agent_look_direction_emb, pad=(0, 0, 0, pad_len))

        # ------ 3. Forward data to the network ------

        # 1) Transform mic signal.
        enc1_pool, enc1 = self.mic_signal_encoder_block1(x)
        enc2_pool, enc2 = self.mic_signal_encoder_block2(enc1_pool)
        enc3_pool, enc3 = self.mic_signal_encoder_block3(enc2_pool)
        # enc3_pool: (bs, 128, T=38, F=32)

        mic_signal_feature = rearrange(enc3_pool, 'b c t f -> b t (c f)')
        # (bs, T, C*F=4096)

        mic_signal_feature = F.leaky_relu_(
            self.mic_signal_fc_reshape(mic_signal_feature), negative_slope=0.01,
        )  # shape: (bs, T, 1024)
        
        # 2) Transform mic position and look direction embeddings.
        mic_position_feature = rearrange(mic_position_emb, 'b c t f -> b t (c f)')
        # shape: (bs, T, mics_num*emb_size=120)

        mic_position_feature = F.leaky_relu_(
            self.mic_position_fc(mic_position_feature)
        )  # shape: (bs, T, 128)

        mic_position_feature = mic_position_feature[:, 0 :: self.time_downsample_ratio, :]
        
        mic_look_direction_feature = rearrange(mic_look_direction_emb, 'b c t f -> b t (c f)')
        # shape: (bs, T, mics_num*emb_size=80)

        mic_look_direction_feature = F.leaky_relu_(
            self.mic_look_direction_fc(mic_look_direction_feature)
        )  # shape: (bs, T, 128)

        mic_look_direction_feature = mic_look_direction_feature[:, 0 :: self.time_downsample_ratio, :]

        # 3) Concatenate mic signal, position, and look direction features.
        mic_feature = torch.cat((
            mic_signal_feature, mic_position_feature, mic_look_direction_feature
        ), dim=-1)  # (batch_size, T, 1280)

        agents_num = agent_position.shape[1]

        mic_feature = torch.tile(mic_feature[:, None, :, :], (1, agents_num, 1, 1))
        # shape: (bs, agents_num, T=38, 1280)

        # x = mic_feature
        # ray_direction_emb = agent_look_direction_emb
        # ray_pos_xyz_emb = agent_position_emb
        # origin_len = frames_num

        # # output angle FC
        # a3 = F.leaky_relu_(self.output_angle_fc(ray_direction_emb))
        # a3 = a3[:, :, 0 :: self.time_downsample_ratio, :]
        # # a3 = torch.tile(a3[:, :, None, :], (1, 1, frames_num, 1))
        # a3_xyz = F.leaky_relu_(self.output_pos_xyz_fc(ray_pos_xyz_emb))
        # a3_xyz = a3_xyz[:, :, 0 :: self.time_downsample_ratio, :]
        # # a3_xyz = torch.tile(a3_xyz[:, :, None, :], (1, 1, frames_num, 1))
        # # from IPython import embed; embed(using=False); os._exit(0)
        # inter_emb = torch.cat((x, a3, a3_xyz), dim=-1)
        # # (batch_size, outputs_num, T, C * 3)
        
        # batch_size, rays_num, _T, _C = inter_emb.shape

        # x = inter_emb.reshape(batch_size * rays_num, _T, _C)
        # # (batch_size * outputs_num, T, C * 3)

        # x = x.transpose(1, 2)[:, :, :, None]
        # x = self.conv_block1(x)
        # x = self.conv_block2(x)
        # x = x[:, :, :, 0].transpose(1, 2)

        # x = x.reshape(batch_size, rays_num, _T, x.shape[-1])
        # # (batch_size, outputs_num, T, C * 2)

        # output_dict = {}

        # if True:
        #     # tracking_output = torch.sigmoid(self.fc_final_tracking(x)).flatten(2)
        #     # output_dict['ray_intersect_source'] = torch.mean(tracking_output, dim=-1)
        #     cla_output = torch.sigmoid(self.fc_final_cla(x))
        #     # cla_output = cla_output.repeat(1, 1, self.time_downsample_ratio, 1)
        #     # from IPython import embed; embed(using=False); os._exit(0)

        #     _bs, _rays_num, _T, _C = cla_output.shape
        #     tmp = cla_output.reshape(_bs * _rays_num, _T, _C)
        #     tmp = interpolate(tmp, self.time_downsample_ratio)
        #     cla_output = tmp.reshape(_bs, _rays_num, -1, _C)

        #     cla_output = cla_output[:, :, 0 : origin_len, :].flatten(2)
        #     output_dict['agent_see_source'] = cla_output

        # # if do_sep:
        # if False:
        #     max_rays_contain_waveform = data_dict['max_rays_contain_waveform']
        #     # from IPython import embed; embed(using=False); os._exit(0)

        #     x = F.leaky_relu_(self.fc_inter(inter_emb[:, 0 : max_rays_contain_waveform, :, :]))

        #     batch_size, sep_rays_num, _T, _C = x.shape
        #     x = x.reshape(batch_size * sep_rays_num, _T, 128, 64)
        #     x = x.permute(0, 2, 1, 3)

        #     enc1 = self.tile_omni(x=enc1, rays_num=max_rays_contain_waveform)
        #     enc2 = self.tile_omni(x=enc2, rays_num=max_rays_contain_waveform)
        #     enc3 = self.tile_omni(x=enc3, rays_num=max_rays_contain_waveform)

        #     # from IPython import embed; embed(using=False); os._exit(0)
        #     x = self.dec_block1(x, enc3)
        #     x = self.dec_block2(x, enc2)
        #     x = self.dec_block3(x, enc1)
        #     x = self.dec_final(x)

        #     # Recover shape
        #     x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257. 

        #     x = x[:, :, 0:origin_len, :]

        #     audio_length = omni_waveform.shape[2]

        #     # from IPython import embed; embed(using=False); os._exit(0)
        #     # import matplotlib.pyplot as plt
        #     # plt.matshow(np.log(omni_mag.data.cpu().numpy()[0, 0]).T, origin='lower', aspect='auto', cmap='jet')
        #     # plt.savefig('_zz.pdf')
        #     # import soundfile
        #     # soundfile.write(file='_zz.wav', data=omni_waveform.data.cpu().numpy()[0, 0], samplerate=24000)

        #     omni_mag0 = self.tile_omni(omni_mag, max_rays_contain_waveform)
        #     omni_cos0 = self.tile_omni(omni_cos, max_rays_contain_waveform)
        #     omni_sin0 = self.tile_omni(omni_sin, max_rays_contain_waveform)

        #     separated_audio = self.feature_maps_to_wav(
        #         input_tensor=x,
        #         sp=omni_mag0,
        #         sin_in=omni_sin0,
        #         cos_in=omni_cos0,
        #         audio_length=audio_length,
        #     )

        #     separated_audio = separated_audio.reshape(batch_size, sep_rays_num, -1)
        #     # from IPython import embed; embed(using=False); os._exit(0)

        #     output_dict['ray_waveform'] = separated_audio

        # return output_dict

        # 4) Calculate agent position and look direction features.
        agent_position_feature = F.leaky_relu_(self.agent_position_fc(agent_position_emb))
        # shape: (bs, agents_num, T, 128)

        agent_position_feature = agent_position_feature[:, :, 0 :: self.time_downsample_ratio, :]
        # shape: (bs, agents_num, T=38, 128)

        agent_look_direction_feature = F.leaky_relu_(
            self.agent_look_direction_fc(agent_look_direction_emb)
        )  # (bs, agents_num, T, 128)

        agent_look_direction_feature = agent_look_direction_feature[:, :, 0 :: self.time_downsample_ratio, :]
        # shape: (bs, agents_num, T=38, 128)

        # Concatenate mic features and agent features.
        shared_feature = torch.cat((mic_feature, agent_position_feature, 
            agent_look_direction_feature), dim=-1)
        # (bs, agents_num, T=38, 1536)

        batch_size = shared_feature.shape[0]

        x = rearrange(shared_feature, 'b n t c -> (b n) c t')[:, :, :, None]
        # y2 = rearrange(shared_feature, 'b n t c -> (b n) c t 1') 
        # x = rearrange(shared_feature, 'b n t c -> (b n) c t 1')
        # (bs * agents_num, C=1536, T, 1)

        x = self.loc_fc_block1(x)
        # (bs * agents_num, 1024, T, 1)

        x = self.loc_fc_block2(x)
        # (bs * agents_num, 1024, T, 1)

        # x = rearrange(x, '(b n) c t 1 -> (b n) t c', b=batch_size)
        x = rearrange(x.flatten(2), '(b n) c t -> (b n) t c', b=batch_size)
        # (bs * agents_num, T, 1024)

        output_dict = {}

        if True:
            x = rearrange(x, '(b n) t c -> b n t c', b=batch_size)
            origin_len = frames_num
            # tracking_output = torch.sigmoid(self.fc_final_tracking(x)).flatten(2)
            # output_dict['ray_intersect_source'] = torch.mean(tracking_output, dim=-1)
            cla_output = torch.sigmoid(self.fc_final_cla(x))
            # cla_output = cla_output.repeat(1, 1, self.time_downsample_ratio, 1)
            # from IPython import embed; embed(using=False); os._exit(0)

            _bs, _rays_num, _T, _C = cla_output.shape
            tmp = cla_output.reshape(_bs * _rays_num, _T, _C)
            tmp = interpolate(tmp, self.time_downsample_ratio)
            cla_output = tmp.reshape(_bs, _rays_num, -1, _C)

            cla_output = cla_output[:, :, 0 : origin_len, :].flatten(2)
            output_dict['agent_see_source'] = cla_output

        # if self.do_localization:
        #     batch_size, agents_num, _T, _C = shared_feature.shape

        #     x = rearrange(shared_feature, 'b n t c -> (b n) c t 1')
        #     # (bs * agents_num, C=1536, T, 1)

        #     x = self.loc_fc_block1(x)
        #     # (bs * agents_num, 1024, T, 1)

        #     x = self.loc_fc_block2(x)
        #     # (bs * agents_num, 1024, T, 1)

        #     x = rearrange(x, '(b n) c t 1 -> (b n) t c', b=batch_size)
        #     # (bs * agents_num, T, 1024)

        #     x = torch.sigmoid(self.loc_fc_final(x))
        #     # (bs * agents_num, T=38, C=1)

        #     x = interpolate(x, self.time_downsample_ratio)[:, 0 : frames_num, :]
        #     # (bs * agents_num, T=301, C=1)

        #     agent_see_source = rearrange(x, '(b n) t 1 -> b n t', b=batch_size)
        #     # (bs, agents_num, T=301)

        #     output_dict['agent_see_source'] = agent_see_source

        # if self.do_separation:

        #     max_agents_contain_waveform = data_dict['max_agents_contain_waveform']

        #     x = shared_feature[:, 0 : max_agents_contain_waveform, :, :]
        #     # (bs, n=max_agents_contain_waveform, T=38, C=1536)

        #     x = F.leaky_relu_(self.sep_fc_reshape(x))
        #     # (bs, n, T=38, 4096)

        #     x = rearrange(x, 'b n t (c f) -> (b n) c t f', c=128, f=32)
        #     # (bs * n, C=128, T=38, F=32)

        #     enc1 = self.repeat_conv_features(x=enc1, repeats_num=max_agents_contain_waveform)
        #     # (bs * n, C=32, T=304, F=256)

        #     enc2 = self.repeat_conv_features(x=enc2, repeats_num=max_agents_contain_waveform)
        #     # (bs * n, C=64, T=152, F=128)

        #     enc3 = self.repeat_conv_features(x=enc3, repeats_num=max_agents_contain_waveform)
        #     # (bs * n, C=128, T=76, F=64)

        #     x = self.sep_decoder_block1(x, enc3)
        #     x = self.sep_decoder_block2(x, enc2)
        #     x = self.sep_decoder_block3(x, enc1)
        #     # (bs * n, C=32, T=304, F=256)

        #     x = self.sep_conv_final(x)
        #     # (bs * n, C=3, T=304, F=256)

        #     # Recover shape
        #     x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257. 
        #     # (bs * n, C=3, T=304, F=257)

        #     x = x[:, :, 0:frames_num, :]
        #     # (bs * n, C=3, T=301, F=257)

        #     repeat_total_mag = self.repeat_conv_features(
        #         x=total_mag, repeats_num=max_agents_contain_waveform
        #     )  # (bs * n, C=1, T=301, F=257)

        #     repeat_total_sin = self.repeat_conv_features(
        #         x=total_sin, repeats_num=max_agents_contain_waveform
        #     )  # (bs * n, C=1, T=301, F=257)

        #     repeat_total_cos = self.repeat_conv_features(
        #         x=total_cos, repeats_num=max_agents_contain_waveform
        #     )  # (bs * n, C=1, T=301, F=257)

        #     audio_length = total_waveform.shape[-1]

        #     x = self.feature_maps_to_wav(
        #         input_tensor=x,
        #         sp=repeat_total_mag,
        #         sin_in=repeat_total_sin,
        #         cos_in=repeat_total_cos,
        #         audio_length=audio_length,
        #     )
        #     # (bs * n, segment_samples)

        #     agent_waveform = rearrange(x, '(b n) t -> b n t', b=batch_size)
        #     # (bs, n=max_agents_contain_waveform, segment_samples)

        #     output_dict['agent_waveform'] = agent_waveform

        return output_dict


    def tile_omni(self, x, rays_num): 
        batch_size, _, _T, _F = x.shape
        return torch.tile(x[:, None, :, :, :], (1, rays_num, 1, 1, 1)).reshape(batch_size * rays_num, -1, _T, _F)


class Model01_bak5(nn.Module, Base):
    def __init__(self, microphones_num, classes_num, do_localization=None, do_sed=None, do_separation=None):
        super(Model01_bak5, self).__init__() 

        window_size = 512
        hop_size = 240
        center = True
        pad_mode = "reflect"
        window = "hann" 
        activation = "relu"
        momentum = 0.01
        classes_num = 1

        self.window_size = window_size
        self.hop_size = hop_size
        self.pad_mode = pad_mode

        positional_embedding_factor = 5
        self.time_downsample_ratio = 2 ** 3

        self.do_localization = True
        self.do_sed = False
        self.do_separation = False

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
            in_channels=microphones_num + 1,
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
            bias=True
        )

        self.phase_fc = nn.Conv2d(
            in_channels=microphones_num * (microphones_num - 1), 
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
            in_features=microphones_num * (positional_embedding_factor * 6), 
            out_features=128, 
            bias=True
        )

        self.mic_look_direction_fc = nn.Linear(
            in_features=microphones_num * (positional_embedding_factor * 4), 
            out_features=128, 
            bias=True
        )

        # self.output_angle_fc = nn.Linear(in_features=(positional_embedding_factor * 4), out_features=128, bias=True)
        # self.output_pos_xyz_fc = nn.Linear(in_features=(positional_embedding_factor * 6), out_features=128, bias=True)

        self.conv_block1 = ConvBlockAfter(in_channels=1536, out_channels=1024)
        self.conv_block2 = ConvBlockAfter(in_channels=1024, out_channels=1024)

        # # self.fc_final_tracking = nn.Linear(1024, 1, bias=True)
        # self.fc_final_cla = nn.Linear(1024, classes_num, bias=True)

        # # Sep
        # self.fc_inter = nn.Linear(in_features=1536, out_features=128 * 32, bias=True)

        # self.dec_block1 = DecoderBlockRes1B(in_channels=128, out_channels=128, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        # self.dec_block2 = DecoderBlockRes1B(in_channels=128, out_channels=64, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        # self.dec_block3 = DecoderBlockRes1B(in_channels=64, out_channels=32, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        # self.dec_final = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

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

        if self.do_localization:
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

    def init_weights(self):

        init_layer(self.mag_fc)
        init_layer(self.phase_fc)
        init_layer(self.mic_signal_fc_reshape)

        init_layer(self.mic_position_fc)
        init_layer(self.mic_look_direction_fc)

        # init_layer(self.output_angle_fc)
        # init_layer(self.output_pos_xyz_fc)

        # init_layer(self.fc_inter)
        # # init_layer(self.fc_final_tracking)
        # init_layer(self.fc_final_cla)

        # init_layer(self.dec_final)

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
            input_tensor: (batch_size, target_sources_num * output_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, input_channels, time_steps, freq_bins)
            sin_in: (batch_size, input_channels, time_steps, freq_bins)
            cos_in: (batch_size, input_channels, time_steps, freq_bins)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (batch_size, target_sources_num * output_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.output_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, output_channels, self.K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        # linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # mask_cos, mask_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)
        # out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        # out_mag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size * target_sources_num * output_channels
        shape = (
            batch_size * self.target_sources_num * self.output_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * output_channels, segments_num)

        # Reshape.
        waveform = x.reshape(
            batch_size, self.target_sources_num * self.output_channels, audio_length
        )
        # (batch_size, target_sources_num * output_channels, segments_num)

        return waveform

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

    def convert_position_to_embedding(self, position):
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

    def forward(self, data_dict, do_sep=True):

        self.eps = 1e-10

        mic_position = data_dict['mic_position']    # (bs, mics_num, frames_num, 3)
        mic_look_direction = data_dict['mic_look_direction']    # (bs, mics_num, frames_num, 3)
        mic_waveform = data_dict['mic_waveform']    # (bs, mics_num, samples_num)
        agent_position = data_dict['agent_position']    # (bs, agents_num, frames_num, 3)
        agent_look_direction = data_dict['agent_look_direction']    # (bs, agents_num, frames_num, 3)

        # ------ 1. Convert positions and look directions to embeddings ------
        mic_position_emb = self.convert_position_to_embedding(position=mic_position)
        # shape: (bs, mics_num, frames_num, emb_size)

        mic_look_direction_emb = self.convert_look_direction_to_embedding(
            look_direction=mic_look_direction,
        )   # shape: (bs, mics_num, frames_num, emb_size)
        
        agent_position_emb = self.convert_position_to_embedding(position=agent_position)
        # shape: (bs, agents_num, frames_num, emb_size)

        agent_look_direction_emb = self.convert_look_direction_to_embedding(
            look_direction=agent_look_direction,
        )   # shape: (bs, agents_num, frames_num, emb_size)

        # ------ 2. Extract features ------
        total_waveform = torch.sum(mic_waveform, dim=1, keepdim=True)
        # (bs, 1, samples_num)

        total_mag, total_cos, total_sin = self.wav_to_spectrogram_phase(total_waveform, self.eps)
        # (bs, 1, frames_num, freq_bins)

        mic_mag, mic_cos, mic_sin = self.wav_to_spectrogram_phase(mic_waveform, self.eps)
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
        mic_position_emb = F.pad(mic_position_emb, pad=(0, 0, 0, pad_len))
        mic_look_direction_emb = F.pad(mic_look_direction_emb, pad=(0, 0, 0, pad_len))
        agent_position_emb = F.pad(agent_position_emb, pad=(0, 0, 0, pad_len))
        agent_look_direction_emb = F.pad(agent_look_direction_emb, pad=(0, 0, 0, pad_len))

        # ------ 3. Forward data to the network ------

        # 1) Transform mic signal.
        enc1_pool, enc1 = self.mic_signal_encoder_block1(x)
        enc2_pool, enc2 = self.mic_signal_encoder_block2(enc1_pool)
        enc3_pool, enc3 = self.mic_signal_encoder_block3(enc2_pool)
        # enc3_pool: (bs, 128, T=38, F=32)

        mic_signal_feature = rearrange(enc3_pool, 'b c t f -> b t (c f)')
        # (bs, T, C*F=4096)

        mic_signal_feature = F.leaky_relu_(
            self.mic_signal_fc_reshape(mic_signal_feature), negative_slope=0.01,
        )  # shape: (bs, T, 1024)
        
        # 2) Transform mic position and look direction embeddings.
        mic_position_feature = rearrange(mic_position_emb, 'b c t f -> b t (c f)')
        # shape: (bs, T, mics_num*emb_size=120)

        mic_position_feature = F.leaky_relu_(
            self.mic_position_fc(mic_position_feature)
        )  # shape: (bs, T, 128)

        mic_position_feature = mic_position_feature[:, 0 :: self.time_downsample_ratio, :]
        
        mic_look_direction_feature = rearrange(mic_look_direction_emb, 'b c t f -> b t (c f)')
        # shape: (bs, T, mics_num*emb_size=80)

        mic_look_direction_feature = F.leaky_relu_(
            self.mic_look_direction_fc(mic_look_direction_feature)
        )  # shape: (bs, T, 128)

        mic_look_direction_feature = mic_look_direction_feature[:, 0 :: self.time_downsample_ratio, :]

        # 3) Concatenate mic signal, position, and look direction features.
        mic_feature = torch.cat((
            mic_signal_feature, mic_position_feature, mic_look_direction_feature
        ), dim=-1)  # (batch_size, T, 1280)

        agents_num = agent_position.shape[1]

        mic_feature = torch.tile(mic_feature[:, None, :, :], (1, agents_num, 1, 1))
        # shape: (bs, agents_num, T=38, 1280)

        # x = mic_feature
        # ray_direction_emb = agent_look_direction_emb
        # ray_pos_xyz_emb = agent_position_emb
        # origin_len = frames_num

        # # output angle FC
        # a3 = F.leaky_relu_(self.output_angle_fc(ray_direction_emb))
        # a3 = a3[:, :, 0 :: self.time_downsample_ratio, :]
        # # a3 = torch.tile(a3[:, :, None, :], (1, 1, frames_num, 1))
        # a3_xyz = F.leaky_relu_(self.output_pos_xyz_fc(ray_pos_xyz_emb))
        # a3_xyz = a3_xyz[:, :, 0 :: self.time_downsample_ratio, :]
        # # a3_xyz = torch.tile(a3_xyz[:, :, None, :], (1, 1, frames_num, 1))
        # # from IPython import embed; embed(using=False); os._exit(0)
        # inter_emb = torch.cat((x, a3, a3_xyz), dim=-1)
        # # (batch_size, outputs_num, T, C * 3)
        
        # batch_size, rays_num, _T, _C = inter_emb.shape

        # x = inter_emb.reshape(batch_size * rays_num, _T, _C)
        # # (batch_size * outputs_num, T, C * 3)

        # x = x.transpose(1, 2)[:, :, :, None]
        # x = self.conv_block1(x)
        # x = self.conv_block2(x)
        # x = x[:, :, :, 0].transpose(1, 2)

        # x = x.reshape(batch_size, rays_num, _T, x.shape[-1])
        # # (batch_size, outputs_num, T, C * 2)

        # output_dict = {}

        # if True:
        #     # tracking_output = torch.sigmoid(self.fc_final_tracking(x)).flatten(2)
        #     # output_dict['ray_intersect_source'] = torch.mean(tracking_output, dim=-1)
        #     cla_output = torch.sigmoid(self.fc_final_cla(x))
        #     # cla_output = cla_output.repeat(1, 1, self.time_downsample_ratio, 1)
        #     # from IPython import embed; embed(using=False); os._exit(0)

        #     _bs, _rays_num, _T, _C = cla_output.shape
        #     tmp = cla_output.reshape(_bs * _rays_num, _T, _C)
        #     tmp = interpolate(tmp, self.time_downsample_ratio)
        #     cla_output = tmp.reshape(_bs, _rays_num, -1, _C)

        #     cla_output = cla_output[:, :, 0 : origin_len, :].flatten(2)
        #     output_dict['agent_see_source'] = cla_output

        # # if do_sep:
        # if False:
        #     max_rays_contain_waveform = data_dict['max_rays_contain_waveform']
        #     # from IPython import embed; embed(using=False); os._exit(0)

        #     x = F.leaky_relu_(self.fc_inter(inter_emb[:, 0 : max_rays_contain_waveform, :, :]))

        #     batch_size, sep_rays_num, _T, _C = x.shape
        #     x = x.reshape(batch_size * sep_rays_num, _T, 128, 64)
        #     x = x.permute(0, 2, 1, 3)

        #     enc1 = self.tile_omni(x=enc1, rays_num=max_rays_contain_waveform)
        #     enc2 = self.tile_omni(x=enc2, rays_num=max_rays_contain_waveform)
        #     enc3 = self.tile_omni(x=enc3, rays_num=max_rays_contain_waveform)

        #     # from IPython import embed; embed(using=False); os._exit(0)
        #     x = self.dec_block1(x, enc3)
        #     x = self.dec_block2(x, enc2)
        #     x = self.dec_block3(x, enc1)
        #     x = self.dec_final(x)

        #     # Recover shape
        #     x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257. 

        #     x = x[:, :, 0:origin_len, :]

        #     audio_length = omni_waveform.shape[2]

        #     # from IPython import embed; embed(using=False); os._exit(0)
        #     # import matplotlib.pyplot as plt
        #     # plt.matshow(np.log(omni_mag.data.cpu().numpy()[0, 0]).T, origin='lower', aspect='auto', cmap='jet')
        #     # plt.savefig('_zz.pdf')
        #     # import soundfile
        #     # soundfile.write(file='_zz.wav', data=omni_waveform.data.cpu().numpy()[0, 0], samplerate=24000)

        #     omni_mag0 = self.tile_omni(omni_mag, max_rays_contain_waveform)
        #     omni_cos0 = self.tile_omni(omni_cos, max_rays_contain_waveform)
        #     omni_sin0 = self.tile_omni(omni_sin, max_rays_contain_waveform)

        #     separated_audio = self.feature_maps_to_wav(
        #         input_tensor=x,
        #         sp=omni_mag0,
        #         sin_in=omni_sin0,
        #         cos_in=omni_cos0,
        #         audio_length=audio_length,
        #     )

        #     separated_audio = separated_audio.reshape(batch_size, sep_rays_num, -1)
        #     # from IPython import embed; embed(using=False); os._exit(0)

        #     output_dict['ray_waveform'] = separated_audio

        # return output_dict

        # 4) Calculate agent position and look direction features.
        agent_position_feature = F.leaky_relu_(self.agent_position_fc(agent_position_emb))
        # shape: (bs, agents_num, T, 128)

        agent_position_feature = agent_position_feature[:, :, 0 :: self.time_downsample_ratio, :]
        # shape: (bs, agents_num, T=38, 128)

        agent_look_direction_feature = F.leaky_relu_(
            self.agent_look_direction_fc(agent_look_direction_emb)
        )  # (bs, agents_num, T, 128)

        agent_look_direction_feature = agent_look_direction_feature[:, :, 0 :: self.time_downsample_ratio, :]
        # shape: (bs, agents_num, T=38, 128)

        # Concatenate mic features and agent features.
        shared_feature = torch.cat((mic_feature, agent_position_feature, 
            agent_look_direction_feature), dim=-1)
        # (bs, agents_num, T=38, 1536)

        output_dict = {}

        # inter_emb = shared_feature

        # batch_size, rays_num, _T, _C = inter_emb.shape

        # x = inter_emb.reshape(batch_size * rays_num, _T, _C)
        # # (batch_size * outputs_num, T, C * 3)

        # x = x.transpose(1, 2)[:, :, :, None]
        # x = self.conv_block1(x)
        # x = self.conv_block2(x)
        # x = x[:, :, :, 0].transpose(1, 2)

        # x = x.reshape(batch_size, rays_num, _T, x.shape[-1])
        # # (batch_size, outputs_num, T, C * 2)

        # output_dict = {}

        # if True:
        #     origin_len = frames_num
        #     # tracking_output = torch.sigmoid(self.fc_final_tracking(x)).flatten(2)
        #     # output_dict['ray_intersect_source'] = torch.mean(tracking_output, dim=-1)
        #     cla_output = torch.sigmoid(self.fc_final_cla(x))
        #     # cla_output = cla_output.repeat(1, 1, self.time_downsample_ratio, 1)
        #     # from IPython import embed; embed(using=False); os._exit(0)

        #     _bs, _rays_num, _T, _C = cla_output.shape
        #     tmp = cla_output.reshape(_bs * _rays_num, _T, _C)
        #     tmp = interpolate(tmp, self.time_downsample_ratio)
        #     cla_output = tmp.reshape(_bs, _rays_num, -1, _C)

        #     cla_output = cla_output[:, :, 0 : origin_len, :].flatten(2)
        #     output_dict['agent_see_source'] = cla_output

        if self.do_localization:
            
            batch_size = shared_feature.shape[0]

            x = rearrange(shared_feature, 'b n t c -> (b n) c t 1')
            # (bs * agents_num, C=1536, T, 1)

            x = self.loc_fc_block1(x)
            # (bs * agents_num, 1024, T, 1)

            x = self.loc_fc_block2(x)
            # (bs * agents_num, 1024, T, 1)

            x = rearrange(x, '(b n) c t 1 -> (b n) t c', b=batch_size)
            # (bs * agents_num, T, 1024)

            x = torch.sigmoid(self.loc_fc_final(x))
            # (bs * agents_num, T=38, C=1)

            x = x.repeat_interleave(repeats=self.time_downsample_ratio, dim=1)[:, 0 : frames_num, :]
            # x = interpolate(x, self.time_downsample_ratio)[:, 0 : frames_num, :]
            # (bs * agents_num, T=301, C=1)

            agent_see_source = rearrange(x, '(b n) t 1 -> b n t', b=batch_size)
            # (bs, agents_num, T=301)

            output_dict['agent_see_source'] = agent_see_source

        # if self.do_separation:

        #     max_agents_contain_waveform = data_dict['max_agents_contain_waveform']

        #     x = shared_feature[:, 0 : max_agents_contain_waveform, :, :]
        #     # (bs, n=max_agents_contain_waveform, T=38, C=1536)

        #     x = F.leaky_relu_(self.sep_fc_reshape(x))
        #     # (bs, n, T=38, 4096)

        #     x = rearrange(x, 'b n t (c f) -> (b n) c t f', c=128, f=32)
        #     # (bs * n, C=128, T=38, F=32)

        #     enc1 = self.repeat_conv_features(x=enc1, repeats_num=max_agents_contain_waveform)
        #     # (bs * n, C=32, T=304, F=256)

        #     enc2 = self.repeat_conv_features(x=enc2, repeats_num=max_agents_contain_waveform)
        #     # (bs * n, C=64, T=152, F=128)

        #     enc3 = self.repeat_conv_features(x=enc3, repeats_num=max_agents_contain_waveform)
        #     # (bs * n, C=128, T=76, F=64)

        #     x = self.sep_decoder_block1(x, enc3)
        #     x = self.sep_decoder_block2(x, enc2)
        #     x = self.sep_decoder_block3(x, enc1)
        #     # (bs * n, C=32, T=304, F=256)

        #     x = self.sep_conv_final(x)
        #     # (bs * n, C=3, T=304, F=256)

        #     # Recover shape
        #     x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257. 
        #     # (bs * n, C=3, T=304, F=257)

        #     x = x[:, :, 0:frames_num, :]
        #     # (bs * n, C=3, T=301, F=257)

        #     repeat_total_mag = self.repeat_conv_features(
        #         x=total_mag, repeats_num=max_agents_contain_waveform
        #     )  # (bs * n, C=1, T=301, F=257)

        #     repeat_total_sin = self.repeat_conv_features(
        #         x=total_sin, repeats_num=max_agents_contain_waveform
        #     )  # (bs * n, C=1, T=301, F=257)

        #     repeat_total_cos = self.repeat_conv_features(
        #         x=total_cos, repeats_num=max_agents_contain_waveform
        #     )  # (bs * n, C=1, T=301, F=257)

        #     audio_length = total_waveform.shape[-1]

        #     x = self.feature_maps_to_wav(
        #         input_tensor=x,
        #         sp=repeat_total_mag,
        #         sin_in=repeat_total_sin,
        #         cos_in=repeat_total_cos,
        #         audio_length=audio_length,
        #     )
        #     # (bs * n, segment_samples)

        #     agent_waveform = rearrange(x, '(b n) t -> b n t', b=batch_size)
        #     # (bs, n=max_agents_contain_waveform, segment_samples)

        #     output_dict['agent_waveform'] = agent_waveform

        return output_dict


    def tile_omni(self, x, rays_num): 
        batch_size, _, _T, _F = x.shape
        return torch.tile(x[:, None, :, :, :], (1, rays_num, 1, 1, 1)).reshape(batch_size * rays_num, -1, _T, _F)


'''
class Model01_win1024(nn.Module, Base):
    def __init__(self, microphones_num, classes_num):
        super(Model01_win1024, self).__init__() 

        window_size = 1024
        hop_size = 240
        center = True
        pad_mode = "reflect"
        window = "hann" 
        activation = "relu"
        momentum = 0.01

        self.window_size = window_size
        self.hop_size = hop_size
        self.pad_mode = pad_mode

        positional_embedding_factor = 5
        self.time_downsample_ratio = 2 ** 3

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

        self.positional_encoder = PositionalEncoder(factor=positional_embedding_factor)
        self.positional_encoder_room_xyz = PositionalEncoderRoomXYZ(factor=positional_embedding_factor)

        self.pre_value_cnn = nn.Conv2d(in_channels=microphones_num + 1, out_channels=32,
                              kernel_size=(1, 1), stride=(1, 1),
                              padding=(0, 0), bias=True)
        self.pre_angle_cnn = nn.Conv2d(in_channels=12, out_channels=32,
                              kernel_size=(1, 1), stride=(1, 1),
                              padding=(0, 0), bias=True)

        self.input_value_cnn1 = EncoderBlockRes1B(in_channels=64, out_channels=32, kernel_size=(3, 3), downsample=(2, 2), momentum=momentum)

        self.input_value_cnn2 = EncoderBlockRes1B(in_channels=32, out_channels=64, kernel_size=(3, 3), downsample=(2, 2), momentum=momentum)

        self.input_value_cnn3 = EncoderBlockRes1B(in_channels=64, out_channels=128, kernel_size=(3, 3), downsample=(2, 2), momentum=momentum)

        # self.input_value_cnn4 = EncoderBlockRes1B(in_channels=128, out_channels=256, kernel_size=(3, 3), downsample=(2, 2), momentum=momentum)

        self.input_value_fc1 = nn.Linear(in_features=64 * 128, out_features=1024)

        self.input_angle_fc = nn.Linear(in_features=(positional_embedding_factor * 4) * microphones_num, out_features=128, bias=True)
        self.input_pos_xyz_fc = nn.Linear(in_features=(positional_embedding_factor * 6) * microphones_num, out_features=128, bias=True)

        self.output_angle_fc = nn.Linear(in_features=(positional_embedding_factor * 4), out_features=128, bias=True)
        self.output_pos_xyz_fc = nn.Linear(in_features=(positional_embedding_factor * 6), out_features=128, bias=True)

        self.conv_block1 = ConvBlockAfter(in_channels=1536, out_channels=1024)
        self.conv_block2 = ConvBlockAfter(in_channels=1024, out_channels=1024)

        # self.fc_final_tracking = nn.Linear(1024, 1, bias=True)
        self.fc_final_cla = nn.Linear(1024, classes_num, bias=True)

        # Sep
        self.fc_inter = nn.Linear(in_features=1536, out_features=128 * 64, bias=True)

        self.dec_block1 = DecoderBlockRes1B(in_channels=128, out_channels=128, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        self.dec_block2 = DecoderBlockRes1B(in_channels=128, out_channels=64, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        self.dec_block3 = DecoderBlockRes1B(in_channels=64, out_channels=32, kernel_size=(3, 3), upsample=(2, 2), momentum=momentum)
        self.dec_final = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.init_weights()

        self.target_sources_num = 1
        self.output_channels = 1
        self.K = 3

    def init_weights(self):

        init_layer(self.pre_value_cnn)
        init_layer(self.pre_angle_cnn)
        init_layer(self.input_value_fc1)

        init_layer(self.input_angle_fc)
        init_layer(self.input_pos_xyz_fc)
        init_layer(self.output_angle_fc)
        init_layer(self.output_pos_xyz_fc)

        init_layer(self.fc_inter)
        # init_layer(self.fc_final_tracking)
        init_layer(self.fc_final_cla)

        init_layer(self.dec_final)

    def pad(self, x, window_size, hop_size):
        # To make show the center of fft window is the center of a frame
        x = F.pad(x, pad=(window_size // 2 - hop_size // 2, window_size // 2 - hop_size // 2), mode=self.pad_mode)
        return x

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
            input_tensor: (batch_size, target_sources_num * output_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, input_channels, time_steps, freq_bins)
            sin_in: (batch_size, input_channels, time_steps, freq_bins)
            cos_in: (batch_size, input_channels, time_steps, freq_bins)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (batch_size, target_sources_num * output_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.output_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, output_channels, self.K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        # linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # mask_cos, mask_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)
        # out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        # out_mag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size * target_sources_num * output_channels
        shape = (
            batch_size * self.target_sources_num * self.output_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * output_channels, segments_num)

        # Reshape.
        waveform = x.reshape(
            batch_size, self.target_sources_num * self.output_channels, audio_length
        )
        # (batch_size, target_sources_num * output_channels, segments_num)

        return waveform


    def forward(self, data_dict, do_sep=True):

        eps = 1e-10

        mic_position = data_dict['mic_position']
        mic_look_direction = data_dict['mic_look_direction']
        mic_waveform = data_dict['mic_waveform']
        ray_direction = data_dict['ray_direction']
        ray_position = data_dict['ray_origin']

        # Mic direction embedding
        _, mic_look_azimuth, mic_look_colatitude = cart2sph_torch(
            x=mic_look_direction[:, :, :, 0], 
            y=mic_look_direction[:, :, :, 1], 
            z=mic_look_direction[:, :, :, 2],
        )
        mic_direction_emb = self.positional_encoder(
            azimuth=mic_look_azimuth,
            elevation=mic_look_colatitude,
        )
        # (batch_size, mics_num, outputs_num)

        # ray direction embedding
        _, ray_di_azimuth, ray_di_zenith = cart2sph_torch(
            x=ray_direction[:, :, :, 0], 
            y=ray_direction[:, :, :, 1], 
            z=ray_direction[:, :, :, 2],
        )
        ray_direction_emb = self.positional_encoder(
            azimuth=ray_di_azimuth,
            elevation=ray_di_zenith,
        )
        # (batch_size, mics_num, outputs_num)

        mic_pos_xyz_emb = self.positional_encoder_room_xyz(
            x=mic_position[:, :, :, 0],
            y=mic_position[:, :, :, 1],
            z=mic_position[:, :, :, 2],
        )
        
        ray_pos_xyz_emb = self.positional_encoder_room_xyz(
            x=ray_position[:, :, :, 0],
            y=ray_position[:, :, :, 1],
            z=ray_position[:, :, :, 2],
        )

        omni_waveform = torch.sum(mic_waveform, dim=1, keepdim=True)
        omni_mag, omni_cos, omni_sin = self.wav_to_spectrogram_phase(omni_waveform, eps)
        mic_mag, mic_cos, mic_sin = self.wav_to_spectrogram_phase(mic_waveform, eps) # (batch_size, 1, L, F)
        # (batch_size, 1, L, F)

        omni_real = omni_mag * omni_cos
        omni_imag = omni_mag * omni_sin

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

        delta_real = omni_mag * delta_cos
        delta_imag = omni_mag * delta_sin

        x1 = torch.cat((omni_mag, mic_mag), dim=1)
        x2 = torch.cat((delta_cos, delta_sin), dim=1)

        x1 = self.pre_value_cnn(x1)
        x2 = self.pre_angle_cnn(x2)
        
        x = torch.cat((x1, x2), dim=1)

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio))
            * self.time_downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        # x: (batch_size, input_channels * subbands_num, padded_time_steps, freq_bins)
        # Let frequency bins be evenly divided by 2, e.g., 257 -> 256
        x = x[..., 0 : x.shape[-1] - 1]  # (bs, input_channels, T, F)

        mic_direction_emb = F.pad(mic_direction_emb, pad=(0, 0, 0, pad_len))
        mic_pos_xyz_emb = F.pad(mic_pos_xyz_emb, pad=(0, 0, 0, pad_len))
        ray_direction_emb = F.pad(ray_direction_emb, pad=(0, 0, 0, pad_len))
        ray_pos_xyz_emb = F.pad(ray_pos_xyz_emb, pad=(0, 0, 0, pad_len))

        # input value FC
        enc1_pool, enc1 = self.input_value_cnn1(x)
        enc2_pool, enc2 = self.input_value_cnn2(enc1_pool)
        enc3_pool, enc3 = self.input_value_cnn3(enc2_pool)
        # a1 = self.input_value_cnn4(a1)
        a1 = enc3_pool

        a1 = a1.permute(0, 2, 1, 3)
        a1 = a1.flatten(2)

        a1 = F.leaky_relu_(self.input_value_fc1(a1), negative_slope=0.01)

        # input angle FC
        a2 = mic_direction_emb.permute(0, 2, 1, 3).flatten(2)
        a2 = F.leaky_relu_(self.input_angle_fc(a2))    # (batch_size, 1, T', C)

        a2_xyz = mic_pos_xyz_emb.permute(0, 2, 1, 3).flatten(2)
        a2_xyz = F.leaky_relu_(self.input_pos_xyz_fc(a2_xyz))

        a2 = a2[:, 0 :: self.time_downsample_ratio, :]
        a2_xyz = a2_xyz[:, 0 :: self.time_downsample_ratio, :]

        x = torch.cat((a1, a2, a2_xyz), dim=-1) # (batch_size, T, C * 2)

        rays_num = ray_direction.shape[1]

        x = torch.tile(x[:, None, :, :], (1, rays_num, 1, 1))
        # (batch_size, outputs_num, T, C * 2)

        # output angle FC
        a3 = F.leaky_relu_(self.output_angle_fc(ray_direction_emb))
        a3 = a3[:, :, 0 :: self.time_downsample_ratio, :]
        # a3 = torch.tile(a3[:, :, None, :], (1, 1, frames_num, 1))
        a3_xyz = F.leaky_relu_(self.output_pos_xyz_fc(ray_pos_xyz_emb))
        a3_xyz = a3_xyz[:, :, 0 :: self.time_downsample_ratio, :]
        # a3_xyz = torch.tile(a3_xyz[:, :, None, :], (1, 1, frames_num, 1))
        # from IPython import embed; embed(using=False); os._exit(0)
        inter_emb = torch.cat((x, a3, a3_xyz), dim=-1)
        # (batch_size, outputs_num, T, C * 3)
        
        batch_size, rays_num, _T, _C = inter_emb.shape

        x = inter_emb.reshape(batch_size * rays_num, _T, _C)
        # (batch_size * outputs_num, T, C * 3)

        x = x.transpose(1, 2)[:, :, :, None]
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x[:, :, :, 0].transpose(1, 2)

        x = x.reshape(batch_size, rays_num, _T, x.shape[-1])
        # (batch_size, outputs_num, T, C * 2)

        output_dict = {}

        if True:
            # tracking_output = torch.sigmoid(self.fc_final_tracking(x)).flatten(2)
            # output_dict['ray_intersect_source'] = torch.mean(tracking_output, dim=-1)
            cla_output = torch.sigmoid(self.fc_final_cla(x))
            # cla_output = cla_output.repeat(1, 1, self.time_downsample_ratio, 1)
            # from IPython import embed; embed(using=False); os._exit(0)

            _bs, _rays_num, _T, _C = cla_output.shape
            tmp = cla_output.reshape(_bs * _rays_num, _T, _C)
            tmp = interpolate(tmp, self.time_downsample_ratio)
            cla_output = tmp.reshape(_bs, _rays_num, -1, _C)

            cla_output = cla_output[:, :, 0 : origin_len, :].flatten(2)
            output_dict['ray_intersect_source'] = cla_output

        # if do_sep:
        if True:
            max_rays_contain_waveform = data_dict['max_rays_contain_waveform']
            # from IPython import embed; embed(using=False); os._exit(0)

            x = F.leaky_relu_(self.fc_inter(inter_emb[:, 0 : max_rays_contain_waveform, :, :]))

            batch_size, sep_rays_num, _T, _C = x.shape
            x = x.reshape(batch_size * sep_rays_num, _T, 128, 64)
            x = x.permute(0, 2, 1, 3)

            enc1 = self.tile_omni(x=enc1, rays_num=max_rays_contain_waveform)
            enc2 = self.tile_omni(x=enc2, rays_num=max_rays_contain_waveform)
            enc3 = self.tile_omni(x=enc3, rays_num=max_rays_contain_waveform)

            # from IPython import embed; embed(using=False); os._exit(0)
            x = self.dec_block1(x, enc3)
            x = self.dec_block2(x, enc2)
            x = self.dec_block3(x, enc1)
            x = self.dec_final(x)

            # Recover shape
            x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257. 

            x = x[:, :, 0:origin_len, :]

            audio_length = omni_waveform.shape[2]

            # from IPython import embed; embed(using=False); os._exit(0)
            # import matplotlib.pyplot as plt
            # plt.matshow(np.log(omni_mag.data.cpu().numpy()[0, 0]).T, origin='lower', aspect='auto', cmap='jet')
            # plt.savefig('_zz.pdf')
            # import soundfile
            # soundfile.write(file='_zz.wav', data=omni_waveform.data.cpu().numpy()[0, 0], samplerate=24000)

            omni_mag0 = self.tile_omni(omni_mag, max_rays_contain_waveform)
            omni_cos0 = self.tile_omni(omni_cos, max_rays_contain_waveform)
            omni_sin0 = self.tile_omni(omni_sin, max_rays_contain_waveform)

            separated_audio = self.feature_maps_to_wav(
                input_tensor=x,
                sp=omni_mag0,
                sin_in=omni_sin0,
                cos_in=omni_cos0,
                audio_length=audio_length,
            )

            separated_audio = separated_audio.reshape(batch_size, sep_rays_num, -1)
            # from IPython import embed; embed(using=False); os._exit(0)

            output_dict['ray_waveform'] = separated_audio

        return output_dict

    def tile_omni(self, x, rays_num): 
        batch_size, _, _T, _F = x.shape
        return torch.tile(x[:, None, :, :, :], (1, rays_num, 1, 1, 1)).reshape(batch_size * rays_num, -1, _T, _F)
'''