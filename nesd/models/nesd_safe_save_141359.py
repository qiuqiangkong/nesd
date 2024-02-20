import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Dict, List, NoReturn, Tuple, Callable, Any

from nesd.models.fourier import Fourier
from nesd.models.base import PositionEncoder, OrientationEncoder, DistanceEncoder


class NeSD(Fourier):
    def __init__(self, 
        mics_num: int,
        n_fft=512,
        hop_length=240,
    ):
        super(NeSD, self).__init__(n_fft=n_fft, hop_length=hop_length) 

        # window_size = n_fft
        # hop_size = hop_length
        # center = True
        # pad_mode = "reflect"
        # window = "hann" 
        # activation = "relu"

        self.downsample_ratio = 8

        pos_encoder_size = 5
        orien_encoder_size = 5
        dist_encoder_size = 5

        self.pos_encoder = PositionEncoder(size=pos_encoder_size)
        self.orien_encoder = OrientationEncoder(size=orien_encoder_size)
        self.dist_encoder = DistanceEncoder(size=dist_encoder_size)

        self.mic_signal_encoder_block1 = EncoderBlock(
            in_channels=mics_num ** 2, 
            out_channels=32, 
            kernel_size=(3, 3), 
            downsample=(2, 2), 
        )

        self.mic_signal_encoder_block2 = EncoderBlock(
            in_channels=32, 
            out_channels=64, 
            kernel_size=(3, 3), 
            downsample=(2, 2), 
        )

        self.mic_signal_encoder_block3 = EncoderBlock(
            in_channels=64, 
            out_channels=128, 
            kernel_size=(3, 3), 
            downsample=(2, 2), 
        )

        # self.look_direction_encoder = LookDirectionEncoder(
        #     factor=positional_embedding_factor
        # )

    def forward(
        self, 
        data: Dict, 
        mode="train",
    ):

        mic_poss = data["mic_positions"]
        mic_oriens = data["mic_orientations"]
        mic_wavs = data["mic_wavs"]

        agent_poss = data["agent_positions"]
        agent_look_at_directions = data["agent_look_at_directions"]
        agent_look_at_distances = data["agent_look_at_distances"]

        agent_detect_idxes = data["agent_detect_idxes"]
        agent_distance_idxes = data["agent_distance_idxes"]
        agent_sep_idxes = data["agent_sep_idxes"]

        # ------ 1. Convert positions and look at directions to embeddings ------
        mic_pos_emb = self.pos_encoder(mic_poss)
        # shape: (bs, mics_num, frames_num, emb_size)

        mic_orien_emb = self.orien_encoder(mic_oriens)
        # shape: (bs, mics_num, frames_num, emb_size)

        agent_pos_emb = self.pos_encoder(agent_poss)
        # shape: (bs, agents_num, frames_num, emb_size)

        agent_look_at_direct_emb = self.orien_encoder(agent_look_at_directions)
        # shape: (bs, agents_num, frames_num, emb_size)

        agent_look_at_dist_emb = self.dist_encoder(agent_look_at_distances)
        # shape: (bs, agents_num, frames_num, emb_size)

        # ------ 2. Extract features ------
        sum_mic_wavs = torch.sum(mic_wavs, dim=1, keepdim=True)
        # (bs, 1, samples_num)

        mic_stfts = self.stft(mic_wavs)
        
        mag_feature = torch.abs(mic_stfts)
        phase_feature = self.compute_phase_feature(mic_stfts)

        x = torch.cat((mag_feature, phase_feature), dim=1)
        # (bs, 64, frames_num, freq_bins)

        x = self.cut_image(x)

        # # Pad zero frames after the last frame.
        # mic_positions_emb = F.pad(mic_positions_emb, pad=(0, 0, 0, pad_len))
        # mic_look_directions_emb = F.pad(mic_look_directions_emb, pad=(0, 0, 0, pad_len))
        # agent_positions_emb = F.pad(agent_positions_emb, pad=(0, 0, 0, pad_len))
        # agent_look_directions_emb = F.pad(agent_look_directions_emb, pad=(0, 0, 0, pad_len))
        # agent_look_depths_emb = F.pad(agent_look_depths_emb, pad=(0, 0, 0, pad_len))
        # agent_look_depths_mask = F.pad(agent_look_depths_mask, pad=(0, 0, 0, pad_len))

        # ------ 3. Forward data to the network ------

        # 1) Transform mic signal.
        x1, latent1 = self.mic_signal_encoder_block1(x)
        x2, latent2 = self.mic_signal_encoder_block2(enc1_pool)
        x3, latent3 = self.mic_signal_encoder_block3(enc2_pool)
        # enc3_pool: (bs, 128, T=38, F=32)

        from IPython import embed; embed(using=False); os._exit(0)

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

        if do_separation:

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

            agent_waveform_echo = rearrange(x, '(b n) t -> b n t', b=batch_size)
            # (bs, n=max_agents_contain_waveform, segment_samples)

            if mode == "train":
                agent_waveform_echo *= agent_active_indexes_mask[:, :, None]
            elif mode == "inference":
                pass
            else:
                raise NotImplementedError

            output_dict['agent_signals_echo'] = agent_waveform_echo

        return output_dict

    def compute_phase_feature(self, stfts):

        mics_num = stfts.shape[1]
        eps = 1e-10

        diffs = []

        for i in range(1, mics_num):
            for j in range(0, i):
                diff = stfts[:, i : i + 1, :, :] / (stfts[:, j : j + 1, :, :] + eps)
                diffs.append(diff)

        diffs = torch.cat(diffs, dim=1)
        # shape: (B, M*(M-1)/2, T, F)

        diffs = torch.view_as_real(diffs)
        diffs = rearrange(diffs, 'b c t f k -> b (c k) t f')
        return diffs

    def cut_image(self, x):
        """Cut a spectrum that can be evenly divided by downsample_ratio.

        Args:
            x: E.g., (B, C, 201, 1025)
        
        Outpus:
            output: E.g., (B, C, 208, 1024)
        """

        B, C, T, Freq = x.shape

        pad_len = (
            int(math.ceil(T / self.downsample_ratio)) * self.downsample_ratio
            - T
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))

        output = x[:, :, :, 0 : Freq - 1]

        return output

    def patch_image(self, x, time_steps):
        """Patch a spectrum to the original shape. E.g.,
        
        Args:
            x: E.g., (B, C, 208, 1024)
        
        Outpus:
            output: E.g., (B, C, 201, 1025)
        """
        x = F.pad(x, pad=(0, 1))

        output = x[:, :, 0 : time_steps, :]

        return output


'''
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
'''

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
    ):
        r"""Residual block."""
        super(ConvBlock, self).__init__()

        padding = [kernel_size[0] // 2, kernel_size[1] // 2]

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

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



class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        downsample: Tuple,
    ):
        r"""Encoder block, contains 8 convolutional layers."""
        super(EncoderBlock, self).__init__()

        self.conv_block = ConvBlock(
            in_channels, out_channels, kernel_size
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
        latent = self.conv_block(input_tensor)
        output = F.avg_pool2d(latent, kernel_size=self.downsample)
        return output, latent


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        upsample: Tuple,
    ):
        r"""Decoder block, contains 1 transposed convolutional and 8 convolutional layers."""
        super(DecoderBlockRes1B, self).__init__()
        self.kernel_size = kernel_size
        self.stride = upsample

        self.upsample = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=self.stride,
            stride=self.stride,
            padding=(0, 0),
            bias=False,
        )

        self.conv = nn.Conv2d(
            in_channels=in_channels * 2, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            padding=padding
        )

        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(
        self, x: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, input_feature_maps, downsampled_time_steps, downsampled_freq_bins)
            concat_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

        Returns:
            output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)
        """
        x = self.upsample(x)

        x = torch.cat((x, latent), dim=1)

        output = F.relu_(self.bn(self.conv(x)))
        
        return output