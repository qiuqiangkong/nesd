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

        self.downsample_ratio = 8

        pos_enc_size = 5
        orien_enc_size = 5
        dist_enc_size = 5

        self.pos_encoder = PositionEncoder(size=pos_enc_size)
        self.orien_encoder = OrientationEncoder(size=orien_enc_size)
        self.dist_encoder = DistanceEncoder(size=dist_enc_size)

        #
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

        #
        self.mic_wav_fc = nn.Linear(
            in_features=4096, 
            out_features=1024
        )

        self.mic_pos_fc = nn.Linear(
            in_features=mics_num * (pos_enc_size * 6), 
            out_features=128, 
            bias=True
        )

        self.mic_orien_fc = nn.Linear(
            in_features=mics_num * (orien_enc_size * 4), 
            out_features=128, 
            bias=True
        )

        # Agent position, direction, distance FC layers.
        self.agent_pos_fc = nn.Linear(
            in_features=pos_enc_size * 6,
            out_features=128, 
            bias=True
        )

        self.agent_dir_fc = nn.Linear(
            in_features=orien_enc_size * 4, 
            out_features=128, 
            bias=True
        )

        self.agent_dist_fc = nn.Linear(
            in_features=dist_enc_size * 2, 
            out_features=128, 
            bias=True
        )

        #
        self.det_mlp = MLP(in_channels=1664, hid_channels=1024, out_channels=1)

        #
        self.dist_mlp = MLP(in_channels=1664, hid_channels=1024, out_channels=1)

        #
        self.sep_expand = nn.Linear(1664, 4096, bias=True)

        self.sep_decoder_block1 = DecoderBlock(
            in_channels=128, 
            out_channels=64, 
            kernel_size=(3, 3), 
            upsample=(2, 2), 
        )

        self.sep_decoder_block2 = DecoderBlock(
            in_channels=64, 
            out_channels=32, 
            kernel_size=(3, 3), 
            upsample=(2, 2), 
        )

        self.sep_decoder_block3 = DecoderBlock(
            in_channels=32, 
            out_channels=32, 
            kernel_size=(3, 3), 
            upsample=(2, 2), 
        )

        self.sep_last_conv = nn.Conv2d(
            in_channels=32,
            out_channels=2,
            kernel_size=(1, 1),
            padding=(0, 0),
            bias=True,
        )


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
        agent_distance_masks = data["agent_distance_masks"]

        agent_detect_idxes = data["agent_detect_idxes"]
        agent_dist_idxes = data["agent_distance_idxes"]
        agent_sep_idxes = data["agent_sep_idxes"]

        # ------ 1. Convert positions and look at directions to embeddings ------
        mic_pos_emb = self.pos_encoder(mic_poss)
        # shape: (bs, mics_num, frames_num, emb_size)

        mic_orien_emb = self.orien_encoder(mic_oriens)
        # shape: (bs, mics_num, frames_num, emb_size)

        agent_pos_emb = self.pos_encoder(agent_poss)
        # shape: (bs, agents_num, frames_num, emb_size)

        agent_dir_emb = self.orien_encoder(agent_look_at_directions)
        # shape: (bs, agents_num, frames_num, emb_size)

        agent_dist_emb = self.dist_encoder(agent_look_at_distances)
        # shape: (bs, agents_num, frames_num, emb_size)

        # ------ 2. Extract features ------

        mic_stfts = self.stft(mic_wavs)

        mag_feat = torch.abs(mic_stfts)
        phase_feat = self.compute_phase_feature(mic_stfts)

        x = torch.cat((mag_feat, phase_feat), dim=1)
        # (bs, 64, frames_num, freq_bins)

        frames_num = x.shape[2]

        x = self.pad_image(x)

        # ------ 3. Forward data to the network ------

        # 1) Transform mic signal.
        x1, latent1 = self.mic_signal_encoder_block1(x)
        x2, latent2 = self.mic_signal_encoder_block2(x1)
        x3, latent3 = self.mic_signal_encoder_block3(x2)
        # enc3_pool: (bs, 128, T=38, F=32)

        mic_wav_feat = rearrange(x3, 'b c t f -> b t (c f)')
        # (bs, T, C*F=4096)

        mic_wav_feat = F.leaky_relu_(self.mic_wav_fc(mic_wav_feat))
        # shape: (bs, T, 1024)

        # Microphone positions feature.
        mic_pos_emb = self.pad_image(mic_pos_emb, freq_axis=False)
        mic_pos_emb = rearrange(mic_pos_emb, 'b c t f -> b t (c f)')
        mic_pos_feat = F.leaky_relu_(self.mic_pos_fc(mic_pos_emb))
        mic_pos_feat = mic_pos_feat[:, 0 :: self.downsample_ratio, :]
        
        # Microphone orientations feature.
        mic_orien_emb = self.pad_image(mic_orien_emb, freq_axis=False)
        mic_orien_emb = rearrange(mic_orien_emb, 'b c t f -> b t (c f)')
        mic_orien_feat = F.leaky_relu_(self.mic_orien_fc(mic_orien_emb))  
        mic_orien_feat = mic_orien_feat[:, 0 :: self.downsample_ratio, :]

        # Concatenate mic signal, position, and look direction features.
        mic_feat = torch.cat((
            mic_wav_feat, mic_pos_feat, mic_orien_feat
        ), dim=-1)  # (batch_size, T, 1280)

        
        agents_num = agent_poss.shape[1]

        mic_feat = mic_feat[:, None, :, :].repeat_interleave(repeats=agents_num, dim=1)

        # ------ 4. Add agents information ------
        agent_pos_emb = self.pad_image(agent_pos_emb, freq_axis=False)
        agent_pos_feat = F.leaky_relu_(self.agent_pos_fc(agent_pos_emb))  
        agent_pos_feat = agent_pos_feat[:, :, 0 :: self.downsample_ratio, :]
        # shape: (bs, T, 128)

        #
        agent_dir_emb = self.pad_image(agent_dir_emb, freq_axis=False)
        agent_dir_feat = F.leaky_relu_(self.agent_dir_fc(agent_dir_emb))
        agent_dir_feat = agent_dir_feat[:, :, 0 :: self.downsample_ratio, :]
        # shape: (bs, T, 128)

        #
        agent_dist_emb = self.pad_image(agent_dist_emb, freq_axis=False)
        agent_dist_feat = F.leaky_relu_(self.agent_dist_fc(agent_dist_emb))
        agent_dist_feat = agent_dist_feat[:, :, 0 :: self.downsample_ratio, :]
        # shape: (B, rays_num, T, C)

        agent_distance_masks = agent_distance_masks[:, :, 0 :: self.downsample_ratio, None]
        agent_dist_feat = agent_distance_masks * agent_dist_feat.nan_to_num(nan=0.)

        total_feat = torch.cat((mic_feat, agent_pos_feat, agent_dir_feat, agent_dist_feat), dim=-1)

        # ------ 5a. Detection. ------
        det_feat = get_tensor_from_indexes(total_feat, agent_detect_idxes)
        det_feat = self.det_mlp(det_feat)
        look_at_direction_has_source = det_feat.repeat_interleave(repeats=self.downsample_ratio, dim=2)[:, :, 0 : frames_num, 0]
        # shape: (B, det_rays, T)

        # ------ 5b. Distance estimation. ------
        dist_feat = get_tensor_from_indexes(total_feat, agent_dist_idxes)
        dist_feat = self.dist_mlp(dist_feat)
        look_at_distance_has_source = det_feat.repeat_interleave(repeats=self.downsample_ratio, dim=2)[:, :, 0 : frames_num, 0]
        # shape: (B, dist_rays, T)

        # ------ 5c. Spatial source separation. ------
        sep_rays = agent_sep_idxes.shape[1]

        sep_feat = get_tensor_from_indexes(total_feat, agent_sep_idxes)

        x = F.leaky_relu_(self.sep_expand(sep_feat))

        x = rearrange(x, 'b n t (c f) -> (b n) c t f', c=128, f=32)
        # (B * sep_rays, C=128, T=38, F=32)

        latent1 = self.repeat_conv_features(x=latent1, repeats_num=sep_rays)
        latent2 = self.repeat_conv_features(x=latent2, repeats_num=sep_rays)
        latent3 = self.repeat_conv_features(x=latent3, repeats_num=sep_rays)
        # (B * sep_rays, C, T, F)

        x = self.sep_decoder_block1(x, latent3)
        x = self.sep_decoder_block2(x, latent2)
        x = self.sep_decoder_block3(x, latent1)
        # (B * sep_rays, C, T, F)

        x = self.sep_last_conv(x)
        # (B * sep_rays, C, T, F)

        x = rearrange(x, '(b n) k t f -> b n t f k', n=sep_rays).contiguous()
        mask = torch.view_as_complex(x)
        mask = self.cut_image(mask, frames_num)

        sum_mic_wav = torch.sum(mic_wavs, dim=1, keepdim=True)
        sum_mic_stft = self.stft(sum_mic_wav)

        sep_stft = mask * sum_mic_stft

        look_at_direction_wav = self.istft(sep_stft)

        output_dict = {
            "agent_look_at_direction_has_source": look_at_direction_has_source,
            "agent_look_at_distance_has_source": look_at_distance_has_source,
            "agent_look_at_direction_reverb_wav": look_at_direction_wav
        }

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

    def pad_image(self, x, time_axis=True, freq_axis=True):
        """Cut a spectrum that can be evenly divided by downsample_ratio.

        Args:
            x: E.g., (B, C, 201, 1025)
        
        Outpus:
            output: E.g., (B, C, 208, 1024)
        """

        B, C, T, Freq = x.shape

        if time_axis:
            pad_len = (
                int(math.ceil(T / self.downsample_ratio)) * self.downsample_ratio
                - T
            )
            x = F.pad(x, pad=(0, 0, 0, pad_len))

        if freq_axis:
            x = x[:, :, :, 0 : Freq - 1]

        return x

    def cut_image(self, x, time_steps, time_axis=True, freq_axis=True):
        """Patch a spectrum to the original shape. E.g.,
        
        Args:
            x: E.g., (B, C, 208, 1024)
        
        Outpus:
            output: E.g., (B, C, 201, 1025)
        """
        if freq_axis:
            x = F.pad(x, pad=(0, 1))

        if time_axis:
            x = x[:, :, 0 : time_steps, :]

        return x

    def repeat_conv_features(self, x, repeats_num):
        x = torch.tile(x[:, None, :, :, :], (1, repeats_num, 1, 1, 1))
        x = rearrange(x, 'b n c t f -> (b n) c t f')
        return x


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
        x = self.conv1(F.leaky_relu_(self.bn1(input_tensor)))
        x = self.conv2(F.leaky_relu_(self.bn2(x)))

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
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size
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
        super(DecoderBlock, self).__init__()
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

        self.conv_block = ConvBlock(
            in_channels=in_channels * 2, 
            out_channels=out_channels, 
            kernel_size=kernel_size
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

        output = self.conv_block(x)

        return output


class MLP(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, hid_channels)
        self.fc2 = nn.Linear(hid_channels, hid_channels)
        self.fc3 = nn.Linear(hid_channels, hid_channels)
        self.fc4 = nn.Linear(hid_channels, out_channels)

    def forward(self, x):
        x = F.leaky_relu_(self.fc1(x))
        x = F.leaky_relu_(self.fc2(x))
        x = F.leaky_relu_(self.fc3(x))
        output = torch.sigmoid(self.fc4(x))
        return output


def get_tensor_from_indexes(x, indexes):
    
    batch_size = x.shape[0]
    indexes = indexes.long()
    output = torch.stack([x[i, indexes[i]] for i in range(batch_size)], dim=0)

    return output