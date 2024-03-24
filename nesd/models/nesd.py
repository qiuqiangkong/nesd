import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Dict, List, NoReturn, Tuple, Callable, Any

from nesd.models.fourier import Fourier
from nesd.models.base import PositionEncoder, OrientationEncoder, DistanceEncoder, AngleEncoder


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

        batch_size = agent_poss.shape[0]

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

        if True:
            agent_dist_feat = agent_distance_masks * agent_dist_feat.nan_to_num(nan=0.)
        else:
            agent_dist_feat = agent_distance_masks * agent_dist_feat

        total_feat = torch.cat((mic_feat, agent_pos_feat, agent_dir_feat, agent_dist_feat), dim=-1)

        # ------ 5a. Detection. ------
        det_feat = get_tensor_from_indexes(total_feat, agent_detect_idxes)
        det_feat = self.det_mlp(det_feat)
        look_at_direction_has_source = det_feat.repeat_interleave(repeats=self.downsample_ratio, dim=2)[:, :, 0 : frames_num, 0]
        # shape: (B, det_rays, T)

        # ------ 5b. Distance estimation. ------
        dist_feat = get_tensor_from_indexes(total_feat, agent_dist_idxes)
        dist_feat = self.dist_mlp(dist_feat)
        look_at_distance_has_source = dist_feat.repeat_interleave(repeats=self.downsample_ratio, dim=2)[:, :, 0 : frames_num, 0]
        # shape: (B, dist_rays, T)

        output_dict = {
            "agent_look_at_direction_has_source": look_at_direction_has_source,
            "agent_look_at_distance_has_source": look_at_distance_has_source
        }

        # ------ 5c. Spatial source separation. ------
        sep_rays = agent_sep_idxes.shape[1]

        if sep_rays > 0:
            
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

        
            # x = rearrange(x, '(b n) k t f -> b n t f k', n=sep_rays).contiguous()
            x = rearrange(x, '(b n) k t f -> b n t f k', b=batch_size).contiguous()
            mask = torch.view_as_complex(x)
            mask = self.cut_image(mask, frames_num)

            sum_mic_wav = torch.sum(mic_wavs, dim=1, keepdim=True)
            sum_mic_stft = self.stft(sum_mic_wav)

            sep_stft = mask * sum_mic_stft

            look_at_direction_wav = self.istft(sep_stft)

            output_dict["agent_look_at_direction_reverb_wav"] = look_at_direction_wav

        # output_dict = {
        #     "agent_look_at_direction_has_source": look_at_direction_has_source,
        #     "agent_look_at_distance_has_source": look_at_distance_has_source,
        #     "agent_look_at_direction_reverb_wav": look_at_direction_wav
        # }

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


###
class NeSD2(Fourier):
    def __init__(self, 
        mics_num: int,
        n_fft=512,
        hop_length=240,
    ):
        super(NeSD2, self).__init__(n_fft=n_fft, hop_length=hop_length) 

        self.downsample_ratio = 8

        pos_enc_size = 5
        orien_enc_size = 5
        dist_enc_size = 5

        self.pos_encoder = PositionEncoder(size=pos_enc_size)
        self.orien_encoder = OrientationEncoder(size=orien_enc_size)
        self.dist_encoder = DistanceEncoder(size=dist_enc_size)

        #
        self.mic_signal_encoder_block1 = EncoderBlock(
            # in_channels=mics_num ** 2, 
            in_channels=64,
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

        batch_size = agent_poss.shape[0]

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

        # ------ 2. Extract features ------
        sum_mic_wavs = torch.sum(mic_wavs, dim=1, keepdim=True)
        # (bs, 1, samples_num)

        sum_mic_stfts = self.stft(sum_mic_wavs)

        frames_num = sum_mic_stfts.shape[2]

        tmp = torch.view_as_real(sum_mic_stfts)
        sum_mic_mag = torch.abs(sum_mic_stfts)
        sum_mic_real = tmp[..., 0]
        sum_mic_imag = tmp[..., 1]

        mic_stfts = self.stft(mic_wavs)
        mic_mag = torch.abs(mic_stfts)
        mic_angle = torch.angle(mic_stfts)
        mic_cos = torch.cos(mic_angle)
        mic_sin = torch.sin(mic_angle)

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

        mag_feature = torch.cat((mic_mag, sum_mic_mag), dim=1)
        # shape: (bs, mics_num + 1, frames_num, freq_bins)

        phase_feature = torch.cat((delta_cos, delta_sin), dim=1)
        # shape: (bs, mics_num * (mics_num - 1), frames_num, freq_bins)

        mag_feature = self.mag_fc(mag_feature)  # (bs, 32, frames_num, freq_bins)
        phase_feature = self.phase_fc(phase_feature)  # (bs, 32, frames_num, freq_bins)
        
        x = torch.cat((mag_feature, phase_feature), dim=1)

        x = self.pad_image(x)

        # from IPython import embed; embed(using=False); os._exit(0)

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

        if True:
            agent_dist_feat = agent_distance_masks * agent_dist_feat.nan_to_num(nan=0.)
        else:
            agent_dist_feat = agent_distance_masks * agent_dist_feat

        total_feat = torch.cat((mic_feat, agent_pos_feat, agent_dir_feat, agent_dist_feat), dim=-1)

        # ------ 5a. Detection. ------
        det_feat = get_tensor_from_indexes(total_feat, agent_detect_idxes)
        det_feat = self.det_mlp(det_feat)
        look_at_direction_has_source = det_feat.repeat_interleave(repeats=self.downsample_ratio, dim=2)[:, :, 0 : frames_num, 0]
        # shape: (B, det_rays, T)

        # ------ 5b. Distance estimation. ------
        dist_feat = get_tensor_from_indexes(total_feat, agent_dist_idxes)
        dist_feat = self.dist_mlp(dist_feat)
        look_at_distance_has_source = dist_feat.repeat_interleave(repeats=self.downsample_ratio, dim=2)[:, :, 0 : frames_num, 0]
        # shape: (B, dist_rays, T)

        output_dict = {
            "agent_look_at_direction_has_source": look_at_direction_has_source,
            "agent_look_at_distance_has_source": look_at_distance_has_source
        }

        # ------ 5c. Spatial source separation. ------
        sep_rays = agent_sep_idxes.shape[1]

        if sep_rays > 0:
            
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

        
            # x = rearrange(x, '(b n) k t f -> b n t f k', n=sep_rays).contiguous()
            x = rearrange(x, '(b n) k t f -> b n t f k', b=batch_size).contiguous()
            mask = torch.view_as_complex(x)
            mask = self.cut_image(mask, frames_num)

            sum_mic_wav = torch.sum(mic_wavs, dim=1, keepdim=True)
            sum_mic_stft = self.stft(sum_mic_wav)

            sep_stft = mask * sum_mic_stft

            look_at_direction_wav = self.istft(sep_stft)

            output_dict["agent_look_at_direction_reverb_wav"] = look_at_direction_wav

        # output_dict = {
        #     "agent_look_at_direction_has_source": look_at_direction_has_source,
        #     "agent_look_at_distance_has_source": look_at_distance_has_source,
        #     "agent_look_at_direction_reverb_wav": look_at_direction_wav
        # }

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


class NeSD3(Fourier):
    def __init__(self, 
        mics_num: int,
        n_fft=512,
        hop_length=240,
    ):
        super(NeSD3, self).__init__(n_fft=n_fft, hop_length=hop_length) 

        self.downsample_ratio = 8

        pos_enc_size = 5
        orien_enc_size = 5
        dist_enc_size = 5

        self.pos_encoder = PositionEncoder(size=pos_enc_size)
        self.orien_encoder = OrientationEncoder(size=orien_enc_size)
        self.dist_encoder = DistanceEncoder(size=dist_enc_size)

        #
        self.mic_signal_encoder_block1 = EncoderBlock(
            # in_channels=mics_num ** 2, 
            in_channels=64,
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

        self.mag_fc = nn.Conv2d(
            in_channels=mics_num + 1,
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
            bias=True
        )

        self.phase_fc = nn.Conv2d(
            # in_channels=mics_num * (mics_num - 1), 
            in_channels=20,
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
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

        batch_size = agent_poss.shape[0]

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

        # ------ 2. Extract features ------
        sum_mic_wavs = torch.sum(mic_wavs, dim=1, keepdim=True)
        # (bs, 1, samples_num)

        sum_mic_stfts = self.stft(sum_mic_wavs)

        frames_num = sum_mic_stfts.shape[2]

        tmp = torch.view_as_real(sum_mic_stfts)
        sum_mic_mag = torch.abs(sum_mic_stfts)
        sum_mic_real = tmp[..., 0]
        sum_mic_imag = tmp[..., 1]

        mic_stfts = self.stft(mic_wavs)
        mic_mag = torch.abs(mic_stfts)
        mic_angle = torch.angle(mic_stfts)
        mic_cos = torch.cos(mic_angle)
        mic_sin = torch.sin(mic_angle)

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

        mag_feature = torch.cat((mic_mag, sum_mic_mag), dim=1)
        # shape: (bs, mics_num + 1, frames_num, freq_bins)

        phase_feature = torch.cat((mic_cos, mic_sin, delta_cos, delta_sin), dim=1)
        # shape: (bs, mics_num * (mics_num - 1), frames_num, freq_bins)

        mag_feature = self.mag_fc(mag_feature)  # (bs, 32, frames_num, freq_bins)
        phase_feature = self.phase_fc(phase_feature)  # (bs, 32, frames_num, freq_bins)
        
        x = torch.cat((mag_feature, phase_feature), dim=1)

        x = self.pad_image(x)

        # from IPython import embed; embed(using=False); os._exit(0)

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

        if True:
            agent_dist_feat = agent_distance_masks * agent_dist_feat.nan_to_num(nan=0.)
        else:
            agent_dist_feat = agent_distance_masks * agent_dist_feat

        total_feat = torch.cat((mic_feat, agent_pos_feat, agent_dir_feat, agent_dist_feat), dim=-1)

        # ------ 5a. Detection. ------
        det_feat = get_tensor_from_indexes(total_feat, agent_detect_idxes)
        det_feat = self.det_mlp(det_feat)
        look_at_direction_has_source = det_feat.repeat_interleave(repeats=self.downsample_ratio, dim=2)[:, :, 0 : frames_num, 0]
        # shape: (B, det_rays, T)

        # ------ 5b. Distance estimation. ------
        dist_feat = get_tensor_from_indexes(total_feat, agent_dist_idxes)
        dist_feat = self.dist_mlp(dist_feat)
        look_at_distance_has_source = dist_feat.repeat_interleave(repeats=self.downsample_ratio, dim=2)[:, :, 0 : frames_num, 0]
        # shape: (B, dist_rays, T)

        output_dict = {
            "agent_look_at_direction_has_source": look_at_direction_has_source,
            "agent_look_at_distance_has_source": look_at_distance_has_source
        }

        # ------ 5c. Spatial source separation. ------
        sep_rays = agent_sep_idxes.shape[1]

        if sep_rays > 0:
            
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

        
            # x = rearrange(x, '(b n) k t f -> b n t f k', n=sep_rays).contiguous()
            x = rearrange(x, '(b n) k t f -> b n t f k', b=batch_size).contiguous()
            mask = torch.view_as_complex(x)
            mask = self.cut_image(mask, frames_num)

            sum_mic_wav = torch.sum(mic_wavs, dim=1, keepdim=True)
            sum_mic_stft = self.stft(sum_mic_wav)

            sep_stft = mask * sum_mic_stft

            look_at_direction_wav = self.istft(sep_stft)

            output_dict["agent_look_at_direction_reverb_wav"] = look_at_direction_wav

        # output_dict = {
        #     "agent_look_at_direction_has_source": look_at_direction_has_source,
        #     "agent_look_at_distance_has_source": look_at_distance_has_source,
        #     "agent_look_at_direction_reverb_wav": look_at_direction_wav
        # }

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


class AudioEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AudioEncoder, self).__init__()

        self.encoder_block1 = EncoderBlock(
            in_channels=in_channels,
            out_channels=32, 
            kernel_size=(3, 3), 
            downsample=(2, 2), 
        )

        self.encoder_block2 = EncoderBlock(
            in_channels=32, 
            out_channels=64, 
            kernel_size=(3, 3), 
            downsample=(2, 2), 
        )

        self.encoder_block3 = EncoderBlock(
            in_channels=64, 
            out_channels=out_channels, 
            kernel_size=(3, 3), 
            downsample=(2, 2), 
        )

    def forward(self, x):
        
        x1, latent1 = self.encoder_block1(x)
        x2, latent2 = self.encoder_block2(x1)
        x3, latent3 = self.encoder_block3(x2)

        output = {
            "block1": {"x": x1, "latent": latent1},
            "block2": {"x": x2, "latent": latent2},
            "block3": {"x": x3, "latent": latent3},
        }

        return output


class AudioDecoder(Fourier):
    def __init__(self, in_channels, n_fft, hop_length):
        super(AudioDecoder, self).__init__(n_fft=n_fft, hop_length=hop_length) 

        self.decoder_block1 = DecoderBlock(
            in_channels=in_channels, 
            out_channels=64, 
            kernel_size=(3, 3), 
            upsample=(2, 2), 
        )

        self.decoder_block2 = DecoderBlock(
            in_channels=64, 
            out_channels=32, 
            kernel_size=(3, 3), 
            upsample=(2, 2), 
        )

        self.decoder_block3 = DecoderBlock(
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

    def forward(self, x, wav_feat_dict, repeat_rays, frames_num, sum_mic_wav):

        latent1 = wav_feat_dict["block1"]["latent"]
        latent2 = wav_feat_dict["block2"]["latent"]
        latent3 = wav_feat_dict["block3"]["latent"]

        batch_size = latent1.shape[0]
        
        latent1 = self.repeat_conv_features(x=latent1, repeats_num=repeat_rays)
        latent2 = self.repeat_conv_features(x=latent2, repeats_num=repeat_rays)
        latent3 = self.repeat_conv_features(x=latent3, repeats_num=repeat_rays)
        # shape: (B, R_sep, C, T, F)

        x = self.decoder_block1(x, latent3)
        x = self.decoder_block2(x, latent2)
        x = self.decoder_block3(x, latent1)
        # shape: (B, R_sep, C, T, F)

        x = self.sep_last_conv(x)
        # shape: (B, R_sep, C, T, F)

        x = rearrange(x, '(b n) k t f -> b n t f k', b=batch_size).contiguous()
        mask = torch.view_as_complex(x)
        mask = self.unprocess_image(mask, frames_num)
        # mask: (B, R_sep, C, T, F)

        sum_mic_stft = self.stft(sum_mic_wav)

        sep_stft = mask * sum_mic_stft

        sep_wav = self.istft(sep_stft)

        return sep_wav

    def repeat_conv_features(self, x, repeats_num):
        x = torch.tile(x[:, None, :, :, :], (1, repeats_num, 1, 1, 1))
        x = rearrange(x, 'b n c t f -> (b n) c t f')
        return x

    def unprocess_image(self, x, time_steps, time_axis=True, freq_axis=True):
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


class NeSD4(Fourier):
    def __init__(self, 
        mics_num: int,
        n_fft=512,
        hop_length=240,
    ):
        super(NeSD4, self).__init__(n_fft=n_fft, hop_length=hop_length) 

        self.downsample_ratio = 8

        pos_enc_size = 5
        orien_enc_size = 5
        dist_enc_size = 5
        angle_enc_size = 5

        self.pos_encoder = PositionEncoder(size=pos_enc_size)
        self.orien_encoder = OrientationEncoder(size=orien_enc_size)
        self.dist_encoder = DistanceEncoder(size=dist_enc_size)
        self.angle_encoder = AngleEncoder(size=angle_enc_size)

        # Mic magnitude process layer.
        self.mic_mag_fc = nn.Conv2d(
            in_channels=mics_num,
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
            bias=True
        )

        # Mic phase process layer.
        in_channels = (mics_num + mics_num * (mics_num - 1) // 2) * (angle_enc_size * 2)
        self.mic_phase_fc = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
            bias=True
        )

        self.mic_wav_encoder = AudioEncoder(
            in_channels=64,
            out_channels=128
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

        # Detection MLP layers.
        self.det_mlp = MLP(in_channels=1664, hid_channels=1024, out_channels=1)

        # Distance MLP layers.
        self.dist_mlp = MLP(in_channels=1664, hid_channels=1024, out_channels=1)

        # Separation layers.
        self.sep_fc = nn.Linear(1664, 4096, bias=True)

        self.sep_reverb_decoder = AudioDecoder(
            in_channels=128, 
            n_fft=n_fft, 
            hop_length=hop_length
        )
        # self.sep_direct_decoder = AudioEncoder(in_channels=128)

    def forward(
        self, 
        data: Dict, 
    ):

        mic_poss = data["mic_positions"]
        mic_oriens = data["mic_orientations"]
        mic_wavs = data["mic_wavs"]

        agent_poss = data["agent_positions"]
        agent_look_at_directions = data["agent_look_at_directions"]
        agent_look_at_distances = data["agent_look_at_distances"]
        agent_distance_masks = data["agent_distance_masks"]

        batch_size = agent_poss.shape[0]

        # ------ 1. Convert positions and look at directions to embeddings ------
        mic_pos_emb = self.pos_encoder(mic_poss)
        # shape: (B, mics_num, T, C)

        mic_orien_emb = self.orien_encoder(mic_oriens)
        # shape: (B, mics_num, T, C)

        agent_pos_emb = self.pos_encoder(agent_poss)
        # shape: (B, agents_num, T, C)

        agent_dir_emb = self.orien_encoder(agent_look_at_directions)
        # shape: (B, agents_num, T, C)

        agent_dist_emb = self.dist_encoder(agent_look_at_distances)
        # shape: (B, agents_num, T, C)

        # ------ 2. Extract features ------

        mic_stfts = self.stft(mic_wavs)
        # shape: (B, mics_num, T, F)

        frames_num = mic_stfts.shape[2]

        # Magnitude feature maps
        mags = torch.abs(mic_stfts)
        mag_feat = self.mic_mag_fc(mags)  # (bs, 32, frames_num, freq_bins)
        # shape: (B, 32, T, F)

        # Feature feature maps
        phases = torch.angle(mic_stfts)
        diff_phases = self.calculate_diff_phases(phases)
        total_phases = torch.cat((phases, diff_phases), dim=1)

        phase_feat = self.angle_encoder(total_phases)
        phase_feat = rearrange(phase_feat, 'b c t f k -> b (c k) t f')
        phase_feat = self.mic_phase_fc(phase_feat)
        # shape: (B, 32, T, F)
        
        x = torch.cat((mag_feat, phase_feat), dim=1)
        # shape: (B, 64, T, F)

        x = self.process_image(x)
        # shape: (B, 64, T, F)

        # ------ 3. Forward data to the network ------

        mic_feat_dict = self.mic_wav_encoder(x)

        mic_wav_feat = rearrange(mic_feat_dict["block3"]["x"], 'b c t f -> b t (c f)')
        # shape: (bs, T, 4096)

        mic_wav_feat = F.leaky_relu_(self.mic_wav_fc(mic_wav_feat))
        # shape: (bs, T, 1024)

        # Microphone positions feature.
        mic_pos_emb = self.process_image(mic_pos_emb, freq_axis=False)
        mic_pos_emb = rearrange(mic_pos_emb, 'b c t f -> b t (c f)')
        mic_pos_feat = F.leaky_relu_(self.mic_pos_fc(mic_pos_emb))
        mic_pos_feat = mic_pos_feat[:, 0 :: self.downsample_ratio, :]
        # mic_pos_feat: (B, T', C)
        
        # Microphone orientations feature.
        mic_orien_emb = self.process_image(mic_orien_emb, freq_axis=False)
        mic_orien_emb = rearrange(mic_orien_emb, 'b c t f -> b t (c f)')
        mic_orien_feat = F.leaky_relu_(self.mic_orien_fc(mic_orien_emb))  
        mic_orien_feat = mic_orien_feat[:, 0 :: self.downsample_ratio, :]
        # mic_orien_feat: (B, T', C)

        # Concatenate mic signal, position, and look direction features.
        mic_feat = torch.cat((mic_wav_feat, mic_pos_feat, mic_orien_feat), dim=-1)
        # mic_feat: (B, T', 1280)

        agents_num = agent_poss.shape[1]
        mic_feat = mic_feat[:, None, :, :].repeat_interleave(repeats=agents_num, dim=1)
        # mic_feat: (B, R, T', 1280)

        # ------ 4. Add agents information ------
        agent_pos_emb = self.process_image(agent_pos_emb, freq_axis=False)
        agent_pos_feat = F.leaky_relu_(self.agent_pos_fc(agent_pos_emb))  
        agent_pos_feat = agent_pos_feat[:, :, 0 :: self.downsample_ratio, :]
        # agent_pos_feat: (B, R, T', 128)

        agent_dir_emb = self.process_image(agent_dir_emb, freq_axis=False)
        agent_dir_feat = F.leaky_relu_(self.agent_dir_fc(agent_dir_emb))
        agent_dir_feat = agent_dir_feat[:, :, 0 :: self.downsample_ratio, :]
        # agent_dir_feat: (B, R, T', 128)

        agent_dist_emb = self.process_image(agent_dist_emb, freq_axis=False)
        agent_dist_feat = F.leaky_relu_(self.agent_dist_fc(agent_dist_emb))
        agent_dist_feat = agent_dist_feat[:, :, 0 :: self.downsample_ratio, :]
        # agent_dist_feat: (B, R, T', 128)

        agent_distance_masks = agent_distance_masks[:, :, 0 :: self.downsample_ratio, None]
        # shape: (B, R, T', 1)

        # If no distance information is provided then set to 0.
        agent_dist_feat = agent_distance_masks * agent_dist_feat
        # shape: (B, R, T', 128)

        total_feat = torch.cat((mic_feat, agent_pos_feat, agent_dir_feat, agent_dist_feat), dim=-1)
        # shape: (B, R, T', 1664)

        # ------ 5a. Spatial detection. ------
        agent_detect_idxes = data["agent_detect_idxes"]
        det_feat = get_tensor_from_indexes(total_feat, agent_detect_idxes)
        det_feat = self.det_mlp(det_feat)
        # shape: (B, R_det, T', 1)

        look_at_direction_has_source = det_feat.repeat_interleave(
            repeats=self.downsample_ratio, dim=2)[:, :, 0 : frames_num, 0]
        # shape: (B, R_det, T)
        
        # ------ 5b. Spatial distance estimation. ------
        agent_dist_idxes = data["agent_distance_idxes"]
        dist_feat = get_tensor_from_indexes(total_feat, agent_dist_idxes)
        dist_feat = self.dist_mlp(dist_feat)
        look_at_distance_has_source = dist_feat.repeat_interleave(repeats=self.downsample_ratio, dim=2)[:, :, 0 : frames_num, 0]
        # shape: (B, dist_rays, T)

        output_dict = {
            "agent_look_at_direction_has_source": look_at_direction_has_source,
            "agent_look_at_distance_has_source": look_at_distance_has_source
        }

        # ------ 5c. Spatial source separation. ------
        agent_sep_idxes = data["agent_sep_idxes"]
        sep_rays = agent_sep_idxes.shape[1]

        if sep_rays > 0:
            
            sep_feat = get_tensor_from_indexes(total_feat, agent_sep_idxes)
            # shape: (B, R_sep, T', 1664)

            x = F.leaky_relu_(self.sep_fc(sep_feat))
            # shape: (B, R_sep, T', 4096)

            x = rearrange(x, 'b n t (c f) -> (b n) c t f', c=128, f=32)
            # shape: (B, R_sep, C, T'=38, F=32)

            sum_mic_wav = torch.sum(mic_wavs, dim=1, keepdim=True)

            look_at_direction_wav = self.sep_reverb_decoder(
                x=x, 
                wav_feat_dict=mic_feat_dict, 
                repeat_rays=sep_rays, 
                frames_num=frames_num,
                sum_mic_wav=sum_mic_wav
            )

            output_dict["agent_look_at_direction_reverb_wav"] = look_at_direction_wav
            
        return output_dict

    def calculate_diff_phases(self, mic_phase):

        mics_num = mic_phase.shape[1]

        diff_phases = []

        for i in range(1, mics_num):
            for j in range(0, i):
                diff_phases.append(mic_phase[:, i, :, :] - mic_phase[:, j, :, :])

        diff_phases = torch.stack(diff_phases, dim=1)

        return diff_phases

    def process_image(self, x, time_axis=True, freq_axis=True):
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


class NeSD4b(Fourier):
    def __init__(self, 
        mics_num: int,
        n_fft=512,
        hop_length=240,
    ):
        super(NeSD4b, self).__init__(n_fft=n_fft, hop_length=hop_length) 

        self.downsample_ratio = 8

        pos_enc_size = 5
        orien_enc_size = 5
        dist_enc_size = 5
        angle_enc_size = 1  # If too large will increase GPU memory.

        self.pos_encoder = PositionEncoder(size=pos_enc_size)
        self.orien_encoder = OrientationEncoder(size=orien_enc_size)
        self.dist_encoder = DistanceEncoder(size=dist_enc_size)
        self.angle_encoder = DistanceEncoder(size=angle_enc_size)

        # Mic magnitude process layer.
        self.mic_mag_fc = nn.Conv2d(
            in_channels=mics_num,
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
            bias=True
        )

        # Mic phase process layer.
        in_channels = (mics_num + mics_num * (mics_num - 1) // 2) * (angle_enc_size * 2)
        self.mic_phase_fc = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
            bias=True
        )

        self.mic_wav_encoder = AudioEncoder(
            in_channels=64,
            out_channels=128
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

        # Detection MLP layers.
        self.det_mlp = MLP(in_channels=1664, hid_channels=1024, out_channels=1)

        # Distance MLP layers.
        self.dist_mlp = MLP(in_channels=1664, hid_channels=1024, out_channels=1)

        # Separation layers.
        self.sep_fc = nn.Linear(1664, 4096, bias=True)

        self.sep_reverb_decoder = AudioDecoder(
            in_channels=128, 
            n_fft=n_fft, 
            hop_length=hop_length
        )

        self.sep_direct_decoder = AudioDecoder(
            in_channels=128, 
            n_fft=n_fft, 
            hop_length=hop_length
        )
        

    def forward(
        self, 
        data: Dict, 
    ):

        mic_poss = data["mic_positions"]
        mic_oriens = data["mic_orientations"]
        mic_wavs = data["mic_wavs"]

        agent_poss = data["agent_positions"]
        agent_look_at_directions = data["agent_look_at_directions"]
        agent_look_at_distances = data["agent_look_at_distances"]
        agent_distance_masks = data["agent_distance_masks"]

        batch_size = agent_poss.shape[0]

        # ------ 1. Convert positions and look at directions to embeddings ------
        mic_pos_emb = self.pos_encoder(mic_poss)
        # shape: (B, mics_num, T, C)

        mic_orien_emb = self.orien_encoder(mic_oriens)
        # shape: (B, mics_num, T, C)

        agent_pos_emb = self.pos_encoder(agent_poss)
        # shape: (B, agents_num, T, C)

        agent_dir_emb = self.orien_encoder(agent_look_at_directions)
        # shape: (B, agents_num, T, C)

        agent_dist_emb = self.dist_encoder(agent_look_at_distances)
        # shape: (B, agents_num, T, C)

        # ------ 2. Extract features ------

        mic_stfts = self.stft(mic_wavs)
        # shape: (B, mics_num, T, F)

        frames_num = mic_stfts.shape[2]

        # Magnitude feature maps
        mags = torch.abs(mic_stfts)
        mag_feat = self.mic_mag_fc(mags)  # (bs, 32, frames_num, freq_bins)
        # shape: (B, 32, T, F)

        # Feature feature maps
        phases = torch.angle(mic_stfts)
        diff_phases = self.calculate_diff_phases(phases)
        total_phases = torch.cat((phases, diff_phases), dim=1)

        phase_feat = self.angle_encoder(total_phases)
        phase_feat = rearrange(phase_feat, 'b c t f k -> b (c k) t f')
        phase_feat = self.mic_phase_fc(phase_feat)
        # shape: (B, 32, T, F)
        
        x = torch.cat((mag_feat, phase_feat), dim=1)
        # shape: (B, 64, T, F)

        x = self.process_image(x)
        # shape: (B, 64, T, F)

        # ------ 3. Forward data to the network ------

        mic_feat_dict = self.mic_wav_encoder(x)

        mic_wav_feat = rearrange(mic_feat_dict["block3"]["x"], 'b c t f -> b t (c f)')
        # shape: (bs, T, 4096)

        mic_wav_feat = F.leaky_relu_(self.mic_wav_fc(mic_wav_feat))
        # shape: (bs, T, 1024)

        # Microphone positions feature.
        mic_pos_emb = self.process_image(mic_pos_emb, freq_axis=False)
        mic_pos_emb = rearrange(mic_pos_emb, 'b c t f -> b t (c f)')
        mic_pos_feat = F.leaky_relu_(self.mic_pos_fc(mic_pos_emb))
        mic_pos_feat = mic_pos_feat[:, 0 :: self.downsample_ratio, :]
        # mic_pos_feat: (B, T', C)
        
        # Microphone orientations feature.
        mic_orien_emb = self.process_image(mic_orien_emb, freq_axis=False)
        mic_orien_emb = rearrange(mic_orien_emb, 'b c t f -> b t (c f)')
        mic_orien_feat = F.leaky_relu_(self.mic_orien_fc(mic_orien_emb))  
        mic_orien_feat = mic_orien_feat[:, 0 :: self.downsample_ratio, :]
        # mic_orien_feat: (B, T', C)

        # Concatenate mic signal, position, and look direction features.
        mic_feat = torch.cat((mic_wav_feat, mic_pos_feat, mic_orien_feat), dim=-1)
        # mic_feat: (B, T', 1280)

        agents_num = agent_poss.shape[1]
        mic_feat = mic_feat[:, None, :, :].repeat_interleave(repeats=agents_num, dim=1)
        # mic_feat: (B, R, T', 1280)

        # ------ 4. Add agents information ------
        agent_pos_emb = self.process_image(agent_pos_emb, freq_axis=False)
        agent_pos_feat = F.leaky_relu_(self.agent_pos_fc(agent_pos_emb))  
        agent_pos_feat = agent_pos_feat[:, :, 0 :: self.downsample_ratio, :]
        # agent_pos_feat: (B, R, T', 128)

        agent_dir_emb = self.process_image(agent_dir_emb, freq_axis=False)
        agent_dir_feat = F.leaky_relu_(self.agent_dir_fc(agent_dir_emb))
        agent_dir_feat = agent_dir_feat[:, :, 0 :: self.downsample_ratio, :]
        # agent_dir_feat: (B, R, T', 128)

        agent_dist_emb = self.process_image(agent_dist_emb, freq_axis=False)
        agent_dist_feat = F.leaky_relu_(self.agent_dist_fc(agent_dist_emb))
        agent_dist_feat = agent_dist_feat[:, :, 0 :: self.downsample_ratio, :]
        # agent_dist_feat: (B, R, T', 128)

        agent_distance_masks = agent_distance_masks[:, :, 0 :: self.downsample_ratio, None]
        # shape: (B, R, T', 1)

        # If no distance information is provided then set to 0.
        agent_dist_feat = agent_distance_masks * agent_dist_feat
        # shape: (B, R, T', 128)

        total_feat = torch.cat((mic_feat, agent_pos_feat, agent_dir_feat, agent_dist_feat), dim=-1)
        # shape: (B, R, T', 1664)

        # ------ 5a. Spatial detection. ------
        agent_detect_idxes = data["agent_detect_idxes"]
        det_feat = get_tensor_from_indexes(total_feat, agent_detect_idxes)
        det_feat = self.det_mlp(det_feat)
        # shape: (B, R_det, T', 1)

        look_at_direction_has_source = det_feat.repeat_interleave(
            repeats=self.downsample_ratio, dim=2)[:, :, 0 : frames_num, 0]
        # shape: (B, R_det, T)
        
        # ------ 5b. Spatial distance estimation. ------
        agent_dist_idxes = data["agent_distance_idxes"]
        dist_feat = get_tensor_from_indexes(total_feat, agent_dist_idxes)
        dist_feat = self.dist_mlp(dist_feat)
        look_at_distance_has_source = dist_feat.repeat_interleave(repeats=self.downsample_ratio, dim=2)[:, :, 0 : frames_num, 0]
        # shape: (B, dist_rays, T)

        output_dict = {
            "agent_look_at_direction_has_source": look_at_direction_has_source,
            "agent_look_at_distance_has_source": look_at_distance_has_source
        }

        # ------ 5c. Spatial source separation. ------
        agent_sep_idxes = data["agent_sep_idxes"]
        sep_rays = agent_sep_idxes.shape[1]

        if sep_rays > 0:
            
            sep_feat = get_tensor_from_indexes(total_feat, agent_sep_idxes)
            # shape: (B, R_sep, T', 1664)

            x = F.leaky_relu_(self.sep_fc(sep_feat))
            # shape: (B, R_sep, T', 4096)

            x = rearrange(x, 'b n t (c f) -> (b n) c t f', c=128, f=32)
            # shape: (B, R_sep, C, T'=38, F=32)

            sum_mic_wav = torch.sum(mic_wavs, dim=1, keepdim=True)

            look_at_direction_reverb_wav = self.sep_reverb_decoder(
                x=x, 
                wav_feat_dict=mic_feat_dict, 
                repeat_rays=sep_rays, 
                frames_num=frames_num,
                sum_mic_wav=sum_mic_wav
            )

            look_at_direction_direct_wav = self.sep_direct_decoder(
                x=x, 
                wav_feat_dict=mic_feat_dict, 
                repeat_rays=sep_rays, 
                frames_num=frames_num,
                sum_mic_wav=sum_mic_wav
            )

            output_dict["agent_look_at_direction_reverb_wav"] = look_at_direction_reverb_wav
            output_dict["agent_look_at_direction_direct_wav"] = look_at_direction_direct_wav

        return output_dict

    def calculate_diff_phases(self, mic_phase):

        mics_num = mic_phase.shape[1]

        diff_phases = []

        for i in range(1, mics_num):
            for j in range(0, i):
                diff_phases.append(mic_phase[:, i, :, :] - mic_phase[:, j, :, :])

        diff_phases = torch.stack(diff_phases, dim=1)

        return diff_phases

    def process_image(self, x, time_axis=True, freq_axis=True):
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


class NeSD5(Fourier):
    def __init__(self, 
        mics_num: int,
        n_fft=512,
        hop_length=240,
    ):
        super(NeSD5, self).__init__(n_fft=n_fft, hop_length=hop_length) 

        self.downsample_ratio = 8

        pos_enc_size = 5
        orien_enc_size = 5
        dist_enc_size = 5
        angle_enc_size = 1  # If too large will increase GPU memory.

        self.pos_encoder = PositionEncoder(size=pos_enc_size)
        self.orien_encoder = OrientationEncoder(size=orien_enc_size)
        self.dist_encoder = DistanceEncoder(size=dist_enc_size)
        self.angle_encoder = DistanceEncoder(size=angle_enc_size)

        # Mic magnitude process layer.
        self.mic_mag_fc = nn.Conv2d(
            in_channels=mics_num + 1,
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
            bias=True
        )

        # Mic phase process layer.
        in_channels = (mics_num + mics_num * (mics_num - 1) // 2) * (angle_enc_size * 2)
        self.mic_phase_fc = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
            bias=True
        )

        self.mic_wav_encoder = AudioEncoder(
            in_channels=64,
            out_channels=128
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

        # Detection MLP layers.
        self.det_mlp = MLP(in_channels=1664, hid_channels=1024, out_channels=1)

        # Distance MLP layers.
        self.dist_mlp = MLP(in_channels=1664, hid_channels=1024, out_channels=1)

        # Separation layers.
        self.sep_fc = nn.Linear(1664, 4096, bias=True)

        self.sep_reverb_decoder = AudioDecoder(
            in_channels=128, 
            n_fft=n_fft, 
            hop_length=hop_length
        )
        # self.sep_direct_decoder = AudioEncoder(in_channels=128)

    def forward(
        self, 
        data: Dict, 
    ):

        mic_poss = data["mic_positions"]
        mic_oriens = data["mic_orientations"]
        mic_wavs = data["mic_wavs"]

        agent_poss = data["agent_positions"]
        agent_look_at_directions = data["agent_look_at_directions"]
        agent_look_at_distances = data["agent_look_at_distances"]
        agent_distance_masks = data["agent_distance_masks"]

        batch_size = agent_poss.shape[0]

        # ------ 1. Convert positions and look at directions to embeddings ------
        mic_pos_emb = self.pos_encoder(mic_poss)
        # shape: (B, mics_num, T, C)

        mic_orien_emb = self.orien_encoder(mic_oriens)
        # shape: (B, mics_num, T, C)

        agent_pos_emb = self.pos_encoder(agent_poss)
        # shape: (B, agents_num, T, C)

        agent_dir_emb = self.orien_encoder(agent_look_at_directions)
        # shape: (B, agents_num, T, C)

        agent_dist_emb = self.dist_encoder(agent_look_at_distances)
        # shape: (B, agents_num, T, C)

        # ------ 2. Extract features ------

        sum_mic_wavs = torch.sum(mic_wavs, dim=1, keepdim=True)
        sum_mic_stfts = self.stft(sum_mic_wavs)

        frames_num = sum_mic_stfts.shape[2]

        tmp = torch.view_as_real(sum_mic_stfts)
        sum_mic_mag = torch.abs(sum_mic_stfts)
        sum_mic_real = tmp[..., 0]
        sum_mic_imag = tmp[..., 1]

        mic_stfts = self.stft(mic_wavs)
        mic_mag = torch.abs(mic_stfts)
        mic_angle = torch.angle(mic_stfts)
        mic_cos = torch.cos(mic_angle)
        mic_sin = torch.sin(mic_angle)

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

        mag_feature = torch.cat((mic_mag, sum_mic_mag), dim=1)
        # shape: (bs, mics_num + 1, frames_num, freq_bins)

        phase_feature = torch.cat((mic_cos, mic_sin, delta_cos, delta_sin), dim=1)
        # shape: (bs, mics_num * (mics_num - 1), frames_num, freq_bins)

        mag_feature = self.mic_mag_fc(mag_feature)  # (bs, 32, frames_num, freq_bins)
        phase_feature = self.mic_phase_fc(phase_feature)  # (bs, 32, frames_num, freq_bins)
        
        x = torch.cat((mag_feature, phase_feature), dim=1)
        # from IPython import embed; embed(using=False); os._exit(0)

        '''
        mic_stfts = self.stft(mic_wavs)
        # shape: (B, mics_num, T, F)

        frames_num = mic_stfts.shape[2]

        # Magnitude feature maps
        mags = torch.abs(mic_stfts)
        mag_feat = self.mic_mag_fc(mags)  # (bs, 32, frames_num, freq_bins)
        # shape: (B, 32, T, F)

        # Feature feature maps
        phases = torch.angle(mic_stfts)
        diff_phases = self.calculate_diff_phases(phases)
        total_phases = torch.cat((phases, diff_phases), dim=1)

        phase_feat = self.angle_encoder(total_phases)
        phase_feat = rearrange(phase_feat, 'b c t f k -> b (c k) t f')
        phase_feat = self.mic_phase_fc(phase_feat)
        # shape: (B, 32, T, F)
        
        x = torch.cat((mag_feat, phase_feat), dim=1)
        # shape: (B, 64, T, F)
        '''

        x = self.process_image(x)
        # shape: (B, 64, T, F)

        # ------ 3. Forward data to the network ------

        mic_feat_dict = self.mic_wav_encoder(x)

        mic_wav_feat = rearrange(mic_feat_dict["block3"]["x"], 'b c t f -> b t (c f)')
        # shape: (bs, T, 4096)

        mic_wav_feat = F.leaky_relu_(self.mic_wav_fc(mic_wav_feat))
        # shape: (bs, T, 1024)

        # Microphone positions feature.
        mic_pos_emb = self.process_image(mic_pos_emb, freq_axis=False)
        mic_pos_emb = rearrange(mic_pos_emb, 'b c t f -> b t (c f)')
        mic_pos_feat = F.leaky_relu_(self.mic_pos_fc(mic_pos_emb))
        mic_pos_feat = mic_pos_feat[:, 0 :: self.downsample_ratio, :]
        # mic_pos_feat: (B, T', C)
        
        # Microphone orientations feature.
        mic_orien_emb = self.process_image(mic_orien_emb, freq_axis=False)
        mic_orien_emb = rearrange(mic_orien_emb, 'b c t f -> b t (c f)')
        mic_orien_feat = F.leaky_relu_(self.mic_orien_fc(mic_orien_emb))  
        mic_orien_feat = mic_orien_feat[:, 0 :: self.downsample_ratio, :]
        # mic_orien_feat: (B, T', C)

        # Concatenate mic signal, position, and look direction features.
        mic_feat = torch.cat((mic_wav_feat, mic_pos_feat, mic_orien_feat), dim=-1)
        # mic_feat: (B, T', 1280)

        agents_num = agent_poss.shape[1]
        mic_feat = mic_feat[:, None, :, :].repeat_interleave(repeats=agents_num, dim=1)
        # mic_feat: (B, R, T', 1280)

        # ------ 4. Add agents information ------
        agent_pos_emb = self.process_image(agent_pos_emb, freq_axis=False)
        agent_pos_feat = F.leaky_relu_(self.agent_pos_fc(agent_pos_emb))  
        agent_pos_feat = agent_pos_feat[:, :, 0 :: self.downsample_ratio, :]
        # agent_pos_feat: (B, R, T', 128)

        agent_dir_emb = self.process_image(agent_dir_emb, freq_axis=False)
        agent_dir_feat = F.leaky_relu_(self.agent_dir_fc(agent_dir_emb))
        agent_dir_feat = agent_dir_feat[:, :, 0 :: self.downsample_ratio, :]
        # agent_dir_feat: (B, R, T', 128)

        agent_dist_emb = self.process_image(agent_dist_emb, freq_axis=False)
        agent_dist_feat = F.leaky_relu_(self.agent_dist_fc(agent_dist_emb))
        agent_dist_feat = agent_dist_feat[:, :, 0 :: self.downsample_ratio, :]
        # agent_dist_feat: (B, R, T', 128)

        agent_distance_masks = agent_distance_masks[:, :, 0 :: self.downsample_ratio, None]
        # shape: (B, R, T', 1)

        # If no distance information is provided then set to 0.
        agent_dist_feat = agent_distance_masks * agent_dist_feat
        # shape: (B, R, T', 128)

        total_feat = torch.cat((mic_feat, agent_pos_feat, agent_dir_feat, agent_dist_feat), dim=-1)
        # shape: (B, R, T', 1664)

        # ------ 5a. Spatial detection. ------
        agent_detect_idxes = data["agent_detect_idxes"]
        det_feat = get_tensor_from_indexes(total_feat, agent_detect_idxes)
        det_feat = self.det_mlp(det_feat)
        # shape: (B, R_det, T', 1)

        look_at_direction_has_source = det_feat.repeat_interleave(
            repeats=self.downsample_ratio, dim=2)[:, :, 0 : frames_num, 0]
        # shape: (B, R_det, T)
        
        # ------ 5b. Spatial distance estimation. ------
        agent_dist_idxes = data["agent_distance_idxes"]
        dist_feat = get_tensor_from_indexes(total_feat, agent_dist_idxes)
        dist_feat = self.dist_mlp(dist_feat)
        look_at_distance_has_source = dist_feat.repeat_interleave(repeats=self.downsample_ratio, dim=2)[:, :, 0 : frames_num, 0]
        # shape: (B, dist_rays, T)

        output_dict = {
            "agent_look_at_direction_has_source": look_at_direction_has_source,
            "agent_look_at_distance_has_source": look_at_distance_has_source
        }

        # ------ 5c. Spatial source separation. ------
        agent_sep_idxes = data["agent_sep_idxes"]
        sep_rays = agent_sep_idxes.shape[1]

        if sep_rays > 0:
            
            sep_feat = get_tensor_from_indexes(total_feat, agent_sep_idxes)
            # shape: (B, R_sep, T', 1664)

            x = F.leaky_relu_(self.sep_fc(sep_feat))
            # shape: (B, R_sep, T', 4096)

            x = rearrange(x, 'b n t (c f) -> (b n) c t f', c=128, f=32)
            # shape: (B, R_sep, C, T'=38, F=32)

            sum_mic_wav = torch.sum(mic_wavs, dim=1, keepdim=True)

            look_at_direction_wav = self.sep_reverb_decoder(
                x=x, 
                wav_feat_dict=mic_feat_dict, 
                repeat_rays=sep_rays, 
                frames_num=frames_num,
                sum_mic_wav=sum_mic_wav
            )

            output_dict["agent_look_at_direction_reverb_wav"] = look_at_direction_wav
            
        return output_dict

    def calculate_diff_phases(self, mic_phase):

        mics_num = mic_phase.shape[1]

        diff_phases = []

        for i in range(1, mics_num):
            for j in range(0, i):
                diff_phases.append(mic_phase[:, i, :, :] - mic_phase[:, j, :, :])

        diff_phases = torch.stack(diff_phases, dim=1)

        return diff_phases

    def process_image(self, x, time_axis=True, freq_axis=True):
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


'''
class NeSD6(Fourier):
    def __init__(self, 
        mics_num: int,
        n_fft=512,
        hop_length=240,
    ):
        super(NeSD6, self).__init__(n_fft=n_fft, hop_length=hop_length) 

        self.downsample_ratio = 8

        pos_enc_size = 5
        orien_enc_size = 5
        dist_enc_size = 5
        angle_enc_size = 1  # If too large will increase GPU memory.

        self.pos_encoder = PositionEncoder(size=pos_enc_size)
        self.orien_encoder = OrientationEncoder(size=orien_enc_size)
        self.dist_encoder = DistanceEncoder(size=dist_enc_size)
        self.angle_encoder = DistanceEncoder(size=angle_enc_size)

        # Mic magnitude process layer.
        self.mic_mag_fc = nn.Conv2d(
            in_channels=mics_num + 1,
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
            bias=True
        )

        # Mic phase process layer.
        in_channels = (mics_num + mics_num * (mics_num - 1) // 2) * (angle_enc_size * 2)
        self.mic_phase_fc = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=(1, 1), 
            stride=(1, 1),
            padding=(0, 0), 
            bias=True
        )

        self.mic_wav_encoder = AudioEncoder(
            in_channels=64,
            out_channels=128
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

        # Detection MLP layers.
        self.det_mlp = MLP(in_channels=1664, hid_channels=1024, out_channels=1)

        # Distance MLP layers.
        self.dist_mlp = MLP(in_channels=1664, hid_channels=1024, out_channels=1)

        # Separation layers.
        self.sep_fc = nn.Linear(1664, 4096, bias=True)

        self.sep_reverb_decoder = AudioDecoder(
            in_channels=128, 
            n_fft=n_fft, 
            hop_length=hop_length
        )
        # self.sep_direct_decoder = AudioEncoder(in_channels=128)

    def forward(
        self, 
        data: Dict, 
    ):

        mic_poss = data["mic_positions"]
        mic_oriens = data["mic_orientations"]
        mic_wavs = data["mic_wavs"]

        agent_poss = data["agent_positions"]
        agent_look_at_directions = data["agent_look_at_directions"]
        agent_look_at_distances = data["agent_look_at_distances"]
        agent_distance_masks = data["agent_distance_masks"]

        batch_size = agent_poss.shape[0]

        # ------ 1. Convert positions and look at directions to embeddings ------
        mic_pos_emb = self.pos_encoder(mic_poss)
        # shape: (B, mics_num, T, C)

        mic_orien_emb = self.orien_encoder(mic_oriens)
        # shape: (B, mics_num, T, C)

        agent_pos_emb = self.pos_encoder(agent_poss)
        # shape: (B, agents_num, T, C)

        agent_dir_emb = self.orien_encoder(agent_look_at_directions)
        # shape: (B, agents_num, T, C)

        agent_dist_emb = self.dist_encoder(agent_look_at_distances)
        # shape: (B, agents_num, T, C)

        # ------ 2. Extract features ------

        """
        sum_mic_wavs = torch.sum(mic_wavs, dim=1, keepdim=True)
        sum_mic_stfts = self.stft(sum_mic_wavs)

        frames_num = sum_mic_stfts.shape[2]

        tmp = torch.view_as_real(sum_mic_stfts)
        sum_mic_mag = torch.abs(sum_mic_stfts)
        sum_mic_real = tmp[..., 0]
        sum_mic_imag = tmp[..., 1]

        mic_stfts = self.stft(mic_wavs)
        mic_mag = torch.abs(mic_stfts)
        mic_angle = torch.angle(mic_stfts)
        mic_cos = torch.cos(mic_angle)
        mic_sin = torch.sin(mic_angle)

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

        mag_feature = torch.cat((mic_mag, sum_mic_mag), dim=1)
        # shape: (bs, mics_num + 1, frames_num, freq_bins)

        phase_feature = torch.cat((mic_cos, mic_sin, delta_cos, delta_sin), dim=1)
        # shape: (bs, mics_num * (mics_num - 1), frames_num, freq_bins)

        mag_feature = self.mic_mag_fc(mag_feature)  # (bs, 32, frames_num, freq_bins)
        phase_feature = self.mic_phase_fc(phase_feature)  # (bs, 32, frames_num, freq_bins)
        
        x = torch.cat((mag_feature, phase_feature), dim=1)
        # from IPython import embed; embed(using=False); os._exit(0)
        """

        sum_mic_wavs = torch.sum(mic_wavs, dim=1, keepdim=True)
        sum_mic_stfts = self.stft(sum_mic_wavs)
        sum_mic_mag = torch.abs(sum_mic_stfts)
        
        mic_stfts = self.stft(mic_wavs)
        # shape: (B, mics_num, T, F)

        frames_num = mic_stfts.shape[2]

        # Magnitude feature maps
        mags = torch.abs(mic_stfts)
        total_mags = torch.cat((mags, sum_mic_mag), dim=1)
        mag_feat = self.mic_mag_fc(total_mags)  # (bs, 32, frames_num, freq_bins)
        # shape: (B, 32, T, F)

        # Feature feature maps
        phases = torch.angle(mic_stfts)
        diff_phases = self.calculate_diff_phases(phases)
        total_phases = torch.cat((phases, diff_phases), dim=1)

        phase_feat = self.angle_encoder(total_phases)
        phase_feat = rearrange(phase_feat, 'b c t f k -> b (c k) t f')
        phase_feat = self.mic_phase_fc(phase_feat)
        # shape: (B, 32, T, F)
        
        x = torch.cat((mag_feat, phase_feat), dim=1)
        # shape: (B, 64, T, F)
        

        x = self.process_image(x)
        # shape: (B, 64, T, F)

        # ------ 3. Forward data to the network ------

        mic_feat_dict = self.mic_wav_encoder(x)

        mic_wav_feat = rearrange(mic_feat_dict["block3"]["x"], 'b c t f -> b t (c f)')
        # shape: (bs, T, 4096)

        mic_wav_feat = F.leaky_relu_(self.mic_wav_fc(mic_wav_feat))
        # shape: (bs, T, 1024)

        # Microphone positions feature.
        mic_pos_emb = self.process_image(mic_pos_emb, freq_axis=False)
        mic_pos_emb = rearrange(mic_pos_emb, 'b c t f -> b t (c f)')
        mic_pos_feat = F.leaky_relu_(self.mic_pos_fc(mic_pos_emb))
        mic_pos_feat = mic_pos_feat[:, 0 :: self.downsample_ratio, :]
        # mic_pos_feat: (B, T', C)
        
        # Microphone orientations feature.
        mic_orien_emb = self.process_image(mic_orien_emb, freq_axis=False)
        mic_orien_emb = rearrange(mic_orien_emb, 'b c t f -> b t (c f)')
        mic_orien_feat = F.leaky_relu_(self.mic_orien_fc(mic_orien_emb))  
        mic_orien_feat = mic_orien_feat[:, 0 :: self.downsample_ratio, :]
        # mic_orien_feat: (B, T', C)

        # Concatenate mic signal, position, and look direction features.
        mic_feat = torch.cat((mic_wav_feat, mic_pos_feat, mic_orien_feat), dim=-1)
        # mic_feat: (B, T', 1280)

        agents_num = agent_poss.shape[1]
        mic_feat = mic_feat[:, None, :, :].repeat_interleave(repeats=agents_num, dim=1)
        # mic_feat: (B, R, T', 1280)

        # ------ 4. Add agents information ------
        agent_pos_emb = self.process_image(agent_pos_emb, freq_axis=False)
        agent_pos_feat = F.leaky_relu_(self.agent_pos_fc(agent_pos_emb))  
        agent_pos_feat = agent_pos_feat[:, :, 0 :: self.downsample_ratio, :]
        # agent_pos_feat: (B, R, T', 128)

        agent_dir_emb = self.process_image(agent_dir_emb, freq_axis=False)
        agent_dir_feat = F.leaky_relu_(self.agent_dir_fc(agent_dir_emb))
        agent_dir_feat = agent_dir_feat[:, :, 0 :: self.downsample_ratio, :]
        # agent_dir_feat: (B, R, T', 128)

        agent_dist_emb = self.process_image(agent_dist_emb, freq_axis=False)
        agent_dist_feat = F.leaky_relu_(self.agent_dist_fc(agent_dist_emb))
        agent_dist_feat = agent_dist_feat[:, :, 0 :: self.downsample_ratio, :]
        # agent_dist_feat: (B, R, T', 128)

        agent_distance_masks = agent_distance_masks[:, :, 0 :: self.downsample_ratio, None]
        # shape: (B, R, T', 1)

        # If no distance information is provided then set to 0.
        agent_dist_feat = agent_distance_masks * agent_dist_feat
        # shape: (B, R, T', 128)

        total_feat = torch.cat((mic_feat, agent_pos_feat, agent_dir_feat, agent_dist_feat), dim=-1)
        # shape: (B, R, T', 1664)

        # ------ 5a. Spatial detection. ------
        agent_detect_idxes = data["agent_detect_idxes"]
        det_feat = get_tensor_from_indexes(total_feat, agent_detect_idxes)
        det_feat = self.det_mlp(det_feat)
        # shape: (B, R_det, T', 1)

        look_at_direction_has_source = det_feat.repeat_interleave(
            repeats=self.downsample_ratio, dim=2)[:, :, 0 : frames_num, 0]
        # shape: (B, R_det, T)
        
        # ------ 5b. Spatial distance estimation. ------
        agent_dist_idxes = data["agent_distance_idxes"]
        dist_feat = get_tensor_from_indexes(total_feat, agent_dist_idxes)
        dist_feat = self.dist_mlp(dist_feat)
        look_at_distance_has_source = dist_feat.repeat_interleave(repeats=self.downsample_ratio, dim=2)[:, :, 0 : frames_num, 0]
        # shape: (B, dist_rays, T)

        output_dict = {
            "agent_look_at_direction_has_source": look_at_direction_has_source,
            "agent_look_at_distance_has_source": look_at_distance_has_source
        }

        # ------ 5c. Spatial source separation. ------
        agent_sep_idxes = data["agent_sep_idxes"]
        sep_rays = agent_sep_idxes.shape[1]

        if sep_rays > 0:
            
            sep_feat = get_tensor_from_indexes(total_feat, agent_sep_idxes)
            # shape: (B, R_sep, T', 1664)

            x = F.leaky_relu_(self.sep_fc(sep_feat))
            # shape: (B, R_sep, T', 4096)

            x = rearrange(x, 'b n t (c f) -> (b n) c t f', c=128, f=32)
            # shape: (B, R_sep, C, T'=38, F=32)

            sum_mic_wav = torch.sum(mic_wavs, dim=1, keepdim=True)

            look_at_direction_wav = self.sep_reverb_decoder(
                x=x, 
                wav_feat_dict=mic_feat_dict, 
                repeat_rays=sep_rays, 
                frames_num=frames_num,
                sum_mic_wav=sum_mic_wav
            )

            output_dict["agent_look_at_direction_reverb_wav"] = look_at_direction_wav
            
        return output_dict

    def calculate_diff_phases(self, mic_phase):

        mics_num = mic_phase.shape[1]

        diff_phases = []

        for i in range(1, mics_num):
            for j in range(0, i):
                diff_phases.append(mic_phase[:, i, :, :] - mic_phase[:, j, :, :])

        diff_phases = torch.stack(diff_phases, dim=1)

        return diff_phases

    def process_image(self, x, time_axis=True, freq_axis=True):
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
'''