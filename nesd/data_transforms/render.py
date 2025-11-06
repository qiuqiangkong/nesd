from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio
import torch
from torch import Tensor
from einops import rearrange
import h5py
import numpy as np
import torch.nn.functional as F
import random

from nesd.utils.torch import (included_angle, get_delayed_filter, convolve, 
    add_delayed_filters, db_to_gain, transform_coordinate, sample_direction, 
    is_within_angle, perturb_direction, normalize)


class AcousticRenderer(nn.Module):
    r"""Render audio of all microphones from sources and spaital impulse 
    responses. Sample listener directions, and compute targets.
    """

    def __init__(self, config: dict):
        super().__init__()

        self.sr = config["sample_rate"]
        self.c = config["sound_speed"]
        self.filter_len = config["filter_len"]
        self.half_angle = np.deg2rad(config["half_angle_deg"]).item()
        self.rays_num = config["rays_num"]
        self.frames_num = int(config["segment_duration"] * config["fps"]) + 1
        
        with h5py.File(config["mic_sir"], 'r') as hf:
            self.register_buffer(name="mic_hs", tensor=torch.Tensor(hf["h"][:]))  # (angles_num, nfft)
            self.h_mic_origin = hf.attrs["origin"]  # scalar

        self._dummy = nn.Parameter(torch.zeros(1))

    def __call__(self, data: dict) -> dict:
        r"""Transform data into latent representations and conditions.

        b: batch_size
        s: srcs_num
        m: mics_num
        i: imgs_num
        l: lis_num
        r: rays_num

        Args:
            data (dict)

        Returns:
            new_data (dict)
        """
        
        device = next(self.parameters()).device

        src = data["src"]  # (b, s, l_audio)
        src_pos = data["src_pos"]  # (b, s, 3)
        src_masks = data["src_mask"]  # (b, s)
        srcs_num = data["srcs_num"]  # (b,)
        max_srcs = data["max_srcs"][0].item()  # (b,)
        max_imgs = data["max_imgs"][0].item()  # (b,)
        mic_pos = data["mic_pos"]  # (b, m, 3)
        mic_dir = data["mic_dir"]  # (b, m, 3)
        mic_imgs = data["mic_img"]  # (b, s, m, i, 3)
        mic_orders = data["mic_order"]  # (b, s, m, i)
        mic_rot_mat = data["mic_rot_mat"]  # (3, 3)
        mic_masks = data["mic_mask"]  # (b, s, m, i)
        lis_pos = data["lis_pos"]  # (b, l, 3)
        lis_imgs = data["lis_img"]  # (b, s, l, i, 3)
        lis_orders = data["lis_order"]  # (b, s, l, i)
        lis_rot_mat = data["lis_rot_mat"]  # (3, 3)
        lis_masks = data["lis_mask"]  # (b, s, l, i)
        lis_num = data["lis_num"]  # (b,)

        # === 1. Compute SIR between source images and microphones ===
        # 1.1 Compute IR of mics
        doa = mic_imgs - mic_pos[:, None, :, None, :]  # (b, s, m, i, 3)
        theta = torch.rad2deg(included_angle(mic_dir[:, None, :, None, :], doa))  # (b, s, m, i)
        theta = theta.round().long()  # (b, s, m, i)
        h_mic = self.mic_hs[theta]  # (b, s, m, i, l_h_mic)

        # 1.2 Compute IR of delays
        dist = torch.norm(doa, dim=-1)  # (b, s, m, i)
        delayed_samples = (dist / self.c) * self.sr  # (b, s, m, i)
        delay_int, h_frac, h_frac_origin = get_delayed_filter(delayed_samples)
        # delay_int: (b, s, m, i), h_frac: (b, s, m, i, l_h_frac)

        # 1.3 Compute amplitude gain
        # Dist amplitude
        dist_gain = 1. / dist  # (b, s, m, i)
        
        #   Reflect amplitude
        alpha_wall = random.uniform(0, 0.5)
        reflect_gain = np.sqrt(1. - alpha_wall) ** mic_orders  # (b, s, m, i)

        #   Rays num amplitude
        imgs_num = torch.sum(mic_masks, dim=-1)  # (b, s, m)
        img_gain = mic_masks / torch.clamp(torch.sqrt(imgs_num[:, :, :, None]), 1e-10)

        #   Total amuplitude
        total_gain = mic_masks * dist_gain * reflect_gain * img_gain  # (b, s, m, i)
        h_frac = h_frac * total_gain[:, :, :, :, None]  # (b, s, m, i, l_h_frac)

        # 1.4 Combine IR of mics and delays
        h_mic_delay_frac = convolve(
            x=h_mic, 
            h=h_frac, 
            origin=h_frac_origin
        )  # (b, s, m, i, l_h)

        # 1.5 Sum IR over all image sources
        h_total = add_delayed_filters(
            h_frac=h_mic_delay_frac, 
            delay_int=delay_int, 
            length=self.filter_len, 
            origin=h_frac_origin
        )  # (b, s, m, l_total)

        # 1.6 Convolve IR with source
        x = convolve(
            x=src[:, :, None, :], 
            h=h_total, 
            origin=self.h_mic_origin
        )[..., 0 : src.shape[-1]]  # (b, s, m, l_audio)

        # 1.7 Normalize energy
        eng_s = (src ** 2).mean(dim=-1)  # (b, s)
        eng_x = (x ** 2).mean(dim=(-2, -1))  # (b, s)
        ratio = torch.sqrt(eng_s / torch.clamp(eng_x, 1e-8))  # (b, s)
        x = x * ratio[:, :, None, None]  # (b, s, m, l_audio)

        # 1.8 Randomly adjust the volume of sources
        db = -40 * torch.rand((src.shape[0], src.shape[1])).to(device)  # (b, s)
        gain = db_to_gain(db)
        x = x * gain[:, :, None, None]

        # 1.9 Merge sources to mic signal
        mic_audio = x.sum(dim=1)  # (b, m, l_audio)

        # === 2. Sample listener directions and compute targets ===
        B = lis_imgs.shape[0]
        L = lis_imgs.shape[2]
        R = self.rays_num
        T = self.frames_num

        # 2.1 Sample negative directions where no sources are present
        doa = lis_imgs[:, :, :, 0, :] - lis_pos[:, None, :, :]  # (b, s, l, 3)

        # Convert source world DOA to local DOA
        local_doa = transform_coordinate(
            x=rearrange(doa, 'b s l d -> b (s l) d'),
            origin_from=torch.zeros(B, 3).to(device),
            origin_to=torch.zeros(B, 3).to(device),
            R_from=torch.eye(3)[None, :, :].repeat(B, 1, 1).to(device),
            R_to=lis_rot_mat
        )
        local_doa = rearrange(local_doa, 'b (s l) d -> b s l d', s=max_srcs)  # (b, s, l, 3)
        
        # Randomly sample directions
        local_lis_dir = sample_direction(size=(B, L, R)).to(device)  # (b, l, r, 3)

        # Set target to 1 when sampled directions coincide with source DOAs.
        flag = is_within_angle(
            vec1=local_doa[:, :, :, None, :],  # (b, s, l, r, 3)
            vec2=local_lis_dir[:, None, :, :, :],  # (b, s, l, r, 3)
            max_ele=self.half_angle, 
            max_azi=self.half_angle
        ).float()  # (b, s, l, r)
        flag *= src_masks[:, :, None, None]  # (b, s, l, r)
        flag = flag.sum(dim=1).bool()  # (b, l, r)

        target = torch.zeros(B, L, R, 1).to(device)  # (b, l, r, 1)
        target[flag] = 1

        # 2.2 Sample positive directions where no sources are present
        positive_dir = perturb_direction(normalize(local_doa), self.half_angle, self.half_angle)  # (b, s, l, 3) 
        positive_dir = rearrange(positive_dir, 'b s l d -> b l s d')  # (b, l, s, 3)
        local_lis_dir[:, :, 0 : max_srcs, :] = positive_dir  # (b, l, r, 3)
        target[:, :, 0 : max_srcs, :] = src_masks[:, None, :, None]  # (b, l, r, 1)
        
        local_lis_dir = local_lis_dir[:, :, :, None, :].repeat(1, 1, 1, T, 1)  # (b, l, r, t, 3)
        target = target[:, :, :, None, :].repeat(1, 1, 1, T, 1)  # (b, l, r, t, 1)

        data = {
            "mic_wav": mic_audio,
            "lis_dir": local_lis_dir,
            "target": target
        }

        return data