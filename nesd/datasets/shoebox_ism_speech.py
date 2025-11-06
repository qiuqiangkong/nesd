from __future__ import annotations

from typing import Dict

import h5py
import numpy as np
import torch
import torchaudio
import random
import pyroomacoustics as pra
import pandas as pd
import librosa
from pathlib import Path
from librosa.util import fix_length

from rooms.shoebox_ism import ShoeboxISM
from nesd.utils.utils import list_of_list_to_array, list_of_list_to_mask


class ShoeboxISMSpeech:
    r"""Convert environment, speech, source, microphone, and image source data 
    into tensors.
    """

    def __init__(self, config: dict, sr: float):

        self.sr = sr

        self.room = ShoeboxISM(
            env_config=config["environment"],
            mic_csv=config["microphones"]["meta"], 
            min_srcs=config["sources"]["min_srcs"], 
            max_srcs=config["sources"]["max_srcs"], 
            min_ism_order=config["image_source_method"]["min_order"], 
            max_ism_order=config["image_source_method"]["max_order"]
        )

        self.audio_paths = sorted(list(Path(config["sources"]["audios_dir"]).glob("*.wav")))
        
    def __getitem__(self, index) -> dict:
        r"""Get an item.

        s: srcs_num
        m: mics_num
        l: lis_num
        i: max_img_srcs

        Returns:
            data: dict
        """
        
        data = self.room.sample()

        srcs_num = data["srcs_num"]
        mics_num = data["mics_num"]
        lis_num = data["lis_num"]
        max_srcs = data["max_srcs"]
        max_imgs = data["max_imgs"]
        
        src_masks = np.zeros(max_srcs, dtype=np.float32)  # (s,)
        src_masks[0 : srcs_num] = 1
        src_pos = fix_length(data=data["src_pos"], size=max_srcs, axis=0)  # (s, 3)
        mic_pos = fix_length(data=data["mic_pos"], size=mics_num, axis=0)  # (m, 3)
        mic_dir = fix_length(data=data["mic_dir"], size=mics_num, axis=0)  # (m, 3)
        lis_pos = fix_length(data=data["lis_pos"], size=lis_num, axis=0)  # (l, 3)

        mic_imgs = list_of_list_to_array((max_srcs, mics_num, max_imgs, 3), data["mic_img"], np.float32)
        mic_orders = list_of_list_to_array((max_srcs, mics_num, max_imgs), data["mic_order"], np.int32)
        mic_masks = list_of_list_to_mask((max_srcs, mics_num, max_imgs), data["mic_img"], np.float32)

        lis_imgs = list_of_list_to_array((max_srcs, lis_num, max_imgs, 3), data["lis_img"], np.float32)
        lis_orders = list_of_list_to_array((max_srcs, lis_num, max_imgs), data["lis_order"], np.int32)
        lis_masks = list_of_list_to_mask((max_srcs, lis_num, max_imgs), data["lis_img"], np.float32)
        
        # Load audio
        srcs = np.zeros((max_srcs, self.sr * 2), dtype=np.float32)  # (s, l_audio)

        for s in range(srcs_num):
            audio_path = random.choice(self.audio_paths)
            audio, _ = librosa.load(path=audio_path, sr=self.sr, mono=True)
            srcs[s] = audio

        new_data = {
            "src": srcs,  # (s, l_audio)
            "src_pos": src_pos,  # (s, 3)
            "src_mask": src_masks,  # (s,)
            "srcs_num": srcs_num,  # scalar
            "max_srcs": max_srcs,  # scalar
            "max_imgs": max_imgs,  # scalar
            "mic_pos": mic_pos,  # (m, 3)
            "mic_dir": mic_dir,  # (m, 3)
            "mic_img": mic_imgs,  # (s, m, i, 3)
            "mic_order": mic_orders,  # (s, m, i)
            "mic_rot_mat": data["mic_rot_mat"],  # (3, 3)
            "mic_mask": mic_masks,  # (s, m, i)
            "mics_num": mics_num,  # scalar
            "lis_pos": lis_pos,  # (l, 3)
            "lis_img": lis_imgs,  # (s, l, i, 3)
            "lis_order": lis_orders,  # (s, l, i)
            "lis_rot_mat": data["lis_rot_mat"],  # (3, 3)
            "lis_mask": lis_masks,  # (s, l, i)
            "lis_num": lis_num  # scalar
        }
        
        return new_data

    def __len__(self) -> int:
        return 10000