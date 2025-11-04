from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile
from torch import Tensor
import torch
from einops import rearrange
import matplotlib.pyplot as plt

from nesd.utils.utils import parse_yaml
from nesd.utils.torch import sph2cart, normalize
from train import get_model
import pickle
import torch.nn.functional as F


def inference(args) -> None:
    r"""Separate an audio."""

    # Arguments and parameters
    config_yaml = args.config_yaml
    ckpt_path = args.ckpt_path
    audio_path = args.audio_path
    frames_num = 201
    
    device = "cuda"
    # batch_size = 4
    
    # Default parameters
    configs = parse_yaml(config_yaml)
    sr = configs["sample_rate"]
    seg_duration = configs["segment_duration"]
    seg_samples = round(seg_duration * sr)

    # Model
    model = get_model(
        configs=configs, 
        ckpt_path=ckpt_path
    ).to(device)

    # Load audio
    # audio, _ = librosa.load(path=audio_path, sr=sr, mono=False)  # shape: (c, l)
    # audio = np.zeros((4, 96000))
    
    # audio = Tensor(audio[None, :, :]).to(device)

    data = pickle.load(open("_zz.pkl", "rb"))
    audio = torch.Tensor(data["mic_wav"]).to(device)
    lis_dir = torch.Tensor(data["lis_dir"]).to(device)
    target = torch.Tensor(data["target"]).to(device)
        
    with torch.no_grad():
        model.eval()
        out = model(audio, lis_dir)
        
    F.binary_cross_entropy(out, target)
    print(out.max())

    # plt.matshow(outputs, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    # plt.savefig("_zz.pdf")
    from IPython import embed; embed(using=False); os._exit(0)

    ele = torch.linspace(0, 180, 181).deg2rad()  # 5 points along x
    azi = torch.linspace(0, 360, 360).deg2rad()  # 3 points along y
    ele, azi = torch.meshgrid(ele, azi, indexing='ij')  # (181, 361)
    r = torch.ones_like(ele)

    sph = torch.stack([r, ele, azi], dim=-1)
    dirn = normalize(sph2cart(sph))
    
    dirn = rearrange(dirn, 'el az d -> (el az) d')

    i = 0
    R = 100
    outputs = []
    while i < dirn.shape[0]:
        print(i)

        lis_dir = dirn[None, None, i : i + R, None, :].repeat(1, 1, 1, frames_num, 1)  # (b, l, r', t, 3)
        lis_dir = lis_dir.to(device)
        
        with torch.no_grad():
            model.eval()
            out = model(audio[0:1, :, :], lis_dir)
            # out = torch.mean(out.cpu()[0, 0, :, :, 0], dim=-1)
            out, _ = torch.max(out.cpu()[0, 0, :, :, 0], dim=-1)
            outputs.append(out)
            
        i += R

    outputs = torch.cat(outputs, axis=0)
    outputs = rearrange(outputs, '(el az) -> el az', el=181)

    plt.matshow(outputs, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    plt.savefig("_zz.pdf")
    from IPython import embed; embed(using=False); os._exit(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--audio_path', type=str, required=True)

    args = parser.parse_args()

    inference(args)