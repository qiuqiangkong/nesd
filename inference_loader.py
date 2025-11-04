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

from nesd.utils.utils import parse_yaml, to_device
from nesd.utils.torch import sph2cart, normalize
from torch.utils.data._utils.collate import default_collate

from train import get_model, get_dataset, get_data_transform



def inference(args) -> None:
    r"""Separate an audio."""

    # Arguments and parameters
    config_yaml = args.config_yaml
    ckpt_path = args.ckpt_path
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

    test_dataset = get_dataset(configs, split="test")
    data_transform = get_data_transform(configs).to(device)

    for n, data in enumerate(test_dataset):
        
        data = default_collate([data])
        data = to_device(data, device)
        data = data_transform(data)

        audio = Tensor(data["mic_wav"]).to(device)
        break

    # audio = Tensor(audio[None, :, :]).to(device)
    
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
            out = model(audio, lis_dir)
            # out = torch.mean(out.cpu()[0, 0, :, :, 0], dim=-1)
            out, _ = torch.max(out.cpu()[0, 0, :, :, 0], dim=-1)
            outputs.append(out)
            
        i += R

    outputs = torch.cat(outputs, axis=0)
    outputs = rearrange(outputs, '(el az) -> el az', el=181)
    # outputs = rearrange(outputs, '(az el) -> el az', el=181)

    print(outputs.max())

    plt.matshow(outputs, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    plt.savefig("_zz.pdf")
    print(data['target'].mean())
    from IPython import embed; embed(using=False); os._exit(0)



    # Foward
    output = separate(
        model=model, 
        audio=audio, 
        clip_samples=clip_samples, 
        batch_size=batch_size
    )  # shape: (c, l)

    # Write out to MIDI
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    soundfile.write(file=output_path, data=output.T, samplerate=sr)
    print("Write out to {}".format(output_path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)

    args = parser.parse_args()

    inference(args)