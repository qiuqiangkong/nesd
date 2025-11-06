from __future__ import annotations

import argparse
import io
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import imageio.v2 as imageio
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import torch
from einops import rearrange
from torch import Tensor
from torch.utils.data._utils.collate import default_collate

from nesd.utils.torch import normalize, sph2cart
from nesd.utils.utils import parse_yaml
from train import get_data_transform, get_dataset, get_model


def inference(args) -> None:
    r"""Separate an audio."""

    # Arguments
    config_yaml = args.config_yaml
    ckpt_path = args.ckpt_path
    audio_path = args.audio_path
    video_path = args.video_path
    out_path = args.out_path
    
    # Parameters
    configs = parse_yaml(config_yaml)
    sr = configs["sample_rate"]
    seg_duration = configs["segment_duration"]
    seg_samples = round(seg_duration * sr)
    frames_num = int(seg_duration * configs["fps"]) + 1
    pred_fps = 10
    device = "cuda"

    if not audio_path:
        test_dataset = get_dataset(configs, split="test")
        data_transform = get_data_transform(configs)
        data = test_dataset[0]
        data = default_collate([data])
        data = data_transform(data)
        audio = data["mic_wav"][0]  # (m, l_audio)

    else:
        audio, _ = librosa.load(path=audio_path, sr=sr, mono=False)  # (m, l_audio)
        # audio = audio[:, 0 : 10 * sr] 
        
    # Model
    model = get_model(
        configs=configs, 
        ckpt_path=ckpt_path
    ).to(device)
    
    # Collect all directions on the panorama.
    ele = torch.linspace(0, 180, 181).deg2rad()  # 5 points along x
    azi = torch.linspace(0, 360, 360).deg2rad()  # 3 points along y
    ele, azi = torch.meshgrid(ele, azi, indexing='ij')  # (181, 360)
    r = torch.ones_like(ele)

    sph = torch.stack([r, ele, azi], dim=-1)  # (181, 360, 3)
    dirn = normalize(sph2cart(sph))  # (181, 360, 3)
    dirn = rearrange(dirn, 'el az d -> (el az) d')

    bgn_sample = 0
    R = 10000  # rays_num
    outputs = []

    # Predict all directions on the panaroma
    while bgn_sample < audio.shape[-1]:

        print("{} s".format(bgn_sample / sr))
        seg = audio[:, bgn_sample : bgn_sample + seg_samples]
        seg = librosa.util.fix_length(seg, size=seg_samples, axis=-1)
        seg = Tensor(seg[None, :, :]).to(device)

        i = 0
        outs = []
    
        while i < dirn.shape[0]:

            lis_dir = dirn[None, None, i : i + R, None, :].repeat(1, 1, 1, frames_num, 1)  # (b, l, r', t, 3)
            lis_dir = lis_dir.to(device)
            
            with torch.no_grad():
                model.eval()
                out = model(seg, lis_dir)
                out = out.cpu()[0, 0, :, 0 : -1, 0]  # (b, t)
                out = rearrange(out, 'b (t1 t2) -> b t1 t2', t2=10)  # Downsample by 10x
                out = out.mean(dim=-1)
                outs.append(out)
                
            i += R

        outs = torch.cat(outs, axis=0)
        outs = rearrange(outs, '(el az) t -> t el az', el=181)
        outputs.append(outs)
        bgn_sample += seg_samples

    outputs = torch.cat(outputs, dim=0)
    indices = ((torch.arange(359, -0.1, -1) - 180) % 360).long()
    outputs = outputs[:, :, indices]

    # Expand predicted panaroma to match video size
    Q = 1
    outputs = outputs.numpy()
    outputs = outputs.repeat(Q, axis=0).repeat(Q, axis=1).repeat(Q, axis=2)

    t1 = time.time()    
    
    if video_path:

        h5_path = Path("./results/video_h5/{}.h5".format(Path(video_path).stem))
        
        if not Path(h5_path).is_file():
            print("Caching video ..")
            cache_video_to_hdf5(video_path, h5_path, pred_fps, Q)    

        with h5py.File(h5_path, 'r') as hf:
            video = hf["x"][:] / 255.  # (T, H, W, 3)
        video = librosa.util.fix_length(data=video, size=outputs.shape[0], axis=0)
        
        cmap = plt.get_cmap('jet')
        outputs = cmap(outputs)[:, :, :, :3]
        
        outputs = 0.7 * video + 0.3 * outputs
        outputs = np.clip(outputs, 0., 1.)
        
    # Parallel matshow
    params = [(outputs[i], i / (pred_fps * Q), Q) for i in range(outputs.shape[0])]

    if False:  # For debug
        for i, param in enumerate(params):
            plot_one_frame(param)
            from IPython import embed; embed(using=False); os._exit(0)
    
    with ProcessPoolExecutor(max_workers=32) as pool: # Maximum workers on the machine.
        results = pool.map(plot_one_frame, params)

    # Save to video
    writer = imageio.get_writer("./_tmp1.mp4", fps=pred_fps*Q, codec="libx264")
    for img in results:
        writer.append_data(img)
    writer.close()

    # Save to audio
    soundfile.write(file="./_tmp1.wav", data=audio.mean(axis=0), samplerate=sr)

    # Combine video and audio
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    os.system(f"ffmpeg -y -i _tmp1.mp4 -i _tmp1.wav -c:v copy -c:a aac -shortest {out_path}")

    writer.close()
    print("Time: {:.02f} s".format(time.time() - t1))
    print("Max value: {:.04f}".format(outputs.max()))


def cache_video_to_hdf5(video_path: str, h5_path: str, pred_fps: int, Q: int) -> None:
    r"""Cache video to hdf5 for fast load"""
    from moviepy import VideoFileClip
    clip = VideoFileClip(video_path)
    clip = clip.resized((360 * Q, 180 * Q)).with_fps(pred_fps * Q)
    frames = [frame for frame in clip.iter_frames()]
    frames = np.stack(frames, axis=0)
    Path(h5_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, 'w') as hf:
        hf.create_dataset('x', data=frames, dtype=np.uint8)
    print(f"Cache video to {h5_path}")


def plot_one_frame(param: tuple) -> np.ndarray:
    frame, time, Q = param
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.matshow(frame, cmap='jet', origin='upper', vmin=0, vmax=1)
    ax.set_title(f"Time {time:.02f} s")
    ax.grid(color='w', linestyle='--', linewidth=0.2)
    ax.xaxis.set_ticks(np.arange(0, 361*Q, 10*Q))
    ax.xaxis.set_ticklabels(np.arange(180, -181, -10), rotation=90)
    ax.yaxis.set_ticks(np.arange(0, 181*Q, 10*Q))
    ax.yaxis.set_ticklabels(np.arange(0, 181, 10))
    ax.set_xlabel("Azimuth")
    ax.set_ylabel("Elevation")
    ax.xaxis.tick_bottom()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = imageio.imread(buf)
    return img


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--audio_path', type=str, required=False)
    parser.add_argument('--video_path', type=str, required=False)
    parser.add_argument('--out_path', type=str, required=True)

    args = parser.parse_args()

    inference(args)