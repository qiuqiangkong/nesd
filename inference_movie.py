from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile
from torch import Tensor
import cv2
import io
import os
import torch
from einops import rearrange
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio.v2 as imageio
import time
import h5py
import torchaudio
from concurrent.futures import ProcessPoolExecutor

from nesd.utils.utils import parse_yaml
from nesd.utils.torch import sph2cart, normalize
from train import get_model


def inference(args) -> None:
    r"""Separate an audio."""

    # Arguments and parameters
    config_yaml = args.config_yaml
    ckpt_path = args.ckpt_path
    audio_path = args.audio_path
    video_path = args.video_path
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
    audio, _ = librosa.load(path=audio_path, sr=sr, mono=False)  # shape: (c, l)
    audio = audio[:, 0:60*sr] 
    # audio = audio[:, 0:210*sr] 
    # audio = np.zeros((4, 96000))
    audio*=1

    if False:
        audio = torchaudio.functional.lowpass_biquad(
            waveform=torch.Tensor(audio),
            sample_rate=sr,
            cutoff_freq=16000,
        ).data.cpu().numpy()
    
    # audio = Tensor(audio[None, :, :]).to(device)
    
    ele = torch.linspace(0, 180, 181).deg2rad()  # 5 points along x
    azi = torch.linspace(0, 360, 360).deg2rad()  # 3 points along y
    ele, azi = torch.meshgrid(ele, azi, indexing='ij')  # (181, 361)
    r = torch.ones_like(ele)

    sph = torch.stack([r, ele, azi], dim=-1)
    dirn = normalize(sph2cart(sph))
    
    dirn = rearrange(dirn, 'el az d -> (el az) d')

    bgn_sample = 0
    R = 10000
    outputs = []

    while bgn_sample < audio.shape[-1]:
        print(bgn_sample / sr)
        seg = librosa.util.fix_length(audio[:, bgn_sample : bgn_sample + seg_samples], size=seg_samples, axis=-1)
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
                out = rearrange(out, 'b (t1 t2) -> b t1 t2', t2=10)
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

    t1 = time.time()    
    
    if video_path:
        h5_path = Path("_tmp/video_h5/{}.h5".format(Path(video_path).stem))
        
        if not Path(h5_path).is_file():
            from moviepy import VideoFileClip
            clip = VideoFileClip(video_path)
            clip = clip.resized((360, 181)).with_fps(10)
            frames = [frame for frame in clip.iter_frames()]
            frames = np.stack(frames, axis=0)
            Path(h5_path).parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(h5_path, 'w') as hf:
                hf.create_dataset('x', data=frames, dtype=np.uint8)

        with h5py.File(h5_path, 'r') as hf:
            video = hf["x"][:] / 255.  # (T, H, W, 3)
        video = librosa.util.fix_length(data=video, size=outputs.shape[0], axis=0)
        
        cmap = plt.get_cmap('jet')
        outputs = cmap(outputs)[:, :, :, :3]
        
        outputs = 0.7 * video + 0.3 * outputs
        outputs = np.clip(outputs, 0., 1.)
        
    
    params = [(outputs[i], i/10) for i in range(outputs.shape[0])]

    # for i, param in enumerate(params):
    #     sub(param)
    #     print(i)
        # from IPython import embed; embed(using=False); os._exit(0)
    
    with ProcessPoolExecutor(max_workers=32) as pool: # Maximum workers on the machine.
        results = pool.map(sub, params)

    writer = imageio.get_writer('matshow_video.mp4', fps=10, codec='libx264')
    for img in results:
        writer.append_data(img)

    writer.close()
    print("Time: {:02f} s".format(time.time() - t1))
    print(outputs.max())
    

    tmp_audio_path = "__zz.wav"
    soundfile.write(file=tmp_audio_path, data=np.mean(audio, axis=0), samplerate=sr)
    os.system(f"ffmpeg -y -i matshow_video.mp4 -i {tmp_audio_path} -c:v copy -c:a aac -shortest matshow_video2.mp4")

    print(outputs.max())
    from IPython import embed; embed(using=False); os._exit(0)



# def sub(param):
#     frame, time, vid = param
#     alpha = 0.5
#     blend = alpha * frame + (1 - alpha) * vid

#     fig, ax = plt.subplots(figsize=(12, 8))
#     im = ax.matshow(blend, cmap='jet', origin="upper", vmin=0, vmax=1)

#     ax.set_title(f"Time {time:.02f} s")
#     ax.grid(color='w', linestyle='--', linewidth=0.2)
#     ax.xaxis.set_ticks(np.arange(0, 361, 10))
#     ax.yaxis.set_ticks(np.arange(0, 181, 10))
#     ax.xaxis.set_ticklabels(np.arange(0, 361, 10), rotation=90)
#     ax.yaxis.set_ticklabels(np.arange(0, 181, 10))
#     ax.set_xlabel('azimuth')
#     ax.set_ylabel('elevation')
#     ax.xaxis.tick_bottom()

#     # 把matplotlib的图转为numpy图像
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
#     plt.close(fig)
#     buf.seek(0)
#     img = imageio.imread(buf)
#     return img

def sub(param):
    frame, time = param
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.matshow(frame, cmap='jet', origin="upper", vmin=0, vmax=1)

    ax.set_title(f"Time {time:.02f} s")
    ax.grid(color='w', linestyle='--', linewidth=0.2)
    ax.xaxis.set_ticks(np.arange(0, 361, 10))
    ax.yaxis.set_ticks(np.arange(0, 181, 10))
    ax.xaxis.set_ticklabels(np.arange(0, 361, 10), rotation=90)
    ax.yaxis.set_ticklabels(np.arange(0, 181, 10))
    ax.set_xlabel('azimuth')
    ax.set_ylabel('elevation')
    ax.xaxis.tick_bottom()

    # 把matplotlib的图转为numpy图像
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
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--video_path', type=str, required=False)

    args = parser.parse_args()

    inference(args)