import librosa
import numpy as np
import pandas as pd
import random
import math
import yaml
import torch
import torch.nn as nn

from nesd.utils.numpy import sph2cart


def remove_silence(audio: np.ndarray, sample_rate: int, threshold=0.02) -> np.ndarray:
    r"""Remove silence from an audio clip."""
    window_size = int(sample_rate * 0.1)

    frames = librosa.util.frame(x=audio, frame_length=window_size, hop_length=window_size).T
    # shape: (frames_num, window_size)

    new_frames = get_active_frames(frames, threshold) # (new_frames_num, window_size)
    new_audio = new_frames.flatten()  # (new_audio_samples,)

    return new_audio


def get_active_frames(frames: np.ndarray, threshold: float) -> np.ndarray:
    r"""Detect and keep active frames."""
    energy = np.max(np.abs(frames), axis=-1)  # (frames_num,)
    active_indexes = np.where(energy > threshold)[0]  # (new_frames_num,)
    new_frames = frames[active_indexes]  # (new_frames_num,)

    return new_frames


def repeat_audio_to_length(audio: np.ndarray, segment_samples: int) -> np.ndarray:
    r"""Repeat audio to length."""
    repeats_num = (segment_samples // audio.shape[-1]) + 1
    audio = np.tile(audio, repeats_num)[0 : segment_samples]

    return audio


def get_eigenmike32(mic_csv: str) -> np.ndarray:
    r"""Get the cart Cartesian coordinates of Eigenmike microphones.

    Args:
        mic_csv: str

    Outputs:
        cart: (mics_num, 3)
    """

    # Parse csv file. θ: elevation, [0, π]. φ: azimuth: [0, 2π)
    df = pd.read_csv(mic_csv, sep=",", index_col="microphone")
    
    indices = [6, 10, 26, 22]
    # 6: (0.042, θ=55° φ=45°), 10: (0.042, θ=125° φ=315°), 
    # 26: (0.042, θ=125° φ=135°), 22: (0.042, θ=55° φ=225°)

    r = df["radius"][indices].values
    theta = np.deg2rad(df["theta"][indices].values)
    phi = np.deg2rad(df["phi"][indices].values)
    
    sph = np.stack([r, theta, phi], axis=-1)
    cart = sph2cart(sph)

    return cart


def log_uniform(a: float, b: float) -> float:
    x = random.uniform(math.log10(a), math.log10(b))
    x = 10 ** x
    return x


def save_list_of_list_to_hdf5(hf, name: str, data: list[list[np.ndarray]], dtype) -> None:
    r"""Save list of list to HDF5."""
    hf.create_group(name)
    for i in range(len(data)):
        hf[name].create_group(str(i))
        for j in range(len(data[i])):
            hf[name][str(i)].create_group(str(j))
            hf[name][str(i)][str(j)].create_dataset(name="x", data=data[i][j], dtype=dtype)


def load_list_of_list_from_hdf5(hf, name: str) -> list:
    r"""Load list of list from HDF5."""
    data = []
    for i in range(len(hf[name])):
        tmp = []
        for j in range(len(hf[name][str(i)])):
            tmp.append(hf[name][str(i)][str(j)]["x"][:])
        data.append(tmp)

    return data


def parse_yaml(config_yaml: str) -> dict:
    r"""Parse yaml file."""
    
    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)


class LinearWarmUp:
    r"""Linear learning rate warm up scheduler.
    """

    def __init__(self, warm_up_steps: int) -> None:
        self.warm_up_steps = warm_up_steps

    def __call__(self, step: int) -> float:
        if step <= self.warm_up_steps:
            return step / self.warm_up_steps
        else:
            return 1.


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float = 0.999) -> None:
    r"""Update EMA model weights and buffers from model."""

    # Parameters
    for e, m in zip(ema.parameters(), model.parameters()):
        e.mul_(decay).add_(m.data.float(), alpha=1 - decay)

    # Buffers (BN running stats, etc)
    for e, m in zip(ema.buffers(), model.buffers()):
        if m.dtype in [torch.bool, torch.long]:
            continue
        e.mul_(decay).add_(m.data.float(), alpha=1 - decay)


def requires_grad(model: nn.Module, flag=True) -> None:
    for p in model.parameters():
        p.requires_grad = flag


def to_device(x: dict, device) -> dict:
    for key in x.keys():
        x[key] = x[key].to(device)
    
    return x


def list_of_list_to_array(size: tuple, x: list, dtype) -> np.ndarray:
    r"""Convert list of list to tensor."""
    y = np.zeros(size, dtype)
    for i in range(len(x)):
        for j in range(len(x[i])):
            y[i, j, 0 : len(x[i][j])] = x[i][j]
    return y


def list_of_list_to_mask(size: tuple, x: list, dtype) -> np.ndarray:
    r"""Convert list of list to mask."""
    y = np.zeros(size, dtype)
    for i in range(len(x)):
        for j in range(len(x[i])):
            y[i, j, 0 : len(x[i][j])] = 1.
    return y