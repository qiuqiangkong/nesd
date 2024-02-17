from typing import Dict
import yaml
import math
import os
import pickle
import datetime
import time
import numpy as np
import logging
import librosa
import random
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def read_yaml(config_yaml: str) -> Dict:
    """Read config file to dictionary.

    Args:
        config_yaml: str

    Returns:
        configs: Dict
    """
    with open(config_yaml, "r") as fr:
        configs = yaml.load(fr, Loader=yaml.FullLoader)

    return configs


def remove_silence(audio, sample_rate, threshold=0.02):

    window_size = int(sample_rate * 0.1)

    frames = librosa.util.frame(x=audio, frame_length=window_size, hop_length=window_size).T
    # shape: (frames_num, window_size)

    new_frames = get_active_frames(frames, threshold)
    # shape: (new_frames_num, window_size)

    new_audio = new_frames.flatten()
    # shape: (new_audio_samples,)

    return new_audio


def get_active_frames(frames, threshold):
    
    energy = np.max(np.abs(frames), axis=-1)
    # shape: (frames_num,)

    active_indexes = np.where(energy > threshold)[0]
    # shape: (new_frames_num,)

    new_frames = frames[active_indexes]
    # shape: (new_frames_num,)

    return new_frames


def scale_to_db(scale):
    db = 20 * np.log10(scale)
    return db


def db_to_scale(db):
    scale = 10 ** (db / 20.)
    return scale


def sph2cart(azimuth, elevation, r):
    x = r * np.cos(azimuth) * np.cos(elevation)
    y = r * np.sin(azimuth) * np.cos(elevation)
    z = r * np.sin(elevation)
    vector = np.stack([x, y, z], axis=-1)
    return vector

'''
def cart2sph(x, y, z):
    azimuth = np.arctan2(y, x)
    elevation = np.arcsin(z / r)
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return azimuth, elevation, r
'''
def cart2sph(vector):
    x = vector[..., 0]
    y = vector[..., 1]
    z = vector[..., 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    azimuth = np.arctan2(y, x)
    elevation = np.arcsin(z / r)
    
    return azimuth, elevation, r


def normalize(x):
    if x.ndim == 1:
        return x / np.linalg.norm(x)
    elif x.ndim == 2:
        return x / np.linalg.norm(x, axis=-1)[:, None]
    else:
        raise NotImplementedError


def random_direction(
    min_azimuth=-math.pi, 
    max_azimuth=math.pi, 
    min_elevation=-math.pi / 2,
    max_elevation=math.pi / 2,
):
    azimuth = random.uniform(a=min_azimuth, b=max_azimuth)
    elevation = random.uniform(a=min_elevation, b=max_elevation)

    orientation = sph2cart(azimuth=azimuth, elevation=elevation, r=1.)

    return orientation


def fractional_delay_filter(delayed_samples):
    r"""Fractional delay with Whittaker–Shannon interpolation formula. 
    Ref: https://tomroelandts.com/articles/how-to-create-a-fractional-delay-filter

    Args:
        x: np.array (1D), input signal
        delay_samples: float >= 0., e.g., 3.3

    Outputs:
        y: np.array (1D), delayed signal
    """

    delayed_samples_integer = math.floor(delayed_samples)
    delayed_samples_fraction = delayed_samples % 1

    N = 99     # Filter length.
    n = np.arange(N)

    # Compute sinc filter.
    center = (N - 1) // 2
    h = np.sinc(n - center - delayed_samples_fraction)

    # Multiply sinc filter by window.
    h *= np.blackman(N)
    
    # Normalize to get unity gain.
    h /= np.sum(h)

    # Delay filter length.
    M = np.abs(delayed_samples_integer) * 2 + 1

    # Combined filter.
    new_len = M + N - 1
    new_h = np.zeros(new_len)

    bgn = new_len // 2 + delayed_samples_integer - (N - 1) // 2
    end = new_len // 2 + delayed_samples_integer + (N - 1) // 2
    new_h[bgn : end + 1] = h

    return new_h


# def get_incident_angle(a, b):
def get_included_angle(a, b):
    cos = np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    angle = np.arccos(cos)
    return angle


def random_positive_direction(source_direction, theta):

    base_direction = np.array([0, 0, 1])
    
    # Compute rotation vector.
    rot_vector = np.cross(a=base_direction, b=source_direction)
    rot_angle = get_included_angle(a=base_direction, b=source_direction)
    rot_vector = rot_angle * normalize(rot_vector)

    # Initiate rotation object.
    rot_obj = Rotation.from_rotvec(rotvec=rot_vector)

    # Sample a direction near the base direction.
    random_direction_near_base = random_direction(min_azimuth=-math.pi, 
        max_azimuth=math.pi, 
        min_elevation=math.pi / 2 - theta,
        max_elevation=math.pi / 2,
    )

    # Rotate the sampled direction.
    output_direction = rot_obj.apply(random_direction_near_base)
    
    return output_direction


def triangle_function(x, r=0.05, low=0.5):
    return 1 - ((1 - low) / r * np.abs(x))


def random_negative_direction(source_directions, theta):

    while True:

        rand_direction = random_direction()
        satisfied = True

        for src_direction in source_directions:
            if get_included_angle(a=rand_direction, b=src_direction) < theta:
                satisfied = False
                break
            
        if satisfied:
            return rand_direction


def random_positive_distance(source_distance, r):
    rand_dist = random.uniform(a=source_distance - r, b=source_distance + r)
    return rand_dist


def random_negative_distance(source_distances, r, max_dist):

    while True:
        
        rand_dist = random.uniform(a=0, b=max_dist)
        satisfied = True

        for src_dist in source_distances:
            if src_dist - r <= rand_dist <= src_dist + r:
                satisfied = False
                break

        if satisfied:
            return rand_dist