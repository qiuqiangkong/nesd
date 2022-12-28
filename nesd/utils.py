from typing import Dict
import yaml
import math
import os
import pickle
import datetime
import logging
from scipy.signal import fftconvolve

import torch
import numpy as np


def float32_to_int16(x: np.float32) -> np.int16:

    x = np.clip(x, a_min=-1, a_max=1)

    return (x * 32767.0).astype(np.int16)


def int16_to_float32(x: np.int16) -> np.float32:

    return (x / 32767.0).astype(np.float32)


def create_logging(log_dir: str, filemode: str) -> logging:
    r"""Create logging to write out log files.

    Args:
        logs_dir, str, directory to write out logs
        filemode: str, e.g., "w" 

    Returns:
        logging
    """
    os.makedirs(log_dir, exist_ok=True)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, "{:04d}.log".format(i1))):
        i1 += 1

    log_path = os.path.join(log_dir, "{:04d}.log".format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
        filename=log_path,
        filemode=filemode,
    )

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    return logging



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

'''
class DirectionSampler:
    def __init__(self, low_zenith, high_zenith, sample_on_sphere_uniformly, random_state):

        self.random_state = random_state
        # self.sample_on_sphere_uniformly = False # Always set to False.

        self.low_zenith = low_zenith
        self.high_zenith = high_zenith
        self.sample_on_sphere_uniformly = sample_on_sphere_uniformly
        self.random_state = random_state

    def sample(self, size=None):

        azimuth = self.random_state.uniform(low=0, high=2 * math.pi, size=size)

        if self.sample_on_sphere_uniformly:

            v = self.random_state.uniform(low=np.cos(self.high_zenith), high=np.cos(self.low_zenith), size=size)
            
            zenith = np.arccos(v)
            # Ref: https://mathworld.wolfram.com/SpherePointPicking.html
            # Ref: https://zhuanlan.zhihu.com/p/26052376

        else:
            zenith = self.random_state.uniform(low=self.low_zenith, high=self.high_zenith, size=size)

        return azimuth, zenith
'''

class DirectionSampler:
    def __init__(self, low_zenith, high_zenith, sample_on_sphere_uniformly, random_state):

        self.random_state = random_state
        # self.sample_on_sphere_uniformly = False # Always set to False.

        self.low_zenith = low_zenith
        self.high_zenith = high_zenith
        self.sample_on_sphere_uniformly = sample_on_sphere_uniformly
        self.random_state = random_state

    def sample(self, size=None):

        azimuth = self.random_state.uniform(low=0, high=2 * math.pi, size=size)

        if self.sample_on_sphere_uniformly:

            v = self.random_state.uniform(low=np.cos(self.high_zenith), high=np.cos(self.low_zenith), size=size)
            
            zenith = np.arccos(v)
            # Ref: https://mathworld.wolfram.com/SpherePointPicking.html
            # Ref: https://zhuanlan.zhihu.com/p/26052376

        else:
            zenith = self.random_state.uniform(low=self.low_zenith, high=self.high_zenith, size=size)

        return azimuth, zenith


def sph2cart(r, azimuth, zenith):
    x = r * np.cos(azimuth) * np.sin(zenith)
    y = r * np.sin(azimuth) * np.sin(zenith)
    z = r * np.cos(zenith)
    return x, y, z


class Position:
    def __init__(self, position):
        self.position = position

'''
class Microphone:
    def __init__(self, directivity):
        self.directivity = directivity

    def look_at(self, position, target, up):
        # https://www.programcreek.com/python/?CodeExample=look+at
        self.position = position

        z = target - position
        z = z / np.linalg.norm(z)

        self.z = z
'''


class Microphone:
    def __init__(self, position, direction, directivity, directivity_object=None):

        self.position = position
        self.direction = direction / np.linalg.norm(direction)
        self.directivity = directivity
        self.directivity_object = directivity_object

    def set_waveform(self, waveform):
        self.waveform = waveform
        

        # z = target - position
        # z = z / np.linalg.norm(z)

        # self.z = z

    # def look_at(self, position, target, up):
    #     # https://www.programcreek.com/python/?CodeExample=look+at
    #     self.position = position

    #     z = target - position
    #     z = z / np.linalg.norm(z)

    #     self.z = z

'''
class Sphere:
    def __init__(self, origin, radius):
        
        self.origin = origin
        self.radius = radius

class SphereSource:
    def __init__(self, origin, radius):
        self.origin = origin
        self.radius = radius

    def set_waveform(self, waveform):
        self.waveform = waveform
'''

class SphereSource:
    def __init__(self, position, radius):
        self.position = position
        self.radius = radius

    def set_waveform(self, waveform):
        self.waveform = waveform 

def get_cos(a, b):
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos


def calculate_microphone_gain(cos, directivity):

    if directivity == "cardioid":
        gain = (1 + cos) / 2

    # elif self.directivity == "omni":
    #     gain = np.ones_like(angle)

    else:
        raise NotImplementedError

    return gain

'''
def get_ir_filter(filter_len, gain, delayed_samples):

    filt = np.zeros(filter_len)
    filt[int(delayed_samples)] = 1 - (delayed_samples - int(delayed_samples))
    filt[int(delayed_samples) + 1] = delayed_samples - int(delayed_samples)
    filt *= gain

    return filt
'''

def get_ir_filter(filter_len, gain_list, delayed_samples_list):

    filt = np.zeros(filter_len)

    for gain, delayed_samples in zip(gain_list, delayed_samples_list):

        if delayed_samples is not None:
            alpha = delayed_samples - int(delayed_samples)
            filt[int(delayed_samples)] += (1 - alpha) * gain
            filt[int(delayed_samples) + 1] += alpha * gain
        
    return filt


def conv_signals(source, filt):
    len_source = len(source)
    source = np.concatenate((np.zeros(len(filt) - 1), source))
    y = fftconvolve(in1=source, in2=filt, mode='valid')
    y = y[0 : len_source]
    return y


'''
def cart2sph(x, y, z):
    azimuth = np.arctan2(y, x)
    azimuth %= math.pi * 2
    zenith = np.arccos(z)
    return azimuth, zenith
'''

'''
def cart2sph(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    x /= r
    y /= r
    z /= r
    azimuth = np.arctan2(y, x)
    azimuth %= math.pi * 2
    zenith = np.arccos(z)
    return r, azimuth, zenith


def cart2sph_torch(x, y, z):
    r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    x /= r
    y /= r
    z /= r
    azimuth = torch.atan2(y, x)
    azimuth %= math.pi * 2
    zenith = torch.acos(z)
    return r, azimuth, zenith
'''

def cart2sph(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    azimuth = np.arctan2(y, x)
    azimuth %= math.pi * 2
    zenith = np.arccos(z / r)
    return r, azimuth, zenith


def cart2sph_torch(x, y, z):
    r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    azimuth = torch.atan2(y, x)
    azimuth %= math.pi * 2
    zenith = torch.acos(z / r)
    return r, azimuth, zenith

# def sphere_to_cart(azimuth: np.ndarray, elevation: np.ndarray):
#     r = 1.
#     x = r * np.cos(azimuth) * np.sin(elevation)
#     y = r * np.sin(azimuth) * np.sin(elevation)
#     z = r * np.cos(elevation)
#     return x, y, z


# def cart_to_sphere(x, y, z):
#     azimuth = np.arctan2(y, x)
#     azimuth %= math.pi * 2
#     elevation = np.arccos(z)
#     return azimuth, elevation


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction / np.linalg.norm(direction)
        # self.waveform = waveform

    def set_waveform(self, waveform):
        self.waveform = waveform

    def set_intersect_source(self, intersect_source):
        self.intersect_source = intersect_source


class Rotator3D:
    def __init__(self):
        pass

    @staticmethod
    def get_rotation_matrix_from_azimuth_zenith(azimuth, zenith):

        alpha, beta, gamma = Rotator3D.get_alpha_beta_gamma(azimuth, zenith)
        rotation_matrix = Rotator3D.get_rotation_matrix_from_alpha_beta_gamma(alpha, beta, gamma)

        return rotation_matrix

    @staticmethod
    def get_alpha_beta_gamma(azimuth, zenith):
        # Rotate along x (roll, gamma) -> along y (pitch, beta) -> along z (yaw, alpha)
        x, y, z = sph2cart(r=1., azimuth=azimuth, zenith=zenith)

        alpha = 0.
        beta = math.atan2(x, z)
        gamma = - math.asin(y)

        return alpha, beta, gamma

    @staticmethod
    # Ref: https://stackoverflow.com/questions/2782647/how-to-get-yaw-pitch-and-roll-from-a-3d-vector
    def get_rotation_matrix_from_alpha_beta_gamma(alpha, beta, gamma):

        R_z = np.array([
            [math.cos(alpha), - math.sin(alpha), 0],
            [math.sin(alpha), math.cos(alpha), 0],
            [0, 0, 1],
        ])

        R_y = np.array([
            [math.cos(beta), 0, math.sin(beta)],
            [0, 1, 0],
            [- math.sin(beta), 0, math.cos(beta)],
        ])

        R_x = np.array([
            [1, 0, 0],
            [0, math.cos(gamma), - math.sin(gamma)],
            [0, math.sin(gamma), math.cos(gamma)],
        ])

        rotation_matrix = R_z.dot(R_y).dot(R_x)
        
        return rotation_matrix

    @staticmethod
    def rotate_azimuth_zenith(rotation_matrix, azimuth, zenith):
        x, y, z = sph2cart(r=1., azimuth=azimuth, zenith=zenith)
        new_x, new_y, new_z = Rotator3D.rotate_x_y_z(rotation_matrix, x, y, z)
        _, new_azimuth, new_zenith = cart2sph(new_x, new_y, new_z)
        return new_azimuth, new_zenith

    @staticmethod
    def rotate_x_y_z(rotation_matrix, x, y, z):
        input_tensor = np.stack((x, y, z), axis=0)
        output_tensor = rotation_matrix.dot(input_tensor)
        new_x = output_tensor[0]
        new_y = output_tensor[1]
        new_z = output_tensor[2]
        return new_x, new_y, new_z


class StatisticsContainer(object):
    def __init__(self, statistics_path):
        self.statistics_path = statistics_path

        self.backup_statistics_path = "{}_{}.pkl".format(
            os.path.splitext(self.statistics_path)[0],
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )

        self.statistics_dict = {"train": [], "test": []}

    def append(self, steps, statistics, split):
        statistics["steps"] = steps
        self.statistics_dict[split].append(statistics)

    def dump(self):
        pickle.dump(self.statistics_dict, open(self.statistics_path, "wb"))
        pickle.dump(self.statistics_dict, open(self.backup_statistics_path, "wb"))
        logging.info("    Dump statistics to {}".format(self.statistics_path))
        logging.info("    Dump statistics to {}".format(self.backup_statistics_path))
