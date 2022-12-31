from typing import Dict, List, NoReturn, Optional, Tuple

import h5py
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy
import math
import os
import time
import yaml
import soundfile
from pytorch_lightning.core.datamodule import LightningDataModule
import pyroomacoustics as pra
from scipy.signal import fftconvolve

from nesd.data.samplers import DistributedSamplerWrapper
from nesd.utils import Microphone, sph2cart, norm, normalize, Source, int16_to_float32, get_cos, calculate_microphone_gain, fractional_delay, DirectionSampler, cart2sph, Rotator3D, Ray, expand_along_time


class DataModule(LightningDataModule):
    def __init__(
        self,
        train_sampler: object,
        train_dataset: object,
        num_workers: int,
        distributed: bool,
    ):
        r"""Data module.

        Args:
            train_sampler: Sampler object
            train_dataset: Dataset object
            num_workers: int
            distributed: bool
        """
        super().__init__()
        self._train_sampler = train_sampler
        self.train_dataset = train_dataset
        self.num_workers = num_workers
        self.distributed = distributed

    def setup(self, stage: Optional[str] = None) -> NoReturn:
        r"""called on every device."""

        # SegmentSampler is used for sampling segment indexes for training.
        # On multiple devices, each SegmentSampler samples a part of mini-batch
        # data. 

        if self.distributed:
            self.train_sampler = DistributedSamplerWrapper(self._train_sampler)

        else:
            self.train_sampler = self._train_sampler

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        r"""Get train loader."""
        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return train_loader


def get_ambisonic_microphone(mic_meta):

    x, y, z = sph2cart(
        r=mic_meta['radius'], 
        azimuth=mic_meta['azimuth'], 
        zenith=mic_meta['zenith']
    )

    position = np.array([x, y, z])
    direction = np.array([x, y, z])

    # target = np.array([2 * x, 2 * y, 2 * z])
    # up = np.array([0, 1, 0])

    # mic = Microphone(directivity=mic_meta['directivity'])

    # mic = Microphone(position=position, direction=direction, directivity=mic_meta['directivity'])

    # mic.look_at(position=mic_position, target=target, up=up)

    return mic


class Dataset:
    def __init__(
        self,
        hdf5s_dir,
    ):
        r"""Used for getting data according to a meta.

        Args:
            input_source_types: list of str, e.g., ['vocals', 'accompaniment']
            target_source_types: list of str, e.g., ['vocals']
            input_channels: int
            augmentor: Augmentor
            segment_samples: int
        """
        self.speed_of_sound = 343.
        self.sample_rate = 24000
        self.rays_num = 100
        self.segment_samples = self.sample_rate * 3
        self.frames_num = 301
        self.max_rays_contain_waveform = 2
        mic_yaml = "ambisonic.yaml"

        with open(mic_yaml, 'r') as f:
            self.mics_meta = yaml.load(f, Loader=yaml.FullLoader)

        # self.hdf5s_dir = "/home/tiger/workspaces/nesd2/hdf5s/vctk/sr=24000/train"
        self.hdf5s_dir = hdf5s_dir
        self.hdf5_names = sorted(os.listdir(self.hdf5s_dir))

    def __getitem__(self, meta: Dict) -> Dict:
        r"""Return data according to a meta. E.g., an input meta looks like: {
            'vocals': [['song_A.h5', 6332760, 6465060], ['song_B.h5', 198450, 330750]],
            'accompaniment': [['song_C.h5', 24232920, 24365250], ['song_D.h5', 1569960, 1702260]]}.
        }

        Then, vocals segments of song_A and song_B will be mixed (mix-audio augmentation).
        Accompaniment segments of song_C and song_B will be mixed (mix-audio augmentation).
        Finally, mixture is created by summing vocals and accompaniment.

        Args:
            meta: dict, e.g., {
                'vocals': [['song_A.h5', 6332760, 6465060], ['song_B.h5', 198450, 330750]],
                'accompaniment': [['song_C.h5', 24232920, 24365250], ['song_D.h5', 1569960, 1702260]]}
            }

        Returns:
            data_dict: dict, e.g., {
                'vocals': (channels, segments_num),
                'accompaniment': (channels, segments_num),
                'mixture': (channels, segments_num),
            }
        """

        # Init mic pos, source pos, target pos

        # Calculate mic signal 
        # - Normal: calculate all rays and convolve with mic IR
        # - Fast: free-field use sparsity

        # Calculate target signal, hard sample
        # - Normal: calculate all signals at all directions
        # - Fast: Hard example directions

        random_seed = meta['random_seed']
        random_state = np.random.RandomState(random_seed)

        # -------- Mic
        new_position = np.array([4, 4, 2])
        new_position = expand_along_time(new_position, self.frames_num)

        mic_center_position = np.array([4, 4, 2])
        mic_center_position = expand_along_time(mic_center_position, self.frames_num)
        
        mics = []

        for mic_meta in self.mics_meta:

            relative_mic_posision = np.array(sph2cart(
                r=mic_meta['radius'], 
                azimuth=mic_meta['azimuth'], 
                colatitude=mic_meta['colatitude']
            ))

            mic_position = mic_center_position + relative_mic_posision[None, :]

            mic_look_direction = normalize(relative_mic_posision)
            mic_look_direction = expand_along_time(mic_look_direction, self.frames_num)

            mic = Microphone(
                position=mic_position,
                look_direction=mic_look_direction,
                directivity=mic_meta['directivity'],
            )
            mics.append(mic)

        # --------- Source
        sources_num = 2
        sources = []

        for i in range(sources_num):
        
            source_position = np.array((
                random_state.uniform(low=0, high=8),
                random_state.uniform(low=0, high=8),
                random_state.uniform(low=0, high=4),
            ))
            source_position = expand_along_time(source_position, self.frames_num)

            h5s_num = len(self.hdf5_names)
            h5_index = random_state.randint(h5s_num)
            hdf5_path = os.path.join(self.hdf5s_dir, self.hdf5_names[h5_index])

            with h5py.File(hdf5_path, 'r') as hf:
                waveform = int16_to_float32(hf['waveform'][:])

            source = Source(
                position=source_position,
                radius=0.1,
                waveform=waveform,
            )
            sources.append(source)

        # --------- Fast simulate mic

        # Microphone signals
        for mic in mics:

            # total = 0
            for source in sources:
                src_to_mic = mic.position - source.position
                delayed_seconds = norm(src_to_mic[0, :]) / self.speed_of_sound
                delayed_samples = self.sample_rate * delayed_seconds

                cos = get_cos(mic.look_direction[0, :], -src_to_mic[0, :])
                gain = calculate_microphone_gain(cos=cos, directivity=mic.directivity)

                y = fractional_delay(x=source.waveform, delayed_samples=delayed_samples)
                y *= gain

                mic.waveform += y

            # soundfile.write(file='_zz.wav', data=mic.waveform, samplerate=24000)
            # from IPython import embed; embed(using=False); os._exit(0)

        # --------- Hard example new position rays

        rays = []

        half_angle = math.atan2(0.1, 1)

        for source in sources:

            newpos_to_src = source.position - new_position

            ray_direction = sample_ray_direction(
                newpos_to_src=newpos_to_src[0, :],
                half_angle=half_angle,
                random_state=random_state,
            )
            ray_direction = expand_along_time(ray_direction, self.frames_num)

            delayed_seconds = norm(newpos_to_src) / self.speed_of_sound
            delayed_samples = self.sample_rate * delayed_seconds

            y = fractional_delay(x=source.waveform, delayed_samples=delayed_samples)
            gain = 1.
            y *= gain

            ray = Ray(
                origin=new_position, 
                direction=ray_direction, 
                waveform=y,
                intersect_source=np.ones(self.frames_num),
            )
            rays.append(ray)

        while len(rays) < self.rays_num:

            _direction_sampler = DirectionSampler(
                low_colatitude=0, 
                high_colatitude=math.pi, 
                sample_on_sphere_uniformly=False, 
                random_state=random_state,
            )

            azimuth, colatitude = _direction_sampler.sample()
            ray_direction = np.array(sph2cart(r=1., azimuth=azimuth, colatitude=colatitude))
            ray_direction = expand_along_time(ray_direction, self.frames_num)

            satisfied = True

            for source in sources:

                newpos_to_src = source.position - new_position
                angle_between_ray_and_src = np.arccos(get_cos(ray_direction[0, :], newpos_to_src[0, :]))

                if angle_between_ray_and_src < half_angle:
                    satisfied = False

            if satisfied:
                ray = Ray(
                    origin=new_position, 
                    direction=ray_direction, 
                    waveform=np.zeros(self.segment_samples),
                    # waveform=None,
                    intersect_source=np.zeros(self.frames_num),
                )
                rays.append(ray)

        data_dict = {
            'source_waveform': np.array([source.waveform for source in sources]),
            'source_position': np.array([source.position for source in sources]),
            'mic_position': np.array([mic.position for mic in mics]),
            'mic_look_direction': np.array([mic.look_direction for mic in mics]),
            'mic_waveform': np.array([mic.waveform for mic in mics]),
            'ray_origin': np.array([ray.origin for ray in rays]),
            'ray_direction': np.array([ray.direction for ray in rays]),
            'ray_intersect_source': np.array([ray.intersect_source for ray in rays]),
            'ray_waveform': np.array([ray.waveform for ray in rays[0 : self.max_rays_contain_waveform]]),
        }
        
        # Plot
        if False:
            azis, cols = [], []
            for ray in rays:
                _, azi, col = cart2sph(
                    x=ray.direction[0],
                    y=ray.direction[1],
                    z=ray.direction[2],
                )
                azis.append(azi)
                cols.append(col)
            plt.scatter(azis, cols, s=4, c='b')

            azis, cols = [], []
            for source in sources:
                newpos_to_src = source.position - new_position
                _, azi, col = cart2sph(
                    x=newpos_to_src[0],
                    y=newpos_to_src[1],
                    z=newpos_to_src[2],
                )
                azis.append(azi)
                cols.append(col)
            plt.scatter(azis, cols, s=20, c='r', marker='+')
                
            plt.xlim(0, 2 * math.pi)
            plt.ylim(0, math.pi)
            plt.savefig('_zz.pdf')
            from IPython import embed; embed(using=False); os._exit(0)

        return data_dict


def sample_ray_direction(newpos_to_src, half_angle, random_state):

    _, newpos_to_src_azimuth, newpos_to_src_colatitude = cart2sph(
        x=newpos_to_src[0], 
        y=newpos_to_src[1], 
        z=newpos_to_src[2],
    )

    rotation_matrix = Rotator3D.get_rotation_matrix_from_azimuth_colatitude(
        azimuth=newpos_to_src_azimuth,
        colatitude=newpos_to_src_colatitude,
    )

    _direction_sampler = DirectionSampler(
        low_colatitude=0, 
        high_colatitude=half_angle, 
        sample_on_sphere_uniformly=False, 
        random_state=random_state,
    )
    _azimuth, _colatitude = _direction_sampler.sample()

    ray_azimuth, ray_colatitude = Rotator3D.rotate_azimuth_colatitude(
        rotation_matrix=rotation_matrix,
        azimuth=_azimuth,
        colatitude=_colatitude,
    )

    ray_direction = np.array(sph2cart(r=1., azimuth=ray_azimuth, colatitude=ray_colatitude))

    return ray_direction

def collate_fn(list_data_dict: List[Dict]) -> Dict:
    r"""Collate mini-batch data to inputs and targets for training.

    Args:
        list_data_dict: e.g., [
            {'vocals': (input_channels, segment_samples),
             'accompaniment': (input_channels, segment_samples),
             'mixture': (input_channels, segment_samples)
            },
            {'vocals': (input_channels, segment_samples),
             'accompaniment': (input_channels, segment_samples),
             'mixture': (input_channels, segment_samples)
            },
            ...]

    Returns:
        data_dict: e.g. {
            'vocals': (batch_size, input_channels, segment_samples),
            'accompaniment': (batch_size, input_channels, segment_samples),
            'mixture': (batch_size, input_channels, segment_samples)
            }
    """
    data_dict = {}
    
    for key in list_data_dict[0].keys():
        data_dict[key] = torch.Tensor(
            np.array([_data_dict[key] for _data_dict in list_data_dict])
        )

    return data_dict