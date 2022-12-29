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
from nesd.utils import Microphone, sph2cart, norm, normalize, Source, int16_to_float32, get_cos, calculate_microphone_gain, fractional_delay
# from nesd.utils import int16_to_float32, Microphone, sph2cart, cart2sph, SphereSource, calculate_microphone_gain, get_ir_filter, conv_signals, DirectionSampler, Ray, Rotator3D, get_cos

# GOLDEN_RATIO = (math.sqrt(5) - 1) / 2


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
        # workspace,
        # source_configs: Dict,
        # room_configs: Dict,
        # microphone_configs: Dict,
    ):
        r"""Used for getting data according to a meta.

        Args:
            input_source_types: list of str, e.g., ['vocals', 'accompaniment']
            target_source_types: list of str, e.g., ['vocals']
            input_channels: int
            augmentor: Augmentor
            segment_samples: int
        """
        '''
        mic_yaml = "ambisonic.yaml"
        self.speed_of_sound = 343.
        self.sample_rate = 16000
        self.filter_len = 16000
        self.segment_samples = 48000
        self.rays_num = 100

        self.hdf5s_dir = "/home/tiger/workspaces/nesd/hdf5s/vctk/sr=16000/train"
        # self.hdf5s_dir = "/home/tiger/workspaces/nesd/hdf5s/vctk/sr=16000/test"
        self.hdf5_names = sorted(os.listdir(self.hdf5s_dir))

        with open(mic_yaml, 'r') as f:
            mics_meta = yaml.load(f, Loader=yaml.FullLoader)

        self.mics = []

        for mic_meta in mics_meta:

            mic = get_ambisonic_microphone(mic_meta)
            self.mics.append(mic)
        '''
        self.speed_of_sound = 343.
        self.sample_rate = 24000
        mic_yaml = "ambisonic.yaml"

        with open(mic_yaml, 'r') as f:
            self.mics_meta = yaml.load(f, Loader=yaml.FullLoader)

        self.hdf5s_dir = "/home/tiger/workspaces/nesd2/hdf5s/vctk/sr=24000/train"
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

        mic_center_position = np.array([4, 4, 2])
        
        mics = []

        for mic_meta in self.mics_meta:

            x = np.array(sph2cart(
                r=mic_meta['radius'], 
                azimuth=mic_meta['azimuth'], 
                colatitude=mic_meta['colatitude']
            ))

            mic_position = mic_center_position + x
            mic_look_direction = normalize(x)

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

        # --------- 

        # Microphone signals
        for mic in mics:

            # total = 0
            for source in sources:

                src_to_mic = mic.position - source.position
                delayed_seconds = norm(src_to_mic) / self.speed_of_sound
                delayed_samples = self.sample_rate * delayed_seconds
                
                cos = get_cos(mic.look_direction, -src_to_mic)
                gain = calculate_microphone_gain(cos=cos, directivity=mic.directivity)

                t1 = time.time()
                y = fractional_delay(x=source.waveform, delayed_samples=delayed_samples)
                y *= gain
                print(time.time() - t1)

                mic.waveform += y

            # mic.set_waveform(waveform=total)

            soundfile.write(file='_zz.wav', data=mic.waveform, samplerate=24000)
            from IPython import embed; embed(using=False); os._exit(0)
    
        random_seed = meta['random_seed']
        random_state = np.random.RandomState(random_seed)

        direction_sampler = DirectionSampler(
            low_zenith=0, 
            high_zenith=math.pi, 
            sample_on_sphere_uniformly=False, 
            random_state=random_state,
        )

        

        # field
        ray_origin = np.array([0, 0, 0])
        rays = []

        for sphere_source in sphere_sources:

            ray_origin_to_src_origin = sphere_source.position - ray_origin
            distance = np.linalg.norm(ray_origin_to_src_origin)
            angle = np.arctan2(sphere_source.radius, distance)

            _, src_azimuth, src_zenith = cart2sph(
                x=sphere_source.position[0], 
                y=sphere_source.position[1], 
                z=sphere_source.position[2],
            )

            rotation_matrix = Rotator3D.get_rotation_matrix_from_azimuth_zenith(
                azimuth=src_azimuth,
                zenith=src_zenith,
            )

            _direction_sampler = DirectionSampler(
                low_zenith=0, 
                high_zenith=angle, 
                sample_on_sphere_uniformly=False, 
                random_state=random_state,
            )
            _azimuth, _zenith = _direction_sampler.sample()

            ray_azimuth, ray_zenith = Rotator3D.rotate_azimuth_zenith(
                rotation_matrix=rotation_matrix,
                azimuth=_azimuth,
                zenith=_zenith,
            )

            ray_direction = sph2cart(r=1., azimuth=ray_azimuth, zenith=ray_zenith)
            delayed_seconds = distance / self.speed_of_sound
            delayed_samples = self.sample_rate * delayed_seconds

            gain = 1
            filt = get_ir_filter(
                filter_len=self.filter_len, 
                gain_list=[gain], 
                delayed_samples_list=[delayed_samples],
            )
            y = conv_signals(source=sphere_source.waveform, filt=filt)

            ray = Ray(origin=ray_origin, direction=ray_direction)
            ray.set_waveform(waveform=y)
            ray.set_intersect_source(intersect_source=1.)
            rays.append(ray)

        while len(rays) < self.rays_num:

            azimuth, zenith = direction_sampler.sample()
            ray_direction = sph2cart(r=1., azimuth=azimuth, zenith=zenith)

            satisfied = True

            for sphere_source in sphere_sources:

                ray_origin_to_src_origin = sphere_source.position - ray_origin
                distance = np.linalg.norm(ray_origin_to_src_origin)
                max_angle = np.arctan2(sphere_source.radius, distance)

                ray_angle = np.arccos(get_cos(ray_origin_to_src_origin, ray_direction))

                if ray_angle < max_angle:
                    satisfied = False

            if satisfied:
                waveform = np.zeros(self.segment_samples)
                ray = Ray(origin=ray_origin, direction=ray_direction)
                ray.set_waveform(waveform=waveform)
                ray.set_intersect_source(intersect_source=0.)
                rays.append(ray)
        
        data_dict = {
            'source_waveform': np.array([sphere_source.waveform for sphere_source in sphere_sources]),
            'source_position': np.array([sphere_source.position for sphere_source in sphere_sources]),
            'source_radius': np.array([sphere_source.radius for sphere_source in sphere_sources]),
            'mic_waveform': np.array([mic.waveform for mic in self.mics]),
            'mic_position': np.array([mic.position for mic in self.mics]),
            'mic_direction': np.array([mic.direction for mic in self.mics]),
            'ray_waveform': np.array([ray.waveform for ray in rays]),
            'ray_origin': np.array([ray.origin for ray in rays]),
            'ray_direction': np.array([ray.direction for ray in rays]),
            'ray_intersect_source': np.array([ray.intersect_source for ray in rays]),
        }

        if False:
            fig, ax = plt.subplots(1, 1, sharex=True)
            ax.scatter(data_dict['mic_origin'][:, 0], data_dict['mic_origin'][:, 1], c='r')
            ax.scatter(data_dict['source_origin'][:, 0], data_dict['source_origin'][:, 1], c='k')
            # for i, sphere_source in enumerate(sphere_sources):
            #     plt.Circle((data_dict['source_origin'][i, 0], data_dict['source_origin'][i, 1]), sphere_source.radius, color='k')
            ax.scatter(ray_origin[0], ray_origin[0], c='b')
            for ray in rays[0 : sources_num]:
                ax.quiver(ray.origin[0], ray.origin[1], ray.direction[0], ray.direction[1], color='k', scale=1)
            for ray in rays[sources_num :]:
                ax.quiver(ray.origin[0], ray.origin[1], ray.direction[0], ray.direction[1], color='pink')
            ax.axis('square')
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            plt.savefig('_zz.pdf')

        return data_dict


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
    
    '''
    for key in list_data_dict[0].keys():
        print(key)
        if key in ['source_waveform', 'source_azimuth_array', 'source_elevation_array', 'source_audio_name', 'source_class_id']:
            data_dict[key] = [data_dict[key] for data_dict in list_data_dict]
        else:
            data_dict[key] = torch.Tensor(
                np.array([data_dict[key] for data_dict in list_data_dict])
            )
    '''
    for key in list_data_dict[0].keys():

        if key in ['source_waveform', 'source_position', 'source_direction', 'source_radius', 'target_position', 'source_class']:
            data_dict[key] = [_data_dict[key] for _data_dict in list_data_dict]
        else:
            try:
                data_dict[key] = torch.Tensor(
                    np.array([_data_dict[key] for _data_dict in list_data_dict])
                )
            except:
                from IPython import embed; embed(using=False); os._exit(0)
    # from IPython import embed; embed(using=False); os._exit(0)
    return data_dict