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
from nesd.utils import Microphone, sph2cart, norm, normalize, Source, int16_to_float32, get_cos, calculate_microphone_gain, fractional_delay, DirectionSampler, cart2sph, Rotator3D, Agent, expand_along_time


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
        self.agents_num = 100
        self.segment_samples = self.sample_rate * 3
        self.frames_num = 301
        self.max_agents_contain_waveform = 2
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
        # - Normal: calculate all agents and convolve with mic IR
        # - Fast: free-field use sparsity

        # Calculate target signal, hard sample
        # - Normal: calculate all signals at all directions
        # - Fast: Hard example directions

        random_seed = meta['random_seed']
        random_state = np.random.RandomState(random_seed)

        # -------- Mic
        agent_position = np.array([4, 4, 2])
        agent_position = expand_along_time(agent_position, self.frames_num)

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
                mic_to_src = source.position - mic.position
                delayed_seconds = norm(mic_to_src[0, :]) / self.speed_of_sound
                delayed_samples = self.sample_rate * delayed_seconds

                cos = get_cos(mic.look_direction[0, :], mic_to_src[0, :])
                gain = calculate_microphone_gain(cos=cos, directivity=mic.directivity)

                y = fractional_delay(x=source.waveform, delayed_samples=delayed_samples)
                y *= gain

                mic.waveform += y

        # --------- Hard example new position agents

        agents = []

        half_angle = math.atan2(0.1, 1)

        for source in sources:

            agent_to_src = source.position - agent_position

            agent_look_direction = sample_agent_look_direction(
                agent_to_src=agent_to_src[0, :],
                half_angle=half_angle,
                random_state=random_state,
            )
            agent_look_direction = expand_along_time(agent_look_direction, self.frames_num)

            delayed_seconds = norm(agent_to_src[0, :]) / self.speed_of_sound
            delayed_samples = self.sample_rate * delayed_seconds

            y = fractional_delay(x=source.waveform, delayed_samples=delayed_samples)
            gain = 1.
            y *= gain

            agent = Agent(
                position=agent_position, 
                look_direction=agent_look_direction, 
                waveform=y,
                see_source=np.ones(self.frames_num),
            )

            agents.append(agent)

        while len(agents) < self.agents_num:

            _direction_sampler = DirectionSampler(
                low_colatitude=0, 
                high_colatitude=math.pi, 
                sample_on_sphere_uniformly=False, 
                random_state=random_state,
            )

            agent_look_azimuth, agent_look_colatitude = _direction_sampler.sample()
            agent_look_direction = np.array(sph2cart(
                r=1., 
                azimuth=agent_look_azimuth, 
                colatitude=agent_look_colatitude
            ))
            agent_look_direction = expand_along_time(agent_look_direction, self.frames_num)

            satisfied = True

            for source in sources:

                agent_to_src = source.position - agent_position
                angle_between_agent_and_src = np.arccos(get_cos(agent_look_direction[0, :], agent_to_src[0, :]))

                if angle_between_agent_and_src < half_angle:
                    satisfied = False

            if satisfied:
                agent = Agent(
                    position=agent_position, 
                    look_direction=agent_look_direction, 
                    waveform=np.zeros(self.segment_samples),
                    see_source=np.zeros(self.frames_num),
                )
                agents.append(agent)

        data_dict = {
            'source_position': np.array([source.position for source in sources]),
            'source_waveform': np.array([source.waveform for source in sources]),
            'mic_position': np.array([mic.position for mic in mics]),
            'mic_look_direction': np.array([mic.look_direction for mic in mics]),
            'mic_waveform': np.array([mic.waveform for mic in mics]),
            'agent_position': np.array([agent.position for agent in agents]),
            'agent_look_direction': np.array([agent.look_direction for agent in agents]),
            'agent_waveform': np.array([agent.waveform for agent in agents[0 : self.max_agents_contain_waveform]]),
            'agent_see_source': np.array([agent.see_source for agent in agents]),
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


def sample_agent_look_direction(agent_to_src, half_angle, random_state):

    _, agent_to_src_azimuth, agent_to_src_colatitude = cart2sph(
        x=agent_to_src[0], 
        y=agent_to_src[1], 
        z=agent_to_src[2],
    )

    rotation_matrix = Rotator3D.get_rotation_matrix_from_azimuth_colatitude(
        azimuth=agent_to_src_azimuth,
        colatitude=agent_to_src_colatitude,
    )

    _direction_sampler = DirectionSampler(
        low_colatitude=0, 
        high_colatitude=half_angle, 
        sample_on_sphere_uniformly=False, 
        random_state=random_state,
    )
    _azimuth, _colatitude = _direction_sampler.sample()

    agent_look_azimuth, agent_look_colatitude = Rotator3D.rotate_azimuth_colatitude(
        rotation_matrix=rotation_matrix,
        azimuth=_azimuth,
        colatitude=_colatitude,
    )

    agent_look_direction = np.array(sph2cart(
        r=1., 
        azimuth=agent_look_azimuth, 
        colatitude=agent_look_colatitude
    ))

    return agent_look_direction


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
        if key in ['source_position', 'source_waveform']:
            data_dict[key] = [_data_dict[key] for _data_dict in list_data_dict]
        else:
            data_dict[key] = torch.Tensor(
                np.array([_data_dict[key] for _data_dict in list_data_dict])
            )

    return data_dict


class Dataset_agent552:
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
        self.agents_num = 100
        self.segment_samples = self.sample_rate * 3
        self.frames_num = 301
        self.max_agents_contain_waveform = 2
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
        # - Normal: calculate all agents and convolve with mic IR
        # - Fast: free-field use sparsity

        # Calculate target signal, hard sample
        # - Normal: calculate all signals at all directions
        # - Fast: Hard example directions

        random_seed = meta['random_seed']
        random_state = np.random.RandomState(random_seed)

        # -------- Mic
        agent_position = np.array([5, 5, 2])
        agent_position = expand_along_time(agent_position, self.frames_num)

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
                mic_to_src = source.position - mic.position
                delayed_seconds = norm(mic_to_src[0, :]) / self.speed_of_sound
                delayed_samples = self.sample_rate * delayed_seconds

                cos = get_cos(mic.look_direction[0, :], mic_to_src[0, :])
                gain = calculate_microphone_gain(cos=cos, directivity=mic.directivity)

                y = fractional_delay(x=source.waveform, delayed_samples=delayed_samples)
                y *= gain

                mic.waveform += y

        # --------- Hard example new position agents

        agents = []

        half_angle = math.atan2(0.1, 1)

        for source in sources:

            agent_to_src = source.position - agent_position

            agent_look_direction = sample_agent_look_direction(
                agent_to_src=agent_to_src[0, :],
                half_angle=half_angle,
                random_state=random_state,
            )
            agent_look_direction = expand_along_time(agent_look_direction, self.frames_num)

            delayed_seconds = norm(agent_to_src[0, :]) / self.speed_of_sound
            delayed_samples = self.sample_rate * delayed_seconds

            y = fractional_delay(x=source.waveform, delayed_samples=delayed_samples)
            gain = 1.
            y *= gain

            agent = Agent(
                position=agent_position, 
                look_direction=agent_look_direction, 
                waveform=y,
                see_source=np.ones(self.frames_num),
            )

            agents.append(agent)

        while len(agents) < self.agents_num:

            _direction_sampler = DirectionSampler(
                low_colatitude=0, 
                high_colatitude=math.pi, 
                sample_on_sphere_uniformly=False, 
                random_state=random_state,
            )

            agent_look_azimuth, agent_look_colatitude = _direction_sampler.sample()
            agent_look_direction = np.array(sph2cart(
                r=1., 
                azimuth=agent_look_azimuth, 
                colatitude=agent_look_colatitude
            ))
            agent_look_direction = expand_along_time(agent_look_direction, self.frames_num)

            satisfied = True

            for source in sources:

                agent_to_src = source.position - agent_position
                angle_between_agent_and_src = np.arccos(get_cos(agent_look_direction[0, :], agent_to_src[0, :]))

                if angle_between_agent_and_src < half_angle:
                    satisfied = False

            if satisfied:
                agent = Agent(
                    position=agent_position, 
                    look_direction=agent_look_direction, 
                    waveform=np.zeros(self.segment_samples),
                    see_source=np.zeros(self.frames_num),
                )
                agents.append(agent)

        data_dict = {
            'source_position': np.array([source.position for source in sources]),
            'source_waveform': np.array([source.waveform for source in sources]),
            'mic_position': np.array([mic.position for mic in mics]),
            'mic_look_direction': np.array([mic.look_direction for mic in mics]),
            'mic_waveform': np.array([mic.waveform for mic in mics]),
            'agent_position': np.array([agent.position for agent in agents]),
            'agent_look_direction': np.array([agent.look_direction for agent in agents]),
            'agent_waveform': np.array([agent.waveform for agent in agents[0 : self.max_agents_contain_waveform]]),
            'agent_see_source': np.array([agent.see_source for agent in agents]),
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


class Dataset_agent_random:
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
        self.agents_num = 100
        self.segment_samples = self.sample_rate * 3
        self.frames_num = 301
        self.max_agents_contain_waveform = 2
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
        # - Normal: calculate all agents and convolve with mic IR
        # - Fast: free-field use sparsity

        # Calculate target signal, hard sample
        # - Normal: calculate all signals at all directions
        # - Fast: Hard example directions

        random_seed = meta['random_seed']
        random_state = np.random.RandomState(random_seed)

        # -------- Mic
        agent_position = np.array((
            random_state.uniform(low=2, high=6),
            random_state.uniform(low=2, high=6),
            random_state.uniform(low=1, high=3),
        ))
        agent_position = expand_along_time(agent_position, self.frames_num)
        
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
                mic_to_src = source.position - mic.position
                delayed_seconds = norm(mic_to_src[0, :]) / self.speed_of_sound
                delayed_samples = self.sample_rate * delayed_seconds

                cos = get_cos(mic.look_direction[0, :], mic_to_src[0, :])
                gain = calculate_microphone_gain(cos=cos, directivity=mic.directivity)

                y = fractional_delay(x=source.waveform, delayed_samples=delayed_samples)
                y *= gain

                mic.waveform += y

        # --------- Hard example new position agents

        agents = []

        half_angle = math.atan2(0.1, 1)

        for source in sources:

            agent_to_src = source.position - agent_position

            agent_look_direction = sample_agent_look_direction(
                agent_to_src=agent_to_src[0, :],
                half_angle=half_angle,
                random_state=random_state,
            )
            agent_look_direction = expand_along_time(agent_look_direction, self.frames_num)

            delayed_seconds = norm(agent_to_src[0, :]) / self.speed_of_sound
            delayed_samples = self.sample_rate * delayed_seconds

            y = fractional_delay(x=source.waveform, delayed_samples=delayed_samples)
            gain = 1.
            y *= gain

            agent = Agent(
                position=agent_position, 
                look_direction=agent_look_direction, 
                waveform=y,
                see_source=np.ones(self.frames_num),
            )

            agents.append(agent)

        while len(agents) < self.agents_num:

            _direction_sampler = DirectionSampler(
                low_colatitude=0, 
                high_colatitude=math.pi, 
                sample_on_sphere_uniformly=False, 
                random_state=random_state,
            )

            agent_look_azimuth, agent_look_colatitude = _direction_sampler.sample()
            agent_look_direction = np.array(sph2cart(
                r=1., 
                azimuth=agent_look_azimuth, 
                colatitude=agent_look_colatitude
            ))
            agent_look_direction = expand_along_time(agent_look_direction, self.frames_num)

            satisfied = True

            for source in sources:

                agent_to_src = source.position - agent_position
                angle_between_agent_and_src = np.arccos(get_cos(agent_look_direction[0, :], agent_to_src[0, :]))

                if angle_between_agent_and_src < half_angle:
                    satisfied = False

            if satisfied:
                agent = Agent(
                    position=agent_position, 
                    look_direction=agent_look_direction, 
                    waveform=np.zeros(self.segment_samples),
                    see_source=np.zeros(self.frames_num),
                )
                agents.append(agent)

        data_dict = {
            'source_position': np.array([source.position for source in sources]),
            'source_waveform': np.array([source.waveform for source in sources]),
            'mic_position': np.array([mic.position for mic in mics]),
            'mic_look_direction': np.array([mic.look_direction for mic in mics]),
            'mic_waveform': np.array([mic.waveform for mic in mics]),
            'agent_position': np.array([agent.position for agent in agents]),
            'agent_look_direction': np.array([agent.look_direction for agent in agents]),
            'agent_waveform': np.array([agent.waveform for agent in agents[0 : self.max_agents_contain_waveform]]),
            'agent_see_source': np.array([agent.see_source for agent in agents]),
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


class Dataset_src1to2:
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
        self.agents_num = 100
        self.segment_samples = self.sample_rate * 3
        self.frames_num = 301
        self.max_agents_contain_waveform = 2
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
        # - Normal: calculate all agents and convolve with mic IR
        # - Fast: free-field use sparsity

        # Calculate target signal, hard sample
        # - Normal: calculate all signals at all directions
        # - Fast: Hard example directions

        random_seed = meta['random_seed']
        random_state = np.random.RandomState(random_seed)

        # -------- Mic
        agent_position = np.array([4, 4, 2])
        agent_position = expand_along_time(agent_position, self.frames_num)

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
        sources_num = random_state.randint(low=1, high=3)
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
                mic_to_src = source.position - mic.position
                delayed_seconds = norm(mic_to_src[0, :]) / self.speed_of_sound
                delayed_samples = self.sample_rate * delayed_seconds

                cos = get_cos(mic.look_direction[0, :], mic_to_src[0, :])
                gain = calculate_microphone_gain(cos=cos, directivity=mic.directivity)

                y = fractional_delay(x=source.waveform, delayed_samples=delayed_samples)
                y *= gain

                mic.waveform += y

        # --------- Hard example new position agents

        agents = []

        half_angle = math.atan2(0.1, 1)

        for source in sources:

            agent_to_src = source.position - agent_position

            agent_look_direction = sample_agent_look_direction(
                agent_to_src=agent_to_src[0, :],
                half_angle=half_angle,
                random_state=random_state,
            )
            agent_look_direction = expand_along_time(agent_look_direction, self.frames_num)

            delayed_seconds = norm(agent_to_src[0, :]) / self.speed_of_sound
            delayed_samples = self.sample_rate * delayed_seconds

            y = fractional_delay(x=source.waveform, delayed_samples=delayed_samples)
            gain = 1.
            y *= gain

            agent = Agent(
                position=agent_position, 
                look_direction=agent_look_direction, 
                waveform=y,
                see_source=np.ones(self.frames_num),
            )

            agents.append(agent)

        while len(agents) < self.agents_num:

            _direction_sampler = DirectionSampler(
                low_colatitude=0, 
                high_colatitude=math.pi, 
                sample_on_sphere_uniformly=False, 
                random_state=random_state,
            )

            agent_look_azimuth, agent_look_colatitude = _direction_sampler.sample()
            agent_look_direction = np.array(sph2cart(
                r=1., 
                azimuth=agent_look_azimuth, 
                colatitude=agent_look_colatitude
            ))
            agent_look_direction = expand_along_time(agent_look_direction, self.frames_num)

            satisfied = True

            for source in sources:

                agent_to_src = source.position - agent_position
                angle_between_agent_and_src = np.arccos(get_cos(agent_look_direction[0, :], agent_to_src[0, :]))

                if angle_between_agent_and_src < half_angle:
                    satisfied = False

            if satisfied:
                agent = Agent(
                    position=agent_position, 
                    look_direction=agent_look_direction, 
                    waveform=np.zeros(self.segment_samples),
                    see_source=np.zeros(self.frames_num),
                )
                agents.append(agent)

        data_dict = {
            'source_position': np.array([source.position for source in sources]),
            'source_waveform': np.array([source.waveform for source in sources]),
            'mic_position': np.array([mic.position for mic in mics]),
            'mic_look_direction': np.array([mic.look_direction for mic in mics]),
            'mic_waveform': np.array([mic.waveform for mic in mics]),
            'agent_position': np.array([agent.position for agent in agents]),
            'agent_look_direction': np.array([agent.look_direction for agent in agents]),
            'agent_waveform': np.array([agent.waveform for agent in agents[0 : self.max_agents_contain_waveform]]),
            'agent_see_source': np.array([agent.see_source for agent in agents]),
        }
        # from IPython import embed; embed(using=False); os._exit(0)
        
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


class Dataset_src4:
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
        self.agents_num = 100
        self.segment_samples = self.sample_rate * 3
        self.frames_num = 301
        self.max_agents_contain_waveform = 4
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
        # - Normal: calculate all agents and convolve with mic IR
        # - Fast: free-field use sparsity

        # Calculate target signal, hard sample
        # - Normal: calculate all signals at all directions
        # - Fast: Hard example directions

        random_seed = meta['random_seed']
        random_state = np.random.RandomState(random_seed)

        # -------- Mic
        agent_position = np.array([4, 4, 2])
        agent_position = expand_along_time(agent_position, self.frames_num)

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
        sources_num = 4
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
                mic_to_src = source.position - mic.position
                delayed_seconds = norm(mic_to_src[0, :]) / self.speed_of_sound
                delayed_samples = self.sample_rate * delayed_seconds

                cos = get_cos(mic.look_direction[0, :], mic_to_src[0, :])
                gain = calculate_microphone_gain(cos=cos, directivity=mic.directivity)

                y = fractional_delay(x=source.waveform, delayed_samples=delayed_samples)
                y *= gain

                mic.waveform += y

        # --------- Hard example new position agents

        agents = []

        half_angle = math.atan2(0.1, 1)

        for source in sources:

            agent_to_src = source.position - agent_position

            agent_look_direction = sample_agent_look_direction(
                agent_to_src=agent_to_src[0, :],
                half_angle=half_angle,
                random_state=random_state,
            )
            agent_look_direction = expand_along_time(agent_look_direction, self.frames_num)

            delayed_seconds = norm(agent_to_src[0, :]) / self.speed_of_sound
            delayed_samples = self.sample_rate * delayed_seconds

            y = fractional_delay(x=source.waveform, delayed_samples=delayed_samples)
            gain = 1.
            y *= gain

            agent = Agent(
                position=agent_position, 
                look_direction=agent_look_direction, 
                waveform=y,
                see_source=np.ones(self.frames_num),
            )

            agents.append(agent)

        while len(agents) < self.agents_num:

            _direction_sampler = DirectionSampler(
                low_colatitude=0, 
                high_colatitude=math.pi, 
                sample_on_sphere_uniformly=False, 
                random_state=random_state,
            )

            agent_look_azimuth, agent_look_colatitude = _direction_sampler.sample()
            agent_look_direction = np.array(sph2cart(
                r=1., 
                azimuth=agent_look_azimuth, 
                colatitude=agent_look_colatitude
            ))
            agent_look_direction = expand_along_time(agent_look_direction, self.frames_num)

            satisfied = True

            for source in sources:

                agent_to_src = source.position - agent_position
                angle_between_agent_and_src = np.arccos(get_cos(agent_look_direction[0, :], agent_to_src[0, :]))

                if angle_between_agent_and_src < half_angle:
                    satisfied = False

            if satisfied:
                agent = Agent(
                    position=agent_position, 
                    look_direction=agent_look_direction, 
                    waveform=np.zeros(self.segment_samples),
                    see_source=np.zeros(self.frames_num),
                )
                agents.append(agent)

        data_dict = {
            'source_position': np.array([source.position for source in sources]),
            'source_waveform': np.array([source.waveform for source in sources]),
            'mic_position': np.array([mic.position for mic in mics]),
            'mic_look_direction': np.array([mic.look_direction for mic in mics]),
            'mic_waveform': np.array([mic.waveform for mic in mics]),
            'agent_position': np.array([agent.position for agent in agents]),
            'agent_look_direction': np.array([agent.look_direction for agent in agents]),
            'agent_waveform': np.array([agent.waveform for agent in agents[0 : self.max_agents_contain_waveform]]),
            'agent_see_source': np.array([agent.see_source for agent in agents]),
        }
        # from IPython import embed; embed(using=False); os._exit(0)
        
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


class Dataset_src8:
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
        self.agents_num = 100
        self.segment_samples = self.sample_rate * 3
        self.frames_num = 301
        self.max_agents_contain_waveform = 4
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
        # - Normal: calculate all agents and convolve with mic IR
        # - Fast: free-field use sparsity

        # Calculate target signal, hard sample
        # - Normal: calculate all signals at all directions
        # - Fast: Hard example directions

        random_seed = meta['random_seed']
        random_state = np.random.RandomState(random_seed)

        # -------- Mic
        agent_position = np.array([4, 4, 2])
        agent_position = expand_along_time(agent_position, self.frames_num)

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
        sources_num = 8
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
                mic_to_src = source.position - mic.position
                delayed_seconds = norm(mic_to_src[0, :]) / self.speed_of_sound
                delayed_samples = self.sample_rate * delayed_seconds

                cos = get_cos(mic.look_direction[0, :], mic_to_src[0, :])
                gain = calculate_microphone_gain(cos=cos, directivity=mic.directivity)

                y = fractional_delay(x=source.waveform, delayed_samples=delayed_samples)
                y *= gain

                mic.waveform += y

        # --------- Hard example new position agents

        agents = []

        half_angle = math.atan2(0.1, 1)

        for source in sources:

            agent_to_src = source.position - agent_position

            agent_look_direction = sample_agent_look_direction(
                agent_to_src=agent_to_src[0, :],
                half_angle=half_angle,
                random_state=random_state,
            )
            agent_look_direction = expand_along_time(agent_look_direction, self.frames_num)

            delayed_seconds = norm(agent_to_src[0, :]) / self.speed_of_sound
            delayed_samples = self.sample_rate * delayed_seconds

            y = fractional_delay(x=source.waveform, delayed_samples=delayed_samples)
            gain = 1.
            y *= gain

            agent = Agent(
                position=agent_position, 
                look_direction=agent_look_direction, 
                waveform=y,
                see_source=np.ones(self.frames_num),
            )

            agents.append(agent)

        while len(agents) < self.agents_num:

            _direction_sampler = DirectionSampler(
                low_colatitude=0, 
                high_colatitude=math.pi, 
                sample_on_sphere_uniformly=False, 
                random_state=random_state,
            )

            agent_look_azimuth, agent_look_colatitude = _direction_sampler.sample()
            agent_look_direction = np.array(sph2cart(
                r=1., 
                azimuth=agent_look_azimuth, 
                colatitude=agent_look_colatitude
            ))
            agent_look_direction = expand_along_time(agent_look_direction, self.frames_num)

            satisfied = True

            for source in sources:

                agent_to_src = source.position - agent_position
                angle_between_agent_and_src = np.arccos(get_cos(agent_look_direction[0, :], agent_to_src[0, :]))

                if angle_between_agent_and_src < half_angle:
                    satisfied = False

            if satisfied:
                agent = Agent(
                    position=agent_position, 
                    look_direction=agent_look_direction, 
                    waveform=np.zeros(self.segment_samples),
                    see_source=np.zeros(self.frames_num),
                )
                agents.append(agent)

        data_dict = {
            'source_position': np.array([source.position for source in sources]),
            'source_waveform': np.array([source.waveform for source in sources]),
            'mic_position': np.array([mic.position for mic in mics]),
            'mic_look_direction': np.array([mic.look_direction for mic in mics]),
            'mic_waveform': np.array([mic.waveform for mic in mics]),
            'agent_position': np.array([agent.position for agent in agents]),
            'agent_look_direction': np.array([agent.look_direction for agent in agents]),
            'agent_waveform': np.array([agent.waveform for agent in agents[0 : self.max_agents_contain_waveform]]),
            'agent_see_source': np.array([agent.see_source for agent in agents]),
        }
        # from IPython import embed; embed(using=False); os._exit(0)
        
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

#################
class DatasetDcase2021Task3:
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
        self.agents_num = 100
        self.segment_samples = self.sample_rate * 3
        self.segment_seconds = 3.
        self.frames_num = 301
        self.max_agents_contain_waveform = 2
        mic_yaml = "eigenmike.yaml"

        with open(mic_yaml, 'r') as f:
            self.mics_meta = yaml.load(f, Loader=yaml.FullLoader)

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
        # - Normal: calculate all agents and convolve with mic IR
        # - Fast: free-field use sparsity

        # Calculate target signal, hard sample
        # - Normal: calculate all signals at all directions
        # - Fast: Hard example directions

        random_seed = meta['random_seed']
        random_state = np.random.RandomState(random_seed)

        # -------- Mic
        agent_position = np.array([4, 4, 2])
        agent_position = expand_along_time(agent_position, self.frames_num)

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

        # --------- Simulate mic

        hdf5s_num = len(self.hdf5_names)
        hdf5_index = random_state.randint(hdf5s_num)

        hdf5_path = os.path.join(self.hdf5s_dir, self.hdf5_names[hdf5_index])

        with h5py.File(hdf5_path, 'r') as hf:

            audio_seconds = hf['waveform'].shape[-1] // self.sample_rate
            begin_second = random_state.uniform(low=0, high=audio_seconds - self.segment_seconds)
            begin_second = np.round(begin_second, decimals=1)
            end_second = begin_second + self.segment_seconds

            begin_sample = int(begin_second * self.sample_rate)
            end_sample = int(begin_sample + self.segment_samples)

            for mic_index, mic in enumerate(mics):
                segment = int16_to_float32(hf['waveform'][mic_index, begin_sample : end_sample])
                mic.waveform = segment

            begin_frame_10fps = int(begin_second * 10)
            end_frame_10fps = int(end_second * 10)

            frame_indexes = hf['frame_index'][:]
            class_indexes = hf['class_index'][:]
            event_indexes = hf['event_index'][:]

            for n in range(len(frame_indexes)):
                frame_index_10fps = frame_indexes[n]

                if begin_frame_10fps <= frame_index_10fps < end_frame_10fps:

                    relative_frame_index_10fps = frame_index_10fps - begin_frame_10fps
                    from IPython import embed; embed(using=False); os._exit(0)

            azimuths = np.deg2rad(hf['azimuth'][:] % 360)
            elevations = np.deg2rad(90 - hf['elevation'][:])

        # --------- Hard example new position agents

        agents = []

        half_angle = math.atan2(0.1, 1)

        for source in sources:

            agent_to_src = source.position - agent_position

            agent_look_direction = sample_agent_look_direction(
                agent_to_src=agent_to_src[0, :],
                half_angle=half_angle,
                random_state=random_state,
            )
            agent_look_direction = expand_along_time(agent_look_direction, self.frames_num)

            delayed_seconds = norm(agent_to_src[0, :]) / self.speed_of_sound
            delayed_samples = self.sample_rate * delayed_seconds

            y = fractional_delay(x=source.waveform, delayed_samples=delayed_samples)
            gain = 1.
            y *= gain

            agent = Agent(
                position=agent_position, 
                look_direction=agent_look_direction, 
                waveform=y,
                see_source=np.ones(self.frames_num),
            )

            agents.append(agent)

        while len(agents) < self.agents_num:

            _direction_sampler = DirectionSampler(
                low_colatitude=0, 
                high_colatitude=math.pi, 
                sample_on_sphere_uniformly=False, 
                random_state=random_state,
            )

            agent_look_azimuth, agent_look_colatitude = _direction_sampler.sample()
            agent_look_direction = np.array(sph2cart(
                r=1., 
                azimuth=agent_look_azimuth, 
                colatitude=agent_look_colatitude
            ))
            agent_look_direction = expand_along_time(agent_look_direction, self.frames_num)

            satisfied = True

            for source in sources:

                agent_to_src = source.position - agent_position
                angle_between_agent_and_src = np.arccos(get_cos(agent_look_direction[0, :], agent_to_src[0, :]))

                if angle_between_agent_and_src < half_angle:
                    satisfied = False

            if satisfied:
                agent = Agent(
                    position=agent_position, 
                    look_direction=agent_look_direction, 
                    waveform=np.zeros(self.segment_samples),
                    see_source=np.zeros(self.frames_num),
                )
                agents.append(agent)

        data_dict = {
            'source_position': np.array([source.position for source in sources]),
            'source_waveform': np.array([source.waveform for source in sources]),
            'mic_position': np.array([mic.position for mic in mics]),
            'mic_look_direction': np.array([mic.look_direction for mic in mics]),
            'mic_waveform': np.array([mic.waveform for mic in mics]),
            'agent_position': np.array([agent.position for agent in agents]),
            'agent_look_direction': np.array([agent.look_direction for agent in agents]),
            'agent_waveform': np.array([agent.waveform for agent in agents[0 : self.max_agents_contain_waveform]]),
            'agent_see_source': np.array([agent.see_source for agent in agents]),
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