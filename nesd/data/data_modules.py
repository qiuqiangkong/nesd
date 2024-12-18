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
from scipy.signal import fftconvolve
import pyroomacoustics as pra
from sklearn.linear_model import LinearRegression

from nesd.data.samplers import DistributedSamplerWrapper
from nesd.utils import Microphone, sph2cart, norm, normalize, Source, int16_to_float32, get_cos, calculate_microphone_gain, fractional_delay, DirectionSampler, cart2sph, Rotator3D, Agent, expand_along_time, CrossFade


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


class DatasetEigenmike:
    def __init__(
        self,
        hdf5s_dir,
        classes_num,
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
        mic_yaml = "eigenmike.yaml"

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


class DatasetEigenmike_VctkMusdb18hqD18t2:
    def __init__(
        self,
        hdf5s_dir,
        classes_num,
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
        mic_yaml = "eigenmike.yaml"

        with open(mic_yaml, 'r') as f:
            self.mics_meta = yaml.load(f, Loader=yaml.FullLoader)

        # self.hdf5s_dir = "/home/tiger/workspaces/nesd2/hdf5s/vctk/sr=24000/train"
        # self.hdf5s_dir = hdf5s_dir
        # self.hdf5_names = sorted(os.listdir(self.hdf5s_dir))

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
        hdf5_path = meta['hdf5_path']
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

            # h5s_num = len(self.hdf5_names)
            # h5_index = random_state.randint(h5s_num)
            # hdf5_path = os.path.join(self.hdf5s_dir, self.hdf5_names[h5_index])

            with h5py.File(hdf5_path, 'r') as hf:
                waveform = int16_to_float32(hf['waveform'][:])

            alpha = random_state.uniform(low=0.02, high=1.)
            waveform *= alpha

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


def get_random_silence_mask(frames_num, frames_per_sec, random_state):

    mask_duration_dict = {
        1: 3,
        2: 1.5,
        3: 1,
        4: 0.75,
    }

    mask = np.ones(frames_num)
    remove_num = random_state.randint(low=0, high=5)

    for i in range(remove_num):
        center = random_state.randint(low=0, high=frames_num)
        radius = random_state.randint(low=0, high=frames_per_sec * mask_duration_dict[remove_num] // 2)
        bgn = max(0, center - radius)
        end = min(center + radius, frames_num - 1)
        mask[bgn : end + 1] = 0

    return mask


def get_random_silence_mask2(frames_num, frames_per_sec, random_state):

    mask = np.zeros(frames_num)

    center = random_state.randint(low=0, high=frames_num)
    radius = random_state.randint(low=0, high=frames_num // 2)

    bgn = max(0, center - radius)
    end = min(center + radius, frames_num - 1)
    mask[bgn : end + 1] = 1

    return mask


class DatasetEigenmikeRandSil:
    def __init__(
        self,
        hdf5s_dir,
        classes_num,
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
        self.frames_per_sec = 100
        self.hop_samples = self.sample_rate // self.frames_per_sec
        self.max_agents_contain_waveform = 2
        mic_yaml = "eigenmike.yaml"

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

            frame_mask = get_random_silence_mask(
                frames_num=self.frames_num, 
                frames_per_sec=self.frames_per_sec, 
                random_state=random_state
            )
            sample_mask = np.repeat(frame_mask, repeats=self.hop_samples)[0 : self.segment_samples]
            waveform *= sample_mask

            source = Source(
                position=source_position,
                radius=0.1,
                waveform=waveform,
            )
            source.frame_mask = frame_mask
            sources.append(source)

        # fig, axs = plt.subplots(2,1)
        # axs[0].plot(source.waveform)
        # axs[1].plot(source.frame_mask)
        # plt.savefig('_zz.pdf')
        # from IPython import embed; embed(using=False); os._exit(0) 

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

            see_source = fractional_delay(x=source.frame_mask, delayed_samples=delayed_samples / self.hop_samples)
            see_source = np.clip(see_source, 0, 1)
            
            agent = Agent(
                position=agent_position, 
                look_direction=agent_look_direction, 
                waveform=y,
                see_source=see_source,
            )
            # soundfile.write(file='_zz.wav', data=y, samplerate=24000)
            # from IPython import embed; embed(using=False); os._exit(0)

            agents.append(agent)

        # fig, axs = plt.subplots(3,1)
        # axs[0].plot(agent.waveform)
        # axs[1].plot(agent.see_source)
        # plt.savefig('_zz.pdf')
        # from IPython import embed; embed(using=False); os._exit(0) 

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


class DatasetEigenmikeRandSil2:
    def __init__(
        self,
        hdf5s_dir,
        classes_num,
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
        self.frames_per_sec = 100
        self.hop_samples = self.sample_rate // self.frames_per_sec
        self.max_agents_contain_waveform = 2
        mic_yaml = "eigenmike.yaml"

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

            frame_mask = get_random_silence_mask(
                frames_num=self.frames_num, 
                frames_per_sec=self.frames_per_sec, 
                random_state=random_state
            )
            sample_mask = np.repeat(frame_mask, repeats=self.hop_samples)[0 : self.segment_samples]
            waveform *= sample_mask

            source = Source(
                position=source_position,
                radius=0.1,
                waveform=waveform,
            )
            source.frame_mask = frame_mask
            sources.append(source)

        # fig, axs = plt.subplots(2,1)
        # axs[0].plot(source.waveform)
        # axs[1].plot(source.frame_mask)
        # plt.savefig('_zz.pdf')
        # from IPython import embed; embed(using=False); os._exit(0) 

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

            see_source = fractional_delay(x=source.frame_mask, delayed_samples=delayed_samples / self.hop_samples)
            see_source = np.clip(see_source, 0, 1)
            
            agent = Agent(
                position=agent_position, 
                look_direction=agent_look_direction, 
                waveform=y,
                see_source=see_source,
            )
            # soundfile.write(file='_zz.wav', data=y, samplerate=24000)

            agents.append(agent)

        # fig, axs = plt.subplots(3,1)
        # axs[0].plot(agent.waveform)
        # axs[1].plot(agent.see_source)
        # plt.savefig('_zz.pdf')
        # from IPython import embed; embed(using=False); os._exit(0) 

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


def sample_agent_look_direction2(agent_to_src_direction, half_angle, random_state):

    frames_num = agent_to_src_direction.shape[0]
    
    _direction_sampler = DirectionSampler(
        low_colatitude=0, 
        high_colatitude=half_angle, 
        sample_on_sphere_uniformly=False, 
        random_state=random_state,
    )
    _azimuth, _colatitude = _direction_sampler.sample()

    _, agent_to_src_azimuth, agent_to_src_colatitude = cart2sph(
        x=agent_to_src_direction[:, 0], 
        y=agent_to_src_direction[:, 1], 
        z=agent_to_src_direction[:, 2],
    )

    agent_look_azimuth = np.zeros(frames_num)
    agent_look_colatitude = np.zeros(frames_num)

    for i in range(frames_num):

        rotation_matrix = Rotator3D.get_rotation_matrix_from_azimuth_colatitude(
            azimuth=agent_to_src_azimuth[i],
            colatitude=agent_to_src_colatitude[i],
        )

        agent_look_azimuth[i], agent_look_colatitude[i] = Rotator3D.rotate_azimuth_colatitude(
            rotation_matrix=rotation_matrix,
            azimuth=_azimuth,
            colatitude=_colatitude,
        )

    agent_look_direction = np.stack(sph2cart(
        r=1., 
        azimuth=agent_look_azimuth, 
        colatitude=agent_look_colatitude
    ), axis=-1)

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
        self.max_agents_contain_waveform = 8
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

def dcase_phase_to_nesd_phase(azimuth, elevation):
    nesd_azimuths = np.deg2rad(azimuth % 360)
    nesd_colatitude = np.deg2rad(90 - elevation)
    return nesd_azimuths, nesd_colatitude

'''
class Dataset4GroupMics:
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

        mics = []

        # mic_center_position = np.array([4, 4, 2])
        for mic_center_position in [np.array([2, 2, 2]), np.array([6, 2, 2]), np.array([6, 6, 2]), np.array([2, 6, 2])]:

            mic_center_position = expand_along_time(mic_center_position, self.frames_num)

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
        from IPython import embed; embed(using=False); os._exit(0)

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
'''

class Dataset4GroupMics:
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

        mics = []

        # mic_center_position = np.array([4, 4, 2])
        for mic_center_position in [np.array([2, 2, 2]), np.array([6, 2, 2]), np.array([6, 6, 2]), np.array([2, 6, 2])]:

            mic_center_position = expand_along_time(mic_center_position, self.frames_num)

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


class Dataset4GroupMicsDepth:
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
        self.max_agents_contain_depths = 20
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

        mics = []

        # mic_center_position = np.array([4, 4, 2])
        for mic_center_position in [np.array([2, 2, 2]), np.array([6, 2, 2]), np.array([6, 6, 2]), np.array([2, 6, 2])]:

            mic_center_position = expand_along_time(mic_center_position, self.frames_num)

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

        # depth agents
        agents2 = []

        for source in sources:

            agent_to_src = source.position - agent_position

            agent_look_direction = sample_agent_look_direction(
                agent_to_src=agent_to_src[0, :],
                half_angle=half_angle,
                random_state=random_state,
            )
            agent_look_direction = expand_along_time(agent_look_direction, self.frames_num)

            max_depth = 6
            sphere_r = 0.2
            distance = norm(agent_to_src[0, :])

            # Positive depth
            depth = random_state.uniform(
                low=max(0, distance - sphere_r), 
                high=min(distance + sphere_r, max_depth),
            )

            agent = Agent(
                position=agent_position, 
                look_direction=agent_look_direction, 
                waveform=np.zeros(self.segment_samples),
                see_source=np.ones(self.frames_num),
            )
            agent.look_depth = depth * np.ones(self.frames_num)
            agent.exist_source = np.ones(self.frames_num)
            agents2.append(agent)

            # Negative depth
            for _ in range(9):
                while True:
                    depth = random_state.uniform(low=0., high=max_depth)

                    if distance - sphere_r < depth < distance + sphere_r:
                        continue
                    else:
                        break

                agent = Agent(
                    position=agent_position, 
                    look_direction=agent_look_direction, 
                    waveform=np.zeros(self.segment_samples),
                    see_source=np.zeros(self.frames_num),
                )
                agent.look_depth = depth * np.ones(self.frames_num)
                agent.exist_source = np.zeros(self.frames_num)
                agents2.append(agent)

        agents.extend(agents2)

        data_dict = {
            'source_position': np.array([source.position for source in sources]),
            'source_waveform': np.array([source.waveform for source in sources]),
            'mic_position': np.array([mic.position for mic in mics]),
            'mic_look_direction': np.array([mic.look_direction for mic in mics]),
            'mic_waveform': np.array([mic.waveform for mic in mics]),
            'agent_position': np.array([agent.position for agent in agents]),
            'agent_look_direction': np.array([agent.look_direction for agent in agents]),
            'agent_waveform': np.array([agent.waveform for agent in agents[0 : self.max_agents_contain_waveform]]),
            'agent_see_source': np.array([agent.see_source for agent in agents[0 : self.agents_num]]),
            'agent_look_depth': np.array([agent.look_depth for agent in agents[self.agents_num : self.agents_num + self.max_agents_contain_depths]]),
            'agent_exist_source': np.array([agent.exist_source for agent in agents[self.agents_num : self.agents_num + self.max_agents_contain_depths]]),
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


#################
class DatasetDcase2021Task3:
    def __init__(
        self,
        hdf5s_dir,
        classes_num,
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

        self.dcase_fps = 10
        self.nesd_fps = 100
        self.classes_num = classes_num

        self.segment_frames_10fps = int(self.segment_seconds * self.dcase_fps) + 1

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

            begin_frame_10fps = int(begin_second * self.dcase_fps)
            end_frame_10fps = int(end_second * self.dcase_fps)
            

            frame_indexes_10fps = hf['frame_index'][:]
            class_indexes = hf['class_index'][:]
            event_indexes = hf['event_index'][:]

            azimuths, colatitudes = dcase_phase_to_nesd_phase(
                azimuth=hf['azimuth'][:],
                elevation=hf['elevation'][:],
            )

        # Collect sources
        event_ids = set(event_indexes)

        events_dict = {}

        for event_id in event_ids:

            event = None

            for n in range(len(frame_indexes_10fps)):

                frame_index_10fps = frame_indexes_10fps[n]

                if event_indexes[n] == event_id and begin_frame_10fps <= frame_index_10fps <= end_frame_10fps:

                    if event is None:
                        event = {
                            'class_id': np.ones(self.segment_frames_10fps, dtype=np.int32) * np.nan,
                            'azimuth': np.ones(self.segment_frames_10fps) * np.nan,
                            'colatitude': np.ones(self.segment_frames_10fps) * np.nan,
                        }

                    relative_frame_10fps = frame_index_10fps - begin_frame_10fps

                    event['class_id'][relative_frame_10fps] = class_indexes[n]
                    event['azimuth'][relative_frame_10fps] = azimuths[n]
                    event['colatitude'][relative_frame_10fps] = colatitudes[n]

            if event:
                events_dict[event_id] = event

        # sources
        sources = []

        for event_id in events_dict.keys():

            agent_to_src_direction = np.stack(sph2cart(
                r=1.,
                azimuth=events_dict[event_id]['azimuth'],
                colatitude=events_dict[event_id]['colatitude'],
            ), axis=-1)
            
            agent_to_src_direction_dcase_fps = extend_dcase_frames_to_nesd_frames(
                x=agent_to_src_direction, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.frames_num,
            )
            
            source_position = agent_to_src_direction_dcase_fps + agent_position

            source = Source(
                position=source_position,
                radius=0.1,
                waveform=np.nan * np.zeros(self.segment_samples),
            )
            sources.append(source)

        # --------- Hard example new position agents

        half_angle = math.atan2(0.1, 1)

        agents = []

        for event_id in events_dict.keys():

            class_id_array = events_dict[event_id]['class_id']
            azimuth_array = events_dict[event_id]['azimuth'].copy()
            colatitude_array = events_dict[event_id]['colatitude'].copy()

            agent_see_source = np.zeros(self.segment_frames_10fps)
            agent_see_source_classwise = np.zeros((self.segment_frames_10fps, self.classes_num))

            for i in range(self.segment_frames_10fps):
                if not math.isnan(class_id_array[i]):
                    agent_see_source[i] = 1
                    agent_see_source_classwise[i, int(class_id_array[i])] = 1

            for i in range(1, self.segment_frames_10fps):
                if math.isnan(azimuth_array[i]):
                    azimuth_array[i] = azimuth_array[i - 1]
                    colatitude_array[i] = colatitude_array[i - 1]

            for i in range(self.segment_frames_10fps - 2, -1, -1):
                if math.isnan(azimuth_array[i]):
                    azimuth_array[i] = azimuth_array[i + 1]
                    colatitude_array[i] = colatitude_array[i + 1]

            agent_to_src_direction = np.stack(sph2cart(
                r=1.,
                azimuth=azimuth_array,
                colatitude=colatitude_array
            ), axis=-1)

            agent_look_direction = sample_agent_look_direction2(
                agent_to_src_direction=agent_to_src_direction,
                half_angle=half_angle,
                random_state=random_state,
            )
            
            agent_look_direction = extend_dcase_frames_to_nesd_frames(
                x=agent_look_direction, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.frames_num,
            )

            agent_see_source = extend_dcase_frames_to_nesd_frames(
                x=agent_see_source, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.frames_num,
            )

            agent_see_source_classwise = extend_dcase_frames_to_nesd_frames(
                x=agent_see_source_classwise, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.frames_num,
            )
            
            agent = Agent(
                position=agent_position, 
                look_direction=agent_look_direction, 
                waveform=np.ones(self.segment_samples) * np.nan,
                see_source=agent_see_source,
                see_source_classwise=agent_see_source_classwise,
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
            agent_look_direction = expand_along_time(agent_look_direction, self.segment_frames_10fps)

            satisfied = True

            for event_id in events_dict.keys():

                azimuth_array = events_dict[event_id]['azimuth']
                colatitude_array = events_dict[event_id]['colatitude']

                agent_to_src_direction = np.stack(sph2cart(
                    r=1.,
                    azimuth=azimuth_array,
                    colatitude=colatitude_array
                ), axis=-1)

                for i in range(self.segment_frames_10fps):

                    if not math.isnan(azimuth_array[i]):

                        angle_between_agent_and_src = np.arccos(get_cos(agent_look_direction[i, :], agent_to_src_direction[i, :]))

                        if angle_between_agent_and_src < half_angle:
                            satisfied = False

            agent_look_direction = extend_dcase_frames_to_nesd_frames(
                x=agent_look_direction, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.frames_num,
            )
            
            if satisfied:
                agent = Agent(
                    position=agent_position, 
                    look_direction=agent_look_direction, 
                    waveform=np.ones(self.segment_samples) * np.nan,
                    see_source=np.zeros(self.frames_num),
                    see_source_classwise=np.zeros((self.frames_num, self.classes_num))
                )
                agents.append(agent)

        data_dict = {
            'source_position': np.array([source.position for source in sources]),
            'mic_position': np.array([mic.position for mic in mics]),
            'mic_look_direction': np.array([mic.look_direction for mic in mics]),
            'mic_waveform': np.array([mic.waveform for mic in mics]),
            'agent_position': np.array([agent.position for agent in agents]),
            'agent_look_direction': np.array([agent.look_direction for agent in agents]),
            'agent_waveform': np.array([agent.waveform for agent in agents[0 : self.max_agents_contain_waveform]]),
            'agent_see_source': np.array([agent.see_source for agent in agents]),
            'agent_see_source_classwise': np.array([agent.see_source_classwise for agent in agents[0 : self.max_agents_contain_waveform]]),
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

'''
def extend_dcase_frames_to_nesd_frames(x, dcase_fps, nesd_fps):
    x = np.repeat(x, repeats=nesd_fps // dcase_fps, axis=0)
    x = np.concatenate((x, x[-1:]), axis=0)
    return x
'''
def extend_dcase_frames_to_nesd_frames(x, dcase_fps, nesd_fps, dcase_segment_frames):
    x = np.repeat(x, repeats=nesd_fps // dcase_fps, axis=0)
    x = x[0 : dcase_segment_frames]
    return x

'''
def extend_look_direction(look_direction):

    frames_num = look_direction.shape[0]

    pairs = []

    if not math.isnan(look_direction[0, 0]):
        bgn = 0

    for i in range(1, frames_num):
        if not math.isnan(look_direction[i, 0]) and math.isnan(look_direction[i - 1, 0]):
            bgn = i

        if math.isnan(look_direction[i, 0]) and not math.isnan(look_direction[i - 1, 0]):
            end = i
            pairs.append((bgn, end))
            bgn = None
            end = None

    if not math.isnan(look_direction[-1, 0]) and bgn is not None:
        end = frames_num - 1
        pairs.append((bgn, end))
        bgn = None
        end = None

    from IPython import embed; embed(using=False); os._exit(0)
'''

def extend_look_direction(look_direction):

    frames_num = look_direction.shape[0]

    tmp = []  # nan
    tmp2 = []  # float

    new_x = look_direction.copy()

    # process begin Nan
    for i in range(frames_num):
        if math.isnan(look_direction[i, 0]):
            if len(tmp2) > 0:
                break
            else:
                tmp.append(i)

        if not math.isnan(look_direction[i, 0]):
            tmp2.append(i)

        if math.isnan(look_direction[i, 0]) and len(tmp2) > 0:
            break

    if len(tmp) > 0 and len(tmp2) > 0:
        X = np.array(tmp2)[:, None]
        y = look_direction[np.array(tmp2)]
        reg = LinearRegression().fit(X, y)
        pred = reg.predict(np.array(tmp)[:, None])
        new_x[np.array(tmp)] = pred

    # process end Nan
    tmp = []  # nan
    tmp2 = []  # float
    new_x = new_x[::-1]
    _look_direction = look_direction[::-1]

    for i in range(frames_num):
        if math.isnan(_look_direction[i, 0]):
            if len(tmp2) > 0:
                break
            else:
                tmp.append(i)

        if not math.isnan(_look_direction[i, 0]):
            tmp2.append(i)

        if math.isnan(_look_direction[i, 0]) and len(tmp2) > 0:
            break

    if len(tmp) > 0 and len(tmp2) > 0:
        X = np.array(tmp2)[:, None]
        y = _look_direction[np.array(tmp2)]
        reg = LinearRegression().fit(X, y)
        pred = reg.predict(np.array(tmp)[:, None])
        new_x[np.array(tmp)] = pred

    new_x = new_x[::-1]
    
    # process Middle Nan
    bgn = None
    end = None

    for i in range(1, frames_num):
        if math.isnan(new_x[i, 0]) and not math.isnan(new_x[i - 1, 0]):
            bgn = i - 1

        if not math.isnan(new_x[i, 0]) and math.isnan(new_x[i - 1, 0]):
            end = i

        if bgn is not None and end is not None:
            X = np.array([bgn, end])[:, None]
            y = np.array([new_x[bgn], new_x[end]])
            reg = LinearRegression().fit(X, y)
            pred = reg.predict(np.arange(bgn + 1, end)[:, None])
            new_x[np.arange(bgn + 1, end)] = pred
            bgn = None
            end = None
            
    for i in range(frames_num):
        new_x[i] = normalize(new_x[i])
    
    return new_x


class DatasetDcase2021Task3_MovB:
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

        self.dcase_fps = 10
        self.nesd_fps = 100
        self.classes_num = 12

        self.segment_frames_10fps = int(self.segment_seconds * self.dcase_fps) + 1

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

            begin_frame_10fps = int(begin_second * self.dcase_fps)
            end_frame_10fps = int(end_second * self.dcase_fps)
            

            frame_indexes_10fps = hf['frame_index'][:]
            class_indexes = hf['class_index'][:]
            event_indexes = hf['event_index'][:]

            azimuths, colatitudes = dcase_phase_to_nesd_phase(
                azimuth=hf['azimuth'][:],
                elevation=hf['elevation'][:],
            )

        # Collect sources
        event_ids = set(event_indexes)

        events_dict = {}

        for event_id in event_ids:

            event = None

            for n in range(len(frame_indexes_10fps)):

                frame_index_10fps = frame_indexes_10fps[n]

                if event_indexes[n] == event_id and begin_frame_10fps <= frame_index_10fps <= end_frame_10fps:

                    if event is None:
                        event = {
                            'class_id': np.ones(self.segment_frames_10fps, dtype=np.int32) * np.nan,
                            'azimuth': np.ones(self.segment_frames_10fps) * np.nan,
                            'colatitude': np.ones(self.segment_frames_10fps) * np.nan,
                        }

                    relative_frame_10fps = frame_index_10fps - begin_frame_10fps

                    event['class_id'][relative_frame_10fps] = class_indexes[n]
                    event['azimuth'][relative_frame_10fps] = azimuths[n]
                    event['colatitude'][relative_frame_10fps] = colatitudes[n]

            if event:
                events_dict[event_id] = event

        # sources
        sources = []

        for event_id in events_dict.keys():

            agent_to_src_direction = np.stack(sph2cart(
                r=1.,
                azimuth=events_dict[event_id]['azimuth'],
                colatitude=events_dict[event_id]['colatitude'],
            ), axis=-1)
            
            agent_to_src_direction_dcase_fps = extend_dcase_frames_to_nesd_frames(
                x=agent_to_src_direction, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.frames_num,
            )
            
            source_position = agent_to_src_direction_dcase_fps + agent_position

            source = Source(
                position=source_position,
                radius=0.1,
                waveform=np.nan * np.zeros(self.segment_samples),
            )
            sources.append(source)

        # --------- Hard example new position agents

        half_angle = math.atan2(0.1, 1)

        agents = []

        for event_id in events_dict.keys():

            class_id_array = events_dict[event_id]['class_id']
            azimuth_array = events_dict[event_id]['azimuth'].copy()
            colatitude_array = events_dict[event_id]['colatitude'].copy()

            agent_see_source = np.zeros(self.segment_frames_10fps)
            agent_see_source_classwise = np.zeros((self.segment_frames_10fps, self.classes_num))

            for i in range(self.segment_frames_10fps):
                if not math.isnan(class_id_array[i]):
                    agent_see_source[i] = 1
                    agent_see_source_classwise[i, int(class_id_array[i])] = 1

            agent_to_src_direction = np.stack(sph2cart(
                r=1.,
                azimuth=azimuth_array,
                colatitude=colatitude_array
            ), axis=-1)

            agent_look_direction = sample_agent_look_direction2(
                agent_to_src_direction=agent_to_src_direction,
                half_angle=half_angle,
                random_state=random_state,
            )

            agent_look_direction = extend_look_direction(agent_to_src_direction)
            
            agent_look_direction = extend_dcase_frames_to_nesd_frames(
                x=agent_look_direction, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.frames_num,
            )

            agent_see_source = extend_dcase_frames_to_nesd_frames(
                x=agent_see_source, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.frames_num,
            )

            agent_see_source_classwise = extend_dcase_frames_to_nesd_frames(
                x=agent_see_source_classwise, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.frames_num,
            )
            
            agent = Agent(
                position=agent_position, 
                look_direction=agent_look_direction, 
                waveform=np.ones(self.segment_samples) * np.nan,
                see_source=agent_see_source,
                see_source_classwise=agent_see_source_classwise,
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
            agent_look_direction = expand_along_time(agent_look_direction, self.segment_frames_10fps)

            satisfied = True

            for event_id in events_dict.keys():

                azimuth_array = events_dict[event_id]['azimuth']
                colatitude_array = events_dict[event_id]['colatitude']

                agent_to_src_direction = np.stack(sph2cart(
                    r=1.,
                    azimuth=azimuth_array,
                    colatitude=colatitude_array
                ), axis=-1)

                for i in range(self.segment_frames_10fps):

                    if not math.isnan(azimuth_array[i]):

                        angle_between_agent_and_src = np.arccos(get_cos(agent_look_direction[i, :], agent_to_src_direction[i, :]))

                        if angle_between_agent_and_src < half_angle:
                            satisfied = False

            agent_look_direction = extend_dcase_frames_to_nesd_frames(
                x=agent_look_direction, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.frames_num,
            )
            
            if satisfied:
                agent = Agent(
                    position=agent_position, 
                    look_direction=agent_look_direction, 
                    waveform=np.ones(self.segment_samples) * np.nan,
                    see_source=np.zeros(self.frames_num),
                    see_source_classwise=np.zeros((self.frames_num, self.classes_num))
                )
                agents.append(agent)

        data_dict = {
            'source_position': np.array([source.position for source in sources]),
            'mic_position': np.array([mic.position for mic in mics]),
            'mic_look_direction': np.array([mic.look_direction for mic in mics]),
            'mic_waveform': np.array([mic.waveform for mic in mics]),
            'agent_position': np.array([agent.position for agent in agents]),
            'agent_look_direction': np.array([agent.look_direction for agent in agents]),
            'agent_waveform': np.array([agent.waveform for agent in agents[0 : self.max_agents_contain_waveform]]),
            'agent_see_source': np.array([agent.see_source for agent in agents]),
            'agent_see_source_classwise': np.array([agent.see_source_classwise for agent in agents[0 : self.max_agents_contain_waveform]]),
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


'''
class DatasetDcase2021Task3_MovC:
    def __init__(
        self,
        hdf5s_dir,
        classes_num,
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

        self.dcase_fps = 10
        self.nesd_fps = 100
        self.classes_num = classes_num

        self.segment_frames_10fps = int(self.segment_seconds * self.dcase_fps) + 1

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

            begin_frame_10fps = int(begin_second * self.dcase_fps)
            end_frame_10fps = int(end_second * self.dcase_fps)
            

            frame_indexes_10fps = hf['frame_index'][:]
            class_indexes = hf['class_index'][:]
            event_indexes = hf['event_index'][:]

            azimuths, colatitudes = dcase_phase_to_nesd_phase(
                azimuth=hf['azimuth'][:],
                elevation=hf['elevation'][:],
            )

        # Collect sources
        event_ids = set(event_indexes)

        events_dict = {}

        for event_id in event_ids:

            event = None

            for n in range(len(frame_indexes_10fps)):

                frame_index_10fps = frame_indexes_10fps[n]

                if event_indexes[n] == event_id and begin_frame_10fps <= frame_index_10fps <= end_frame_10fps:

                    if event is None:
                        event = {
                            'class_id': np.ones(self.segment_frames_10fps, dtype=np.int32) * np.nan,
                            'azimuth': np.ones(self.segment_frames_10fps) * np.nan,
                            'colatitude': np.ones(self.segment_frames_10fps) * np.nan,
                        }

                    relative_frame_10fps = frame_index_10fps - begin_frame_10fps

                    event['class_id'][relative_frame_10fps] = class_indexes[n]
                    event['azimuth'][relative_frame_10fps] = azimuths[n]
                    event['colatitude'][relative_frame_10fps] = colatitudes[n]

            if event:
                events_dict[event_id] = event

        # sources
        sources = []

        for event_id in events_dict.keys():

            agent_to_src_direction = np.stack(sph2cart(
                r=1.,
                azimuth=events_dict[event_id]['azimuth'],
                colatitude=events_dict[event_id]['colatitude'],
            ), axis=-1)
            
            agent_to_src_direction_dcase_fps = extend_dcase_frames_to_nesd_frames(
                x=agent_to_src_direction, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.frames_num,
            )
            
            source_position = agent_to_src_direction_dcase_fps + agent_position

            source = Source(
                position=source_position,
                radius=0.1,
                waveform=np.nan * np.zeros(self.segment_samples),
            )
            sources.append(source)

        # --------- Hard example new position agents

        half_angle = math.atan2(0.1, 1)

        agents = []

        for event_id in events_dict.keys():

            class_id_array = events_dict[event_id]['class_id']
            azimuth_array = events_dict[event_id]['azimuth'].copy()
            colatitude_array = events_dict[event_id]['colatitude'].copy()

            agent_see_source = np.zeros(self.segment_frames_10fps)
            agent_see_source_classwise = np.zeros((self.segment_frames_10fps, self.classes_num))

            for i in range(self.segment_frames_10fps):
                if not math.isnan(class_id_array[i]):
                    agent_see_source[i] = 1
                    agent_see_source_classwise[i, int(class_id_array[i])] = 1

            agent_to_src_direction = np.stack(sph2cart(
                r=1.,
                azimuth=azimuth_array,
                colatitude=colatitude_array
            ), axis=-1)

            agent_look_direction = sample_agent_look_direction2(
                agent_to_src_direction=agent_to_src_direction,
                half_angle=half_angle,
                random_state=random_state,
            )

            agent_look_direction = extend_look_direction(agent_to_src_direction)
            
            agent_look_direction = extend_dcase_frames_to_nesd_frames(
                x=agent_look_direction, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.frames_num,
            )
            
            agent_see_source = extend_dcase_frames_to_nesd_frames(
                x=agent_see_source, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.frames_num,
            )

            agent_see_source_classwise = extend_dcase_frames_to_nesd_frames(
                x=agent_see_source_classwise, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.frames_num,
            )
            
            agent = Agent(
                position=agent_position, 
                look_direction=agent_look_direction, 
                waveform=np.ones(self.segment_samples) * np.nan,
                see_source=agent_see_source,
                see_source_classwise=agent_see_source_classwise,
            )

            agents.append(agent)

        #
        for event_id in events_dict.keys():

            for _ in range(5):

                class_id_array = events_dict[event_id]['class_id']
                azimuth_array = events_dict[event_id]['azimuth'].copy()
                colatitude_array = events_dict[event_id]['colatitude'].copy()

                agent_see_source = np.zeros(self.segment_frames_10fps)
                agent_see_source_classwise = np.zeros((self.segment_frames_10fps, self.classes_num))

                # for i in range(self.segment_frames_10fps):
                #     if not math.isnan(class_id_array[i]):
                #         agent_see_source[i] = 1
                #         agent_see_source_classwise[i, int(class_id_array[i])] = 1

                agent_to_src_direction = np.stack(sph2cart(
                    r=1.,
                    azimuth=azimuth_array,
                    colatitude=colatitude_array
                ), axis=-1)

                _frame_candidates = np.where(~np.isnan(agent_to_src_direction[:, 0]))[0]
                _frame_index = random_state.choice(_frame_candidates)
                agent_look_direction = agent_to_src_direction[_frame_index]

                agent_look_direction = expand_along_time(agent_look_direction, self.frames_num)

                agent_look_direction = sample_agent_look_direction2(
                    agent_to_src_direction=agent_look_direction,
                    half_angle=half_angle,
                    random_state=random_state,
                )

                for i in range(self.segment_frames_10fps):

                    if not math.isnan(azimuth_array[i]):

                        angle_between_agent_and_src = np.arccos(get_cos(agent_look_direction[i, :], agent_to_src_direction[i, :]))

                        if angle_between_agent_and_src < half_angle:
                            agent_see_source[i] = 1
                            agent_see_source_classwise[i, int(class_id_array[i])] = 1

                agent_see_source = extend_dcase_frames_to_nesd_frames(
                    x=agent_see_source, 
                    dcase_fps=self.dcase_fps,
                    nesd_fps=self.nesd_fps,
                    dcase_segment_frames=self.frames_num,
                )

                agent_see_source_classwise = extend_dcase_frames_to_nesd_frames(
                    x=agent_see_source_classwise, 
                    dcase_fps=self.dcase_fps,
                    nesd_fps=self.nesd_fps,
                    dcase_segment_frames=self.frames_num,
                )
                
                agent = Agent(
                    position=agent_position, 
                    look_direction=agent_look_direction, 
                    waveform=np.ones(self.segment_samples) * np.nan,
                    see_source=agent_see_source,
                    see_source_classwise=agent_see_source_classwise,
                )

                agents.append(agent)

        #
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
            agent_look_direction = expand_along_time(agent_look_direction, self.segment_frames_10fps)

            satisfied = True

            for event_id in events_dict.keys():

                azimuth_array = events_dict[event_id]['azimuth']
                colatitude_array = events_dict[event_id]['colatitude']

                agent_to_src_direction = np.stack(sph2cart(
                    r=1.,
                    azimuth=azimuth_array,
                    colatitude=colatitude_array
                ), axis=-1)

                for i in range(self.segment_frames_10fps):

                    if not math.isnan(azimuth_array[i]):

                        angle_between_agent_and_src = np.arccos(get_cos(agent_look_direction[i, :], agent_to_src_direction[i, :]))

                        if angle_between_agent_and_src < half_angle:
                            satisfied = False

            agent_look_direction = extend_dcase_frames_to_nesd_frames(
                x=agent_look_direction, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.frames_num,
            )
            
            if satisfied:
                agent = Agent(
                    position=agent_position, 
                    look_direction=agent_look_direction, 
                    waveform=np.ones(self.segment_samples) * np.nan,
                    see_source=np.zeros(self.frames_num),
                    see_source_classwise=np.zeros((self.frames_num, self.classes_num))
                )
                agents.append(agent)

        data_dict = {
            'source_position': np.array([source.position for source in sources]),
            'mic_position': np.array([mic.position for mic in mics]),
            'mic_look_direction': np.array([mic.look_direction for mic in mics]),
            'mic_waveform': np.array([mic.waveform for mic in mics]),
            'agent_position': np.array([agent.position for agent in agents]),
            'agent_look_direction': np.array([agent.look_direction for agent in agents]),
            'agent_waveform': np.array([agent.waveform for agent in agents[0 : self.max_agents_contain_waveform]]),
            'agent_see_source': np.array([agent.see_source for agent in agents]),
            'agent_see_source_classwise': np.array([agent.see_source_classwise for agent in agents[0 : self.max_agents_contain_waveform]]),
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
'''

class DatasetDcase2021Task3_MovC:
    def __init__(
        self,
        hdf5s_dir,
        classes_num,
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
        # self.dcase_frames_num = 31
        self.max_agents_contain_waveform = 2
        mic_yaml = "eigenmike.yaml"

        self.dcase_fps = 10
        self.nesd_fps = 100
        self.classes_num = classes_num

        self.segment_frames_10fps = int(self.segment_seconds * self.dcase_fps) + 1

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

            begin_frame_10fps = int(begin_second * self.dcase_fps)
            end_frame_10fps = int(end_second * self.dcase_fps)
            

            frame_indexes_10fps = hf['frame_index'][:]
            class_indexes = hf['class_index'][:]
            event_indexes = hf['event_index'][:]

            azimuths, colatitudes = dcase_phase_to_nesd_phase(
                azimuth=hf['azimuth'][:],
                elevation=hf['elevation'][:],
            )

        # Collect sources
        event_ids = set(event_indexes)

        events_dict = {}

        for event_id in event_ids:

            event = None

            for n in range(len(frame_indexes_10fps)):

                frame_index_10fps = frame_indexes_10fps[n]

                if event_indexes[n] == event_id and begin_frame_10fps <= frame_index_10fps <= end_frame_10fps:

                    if event is None:
                        event = {
                            'class_id': np.ones(self.segment_frames_10fps, dtype=np.int32) * np.nan,
                            'azimuth': np.ones(self.segment_frames_10fps) * np.nan,
                            'colatitude': np.ones(self.segment_frames_10fps) * np.nan,
                        }

                    relative_frame_10fps = frame_index_10fps - begin_frame_10fps

                    event['class_id'][relative_frame_10fps] = class_indexes[n]
                    event['azimuth'][relative_frame_10fps] = azimuths[n]
                    event['colatitude'][relative_frame_10fps] = colatitudes[n]

            if event:
                events_dict[event_id] = event

        # sources
        sources = []

        for event_id in events_dict.keys():

            agent_to_src_direction = np.stack(sph2cart(
                r=1.,
                azimuth=events_dict[event_id]['azimuth'],
                colatitude=events_dict[event_id]['colatitude'],
            ), axis=-1)
            
            agent_to_src_direction_dcase_fps = extend_dcase_frames_to_nesd_frames(
                x=agent_to_src_direction, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.frames_num,
            )
            
            source_position = agent_to_src_direction_dcase_fps + agent_position

            source = Source(
                position=source_position,
                radius=0.1,
                waveform=np.nan * np.zeros(self.segment_samples),
            )
            sources.append(source)

        # --------- Hard example new position agents

        half_angle = math.atan2(0.1, 1)

        agents = []

        for event_id in events_dict.keys():

            class_id_array = events_dict[event_id]['class_id']
            azimuth_array = events_dict[event_id]['azimuth'].copy()
            colatitude_array = events_dict[event_id]['colatitude'].copy()

            agent_see_source = np.zeros(self.segment_frames_10fps)
            agent_see_source_classwise = np.zeros((self.segment_frames_10fps, self.classes_num))

            for i in range(self.segment_frames_10fps):
                if not math.isnan(class_id_array[i]):
                    agent_see_source[i] = 1
                    agent_see_source_classwise[i, int(class_id_array[i])] = 1

            agent_to_src_direction = np.stack(sph2cart(
                r=1.,
                azimuth=azimuth_array,
                colatitude=colatitude_array
            ), axis=-1)

            agent_look_direction = sample_agent_look_direction2(
                agent_to_src_direction=agent_to_src_direction,
                half_angle=half_angle,
                random_state=random_state,
            )

            agent_look_direction = extend_look_direction(agent_to_src_direction)
            
            agent_look_direction = extend_dcase_frames_to_nesd_frames(
                x=agent_look_direction, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.frames_num,
            )
            
            agent_see_source = extend_dcase_frames_to_nesd_frames(
                x=agent_see_source, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.frames_num,
            )

            agent_see_source_classwise = extend_dcase_frames_to_nesd_frames(
                x=agent_see_source_classwise, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.frames_num,
            )
            
            agent = Agent(
                position=agent_position, 
                look_direction=agent_look_direction, 
                waveform=np.ones(self.segment_samples) * np.nan,
                see_source=agent_see_source,
                see_source_classwise=agent_see_source_classwise,
            )

            agents.append(agent)

        #
        for event_id in events_dict.keys():

            for _ in range(5):

                class_id_array = events_dict[event_id]['class_id']
                azimuth_array = events_dict[event_id]['azimuth'].copy()
                colatitude_array = events_dict[event_id]['colatitude'].copy()

                agent_see_source = np.zeros(self.segment_frames_10fps)
                agent_see_source_classwise = np.zeros((self.segment_frames_10fps, self.classes_num))

                # for i in range(self.segment_frames_10fps):
                #     if not math.isnan(class_id_array[i]):
                #         agent_see_source[i] = 1
                #         agent_see_source_classwise[i, int(class_id_array[i])] = 1

                agent_to_src_direction = np.stack(sph2cart(
                    r=1.,
                    azimuth=azimuth_array,
                    colatitude=colatitude_array
                ), axis=-1)

                _frame_candidates = np.where(~np.isnan(agent_to_src_direction[:, 0]))[0]
                _frame_index = random_state.choice(_frame_candidates)
                agent_look_direction = agent_to_src_direction[_frame_index]

                agent_look_direction = expand_along_time(agent_look_direction, self.segment_frames_10fps)

                agent_look_direction = sample_agent_look_direction2(
                    agent_to_src_direction=agent_look_direction,
                    half_angle=half_angle,
                    random_state=random_state,
                )

                for i in range(self.segment_frames_10fps):

                    if not math.isnan(azimuth_array[i]):

                        angle_between_agent_and_src = np.arccos(get_cos(agent_look_direction[i, :], agent_to_src_direction[i, :]))

                        if angle_between_agent_and_src < half_angle:
                            agent_see_source[i] = 1
                            agent_see_source_classwise[i, int(class_id_array[i])] = 1

                agent_look_direction = extend_dcase_frames_to_nesd_frames(
                    x=agent_look_direction, 
                    dcase_fps=self.dcase_fps,
                    nesd_fps=self.nesd_fps,
                    dcase_segment_frames=self.frames_num,
                )

                agent_see_source = extend_dcase_frames_to_nesd_frames(
                    x=agent_see_source, 
                    dcase_fps=self.dcase_fps,
                    nesd_fps=self.nesd_fps,
                    dcase_segment_frames=self.frames_num,
                )

                agent_see_source_classwise = extend_dcase_frames_to_nesd_frames(
                    x=agent_see_source_classwise, 
                    dcase_fps=self.dcase_fps,
                    nesd_fps=self.nesd_fps,
                    dcase_segment_frames=self.frames_num,
                )
                
                agent = Agent(
                    position=agent_position, 
                    look_direction=agent_look_direction, 
                    waveform=np.ones(self.segment_samples) * np.nan,
                    see_source=agent_see_source,
                    see_source_classwise=agent_see_source_classwise,
                )

                agents.append(agent)

        #
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
            agent_look_direction = expand_along_time(agent_look_direction, self.segment_frames_10fps)

            satisfied = True

            for event_id in events_dict.keys():

                azimuth_array = events_dict[event_id]['azimuth']
                colatitude_array = events_dict[event_id]['colatitude']

                agent_to_src_direction = np.stack(sph2cart(
                    r=1.,
                    azimuth=azimuth_array,
                    colatitude=colatitude_array
                ), axis=-1)

                for i in range(self.segment_frames_10fps):

                    if not math.isnan(azimuth_array[i]):

                        angle_between_agent_and_src = np.arccos(get_cos(agent_look_direction[i, :], agent_to_src_direction[i, :]))

                        if angle_between_agent_and_src < half_angle:
                            satisfied = False

            agent_look_direction = extend_dcase_frames_to_nesd_frames(
                x=agent_look_direction, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.frames_num,
            )
            
            if satisfied:
                agent = Agent(
                    position=agent_position, 
                    look_direction=agent_look_direction, 
                    waveform=np.ones(self.segment_samples) * np.nan,
                    see_source=np.zeros(self.frames_num),
                    see_source_classwise=np.zeros((self.frames_num, self.classes_num))
                )
                agents.append(agent)

        data_dict = {
            'source_position': np.array([source.position for source in sources]),
            'mic_position': np.array([mic.position for mic in mics]),
            'mic_look_direction': np.array([mic.look_direction for mic in mics]),
            'mic_waveform': np.array([mic.waveform for mic in mics]),
            'agent_position': np.array([agent.position for agent in agents]),
            'agent_look_direction': np.array([agent.look_direction for agent in agents]),
            'agent_waveform': np.array([agent.waveform for agent in agents[0 : self.max_agents_contain_waveform]]),
            'agent_see_source': np.array([agent.see_source for agent in agents]),
            'agent_see_source_classwise': np.array([agent.see_source_classwise for agent in agents[0 : self.max_agents_contain_waveform]]),
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


##############
class DatasetPra:
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
        debug = False

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

        if debug:
            mics.append(Microphone(
                position=mic_center_position,
                look_direction=mic_look_direction,
                directivity=mic_meta['directivity'],
            ))

        # --------- Source
        if debug:
            sources_num = 1
        else:
            sources_num = 2
        sources = []

        success_flag = False
        
        while not success_flag:

            for i in range(sources_num):
            
                source_position = mic_center_position + normalize(random_state.uniform(low=-1, high=1, size=3))

                h5s_num = len(self.hdf5_names)
                h5_index = random_state.randint(h5s_num)
                hdf5_path = os.path.join(self.hdf5s_dir, self.hdf5_names[h5_index])

                with h5py.File(hdf5_path, 'r') as hf:
                    waveform = int16_to_float32(hf['waveform'][:])

                if debug:
                    waveform = np.zeros(72000)
                    waveform[100] = 1

                source = Source(
                    position=source_position,
                    radius=0.1,
                    waveform=waveform,
                )
                sources.append(source)

            # --------- Fast simulate mic
            
            corners = np.array([
                [8, 8], 
                [0, 8], 
                [0, 0], 
                [8, 0],
            ]).T
            height = 4

            r = 0.2
            wall_mat = {
                "description": "Example wall material",
                "coeffs": [r, r, r, r, r, r, r, r, ],
                "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000, 16000],
            }

            materials = pra.Material(
                energy_absorption=wall_mat,
            )

            if debug:
                is_raytracing = True
            else:
                is_raytracing = True

            t1 = time.time()
            if debug:
                room = pra.Room.from_corners(
                    corners=corners,
                    fs=self.sample_rate,
                    materials=materials,
                    max_order=3,
                    ray_tracing=is_raytracing,
                    air_absorption=False,
                )
            else:
                room = pra.Room.from_corners(
                    corners=corners,
                    fs=self.sample_rate,
                    materials=materials,
                    max_order=3,
                    ray_tracing=is_raytracing,
                    air_absorption=False,
                )

            room.extrude(
                height=height, 
                materials=materials,
            )

            if is_raytracing:
                room.set_ray_tracing(
                    n_rays=1000,
                    receiver_radius=0.5,
                    energy_thres=1e-7,
                    time_thres=1.0,
                    hist_bin_size=0.004,
                )

            try:
                for source in sources:
                    room.add_source(
                        position=source.position[0],
                        signal=source.waveform,
                    )

                directivity_object = None

                for mic in mics:
                    room.add_microphone(
                        loc=mic.position[0], 
                        directivity=directivity_object,
                    )

                room.compute_rir()
                success_flag = True
                
            except:
                success_flag = False

        room.simulate()
        # print(time.time() - t1)

        for j, mic in enumerate(mics):
            mic.waveform = room.mic_array.signals[j, 40 : 40 + self.segment_samples]
        
        # soundfile.write(file='_zz.wav', data=mics[0].waveform, samplerate=24000)
        # soundfile.write(file='_zz1.wav', data=sources[0].waveform, samplerate=24000)
        # fig, axs = plt.subplots(2,1, sharex=True)
        # axs[0].stem(sources[0].waveform[0:3000])
        # axs[1].stem(mics[-1].waveform[0:3000])
        # plt.savefig('_zz.pdf')
        # np.argmax(mics[-1].waveform)
        # from IPython import embed; embed(using=False); os._exit(0)

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
            gain = 1. / norm(agent_to_src[0, :])
            y *= gain

            # fig, axs = plt.subplots(2,1, sharex=True)
            # axs[0].stem(mics[-1].waveform[0:1000])
            # axs[1].stem(y[0:1000])
            # plt.savefig('_zz.pdf')

            # from IPython import embed; embed(using=False); os._exit(0)

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


class DatasetPraFreefield:
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
        debug = False

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

        if debug:
            mics.append(Microphone(
                position=mic_center_position,
                look_direction=mic_look_direction,
                directivity=mic_meta['directivity'],
            ))

        # --------- Source
        if debug:
            sources_num = 1
        else:
            sources_num = 2
        sources = []

        for i in range(sources_num):
        
            source_position = mic_center_position + normalize(random_state.uniform(low=-1, high=1, size=3))

            h5s_num = len(self.hdf5_names)
            h5_index = random_state.randint(h5s_num)
            hdf5_path = os.path.join(self.hdf5s_dir, self.hdf5_names[h5_index])

            with h5py.File(hdf5_path, 'r') as hf:
                waveform = int16_to_float32(hf['waveform'][:])

            if debug:
                waveform = np.zeros(72000)
                waveform[100] = 1

            source = Source(
                position=source_position,
                radius=0.1,
                waveform=waveform,
            )
            sources.append(source)

        # --------- Fast simulate mic
        
        corners = np.array([
            [8, 8], 
            [0, 8], 
            [0, 0], 
            [8, 0],
        ]).T
        height = 4

        r = 0.2
        wall_mat = {
            "description": "Example wall material",
            "coeffs": [r, r, r, r, r, r, r, r, ],
            "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000, 16000],
        }

        materials = pra.Material(
            energy_absorption=wall_mat,
        )

        if debug:
            is_raytracing = False
        else:
            is_raytracing = False

        t1 = time.time()
        if debug:
            room = pra.Room.from_corners(
                corners=corners,
                fs=self.sample_rate,
                materials=materials,
                max_order=0,
                ray_tracing=is_raytracing,
                air_absorption=False,
            )
        else:
            room = pra.Room.from_corners(
                corners=corners,
                fs=self.sample_rate,
                materials=materials,
                max_order=0,
                ray_tracing=False,
                air_absorption=False,
            )

        room.extrude(
            height=height, 
            materials=materials,
        )

        if is_raytracing:
            room.set_ray_tracing(
                n_rays=1000,
                receiver_radius=0.5,
                energy_thres=1e-7,
                time_thres=1.0,
                hist_bin_size=0.004,
            )

        for source in sources:
            room.add_source(
                position=source.position[0],
                signal=source.waveform,
            )

        directivity_object = None

        for mic in mics:
            room.add_microphone(
                loc=mic.position[0], 
                directivity=directivity_object,
            )

        room.compute_rir()

        room.simulate()
        # print(time.time() - t1)

        for j, mic in enumerate(mics):
            mic.waveform = room.mic_array.signals[j, 40 : 40 + self.segment_samples]
        
        # soundfile.write(file='_zz.wav', data=mics[0].waveform, samplerate=24000)
        # soundfile.write(file='_zz1.wav', data=sources[0].waveform, samplerate=24000)
        # fig, axs = plt.subplots(2,1, sharex=True)
        # axs[0].stem(sources[0].waveform[0:1000])
        # axs[1].stem(mics[-1].waveform[0:1000])
        # plt.savefig('_zz.pdf')
        # np.argmax(mics[-1].waveform)
        # from IPython import embed; embed(using=False); os._exit(0)

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
            gain = 1. / norm(agent_to_src[0, :])
            y *= gain

            # fig, axs = plt.subplots(2,1, sharex=True)
            # axs[0].stem(mics[-1].waveform[0:1000])
            # axs[1].stem(y[0:1000])
            # plt.savefig('_zz.pdf')

            # from IPython import embed; embed(using=False); os._exit(0)

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


class DatasetPraOrd1:
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
        debug = False

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

        if debug:
            mics.append(Microphone(
                position=mic_center_position,
                look_direction=mic_look_direction,
                directivity=mic_meta['directivity'],
            ))

        # --------- Source
        if debug:
            sources_num = 1
        else:
            sources_num = 2
        sources = []

        success_flag = False

        while not success_flag:

            for i in range(sources_num):
            
                source_position = mic_center_position + normalize(random_state.uniform(low=-1, high=1, size=3))

                h5s_num = len(self.hdf5_names)
                h5_index = random_state.randint(h5s_num)
                hdf5_path = os.path.join(self.hdf5s_dir, self.hdf5_names[h5_index])

                with h5py.File(hdf5_path, 'r') as hf:
                    waveform = int16_to_float32(hf['waveform'][:])

                if debug:
                    waveform = np.zeros(72000)
                    waveform[100] = 1

                source = Source(
                    position=source_position,
                    radius=0.1,
                    waveform=waveform,
                )
                sources.append(source)

            # --------- Fast simulate mic
            
            corners = np.array([
                [8, 8], 
                [0, 8], 
                [0, 0], 
                [8, 0],
            ]).T
            height = 4

            r = 0.2
            wall_mat = {
                "description": "Example wall material",
                "coeffs": [r, r, r, r, r, r, r, r, ],
                "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000, 16000],
            }

            materials = pra.Material(
                energy_absorption=wall_mat,
            )

            if debug:
                is_raytracing = False
            else:
                is_raytracing = False

            t1 = time.time()
            if debug:
                room = pra.Room.from_corners(
                    corners=corners,
                    fs=self.sample_rate,
                    materials=materials,
                    max_order=5,
                    ray_tracing=is_raytracing,
                    air_absorption=False,
                )
            else:
                room = pra.Room.from_corners(
                    corners=corners,
                    fs=self.sample_rate,
                    materials=materials,
                    max_order=5,
                    ray_tracing=False,
                    air_absorption=False,
                )

            room.extrude(
                height=height, 
                materials=materials,
            )

            if is_raytracing:
                room.set_ray_tracing(
                    n_rays=1000,
                    receiver_radius=0.5,
                    energy_thres=1e-7,
                    time_thres=1.0,
                    hist_bin_size=0.004,
                )

            try:
                for source in sources:
                    room.add_source(
                        position=source.position[0],
                        signal=source.waveform,
                    )

                directivity_object = None

                for mic in mics:
                    room.add_microphone(
                        loc=mic.position[0], 
                        directivity=directivity_object,
                    )

                room.compute_rir()
                success_flag = True

            except:
                success_flag = False

        room.simulate()
        # print(time.time() - t1)

        for j, mic in enumerate(mics):
            mic.waveform = room.mic_array.signals[j, 40 : 40 + self.segment_samples]
        
        # soundfile.write(file='_zz.wav', data=mics[0].waveform, samplerate=24000)
        # soundfile.write(file='_zz1.wav', data=sources[0].waveform, samplerate=24000)
        # fig, axs = plt.subplots(2,1, sharex=True)
        # axs[0].stem(sources[0].waveform[0:5000])
        # axs[1].stem(mics[-1].waveform[0:5000])
        # plt.savefig('_zz.pdf')
        # np.argmax(mics[-1].waveform)
        # from IPython import embed; embed(using=False); os._exit(0)

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
            gain = 1. / norm(agent_to_src[0, :])
            y *= gain

            # fig, axs = plt.subplots(2,1, sharex=True)
            # axs[0].stem(mics[-1].waveform[0:1000])
            # axs[1].stem(y[0:1000])
            # plt.savefig('_zz.pdf')

            # from IPython import embed; embed(using=False); os._exit(0)

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

### Mov src
'''
class DatasetEigenmikeMovSrc:
    def __init__(
        self,
        hdf5s_dir,
        classes_num,
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
        self.cross_fade_window = 2400
        self.max_agents_contain_waveform = 2
        mic_yaml = "eigenmike.yaml"

        self.dcase_segment_frames = 31
        self.dcase_fps = 10
        self.nesd_fps = 100

        with open(mic_yaml, 'r') as f:
            self.mics_meta = yaml.load(f, Loader=yaml.FullLoader)

        # self.hdf5s_dir = "/home/tiger/workspaces/nesd2/hdf5s/vctk/sr=24000/train"
        self.hdf5s_dir = hdf5s_dir
        self.hdf5_names = sorted(os.listdir(self.hdf5s_dir))

        self.cross_fader = CrossFade(
            window_samples=self.cross_fade_window, 
            segment_samples=self.segment_samples
        )

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

            is_moving = random_state.choice([True, False])

            if is_moving:
                deg_per_sec = random_state.uniform(low=-40, high=40)

                source_position = get_moving_source_positions(source_position, mic_center_position[0], deg_per_sec, self.dcase_segment_frames, self.dcase_fps)

                source_position = extend_dcase_frames_to_nesd_frames(
                    x=source_position, 
                    dcase_fps=self.dcase_fps,
                    nesd_fps=self.nesd_fps,
                    dcase_segment_frames=self.frames_num,
                )

            else:
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

                window_wavs = []
                for t in range(0, self.frames_num - 1, self.nesd_fps // self.dcase_fps):

                    delayed_seconds = norm(mic_to_src[t, :]) / self.speed_of_sound
                    delayed_samples = self.sample_rate * delayed_seconds

                    cos = get_cos(mic.look_direction[t, :], mic_to_src[t, :])
                    gain = calculate_microphone_gain(cos=cos, directivity=mic.directivity)

                    y = fractional_delay(x=source.waveform, delayed_samples=delayed_samples)
                    y *= gain
                    window_wavs.append(y)
                    # print(norm(mic_to_src[t, :]), delayed_seconds, delayed_samples)

                window_wavs = np.stack(window_wavs, axis=0)
                mic.waveform += self.cross_fader.forward(window_wavs)

        # --------- Hard example new position agents

        agents = []

        half_angle = math.atan2(0.1, 1)

        for source in sources:

            agent_to_src = source.position - agent_position
            agent_to_src = agent_to_src[0 :: 10, :]

            agent_to_src_direction = normalize(agent_to_src)

            for _ in range(5):

                _frame_index = random_state.randint(self.dcase_segment_frames)

                agent_look_direction = agent_to_src_direction[_frame_index]

                agent_look_direction = expand_along_time(agent_look_direction, self.dcase_segment_frames)

                agent_look_direction = sample_agent_look_direction2(
                    agent_to_src_direction=agent_look_direction,
                    half_angle=half_angle,
                    random_state=random_state,
                )

                agent_see_source = np.zeros(self.dcase_segment_frames)
                agent_see_source_classwise = np.zeros((self.dcase_segment_frames, self.classes_num))

                for t in range(self.dcase_segment_frames):

                    angle_between_agent_and_src = np.arccos(get_cos(agent_look_direction[t, :], agent_to_src_direction[t, :]))

                    if angle_between_agent_and_src < half_angle:
                        agent_see_source[i] = 1
                        agent_see_source_classwise[i, int(class_id_array[i])] = 1
            
                from IPython import embed; embed(using=False); os._exit(0)

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
'''

class DatasetEigenmikeMovSrc:
    def __init__(
        self,
        hdf5s_dir,
        classes_num,
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
        self.cross_fade_window = 2400
        self.max_agents_contain_waveform = 2
        mic_yaml = "eigenmike.yaml"

        self.dcase_frames_num = 31
        self.nesd_frames_num = 301
        self.dcase_fps = 10
        self.nesd_fps = 100

        with open(mic_yaml, 'r') as f:
            self.mics_meta = yaml.load(f, Loader=yaml.FullLoader)

        # self.hdf5s_dir = "/home/tiger/workspaces/nesd2/hdf5s/vctk/sr=24000/train"
        self.hdf5s_dir = hdf5s_dir
        self.hdf5_names = sorted(os.listdir(self.hdf5s_dir))

        self.cross_fader = CrossFade(
            window_samples=self.cross_fade_window, 
            segment_samples=self.segment_samples
        )

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
        agent_position = expand_along_time(agent_position, self.dcase_frames_num)

        mic_center_position = np.array([4, 4, 2])
        mic_center_position = expand_along_time(mic_center_position, self.dcase_frames_num)
        
        mics = []

        for mic_meta in self.mics_meta:

            relative_mic_posision = np.array(sph2cart(
                r=mic_meta['radius'], 
                azimuth=mic_meta['azimuth'], 
                colatitude=mic_meta['colatitude']
            ))

            mic_position = mic_center_position + relative_mic_posision[None, :]

            mic_look_direction = normalize(relative_mic_posision)
            mic_look_direction = expand_along_time(mic_look_direction, self.dcase_frames_num)

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

            is_moving = random_state.choice([True, False])

            if is_moving:
                deg_per_sec = random_state.uniform(low=-40, high=40)

                source_position = get_moving_source_positions(source_position, mic_center_position[0], deg_per_sec, self.dcase_frames_num, self.dcase_fps)

            else:
                source_position = expand_along_time(source_position, self.dcase_frames_num)

            # plt.scatter(source_position[:, 0], source_position[:, 1], s=10, c='b')
            # plt.xlim(0, 8)
            # plt.ylim(0, 8)
            # plt.savefig('_zz.pdf')

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

        # t1 = time.time()
        # Microphone signals
        for mic in mics:

            # total = 0
            for source in sources:
                mic_to_src = source.position - mic.position

                window_wavs = []
                for t in range(0, self.dcase_frames_num - 1):

                    delayed_seconds = norm(mic_to_src[t, :]) / self.speed_of_sound
                    delayed_samples = self.sample_rate * delayed_seconds

                    cos = get_cos(mic.look_direction[t, :], mic_to_src[t, :])
                    gain = calculate_microphone_gain(cos=cos, directivity=mic.directivity)

                    y = fractional_delay(x=source.waveform, delayed_samples=delayed_samples)
                    y *= gain
                    window_wavs.append(y)
                
                window_wavs = np.stack(window_wavs, axis=0)
                mic.waveform += self.cross_fader.forward(window_wavs)

        # print("t1: {:.3f}".format(time.time() - t1))
        # from IPython import embed; embed(using=False); os._exit(0)
        # soundfile.write(file='_zz.wav', data=mic.waveform, samplerate=self.sample_rate)

        # --------- Hard example new position agents
        
        agents = []

        half_angle = math.atan2(0.1, 1)

        for source in sources:

            agent_to_src = source.position - agent_position
            agent_to_src_direction = normalize(agent_to_src)

            for _ in range(5):

                _frame_index = random_state.randint(self.dcase_frames_num)

                agent_look_direction = agent_to_src_direction[_frame_index]

                agent_look_direction = expand_along_time(agent_look_direction, self.dcase_frames_num)

                agent_look_direction = sample_agent_look_direction2(
                    agent_to_src_direction=agent_look_direction,
                    half_angle=half_angle,
                    random_state=random_state,
                )

                agent_see_source = np.zeros(self.dcase_frames_num)
                # agent_see_source_classwise = np.zeros((self.dcase_frames_num, self.classes_num))

                # Detect 
                for t in range(self.dcase_frames_num):

                    angle_between_agent_and_src = np.arccos(get_cos(agent_look_direction[t, :], agent_to_src_direction[t, :]))

                    if angle_between_agent_and_src < half_angle:
                        agent_see_source[t] = 1

                # Build wavs
                window_wavs = []
                for t in range(0, self.dcase_frames_num - 1):

                    if agent_see_source[t] == 1:

                        delayed_seconds = norm(agent_to_src[t, :]) / self.speed_of_sound
                        delayed_samples = self.sample_rate * delayed_seconds

                        gain = 1.
                        y = fractional_delay(x=source.waveform, delayed_samples=delayed_samples)
                        y *= gain
                        tmp = y
                        y = np.zeros(self.segment_samples)
                        y[t * self.cross_fade_window : (t + 1) * self.cross_fade_window] = tmp[t * self.cross_fade_window : (t + 1) * self.cross_fade_window]

                    else:
                        y = np.zeros(self.segment_samples)
                        
                    window_wavs.append(y)
                
                window_wavs = np.stack(window_wavs, axis=0)
                wav = self.cross_fader.forward(window_wavs)
                # soundfile.write(file='_zz.wav', data=a1, samplerate=self.sample_rate)
                # from IPython import embed; embed(using=False); os._exit(0)

                _agent_position = extend_dcase_frames_to_nesd_frames(
                    x=agent_position, 
                    dcase_fps=self.dcase_fps,
                    nesd_fps=self.nesd_fps,
                    dcase_segment_frames=self.nesd_frames_num,
                )

                _agent_look_direction = extend_dcase_frames_to_nesd_frames(
                    x=agent_look_direction, 
                    dcase_fps=self.dcase_fps,
                    nesd_fps=self.nesd_fps,
                    dcase_segment_frames=self.nesd_frames_num,
                )

                _agent_see_source = extend_dcase_frames_to_nesd_frames(
                    x=agent_see_source, 
                    dcase_fps=self.dcase_fps,
                    nesd_fps=self.nesd_fps,
                    dcase_segment_frames=self.nesd_frames_num,
                )

                agent = Agent(
                    position=_agent_position, 
                    look_direction=_agent_look_direction, 
                    waveform=wav,
                    see_source=_agent_see_source,
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
            agent_look_direction = expand_along_time(agent_look_direction, self.dcase_frames_num)

            satisfied = True

            for source in sources:

                agent_to_src = source.position - agent_position

                for t in range(0, self.dcase_frames_num - 1):
                    angle_between_agent_and_src = np.arccos(get_cos(agent_look_direction[t, :], agent_to_src[t, :]))

                    if angle_between_agent_and_src < half_angle:
                        satisfied = False

            if satisfied:

                _agent_position = extend_dcase_frames_to_nesd_frames(
                    x=agent_position, 
                    dcase_fps=self.dcase_fps,
                    nesd_fps=self.nesd_fps,
                    dcase_segment_frames=self.nesd_frames_num,
                )

                _agent_look_direction = extend_dcase_frames_to_nesd_frames(
                    x=agent_look_direction, 
                    dcase_fps=self.dcase_fps,
                    nesd_fps=self.nesd_fps,
                    dcase_segment_frames=self.nesd_frames_num,
                )

                agent = Agent(
                    position=_agent_position, 
                    look_direction=_agent_look_direction, 
                    waveform=np.zeros(self.segment_samples),
                    see_source=np.zeros(self.nesd_frames_num),
                )
                agents.append(agent)

        for source in sources:
            source.position = extend_dcase_frames_to_nesd_frames(
                x=source.position, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.nesd_frames_num,
            )

        for mic in mics:
            mic.position = extend_dcase_frames_to_nesd_frames(
                x=mic.position, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.nesd_frames_num,
            )

            mic.look_direction = extend_dcase_frames_to_nesd_frames(
                x=mic.look_direction, 
                dcase_fps=self.dcase_fps,
                nesd_fps=self.nesd_fps,
                dcase_segment_frames=self.nesd_frames_num,
            )

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


def get_moving_source_positions(source_position, mic_center_position, deg_per_sec, dcase_segment_frames, dcase_fps):

    relative_x = source_position[0] - mic_center_position[0]
    relative_y = source_position[1] - mic_center_position[1]
    relative_z = source_position[2] - mic_center_position[2]
    xy = np.array([relative_x, relative_y])

    deg_per_frame = deg_per_sec / dcase_fps

    R = rotation_matrix(np.deg2rad(deg_per_frame))

    source_positions = [source_position]

    for _ in range(dcase_segment_frames - 1):
        xy = np.dot(R, xy)
        source_positions.append([
            xy[0] + mic_center_position[0],
            xy[1] + mic_center_position[1],
            relative_z + mic_center_position[2],
        ])
        # source_positions.append(np.concatenate((xy, [relative_z])))

    source_positions = np.stack(source_positions, axis=0)

    # source_positions = extend_dcase_frames_to_nesd_frames(
    #     x=source_positions, 
    #     dcase_fps=dcase_fps,
    #     nesd_fps=nesd_fps,
    #     dcase_segment_frames=nesd_segment_frames,
    # )

    # plt.scatter(source_positions[:, 0], source_positions[:, 1], s=10, c='b')
    # plt.xlim(-4, 4)
    # plt.ylim(-4, 4)
    # plt.savefig('_zz.pdf')
    # from IPython import embed; embed(using=False); os._exit(0)

    return source_positions



def rotation_matrix(theta):
    R = np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)],
        ])

    return R