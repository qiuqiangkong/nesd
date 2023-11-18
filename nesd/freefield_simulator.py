import lightning.pytorch as pl
from typing import List, Dict, NoReturn, Callable, Union, Optional
import torch
from torch.utils.data import DataLoader
import random
import soundfile
import numpy as np
import math
import numpy as np
import pyroomacoustics as pra
import librosa
import torchaudio
from room import Room
from nesd.utils import norm, fractional_delay_filter, FractionalDelay, DirectionSampler, sph2cart, get_cos, Agent, repeat_to_length, cart2sph, Rotator3D, sample_agent_look_direction, fractional_delay, calculate_microphone_gain
import time
from pathlib import Path
import yaml


class DatasetFreefield:
    def __init__(self, audios_dir, expand_frames=None, simulator_configs=None, lowpass_freq=None):
        self.audios_dir = audios_dir
        self.expand_frames = expand_frames
        self.simulator_configs = simulator_configs

        self.simulator = FreefieldSimulator(
            audios_dir=self.audios_dir, 
            expand_frames=self.expand_frames,
            simulator_configs=self.simulator_configs,
            lowpass_freq=lowpass_freq
        )

    def __getitem__(self, meta):
        
        data = self.simulator.sample()

        return data


class FreefieldSimulator:

    def __init__(self, audios_dir, expand_frames=None, simulator_configs=None, lowpass_freq=None): 

        self.expand_frames = expand_frames
        self.simulator_configs = simulator_configs

        self.sample_rate = 24000
        self.segment_samples = self.sample_rate * 2
        self.speed_of_sound = 343
        self.half_angle = math.atan2(0.1, 1)
        self.exclude_raidus = 0.5

        self.audio_paths = list(Path(audios_dir).glob("*.wav"))
        self.mics_yaml = simulator_configs["mics_yaml"]

        self.lowpass_freq = lowpass_freq

    def sample(self):

        simulator_configs = self.simulator_configs

        self.positive_rays = simulator_configs["positive_rays"]
        self.total_rays = simulator_configs["total_rays"]

        min_sources_num = simulator_configs["min_sources_num"]
        max_sources_num = simulator_configs["max_sources_num"]
        self.sources_num = np.random.randint(min_sources_num, max_sources_num + 1)

        self.mics_position_type = simulator_configs["mics_position_type"]
        agent_position_type = simulator_configs["agent_position_type"]
        sources_position_type = simulator_configs["sources_position_type"]

        debug = False

        if debug:
            pass
        else:

            self.length = 8
            self.width = 8
            self.height = 4

            # Sample mic
            self.mic_positions, _mic_look_directions, mic_directivities = self.sample_mic_positions(
                exclude_positions=None, 
                exclude_raidus=None,
            )

            # Sample source
            self.sources = self.sample_sources()

            if self.lowpass_freq is not None:

                # for source in self.sources:
                for i in range(len(self.sources)):
                    self.sources[i] = torchaudio.functional.lowpass_biquad(
                        waveform=torch.Tensor(self.sources[i]),
                        sample_rate=self.sample_rate,
                        cutoff_freq=500,
                    ).data.cpu().numpy()

            # Sample source positions
            if sources_position_type == "unit_sphere":
                mics_center_pos = np.mean(self.mic_positions, axis=0)
                self.source_positions = self.sample_source_positions_on_unit_sphere(mics_center_pos)

            else:
                self.source_positions = self.sample_source_positions(
                    exclude_positions=self.mic_positions, 
                    exclude_raidus=self.exclude_raidus
                )

            mics_num = len(self.mic_positions)

            self.mic_look_directions = np.ones((mics_num, 3))

            if agent_position_type == "center_of_mics":
                self.agent_position = np.mean(self.mic_positions, axis=0)

            elif agent_position_type == "random":
                self.agent_position = self.sample_position_in_room(
                    exclude_positions=self.source_positions, 
                    exclude_raidus=self.exclude_raidus,
                )
            else:
                raise NotImplementedError

        self.mic_signals = []

        # TODO, mic signals
        for mic_position, _mic_look_direction, mic_directivity in zip(self.mic_positions, _mic_look_directions, mic_directivities):

            mic_signal = 0

            for source, source_position in zip(self.sources, self.source_positions):
                mic_to_src = source_position - mic_position
                delayed_seconds = norm(mic_to_src) / self.speed_of_sound
                delayed_samples = self.sample_rate * delayed_seconds

                cos = get_cos(_mic_look_direction, mic_to_src)
                gain = calculate_microphone_gain(cos=cos, directivity=mic_directivity)

                y = fractional_delay(x=source, delayed_samples=delayed_samples)
                y *= gain

                mic_signal += y

            self.mic_signals.append(mic_signal)

        self.mic_signals = np.stack(self.mic_signals, axis=0)
        # soundfile.write(file="_zz.wav", data=self.mic_signals.T, samplerate=self.sample_rate)

        # Simluate agent signals
        self.agents = self.sample_and_simulate_agent_signals()
        
        self.mic_look_directions = np.stack(self.mic_look_directions, axis=0)
        self.mic_positions = np.stack(self.mic_positions, axis=0)
        self.mic_signals = np.stack(self.mic_signals, axis=0)

        self.agent_positions = np.stack([agent.position for agent in self.agents], axis=0)
        self.agent_look_directions = np.stack([agent.look_direction for agent in self.agents], axis=0)
        self.agent_look_directions_has_source = np.stack([agent.look_direction_has_source for agent in self.agents], axis=0)
        self.agent_waveforms = np.stack([agent.waveform for agent in self.agents], axis=0)
        self.agent_ray_types = np.stack([agent.ray_type for agent in self.agents], axis=0)

        if self.expand_frames:
            self.mic_positions = expand_frame_dim(self.mic_positions, self.expand_frames)
            self.mic_look_directions = expand_frame_dim(self.mic_look_directions, self.expand_frames)
            self.agent_positions = expand_frame_dim(self.agent_positions, self.expand_frames)
            self.agent_look_directions = expand_frame_dim(self.agent_look_directions, self.expand_frames)

            for i in range(len(self.source_positions)):
                self.source_positions[i] = expand_frame_dim(self.source_positions[i], self.expand_frames)

        data = {
            "room_length": self.length,
            "room_width": self.width,
            "room_height": self.height,
            "source_positions": self.source_positions,
            "source_signals": self.sources,
            "mic_positions": self.mic_positions,
            "mic_look_directions": self.mic_look_directions,
            "mic_signals": self.mic_signals,
            "agent_positions": self.agent_positions,
            "agent_look_directions": self.agent_look_directions,
            "agent_signals": self.agent_waveforms,
            "agent_look_directions_has_source": self.agent_look_directions_has_source,
            "agent_ray_types": self.agent_ray_types
        }

        return data

    def sample_shoebox_room(self):

        length = np.random.uniform(low=self.min_length, high=self.max_length)
        width = np.random.uniform(low=self.min_width, high=self.max_width)
        height = np.random.uniform(low=self.min_height, high=self.max_height)

        return length, width, height

    def sample_sources(self):

        audio_paths = np.random.choice(self.audio_paths, size=self.sources_num, replace=False)

        sources = []

        for i, audio_path in enumerate(audio_paths):
            audio, _ = librosa.load(path=audio_path, sr=self.sample_rate, mono=True)
            audio = repeat_to_length(audio=audio, segment_samples=self.segment_samples)

            sources.append(audio)

        return sources

    def sample_source_positions(self, exclude_positions=None, exclude_raidus=None):

        source_positions = []

        for _ in range(self.sources_num):
            position = self.sample_position_in_room(exclude_positions, exclude_raidus)
            source_positions.append(position)

        return source_positions

    def sample_source_positions_on_unit_sphere(self, mics_center_pos):

        _direction_sampler = DirectionSampler(
            low_colatitude=0, 
            high_colatitude=math.pi, 
            sample_on_sphere_uniformly=False, 
        )

        source_positions = []

        for _ in range(self.sources_num):
            
            agent_look_azimuth, agent_look_colatitude = _direction_sampler.sample()

            agent_look_direction = np.array(sph2cart(
                r=1., 
                azimuth=agent_look_azimuth, 
                colatitude=agent_look_colatitude
            ))

            source_position = mics_center_pos + agent_look_direction
            source_positions.append(source_position)

        return source_positions

    def sample_mic_positions(self, exclude_positions, exclude_raidus):

        if self.mics_position_type == "random":
            center_pos = self.sample_position_in_room(exclude_positions, exclude_raidus)

        elif self.mics_position_type == "center_of_room":
            center_pos = np.array([self.length / 2, self.width / 2, self.height / 2])

        else:
            raise NotImplementedError

        mic_look_directions = []
        directivities = []

        # mic_array_type = "linear"
        mic_array_type = "eigenmike"

        if mic_array_type == "single":
            mics_pos = [center_pos]

        elif mic_array_type == "linear":
            mics_pos = [center_pos, center_pos + 0.04, center_pos + 0.08, center_pos + 0.12]

        elif mic_array_type == "eigenmike":

            mics_pos = []

            # mics_yaml = "./nesd/microphones/eigenmike.yaml"

            with open(self.mics_yaml, 'r') as f:
                mics_meta = yaml.load(f, Loader=yaml.FullLoader)

            for mic_meta in mics_meta:

                relative_mic_pos = np.array(sph2cart(
                    r=mic_meta['radius'], 
                    azimuth=mic_meta['azimuth'], 
                    colatitude=mic_meta['colatitude']
                ))

                mic_pos = center_pos + relative_mic_pos

                mics_pos.append(mic_pos)
                mic_look_directions.append(relative_mic_pos)
                directivities.append(mic_meta["directivity"])

        elif mic_array_type == "mutli_array":
            center_pos = np.array([self.length / 2, self.width / 2, self.height / 2])

        #todo
        return mics_pos, mic_look_directions, directivities

    def sample_position_in_room(self, exclude_positions=None, exclude_raidus=None):

        pos = np.zeros(self.ndim)

        while True:

            for dim, edge in enumerate((self.length, self.width, self.height)):
                pos[dim] = np.random.uniform(low=self.margin, high=edge - self.margin)

            if exclude_positions is None:
                break

            elif self.pass_position_check(pos, exclude_positions, exclude_raidus):
                break

            else:
                # Resample position if not pass the check.
                pass

        return pos

    def pass_position_check(self, pos, exclude_positions, exclude_raidus):
        # To check

        for exclude_position in exclude_positions:

            if norm(pos - exclude_position) < exclude_raidus:
                return False

        return True


    def create_images(self):

        image_meta_list = []

        # Create images
        for source_index in range(self.sources_num):

            corners = np.array([
                [0, 0], 
                [0, self.width], 
                [self.length, self.width], 
                [self.length, 0]
            ]).T
            # shape: (2, 4)

            room = pra.Room.from_corners(
                corners=corners,
                max_order=self.max_order,
            )

            room.extrude(height=self.height)

            source_position = self.source_positions[source_index]
            room.add_source(source_position)

            room.add_microphone([0.1, 0.1, 0.1])    # dummy

            room.image_source_model()

            images = room.sources[0].images.T
            # (images_num, ndim)

            orders = room.sources[0].orders

            unique_images = []

            for i, image in enumerate(images):

                image = image.tolist()

                if image not in unique_images:
                    
                    unique_images.append(image)

                    meta = {
                        "source_index": source_index,
                        "order": room.sources[0].orders[i],
                        "position": np.array(image),
                    }
                    image_meta_list.append(meta)
        
        return image_meta_list

    def simulate_microphone_signals(self, mics_pos):

        mic_signals = []

        for mic_pos in self.mic_positions:
            mic_signal = self.simulate_microphone_signal(mic_pos)
            mic_signals.append(mic_signal)

        return mic_signals

    def simulate_microphone_signal(self, mic_pos):

        hs_dict = {source_index: [] for source_index in range(self.sources_num)}

        for image_meta in self.image_meta_list:

            source_index = image_meta["source_index"]

            direction = image_meta["position"] - mic_pos
            distance = norm(direction)
            delayed_samples = distance / self.speed_of_sound * self.sample_rate

            decay_factor = 1 / distance

            angle_factor = 1.

            normalized_direction = direction / distance

            h = decay_factor * angle_factor * fractional_delay_filter(delayed_samples)
            hs_dict[source_index].append(h)

        y_dict = {}

        for source_index in range(self.sources_num):

            hs = hs_dict[source_index]
            source = self.sources[source_index]

            max_filter_len = max([len(h) for h in hs])
            
            sum_h = np.zeros(max_filter_len)
            for h in hs:
                sum_h[0 : len(h)] += h

            y = self.convolve_source_filter(x=source, h=sum_h)

            y_dict[source_index] = y

            # soundfile.write(file="_zz{}.wav".format(source_index), data=y, samplerate=16000)
            
        y_total = np.sum([y for y in y_dict.values()], axis=0)

        return y_total

    def convolve_source_filter(self, x, h):
        return np.convolve(x, h, mode='full')[0 : len(x)]

    def sample_and_simulate_agent_signals(self):

        agents = []

        _direction_sampler = DirectionSampler(
            low_colatitude=0, 
            high_colatitude=math.pi, 
            sample_on_sphere_uniformly=False, 
        )

        # positive
        # positive_image_meta_list = self.get_image_meta_by_order(self.image_meta_list, orders=[0])
        # positive_image_meta_list = self.source_positions

        # if len(positive_image_meta_list) > self.positive_rays:
        #     positive_image_meta_list = np.random.choice(positive_image_meta_list, size=positive_num, replace=False)

        # for image_meta in positive_image_meta_list:
        for source, source_position in zip(self.sources, self.source_positions):
            
            # source_index = image_meta["source_index"]

            agent_to_image = source_position - self.agent_position

            agent_look_direction = sample_agent_look_direction(
                agent_to_src=agent_to_image, 
                half_angle=self.half_angle, 
            )

            distance = norm(agent_to_image)
            delayed_samples = distance / self.speed_of_sound * self.sample_rate
            decay_factor = 1 / distance
            angle_factor = 1.

            h = decay_factor * angle_factor * fractional_delay_filter(delayed_samples)

            y = self.convolve_source_filter(x=source, h=h)

            agent = Agent(
                position=self.agent_position, 
                look_direction=agent_look_direction, 
                waveform=y,
                look_direction_has_source=np.ones(self.expand_frames),
                ray_type="positive",
            )

            agents.append(agent)
            
        # middle
        # todo
        
        # negative
        # t1 = time.time()
        while len(agents) < self.total_rays:
            
            agent_look_direction = self.sample_negative_direction(
                agent_position=self.agent_position,
                direction_sampler=_direction_sampler,
                positive_image_meta_list=self.source_positions,
                half_angle=self.half_angle,
            )

            agent = Agent(
                position=self.agent_position, 
                look_direction=agent_look_direction, 
                waveform=np.zeros(self.segment_samples),
                look_direction_has_source=np.zeros(self.expand_frames),
                ray_type="negative",
            )
            agents.append(agent)

        # print("a4", time.time() - t1)
        return agents
        # from IPython import embed; embed(using=False); os._exit(0)

    def sample_negative_direction(self, agent_position, direction_sampler, positive_image_meta_list, half_angle):

        while True:
            
            agent_look_azimuth, agent_look_colatitude = direction_sampler.sample()

            agent_look_direction = np.array(sph2cart(
                r=1., 
                azimuth=agent_look_azimuth, 
                colatitude=agent_look_colatitude
            ))

            if self.pass_direction_check(agent_position, agent_look_direction, positive_image_meta_list):
                break

        return agent_look_direction

    def pass_direction_check(self, agent_position, agent_look_direction, positive_image_meta_list):

        for source_position in self.source_positions:

            agent_to_image = source_position - agent_position

            angle_between_agent_and_src = np.arccos(get_cos(
                agent_look_direction, agent_to_image))

            if angle_between_agent_and_src < self.half_angle:
                return False

        return True


'''
class FreefieldSimulator:

    def __init__(self, audios_dir, expand_frames=None, simulator_configs=None): 

        t1 = time.time()
        # self.sample_rate = 16000
        self.sample_rate = 24000
        self.segment_samples = self.sample_rate * 2
        self.speed_of_sound = 343
        self.half_angle = math.atan2(0.1, 1)

        self.exclude_raidus = 0.5

        min_sources_num = simulator_configs["min_sources_num"]
        max_sources_num = simulator_configs["max_sources_num"]
        self.sources_num = np.random.randint(min_sources_num, max_sources_num + 1)
        # self.sources_num = 2

        self.expand_frames = expand_frames

        self.positive_rays = simulator_configs["positive_rays"]
        self.total_rays = simulator_configs["total_rays"]

        t1 = time.time()
        self.audio_paths = list(Path(audios_dir).glob("*.wav"))
        print("x1", time.time() - t1)

        debug = False

        self.mics_position_type = simulator_configs["mics_position_type"]
        agent_position_type = simulator_configs["agent_position_type"]
        # agent_in_center_of_mics = True
        sources_position_type = simulator_configs["sources_position_type"]
        print("a1", time.time() - t1)

        if debug:
            pass
        else:

            t1 = time.time()
            self.sources = self.sample_sources()
            # print("a2", time.time() - t1)

            # t1 = time.time()
            # self.source_positions = self.sample_source_positions()
            # (sources_num, ndim)
            # print("a3", time.time() - t1)

            # t1 = time.time()
            # if mics_position_type == "random":
            # self.mic_positions = self.sample_mic_positions(
            #     exclude_positions=self.source_positions, 
            #     exclude_raidus=self.exclude_raidus,
            # )
            self.length = 8
            self.width = 8
            self.height = 4

            self.mic_positions, _mic_look_directions = self.sample_mic_positions(
                exclude_positions=None, 
                exclude_raidus=None,
            )

            if sources_position_type == "unit_sphere":
                mics_center_pos = np.mean(self.mic_positions, axis=0)
                self.source_positions = self.sample_source_positions_on_unit_sphere(mics_center_pos)
                # from IPython import embed; embed(using=False); os._exit(0)

            else:
                self.source_positions = self.sample_source_positions(
                    exclude_positions=self.mic_positions, 
                    exclude_raidus=self.exclude_raidus
                )
            
            # (mics_num, ndim)
            # print("a4", time.time() - t1)


            mics_num = len(self.mic_positions)

            self.mic_look_directions = np.ones((mics_num, 3))

            # t1 = time.time()
            if agent_position_type == "center_of_mics":
                self.agent_position = np.mean(self.mic_positions, axis=0)

            elif agent_position_type == "random":
                self.agent_position = self.sample_position_in_room(
                    exclude_positions=self.source_positions, 
                    exclude_raidus=self.exclude_raidus,
                )
            else:
                raise NotImplementedError
            # (ndim,)
            # print("a5", time.time() - t1)
            print("b1", time.time() - t1)

        self.mic_signals = []

        t1 = time.time()

        # TODO, mic signals
        for mic_position, _mic_look_direction in zip(self.mic_positions, _mic_look_directions):

            mic_signal = 0

            # total = 0
            for source, source_position in zip(self.sources, self.source_positions):
                mic_to_src = source_position - mic_position
                delayed_seconds = norm(mic_to_src) / self.speed_of_sound
                delayed_samples = self.sample_rate * delayed_seconds

                
                # cos = get_cos(_mic_look_direction, mic_to_src)
                # gain = calculate_microphone_gain(cos=cos, directivity="cardioid")
                gain = 1

                y = fractional_delay(x=source, delayed_samples=delayed_samples)
                y *= gain

                mic_signal += y
                # from IPython import embed; embed(using=False); os._exit(0)

            self.mic_signals.append(mic_signal)

        self.mic_signals = np.stack(self.mic_signals, axis=0)
        # soundfile.write(file="_zz.wav", data=self.mic_signals.T, samplerate=self.sample_rate)
        print("c1", time.time() - t1)

        # from IPython import embed; embed(using=False); os._exit(0)

        # Simulate microphone signals
        t1 = time.time()
        # self.mic_signals = self.simulate_microphone_signals(self.mic_positions)
        # print("a7", time.time() - t1)

        # Simluate agent signals
        # t1 = time.time()
        self.agents = self.sample_and_simulate_agent_signals()
        # print("a8", time.time() - t1)

        self.mic_look_directions = np.stack(self.mic_look_directions, axis=0)
        self.mic_positions = np.stack(self.mic_positions, axis=0)
        self.mic_signals = np.stack(self.mic_signals, axis=0)

        self.agent_positions = np.stack([agent.position for agent in self.agents], axis=0)
        self.agent_look_directions = np.stack([agent.look_direction for agent in self.agents], axis=0)
        self.agent_look_directions_has_source = np.stack([agent.look_direction_has_source for agent in self.agents], axis=0)
        # from IPython import embed; embed(using=False); os._exit(0)
        self.agent_waveforms = np.stack([agent.waveform for agent in self.agents], axis=0)
        # self.agent_look_direction_has_source = np.stack([agent.look_direction_has_source for agent in self.agents], axis=0)
        self.agent_ray_types = np.stack([agent.ray_type for agent in self.agents], axis=0)

        if expand_frames:
            self.mic_positions = expand_frame_dim(self.mic_positions, expand_frames)
            self.mic_look_directions = expand_frame_dim(self.mic_look_directions, expand_frames)
            self.agent_positions = expand_frame_dim(self.agent_positions, expand_frames)
            self.agent_look_directions = expand_frame_dim(self.agent_look_directions, expand_frames)

            for i in range(len(self.source_positions)):
                self.source_positions[i] = expand_frame_dim(self.source_positions[i], expand_frames)
            
            # self.agent_look_directions_has_source = expand_frame_dim(self.agent_look_directions_has_source, expand_frames)

        # self.agent_positions = np.stack(self.agent_positions)
        # from IPython import embed; embed(using=False); os._exit(0)
        print("d1", time.time() - t1)

    def sample_shoebox_room(self):

        length = np.random.uniform(low=self.min_length, high=self.max_length)
        width = np.random.uniform(low=self.min_width, high=self.max_width)
        height = np.random.uniform(low=self.min_height, high=self.max_height)

        return length, width, height

    def sample_sources(self):

        audio_paths = np.random.choice(self.audio_paths, size=self.sources_num, replace=False)

        sources = []

        for i, audio_path in enumerate(audio_paths):
            audio, _ = librosa.load(path=audio_path, sr=self.sample_rate, mono=True)
            audio = repeat_to_length(audio=audio, segment_samples=self.segment_samples)

            sources.append(audio)

        return sources

    def sample_source_positions(self, exclude_positions=None, exclude_raidus=None):

        source_positions = []

        for _ in range(self.sources_num):
            position = self.sample_position_in_room(exclude_positions, exclude_raidus)
            source_positions.append(position)

        return source_positions

    def sample_source_positions_on_unit_sphere(self, mics_center_pos):

        _direction_sampler = DirectionSampler(
            low_colatitude=0, 
            high_colatitude=math.pi, 
            sample_on_sphere_uniformly=False, 
        )

        source_positions = []

        for _ in range(self.sources_num):
            
            agent_look_azimuth, agent_look_colatitude = _direction_sampler.sample()

            agent_look_direction = np.array(sph2cart(
                r=1., 
                azimuth=agent_look_azimuth, 
                colatitude=agent_look_colatitude
            ))

            source_position = mics_center_pos + agent_look_direction
            source_positions.append(source_position)

        return source_positions

    def sample_mic_positions(self, exclude_positions, exclude_raidus):

        if self.mics_position_type == "random":
            center_pos = self.sample_position_in_room(exclude_positions, exclude_raidus)

        elif self.mics_position_type == "center_of_room":
            center_pos = np.array([self.length / 2, self.width / 2, self.height / 2])

        else:
            raise NotImplementedError

        mic_look_directions = []

        # mic_array_type = "linear"
        mic_array_type = "eigenmike"

        if mic_array_type == "single":
            mics_pos = [center_pos]

        elif mic_array_type == "linear":
            mics_pos = [center_pos, center_pos + 0.04, center_pos + 0.08, center_pos + 0.12]

        elif mic_array_type == "eigenmike":

            mics_pos = []

            mics_yaml = "./nesd/microphones/eigenmike.yaml"

            with open(mics_yaml, 'r') as f:
                mics_meta = yaml.load(f, Loader=yaml.FullLoader)

            for mic_meta in mics_meta:

                relative_mic_pos = np.array(sph2cart(
                    r=mic_meta['radius'], 
                    azimuth=mic_meta['azimuth'], 
                    colatitude=mic_meta['colatitude']
                ))

                mic_pos = center_pos + relative_mic_pos

                mics_pos.append(mic_pos)
                mic_look_directions.append(relative_mic_pos)

        elif mic_array_type == "mutli_array":
            center_pos = np.array([self.length / 2, self.width / 2, self.height / 2])

        #todo
        return mics_pos, mic_look_directions

    def sample_position_in_room(self, exclude_positions=None, exclude_raidus=None):

        pos = np.zeros(self.ndim)

        while True:

            for dim, edge in enumerate((self.length, self.width, self.height)):
                pos[dim] = np.random.uniform(low=self.margin, high=edge - self.margin)

            if exclude_positions is None:
                break

            elif self.pass_position_check(pos, exclude_positions, exclude_raidus):
                break

            else:
                # Resample position if not pass the check.
                pass

        return pos

    def pass_position_check(self, pos, exclude_positions, exclude_raidus):
        # To check

        for exclude_position in exclude_positions:

            if norm(pos - exclude_position) < exclude_raidus:
                return False

        return True


    def create_images(self):

        image_meta_list = []

        # Create images
        for source_index in range(self.sources_num):

            corners = np.array([
                [0, 0], 
                [0, self.width], 
                [self.length, self.width], 
                [self.length, 0]
            ]).T
            # shape: (2, 4)

            room = pra.Room.from_corners(
                corners=corners,
                max_order=self.max_order,
            )

            room.extrude(height=self.height)

            source_position = self.source_positions[source_index]
            room.add_source(source_position)

            room.add_microphone([0.1, 0.1, 0.1])    # dummy

            room.image_source_model()

            images = room.sources[0].images.T
            # (images_num, ndim)

            orders = room.sources[0].orders

            unique_images = []

            for i, image in enumerate(images):

                image = image.tolist()

                if image not in unique_images:
                    
                    unique_images.append(image)

                    meta = {
                        "source_index": source_index,
                        "order": room.sources[0].orders[i],
                        "position": np.array(image),
                    }
                    image_meta_list.append(meta)
        
        return image_meta_list

    def simulate_microphone_signals(self, mics_pos):

        mic_signals = []

        for mic_pos in self.mic_positions:
            mic_signal = self.simulate_microphone_signal(mic_pos)
            mic_signals.append(mic_signal)

        return mic_signals

    def simulate_microphone_signal(self, mic_pos):

        hs_dict = {source_index: [] for source_index in range(self.sources_num)}

        for image_meta in self.image_meta_list:

            source_index = image_meta["source_index"]

            direction = image_meta["position"] - mic_pos
            distance = norm(direction)
            delayed_samples = distance / self.speed_of_sound * self.sample_rate

            decay_factor = 1 / distance

            angle_factor = 1.

            normalized_direction = direction / distance

            h = decay_factor * angle_factor * fractional_delay_filter(delayed_samples)
            hs_dict[source_index].append(h)

        y_dict = {}

        for source_index in range(self.sources_num):

            hs = hs_dict[source_index]
            source = self.sources[source_index]

            max_filter_len = max([len(h) for h in hs])
            
            sum_h = np.zeros(max_filter_len)
            for h in hs:
                sum_h[0 : len(h)] += h

            y = self.convolve_source_filter(x=source, h=sum_h)

            y_dict[source_index] = y

            # soundfile.write(file="_zz{}.wav".format(source_index), data=y, samplerate=16000)
            
        y_total = np.sum([y for y in y_dict.values()], axis=0)

        return y_total

    def convolve_source_filter(self, x, h):
        return np.convolve(x, h, mode='full')[0 : len(x)]

    def sample_and_simulate_agent_signals(self):

        agents = []

        _direction_sampler = DirectionSampler(
            low_colatitude=0, 
            high_colatitude=math.pi, 
            sample_on_sphere_uniformly=False, 
        )

        # positive
        # positive_image_meta_list = self.get_image_meta_by_order(self.image_meta_list, orders=[0])
        # positive_image_meta_list = self.source_positions

        # if len(positive_image_meta_list) > self.positive_rays:
        #     positive_image_meta_list = np.random.choice(positive_image_meta_list, size=positive_num, replace=False)

        # for image_meta in positive_image_meta_list:
        for source, source_position in zip(self.sources, self.source_positions):
            
            # source_index = image_meta["source_index"]

            agent_to_image = source_position - self.agent_position

            agent_look_direction = sample_agent_look_direction(
                agent_to_src=agent_to_image, 
                half_angle=self.half_angle, 
            )

            distance = norm(agent_to_image)
            delayed_samples = distance / self.speed_of_sound * self.sample_rate
            decay_factor = 1 / distance
            angle_factor = 1.

            h = decay_factor * angle_factor * fractional_delay_filter(delayed_samples)

            y = self.convolve_source_filter(x=source, h=h)

            agent = Agent(
                position=self.agent_position, 
                look_direction=agent_look_direction, 
                waveform=y,
                look_direction_has_source=np.ones(self.expand_frames),
                ray_type="positive",
            )

            agents.append(agent)
            
        # middle
        # todo
        
        # negative
        # t1 = time.time()
        while len(agents) < self.total_rays:
            
            agent_look_direction = self.sample_negative_direction(
                agent_position=self.agent_position,
                direction_sampler=_direction_sampler,
                positive_image_meta_list=self.source_positions,
                half_angle=self.half_angle,
            )

            agent = Agent(
                position=self.agent_position, 
                look_direction=agent_look_direction, 
                waveform=np.zeros(self.segment_samples),
                look_direction_has_source=np.zeros(self.expand_frames),
                ray_type="negative",
            )
            agents.append(agent)

        # print("a4", time.time() - t1)
        return agents
        # from IPython import embed; embed(using=False); os._exit(0)

    def sample_negative_direction(self, agent_position, direction_sampler, positive_image_meta_list, half_angle):

        while True:
            
            agent_look_azimuth, agent_look_colatitude = direction_sampler.sample()

            agent_look_direction = np.array(sph2cart(
                r=1., 
                azimuth=agent_look_azimuth, 
                colatitude=agent_look_colatitude
            ))

            if self.pass_direction_check(agent_position, agent_look_direction, positive_image_meta_list):
                break

        return agent_look_direction

    def pass_direction_check(self, agent_position, agent_look_direction, positive_image_meta_list):

        for source_position in self.source_positions:

            agent_to_image = source_position - agent_position

            angle_between_agent_and_src = np.arccos(get_cos(
                agent_look_direction, agent_to_image))

            if angle_between_agent_and_src < self.half_angle:
                return False

        return True
'''

def expand_frame_dim(x, frames_num):
    x = np.expand_dims(x, axis=-2)
    x = np.repeat(a=x, repeats=frames_num, axis=-2)
    return x