import math
import numpy as np
import pyroomacoustics as pra
import librosa
from room import Room
from nesd.utils import norm, fractional_delay_filter, FractionalDelay, DirectionSampler, sph2cart, get_cos, Agent, repeat_to_length, cart2sph, Rotator3D, sample_agent_look_direction
import time
from pathlib import Path
import yaml


class ImageSourceSimulator:

    def __init__(self, audios_dir, expand_frames=None, simulator_configs=None): 

        self.ndim = 3
        self.margin = 0.2

        # self.sample_rate = 16000
        self.sample_rate = 24000
        self.segment_samples = self.sample_rate * 2
        self.speed_of_sound = 343
        self.max_order = simulator_configs["image_source_max_order"]
        self.half_angle = math.atan2(0.1, 1)

        self.exclude_raidus = 0.5

        min_sources_num = simulator_configs["min_sources_num"]
        max_sources_num = simulator_configs["max_sources_num"]
        self.sources_num = np.random.randint(min_sources_num, max_sources_num + 1)
        # self.sources_num = 2

        self.expand_frames = expand_frames

        self.positive_rays = simulator_configs["positive_rays"]
        self.total_rays = simulator_configs["total_rays"]

        # audios_dir = "./resources"
        # audios_dir = "/home/qiuqiangkong/workspaces/nesd2/audios/vctk_2s_segments/train"
        self.audio_paths = list(Path(audios_dir).glob("*.wav"))

        debug = False

        self.mics_position_type = simulator_configs["mics_position_type"]
        agent_position_type = simulator_configs["agent_position_type"]
        # agent_in_center_of_mics = True

        if debug:

            self.length = 5
            self.width = 4
            self.height = 3

            x0, fs = librosa.load(path="./resources/p226_001.wav", sr=self.sample_rate, mono=True)
            x1, fs = librosa.load(path="./resources/p232_006.wav", sr=self.sample_rate, mono=True)
            a = 0.5

            x0 = repeat_to_length(audio=x0, segment_samples=segment_samples)
            x1 = repeat_to_length(audio=x1, segment_samples=segment_samples)

            self.source_dict = {
                0: a * x0, 
                1: a * x1,
            }

            self.sources_pos = [
                [1, 1, 1],
                [2, 2, 2],
            ]

            self.mic_pos = np.array([0.2, 0.3, 0.4])

            self.agent_pos = np.array([0.2, 0.3, 0.4])

        else:

            self.min_length = simulator_configs["room_min_length"]
            self.max_length = simulator_configs["room_max_length"]
            self.min_width = simulator_configs["room_min_width"]
            self.max_width = simulator_configs["room_max_width"]
            self.min_height = simulator_configs["room_min_height"]
            self.max_height = simulator_configs["room_max_height"]
            # self.max_length = 8
            # self.min_width = 4
            # self.max_width = 8
            # self.min_height = 2
            # self.max_height = 4

            # t1 = time.time()
            self.length, self.width, self.height = self.sample_shoebox_room()
            # print("a1", time.time() - t1)

            # t1 = time.time()
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

            self.mic_positions = self.sample_mic_positions(
                exclude_positions=None, 
                exclude_raidus=None,
            )

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

        # Create images
        # t1 = time.time()
        self.image_meta_list = self.create_images()
        # E.g., [
        #     {"source_index": 0, "order": 0, pos: [4.5, 3.4, 2.3]},
        #     ...
        # ]
        # print("a6", time.time() - t1)

        # Simulate microphone signals
        # t1 = time.time()
        self.mic_signals = self.simulate_microphone_signals(self.mic_positions)
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
        self.agent_look_direction_has_source = np.stack([agent.waveform for agent in self.agents], axis=0)
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

    def sample_mic_positions(self, exclude_positions, exclude_raidus):

        if self.mics_position_type == "random":
            center_pos = self.sample_position_in_room(exclude_positions, exclude_raidus)

        elif self.mics_position_type == "center_of_room":
            center_pos = np.array([self.length / 2, self.width / 2, self.height / 2])

        else:
            raise NotImplementedError

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

        elif mic_array_type == "mutli_array":
            center_pos = np.array([self.length / 2, self.width / 2, self.width / 2])

        #todo
        return mics_pos

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
        positive_image_meta_list = self.get_image_meta_by_order(self.image_meta_list, orders=[0])

        if len(positive_image_meta_list) > self.positive_rays:
            positive_image_meta_list = np.random.choice(positive_image_meta_list, size=positive_num, replace=False)

        for image_meta in positive_image_meta_list:
            
            source_index = image_meta["source_index"]

            agent_to_image = image_meta["position"] - self.agent_position

            agent_look_direction = sample_agent_look_direction(
                agent_to_src=agent_to_image, 
                half_angle=self.half_angle, 
            )

            distance = norm(agent_to_image)
            delayed_samples = distance / self.speed_of_sound * self.sample_rate
            decay_factor = 1 / distance
            angle_factor = 1.

            h = decay_factor * angle_factor * fractional_delay_filter(delayed_samples)

            y = self.convolve_source_filter(x=self.sources[source_index], h=h)

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
                positive_image_meta_list=self.get_image_meta_by_order(self.image_meta_list, orders=[0]),
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

    def get_image_meta_by_order(self, image_meta_list, orders):

        new_image_meta_list = []

        for image_meta in image_meta_list:
            if image_meta["order"] in orders:
                new_image_meta_list.append(image_meta)

        return new_image_meta_list

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

        for image_meta in positive_image_meta_list:

            agent_to_image = image_meta["position"] - agent_position

            angle_between_agent_and_src = np.arccos(get_cos(
                agent_look_direction, agent_to_image))

            if angle_between_agent_and_src < self.half_angle:
                return False

        return True


def expand_frame_dim(x, frames_num):
    x = np.expand_dims(x, axis=-2)
    x = np.repeat(a=x, repeats=frames_num, axis=-2)
    return x


def add():

    pass


if __name__ == '__main__':

    add()