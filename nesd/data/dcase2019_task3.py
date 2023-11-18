import os
import numpy as np
from pathlib import Path
import h5py
import soundfile
import yaml
import math

from nesd.utils import int16_to_float32, sph2cart, sample_agent_look_direction, Agent, DirectionSampler, get_cos
from nesd.image_source_simulator import expand_frame_dim


class DCASE2019Task3Dataset:
    # def __init__(self, audios_dir, expand_frames=None, simulator_configs=None):
    def __init__(self, hdf5s_dir):
        self.hdf5s_dir = hdf5s_dir
        self.hdf5_paths = sorted(Path(self.hdf5s_dir).glob("*.h5"))
        self.hdf5s_num = len(self.hdf5_paths)
        self.sample_rate = 24000
        self.segment_seconds = 2
        self.segment_samples = int(self.sample_rate * self.segment_seconds)
        self.frames_per_sec = 100
        self.segment_frames = int(self.segment_seconds * self.frames_per_sec)
        self.expand_frames = 201

        self.mics_position_type = "center_of_room"

        self.length = 8
        self.width = 8
        self.height = 4
        self.half_angle = math.atan2(0.1, 1)
        self.total_rays = 20

        # self.hdf5_paths = self.hdf5_paths[0:1]
        
    def __getitem__(self, meta):


        hdf5_path = np.random.choice(self.hdf5_paths)

        with h5py.File(hdf5_path, 'r') as hf:

            bgn_sec = np.random.uniform(low=0, high=hf.attrs["duration"] - self.segment_seconds - 1)
            bgn_sec = np.round(a=bgn_sec, decimals=2)
            end_sec = bgn_sec + self.segment_seconds
            bgn_sample = int(bgn_sec * self.sample_rate)
            end_sample = bgn_sample + self.segment_samples
            bgn_frame = int(bgn_sec * self.frames_per_sec)
            end_frame = bgn_frame + self.segment_frames

            mic_signals = int16_to_float32(hf["waveform"][:, bgn_sample : end_sample])

            # frame_indexes = hf["frame_index"][:]
            has_sources_array = hf["has_source"][:]
            class_indexes_array = hf["class_index"][:]
            # event_indexes = hf["event_index"][:]
            azimuths_array = hf["azimuth"][:]
            elevations_array = hf["elevation"][:]
            distances_array = hf["distance"][:]

            # soundfile.write(file="_zz.wav", data=segment.T, samplerate=self.sample_rate)

        # microphone
        mic_positions = self.sample_mic_positions(
            exclude_positions=None, 
            exclude_raidus=None,
        )
        mic_positions = expand_frame_dim(mic_positions, self.expand_frames)

        mics_num = mic_positions.shape[0]
        mic_look_directions = np.ones((mics_num, 3))
        mic_look_directions = expand_frame_dim(mic_look_directions, self.expand_frames)

        # has_sources
        max_sources_num = has_sources_array.shape[0]
        sources = []

        agent_position = np.array((self.length / 2, self.width / 2, self.height / 2))

        source_positions = []
        agents = []

        for source_id in range(max_sources_num):

            has_sources = has_sources_array[source_id, bgn_frame : end_frame + 1]
            class_indexes = class_indexes_array[source_id, bgn_frame : end_frame + 1]
            azimuths = np.deg2rad(azimuths_array[source_id, bgn_frame : end_frame + 1] % 360)
            colatitudes = np.deg2rad(90 - np.array(elevations_array[source_id, bgn_frame : end_frame + 1]))
            distances = distances_array[source_id, bgn_frame : end_frame + 1]

            sound_classes = []

            for i in range(len(has_sources)):
                if has_sources[i] == 1:
                    sound_classes.append(class_indexes[i])

            unique_sound_classes = list(set(sound_classes))

            for class_index in unique_sound_classes:

                active_indexes = np.where(class_indexes == class_index)[0]

                avg_azi = np.mean(azimuths[active_indexes])
                avg_col = np.mean(colatitudes[active_indexes])

                agent_to_source = np.array(sph2cart(
                    r=1., 
                    azimuth=avg_azi, 
                    colatitude=avg_col
                ))
                source_position = agent_position + agent_to_source
                source_positions.append(source_position)

                agent_look_direction = sample_agent_look_direction(
                    agent_to_src=agent_to_source, 
                    half_angle=self.half_angle, 
                )

                look_direction_has_source = np.zeros(self.expand_frames)
                look_direction_has_source[active_indexes] = 1
                
                agent = Agent(
                    position=agent_position, 
                    look_direction=agent_look_direction, 
                    waveform=np.zeros(self.segment_samples),
                    look_direction_has_source=look_direction_has_source,
                    ray_type="positive",
                )
                agents.append(agent)

            # soundfile.write(file="_zz.wav", data=mic_signals.T, samplerate=self.sample_rate) 
            # from IPython import embed; embed(using=False); os._exit(0)

        _direction_sampler = DirectionSampler(
            low_colatitude=0, 
            high_colatitude=math.pi, 
            sample_on_sphere_uniformly=False, 
        )

        while len(agents) < self.total_rays:
            
            agent_look_direction = self.sample_negative_direction(
                agent_position=agent_position,
                direction_sampler=_direction_sampler,
                positive_image_meta_list=source_positions,
                half_angle=self.half_angle,
            )

            agent = Agent(
                position=agent_position, 
                look_direction=agent_look_direction, 
                waveform=np.zeros(self.segment_samples),
                look_direction_has_source=np.zeros(self.expand_frames),
                ray_type="negative",
            )
            agents.append(agent)

        agent_positions = np.stack([agent.position for agent in agents], axis=0)
        agent_look_directions = np.stack([agent.look_direction for agent in agents], axis=0)

        agent_positions = expand_frame_dim(agent_positions, self.expand_frames)
        agent_look_directions = expand_frame_dim(agent_look_directions, self.expand_frames)
        agent_look_directions_has_source = np.stack([agent.look_direction_has_source for agent in agents], axis=0)
        agent_waveforms = np.stack([agent.waveform for agent in agents], axis=0)
        
        # from IPython import embed; embed(using=False); os._exit(0)

        data = {
            "source_positions": source_positions,
            "mic_positions": mic_positions,
            "mic_look_directions": mic_look_directions,
            "mic_signals": mic_signals,
            "agent_positions": agent_positions,
            "agent_look_directions": agent_look_directions,
            "agent_look_directions_has_source": agent_look_directions_has_source,
            # "agent_ray_types": agent_ray_types
        }

        return data

    def sample_mic_positions(self, exclude_positions, exclude_raidus):

        if self.mics_position_type == "random":
            center_pos = self.sample_position_in_room(exclude_positions, exclude_raidus)

        elif self.mics_position_type == "center_of_room":
            center_pos = np.array([self.length / 2, self.width / 2, self.height / 2])

        else:
            raise NotImplementedError

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

        for image_position in positive_image_meta_list:

            agent_to_image = image_position - agent_position

            angle_between_agent_and_src = np.arccos(get_cos(
                agent_look_direction, agent_to_image))

            if angle_between_agent_and_src < self.half_angle:
                return False

        return True