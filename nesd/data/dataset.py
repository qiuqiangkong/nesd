import random
from pathlib import Path
import time

import math
import librosa
import soundfile
import numpy as np
import yaml
import pickle
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

from nesd.utils import read_yaml, db_to_scale, sph2cart, normalize, random_direction, get_included_angle, random_positive_direction, triangle_function, random_negative_direction, random_positive_distance, random_negative_distance, expand_along_frame_axis, Agent, apply_lowpass_filter, is_collide
from nesd.data.engine import ImageSourceEngine
from nesd.constants import PAD


class Dataset:
    def __init__(self, simulator_configs, split):

        self.simulator_configs = simulator_configs

        # General
        self.speed_of_sound = simulator_configs["speed_of_sound"]
        self.sample_rate = simulator_configs["sample_rate"]
        self.segment_seconds = simulator_configs["segment_seconds"]
        self.frames_per_sec = simulator_configs["frames_per_sec"]
        
        self.frames_num = int(self.frames_per_sec * self.segment_seconds) + 1
        self.segment_samples = int(self.segment_seconds * self.sample_rate)

        # Environment
        self.min_room_length = simulator_configs["room_length"]["min"]
        self.max_room_length = simulator_configs["room_length"]["max"]
        self.min_room_width = simulator_configs["room_width"]["min"]
        self.max_room_width = simulator_configs["room_width"]["max"]
        self.min_room_height = simulator_configs["room_height"]["min"]
        self.max_room_height = simulator_configs["room_height"]["max"]
        self.room_margin = simulator_configs["room_margin"]
        if "collision_raidus" in simulator_configs.keys():
            self.collision_raidus = simulator_configs["collision_raidus"]
        else:
            self.collision_raidus = 0.5
        self.ndim = 3

        # Sources
        self.audios_dir = simulator_configs["{}_audios".format(split)]
        self.min_sources_num = simulator_configs["sources_num"]["min"]
        self.max_sources_num = simulator_configs["sources_num"]["max"]
        self.sample_rate = simulator_configs["sample_rate"]
        self.min_source_gain_db = simulator_configs["source_gain_db"]["min"]
        self.max_source_gain_db = simulator_configs["source_gain_db"]["max"]
        self.source_apprent_diameter_deg = simulator_configs["source_apprent_diameter_deg"]
        self.source_radius = simulator_configs["source_radius"]

        self.audio_paths = sorted(list(Path(self.audios_dir).glob("*.wav")))

        # Microphones
        self.mics_meta = read_yaml(simulator_configs["mics_yaml"])
        if "mic_cutoff_freq" in simulator_configs.keys():
            self.mic_cutoff_freq = simulator_configs["mic_cutoff_freq"]
        else:
            self.mic_cutoff_freq = None
        if "mic_positions_type" in simulator_configs.keys():
            self.mic_positions_type = simulator_configs["mic_positions_type"]
        else:
            self.mic_positions_type = None
        self.mic_spatial_irs_path = simulator_configs["mic_spatial_irs_path"]
        self.mic_noise_dir = simulator_configs["{}_mic_noise".format(split)]

        if self.mic_spatial_irs_path:
            self.mic_spatial_irs = pickle.load(open(self.mic_spatial_irs_path, "rb"))
        else:
            self.mic_spatial_irs = None
        self.mic_noise_paths = self.load_mic_noise_paths(self.mic_noise_dir)
        self.min_noise_gain_db = simulator_configs["noise_gain_db"]["min"]
        self.max_noise_gain_db = simulator_configs["noise_gain_db"]["max"]

        # Simulator
        self.image_source_order = simulator_configs["image_source_order"]

        # Agents
        self.agent_positions_type = simulator_configs["agent_positions_type"]
        self.agent_det_max_pos_rays = simulator_configs["agent_detection_rays"]["max_positive"]
        self.agent_det_total_rays = simulator_configs["agent_detection_rays"]["total"]
        self.agent_dist_max_pos_rays = simulator_configs["agent_distance_rays"]["max_positive"]
        self.agent_dist_total_rays = simulator_configs["agent_distance_rays"]["total"]
        self.agent_sep_max_pos_rays = simulator_configs["agent_sep_rays"]["max_positive"]
        self.agent_sep_total_rays = simulator_configs["agent_sep_rays"]["total"]

        # self.full_random = True

    def __getitem__(self, _):

        # ------ 1. Sample environment. ------
        environment = self.sample_environment()
        
        # ------ 2. Sample sources, mics, and agents positions. ------
        srcs_num = random.randint(a=self.min_sources_num, b=self.max_sources_num)
        # srcs_num = 2

        # Sample until source positions do not collide with mics and agent positions.
        while True:

            # Sample source positions.
            src_positions = self.sample_source_positions(
                sources_num=srcs_num,
                environment=environment,
            )
            # shape: (srcs_num, frames_num, ndim)
            # from IPython import embed; embed(using=False); os._exit(0)

            # Sample mic positions.
            mic_positions = self.sample_mic_positions(
                environment=environment,
                mics_meta=self.mics_meta
            )
            # (mics_num, frames_num, ndim)

            # Sample agent position. All agents have the same position.
            agent_position = self.sample_agent_positions(
                environment=environment,
                mic_positions=mic_positions,
            )
            # (frames_num, ndim)

            src_mic_collide = is_collide(
                trajs1=src_positions, 
                trajs2=mic_positions, 
                collision_raidus=self.collision_raidus
            )

            src_agent_collide = is_collide(
                trajs1=src_positions, 
                trajs2=agent_position[None, :, :], 
                collision_raidus=self.collision_raidus
            )

            if src_mic_collide or src_agent_collide:
                # Resample sources, mics, and agents positions to avoid collision.
                continue
            else:
                break

        # ------ 3. Sample sources and mics orientations. ------
        src_oriens = self.sample_source_orientations(
            sources_num=srcs_num,
        )
        # shape: (srcs_num, frames_num, ndim)

        mic_oriens = self.sample_mic_orientations()
        # (mics_num, frames_num, ndim)

        srcs = self.sample_sources(srcs_num)
        # shape: (srcs_num, seg_samples)
        # from IPython import embed; embed(using=False); os._exit(0)

        # ------ 4. Simulate mics signals. ------

        mic_wavs = self.simulate_mic_waveforms(
            environment=environment, 
            sources=srcs, 
            source_positions=src_positions, 
            mic_positions=mic_positions, 
            mic_orientations=mic_oriens
        )

        # ------ 5. Sample agents orientations and signals. ------
        
        agents = self.sample_agents(
            sources=srcs, 
            source_positions=src_positions, 
            agent_position=agent_position, 
            environment=environment
        )

        # ------ 6. Collect data. ------
        agents_num = len(agents)
        agent_positions = np.stack([a.position for a in agents], axis=0)
        agent_look_at_directions = np.stack([a.look_at_direction for a in agents], axis=0)
        agent_look_at_distances = np.stack([a.look_at_distance for a in agents], axis=0)
        agent_distance_masks = (agent_look_at_distances >= 0).astype(np.float32)

        # Task dependent indexes
        agent_detect_idxes = np.array([i for i in range(agents_num) if agents[i].look_at_direction_has_source is not None])

        agent_distance_idxes = np.array([i for i in range(agents_num) if agents[i].look_at_distance_has_source is not None])

        agent_sep_idxes = np.array([i for i in range(agents_num) if agents[i].look_at_direction_direct_waveform is not None])

        # Targets
        agent_look_at_direction_has_source = np.array([agents[i].look_at_direction_has_source for i in agent_detect_idxes])

        agent_look_at_distance_has_source = np.array([agents[i].look_at_distance_has_source for i in agent_distance_idxes])

        agent_look_at_direction_direct_wav = np.array([agents[i].look_at_direction_direct_waveform for i in agent_sep_idxes])

        agent_look_at_direction_reverb_wav = np.array([agents[i].look_at_direction_reverb_waveform for i in agent_sep_idxes])

        data = {
            "source": srcs,
            "source_position": src_positions,
            "mic_wavs": mic_wavs,
            "mic_positions": mic_positions,
            "mic_orientations": mic_oriens,
            "agent_positions": agent_positions,
            "agent_look_at_directions": agent_look_at_directions,
            "agent_look_at_distances": agent_look_at_distances,
            "agent_distance_masks": agent_distance_masks,
            "agent_detect_idxes": agent_detect_idxes,
            "agent_distance_idxes": agent_distance_idxes,
            "agent_sep_idxes": agent_sep_idxes,
            "agent_look_at_direction_has_source": agent_look_at_direction_has_source,
            "agent_look_at_distance_has_source": agent_look_at_distance_has_source,
            "agent_look_at_direction_direct_wav": agent_look_at_direction_direct_wav,
            "agent_look_at_direction_reverb_wav": agent_look_at_direction_reverb_wav
        }
        # if len(srcs) == 0:
        #     from IPython import embed; embed(using=False); os._exit(0)
        # soundfile.write(file="_zz.wav", data=mic_wavs[0], samplerate=24000)
        # soundfile.write(file="_zz.wav", data=agent_look_at_direction_direct_wav[0], samplerate=24000)
        # soundfile.write(file="_zz2.wav", data=agent_look_at_direction_reverb_wav[0], samplerate=24000)

        # sources_num = 
        # tmp = np.zeros(16)
        # tmp[srcs_num] = 1
        frames_num = mic_positions.shape[1]
        data["sources_num"] = np.ones(frames_num) * srcs_num

        return data

    def load_mic_noise_paths(self, mic_noise_dir):
        if mic_noise_dir:
            return sorted(list(Path(mic_noise_dir).glob("*.wav")))
        else:
            return None

    def sample_environment(self):

        room_length = random.uniform(a=self.min_room_length, b=self.max_room_length)
        room_width = random.uniform(a=self.min_room_width, b=self.max_room_width)
        room_height = random.uniform(a=self.min_room_height, b=self.max_room_height)
        max_room_distance = math.sqrt(room_length ** 2 + room_width ** 2 + room_height ** 2)

        # x=0, y=0 is the center of a 2D polygon. z=0 is the floor.
        x1 = random.uniform(a=self.room_margin, b=room_length - self.room_margin)
        x0 = x1 - room_length

        y1 = random.uniform(a=self.room_margin, b=room_width - self.room_margin)
        y0 = y1 - room_width

        z1 = room_height
        z0 = 0.

        environment = {
            "x0": x0,
            "x1": x1,
            "y0": y0,
            "y1": y1,
            "z0": z0,
            "z1": z1,
            "ndim": self.ndim,
            "room_length": room_length,
            "room_width": room_width,
            "room_height": room_height,
            "max_room_distance": max_room_distance,
            "room_margin": self.room_margin
        }

        return environment

    def sample_source_positions(self, sources_num, environment):

        src_poss = []

        for _ in range(sources_num):

            src_pos = self.sample_static_position_in_room(environment)
            # shape: (ndim,)

            src_pos = expand_along_frame_axis(x=src_pos, repeats=self.frames_num)
            # shape: (frames_num, ndim)

            src_poss.append(src_pos)

        src_poss = np.array(src_poss)
        # shape: (srcs_num, frames_num, ndim)

        return src_poss

    def sample_mic_positions(self, environment, mics_meta):

        coordinate_type = mics_meta["coordinate_type"]
        mics_coords = mics_meta["mic_coordinates"]
        mic_poss = []

        for mic_coord in mics_coords:

            mic_pos = self.coordinate_to_position(mic_coord)
            # shape: (ndim,)

            mic_pos = expand_along_frame_axis(x=mic_pos, repeats=self.frames_num)
            # shape: (frames_num, ndim)

            mic_poss.append(mic_pos)

        mic_poss = np.stack(mic_poss, axis=0)
        # (mics_num, frames_num, ndim)

        if coordinate_type == "absolute": 

            return mic_poss

        elif coordinate_type == "relative":

            mics_center_pos = self.sample_static_position_in_room(environment)
            # shape: (ndim,)

            mics_center_pos = expand_along_frame_axis(
                x=mics_center_pos, 
                repeats=self.frames_num
            )
            # shape: (frames_num, ndim)

            mic_poss += mics_center_pos
            # (mics_num, frames_num, ndim)

            return mic_poss

        else:
            raise NotImplementedError

    def sample_agent_positions(self, environment, mic_positions):

        if self.agent_positions_type == "center_of_mics":

            agent_pos = np.mean(mic_positions, axis=0)
            # shape: (frames_num, ndim)

            return agent_pos

        elif self.agent_positions_type == "random_in_environment":

            agent_pos = self.sample_static_position_in_room(environment)
            # shape: (ndim,)

            agent_pos = expand_along_frame_axis(x=agent_pos, repeats=self.frames_num)
            # shape: (frames_num, ndim)
            
            return agent_pos

        else:
            raise NotImplementedError

    def coordinate_to_position(self, coordinate):

        if set(["x", "y", "z"]) <= set(coordinate.keys()):
            pos = np.array([coordinate["x"], coordinate["y"], coordinate["z"]])

        if set(["azimuth_deg", "elevation_deg", "radius"]) <= set(coordinate.keys()):
            pos = sph2cart(
                azimuth=np.deg2rad(coordinate["azimuth_deg"]), 
                elevation=np.deg2rad(coordinate["elevation_deg"]),
                r=coordinate["radius"],
            )

        return pos

    def sample_mic_orientations(self):

        raw_mics_oriens = self.mics_meta["mic_orientations"]
        mic_oriens = []

        for mic_orien in raw_mics_oriens:

            mic_orien = self.coordinate_to_orientation(mic_orien)
            # shape: (ndim,)

            mic_orien = expand_along_frame_axis(x=mic_orien, repeats=self.frames_num)
            # shape: (frames_num, dim)

            mic_oriens.append(mic_orien)

        mic_oriens = np.array(mic_oriens)
        # shape: (mics_num, frames_num, dim)

        return mic_oriens


    def coordinate_to_orientation(self, orientation):

        if set(["x", "y", "z"]) <= set(orientation.keys()):
            orientation = np.array([orientation["x"], orientation["y"], orientation["z"]])

        if set(["azimuth_deg", "elevation_deg"]) <= set(orientation.keys()):
            orientation = sph2cart(
                azimuth=np.deg2rad(orientation["azimuth_deg"]), 
                elevation=np.deg2rad(orientation["elevation_deg"]),
                r=1.,
            )

        orientation = normalize(orientation)

        return orientation

    def sample_static_position_in_room(self, environment):
        
        env = environment
        margin = env["room_margin"]
        position = np.zeros(env["ndim"])

        position[0] = random.uniform(a=env["x0"] + margin, b=env["x1"] - margin)
        position[1] = random.uniform(a=env["y0"] + margin, b=env["y1"] - margin)
        position[2] = random.uniform(a=env["z0"] + margin, b=env["z1"] - margin)

        return position

    def sample_sources(self, sources_num):

        srcs = []

        # Sample source.
        for _ in range(sources_num):

            # Load audio.
            src_path = random.choice(self.audio_paths)
            
            src, _ = librosa.load(path=src_path, sr=self.sample_rate, mono=True)
            # shape: (seg_samples,)

            # Get random gain.
            src_gain_db = random.uniform(a=self.min_source_gain_db, b=self.max_source_gain_db)
            src_gain_scale = db_to_scale(src_gain_db)

            # Gain augmentation.
            src *= src_gain_scale

            srcs.append(src)

        srcs = np.array(srcs)
        # shape: (srcs_num, seg_samples)

        return srcs

    def sample_source_orientations(self, sources_num):

        src_oriens = []

        for _ in range(sources_num):

            src_orien = random_direction()
            # shape: (ndim,)

            src_orien = expand_along_frame_axis(x=src_orien, repeats=self.frames_num)
            # shape: (frames_num, ndim)

            src_oriens.append(src_orien)

        src_oriens = np.array(src_oriens)
        # shape: (srcs, frames_num, ndim)

        return src_oriens

    def get_static_position(self, position):

        if np.array_equiv(a1=position[0], a2=position):
            return position[0]

        else:
            raise NotImplementedError("Only support static source for now!")

    def simulate_mic_waveforms(
        self,
        environment, 
        sources, 
        source_positions,
        mic_positions, 
        mic_orientations
    ):

        srcs_num = len(source_positions)
        mics_num = len(mic_positions)

        if self.mic_cutoff_freq is not None:
            for i in range(srcs_num):
                sources[i] = apply_lowpass_filter(
                    audio=sources[i], 
                    cutoff_freq=self.mic_cutoff_freq,
                    sample_rate=self.sample_rate,
                )

        # Get static source and mic positions.
        static_src_poss = [self.get_static_position(position=src_pos) for src_pos in source_positions]

        static_mic_poss = [self.get_static_position(position=mic_pos) for mic_pos in mic_positions]

        static_mic_oriens = [self.get_static_position(position=mic_orien) for mic_orien in mic_orientations]

        mic_wavs = []

        for mic_idx in range(mics_num):
            # from IPython import embed; embed(using=False); os._exit(0) 

            engine = ImageSourceEngine(
                environment=environment, 
                source_positions=static_src_poss,
                mic_position=static_mic_poss[mic_idx], 
                mic_orientation=static_mic_oriens[mic_idx],
                mic_spatial_irs=self.mic_spatial_irs,
                image_source_order=self.image_source_order,
                speed_of_sound=self.speed_of_sound,
                sample_rate=self.sample_rate,
                compute_direct_ir_only=False,
                center_mic_pos=np.mean(static_mic_poss, axis=0)
            )

            _, srcs_h_reverb = engine.compute_spatial_ir()
            # srcs_h_reverb: (2, var_len)

            mic_wav = np.zeros(self.segment_samples)
            # shape: (seg_samples,)

            for src, h in zip(sources, srcs_h_reverb):

                # new
                mic_wav += fftconvolve(in1=src, in2=h, mode="same")
                # shape: (seg_samples,)

            if self.mic_noise_paths is not None:

                mic_noise = self.sample_mic_noise()
                # shape: (seg_samples,)

                mic_wav += mic_noise
                # shape: (seg_samples,)

            mic_wavs.append(mic_wav)

        mic_wavs = np.stack(mic_wavs, axis=0)

        # from IPython import embed; embed(using=False); os._exit(0)
        # soundfile.write(file="_zz.wav", data=mic_wav, samplerate=self.sample_rate)

        return mic_wavs

    def sample_mic_noise(self):
        
        audio_path = random.choice(self.mic_noise_paths)
        audio_duration = librosa.get_duration(path=audio_path)

        bgn_sec = random.uniform(a=0, b=audio_duration - self.segment_seconds)

        audio, _ = librosa.load(
            path=audio_path, 
            offset=bgn_sec, 
            duration=self.segment_seconds, 
            sr=self.sample_rate
        )

        gain_db = random.uniform(a=self.min_noise_gain_db, b=self.max_noise_gain_db)
        gain_scale = db_to_scale(gain_db)

        audio *= gain_scale
        # from IPython import embed; embed(using=False); os._exit(0)

        return audio

    def sample_agents(self, sources, source_positions, agent_position, environment): 

        srcs_num = len(source_positions)

        # --- 1. Detection agents ---
        detection_agents = []
        assert self.agent_det_max_pos_rays <= self.agent_det_total_rays

        src_idxes = self.sample_source_indexes(
            sources_num=srcs_num, 
            max_positive_rays=self.agent_det_max_pos_rays,
        )

        # Positive agents.
        for src_idx in src_idxes:

            agent = self.random_positive_detection_agent(
                source_position=source_positions[src_idx],
                agent_position=agent_position
            )
            detection_agents.append(agent)

        # Negative agents.
        while len(detection_agents) < self.agent_det_total_rays:

            agent = self.random_negative_detection_agent(
                source_positions=source_positions,
                agent_position=agent_position
            )
            detection_agents.append(agent)

        # --- 2. Distance estimation agents ---
        distance_agents = []
        assert self.agent_dist_max_pos_rays <= self.agent_dist_total_rays

        src_idxes = self.sample_source_indexes(
            sources_num=srcs_num, 
            max_positive_rays=self.agent_dist_max_pos_rays,
        )

        # Positive agents.
        for src_idx in src_idxes:

            agent = self.random_positive_distance_agent(
                source_position=source_positions[src_idx],
                agent_position=agent_position
            )
            distance_agents.append(agent)

        # Negative agents.
        while len(distance_agents) < self.agent_dist_total_rays:

            agent = self.random_negative_distance_agent(
                source_positions=source_positions,
                agent_position=agent_position,
                max_room_distance=environment["max_room_distance"]
            )
            distance_agents.append(agent)

        # --- 3. Spatial source separation agents ---
        sep_agents = []
        assert self.agent_sep_max_pos_rays <= self.agent_sep_total_rays

        src_idxes = self.sample_source_indexes(
            sources_num=srcs_num, 
            max_positive_rays=self.agent_sep_max_pos_rays,
        )

        # Positive agents.
        for src_idx in src_idxes:

            agent = self.random_positive_sep_agent(
                environment=environment,
                source=sources[src_idx],
                source_position=source_positions[src_idx],
                agent_position=agent_position
            )

            sep_agents.append(agent)

        # Negative agents.
        while len(sep_agents) < self.agent_sep_total_rays:

            agent = self.random_negative_sep_agent(
                source_positions=source_positions,
                agent_position=agent_position,
            )
            sep_agents.append(agent)

        agents = detection_agents + distance_agents + sep_agents
        
        return agents

    def sample_source_indexes(self, sources_num, max_positive_rays):

        if sources_num <= max_positive_rays:
            return range(sources_num)
        else:
            return random.sample(range(sources_num), k=max_positive_rays)

    def random_positive_detection_agent(self, source_position, agent_position):

        agent_to_src = source_position - agent_position
        static_agent_to_src = self.get_static_position(agent_to_src)

        look_at_direction = random_positive_direction(
            direction=static_agent_to_src, 
            theta=np.deg2rad(self.source_apprent_diameter_deg / 2)
        )
        look_at_direction = expand_along_frame_axis(
            x=look_at_direction, 
            repeats=self.frames_num
        )

        look_at_distance = PAD * np.ones(self.frames_num)

        # 
        """
        included_angles = get_included_angle(look_at_direction, agent_to_src)

        look_at_direction_has_source = triangle_function(
            x=included_angles, 
            r=np.deg2rad(self.source_apprent_diameter_deg / 2)
        )
        """
        look_at_direction_has_source = np.ones(self.frames_num)

        agent = Agent(
            position=agent_position,
            look_at_direction=look_at_direction,
            look_at_distance=look_at_distance,
            look_at_direction_has_source=look_at_direction_has_source
        )
        
        return agent

    def random_negative_detection_agent(self, source_positions, agent_position):

        directions = [self.get_static_position(src_pos - agent_position) for src_pos in source_positions]
        
        look_at_direction = random_negative_direction(
            directions=directions, 
            theta=np.deg2rad(self.source_apprent_diameter_deg / 2)
        )
        look_at_direction = expand_along_frame_axis(
            x=look_at_direction, 
            repeats=self.frames_num
        )

        look_at_distance = PAD * np.ones(self.frames_num)

        look_at_direction_has_source = np.zeros(self.frames_num)

        agent = Agent(
            position=agent_position,
            look_at_direction=look_at_direction,
            look_at_distance=look_at_distance,
            look_at_direction_has_source=look_at_direction_has_source
        )

        return agent

    def random_positive_distance_agent(self, source_position, agent_position):

        agent_to_src = source_position - agent_position
        static_agent_to_src = self.get_static_position(agent_to_src)

        look_at_direction = random_positive_direction(
            direction=static_agent_to_src, 
            theta=np.deg2rad(self.source_apprent_diameter_deg / 2)
        )
        look_at_direction = expand_along_frame_axis(
            x=look_at_direction, 
            repeats=self.frames_num
        )
        # shape: (frames_num, ndim)

        src_distance = np.linalg.norm(static_agent_to_src)
        # shape: (1,)

        look_at_distance = random_positive_distance(
            distance=src_distance, 
            r=self.source_radius
        )
        # shape: (1,)

        # relative_dist = look_at_distance - src_distance
        # shape: (1,)

        """
        look_at_distance_has_source = triangle_function(
            x=relative_dist,
            r=self.source_radius,
        )
        """
        
        # shape: (1,)
        
        look_at_distance = look_at_distance * np.ones(self.frames_num)
        # look_at_distance_has_source = look_at_distance_has_source * np.ones(self.frames_num)

        look_at_distance_has_source = np.ones(self.frames_num)

        agent = Agent(
            position=agent_position,
            look_at_direction=look_at_direction,
            look_at_distance=look_at_distance,
            look_at_distance_has_source=look_at_distance_has_source
        )

        return agent

    def random_negative_distance_agent(self, source_positions, agent_position, max_room_distance):

        if len(source_positions) == 0:

            look_at_direction = random_direction()
            # shape: (1,)

            look_at_distance = random.uniform(a=0., b=max_room_distance)
            # shape: (1,)
            
        else:
            src_pos = random.choice(source_positions)

            agent_to_src = src_pos - agent_position
            static_agent_to_src = self.get_static_position(agent_to_src)

            look_at_direction = random_positive_direction(
                direction=static_agent_to_src, 
                theta=np.deg2rad(self.source_apprent_diameter_deg / 2)
            )
            
            src_distance = np.linalg.norm(static_agent_to_src)
            # shape: (1,)

            look_at_distance = random_negative_distance(
                distances=[src_distance], 
                r=self.source_radius,
                max_dist=max_room_distance,
            )
            # shape: (1,)

        look_at_direction = expand_along_frame_axis(
                x=look_at_direction, 
                repeats=self.frames_num
            )
            # shape: (frames_num, ndim)

        look_at_distance = look_at_distance * np.ones(self.frames_num)
        look_at_distance_has_source = np.zeros(self.frames_num)

        agent = Agent(
            position=agent_position,
            look_at_direction=look_at_direction,
            look_at_distance=look_at_distance,
            look_at_distance_has_source=look_at_distance_has_source
        )

        return agent

    def random_positive_sep_agent(self, environment, source, source_position, agent_position):

        agent_to_src = source_position - agent_position
        static_agent_to_src = self.get_static_position(agent_to_src)

        look_at_direction = random_positive_direction(
            direction=static_agent_to_src, 
            theta=np.deg2rad(self.source_apprent_diameter_deg / 2)
        )
        look_at_direction = expand_along_frame_axis(
            x=look_at_direction, 
            repeats=self.frames_num
        )

        look_at_distance = PAD * np.ones(self.frames_num)

        engine = ImageSourceEngine(
            environment=environment, 
            source_positions=[self.get_static_position(source_position)],
            mic_position=self.get_static_position(agent_position), 
            mic_orientation=None,
            mic_spatial_irs=None,
            image_source_order=self.image_source_order,
            speed_of_sound=self.speed_of_sound,
            sample_rate=self.sample_rate,
            compute_direct_ir_only=False,
            center_mic_pos=None,
        )

        srcs_h_direct, srcs_h_reverb = engine.compute_spatial_ir()
        h_direct = srcs_h_direct[0]
        h_reverb = srcs_h_reverb[0]
        
        direct_wav = fftconvolve(in1=source, in2=h_direct, mode="same")
        reverb_wav = fftconvolve(in1=source, in2=h_reverb, mode="same")

        agent = Agent(
            position=agent_position,
            look_at_direction=look_at_direction,
            look_at_distance=look_at_distance,
            look_at_direction_direct_waveform=direct_wav,
            look_at_direction_reverb_waveform=reverb_wav
        )

        return agent

    def random_negative_sep_agent(self, source_positions, agent_position):

        directions = [self.get_static_position(src_pos - agent_position) for src_pos in source_positions]
        
        look_at_direction = random_negative_direction(
            directions=directions, 
            theta=np.deg2rad(self.source_apprent_diameter_deg / 2)
        )
        look_at_direction = expand_along_frame_axis(
            x=look_at_direction, 
            repeats=self.frames_num
        )

        look_at_distance = PAD * np.ones(self.frames_num)

        direct_wav = np.zeros(self.segment_samples)
        reverb_wav = np.zeros(self.segment_samples)

        agent = Agent(
            position=agent_position,
            look_at_direction=look_at_direction,
            look_at_distance=look_at_distance,
            look_at_direction_direct_waveform=direct_wav,
            look_at_direction_reverb_waveform=reverb_wav
        )

        return agent

    # def __len__(self):
    #     return 100