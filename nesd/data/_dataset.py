import random
from pathlib import Path
import time

import math
import librosa
import soundfile
import numpy as np
import yaml
import pyroomacoustics as pra
import pickle
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

from nesd.utils import db_to_scale, sph2cart, normalize, random_direction, fractional_delay_filter, get_included_angle, random_positive_direction, triangle_function, random_negative_direction, random_positive_distance, random_negative_distance


class Dataset:
    def __init__(self, simulator_configs, split):
        # self.audios_dir = audios_dir
        # self.expand_frames = expand_frames

        self.simulator_configs = simulator_configs

        # General
        self.speed_of_sound = simulator_configs["speed_of_sound"]
        self.sample_rate = simulator_configs["sample_rate"]
        self.segment_seconds = simulator_configs["segment_seconds"]
        self.frames_per_sec = simulator_configs["frames_per_sec"]
        self.collision_radius = simulator_configs["collision_radius"]
        self.frames_num = int(self.frames_per_sec * self.segment_seconds) + 1
        self.segment_samples = int(self.segment_seconds * self.sample_rate)

        # Env
        self.min_room_length = simulator_configs["room_length"]["min"]
        self.max_room_length = simulator_configs["room_length"]["max"]
        self.min_room_width = simulator_configs["room_width"]["min"]
        self.max_room_width = simulator_configs["room_width"]["max"]
        self.min_room_height = simulator_configs["room_height"]["min"]
        self.max_room_height = simulator_configs["room_height"]["max"]
        self.room_margin = simulator_configs["room_margin"]
        self.ndim = 3

        # Mic
        # self.mic_positions_type = simulator_configs["mic_positions_type"]
        self.mics_meta = self.load_mics_meta(mics_yaml=simulator_configs["mics_yaml"])
        self.mic_spatial_irs_path = simulator_configs["mic_spatial_irs_path"]
        self.mic_spatial_irs = pickle.load(open(self.mic_spatial_irs_path, "rb"))

        self.audios_dir = simulator_configs["{}_audios_dir".format(split)]
        self.min_sources_num = simulator_configs["sources_num"]["min"]
        self.max_sources_num = simulator_configs["sources_num"]["max"]
        self.sample_rate = simulator_configs["sample_rate"]
        self.min_source_gain_db = simulator_configs["source_gain_db"]["min"]
        self.max_source_gain_db = simulator_configs["source_gain_db"]["max"]
        self.source_apprent_diameter_deg = simulator_configs["source_apprent_diameter_deg"]
        self.source_radius = simulator_configs["source_radius"]

        # Simulator
        self.image_source_order = simulator_configs["image_source_order"]

        self.agent_positions_type = simulator_configs["agent_positions_type"]
        self.agent_det_max_pos_rays = simulator_configs["agent_detection_rays"]["max_positive"]
        self.agent_det_total_rays = simulator_configs["agent_detection_rays"]["total"]
        self.agent_dist_max_pos_rays = simulator_configs["agent_distance_rays"]["max_positive"]
        self.agent_dist_total_rays = simulator_configs["agent_distance_rays"]["total"]
        self.agent_sep_max_pos_rays = simulator_configs["agent_sep_rays"]["max_positive"]
        self.agent_sep_total_rays = simulator_configs["agent_sep_rays"]["total"]


        self.audio_paths = sorted(list(Path(self.audios_dir).glob("*.wav")))

        # from IPython import embed; embed(using=False); os._exit(0)       

        # sources

        # envs

        # microp

        # agents

    def __getitem__(self, meta):
        # print(meta)

        # ------ 1. Sample environment. ------
        room_length, room_width, room_height = self.sample_environment()

        environment = {
            "room_length": room_length,
            "room_width": room_width,
            "room_height": room_height,
            "room_margin": self.room_margin,
            "max_room_dist": math.sqrt(room_length ** 2 + room_width ** 2 + room_height ** 2)
        }

        # ------ 2. Sample sources and their positions and orientations. ------

        sources = self.sample_sources()
        # shape: (sources_num, samples_num)

        source_positions = self.sample_source_positions(
            sources_num=len(sources),
            environment=environment,
            # exclude_area_centers=mic_positions, 
            # exclude_area_raidus=self.collision_radius
        )
        # shape: (sources_num, frames_num, ndim)

        source_orientations = self.sample_source_orientations(
            sources_num=len(sources),
        )
        # shape: (sources_num, frames_num, ndim)

        # ------ 3. Sample microphone positions and orientations ------
        mic_positions = self.sample_mic_positions(
            environment=environment,
            exclude_area_centers=source_positions, 
            exclude_area_raidus=self.collision_radius
        )
        # (mics_num, frames_num, ndim)

        mic_orientations = self.sample_mic_orientations()
        # (mics_num, frames_num, ndim)

        # ------ 4. Simulate microphone signals ------

        t1 = time.time()

        mic_signals = self.simulate_mic_signals(
            environment=environment, 
            sources=sources, 
            source_positions=source_positions, 
            mic_positions=mic_positions, 
            mic_orientations=mic_orientations
        )

        print(time.time() - t1)

        # ------ 4. Sample agents ------

        agent_position = self.sample_agent_positions(
            environment=environment,
            mic_positions=mic_positions,
            exclude_area_centers=source_positions, 
            exclude_area_raidus=self.collision_radius
        )
        # (frames_num, ndim)

        agents = self.simulate_agents(sources, source_positions, agent_position, environment)
        agents_num = len(agents)

        agent_positions = np.stack([a.position for a in agents], axis=0)
        agent_look_at_directions = np.stack([a.look_at_direction for a in agents], axis=0)

        #
        agent_look_at_direction_has_source_idxes = np.stack([i for i in range(agents_num) if agents[i].look_at_direction_has_source is not None], axis=0)

        agent_look_at_distance_has_source_idxes = np.stack([i for i in range(agents_num) if agents[i].look_at_distance_has_source is not None], axis=0)

        agent_look_at_direction_has_wav_idxes = np.stack([i for i in range(agents_num) if agents[i].look_at_direction_direct_waveform is not None], axis=0)

        #
        a1 = [agents[i].look_at_distance for i in agent_look_at_distance_has_source_idxes]
        
        agent_look_at_distances = np.stack([agents[i].look_at_distance for i in agent_look_at_distance_has_source_idxes], axis=0)

        agent_look_at_direction_has_source = np.stack([agents[i].look_at_direction_has_source for i in agent_look_at_direction_has_source_idxes], axis=0)

        agent_look_at_distance_has_source = np.stack([agents[i].look_at_distance_has_source for i in agent_look_at_distance_has_source_idxes], axis=0)

        agent_look_at_direction_direct_wav = np.stack([agents[i].look_at_direction_direct_waveform for i in agent_look_at_direction_has_wav_idxes], axis=0)

        agent_look_at_direction_reverb_wav = np.stack([agents[i].look_at_direction_reverb_waveform for i in agent_look_at_direction_has_wav_idxes], axis=0)

        data = {
            "source": sources,
            "source_position": source_positions,
            "mic": mic_signals,
            "mic_position": mic_positions,
            "mic_orientation": mic_orientations,
            "agent_position": agent_positions,
            "agent_look_at_direction": agent_look_at_directions,
            "agent_look_at_distances": agent_look_at_distances,
            "agent_look_at_direction_has_source": agent_look_at_distance_has_source,
            "agent_look_at_distance_has_source": agent_look_at_distance_has_source,
            "agent_look_at_direction_direct_wav": agent_look_at_direction_direct_wav,
            "agent_look_at_direction_reverb_wav": agent_look_at_direction_reverb_wav,
            "agent_detect_idxes": agent_look_at_direction_has_source_idxes,
            "agent_distance_idxes": agent_look_at_distance_has_source_idxes,
            "agent_sep_idxes": agent_look_at_direction_has_wav_idxes
        }

        from IPython import embed; embed(using=False); os._exit(0)

        return data


    def load_mics_meta(self, mics_yaml):

        with open(mics_yaml, 'r') as f:
            mics_meta = yaml.load(f, Loader=yaml.FullLoader)

        return mics_meta


    def sample_environment(self):

        room_length = random.uniform(a=self.min_room_length, b=self.max_room_length)
        room_width = random.uniform(a=self.min_room_width, b=self.max_room_width)
        room_height = random.uniform(a=self.min_room_height, b=self.max_room_height)

        return room_length, room_width, room_height

    def sample_mic_positions(self, environment, exclude_area_centers, exclude_area_raidus):

        coordinate_type = self.mics_meta["coordinate_type"]
        mics_coords = self.mics_meta["microphone_coordinates"]
        mic_poss = []

        for mic_coord in mics_coords:
            mic_pos = self.coordinate_to_position(mic_coord)
            mic_pos = expand_along_frame_axis(x=mic_pos, repeats=self.frames_num)
            mic_poss.append(mic_pos)

        mic_poss = np.stack(mic_poss, axis=0)
        # (mics_num, frames_num, ndim)

        if coordinate_type == "absolute": 
            return mic_poss

        elif coordinate_type == "relative":

            bool_valid = False

            while bool_valid is False:

                mics_center_pos = self.sample_static_position_in_room(
                    room_length=environment["room_length"], 
                    room_width=environment["room_width"], 
                    room_height=environment["room_height"],
                    room_margin=environment["room_margin"]
                )

                mics_center_pos = expand_along_frame_axis(
                    x=mics_center_pos, 
                    repeats=self.frames_num
                )
                # (frames_num, ndim)

                bool_valid = self.verify_position(
                    position=mics_center_pos, 
                    exclude_area_centers=exclude_area_centers, 
                    exclude_area_raidus=exclude_area_raidus
                )

            mic_poss += mics_center_pos
                
            return mic_poss

        else:
            raise NotImplementedError


    def coordinate_to_position(self, coordinate):

        if set(["x", "y", "z"]) <= set(coordinate.keys()):
            position = np.array([coordinate["x"], coordinate["y"], coordinate["z"]])

        if set(["azimuth_deg", "elevation_deg", "radius"]) <= set(coordinate.keys()):
            position = sph2cart(
                azimuth=np.deg2rad(coordinate["azimuth_deg"]), 
                elevation=np.deg2rad(coordinate["elevation_deg"]),
                r=coordinate["radius"],
            )

        return position


    def sample_mic_orientations(self):

        raw_mics_oriens = self.mics_meta["microphone_orientations"]
        mics_oriens = []

        for mic_orien in raw_mics_oriens:

            mic_orien = self.coordinate_to_orientation(mic_orien)
            mic_orien = expand_along_frame_axis(x=mic_orien, repeats=self.frames_num)

            mics_oriens.append(mic_orien)

        mics_oriens = np.stack(mics_oriens, axis=0)

        return mics_oriens


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

    '''
    def sample_static_position_in_room(self, 
        room_length, 
        room_width, 
        room_height, 
        room_margin, 
        exclude_area_centers=None, 
        exclude_area_raidus=None
    ):

        position = np.zeros(self.ndim)

        while True:

            for dim, edge in enumerate((room_length, room_width, room_height)):
                position[dim] = random.uniform(a=room_margin, b=edge - room_margin)

            if exclude_area_centers is None or exclude_area_raidus is None:
                break

            bool_valid_position = self.verify_position(
                position=position, 
                exclude_area_centers=exclude_area_centers, 
                exclude_area_raidus=exclude_area_raidus
            )

            if bool_valid_position:
                break

        return position
    '''

    def sample_static_position_in_room(self, 
        room_length, 
        room_width, 
        room_height, 
        room_margin,
    ):
        position = np.zeros(self.ndim)

        for dim, edge in enumerate((room_length, room_width, room_height)):
            position[dim] = random.uniform(a=room_margin, b=edge - room_margin)

        return position


    def sample_sources(self):
        sources_num = random.randint(a=self.min_sources_num, b=self.max_sources_num)

        sources = []

        # Sample source.
        for _ in range(sources_num):

            # Load audio.
            source_path = random.choice(self.audio_paths)
            source, _ = librosa.load(path=source_path, sr=self.sample_rate, mono=True)

            # Get random gain.
            source_gain_db = random.uniform(a=self.min_source_gain_db, b=self.max_source_gain_db)
            source_gain_scale = db_to_scale(source_gain_db)

            # Gain augmentation.
            source *= source_gain_scale

            sources.append(source)

        if sources_num > 0:
            sources = np.stack(sources, axis=0)
        
        return sources

        # print(source_gain_db, source_gain_scale)
        
        # soundfile.write(file="_zz.wav", data=source, samplerate=self.sample_rate)

    '''
    def verify_position(self, position, exclude_area_centers, exclude_area_raidus):
        
        for exclude_area_center in exclude_area_centers:

            if np.linalg.norm(position - exclude_area_center) < exclude_area_raidus:
                return False

        return True
    '''

    '''
    def sample_source_positions(self, 
        sources_num, 
        environment, 
        exclude_area_centers, 
        exclude_area_raidus
    ):

        source_positions = []

        for _ in range(sources_num):

            bool_valid = False

            while bool_valid is False:

                source_pos = self.sample_static_position_in_room(
                    room_length=environment["room_length"], 
                    room_width=environment["room_width"], 
                    room_height=environment["room_height"],
                    room_margin=environment["room_margin"]
                )
                source_pos = expand_along_frame_axis(x=source_pos, repeats=self.frames_num)
                # shape: (frames_num, ndim)

                bool_valid = self.verify_position(
                    position=source_pos, 
                    exclude_area_centers=exclude_area_centers, 
                    exclude_area_raidus=exclude_area_raidus
                )

            source_positions.append(source_pos)

        if sources_num > 0:
            source_positions = np.stack(source_positions, axis=0)

        return source_positions
    '''
    def sample_source_positions(self, 
        sources_num, 
        environment, 
    ):

        source_positions = []

        for _ in range(sources_num):

            source_pos = self.sample_static_position_in_room(
                room_length=environment["room_length"], 
                room_width=environment["room_width"], 
                room_height=environment["room_height"],
                room_margin=environment["room_margin"]
            )
            source_pos = expand_along_frame_axis(x=source_pos, repeats=self.frames_num)
            # shape: (frames_num, ndim)

            source_positions.append(source_pos)

        if sources_num > 0:
            source_positions = np.stack(source_positions, axis=0)

        return source_positions
    
    def verify_position(self, position, exclude_area_centers, exclude_area_raidus):
        
        for exclude_area_center in exclude_area_centers:

            distances = np.linalg.norm(x=position - exclude_area_center, axis=-1)

            if any(distances < exclude_area_raidus):
                return False

        return True

    def sample_source_orientations(self, sources_num):

        source_orientations = []

        for _ in range(sources_num):

            source_orien = random_direction()
            source_orien = expand_along_frame_axis(x=source_orien, repeats=self.frames_num)
            # shape: (frames_num, ndim)

            source_orientations.append(source_orien)

        if sources_num > 0:
            source_orientations = np.stack(source_orientations, axis=0)

        return source_orientations

    def render_image_sources(self, environment, source_positions):

        image_metas = []
        sources_num = len(source_positions)

        # Render image sources
        for source_index in range(sources_num):

            corners = np.array([
                [0, 0], 
                [0, environment["room_width"]], 
                [environment["room_length"], environment["room_width"]], 
                [environment["room_length"], 0]
            ]).T
            # shape: (2, 4)

            room = pra.Room.from_corners(
                corners=corners,
                max_order=self.image_source_order,
            )

            room.extrude(height=environment["room_height"])

            source_pos = source_positions[source_index]

            static_source_pos = self.get_static_source_position(position=source_pos)

            while True:
                try:
                    room.add_source(static_source_pos)
                    break
                except:
                    continue

            # Add a dummy microphone that will not be used. This is a syntax required by PyRoomAcoustics.
            room.add_microphone([0.1, 0.1, 0.1])

            room.image_source_model()

            images = room.sources[0].images.T
            # (images_num, ndim)

            orders = room.sources[0].orders

            unique_images = []

            for i, image in enumerate(images):

                    unique_images.append(image)

                    image_meta = {
                        "source_index": source_index,
                        "order": room.sources[0].orders[i],
                        "position": np.array(image),
                    }
                    image_metas.append(image_meta)
        
        return image_metas

    def get_static_position(self, position):

        static_position = position[0]

        if np.array_equiv(a1=static_position, a2=position):
            return static_position

        else:
            raise NotImplementedError("Only support static source for now!")

    def simulate_mic_signals(
        self,
        environment, 
        sources, 
        source_positions,
        mic_positions, 
        mic_orientations
    ):

        sources_num = len(sources)
        mics_num = len(mic_positions)

        # Get static source positions
        static_source_positions = [self.get_static_position(position=source_pos) for source_pos in source_positions]

        static_mic_positions = [self.get_static_position(position=mic_pos) for mic_pos in mic_positions]

        static_mic_oriens = [self.get_static_position(position=mic_pos) for mic_pos in mic_positions]

        mic_signals = []

        # Calculate all mics signals.
        for static_mic_pos, static_mic_orien in zip(static_mic_positions, static_mic_oriens):

            # t1 = time.time()

            # Initialize a room.
            corners = np.array([
                [0, 0], 
                [0, environment["room_width"]], 
                [environment["room_length"], environment["room_width"]], 
                [environment["room_length"], 0]
            ]).T
            # shape: (2, 4)

            room = pra.Room.from_corners(
                corners=corners,
                max_order=self.image_source_order,
            )

            room.extrude(height=environment["room_height"])

            # print("a1", time.time() - t1)
            # t1 = time.time()

            # Add sources to the room.
            for static_src_pos in static_source_positions:
                room.add_source(static_src_pos)

            # Add microphone to the room.
            room.add_microphone(static_mic_pos)

            # Render image sources
            room.image_source_model()

            sources_images = []

            for s in range(sources_num):
                images = room.sources[s].images.T  # (images_num, ndim)
                sources_images.append(images)

            # print("a2", time.time() - t1)
            # t1 = time.time()

            # Go through all sources.
            mic_signal = []

            for source, source_images in zip(sources, sources_images):

                h_list = []

                # Compute the IR of the images of each source.
                for image in source_images:

                    # t1 = time.time()

                    mic_to_img = image - static_mic_pos
                    distance = np.linalg.norm(mic_to_img)

                    # Delay IR.
                    delayed_samples = (distance / self.speed_of_sound) * self.sample_rate
                    distance_gain = 1. / np.clip(a=distance, a_min=0.01, a_max=None)
                    h_delay = distance_gain * fractional_delay_filter(delayed_samples)

                    # print("b1", time.time() - t1)
                    # t1 = time.time()

                    # Mic spatial IR.
                    incident_angle = get_included_angle(a=static_mic_orien, b=mic_to_img)
                    incident_angle_deg = np.rad2deg(incident_angle)
                    h_mic = self.mic_spatial_irs[round(incident_angle_deg)]

                    # print("b2", time.time() - t1)
                    # t1 = time.time()

                    # Composed IR.
                    h_composed = fftconvolve(in1=h_delay, in2=h_mic, mode="full")
                    h_list.append(h_composed)

                    # print("b3", time.time() - t1)

                # Sum the IR of all images.
                h_sum = self.sum_impulse_responses(h_list=h_list)

                # Convolve the source with the summed IR.
                y = fftconvolve(in1=source, in2=h_sum, mode="same")
                mic_signal.append(y)

                # soundfile.write(file="_zz.wav", data=mic_signal, samplerate=24000)

            mic_signal = np.sum(mic_signal, axis=0)
            mic_signals.append(mic_signal)

            # print("a3", time.time() - t1)
            

        mic_signals = np.stack(mic_signals, axis=0)

        return mic_signals

    def sum_impulse_responses(self, h_list):

        max_filter_len = max([len(h) for h in h_list])

        new_h = np.zeros(max_filter_len)

        for h in h_list:
            bgn_sample = max_filter_len // 2 - len(h) // 2
            end_sample = max_filter_len // 2 + len(h) // 2
            new_h[bgn_sample : end_sample + 1] += h

        return new_h

    def sample_agent_positions(self, 
        environment, 
        mic_positions, 
        exclude_area_centers, 
        exclude_area_raidus
    ):

        if self.agent_positions_type == "center_of_mics":

            agent_pos = np.mean(mic_positions, axis=0)
            return agent_pos

        elif self.agent_positions_type == "random_in_environment":
            
            bool_valid = False

            while bool_valid is False:

                agent_pos = self.sample_static_position_in_room(
                    room_length=environment["room_length"], 
                    room_width=environment["room_width"], 
                    room_height=environment["room_height"],
                    room_margin=environment["room_margin"]
                )
                agent_pos = expand_along_frame_axis(x=agent_pos, repeats=self.frames_num)
                # shape: (frames_num, ndim)

                bool_valid = self.verify_position(
                    position=agent_pos, 
                    exclude_area_centers=exclude_area_centers, 
                    exclude_area_raidus=exclude_area_raidus
                )

            return agent_pos

        else:
            raise NotImplementedError

    def sample_source_indexes(self, sources_num, max_positive_rays):

        if sources_num <= max_positive_rays:
            return range(sources_num)
        else:
            return random.sample(range(sources_num), k=max_positive_rays)


    def simulate_agents(self, sources, source_positions, agent_position, environment): 

        sources_num = len(source_positions)

        src_idxes = self.sample_source_indexes(
            sources_num=sources_num, 
            max_positive_rays=self.agent_det_max_pos_rays,
        )

        # Agents for detection
        agents_detect = []

        # for src_pos in source_positions:
        for src_idx in src_idxes:

            src_pos = source_positions[src_idx]

            agent_to_src = src_pos - agent_position
            static_agent_to_src = self.get_static_position(agent_to_src)

            look_at_direction = random_positive_direction(
                source_direction=static_agent_to_src, 
                theta=np.deg2rad(self.source_apprent_diameter_deg / 2)
            )
            look_at_direction = expand_along_frame_axis(
                x=look_at_direction, 
                repeats=self.frames_num
            )

            included_angles = get_included_angle(look_at_direction, agent_to_src)

            look_at_direction_has_source = triangle_function(
                x=included_angles, 
                r=np.deg2rad(self.source_apprent_diameter_deg / 2)
            )

            agent = Agent(
                position=agent_position,
                look_at_direction=look_at_direction,
                look_at_direction_has_source=look_at_direction_has_source
            )

            agents_detect.append(agent)
            
        static_source_poss = [self.get_static_position(pos) for pos in source_positions]

        while len(agents_detect) < self.agent_det_total_rays:

            look_at_direction = random_negative_direction(
                source_directions=static_source_poss, 
                theta=np.deg2rad(self.source_apprent_diameter_deg / 2)
            )
            look_at_direction = expand_along_frame_axis(
                x=look_at_direction, 
                repeats=self.frames_num
            )

            look_at_direction_has_source = np.zeros(self.frames_num)

            agent = Agent(
                position=agent_position,
                look_at_direction=look_at_direction,
                look_at_direction_has_source=look_at_direction_has_source
            )

            agents_detect.append(agent)

        # Agents for distance estimation
        agents_dist = []

        src_idxes = self.sample_source_indexes(
            sources_num=sources_num, 
            max_positive_rays=self.agent_dist_max_pos_rays,
        )

        for src_idx in src_idxes:

            src_pos = source_positions[src_idx]

            agent_to_src = src_pos - agent_position
            static_agent_to_src = self.get_static_position(agent_to_src)

            look_at_direction = random_positive_direction(
                source_direction=static_agent_to_src, 
                theta=np.deg2rad(self.source_apprent_diameter_deg / 2)
            )
            look_at_direction = expand_along_frame_axis(
                x=look_at_direction, 
                repeats=self.frames_num
            )

            src_dist = np.linalg.norm(static_agent_to_src)

            look_at_distance = random_positive_distance(
                source_distance=src_dist, 
                r=self.source_radius
            )
            # look_at_distance = expand_along_frame_axis(
            #     x=look_at_distance, 
            #     repeats=self.frames_num,
            # )

            relative_dist = look_at_distance - src_dist

            look_at_distance_has_source = triangle_function(
                x=relative_dist,
                r=self.source_radius,
            )
            # look_at_distance_has_source = expand_along_frame_axis(
            #     x=look_at_distance_has_source, 
            #     repeats=self.frames_num,
            # )
            look_at_distance = look_at_distance * np.ones(self.frames_num)
            look_at_distance_has_source = look_at_distance_has_source * np.ones(self.frames_num)

            agent = Agent(
                position=agent_position,
                look_at_direction=look_at_direction,
                look_at_distance=look_at_distance,
                look_at_distance_has_source=look_at_distance_has_source
            )

            agents_dist.append(agent)

        while len(agents_dist) < self.agent_dist_total_rays:

            src_pos = random.choice(source_positions)

            agent_to_src = src_pos - agent_position
            static_agent_to_src = self.get_static_position(agent_to_src)

            look_at_direction = random_positive_direction(
                source_direction=static_agent_to_src, 
                theta=np.deg2rad(self.source_apprent_diameter_deg / 2)
            )
            look_at_direction = expand_along_frame_axis(
                x=look_at_direction, 
                repeats=self.frames_num
            )

            src_dist = np.linalg.norm(static_agent_to_src)

            look_at_distance = random_negative_distance(
                source_distances=[src_dist], 
                r=self.source_radius,
                max_dist=environment["max_room_dist"]
            )
            look_at_distance = look_at_distance * np.ones(self.frames_num)

            look_at_distance_has_source = np.zeros(self.frames_num)

            agent = Agent(
                position=agent_position,
                look_at_direction=look_at_direction,
                look_at_distance=look_at_distance,
                look_at_distance_has_source=look_at_distance_has_source
            )

            agents_dist.append(agent)

        # Agents for spatial source separation
        agents_sep = []

        src_idxes = self.sample_source_indexes(
            sources_num=sources_num, 
            max_positive_rays=self.agent_sep_max_pos_rays,
        )

        for src_idx in src_idxes:

            src = sources[src_idx]
            src_pos = source_positions[src_idx]

            agent_to_src = src_pos - agent_position
            static_agent_to_src = self.get_static_position(agent_to_src)

            look_at_direction = random_positive_direction(
                source_direction=static_agent_to_src, 
                theta=np.deg2rad(self.source_apprent_diameter_deg / 2)
            )
            look_at_direction = expand_along_frame_axis(
                x=look_at_direction, 
                repeats=self.frames_num
            )

            # Initialize a room.
            corners = np.array([
                [0, 0], 
                [0, environment["room_width"]], 
                [environment["room_length"], environment["room_width"]], 
                [environment["room_length"], 0]
            ]).T
            # shape: (2, 4)

            room = pra.Room.from_corners(
                corners=corners,
                max_order=self.image_source_order,
            )

            room.extrude(height=environment["room_height"])

            # print("a1", time.time() - t1)
            # t1 = time.time()

            # Add sources to the room.
            static_src_pos = self.get_static_position(src_pos)
            room.add_source(static_src_pos)

            static_agent_pos = self.get_static_position(agent_position)
            
            # Add microphone to the room.
            room.add_microphone(static_agent_pos)

            # Render image sources
            room.image_source_model()

            assert len(room.sources) == 1
            source_images = room.sources[0].images.T

            h_list = []

            for src_img in source_images:

                # t1 = time.time()

                agent_to_img = src_img - static_agent_pos
                dist = np.linalg.norm(agent_to_img)

                # Delay IR.
                delayed_samples = (dist / self.speed_of_sound) * self.sample_rate
                distance_gain = 1. / np.clip(a=dist, a_min=0.01, a_max=None)
                h_delay = distance_gain * fractional_delay_filter(delayed_samples)

                # print("b1", time.time() - t1)
                # t1 = time.time()

                # Composed IR.
                h_list.append(h_delay)

                # print("b3", time.time() - t1)

            # Sum the IR of all images.
            h_sum = self.sum_impulse_responses(h_list=h_list)

            # Convolve the source with the summed IR.
            direct_wav = fftconvolve(in1=src, in2=h_list[0], mode="same")
            reverb_wav = fftconvolve(in1=src, in2=h_sum, mode="same")
            

            # soundfile.write(file="_zz.wav", data=y, samplerate=self.sample_rate)
            # soundfile.write(file="_zz0.wav", data=y0, samplerate=self.sample_rate)

            agent = Agent(
                position=agent_position,
                look_at_direction=look_at_direction,
                look_at_direction_direct_waveform=direct_wav,
                look_at_direction_reverb_waveform=reverb_wav
            )

            agents_sep.append(agent)

        while len(agents_sep) < self.agent_sep_total_rays:

            look_at_direction = random_negative_direction(
                source_directions=static_source_poss, 
                theta=np.deg2rad(self.source_apprent_diameter_deg / 2)
            )
            look_at_direction = expand_along_frame_axis(
                x=look_at_direction, 
                repeats=self.frames_num
            )

            agent = Agent(
                position=agent_position,
                look_at_direction=look_at_direction,
                look_at_direction_has_source=look_at_direction_has_source
            )

            direct_wav = np.zeros(self.segment_samples)
            reverb_wav = np.zeros(self.segment_samples)

            agent = Agent(
                position=agent_position,
                look_at_direction=look_at_direction,
                look_at_direction_direct_waveform=direct_wav,
                look_at_direction_reverb_waveform=reverb_wav
            )

            agents_sep.append(agent)

        agents = agents_detect + agents_dist + agents_sep
            
        return agents


    def __len__(self):
        return 10000


class Agent:
    def __init__(self, 
        position, 
        look_at_direction, 
        look_at_direction_has_source=None,
        look_at_direction_direct_waveform=None,
        look_at_direction_reverb_waveform=None,
        look_at_distance=None, 
        look_at_distance_has_source=None,
    ):
        self.position = position
        self.look_at_direction = look_at_direction
        self.look_at_direction_has_source = look_at_direction_has_source
        self.look_at_direction_direct_waveform = look_at_direction_direct_waveform
        self.look_at_direction_reverb_waveform = look_at_direction_reverb_waveform
        self.look_at_distance = look_at_distance
        self.look_at_distance_has_source = look_at_distance_has_source



def expand_along_frame_axis(x, repeats):
    
    output = np.repeat(
        a=np.expand_dims(x, axis=-2), 
        repeats=repeats, 
        axis=-2
    )
    return output