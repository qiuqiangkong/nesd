import numpy as np
import pyroomacoustics as pra
from scipy.signal import fftconvolve

from nesd.utils import fractional_delay_filter, get_included_angle


class ImageSourceEngine:
    def __init__(self, 
        environment, 
        source_positions,
        mic_position, 
        mic_orientation,
        mic_spatial_irs,
        image_source_order,
        speed_of_sound,
        sample_rate,
        compute_direct_ir_only,
    ):

        self.environment = environment
        self.source_positions = source_positions
        self.mic_position = mic_position
        self.mic_orientation = mic_orientation
        self.mic_spatial_irs = mic_spatial_irs
        self.image_source_order = image_source_order
        self.speed_of_sound = speed_of_sound
        self.sample_rate = sample_rate
        self.compute_direct_ir_only = compute_direct_ir_only
        
        self.min_distance = 0.1
        
    def compute_spatial_ir(self):

        srcs_num = len(self.source_positions)

        room = self.build_shoebox_room(self.environment)

        # Add sources to the room.
        for src_pos in self.source_positions:
            room.add_source(src_pos)

        # Add microphone to the room.
        room.add_microphone(self.mic_position)

        # Render image sources
        room.image_source_model()

        srcs_images = []

        for s in range(srcs_num):
            images = room.sources[s].images.T  # (images_num, ndim)
            srcs_images.append(images)

        srcs_h_direct = []
        srcs_h_reverb = []

        for src_images in srcs_images:

            h_list = []

            # Compute the IR of the images of each source.
            for img in src_images:

                mic_to_img = img - self.mic_position
                distance = np.linalg.norm(mic_to_img)

                # Delay IR.
                delayed_samples = (distance / self.speed_of_sound) * self.sample_rate
                distance_gain = 1. / np.clip(a=distance, a_min=self.min_distance, a_max=None)
                h_delay = distance_gain * fractional_delay_filter(delayed_samples)

                if self.mic_spatial_irs:
                    # Mic spatial IR.
                    incident_angle = get_included_angle(a=self.mic_orientation, b=mic_to_img)
                    incident_angle_deg = np.rad2deg(incident_angle)
                    h_mic = self.mic_spatial_irs[round(incident_angle_deg)]

                    # Composed IR.
                    h_composed = fftconvolve(in1=h_delay, in2=h_mic, mode="full")

                else:
                    h_composed = h_delay
                
                h_list.append(h_composed)

            # Sum the IR of all images.
            h_direct = h_list[0]
            h_reverb = self.sum_impulse_responses(h_list=h_list)

            srcs_h_direct.append(h_direct)
            srcs_h_reverb.append(h_reverb)

        return srcs_h_direct, srcs_h_reverb


    def build_shoebox_room(self, environment):

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

        return room

    def sum_impulse_responses(self, h_list):

        max_filter_len = max([len(h) for h in h_list])

        new_h = np.zeros(max_filter_len)

        for h in h_list:
            bgn_sample = max_filter_len // 2 - len(h) // 2
            end_sample = max_filter_len // 2 + len(h) // 2
            new_h[bgn_sample : end_sample + 1] += h

        return new_h