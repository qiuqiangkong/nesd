import argparse
import librosa
import numpy as np
import yaml
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import h5py
import os
import soundfile

from nesd.utils import sph2cart, int16_to_float32, normalize

methods = ["MUSIC", "FRIDA", "WAVES", "TOPS", "CSSM", "SRP", "NormMUSIC"]

if __name__ == "__main__":

    sample_rate = 24000
    random_state = np.random.RandomState(1234)
    sources_num = 2
    
    corners = np.array([
        [8, 8], 
        [0, 8], 
        [0, 0], 
        [8, 0],
    ]).T
    height = 4

    materials = None

    room = pra.Room.from_corners(
        corners=corners,
        fs=sample_rate,
        materials=materials,
        max_order=3,
        ray_tracing=False,
        air_absorption=False,
    )
    room.extrude(
        height=height, 
        materials=materials
    )

    if False:
        audio_path = './resources/p360_396_mic1.flac.wav'
        source, fs = librosa.load(audio_path, sr=sample_rate, mono=True)
        room.add_source(position=np.array([6, 6, 3]), signal=source)
    else:
        for _ in range(sources_num):
            # source_position = np.array((
            #     random_state.uniform(low=0, high=8),
            #     random_state.uniform(low=0, high=8),
            #     random_state.uniform(low=0, high=4),
            # ))

            source_position = np.array([4, 4, 2]) + normalize(random_state.uniform(low=-1, high=1, size=3))

            hdf5s_dir = "/home/tiger/workspaces/nesd2/hdf5s/vctk/sr=24000/test"
            hdf5_names = sorted(os.listdir(hdf5s_dir))
            hdf5_name = random_state.choice(hdf5_names)
            hdf5_path = os.path.join(hdf5s_dir, hdf5_name)
            with h5py.File(hdf5_path, 'r') as hf:
                source = int16_to_float32(hf['waveform'][:])

            room.add_source(position=source_position, signal=source)
            # soundfile.write(file='_zz.wav', data=source, samplerate=sample_rate)


    # Add microphone
    mic_yaml = "ambisonic.yaml"
    with open(mic_yaml, 'r') as f:
        mics_meta = yaml.load(f, Loader=yaml.FullLoader)

    mic_center_position = np.array([4, 4, 2])

    mic_positions = []

    for mic_meta in mics_meta:
        relative_mic_posision = np.array(sph2cart(
            r=mic_meta['radius'], 
            azimuth=mic_meta['azimuth'], 
            colatitude=mic_meta['colatitude']
        ))

        mic_position = mic_center_position + relative_mic_posision
        mic_positions.append(mic_position)

        directivity_object = None
        room.add_microphone(loc=mic_position, directivity=directivity_object)

        # print(mic_position)

    mic_positions = np.stack(mic_positions, axis=1)
    room.compute_rir()

    room.simulate()
    
    