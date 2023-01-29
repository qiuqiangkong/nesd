import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import math
import librosa
import soundfile
import time


wall_mat = {
    "description": "Example wall material",
    "coeffs": [0.1, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05],
    "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000, 16000],
}
r = 0.2
wall_mat2 = {
    "description": "Example wall material",
    "coeffs": [r, r, r, r, r, r, r, r, ],
    "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000, 16000],
}

def add():
    sample_rate = 24000
    # sample_rate = 8000

    select = '3'

    if select == '1':
        t = np.arange(sample_rate)
        source = np.cos(t * 2 * math.pi * 440 / sample_rate) * 0.1
    elif select == '2':
        t = np.arange(sample_rate * 3)
        source = np.zeros(len(t))
        source[100] = 1
    elif select == '3':
        audio_path = './resources/p360_396_mic1.flac.wav'
        source, fs = librosa.load(audio_path, sr=sample_rate, mono=True)

    corners = np.array([
        [8, 8], 
        [0, 8], 
        [0, 0], 
        [8, 0],
    ]).T
    height = 4

    materials = pra.Material(energy_absorption=wall_mat2)
    # materials = None

    is_raytracing = True

    t1 = time.time()
    room = pra.Room.from_corners(
        corners=corners,
        fs=sample_rate,
        materials=materials,
        max_order=3,
        ray_tracing=is_raytracing,
        air_absorption=False,
    )
    room.extrude(
        height=height, 
        materials=materials
    )

    if is_raytracing:
        room.set_ray_tracing(
            n_rays=1000,
            receiver_radius=0.5,
            energy_thres=1e-7,
            time_thres=1.0,
            hist_bin_size=0.004,
        )

    room.add_source(position=np.array([3.8, 7.43, 2.7]), signal=source)

    # if True:
    #     zenith = math.pi / 2
    #     directivity_object = pra.CardioidFamily(
    #         orientation=pra.DirectionVector(azimuth=azimuth, colatitude=zenith, degrees=False),
    #         pattern_enum=pra.DirectivityPattern.CARDIOID,
    #     )

    directivity_object = None
    room.add_microphone(loc=np.array([4, 4, 2]), directivity=directivity_object)

    room.compute_rir()

    room.simulate()
    print(time.time() - t1)
    
    #
    mic_signals = room.mic_array.signals[0]

    soundfile.write(file='_zz.wav', data=mic_signals, samplerate=sample_rate)
    fig, axs = plt.subplots(2,1, sharex=True)
    axs[0].stem(source[0:300])
    axs[1].stem(mic_signals[0:1000])
    plt.savefig('_zz.pdf')
    from IPython import embed; embed(using=False); os._exit(0)

    np.argmax(mic_signals)

    # pad 40 in the beginning
    # gain = 1/r

    # 210
    # 280


if __name__ == '__main__':

    add()