import mat73
from scipy import io
import os
import matplotlib.pyplot as plt
import math
import h5py
import numpy as np
import pickle
import librosa
import soundfile
from pathlib import Path
from scipy.signal import fftconvolve

from nesd.utils import sph2cart, fractional_delay_filter


def add():
    sample_rate = 24000

    rirdata_mat_path = os.path.join("/datasets/tau-srir/TAU-SRIR_DB/rirdata.mat")
    rir_traj = io.loadmat(rirdata_mat_path)['rirdata']
    
    # rir_traj[0][0][1][8][0][2][1][0][0].shape

    mat_path = "/datasets/tau-srir/TAU-SRIR_DB/rirs_10_tc352.mat"
    rir_data = mat73.loadmat(mat_path)
    # print("Mic pos: {}".format(rir_traj[0][0][3]))


    data = rir_traj[0][0]
    # data[0]   # 24000
    # data[2]   # 0.042
    # data[3]   # (4, 2)
    # data[4]   # "sn3d"
    # data[5]   # "wyzx"
    # data[1]   # (9,) number of rooms

    data[1][8][0][0]    # "tc352"
    data[1][8][0][1]    # 2019
    data[1][8][0][3]    # 360, 360, ...

    data[1][8][0][2]    # (2, 9)
    data[1][8][0][2][0][0][0]   # (360, 3)

    ###
    traj1 = 0   # 2
    traj2 = 0   # 9
    a1 = rir_data['rirs']['mic'][traj1][traj2]   # (7200, 4, 360)
    plt.plot(a1[:, 0, 0])
    plt.savefig("_zz.pdf")

    audio_path = "/datasets/vctk/wav48/p225/p225_366.wav"
    audio, fs = librosa.load(path=audio_path, sr=sample_rate, mono=True)
    audio = audio[sample_rate : sample_rate * 3]
    

    n = 0
    ys = []

    for i  in range(4):
        y = fftconvolve(in1=audio, in2=a1[:, i, n], mode="same")
        ys.append(y)

    ys = np.stack(ys, axis=0)

    soundfile.write(file="_uu.wav", data=ys.T, samplerate=sample_rate)
    ys0 = np.tile(ys, 30)
    soundfile.write(file="_uu0.wav", data=ys0.T, samplerate=sample_rate)
    from IPython import embed; embed(using=False); os._exit(0)

    fig, axs = plt.subplots(4, 1, sharex=True)
    for i in range(4):
        axs[i].stem(a1[:, i, n][0:50])
    plt.savefig("_zz.pdf")

    


def add2():

    sample_rate = 24000
    audio_path = "/datasets/vctk/wav48/p225/p225_366.wav"
    audio, _ = librosa.load(path=audio_path, sr=sample_rate, mono=True)

    tmp = sph2cart(
        azimuth=np.deg2rad(45),
        elevation=np.deg2rad(35),
        r=0.042,
    )[0]
    delayed_samples = (tmp * 2) / 343 * sample_rate

    h = fractional_delay_filter(delayed_samples)
    delayed_audio = fftconvolve(in1=audio, in2=h, mode="same")

    y = np.array([
        audio,
        audio,
        delayed_audio,
        delayed_audio,
    ])

    # fig, axs = plt.subplots(2, 1, sharex=True)
    # axs[0].stem(audio[0:100])
    # axs[1].stem(delayed_audio[0:100])
    # plt.savefig("_zz.pdf")

    soundfile.write(file="_uu3.wav", data=y.T, samplerate=sample_rate)

    from IPython import embed; embed(using=False); os._exit(0)
    # fractional_delay_filter()



if __name__ == '__main__':

    add()