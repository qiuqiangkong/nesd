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

    from IPython import embed; embed(using=False); os._exit(0)


if __name__ == '__main__':

    add()