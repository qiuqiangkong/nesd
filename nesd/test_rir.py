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

from nesd.utils import cart2sph


def add():
    rirdata_mat_path = os.path.join("/home/qiuqiangkong/datasets/dcase2022/tau-srir/TAU-SRIR_DB/rirdata.mat")
    rir_traj = io.loadmat(rirdata_mat_path)['rirdata']
    
    # rir_traj[0][0][1][8][0][2][1][0][0].shape

    mat_path = "/home/qiuqiangkong/datasets/dcase2022/tau-srir/TAU-SRIR_DB/rirs_10_tc352.mat"
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

    from IPython import embed; embed(using=False); os._exit(0)


def add2():

    rirdata_mat_path = os.path.join("/home/qiuqiangkong/datasets/dcase2022/tau-srir/TAU-SRIR_DB/rirdata.mat")
    rir_traj = io.loadmat(rirdata_mat_path)['rirdata']

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
    

    for i in range(2):

        for j in range(9):
            a1 = data[1][8][0][2][i][j][0]   # (360, 3)
            r, azi, col = cart2sph(x=a1[:, 0], y=a1[:, 1], z=a1[:, 2])
            plt.scatter(azi, col)
            plt.xlim(0, 2 * math.pi)
            plt.ylim(0, math.pi)
    

    plt.savefig("_zz.pdf")

    from IPython import embed; embed(using=False); os._exit(0)


def add3():

    mat_path = os.path.join("/home/qiuqiangkong/datasets/dcase2022/tau-srir/TAU-SRIR_DB/measinfo.mat")
    x = io.loadmat(mat_path)

    x['measinfo']   # (9, 1)
    x['measinfo'][0, 0]
    from IPython import embed; embed(using=False); os._exit(0)


# Create RIR
def add4():

    traj_mat_path = os.path.join("/home/qiuqiangkong/datasets/dcase2022/tau-srir/TAU-SRIR_DB/rirdata.mat")

    mats_dir = "/home/qiuqiangkong/datasets/dcase2022/tau-srir/TAU-SRIR_DB"

    hdf5s_dir = "/home/qiuqiangkong/workspaces/nesd2/tau-srir"
    os.makedirs(hdf5s_dir, exist_ok=True)

    path_mapping = {
        0: "rirs_01_bomb_shelter.mat",
        1: "rirs_02_gym.mat",
        2: "rirs_03_pb132.mat",
        3: "rirs_04_pc226.mat",
        4: "rirs_05_sa203.mat",
        5: "rirs_06_sc203.mat",
        6: "rirs_08_se203.mat",
        7: "rirs_09_tb103.mat",
        8: "rirs_10_tc352.mat",
    }

    rir_traj = io.loadmat(traj_mat_path)['rirdata']

    data1 = rir_traj[0][0]
    # data[0]   # 24000
    # data[2]   # 0.042
    # data[3]   # (4, 2)
    # data[4]   # "sn3d"
    # data[5]   # "wyzx"
    # data[1]   # (9,) number of rooms

    rooms_num = 9

    # data[1][8][0][0]    # "tc352"
    # data[1][8][0][1]    # 2019
    # data[1][8][0][3]    # 360, 360, ...
    # data[1][8][0][2]    # (2, 9)

    data_list = []

    for room_index in range(rooms_num):

        print(room_index)
        
        mat_path = os.path.join(mats_dir, path_mapping[room_index])
        rir_data = mat73.loadmat(mat_path)

        data2 = data1[1][room_index][0][2]

        traj1s_num = len(data2)

        for traj1 in range(traj1s_num):
            # print(traj1)

            traj2s_num = len(data2[traj1])

            for traj2 in range(traj2s_num):
                # print(traj2) 
                # from IPython import embed; embed(using=False); os._exit(0)

                traj_tensor = data2[traj1][traj2][0]  # (360, 3)

                rir_tensor = rir_data['rirs']['mic'][traj1][traj2]   # (7200, 4, 360)
                rir_tensor = rir_tensor.transpose(2, 1, 0)  # (360, 4, 7200)

                data = {
                    "room_name": path_mapping[room_index],
                    "traj": traj_tensor,
                    "rir": rir_tensor,
                }
                data_list.append(data)

                hdf5_path = Path(hdf5s_dir, "{}_traj_{}_{}.h5".format(Path(mat_path).stem, traj1, traj2))

                with h5py.File(hdf5_path, 'w') as hf:
                    hf.create_dataset('traj', data=data["traj"], dtype=np.float32)
                    hf.create_dataset('rir', data=data["rir"], dtype=np.float32)

                print("Write out to {}".format(hdf5_path))

    # pickle.dump(data_list, open("rir_data.pkl", "wb"))

    from IPython import embed; embed(using=False); os._exit(0)


def add5():

    hdf5_path = "/home/qiuqiangkong/workspaces/nesd2/tau-srir/rirs_10_tc352_traj_1_7.h5"

    with h5py.File(hdf5_path, 'r') as hf:

        i = 99
        traj = hf['traj'][i]
        rir = hf['rir'][i]

    audio_path = "resources/p226_001.wav"
    audio, _ = librosa.load(path=audio_path, sr=24000, mono=True)

    from scipy.signal import fftconvolve
    y = fftconvolve(in1=audio, in2=rir[0])

    x = librosa.util.fix_length(data=audio, size=y.shape[0], axis=0)

    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(x[16000:18000])
    axs[1].plot(y[16000:18000])
    plt.savefig("_zz.pdf")

    soundfile.write(file="_zz.wav", data=y, samplerate=24000)

    from IPython import embed; embed(using=False); os._exit(0) 


if __name__ == '__main__':

    # add()
    add2()
    # add3()
    # add4()
    # add5()