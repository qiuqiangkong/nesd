import os
import librosa
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = ['alarm', 'crying baby', 'crash', 'barking dog', 'female scream', 'female speech', 'footsteps', 'knocking on door', 'male scream', 'male speech', 'ringing phone', 'piano']


def add():

    begin_sec = 0.
    end_sec = 10.

    audios_dir = "/home/tiger/datasets/dcase2021/task3/mic_dev/dev-train"
    csvs_dir = "/home/tiger/datasets/dcase2021/task3/metadata_dev/dev-train"

    audio_names = sorted(os.listdir(audios_dir))

    for audio_index, audio_name in enumerate(audio_names):
        print(audio_index) 

        audio_path = os.path.join(audios_dir, audio_name)
        csv_path = os.path.join(csvs_dir, "{}.csv".format(pathlib.Path(audio_name).stem))

        audio, fs = librosa.load(audio_path, sr=None, mono=True)

        mel = librosa.feature.melspectrogram(audio, sr=fs, n_fft=1024, hop_length=240, n_mels=64, fmin=0, fmax=fs//2).T

        begin_frame_100fps = int(begin_sec * 100)
        end_frame_100fps = int(end_sec * 100) + 1
        mel = mel[begin_frame_100fps : end_frame_100fps : 10, :]

        df = pd.read_csv(csv_path, sep=',', header=None)
        frame_indexes = df[0].values
        class_indexes = df[1].values
        event_indexes = df[2].values
        azimuths = df[3].values
        elevations = df[4].values

        classes_num = 12
        gt_mat = np.zeros((601, classes_num))

        for n in range(len(frame_indexes)):
            frame_index = frame_indexes[n]
            class_id = class_indexes[n]
            gt_mat[frame_index, class_id] = 1
            # gt_mat[frame_index : frame_index + 2, class_id] = 1

        begin_frame_10fps = int(begin_sec * 10)
        end_frame_10fps = int(end_sec * 10) + 1
        gt_mat = gt_mat[begin_frame_10fps : end_frame_10fps, :]

        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].matshow(np.log(mel).T, origin='lower', aspect='auto', cmap='jet')
        axs[1].matshow(gt_mat.T, origin='lower', aspect='auto', cmap='jet')
        axs[0].xaxis.set_ticks(np.arange(0, 101, 10))
        axs[0].xaxis.set_ticklabels(np.arange(11))
        axs[1].yaxis.set_ticks(np.arange(classes_num))
        axs[1].yaxis.set_ticklabels([lb[0:10] for lb in LABELS], fontsize=8)
        axs[0].xaxis.grid(color='k', linestyle='solid', linewidth=1)
        axs[1].xaxis.grid(color='w', linestyle='solid', linewidth=1)
        axs[1].yaxis.grid(color='w', linestyle='solid', linewidth=0.2)
        os.makedirs('_tmp', exist_ok=True)
        plt.savefig('_tmp/_zz_{:03d}.pdf'.format(audio_index))

        if audio_index == 100:
            break
        
    from IPython import embed; embed(using=False); os._exit(0)


if __name__ == '__main__':
    add()