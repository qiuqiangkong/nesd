import numpy as np
from pathlib import Path
import librosa
import matplotlib.pyplot as plt


def add():

    sample_rate = 24000
    segment_samples = 24000 * 2

    audios_dir = "/home/qiuqiangkong/workspaces/nesd/audios/tau_noise/train"
    # audios_dir = "/home/qiuqiangkong/workspaces/nesd/audios/vctk_2s_segments/train"
    # audios_dir = "/datasets/dcase2019/task3/mic_dev"

    audio_paths = sorted(list(Path(audios_dir).glob("*.wav")))

    energies = []

    for audio_index, audio_path in enumerate(audio_paths):
        print(audio_index)

        audio, _ = librosa.load(path=audio_path, sr=sample_rate, mono=True)
        frames = librosa.util.frame(audio, frame_length=segment_samples, hop_length=segment_samples).T

        energies.extend([energy(frame) for frame in frames])

    energies = np.array(energies)
    dbs = energy_to_db(energies)
    median_db = np.median(dbs)
    print(median_db)

    hist, bin_edges = np.histogram(dbs, bins=20)
    plt.title("median DB: {:.3f}".format(median_db))
    plt.stem(bin_edges[:-1], hist)
    plt.savefig("_zz.pdf")

    from IPython import embed; embed(using=False); os._exit(0)


def energy(x):
    return np.mean(x ** 2)


def energy_to_db(energy):
    db = 10 * np.log10(energy)
    return db


if __name__ == "__main__":

    add()