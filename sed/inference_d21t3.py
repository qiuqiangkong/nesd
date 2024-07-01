import torch
import time
import librosa
import numpy as np
import soundfile
from pathlib import Path
import torch.optim as optim
# from data.musdb18hq import Musdb18HQ
# from data.collate import collate_fn
from sed.models.sed import Cnn, CRnn
from tqdm import tqdm
import pickle
# import museval
import argparse
import matplotlib.pyplot as plt

from sed.train_d21t3 import LABELS, LB_TO_ID, ID_TO_LB, LABELS_NUM
from sed.train_d21t3 import get_model
from sed.data.dcase2021_task3 import read_dcase2021_task3_csv


dataset_dir = "/datasets/dcase2021/task3"
workspace = "/home/qiuqiangkong/workspaces/nesd"
select = "1"

if select == "1": 

    audio_paths = [Path(dataset_dir, "mic_dev", "dev-test", "fold6_room1_mix001.wav")]
    gt_csv_path = Path(dataset_dir, "metadata_dev", "dev-test", "fold6_room1_mix001.csv")

    sed_dir = Path(workspace, "results/dcase2021_task3/sed")
    sed_paths = [Path(sed_dir, "fold6_room1_mix001.pkl")]


def inference(args):

    # Arguments
    model_name = args.model_name

    # Default parameters
    segment_seconds = 2.
    sample_rate = 24000
    device = "cuda"
    
    segment_samples = int(segment_seconds * sample_rate)
    classes_num = LABELS_NUM

    # Load checkpoint
    # checkpoint_path = Path("checkpoints", model_name, "latest.pth")
    checkpoint_path = "checkpoints/train_d21t3/{}/step=10000.pth".format(model_name)
    # from IPython import embed; embed(using=False); os._exit(0)

    Path(sed_dir).mkdir(parents=True, exist_ok=True)

    model = get_model(model_name, classes_num)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    for audio_path in audio_paths:

        audio, _ = librosa.load(path=audio_path, sr=sample_rate, mono=True)
        # (channels_num, audio_samples)

        bgn = 0
        segment_samples = int(segment_seconds * sample_rate)
        audio_samples = audio.shape[0]

        outputs = []

        # Do separation
        while bgn < audio_samples:
            
            segment = audio[bgn : bgn + segment_samples]
            segment = librosa.util.fix_length(data=segment, size=segment_samples, axis=0)
            segment = torch.Tensor(segment).to(device)

            # Separate a segment
            with torch.no_grad():
                model.eval()
                output = model(segment[None, :])[0][0 : -1: 10]
                outputs.append(output.cpu().numpy())

            bgn += segment_samples
            
        outputs = np.concatenate(outputs, axis=0)

        targets = read_dcase2021_task3_csv(gt_csv_path, LB_TO_ID)
        targets = targets[0 :: 10][0 : len(outputs)]

        fig, axs = plt.subplots(2,1, sharex=True)
        axs[0].matshow(targets.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
        axs[1].matshow(outputs.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
        for i in range(2):
            axs[i].yaxis.set_ticks(np.arange(len(LABELS)))
            axs[i].yaxis.set_ticklabels(LABELS)
        plt.savefig("_zz_sed.pdf")

        pickle_path = Path(sed_dir, "{}.pkl".format(Path(audio_path).stem))
        pickle.dump(outputs, open(pickle_path, "wb"))
        print("Write out to {}".format(pickle_path))

        soundfile.write(file="_zz.wav", data=audio, samplerate=sample_rate)

        from IPython import embed; embed(using=False); os._exit(0)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Cnn")
    args = parser.parse_args()

    inference(args)