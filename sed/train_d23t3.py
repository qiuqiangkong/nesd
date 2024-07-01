import time
import random
import librosa
import numpy as np
import soundfile
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import torch.optim as optim
# from data.maestro import Maestro
from sed.data.collate import collate_fn
# from sed.models.crnn import CRnn
from data.dcase2023_task3 import DCASE2023Task3
from sed.models.sed import Cnn, CRnn
from tqdm import tqdm
import argparse
import wandb
import torch
import torch.nn.functional as F

# from nesd.evaluate.dcase2019_task3 import LABELS, LABELS_NUM, LB_TO_ID, ID_TO_LB

# from data.tokenizers import Tokenizer
# from losses import regress_onset_offset_frame_velocity_bce, regress_onset_offset_frame_velocity_bce2

LABELS = [
    "Female speech, woman speaking",
    "Male speech, man speaking",
    "Clapping",
    "Telephone",
    "Laughter",
    "Domestic sounds",
    "Walk, footsteps",
    "Door, open or close",
    "Music",
    "Musical instrument",
    "Water tap, faucet",
    "Bell",
    "Knock",
]

LB_TO_ID = {lb: id for id, lb in enumerate(LABELS)}
ID_TO_LB = {id: lb for id, lb in enumerate(LABELS)}
LABELS_NUM = len(LABELS)


def train(args):

    # Arguments
    model_name = args.model_name

    # Default parameters
    device = "cuda"
    batch_size = 16
    num_workers = 32
    # num_workers = 0
    save_step_frequency = 1000
    training_steps = 20000
    debug = False
    filename = Path(__file__).stem
    segment_seconds = 4.
    sample_rate = 24000
    wandb_log = False

    root = "/datasets/dcase2023/task3"
    classes_num = LABELS_NUM

    if wandb_log:
        wandb.init(project="nsed_sed")

    checkpoints_dir = Path("./checkpoints", filename, model_name)
    

    train_dataset = DCASE2023Task3(
        root=root,
        split="train",
        segment_seconds=segment_seconds,
        sample_rate=sample_rate,
        lb_to_id=LB_TO_ID
    )

    # Sampler
    train_sampler = Sampler(dataset_size=len(train_dataset))

    # Dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers, 
        pin_memory=True
    )

    # Model
    model = get_model(model_name, classes_num)
    model.to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Create checkpoints directory
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        audio = data["audio"].to(device)
        target = data["target"].to(device)

        # soundfile.write(file="_zz.wav", data=audio.cpu().numpy()[0] * 10, samplerate=16000)

        # fig, axs = plt.subplots(2,1, sharex=True)
        # axs[0].matshow(data["target"][0].cpu().numpy().T, origin='lower', aspect='auto', cmap='jet')
        # axs[0].yaxis.set_ticks(np.arange(classes_num))
        # axs[0].yaxis.set_ticklabels(LABELS)
        # plt.savefig("_zz.pdf")
        # from IPython import embed; embed(using=False); os._exit(0)

        # Play the audio.
        if debug:
            play_audio(mixture, target)

        model.train()
        frame_roll = model(audio=audio)

        loss = bce_loss(frame_roll, target)

        optimizer.zero_grad()    
        loss.backward()

        optimizer.step()

        if step % 100 == 0:
            print("step: {}, loss: {:.3f}".format(step, loss.item()))

            if wandb_log:
                wandb.log({"train loss": loss.item()})

        # Save model
        if step % save_step_frequency == 0:
            checkpoint_path = Path(checkpoints_dir, "step={}.pth".format(step))
            torch.save(model.state_dict(), checkpoint_path)
            print("Save model to {}".format(checkpoint_path))

            checkpoint_path = Path(checkpoints_dir, "latest.pth")
            torch.save(model.state_dict(), Path(checkpoint_path))
            print("Save model to {}".format(checkpoint_path))

        if step == training_steps:
            break


def get_model(model_name, classes_num):
    if model_name == "Cnn":
        return Cnn(classes_num=classes_num)
    elif model_name == "CRnn":
        return CRnn(classes_num=classes_num)
    else:
        raise NotImplementedError


class Sampler:
    def __init__(self, dataset_size):
        self.indexes = list(range(dataset_size))
        random.shuffle(self.indexes)
        
    def __iter__(self):

        pointer = 0

        while True:

            if pointer == len(self.indexes):
                random.shuffle(self.indexes)
                pointer = 0
                
            index = self.indexes[pointer]
            pointer += 1

            yield index


def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)


def play_audio(mixture, target):
    soundfile.write(file="tmp_mixture.wav", data=mixture[0].cpu().numpy().T, samplerate=44100)
    soundfile.write(file="tmp_target.wav", data=target[0].cpu().numpy().T, samplerate=44100)
    from IPython import embed; embed(using=False); os._exit(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="CRnn")
    args = parser.parse_args()

    train(args)