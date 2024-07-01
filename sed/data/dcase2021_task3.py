import os
import torch
from pathlib import Path
import pandas as pd
import random
import soundfile
import re
import time
import librosa
import torchaudio
import numpy as np
import yaml
import matplotlib.pyplot as plt


class DCASE2021Task3:
    def __init__(
        self, 
        root: str = None, 
        split: str = "train",
        segment_seconds: float = 4.,
        sample_rate: float = 24000,
        lb_to_id=None,
    ):

        self.root = root
        self.split = split
        self.segment_seconds = segment_seconds
        self.sample_rate = sample_rate
        self.lb_to_id = lb_to_id

        self.frames_per_sec = 100

        audios_dir = Path(self.root, "mic_dev")
        audio_paths = list(Path(audios_dir).glob("*.wav"))

        if split == "train":
            self.audios_dir = Path(self.root, "mic_dev/dev-train")
            self.metas_dir = Path(self.root, "metadata_dev/dev-train")
        
        elif split == "test":
            self.audios_dir = Path(self.root, "mic_dev/dev-test")
            self.metas_dir = Path(self.root, "metadata_eval/dev-train")

        else:
            raise NotImplementedError

        self.audio_paths = sorted(list(Path(self.audios_dir).glob("*.wav")))
        self.audios_num = len(self.audio_paths)

    def __getitem__(self, index):

        audio_path = self.audio_paths[index]
        meta_path = Path(self.metas_dir, "{}.csv".format(audio_path.stem))

        duration = librosa.get_duration(path=audio_path)
        segment_start_time = random.uniform(0, duration - self.segment_seconds)

        # Load audio.
        audio = self.load_audio(audio_path, segment_start_time)

        targets = self.load_targets(meta_path, segment_start_time)

        data = {
            "audio": audio,
            "target": targets
        }
        # print(audio.shape, targets.shape)

        return data

    def load_audio(self, audio_path, segment_start_time):

        orig_sr = librosa.get_samplerate(audio_path)

        segment_start_sample = int(segment_start_time * orig_sr)
        segment_samples = int(self.segment_seconds * orig_sr)

        audio, fs = torchaudio.load(
            audio_path, 
            frame_offset=segment_start_sample, 
            num_frames=segment_samples
        )
        # (channels, audio_samples)

        audio = torch.mean(audio, dim=0)
        # shape: (audio_samples,)

        audio = torchaudio.functional.resample(
            waveform=audio, 
            orig_freq=orig_sr, 
            new_freq=self.sample_rate
        )
        # shape: (audio_samples,)

        return audio

    def load_targets(self, meta_path, segment_start_time):

        start_frame = round(segment_start_time * self.frames_per_sec)
        segment_frames = int(self.segment_seconds * self.frames_per_sec)
        end_frame = start_frame + segment_frames

        targets = read_dcase2021_task3_csv(meta_path, self.lb_to_id)

        tmp = targets[start_frame : end_frame + 1, :]

        return tmp

    def __len__(self):
        return self.audios_num


def read_dcase2021_task3_csv(meta_path, lb_to_id):

    df = pd.read_csv(meta_path, sep=',', header=None)

    frame_indexes = df[0].values
    class_indexes = df[1].values
    azimuths = df[3].values
    elevations = df[4].values
    distances = np.zeros(len(frame_indexes))

    azimuths = np.deg2rad(np.array(azimuths))
    elevations = np.deg2rad(np.array(elevations))

    classes_num = len(lb_to_id)
    targets = np.zeros((7000, classes_num))

    for i in range(len(frame_indexes)):
        frame_index = frame_indexes[i]
        class_index = class_indexes[i]
        for j in range(10):
            targets[frame_index * 10 + j, class_index] = 1

    return targets