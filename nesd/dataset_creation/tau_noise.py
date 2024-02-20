import argparse
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor
from typing import NoReturn, List
import h5py
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
import soundfile


TEST_ROOMS = ["09_tb103_tietotalo_lecturehall", "10_tc352_tietotalo_meetingroom"]


def process_audio_into_clips(args) -> NoReturn:
    r"""Pack (resampled) audio files into hdf5 files to speed up loading.

    Args:
        dataset_dir: str
        split: str, 'train' | 'test'
        hdf5s_dir: str, directory to write out hdf5 files
        sample_rate: int
        channels_num: int
        mono: bool

    Returns:
        NoReturn
    """

    # arguments & parameters
    audios_dir = args.audios_dir
    split = args.split
    output_dir = args.output_dir
    sample_rate = args.sample_rate
    clip_seconds = args.clip_seconds

    clip_samples = int(sample_rate * clip_seconds)

    output_dir = Path(output_dir, split)
    os.makedirs(output_dir, exist_ok=True)

    params = []
    audio_index = 0

    audio_paths = get_audio_paths(audios_dir=audios_dir, split=split)

    for audio_index, audio_path in enumerate(audio_paths):

        param = (
            audio_index, audio_path, sample_rate, clip_samples, output_dir,
        )
        params.append(param)

    for param in params:
        write_single_audio(param)


def get_audio_paths(audios_dir, split):

    room_ids = sorted(os.listdir(audios_dir))

    if split == "train":
        room_ids = [e for e in room_ids if e not in TEST_ROOMS]
    elif split == "test":
        room_ids = [e for e in room_ids if e in TEST_ROOMS]
    else:
        raise NotImplementedError

    audio_paths = [Path(audios_dir, room_id, "ambience_tetra_24k_edited.wav") for room_id in room_ids]

    return audio_paths


def write_single_audio(param: List) -> NoReturn:
    r"""Write single audio into hdf5 file."""

    (
        audio_index,
        audio_path,
        sample_rate,
        clip_samples, 
        output_dir,
    ) = param

    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=False)
    audio_length = audio.shape[-1]
    clip_samples = int(60 * sample_rate)

    pointer = 0
    index = 0

    while pointer < audio_length:

        clip = audio[:, pointer : pointer + clip_samples]
        print(np.max(clip))

        bare_name = ",".join([Path(audio_path).parent.stem, Path(audio_path).stem])
        output_path = Path(output_dir, "{}_{:03d}.wav".format(bare_name, index))

        soundfile.write(file=output_path, data=clip.T, samplerate=sample_rate)
        print("write out to {}".format(output_path))

        index += 1
        pointer += clip_samples

        
def meta_file_to_event_list(metadata_path):

    df = pd.read_csv(metadata_path, sep=',', header=None)
    frame_indexes = df[0].values
    class_indexes = df[1].values
    event_indexes = df[2].values
    azimuths = df[3].values
    elevations = df[4].values

    repeats = 10
    # frame_indexes = np.repeat(frame_indexes, repeats=repeats)
    class_indexes = np.repeat(class_indexes, repeats=repeats)
    event_indexes = np.repeat(event_indexes, repeats=repeats)
    azimuths = np.repeat(azimuths, repeats=repeats)
    elevations = np.repeat(elevations, repeats=repeats)

    tmp = []
    for frame_index in frame_indexes:
        for i in range(repeats):
            tmp.append(frame_index * repeats + i)

    frame_indexes = np.array(tmp)

    unique_class_indexes = list(set(event_indexes))

    event_list = []

    for event_index in unique_class_indexes:
        event_frame_indexes = frame_indexes[event_indexes == event_index]

        event = {
            # "onset_frame": event_frame_indexes[0],
            # "offset_frame": event_frame_indexes[-1],
            "frame_indexes": frame_indexes[event_indexes == event_index],
            "azimuths": azimuths[event_indexes == event_index],
            "elevations": elevations[event_indexes == event_index],
            "class_indexes": class_indexes[event_indexes == event_index],
        }
        
        event_list.append(event)

    return event_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audios_dir",
        type=str,
        required=True,
        help="Directory of the VCTK dataset.",
    )
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"])
    parser.add_argument("--sample_rate", type=int, required=True, help="Sample rate.")
    parser.add_argument("--clip_seconds", type=float, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Parse arguments.
    args = parser.parse_args()

    # Parse arguments.
    args = parser.parse_args()

    # Process audios into segments.
    process_audio_into_clips(args)
