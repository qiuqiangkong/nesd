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

from nesd.utils import float32_to_int16


def pack_audios_to_hdf5s(args) -> NoReturn:
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
    dataset_dir = args.dataset_dir
    split = args.split
    hdf5s_dir = args.hdf5s_dir
    sample_rate = args.sample_rate

    os.makedirs(hdf5s_dir, exist_ok=True)
    params = []
    audio_index = 0

    audio_paths = sorted(list(Path(dataset_dir).rglob("*.wav")))

    for audio_index, audio_path in enumerate(audio_paths):

        # bare_name = Path(audio_path).stem
        # hdf5_path = Path(hdf5s_dir, "{}.h5".format(bare_name))

        param = (
            audio_index, audio_path, sample_rate, hdf5s_dir,
        )
        params.append(param)

    # Uncomment for debug.
    # write_single_audio_to_hdf5(params[0])
    # os._exit(0)
    for param in params:
        write_single_audio_to_hdf5(param)
    # asdf
    pack_hdf5s_time = time.time()

    # with ProcessPoolExecutor(max_workers=16) as pool:
    #     # Maximum works on the machine
    #     pool.map(write_single_audio_to_hdf5, params)

    print("Pack hdf5 time: {:.3f} s".format(time.time() - pack_hdf5s_time))


def write_single_audio_to_hdf5(param: List) -> NoReturn:
    r"""Write single audio into hdf5 file."""

    (
        audio_index,
        audio_path,
        sample_rate,
        hdf5s_dir
    ) = param

    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=False)
    audio_length = audio.shape[-1]
    clip_samples = int(60 * sample_rate)

    pointer = 0
    index = 0

    while pointer < audio_length:

        clip = audio[:, pointer : pointer + clip_samples]

        bare_name = Path(audio_path).stem
        hdf5_path = Path(hdf5s_dir, "{}_{:03d}.h5".format(bare_name, index))

        with h5py.File(hdf5_path, "w") as hf:
            hf.create_dataset(name="waveform", data=float32_to_int16(clip), dtype=np.int16)

        index += 1
        pointer += clip_samples

        print("write out to {}".format(hdf5_path))


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
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory of the VCTK dataset.",
    )
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument(
        "--hdf5s_dir",
        type=str,
        required=True,
        help="Directory to write out hdf5 files.",
    )
    parser.add_argument("--sample_rate", type=int, required=True, help="Sample rate.")

    # Parse arguments.
    args = parser.parse_args()

    # Pack audios into hdf5 files.
    pack_audios_to_hdf5s(args)
