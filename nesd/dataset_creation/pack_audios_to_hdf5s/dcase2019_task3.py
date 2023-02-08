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

from nesd.utils import float32_to_int16


# LABELS = ['alarm', 'crying baby', 'crash', 'barking dog', 'female scream', 'female speech', 'footsteps', 'knocking on door', 'male scream', 'male speech', 'ringing phone', 'piano']
LABELS = ['clearthroat', 'cough', 'doorslam', 'drawer', 'keyboard', 'keysDrop', 'knock', 'laughter', 'pageturn', 'phone', 'speech']

LB_TO_ID = {lb: id for id, lb in enumerate(LABELS)}
ID_TO_LB = {id: lb for id, lb in enumerate(LABELS)}


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

    if split == "train":
        audios_dir = os.path.join(dataset_dir, "mic_dev")
        metadatas_dir = os.path.join(dataset_dir, "metadata_dev")
    elif split == "test":
        audios_dir = os.path.join(dataset_dir, "mic_eval")
        metadatas_dir = os.path.join(dataset_dir, "metadata_eval")
    else:
        raise NotImplementedError

    os.makedirs(hdf5s_dir, exist_ok=True)

    audio_names = sorted(os.listdir(audios_dir))

    params = []
    audio_index = 0

    for audio_name in audio_names:

        audio_path = os.path.join(audios_dir, audio_name)

        bare_name = pathlib.Path(audio_name).stem
        metadata_path = os.path.join(metadatas_dir, "{}.csv".format(bare_name))

        hdf5_path = os.path.join(hdf5s_dir, "{}.h5".format(bare_name))

        param = (
            audio_index,
            bare_name,
            audio_path,
            metadata_path,
            sample_rate,
            hdf5_path,
        )
        params.append(param)

        audio_index += 1

    # Uncomment for debug.
    # write_single_audio_to_hdf5(params[0])
    # os._exit(0)
    # for param in params:
    #     write_single_audio_to_hdf5(param)

    pack_hdf5s_time = time.time()

    with ProcessPoolExecutor(max_workers=16) as pool:
        # Maximum works on the machine
        pool.map(write_single_audio_to_hdf5, params)

    print("Pack hdf5 time: {:.3f} s".format(time.time() - pack_hdf5s_time))


def write_single_audio_to_hdf5(param: List) -> NoReturn:
    r"""Write single audio into hdf5 file."""

    (
        audio_index,
        bare_name,
        audio_path,
        metadata_path,
        sample_rate,
        hdf5_path,
    ) = param

    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=False)
    
    df = pd.read_csv(metadata_path, sep=',')
    labels = df['sound_event_recording'].values
    onsets = df['start_time'].values
    offsets = df['end_time'].values
    azimuths = df['azi'].values
    elevations = df['ele'].values

    events_num = len(labels)
    frames_per_sec = 10

    frame_indexes = []
    class_indexes = []
    event_indexes = []
    _azimuths = []
    _elevations = []

    for n in range(events_num):

        onset_frame = int(onsets[n] * frames_per_sec)
        offset_frame = int(offsets[n] * frames_per_sec)

        for frame_index in np.arange(onset_frame, offset_frame + 1):
            frame_indexes.append(frame_index)
            class_indexes.append(LB_TO_ID[labels[n]])
            event_indexes.append(n)
            _azimuths.append(azimuths[n])
            _elevations.append(elevations[n])

    duration = audio.shape[-1] / sample_rate

    with h5py.File(hdf5_path, "w") as hf:
        hf.create_dataset(name="waveform", data=float32_to_int16(audio), dtype=np.int16)
        hf.create_dataset(name="frame_index", data=frame_indexes, dtype=np.int32)
        hf.create_dataset(name="class_index", data=class_indexes, dtype=np.int32)
        hf.create_dataset(name="event_index", data=event_indexes, dtype=np.int32)
        hf.create_dataset(name="azimuth", data=_azimuths, dtype=np.float32)
        hf.create_dataset(name="elevation", data=_elevations, dtype=np.float32)
        hf.attrs.create("audio_name", data=bare_name.encode(), dtype="S100")
        hf.attrs.create("sample_rate", data=sample_rate, dtype=np.int32)
        hf.attrs.create("duration", data=duration, dtype=np.float32)

    print('{} Write hdf5 to {}'.format(audio_index, hdf5_path))

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
