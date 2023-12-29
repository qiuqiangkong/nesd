import argparse
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor
from typing import NoReturn, List
import h5py
import numpy as np
import librosa

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
    segment_seconds = args.segment_seconds

    segment_samples = int(sample_rate * segment_seconds)

    audios_dir = os.path.join(dataset_dir, "FSDKaggle2018.audio_{}".format(split))

    os.makedirs(hdf5s_dir, exist_ok=True)

    params = []
    audio_index = 0

    audio_names = sorted(os.listdir(audios_dir))

    for audio_name in audio_names:

        audio_path = os.path.join(audios_dir, audio_name)

        param = (
            audio_index,
            audio_name,
            audio_path,
            sample_rate,
            hdf5s_dir,
            segment_samples,
        )
        params.append(param)

        audio_index += 1

    # Uncomment for debug.
    # write_single_audio_to_hdf5(params[0])
    # os._exit(0)
    # for param in params:
    #     write_single_audio_to_hdf5(param)

    pack_hdf5s_time = time.time()

    with ProcessPoolExecutor(max_workers=None) as pool:
        # Maximum works on the machine
        pool.map(write_single_audio_to_hdf5, params)

    print("Pack hdf5 time: {:.3f} s".format(time.time() - pack_hdf5s_time))


def write_single_audio_to_hdf5(param: List) -> NoReturn:
    r"""Write single audio into hdf5 file."""

    (
        audio_index,
        audio_name,
        audio_path,
        sample_rate,
        hdf5s_dir,
        segment_samples,
    ) = param

    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)

    audio = remove_silence(audio, sample_rate)

    audio_samples = audio.shape[-1]

    if audio_samples > 0:

        segments = []

        if audio_samples < segment_samples:
            segment = fix_length(audio, segment_samples)
            segments.append(segment)
            
        else:
            pointer = 0
            while pointer + segment_samples < audio_samples:
                segment = audio[pointer : pointer + segment_samples]
                segments.append(segment)
                pointer += segment_samples

    for n, segment in enumerate(segments):

        bare_name = "{}_{:04d}".format(pathlib.Path(audio_name).stem, n)
        hdf5_path = os.path.join(hdf5s_dir, "{}.h5".format(bare_name, n))

        frame_contains_source = np.ones(301)

        with h5py.File(hdf5_path, "w") as hf:
            hf.create_dataset(name="waveform", data=float32_to_int16(segment), dtype=np.int16)
            hf.create_dataset(name="frame_contains_source", data=frame_contains_source, dtype=np.float32)
            hf.attrs.create("audio_name", data=bare_name.encode(), dtype="S100")
            hf.attrs.create("sample_rate", data=sample_rate, dtype=np.int32)
            # hf.attrs.create("label", data=label, dtype="S100")

        print('{} Write hdf5 to {}'.format(audio_index, hdf5_path))

        # import soundfile
        # out_path = os.path.join('_tmp', "{}.wav".format(bare_name, n))
        # soundfile.write(file=out_path, data=segment, samplerate=sample_rate)


def remove_silence(audio, sample_rate):

    window_size = int(sample_rate * 0.1)
    threshold = 0.02

    frames = librosa.util.frame(x=audio, frame_length=window_size, hop_length=window_size).T
    # shape: (frames_num, window_size)

    new_frames = get_active_frames(frames, threshold)
    # shape: (new_frames_num, window_size)

    new_audio = new_frames.flatten()
    # shape: (new_audio_samples,)

    return new_audio


def get_active_frames(frames, threshold):
    
    energy = np.max(np.abs(frames), axis=-1)
    # shape: (frames_num,)

    active_indexes = np.where(energy > threshold)[0]
    # shape: (new_frames_num,)

    new_frames = frames[active_indexes]
    # shape: (new_frames_num,)

    return new_frames


def fix_length(audio, segment_samples):
    repeats_num = (segment_samples // audio.shape[-1]) + 1
    audio = np.tile(audio, repeats_num)[0 : segment_samples]
    return audio



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory of the VCTK dataset.",
    )
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"])
    parser.add_argument(
        "--hdf5s_dir",
        type=str,
        required=True,
        help="Directory to write out hdf5 files.",
    )
    parser.add_argument("--sample_rate", type=int, required=True, help="Sample rate.")
    parser.add_argument("--segment_seconds", type=float, required=True, help="Sample rate.")

    # Parse arguments.
    args = parser.parse_args()

    # Pack audios into hdf5 files.
    pack_audios_to_hdf5s(args)
