import argparse
import os
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor
from typing import NoReturn, List
import h5py
import numpy as np
import librosa
import soundfile

from nesd.utils import remove_silence


TEST_SPEAKER_IDS = ["p345", "p347", "p351", "p360", "p361", "p362", "p363", "p364", "p374", "p376"]


def process_audio_into_segments(args) -> NoReturn:
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
    output_dir = args.output_dir
    sample_rate = args.sample_rate
    segment_seconds = args.segment_seconds

    segment_samples = int(sample_rate * segment_seconds)

    # Get speaker ids.
    audios_dir = Path(dataset_dir, "wav48")    
    speaker_ids = get_speaker_ids(audios_dir=audios_dir, split=split)

    output_dir = Path(output_dir, split)
    os.makedirs(output_dir, exist_ok=True)

    # Prepare information of audio for processing them in parallel.
    params = []
    audio_index = 0

    for speaker_id in speaker_ids:

        speaker_audios_dir = Path(audios_dir, speaker_id)

        audio_names = sorted(os.listdir(speaker_audios_dir))

        for audio_name in audio_names:

            audio_path = Path(speaker_audios_dir, audio_name)

            param = (
                audio_index,
                audio_path,
                sample_rate,
                segment_samples,
                output_dir,
            )
            params.append(param)

            audio_index += 1

    # Uncomment for debug.
    # write_single_audio(params[0])
    # os._exit(0)
    # for param in params:
    #     write_single_audio(param)

    # Process audio in parallel.
    pack_hdf5s_time = time.time()

    with ProcessPoolExecutor(max_workers=32) as pool:
        # Maximum works on the machine
        pool.map(write_single_audio, params)

    print("Pack hdf5 time: {:.3f} s".format(time.time() - pack_hdf5s_time))


def get_speaker_ids(audios_dir, split):

    all_speaker_ids = sorted(os.listdir(audios_dir))

    if split == "train":
        speaker_ids = [speaker_id for speaker_id in all_speaker_ids if speaker_id not in TEST_SPEAKER_IDS]
    elif split == "test":
        speaker_ids = [speaker_id for speaker_id in all_speaker_ids if speaker_id in TEST_SPEAKER_IDS]
    else:
        raise NotImplementedError 

    return speaker_ids


def write_single_audio(param: List) -> NoReturn:
    r"""Write single audio into hdf5 file."""

    (
        audio_index,
        audio_path,
        sample_rate,
        segment_samples,
        output_dir,
    ) = param

    # Load audio.
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)

    audio = remove_silence(audio=audio, sample_rate=sample_rate)

    audio_samples = audio.shape[-1]

    # Repat or split audio into fixed length segments.
    if audio_samples > 0:

        segments = []

        if audio_samples < segment_samples:
            segment = repeat_to_length(audio=audio, segment_samples=segment_samples)
            segments.append(segment)
            
        else:
            pointer = 0
            while pointer + segment_samples < audio_samples:
                segment = audio[pointer : pointer + segment_samples]
                segments.append(segment)
                pointer += segment_samples

    # Write out segments.
    for n, segment in enumerate(segments):

        bare_name = "{}_{:04d}".format(Path(audio_path).stem, n)
        output_path = Path(output_dir, "{}.wav".format(bare_name))

        soundfile.write(file=output_path, data=segment, samplerate=sample_rate)
        print('{} Write audio to {}'.format(audio_index, output_path))


def repeat_to_length(audio: np.ndarray, segment_samples: int) -> np.ndarray:
    r"""Repeat audio to length."""
    
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
    parser.add_argument("--sample_rate", type=int, required=True, help="Sample rate.")
    parser.add_argument("--segment_seconds", type=float, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Parse arguments.
    args = parser.parse_args()

    # Process audios into segments.
    process_audio_into_segments(args)
