from pathlib import Path
import librosa
import soundfile
import pandas as pd

from nesd.utils import remove_silence, repeat_to_length

def add():
    
    sample_rate = 24000
    segment_seconds = 2
    segment_samples = int(sample_rate * segment_seconds) 
    split = "train"

    dataset_dir = Path("/home/qiuqiangkong/datasets/fsd50k")

    meta_csv_path = Path(dataset_dir, "FSD50K.metadata/collection/collection_dev.csv")
    audios_dir = Path(dataset_dir, "FSD50K.dev_audio")

    out_audios_dir = Path("/home/qiuqiangkong/workspaces/nesd2/audios/fsd50k_2s_segments", split)
    out_audios_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(meta_csv_path, sep=',')
    meta_dict = {}
    meta_dict["audio_names"] = df["fname"].values
    meta_dict["labels"] = df["labels"].values
    meta_dict["mids"] = df["mids"].values

    audios_num = len(meta_dict["audio_names"])

    for n in range(audios_num):

        print(n)

        audio_path = Path(audios_dir, "{}.wav".format(meta_dict["audio_names"][n]))

        audio, _ = librosa.load(path=audio_path, sr=sample_rate, mono=True)

        audio = remove_silence(audio=audio, sample_rate=sample_rate)

        if len(audio) == 0:
            continue

        audio = repeat_to_length(audio=audio, segment_samples=segment_samples)

        out_audio_path = Path(out_audios_dir, Path(audio_path).name)

        soundfile.write(file=out_audio_path, data=audio, samplerate=sample_rate)

        print("Write out to {}".format(out_audio_path))


if __name__ == '__main__':

    add()