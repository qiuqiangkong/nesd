from pathlib import Path
import librosa
import soundfile

from nesd.utils import remove_silence, repeat_to_length

def add():
    
    sample_rate = 24000
    segment_seconds = 2
    segment_samples = int(sample_rate * segment_seconds) 
    split = "train"

    audios_dir = Path("/home/qiuqiangkong/datasets/dcase2016/task2/dcase2016_task2_train_dev/dcase2016_task2_train")
    out_audios_dir = Path("/home/qiuqiangkong/workspaces/nesd2/audios/dcase2016_task2_2s_segments", split)
    out_audios_dir.mkdir(parents=True, exist_ok=True)

    audio_paths = sorted(list(Path(audios_dir).glob("*.wav")))

    for n ,audio_path in enumerate(audio_paths):

        print(n)

        audio, _ = librosa.load(path=audio_path, sr=sample_rate, mono=True)

        audio = remove_silence(audio=audio, sample_rate=sample_rate)

        if len(audio) == 0:
            continue

        audio = repeat_to_length(audio=audio, segment_samples=segment_samples)

        out_audio_path = Path(out_audios_dir, Path(audio_path).name)

        soundfile.write(file=out_audio_path, data=audio, samplerate=sample_rate)

        print("Write out to {}".format(out_audio_path))

        # from IPython import embed; embed(using=False); os._exit(0)


if __name__ == '__main__':

    add()