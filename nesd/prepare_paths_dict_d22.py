from pathlib import Path
import pandas as pd
import math
import pickle


LABELS = ['Female speech, woman speaking', 
'Male speech, man speaking', 
'Clapping', 
'Telephone', 
'Laughter', 
'Domestic sounds', 
'Walk, footsteps' ,
'Door, open or close', 
'Music', 
'Musical instrument', 
'Water tap, faucet', 
'Bell', 
'Knock']

LB_TO_ID = {lb: id for id, lb in enumerate(LABELS)}
ID_TO_LB = {id: lb for id, lb in enumerate(LABELS)}


MID_TO_LABEL = {
    "/m/02zsn": 'Female speech, woman speaking',
    "/m/05zppz": 'Male speech, man speaking',
    "/m/0l15bq": 'Clapping',
    "/m/07cx4": 'Telephone',
    "/m/07pp8cl": 'Telephone',
    "/m/01j3sz": 'Laughter',
    "/m/0d31p": 'Domestic sounds',
    "/m/0dv3j": 'Domestic sounds',
    "/m/02x984l": 'Domestic sounds',
    "/m/07pbtc8": 'Walk, footsteps',
    "/m/0fqfqc": 'Door, open or close',

    "/m/042v_gx": 'Musical instrument',
    "/m/0dwsp": 'Musical instrument',
    "/m/0239kh": 'Musical instrument',
    "/m/05r5c": 'Musical instrument',
    "/m/05r5wn": 'Musical instrument',
    "/m/02jz0l": 'Water tap, faucet',
    "/m/0gy1t2s": 'Bell',
    "/m/0f8s22": 'Bell',
    "/m/07r4wb8": 'Knock',
}

LABEL_TO_MIDS = {label: [] for label in LABELS}

for id in MID_TO_LABEL.keys():
    label = MID_TO_LABEL[id]
    LABEL_TO_MIDS[label].append(id)


def dcase2022_task3():
    
    vctk_dir = "/home/qiuqiangkong/workspaces/nesd2/audios/vctk_2s_segments/train"

    fsd50k_dir = "/home/qiuqiangkong/workspaces/nesd2/audios/fsd50k_2s_segments/train"

    musdb18hq_dir = "/home/qiuqiangkong/workspaces/nesd2/audios/musdb18hq_2s_segments/train"

    meta_csv_path = "/home/qiuqiangkong/datasets/fsd50k/FSD50K.metadata/collection/collection_dev.csv"

    df = pd.read_csv(meta_csv_path, sep=',')
    meta_dict = {}
    meta_dict["audio_names"] = df["fname"].values
    meta_dict["labels"] = df["labels"].values
    meta_dict["mids"] = df["mids"].values
    audios_num = len(meta_dict["audio_names"])

    paths_dict = {}

    for label in LABELS:
        # tmp[label] = None

        ids = LABEL_TO_MIDS[label]

        indexes = []
        audio_paths = []

        for id in ids:

            for n in range(audios_num):
                
                if not isinstance(meta_dict["mids"][n], str):
                    continue

                _mids = meta_dict["mids"][n].split(",")

                if id in _mids:
                    indexes.append(n)
                    audio_path = Path(fsd50k_dir, "{}.wav".format(meta_dict["audio_names"][n]))
                    if Path(audio_path).exists():
                        audio_paths.append(audio_path)

        paths_dict[label] = audio_paths

    # vctk
    audio_paths = sorted(list(Path(vctk_dir).glob("*.wav")))
    paths_dict["Unknown"] = audio_paths

    # music
    audio_paths = sorted(list(Path(musdb18hq_dir).glob("*.wav")))
    paths_dict["Music"] = audio_paths

    for key in paths_dict.keys():
        print(len(paths_dict[key]))

    out_pickle_path = "paths_dict_d22.pkl"
    pickle.dump([paths_dict, LABELS, LB_TO_ID, ID_TO_LB], open(out_pickle_path, "wb"))
    print("Write out to {}".format(out_pickle_path))


def add2():

    tmp = [
        [4, 80],
        [3.5, 83],
        [5.0, 83],
        [3.0, 85],  # B
        [4.0, 86],
        [3.0, 87],
        [1.0, 84],
        [3.0, 75],
        [2.0, 80],
        [4.0, 79],
        [1.0, 65],
        [4.0, 79],
        [0.5, 75],
        [0.5, 65],
        [2.0, 79],
        [7.0, 77],
        [1.0, 85],
        [1.0, 85],
        [1.0, 66],
        [4.0, 77],
        [4.0, 78],
        [0.5, 65],
        [1.0, 88],
        [2.0, 65],
        [1.0, 83],
        [3.0, 76],
        [1.0, 65],
        [2.0, 85],
        [2.5, 84],
        [4.5, 80],
        [2.0, 78],
        [6.0, 66],
        [4.0, 76],
        [0.5, 61],
        [2.0, 65],
        [1.0, 75],
        [4.0, 77],
        [3.0, 80],
        [3.5, 89],
        [3.0, 88],
        [1.0, 85],
        [1.0, 60],
        [1.0, 75],
        [4.0, 86],
        [0.5, 85],
        [2.0, 60],
        [1.0, 82],
        [2.0, 71],
        [2.5, 69],
        [1.5, 88],
        [3.5, 93],
        [2.0, 90],
        [3.0, 94],
        [0.5, 94],
        [3.5, 87],
        [2.0, 70],
        [3.0, 94],
        [1.0, 91],
        [3.5, 87],
        [1.0, 90],
        [1.0, 93],
        [2.0, 91],
        [2.0, 91],
        [2.0, 97],
        [2.0, 82],
        [4.0, 89],
        [3.0, 94],
        [3.0, 86],
        [3.0, 94],
        [4.0, 85],
    ]

    def score_to_gpa(score):
        if 100 >= score >= 90:
            gpa = 4.0
        elif 90 > score >= 85:
            gpa = 3.7
        elif 85 > score >= 82:
            gpa = 3.3
        elif 82 > score >= 78:
            gpa = 3.0
        elif 78 > score >= 75:
            gpa = 2.7
        elif 75 > score >= 71:
            gpa = 2.3
        elif 71 > score >= 66:
            gpa = 2.0
        elif 66 > score >= 62:
            gpa = 1.7
        elif 62 > score >= 60:
            gpa = 1.3
        elif 60 > score:
            gpa = 0
        else:
            raise NotImplementedError

        return gpa

    tmp2 = []
    weights = 0
    total = 0

    for i in range(len(tmp)):
        weight = tmp[i][0]
        gpa = score_to_gpa(tmp[i][1])
        total += weight * gpa
        weights += weight

    print(total / weights)


if __name__ == '__main__':

    dcase2022_task3()
    # add2()