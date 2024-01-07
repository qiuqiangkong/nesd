from pathlib import Path
import pandas as pd
import math
import pickle


LABELS = ['clearthroat', 'cough', 'doorslam', 'drawer', 'keyboard', 'keysDrop', 'knock', 'laughter', 'pageturn', 'phone', 'speech']

LB_TO_ID = {lb: id for id, lb in enumerate(LABELS)}
ID_TO_LB = {id: lb for id, lb in enumerate(LABELS)}


def dcase2022_task3():
    
    d16t2_dir = "/home/qiuqiangkong/workspaces/nesd2/audios/dcase2016_task2_2s_segments/train"

    paths_dict = {}

    for label in LABELS:
        
        audio_paths = sorted(list(Path(d16t2_dir).glob("{}*.wav".format(label))))
        audio_paths = [str(audio_path) for audio_path in audio_paths]

        paths_dict[label] = audio_paths

    out_pickle_path = "paths_dict_d16t2.pkl"
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