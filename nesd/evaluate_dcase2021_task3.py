import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import pathlib
import scipy.signal


D21T3_LABELS = [
    "alarm",
    "crying baby",
    "crash",
    "barking dog",
    "female scream",
    "female speech",
    "footsteps",
    "knocking on door",
    "male scream",
    "male speech",
    "ringing phone",
    "piano",
]

# NIGNES_ID_TO_D21T3_ID = {
#     0: 0,
#     1: 1,
#     2: 2,
#     3: 3,
#     5: 4,
#     6: 5,
#     8: 6,
#     9: 7,
#     10: 8,
#     11: 9,
#     12: 10,
#     13: 11,
# }


def detect_max(x):

    azi_grids, ele_grids = x.shape

    if True:
        x = scipy.signal.medfilt2d(input=x, kernel_size=5)

    locts = []
    i = 0

    while True:

        if False:
            plt.matshow(x.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
            plt.savefig('_zz{}.pdf'.format(i))
            i += 1

        max_value = np.max(x)
        azi, ele = np.unravel_index(np.argmax(x), x.shape)
        r = 10
        x[max(azi - r, 0) : min(azi + r, azi_grids) + 1, max(ele - r, 0) : min(ele + r, ele_grids) + 1] = 0
        
        if max_value > 0.8:
            locts.append((azi, ele))
        else:
            break

    return locts


def process_mat_write_csv(args):

    # frames_per_sec = 10

    csv_path = args.csv_path
    
    pred_mat_timelapse = pickle.load(open('_all_pred_mat_timelapse.pkl', 'rb'))
    pred_mat_classwise_timelapse = pickle.load(open('_all_pred_mat_classwise_timelapse.pkl', 'rb'))
    
    frames_num, azi_grids, col_grids, classes_num = pred_mat_classwise_timelapse.shape

    tmp = []
    events = []
    grid_deg = 2

    # For debug
    
    # detect_max(pred_mat_timelapse[100, :, :])
    
    for t in range(frames_num):
        # print(t)
        loct_pairs = detect_max(pred_mat_timelapse[t, :, :])

        if len(loct_pairs) > 0:
            for (azi, col) in loct_pairs:
                tmp = pred_mat_classwise_timelapse[t, azi, col]
                pred_id = np.argmax(tmp)
                prob = np.max(tmp)
                
                # if prob > 0.2:
                if prob > 0.:
                    dcase_azimuth = azi * grid_deg
                    if dcase_azimuth > 180:
                        dcase_azimuth -= 360
                    dcase_colatitude = 90 - col * grid_deg

                    event = [t, pred_id, dcase_azimuth, dcase_colatitude]  # TODO
                    events.append(event)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, 'w') as fw:
        
        for event in events:
            frame_index, class_id, azimuth, colatitude = event
            fw.write("{},{},{},{}\n".format(frame_index, class_id, azimuth, colatitude))

    from IPython import embed; embed(using=False); os._exit(0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest="mode")

    parser_process_mat_write_csv = subparsers.add_parser("process_mat_write_csv")
    parser_process_mat_write_csv.add_argument(
        "--csv_path", type=str, required=True, help="Directory of workspace."
    )

    args = parser.parse_args()
    args.filename = pathlib.Path(__file__).stem 
    
    if args.mode == "process_mat_write_csv":
        process_mat_write_csv(args)

    else:
        raise Exception("Error argument!")
