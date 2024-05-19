import argparse
import numpy as np
import pickle
from pathlib import Path
import torch
import math
import soundfile
from sklearn.cluster import KMeans
from nesd.utils import read_yaml, sph2cart, expand_along_frame_axis, get_included_angle, normalize
from nesd.train import get_model
import librosa
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from nesd.inference import get_all_agent_look_at_directions
from nesd.constants import PAD
from nesd.utils import get_included_angle, apply_lowpass_filter


LABELS = [
    "clearthroat", 
    "cough", 
    "doorslam", 
    "drawer", 
    "keyboard", 
    "keysDrop", 
    "knock", 
    "laughter", 
    "pageturn", 
    "phone", 
    "speech"
]

LB_TO_ID = {lb: id for id, lb in enumerate(LABELS)}
ID_TO_LB = {id: lb for id, lb in enumerate(LABELS)}


split = "test"
dataset_dir = "/datasets/dcase2019/task3"

# audio_path = Path(dataset_dir, "mic_eval", "split0_1.wav")
csv_path = Path(dataset_dir, "metadata_eval", "split0_1.csv")

audio_path = "_uu.wav"


def coordinate_to_position(coordinate):

    if set(["x", "y", "z"]) <= set(coordinate.keys()):
        pos = np.array([coordinate["x"], coordinate["y"], coordinate["z"]])

    if set(["azimuth_deg", "elevation_deg", "radius"]) <= set(coordinate.keys()):
        pos = sph2cart(
            azimuth=np.deg2rad(coordinate["azimuth_deg"]), 
            elevation=np.deg2rad(coordinate["elevation_deg"]),
            r=coordinate["radius"],
        )

    return pos


def coordinate_to_orientation(orientation):

    if set(["x", "y", "z"]) <= set(orientation.keys()):
        orientation = np.array([orientation["x"], orientation["y"], orientation["z"]])

    if set(["azimuth_deg", "elevation_deg"]) <= set(orientation.keys()):
        orientation = sph2cart(
            azimuth=np.deg2rad(orientation["azimuth_deg"]), 
            elevation=np.deg2rad(orientation["elevation_deg"]),
            r=1.,
        )

    orientation = normalize(orientation)

    return orientation



def inference(args):

    workspace = args.workspace
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    filename = Path(__file__).stem
    
    configs = read_yaml(config_yaml)

    simulator_configs = configs["simulator_configs"]
    # sample_rate = simulator_configs["sample_rate"]
    # segment_seconds = simulator_configs["segment_seconds"]
    # frames_per_sec = simulator_configs["frames_per_sec"]
    sample_rate = 24000
    segment_seconds = 2.0
    frames_per_sec = 100
    mics_meta = read_yaml("microphones/eigenmike.yaml")
    # mics_num = len(mics_meta["mic_coordinates"])
    mics_num = 4

    device = configs["train"]["device"]
    model_name = configs["train"]["model_name"]

    segment_samples = int(segment_seconds * sample_rate)
    frames_num = int(segment_seconds * frames_per_sec) + 1

    # Load checkpoint
    model = get_model(model_name, mics_num)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    # Load audio
    audio, fs = librosa.load(path=audio_path, sr=sample_rate, mono=False)
    audio_samples = audio.shape[-1]

    # audio *= 10
    # audi

    if True:
        audio = apply_lowpass_filter(
            audio=audio, 
            # cutoff_freq=simulator_configs["mic_cutoff_freq"], 
            cutoff_freq=1000,
            sample_rate=sample_rate,
        )

    # if True:
    #     lowpass = 1000
    #     import torchaudio
    #     audio = torchaudio.functional.lowpass_biquad(
    #         waveform=torch.Tensor(audio),
    #         sample_rate=sample_rate,
    #         cutoff_freq=lowpass,
    #     ).data.cpu().numpy()
    #     soundfile.write(file="_zz.wav", data=audio.T, samplerate=sample_rate)

    # soundfile.write(file="_zz.wav", data=audio.T, samplerate=sample_rate)
    # from IPython import embed; embed(using=False); os._exit(0)

    grid_deg = 2
    azi_grids = 360 // grid_deg
    ele_grids = 180 // grid_deg

    # Mic positions
    # mics_center_pos = np.array([3, 3, 1])
    # mics_center_pos = np.array([3, 3, 3])
    mics_center_pos = np.array([4, 4, 2])
    mic_poss = get_mic_positions(mics_meta, frames_num) + mics_center_pos
    mic_poss = mic_poss[None, :, :, :]
    
    # Mic orientations
    # mic_oriens = get_mic_orientations(mics_meta, frames_num)
    # mic_oriens = mic_oriens[None, :, :, :]

    mic_oriens = np.array(
        [[ 0.5792,  0.5792,  0.5736],
        [ 0.5792, -0.5792, -0.5736],
        [-0.5792,  0.5792, -0.5736],
        [-0.5792, -0.5792,  0.5736]]
    )
    mic_oriens = np.tile(mic_oriens[None, :, None, :], (1, 1, 201, 1))

    ##
    agent_look_at_directions = get_all_agent_look_at_directions(grid_deg)
    # (rays_num, ndim)

    agent_look_at_directions = expand_along_frame_axis(
        x=agent_look_at_directions[None, :, :], 
        repeats=frames_num
    )
    # (1, rays_num, frames_num, ndim)

    rays_num = agent_look_at_directions.shape[1]

    ##
    agent_poss = np.repeat(mics_center_pos[None, None, None, :], repeats=frames_num, axis=2)
    agent_poss = np.repeat(agent_poss, repeats=rays_num, axis=1)
    # (1, rays_num, frames_num, 3)
    
    agent_look_at_distances = np.ones((1, rays_num, frames_num)) * PAD
    # (1, rays_num, frames_num)

    agent_distance_masks = np.zeros((1, rays_num, frames_num))
    # (1, rays_num, frames_num)

    ##
    bgn_sample = 0
    pred_directions = []

    while bgn_sample < audio_samples:

        print(bgn_sample / sample_rate)

        mic_wavs = audio[None, :, bgn_sample : bgn_sample + segment_samples]
        mic_wavs = librosa.util.fix_length(data=mic_wavs, size=segment_samples, axis=-1)

        pointer = 0
        batch_size = 2000
        output_dict = {}

        while pointer < rays_num:
            # print(pointer)

            _len = min(batch_size, rays_num - pointer)

            agent_detect_idxes = torch.Tensor(np.arange(_len)[None, :])
            agent_distance_idxes = torch.Tensor(np.arange(0)[None, :])
            agent_sep_idxes = torch.Tensor(np.arange(0)[None, :])

            batch_data = {
                "mic_wavs": mic_wavs,
                "mic_positions": mic_poss,
                "mic_orientations": mic_oriens,
                "agent_positions": agent_poss[:, pointer : pointer + batch_size, :, :],
                "agent_look_at_directions": agent_look_at_directions[:, pointer : pointer + batch_size, :, :],
                "agent_look_at_distances": agent_look_at_distances[:, pointer : pointer + batch_size, :],
                "agent_distance_masks": agent_distance_masks[:, pointer : pointer + batch_size, :],
                "agent_detect_idxes": agent_detect_idxes,
                "agent_distance_idxes": agent_distance_idxes,
                "agent_sep_idxes": agent_sep_idxes
            }
            
            for key in batch_data.keys():
                batch_data[key] = torch.Tensor(batch_data[key]).to(device)

            with torch.no_grad():
                model.eval()
                batch_output_dict = model(batch_data)

            if len(output_dict) == 0:
                for key in batch_output_dict.keys():
                    output_dict[key] = []

            for key in batch_output_dict.keys():
                output_dict[key].append(batch_output_dict[key].cpu().numpy())

            pointer += batch_size

        bgn_sample += segment_samples

        for key in output_dict.keys():
            output_dict[key] = np.concatenate(output_dict[key], axis=1)

        pred_direction = output_dict["agent_look_at_direction_has_source"][0].transpose(1, 0).reshape(frames_num, azi_grids, ele_grids)

        pred_direction = pred_direction[0 : -1 : 10, :, :]
        pred_directions.append(pred_direction)

    pred_directions = np.concatenate(pred_directions, axis=0)
    pickle.dump(pred_directions, open("_zz.pkl", "wb"))


    # soundfile.write(file="_zz.wav", data=audio.T, samplerate=sample_rate)
    # soundfile.write(file="_zz.wav", data=mic_wavs[0].T, samplerate=sample_rate)

    from IPython import embed; embed(using=False); os._exit(0)


'''
def get_mic_positions(mics_meta, mics_center_pos, frames_num):

    mics_coords = mics_meta["mic_coordinates"]
    
    mic_poss = []
    for mic_coord in mics_coords:
        mic_pos = coordinate_to_position(mic_coord)
        mic_pos = expand_along_frame_axis(x=mic_pos, repeats=frames_num)
        mic_poss.append(mic_pos)

    mic_poss = np.stack(mic_poss, axis=0)
    # (mics_num, frames_num, ndim)

    mic_poss += mics_center_pos

    return mic_poss
'''

def get_mic_positions(mics_meta, frames_num):

    mics_coords = mics_meta["mic_coordinates"]
    
    mic_poss = []
    for mic_coord in mics_coords:
        mic_pos = coordinate_to_position(mic_coord)
        mic_pos = expand_along_frame_axis(x=mic_pos, repeats=frames_num)
        mic_poss.append(mic_pos)

    mic_poss = np.stack(mic_poss, axis=0)
    # (mics_num, frames_num, ndim)

    return mic_poss


def get_mic_orientations(mics_meta, frames_num):

    raw_mics_oriens = mics_meta["mic_orientations"]
    mic_oriens = []

    for mic_orien in raw_mics_oriens:

        mic_orien = coordinate_to_orientation(mic_orien)
        mic_orien = expand_along_frame_axis(x=mic_orien, repeats=frames_num)

        mic_oriens.append(mic_orien)

    mic_oriens = np.stack(mic_oriens, axis=0)
    return mic_oriens


def read_dcase2019_task3_csv(csv_path):

    df = pd.read_csv(csv_path, sep=',')
    labels = df['sound_event_recording'].values
    onsets = df['start_time'].values
    offsets = df['end_time'].values
    azimuths = df['azi'].values
    elevations = df['ele'].values
    distances = df['dist'].values

    events_num = len(labels)
    frames_per_sec = 10

    frame_indexes = []
    class_indexes = []
    event_indexes = []
    _azimuths = []
    _elevations = []
    _distances = []

    # from nesd.dataset_creation.pack_audios_to_hdf5s.dcase2019_task3 import LB_TO_ID

    for n in range(events_num):

        onset_frame = int(onsets[n] * frames_per_sec)
        offset_frame = int(offsets[n] * frames_per_sec)

        for frame_index in np.arange(onset_frame, offset_frame + 1):
            frame_indexes.append(frame_index)
            class_indexes.append(LB_TO_ID[labels[n]])
            event_indexes.append(n)
            _azimuths.append(azimuths[n])
            _elevations.append(elevations[n])
            _distances.append(distances[n])

    # azimuths = np.deg2rad(np.array(_azimuths) % 360)
    # colatitudes = np.deg2rad(90 - np.array(_elevations))
    # distances = np.array(_distances)
    azimuths = np.deg2rad(np.array(_azimuths))
    elevations = np.deg2rad(np.array(_elevations))

    return frame_indexes, class_indexes, azimuths, elevations, distances


def plot_panaroma(args):

    frame_indexes, class_indexes, azimuths, elevations, distances = read_dcase2019_task3_csv(csv_path=csv_path)

    #
    pred_tensor = pickle.load(open("_zz.pkl", "rb"))
    frames_num = pred_tensor.shape[0]

    grid_deg = 2
    azi_grids = 360 // grid_deg
    ele_grids = 180 // grid_deg
    
    # gt_tensor = np.zeros((frames_num + 20, azi_grids, ele_grids))
    gt_tensor = np.zeros((600, azi_grids, ele_grids))
    # half_angle = math.atan2(0.1, 1)
    half_angle = np.deg2rad(5)

    # Get GT
    params = []

    for n in range(len(frame_indexes)):

        frame_index = frame_indexes[n]
        class_index = class_indexes[n]
        source_azi = azimuths[n]
        source_ele = elevations[n]

        param = (frame_index, class_index, source_azi, source_ele, azi_grids, ele_grids, grid_deg, half_angle)
        params.append(param)

    # for param in params:
    #     _multiple_process_gt_mat(param)

    with ProcessPoolExecutor(max_workers=None) as pool: # Maximum workers on the machine.
        results = pool.map(_multiple_process_gt_mat, params)

    gt_texts = [""] * gt_tensor.shape[0]
    for (frame_index, class_index, gt_mat) in results:
        gt_tensor[frame_index] += gt_mat
        gt_texts[frame_index] += ID_TO_LB[class_index]

    Path("_tmp").mkdir(parents=True, exist_ok=True)

    # Plot
    params = []

    for n in range(min(pred_tensor.shape[0], gt_tensor.shape[0])):
        param = (n, gt_texts[n], gt_tensor[n], pred_tensor[n], grid_deg)
        params.append(param)

    # for param in params:
    #     _multiple_process_plot(param)

    with ProcessPoolExecutor(max_workers=None) as pool: # Maximum workers on the machine.
        results = pool.map(_multiple_process_plot, params)

    results = list(results)

    pickle.dump(results, open("_zz_centers.pkl", "wb"))


def _multiple_process_gt_mat(param):

    frame_index, class_index, source_azi, source_ele, azi_grids, ele_grids, grid_deg, half_angle = param
    print(frame_index)

    gt_mat = np.zeros((azi_grids, ele_grids))

    source_direction = np.array(sph2cart(source_azi, source_ele, 1.))

    tmp = []
    azi_grids, ele_grids = gt_mat.shape

    for i in range(azi_grids):
        for j in range(ele_grids):

            _azi = np.deg2rad(i * grid_deg - 180)
            _ele = np.deg2rad(j * grid_deg - 90)

            plot_direction = np.array(sph2cart(_azi, _ele, 1))

            ray_angle = get_included_angle(source_direction, plot_direction)
            # from IPython import embed; embed(using=False); os._exit(0)
            # print(i * grid_deg - 180, j * grid_deg - 90, ray_angle)

            if ray_angle < half_angle:
                gt_mat[i, j] = 1
                tmp.append((i, j))

    # if class_index == 0:
    #     from IPython import embed; embed(using=False); os._exit(0)

    return frame_index, class_index, gt_mat


def _multiple_process_plot(param):

    n, gt_text, gt_mat, pred_mat, grid_deg = param
    print("Plot: {}".format(n))

    azi_grids, ele_grids = gt_mat.shape

    if True:
        centers = calculate_centers(x=pred_mat)
        pred_mat = plot_center_to_mat(centers=centers, x=pred_mat)

    plt.figure(figsize=(20, 10))
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].matshow(gt_mat.T, origin='upper', aspect='equal', cmap='jet', vmin=0, vmax=1)
    axs[1].matshow(pred_mat.T, origin='upper', aspect='equal', cmap='jet', vmin=0, vmax=1)
    for i in range(2):
        axs[i].grid(color='w', linestyle='--', linewidth=0.1)
        axs[i].xaxis.set_ticks(np.arange(0, azi_grids + 1, 10))
        axs[i].yaxis.set_ticks(np.arange(0, ele_grids + 1, 10))
        axs[i].xaxis.set_ticklabels(np.arange(0, 361, 10 * grid_deg), rotation=90)
        axs[i].yaxis.set_ticklabels(np.arange(0, 181, 10 * grid_deg))
    axs[0].set_title(gt_text)

    plt.savefig('_tmp/{:04d}.png'.format(n))

    return n, centers


def calculate_centers(x):

    tmp = np.stack(np.where(x > 0.8), axis=1)

    if len(tmp) == 0:
        return []
    
    for n_clusters in range(1, 10):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(tmp)
        error = np.mean(np.abs(tmp - kmeans.cluster_centers_[kmeans.labels_]))
        if error < 10:
            break
    
    return kmeans.cluster_centers_


def plot_center_to_mat(centers, x):
    
    for center in centers:
        center_azi = int(center[0])
        center_col = int(center[1])
        x[center_azi - 5 : center_azi + 6, center_col] = np.nan
        x[center_azi, center_col - 5 : center_col + 6] = np.nan

    return x



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest="mode")

    parser_inference = subparsers.add_parser("inference")
    parser_inference.add_argument('--workspace', type=str)
    parser_inference.add_argument('--config_yaml', type=str)
    parser_inference.add_argument("--checkpoint_path", type=str)

    parser_plot_panaroma = subparsers.add_parser("plot_panaroma")

    args = parser.parse_args()

    if args.mode == "inference":
        inference(args)

    if args.mode == "plot_panaroma":
        plot_panaroma(args)

    else:
        raise Exception("Error argument!")