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
import os
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from nesd.inference import get_all_agent_look_at_directions
from nesd.constants import PAD
from nesd.utils import get_included_angle, apply_lowpass_filter
from evaluate.dcase2019_task3 import coordinate_to_position, coordinate_to_orientation, get_mic_positions, get_mic_orientations
from evaluate.dcase2020_task3 import read_dcase2020_task3_csv as read_dcase2022_task3_csv
from evaluate.utils import get_locss, _multiple_process_gt_mat, _multiple_process_plot, _calculate_centers, grouping


LABELS = [
    "Female speech, woman speaking",
    "Male speech, man speaking",
    "Clapping",
    "Telephone",
    "Laughter",
    "Domestic sounds",
    "Walk, footsteps",
    "Door, open or close",
    "Music",
    "Musical instrument",
    "Water tap, faucet",
    "Bell",
    "Knock",
]

LB_TO_ID = {lb: id for id, lb in enumerate(LABELS)}
ID_TO_LB = {id: lb for id, lb in enumerate(LABELS)}

SCALE = 1


dataset_dir = "/datasets/dcase2022/task3"
workspace = "/home/qiuqiangkong/workspaces/nesd"
results_dir = Path(workspace, "results/dcase2022_task3")

select = "2"

if select == "1":
    audio_paths = [Path(dataset_dir, "mic_dev", "dev-test-sony", "fold4_room23_mix001.wav")]

    panaromas_dir = Path(results_dir, "panaroma")
    panaroma_paths = [Path(panaromas_dir, "fold4_room23_mix001.pkl")]

    gt_csvs_dir1 = Path(dataset_dir, "metadata_dev", "dev-test-sony")
    gt_csvs_dir2 = Path(dataset_dir, "metadata_dev", "dev-test-tau")

    pred_csvs_dir = Path(results_dir, "pred_csvs")

elif select == "2":

    audios_dir1 = Path(dataset_dir, "mic_dev", "dev-test-sony")
    audios_dir2 = Path(dataset_dir, "mic_dev", "dev-test-tau")

    audio_paths = sorted(
        list(Path(audios_dir1).glob("*.wav")) + list(Path(audios_dir2).glob("*.wav"))
    )

    panaromas_dir = Path(results_dir, "panaroma")
    panaroma_paths = sorted(list(Path(panaromas_dir).glob("*.pkl")))

    pred_csvs_dir = Path(results_dir, "pred_csvs")

    gt_csvs_dir1 = Path(dataset_dir, "metadata_dev", "dev-test-sony")
    gt_csvs_dir2 = Path(dataset_dir, "metadata_dev", "dev-test-tau")


def inference(args):

    workspace = args.workspace
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    filename = Path(__file__).stem
    
    configs = read_yaml(config_yaml)

    simulator_configs = configs["simulator_configs"]
    sample_rate = simulator_configs["sample_rate"]
    segment_seconds = simulator_configs["segment_seconds"]
    frames_per_sec = simulator_configs["frames_per_sec"]
    mics_meta = read_yaml(simulator_configs["mics_yaml"])
    mics_num = len(mics_meta["mic_coordinates"])

    device = configs["train"]["device"]
    model_name = configs["train"]["model_name"]

    segment_samples = int(segment_seconds * sample_rate)
    frames_num = int(segment_seconds * frames_per_sec) + 1

    Path(panaromas_dir).mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    model = get_model(model_name, mics_num)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    for audio_path in audio_paths:

        # Load audio
        audio, fs = librosa.load(path=audio_path, sr=sample_rate, mono=False)
        audio_samples = audio.shape[-1]

        audio *= SCALE

        if False:
            audio = apply_lowpass_filter(
                audio=audio, 
                # cutoff_freq=simulator_configs["mic_cutoff_freq"], 
                cutoff_freq=1000,
                sample_rate=sample_rate,
            )

        grid_deg = 2
        azi_grids = 360 // grid_deg
        ele_grids = 180 // grid_deg

        # Mic positions
        mics_center_pos = np.array([0, 0, 2])
        mic_poss = get_mic_positions(mics_meta, frames_num)
        mic_poss = mic_poss[None, :, :, :]

        # Mic orientations
        mic_oriens = get_mic_orientations(mics_meta, frames_num)
        mic_oriens = mic_oriens[None, :, :, :]

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
                if key not in ["sources_num"]:
                    output_dict[key] = np.concatenate(output_dict[key], axis=1)

            pred_direction = output_dict["agent_look_at_direction_has_source"][0].transpose(1, 0).reshape(frames_num, azi_grids, ele_grids)

            pred_direction = pred_direction[0 : -1 : 10, :, :]
            pred_directions.append(pred_direction)

            if "sources_num" in output_dict.keys():
                pred_sources_num = np.argmax(output_dict["sources_num"][0][0], axis=-1)

        pred_directions = np.concatenate(pred_directions, axis=0)
        pickle_path = Path(panaromas_dir, "{}.pkl".format(Path(audio_path).stem))
        pickle.dump(pred_directions, open(pickle_path, "wb"))
        print("Write out to {}".format(pickle_path))


def write_loc_csv(args):

    workspace = args.workspace
    grid_deg = 2

    # Calculate prediction center.
    for panaroma_path in panaroma_paths:

        csv_path = Path(pred_csvs_dir, "{}.csv".format(Path(panaroma_path).stem))

        pred_tensor = pickle.load(open(panaroma_path, "rb"))
        frames_num, azi_grids, ele_grids = pred_tensor.shape

        params = []
        for frame_index in range(frames_num):
            param = (frame_index, pred_tensor[frame_index], grid_deg)
            params.append(param)

        # No parallel is faster
        results = []
        for param in params:
            result = _calculate_centers(param)
            results.append(result)

        # with ProcessPoolExecutor(max_workers=None) as pool: # Maximum workers on the machine.
        #     results = pool.map(_calculate_centers, params)

        locss = list(results)

        # Write result to csv
        class_ids = [[1] * len(locs) for locs in locss]
        write_locss_to_csv(locss, class_ids, csv_path, grid_deg)

    return pred_tensor, locss


def plot_panaroma(args):

    workspace = args.workspace
    grid_deg = 2

    for panaroma_path in panaroma_paths:

        csv_path = Path(pred_csvs_dir, "{}.csv".format(Path(panaroma_path).stem))

        pred_tensor = pickle.load(open(panaroma_path, "rb"))

        locss = get_locss(pred_tensor)
        frames_num, azi_grids, ele_grids = pred_tensor.shape

        # Get GT panaroma.
        gt_csv_path1 = Path(gt_csvs_dir1, "{}.csv".format(panaroma_path.stem))
        gt_csv_path2 = Path(gt_csvs_dir2, "{}.csv".format(panaroma_path.stem))

        if gt_csv_path1.exists():
            gt_csv_path = gt_csv_path1

        elif gt_csv_path2.exists():
            gt_csv_path = gt_csv_path2

        frame_indexes, class_indexes, azimuths, elevations, distances = read_dcase2022_task3_csv(csv_path=gt_csv_path)

        gt_tensor = np.zeros_like(pred_tensor)
        half_angle = np.deg2rad(5)

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

        gt_texts = [""] * frames_num
        for (frame_index, class_index, gt_mat) in results:
            if frame_index >= frames_num:
                break
            gt_tensor[frame_index] += gt_mat
            gt_texts[frame_index] += ID_TO_LB[class_index]

        # Get predicted panaroma.
        params = []

        for frame_index in range(frames_num):
            png_path = Path("_tmp", panaroma_path.stem, "{:04d}.png".format(frame_index))
            Path(png_path.parent).mkdir(parents=True, exist_ok=True)
            param = (
                frame_index, 
                gt_texts[frame_index], 
                gt_tensor[frame_index], 
                pred_tensor[frame_index], 
                locss[frame_index],
                grid_deg,
                png_path
            )
            params.append(param)

        # for param in params:
        #     _multiple_process_plot(param)

        with ProcessPoolExecutor(max_workers=None) as pool: # Maximum workers on the machine.
            pool.map(_multiple_process_plot, params)

        out_video_path = Path(results_dir, "videos", "{}.mp4".format(panaroma_path.stem))
        Path(out_video_path.parent).mkdir(parents=True, exist_ok=True)
        os.system("ffmpeg -y -framerate 10 -i '{}/%04d.png' -r 30 -pix_fmt yuv420p {}".format(png_path.parent, out_video_path))
        print("Write out to {}".format(out_video_path))


def write_locss_to_csv(locss, class_ids, csv_path, grid_deg):

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    frames_num = len(locss)

    with open(csv_path, 'w') as fw:
        for frame_index in range(frames_num):
            centers = locss[frame_index]

            for i in range(len(centers)):

                azi = np.rad2deg(centers[i][0])
                ele = np.rad2deg(centers[i][1])

                track_id = 0

                fw.write("{},{},{},{},{}\n".format(
                    frame_index, 
                    class_ids[frame_index][i], 
                    track_id,
                    int(np.around(azi)), 
                    int(np.around(ele))
                )) 

    print("Write out to {}".format(csv_path))


def inference_sep(args):

    workspace = args.workspace
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    filename = Path(__file__).stem
    
    configs = read_yaml(config_yaml)

    simulator_configs = configs["simulator_configs"]
    sample_rate = simulator_configs["sample_rate"]
    segment_seconds = simulator_configs["segment_seconds"]
    frames_per_sec = simulator_configs["frames_per_sec"]
    mics_meta = read_yaml(simulator_configs["mics_yaml"])
    mics_num = len(mics_meta["mic_coordinates"])

    device = configs["train"]["device"]
    model_name = configs["train"]["model_name"]

    segment_samples = int(segment_seconds * sample_rate)
    frames_num = int(segment_seconds * frames_per_sec) + 1

    frame_indexes, class_indexes, azimuths, elevations, distances = read_dcase2020_task3_csv(csv_path=gt_csv_path)

    # Load checkpoint
    model = get_model(model_name, mics_num)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    # Load audio
    audio, fs = librosa.load(path=audio_paths[0], sr=sample_rate, mono=False)
    audio_samples = audio.shape[-1]

    audio *= SCALE 
    # audi

    if False:
        audio = apply_lowpass_filter(
            audio=audio, 
            # cutoff_freq=simulator_configs["mic_cutoff_freq"], 
            cutoff_freq=1000,
            sample_rate=sample_rate,
        )

    # Mic positions
    mics_center_pos = np.array([0, 0, 2])
    mic_poss = get_mic_positions(mics_meta, frames_num)
    mic_poss = mic_poss[None, :, :, :]

    # Mic orientations
    mic_oriens = get_mic_orientations(mics_meta, frames_num)
    mic_oriens = mic_oriens[None, :, :, :]

    ##
    rays_num = 4
    agent_poss = np.repeat(mics_center_pos[None, None, None, :], repeats=frames_num, axis=2)
    agent_poss = np.repeat(agent_poss, repeats=rays_num, axis=1)
    # (1, rays_num, frames_num, 3)

    agent_look_at_distances = np.ones((1, rays_num, frames_num)) * PAD
    # (1, rays_num, frames_num)

    agent_distance_masks = np.zeros((1, rays_num, frames_num))
    # (1, rays_num, frames_num)

    ##
    bgn_sample = 0
    pred_wavs = []

    while bgn_sample < audio_samples:

        bgn_sec = bgn_sample / sample_rate
        print(bgn_sec)

        curr_frame = int(bgn_sec * 10)

        _azis, _eles = [], []
        tmp2 = []

        for i in range(len(frame_indexes)):
            if curr_frame < frame_indexes[i] < curr_frame + 20:
                if class_indexes[i] not in tmp2:
                    _azis.append(azimuths[i])
                    _eles.append(elevations[i])
                    tmp2.append(class_indexes[i])

        azis = np.zeros(rays_num)
        eles = np.zeros(rays_num)
        azis[0 : len(_azis)] = np.array(_azis)
        eles[0 : len(_eles)] = np.array(_eles)

        agent_look_at_directions = np.array(sph2cart(azis, eles, 1.))
        agent_look_at_directions = agent_look_at_directions[None, :, None, :]
        # agent_look_at_directions = np.repeat(a=agent_look_at_directions, repeats=rays_num, axis=1)
        agent_look_at_directions = np.repeat(a=agent_look_at_directions, repeats=frames_num, axis=2)

        #
        mic_wavs = audio[None, :, bgn_sample : bgn_sample + segment_samples]
        mic_wavs = librosa.util.fix_length(data=mic_wavs, size=segment_samples, axis=-1)

        pointer = 0
        batch_size = 2000
        output_dict = {}

        while pointer < rays_num:
            # print(pointer)

            _len = min(batch_size, rays_num - pointer)

            agent_detect_idxes = torch.Tensor(np.arange(0)[None, :])
            agent_distance_idxes = torch.Tensor(np.arange(0)[None, :])
            agent_sep_idxes = torch.Tensor(np.arange(_len)[None, :])

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

        pred_wav = output_dict["agent_look_at_direction_reverb_wav"][0]
        pred_wavs.append(pred_wav)

    pred_wavs = np.concatenate(pred_wavs, axis=-1)
    for i in range(pred_wavs.shape[0]):
        audio_path = "_zz_{}.wav".format(i)
        soundfile.write(file=audio_path, data=pred_wavs[i], samplerate=sample_rate)
        print(audio_path)


def panaroma_to_events(args):

    workspace = args.workspace
    sample_rate = 24000
    grid_deg = 2

    # Calculate prediction center.
    for name_idx, panaroma_path in enumerate(panaroma_paths):

        csv_path = Path(pred_csvs_dir, "{}.csv".format(Path(panaroma_path).stem))

        pana_tensor = pickle.load(open(panaroma_path, "rb"))
        frames_num, azi_grids, ele_grids = pana_tensor.shape

        params = []
        for frame_index in range(frames_num):
            # param = (frame_index, pred_tensor[frame_index], sed_mat[frame_index], grid_deg)
            param = (frame_index, pana_tensor[frame_index], grid_deg)
            params.append(param)

        # No parallel is faster
        results = []
        for param in params:
            result = _calculate_centers(param)
            results.append(result)

        # with ProcessPoolExecutor(max_workers=None) as pool: # Maximum workers on the machine.
        #     results = pool.map(_calculate_centers, params)

        locss = list(results)

        bgn = 0
        buffer = grouping(locss, bgn)
        
        # 
        new_buffer = {}

        for key in buffer.keys():
            if len(buffer[key]["frame_index"]) > 2:
                new_buffer[key] = buffer[key]

        buffer = new_buffer

        #
        audio_path = audio_paths[name_idx]
        audio, fs = librosa.load(path=audio_path, sr=sample_rate, mono=False)
        # audio_samples = audio.shape[-1]

        segments_dir = Path(results_dir, "segs", panaroma_path.stem)
        Path(segments_dir).mkdir(parents=True, exist_ok=True)

        # 
        for key in buffer.keys():

            frame_indexes = buffer[key]["frame_index"]
            locs = np.repeat(np.stack(buffer[key]["loc"], axis=0), repeats=10, axis=0)
            locs = np.concatenate((locs, locs[-1:]), axis=0)

            bgn_sec = frame_indexes[0] / 10
            end_sec = (frame_indexes[-1] + 1) / 10
            bgn_sample = round(bgn_sec * sample_rate)
            end_sample = round(end_sec * sample_rate)
            segment = audio[:, bgn_sample : end_sample]
            segment = librosa.util.fix_length(data=segment, size=end_sample - bgn_sample, axis=-1)
            # print(bgn_sec, end_sec, segment.shape)

            event = {
                "begin_time": bgn_sec,
                "end_time": end_sec,
                "locs": locs,
            }
            
            out_segment_path = Path(segments_dir, "{}_{:04.1f}s_{:04.1f}s.wav".format(panaroma_path.stem, bgn_sec, end_sec))
            out_event_path = Path(segments_dir, "{}_{:04.1f}s_{:04.1f}s.pkl".format(panaroma_path.stem, bgn_sec, end_sec))

            soundfile.write(file=out_segment_path, data=segment.T, samplerate=sample_rate)
            pickle.dump(event, open(out_event_path, "wb"))
            print("Write out to {}".format(out_segment_path))
            print("Write out to {}".format(out_event_path))


def segs_sep(args):

    workspace = args.workspace
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    filename = Path(__file__).stem
    
    configs = read_yaml(config_yaml)

    simulator_configs = configs["simulator_configs"]
    sample_rate = simulator_configs["sample_rate"]
    segment_seconds = simulator_configs["segment_seconds"]
    frames_per_sec = simulator_configs["frames_per_sec"]
    mics_meta = read_yaml(simulator_configs["mics_yaml"])
    mics_num = len(mics_meta["mic_coordinates"])

    device = configs["train"]["device"]
    model_name = configs["train"]["model_name"]

    segment_samples = int(segment_seconds * sample_rate)

    # Load checkpoint
    model = get_model(model_name, mics_num)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    for audio_path in audio_paths:
        segs_dir = Path(results_dir, "segs/{}".format(audio_path.stem))

        seg_paths = sorted(list(Path(segs_dir).glob("*.wav")))

        for seg_path in seg_paths:
            print(seg_path)

            segment, fs = librosa.load(path=seg_path, sr=sample_rate, mono=False)

            #
            frames_num = segment.shape[-1] // 240 + 1
            mics_center_pos = np.array([0, 0, 2])
            mic_poss = get_mic_positions(mics_meta, frames_num)
            mic_poss = mic_poss[None, :, :, :]

            # Mic orientations
            mic_oriens = get_mic_orientations(mics_meta, frames_num)
            mic_oriens = mic_oriens[None, :, :, :]

            rays_num = 1
            agent_poss = np.repeat(mics_center_pos[None, None, None, :], repeats=frames_num, axis=2)
            agent_poss = np.repeat(agent_poss, repeats=rays_num, axis=1)
            # (1, rays_num, frames_num, 3)

            event_path = Path(segs_dir, "{}.pkl".format(seg_path.stem))
            event = pickle.load(open(event_path, "rb"))
            locs = event["locs"]
            
            agent_look_at_directions = np.array(sph2cart(locs[:, 0], locs[:, 1], 1.))
            agent_look_at_directions = agent_look_at_directions[None, None, :, :]

            # Sep.
            agent_look_at_distances = np.ones((1, rays_num, frames_num)) * PAD
            # (1, rays_num, frames_num)

            agent_distance_masks = np.zeros((1, rays_num, frames_num))
            # (1, rays_num, frames_num)

            agent_detect_idxes = torch.Tensor(np.arange(0)[None, :])
            agent_distance_idxes = torch.Tensor(np.arange(0)[None, :])
            agent_sep_idxes = torch.Tensor(np.arange(rays_num)[None, :])

            batch_data = {
                "mic_wavs": segment[None, :, :],
                "mic_positions": mic_poss,
                "mic_orientations": mic_oriens,
                "agent_positions": agent_poss,
                "agent_look_at_directions": agent_look_at_directions,
                "agent_look_at_distances": agent_look_at_distances,
                "agent_distance_masks": agent_distance_masks,
                "agent_detect_idxes": agent_detect_idxes,
                "agent_distance_idxes": agent_distance_idxes,
                "agent_sep_idxes": agent_sep_idxes
            }
            
            for key in batch_data.keys():
                batch_data[key] = torch.Tensor(batch_data[key]).to(device)

            with torch.no_grad():
                model.eval()
                batch_output_dict = model(batch_data)

            out_wav = batch_output_dict["agent_look_at_direction_reverb_wav"].data.cpu().numpy()[0, 0]

            out_path = Path(results_dir, "segs_wavs/{}/{}.wav".format(audio_path.stem, seg_path.stem))
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            soundfile.write(file=out_path, data=out_wav, samplerate=sample_rate)
            print("write out to {}".format(out_path))


def segs_distance(args):

    workspace = args.workspace
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    filename = Path(__file__).stem
    
    configs = read_yaml(config_yaml)

    simulator_configs = configs["simulator_configs"]
    sample_rate = simulator_configs["sample_rate"]
    segment_seconds = simulator_configs["segment_seconds"]
    frames_per_sec = simulator_configs["frames_per_sec"]
    mics_meta = read_yaml(simulator_configs["mics_yaml"])
    mics_num = len(mics_meta["mic_coordinates"])

    device = configs["train"]["device"]
    model_name = configs["train"]["model_name"]

    segment_samples = int(segment_seconds * sample_rate)

    # Load checkpoint
    model = get_model(model_name, mics_num)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    for audio_path in audio_paths:
        segs_dir = Path(results_dir, "segs", audio_path.stem)

        seg_paths = sorted(list(Path(segs_dir).glob("*.wav")))

        for seg_path in seg_paths:
            print(seg_path)

            segment, fs = librosa.load(path=seg_path, sr=sample_rate, mono=False)

            #
            frames_num = segment.shape[-1] // 240 + 1
            mics_center_pos = np.array([0, 0, 2])
            mic_poss = get_mic_positions(mics_meta, frames_num)
            mic_poss = mic_poss[None, :, :, :]

            # Mic orientations
            mic_oriens = get_mic_orientations(mics_meta, frames_num)
            mic_oriens = mic_oriens[None, :, :, :]

            # Agents
            agent_look_at_distances = np.arange(0, 10, 0.01)
            agent_look_at_distances = agent_look_at_distances[None, :, None]
            agent_look_at_distances = np.repeat(a=agent_look_at_distances, repeats=frames_num, axis=2)
            rays_num = agent_look_at_distances.shape[1]

            agent_poss = np.repeat(mics_center_pos[None, None, None, :], repeats=frames_num, axis=2)
            agent_poss = np.repeat(agent_poss, repeats=rays_num, axis=1)
            # (1, rays_num, frames_num, 3)

            event_path = Path(segs_dir, "{}.pkl".format(seg_path.stem))
            event = pickle.load(open(event_path, "rb"))
            locs = event["locs"]
            
            agent_look_at_directions = np.array(sph2cart(locs[:, 0], locs[:, 1], 1.))
            agent_look_at_directions = np.repeat(agent_look_at_directions[None, None, :, :], repeats=rays_num, axis=1)

            agent_distance_masks = np.ones((1, rays_num, frames_num))
            # (1, rays_num, frames_num)

            pointer = 0
            batch_size = 200
            output_dict = {}

            while pointer < rays_num:
                # print(pointer)

                _len = min(batch_size, rays_num - pointer)

                agent_detect_idxes = torch.Tensor(np.arange(0)[None, :])
                agent_distance_idxes = torch.Tensor(np.arange(_len)[None, :])
                agent_sep_idxes = torch.Tensor(np.arange(0)[None, :])

                batch_data = {
                    "mic_wavs": segment[None, :, :],
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

            for key in output_dict.keys():
                output_dict[key] = np.concatenate(output_dict[key], axis=1)

            pred_distance = output_dict["agent_look_at_distance_has_source"][0].transpose(1, 0)

            # out_path = "_tmp_segments_results/{}/{}.pkl".format(audio_path.stem, seg_path.stem)
            out_path = Path(results_dir, "segs_dists", audio_path.stem, "{}.pkl".format(seg_path.stem))
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(pred_distance, open(out_path, "wb"))
            print("write out to {}".format(out_path))

            fig, axs = plt.subplots(2, 1, sharex=True)
            axs[1].matshow(pred_distance.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1.)
            out_path = Path(results_dir, "segs_dists", audio_path.stem, "{}.png".format(seg_path.stem))
            plt.savefig(out_path)
            print("write out to {}".format(out_path))


def combine_results(args):

    frames_per_sec = 100
    sample_rate = 24000

    for name_idx, panaroma_path in enumerate(panaroma_paths):

        pana_tensor = pickle.load(open(panaroma_path, "rb"))
        frames_num, azi_grids, ele_grids = pana_tensor.shape

        sed_path = Path(results_dir, "sed", "{}.pkl".format(Path(panaroma_path).stem))
        sed_tensor = pickle.load(open(sed_path, "rb"))

        segs_dir = Path(results_dir, "segs", panaroma_path.stem)
        event_paths = sorted(list(Path(segs_dir).glob("*.pkl")))

        list_tuples = []

        for event_path in event_paths:

            # Load part SED
            # print(event_path)
            event = pickle.load(open(event_path, "rb"))

            bgn_time = event["begin_time"]
            end_time = event["end_time"]
            locs = event["locs"]

            bgn_frame = round(bgn_time * frames_per_sec)
            end_frame = round(end_time * frames_per_sec)

            sed_part = sed_tensor[bgn_frame : end_frame + 1, :]

            # Load part sep wav
            wav_part_path = Path(results_dir, "segs_wavs", panaroma_path.stem, "{}.wav".format(event_path.stem))
            wav_part, _ = librosa.load(path=wav_part_path, sr=sample_rate, mono=False)

            # Load part distance
            dist_path = Path(results_dir, "segs_dists", panaroma_path.stem, "{}.pkl".format(event_path.stem))
            dist_part = pickle.load(open(dist_path, "rb"))

            sep_sed_path = Path(results_dir, "segs_sed", panaroma_path.stem, "{}.pkl".format(event_path.stem))
            sep_sed_part = pickle.load(open(sep_sed_path, "rb"))

            if False:
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(2, 1, sharex=True)
                axs[1].matshow(sed_part.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
                axs[1].yaxis.set_ticks(np.arange(len(LABELS)))
                axs[1].yaxis.set_ticklabels(LABELS)
                # plt.savefig("_zz.pdf")
                plt.savefig("_tmp2/{}.png".format(event_path.stem))

            if False:
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(2, 1, sharex=True)
                axs[1].matshow(sed_tensor.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
                axs[1].yaxis.set_ticks(np.arange(len(LABELS)))
                axs[1].yaxis.set_ticklabels(LABELS)
                plt.savefig("_zz.pdf")

            for t in range(sed_part.shape[0]):
                if np.max(sed_part[t]) > 0.8:
                    mask = sed_part[t] > 0.8
                    class_index = np.argmax(sep_sed_part[t] * mask)

                    frame_index = int(bgn_time * frames_per_sec) + t
                    azi = locs[t, 0]
                    ele = locs[t, 1]
                    dist = np.argmax(dist_part[t])

                    _tuple = (frame_index, class_index, azi, ele, dist)
                    list_tuples.append(_tuple)

        # Write
        csv_path = Path(pred_csvs_dir, "{}.csv".format(Path(panaroma_path).stem))
        Path(csv_path.parent).mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w') as fw:
            for _tuple in list_tuples[0 :: 10]:

                frame_index = _tuple[0] // 10
                class_index = _tuple[1]
                azi = np.rad2deg(_tuple[2])
                ele = np.rad2deg(_tuple[3])
                dist = _tuple[4]

                track_id = 0

                fw.write("{},{},{},{},{}\n".format(
                    frame_index, 
                    class_index, 
                    track_id,
                    int(np.around(azi)), 
                    int(np.around(ele))
                ))

        print("Write out to {}".format(csv_path))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest="mode")

    parser_inference = subparsers.add_parser("inference")
    parser_inference.add_argument('--workspace', type=str)
    parser_inference.add_argument('--config_yaml', type=str)
    parser_inference.add_argument("--checkpoint_path", type=str)

    parser_write_loc_csv = subparsers.add_parser("write_loc_csv")
    parser_write_loc_csv.add_argument('--workspace', type=str)

    parser_plot_panaroma = subparsers.add_parser("plot_panaroma")
    parser_plot_panaroma.add_argument('--workspace', type=str)

    parser_panaroma_to_events = subparsers.add_parser("panaroma_to_events")
    parser_panaroma_to_events.add_argument('--workspace', type=str)

    parser_inference = subparsers.add_parser("inference_sep")
    parser_inference.add_argument('--workspace', type=str)
    parser_inference.add_argument('--config_yaml', type=str)
    parser_inference.add_argument("--checkpoint_path", type=str)

    parser_segs_sep = subparsers.add_parser("segs_sep")
    parser_segs_sep.add_argument('--workspace', type=str)
    parser_segs_sep.add_argument('--config_yaml', type=str)
    parser_segs_sep.add_argument("--checkpoint_path", type=str)

    parser_segs_distance = subparsers.add_parser("segs_distance")
    parser_segs_distance.add_argument('--workspace', type=str)
    parser_segs_distance.add_argument('--config_yaml', type=str)
    parser_segs_distance.add_argument("--checkpoint_path", type=str)

    parser_combine_results = subparsers.add_parser("combine_results")

    args = parser.parse_args()

    if args.mode == "inference":
        inference(args)

    elif args.mode == "write_loc_csv":
        write_loc_csv(args)

    elif args.mode == "plot_panaroma":
        plot_panaroma(args)

    elif args.mode == "panaroma_to_events":
        panaroma_to_events(args)

    elif args.mode == "inference_sep":
        inference_sep(args)

    elif args.mode == "segs_sep":
        segs_sep(args)

    elif args.mode == "segs_distance":
        segs_distance(args)

    elif args.mode == "combine_results":
        combine_results(args)

    else:
        raise Exception("Error argument!")