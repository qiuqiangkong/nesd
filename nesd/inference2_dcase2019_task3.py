import argparse
import logging
import os
import pathlib
from functools import partial
from typing import Dict, List, NoReturn
import h5py
import time
import glob
import soundfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import pickle
import lightning as L
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import torchaudio
import pytorch_lightning as pl
# from pytorch_lightning.plugins import DDPPlugin
import torch 
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint

from nesd.utils import read_yaml, create_logging, PAD
from nesd.data.samplers import BatchSampler, DistributedSamplerWrapper
from nesd.data.data_modules import DataModule, Dataset
from nesd.data.data_modules import *
from nesd.models.models01 import *
from nesd.models.models02 import *
# from nesd.models.lightning_modules import LitModel
from nesd.optimizers.lr_schedulers import get_lr_lambda
from nesd.losses import *
# from nesd.callbacks.callback import get_callback

from nesd.test_dataloader import Dataset3, collate_fn
from nesd.train2 import LitModel
from nesd.image_source_simulator import expand_frame_dim
from nesd.inference2 import get_all_agent_look_directions, forward_in_batch


LABELS = ['clearthroat', 'cough', 'doorslam', 'drawer', 'keyboard', 'keysDrop', 'knock', 'laughter', 'pageturn', 'phone', 'speech']

LB_TO_ID = {lb: id for id, lb in enumerate(LABELS)}
ID_TO_LB = {id: lb for id, lb in enumerate(LABELS)}


def inference(args):

    workspace = args.workspace
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    filename = args.filename
    device = "cuda"

    configs = read_yaml(config_yaml)
    model_type = configs['train']['model_type']
    simulator_configs = configs["simulator_configs"]
    mics_yaml = simulator_configs["mics_yaml"]
    lowpass_freq = configs["lowpass_freq"] if "lowpass_freq" in configs.keys() else None

    num_workers = 0
    batch_size = 32
    frames_num = 201
    sample_rate = 24000
    segment_seconds = 2
    segment_samples = int(sample_rate * segment_seconds)

    # Load checkpoint
    # checkpoint_path = "./tmp/epoch=8-step=9000-test_loss=0.094.ckpt"
    checkpoint = torch.load(checkpoint_path)
    # device = "cuda"

    Net = eval(model_type)
    net = Net(mics_num=4)

    model = LitModel.load_from_checkpoint(
        net=net, 
        loss_function=None,
        learning_rate=None,
        checkpoint_path=checkpoint_path,

    )

    # Load audio
    # audio_path = "/home/qiuqiangkong/datasets/dcase2019/task3/downloaded_package/mic_dev/split1_ir0_ov1_1.wav"
    audio_path = "/home/qiuqiangkong/datasets/dcase2019/task3/downloaded_package/mic_eval/split0_1.wav"
    audio, fs = librosa.load(path=audio_path, sr=sample_rate, mono=False)

    # audio *= 50
    # soundfile.write(file="_zz.wav", data=audio.T, samplerate=sample_rate)
    # from IPython import embed; embed(using=False); os._exit(0)

    if lowpass_freq is not None:
        audio = torchaudio.functional.lowpass_biquad(
            waveform=torch.Tensor(audio),
            sample_rate=sample_rate,
            cutoff_freq=500,
        ).data.cpu().numpy()

    audio_samples = audio.shape[1]

    # agents
    grid_deg = 2
    azimuth_grids = 360 // grid_deg
    elevation_grids = 180 // grid_deg

    agent_look_azimuths, agent_look_colatitudes = get_all_agent_look_directions(grid_deg)

    agent_look_directions = np.stack(sph2cart(
        r=1., azimuth=agent_look_azimuths, colatitude=agent_look_colatitudes), axis=-1)

    rays_num = agent_look_directions.shape[0]

    # center_pos = np.array([
    #     simulator_configs["room_min_length"] / 2,
    #     simulator_configs["room_min_width"] / 2,
    #     simulator_configs["room_min_height"] / 2
    # ])
    center_pos = np.array([4, 4, 2])
    agent_positions = center_pos[None, None, None, :]
    agent_positions = np.repeat(a=agent_positions, repeats=rays_num, axis=1)
    agent_positions = np.repeat(a=agent_positions, repeats=frames_num, axis=2)

    agent_look_directions = np.repeat(
        a=agent_look_directions[None, :, None, :],
        repeats=frames_num,
        axis=-2
    )

    #
    mics_num = 4
    mic_look_directions = np.ones((mics_num, 3))
    mic_look_directions = np.repeat(
        a=mic_look_directions[None, :, None, :],
        repeats=frames_num,
        axis=-2
    )

    mic_positions = get_mic_positions(center_pos, mics_yaml)
    mic_positions = np.repeat(
        a=mic_positions[None, :, None, :],
        repeats=frames_num,
        axis=-2
    )

    pointer = 0

    pred_tensor = []

    while pointer + segment_samples < audio_samples:

        print(pointer / sample_rate)

        segment = audio[:, pointer : pointer + segment_samples]

        input_dict = {
            "mic_positions": mic_positions,
            "mic_look_directions": mic_look_directions,
            "mic_signals": segment[None, ...],
            "agent_positions": agent_positions,
            "agent_look_directions": agent_look_directions,
        }

        input_dict["agent_look_depths"] = PAD * np.ones(agent_look_directions.shape[0:-1] + (1,))

        for key in input_dict.keys():
            input_dict[key] = torch.Tensor(input_dict[key]).to(device)

        output_dict = forward_in_batch(model=model, input_dict=input_dict, mode="inference") 

        tmp = output_dict["agent_look_directions_has_source"][:, 0:-1:10].reshape((azimuth_grids, elevation_grids, -1)).transpose(2, 0, 1)

        pred_tensor.append(tmp)

        pointer += segment_samples

        # agent_look_directions_has_source = np.mean(output_dict["agent_look_directions_has_source"], axis=1)

        # source_positions = [e[0] for e in data_dict["source_positions"][i]] # (sources_num, 3)
        # agent_position = data_dict["agent_positions"][i][0, 0]  # (3,)

        # pickle.dump([agent_look_directions_has_source, source_positions, agent_position.data.cpu().numpy()], open("_zz.pkl", "wb"))

        # add(None)

    pred_tensor = np.concatenate(pred_tensor, axis=0)

    pickle.dump(pred_tensor, open("_zz2.pkl", "wb"))
    from IPython import embed; embed(using=False); os._exit(0)


def get_mic_positions(center_pos, mics_yaml):

    mics_pos = []

    # mics_yaml = "./microphones/eigenmike.yaml"

    with open(mics_yaml, 'r') as f:
        mics_meta = yaml.load(f, Loader=yaml.FullLoader)

    for mic_meta in mics_meta:

        relative_mic_pos = np.array(sph2cart(
            r=mic_meta['radius'], 
            azimuth=mic_meta['azimuth'], 
            colatitude=mic_meta['colatitude']
        ))

        mic_pos = center_pos + relative_mic_pos

        mics_pos.append(mic_pos)

    return np.stack(mics_pos, axis=0)


'''
def plot(args):

    csv_path = "/home/qiuqiangkong/datasets/dcase2019/task3/downloaded_package/metadata_dev/split1_ir0_ov1_1.csv"

    frame_indexes, class_indexes, azimuths, colatitudes = read_dcase2019_task3_csv(csv_path=csv_path)

    #
    pred_tensor = pickle.load(open("_zz2.pkl", "rb"))
    frames_num = pred_tensor.shape[0]

    grid_deg = 2
    azimuth_grids = 360 // grid_deg
    elevation_grids = 180 // grid_deg
    
    gt_tensor = np.zeros((frames_num + 20, azimuth_grids, elevation_grids))
    half_angle = math.atan2(0.1, 1)

    for l in range(len(frame_indexes)):
        print(l)
        
        frame_index = frame_indexes[l]
        source_azi = azimuths[l]
        source_col = colatitudes[l]

        source_direction = np.array(sph2cart(1., source_azi, source_col))

        for i in range(gt_tensor[frame_index].shape[0]):
            for j in range(gt_tensor[frame_index].shape[1]):
                _azi = np.deg2rad(i * grid_deg)
                _col = np.deg2rad(j * grid_deg)

                plot_direction = np.array(sph2cart(1., _azi, _col))

                ray_angle = np.arccos(get_cos(source_direction, plot_direction))

                if ray_angle < half_angle:
                    gt_tensor[frame_index, i, j] = 1
    
    Path("_tmp").mkdir(parents=True, exist_ok=True)

    for n in range(pred_tensor.shape[0]):
        print(n)

        plt.figure(figsize=(20, 10))
        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].matshow(gt_tensor[n].T, origin='upper', aspect='equal', cmap='jet', vmin=0, vmax=1)
        axs[1].matshow(pred_tensor[n].T, origin='upper', aspect='equal', cmap='jet', vmin=0, vmax=1)
        for i in range(2):
            axs[i].grid(color='w', linestyle='--', linewidth=0.1)
            axs[i].xaxis.set_ticks(np.arange(0, azimuth_grids+1, 10))
            axs[i].yaxis.set_ticks(np.arange(0, elevation_grids+1, 10))
            axs[i].xaxis.set_ticklabels(np.arange(0, 361, 10 * grid_deg), rotation=90)
            axs[i].yaxis.set_ticklabels(np.arange(0, 181, 10 * grid_deg))

        plt.savefig('_tmp/_zz_{:04d}.png'.format(n))

    from IPython import embed; embed(using=False); os._exit(0)
'''

def plot(args):

    # csv_path = "/home/qiuqiangkong/datasets/dcase2019/task3/downloaded_package/metadata_dev/split1_ir0_ov1_1.csv"
    csv_path = "/home/qiuqiangkong/datasets/dcase2019/task3/downloaded_package/metadata_eval/split0_1.csv"

    frame_indexes, class_indexes, azimuths, colatitudes, distances = read_dcase2019_task3_csv(csv_path=csv_path)

    #
    pred_tensor = pickle.load(open("_zz2.pkl", "rb"))
    frames_num = pred_tensor.shape[0]

    grid_deg = 2
    azimuth_grids = 360 // grid_deg
    elevation_grids = 180 // grid_deg
    
    gt_tensor = np.zeros((frames_num + 20, azimuth_grids, elevation_grids))
    half_angle = math.atan2(0.1, 1)

    # Get GT
    params = []

    for n in range(len(frame_indexes)):

        frame_index = frame_indexes[n]
        class_index = class_indexes[n]
        source_azi = azimuths[n]
        source_col = colatitudes[n]

        param = (frame_index, class_index, source_azi, source_col, azimuth_grids, elevation_grids, grid_deg, half_angle)
        params.append(param)

    # for param in params:
    #     _multiple_process_gt_mat(param)

    with ProcessPoolExecutor(max_workers=None) as pool: # Maximum workers on the machine.
        results = pool.map(_multiple_process_gt_mat, params)

    gt_texts = [""] * gt_tensor.shape[0]
    for (frame_index, class_index, gt_mat) in results:
        gt_tensor[frame_index] = gt_mat
        gt_texts[frame_index] = ID_TO_LB[class_index]

    Path("_tmp").mkdir(parents=True, exist_ok=True)

    # Plot
    params = []

    for n in range(min(pred_tensor.shape[0], gt_tensor.shape[0])):
        param = (n, gt_texts[n], gt_tensor[n], pred_tensor[n], azimuth_grids, elevation_grids, grid_deg)
        params.append(param)

    # for param in params:
    #     _multiple_process_plot(param)

    with ProcessPoolExecutor(max_workers=None) as pool: # Maximum workers on the machine.
        results = pool.map(_multiple_process_plot, params)


def _multiple_process_gt_mat(param):

    frame_index, class_index, source_azi, source_col, azimuth_grids, elevation_grids, grid_deg, half_angle = param
    print(frame_index)

    gt_mat = np.zeros((azimuth_grids, elevation_grids))

    source_direction = np.array(sph2cart(1., source_azi, source_col))

    tmp = []

    for i in range(gt_mat.shape[0]):
        for j in range(gt_mat.shape[1]):
            _azi = np.deg2rad(i * grid_deg)
            _col = np.deg2rad(j * grid_deg)

            plot_direction = np.array(sph2cart(1., _azi, _col))

            ray_angle = np.arccos(get_cos(source_direction, plot_direction))

            if ray_angle < half_angle:
                gt_mat[i, j] = 1
                tmp.append((i, j))

    # if class_index == 0:
    #     from IPython import embed; embed(using=False); os._exit(0)

    return frame_index, class_index, gt_mat


def _multiple_process_plot(param):

    n, gt_text, gt_mat, pred_mat, azimuth_grids, elevation_grids, grid_deg = param
    print("Plot: {}".format(n))

    plt.figure(figsize=(20, 10))
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].matshow(gt_mat.T, origin='upper', aspect='equal', cmap='jet', vmin=0, vmax=1)
    axs[1].matshow(pred_mat.T, origin='upper', aspect='equal', cmap='jet', vmin=0, vmax=1)
    for i in range(2):
        axs[i].grid(color='w', linestyle='--', linewidth=0.1)
        axs[i].xaxis.set_ticks(np.arange(0, azimuth_grids+1, 10))
        axs[i].yaxis.set_ticks(np.arange(0, elevation_grids+1, 10))
        axs[i].xaxis.set_ticklabels(np.arange(0, 361, 10 * grid_deg), rotation=90)
        axs[i].yaxis.set_ticklabels(np.arange(0, 181, 10 * grid_deg))
    axs[0].set_title(gt_text)

    plt.savefig('_tmp/_zz_{:04d}.png'.format(n))


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

    azimuths = np.deg2rad(np.array(_azimuths) % 360)
    colatitudes = np.deg2rad(90 - np.array(_elevations))
    distances = np.array(_distances)

    return frame_indexes, class_indexes, azimuths, colatitudes, distances


def inference_depth(args):

    workspace = args.workspace
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    filename = args.filename
    device = "cuda"

    configs = read_yaml(config_yaml)
    model_type = configs['train']['model_type']
    simulator_configs = configs["simulator_configs"]
    mics_yaml = simulator_configs["mics_yaml"]
    lowpass_freq = configs["lowpass_freq"] if "lowpass_freq" in configs.keys() else None

    num_workers = 0
    batch_size = 32
    frames_num = 201
    sample_rate = 24000
    segment_seconds = 2
    segment_samples = int(sample_rate * segment_seconds)

    # Load checkpoint
    # checkpoint_path = "./tmp/epoch=8-step=9000-test_loss=0.094.ckpt"
    checkpoint = torch.load(checkpoint_path)
    # device = "cuda"

    Net = eval(model_type)
    net = Net(mics_num=4)

    model = LitModel.load_from_checkpoint(
        net=net, 
        loss_function=None,
        learning_rate=None,
        checkpoint_path=checkpoint_path,

    )

    csv_path = "/home/qiuqiangkong/datasets/dcase2019/task3/downloaded_package/metadata_eval/split0_1.csv"
    frame_indexes, class_indexes, azimuths, colatitudes, distances = read_dcase2019_task3_csv(csv_path=csv_path)

    # Load audio
    # audio_path = "/home/qiuqiangkong/datasets/dcase2019/task3/downloaded_package/mic_dev/split1_ir0_ov1_1.wav"
    audio_path = "/home/qiuqiangkong/datasets/dcase2019/task3/downloaded_package/mic_eval/split0_1.wav"
    audio, fs = librosa.load(path=audio_path, sr=sample_rate, mono=False)

    # audio *= 50
    # soundfile.write(file="_zz.wav", data=audio.T, samplerate=sample_rate)
    # from IPython import embed; embed(using=False); os._exit(0)

    if lowpass_freq is not None:
        audio = torchaudio.functional.lowpass_biquad(
            waveform=torch.Tensor(audio),
            sample_rate=sample_rate,
            cutoff_freq=500,
        ).data.cpu().numpy()

    audio_samples = audio.shape[1]

    # agents
    grid_deg = 2
    azimuth_grids = 360 // grid_deg
    elevation_grids = 180 // grid_deg

    # agent_look_azimuths, agent_look_colatitudes = get_all_agent_look_directions(grid_deg)

    # agent_look_directions = np.stack(sph2cart(
    #     r=1., azimuth=agent_look_azimuths, colatitude=agent_look_colatitudes), axis=-1)

    # rays_num = agent_look_directions.shape[0]

    agent_look_depths = np.arange(0, 10, 0.01)
    agent_look_depths = agent_look_depths[None, :, None, None]
    agent_look_depths = np.repeat(a=agent_look_depths, repeats=frames_num, axis=2)
    rays_num = agent_look_depths.shape[1]

    # center_pos = np.array([
    #     simulator_configs["room_min_length"] / 2,
    #     simulator_configs["room_min_width"] / 2,
    #     simulator_configs["room_min_height"] / 2
    # ])
    center_pos = np.array([4, 4, 2])
    agent_positions = center_pos[None, None, None, :]
    agent_positions = np.repeat(a=agent_positions, repeats=rays_num, axis=1)
    agent_positions = np.repeat(a=agent_positions, repeats=frames_num, axis=2)

    # agent_look_directions = np.repeat(
    #     a=agent_look_directions[None, :, None, :],
    #     repeats=frames_num,
    #     axis=-2
    # )

    #
    mics_num = 4
    mic_look_directions = np.ones((mics_num, 3))
    mic_look_directions = np.repeat(
        a=mic_look_directions[None, :, None, :],
        repeats=frames_num,
        axis=-2
    )

    mic_positions = get_mic_positions(center_pos, mics_yaml)
    mic_positions = np.repeat(
        a=mic_positions[None, :, None, :],
        repeats=frames_num,
        axis=-2
    )

    pointer = 0

    pred_tensor = []

    while pointer + segment_samples < audio_samples:

        print(pointer / sample_rate)

        curr_sec = pointer / sample_rate
        curr_frame = curr_sec * 10

        for i in range(len(frame_indexes)):
            if frame_indexes[i] > curr_frame:
                break

        azi = azimuths[i]
        col = colatitudes[i]
        dist = distances[i]

        agent_look_directions = np.array(sph2cart(r=1., azimuth=azi, colatitude=col))
        agent_look_directions = agent_look_directions[None, None, None, :]
        agent_look_directions = np.repeat(a=agent_look_directions, repeats=rays_num, axis=1)
        agent_look_directions = np.repeat(a=agent_look_directions, repeats=frames_num, axis=2)

        segment = audio[:, pointer : pointer + segment_samples]

        input_dict = {
            "mic_positions": mic_positions,
            "mic_look_directions": mic_look_directions,
            "mic_signals": segment[None, ...],
            "agent_positions": agent_positions,
            "agent_look_depths": agent_look_depths,
            "agent_look_directions": agent_look_directions,
        }

        for key in input_dict.keys():
            input_dict[key] = torch.Tensor(input_dict[key]).to(device)

        # from IPython import embed; embed(using=False); os._exit(0)

        output_dict = forward_in_batch(model=model, input_dict=input_dict) 

        tmp = output_dict["agent_look_depths_has_source"][:, 0:-1:10].transpose(1, 0)

        pred_tensor.append(tmp)

        pointer += segment_samples

        # agent_look_directions_has_source = np.mean(output_dict["agent_look_directions_has_source"], axis=1)

        # source_positions = [e[0] for e in data_dict["source_positions"][i]] # (sources_num, 3)
        # agent_position = data_dict["agent_positions"][i][0, 0]  # (3,)

        # pickle.dump([agent_look_directions_has_source, source_positions, agent_position.data.cpu().numpy()], open("_zz.pkl", "wb"))

        # add(None)

    pred_tensor = np.concatenate(pred_tensor, axis=0)

    pickle.dump(pred_tensor, open("_zz2_depth.pkl", "wb"))
    from IPython import embed; embed(using=False); os._exit(0)


def plot_depth(args):

    # csv_path = "/home/qiuqiangkong/datasets/dcase2019/task3/downloaded_package/metadata_dev/split1_ir0_ov1_1.csv"
    csv_path = "/home/qiuqiangkong/datasets/dcase2019/task3/downloaded_package/metadata_eval/split0_1.csv"

    frame_indexes, class_indexes, azimuths, colatitudes, distances = read_dcase2019_task3_csv(csv_path=csv_path)

    #
    pred_tensor = pickle.load(open("_zz2_depth.pkl", "rb"))
    frames_num = pred_tensor.shape[0]

    fig, ax = plt.subplots(1, 1, sharex=True)
    ax.matshow(pred_tensor.T, origin='lower', aspect='auto', cmap='jet')
    plt.savefig('_zz.pdf')

    from IPython import embed; embed(using=False); os._exit(0)


def inference_sep(args):

    workspace = args.workspace
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    filename = args.filename
    device = "cuda"

    configs = read_yaml(config_yaml)
    model_type = configs['train']['model_type']
    simulator_configs = configs["simulator_configs"]
    mics_yaml = simulator_configs["mics_yaml"]
    lowpass_freq = configs["lowpass_freq"] if "lowpass_freq" in configs.keys() else None

    num_workers = 0
    batch_size = 32
    frames_num = 201
    sample_rate = 24000
    segment_seconds = 2
    segment_samples = int(sample_rate * segment_seconds)

    # Load checkpoint
    # checkpoint_path = "./tmp/epoch=8-step=9000-test_loss=0.094.ckpt"
    checkpoint = torch.load(checkpoint_path)
    # device = "cuda"

    Net = eval(model_type)
    net = Net(mics_num=4)

    model = LitModel.load_from_checkpoint(
        net=net, 
        loss_function=None,
        learning_rate=None,
        checkpoint_path=checkpoint_path,

    )

    csv_path = "/home/qiuqiangkong/datasets/dcase2019/task3/downloaded_package/metadata_eval/split0_1.csv"
    frame_indexes, class_indexes, azimuths, colatitudes, distances = read_dcase2019_task3_csv(csv_path=csv_path)

    # Load audio
    # audio_path = "/home/qiuqiangkong/datasets/dcase2019/task3/downloaded_package/mic_dev/split1_ir0_ov1_1.wav"
    audio_path = "/home/qiuqiangkong/datasets/dcase2019/task3/downloaded_package/mic_eval/split0_1.wav"
    audio, fs = librosa.load(path=audio_path, sr=sample_rate, mono=False)

    # audio *= 50
    # soundfile.write(file="_zz.wav", data=audio.T, samplerate=sample_rate)
    # from IPython import embed; embed(using=False); os._exit(0)

    if lowpass_freq is not None:
        audio = torchaudio.functional.lowpass_biquad(
            waveform=torch.Tensor(audio),
            sample_rate=sample_rate,
            cutoff_freq=500,
        ).data.cpu().numpy()

    audio_samples = audio.shape[1]

    # agents
    grid_deg = 2
    azimuth_grids = 360 // grid_deg
    elevation_grids = 180 // grid_deg

    # agent_look_azimuths, agent_look_colatitudes = get_all_agent_look_directions(grid_deg)

    # agent_look_directions = np.stack(sph2cart(
    #     r=1., azimuth=agent_look_azimuths, colatitude=agent_look_colatitudes), axis=-1)

    # rays_num = agent_look_directions.shape[0]

    # agent_look_depths = np.arange(0, 10, 0.01)
    # agent_look_depths = agent_look_depths[None, :, None, None]
    # agent_look_depths = np.repeat(a=agent_look_depths, repeats=frames_num, axis=2)
    rays_num = 2

    # center_pos = np.array([
    #     simulator_configs["room_min_length"] / 2,
    #     simulator_configs["room_min_width"] / 2,
    #     simulator_configs["room_min_height"] / 2
    # ])
    center_pos = np.array([4, 4, 2])
    agent_positions = center_pos[None, None, None, :]
    agent_positions = np.repeat(a=agent_positions, repeats=rays_num, axis=1)
    agent_positions = np.repeat(a=agent_positions, repeats=frames_num, axis=2)

    # agent_look_directions = np.repeat(
    #     a=agent_look_directions[None, :, None, :],
    #     repeats=frames_num,
    #     axis=-2
    # )

    #
    mics_num = 4
    mic_look_directions = np.ones((mics_num, 3))
    mic_look_directions = np.repeat(
        a=mic_look_directions[None, :, None, :],
        repeats=frames_num,
        axis=-2
    )

    mic_positions = get_mic_positions(center_pos, mics_yaml)
    mic_positions = np.repeat(
        a=mic_positions[None, :, None, :],
        repeats=frames_num,
        axis=-2
    )

    pointer = 0

    pred_tensor = []

    while pointer + segment_samples < audio_samples:

        print(pointer / sample_rate)

        curr_sec = pointer / sample_rate
        curr_frame = curr_sec * 10

        for i in range(len(frame_indexes)):
            if frame_indexes[i] > curr_frame:
                break

        azi = azimuths[i]
        col = colatitudes[i]
        dist = distances[i]

        agent_look_directions = np.array(sph2cart(r=1., azimuth=azi, colatitude=col))
        agent_look_directions = agent_look_directions[None, None, None, :]
        agent_look_directions = np.repeat(a=agent_look_directions, repeats=rays_num, axis=1)
        agent_look_directions = np.repeat(a=agent_look_directions, repeats=frames_num, axis=2)

        segment = audio[:, pointer : pointer + segment_samples]

        input_dict = {
            "mic_positions": mic_positions,
            "mic_look_directions": mic_look_directions,
            "mic_signals": segment[None, ...],
            "agent_positions": agent_positions,
            # "agent_look_depths": agent_look_depths,
            "agent_look_directions": agent_look_directions,
        }

        input_dict["agent_look_depths"] = PAD * np.ones(agent_look_directions.shape[0:-1] + (1,))

        for key in input_dict.keys():
            input_dict[key] = torch.Tensor(input_dict[key]).to(device)

        output_dict = forward_in_batch(model=model, input_dict=input_dict)

        # from IPython import embed; embed(using=False); os._exit(0)

        tmp = output_dict["agent_signals"][0]

        pred_tensor.append(tmp)

        pointer += segment_samples

        # agent_look_directions_has_source = np.mean(output_dict["agent_look_directions_has_source"], axis=1)

        # source_positions = [e[0] for e in data_dict["source_positions"][i]] # (sources_num, 3)
        # agent_position = data_dict["agent_positions"][i][0, 0]  # (3,)

        # pickle.dump([agent_look_directions_has_source, source_positions, agent_position.data.cpu().numpy()], open("_zz.pkl", "wb"))

        # add(None)

    pred_tensor = np.concatenate(pred_tensor, axis=0)

    soundfile.write(file="_zz.wav", data=pred_tensor, samplerate=sample_rate)

    # pickle.dump(pred_tensor, open("_zz2_depth.pkl", "wb"))
    from IPython import embed; embed(using=False); os._exit(0)


def sub(x):
    x[0] = 10
    # time.sleep(6 - x) 
    # print(x)
    return x


def add(args):
    # params = [1,2,3,4,5]
    params = [np.zeros(3), np.zeros(4)]

    # for param in params:
    #     sub(param)

    with ProcessPoolExecutor(max_workers=4) as pool: # Maximum workers on the machine.
        results = pool.map(sub, params)

    for x in results:
        print(x)
    from IPython import embed; embed(using=False); os._exit(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest="mode")

    parser_inference = subparsers.add_parser("inference")
    parser_inference.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
    parser_inference.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path of config file for training.",
    )
    parser_inference.add_argument(
        "--checkpoint_path", type=str, required=True, help="Directory of workspace."
    )

    parser_inference_depth = subparsers.add_parser("inference_depth")
    parser_inference_depth.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
    parser_inference_depth.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path of config file for training.",
    )
    parser_inference_depth.add_argument(
        "--checkpoint_path", type=str, required=True, help="Directory of workspace."
    )

    parser_inference_sep = subparsers.add_parser("inference_sep")
    parser_inference_sep.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
    parser_inference_sep.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path of config file for training.",
    )
    parser_inference_sep.add_argument(
        "--checkpoint_path", type=str, required=True, help="Directory of workspace."
    )

    parser_inference = subparsers.add_parser("plot")
    parser_inference = subparsers.add_parser("plot_depth")

    parser_inference = subparsers.add_parser("add")
    
    args = parser.parse_args()
    args.filename = pathlib.Path(__file__).stem 

    if args.mode == "inference":
        inference(args)

    elif args.mode == "inference_depth":
        inference_depth(args)

    elif args.mode == "inference_sep":
        inference_sep(args)

    elif args.mode == "plot": 
        plot(args)

    elif args.mode == "plot_depth": 
        plot_depth(args)

    elif args.mode == "add": 
        add(args)

    else:
        raise Exception("Error argument!")