import argparse
import pathlib
import os
import math
import numpy as np
import torch
import soundfile
import matplotlib.pyplot as plt
import pandas as pd

from nesd.data.samplers import Sampler
from nesd.data.data_modules import DataModule, Dataset
from nesd.data.data_modules import *
from nesd.utils import read_yaml, create_logging, sph2cart, get_cos, norm
from nesd.models.models01 import *


def inference(args):

    workspace = args.workspace
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    gpus = args.gpus
    filename = args.filename

    configs = read_yaml(config_yaml) 
    sampler_type = configs['sampler_type']
    dataset_type = configs['dataset_type']
    model_type = configs['train']['model_type']
    do_localization = configs['train']['do_localization']
    do_sed = configs['train']['do_sed']
    do_separation = configs['train']['do_separation']
    evaluate_step_frequency = configs['train']['evaluate_step_frequency']
    save_step_frequency = configs['train']['save_step_frequency']
    batch_size = configs['train']['batch_size']
    steps_per_epoch = configs['train']['steps_per_epoch']

    split = 'test' 

    if split == 'train':
        hdf5s_dir = os.path.join(workspace, configs['sources']['train_hdf5s_dir'])
        random_seed = 1234

    elif split == 'test':
        hdf5s_dir = os.path.join(workspace, configs['sources']['test_hdf5s_dir'])
        random_seed = 2345

    num_workers = 8
    distributed = True if gpus > 1 else False
    device = 'cuda'

    frames_num = 301
    classes_num = -1

    Model = eval(model_type)

    model = Model(
        microphones_num=4, 
        classes_num=classes_num, 
        do_localization=do_localization,
        do_sed=do_sed,
        do_separation=do_separation,
    )

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    print(
        "Load pretrained checkpoint from {}".format(checkpoint_path)
    )
    model.to(device)

    _Sampler = eval(sampler_type)
    _Dataset = eval(dataset_type)

    # sampler
    train_sampler = _Sampler(
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        random_seed=random_seed,
    )

    train_dataset = _Dataset(
        hdf5s_dir=hdf5s_dir,
    )

    # data module
    data_module = DataModule(
        train_sampler=train_sampler,
        train_dataset=train_dataset,
        num_workers=num_workers,
        distributed=distributed,
    )

    data_module.setup()

    cnt = 0
    losses = []

    grid_deg = 2
    azimuth_grids = 360 // grid_deg
    elevation_grids = 180 // grid_deg

    agent_look_azimuths, agent_look_colatitudes = get_all_agent_look_directions(grid_deg)

    agent_look_directions = np.stack(sph2cart(
        r=1., azimuth=agent_look_azimuths, colatitude=agent_look_colatitudes), axis=-1)

    agent_look_directions = np.tile(agent_look_directions[None, :, None, :], (1, 1, frames_num, 1))

    agents_num = agent_look_directions.shape[1]

    for batch_data_dict in data_module.train_dataloader():
        
        i = 0
        max_agents_contain_waveform = 2

        input_dict = {
            'mic_position': batch_data_dict['mic_position'][i : i + 1, :, :, :].to(device),
            'mic_look_direction': batch_data_dict['mic_look_direction'][i : i + 1, :, :, :].to(device),
            'mic_waveform': batch_data_dict['mic_waveform'][i : i + 1, :, :].to(device),
            'agent_position': batch_data_dict['agent_position'][i : i + 1, 0 : 1, :, :].repeat(1, agents_num, 1, 1).to(device),
            'agent_look_direction': torch.Tensor(agent_look_directions).to(device),
            'max_agents_contain_waveform': max_agents_contain_waveform,
        }

        # from IPython import embed; embed(using=False); os._exit(0)
        t = -1
        source_positions = batch_data_dict['source_position'][i][:, t, :]
        agent_position = batch_data_dict['agent_position'][i][0, t, :].data.cpu().numpy()
        sources_num = source_positions.shape[0]
        break

    output_dict = forward_in_batch(model, input_dict, do_separation=False)

    pred_mat = np.mean(output_dict['agent_see_source'], axis=1).reshape(azimuth_grids, elevation_grids)
    # pred_mat = output_dict['agent_see_source'][:, t].reshape(azimuth_grids, elevation_grids) 

    gt_mat = np.zeros((azimuth_grids, elevation_grids))
    half_angle = math.atan2(0.1, 1)

    for i in range(gt_mat.shape[0]):
        for j in range(gt_mat.shape[1]):
            _azi = np.deg2rad(i * grid_deg)
            _zen = np.deg2rad(j * grid_deg)
            plot_direction = np.array(sph2cart(1., _azi, _zen))

            for k in range(sources_num):
                new_to_src = source_positions[k] - agent_position
                ray_angle = np.arccos(get_cos(new_to_src, plot_direction))

                if ray_angle < half_angle:
                    gt_mat[i, j] = 1

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

    plt.savefig('_zz.pdf')

    ###
    if do_separation:
        sources_num = source_positions.shape[0]
        agent_to_src = source_positions - agent_position
        agent_to_src = np.tile(agent_to_src[None, :, None, :], (1, 1, frames_num, 1))

        i = 0
        max_agents_contain_waveform = 2

        input_dict = {
            'mic_position': batch_data_dict['mic_position'][i : i + 1, :, :, :].to(device),
            'mic_look_direction': batch_data_dict['mic_look_direction'][i : i + 1, :, :, :].to(device),
            'mic_waveform': batch_data_dict['mic_waveform'][i : i + 1, :, :].to(device),
            'agent_position': batch_data_dict['agent_position'][i : i + 1, 0 : 1, :, :].repeat(1, sources_num, 1, 1).to(device),
            'agent_look_direction': torch.Tensor(agent_to_src).to(device),
            'max_agents_contain_waveform': sources_num,
        }
        output_dict = forward_in_batch(model, input_dict, do_separation=True)

        for j in range(sources_num):
            soundfile.write(file='_zz{}.wav'.format(j), data=output_dict['agent_waveform'][j], samplerate=24000)

        soundfile.write(file='_zz.wav', data=batch_data_dict['mic_waveform'][i, 0].data.cpu().numpy(), samplerate=24000)
        batch_data_dict['mic_waveform']
        

    from IPython import embed; embed(using=False); os._exit(0)


def get_all_agent_look_directions(grid_deg):
    delta = 2 * np.pi * (grid_deg / 360)
    
    tmp = []
    for i in np.arange(0, 2 * np.pi, delta):
        for j in np.arange(0, np.pi, delta):
            tmp.append([i, j])

    tmp = np.array(tmp)
    agent_look_azimuths = tmp[:, 0]
    agent_look_colatitudes = tmp[:, 1]
    return agent_look_azimuths, agent_look_colatitudes


def forward_in_batch(model, input_dict, do_separation):

    N = input_dict['agent_position'].shape[1]
    batch_size = 200
    pointer = 0

    output_dicts = []

    while pointer < N:
        
        batch_input_dict = {}

        for key in input_dict.keys():
            if key in ['agent_position', 'agent_look_direction', 'agent_look_depth']:
                batch_input_dict[key] = input_dict[key][:, pointer : pointer + batch_size, :]
            else:
                batch_input_dict[key] = input_dict[key]

        with torch.no_grad():
            model.eval()
            batch_output_dict = model(data_dict=batch_input_dict, do_separation=do_separation)

        output_dicts.append(batch_output_dict)
        pointer += batch_size

    output_dict = {}
    
    for key in output_dicts[0].keys():
        output_dict[key] = np.concatenate([batch_output_dict[key].data.cpu().numpy() for batch_output_dict in output_dicts], axis=1)[0]

    return output_dict


def forward_depth_in_batch(model, input_dict, do_separation):

    N = input_dict['agent_position'].shape[1]
    batch_size = 200
    pointer = 0

    output_dicts = []

    while pointer < N:
        
        batch_input_dict = {}

        for key in input_dict.keys():
            if key in ['agent_position', 'agent_look_direction', 'agent_look_depth']:
                batch_input_dict[key] = input_dict[key][:, pointer : pointer + batch_size, :]
            else:
                batch_input_dict[key] = input_dict[key]

        with torch.no_grad():
            model.eval()
            batch_output_dict = model(data_dict=batch_input_dict, do_separation=do_separation, mode='eval')

        output_dicts.append(batch_output_dict)
        pointer += batch_size

    output_dict = {}
    
    for key in output_dicts[0].keys():
        output_dict[key] = np.concatenate([batch_output_dict[key].data.cpu().numpy() for batch_output_dict in output_dicts], axis=1)[0]

    return output_dict


def inference_timelapse(args):

    workspace = args.workspace
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    gpus = args.gpus
    filename = args.filename

    configs = read_yaml(config_yaml) 
    sampler_type = configs['sampler_type']
    dataset_type = configs['dataset_type']
    model_type = configs['train']['model_type']
    do_localization = configs['train']['do_localization']
    do_sed = configs['train']['do_sed']
    do_separation = configs['train']['do_separation']
    evaluate_step_frequency = configs['train']['evaluate_step_frequency']
    save_step_frequency = configs['train']['save_step_frequency']
    batch_size = configs['train']['batch_size']
    steps_per_epoch = configs['train']['steps_per_epoch']

    split = 'test' 

    if split == 'train':
        hdf5s_dir = os.path.join(workspace, configs['sources']['train_hdf5s_dir'])
        random_seed = 1234

    elif split == 'test':
        hdf5s_dir = os.path.join(workspace, configs['sources']['test_hdf5s_dir'])
        random_seed = 2345

    num_workers = 0
    distributed = True if gpus > 1 else False
    device = 'cuda'

    frames_num = 301
    classes_num = -1

    Model = eval(model_type)

    model = Model(
        microphones_num=4, 
        classes_num=classes_num, 
        do_localization=do_localization,
        do_sed=do_sed,
        do_separation=do_separation,
    )

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    print(
        "Load pretrained checkpoint from {}".format(checkpoint_path)
    )
    model.to(device)

    _Sampler = eval(sampler_type)
    _Dataset = eval(dataset_type)

    # sampler
    train_sampler = _Sampler(
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        random_seed=random_seed,
    )

    train_dataset = _Dataset(
        hdf5s_dir=hdf5s_dir,
    )

    # data module
    data_module = DataModule(
        train_sampler=train_sampler,
        train_dataset=train_dataset,
        num_workers=num_workers,
        distributed=distributed,
    )

    data_module.setup()

    cnt = 0
    losses = []

    grid_deg = 2
    azimuth_grids = 360 // grid_deg
    elevation_grids = 180 // grid_deg

    agent_look_azimuths, agent_look_colatitudes = get_all_agent_look_directions(grid_deg)

    agent_look_directions = np.stack(sph2cart(
        r=1., azimuth=agent_look_azimuths, colatitude=agent_look_colatitudes), axis=-1)

    agent_look_directions = np.tile(agent_look_directions[None, :, None, :], (1, 1, frames_num, 1))

    agents_num = agent_look_directions.shape[1]

    for batch_data_dict in data_module.train_dataloader():
        
        i = 3
        max_agents_contain_waveform = 2

        input_dict = {
            'mic_position': batch_data_dict['mic_position'][i : i + 1, :, :, :].to(device),
            'mic_look_direction': batch_data_dict['mic_look_direction'][i : i + 1, :, :, :].to(device),
            'mic_waveform': batch_data_dict['mic_waveform'][i : i + 1, :, :].to(device),
            'agent_position': batch_data_dict['agent_position'][i : i + 1, 0 : 1, :, :].repeat(1, agents_num, 1, 1).to(device),
            'agent_look_direction': torch.Tensor(agent_look_directions).to(device),
            'max_agents_contain_waveform': max_agents_contain_waveform,
        }

        source_positions = batch_data_dict['source_position'][i]  # (2, 301, 3)
        agent_position = batch_data_dict['agent_position'][i][0, :, :].data.cpu().numpy()
        sources_num = source_positions.shape[0]
        break

    output_dict = forward_in_batch(model, input_dict, do_separation=False)

    pred_mat_timelapse = output_dict['agent_see_source'][:, 0 : -1 : 10].T.reshape(30, azimuth_grids, elevation_grids)
    
    pred_mat = np.max(pred_mat_timelapse, axis=0)

    gt_mat_timelapse = np.zeros((30, azimuth_grids, elevation_grids))
    half_angle = math.atan2(0.1, 1)

    for i in range(azimuth_grids):
        for j in range(elevation_grids):
            _azi = np.deg2rad(i * grid_deg)
            _zen = np.deg2rad(j * grid_deg)
            plot_direction = np.array(sph2cart(1., _azi, _zen))

            for k in range(sources_num):
                for t in range(0, 300, 10):
                    new_to_src = source_positions[k, t] - agent_position[t]
                    ray_angle = np.arccos(get_cos(new_to_src, plot_direction))

                    if ray_angle < half_angle:
                        gt_mat_timelapse[t // 10, i, j] = 1

    gt_mat = np.max(gt_mat_timelapse, axis=0)

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

    plt.savefig('_zz.pdf')

    for t in range(gt_mat_timelapse.shape[0]):
        plt.figure(figsize=(20, 10))
        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].matshow(gt_mat_timelapse[t].T, origin='upper', aspect='equal', cmap='jet', vmin=0, vmax=1)
        axs[1].matshow(pred_mat_timelapse[t].T, origin='upper', aspect='equal', cmap='jet', vmin=0, vmax=1)
        for i in range(2):
            axs[i].grid(color='w', linestyle='--', linewidth=0.1)
            axs[i].xaxis.set_ticks(np.arange(0, azimuth_grids+1, 10))
            axs[i].yaxis.set_ticks(np.arange(0, elevation_grids+1, 10))
            axs[i].xaxis.set_ticklabels(np.arange(0, 361, 10 * grid_deg), rotation=90)
            axs[i].yaxis.set_ticklabels(np.arange(0, 181, 10 * grid_deg))

        os.makedirs('_tmp', exist_ok=True)
        plt.savefig('_tmp/_zz_{:03d}.jpg'.format(t))

    ###
    if do_separation:
        sources_num = source_positions.shape[0]
        agent_to_src = source_positions - agent_position
        agent_to_src = np.tile(agent_to_src[None, :, None, :], (1, 1, frames_num, 1))

        i = 0
        max_agents_contain_waveform = 2

        input_dict = {
            'mic_position': batch_data_dict['mic_position'][i : i + 1, :, :, :].to(device),
            'mic_look_direction': batch_data_dict['mic_look_direction'][i : i + 1, :, :, :].to(device),
            'mic_waveform': batch_data_dict['mic_waveform'][i : i + 1, :, :].to(device),
            'agent_position': batch_data_dict['agent_position'][i : i + 1, 0 : 1, :, :].repeat(1, sources_num, 1, 1).to(device),
            'agent_look_direction': torch.Tensor(agent_to_src).to(device),
            'max_agents_contain_waveform': sources_num,
        }
        output_dict = forward_in_batch(model, input_dict, do_separation=True)

        for i in range(sources_num):
            soundfile.write(file='_zz{}.wav'.format(i), data=output_dict['agent_waveform'][i], samplerate=24000)

    from IPython import embed; embed(using=False); os._exit(0)


def inference_dcase2021_multi_maps(args):

    workspace = args.workspace
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    gpus = args.gpus
    filename = args.filename

    configs = read_yaml(config_yaml) 
    sampler_type = configs['sampler_type']
    dataset_type = configs['dataset_type']
    classes_num = configs['sources']['classes_num']
    sample_rate = configs['train']['sample_rate']
    model_type = configs['train']['model_type']
    do_localization = configs['train']['do_localization']
    do_sed = configs['train']['do_sed']
    do_separation = configs['train']['do_separation']
    evaluate_step_frequency = configs['train']['evaluate_step_frequency']
    save_step_frequency = configs['train']['save_step_frequency']
    batch_size = configs['train']['batch_size']
    steps_per_epoch = configs['train']['steps_per_epoch']

    audio_path = "/home/tiger/datasets/dcase2021/task3/mic_dev/dev-test/fold6_room1_mix001.wav"
    csv_path = "/home/tiger/datasets/dcase2021/task3/metadata_dev/dev-test/fold6_room1_mix001.csv"

    # audio_path = "/home/tiger/datasets/dcase2021/task3/mic_dev/dev-test/fold6_room2_mix050.wav"
    # csv_path = "/home/tiger/datasets/dcase2021/task3/metadata_dev/dev-test/fold6_room2_mix050.csv"

    # audio_path = "/home/tiger/datasets/dcase2021/task3/mic_dev/dev-train/fold1_room1_mix001.wav"
    # csv_path = "/home/tiger/datasets/dcase2021/task3/metadata_dev/dev-train/fold1_room1_mix001.csv"

    df = pd.read_csv(csv_path, sep=',', header=None)
    frame_indexes = df[0].values
    class_ids = df[1].values
    azimuths = df[3].values % 360
    elevations = 90 - df[4].values

    grid_deg = 4
    azimuth_grids = 360 // grid_deg
    elevation_grids = 180 // grid_deg
    classwise = True
    
    gt_mat_timelapse = np.zeros((600, azimuth_grids, elevation_grids))
    gt_mat_classwise_timelapse = np.zeros((600, azimuth_grids, elevation_grids, classes_num))

    for n in range(len(frame_indexes)):
        i = int(azimuths[n] / grid_deg)
        j = int(elevations[n] / grid_deg)
        class_id = class_ids[n]
        r = 2
        gt_mat_timelapse[frame_indexes[n], max(i - r, 0) : min(i + r, azimuth_grids), max(j - r, 0) : min(j + r, elevation_grids)] = 1

        gt_mat_classwise_timelapse[frame_indexes[n], max(i - r, 0) : min(i + r, azimuth_grids), max(j - r, 0) : min(j + r, elevation_grids), class_id] = 1

    device = 'cuda'
    frames_num = 301
    # classes_num = -1
    segment_samples = int(3 * sample_rate)

    if True:
        hdf5s_dir = os.path.join(workspace, configs['sources']['train_hdf5s_dir'])
        random_seed = 1234

        num_workers = 0
        distributed = True if gpus > 1 else False

        _Sampler = eval(sampler_type)
        _Dataset = eval(dataset_type)

        # sampler
        train_sampler = _Sampler(
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            random_seed=random_seed,
        )

        train_dataset = _Dataset(
            hdf5s_dir=hdf5s_dir,
        )

        # data module
        data_module = DataModule(
            train_sampler=train_sampler,
            train_dataset=train_dataset,
            num_workers=num_workers,
            distributed=distributed,
        )

        data_module.setup()

    Model = eval(model_type)

    model = Model(
        microphones_num=4, 
        classes_num=classes_num, 
        do_localization=do_localization,
        do_sed=do_sed,
        do_separation=do_separation,
    )

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    print(
        "Load pretrained checkpoint from {}".format(checkpoint_path)
    )
    model.to(device)

    cnt = 0
    losses = []

    agent_look_azimuths, agent_look_colatitudes = get_all_agent_look_directions(grid_deg)

    agent_look_directions = np.stack(sph2cart(
        r=1., azimuth=agent_look_azimuths, colatitude=agent_look_colatitudes), axis=-1)

    agent_look_directions = np.tile(agent_look_directions[None, :, None, :], (1, 1, frames_num, 1))

    agents_num = agent_look_directions.shape[1]

    pointer = 0

    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=False)
    audio_samples = audio.shape[-1]

    for batch_data_dict in data_module.train_dataloader():
        
        i = 0

        input_dict = {
            'mic_position': batch_data_dict['mic_position'][i : i + 1, :, :, :].to(device),
            'mic_look_direction': batch_data_dict['mic_look_direction'][i : i + 1, :, :, :].to(device),
            # 'mic_waveform': batch_data_dict['mic_waveform'][i : i + 1, :, :].to(device),
            'agent_position': batch_data_dict['agent_position'][i : i + 1, 0 : 1, :, :].repeat(1, agents_num, 1, 1).to(device),
            'agent_look_direction': torch.Tensor(agent_look_directions).to(device),
        }

        source_positions = batch_data_dict['source_position'][i]  # (2, 301, 3)
        agent_position = batch_data_dict['agent_position'][i][0, :, :].data.cpu().numpy()
        sources_num = source_positions.shape[0]
        break
    
    global_t = 0
    while pointer + segment_samples < audio_samples:
        print(global_t)

        segment = audio[:, pointer : pointer + segment_samples]

        input_dict['mic_waveform'] = torch.Tensor(segment[None, :, :]).to(device)
        input_dict['max_agents_contain_waveform'] = input_dict['agent_position'].shape[1]

        output_dict = forward_in_batch(model, input_dict, do_separation=False)

        pred_mat_timelapse = output_dict['agent_see_source'][:, 0 : -1 : 10].T.reshape(30, azimuth_grids, elevation_grids)

        if classwise:
            pred_mat_classwise_timelapse = output_dict['agent_see_source_classwise'][:, 0 : -1 : 10, :].transpose(1, 0, 2).reshape(30, azimuth_grids, elevation_grids, classes_num)

            pred_mat_classwise_timelapse = pred_mat_timelapse[:, :, :, None] * pred_mat_classwise_timelapse

        for t in range(pred_mat_timelapse.shape[0]):
            print(t)
            plt.figure(figsize=(20, 10))
            fig, axs = plt.subplots(2, 1, sharex=True)
            axs[0].matshow(gt_mat_timelapse[global_t].T, origin='upper', aspect='equal', cmap='jet', vmin=0, vmax=1)
            axs[1].matshow(pred_mat_timelapse[t].T, origin='upper', aspect='equal', cmap='jet', vmin=0, vmax=1)
            for i in range(2):
                axs[i].grid(color='w', linestyle='--', linewidth=0.1)
                axs[i].xaxis.set_ticks(np.arange(0, azimuth_grids+1, 10))
                axs[i].yaxis.set_ticks(np.arange(0, elevation_grids+1, 10))
                axs[i].xaxis.set_ticklabels(np.arange(0, 361, 10 * grid_deg), rotation=90)
                axs[i].yaxis.set_ticklabels(np.arange(0, 181, 10 * grid_deg))

            os.makedirs('_tmp', exist_ok=True)
            plt.savefig('_tmp/_zz_{:03d}.jpg'.format(global_t))
            
            if classwise:
                plt.figure(figsize=(20, 20))
                fig, axs = plt.subplots(6, 4, sharex=True)

                for k in range(classes_num):
                    axs[k // 4, k % 4].matshow(gt_mat_classwise_timelapse[global_t, :, :, k].T, origin='upper', aspect='equal', cmap='jet', vmin=0, vmax=1)

                    axs[3 + k // 4, k % 4].matshow(pred_mat_classwise_timelapse[t, :, :, k].T, origin='upper', aspect='equal', cmap='jet', vmin=0, vmax=1)

                os.makedirs('_tmp2', exist_ok=True)
                plt.savefig('_tmp2/_zz_{:03d}.jpg'.format(global_t))

            global_t += 1

            # from IPython import embed; embed(using=False); os._exit(0)

        pointer += segment_samples



def inference_dcase2021_single_map(args):

    workspace = args.workspace
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    gpus = args.gpus
    filename = args.filename

    configs = read_yaml(config_yaml) 
    sampler_type = configs['sampler_type']
    dataset_type = configs['dataset_type']
    classes_num = configs['sources']['classes_num']
    sample_rate = configs['train']['sample_rate']
    model_type = configs['train']['model_type']
    do_localization = configs['train']['do_localization']
    do_sed = configs['train']['do_sed']
    do_separation = configs['train']['do_separation']
    evaluate_step_frequency = configs['train']['evaluate_step_frequency']
    save_step_frequency = configs['train']['save_step_frequency']
    batch_size = configs['train']['batch_size']
    steps_per_epoch = configs['train']['steps_per_epoch']

    audio_path = "/home/tiger/datasets/dcase2021/task3/mic_dev/dev-test/fold6_room1_mix001.wav"
    csv_path = "/home/tiger/datasets/dcase2021/task3/metadata_dev/dev-test/fold6_room1_mix001.csv"

    # audio_path = "/home/tiger/datasets/dcase2021/task3/mic_dev/dev-test/fold6_room2_mix050.wav"
    # csv_path = "/home/tiger/datasets/dcase2021/task3/metadata_dev/dev-test/fold6_room2_mix050.csv"

    # audio_path = "/home/tiger/datasets/dcase2021/task3/mic_dev/dev-train/fold1_room1_mix001.wav"
    # csv_path = "/home/tiger/datasets/dcase2021/task3/metadata_dev/dev-train/fold1_room1_mix001.csv"

    df = pd.read_csv(csv_path, sep=',', header=None) 
    frame_indexes = df[0].values
    class_ids = df[1].values
    azimuths = df[3].values % 360
    elevations = 90 - df[4].values

    grid_deg = 2
    azimuth_grids = 360 // grid_deg
    elevation_grids = 180 // grid_deg
    classwise = True
    
    gt_mat_timelapse = np.zeros((600, azimuth_grids, elevation_grids))
    strs = [''] * 600

    from nesd.dataset_creation.pack_audios_to_hdf5s.dcase2021_task3 import ID_TO_LB
    
    for n in range(len(frame_indexes)):
        i = int(azimuths[n] / grid_deg)
        j = int(elevations[n] / grid_deg)
        class_id = class_ids[n]
        r = 2
        gt_mat_timelapse[frame_indexes[n], max(i - r, 0) : min(i + r, azimuth_grids), max(j - r, 0) : min(j + r, elevation_grids)] = 1

        strs[frame_indexes[n]] += '({}, {}): {} '.format(azimuths[n], elevations[n], ID_TO_LB[class_id])
    
    device = 'cuda'
    frames_num = 301
    # classes_num = -1
    segment_samples = int(3 * sample_rate)

    if True:
        hdf5s_dir = os.path.join(workspace, configs['sources']['train_hdf5s_dir'])
        random_seed = 1234

        num_workers = 0
        distributed = True if gpus > 1 else False

        _Sampler = eval(sampler_type)
        _Dataset = eval(dataset_type)

        # sampler
        train_sampler = _Sampler(
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            random_seed=random_seed,
        )

        train_dataset = _Dataset(
            hdf5s_dir=hdf5s_dir,
        )

        # data module
        data_module = DataModule(
            train_sampler=train_sampler,
            train_dataset=train_dataset,
            num_workers=num_workers,
            distributed=distributed,
        )

        data_module.setup()

    Model = eval(model_type)

    model = Model(
        microphones_num=4, 
        classes_num=classes_num, 
        do_localization=do_localization,
        do_sed=do_sed,
        do_separation=do_separation,
    )

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    print(
        "Load pretrained checkpoint from {}".format(checkpoint_path)
    )
    model.to(device)

    cnt = 0
    losses = []

    agent_look_azimuths, agent_look_colatitudes = get_all_agent_look_directions(grid_deg)

    agent_look_directions = np.stack(sph2cart(
        r=1., azimuth=agent_look_azimuths, colatitude=agent_look_colatitudes), axis=-1)

    agent_look_directions = np.tile(agent_look_directions[None, :, None, :], (1, 1, frames_num, 1))

    agents_num = agent_look_directions.shape[1]

    pointer = 0

    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=False)
    audio_samples = audio.shape[-1]

    for batch_data_dict in data_module.train_dataloader():
        
        i = 0

        input_dict = {
            'mic_position': batch_data_dict['mic_position'][i : i + 1, :, :, :].to(device),
            'mic_look_direction': batch_data_dict['mic_look_direction'][i : i + 1, :, :, :].to(device),
            # 'mic_waveform': batch_data_dict['mic_waveform'][i : i + 1, :, :].to(device),
            'agent_position': batch_data_dict['agent_position'][i : i + 1, 0 : 1, :, :].repeat(1, agents_num, 1, 1).to(device),
            'agent_look_direction': torch.Tensor(agent_look_directions).to(device),
        }

        source_positions = batch_data_dict['source_position'][i]  # (2, 301, 3)
        agent_position = batch_data_dict['agent_position'][i][0, :, :].data.cpu().numpy()
        sources_num = source_positions.shape[0]
        break
    
    global_t = 0
    while pointer + segment_samples < audio_samples:
        print(global_t)

        segment = audio[:, pointer : pointer + segment_samples]

        input_dict['mic_waveform'] = torch.Tensor(segment[None, :, :]).to(device)
        input_dict['max_agents_contain_waveform'] = input_dict['agent_position'].shape[1]

        output_dict = forward_in_batch(model, input_dict, do_separation=False)

        pred_mat_timelapse = output_dict['agent_see_source'][:, 0 : -1 : 10].T.reshape(30, azimuth_grids, elevation_grids)

        if classwise:
            pred_mat_classwise_timelapse = output_dict['agent_see_source_classwise'][:, 0 : -1 : 10, :].transpose(1, 0, 2).reshape(30, azimuth_grids, elevation_grids, classes_num)

            pred_mat_classwise_timelapse = pred_mat_timelapse[:, :, :, None] * pred_mat_classwise_timelapse

        for t in range(pred_mat_timelapse.shape[0]):
            
            tmp = pred_mat_timelapse[t, :, :]
            max_coord = np.array(np.unravel_index(np.argmax(tmp), tmp.shape))
            max_prob = np.max(tmp)

            if classwise:
                tmp = pred_mat_classwise_timelapse[t, max_coord[0], max_coord[1], :]
                pred_class_id = np.argmax(tmp)
                pred_prob = np.max(tmp)
                
                if max_prob > 0.5:
                    pred_str = '({},{}):{},{:.3f}'.format(max_coord[0] * grid_deg, max_coord[1] * grid_deg, ID_TO_LB[pred_class_id], pred_prob)
                else:
                    pred_str = ''
            else:
                pred_str = ''

            plt.figure(figsize=(20, 10))
            fig, axs = plt.subplots(2, 1, sharex=True)
            axs[0].matshow(gt_mat_timelapse[global_t].T, origin='upper', aspect='equal', cmap='jet', vmin=0, vmax=1)
            axs[1].matshow(pred_mat_timelapse[t].T, origin='upper', aspect='equal', cmap='jet', vmin=0, vmax=1)
            axs[0].set_title(strs[global_t])
            axs[1].set_title(pred_str)
            for i in range(2):
                axs[i].grid(color='w', linestyle='--', linewidth=0.1)
                axs[i].xaxis.set_ticks(np.arange(0, azimuth_grids+1, 10))
                axs[i].yaxis.set_ticks(np.arange(0, elevation_grids+1, 10))
                axs[i].xaxis.set_ticklabels(np.arange(0, 361, 10 * grid_deg), rotation=90)
                axs[i].yaxis.set_ticklabels(np.arange(0, 181, 10 * grid_deg))
                axs[i].xaxis.tick_bottom()

            # plt.tight_layout(pad=0, h_pad=0, w_pad=0)
            os.makedirs('_tmp', exist_ok=True)
            plt.savefig('_tmp/_zz_{:03d}.jpg'.format(global_t))

            global_t += 1

        pointer += segment_samples


def inference_depth(args):

    workspace = args.workspace
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    gpus = args.gpus
    filename = args.filename

    configs = read_yaml(config_yaml) 
    sampler_type = configs['sampler_type']
    dataset_type = configs['dataset_type']
    model_type = configs['train']['model_type']
    do_localization = configs['train']['do_localization']
    do_sed = configs['train']['do_sed']
    do_separation = configs['train']['do_separation']
    evaluate_step_frequency = configs['train']['evaluate_step_frequency']
    save_step_frequency = configs['train']['save_step_frequency']
    batch_size = configs['train']['batch_size']
    steps_per_epoch = configs['train']['steps_per_epoch']

    split = 'test' 

    if split == 'train':
        hdf5s_dir = os.path.join(workspace, configs['sources']['train_hdf5s_dir'])
        random_seed = 1234

    elif split == 'test':
        hdf5s_dir = os.path.join(workspace, configs['sources']['test_hdf5s_dir'])
        random_seed = 2345

    num_workers = 8
    distributed = True if gpus > 1 else False
    device = 'cuda'

    frames_num = 301
    classes_num = -1

    Model = eval(model_type)

    model = Model(
        microphones_num=4, 
        classes_num=classes_num, 
        do_localization=do_localization,
        do_sed=do_sed,
        do_separation=do_separation,
    )

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    print(
        "Load pretrained checkpoint from {}".format(checkpoint_path)
    )
    model.to(device)

    _Sampler = eval(sampler_type)
    _Dataset = eval(dataset_type)

    # sampler
    train_sampler = _Sampler(
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        random_seed=random_seed,
    )

    train_dataset = _Dataset(
        hdf5s_dir=hdf5s_dir,
    )

    # data module
    data_module = DataModule(
        train_sampler=train_sampler,
        train_dataset=train_dataset,
        num_workers=num_workers,
        distributed=distributed,
    )

    data_module.setup()

    cnt = 0
    losses = []

    grid_deg = 2
    azimuth_grids = 360 // grid_deg
    elevation_grids = 180 // grid_deg

    agent_look_azimuths, agent_look_colatitudes = get_all_agent_look_directions(grid_deg)

    agent_look_directions = np.stack(sph2cart(
        r=1., azimuth=agent_look_azimuths, colatitude=agent_look_colatitudes), axis=-1)

    agent_look_directions = np.tile(agent_look_directions[None, :, None, :], (1, 1, frames_num, 1))

    agents_num = agent_look_directions.shape[1]

    for batch_data_dict in data_module.train_dataloader():
        
        i = 0
        max_agents_contain_waveform = 2

        input_dict = {
            'mic_position': batch_data_dict['mic_position'][i : i + 1, :, :, :].to(device),
            'mic_look_direction': batch_data_dict['mic_look_direction'][i : i + 1, :, :, :].to(device),
            'mic_waveform': batch_data_dict['mic_waveform'][i : i + 1, :, :].to(device),
            'agent_position': batch_data_dict['agent_position'][i : i + 1, 0 : 1, :, :].repeat(1, agents_num, 1, 1).to(device),
            'agent_look_direction': torch.Tensor(agent_look_directions).to(device),
            'max_agents_contain_waveform': max_agents_contain_waveform,
        }

        if 'agent_look_depth' in batch_data_dict.keys():
            input_dict['agent_look_depth'] = batch_data_dict['agent_look_depth'][i : i + 1, 0 : 1, :].repeat(1, agents_num, 1).to(device)
            input_dict['max_agents_contain_depth'] = batch_data_dict['agent_look_depth'].shape[1]

        t = -1
        source_positions = batch_data_dict['source_position'][i][:, t, :]
        agent_position = batch_data_dict['agent_position'][i][0, t, :].data.cpu().numpy()
        sources_num = source_positions.shape[0]
        break

    output_dict = forward_depth_in_batch(model, input_dict, do_separation=False)

    pred_mat = np.mean(output_dict['agent_see_source'], axis=1).reshape(azimuth_grids, elevation_grids)
    # pred_mat = output_dict['agent_see_source'][:, t].reshape(azimuth_grids, elevation_grids) 

    gt_mat = np.zeros((azimuth_grids, elevation_grids))
    half_angle = math.atan2(0.1, 1)

    for i in range(gt_mat.shape[0]):
        for j in range(gt_mat.shape[1]):
            _azi = np.deg2rad(i * grid_deg)
            _zen = np.deg2rad(j * grid_deg)
            plot_direction = np.array(sph2cart(1., _azi, _zen))

            for k in range(sources_num):
                new_to_src = source_positions[k] - agent_position
                ray_angle = np.arccos(get_cos(new_to_src, plot_direction))

                if ray_angle < half_angle:
                    gt_mat[i, j] = 1

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

    plt.savefig('_zz.pdf')

    if True:
        sources_num = source_positions.shape[0]
        agent_to_src = source_positions - agent_position
        agent_to_src = np.tile(agent_to_src[None, :, None, :], (1, 1, frames_num, 1))

        depths_num = 60
        agent_look_direction = torch.Tensor(agent_to_src[:, 0:1, :, :]).repeat(1, depths_num, 1, 1).to(device)
        a1 = norm(agent_to_src[0, 0, 0])
        gt_depth = np.zeros(depths_num)
        gt_depth[int(np.around(a1 / 0.1))] = 1

        i = 0
        depth_num = 100
        agent_look_depth = torch.Tensor(np.arange(0, 6, 0.1)).to(device)[None, :, None].repeat(1, 1, frames_num)

        input_dict = {
            'mic_position': batch_data_dict['mic_position'][i : i + 1, :, :, :].to(device),
            'mic_look_direction': batch_data_dict['mic_look_direction'][i : i + 1, :, :, :].to(device),
            'mic_waveform': batch_data_dict['mic_waveform'][i : i + 1, :, :].to(device),
            'agent_position': batch_data_dict['agent_position'][i : i + 1, 0 : 1, :, :].repeat(1, depths_num, 1, 1).to(device),
            'agent_look_direction': agent_look_direction,
            'agent_look_depth': agent_look_depth,
            'max_agents_contain_waveform': None,
            'max_agents_contain_depth': None,
        }
        
        output_dict = forward_depth_in_batch(model, input_dict, do_separation=False)

        tmp = output_dict['agent_exist_source'][:, 0]
        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].plot(tmp)
        axs[1].plot(gt_depth)

        plt.savefig('_zz2.pdf')

        from IPython import embed; embed(using=False); os._exit(0)

    ###
    '''
    if do_separation:
        sources_num = source_positions.shape[0]
        agent_to_src = source_positions - agent_position
        agent_to_src = np.tile(agent_to_src[None, :, None, :], (1, 1, frames_num, 1))

        i = 0
        max_agents_contain_waveform = 2

        input_dict = {
            'mic_position': batch_data_dict['mic_position'][i : i + 1, :, :, :].to(device),
            'mic_look_direction': batch_data_dict['mic_look_direction'][i : i + 1, :, :, :].to(device),
            'mic_waveform': batch_data_dict['mic_waveform'][i : i + 1, :, :].to(device),
            'agent_position': batch_data_dict['agent_position'][i : i + 1, 0 : 1, :, :].repeat(1, sources_num, 1, 1).to(device),
            'agent_look_direction': torch.Tensor(agent_to_src).to(device),
            'max_agents_contain_waveform': sources_num,
        }

        output_dict = forward_in_batch(model, input_dict, do_separation=True)

        for j in range(sources_num):
            soundfile.write(file='_zz{}.wav'.format(j), data=output_dict['agent_waveform'][j], samplerate=24000)

        soundfile.write(file='_zz.wav', data=batch_data_dict['mic_waveform'][i, 0].data.cpu().numpy(), samplerate=24000)
        batch_data_dict['mic_waveform']
        

    from IPython import embed; embed(using=False); os._exit(0)
    '''

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest="mode")

    #
    parser_train = subparsers.add_parser("inference")
    parser_train.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
    parser_train.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path of config file for training.",
    )
    parser_train.add_argument(
        "--checkpoint_path", type=str,
    )
    parser_train.add_argument("--gpus", type=int, required=True)

    #
    parser_inference_timelapse = subparsers.add_parser("inference_timelapse")
    parser_inference_timelapse.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
    parser_inference_timelapse.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path of config file for training.",
    )
    parser_inference_timelapse.add_argument(
        "--checkpoint_path", type=str,
    )
    parser_inference_timelapse.add_argument("--gpus", type=int, required=True)

    #
    parser_inference_dcase2021_multi_maps = subparsers.add_parser("inference_dcase2021_multi_maps")
    parser_inference_dcase2021_multi_maps.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
    parser_inference_dcase2021_multi_maps.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path of config file for training.",
    )
    parser_inference_dcase2021_multi_maps.add_argument(
        "--checkpoint_path", type=str,
    )
    parser_inference_dcase2021_multi_maps.add_argument("--gpus", type=int, required=True)

    #
    parser_inference_dcase2021_multi_maps = subparsers.add_parser("inference_dcase2021_single_map")
    parser_inference_dcase2021_multi_maps.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
    parser_inference_dcase2021_multi_maps.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path of config file for training.",
    )
    parser_inference_dcase2021_multi_maps.add_argument(
        "--checkpoint_path", type=str,
    )
    parser_inference_dcase2021_multi_maps.add_argument("--gpus", type=int, required=True)

    #
    parser_train = subparsers.add_parser("inference_depth")
    parser_train.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
    parser_train.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path of config file for training.",
    )
    parser_train.add_argument(
        "--checkpoint_path", type=str,
    )
    parser_train.add_argument("--gpus", type=int, required=True)

    #
    args = parser.parse_args()
    args.filename = pathlib.Path(__file__).stem 

    if args.mode == "inference":
        inference(args)

    elif args.mode == "inference_timelapse":
        inference_timelapse(args)

    elif args.mode == "inference_dcase2021_multi_maps":
        inference_dcase2021_multi_maps(args)

    elif args.mode == "inference_dcase2021_single_map":
        inference_dcase2021_single_map(args)

    if args.mode == "inference_depth":
        inference_depth(args)

    else:
        raise Exception("Error argument!")