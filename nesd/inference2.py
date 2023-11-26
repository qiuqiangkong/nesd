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

import pytorch_lightning as pl
# from pytorch_lightning.plugins import DDPPlugin
import torch 
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint

from nesd.utils import read_yaml, create_logging
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


def inference(args):

    workspace = args.workspace
    checkpoint_path = args.checkpoint_path
    config_yaml = args.config_yaml
    filename = args.filename
    device = "cuda"

    configs = read_yaml(config_yaml)
    # device = configs["train"]["device"]
    # devices_num = configs["train"]["devices_num"]

    # num_workers = configs["train"]["num_workers"]
    model_type = configs['train']['model_type']
    simulator_configs = configs["simulator_configs"]

    num_workers = 0
    batch_size = 32
    frames_num = 201

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
        checkpoint_path=checkpoint_path
    )

    # Data
    test_audios_dir = "/home/qiuqiangkong/workspaces/nesd2/audios/vctk_2s_segments/test"
    dataset = Dataset3(audios_dir=test_audios_dir, expand_frames=201, simulator_configs=simulator_configs)
    
    batch_sampler = BatchSampler(batch_size=batch_size, iterations_per_epoch=5)
    batch_sampler = DistributedSamplerWrapper(batch_sampler)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    grid_deg = 2
    azimuth_grids = 360 // grid_deg
    elevation_grids = 180 // grid_deg

    agent_look_azimuths, agent_look_colatitudes = get_all_agent_look_directions(grid_deg)

    agent_look_directions = np.stack(sph2cart(
        r=1., azimuth=agent_look_azimuths, colatitude=agent_look_colatitudes), axis=-1)

    rays_num = agent_look_directions.shape[0]

    for _, data_dict in enumerate(dataloader):

        i = 0

        agent_positions = np.repeat(
            a=data_dict["agent_positions"][i : i + 1, 0 : 1, ...], 
            repeats=rays_num, 
            axis=1
        )

        agent_look_directions = np.repeat(
            a=agent_look_directions[None, :, None, :],
            repeats=frames_num,
            axis=-2
        )

        input_dict = {
            "mic_positions": data_dict["mic_positions"][i : i + 1, ...],
            "mic_look_directions": data_dict["mic_look_directions"][i : i + 1, ...],
            "mic_signals": data_dict["mic_signals"][i : i + 1, ...],
            "agent_positions": agent_positions,
            "agent_look_directions": agent_look_directions,
        }

        input_dict["agent_look_depths"] = PAD * np.ones(agent_look_directions.shape[0:-1] + (1,))

        for key in input_dict.keys():
            input_dict[key] = torch.Tensor(input_dict[key]).to(device)

        output_dict = forward_in_batch(model=model, input_dict=input_dict)

        agent_look_directions_has_source = np.mean(output_dict["agent_look_directions_has_source"], axis=1)

        source_positions = [e[0] for e in data_dict["source_positions"][i]] # (sources_num, 3)
        agent_position = data_dict["agent_positions"][i][0, 0]  # (3,)

        pickle.dump([agent_look_directions_has_source, source_positions, agent_position.data.cpu().numpy()], open("_zz.pkl", "wb"))

        plot(None)
        from IPython import embed; embed(using=False); os._exit(0)


def forward_in_batch(model, input_dict, mode="inference"):

    rays_num = input_dict['agent_positions'].shape[1]
    batch_size = 1000
    pointer = 0

    output_dicts = []

    while pointer < rays_num:
        # print(pointer)
        
        batch_input_dict = {}

        for key in input_dict.keys():
            if key in ['agent_positions', 'agent_look_directions', 'agent_look_depths']:
                batch_input_dict[key] = input_dict[key][:, pointer : pointer + batch_size, :]
            else:
                batch_input_dict[key] = input_dict[key]

        with torch.no_grad():
            model.eval()
            batch_output_dict = model.net(data_dict=batch_input_dict, mode=mode)

        output_dicts.append(batch_output_dict)
        pointer += batch_size

    output_dict = {}
    
    for key in output_dicts[0].keys():
        output_dict[key] = np.concatenate([batch_output_dict[key].data.cpu().numpy() for batch_output_dict in output_dicts], axis=1)[0]

    # from IPython import embed; embed(using=False); os._exit(0)
    return output_dict


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


def plot(args):

    agent_look_directions_has_source, source_positions, agent_position = pickle.load(open("_zz.pkl", "rb"))

    grid_deg = 2
    azimuth_grids = 360 // grid_deg
    elevation_grids = 180 // grid_deg
    sources_num = len(source_positions)

    pred_mat = agent_look_directions_has_source.reshape(azimuth_grids, elevation_grids)

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

    from IPython import embed; embed(using=False); os._exit(0)


def inference_depth(args):

    workspace = args.workspace
    checkpoint_path = args.checkpoint_path
    config_yaml = args.config_yaml
    filename = args.filename
    device = "cuda"

    configs = read_yaml(config_yaml)
    # device = configs["train"]["device"]
    # devices_num = configs["train"]["devices_num"]

    # num_workers = configs["train"]["num_workers"]
    model_type = configs['train']['model_type']
    simulator_configs = configs["simulator_configs"]

    num_workers = 0
    batch_size = 32
    frames_num = 201

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
        checkpoint_path=checkpoint_path
    )

    # Data
    test_audios_dir = "/home/qiuqiangkong/workspaces/nesd2/audios/vctk_2s_segments/test"
    dataset = Dataset3(audios_dir=test_audios_dir, expand_frames=201, simulator_configs=simulator_configs)
    
    batch_sampler = BatchSampler(batch_size=batch_size, iterations_per_epoch=5)
    batch_sampler = DistributedSamplerWrapper(batch_sampler)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    grid_deg = 2
    azimuth_grids = 360 // grid_deg
    elevation_grids = 180 // grid_deg

    agent_look_depths = np.arange(0, 10, 0.01)
    agent_look_depths = agent_look_depths[None, :, None, None]
    agent_look_depths = np.repeat(a=agent_look_depths, repeats=frames_num, axis=2)
    rays_num = agent_look_depths.shape[1]

    # agent_look_azimuths, agent_look_colatitudes = get_all_agent_look_directions(grid_deg)

    # agent_look_directions = np.stack(sph2cart(
    #     r=1., azimuth=agent_look_azimuths, colatitude=agent_look_colatitudes), axis=-1)

    # rays_num = agent_look_directions.shape[0]

    for _, data_dict in enumerate(dataloader):

        i = 0

        agent_positions = np.repeat(
            a=data_dict["agent_positions"][i : i + 1, 0 : 1, ...], 
            repeats=rays_num, 
            axis=1
        )

        tmp = data_dict["agent_look_directions"][i : i + 1, 0 : 1, ...]
        agent_look_directions = np.repeat(a=tmp, repeats=rays_num, axis=1)

        input_dict = {
            "mic_positions": data_dict["mic_positions"][i : i + 1, ...],
            "mic_look_directions": data_dict["mic_look_directions"][i : i + 1, ...],
            "mic_signals": data_dict["mic_signals"][i : i + 1, ...],
            "agent_positions": agent_positions,
            "agent_look_directions": agent_look_directions,
            "agent_look_depths": agent_look_depths
        }

        for key in input_dict.keys():
            input_dict[key] = torch.Tensor(input_dict[key]).to(device)

        output_dict = forward_in_batch(model=model, input_dict=input_dict)

        tmp = output_dict["agent_look_depths_has_source"]

        fig, axs = plt.subplots(1, 1, sharex=True)
        axs.matshow(tmp, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
        plt.savefig("_zz.pdf")

        from IPython import embed; embed(using=False); os._exit(0)


def inference_depth2(args):

    workspace = args.workspace
    checkpoint_path = args.checkpoint_path
    config_yaml = args.config_yaml
    filename = args.filename
    device = "cuda"

    configs = read_yaml(config_yaml)
    # device = configs["train"]["device"]
    # devices_num = configs["train"]["devices_num"]

    # num_workers = configs["train"]["num_workers"]
    model_type = configs['train']['model_type']
    simulator_configs = configs["simulator_configs"]

    num_workers = 0
    batch_size = 32
    frames_num = 201

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
        checkpoint_path=checkpoint_path
    )

    # Data
    test_audios_dir = "/home/qiuqiangkong/workspaces/nesd2/audios/vctk_2s_segments/test"
    dataset = Dataset3(audios_dir=test_audios_dir, expand_frames=201, simulator_configs=simulator_configs)
    
    batch_sampler = BatchSampler(batch_size=batch_size, iterations_per_epoch=5)
    batch_sampler = DistributedSamplerWrapper(batch_sampler)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    grid_deg = 2
    azimuth_grids = 360 // grid_deg
    elevation_grids = 180 // grid_deg

    for _, data_dict in enumerate(dataloader):

        i = 0

        input_dict = {
            "mic_positions": data_dict["mic_positions"][i : i + 1, ...],
            "mic_look_directions": data_dict["mic_look_directions"][i : i + 1, ...],
            "mic_signals": data_dict["mic_signals"][i : i + 1, ...],
            "agent_positions": data_dict["agent_positions"][i : i + 1, ...],
            "agent_look_directions": data_dict["agent_look_directions"][i : i + 1, ...],
            "agent_look_depths": data_dict["agent_look_depths"][i : i + 1, ...]
        }

        for key in input_dict.keys():
            input_dict[key] = torch.Tensor(input_dict[key]).to(device)

        output_dict = forward_in_batch(model=model, input_dict=input_dict)

        tmp = output_dict["agent_look_depths_has_source"]

        fig, axs = plt.subplots(1, 1, sharex=True)
        axs.matshow(tmp, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
        plt.savefig("_zz.pdf")

        from IPython import embed; embed(using=False); os._exit(0)


def inference_sep(args):

    workspace = args.workspace
    checkpoint_path = args.checkpoint_path
    config_yaml = args.config_yaml
    filename = args.filename
    device = "cuda"

    configs = read_yaml(config_yaml)
    # device = configs["train"]["device"]
    # devices_num = configs["train"]["devices_num"]

    # num_workers = configs["train"]["num_workers"]
    model_type = configs['train']['model_type']
    simulator_configs = configs["simulator_configs"]

    num_workers = 0
    batch_size = 32
    frames_num = 201
    sample_rate = 24000

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
        checkpoint_path=checkpoint_path
    )

    # Data
    test_audios_dir = "/home/qiuqiangkong/workspaces/nesd2/audios/vctk_2s_segments/test"
    dataset = Dataset3(audios_dir=test_audios_dir, expand_frames=201, simulator_configs=simulator_configs)
    
    batch_sampler = BatchSampler(batch_size=batch_size, iterations_per_epoch=5)
    batch_sampler = DistributedSamplerWrapper(batch_sampler)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    grid_deg = 2
    azimuth_grids = 360 // grid_deg
    elevation_grids = 180 // grid_deg

    agent_look_depths = np.arange(0, 10, 0.01)
    agent_look_depths = agent_look_depths[None, :, None, None]
    agent_look_depths = np.repeat(a=agent_look_depths, repeats=frames_num, axis=2)
    rays_num = agent_look_depths.shape[1]

    # agent_look_azimuths, agent_look_colatitudes = get_all_agent_look_directions(grid_deg)

    # agent_look_directions = np.stack(sph2cart(
    #     r=1., azimuth=agent_look_azimuths, colatitude=agent_look_colatitudes), axis=-1)

    # rays_num = agent_look_directions.shape[0]

    for _, data_dict in enumerate(dataloader):

        i = 0

        # agent_positions = np.repeat(
        #     a=data_dict["agent_positions"][i : i + 1, 0 : 1, ...], 
        #     repeats=rays_num, 
        #     axis=1
        # )

        sources_num = 2
        agent_positions = data_dict["agent_positions"][i : i + 1, 0 : sources_num, ...]
        agent_look_directions = data_dict["agent_look_directions"][i : i + 1, 0 : sources_num, ...]

        input_dict = {
            "mic_positions": data_dict["mic_positions"][i : i + 1, ...],
            "mic_look_directions": data_dict["mic_look_directions"][i : i + 1, ...],
            "mic_signals": data_dict["mic_signals"][i : i + 1, ...],
            "agent_positions": agent_positions,
            "agent_look_directions": agent_look_directions,

        }

        input_dict["agent_look_depths"] = PAD * np.ones(agent_look_directions.shape[0:-1] + (1,))

        for key in input_dict.keys():
            input_dict[key] = torch.Tensor(input_dict[key]).to(device)

        output_dict = forward_in_batch(model=model, input_dict=input_dict)

        out_wavs = output_dict["agent_signals"]

        for i in range(len(out_wavs)):
            soundfile.write(file="_zz_{:02d}.wav".format(i), data=out_wavs[i], samplerate=sample_rate)

        from IPython import embed; embed(using=False); os._exit(0)

        # tmp = output_dict["agent_look_depths_has_source"]

        # fig, axs = plt.subplots(1, 1, sharex=True)
        # axs.matshow(tmp, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
        # plt.savefig("_zz.pdf")

        from IPython import embed; embed(using=False); os._exit(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest="mode")

    parser_inference = subparsers.add_parser("inference")
    parser_inference.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
    parser_inference.add_argument(
        "--checkpoint_path", type=str, required=True, help="Directory of workspace."
    )
    parser_inference.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path of config file for training.",
    )

    parser_inference_depth = subparsers.add_parser("inference_depth")
    parser_inference_depth.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
    parser_inference_depth.add_argument(
        "--checkpoint_path", type=str, required=True, help="Directory of workspace."
    )
    parser_inference_depth.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path of config file for training.",
    )

    parser_inference_sep = subparsers.add_parser("inference_sep")
    parser_inference_sep.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
    parser_inference_sep.add_argument(
        "--checkpoint_path", type=str, required=True, help="Directory of workspace."
    )
    parser_inference_sep.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path of config file for training.",
    )

    parser_plot = subparsers.add_parser("plot")
    
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

    else:
        raise Exception("Error argument!")