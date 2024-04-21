import argparse
import os
import pathlib
from typing import Dict, List, NoReturn
import torch
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import math
import soundfile

from nesd.utils import read_yaml, sph2cart, expand_along_frame_axis, get_included_angle
from nesd.data.dataset import Dataset
from nesd.data.collate import collate_fn
from nesd.data.samplers import InfiniteRandomSampler
from nesd.train import get_model


def inference(args):

    workspace = args.workspace
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    filename = pathlib.Path(__file__).stem
    
    configs = read_yaml(config_yaml)

    simulator_configs = configs["simulator_configs"]
    mics_meta = read_yaml(simulator_configs["mics_yaml"])
    mics_num = len(mics_meta["microphone_coordinates"])

    device = configs["train"]["device"]
    batch_size_per_device = configs["train"]["batch_size_per_device"]
    num_workers = configs["train"]["num_workers"]
    model_name = configs["train"]["model_name"]
    
    batch_size = batch_size_per_device
    split = "test"
    # split = "train"

    # Dataset
    train_dataset = Dataset(
        simulator_configs=simulator_configs,
        split=split
    )

    sampler = InfiniteRandomSampler()

    # Load checkpoint
    model = get_model(model_name, mics_num)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    # Dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers, 
        pin_memory=True
    )

    for data in dataloader:

        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)

        with torch.no_grad():
            model.eval()
            output_dict = model(data)

        gt_dir = data["agent_look_at_direction_has_source"].cpu().numpy()
        gt_dist = data["agent_look_at_distance_has_source"].cpu().numpy()
        gt_wav = data["agent_look_at_direction_reverb_wav"].cpu().numpy()
        
        pred_dir = output_dict["agent_look_at_direction_has_source"].cpu().numpy()
        pred_dist = output_dict["agent_look_at_distance_has_source"].cpu().numpy()
        # pred_wav = output_dict["agent_look_at_direction_reverb_wav"].cpu().numpy()

        n = 0
        fig, axs = plt.subplots(2, 1, sharex=True) 
        axs[0].matshow(gt_dir[n], origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
        axs[1].matshow(pred_dir[n], origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
        plt.savefig("_zz.pdf")

        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].matshow(gt_dist[n], origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
        axs[1].matshow(pred_dist[n], origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
        plt.savefig("_zz2.pdf") 

        from IPython import embed; embed(using=False); os._exit(0)


def inference_panaroma(args):

    workspace = args.workspace
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    filename = pathlib.Path(__file__).stem
    devices_num = torch.cuda.device_count()
    
    configs = read_yaml(config_yaml)

    simulator_configs = configs["simulator_configs"]
    mics_meta = read_yaml(simulator_configs["mics_yaml"])
    mics_num = len(mics_meta["mic_coordinates"])

    device = configs["train"]["device"]
    batch_size_per_device = configs["train"]["batch_size_per_device"]
    num_workers = configs["train"]["num_workers"]
    model_name = configs["train"]["model_name"]
    
    batch_size = batch_size_per_device
    split = "test"
    grid_deg = 2
    azi_grids = 360 // grid_deg
    ele_grids = 180 // grid_deg

    # Dataset
    dataset = Dataset(
        simulator_configs=simulator_configs,
        split=split 
    )
    
    sampler = InfiniteRandomSampler()

    # Load checkpoint
    model = get_model(model_name, mics_num)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    # Dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers, 
        pin_memory=True
    )

    for data in dataloader:

        n = 0
        mic_wavs = data["mic_wavs"][n : n + 1]
        mic_poss = data["mic_positions"][n : n + 1]
        mic_oriens = data["mic_orientations"][n : n + 1]

        agent_poss = data["agent_positions"][n : n + 1, 0 : 1]
        # data["agent_look_at_directions"][n : n + 1]
        # agent_look_at_distances = data["agent_look_at_distances"][n : n + 1, 0 : 1]
        # data["agent_distance_masks"][n : n + 1]
        # data["agent_detect_idxes"][n : n + 1]
        # data["agent_look_at_direction_has_source"][n : n + 1]

        frames_num = agent_poss.shape[2]

        agent_look_at_directions = get_all_agent_look_at_directions(grid_deg)
        # (rays_num, ndim)

        agent_look_at_directions = expand_along_frame_axis(
            x=agent_look_at_directions[None, :, :], 
            repeats=frames_num
        )
        # (1, rays_num, frames_num, ndim)

        rays_num = agent_look_at_directions.shape[1]

        agent_poss = np.repeat(agent_poss, repeats=rays_num, axis=1)

        ##
        agent_look_at_distances = data["agent_look_at_distances"][n : n + 1, 0 : 1]
        agent_look_at_distances = np.repeat(
            agent_look_at_distances, 
            repeats=rays_num,
            axis=1
        )

        agent_distance_masks = data["agent_distance_masks"][n : n + 1, 0 : 1]
        agent_distance_masks = np.repeat(
            agent_distance_masks, 
            repeats=rays_num,
            axis=1
        )

        ##
        rays_num = agent_poss.shape[1]
        batch_size = 2000
        pointer = 0

        # output_dicts = []
        output_dict = {}

        while pointer < rays_num:

            print(pointer)

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
            # from IPython import embed; embed(using=False); os._exit(0)

            for key in batch_data.keys():
                batch_data[key] = torch.Tensor(batch_data[key]).to(device)

            with torch.no_grad():
                model.eval()
                batch_output_dict = model(batch_data)

            # output_dicts.append(batch_output_dict)

            if len(output_dict) == 0:
                for key in batch_output_dict.keys():
                    output_dict[key] = []

            for key in batch_output_dict.keys():
                output_dict[key].append(batch_output_dict[key].cpu().numpy())

            pointer += batch_size

        for key in output_dict.keys():
            output_dict[key] = np.concatenate(output_dict[key], axis=1)

        ##
        pred_direction = output_dict["agent_look_at_direction_has_source"]
        # gt_dist = data["agent_look_at_distance_has_source"].cpu().numpy()
        # gt_wav = data["agent_look_at_direction_reverb_wav"].cpu().numpy()
        
        # pred_dir = output_dict["agent_look_at_direction_has_source"].cpu().numpy()
        # pred_dist = output_dict["agent_look_at_distance_has_source"].cpu().numpy()
        # pred_wav = output_dict["agent_look_at_direction_reverb_wav"].cpu().numpy()

        pred_direction = pred_direction[0].reshape(azi_grids, ele_grids, frames_num)
        pred_direction = np.mean(pred_direction, axis=-1)


        n = 0
        static_src_poss = [pos[0, :] for pos in data["source_position"][n]]
        agent_pos = data["agent_positions"][n, 0, 0].cpu().numpy()
        src_directions = [pos - agent_pos for pos in static_src_poss]

        gt_dir = np.zeros((azi_grids, ele_grids))
        for i in range(azi_grids):
            for j in range(ele_grids):
                azi = np.deg2rad(i * grid_deg) - math.pi
                ele = np.deg2rad(j * grid_deg) - math.pi / 2
                agent_look_at_dir = sph2cart(azimuth=azi, elevation=ele, r=1.)
                for src_dir in src_directions:
                    angle = get_included_angle(agent_look_at_dir, src_dir)
                    if np.rad2deg(angle) < 5:
                        gt_dir[i, j] = 1


        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].matshow(gt_dir.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
        axs[1].matshow(pred_direction.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
        for i in range(2):
            axs[i].grid(color='w', linestyle='--', linewidth=0.1)
            axs[i].xaxis.set_ticks(np.arange(0, azi_grids + 1, 10))
            axs[i].yaxis.set_ticks(np.arange(0, ele_grids + 1, 10))
            axs[i].xaxis.set_ticklabels(np.arange(-180, 181, 10 * grid_deg), rotation=90)
            axs[i].yaxis.set_ticklabels(np.arange(-90, 91, 10 * grid_deg))
        plt.savefig("_zz.pdf")

        

        # fig, axs = plt.subplots(2, 1, sharex=True)
        # axs[0].matshow(gt_dist[n], origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
        # axs[1].matshow(pred_dist[n], origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
        # plt.savefig("_zz2.pdf") 

        from IPython import embed; embed(using=False); os._exit(0)


def inference_distance(args):

    workspace = args.workspace
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    filename = pathlib.Path(__file__).stem
    devices_num = torch.cuda.device_count()
    
    configs = read_yaml(config_yaml)

    simulator_configs = configs["simulator_configs"]
    mics_meta = read_yaml(simulator_configs["mics_yaml"])
    mics_num = len(mics_meta["microphone_coordinates"])

    device = configs["train"]["device"]
    batch_size_per_device = configs["train"]["batch_size_per_device"]
    num_workers = configs["train"]["num_workers"]
    model_name = configs["train"]["model_name"]
    
    batch_size = batch_size_per_device
    split = "test"
    grid_deg = 2
    azi_grids = 360 // grid_deg
    ele_grids = 180 // grid_deg

    # Dataset
    train_dataset = Dataset(
        simulator_configs=simulator_configs,
        split=split
    )

    sampler = InfiniteRandomSampler()

    # Load checkpoint
    model = get_model(model_name, mics_num)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    # Dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers, 
        pin_memory=True
    )

    for data in dataloader:

        n = 0
        mic_wavs = data["mic_wavs"][n : n + 1]
        mic_poss = data["mic_positions"][n : n + 1]
        mic_oriens = data["mic_orientations"][n : n + 1]

        agent_poss = data["agent_positions"][n : n + 1, 0 : 1]
        agent_look_at_directions = data["agent_look_at_directions"][n : n + 1, 0 : 1]
        
        frames_num = agent_poss.shape[2]
        
        agent_look_at_distances = get_all_agent_look_at_distances()

        # agent_look_at_directions = get_all_agent_look_at_directions(grid_deg)
        # (rays_num, ndim)

        agent_look_at_distances = expand_along_frame_axis(
            x=agent_look_at_distances[None, :, None], 
            repeats=frames_num
        )[:, :, :, 0]
        # (1, rays_num, frames_num, ndim)

        rays_num = agent_look_at_distances.shape[1]

        agent_poss = np.repeat(agent_poss, repeats=rays_num, axis=1)

        ##
        # agent_look_at_distances = data["agent_look_at_distances"][n : n + 1, 0 : 1]
        agent_look_at_directions = np.repeat(
            agent_look_at_directions, 
            repeats=rays_num,
            axis=1
        )
        
        # agent_distance_masks = data["agent_distance_masks"][n : n + 1, 0 : 1]
        # agent_distance_masks = np.repeat(
        #     agent_distance_masks, 
        #     repeats=rays_num,
        #     axis=1
        # )
        agent_distance_masks = np.ones_like(agent_look_at_distances)
        
        ##
        rays_num = agent_poss.shape[1]
        batch_size = 200
        pointer = 0

        # output_dicts = []
        output_dict = {}

        while pointer < rays_num:

            print(pointer)

            _len = min(batch_size, rays_num - pointer)

            agent_detect_idxes = torch.Tensor(np.arange(0)[None, :])
            agent_distance_idxes = torch.Tensor(np.arange(_len)[None, :])
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

            # output_dicts.append(batch_output_dict)

            if len(output_dict) == 0:
                for key in batch_output_dict.keys():
                    output_dict[key] = []

            for key in batch_output_dict.keys():
                output_dict[key].append(batch_output_dict[key].cpu().numpy())

            pointer += batch_size

        for key in output_dict.keys():
            output_dict[key] = np.concatenate(output_dict[key], axis=1)

        ##
        pred_distance = np.mean(output_dict["agent_look_at_distance_has_source"][0, :, :], axis=-1)

        n = 0
        static_src_pos = data["source_position"][n][0][0]
        agent_pos = data["agent_positions"][n, 0, 0].cpu().numpy()
        agent_to_src = static_src_pos - agent_pos
        gt_dist = np.linalg.norm(agent_to_src)
        gt_distance = np.zeros(2000)
        gt_distance[round(gt_dist * 100)] = 1
        
        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].stem(gt_distance)
        axs[1].plot(pred_distance)
        plt.savefig("_zz.pdf")
        
        from IPython import embed; embed(using=False); os._exit(0)


def inference_sep(args):

    workspace = args.workspace
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    filename = pathlib.Path(__file__).stem
    devices_num = torch.cuda.device_count()
    
    configs = read_yaml(config_yaml)

    simulator_configs = configs["simulator_configs"]
    mics_meta = read_yaml(simulator_configs["mics_yaml"])
    mics_num = len(mics_meta["microphone_coordinates"])

    device = configs["train"]["device"]
    batch_size_per_device = configs["train"]["batch_size_per_device"]
    num_workers = configs["train"]["num_workers"]
    model_name = configs["train"]["model_name"]
    
    batch_size = batch_size_per_device
    split = "test"
    grid_deg = 2
    azi_grids = 360 // grid_deg
    ele_grids = 180 // grid_deg

    # Dataset
    train_dataset = Dataset(
        simulator_configs=simulator_configs,
        split=split
    )

    sampler = InfiniteRandomSampler()

    # Load checkpoint
    model = get_model(model_name, mics_num)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    # Dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers, 
        pin_memory=True
    )

    for data in dataloader:

        n = 0
        sources = data["source"][0]
        mic_wavs = data["mic_wavs"][n : n + 1]
        mic_poss = data["mic_positions"][n : n + 1]
        mic_oriens = data["mic_orientations"][n : n + 1]

        S = 2
        agent_poss = data["agent_positions"][n : n + 1, 0 : S]
        agent_look_at_directions = data["agent_look_at_directions"][n : n + 1, 0 : S]
        agent_look_at_distances = data["agent_look_at_distances"][n : n + 1, 0 : S]
        agent_distance_masks = data["agent_distance_masks"][n : n + 1, 0 : S]
        
        frames_num = agent_poss.shape[2]
        
        rays_num = agent_look_at_distances.shape[1]

        # agent_distance_masks = np.zeros_like(agent_look_at_distances)
        
        ##
        rays_num = agent_poss.shape[1]
        batch_size = 10
        pointer = 0

        # output_dicts = []
        output_dict = {}

        while pointer < rays_num:

            print(pointer)

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
            # from IPython import embed; embed(using=False); os._exit(0)

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

        sep_wavs = output_dict["agent_look_at_direction_reverb_wav"][0]

        if "agent_look_at_direction_direct_wav" in output_dict.keys():
            sep_direct_wavs = output_dict["agent_look_at_direction_direct_wav"][0]

        for i in range(sep_wavs.shape[0]):
            soundfile.write(file="_zz_{}.wav".format(i), data=sep_wavs[i], samplerate=24000)
            soundfile.write(file="_yy_{}.wav".format(i), data=sources[i], samplerate=24000)

            if "agent_look_at_direction_direct_wav" in output_dict.keys():
                soundfile.write(file="_zz_direct_{}.wav".format(i), data=sep_direct_wavs[i], samplerate=24000)

        from IPython import embed; embed(using=False); os._exit(0)
        soundfile.write(file="_uu2.wav", data=mic_wavs[0].data.cpu().numpy().T, samplerate=24000)


def get_all_agent_look_at_directions(grid_deg):
    
    grid_rad = np.deg2rad(grid_deg)
    azis = []
    eles = []

    for azi in np.arange(-math.pi, math.pi, grid_rad):
        for ele in np.arange(-math.pi / 2, math.pi / 2, grid_rad):
            azis.append(azi)
            eles.append(ele)
    
    azis = np.stack(azis, axis=0)
    eles = np.stack(eles, axis=0)

    agent_look_at_directions = sph2cart(azimuth=azis, elevation=eles, r=1.)

    # agent_look_at_directions = np.zeros((16200, 3))

    # agent_look_at_directions[0 : 2] = np.array([
    #     [1,2,3],
    #     [-1,2,3]
    # ])

    # from IPython import embed; embed(using=False); os._exit(0)

    return agent_look_at_directions


def get_all_agent_look_at_distances():
    agent_look_at_distances = np.arange(0, 20, 0.01)
    return agent_look_at_distances


def forward_in_batch(model, data):

    rays_num = input_dict['agent_positions'].shape[1]
    batch_size = 10000
    pointer = 0

    output_dicts = []

    while pointer < rays_num:
        print(pointer)
        
        data = {
            "mic_wavs": mic_wavs,
            "mic_positions": mic_poss,
            "mic_orientations": mic_oriens,
            "agent_positions": agent_poss,
            "agent_look_directions": agent_look_at_directions
        }

        # for key in input_dict.keys():
        #     if key in ['agent_positions', 'agent_look_directions', 'agent_look_depths']:
        #         batch_input_dict[key] = input_dict[key][:, pointer : pointer + batch_size, :]
        #     else:
        #         batch_input_dict[key] = input_dict[key]

        with torch.no_grad():
            model.eval()
            batch_output_dict = model(batch_data)

        output_dicts.append(batch_output_dict)
        pointer += batch_size

    output_dict = {}
    
    for key in output_dicts[0].keys():
        output_dict[key] = np.concatenate([batch_output_dict[key].data.cpu().numpy() for batch_output_dict in output_dicts], axis=1)[0]

    # from IPython import embed; embed(using=False); os._exit(0)
    return output_dict


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
    
    parser_inference_pana = subparsers.add_parser("inference_panaroma")
    parser_inference_pana.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
    parser_inference_pana.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path of config file for training.",
    )
    parser_inference_pana.add_argument(
        "--checkpoint_path", type=str, required=True, help="Directory of workspace."
    )

    parser_inference_dist = subparsers.add_parser("inference_distance")
    parser_inference_dist.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
    parser_inference_dist.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path of config file for training.",
    )
    parser_inference_dist.add_argument(
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

    args = parser.parse_args()
    args.filename = pathlib.Path(__file__).stem 

    if args.mode == "inference":
        inference(args)

    elif args.mode == "inference_panaroma":
        inference_panaroma(args)

    elif args.mode == "inference_distance":
        inference_distance(args)

    elif args.mode == "inference_sep":
        inference_sep(args)

    else:
        raise Exception("Error argument!")