import argparse
import pathlib
import os
import math
import numpy as np
import torch
import soundfile
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from nesd.data.samplers import Sampler
from nesd.data.samplers import *
from nesd.data.data_modules import DataModule, Dataset
from nesd.data.data_modules import *
from nesd.data.data_modules_sed import *
from nesd.utils import read_yaml, create_logging, sph2cart, get_cos, norm, sph2cart
from nesd.models.sed_models01 import *
from nesd.evaluate_dcase2021_task3 import detect_max
from nesd.models.sed_fz_loc_models01 import *


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



def read_dcase2019_task3_csv(csv_path):

    df = pd.read_csv(csv_path, sep=',')
    labels = df['sound_event_recording'].values
    onsets = df['start_time'].values
    offsets = df['end_time'].values
    azimuths = df['azi'].values
    elevations = df['ele'].values

    events_num = len(labels)
    frames_per_sec = 10

    frame_indexes = []
    class_indexes = []
    event_indexes = []
    _azimuths = []
    _elevations = []

    from nesd.dataset_creation.pack_audios_to_hdf5s.dcase2019_task3 import LB_TO_ID

    for n in range(events_num):

        onset_frame = int(onsets[n] * frames_per_sec)
        offset_frame = int(offsets[n] * frames_per_sec)

        for frame_index in np.arange(onset_frame, offset_frame + 1):
            frame_indexes.append(frame_index)
            class_indexes.append(LB_TO_ID[labels[n]])
            event_indexes.append(n)
            _azimuths.append(azimuths[n])
            _elevations.append(elevations[n])

    azimuths = np.array(_azimuths) % 360
    elevations = 90 - np.array(_elevations)

    return frame_indexes, class_indexes, azimuths, elevations


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

    gt_strs = pickle.load(open('_gt_strs.pkl', 'rb'))
    gt_mat_timelapse = pickle.load(open('_all_gt_mat_timelapse.pkl', 'rb'))
    pred_mat_timelapse = pickle.load(open('_all_pred_mat_timelapse.pkl', 'rb'))
    # pred_mat_classwise_timelapse = pickle.load(open('_all_pred_mat_classwise_timelapse.pkl', 'rb'))

    task = "dcase2019_task3"  
    split = 'test'
    device = 'cuda'
    segment_seconds = 3
    segment_samples = segment_seconds * sample_rate
    frames_num = 301

    if task == "dcase2019_task3":
        if split == 'train':
            audio_path = "/home/tiger/datasets/dcase2019/task3/mic_dev/split1_ir0_ov1_1.wav"
            csv_path = "/home/tiger/datasets/dcase2019/task3/metadata_dev/split1_ir0_ov1_1.csv"

        elif split == 'test':
            audio_path = "/home/tiger/datasets/dcase2019/task3/mic_eval/split0_100.wav"
            csv_path = "/home/tiger/datasets/dcase2019/task3/metadata_eval/split0_100.csv"

        frame_indexes, class_ids, azimuths, elevations = read_dcase2019_task3_csv(csv_path)
        from nesd.dataset_creation.pack_audios_to_hdf5s.dcase2019_task3 import ID_TO_LB

    elif task == "dcase2021_task3":
        if split == 'train':
            audio_path = "/home/tiger/datasets/dcase2021/task3/mic_dev/dev-train/fold1_room1_mix001.wav"
            csv_path = "/home/tiger/datasets/dcase2021/task3/metadata_dev/dev-train/fold1_room1_mix001.csv"

        elif split == 'test':
            audio_path = "/home/tiger/datasets/dcase2021/task3/mic_dev/dev-test/fold6_room1_mix001.wav"
            csv_path = "/home/tiger/datasets/dcase2021/task3/metadata_dev/dev-test/fold6_room1_mix001.csv"

        # audio_path = "/home/tiger/datasets/dcase2021/task3/mic_dev/dev-test/fold6_room2_mix050.wav"
        # csv_path = "/home/tiger/datasets/dcase2021/task3/metadata_dev/dev-test/fold6_room2_mix050.csv"

        frame_indexes, class_ids, azimuths, elevations = read_dcase2021_task3_csv(csv_path)
        from nesd.dataset_creation.pack_audios_to_hdf5s.dcase2021_task3 import ID_TO_LB

    elif task == "dcase2022_task3":
        if split == 'train':
            audio_path = "/home/tiger/datasets/dcase2022/task3/mic_dev/dev-train-sony/fold3_room21_mix001.wav"
            csv_path = "/home/tiger/datasets/dcase2022/task3/metadata_dev/dev-train-sony/fold3_room21_mix001.csv"

        elif split == 'test':
            audio_path = "/home/tiger/datasets/dcase2022/task3/mic_dev/dev-test-sony/fold4_room23_mix001.wav"
            csv_path = "/home/tiger/datasets/dcase2022/task3/metadata_dev/dev-test-sony/fold4_room23_mix001.csv"

        frame_indexes, class_ids, azimuths, elevations = read_dcase2022_task3_csv(csv_path)
        from nesd.dataset_creation.pack_audios_to_hdf5s.dcase2022_task3 import ID_TO_LB

    grid_deg = 2
    azimuth_grids = 360 // grid_deg
    elevation_grids = 180 // grid_deg
    classwise = True

    Model = eval(model_type)

    model = Model(
        microphones_num=4, 
        classes_num=classes_num,
    )

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    print(
        "Load pretrained checkpoint from {}".format(checkpoint_path)
    )
    model.to(device)
    
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
            classes_num=classes_num,
        )

        # data module
        data_module = DataModule(
            train_sampler=train_sampler,
            train_dataset=train_dataset,
            num_workers=num_workers,
            distributed=distributed,
        )

        data_module.setup()

    for batch_data_dict in data_module.train_dataloader():
        
        i = 0
        input_dict = {
            'mic_position': batch_data_dict['mic_position'][i : i + 1, :, :, :].to(device),
            'mic_look_direction': batch_data_dict['mic_look_direction'][i : i + 1, :, :, :].to(device),
            # 'agent_position': batch_data_dict['agent_position'][i : i + 1, 0 : 1, :, :].repeat(1, agents_num, 1, 1).to(device),
        }

        source_positions = batch_data_dict['source_position'][i]  # (2, 301, 3)
        agent_position = batch_data_dict['agent_position'][i][0, :, :].data.cpu().numpy()
        sources_num = source_positions.shape[0]
        break

    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=False)
    audio_samples = audio.shape[-1]

    pointer = 0
    t = 0

    while pointer < audio_samples:

        segment = audio[:, pointer : pointer + segment_samples]
        segment = librosa.util.fix_length(data=segment, size=segment_samples, axis=-1)

        input_dict['mic_waveform'] = torch.Tensor(segment[None, :, :]).to(device)

        tmp = np.max(gt_mat_timelapse[t * 10 : (t + segment_seconds) * 10], axis=0)

        pairs = detect_max(tmp)

        # plt.matshow(tmp.T, origin='lower', aspect='auto', cmap='jet')
        # plt.savefig('_zz.pdf')

        for azimuth, colatitude in pairs:
            azimuth = np.deg2rad(azimuth * grid_deg)
            colatitude = np.deg2rad(colatitude * grid_deg)
            agent_look_direction = np.array(sph2cart(r=1., azimuth=azimuth, colatitude=colatitude))
            agent_look_direction = np.tile(agent_look_direction[None, None, None, :], (1, 1, frames_num, 1))
            input_dict['agent_look_direction'] = torch.Tensor(agent_look_direction).to(device)

            with torch.no_grad():
                model.eval()
                output_dict = model(data_dict=input_dict)

            classwise_output = output_dict['classwise_output'][0, 0, :, :].data.cpu().numpy()

            plt.matshow(classwise_output.T, origin='lower', aspect='auto', cmap='jet')
            plt.savefig('_zz.pdf')
            # gt_mat_timelapse[t * 10 : (t + segment_seconds) * 10, :, :]
            # from IPython import embed; embed(using=False); os._exit(0)

        pointer += segment_samples
        t += segment_seconds

        from IPython import embed; embed(using=False); os._exit(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest="mode")

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
    args = parser.parse_args()
    args.filename = pathlib.Path(__file__).stem 

    if args.mode == "inference_dcase2021_single_map":
        inference_dcase2021_single_map(args)

    else:
        raise Exception("Error argument!")