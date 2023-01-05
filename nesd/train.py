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

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
import torch 
import torch.nn.functional as F

from nesd.utils import read_yaml, create_logging
from nesd.data.samplers import Sampler
from nesd.data.samplers import *
from nesd.data.data_modules import DataModule, Dataset
from nesd.data.data_modules import *
from nesd.models.models01 import *
from nesd.models.lightning_modules import LitModel
from nesd.optimizers.lr_schedulers import get_lr_lambda
from nesd.losses import *
from nesd.callbacks.callback import get_callback


def get_dirs(
    workspace: str,
    filename: str,
    config_yaml: str,
    gpus: int,
) -> List[str]:
    r"""Get directory paths.

    Args:
        workspace: str
        task_name, str, e.g., 'musdb18'
        filenmae: str
        config_yaml: str
        gpus: int, e.g., 0 for cpu and 8 for training with 8 gpu cards

    Returns:
        checkpoints_dir: str
        logs_dir: str
        logger: pl.loggers.TensorBoardLogger
        statistics_path: str
    """

    # save checkpoints dir
    checkpoints_dir = os.path.join(
        workspace,
        "checkpoints",
        filename,
        "config={},gpus={}".format(pathlib.Path(config_yaml).stem, gpus),
    )
    os.makedirs(checkpoints_dir, exist_ok=True)

    # logs dir
    logs_dir = os.path.join(
        workspace,
        "logs",
        filename,
        "config={},gpus={}".format(pathlib.Path(config_yaml).stem, gpus),
    )
    os.makedirs(logs_dir, exist_ok=True)

    # loggings
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    # tensorboard logs dir
    tb_logs_dir = os.path.join(workspace, "tensorboard_logs")
    os.makedirs(tb_logs_dir, exist_ok=True)

    experiment_name = os.path.join(filename, pathlib.Path(config_yaml).stem)
    logger = pl.loggers.TensorBoardLogger(save_dir=tb_logs_dir, name=experiment_name)

    # statistics path
    statistics_path = os.path.join(
        workspace,
        "statistics",
        filename,
        "config={},gpus={}".format(pathlib.Path(config_yaml).stem, gpus),
        "statistics.pkl",
    )
    os.makedirs(os.path.dirname(statistics_path), exist_ok=True)

    return checkpoints_dir, logs_dir, logger, statistics_path


def get_data_module(
    workspace: str,
    config_yaml: str,
    num_workers: int,
    distributed: bool,
):# -> DataModule:
    r"""Create data_module. Here is an example to fetch a mini-batch:

    code-block:: python

        data_module.setup()
        for batch_data_dict in data_module.train_dataloader():
            print(batch_data_dict.keys())
            break

    Args:
        workspace: str
        config_yaml: str
        num_workers: int, e.g., 0 for non-parallel and 8 for using cpu cores
            for preparing data in parallel
        distributed: bool

    Returns:
        data_module: DataModule
    """

    configs = read_yaml(config_yaml)

    sampler_type = configs['sampler_type']
    dataset_type = configs['dataset_type']
    train_hdf5s_dir = os.path.join(workspace, configs['sources']['train_hdf5s_dir'])
    test_hdf5s_dir = os.path.join(workspace, configs['sources']['test_hdf5s_dir'])
    batch_size = configs['train']['batch_size']
    steps_per_epoch = configs['train']['steps_per_epoch']

    _Sampler = eval(sampler_type)
    _Dataset = eval(dataset_type)
    
    # sampler
    train_sampler = _Sampler(
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        random_seed=1234,
    )

    train_dataset = _Dataset(
        hdf5s_dir=train_hdf5s_dir,
    )

    # data module
    data_module = DataModule(
        train_sampler=train_sampler,
        train_dataset=train_dataset,
        num_workers=num_workers,
        distributed=distributed,
    )

    return data_module

'''
def loc_bce(model, output_dict, target_dict):
    loss = F.binary_cross_entropy(output_dict['ray_intersect_source'], target_dict['ray_intersect_source'])
    return loss

def loc_bce_sep_l1(model, output_dict, target_dict):
    loc_loss = F.binary_cross_entropy(output_dict['ray_intersect_source'], target_dict['ray_intersect_source'])
    sep_loss = torch.mean(torch.abs(output_dict['ray_waveform'] - target_dict['ray_waveform'][:, 0 : 2, :]))
    sep_loss *= 10.

    total_loss = loc_loss + sep_loss
    print(loc_loss.item(), sep_loss.item(), torch.max(output_dict['ray_waveform']).item()) 

    return total_loss


def sep_l1(model, output_dict, target_dict):
    # from IPython import embed; embed(using=False); os._exit(0)
    # loc_loss = F.binary_cross_entropy(output_dict['ray_intersect_source'], target_dict['ray_intersect_source'])
    sep_loss = torch.mean(torch.abs(output_dict['ray_waveform'] - target_dict['ray_waveform'][:, 0 : 2, :]))
    sep_loss *= 10.

    total_loss = sep_loss
    print(sep_loss.item(), torch.max(output_dict['ray_waveform']).item()) 

    return total_loss


def loc_bce_cla_bce_sep_l1(model, output_dict, target_dict):
    
    eng_mat = torch.mean(torch.abs(target_dict['waveform']), dim=-1)
    wav_loss_mat = torch.mean(torch.abs(output_dict['waveform'] - target_dict['waveform']), dim=-1)

    weight_mat = eng_mat / torch.max(eng_mat)
    weight_mat = torch.clamp(weight_mat, 0.01, 1.)

    wav_loss = torch.sum(wav_loss_mat * weight_mat / torch.sum(weight_mat))
    wav_loss *= 10

    loc_loss = F.binary_cross_entropy(output_dict['has_energy_array'], target_dict['has_energy_array'])
    cla_loss = F.binary_cross_entropy(output_dict['class_id_mat'], target_dict['class_id_mat'])
    
    total_loss = loc_loss + wav_loss + cla_loss
    
    return total_loss
'''

def train(args) -> NoReturn:
    r"""Train & evaluate and save checkpoints.

    Args:
        workspace: str, directory of workspace
        gpus: int
        config_yaml: str, path of config file for training
    """

    # arugments & parameters
    workspace = args.workspace
    gpus = args.gpus
    config_yaml = args.config_yaml
    filename = args.filename

    num_workers = 8
    distributed = True if gpus > 1 else False
    evaluate_device = "cuda" if gpus > 0 else "cpu"

    configs = read_yaml(config_yaml)
    model_type = configs['train']['model_type']
    do_localization = configs['train']['do_localization']
    do_sed = configs['train']['do_sed']
    do_separation = configs['train']['do_separation']
    loss_type = configs['train']['loss_type']
    optimizer_type = configs['train']['optimizer_type']
    learning_rate = float(configs['train']['learning_rate'])
    warm_up_steps = int(configs['train']['warm_up_steps'])
    reduce_lr_steps = int(configs['train']['reduce_lr_steps'])
    early_stop_steps = int(configs['train']['early_stop_steps'])
    precision = int(configs['train']['precision'])

    # paths
    checkpoints_dir, logs_dir, logger, statistics_path = get_dirs(
        workspace, filename, config_yaml, gpus,
    )

    # training data module
    data_module = get_data_module(
        workspace=workspace,
        config_yaml=config_yaml,
        num_workers=num_workers,
        distributed=distributed,
    )

    # model
    classes_num = -1
    Model = eval(model_type)

    model = Model(
        microphones_num=4, 
        classes_num=classes_num, 
        do_localization=do_localization,
        do_sed=do_sed,
        do_separation=do_separation,
    )

    loss_function = eval(loss_type)
    
    # callbacks
    callbacks = get_callback(
        config_yaml=config_yaml,
        workspace=workspace,
        checkpoints_dir=checkpoints_dir,
        statistics_path=statistics_path,
        logger=logger,
        model=model,
        loss_function=loss_function,
        evaluate_device=evaluate_device,
    )

    # learning rate reduce function
    lr_lambda = partial(
        get_lr_lambda, warm_up_steps=warm_up_steps, reduce_lr_steps=reduce_lr_steps
    )

    # pytorch-lightning model
    pl_model = LitModel(
        model=model,
        optimizer_type=optimizer_type,
        loss_function=loss_function,
        learning_rate=learning_rate,
        lr_lambda=lr_lambda,
        # max_sep_rays=max_sep_rays,
        # target_configs=target_configs,
    )

    # trainer
    trainer = pl.Trainer(
        checkpoint_callback=False,
        gpus=gpus,
        callbacks=callbacks,
        max_steps=early_stop_steps,
        accelerator="ddp",
        sync_batchnorm=True,
        precision=precision,
        replace_sampler_ddp=False,
        plugins=[DDPPlugin(find_unused_parameters=True)],
        profiler='simple',
    )

    # Fit, evaluate, and save checkpoints.
    trainer.fit(pl_model, data_module)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest="mode")

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
    parser_train.add_argument("--gpus", type=int, required=True)
    parser_train.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path of config file for training.",
    )
    
    args = parser.parse_args()
    args.filename = pathlib.Path(__file__).stem 

    if args.mode == "train":
        train(args)

    else:
        raise Exception("Error argument!")
