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
from nesd.models.lightning_modules import LitModel
from nesd.optimizers.lr_schedulers import get_lr_lambda
from nesd.losses import *
# from nesd.callbacks.callback import get_callback

from nesd.test_dataloader import collate_fn
from nesd.freefield_simulator import DatasetFreefield


def get_dirs(
    workspace: str,
    filename: str,
    config_yaml: str,
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
        "config={}".format(pathlib.Path(config_yaml).stem),
    )
    os.makedirs(checkpoints_dir, exist_ok=True)

    # logs dir
    logs_dir = os.path.join(
        workspace,
        "logs",
        filename,
        "config={}".format(pathlib.Path(config_yaml).stem),
    )
    os.makedirs(logs_dir, exist_ok=True)

    # loggings
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    # tensorboard logs dir
    tb_logs_dir = os.path.join(workspace, "tensorboard_logs")
    os.makedirs(tb_logs_dir, exist_ok=True)

    experiment_name = os.path.join(filename, pathlib.Path(config_yaml).stem)
    # logger = pl.loggers.TensorBoardLogger(save_dir=tb_logs_dir, name=experiment_name)

    # statistics path
    statistics_path = os.path.join(
        workspace,
        "statistics",
        filename,
        "config={}".format(pathlib.Path(config_yaml).stem),
        "statistics.pkl",
    )
    os.makedirs(os.path.dirname(statistics_path), exist_ok=True)

    return checkpoints_dir, logs_dir, statistics_path


def get_data_module(
    workspace: str,
    config_yaml: str,
    devices_num: int,
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
    simulator_configs = configs["simulator_configs"]
    # train_hdf5s_dir = os.path.join(workspace, configs['sources']['train_hdf5s_dir'])
    # test_hdf5s_dir = os.path.join(workspace, configs['sources']['test_hdf5s_dir'])
    # classes_num = configs['sources']['classes_num']
    batch_size_per_device = configs['train']['batch_size_per_device']
    steps_per_epoch = configs['train']['steps_per_epoch']
    batch_size = batch_size_per_device * devices_num

    train_audios_dir = "/home/qiuqiangkong/workspaces/nesd2/audios/vctk_2s_segments/train"
    test_audios_dir = "/home/qiuqiangkong/workspaces/nesd2/audios/vctk_2s_segments/test"

    train_dataset = DatasetFreefield(
        audios_dir=train_audios_dir, 
        expand_frames=201, 
        simulator_configs=simulator_configs
    )
    val_dataset = DatasetFreefield(
        audios_dir=test_audios_dir, 
        expand_frames=201,
        simulator_configs=simulator_configs
    )

    train_batch_sampler = BatchSampler(batch_size=batch_size, iterations_per_epoch=1000)
    train_batch_sampler = DistributedSamplerWrapper(train_batch_sampler)

    val_batch_sampler = BatchSampler(batch_size=batch_size, iterations_per_epoch=5)
    val_batch_sampler = DistributedSamplerWrapper(val_batch_sampler)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        # pin_memory=True
        pin_memory=False
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_sampler=val_batch_sampler, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    # for i, data in enumerate(train_loader):
    #     print(i)
        
    # return data_module
    return train_dataloader, val_dataloader


class LitModel(L.LightningModule):
    def __init__(self, net, loss_function, learning_rate):
        super().__init__()
        self.net = net
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        
    def training_step(self, batch_dict, batch_idx):
        output_dict = self.net(batch_dict)
        loss = self.loss_function(output_dict=output_dict, target_dict=batch_dict)
        return loss

    def validation_step(self, batch_dict, batch_idx):
        output_dict = self.net(batch_dict)
        loss = self.loss_function(output_dict=output_dict, target_dict=batch_dict)
        print(loss)
        self.log("test_loss", loss.item())

    def predict_step(self, batch_dict):
        self.eval()
        with torch.no_grad():
            output_dict = self.net(batch_dict)
        
        return output_dict

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.net.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0,
            amsgrad=True,
        )
        '''
        # scheduler = LambdaLR(optimizer, self.lr_lambda_func)
        scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
            optimizer=optimizer, 
            warmup_epochs=10, 
            max_epochs=10000, 
            warmup_start_lr=0.0, 
            eta_min=0.0, 
            last_epoch=-1
        )

        output_dict = {
            "optimizer": optimizer,
            "lr_scheduler": {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }
        '''
        return optimizer


# def loss1(output_dict, target_dict):
#     from IPython import embed; embed(using=False); os._exit(0)


def train(args) -> NoReturn:
    r"""Train & evaluate and save checkpoints.

    Args:
        workspace: str, directory of workspace
        gpus: int
        config_yaml: str, path of config file for training
    """

    # arugments & parameters
    workspace = args.workspace
    config_yaml = args.config_yaml
    filename = args.filename

    # distributed = True if gpus > 1 else False
    # evaluate_device = "cuda" if gpus > 0 else "cpu"

    devices_num = torch.cuda.device_count()

    configs = read_yaml(config_yaml)
    device = configs["train"]["device"]
    # devices_num = configs["train"]["devices_num"]

    num_workers = configs["train"]["num_workers"]
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

    distributed = True if devices_num > 1 else False

    # paths
    checkpoints_dir, logs_dir, statistics_path = get_dirs(
        workspace, filename, config_yaml,
    )

    # training data module
    train_dataloader, val_dataloader = get_data_module(
        workspace=workspace,
        config_yaml=config_yaml,
        devices_num=devices_num,
        num_workers=num_workers,
        distributed=distributed,
    )

    # model
    Net = eval(model_type)
    loss_function = eval(loss_type)

    net = Net(mics_num=4)

    lit_model = LitModel(
        net=net, 
        loss_function=loss_function, 
        learning_rate=learning_rate
    )

    # callbacks = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="{epoch}-{step}-{test_loss:.3f}",
        verbose=True,
        save_last=False,
        save_weights_only=True,
        # every_n_train_steps=50,
        save_top_k=3,
        monitor="test_loss",
    )

    callbacks = [checkpoint_callback]
    
    trainer = L.Trainer(
        accelerator="auto",
        devices=devices_num,
        max_epochs=50,
        num_nodes=1,
        precision="32-true",
        callbacks=callbacks,
        # enable_checkpointing=True,
        num_sanity_val_steps=0,
        use_distributed_sampler=False, 
    )

    # Train the model
    trainer.fit(
        model=lit_model, 
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=None
    )



    '''
    # callbacks
    callbacks = get_callback(
        config_yaml=config_yaml,
        workspace=workspace,
        checkpoints_dir=checkpoints_dir,
        statistics_path=statistics_path,
        logger=None,
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
    )

    # trainer
    trainer = pl.Trainer(
        checkpoint_callback=False,
        gpus=devices_num,
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
    '''


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest="mode")

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
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
