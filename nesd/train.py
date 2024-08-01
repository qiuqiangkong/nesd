import argparse
import os
from typing import Dict, List, NoReturn
import torch
from torch.optim.lr_scheduler import LambdaLR
import time
from pathlib import Path

from nesd.utils import read_yaml
from nesd.data.dataset import Dataset
from nesd.data.collate import collate_fn
from nesd.losses import get_loss
from nesd.data.samplers import InfiniteRandomSampler

import random
import numpy as np
random.seed(0)
np.random.seed(0)


def warmup_lambda(step, warm_up_steps=1000):
    if step <= warm_up_steps:
        return step / warm_up_steps
    else:
        return 1.


def train(args) -> NoReturn:
    r"""Train & evaluate and save checkpoints.

    Args:
        workspace: str, directory of workspace
        gpus: int
        config_yaml: str, path of config file for training
    """

    # Arugments & parameters
    workspace = args.workspace
    config_yaml = args.config_yaml
    filename = Path(__file__).stem
    devices_num = torch.cuda.device_count()
    
    configs = read_yaml(config_yaml)

    simulator_configs = configs["simulator_configs"]
    mics_meta = read_yaml(simulator_configs["mics_yaml"])
    mics_num = len(mics_meta["mic_coordinates"])

    device = configs["train"]["device"]
    batch_size_per_device = configs["train"]["batch_size_per_device"]
    num_workers = configs["train"]["num_workers"]
    model_name = configs["train"]["model_name"]
    loss_name = configs["train"]["loss_name"]
    lr = float(configs["train"]["learning_rate"])

    # batch_size = batch_size_per_device * 
    # checkpoints_dir = Path(workspace, "checkpoints", filename, model_name)
    checkpoints_dir = Path(workspace, "checkpoints", filename, Path(config_yaml).stem)
    # from IPython import embed; embed(using=False); os._exit(0) 
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    batch_size = batch_size_per_device
    loss_func = get_loss(loss_name)

    # Dataset
    train_dataset = Dataset(
        simulator_configs=simulator_configs,
        split="train"
    )

    sampler = InfiniteRandomSampler()

    # Dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers, 
        pin_memory=True
    )

    # Model
    model = get_model(model_name, mics_num)

    # checkpoint_path = "/home/qiuqiangkong/workspaces/nesd/checkpoints/train/07a/step=300000.pth"
    # model.load_state_dict(torch.load(checkpoint_path)) 

    model.to(device)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.0,
        amsgrad=True,
    )

    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=warmup_lambda)

    for step, data in enumerate(dataloader):
        
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        
        optimizer.zero_grad()
        # from IPython import embed; embed(using=False); os._exit(0)

        model.train()
        output_dict = model(data) 

        loss = loss_func(output_dict, data)
        loss.backward()

        optimizer.step()
        scheduler.step()

        if step % 10000 == 0: 
            checkpoint_path = Path(checkpoints_dir, "step={}.pth".format(step))
            torch.save(model.state_dict(), checkpoint_path)
            print("Save model to {}".format(checkpoint_path))

            checkpoint_path = Path(checkpoints_dir, "latest.pth")
            torch.save(model.state_dict(), Path(checkpoint_path))
            print("Save model to {}".format(checkpoint_path))

        if step % 1 == 0:
            print(step, loss.item())


def get_model(model_name, mics_num):
    
    if model_name == "NeSD":
        from nesd.models.nesd import NeSD
        return NeSD(mics_num=mics_num)

    elif model_name == "NeSD2":
        from nesd.models.nesd import NeSD2
        return NeSD2(mics_num=mics_num)

    elif model_name == "NeSD3":
        from nesd.models.nesd import NeSD3
        return NeSD3(mics_num=mics_num)

    elif model_name == "NeSD4":
        from nesd.models.nesd import NeSD4
        return NeSD4(mics_num=mics_num)

    elif model_name == "NeSD4b":
        from nesd.models.nesd import NeSD4b
        return NeSD4b(mics_num=mics_num)

    elif model_name == "NeSD5":
        from nesd.models.nesd import NeSD5
        return NeSD5(mics_num=mics_num)

    elif model_name == "NeSD6":
        from nesd.models.nesd import NeSD6
        return NeSD6(mics_num=mics_num)

    elif model_name == "NeSD7":
        from nesd.models.nesd import NeSD7
        return NeSD7(mics_num=mics_num)

    elif model_name == "NeSD7_srcs_num":
        from nesd.models.nesd import NeSD7_srcs_num
        return NeSD7_srcs_num(mics_num=mics_num)

    elif model_name == "NeSD8":
        from nesd.models.nesd import NeSD8
        return NeSD8(mics_num=mics_num)

    elif model_name == "NeSD9":
        from nesd.models.nesd import NeSD9
        return NeSD9(mics_num=mics_num)

    elif model_name == "NeSD9b":
        from nesd.models.nesd import NeSD9b
        return NeSD9b(mics_num=mics_num)

    elif model_name == "Model02":
        from nesd.old_models.models02 import Model02
        return Model02(mics_num=mics_num)

    else:
        raise NotImplementedError


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str)
    parser.add_argument('--config_yaml', type=str)
    
    args = parser.parse_args()

    train(args)