import argparse
import os
import pathlib
from typing import Dict, List, NoReturn
import torch
import time
from pathlib import Path

from nesd.utils import read_yaml, load_mics_meta
from nesd.data.dataset import Dataset
from nesd.data.collate import collate_fn
from nesd.losses import get_loss


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
    filename = pathlib.Path(__file__).stem
    devices_num = torch.cuda.device_count()
    # from IPython import embed; embed(using=False); os._exit(0)

    configs = read_yaml(config_yaml)

    simulator_configs = configs["simulator_configs"]
    mics_meta = load_mics_meta(simulator_configs["mics_yaml"])
    mics_num = len(mics_meta["microphone_coordinates"])

    device = configs["train"]["device"]
    batch_size_per_device = configs["train"]["batch_size_per_device"]
    num_workers = configs["train"]["num_workers"]
    model_name = configs["train"]["model_name"]
    loss_name = configs["train"]["loss_name"]
    lr = float(configs["train"]["learning_rate"])

    # batch_size = batch_size_per_device * 
    checkpoints_dir = Path(workspace, "checkpoints", model_name)
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
    model.to(device)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.0,
        amsgrad=True,
    )

    for step, data in enumerate(dataloader):
        
        for key in data.keys():
            data[key] = data[key].to(device)
        
        optimizer.zero_grad()

        model.train()
        output_dict = model(data=data) 

        loss = loss_func(output_dict, data)
        loss.backward()

        optimizer.step()

        if step % 10000 == 0:
            checkpoint_path = Path(checkpoints_dir, "step={}.pth".format(step))
            torch.save(model.state_dict(), checkpoint_path)
            print("Save model to {}".format(checkpoint_path))

            checkpoint_path = Path(checkpoints_dir, "latest.pth")
            torch.save(model.state_dict(), Path(checkpoint_path))
            print("Save model to {}".format(checkpoint_path))

        print(step, loss.item())


def get_model(model_name, mics_num):
    if model_name == "NeSD":
        from nesd.models.nesd import NeSD
        return NeSD(mics_num=mics_num)
    else:
        raise NotImplementedError


class InfiniteRandomSampler:
    def __init__(self):
        pass

    def __iter__(self):
        while True:
            yield 0




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str)
    parser.add_argument('--config_yaml', type=str)
    
    args = parser.parse_args()

    train(args)