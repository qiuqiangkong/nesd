from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

import wandb
from nesd.utils.utils import parse_yaml, requires_grad, update_ema, LinearWarmUp, to_device


def train(args) -> None:
    r"""Train a neural sound field decomposition (NeSD) system."""

    # Arguments
    wandb_log = not args.no_log
    config_path = args.config
    filename = Path(__file__).stem
    
    # Configs
    configs = parse_yaml(config_path)
    device = configs["train"]["device"]
    ckpt_path = configs["train"]["resume_ckpt_path"]

    # Checkpoints directory
    config_name = Path(config_path).stem
    ckpts_dir = Path("./checkpoints", filename, config_name)
    Path(ckpts_dir).mkdir(parents=True, exist_ok=True)

    # Datasets
    train_dataset = get_dataset(configs, split="train")
    test_dataset = get_dataset(configs, split="test")

    # Sampler
    train_sampler = get_sampler(configs, train_dataset)

    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=configs["train"]["batch_size_per_device"],
        sampler=train_sampler,
        num_workers=configs["train"]["num_workers"], 
        pin_memory=True
    )
    
    # Data processor
    data_transform = get_data_transform(configs).to(device)

    # Model
    model = get_model(configs).to(device)

    # EMA (optional)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    ema.eval()  # EMA model should always be in eval mode

    # Optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(
        configs=configs, 
        params=model.parameters()
    )
    
    # Logger
    if wandb_log:
        wandb.init(project="nesd", name=f"{filename}_{config_name}")

    loss_fn = get_loss_fn(configs)
    
    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        data = to_device(data, device)
        data = data_transform(data)

        # ------ 1. Training ------
        # 1.1 Forward
        model.train()
        output = model(data["mic_wav"], data["lis_dir"])

        # 1.2 Loss
        loss = loss_fn(input=output, target=data["target"])
        
        # 1.3 Optimize
        optimizer.zero_grad()  # Reset all parameter.grad to 0
        loss.backward()  # Update all parameter.grad
        optimizer.step()  # Update all parameters based on all parameter.grad
        update_ema(ema, model, decay=0.999)

        # 1.4 Learning rate scheduler
        if scheduler:
            scheduler.step()

        if step % 100 == 0:
            print(loss)

        # ------ 2. Evaluation ------
        # 2.1 Evaluate
        if step % configs["train"]["test_every_n_steps"] == 0:

            test_loss = validate(configs, test_dataset, data_transform, model)

            if wandb_log:
                wandb.log(
                    data={"test_loss": test_loss},
                    step=step
                )

            print("====== Overall metrics ====== ")
            print("Test loss: {:.3f}".format(test_loss))
        
        # 2.2 Save model
        if step % configs["train"]["save_every_n_steps"] == 0:
            
            ckpt_path = Path(ckpts_dir, f"step={step}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Save model to {ckpt_path}")

            ckpt_path = Path(ckpts_dir, f"step={step}_ema.pth")
            torch.save(ema.state_dict(), ckpt_path)
            print(f"Save model to {ckpt_path}")

        if step == configs["train"]["training_steps"]:
            break
        
def get_dataset(
    configs: dict, 
    split: str
) -> Dataset:
    r"""Get datasets."""

    ds = f"{split}_datasets"
    sr = configs["sample_rate"]
    
    for name in configs[ds].keys():
    
        if name == "ShoeboxISMSpeech":
            from nesd.datasets.shoebox_ism_speech import ShoeboxISMSpeech
            return ShoeboxISMSpeech(configs[ds][name], sr)

        else:
            raise ValueError(name)


def get_sampler(configs: dict, dataset: Dataset) -> Iterable:
    r"""Get sampler."""

    name = configs["sampler"]

    if name == "RepeatShuffleSampler":
        from nesd.samplers.repeat_shuffle_sampler import RepeatShuffleSampler
        return RepeatShuffleSampler(dataset)

    else:
        raise ValueError(name)


def get_data_transform(configs: dict):
    r"""Transform data into latent representations and conditions."""

    name = configs["data_transform"]["name"]
    sr = configs["sample_rate"]

    if name == "AcousticRenderer":
        from nesd.data_transforms.render import AcousticRenderer
        return AcousticRenderer(configs["data_transform"])

    else:
        raise ValueError(name)


def get_model(
    configs: dict, 
    ckpt_path: str = None
) -> nn.Module:
    r"""Initialize base model."""

    name = configs["model"]["name"]

    if name == "ConvDNN":
        from nesd.models.convdnn import ConvDNN
        model = ConvDNN(**configs["model"])

    else:
        raise ValueError(name)

    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt)

    return model


def get_loss_fn(configs: dict) -> callable:
    r"""Get loss function."""

    loss_type = configs["train"]["loss"]

    if loss_type == "bce":
        return F.binary_cross_entropy

    else:
        raise ValueError(loss_type)


def get_optimizer_and_scheduler(
    configs: dict, 
    params: list[torch.Tensor]
) -> tuple[optim.Optimizer, None | optim.lr_scheduler.LambdaLR]:
    r"""Get optimizer and scheduler."""

    lr = float(configs["train"]["lr"])
    warm_up_steps = configs["train"]["warm_up_steps"]
    optimizer_name = configs["train"]["optimizer"]

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(params=params, lr=lr)

    if warm_up_steps:
        lr_lambda = LinearWarmUp(warm_up_steps)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None

    return optimizer, scheduler


def validate(configs, dataset, data_transform, model) -> float:
    r"""Validate the model on part of data.
    """

    device = next(model.parameters()).device
    loss_fn = get_loss_fn(configs)
    losses = []
    outputs = []

    for n, data in enumerate(dataset):
        
        data = default_collate([data])
        data = to_device(data, device)
        data = data_transform(data)

        with torch.no_grad():
            model.eval()
            output = model(data["mic_wav"], data["lis_dir"])

        loss = loss_fn(input=output, target=data["target"]).item()
        losses.append(loss)
        outputs.append(output.cpu())

        if n == 20:
            break

    loss = np.mean(losses)
    outputs = torch.stack(outputs, dim=0)
    print("max: {:.3f}".format(torch.max(outputs)))

    return loss

     
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)
    args = parser.parse_args()

    train(args)