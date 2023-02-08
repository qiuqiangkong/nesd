import logging
import os
import time
from typing import Dict, List, NoReturn

import librosa
import h5py
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch
from pytorch_lightning.utilities import rank_zero_only

# from spatial_audio_nn5.callbacks.base import SaveCheckpointsCallback
from nesd.utils import StatisticsContainer, read_yaml
from nesd.callbacks.base import SaveCheckpointsCallback
from nesd.data.samplers import Sampler
from nesd.data.data_modules import Dataset, DataModule
from nesd.data.data_modules import *
# from nesd.data.samplers import SegmentSampler
# from nesd.data.data_modules import DatasetBaseline, DatasetPyRoomAcoustics, DataModule, DatasetPyRoomAcousticsFramewise, DatasetDcaseRir, DatasetDcaseRirNoise
# from spatial_audio_nn5.data.data_modules import Dataset, DataModule, MicrophoneTrajectory
# from spatial_audio_nn5.data.samplers import SegmentSampler


def get_callback(
    config_yaml: str,
    workspace: str,
    checkpoints_dir: str,
    statistics_path: str,
    logger: pl.loggers.TensorBoardLogger,
    model: nn.Module,
    loss_function,
    evaluate_device: str,
) -> List[pl.Callback]:
    r"""Get MUSDB18 callbacks of a config yaml.

    Args:
        config_yaml: str
        workspace: str
        checkpoints_dir: str, directory to save checkpoints
        statistics_dir: str, directory to save statistics
        logger: pl.loggers.TensorBoardLogger
        model: nn.Module
        evaluate_device: str

    Return:
        callbacks: List[pl.Callback]
    """
    
    configs = read_yaml(config_yaml)
    dataset_type = configs['dataset_type']
    sampler_type = configs['sampler_type']
    test_hdf5s_dir = os.path.join(workspace, configs['sources']['test_hdf5s_dir'])
    classes_num = configs['sources']['classes_num']
    evaluate_step_frequency = configs['train']['evaluate_step_frequency']
    save_step_frequency = configs['train']['save_step_frequency']
    batch_size = configs['train']['batch_size']
    
    # source_configs = configs['sources']
    # background_configs = configs['background'] if 'background' in configs.keys() else None
    # room_configs = configs['rooms'] if 'room' in configs.keys() else None
    # microphone_configs = configs['microphones']
    # rendering_configs = configs['rendering']
    # target_configs = configs['targets']

    # save checkpoint callback
    save_checkpoints_callback = SaveCheckpointsCallback(
        model=model,
        checkpoints_dir=checkpoints_dir,
        save_step_frequency=save_step_frequency,
    )

    # statistics container
    statistics_container = StatisticsContainer(statistics_path)

    evaluate_test_callback = EvaluationCallback(
        sampler_type=sampler_type,
        dataset_type=dataset_type,
        classes_num=classes_num,
        split="test",
        hdf5s_dir=test_hdf5s_dir,
        model=model,
        loss_function=loss_function,
        batch_size=batch_size,
        device=evaluate_device,
        evaluate_step_frequency=evaluate_step_frequency,
        logger=logger,
        statistics_container=statistics_container,
    )

    # callbacks = [save_checkpoints_callback, evaluate_train_callback, evaluate_test_callback]
    callbacks = [save_checkpoints_callback, evaluate_test_callback]
    # callbacks = [save_checkpoints_callback]

    return callbacks


class EvaluationCallback(pl.Callback):
    def __init__(
        self,
        sampler_type,
        dataset_type,
        classes_num,
        split: str,
        hdf5s_dir,
        model: nn.Module,
        loss_function,
        batch_size: int,
        device: str,
        evaluate_step_frequency: int,
        logger: pl.loggers.TensorBoardLogger,
        statistics_container: StatisticsContainer,
    ):
        r"""Callback to evaluate every #save_step_frequency steps.

        Args:
            dataset_dir: str
            model: nn.Module
            target_source_types: List[str], e.g., ['vocals', 'bass', ...]
            input_channels: int
            split: 'train' | 'test'
            sample_rate: int
            segment_samples: int, length of segments to be input to a model, e.g., 44100*30
            batch_size, int, e.g., 12
            device: str, e.g., 'cuda'
            evaluate_step_frequency: int, evaluate every #save_step_frequency steps
            logger: object
            statistics_container: StatisticsContainer
        """
        self.sampler_type = sampler_type
        self.dataset_type = dataset_type
        self.classes_num = classes_num
        self.split = split
        self.model = model
        self.loss_function = loss_function
        self.hdf5s_dir = hdf5s_dir
        self.device = device
        self.evaluate_step_frequency = evaluate_step_frequency
        self.logger = logger
        self.statistics_container = statistics_container

        self.batch_size = 16
        self.steps_per_epoch = 10000
        self.num_workers = 8
        self.distributed = False

        if split == "train":
            self.random_seed = 1234

        elif split == "test":
            self.random_seed = 2345
            
        else:
            raise NotImplementedError

        
    @rank_zero_only
    def on_batch_end(self, trainer: pl.Trainer, _) -> NoReturn:
        r"""Evaluate separation SDRs of audio recordings."""
        global_step = trainer.global_step

        if global_step % self.evaluate_step_frequency == 0:

            _Sampler = eval(self.sampler_type)
            _Dataset = eval(self.dataset_type)

            # sampler
            train_sampler = _Sampler(
                batch_size=self.batch_size,
                steps_per_epoch=self.steps_per_epoch,
                random_seed=self.random_seed,
            )

            train_dataset = _Dataset(
                hdf5s_dir=self.hdf5s_dir,
                classes_num=self.classes_num,
            )

            # data module
            data_module = DataModule(
                train_sampler=train_sampler,
                train_dataset=train_dataset,
                num_workers=self.num_workers,
                distributed=self.distributed,
            )

            data_module.setup()

            cnt = 0
            losses = []

            for batch_data_dict in data_module.train_dataloader():
                
                max_agents_contain_waveform = batch_data_dict['agent_waveform'].shape[1]
                
                input_dict = {
                    'mic_position': batch_data_dict['mic_position'].to(self.device),
                    'mic_look_direction': batch_data_dict['mic_look_direction'].to(self.device),
                    'mic_waveform': batch_data_dict['mic_waveform'].to(self.device),
                    'agent_position': batch_data_dict['agent_position'].to(self.device),
                    'agent_look_direction': batch_data_dict['agent_look_direction'].to(self.device),
                    'max_agents_contain_waveform': max_agents_contain_waveform,
                }

                if 'agent_look_depth' in batch_data_dict.keys():
                    input_dict['agent_look_depth'] = batch_data_dict['agent_look_depth'].to(self.device)
                    input_dict['max_agents_contain_depth'] = batch_data_dict['agent_look_depth'].shape[1]

                target_dict = {
                    'agent_see_source': batch_data_dict['agent_see_source'].to(self.device),
                    'agent_waveform': batch_data_dict['agent_waveform'].to(self.device),
                }

                if 'agent_see_source_classwise' in batch_data_dict.keys():
                    target_dict['agent_see_source_classwise'] = batch_data_dict['agent_see_source_classwise'].to(self.device)

                if 'agent_exist_source' in batch_data_dict.keys():
                    target_dict['agent_exist_source'] = batch_data_dict['agent_exist_source'].to(self.device)

                with torch.no_grad():
                    self.model.eval()
                    output_dict = self.model(data_dict=input_dict)

                loss = self.loss_function(
                    model=self.model,
                    output_dict=output_dict,
                    target_dict=target_dict,
                )
                losses.append(loss.item())

                if cnt == 10:
                    break

                cnt += 1

            loss = np.mean(losses)
            logging.info("{} loss: {:.3f}".format(self.split, loss))

            statistics = {"loss": loss}
            self.statistics_container.append(global_step, statistics, self.split)
            self.statistics_container.dump()
