import pytorch_lightning as pl
import torch.nn as nn
import torch
from typing import Dict, List, NoReturn, Tuple, Callable, Any

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


class LitModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_function: Callable,
        optimizer_type: str,
        learning_rate: float,
        lr_lambda: Callable,
    ):
        r"""Pytorch Lightning wrapper of PyTorch model, including forward,
        optimization of model, etc.

        Args:
            batch_data_preprocessor: object, used for preparing inputs and
                targets for training. E.g., BasicBatchDataPreprocessor is used
                for preparing data in dictionary into tensor.
            model: nn.Module
            loss_function: function
            learning_rate: float
            lr_lambda: function
        """
        super().__init__()

        # self.task_type = task_type
        self.model = model
        self.optimizer_type = optimizer_type
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.lr_lambda = lr_lambda

    def training_step(self, batch_data_dict: Dict, batch_idx: int) -> torch.float:
        r"""Forward a mini-batch data to model, calculate loss function, and
        train for one step. A mini-batch data is evenly distributed to multiple
        devices (if there are) for parallel training.

        Args:
            batch_data_dict: e.g. {
                'vocals': (batch_size, channels_num, segment_samples),
                'accompaniment': (batch_size, channels_num, segment_samples),
                'mixture': (batch_size, channels_num, segment_samples)
            }
            batch_idx: int

        Returns:
            loss: float, loss function of this mini-batch
        """

        # Forward.
        self.model.train()

        max_agents_contain_waveform = batch_data_dict['agent_waveform'].shape[1]

        input_dict = {
            'mic_position': batch_data_dict['mic_position'],
            'mic_look_direction': batch_data_dict['mic_look_direction'],
            'mic_waveform': batch_data_dict['mic_waveform'],
            'agent_position': batch_data_dict['agent_position'],
            'agent_look_direction': batch_data_dict['agent_look_direction'],
            'max_agents_contain_waveform': max_agents_contain_waveform,
        }

        target_dict = {
            'agent_see_source': batch_data_dict['agent_see_source'],
            'agent_waveform': batch_data_dict['agent_waveform'],
        }

        output_dict = self.model(data_dict=input_dict)

        loss = self.loss_function(
            model=self.model,
            output_dict=output_dict,
            target_dict=target_dict,
        )

        return loss

    def configure_optimizers(self) -> Any:
        r"""Configure optimizer."""

        if self.optimizer_type == "AdamW":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.0,
                amsgrad=True,
            )

        else:
            raise NotImplementedError

        scheduler = {
            'scheduler': LambdaLR(optimizer, self.lr_lambda),
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [scheduler]