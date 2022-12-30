from typing import List, Dict
import numpy as np


class Sampler:
    def __init__(
        self,
        batch_size,
        steps_per_epoch: int,
        random_seed=1234,
    ):
        r"""Sample training indexes of sources.

        Args:
            steps_per_epoch: int, #steps_per_epoch is called an `epoch`
            random_seed: int
        """
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.random_state = np.random.RandomState(random_seed)
        
    def __iter__(self) -> List[Dict]:
        r"""Yield a batch of meta info.

        Returns:
        """
        batch_size = self.batch_size

        while True:

            batch_meta_list = []

            while len(batch_meta_list) < batch_size:

                meta = {'random_seed': self.random_state.randint(9999)}
                batch_meta_list.append(meta)
                
            yield batch_meta_list

    def __len__(self) -> int:
        return self.steps_per_epoch


class DistributedSamplerWrapper:
    def __init__(self, sampler):
        r"""Distributed wrapper of sampler."""
        self.sampler = sampler

    def __iter__(self) -> List[Dict]:

        num_replicas = dist.get_world_size()  # number of GPUs.
        rank = dist.get_rank()  # rank of current GPU

        for batch_meta_list in self.sampler:

            # Yield a subset of batch_meta_list on one GPU.
            yield batch_meta_list[rank::num_replicas]

    def __len__(self) -> int:
        return len(self.sampler)
