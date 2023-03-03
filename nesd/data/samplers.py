from typing import List, Dict
import numpy as np
import os


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


class Sampler_VctkMusdb18hqD18t2:
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

        self.hdf5s_dirs = [
            "/home/tiger/workspaces/nesd2/hdf5s/vctk/sr=24000/train",
            "/home/tiger/workspaces/nesd2/hdf5s/musdb18hq/sr=24000/train",
            "/home/tiger/workspaces/nesd2/hdf5s/dcase2018_task2/sr=24000/train",
        ]
        self.hdf5_pathss = [[os.path.join(hdf5s_dir, hdf5_name) for hdf5_name in sorted(os.listdir(hdf5s_dir))] for hdf5s_dir in self.hdf5s_dirs]

    def __iter__(self) -> List[Dict]:
        r"""Yield a batch of meta info.

        Returns:
        """
        batch_size = self.batch_size

        while True:

            batch_meta_list = []

            while len(batch_meta_list) < batch_size:

                i = self.random_state.randint(len(self.hdf5s_dirs))
                j = self.random_state.randint(len(self.hdf5_pathss[i]))

                hdf5_path = self.hdf5_pathss[i][j]
                meta = {
                    'random_seed': self.random_state.randint(9999),
                    'hdf5_path': hdf5_path,
                }
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
