import random
from typing import Sized


class RepeatShuffleSampler:
    def __init__(self, dataset: Sized) -> None:
        r"""Randomly sample indices of a dataset without replacement. Execute
        this process infinitely.
        """

        self.indices = list(range(len(dataset)))
        random.shuffle(self.indices)  # self.indices: [3, 7, 0, ...]
        self.p = 0  # pointer
        
    def __iter__(self) -> int:
        r"""Yield an index."""

        while True:

            if self.p == len(self.indices):
                random.shuffle(self.indices)
                self.p = 0
                
            index = self.indices[self.p]
            self.p += 1

            yield index