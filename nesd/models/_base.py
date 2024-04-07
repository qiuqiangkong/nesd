import math
import torch


class PositionEncoder:
    def __init__(self, size):
        self.size = size
        self.max_distance = 20


    def __call__(self, vector):

        angles = []

        for i in range(self.size):
            angles.append((2 ** i) * (2 * math.pi * vector / self.max_distance))

        angles = torch.cat(angles, dim=-1)
        # (bs, mics_num, frames_num, size * 3)

        pos_emb = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
        # (bs, mics_num, frames_num, size * 6)

        return pos_emb


class OrientationEncoder:
    def __init__(self, size):
        self.size = size

    def __call__(self, direction):

        azi, ele, _ = cart2sph_torch(direction)

        angles = []

        for i in range(self.size):
            angles.append((2 ** i) * azi)
            angles.append((2 ** i) * ele)

        angles = torch.stack(angles, dim=-1)
        # (bs, mics_num, frames_num, size * 2)

        orien_emb = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
        # (bs, mics_num, frames_num, size * 4)

        return orien_emb


class DistanceEncoder:
    def __init__(self, size):
        self.size = size
        self.max_distance = 20

    def __call__(self, distance):

        angles = []

        for i in range(self.size):
            angles.append((2 ** i) * (2 * math.pi * distance / self.max_distance))

        angles = torch.stack(angles, dim=-1)
        # (bs, mics_num, frames_num, size)

        dist_emb = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
        # (bs, mics_num, frames_num, size * 2)

        return dist_emb


class AngleEncoder:
    def __init__(self, size):
        self.size = size

    def __call__(self, angle):

        angles = []

        for i in range(self.size):
            angles.append((2 ** i) * angle)

        angles = torch.stack(angles, dim=-1)
        # (bs, mics_num, frames_num, size)

        angle_emb = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
        # (bs, mics_num, frames_num, size * 2)

        return angle_emb


def cart2sph_torch(vector):
    x = vector[..., 0]
    y = vector[..., 1]
    z = vector[..., 2]
    r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    azimuth = torch.atan2(y, x)
    elevation = torch.asin(z / r)
    
    return azimuth, elevation, r
