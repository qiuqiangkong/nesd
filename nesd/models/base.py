import math
import torch


# X_LOW = -10.
# X_HIGH = 10.
# Y_LOW = -10.
# Y_HIGH = 10.
# Z_LOW = -10.
# Z_HIGH = 10.
# LENGTH = X_HIGH - X_LOW
# WIDTH = Y_HIGH - Y_LOW
# HEIGHT = Z_HIGH - Z_LOW
# MAX_DISTANCE = math.sqrt(LENGTH ** 2 + WIDTH ** 2 + HEIGHT ** 2)
BOUNDARY_LOW = -10.
BOUNDARY_HIGH = 10.
BOUNDARY_LENGTH = BOUNDARY_HIGH - BOUNDARY_LOW
MAX_DISTANCE = math.sqrt(BOUNDARY_LENGTH ** 2 + BOUNDARY_LENGTH ** 2 + BOUNDARY_LENGTH ** 2)


class PositionEncoder:
    def __init__(self, size):
        self.size = size

    def __call__(self, vector):

        vector = (vector - BOUNDARY_LOW) / BOUNDARY_LENGTH  # Normalize to [0, 1]
        vector *= (math.pi / 2)  # Normalize to [0, pi/2]
        
        angles = []

        for i in range(self.size):
            angles.append((2 ** i) * vector)

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

        azi = (azi + math.pi) / 4   # Normalize to [0, pi/2]
        ele = (ele + math.pi) / 4   # Normalize to [0, pi/2]

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

    def __call__(self, distance):

        distance = distance / MAX_DISTANCE  # Normalize to [0, 1]
        distance *= (math.pi / 2)  # Normalize to [0, pi/2]

        angles = []

        for i in range(self.size):
            angles.append((2 ** i) * distance)

        angles = torch.stack(angles, dim=-1)
        # (bs, mics_num, frames_num, size)

        dist_emb = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
        # (bs, mics_num, frames_num, size * 2)

        return dist_emb


class AngleEncoder:
    def __init__(self, size):
        self.size = size

    def __call__(self, angle):
        # angle: [-pi, pi]. Diff_phase: [0, 2*pi]

        angle = (angle + 2 * math.pi) / 8      # Normalize to [0, pi/2]

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
