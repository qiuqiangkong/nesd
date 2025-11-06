import numpy as np
import torch
import math
import random

import nesd


def sph2cart(sph: np.ndarray) -> np.ndarray:
    r"""Convert spherical coordinates to Cartesian coordinates."""
    return nesd.utils.torch.sph2cart(torch.from_numpy(sph)).numpy()


def normalize(x: np.ndarray) -> np.ndarray:
    r"""Convert spherical coordinates to Cartesian coordinates."""
    return nesd.utils.torch.normalize(torch.from_numpy(x)).numpy()


def sample_rotation_matrix(up=np.array([0., 0., 1.])) -> np.ndarray:
    r"""Sample a rotation matrix.

    Args:
        up: (3,), up direction

    Returns:
        R: (3, 3), rotation matrix
    """
    
    front = sample_direction(
        min_azi=0, 
        max_azi=2 * math.pi, 
        min_ele=math.pi / 2, 
        max_ele=math.pi / 2
    )
    x_axis = front
    y_axis = np.cross(up, x_axis)
    z_axis = np.cross(x_axis, y_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=-1)
    return R


def sample_direction(
    min_ele=0.,
    max_ele=math.pi,
    min_azi=0., 
    max_azi=2. * math.pi, 
):
    r"""Uniformly sample a 3D direction based on elevation θ ∈ [0, π] and 
    azimuth φ ∈ [0, 2π).
    """

    azi = random.uniform(a=min_azi, b=max_azi)
    ele = random.uniform(a=min_ele, b=max_ele)
    sph = np.array([1., ele, azi])
    direction = sph2cart(sph)  # (3,)
    return direction


def transform_coordinate(
    x: np.ndarray, 
    origin_from=np.zeros(3), 
    origin_to=np.zeros(3), 
    R_from=np.eye(3), 
    R_to=np.eye(3)
) -> np.ndarray:
    r"""Transform coordinates from the first coordinate system to the second.

    Args:
        x: (any, 3), coordinate in the 1st coordinate system
        origin_from: (3,), origin in the 1st coordinate system
        origin_to: (3,), origin in the 2nd coordinate system
        R_from: (3, 3), rotation matrix in the 1st coordinate system
        R_to: (3, 3), rotation matrix in the 2nd coordinate system

    Returns:
        out: (any, 3), coordinate in the 2nd coordinate system
    """
    
    return (x @ R_from.T + origin_from - origin_to) @ R_to