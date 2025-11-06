import math
import torch
from torch import Tensor, LongTensor, BoolTensor
import torchaudio
from einops import rearrange


def get_delayed_filter(
    delay: Tensor, 
    mask: Tensor | None = None, 
    N=99
) -> Tensor:
    r"""Fractional-delay filter: compute integer delay and fractional-delay sinc filter.
    The filter is split into integer and fractional parts to save memory.

    Args:
        delay (Tensor): (any,)
        mask (Tensor): (any,)
        N: (int)

    Returns:
        delay_int (LongTensor): (any,). Integer part of the delay
        h (Tensor): (any, filter_len). Fractional-delay sinc filter
        origin (int): Origin index
    """

    device = delay.device
    half = (N - 1) // 2
    
    delay_int = torch.floor(delay).long()  # (any,)
    delay_frac = delay - delay_int  # (any,)
    x = torch.arange(-half, half + 1).to(device) - delay_frac[..., None]  # (any, N)
    h_frac = torch.sinc(x) * torch.blackman_window(N).to(device)  # (any, N)
    h_frac = h_frac / torch.sum(h_frac, dim=-1)[..., None]  # (any, N)

    if mask is not None:
        h_frac = h_frac * mask[..., None]
        delay_int = (delay_int * mask).long()

    origin = half

    return delay_int, h_frac, origin


def add_delayed_filters(
    delay_int: LongTensor, 
    h_frac: Tensor, 
    origin: int, 
    length=48000, 
    strict=True
) -> Tensor:
    r"""Recombine the fractional-delay filter from its integer and fractional 
    parts. Sum all filters.
    
    Args:
        delay_int (LongTensor): (any, filters_num)
        h_frac (Tensor): (any, filters_num, filter_len)
        origin (int): origin of h_frac
        length (int): combined filter length
        strict (bool): if True, then check all indices are in the domain
        
    Returns:
        h_sum: (any, length)
    """

    device = h_frac.device
    h_sum = torch.zeros(list(h_frac.shape[0 : -2]) + [length,]).to(device)
    indices = origin + delay_int[..., None] + torch.arange(-origin, -origin + h_frac.shape[-1]).to(device)
    
    if strict:
        assert 0 <= indices.max() < length
    else:
        mask = ((indices >= 0) & (indices < length)).float()
        indices = (indices * mask).long()
        h_frac = h_frac * mask

    h_sum.scatter_add_(dim=-1, index=indices.flatten(-2), src=h_frac.flatten(-2))

    return h_sum


def add_delayed_filter(
    delay_int: LongTensor, 
    h_frac: Tensor, 
    origin: int, 
    length=48000, 
    strict=True
) -> Tensor:
    r"""Recombine the fractional-delay filter from its integer and fractional parts."""
    return add_delayed_filters(delay_int[..., None], h_frac[..., None, :], origin, length, strict)


def convolve(x: Tensor, h: Tensor, origin: int) -> Tensor:
    r"""
    Args:
        x (Tensor): (any, x_len)
        h (Tensor): (any, h_len)
        origin (int), origin of h

    Returns:
        out: (any, x_len+h_len/2)
    """

    return torchaudio.functional.convolve(x, h, mode="full")[..., origin :]


def sph2cart(sph: Tensor) -> Tensor:
    r"""Convert spherical coordinates to Cartesian coordinates.

    Args:
        sph: (any, 3), rθφ, θ ∈ [0, π], φ ∈ [0, 2π)

    Returns:
        out: (any, 3), xyz
    """

    r, theta, phi = sph[..., 0], sph[..., 1], sph[..., 2]
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)


def cart2sph(cart: Tensor) -> Tensor:
    r"""Convert spherical coordinates to Cartesian coordinates.

    Args:
        out: (any, 3), xyz
        
    Returns:
        sph: (any, 3), rθφ, θ ∈ [0, π], φ ∈ [0, 2π)
    """

    x, y, z = cart[..., 0], cart[..., 1], cart[..., 2]
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.acos(z / r)
    phi = torch.atan2(y, x) % (2 * math.pi)
    return torch.stack([r, theta, phi], dim=-1)


def normalize(x: Tensor) -> Tensor:
    r"""Normalize tensor over the last dimension."""
    return x / torch.norm(x, dim=-1, keepdim=True)


def included_angle(a: Tensor, b: Tensor, eps=1e-8) -> Tensor:
    r"""Compute included angle (rad) between two tensors."""
    dot = torch.sum(a * b, dim=-1)
    norm_a = torch.norm(a, dim=-1)
    norm_b = torch.norm(b, dim=-1)
    cos_theta = torch.clamp(dot / (norm_a * norm_b + eps), -1.0, 1.0)
    return torch.acos(cos_theta)


def convolve(x: Tensor, h: Tensor, origin: int) -> Tensor:
    r"""
    Args:
        x: (any, x_len)
        h: (any, h_len)
        origin: int, origin of h

    Returns:
        out: (any, x_len)
    """
    
    return torchaudio.functional.convolve(x, h, mode="full")[..., origin :]


def db_to_gain(db: Tensor) -> Tensor:
    r"""Conver dB to gain."""
    return 10 ** (db / 20.)


def transform_coordinate(
    x: Tensor, 
    origin_from=torch.zeros(3), 
    origin_to=torch.zeros(3), 
    R_from=torch.eye(3), 
    R_to=torch.eye(3)
) -> Tensor:
    r"""Batch transform coordinates from the first coordinate system to the second.
    Batch version of (x @ R_from.T + origin_from - origin_to) @ R_to

    b: batch_size
    n: data_num

    Args:
        x: (b, n, 3), coordinate in the 1st coordinate system
        origin_from: (b, 3,), origin in the 1st coordinate system
        origin_to: (b, 3,), origin in the 2nd coordinate system
        R_from: (b, 3, 3), rotation matrix in the 1st coordinate system
        R_to: (b, 3, 3), rotation matrix in the 2nd coordinate system

    Returns:
        out: (b, n, 3), coordinate in the 2nd coordinate system
    """

    x = torch.bmm(x, rearrange(R_from, 'b i j -> b j i'))  # (b, n, 3)
    x = x + origin_from[:, None, :] - origin_to[:, None, :]  # (b, n, 3)
    out = torch.bmm(x, R_to)  # (b, n, 3)
    return out


def sample_direction(
    size=(1,),
    min_ele=0.,
    max_ele=math.pi,
    min_azi=0., 
    max_azi=2. * math.pi
) -> Tensor:
    r"""Uniformly sample 3D directions based on elevation θ ∈ [0, π] and 
    azimuth φ ∈ [0, 2π).
    """

    r = torch.ones(size)
    ele = torch.distributions.Uniform(min_ele, max_ele).sample(size)
    azi = torch.distributions.Uniform(min_azi, max_azi).sample(size)
    sph = torch.stack([r, ele, azi], dim=-1)
    direction = sph2cart(sph)
    return direction


def is_within_angle(vec1: Tensor, vec2: Tensor, max_ele: float, max_azi: float) -> BoolTensor:
    r"""Batch check whether the angles between two tensors are within the 
    specified azimuth and elevation thresholds.

    Args:
        vec1 (Tensor): (any, 3)
        vec2 (Tensor): (any, 3)
        max_ele (float)
        max_azi (float)

    Returns:
        out (BoolTensor): (any,)
    """

    sph1 = cart2sph(vec1)
    sph2 = cart2sph(vec2)
    _, theta1, phi1 = sph1[..., 0], sph1[..., 1], sph1[..., 2]
    _, theta2, phi2 = sph2[..., 0], sph2[..., 1], sph2[..., 2]
    is_within = (wrap_angle(theta1 - theta2).abs() <= max_ele) & (wrap_angle(phi1 - phi2).abs() <= max_azi)
    return is_within


def wrap_angle(angle: Tensor) -> Tensor:
    r"""Wrap angle between 0 and 2π."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def perturb_direction(direction: Tensor, ele: float, azi: float) -> Tensor:
    r"""Perturb directions.

    Args:
        direction: (any, 3)
        ele (float)
        azi (float)

    Returns:
        perturb_dir: (any, 3)
    """

    device = direction.device

    sph = cart2sph(direction)
    r, theta, phi = sph[..., 0], sph[..., 1], sph[..., 2]

    size = direction.shape[0 : -1]
    theta = theta + torch.distributions.Uniform(-ele, ele).sample(size).to(device)
    phi = phi + torch.distributions.Uniform(-azi, azi).sample(size).to(device)
    perturb_sph = torch.stack([r, theta, phi], dim=-1)
    perturb_dir = sph2cart(perturb_sph)
    return perturb_dir