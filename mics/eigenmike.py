import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.special import spherical_jn, spherical_yn
from torch import Tensor

from nesd.utils.torch import add_delayed_filter, convolve, get_delayed_filter


def compute_spatial_impulse_responses(args) -> None:
    r"""Compuate rigid sphere microphone spatial impulse response (SIR).
    Author 1: Yin Cao
    Author 2: Qiuqiang Kong

    Ref: [1] Rafaely, B. (2015). Fundamentals of Spherical Array Processing. Springer.
    """

    # Arguments
    sphere_type = args.sphere_type
    out_path = args.out_path

    # Parameters
    c = 343.
    r = 0.042
    sr = 48000
    nfft = 256
    order = 30
    
    angles = np.deg2rad(range(0, 181, 1))
    hs = []

    for angle in angles:
        
        h = compute_impulse_response(angle, nfft, r, sr, c, order, sphere_type)  # (nfft,)
        hs.append(h)
        print("angle: {:.2f}".format(np.rad2deg(angle))) 

        if False:
            visualize(h)  # For debug. When sphere_type="open" will produce δ[n] centered at nfft/2
            sys.exit()
        
    hs = np.stack(hs, axis=0)  # (angles_num, n_fft)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset("angle", data=angles, dtype=np.float32)
        hf.create_dataset("h", data=hs, dtype=np.float32)
        hf.attrs.create("origin", data=nfft // 2, dtype=np.int32)

    print(f"Write out to {out_path}")


def compute_impulse_response(
    angle: float, 
    nfft: int, 
    r: float, 
    sr: int, 
    c: float, 
    order: int, 
    sphere_type: str
) -> np.ndarray:
    r"""Compute rigid sphere microphone impulse response at a given angle.

    Reference:
    [1] Rafaely, B. (2015). Fundamentals of Spherical Array Processing. Springer.

    Args:
        angle (float): Angle in radians
        nfft (int): Number of FFT points
        r (float): Sphere radius
        sr (int): Sampling rate
        c (float): Speed of sound
        order (int): Maximum spherical harmonics order
        sphere_type (str): "open" or "rigid"

    Returns:
        h (np.ndarray): Impulse response of shape (nfft,).
    """
    
    # --- 1. Frequency response ---
    # See Eq. 2.37, 2.38, 2.43, 2.46, 2.47, 4.4, 4.5 of [1]
    f = np.linspace(0, sr // 2, nfft // 2 + 1)  # (nfft/2+1,)
    kr = 2 * np.pi * f / c * r  # (nfft/2+1,)
    H = np.zeros(len(f), dtype=np.complex128)  # (nfft/2+1,)

    for n in range(order + 1):
        bn = compute_bn(n, kr, sphere_type)  # (nfft/2+1,)
        Pn = scipy.special.lpmv(0, n, np.cos(angle))  # scalar
        H += bn / (4 * np.pi) * (2 * n + 1) * Pn  # (nfft/2+1,)

    # --- 2. Impulse response ---
    h = np.fft.irfft(H)  # (nfft,)
    h = np.fft.fftshift(h)  # (nfft,)
    
    # 1. IR for delay correction via r·cosθ
    dist = r * np.cos(angle)  # scalar
    delayed_samples = (dist / c) * sr  # scalar

    delay_int, h_frac, origin = get_delayed_filter(
        delay=Tensor([delayed_samples]),  # (1,)
        mask=None,
        N=199
    )  # delay_int: (1,), h_frac: (1, l_frac)
    
    # IR after delay correction
    h_mic_delay = convolve(
        x=Tensor(h[None, :]), # (1, n_fft)
        h=h_frac,
        origin=origin
    )[:, 0 : nfft]  # (1, nfft)
    
    h_total = add_delayed_filter(
        delay_int=delay_int, 
        h_frac=h_mic_delay, 
        length=nfft, 
        origin=origin,
        strict=False
    )[:, 0 : nfft]  # (1, nfft)

    h_total = h_total[0, :].numpy()  # (nfft,)
    
    return h_total


def compute_bn(n: int, kr: np.ndarray, sphere_type: str) -> np.ndarray:
    """Compute mode strength bn(kr) for an incident plane wave on a sphere.

    Reference:
    [1] Rafaely, B. (2015). Fundamentals of Spherical Array Processing. Springer.

    Args:
        n (int)
        kr (np.ndarray), (nfft/2+1,)
        sphere_type (str), "open" | rigid

    Returns:
        bn (np.ndarray), (nfft/2+1,)
    """
    
    # Eq. 4.4 of [1]
    if sphere_type == 'open':
        bn = 4 * np.pi * (1j ** n) * spherical_jn(n, kr)

    # Eq. 4.5 of [1]
    elif sphere_type == 'rigid':
        with np.errstate(divide="ignore", invalid="ignore"):
            sub = (spherical_jn(n, kr, True) / spherical_hn2(n, kr, True)) * spherical_hn2(n, kr)
        sub[0] = 0.
        bn = 4 * np.pi * (1j ** n) * (spherical_jn(n, kr) - sub)
            
    else:
        raise ValueError('sphere_type Not implemented.')

    return bn


def spherical_hn2(n: int, x: np.ndarray, derivative=False) -> np.ndarray:
    """Spherical Hankel function of the second kind:
    hn2(x) = jn(x) - i * yn(x).
    """

    return spherical_jn(n, x, derivative) - 1j * spherical_yn(n, x, derivative)


def visualize(h):
    fig, axes = plt.subplots(2, 1, sharex=False)
    axes[0].stem(h)
    axes[1].stem(h[110:150])
    out_path = "_zz.pdf"
    plt.savefig(out_path)
    print(f"Write out to {out_path}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--sphere_type', type=str, required=True, choices=["open", "rigid"])
    parser.add_argument('--out_path', type=str, required=True)
    
    args = parser.parse_args()
    compute_spatial_impulse_responses(args)
    