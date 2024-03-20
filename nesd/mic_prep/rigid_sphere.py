import matplotlib.pyplot as plt
import numpy as np
import scipy
import pickle
import argparse
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from pathlib import Path

from nesd.utils import fractional_delay_filter


SPEED_OF_SOUND = 343.
RADIUS = 0.042  # Radius of the Eigenmic baffle
SAMPLE_RATE = 24000
NFFT = 2048


def calcualte_spaital_impulse_responses(args):
    
    mic_spatial_irs_path = args.mic_spatial_irs_path

    spatial_irs = {}

    # Inclined angle between mic_orientation and mic_to_source.
    for deg in range(0, 181, 1):

        print("Inclined angle degree: {}".format(deg))

        h_rigid, H_rigid = get_rigid_sph_array(
            angle=np.deg2rad(deg), 
            nfft=NFFT, 
            radius=RADIUS,
            fs=SAMPLE_RATE,
            c=SPEED_OF_SOUND,
            order=30
        )

        # Get delay impulse response.
        # distance = - RADIUS * np.cos(np.deg2rad(deg))
        # delayed_samples = (distance / SPEED_OF_SOUND) * SAMPLE_RATE
        # h_delay = fractional_delay_filter(delayed_samples)

        # Total impulse response.
        # h_total = fftconvolve(in1=h_rigid, in2=h_delay, mode="full")

        # spatial_irs.append(h)
        spatial_irs[deg] = h_rigid

    Path(mic_spatial_irs_path).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(spatial_irs, open(mic_spatial_irs_path, "wb"))
    print("Write out to {}".format(mic_spatial_irs_path))
    

def get_rigid_sph_array(angle, nfft, radius, fs, c, order):
    """Rigid spherical array response simulation.

    Parameters
    ----------
    src_pos : (num_src, 3) array_like
        Source positions in meters
    mic_pos : (num_mic, 3) array_like
        Microphone positions in meters
    n_points : even int
        Points of filter.
    order : int, optional
        Order of expension term, by default 30
        The expansion is limited to 30 terms which provide 
            negligible modeling error up to 20 kHz.

    Returns
    -------
    h_mic: (num_src, num_mic, n_points) array_like
        Spherical array response.
    """
    # Compute the frequency-dependent part of the microphone responses
    f = np.linspace(0, fs // 2, nfft // 2 + 1)
    kr = 2 * np.pi * f / c * radius

    b_n = np.zeros((order + 1, len(f)), dtype=np.complex128)
    
    for n in range(order + 1):
        b_n[n] = mode_strength(n=n, kr=kr, sphere_type='rigid')
        # b_n[n] = mode_strength(n=n, kr=kr, sphere_type='open')

    b_nt = np.fft.irfft(b_n)    # (order + 1, nfft)
    
    P = np.zeros(order + 1)
    for n in range(order + 1):
        Pn = scipy.special.lpmv(0, n, np.cos(angle))
        P[n] = (2 * n + 1) / (4 * np.pi) * Pn
    # P: (orders + 1)
        
    h_mic = b_nt.T @ P  # (nfft,)
    H_mic = b_n.T @ P   # (nfft // 2 + 1)

    # Shift the origin to the center
    h_mic = np.fft.fftshift(h_mic)

    h_mic = np.pad(array=h_mic, pad_width=((0, 1)), constant_values=0.)

    # print(np.sum(h_mic))

    # Multiply sinc filter by window.
    # h_mic *= np.blackman(len(h_mic))
    
    # Normalize to get unity gain.
    # h_mic /= np.sum(h_mic)

    # fig, axes = plt.subplots(2, 1, sharex=True)
    # axes[0].plot(np.angle(H_mic))
    # axes[1].plot(np.abs(H_mic))
    # axes[0].set_ylim(-10, 10)
    # axes[1].set_ylim(-3, 3)
    # plt.savefig("_zz.pdf")

    # M = 100 
    # plt.stem(h_mic[1024 - M : 1024 + M])
    # # plt.stem(h_mic)
    # plt.savefig('_zz.pdf') 

    # from IPython import embed; embed(using=False); os._exit(0)

    return h_mic, H_mic


def mode_strength(n, kr, sphere_type='rigid'):
    """Mode strength b_n(kr) for an incident plane wave on sphere.

    Parameters
    ----------
    n : int
        Degree.
    kr : array_like
        kr vector, product of wavenumber k and radius r_0.
    sphere_type : 'rigid' or 'open'

    Returns
    -------
    b_n : array_like
        Mode strength b_n(kr).

    References
    ----------
    Rafaely, B. (2015). Fundamentals of Spherical Array Processing. Springer.
    eq. (4.4) and (4.5).
    """

    np.seterr(divide='ignore', invalid='ignore')

    kr = np.asarray(kr)

    if sphere_type == 'open':
        b_n = 4 * np.pi * (1j ** n) * scipy.special.spherical_jn(n, kr)

    elif sphere_type == 'rigid':
        b_n = 4 * np.pi * (1j ** n) * (scipy.special.spherical_jn(n, kr) -
                            (scipy.special.spherical_jn(n, kr, True) /
                            spherical_hn2(n, kr, True)) *
                            spherical_hn2(n, kr))
    else:
        raise ValueError('sphere_type Not implemented.')
    
    idx_kr0 = np.where(kr == 0)[0]
    idx_nan = np.where(np.isnan(b_n))[0]
    b_n[idx_nan] = 0
    if n == 0:
        b_n[idx_kr0] = 4 * np.pi
    else:
        b_n[idx_kr0] = 0

    return b_n

def spherical_hn2(n, z, derivative=False):
        """Spherical Hankel function of the second kind.

        Parameters
        ----------
        n : int, array_like
            Order of the spherical Hankel function (n >= 0).
        z : complex or float, array_like
            Argument of the spherical Hankel function.
        derivative : bool, optional
            If True, the value of the derivative (rather than the function
            itself) is returned.

        Returns
        -------
        hn2 : array_like


        References
        ----------
        http://mathworld.wolfram.com/SphericalHankelFunctionoftheSecondKind.html
        """
        with np.errstate(invalid='ignore'):
            yi = 1j * scipy.special.spherical_yn(n, z, derivative)
        return scipy.special.spherical_jn(n, z, derivative) - yi


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mic_spatial_irs_path', type=str, required=True)
    
    args = parser.parse_args()

    calcualte_spaital_impulse_responses(args)
    