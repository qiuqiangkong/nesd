import matplotlib.pyplot as plt
import numpy as np
import pudb
import scipy.signal as scysignal
import scipy.special as scyspecial
import h5py

c = 343 # sound speed
radius = 0.042 # radius of the eigenmic baffle
fs = 24000
nfft=2048

def get_rigid_sph_array(src_pos, mic_pos, n_points, order=30):
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
    f = np.linspace(0, fs//2, n_points//2 + 1)
    kr = 2 * np.pi * f / c * radius

    b_n = np.zeros((order+1, len(f)), dtype=np.complex128)
    for n in range(order+1):
        b_n[n] = mode_strength(n=n, kr=kr, sphere_type='rigid')
    temp = b_n
    temp[:, 0] = np.abs(temp[:, 0])
    temp[:, -1] = np.abs(temp[:, -1])
    temp = np.concatenate((temp, temp[:,-2:0:-1].conj()), axis=1)
    b_nt = np.fft.fftshift(np.fft.ifft(temp, axis=1).real)

    # Compute angular-dependent part of the microphone responses
    # unit vectors of DOAs and microphones
    N_src = len(src_pos)
    N_mic = len(mic_pos)
    h_mic = np.zeros((n_points, N_mic, N_src))
    H_mic = np.zeros((n_points//2+1, N_mic, N_src), dtype=np.complex128)
    for i in range(N_src):
        cosAngle = np.dot(
            mic_pos / radius, 
            src_pos[i,:] / np.linalg.norm(src_pos[i,:]))
        P = np.zeros((order+1, N_mic))
        for n in range(order+1):
            Pn = scyspecial.lpmv(0, n, cosAngle)
            P[n, :] = (2*n+1) / (4 * np.pi) * Pn
        
        h_mic[:,:,i] = b_nt.T @ P
        H_mic[:,:,i] = b_n.T @ P

    from IPython import embed; embed(using=False); os._exit(0)
    return h_mic.transpose(2,1,0), H_mic.transpose(2,1,0)


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
            yi = 1j * scyspecial.spherical_yn(n, z, derivative)
        return scyspecial.spherical_jn(n, z, derivative) - yi

    kr = np.asarray(kr)
    if sphere_type == 'open':
        b_n = 4*np.pi*1j**n * scyspecial.spherical_jn(n, kr)
    elif sphere_type == 'rigid':
        b_n = 4*np.pi*1j**n * (scyspecial.spherical_jn(n, kr) -
                            (scyspecial.spherical_jn(n, kr, True) /
                            spherical_hn2(n, kr, True)) *
                            spherical_hn2(n, kr))
    else:
        raise ValueError('sphere_type Not implemented.')
    
    idx_kr0 = np.where(kr==0)[0]
    idx_nan = np.where(np.isnan(b_n))[0]
    b_n[idx_nan] = 0
    if n == 0:
        b_n[idx_kr0] = 4*np.pi
    else:
        b_n[idx_kr0] = 0

    return b_n


def get_rigid_sph_array2(angle, n_points, order=30):
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
    f = np.linspace(0, fs//2, n_points//2 + 1)
    kr = 2 * np.pi * f / c * radius

    b_n = np.zeros((order+1, len(f)), dtype=np.complex128)
    for n in range(order+1):
        b_n[n] = mode_strength(n=n, kr=kr, sphere_type='rigid')
    temp = b_n
    temp[:, 0] = np.abs(temp[:, 0])
    temp[:, -1] = np.abs(temp[:, -1])
    temp = np.concatenate((temp, temp[:,-2:0:-1].conj()), axis=1)
    b_nt = np.fft.fftshift(np.fft.ifft(temp, axis=1).real)

    P = np.zeros(order+1)
    for n in range(order+1):
        Pn = scyspecial.lpmv(0, n, np.cos(angle))
        P[n] = (2*n+1) / (4 * np.pi) * Pn
        
    h_mic = b_nt.T @ P
    H_mic = b_n.T @ P

    return h_mic, H_mic

    # from IPython import embed; embed(using=False); os._exit(0)
    # return h_mic.transpose(2,1,0), H_mic.transpose(2,1,0)

'''
def get_rigid_sph_array3(angle_mat, n_points, order=30):
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
    f = np.linspace(0, fs//2, n_points//2 + 1)
    kr = 2 * np.pi * f / c * radius

    b_n = np.zeros((order+1, len(f)), dtype=np.complex128)
    for n in range(order+1):
        b_n[n] = mode_strength(n=n, kr=kr, sphere_type='rigid')
    temp = b_n
    temp[:, 0] = np.abs(temp[:, 0])
    temp[:, -1] = np.abs(temp[:, -1])
    temp = np.concatenate((temp, temp[:,-2:0:-1].conj()), axis=1)
    b_nt = np.fft.fftshift(np.fft.ifft(temp, axis=1).real)

    # P = np.zeros(order+1)
    # for n in range(order+1):
    #     Pn = scyspecial.lpmv(0, n, np.cos(angle))
    #     P[n] = (2*n+1) / (4 * np.pi) * Pn
        
    # h_mic = b_nt.T @ P
    # H_mic = b_n.T @ P


    h_mic = np.zeros((azi_grids, col_grids, n_points))
    H_mic = np.zeros((azi_grids, col_grids, n_points//2+1), dtype=np.complex128)
    P = np.zeros((azi_grids, col_grids, order+1))

    for i in range(azi_grids):
        for j in range(col_grids):
            for n in range(order+1):
                Pn = scyspecial.lpmv(0, n, np.cos(angle_mat[i, j]))
                P[i, j, n] = (2*n+1) / (4 * np.pi) * Pn

    # h_mic[:,:,i] = b_nt.T @ P
    # H_mic[:,:,i] = b_n.T @ P
    h_mic = P @ b_nt
    H_mic = P @ b_n

    return h_mic, H_mic

    # from IPython import embed; embed(using=False); os._exit(0)
    # return h_mic.transpose(2,1,0), H_mic.transpose(2,1,0)
'''

# Yin's main
def add():
    src_pos = np.array([[10, 1, 1]])
    mic_pos_sph = [45/180*np.pi, 35/180*np.pi, 0.042]
    mic_pos_cart = [mic_pos_sph[2] * np.sin(mic_pos_sph[1]) * np.cos(mic_pos_sph[0]),
                    mic_pos_sph[2] * np.sin(mic_pos_sph[1]) * np.sin(mic_pos_sph[0]),
                    mic_pos_sph[2] * np.cos(mic_pos_sph[1])]
    mic_pos = np.array([mic_pos_cart])
    h_mic, H_mic = get_rigid_sph_array(src_pos=src_pos, mic_pos=mic_pos, n_points=nfft)

    fig, (ax_T, ax_F) = plt.subplots(2, 1)
    ax_T.plot(h_mic[0, 0])
    ax_T.set_title('h_mic')
    ax_F.plot(np.abs(H_mic[0, 0]))
    ax_F.set_title('H_mic')
    fig.tight_layout()
    fig.show()
    fig.savefig('./figure.png', bbox_inches='tight')
    from IPython import embed; embed(using=False); os._exit(0)

    # # To convolve an audio signal:
    # num_src = 1
    # num_mic = 1
    # num_points=nfft
    # multichannel_signals = np.random.randn(num_src, num_mic, num_points)
    # scysignal.fftconvolve(multichannel_signals, h_mic, axes=-1)[:,:,:num_points].sum(axis=0)


# kqq's main, 
def add2():
    # angle = np.deg2rad(0)

    # h, H = get_rigid_sph_array2(angle, n_points=nfft, order=30)

    hs = []
    Hs = []

    for angle in range(0, 360, 1):

        print(angle)

        h, H = get_rigid_sph_array2(np.deg2rad(angle), n_points=nfft, order=30)
        Hs.append(H)
        hs.append(h)

    Hs = np.stack(Hs, axis=0)
    hs = np.stack(hs, axis=0)
    fig, ax = plt.subplots(1, 1, sharex=True)
    ax.matshow(np.abs(Hs), origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=2)
    plt.savefig("_zz.pdf")

    plt.figure()
    plt.plot(np.abs(Hs[3]))
    plt.savefig("_zz2.pdf")

    plt.figure()
    plt.stem(np.abs(hs[0])[1000:1200])
    plt.savefig("_zz3.pdf")


    with h5py.File('rigid_eig_ir.h5', 'w') as hf:   # 'a' for append
        hf.create_dataset('h', data=hs, dtype=np.float32)

    from IPython import embed; embed(using=False); os._exit(0)


# def add3():

#     azi, col = np.meshgrid(np.linspace(0,359,360), np.linspace(0, 179 ,180))


if __name__ == '__main__':

    add2()