import numpy as np
import matplotlib.pyplot as plt


def fractional_delay(x, delay_samples):
    r"""Fractional delay with Whittaker–Shannon interpolation formula. 
    Ref: https://tomroelandts.com/articles/how-to-create-a-fractional-delay-filter

    Args:
        x: np.array (1D), input signal
        delay_samples: float >= 0., e.g., 3.3

    Outputs:
        y: np.array (1D), delayed signal
    """
    integer = int(delay_samples)
    fraction = delay_samples % 1

    x = np.concatenate((np.zeros(integer), x), axis=0)[0 : len(x)]

    N = 21     # Filter length.
    n = np.arange(N)

    # Compute sinc filter.
    h = np.sinc(n - (N - 1) / 2 - fraction)
     
    # Multiply sinc filter by window
    h *= np.blackman(N)
     
    # Normalize to get unity gain.
    h /= np.sum(h)
    
    y = np.convolve(x, h, mode='same')

    return y


def add():
    t = np.arange(40)
    x = np.sin(t)
    # plt.plot(x)
    # plt.savefig('_zz.pdf')

    # tau = 0.3  # Fractional delay [samples].
    tau = 1
    N = 21     # Filter length.
    n = np.arange(N)
     
    # Compute sinc filter.
    h = np.sinc(n - (N - 1) / 2 - tau)
     
    # Multiply sinc filter by window
    h *= np.blackman(N)
     
    # Normalize to get unity gain.
    h /= np.sum(h)

    y = np.convolve(x, h, mode='same')
    # y = np.convolve(x, h)

    # plt.plot(x)
    # plt.plot(y)
    # plt.savefig('_zz.pdf')

    plt.plot(h)
    plt.savefig('_zz.pdf')

    # a1 = np.array([1,2,3,4,5])
    # a2 = np.array([1,2,3])
    # np.convolve(a1, a2, mode='same')

    from IPython import embed; embed(using=False); os._exit(0)


def add2():

    t = np.arange(40)
    x = np.sin(t)

    delay_samples = 3.3

    y = fractional_delay(x, delay_samples)

    plt.plot(x)
    plt.plot(y)
    plt.savefig('_zz.pdf')


if __name__ == '__main__':

    # add()
    add2()