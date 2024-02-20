from nesd.utils import *
import random
import time
import copy
import soundfile
import math
import h5py
import torch
# import scipy
from scipy.signal import fftconvolve
from scipy.spatial.transform import Rotation

import pyroomacoustics as pra
from nesd.models.base import *


def add():

    a1 = ["asdfasdfasdfasdf"] * 10000000

    t1 = time.time()
    # b1 = random.choice(a1)
    randint()
    print(time.time() - t1)


def add2():

    for _ in range(100):
        x = random.randint(0, 2)
        print(x)


def scale_to_db(scale):
    db = 20 * np.log10(scale)
    return db


def db_to_scale(db):
    scale = 10 ** (db / 20.)
    return scale


def add3():
    # db = random.uniform(-12, 12)
    db = 6
    # 10 * np.log10(10)

    scale = db_to_scale(db)
    print(scale)
    print(scale_to_db(scale))


def add4():

    t1 = time.time()

    corners = np.array([
        [0, 0], 
        [0, 3], 
        [4, 3], 
        [4, 0]
    ]).T
    # shape: (2, 4)

    room = pra.Room.from_corners(
        corners=corners,
        max_order=3,
    )
    print("a0", time.time() - t1)

    t1 = time.time()
    room.extrude(height=2)
    print("a1", time.time() - t1)

    # t1 = time.time()
    # a1 = copy.deepcopy(room)
    # print("b1", time.time() - t1)

    t1 = time.time()
    # for e in np.arange(0, 1, 0.1):
    #     room.add_source([1+e, 1+e, 1+e])
    # room.add_source([0.5, 1.5, 1])
    room.add_source([1, 1, 1])
    print("a2", time.time() - t1)

    t1 = time.time()
    # room.add_microphone([1.5, 0.5, 1])
    room.add_microphone([1.5, 1.501, 1.502])
    # room.add_microphone([3.1, 3.1, 1.6])
    # room.add_microphone_array(mic_array=np.array([[1, 1, 1], [1.2, 1.2, 1.2]]).T)

    # for e in np.arange(0, 1, 0.1):
    #     room.add_microphone([1+e, 1+e, 1+e])

    print("a3", time.time() - t1)

    t1 = time.time()
    room.image_source_model()
    images = room.sources[0].images.T
    print("a4", time.time() - t1)

    from IPython import embed; embed(using=False); os._exit(0)


def add5():

    t1 = time.time()
    x = np.ones(1000)
    y = np.ones(48000)
    # x = np.array([0, 1, 2, 3, 4, 5, 6])
    # y = np.array([0, 1, 2, 3, 4])
    for _ in range(100):
        z = fftconvolve(in1=x, in2=y, mode="same") 
    # np.convolve(x, y, mode='full')
    print(time.time() - t1)

    from IPython import embed; embed(using=False); os._exit(0)


def add6():
    h = fractional_delay_filter(-1.7)

    '''
    audio_path = "resources/p226_001.wav"
    x, _ = librosa.load(path=audio_path, sr=24000, mono=True)

    y = fftconvolve(in1=x, in2=h, mode="same")

    print(np.mean(x**2))
    print(np.mean(y**2))

    soundfile.write(file="_zz.wav", data=y, samplerate=24000)
    '''

    n = np.arange(20)
    x = np.sin(2 * math.pi * n / 20)
    # x = np.arange(500)
    y = fftconvolve(in1=x, in2=h, mode="same")

    fig, axs = plt.subplots(2,1, sharex=True)
    axs[0].stem(x)
    axs[1].stem(y)
    plt.savefig("_zz.pdf")

    from IPython import embed; embed(using=False); os._exit(0)


def add7():

    x = np.array([1, 0, 0])
    y = np.array([1, 1, 0])

    print(np.rad2deg(get_incident_angle(x, y)))


def add8():

    with h5py.File('rigid_eig_ir.h5', 'r') as hf:
        h1 = hf["h"][:]

    with h5py.File('rigid_sphere_ir.h5', 'r') as hf:
        h2 = hf["h"][:]

    from IPython import embed; embed(using=False); os._exit(0)


def add9():

    x = [0, 1, 2, 3]
    y = np.fft.fftshift(x)

    a1 = np.ones(9)
    b1 = np.fft.irfft(a1)
    c1 = np.fft.fftshift(b1)

    d1 = np.pad(array=c1, pad_width=((0, 1)), constant_values=0.)

    from IPython import embed; embed(using=False); os._exit(0)


def add10():

    import librosa
    from scipy.signal import fftconvolve
    import soundfile

    with h5py.File('rigid_eig_ir.h5', 'r') as hf:
        hs = hf["h"][:]

    audio, _ = librosa.load(path="./resources/p226_001.wav", sr=fs)
    y1 = fftconvolve(in1=audio, in2=hs[0])
    y2 = fftconvolve(in1=audio, in2=hs[180])

    soundfile.write(file="_zz.wav", data=audio, samplerate=fs)
    soundfile.write(file="_zz1.wav", data=y1, samplerate=fs)
    soundfile.write(file="_zz2.wav", data=y2, samplerate=fs)

    from IPython import embed; embed(using=False); os._exit(0)


def get_alpha_beta_gamma(azimuth, elevation):
    # Rotate along x (roll, gamma) -> along y (pitch, beta) -> along z (yaw, alpha)
    x, y, z = sph2cart(azimuth=azimuth, elevation=elevation, r=1.)

    alpha = 0.
    beta = math.atan2(x, z)
    gamma = - math.asin(y)

    return alpha, beta, gamma


def add11():

    t1 = time.time()
    azi = math.pi / 4
    ele = math.pi / 6
    ele = 0

    a1 = np.array([1, 0, 0])

    b1 = sph2cart(azimuth=azi, elevation=ele, r=1.)

    print("a0", time.time() - t1)
    t1 = time.time()

    c1 = (a1 + b1) / 2
    c1 = c1 / np.linalg.norm(c1)

    # angle = get_included_angle(a1, c1)
    c1 *= math.pi

    print("a1", time.time() - t1)
    t1 = time.time()

    r = Rotation.from_rotvec(rotvec=c1)

    print("a2", time.time() - t1)
    t1 = time.time()

    y = r.apply(np.array([1, 0, 0]))

    print("a3", time.time() - t1)
    t1 = time.time()


    
    from IPython import embed; embed(using=False); os._exit(0)


def add12():

    # r = Rotation.from_rotvec(np.array([0, math.pi / 2 / math.sqrt(2), math.pi / 2 / math.sqrt(2)]))

    r = Rotation.from_rotvec(np.array([math.pi / 2 / math.sqrt(2), math.pi / 2 / math.sqrt(2), 0]))

    y = r.apply(np.array([1, 0, 0]))

    from IPython import embed; embed(using=False); os._exit(0)


def random_direction_from_spherical_cap(direction, theta):

    cap_direction = np.array([0, 0, 1])

    rot_axis = (direction + cap_direction) / 2
    rot_axis /= np.linalg.norm(rot_axis)

    rotvec = rot_axis * math.pi

    r = Rotation.from_rotvec(rotvec=rotvec)

    randomdirection_from_cap = random_direction(min_azimuth=-math.pi, 
        max_azimuth=math.pi, 
        min_elevation=math.pi / 2 - theta,
        max_elevation=math.pi / 2,
    )

    output_direction = r.apply(randomdirection_from_cap)

    return output_direction


def add13():

    t1 = time.time()
    direction = random_direction()

    ys = []
    zs = []

    for _ in range(10):
        y = random_direction_from_spherical_cap(direction=direction, theta=np.deg2rad(20))
        ys.append(y)

        # z = 

    print(time.time() - t1)
    from IPython import embed; embed(using=False); os._exit(0)


def add14():

    tmp = []
    for _ in range(100):
        a1 = random_direction()
        azi, ele, r = cart2sph(vector=a1)
        c1 = sph2cart(azi, ele, r)
        tmp.append(np.sum(np.abs(a1 - c1)))

    print(np.max(tmp))
    from IPython import embed; embed(using=False); os._exit(0)
    

def add15():

    # base_direction = 

    a1 = np.array([0, 0, 1])

    for _ in range(100):
        b1 = random_direction()

        rot_vector = np.cross(a=a1, b=b1)

        rot_angle = get_included_angle(a1, b1)
        rot_vector = rot_vector / np.linalg.norm(rot_vector) * rot_angle

        r = Rotation.from_rotvec(rotvec=rot_vector)

        y = r.apply(a1)

        print(np.abs(y - b1))

    from IPython import embed; embed(using=False); os._exit(0)



def add16():

    a1 = np.array([0, 0, 1])

    for _ in range(100):
        b1 = random_direction()

        ys = []
        tmp = []
        X = np.zeros((360, 180))
        for i in range(20000):
            y = random_direction_within_region(direction=b1, theta=np.deg2rad(30))
            # print(np.linalg.norm(y - b1))
            ys.append(y)
            azi, ele, r = cart2sph(y)
            # tmp.append()
            # try:
            X[int(np.rad2deg(azi)) + 180, int(np.rad2deg(ele) + 90)] = 1
            # except:
                # from IPython import embed; embed(using=False); os._exit(0)

        plt.matshow(X.T, origin='lower', aspect='auto', cmap='jet')
        plt.savefig("_zz.pdf")

        from IPython import embed; embed(using=False); os._exit(0)


def add17():

    a1 = np.array([1, 0, 0])
    b1 = np.array([0, 1, 0])
    print(np.cross(a1, b1))


def add18():

    ys = []
    for x in np.arange(-0.05, 0.05, 0.002):
        ys.append(triangle_function(x))
    plt.plot(ys)
    plt.savefig("_zz.pdf")
    from IPython import embed; embed(using=False); os._exit(0)





def add19():

    negative_directions = [np.array([1, 0, 0]), np.array([0, 1, 0])]

    X = np.zeros((360, 180))

    for i in range(20000):
        y=random_negative_direction(negative_directions, theta=0.5)

        azi, ele, r = cart2sph(y)
        X[int(np.rad2deg(azi)) + 180, int(np.rad2deg(ele) + 90)] = 1
        
    plt.matshow(X.T, origin='lower', aspect='auto', cmap='jet')
    plt.savefig("_zz.pdf")
    


def add20():

    x = np.zeros(1000)

    for _ in range(20000):
        # dist = random_positive_distance(source_distance=3., r=0.1)
        dist = random_negative_distance(source_distances=[3., 4.5], r=0.1, max_dist=10)
        x[int(dist * 100)] = 1

    plt.plot(x)
    plt.savefig("_zz.pdf")


def sample_source_indexes(sources_num, max_positive_rays):

    if sources_num <= max_positive_rays:
        return range(sources_num)
    else:
        return random.sample(range(sources_num), k=max_positive_rays)

def add21():

    print(sample_source_indexes(5, 10))

# test PositionEncoder
def add22():

    pos_encoder = PositionEncoder(size=5)
    vector = torch.Tensor(np.arange(0, 20, 0.01))[:, None]
    pos_emb = pos_encoder(vector)
    plt.matshow(pos_emb.T, origin='lower', aspect='auto', vmin=-1., vmax=1., cmap='jet')
    plt.savefig("_zz.pdf")
    from IPython import embed; embed(using=False); os._exit(0)
    

def add23():

    a1 = torch.Tensor([1,2,3])[None, :, None]
    y = torch.repeat_interleave(a1, repeats=4, dim=2)
    print(y)





def add24():
    a1 = InfiniteSampler()
    for e in a1:
        print(e)


if __name__ == "__main__":

    add24()