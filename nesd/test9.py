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


class InfiniteRandomSampler:
    def __init__(self):
        self.indexes = list(range(10))
        
    def __iter__(self):

        pointer = 0

        while True:

            if pointer == len(self.indexes):
                pointer = 0
                random.shuffle(self.indexes)

            index = self.indexes[pointer]
            pointer += 1

            yield index

            


def add24():
    a1 = InfiniteRandomSampler()
    for e in a1:
        print(e)


def add25():

    direction = np.array([1, 0, 0])
    theta = 0
    y = random_positive_direction(direction, theta)
    from IPython import embed; embed(using=False); os._exit(0)


def add26():

    directions = [np.array([1, 0, 0]), np.array([1, 1, 1])]
    tmp = np.zeros((360, 180))

    for _ in range(100000):
        dirs = random_negative_direction(directions, theta = np.deg2rad(20))
        azi, ele, r = cart2sph(dirs)
        tmp[int(np.rad2deg(azi)) + 180, int(np.rad2deg(ele)) + 90] = 1

    # fig, axs = plt.subplots(2,2, sharex=True)
    plt.matshow(tmp.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    plt.savefig("_zz.pdf")
    from IPython import embed; embed(using=False); os._exit(0)


def add27():
    pkl_path = "/home/qiuqiangkong/workspaces/nesd/mic_spatial_irs/rigid_sphere.pkl"
    # pkl_path = "/home/qiuqiangkong/workspaces/nesd/mic_spatial_irs/open_sphere.pkl"
    a1 = pickle.load(open(pkl_path, "rb"))

    N = 2049
    L = 20
    fig, axs = plt.subplots(4, 1, sharex=True)
    axs[0].stem(a1[0][N//2 - L : N//2 + L + 1])
    axs[1].stem(a1[90][N//2 - L : N//2 + L + 1])
    axs[2].stem(a1[180][N//2 - L : N//2 + L + 1])
    plt.savefig("_zz.pdf")

    from IPython import embed; embed(using=False); os._exit(0)


def add28():

    a1 = np.array([1, 0.1, 0.2, 0.3, 0.4, 0.5])
    a1 = np.fft.fftshift(a1)
    a1 = np.pad(array=a1, pad_width=((0, 1)), constant_values=0.)
    from IPython import embed; embed(using=False); os._exit(0)


def add29():

    azi_deg = -135
    ele_deg = 35
    r = 0.042

    pos = sph2cart(
        azimuth=np.deg2rad(azi_deg), 
        elevation=np.deg2rad(ele_deg),
        r=r,
    )

    pos = pos + np.array([3,3,1])

    print(pos)


def add30():

    from nesd.data.engine import ImageSourceEngine

    environment = {
     'x0': -2.,
     'x1': 2.,
     'y0': -2.,
     'y1': 2.,
     'z0': 0.0,
     'z1': 4.0,
     'ndim': 3,
     'room_length': 4.,
     'room_width': 4.,
     'room_height': 4.0,
     'max_room_distance': 10,
     'room_margin': 0.2}

    engine = ImageSourceEngine(
        environment=environment, 
        source_positions=np.array([[1., 1., 1.]]),
        mic_position=np.array([0., 0., 2.]), 
        mic_orientation=np.array([1., 1., 1.]),
        mic_spatial_irs=None,
        image_source_order=3,
        speed_of_sound=343.,
        sample_rate=24000.,
        compute_direct_ir_only=False)

    engine.compute_spatial_ir()


def add31():

    from nesd.utils import fractional_delay_filter
    h = fractional_delay_filter(-2.3)
    
    x1 = np.zeros(200)
    x1[10] = 1
    y1 = fftconvolve(in1=x1, in2=h, mode="same")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, sharex=False)
    axs[0].stem(h)
    axs[1].stem(y1[0:20])
    plt.savefig("_zz.pdf")

    from IPython import embed; embed(using=False); os._exit(0)


def add32():
    audio_path = "/datasets/tau-srir/TAU-SNoise_DB/01_bomb_center/ambience_tetra_24k_edited.wav"
    audio, _ = librosa.load(path=audio_path, sr=24000, mono=False)
    soundfile.write(file="_uu2.wav", data=audio[:, 0 : 10 * 24000].T, samplerate=24000)


def add33():
    soundfile.write(file="_uu3.wav", data=np.zeros((24000 * 10, 4)), samplerate=24000)


def load_directions_from_pred(
    audio_path="/datasets/dcase2023/task3/mic_dev/dev-test-sony/fold4_room23_mix001.wav",
    pred_csv_path="/home/qiuqiangkong/workspaces/nesd/results/dcase2024_task3/pred_csvs/fold4_room23_mix001.csv"):

    import pandas as pd
    df = pd.read_csv(pred_csv_path, sep=',', header=None)
    frame_indexes = df[0].values
    class_indexes = df[1].values
    azis = df[3].values
    eles = df[4].values
    distances = df[5].values

    azis = np.deg2rad(azis)
    eles = np.deg2rad(eles)

    buffer = []

    for n in range(len(frame_indexes)):

        tup = (frame_indexes[n], azis[n], eles[n])

        finish = False

        for tups in buffer:
            if is_continus_event(tup, tups[-1]):
                tups.append(tup)
                finish = True
        
        if finish is False:
            buffer.append([tup])

    frames_per_sec = 10
    segment_frames = 20

    sample_rate = 24000
    
    # audio = np.zeros((4, sample_rate * 60))
    audio, fs = librosa.load(path=audio_path, sr=sample_rate, mono=False)

    list_data = []

    #
    for tups in buffer:

        segment = None
        look_at_directions = None

        first_index = tups[0][0]
        last_index = tups[-1][0]

        seg_bgn_index = (first_index // 20) * 20
        seg_end_index = int(np.ceil(last_index / 20) * 20)

        seg_bgn_sample = seg_bgn_index * (sample_rate // 10)
        seg_end_sample = seg_end_index * (sample_rate // 10)

        clip = audio[:, seg_bgn_sample : seg_end_sample]
        _azis = np.ones(seg_end_index - seg_bgn_index + 1) * math.nan
        _eles = np.ones(seg_end_index - seg_bgn_index + 1) * math.nan
        _masks = np.zeros(seg_end_index - seg_bgn_index + 1) 

        for tup in tups:
            frame_index, azi, ele = tup
            _azis[frame_index - seg_bgn_index] = azi
            _eles[frame_index - seg_bgn_index] = ele

        for i in range(len(_azis)):
            if not math.isnan(_azis[i]):
                _masks[i] = 1

        # Remove nan
        for i in range(len(_azis)):
            if math.isnan(_azis[i]):
                for j in range(i, -1, -1):
                    if not math.isnan(_azis[j]):
                        _azis[i] = _azis[j]
                        _eles[i] = _eles[j]
                        break

            if math.isnan(_azis[i]):
                for j in range(i, len(_azis)):
                    if not math.isnan(_azis[j]):
                        _azis[i] = _azis[j]
                        _eles[i] = _eles[j]
                        break

        look_at_directions = sph2cart(azimuth=_azis, elevation=_eles, r=1.)
        look_at_directions = np.repeat(look_at_directions, repeats=10, axis=0)[0 : -9]
        masks = np.repeat(_masks, repeats=10, axis=0)[0 : -9]

        data = {
            "begin_second": seg_bgn_index * 10,
            "audio": clip,
            "look_at_directions": look_at_directions,
            "mask": masks
        }
        # from IPython import embed; embed(using=False); os._exit(0)

        list_data.append(data)

    return list_data


def is_continus_event(tup, prev_tup):

    if 0 < tup[0] - prev_tup[0] < 10:
        curr_dir = sph2cart(azimuth=tup[1], elevation=tup[2], r=1.)
        prev_dir = sph2cart(azimuth=prev_tup[1], elevation=prev_tup[2], r=1.)
        if np.rad2deg(get_included_angle(curr_dir, prev_dir)) < 10:
            return True
        
    return False


if __name__ == "__main__":

    load_directions_from_pred()