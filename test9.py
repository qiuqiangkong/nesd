from nesd.datasets.aa import *
from nesd.utils import *
import torch.nn.functional as F
# from sim.test8 import *
import matplotlib.pyplot as plt
import torchaudio
import time
from torch import Tensor, LongTensor
import torch
from torch import Tensor
import scipy
import matplotlib.pyplot as plt
import math
import soundfile
from pathlib import Path
from torchvision.io import read_video
from einops import rearrange


def add():

    cart = np.array([1, 2, 3])
    a1 = cart2sph(cart)
    print(a1)

    cart = np.array([[1, 2, 3], [1, 2, 3]])
    a2 = cart2sph(cart)
    print(a2)

    b1 = sph2cart(a1)
    print(b1)

    b2 = sph2cart(a2)
    print(b2)

    c1 = normalize(cart)
    print(c1)

    c2 = normalize(b2)
    print(c2)


def add2():

    # N = 1000
    device = "cuda"
    x = torch.zeros(1, 1, 500, 500, 200).to(device)
    w = torch.zeros(1, 1, 3, 3, 3).to(device)

    for i in range(100000):
        if i%100 == 0:
            print(i)
        x = F.conv3d(x, w, padding=1)

    from IPython import embed; embed(using=False); os._exit(0)


def add3():

    a1 = sample_direction(min_ele=0, max_ele=0)
    print(a1)


def add4():

    rot = sample_rotation_matrix()
    v = np.array([1, 0, 0])
    y = rot @ v
    print(y)
    from scipy.spatial.transform import Rotation as R
    

def test_shoebox():
    for _ in range(10):
        print("---")
        env = Shoebox()
        print(env.x_bnd, env.y_bnd, env.height)
        print(env.sample_pos())




def sample_rotation_matrix_(deg) -> np.ndarray:
    front = sample_direction(
        min_azi=np.deg2rad(deg), 
        max_azi=np.deg2rad(deg), 
        min_ele=math.pi / 2, 
        max_ele=math.pi / 2
    )
    up = np.array([0, 0, 1])
    x_axis = front
    y_axis = np.cross(up, x_axis)
    z_axis = np.cross(x_axis, y_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    return R


# def transform_coordinate(x: np.ndarray, R_from: np.ndarray, R_to: np.ndarray):
#     return x @ R_from.T @ R_to


def add5():
    for _ in range(10):
        R = sample_rotation_matrix_(45)
        R2 = sample_rotation_matrix_(60)

        x = np.array([1, 1, 0])
        x2 = transform_coordinate(
            x=x,
            origin_from=np.array([2, 1, 0]),
            origin_to=np.zeros(3),
            R_from=R,
            R_to=np.eye(3)
        )
        x3 = transform_coordinate(
            x=x2,
            origin_from=np.zeros(3),
            origin_to=np.array([2, 1, 0]),
            R_from=np.eye(3),
            R_to=R
        )

        from IPython import embed; embed(using=False); os._exit(0)
        x2 = transform_coordinate(x, R, R2)
        x3 = transform_coordinate(x2, R2, R)
        from IPython import embed; embed(using=False); os._exit(0)

        x2 = transform_coordinate(x, np.eye(3), R)
        x3 = transform_coordinate(x2, R, np.eye(3))

        b1 = transform_coordinate(x, R, np.eye(3))
        from IPython import embed; embed(using=False); os._exit(0)
        direction, up = sample_direction_up()

        x = np.array([1, 1, 1])
        eye = np.array([0, 0, 0])
        rotate(x, pos, direction, up)
        print(lookat, up)


def add6():
    x = torch.zeros(4, 5, 100)
    x[0, 0, 0 : 5] = torch.Tensor([100.3, 100.7, 121.8, 122, 122.2])

    h = get_delay_filter(x)
    fig, axes = plt.subplots(4, 1, sharex=True)
    for i in range(4):
        axes[i].stem(h[0, 0, i, 0:400])
    plt.savefig("_zz.pdf")
    from IPython import embed; embed(using=False); os._exit(0)


def add6b():
    x = torch.zeros(4, 5, 100)
    # x[0, 0, 0 : 5] = torch.Tensor([100.3, 100.7, 121.8, 122, 122.2])
    # x[0, 0, 0 : 5] = torch.Tensor([80.0, 100.3, 121.8, 122, 122.2])
    x[0, 0, 0 : 6] = torch.Tensor([-10, 80.0, 100.3, 121.8, 122, 122.2])
    # x[0, 0, 0 : 1] = torch.Tensor([100.3])

    mask = (x !=0).float()
    h, delay_int = get_delayed_filter(x, mask)
    fig, axes = plt.subplots(4, 1, sharex=True)
    for i in range(4):
        axes[i].stem(h[0, 0, i])
    plt.savefig("_zz.pdf")

    h_sum = add_delayed_filters(h, delay_int)

    h_sum[0, 0, 75:85]

    fig, axes = plt.subplots(4, 1, sharex=True)
    for i in range(4):
        axes[i].stem(h_sum[0, 0, 0:300])
    plt.savefig("_zz.pdf")

    b1 = torch.zeros((4, 5, 96000))
    b1[0, 0, 100] = 1

    device = "cuda"
    b1 = b1.to(device)
    h_sum = h_sum.to(device)
    for _ in range(10):
        t1 = time.time()
        b2 = convolve(b1, h_sum, origin=100)
        print(time.time() - t1)


    from IPython import embed; embed(using=False); os._exit(0)
    


def add7():

    h0 = np.zeros(255)
    h0[0] = 1
    # h0[127] = 1
    h = np.pad(h0, pad_width=((0, 1)), constant_values=0.)
    H = np.fft.rfft(h)
    h2 = np.fft.irfft(H).real
    h2 = h2[0 : -1]

    print(np.sum(np.abs(h2 - h0)))
    from IPython import embed; embed(using=False); os._exit(0)


def add7b():

    h0 = np.zeros(255)
    h0[0] = 1
    h = np.pad(h0, pad_width=((0, 1)), constant_values=0.)
    H = np.fft.rfft(h)

    h2 = np.fft.irfft(H).real
    h2 = np.fft.fftshift(h2)
    # h2 = h2[0 : -1]

    # print(np.sum(np.abs(h2 - h0)))
    from IPython import embed; embed(using=False); os._exit(0)

    

def add8():
    direction = normalize(np.array([1, 1, 1]))
    for _ in range(10):
        perturb_dir = sample_perturbed_direction(direction, np.deg2rad(10))
        print(np.rad2deg(included_angle(Tensor(direction), Tensor(perturb_dir)).numpy()))


# 
def add9():
    from nesd.utils.torch import perturb_direction, sph2cart, cart2sph
    
    # dirn = torch.rand((5, 4, 3))
    dirn = torch.Tensor([10, -1, 1])
    a1 = cart2sph(dirn)
    a2 = sph2cart(a1)
    from IPython import embed; embed(using=False); os._exit(0)
    

def add10():
    from nesd.utils.torch import perturb_direction, sph2cart, cart2sph, included_angle
    
    dirn = torch.rand((5, 4, 3))
    
    a1 = perturb_direction(dirn, d_angle=np.deg2rad(10))
    included_angle(a1, dirn)
    from IPython import embed; embed(using=False); os._exit(0)


def add11():

    from nesd.utils.torch import is_within_angle, wrap_angle
    dirn1 = torch.rand((5, 4, 3))
    dirn2 = torch.rand((5, 4, 3))
    is_within_angle(dirn1, dirn2, np.deg2rad(20), np.deg2rad(20))


def add12():
    hdf5_path = "./_tmp/eigenmike_open.h5"
    with h5py.File(hdf5_path, 'r') as hf:
        hs = hf["h"][:]
        h_mic_origin = hf.attrs["origin"]

    fig, axes = plt.subplots(4, 1, sharex=True)
    axes[0].stem(hs[0])
    axes[1].stem(hs[45])
    axes[2].stem(hs[90])
    axes[3].stem(hs[180])
    plt.savefig("_zz.pdf")

    from IPython import embed; embed(using=False); os._exit(0)


# test sample direction, is_within angle
def add13():
    from nesd.utils.torch import sample_direction, cart2sph, is_within_angle
    dirn = sample_direction(size=(1, 1, 10000))  # (b, l, r, 3)
    sph = cart2sph(dirn)
    r, ele, azi = sph[..., 0], sph[..., 1], sph[..., 2]

    vec1 = torch.Tensor([[1, 1, 1], [-1, -1, -1]])
    vec1 = vec1[None, :, None, :]  # (b, s, l, 3)

    indices = is_within_angle(
        vec1=vec1[:, :, :, None, :], 
        vec2=dirn[:, None, :, :, :], 
        max_ele=np.deg2rad(10), 
        max_azi=np.deg2rad(10)
    ).sum(dim=1).bool()  # (b, s, r)


    plt.scatter(azi, ele, s=4, c="b")
    plt.scatter(azi[indices], ele[indices], s=4, c="r")
    plt.xlim(0, math.pi * 2)
    plt.ylim(0, math.pi)
    plt.savefig("_zz.pdf")

    from IPython import embed; embed(using=False); os._exit(0)


def add14():
    from nesd.utils.torch import perturb_direction, cart2sph
    vec1 = torch.Tensor([[1, 1, 1], [-1, -1, -1]])

    vecs = []
    for _ in range(100):
        vec2 = perturb_direction(vec1, np.deg2rad(10))
        vecs.append(vec2)

    vecs = torch.cat(vecs, dim=0)
    sph = cart2sph(vecs)
    r, ele, azi = sph[..., 0], sph[..., 1], sph[..., 2]
    plt.scatter(azi, ele, s=4, c="r")
    plt.xlim(0, math.pi * 2)
    plt.ylim(0, math.pi)
    plt.savefig("_zz.pdf")

    from IPython import embed; embed(using=False); os._exit(0)


def add15():
    from nesd.models.sinusoidal_pe import SinusoidalPE
    x = torch.arange(0, 6.29, 0.01)  # (b,)
    pe = SinusoidalPE(dim=16, scale=100.)
    emb = pe(x)  # (b, d)

    fig, ax = plt.subplots()
    ax.matshow(emb.T, origin='lower', aspect='auto', cmap='jet')
    ax.set_title("")
    ax.set_xlabel("input")
    ax.set_ylabel("dim")
    ax.xaxis.set_ticks([])
    ax.xaxis.tick_bottom()
    plt.savefig("_zz.pdf")


def add16():
    audio, fs = librosa.load(path="split0_1.wav", sr=None, mono=False)
    out = audio[:, 52 * fs : 54 * fs]

    soundfile.write(file="_test_d19.wav", data=out.T, samplerate=fs)
    from IPython import embed; embed(using=False); os._exit(0)


def add17():
    import pyloudnorm as pyln
    import soundfile as sf

    audio_paths = sorted(list(Path("test_wavs").rglob('*.wav')))
    for path in audio_paths:
        audio, fs = librosa.load(path=path, sr=None, mono=True)
        meter = pyln.Meter(fs)  # 默认为 ITU-R BS.1770 标准
        lufs = meter.integrated_loudness(audio)
        print(f"{path}, Integrated loudness: {lufs:.2f} LUFS")


def add18():
    a1 = np.zeros((181, 360))
    a1[100:120, 100:120] = 1

    a2 = np.zeros((181, 360))
    a2[100:120, 200:220] = 1

    alpha = 0.5
    blend = alpha * a1 + (1 - alpha) * a2

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.matshow(blend, cmap='jet', origin="upper", vmin=0, vmax=1)

    # ax.set_title(f"Time {time:.02f} s")
    ax.grid(color='w', linestyle='--', linewidth=0.2)
    plt.savefig("_zz.pdf")
    from IPython import embed; embed(using=False); os._exit(0)


def add19():

    video, audio, info = read_video("test_wavs/d23_fold4_room16_mix014.mp4", pts_unit="sec")

    with h5py.File('_video.h5', 'w') as hf:
        hf.create_dataset('x', data=video, dtype=np.uint8)
    from IPython import embed; embed(using=False); os._exit(0)


# Test combine colormaps
def add19b():

    import torchvision.transforms as T

    with h5py.File('_video.h5', 'r') as hf:
        x = hf["x"][0] / 255.  # (T, H, W, 3)


    a1 = np.zeros((181, 360))
    a1[100:120, 100:120] = 1
    cmap = plt.get_cmap('jet')
    a1 = cmap(a1)[:, :, :3]

    

    x1 = x
    transform = T.Resize((181, 360))  # 注意传入 (H, W)
    x1 = rearrange(Tensor(x1), 'h w c -> c h w')
    x1 = transform(x1)
    x1 = rearrange(x1, 'c h w -> h w c').numpy()
    
    blend = 0.5 * x1 + a1
    blend = np.clip(blend, 0., 1.)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.matshow(blend, cmap='jet', origin="upper", vmin=0, vmax=1)
    ax.grid(color='w', linestyle='--', linewidth=0.2)
    plt.savefig("_zz.pdf")

    from IPython import embed; embed(using=False); os._exit(0)


def add20():

    # from moviepy.editor import VideoFileClip
    from moviepy import VideoFileClip
    clip = VideoFileClip("test_wavs/d23_fold4_room16_mix014.mp4")
    clip = clip.resized((360, 181)).with_fps(10)
    frames = [frame for frame in clip.iter_frames()]
    frames = np.stack(frames, axis=0)
    with h5py.File('_video.h5', 'w') as hf:
        hf.create_dataset('x', data=frames, dtype=np.uint8)

    from IPython import embed; embed(using=False); os._exit(0)


def add21():
    path = "/home/qiuqiangkong/workspaces/nesd/audios/vctk_2s_segments/train/p329_085_0000.wav"
    sr = 48000
    audio, fs = librosa.load(path=path, sr=sr, mono=True) 
    soundfile.write(file="_zz.wav", data=audio*0.01, samplerate=sr)


if __name__ == '__main__':

    # test_shoebox()
    # add7b()
    # add6b()
    # add8()
    add21()