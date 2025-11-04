from rooms.shoebox_ism import Shoebox
# from nesd.utils import *
import pyroomacoustics as pra
import random
import time
import matplotlib.pyplot as plt
import math
import librosa
import soundfile
from torch import Tensor
import h5py
import numpy as np
from nesd.utils.utils import *
from nesd.utils.numpy import sample_direction, transform_coordinate, normalize
from nesd.utils.torch import get_delayed_filter, add_delayed_filters


# test shoebox
def add():

    for _ in range(10):
        room = Shoebox()
        print(room.x_bnd, room.y_bnd, room.z_bnd)
    

# test shoeboex
def add2():
    room = Shoebox(10, 10, 8, 8, 3, 3)

    for _ in range(10):
        pos = room.sample_position()
        print(pos)

# test rotation matrix
def add3():
    for _ in range(10):
        R = sample_rotation_matrix()
        print(np.linalg.norm(R, axis=0))
        print(R)


def _sample_rotation_matrix(up=np.array([0, 0, 1])) -> np.ndarray:
    front = sample_direction(
        min_ele=math.pi / 2, 
        max_ele=math.pi / 2,
        min_azi=math.pi / 6, 
        max_azi=math.pi / 6, 
        
    )
    x_axis = front
    y_axis = np.cross(up, x_axis)
    z_axis = np.cross(x_axis, y_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=-1)
    return R


# test rotation matrix
def add4():
    
    world_origin = np.zeros(3)
    world_R = np.eye(3)

    local_origin = np.array([4,5,6])
    local_R = _sample_rotation_matrix()

    mic_local_pos = np.array([[1,1,1], [-1,-1,-1]])

    mic_world_pos = transform_coordinate(
        x=mic_local_pos, 
        origin_from=local_origin,
        origin_to=world_origin,
        R_from=local_R,
        R_to=world_R
    )  # (mics_num, 3)

    mic_dir = normalize(transform_coordinate(
        x=mic_local_pos,
        origin_from=np.zeros(3),
        origin_to=np.zeros(3),
        R_from=local_R,
        R_to=world_R
    ))
    print(mic_dir)

    from IPython import embed; embed(using=False); os._exit(0)


# test ISM
def add5():

    sr = 48000
    c = 343

    for _ in range(10):
        t1 = time.time()
        corners = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])

        room = pra.Room.from_corners(
            corners=corners.T,
            max_order=5,
        )
        room.extrude(height=3)

        # pos = np.array([random.uniform(0.1, 9.9), random.uniform(0.1, 9.9), random.uniform(0.1, 2.9)])
        src_pos = np.array([[2,2,2],[1,1,1]])
        room.add_source(src_pos[0].T)
        room.add_source(src_pos[1].T)

        # pos = np.array([random.uniform(0.1, 9.9), random.uniform(0.1, 9.9), random.uniform(0.1, 2.9)])
        # pos = np.array([[8,7,1]])
        mic_pos = np.array([[8,7,1], [3, 3, 1], [4, 4, 1]])
        room.add_microphone(mic_pos.T)

        room.image_source_model()

        # room.visibility  # (src_num, mic_num)
        src_imgs = [s.images.T for s in room.sources]  # (src_num, imgs_num, 3)

        src_num = len(room.sources)
        mic_num = room.n_mics

        imgs = [[None] * mic_num] * src_num
        orders = [[None] * mic_num] * src_num

        for s in range(src_num):
            for m in range(mic_num):
                vis_idx = room.visibility[s][m]  # (imgs_num,)
                imgs[s][m] = room.sources[s].images.T[vis_idx]  # (imgs_num, 3)
                orders[s][m] = room.sources[s].orders[vis_idx]  # (imgs_num,)

        print(time.time() - t1)

        doa = imgs[0][0] - src_pos[0]
        ray_num = len(doa)
        dist = np.linalg.norm(doa, axis=-1)
        
        delay_samples = (dist / c) * sr
        dist = Tensor(dist)
        delay_int, h_frac, origin = get_delayed_filter(Tensor(delay_samples))
        h_frac = h_frac / dist[:, None] / math.sqrt(ray_num)

        h_sum = add_delayed_filters(delay_int, h_frac, origin=origin, length=48000)
        plt.plot(h_sum[0:10000])
        plt.savefig("_zz.pdf")

        audio, _ = librosa.load(path="./assets/p226_001.wav", sr=sr, mono=True)
        from IPython import embed; embed(using=False); os._exit(0)
        y = convolve(Tensor(audio), h_sum, origin=0)

        y = y / y.abs().max()
        soundfile.write(file="_zz.wav", data=y.numpy(), samplerate=sr)
        # np.convolve(x, h, mode='same')
        


# test render
def add6():

    from nesd.utils.torch import get_delayed_filter, add_delayed_filters, convolve

    sr = 48000
    c = 343
    audio, _ = librosa.load(path="./assets/p226_001.wav", sr=sr, mono=True)

    with h5py.File('_tmp/0000.h5', 'r') as hf:
        src_pos = hf["src_pos"][:]  # (src_num, 3)
        mic_pos = hf["mic_pos"][:]  # (mic_num, 3)
        mic_dir = hf["mic_dir"][:]  # (mic_num, 3)
        mic_idx = hf["mic_idx"][:]  # (mic_num,)
        lis_pos = hf["lis_pos"][:]  # (lis_num, 3)
        lis_idx = hf["lis_idx"][:]  # (lis_num,)
        imgs = load_list_of_list_from_hdf5(hf, "img")  # (src_num, receiver_num, img_num, 3)
        orders = load_list_of_list_from_hdf5(hf, "order")  # (src_num, receiver_num, img_num)

        s = 0
        m = 0
        doa = imgs[s][m] - src_pos[s]  # (img_num, 3)
        dist = np.linalg.norm(doa, axis=-1)  # (img_num,)
        
        delay_samples = (dist / c) * sr  # (img_num,)
        dist = Tensor(dist)
        
        delay_int, h_frac, origin = get_delayed_filter(Tensor(delay_samples))
        # h: (img_num, filter_len,), delay_int: (img_num,), origin: (img_num,)

        dist_amp = 1. / dist

        alpha_wall = random.uniform(0, 0.5)
        reflect_amp = np.sqrt(1. - alpha_wall) ** orders[s][m]

        ray_num = len(imgs[s][m])
        total_amp = dist_amp * Tensor(reflect_amp) / math.sqrt(ray_num)

        h_frac = h_frac * total_amp[:, None]

        #
        h_sum = add_delayed_filters(delay_int, h_frac, length=48000, origin=origin)
        plt.plot(h_sum[0:10000])
        plt.savefig("_zz.pdf")

        audio, _ = librosa.load(path="./assets/p226_001.wav", sr=sr, mono=True)
        y = convolve(Tensor(audio), h_sum, origin=0)
        y = y / y.abs().max()
        soundfile.write(file="_zz.wav", data=y.numpy(), samplerate=sr)
        # np.convolve(x, h, mode='same')
        from IPython import embed; embed(using=False); os._exit(0)

        from IPython import embed; embed(using=False); os._exit(0)


# Test normalize
def add7():
    from nesd.utils.numpy import normalize

    x = np.array([[1.,1,1],[2,2,2]])
    y = normalize(x)
    print(y)


# Test transform coordinate
def add8():

    from nesd.utils.torch import transform_coordinate, normalize

    world_origin = torch.zeros(3)
    world_R = torch.eye(3)

    local_origin = Tensor([4,5,6])  # (3,)
    local_R = _sample_rotation_matrix()  # (3, 3)
    local_R = Tensor(local_R)

    mic_local_pos = Tensor([[1,1,1], [-1,-1,-1]])

    mic_world_pos = transform_coordinate(
        x=mic_local_pos, 
        origin_from=local_origin,
        origin_to=world_origin,
        R_from=local_R,
        R_to=world_R
    )  # (mics_num, 3)

    mic_dir = normalize(transform_coordinate(
        x=mic_local_pos,
        origin_from=torch.zeros(3),
        origin_to=torch.zeros(3),
        R_from=local_R,
        R_to=world_R
    ))
    print(mic_dir)

    b1 = transform_coordinate(
        x=mic_world_pos, 
        origin_from=world_origin,
        origin_to=local_origin,
        R_from=world_R,
        R_to=local_R
    )  # (mics_num, 3)

    from IPython import embed; embed(using=False); os._exit(0)


# Test dataloader
def add9():

    from train import get_model, get_dataset, get_data_transform
    from torch.utils.data._utils.collate import default_collate

    config_yaml = "./kqq_configs/07a.yaml"
    configs = parse_yaml(config_yaml)
    sr = configs["sample_rate"]
    device = "cuda"

    test_dataset = get_dataset(configs, split="test")
    data_transform = get_data_transform(configs).to(device)

    out_dir = "_tmp/audio"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for n, data in enumerate(test_dataset):
        
        data = default_collate([data])
        data = to_device(data, device)
        data = data_transform(data)

        audio = Tensor(data["mic_wav"]).to(device)

        path = Path(out_dir, f"{n:04d}.wav")
        soundfile.write(file=path, data=audio.data.cpu().numpy()[0].mean(axis=0), samplerate=sr)
        print(f"Write out to {path}")
        
        if n == 10:
            break


if __name__ == '__main__':

    # test_shoebox()
    add9() 