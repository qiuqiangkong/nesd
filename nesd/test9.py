import h5py
import yaml
import numpy as np
import os
from scipy import io
import soundfile
import librosa
import matplotlib.pyplot as plt
import time
import math
import mat73
from scipy.signal import fftconvolve
import pyroomacoustics as pra

from nesd.utils import int16_to_float32, sph2cart#, cart2sph


def add():

    yaml_path = "_zz.yaml"
    hdf5_path = "_zz_mic.h5"
    sample_rate = 16000
    segment_samples = 48000

    with open(yaml_path, 'r') as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)

    # sources_dict = data_dict['sources']

    with h5py.File(hdf5_path, 'r') as hf:
        tensor = hf['data'][:]

    mics_num = len(data_dict['microphones'])
    integrated_waveform = np.zeros((mics_num, segment_samples))

    t1 = time.time()

    for mic_index, mic_dict in enumerate(data_dict['microphones']):

        for source_index, source_dict in enumerate(data_dict['sources']):

            waveform_hdf5_path = os.path.join("/home/tiger/workspaces/nesd/hdf5s/vctk/sr=16000/train", source_dict['hdf5_name'])

            with h5py.File(waveform_hdf5_path, 'r') as hf:
                origin_waveform = int16_to_float32(hf['waveform'][:])

            valid_indexes = np.where(tensor[mic_index, :, 3] == source_index)[0]

            N = tensor[mic_index, valid_indexes].shape[0]
            delays = tensor[mic_index, valid_indexes, 4]
            gains =  tensor[mic_index, valid_indexes, 5]

            '''
            waveforms = np.tile(waveform[None, :], (N, 1))
            waveforms *= gain[:, None]# * integral_scale

            tmp += np.sum(waveforms, axis=0) / N

            soundfile.write(file='_zz.wav', data=tmp, samplerate=16000)

            from IPython import embed; embed(using=False); os._exit(0)
            '''
            integrated_waveform_per_source = np.zeros(segment_samples)

            for n in range(N):
                waveform = gains[n] * origin_waveform
                waveform = delay_waveform(waveform, delay_seconds=delays[n], sample_rate=sample_rate)
                integrated_waveform_per_source += waveform

            integrated_waveform_per_source /= N

            integrated_waveform[mic_index, :] += integrated_waveform_per_source

    print(time.time() - t1)
    # soundfile.write(file='_zz.wav', data=integrated_waveform[0, :], samplerate=16000)

    # from IPython import embed; embed(using=False); os._exit(0)

def add2():

    yaml_path = "_zz.yaml"
    hdf5_path = "_zz_field.h5"

    with open(yaml_path, 'r') as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)

    with h5py.File(hdf5_path, 'r') as hf:
        tensor = hf['data'][:]

    random_state = np.random.RandomState(1234)

    for field_index, field_dict in enumerate(data_dict['fields']):

        valid_indexes = np.where(tensor[field_index, :, 3] != -1)[0]

        ray_index = random_state.choice(valid_indexes)

        source_index = int(tensor[field_index, ray_index, 3])
        delay = tensor[field_index, ray_index, 4]
        gain = tensor[field_index, ray_index, 5]

        waveform_hdf5_path = os.path.join("/home/tiger/workspaces/nesd/hdf5s/vctk/sr=16000/train", data_dict['sources'][source_index]['hdf5_name'])

        with h5py.File(waveform_hdf5_path, 'r') as hf:
            origin_waveform = int16_to_float32(hf['waveform'][:])
        
        waveform = gain * origin_waveform

        soundfile.write(file='_zz2.wav', data=waveform, samplerate=16000)

        from IPython import embed; embed(using=False); os._exit(0)


from scipy import interpolate

# def delay_waveform(waveform, distance_between_source_and_mic, step_samples, sample_rate, speed_of_sound):
def delay_waveform(waveform, delay_seconds, sample_rate):

    segment_samples = waveform.shape[0]
    delay_samples = delay_seconds * sample_rate
    # delay_samples += np.arange(segment_samples)

    # new_waveform = np.interp(x=np.arange(segment_samples), xp=delay_samples, fp=waveform, mode='bicubic')
    # new_waveform[0 : int(delayed_samples[0])] = 0
    f = interpolate.interp1d(x=np.arange(segment_samples), y=waveform, kind='cubic')
    new_x = - delay_samples + np.arange(segment_samples)
    new_x = np.clip(new_x, a_min=0., a_max=np.inf)
    new_waveform = f(new_x)

    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(2,1, sharex=True)
    # axs[0].plot(waveform[100:350], c='r')
    # axs[0].plot(new_waveform[100:350], c='b')
    # plt.savefig('_zz.pdf')
    # from IPython import embed; embed(using=False); os._exit(0)

    return new_waveform


class Microphone:
    def __init__(self):
        pass

    def look_at(self, position, target, up):
        # https://www.programcreek.com/python/?CodeExample=look+at
        self.position = position

        z = target - position
        z = z / np.linalg.norm(z)

        self.z = z


class Sphere:
    def __init__(self, center, radius):
        
        self.center = center
        self.radius = radius


class SphereSource:
    def __init__(self, source, sphere):
        self.source = source
        self.sphere = sphere


def add3():

    # build scene

    # mic_position = np.array([0, 0, 0])
    # center = np.array([1, 1, 1])
    # up = np.array([0, 1, 0])
    # right = None

    # sphere = None
    sample_rate = 16000
    speed_of_sound = 343.

    random_state = np.random.RandomState(1234)

    mic_position = np.array([0, 0, 0])
    # target = random_state.uniform(low=-3, high=3, size=3)
    target = np.array([1, 0, 0])
    up = np.array([0, 1, 0])

    mic = Microphone()
    mic.look_at(position=mic_position, target=target, up=up)

    # sphere_sources = []

    # for _ in range(2):
        
    sphere = Sphere(
        center=random_state.uniform(low=-3, high=3, size=3), 
        radius=0.1
    )

    # mic_to_src = sphere.center - mic.position
    # mic.z
    src_to_mic = mic.position - sphere.center

    distance = np.linalg.norm(x=src_to_mic, ord=2)
    delayed_seconds = distance / speed_of_sound
    delayed_samples = sample_rate * delayed_seconds

    #
    t = np.arange(48000)
    source = 0.1 * np.sin(2 * np.pi * 440 / 16000 * t)

    filter_len = 100
    delayed_samples = 3
    filt = get_filter(filter_len=filter_len, delayed_samples=delayed_samples)

    # source = source[0 : len(source) - len(filt) + 1]
    
    t1 = time.time()
    # y = fftconvolve(in1=source, in2=filt, mode='full')
    y = conv_signals(source=source, filt=filt)
    print(time.time() - t1)

    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].stem(source[0 : 100])
    axs[1].stem(y[0 : 100])
    plt.savefig('_zz.pdf')
    soundfile.write(file='_zz.wav', data=y, samplerate=16000)

    from IPython import embed; embed(using=False); os._exit(0)


def add4():

    sample_rate = 16000
    speed_of_sound = 343.
    filter_len = 10000

    random_state = np.random.RandomState(1234)

    mic_position = np.array([0, 0, 0])
    target = np.array([1, 0, 0])
    up = np.array([0, 1, 0])

    mic = Microphone()
    mic.look_at(position=mic_position, target=target, up=up)

    sphere_sources = []

    fs = [440, 800]

    for i in range(2):
        
        sphere = Sphere(
            center=random_state.uniform(low=-3, high=3, size=3), 
            radius=0.1
        )

        t = np.arange(48000)
        source = 0.1 * np.sin(2 * np.pi * fs[i] / 16000 * t)

        sphere_source = SphereSource(sphere=sphere, source=source)
        sphere_sources.append(sphere_source)

    total = np.zeros(48000)

    # ys.append(y)
    for sphere_source in sphere_sources:

        src_to_mic = mic.position - sphere_source.sphere.center

        distance = np.linalg.norm(x=src_to_mic, ord=2)
        delayed_seconds = distance / speed_of_sound
        delayed_samples = sample_rate * delayed_seconds

        filt = get_filter(filter_len=filter_len, delayed_samples=delayed_samples)

        t1 = time.time()
        y = conv_signals(source=sphere_source.source, filt=filt)
        print(time.time() - t1)
        print(y.shape)
    
        total += y

    soundfile.write(file='_zz.wav', data=total, samplerate=16000)
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(y)
    axs[1].plot(total)
    plt.savefig('_zz.pdf')
    from IPython import embed; embed(using=False); os._exit(0)


def get_filter(filter_len, gain, delayed_samples):
    filt = np.zeros(filter_len)

    filt[int(delayed_samples)] = 1 - (delayed_samples - int(delayed_samples))
    filt[int(delayed_samples) + 1] = delayed_samples - int(delayed_samples)
    filt *= gain

    return filt


def conv_signals(source, filt):
    len_source = len(source)
    source = np.concatenate((np.zeros(len(filt) - 1), source))
    y = fftconvolve(in1=source, in2=filt, mode='valid')
    y = y[0 : len_source]
    return y


def add5():

    mic_yaml = "ambisonic.yaml"

    with open(mic_yaml, 'r') as f:
        mics_meta = yaml.load(f, Loader=yaml.FullLoader)


    for mic_meta in mics_meta:

        x, y, z = sph2cart(
            r=mic_meta['radius'], 
            azimuth=mic_meta['azimuth'], 
            zenith=mic_meta['zenith']
        )

        mic_position = np.array([x, y, z])
        target = np.array([2 * x, 2 * y, 2 * z])
        up = np.array([0, 1, 0])

        mic = Microphone()
        mic.look_at(position=mic_position, target=target, up=up)

        from IPython import embed; embed(using=False); os._exit(0)


def add6():
    from scipy.signal import fftconvolve
    source = np.arange(100)
    filt = np.zeros(20)
    filt[2] = 1
    y = fftconvolve(in1=source, in2=filt)

    from IPython import embed; embed(using=False); os._exit(0)


def fix_length(audio, segment_samples):
    repeats_num = (segment_samples // audio.shape[-1]) + 1
    audio = np.tile(audio, repeats_num)[0 : segment_samples]
    return audio

def add7():

    audio_path = "./resources/AmbiX_360_speech_a_format_front_tmp.wav"
    out_path = "./resources/AmbiX_360_speech_a_format_front.wav"
    audio, fs = librosa.load(audio_path, sr=None, mono=False)   # (channels, audio_samples)

    if audio.shape[1] < fs * 3:
        audio = fix_length(audio, fs * 3)

    W = np.array([
        [1, 1, 1, 1],
        [1, 1, -1, -1],
        [1, -1, 1, -1],
        [1, -1, -1, 1],
    ]) / 4

    a_format = np.dot(W, audio)

    soundfile.write(file=out_path, data=a_format.T, samplerate=fs)


def add8():

    sample_rate = 16000

    tmps = []

    for azimuth in np.arange(0, 2 * math.pi, 0.1):

        corners = np.array([
            [8, 8], 
            [0, 8], 
            [0, 0], 
            [8, 0],
        ]).T
        height = 8

        room = pra.Room.from_corners(
            corners=corners,
            fs=16000,
            materials=None,
            max_order=0,
            ray_tracing=False,
            air_absorption=False,
        )
        room.extrude(height=height, materials=None)

        t = np.arange(sample_rate)
        signal = np.cos(t * 2 * math.pi * 440 / sample_rate) * 0.1

        room.add_source(position=np.array([5, 4, 4]), signal=signal)

        if True:
            zenith = math.pi / 2
            directivity_object = pra.CardioidFamily(
                orientation=pra.DirectionVector(azimuth=azimuth, colatitude=zenith, degrees=False),
                pattern_enum=pra.DirectivityPattern.CARDIOID,
            )

            room.add_microphone(loc=np.array([4, 4, 4]), directivity=directivity_object)

        room.compute_rir()

        room.simulate()

        #
        mic_signals = room.mic_array.signals[0]

        tmp = np.max(mic_signals)
        print(tmp)
        tmps.append(tmp)

    # soundfile.write(file='_zz.wav', data=signal, samplerate=sample_rate)
    plt.plot(tmps)
    plt.savefig('_zz.pdf')
    from IPython import embed; embed(using=False); os._exit(0)


def add9():
    data = {'1': 11}
    func2(data)

    print(data)

def func(data):
    data['1'] = 22

def func2(data):
    # global data
    tmp_data = data.copy()
    tmp_data['1'] = 22
    # data = tmp_data
    data = None


def add10():
    from scipy import io
    import mat73

    mat_path = '/home/tiger/datasets/dcase2022/tau-srir/TAU-SRIR_DB/rirdata.mat'
    rirdata = io.loadmat(mat_path)['rirdata']
    # rirdata[0][0][1][0][0][2][0][0][0].shape
    # (1, 1, 1, 9 rooms, 1, 4 mics, 2, 9, 1, 360, 3)

    # from IPython import embed; embed(using=False); os._exit(0)

    mat_path = '/home/tiger/datasets/dcase2022/tau-srir/TAU-SRIR_DB/measinfo.mat'
    mea = io.loadmat(mat_path)
    

    mat_path = '/home/tiger/datasets/dcase2022/tau-srir/TAU-SRIR_DB/rirs_01_bomb_shelter.mat'
    tmp = mat73.loadmat(mat_path)

    # (?, n_rooms?, len, mics_num, deg?)
    tmp['rirs']['mic'][0][0]
    from IPython import embed; embed(using=False); os._exit(0)
    X = tmp['X']        # X is the var in .mat


def add11():

    with h5py.File('_zz.h5', 'w') as hf:   # 'a' for append
        hf.create_dataset('x', data=np.zeros((10, 20, 30)), dtype=np.float32)

    with h5py.File('_zz.h5', 'r') as hf:
        hf['x'][0, np.array([0, 1, 2]), :]
        from IPython import embed; embed(using=False); os._exit(0)


def add12():

    rirdata_mat_path = os.path.join("/home/tiger/datasets/dcase2022/tau-srir/TAU-SRIR_DB/rirdata.mat")
    rir_traj = io.loadmat(rirdata_mat_path)['rirdata']
    # from IPython import embed; embed(using=False); os._exit(0)
    # rir_traj[0][0][1][8][0][2][1][0][0].shape

    mat_path = "/home/tiger/datasets/dcase2022/tau-srir/TAU-SRIR_DB/rirs_10_tc352.mat"
    rir_data = mat73.loadmat(mat_path)
    # print("Mic pos: {}".format(rir_traj[0][0][3]))

    # rir_data['rirs']['mic'][1][8] # (7200, 4, 360)
    # rir_data['rirs']['mic'][0][8]

    from IPython import embed; embed(using=False); os._exit(0)

    traj1s_num = len(rir_traj[0][0][1][room_id][0][2])
    # print(traj1s_num)
    
    for traj1 in range(traj1s_num):
        
        traj2s_num = len(rir_traj[0][0][1][room_id][0][2][traj1])

        for traj2 in range(traj2s_num):

            rir_tensor = rir_data['rirs']['mic'][traj1][traj2]
            # (7200, 4, 360)

            rir_tensor = rir_tensor.transpose(1, 2, 0)
            # (4, 360, 7200)

            traj_tensor = rir_traj[0][0][1][room_id][0][2][traj1][traj2][0]
            # (360, 3)
            # print(room_id, traj_id1, traj_id2)



            hdf5_path = os.path.join(hdf5s_dir, "room_{}_traj_{}_height_{}.h5".format(room_id, traj1, traj2))

            with h5py.File(hdf5_path, 'w') as hf:   # 'a' for append
                hf.create_dataset('traj', data=traj_tensor, dtype=np.float32)
                hf.create_dataset(name='rir', data=rir_tensor, dtype=np.float32)

            print("Write out to {}".format(hdf5_path))


# test yin's rigid
def add13():
    from nesd.rigid import get_rigid_sph_array

    src_pos = np.zeros((1, 0, 0))
    mic_pos = np.zeros((0, 0, 0))


def add14():

    x = np.array([0, 4, 3, 5, 2, 7])

    X = np.fft.rfft(x)

    y = np.fft.irfft(X)

    from IPython import embed; embed(using=False); os._exit(0)


def add15():

    # hdf5_path = "/home/qiuqiangkong/workspaces/nesd2/hdf5s/tau-noise/ambience_foa_sn3d_24k_edited_000.h5"

    hdf5_path = "/home/qiuqiangkong/workspaces/nesd2/hdf5s/tau-noise/ambience_tetra_24k_edited_024.h5"

    with h5py.File(hdf5_path, 'r') as hf:

        x = int16_to_float32(hf["waveform"][:])

        print(np.max(x))
        print(np.mean(x**2))

        plt.plot(x[0][0:100000])
        plt.savefig("_zz.pdf")

        from IPython import embed; embed(using=False); os._exit(0)


def add16():

    audio_path = "/home/qiuqiangkong/datasets/dcase2019/task3/downloaded_package/mic_eval/split0_1.wav"

    audio, _ = librosa.load(path=audio_path, sr=None, mono=False)

    # plt.plot(audio[0][0:100000])
    plt.plot(audio[0])
    plt.savefig("_zz.pdf")

    from IPython import embed; embed(using=False); os._exit(0)


def add17():

    # audio_path = "/home/qiuqiangkong/datasets/dcase2016/task2/dcase2016_task2_train_dev/dcase2016_task2_train/phone059.wav"
    # audio_path = "/home/qiuqiangkong/workspaces/nesd2/audios/musdb18hq_2s_segments/test/Nerve 9 - Pray For The Rain_0010.wav"
    audio_path = "/home/qiuqiangkong/datasets/dcase2022/task3/mic_dev/dev-train-sony/fold3_room21_mix001.wav" 

    audio, _ = librosa.load(path=audio_path, sr=None, mono=False)

    # plt.plot(audio[0][0:100000]) 
    # plt.plot(audio)
    # plt.savefig("_zz.pdf")
    print(np.max(audio))

    from IPython import embed; embed(using=False); os._exit(0)


if __name__ == "__main__":

    # add()
    # add2()
    # add3()
    # add4()
    # add5()
    # add6()
    # add7()
    # add8()
    # add9()
    # add10()
    # add11()
    # add12()
    # add13()
    # add14()
    # add15()
    # add16()
    add17()