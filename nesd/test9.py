import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import math
import librosa
import soundfile
import time

from nesd.utils import normalize


wall_mat = {
    "description": "Example wall material",
    "coeffs": [0.1, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05],
    "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000, 16000],
}
r = 0.2
wall_mat2 = {
    "description": "Example wall material",
    "coeffs": [r, r, r, r, r, r, r, r, ],
    "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000, 16000],
}

def add():
    sample_rate = 24000
    # sample_rate = 8000

    select = '3'

    if select == '1':
        t = np.arange(sample_rate)
        source = np.cos(t * 2 * math.pi * 440 / sample_rate) * 0.1
    elif select == '2':
        t = np.arange(sample_rate * 3)
        source = np.zeros(len(t))
        source[100] = 1
    elif select == '3':
        audio_path = './resources/p360_396_mic1.flac.wav'
        source, fs = librosa.load(audio_path, sr=sample_rate, mono=True)

    corners = np.array([
        [8, 8], 
        [0, 8], 
        [0, 0], 
        [8, 0],
    ]).T
    height = 4

    materials = pra.Material(energy_absorption=wall_mat2)
    # materials = None

    is_raytracing = True

    t1 = time.time()
    room = pra.Room.from_corners(
        corners=corners,
        fs=sample_rate,
        materials=materials,
        max_order=3,
        ray_tracing=is_raytracing,
        air_absorption=False,
    )
    room.extrude(
        height=height, 
        materials=materials
    )

    if is_raytracing:
        room.set_ray_tracing(
            n_rays=1000,
            receiver_radius=0.5,
            energy_thres=1e-7,
            time_thres=1.0,
            hist_bin_size=0.004,
        )

    room.add_source(position=np.array([3.8, 7.43, 2.7]), signal=source)

    # if True:
    #     zenith = math.pi / 2
    #     directivity_object = pra.CardioidFamily(
    #         orientation=pra.DirectionVector(azimuth=azimuth, colatitude=zenith, degrees=False),
    #         pattern_enum=pra.DirectivityPattern.CARDIOID,
    #     )

    directivity_object = None
    room.add_microphone(loc=np.array([4, 4, 2]), directivity=directivity_object)

    room.compute_rir()

    room.simulate()
    print(time.time() - t1)
    
    #
    mic_signals = room.mic_array.signals[0]

    soundfile.write(file='_zz.wav', data=mic_signals, samplerate=sample_rate)
    fig, axs = plt.subplots(2,1, sharex=True)
    axs[0].stem(source[0:300])
    axs[1].stem(mic_signals[0:1000])
    plt.savefig('_zz.pdf')
    from IPython import embed; embed(using=False); os._exit(0)

    np.argmax(mic_signals)

    # pad 40 in the beginning
    # gain = 1/r

    # 210
    # 280


def add2():

    sample_rate = 24000
        
    t = np.arange(sample_rate)
    source = np.cos(t * 2 * math.pi * 440 / sample_rate) * 0.1
    
    corners = np.array([
        [8, 8], 
        [0, 8], 
        [0, 0], 
        [8, 0],
    ]).T
    height = 4
    cnt = 0
    random_state = np.random.RandomState(1234)

    while True:
        
        try:
            t1 = time.time()
            room = pra.Room.from_corners(
                corners=corners,
                fs=sample_rate,
                materials=None,
                max_order=5,
                ray_tracing=False,
                air_absorption=False,
            )
            room.extrude(
                height=height, 
                materials=None
            )


            a1 = normalize(random_state.uniform(low=-1, high=1, size=3))
            source_position = np.array([4, 4, 2]) + a1
            print(cnt, a1, source_position)

            room.add_source(position=source_position, signal=source)

            directivity_object = None
            room.add_microphone(loc=np.array([4, 4, 2]), directivity=directivity_object)

            room.compute_rir()
            
            # room.simulate()
            # print(cnt, time.time() - t1)
            cnt += 1
        
        except:
            from IPython import embed; embed(using=False); os._exit(0)


def add3():

    select = '5'

    if select == '1':
        x = np.array([np.nan, np.nan, 1.3, 1.5, 1.8, 2.0, np.nan, np.nan, 4.3, 4.6, 4.7, np.nan, np.nan])
    elif select == '2':
        x = np.array([np.nan, np.nan, 1.3, 1.5, 1.8, 2.0, np.nan, np.nan, 4.3, 4.6, 4.7, np.nan, np.nan, np.nan, 6.1, 6.3, 6.4, np.nan, np.nan])
    elif select == '3':
        x = np.array([1.3, 1.5, 1.8, 2.0, np.nan, np.nan])
    elif select == '4':
        x = np.array([np.nan, np.nan, 1.3, 1.5, 1.8, 2.0])
    elif select == '5':  # will not happen
        x = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    elif select == '6':
        x = np.array([1.3, 1.5, 1.8, 2.0])


    x = x[:, None]
    extend_look_direction(x)

from sklearn.linear_model import LinearRegression


def extend_look_direction(look_direction):

    frames_num = look_direction.shape[0]

    tmp = []  # nan
    tmp2 = []  # float

    new_x = look_direction.copy()

    # process begin Nan
    for i in range(frames_num):
        if math.isnan(look_direction[i, 0]):
            if len(tmp2) > 0:
                break
            else:
                tmp.append(i)

        if not math.isnan(look_direction[i, 0]):
            tmp2.append(i)

        if math.isnan(look_direction[i, 0]) and len(tmp2) > 0:
            break

    if len(tmp) > 0 and len(tmp2) > 0:
        X = np.array(tmp2)[:, None]
        y = look_direction[np.array(tmp2)]
        reg = LinearRegression().fit(X, y)
        pred = reg.predict(np.array(tmp)[:, None])
        new_x[np.array(tmp)] = pred

    # process end Nan
    tmp = []  # nan
    tmp2 = []  # float
    new_x = new_x[::-1]
    _look_direction = look_direction[::-1]

    for i in range(frames_num):
        if math.isnan(_look_direction[i, 0]):
            if len(tmp2) > 0:
                break
            else:
                tmp.append(i)

        if not math.isnan(_look_direction[i, 0]):
            tmp2.append(i)

        if math.isnan(_look_direction[i, 0]) and len(tmp2) > 0:
            break

    if len(tmp) > 0 and len(tmp2) > 0:
        X = np.array(tmp2)[:, None]
        y = _look_direction[np.array(tmp2)]
        reg = LinearRegression().fit(X, y)
        pred = reg.predict(np.array(tmp)[:, None])
        new_x[np.array(tmp)] = pred

    new_x = new_x[::-1]
    
    # process Middle Nan
    bgn = None
    end = None

    for i in range(1, frames_num):
        if math.isnan(new_x[i, 0]) and not math.isnan(new_x[i - 1, 0]):
            bgn = i - 1

        if not math.isnan(new_x[i, 0]) and math.isnan(new_x[i - 1, 0]):
            end = i

        if bgn is not None and end is not None:
            X = np.array([bgn, end])[:, None]
            y = np.array([new_x[bgn], new_x[end]])
            reg = LinearRegression().fit(X, y)
            pred = reg.predict(np.arange(bgn + 1, end)[:, None])
            new_x[np.arange(bgn + 1, end)] = pred
            bgn = None
            end = None
            
    for i in range(frames_num):
        new_x[i] = normalize(new_x[i])
    
    return new_x

if __name__ == '__main__':

    # add()
    # add2()
    add3()