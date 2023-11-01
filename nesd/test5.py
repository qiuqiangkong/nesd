
import numpy as np
import pyroomacoustics as pra
import librosa
import soundfile
import matplotlib.pyplot as plt
import random
import math

import time
from room import Room
from nesd.utils import norm, fractional_delay_filter, FractionalDelay, DirectionSampler, sph2cart, get_cos, Agent, repeat_to_length, cart2sph, Rotator3D, sample_agent_look_direction


def add():

    a = np.array([1,2,3,4,5])
    b = np.array([1, 0, 0])
    y = np.convolve(a, b, mode="valid")

    from IPython import embed; embed(using=False); os._exit(0)


def add2():

    
    length = 5
    width = 4
    height = 3

    sources_pos = [[1, 1, 1]]
    
    sample_rate = 16000
    speed_of_sound = 343

    x, fs = librosa.load(path="./resources/p226_001.wav", sr=sample_rate, mono=True)
    
    # ===
    corners = np.array([[0, 0], [0, width], [length, width], [length, 0]]).T
    # shape: (2, 4)

    room = pra.Room.from_corners(
        corners=corners,
        max_order=3,
    )

    room.extrude(height=height)

    for source_pos in sources_pos:
        room.add_source(source_pos)

    room.add_microphone([0.1, 0.1, 0.1])    # dummy

    room.image_source_model()

    images = room.sources[0].images.T
    # (N, 3)

    tmp = [str(image) for image in images]
    # print(len(tmp))
    # print(len(set(tmp)))

    images = np.array(list(set(map(tuple, images))))
    # (N, 3)

    # ----
    mic_pos = [0.2, 0.3, 0.4]

    hs = []

    for image in images:
        # t2 = time.time()
        direction = image - mic_pos
        distance = norm(direction)
        delayed_samples = distance / speed_of_sound * sample_rate
        # print("b1", time.time() - t2)

        # t2 = time.time()
        decay_factor = 1 / distance
        h = decay_factor * fractional_delay_filter(delayed_samples)
        hs.append(h)
        # print("b2", time.time() - t2)

    # print(time.time() - t1)
        
    t1 = time.time()
    max_filter_len = max([len(h) for h in hs])
    # hs = [librosa.util.fix_length(data=h, size=max_filter_len, axis=0) for h in hs]
    # sum_h = np.sum(hs, axis=0)
    sum_h = np.zeros(max_filter_len)
    for h in hs:
        sum_h[0 : len(h)] += h

    # ----

    ##
    # mic signal, sampled direction, sampled signal


def add3():

    length = 5
    width = 4
    height = 3

    sample_rate = 16000
    speed_of_sound = 343
    max_order = 3

    x1, fs = librosa.load(path="./resources/p226_001.wav", sr=sample_rate, mono=True)
    x2, fs = librosa.load(path="./resources/p232_006.wav", sr=sample_rate, mono=True)

    sources = [x1, x2]
    sources_num = len(sources)

    sources_pos = [
        [1, 1, 1],
        [2, 2, 2],
    ]

    mic_pos = [0.2, 0.3, 0.4]

    ys = []

    # tmp = []
    image_meta_list = []

    t0 = time.time()

    # ===
    # for source_pos in sources_pos:
    for source_index in range(sources_num):

        source = 0.5 * sources[source_index]
        source_pos = sources_pos[source_index]
    
        t1 = time.time()

        corners = np.array([[0, 0], [0, width], [length, width], [length, 0]]).T
        # shape: (2, 4)

        room = pra.Room.from_corners(
            corners=corners,
            max_order=max_order,
        )

        room.extrude(height=height)

        room.add_source(source_pos)

        room.add_microphone([0.1, 0.1, 0.1])    # dummy

        room.image_source_model()

        images = room.sources[0].images.T
        # (N, 3)

        orders = room.sources[0].orders

        tmp = [str(image) for image in images]
        # print(len(tmp))
        # print(len(set(tmp)))

        images = np.array(list(set(map(tuple, images))))
        # (N, 3)

        hs = []

        print("a1", time.time() - t1)

        for image_index, image in enumerate(images):

            order = orders[image_index]

            t2 = time.time()
            direction = image - mic_pos
            distance = norm(direction)
            delayed_samples = distance / speed_of_sound * sample_rate
            print("b1", time.time() - t2)

            t2 = time.time()
            decay_factor = 1 / distance

            angle_factor = 1.

            h = decay_factor * fractional_delay_filter(delayed_samples)
            hs.append(h)
            print("b2", time.time() - t2)

            normalized_direction = direction / distance
            # tmp.append((source_index, order, normalized_direction, distance))
            image_meta = {
                "source_index": source_index,
                "order": order,
                "direction": normalized_direction,
                "distance": distance,
            }
            image_meta_list.append(image_meta)


        # print(time.time() - t1)
        
        t1 = time.time()
        max_filter_len = max([len(h) for h in hs])
        # hs = [librosa.util.fix_length(data=h, size=max_filter_len, axis=0) for h in hs]
        # sum_h = np.sum(hs, axis=0)
        sum_h = np.zeros(max_filter_len)
        for h in hs:
            sum_h[0 : len(h)] += h

        t1 = time.time()
        y = convolve_source_filter(x=source, h=sum_h)
        print("a3", time.time() - t1)

        # soundfile.write(file="_zz.wav", data=y, samplerate=16000)
        # plt.plot(sum_h)
        # plt.savefig("_zz.pdf")

        ys.append(y)


    max_y_len = max([len(y) for y in ys])
    sum_y = np.zeros(max_y_len)
    for y in ys:
        sum_y[0 : len(y)] += y

    # soundfile.write(file="_zz.wav", data=sum_y, samplerate=16000)

    print("z0", time.time() - t0)

    # sample_positive_negative(image_meta_list)
    t0 = time.time()

    ##################
    agent_pos = [1.2, 1.3, 1.4]
    image_meta_list = []

    for source_index in range(sources_num):

        source = 0.5 * sources[source_index]
        source_pos = sources_pos[source_index]
    
        t1 = time.time()

        corners = np.array([[0, 0], [0, width], [length, width], [length, 0]]).T
        # shape: (2, 4)

        room = pra.Room.from_corners(
            corners=corners,
            max_order=max_order,
        )

        room.extrude(height=height)

        room.add_source(source_pos)

        room.add_microphone([0.1, 0.1, 0.1])    # dummy

        room.image_source_model()

        images = room.sources[0].images.T
        # (N, 3)

        orders = room.sources[0].orders

        tmp = [str(image) for image in images]
        # print(len(tmp))
        # print(len(set(tmp)))

        images = np.array(list(set(map(tuple, images))))
        # (N, 3)

        for image_index, image in enumerate(images):

            order = orders[image_index]

            t2 = time.time()
            direction = image - agent_pos
            distance = norm(direction)
            delayed_samples = distance / speed_of_sound * sample_rate
            print("b1", time.time() - t2)

            t2 = time.time()
            decay_factor = 1 / distance

            angle_factor = 1.

            # h = decay_factor * fractional_delay_filter(delayed_samples)
            # hs.append(h)
            print("b2", time.time() - t2)

            normalized_direction = direction / distance
            # tmp.append((source_index, order, normalized_direction, distance))
            image_meta = {
                "source_index": source_index,
                "order": order,
                "direction": normalized_direction,
                "distance": distance,
            }
            image_meta_list.append(image_meta)

    sample_positive_negative(image_meta_list, agent_pos)
    print("z0", time.time() - t0)

    from IPython import embed; embed(using=False); os._exit(0)


def convolve_source_filter(x, h):
    return np.convolve(x, h, mode='full')[0 : len(x)]


# def sample_6dof(image_meta_list)


def sample_positive_negative(image_meta_list, agent_position):

    '''
    for image_meta in image_meta_list:
        
        if image_meta["order"] == 0:
            pass
        else:
            pass
    '''
    positive_num = 4
    total_num = 20
    agents = []

    random_state = np.random.RandomState(1234)

    positive_metas = [image_meta for image_meta in image_meta_list if image_meta["order"] == 0]

    if len(positive_metas) <= positive_num:

        for meta in image_meta_list:
            agent = Agent(
                position=agent_position, 
                look_direction=meta["direction"], 
                # waveform=np.zeros(self.segment_samples),
                waveform=None,
                # see_source=np.zeros(self.frames_num),
                see_source=None,
            )
            agents.append(agent)
            

    else:
        _metas = random_state.choice(positive_metas, size=positive_num, replace=False)
        for meta in _metas:

            agent = Agent(
                position=agent_position, 
                look_direction=meta["direction"], 
                # waveform=np.zeros(self.segment_samples),
                waveform=None,
                # see_source=np.zeros(self.frames_num),
                see_source=None,
            )
            agents.append(agent)

    

    _direction_sampler = DirectionSampler(
        low_colatitude=0, 
        high_colatitude=math.pi, 
        sample_on_sphere_uniformly=False, 
        random_state=random_state,
    )

    half_angle = math.atan2(0.1, 1)

    while len(agents) < total_num:

        agent_look_azimuth, agent_look_colatitude = _direction_sampler.sample()

        agent_look_direction = np.array(sph2cart(
            r=1., 
            azimuth=agent_look_azimuth, 
            colatitude=agent_look_colatitude
        ))

        satisfied = True

        for meta in positive_metas:

            if meta["order"] == 0:
                agent_to_image = meta["direction"]

                angle_between_agent_and_src = np.arccos(get_cos(
                    agent_look_direction, agent_to_image))

                if angle_between_agent_and_src < half_angle:
                    satisfied = False

        if satisfied:
            agent = Agent(
                position=agent_position, 
                look_direction=agent_look_direction, 
                # waveform=np.zeros(self.segment_samples),
                waveform=None,
                # see_source=np.zeros(self.frames_num),
                see_source=None,
            )
            agents.append(agent)

    # from IPython import embed; embed(using=False); os._exit(0)



def add4():

    length = 5
    width = 4
    height = 3

    sample_rate = 16000
    segment_samples = sample_rate * 2
    speed_of_sound = 343
    max_order = 3
    half_angle = math.atan2(0.1, 1)
    random_state = np.random.RandomState(1234)

    positive_rays = 4
    total_rays = 20

    x0, fs = librosa.load(path="./resources/p226_001.wav", sr=sample_rate, mono=True)
    x1, fs = librosa.load(path="./resources/p232_006.wav", sr=sample_rate, mono=True)
    a = 0.5

    x0 = repeat_to_length(audio=x0, segment_samples=segment_samples)
    x1 = repeat_to_length(audio=x1, segment_samples=segment_samples)

    source_dict = {
        0: a * x0, 
        1: a * x1,
    }
    sources_num = len(source_dict)

    sources_pos = [
        [1, 1, 1],
        [2, 2, 2],
    ]

    image_meta_list = []

    # Create images
    t1 = time.time()
    for source_index in range(sources_num):

        corners = np.array([[0, 0], [0, width], [length, width], [length, 0]]).T
        # shape: (2, 4)

        room = pra.Room.from_corners(
            corners=corners,
            max_order=max_order,
        )

        room.extrude(height=height)

        source_pos = sources_pos[source_index]
        room.add_source(source_pos)

        room.add_microphone([0.1, 0.1, 0.1])    # dummy

        room.image_source_model()

        images = room.sources[0].images.T
        # (N, 3)

        orders = room.sources[0].orders

        unique_images = []

        for i, image in enumerate(images):

            image = image.tolist()

            if image not in unique_images:
                
                unique_images.append(image)

                meta = {
                    "source_index": source_index,
                    "order": room.sources[0].orders[i],
                    "pos": np.array(image),
                }
                image_meta_list.append(meta)
    
    print("a1", time.time() - t1)

    ############ Mic
    t1 = time.time()
    mic_pos = np.array([0.2, 0.3, 0.4])

    hs_dict = {source_index: [] for source_index in range(sources_num)}

    for image_meta in image_meta_list:

        source_index = image_meta["source_index"]

        direction = image_meta["pos"] - mic_pos
        distance = norm(direction)
        delayed_samples = distance / speed_of_sound * sample_rate

        decay_factor = 1 / distance

        angle_factor = 1.

        normalized_direction = direction / distance

        h = decay_factor * angle_factor * fractional_delay_filter(delayed_samples)
        hs_dict[source_index].append(h)

    y_dict = {}

    for source_index in range(sources_num):

        hs = hs_dict[source_index]
        source = source_dict[source_index]

        max_filter_len = max([len(h) for h in hs])
        
        sum_h = np.zeros(max_filter_len)
        for h in hs:
            sum_h[0 : len(h)] += h

        y = convolve_source_filter(x=source, h=sum_h)

        y_dict[source_index] = y

        soundfile.write(file="_zz{}.wav".format(source_index), data=y, samplerate=16000)
        
    y_total = np.sum([y for y in y_dict.values()], axis=0)

    print("a2", time.time() - t1)

    ###### Agent
    t1 = time.time()
    agent_pos = np.array([0.2, 0.3, 0.4])
    agents = []

    _direction_sampler = DirectionSampler(
        low_colatitude=0, 
        high_colatitude=math.pi, 
        sample_on_sphere_uniformly=False, 
        random_state=random_state,
    )

    # positive
    positive_image_meta_list = filter_image_meta_by_order(image_meta_list, orders=[0])

    if len(positive_image_meta_list) > positive_rays:
        positive_image_meta_list = random_state.choice(positive_image_meta_list, size=positive_num, replace=False)

    for image_meta in positive_image_meta_list:
        
        source_index = image_meta["source_index"]

        agent_to_image = image_meta["pos"] - agent_pos

        agent_look_direction = sample_agent_look_direction(
            agent_to_src=agent_to_image, 
            half_angle=half_angle, 
            random_state=random_state,
        )

        distance = norm(agent_to_image)
        delayed_samples = distance / speed_of_sound * sample_rate
        decay_factor = 1 / distance
        angle_factor = 1.

        h = decay_factor * angle_factor * fractional_delay_filter(delayed_samples)

        y = convolve_source_filter(x=source_dict[source_index], h=h)

        agent = Agent(
            position=agent_pos, 
            look_direction=agent_look_direction, 
            waveform=y,
        )

        agents.append(agent)

    print("a3", time.time() - t1)
    # middle

    # negative
    t1 = time.time()
    while len(agents) < total_rays:

        agent_look_direction = sample_negative_direction(
            agent_pos=agent_pos,
            direction_sampler=_direction_sampler,
            positive_image_meta_list=filter_image_meta_by_order(image_meta_list, orders=[0]),
            half_angle=half_angle,
        )

        agent = Agent(
            position=agent_pos, 
            look_direction=agent_look_direction, 
            waveform=0,
        )
        agents.append(agent)

    print("a4", time.time() - t1)
    from IPython import embed; embed(using=False); os._exit(0)


def filter_image_meta_by_order(image_meta_list, orders):

    new_image_meta_list = []

    for image_meta in image_meta_list:
        if image_meta["order"] in orders:
            new_image_meta_list.append(image_meta)

    return new_image_meta_list


def sample_negative_direction(agent_pos, direction_sampler, positive_image_meta_list, half_angle):

    agent_look_azimuth, agent_look_colatitude = direction_sampler.sample()

    agent_look_direction = np.array(sph2cart(
        r=1., 
        azimuth=agent_look_azimuth, 
        colatitude=agent_look_colatitude
    ))

    while True:
        flag = True

        for image_meta in positive_image_meta_list:

            agent_to_image = image_meta["pos"] - agent_pos

            angle_between_agent_and_src = np.arccos(get_cos(
                agent_look_direction, agent_to_image))

            if angle_between_agent_and_src < half_angle:
                flag = False

        if flag is True:
            return agent_look_direction



def sample_agent_look_direction(agent_to_src, half_angle, random_state):

    _, agent_to_src_azimuth, agent_to_src_colatitude = cart2sph(
        x=agent_to_src[0], 
        y=agent_to_src[1], 
        z=agent_to_src[2],
    )

    rotation_matrix = Rotator3D.get_rotation_matrix_from_azimuth_colatitude(
        azimuth=agent_to_src_azimuth,
        colatitude=agent_to_src_colatitude,
    )

    _direction_sampler = DirectionSampler(
        low_colatitude=0, 
        high_colatitude=half_angle, 
        sample_on_sphere_uniformly=False, 
        random_state=random_state,
    )
    _azimuth, _colatitude = _direction_sampler.sample()

    agent_look_azimuth, agent_look_colatitude = Rotator3D.rotate_azimuth_colatitude(
        rotation_matrix=rotation_matrix,
        azimuth=_azimuth,
        colatitude=_colatitude,
    )

    agent_look_direction = np.array(sph2cart(
        r=1., 
        azimuth=agent_look_azimuth, 
        colatitude=agent_look_colatitude
    ))

    return agent_look_direction


def sample_agent_look_direction2(agent_to_src_direction, half_angle, random_state):

    frames_num = agent_to_src_direction.shape[0]
    
    _direction_sampler = DirectionSampler(
        low_colatitude=0, 
        high_colatitude=half_angle, 
        sample_on_sphere_uniformly=False, 
        random_state=random_state,
    )
    _azimuth, _colatitude = _direction_sampler.sample()

    _, agent_to_src_azimuth, agent_to_src_colatitude = cart2sph(
        x=agent_to_src_direction[:, 0], 
        y=agent_to_src_direction[:, 1], 
        z=agent_to_src_direction[:, 2],
    )

    agent_look_azimuth = np.zeros(frames_num)
    agent_look_colatitude = np.zeros(frames_num)

    for i in range(frames_num):

        rotation_matrix = Rotator3D.get_rotation_matrix_from_azimuth_colatitude(
            azimuth=agent_to_src_azimuth[i],
            colatitude=agent_to_src_colatitude[i],
        )

        agent_look_azimuth[i], agent_look_colatitude[i] = Rotator3D.rotate_azimuth_colatitude(
            rotation_matrix=rotation_matrix,
            azimuth=_azimuth,
            colatitude=_colatitude,
        )

    agent_look_direction = np.stack(sph2cart(
        r=1., 
        azimuth=agent_look_azimuth, 
        colatitude=agent_look_colatitude
    ), axis=-1)

    return agent_look_direction


def add5():

    np.random.seed(1234)
    a = np.random.uniform(low=0, high=3)
    print(a)
    # from IPython import embed; embed(using=False); os._exit(0)


if __name__ == "__main__":

    # add()
    # add2()
    # add3()

    # add4()
    add5()