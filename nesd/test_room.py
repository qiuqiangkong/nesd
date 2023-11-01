# class Room:

# 	def __init__(self):
# 		pass


import time
import soundfile

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import librosa

import pyroomacoustics as pra
from room import Room
from nesd.utils import norm, fractional_delay_filter

width = 4
corners = np.array([[0, 0], [0, width], [width, width], [width, 0]]).T
room = pra.Room.from_corners(
	corners=corners,
	max_order=3,
)

height = 2
room.extrude(height=height)

room.add_source([1, 1, 1])

room.add_microphone([1, 1, 1])

room.image_source_model()
# room.sources[0].images.shape

images = room.sources[0].images.T

tmp = [str(image) for image in images]
# print(len(tmp))
# print(len(set(tmp)))

images = np.array(list(set(map(tuple, images))))

microphone_xyz = np.array([0.2, 0.3, 0.4])

# delayed_samples = None
sample_rate = 16000
speed_of_sound = 343

x, fs = librosa.load(path="./resources/p226_001.wav", sr=sample_rate, mono=True)

hs = []

t1 = time.time()
for image in images:
	# t2 = time.time()
	direction = image - microphone_xyz
	distance = norm(direction)
	delayed_samples = distance / speed_of_sound * sample_rate
	# print("b1", time.time() - t2)

	# t2 = time.time()
	decay_factor = 1 / distance
	# print(image, decay_factor)
	h = decay_factor * fractional_delay_filter(delayed_samples)
	hs.append(h)
	# print("b2", time.time() - t2)

print(time.time() - t1)
	
t1 = time.time()
max_filter_len = max([len(h) for h in hs])
# hs = [librosa.util.fix_length(data=h, size=max_filter_len, axis=0) for h in hs]
# sum_h = np.sum(hs, axis=0)
sum_h = np.zeros(max_filter_len)
for h in hs:
	sum_h[0 : len(h)] += h
print(time.time() - t1)

t1 = time.time()
y = np.convolve(x, sum_h, mode='full')
print(time.time() - t1)

plt.plot(sum_h)
plt.savefig("_zz.pdf")

soundfile.write(file="_zz.wav", data=y, samplerate=sample_rate)
from IPython import embed; embed(using=False); os._exit(0)