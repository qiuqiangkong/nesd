import numpy as np
import h5py
import os


def add():

	hdf5_path = "/home/tiger/workspaces/nesd2/hdf5s/dcase2019_task3/sr=24000/test/split0_1.h5"

	with h5py.File(hdf5_path, 'r') as hf:

		frame_indexes = hf['frame_index'][:]
		event_indexes = hf['event_index'][:]
		class_indexes = hf['class_index'][:]
		azimuths = hf['azimuth'][:].astype(np.int32)
		elevations = hf['elevation'][:].astype(np.int32)

	out_csv_path = "./d19t3_pred/split0_1.csv"
	os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)

	with open(out_csv_path, 'w') as fw:
		for n in range(len(frame_indexes)):

			for i in range(5):
				fw.write("{},{},{},{}\n".format(frame_indexes[n] * 5 + i, class_indexes[n], azimuths[n], elevations[n]))
			

	from IPython import embed; embed(using=False); os._exit(0)


if __name__ == '__main__':
	add()