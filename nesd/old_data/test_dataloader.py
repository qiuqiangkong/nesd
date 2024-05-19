# import lightning.pytorch as pl
from typing import List, Dict, NoReturn, Callable, Union, Optional
import torch
from torch.utils.data import DataLoader
import random
import soundfile
import numpy as np
from nesd.old_data.image_source_simulator import ImageSourceSimulator
# from nesd.old_data.em_rir import _DatasetEmRir
# from nesd.test_plot import plot_top_view


# class AudioSetDataset(Dataset):
#     def __init__(self, features_dir):
        
#         self.features_dir = features_dir
#         self.feature_names = os.listdir(features_dir)
#         self.features_num = len(self.feature_names)
        
#     def __getitem__(self, index):
        
#         feature_name = self.feature_names[index]
#         feature_path = os.path.join(self.features_dir, feature_name)
        
#         with h5py.File(feature_path, 'r') as hf:
#             feature = hf['feature'][:]
#             target = hf['target'][:].astype(np.float32)
#             audio_id = hf.attrs['audio_id'].decode()
            
#         return feature, target
        
#     def __len__(self):
#         return self.features_num



class Dataset3:
    def __init__(self, audios_dir, expand_frames=None, simulator_configs=None):
        self.audios_dir = audios_dir
        self.expand_frames = expand_frames
        self.simulator_configs = simulator_configs

    def __getitem__(self, meta):
        # print(meta)
        
        iss_data = ImageSourceSimulator(
            audios_dir=self.audios_dir, 
            expand_frames=self.expand_frames,
            simulator_configs=self.simulator_configs
        )

        data = {
            "room_length": iss_data.length,
            "room_width": iss_data.width,
            "room_height": iss_data.height,
            "source_positions": iss_data.source_positions,
            "source_signals": iss_data.sources,
            "mic_positions": iss_data.mic_positions,
            "mic_look_directions": iss_data.mic_look_directions,
            "mic_signals": iss_data.mic_signals,
            "agent_positions": iss_data.agent_positions,
            "agent_look_directions": iss_data.agent_look_directions,
            "agent_look_depths": iss_data.agent_look_depths,
            # "agent_signals": iss_data.agent_waveforms,
            "agent_look_directions_has_source": iss_data.agent_look_directions_has_source,
            "agent_look_depths_has_source": iss_data.agent_look_depths_has_source,
            # "agent_ray_types": iss_data.agent_ray_types,
            "agent_active_indexes": iss_data.active_indexes,
            "agent_active_indexes_mask": iss_data.active_indexes_mask,
            "agent_signals": iss_data.agent_waveforms,
            "agent_signals_echo": iss_data.agent_waveforms_echo
        }

        if hasattr(iss_data, "sed"):
            data["agent_sed"] = iss_data.sed
            data["agent_sed_mask"] = iss_data.sed_mask

        return data

    # def __len__(self):
    #     return 10000


'''
class Dataset3:
    def __init__(self, audios_dir, expand_frames=None, simulator_configs=None):
        self.audios_dir = audios_dir
        self.expand_frames = expand_frames
        self.simulator_configs = simulator_configs

        self.iss_data = ImageSourceSimulator(
            audios_dir=self.audios_dir, 
            expand_frames=self.expand_frames,
            simulator_configs=self.simulator_configs
        )

    def __getitem__(self, meta):
        # print(meta)
        
        self.iss_data.sample()

        from IPython import embed; embed(using=False); os._exit(0)

        data = {
            "room_length": iss_data.length,
            "room_width": iss_data.width,
            "room_height": iss_data.height,
            "source_positions": iss_data.source_positions,
            "source_signals": iss_data.sources,
            "mic_positions": iss_data.mic_positions,
            "mic_look_directions": iss_data.mic_look_directions,
            "mic_signals": iss_data.mic_signals,
            "agent_positions": iss_data.agent_positions,
            "agent_look_directions": iss_data.agent_look_directions,
            "agent_look_depths": iss_data.agent_look_depths,
            # "agent_signals": iss_data.agent_waveforms,
            "agent_look_directions_has_source": iss_data.agent_look_directions_has_source,
            "agent_look_depths_has_source": iss_data.agent_look_depths_has_source,
            # "agent_ray_types": iss_data.agent_ray_types,
            "agent_active_indexes": iss_data.active_indexes,
            "agent_active_indexes_mask": iss_data.active_indexes_mask,
            "agent_signals": iss_data.agent_waveforms
        }

        if hasattr(iss_data, "sed"):
            data["agent_sed"] = iss_data.sed
            data["agent_sed_mask"] = iss_data.sed_mask

        return data

    # def __len__(self):
    #     return 10000
'''

class DatasetEmRir:
    def __init__(self, audios_dir, rir_hdf5s_dir, expand_frames=None, simulator_configs=None):
        self.audios_dir = audios_dir
        self.rir_hdf5s_dir = rir_hdf5s_dir
        self.expand_frames = expand_frames
        self.simulator_configs = simulator_configs

    def __getitem__(self, meta):
        # print(meta)
        
        iss_data = _DatasetEmRir(
            audios_dir=self.audios_dir, 
            rir_hdf5s_dir=self.rir_hdf5s_dir,
            expand_frames=201, 
            simulator_configs=self.simulator_configs
        )

        data = {
            "room_length": iss_data.length,
            "room_width": iss_data.width,
            "room_height": iss_data.height,
            "source_positions": iss_data.source_positions,
            "source_signals": iss_data.sources,
            "mic_positions": iss_data.mic_positions,
            "mic_look_directions": iss_data.mic_look_directions,
            "mic_signals": iss_data.mic_signals,
            "agent_positions": iss_data.agent_positions,
            "agent_look_directions": iss_data.agent_look_directions,
            "agent_look_depths": iss_data.agent_look_depths,
            # "agent_signals": iss_data.agent_waveforms,
            "agent_look_directions_has_source": iss_data.agent_look_directions_has_source,
            "agent_look_depths_has_source": iss_data.agent_look_depths_has_source,
            # "agent_ray_types": iss_data.agent_ray_types,
            "agent_active_indexes": iss_data.active_indexes,
            "agent_active_indexes_mask": iss_data.active_indexes_mask,
            "agent_signals": iss_data.agent_waveforms
        }

        return data


def collate_fn(list_data_dict):
    data_dict = {}

    for key in list_data_dict[0].keys():
        
        data_dict[key] = [dd[key] for dd in list_data_dict]
        # from IPython import embed; embed(using=False); os._exit(0)

        try:
            if key in ["mic_positions", "mic_look_directions", "mic_signals", "agent_positions", "agent_look_directions", "agent_look_depths", "agent_look_directions_has_source", "agent_look_depths_has_source", "agent_active_indexes_mask", "agent_signals", "agent_sed", "agent_sed_mask"]:#, "agent_signals_echo"]:
                data_dict[key] = torch.Tensor(np.stack(data_dict[key], axis=0))

            elif key in ["agent_active_indexes"]:
                data_dict[key] = torch.LongTensor(np.stack(data_dict[key], axis=0))
        except:
            from IPython import embed; embed(using=False); os._exit(0) 
    
    return data_dict


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):

        print(i)
        d = random.choice(self.datasets)

        return d[i]


    def __len__(self):
        # from IPython import embed; embed(using=False); os._exit(0)
        # return min(len(d) for d in self.datasets)
        return 10000


def add():
    train_dataset = Dataset1()
    num_workers = 0

    # print(dataset["a1"])

    # sampler
    # train_sampler = BalancedSampler(
    #     indexes_hdf5_path=indexes_hdf5_path,
    #     batch_size=batch_size,
    #     steps_per_epoch=steps_per_epoch,
    # )

    # data module
    data_module = DataModule(
        # train_sampler=train_sampler,
        train_dataset=train_dataset,
        num_workers=num_workers,
    )

    data_module.setup()

    for batch_data_dict in data_module.train_dataloader():
        print(batch_data_dict)


# test concate datasets
def add2():

    train_dataset1 = Dataset1()
    train_dataset2 = Dataset2()
    num_workers = 0
    batch_size = 8

    concat_dataset = ConcatDataset(
        train_dataset1, train_dataset2
    )

    data_loader = torch.utils.data.DataLoader(
        concat_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    for data in data_loader:
        print(data)


def add3():

    train_dataset1 = Dataset3()
    num_workers = 0
    batch_size = 8

    data_loader = torch.utils.data.DataLoader(
        train_dataset1,
        # batch_sampler=,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    for i, data in enumerate(data_loader):
        n = 0
        plot_top_view(data["room_length"][n], data["room_width"][n], data["room_height"][n], data["source_positions"][n], data["mic_positions"][n], data["agent_positions"][n], data["agent_look_directions"][n], data["agent_ray_types"][n])


        x = data["mic_signals"][0][0].data.cpu().numpy()
        soundfile.write(file="_zz.wav", data=x, samplerate=24000) 
        from IPython import embed; embed(using=False); os._exit(0)
        print(i)


def add4():
    train_dataset1 = Dataset3(expand_frames=201)
    num_workers = 0
    batch_size = 8

    data_loader = torch.utils.data.DataLoader(
        train_dataset1,
        # batch_sampler=,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    for i, data in enumerate(data_loader):
        from IPython import embed; embed(using=False); os._exit(0)

if __name__ == "__main__": 

    # add()
    # add2()
    add3()

    # add4()