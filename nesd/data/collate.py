import numpy as np
import torch
import librosa

'''
def collate_fn(list_data_dict):
    """Convert list of data to Tensor."""

    data_dict = {}

    for key in list_data_dict[0].keys():

        if key in ["source", "source_position"]:
            data_dict[key] = [dd[key] for dd in list_data_dict]
        else:
            data_dict[key] = torch.Tensor(
                np.stack([dd[key] for dd in list_data_dict], axis=0)
            )

    return data_dict
'''

def collate_fn(list_data_dict):
    """Convert list of data to Tensor."""

    data_dict = {}

    for key in list_data_dict[0].keys():

        try:
            # if key in ["source", "source_position"]:
            if key in ["source", "source_position", "source_positions", "source_signals", "agent_signals_echo"]:
                data_dict[key] = [dd[key] for dd in list_data_dict]

            elif key in ["sources_num"]:
                data_dict[key] = torch.LongTensor(
                    np.stack([dd[key] for dd in list_data_dict], axis=0)
                )
            else:
                data_dict[key] = torch.Tensor(
                    np.stack([dd[key] for dd in list_data_dict], axis=0)
                )
        except:
            from IPython import embed; embed(using=False); os._exit(0)

    return data_dict