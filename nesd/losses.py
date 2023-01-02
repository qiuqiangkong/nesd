import torch.nn.functional as F
import torch


def loc_bce(model, output_dict, target_dict):
    loss = F.binary_cross_entropy(
        input=output_dict['ray_intersect_source'], 
        target=target_dict['ray_intersect_source']
    )
    return loss


def sep_l1(model, output_dict, target_dict):
    loss = torch.mean(torch.abs(output_dict['ray_waveform'] - target_dict['ray_waveform']))
    return loss

def loc_bce_sep_l1(model, output_dict, target_dict):

    loc_loss = loc_bce(
        model=model, 
        output_dict=output_dict, 
        target_dict=target_dict
    )

    sep_loss = sep_l1(
        model=model, 
        output_dict=output_dict, 
        target_dict=target_dict
    )

    sep_loss *= 10.

    total_loss = loc_loss + sep_loss
    # total_loss = sep_loss

    print(torch.mean(torch.abs(target_dict['ray_waveform'])).item(), torch.mean(torch.abs(output_dict['ray_waveform'])).item())
    # print(torch.max(target_dict['ray_waveform']).item(), torch.max(output_dict['ray_waveform']).item())
    print(loc_loss.item(), sep_loss.item())
    # from IPython import embed; embed(using=False); os._exit(0)
    # import numpy as np
    # np.max(target_dict['ray_waveform'].data.cpu().numpy(), axis=-1)
    # import soundfile
    # soundfile.write(file='_zz.wav', data=target_dict['ray_waveform'].data.cpu().numpy()[0, 0], samplerate=24000)
    # soundfile.write(file='_zz2.wav', data=target_dict['ray_waveform'].data.cpu().numpy()[0, 1], samplerate=24000) 
    
    return total_loss