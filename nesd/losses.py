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
    
    return total_loss