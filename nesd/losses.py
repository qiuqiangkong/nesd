import torch.nn.functional as F
import torch

from nesd.utils import PAD

'''
def loc_bce(model, output_dict, target_dict):
    loss = F.binary_cross_entropy(
        input=output_dict['agent_see_source'], 
        target=target_dict['agent_see_source'],
    )
    return loss
'''

'''
def l1(x, y, mask):
    from IPython import embed; embed(using=False); os._exit(0)
    loss = torch.sum(torch.abs(x - y) * mask) / torch.clamp(torch.sum(mask), 1e-4)
    return loss
    # return torch.mean(torch.abs(x - y))
'''

def loc_bce(output_dict, target_dict):
    loss = F.binary_cross_entropy(
        input=output_dict['agent_look_directions_has_source'], 
        target=target_dict['agent_look_directions_has_source'],
    )
    return loss


# def loc_bce_mask(output_dict, target_dict, mask):

#     output = output_dict['agent_look_directions_has_source']
#     target = target_dict['agent_look_directions_has_source']

def bce_mask(output, target, mask):

    eps = 1e-7
    output = torch.clamp(output, eps, 1. - eps)
    matrix = - target * torch.log(output) - (1. - target) * torch.log(1. - output)
    loss = torch.sum(matrix * mask) / torch.clamp(torch.sum(mask), 1e-4)

    return loss


def loc_bce_depth_bce(output_dict, target_dict):

    agent_look_depths_mask = (target_dict['agent_look_depths_has_source'] != PAD) * 1.

    loc_bce_loss = loc_bce(output_dict, target_dict)

    depth_bce_loss = bce_mask(
        output=output_dict['agent_look_depths_has_source'], 
        target=target_dict['agent_look_depths_has_source'], 
        mask=agent_look_depths_mask
    )
    # from IPython import embed; embed(using=False); os._exit(0)
    loss = loc_bce_loss + depth_bce_loss

    return loss


def loc_bce_cla_bce(output_dict, target_dict):

    agent_sed_mask = (target_dict['agent_sed_mask'] != PAD) * 1.
    # agent_sed_mask = agent_sed_mask[:, :, :, None].repeat(1, 1, 1, )

    loc_bce_loss = loc_bce(output_dict, target_dict)

    cla_bce_loss = bce_mask(
        output=output_dict['agent_sed'], 
        target=target_dict['agent_sed'], 
        mask=agent_sed_mask
    )
    
    loss = loc_bce_loss + cla_bce_loss
    # from IPython import embed; embed(using=False); os._exit(0)

    return loss


def l1(x, y, mask):
    loss = torch.sum(torch.abs(x - y) * mask[:, :, None]) / torch.clamp(torch.sum(mask), 1e-4) / x.shape[-1]
    return loss


def loc_bce_sep_l1(output_dict, target_dict):
    loc_bce_loss = loc_bce(output_dict, target_dict)
    
    sep_l1_loss = l1(
        x=output_dict["agent_signals"],
        y=target_dict["agent_signals"],
        mask=target_dict["agent_active_indexes_mask"]
    )

    # soundfile.write(file="_zz.wav", data=output_dict["agent_signals"].data.cpu().numpy()[0, 2], samplerate=24000)
    # from IPython import embed; embed(using=False); os._exit(0)

    sep_l1_loss *= 10
    loss = loc_bce_loss + sep_l1_loss
    return loss


###################
'''
def depth_bce(model, output_dict, target_dict):
    loss = F.binary_cross_entropy(
        input=output_dict['agent_exist_source'], 
        target=target_dict['agent_exist_source'],
    )
    return loss


def loc_bce_classwise_bce(model, output_dict, target_dict):
    loc_loss = F.binary_cross_entropy(
        input=output_dict['agent_see_source'], 
        target=target_dict['agent_see_source'],
    )
    classwise_loss = F.binary_cross_entropy(
        input=output_dict['agent_see_source_classwise'], 
        target=target_dict['agent_see_source_classwise'],
    )
    total_loss = loc_loss + classwise_loss
    print(loc_loss.item(), classwise_loss.item())
    
    return total_loss


def sep_l1(model, output_dict, target_dict):
    loss = torch.mean(torch.abs(output_dict['agent_waveform'] - target_dict['agent_waveform']))
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
    
    print(loc_loss.item(), sep_loss.item())

    return total_loss


def loc_bce_sep_l1_depth_bce(model, output_dict, target_dict):

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

    depth_loss = depth_bce(
        model=model, 
        output_dict=output_dict, 
        target_dict=target_dict
    )

    sep_loss *= 10.

    total_loss = loc_loss + sep_loss + depth_loss
    
    print(loc_loss.item(), sep_loss.item(), depth_loss.item())

    return total_loss


def classwise_bce(model, output_dict, target_dict):

    classwise_loss = F.binary_cross_entropy(
        input=output_dict['classwise_output'], 
        target=target_dict['target'],
    )
    
    return classwise_loss


def classwise_bce_mul_agents(model, output_dict, target_dict):

    classwise_loss = F.binary_cross_entropy(
        input=output_dict['classwise_output'], 
        target=target_dict['agent_see_source_classwise'],
    )
    
    return classwise_loss
'''