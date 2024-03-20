import torch
import torch.nn.functional as F


def get_loss(loss_name):
    if loss_name == "detection_distance_sep_loss":
        return detection_distance_sep_loss

    elif loss_name == "detection_distance_sep_direct_reverb_loss":
        return detection_distance_sep_direct_reverb_loss

    elif loss_name == "detection_bce":
        return detection_bce

    else:
        raise NotImplementedError

'''
def detection_distance_sep_loss(output_dict, target_dict):

    det_loss = detection_bce(output_dict, target_dict)
    dist_loss = distance_bce(output_dict, target_dict)
    sep_loss = sep_l1(output_dict, target_dict)

    total_loss = det_loss + dist_loss + sep_loss
    return total_loss
'''

def detection_distance_sep_loss(output_dict, target_dict):

    det_loss = detection_bce(output_dict, target_dict)
    dist_loss = distance_bce(output_dict, target_dict)
    sep_loss = sep_reverb_l1(output_dict, target_dict)

    total_loss = det_loss + dist_loss + sep_loss

    return total_loss


def detection_distance_sep_direct_reverb_loss(output_dict, target_dict):

    det_loss = detection_bce(output_dict, target_dict)
    dist_loss = distance_bce(output_dict, target_dict)
    sep_direct_loss = sep_direct_l1(output_dict, target_dict)
    sep_reverb_loss = sep_reverb_l1(output_dict, target_dict)

    total_loss = det_loss + dist_loss + sep_direct_loss + sep_reverb_loss

    return total_loss


def detection_bce(output_dict, target_dict):
    loss = F.binary_cross_entropy(
        input=output_dict['agent_look_at_direction_has_source'], 
        target=target_dict['agent_look_at_direction_has_source'],
    )
    return loss


def distance_bce(output_dict, target_dict):
    loss = F.binary_cross_entropy(
        input=output_dict['agent_look_at_distance_has_source'], 
        target=target_dict['agent_look_at_distance_has_source'],
    )
    return loss


def sep_reverb_l1(output_dict, target_dict):
    loss = sep_reverb_l1(
        x=output_dict["agent_look_at_direction_reverb_wav"],
        y=target_dict["agent_look_at_direction_reverb_wav"],
    )
    return loss


def sep_direct_l1(output_dict, target_dict):
    loss = sep_reverb_l1(
        x=output_dict["agent_look_at_direction_direct_wav"],
        y=target_dict["agent_look_at_direction_direct_wav"],
    )
    return loss



def l1(x, y):
    loss = torch.mean(torch.abs(x - y))
    return loss