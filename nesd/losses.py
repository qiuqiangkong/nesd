import torch
import torch.nn.functional as F


def get_loss(loss_name):
    if loss_name == "detection_distance_sep_loss":
        return detection_distance_sep_loss

    elif loss_name == "detection_distance_sep_direct_reverb_loss":
        return detection_distance_sep_direct_reverb_loss

    elif loss_name == "detection_bce":
        return detection_bce

    elif loss_name == "detection_bce_srcs_num_ce":
        return detection_bce_srcs_num_ce

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
    
    print(det_loss.item(), dist_loss.item(), sep_loss.item())
    # total_loss = detection_bce(output_dict, target_dict)

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
    loss = l1(
        x=output_dict["agent_look_at_direction_reverb_wav"],
        y=target_dict["agent_look_at_direction_reverb_wav"],
    )
    return loss


def sep_direct_l1(output_dict, target_dict):
    loss = l1(
        x=output_dict["agent_look_at_direction_direct_wav"],
        y=target_dict["agent_look_at_direction_direct_wav"],
    )
    return loss



def l1(x, y):
    loss = torch.mean(torch.abs(x - y))
    return loss


def sources_num_ce(output_dict, target_dict):
    
    loss = F.nll_loss(
        input=output_dict["sources_num"].flatten(0, 1),
        target=target_dict["sources_num"].flatten()
    )
    return loss


def detection_bce_srcs_num_ce(output_dict, target_dict):
    
    det_loss = detection_bce(output_dict, target_dict)
    srcs_num_loss = sources_num_ce(output_dict, target_dict)

    total_loss = det_loss + srcs_num_loss
    
    return total_loss