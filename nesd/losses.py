import torch.nn.functional as F
import torch


def loc_bce(model, output_dict, target_dict):
    loss = F.binary_cross_entropy(
        input=output_dict['agent_see_source'], 
        target=target_dict['agent_see_source'],
    )
    print(loss.item())
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