import torch.nn.functional as F


def intersect_source_bce(model, output_dict, target_dict):
    loss = F.binary_cross_entropy(
    	input=output_dict['ray_intersect_source'], 
    	target=target_dict['ray_intersect_source']
    )
    return loss