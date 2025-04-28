from enum import Enum

import torch


class AttackMask(Enum):
    Full = [512,512,0,0]
    FirstQuadrant = [256,256,0,0]
    SecondQuadrant = [256, 256, 256, 0]
    ThirdQuadrant = [256, 256, 0, 256]
    FourthQuadrant = [256, 256, 256, 256]

def get_attack_mask(attack_mask:AttackMask, raw_image:torch.Tensor):
    if attack_mask is None:
        # If no mask is provided then perturb all pixels
        mask_tensor = torch.ones_like(raw_image)
    else:
        # Make mask tensor with the same shape as the raw_image
        mask_tensor = torch.zeros(raw_image[0].shape)
        x_start = attack_mask.value[2]
        x_end = x_start + attack_mask.value[0]
        y_start = attack_mask.value[3]
        y_end = y_start + attack_mask.value[1]
        mask_tensor[x_start:x_end, y_start:y_end] = 1

        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        if mask_tensor.shape[0] == 1 and raw_image.shape[0] > 1:
            mask_tensor = mask_tensor.expand(raw_image.shape[0], -1, -1)

    return mask_tensor