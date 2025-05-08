from enum import Enum

import torch


class AttackMask(Enum):
    # Mask = [% of X, % of Y, Starting X%, Starting Y%]
    Full = [1,1,0,0]
    FirstQuadrant = [0.5,0.5,0,0]
    SecondQuadrant = [0.5, 0.5, 0.5, 0]
    ThirdQuadrant = [0.5, 0.5, 0, 0.5]
    FourthQuadrant = [0.5, 0.5, 0.5, 0.5]
    Figure = [0.1953, 0.0977, 0.3125, 0.5273]

def get_attack_mask(attack_mask:AttackMask, raw_image:torch.Tensor, image_size:list[int]):
    if attack_mask is None:
        # If no mask is provided then perturb all pixels
        mask_tensor = torch.ones_like(raw_image)
    else:
        # Make mask tensor with the same shape as the raw_image
        mask_tensor = torch.zeros(raw_image[0].shape)
        x_start = int(attack_mask.value[2] * image_size[0])
        x_end = int(x_start + attack_mask.value[0] * image_size[0])
        y_start = int(attack_mask.value[3] * image_size[1])
        y_end = int(y_start + attack_mask.value[1] * image_size[1])
        mask_tensor[x_start:x_end, y_start:y_end] = 1

        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        if mask_tensor.shape[0] == 1 and raw_image.shape[0] > 1:
            mask_tensor = mask_tensor.expand(raw_image.shape[0], -1, -1)

    return mask_tensor