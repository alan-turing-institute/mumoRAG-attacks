import torch
from strenum import StrEnum


class DefenceName(StrEnum):
    NONE = ""
    PARAPHRASE = "paraphrase-queries"
    NOISE = "add-noise"


def add_noise(image, noise_level):
    image = image.float()
    torch.clip(image + noise_level * torch.randn_like(image), min=0, max=255, out=image)
    image = image.type(torch.uint8)
    return image
