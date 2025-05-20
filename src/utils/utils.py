import os
import torch
from pdf2image import convert_from_path
import matplotlib.pyplot as plt

from .logger import logger


def get_device(prefer_mps=False):
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and prefer_mps:
        return "mps"
    return "cpu"

def get_memory_consumption(device):
    if device == "mps":
        return torch.mps.current_allocated_memory()/1e9
    elif device == "cuda":
        return torch.cuda.memory_allocated(0)/1e9
    raise NotImplementedError

def print_memory_consumption(device):
    if device == "mps":
        allocated_ram = torch.mps.current_allocated_memory()
        logger.info(f"Allocated RAM: {allocated_ram/1e9:.2f} GB")
        return
    elif device == "cuda":
        allocated_ram = torch.cuda.memory_allocated(0)
        max_allocated_ram = torch.cuda.max_memory_allocated(0)
        logger.info(f"Current allocated RAM: {allocated_ram/1e9:.2f} GB, max: {max_allocated_ram/1e9:.2f} GB")
        return
    raise NotImplementedError


# plot a list of images side by side
def plot_images(images, n_subplots):
    fig, axes = plt.subplots(1, min(len(images),n_subplots), figsize=(15, 10))

    for i, ax in enumerate(axes.flat):
        img = images[i]
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
