import os
import torch
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
from collections import defaultdict
from datasets import Dataset


def get_device(prefer_mps=False):
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and prefer_mps:
        return "mps"
    return "cpu"

def print_memory_consumption(device):
    if device == "mps":
        allocated_ram = torch.mps.current_allocated_memory()
        print(f"Allocated RAM: {allocated_ram/1e9:.2f} GB")
    elif device == "cuda":
        allocated_ram = torch.cuda.memory_allocated(0)
        max_allocated_ram = torch.cuda.max_memory_allocated(0)
        print(f"Current allocated RAM: {allocated_ram/1e9:.2f} GB, max: {max_allocated_ram/1e9:.2f} GB")


# converts all pdfs in the specified folder to a list of images (one per page) 
def convert_pdfs_to_images(pdf_folder):
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    print(pdf_files)
    all_images = {}

    for doc_id, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        images = convert_from_path(pdf_path)
        all_images[doc_id] = images

    return all_images

# plot a list of images side by side
def plot_images(images, n_subplots):
    fig, axes = plt.subplots(1, min(len(images),n_subplots), figsize=(15, 10))

    for i, ax in enumerate(axes.flat):
        img = images[i]
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# cretaes a huggingface datset given a dict of pdfs, each pdf is a list of images 
def create_hf_dataset(images: dict):

    ds_dict = defaultdict(list)
    for i, pdf in images.items():
        for j, image in enumerate(pdf):
            ds_dict['file'].append(i)
            ds_dict['page'].append(j)
            ds_dict['image'].append(image)

    return Dataset.from_dict(ds_dict)

# reverse the effect of image normalization
def unnormalize_image(image, processor):
    image_mean, image_std = processor.image_processor.image_mean, processor.image_processor.image_std
    image_mean = torch.tensor(image_mean).to(image.device)
    image_std = torch.tensor(image_std).to(image.device)
    image2 = image * image_std.unsqueeze(-1).unsqueeze(-1) + image_mean.unsqueeze(-1).unsqueeze(-1)
    return image2.clip(0,1)

# get the min and max values of an image processor
def get_min_max_image(processor):
    image_mean, image_std = processor.image_processor.image_mean, processor.image_processor.image_std
    image_mean = torch.tensor(image_mean)
    image_std = torch.tensor(image_std)

    rgb_range = [(0-image_mean)/image_std, (1-image_mean)/image_std]
    rgb_range[0] = rgb_range[0].unsqueeze(-1).unsqueeze(-1)#.repeat(1,224,224)
    rgb_range[1] = rgb_range[1].unsqueeze(-1).unsqueeze(-1)#.repeat(1,224,224)

    return rgb_range

def normalize_image(image, processor):
    image_mean = torch.tensor(processor.image_processor.image_mean).unsqueeze(-1).unsqueeze(-1).to(image.device)
    image_std = torch.tensor(processor.image_processor.image_std).unsqueeze(-1).unsqueeze(-1).to(image.device)
    return (image - image_mean) / image_std