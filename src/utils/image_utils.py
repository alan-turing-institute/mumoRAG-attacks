import torch
from transformers import AutoProcessor
import torchvision.transforms as T
from enum import IntEnum

"""
Meaning of resample ints fetched from https://github.com/python-pillow/Pillow/blob/main/src/PIL/Image.py#L164
(Needed when resizing)
class Resampling(IntEnum):
    NEAREST = 0
    BOX = 4
    BILINEAR = 2
    HAMMING = 5
    BICUBIC = 3
    LANCZOS = 1
"""

def process_image(image: torch.tensor, processor: AutoProcessor):
    """
    Simulates the functionalitly of the Huggingface Processor call:
    >>> processor(images=[image], return_tensors="pt").to(device)

    According to https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/image_processing_siglip.py
    the order of operations is:
    1. resize
    2. center crop
    3. rescale
    4. normalize

    TODO: this is still not perfect, as converting our adversarial images through the HF processor kinda removes the attack. Maybe related to processor["resample"] (Lanczos resampling not implemented in PyTorch)?
    """
    p = processor.image_processor

    # Lanczos resampling (code 1) is not implemented by pytorch, so we replace by bicubic (3)
    if p.__dict__.get("resample",-1) != -1:
        if p.resample == 1: p.resample = 3 # bicubic

    # Resizing
    if p.do_resize == True:
        _,h,w = image.shape
        if p.size.get("height", -1) != -1:
            image_resized = T.Resize([p.size['height'],p.size['width']], interpolation=p.resample)(image)
        elif p.__dict__.get("max_image_size", -1) != -1:
            image_resized = T.Resize([p.max_image_size['longest_edge'], p.max_image_size['longest_edge']], interpolation=3)(image)
        elif p.size.get("shortest_edge", -1) != -1: #and p.__dict__.get("do_center_crop", -1) != True:
            image_resized = T.Resize(p.size['shortest_edge'], interpolation=p.resample)(image)
        else:
            image_resized = image
    else:
        image_resized = image

    # center cropping
    if p.__dict__.get("do_center_crop", -1) == True:
        image_cped = center_crop(image_resized, size=[p.crop_size['height'], p.crop_size['width']])
        # image_cped = T.CenterCrop([p.crop_size['height'], p.crop_size['width']])(image)
    else:
        image_cped = image_resized
    
    # Recaling
    if p.rescale_factor:
        image_rescaled = image_cped*p.rescale_factor
    else:
        image_rescaled = image_cped

    # Normalization
    if p.do_normalize == True:
        image_mean = torch.tensor(p.image_mean).to(image.device).unsqueeze(-1).unsqueeze(-1)
        image_std = torch.tensor(p.image_std).to(image.device).unsqueeze(-1).unsqueeze(-1)
        image_normalized = (image_rescaled - image_mean) / image_std
    else:
        image_normalized = image_rescaled
    
    return image_normalized


def center_crop(image: torch.tensor, size):
    """
    An attempt to mimic transformer's center crop transformation from:
    https://github.com/huggingface/transformers/blob/main/src/transformers/image_transforms.py
    """
    _,h,w = image.shape
    ch,cw = size[0], size[1]

    # In case size is odd, (image_shape[0] + size[0]) // 2 won't give the proper result.
    top = (h - ch) // 2
    bottom = top + ch
    # In case size is odd, (image_shape[1] + size[1]) // 2 won't give the proper result.
    left = (w - cw) // 2
    right = left + cw

    # Check if cropped area is within image boundaries
    if top >= 0 and bottom <= h and left >= 0 and right <= w:
        image = image[..., top:bottom, left:right]
        return image
    
    # Otherwise, we may need to pad if the image is too small. Oh joy...
    # TODO: adjust original code from https://github.com/huggingface/transformers/blob/main/src/transformers/image_transforms.py





if __name__ == "__main__":
    """
    Here we test the effectiveness of our "differentiable preprocessing `process_image() function, as compared to the HF processor`"
    """
    from transformers.image_utils import load_image
    from utils import plot_images

    vlm_model_name = "HuggingFaceTB/SmolVLM-256M-Instruct" # "HuggingFaceTB/SmolVLM-256M-Instruct" or "openai/clip-vit-base-patch16", 
    # image = load_image("https://raulperez.tieneblog.net/wp-content/uploads/2015/09/tux.jpg")
    image = load_image("https://farm9.staticflickr.com/8096/8445896722_e28fb3f055_z.jpg")
    image_tensor = T.PILToTensor()(image)
    image_tensor = image_tensor.float()
    image_tensor.requires_grad = True
    processor = AutoProcessor.from_pretrained(vlm_model_name)
    processor.image_processor.do_image_splitting = False
    image_p_hf = processor(images=[image], return_tensors="pt")['pixel_values']
    image_p_me = process_image(image_tensor, processor)

    if len(image_p_hf.shape) == 5: image_p_hf = image_p_hf[0][0]
    if len(image_p_hf.shape) == 4: image_p_hf = image_p_hf[0]
    print(f"Size HF: {image_p_hf.shape}, Size ME: {image_p_me.shape}")
    print(f"MSE: {torch.nn.functional.mse_loss(image_p_me, image_p_hf)}")
    print(f"Linf: {(image_p_hf - image_p_me).norm(p=float('inf'))}")
    print(f"Diff: {(image_p_hf - image_p_me)[0][:4,:4]}")
    print(f"Grad HF: {image_p_hf.requires_grad}, Grad ME: {image_p_me.requires_grad}")
    plot_images([T.ToPILImage()(image_p_hf/2+0.5), T.ToPILImage()(image_p_me/2+0.5)], n_subplots=2)