from typing import TYPE_CHECKING

import torch
from transformers import AutoProcessor, LlavaNextProcessor
import torchvision.transforms.v2 as T
from PIL import Image

from .logger import logger
from config import DATA_FOLDER

if TYPE_CHECKING:
    from config.task import TaskConfig
    from config.train import ExperimentTrainConfig

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

def process_image(image: torch.tensor, model_instance):
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
    
    Potential sources of errors:
    1. (FAIL) resize:       Significant MSEs even when all other transforms are disabled
    2. (PASS) rescale:      MSE ~ 1e-16 when all other transforms are disabled
    3. (PASS) Normalize:    MSE=0 when all other transforms are disabled
    4. (PASS) Center Crop:  MSE=0 when all other transforms are disabled
    5. (PASS) Padding:      MSE=0 when all other transforms are disabled
    - When only resize is disabled --> MSE ~ 1e-15 (for both CLIP and SmolVLM)
    NOTE: this a known issue https://github.com/pytorch/vision/issues/2950
    NOTE: when resample = 0 -> we get 0 erorrs (only if we do not need resizing)
    """
    processor = model_instance.processor
    p = processor.image_processor

    # Lanczos resampling (code 1) is not implemented by pytorch, so we replace by bicubic (3)
    if p.__dict__.get("resample",-1) != -1:
        if p.resample == 1: p.resample = 3 # bicubic

    # Resizing
    if p.do_resize == True:
        _,h,w = image.shape
        if p.size.get("height", -1) != -1:
            image = T.Resize([p.size['height'],p.size['width']], interpolation=p.resample)(image)
        elif p.__dict__.get("max_image_size", -1) != -1:
            image = T.Resize([p.max_image_size['longest_edge'], p.max_image_size['longest_edge']], interpolation=p.resample, antialias=True)(image)
        elif p.size.get("shortest_edge", -1) != -1: #and p.__dict__.get("do_center_crop", -1) != True:
            image = T.Resize(p.size['shortest_edge'], interpolation=p.resample)(image)
        image = image.clamp(min=0, max=255)

    # center cropping
    if p.__dict__.get("do_center_crop", -1) == True:
        image = center_crop(image, size=[p.crop_size['height'], p.crop_size['width']])

    # Rescaling
    if p.__dict__.get("do_rescale", 0) == True:
    # if p.rescale_factor:
        image = image*p.rescale_factor
        # image = image.clamp(0, 1)

    # Normalization
    if p.do_normalize == True:
        image = T.Normalize(p.image_mean, p.image_std)(image)
    
    return image


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


def is_image_processing_close(image: torch.tensor, processor):
    if isinstance(processor, LlavaNextProcessor):
        llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
        img_prompt = llama3_template.format('<image>\nSummary above image in one word: ')
        image_p_hf = processor(text=img_prompt, images=image, return_tensors="pt", padding=True)['pixel_values'][:,0]
    else:
        image_p_hf = processor(images=[image], return_tensors="pt")['pixel_values']
    image_p_me = process_image(image, processor)
    logger.info(f"MSE: {torch.nn.functional.mse_loss(image_p_me, image_p_hf)}")
    logger.info(f"Linf: {(image_p_hf - image_p_me).norm(p=float('inf'))}")
    logger.info(f"L1: {(image_p_hf - image_p_me).norm(p=1)}")
    logger.info(f"L1: {torch.nn.functional.l1_loss(image_p_hf, image_p_me)}")


if __name__ == "__main__":
    """
    Here we test the effectiveness of our "differentiable preprocessing `process_image() function, as compared to the HF processor`"
    """
    from transformers.image_utils import load_image
    from utils import plot_images

    vlm_model_name = "HuggingFaceTB/SmolVLM-256M-Instruct" # "HuggingFaceTB/SmolVLM-256M-Instruct" or "openai/clip-vit-base-patch16", "llava-hf/llava-1.5-7b-hf"
    # image = load_image("https://raulperez.tieneblog.net/wp-content/uploads/2015/09/tux.jpg")
    image = load_image("https://farm9.staticflickr.com/8096/8445896722_e28fb3f055_z.jpg")
    # image = image.resize((512,512))
    image_tensor = T.PILToTensor()(image)
    image_tensor = image_tensor.float()
    image_tensor.requires_grad = True
    processor = AutoProcessor.from_pretrained(vlm_model_name)
    processor.image_processor.do_image_splitting = False
    processor.image_processor.resample = 3
    # if processor.image_processor.resample == 1: processor.image_processor.resample = 3
    if processor.image_processor.__dict__.get("do_convert_rgb", -1) != -1: processor.image_processor.do_convert_rgb = False
    
    image_p_hf = processor(images=[image], return_tensors="pt")['pixel_values']
    image_p_me = process_image(image_tensor, processor)

    if len(image_p_hf.shape) == 5: image_p_hf = image_p_hf[0][0]
    if len(image_p_hf.shape) == 4: image_p_hf = image_p_hf[0]
    logger.info(f"Size orig: {image_tensor.shape}, Size HF: {image_p_hf.shape}, Size ME: {image_p_me.shape}")
    logger.info(f"MSE: {torch.nn.functional.mse_loss(image_p_me, image_p_hf)}")
    logger.info(f"Linf: {(image_p_hf - image_p_me).norm(p=float('inf'))}")
    logger.info(f"Diff: {(image_p_hf - image_p_me)[0][:4,:4]}")
    logger.info(f"Part of mine: {image_p_me[0,:4,:4]}")
    logger.info(f"Grad HF: {image_p_hf.requires_grad}, Grad ME: {image_p_me.requires_grad}")
    # plot_images([T.ToPILImage()(image_p_hf/2+0.5), T.ToPILImage()(image_p_me/2+0.5)], n_subplots=2)
    plot_images([T.ToPILImage()(image_p_hf/255), T.ToPILImage()(image_p_me/255)], n_subplots=2)


def load_adv_image(task_config: "TaskConfig", exp_config_train: "ExperimentTrainConfig") -> torch.tensor:
    filename = exp_config_train.save_folder / task_config.create_filename()
    try:
        return torch.load(filename, weights_only=False)["image_adv"]
    except FileNotFoundError:
        raise ValueError(
            f"Error! Could not find file: {filename}! You need to train an attack with this configuration first"
        )
    
def load_gpt_image(task_config: "TaskConfig") -> torch.tensor:
    filename = DATA_FOLDER / "attacks" / "gpt-attacks" / f"attack-{gpt_filename(task_config)}.png"
    try:
        image_pil = Image.open(filename)
    except FileNotFoundError:
        raise ValueError(
            f"Error! Could not find file: {filename}! Make sure to include the images generated by ChatGPT in the right folder"
        )
    return T.PILToTensor()(image_pil)


def gpt_filename(task_config: "TaskConfig"):
    if not task_config.is_targeted:
        filename = "gpt-universal"
    elif len(task_config.target_query_idx) == 1 and task_config.n_knn_target_queries == 1:
        filename = "gpt-targeted-one-one"
    elif len(task_config.vlm.target_answers) == 1:
        filename = "gpt-targeted-many-one"
    else:
        filename = "gpt-targeted-many-many"

    return filename 
