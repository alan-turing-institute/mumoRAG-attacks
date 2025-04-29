import random, math

import torch
import torchvision.transforms as T

from config.task import TaskConfig
from wrappers.embedding import EmbeddingModel
from wrappers.judge import JudgeVLM
from wrappers.vlm import VLM

from .scheduler import LearningRateScheduler
from .utils import get_memory_consumption
from .logger import logger



def rag_attack(
        raw_image: torch.tensor,
        embedder: EmbeddingModel,
        vlm: VLM,
        jdg: JudgeVLM,
        user_query: str | list[str],
        config: TaskConfig,
        attack_images: list,
        print_every: int,
        device: str):
    """
    Simulates an attack against the full RAG pipeline.
    The input image is jointly optimized w.r.t. the retriever and the VLM outputs

    NOTE:
    1. max_batch_size_per_iter is the batch size per iteration
    1. effective batch_size = batch_size_per_iter * gradient_acc_steps
    2. n_gradient_steps is the number of gradient updates
    3. total number of iterations = n_gradient_steps * gradient_acc_steps

    OBSERVATION:
    1. Using batch_size=1 with gradient accumulation is much more effective than using larger batch_size. No idea why?
    """
    # extract config variables
    target_answer_vlm = config.target_answer_vlm
    max_perturbation = config.max_perturbation
    n_gradient_steps = config.n_gradient_steps
    lr_scheduler = LearningRateScheduler(lr_start=config.lr_start, lr_end=config.lr_end, n_iter=config.n_gradient_steps)
    max_batch_size_per_iter = config.max_batch_size_per_iter
    gradient_acc_steps = config.gradient_acc_steps
    lambda_emb = config.lambda_emb
    lambda_vlm = config.lambda_vlm
    emb_loss_type = config.emb_train_loss_type
    is_adaptive = config.is_adaptive
    lambda_constant = config.lambda_constant
    gen_topk = config.gen_topk
    lambda_jdg = config.lambda_jdg
    target_answer_jdg = config.target_answer_jdg
    jdg_metric_list = config.train_jdg_metric_list
    target_query_idx = config.target_query_idx

    initial_image = raw_image.clone().float() if device == "cuda" else raw_image.clone()
    max_perturbation_pixels = max_perturbation*255
    batch_size_per_iter = min(len(user_query), max_batch_size_per_iter)
    n_iter = n_gradient_steps * gradient_acc_steps
    if type(user_query) == str: user_query = [user_query]
    is_targeted = len(target_query_idx) > 0 

    # pre-computing embeddings and prompts for all data 
    if lambda_emb > 0: 
        user_query_embedding = embedder.compute_txt_embedding(user_query)
    if lambda_vlm > 0: 
        full_text_vlm_prompt, target_tokens_vlm = vlm.get_training_prompt(user_query, target_answer_vlm, gen_topk)
    if lambda_jdg > 0: 
        full_text_jdg_prompt, target_tokens_jdg = jdg.get_training_prompt(user_query, target_answer_vlm, target_answer_jdg, jdg_metric_list, gen_topk)


    # initial values for loss
    loss_emb, loss_vlm, loss_jdg = torch.tensor([0]).to(device), torch.tensor([0]).to(device), torch.tensor([0]).to(device)
    grads = torch.zeros_like(raw_image)

    # attack iterations
    for i in range(n_iter):

        # ensure that raw_image requires grad
        raw_image.requires_grad = True

        # sample minibatch
        samples_idx = sample_minibatch(n_population=len(user_query), batch_size=batch_size_per_iter, target_idx=target_query_idx)
        positive_idx = [i for i in range(len(samples_idx)) if samples_idx[i] in target_query_idx]
        if lambda_emb > 0: user_query_embedding_batch = user_query_embedding[samples_idx,:]
        if lambda_vlm > 0: full_text_vlm_prompt_batch = [full_text_vlm_prompt[i] for i in samples_idx]
        if lambda_jdg > 0: 
            samples_jdg_idx = torch.randint(0, len(user_query)*len(jdg_metric_list), (batch_size_per_iter,))
            full_text_jdg_prompt_batch = [full_text_jdg_prompt[i] for i in samples_jdg_idx]

        if lambda_emb > 0:
            # retrieval loss function
            image_embedding = embedder.compute_img_embedding(raw_image, initial_image, overwrite=True)
            loss_emb = embedder.compute_embedding_loss(image_embedding, user_query_embedding_batch, emb_loss_type, is_targeted, positive_idx)


        if lambda_vlm > 0:
            # generation loss function
            context_images, adv_indices = prepare_context_images(attack_images, T.ToPILImage()(raw_image), batch_size_per_iter, config.gen_topk)
            out = vlm.forward(raw_image, full_text_vlm_prompt_batch, context_images, adv_indices, overwrite=True)
            loss_vlm = vlm.compute_gen_loss(out, target_tokens_vlm)
        
        if lambda_jdg > 0:
            # judge loss function
            out = jdg.forward(raw_image, full_text_jdg_prompt_batch, context_images, adv_indices, overwrite=True)
            loss_jdg = jdg.compute_gen_loss(out, target_tokens_jdg)

        # update loss coefficients if we use the adaptive attack
        if i==0 and is_adaptive and lambda_emb>0 and lambda_vlm>0:
            lambda_emb, lambda_vlm = adaptive_attack_coefficients(loss_emb, loss_vlm, lambda_constant)

        # total loss function
        total_loss = lambda_emb * loss_emb + lambda_vlm * loss_vlm + lambda_jdg * loss_jdg
        if i==0 or ((i+1)/gradient_acc_steps)%print_every==0:
            logger.info(f"Iter {(i//gradient_acc_steps)+1:4d}/{n_gradient_steps}, RAM usage -> {get_memory_consumption(device):.2f} GB, Losses -> Embedding: {loss_emb.item():.8f}, VLM: {loss_vlm.item():.8f}, Judge: {loss_jdg.item():.8f}, Total: {total_loss.item():.8f}")

        # backpropagation
        grads += torch.autograd.grad(total_loss, raw_image)[0]

        if (i+1)%gradient_acc_steps == 0:
            # compute average gradient
            grads /= gradient_acc_steps

            # get learning rate from scheduler
            lr = lr_scheduler.get_lr(i//gradient_acc_steps)

            # optimization step
            with torch.no_grad():
                raw_image = attack_step_pgd(raw_image, grads, lr, max_perturbation_pixels, initial_image)

            # zero the gradient
            grads = torch.zeros_like(raw_image)

    return raw_image


def attack_step_pgd(
        raw_image: torch.tensor,
        grads: torch.tensor,
        lr: float,
        max_perturbation_pixels: int,
        initial_image: torch.tensor):
    """
    Implement the projected gradient descent (PGD) attack proposed by Madry et al. (2018) 
    """
    # take step
    raw_image -= lr * torch.sign(grads)
    # clip according to attack budget
    torch.clip(raw_image, min=initial_image-max_perturbation_pixels, max=initial_image+max_perturbation_pixels, out=raw_image)
    # clip to make sure we stay within allowed RGB values
    torch.clip(raw_image, min=0, max=255, out=raw_image)

    return raw_image

@torch.no_grad()
def adaptive_attack_coefficients(loss_emb, loss_vlm, lambda_constant):
    lambda_vlm = 1
    lambda_emb = lambda_constant * abs(loss_vlm) / abs(loss_emb)

    return lambda_emb, lambda_vlm

def prepare_context_images(attack_images, mock_image_pil, batch_size_per_iter, gen_topk):
    """
    Randomly selects the order of images in the context for each element in the batch
    Returns:
        context_images: list[list[PIL.Image]] --> [batch_size x topk]
        adv_indices: list[int] -> [batch_size]
    """
    context_images, adv_indices = [], []
    
    for  i in range(batch_size_per_iter):
        n_samples = gen_topk-1
        adv_idx = random.randint(0, n_samples)
        sampled_images = random.sample(attack_images, k=n_samples)
        sampled_images.insert(adv_idx, mock_image_pil)
        
        context_images.append(sampled_images)
        adv_indices.append(adv_idx)

    return context_images, adv_indices

def sample_minibatch(n_population, batch_size, target_idx: list[int]):
    if target_idx == []:
        return torch.randint(0, n_population, (batch_size,))
    else:
        # 50% positive samples, 50% negative samples on average
        samples_pos = random.sample([i for i in range(n_population) if i in target_idx], batch_size)
        samples_neg = random.sample([i for i in range(n_population) if i not in target_idx], batch_size)
        return random.sample(samples_pos + samples_neg, batch_size)