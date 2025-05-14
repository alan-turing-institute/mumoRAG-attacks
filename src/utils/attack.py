import random

import torch
import torchvision.transforms.v2 as T

from config.task import TaskConfig
from wrappers.attack_mask import get_attack_mask
from wrappers.cache import get_text_embedder, get_embedder
from wrappers.embedding import EmbeddingModel, EmbedderName, COLPALI_LOSSES, get_loss_with_default
from wrappers.judge import JudgeVLM
from wrappers.text_embedding import TextEmbedderName
from wrappers.vlm import VLM
from .logger import logger
from .scheduler import LearningRateScheduler
from .utils import get_memory_consumption


def rag_attack(
    raw_image: torch.Tensor,
    embedders: list[EmbeddingModel],
    vlms: list[VLM] | None,
    jdg: JudgeVLM | None,
    train_user_queries: list[str],
    train_ground_truth_vlm_answers: list[str],
    config: TaskConfig,
    attack_images: list,
    print_every: int,
    device: str,
):
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
    max_perturbation = config.max_perturbation
    n_gradient_steps = config.n_gradient_steps
    lr_scheduler = LearningRateScheduler(lr=config.lr, n_iter=config.n_gradient_steps)
    max_batch_size_per_iter = config.max_batch_size_per_iter
    gradient_acc_steps = config.gradient_acc_steps
    lambda_emb = config.lambda_emb
    is_adaptive = config.is_adaptive
    lambda_constant = config.lambda_constant
    is_targeted = config.is_targeted
    target_query_idx = config.target_query_idx
    n_knn_target_queries = config.n_knn_target_queries
    optimize_nontargeted_queries = config.optimize_nontargeted_queries

    initial_image = raw_image.clone().float() if device == "cuda" else raw_image.clone()
    attack_mask = get_attack_mask(config.attack_mask, initial_image, config.image_size)
    max_perturbation_pixels = max_perturbation*255
    batch_size_per_iter = min(len(train_user_queries), max_batch_size_per_iter)
    n_iter = n_gradient_steps * gradient_acc_steps

    # embedder only required for nearest neighbour attacks which are not currently supported with multi-embedder
    embedder = embedders[0]
    emb_loss_type = get_loss_with_default(embedder.name, config.emb_train_loss_type)

    target_query_idx, _, all_answers_vlm = get_all_target_queries_and_answers(
        is_targeted,
        target_query_idx,
        config.vlm.target_answers if config.vlm else None,
        train_user_queries,
        n_knn_target_queries,
        train_ground_truth_vlm_answers,
        embedder.name,
        emb_loss_type,
        device,
    )
    user_query_embeddings = dict()
    # pre-computing embeddings and prompts for all data
    if lambda_emb > 0:
        for embedder in embedders:
            user_query_embeddings[embedder.name] = embedder.compute_txt_embedding(train_user_queries)

    vlm_info = dict()
    if config.vlm:
        for vlm in vlms:
            full_text_vlm_prompts, target_tokens_vlm = vlm.get_training_prompts(train_user_queries, all_answers_vlm, config.vlm.gen_topk)
            vlm_info[vlm.name] = {
                "full_text_vlm_prompts": full_text_vlm_prompts,
                "target_tokens_vlm": target_tokens_vlm,
            }
            if config.judge:
                full_text_jdg_prompts, target_tokens_jdg = jdg.get_training_prompts(
                    train_user_queries,
                    all_answers_vlm,
                    config.judge.target_answer,
                    config.judge.metrics,
                    config.vlm.gen_topk,
                )
                vlm_info[vlm.name].update({
                    "full_text_jdg_prompts": full_text_jdg_prompts,
                    "target_tokens_jdg": target_tokens_jdg,
                })

    grads = torch.zeros_like(raw_image)

    for i in range(n_iter):
        raw_image.requires_grad = True

        loss_emb, loss_vlm, loss_jdg = torch.tensor([0.]).to(device), torch.tensor([0.]).to(device), torch.tensor([0.]).to(device)

        samples_idx = sample_minibatch(
            n_population=len(train_user_queries),
            batch_size=batch_size_per_iter,
            is_targeted=is_targeted,
            target_idx=target_query_idx,
            optimize_nontargeted_queries=optimize_nontargeted_queries,
        )
        positive_idx = [i for i in range(len(samples_idx)) if samples_idx[i] in target_query_idx]

        if lambda_emb > 0:
            for embedder in embedders:
                emb_loss_type = get_loss_with_default(embedder.name, config.emb_train_loss_type)
                # retrieval loss function
                user_query_embedding = user_query_embeddings[embedder.name]
                user_query_embedding_batch = user_query_embedding[samples_idx, :]
                image_embedding = embedder.compute_img_embedding(raw_image, initial_image, overwrite=True)
                loss_emb += embedder.compute_embedding_loss(image_embedding, user_query_embedding_batch, emb_loss_type, is_targeted, positive_idx)

        if config.vlm:
            # generation loss function
            for vlm in vlms:
                full_text_vlm_prompts = vlm_info[vlm.name]["full_text_vlm_prompts"]
                target_tokens_vlm = vlm_info[vlm.name]["target_tokens_vlm"]

                full_text_vlm_prompt_batch = [full_text_vlm_prompts[i] for i in samples_idx]
                target_tokens_vlm_batch = [target_tokens_vlm[i] for i in samples_idx]
                context_images, adv_indices = prepare_context_images(attack_images, T.ToPILImage()(raw_image), batch_size_per_iter, config.vlm.gen_topk)
                out = vlm.forward(raw_image, full_text_vlm_prompt_batch, context_images, adv_indices, overwrite=True)
                loss_vlm += vlm.compute_gen_loss(out, target_tokens_vlm_batch, positive_idx)
                if config.judge:
                    full_text_jdg_prompts = vlm_info[vlm.name]["full_text_jdg_prompts"]
                    target_tokens_jdg = vlm_info[vlm.name]["target_tokens_jdg"]

                    samples_jdg_idx = torch.randint(0, len(train_user_queries)*len(config.judge.metrics), (batch_size_per_iter,))
                    full_text_jdg_prompt_batch = [full_text_jdg_prompts[i] for i in samples_jdg_idx]
                    target_tokens_jdg_batch = [target_tokens_jdg[i] for i in samples_jdg_idx]
                    # judge loss function
                    out = jdg.forward(raw_image, full_text_jdg_prompt_batch, context_images, adv_indices, overwrite=True)
                    loss_jdg += jdg.compute_gen_loss(out, target_tokens_jdg_batch)

        # update loss coefficients if we use the adaptive attack
        if i==0 and is_adaptive and lambda_emb>0 and config.vlm:
            lambda_emb, lambda_vlm = adaptive_attack_coefficients(loss_emb, loss_vlm, lambda_constant)

        # total loss function
        total_loss = lambda_emb * loss_emb
        if config.vlm:
            total_loss += config.vlm.lambda_ * loss_vlm
        if config.judge:
            total_loss += config.judge.lambda_ * loss_jdg
        if i==0 or ((i+1)/gradient_acc_steps)%print_every==0:
            logger.info(f"Iter {(i//gradient_acc_steps)+1:4d}/{n_gradient_steps}, RAM usage -> {get_memory_consumption(device):.2f} GB, Losses -> Embedding: {loss_emb.item():.8f}, VLM: {loss_vlm.item():.8f}, Judge: {loss_jdg.item():.8f}, Total: {total_loss.item():.8f}")

        # backpropagation
        grads += torch.autograd.grad(total_loss, raw_image)[0]
        # Applies perturbation mask
        grads *= attack_mask

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
    raw_image: torch.Tensor,
    grads: torch.Tensor,
    lr: float,
    max_perturbation_pixels: int,
    initial_image: torch.Tensor,
):
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

def sample_minibatch(n_population, batch_size, is_targeted: bool, target_idx: list[int], optimize_nontargeted_queries: bool):
    if not is_targeted:
        return torch.randint(0, n_population, (batch_size,))
    else:
        # 50% positive samples, 50% negative samples on average
        samples_pos = random.choices([i for i in range(n_population) if i in target_idx], k=batch_size)
        samples_neg = random.sample([i for i in range(n_population) if i not in target_idx], batch_size) if optimize_nontargeted_queries else []
        return random.sample(samples_pos + samples_neg, batch_size)

def get_all_target_queries_and_answers(
        is_targeted: bool,
        target_query_idx: list[int],
        target_answer_vlm: list[str] | None,
        train_user_queries: list[str],
        n_knn_target_queries: int,
        ground_truth_answers: list[str],
        attack_embedder_name: EmbedderName | TextEmbedderName,
        emb_loss_type,
        device,
    ):
    # if universal attack, all queries are targeted
    if not is_targeted:
        target_query_idx = [i for i in range(len(ground_truth_answers))]

    # make number of answers match number of target queries
    if target_answer_vlm is None:
        target_answer_vlm = [""]
    if len(target_answer_vlm) == 1:
        target_answer_vlm = [target_answer_vlm[0] for _ in target_query_idx]

    # extend target query indices and target answers to include nearest neighbours
    if (not is_targeted) or n_knn_target_queries == 1:
       # only include one nearest neighbors (a.k.a. self)
       extended_target_idx, extended_target_answers = target_query_idx, target_answer_vlm
    else:
        similarity = get_embedding_similarity(train_user_queries, attack_embedder_name, emb_loss_type, device)
        extended_target_idx, extended_target_answers = [], []
        for i, q_idx in enumerate(target_query_idx):
            topk_similar = similarity[q_idx,:].topk(n_knn_target_queries, sorted=True).indices
            # remove duplicates
            topk_similar = [x for x in topk_similar if x not in extended_target_idx]
            extended_target_idx.extend(topk_similar)
            extended_target_answers.extend([target_answer_vlm[i] for _ in range(len(topk_similar))])

    # update ground truth answers by malicious answers
    all_answers = ground_truth_answers
    for idx, answer in zip(extended_target_idx, extended_target_answers):
        all_answers[idx] = answer

    return extended_target_idx, extended_target_answers, all_answers

def get_embedding_similarity(train_user_queries: list[str], attack_embedder_name: EmbedderName | TextEmbedderName, emb_loss_type, device):
    if isinstance(attack_embedder_name, TextEmbedderName):
        text_embedder = get_text_embedder(attack_embedder_name, device=device)
        similarity = text_embedder.compare_embeddings(train_user_queries, train_user_queries, similarity_metric="cos")
        del text_embedder
    elif isinstance(attack_embedder_name, EmbedderName):
        embedder = get_embedder(attack_embedder_name, device=device, quantize=False, colpali_only_images=False)
        query_embeddings = embedder.compute_txt_embedding(train_user_queries)
        num_queries = query_embeddings.shape[0]
        similarity = torch.zeros((num_queries, num_queries))
        for i in range(num_queries):
            query_i = query_embeddings[i].unsqueeze(0) if emb_loss_type in COLPALI_LOSSES else query_embeddings[i]
            similarity[i] = embedder.compare_embeddings(query_i, query_embeddings, emb_loss_type)
        del embedder
    else:
        raise ValueError(f"Unknown embedder {attack_embedder_name}")

    return similarity
