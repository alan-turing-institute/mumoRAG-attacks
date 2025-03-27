import torch
import torch.autograd.profiler as profiler
from .embedding import EmbeddingModel
from .vlm import VLM
from .scheduler import LearningRateScheduler
from .utils import get_memory_consumption
from .attack_config import AttackConfig



def rag_attack(
        raw_image: torch.tensor, 
        embedder: EmbeddingModel,
        vlm: VLM,
        user_query: list, # but maybe string also
        config: AttackConfig,
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
    target_answer = config.target_answer
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

    initial_image = raw_image.clone().float() if device=="cuda" else raw_image.clone()
    max_perturbation_pixels = max_perturbation*255
    batch_size_per_iter = min(len(user_query), max_batch_size_per_iter)
    n_iter = n_gradient_steps * gradient_acc_steps


    # retrieval processing
    if lambda_emb > 0:
        if type(user_query) == str: user_query = [user_query]
        user_query_embedding = embedder.compute_txt_embedding(user_query)

    # VLM processing
    if lambda_vlm > 0:
        full_text_vlm_prompt, target_tokens = vlm.get_training_prompt(user_query, target_answer)
        mock_images = [initial_image for _ in range(batch_size_per_iter)]
        # TODO: maybe here create the whole model_inputs list (to save time)


    # initial values for loss
    loss_emb, loss_vlm = torch.tensor([0]).to(device), torch.tensor([0]).to(device)
    grads = torch.zeros_like(raw_image)

    # attack iterations
    for i in range(n_iter):

        # ensure that raw_image requires grad
        raw_image.requires_grad = True

        # sample minibatch
        samples_idx = torch.randint(0, len(user_query), (batch_size_per_iter,)).type(torch.LongTensor)
        if lambda_emb > 0: user_query_embedding_batch = user_query_embedding[samples_idx,:]
        if lambda_vlm > 0: full_text_vlm_prompt_batch = [full_text_vlm_prompt[i] for i in samples_idx]

        # -- if code is slow uncomment the following line and indent the following code --
        # with profiler.profile(use_cuda=False) as prof:

        if lambda_emb > 0:
            # retrieval loss function
            image_embedding = embedder.compute_img_embedding(raw_image, initial_image, overwrite=True)
            loss_emb = embedder.compute_embedding_loss(image_embedding, user_query_embedding_batch, emb_loss_type)
    

        if lambda_vlm > 0:
            # generation loss function
            out = vlm.forward(raw_image, mock_images, full_text_vlm_prompt_batch, overwrite=True)
            loss_vlm = vlm.compute_gen_loss(out, target_tokens)

        # update loss coefficients if we use the adaptive attack
        if i==0 and is_adaptive and lambda_emb>0 and lambda_vlm>0:
            lambda_emb, lambda_vlm = adaptive_attack_coefficients(loss_emb, loss_vlm, lambda_constant)
        
        # total loss function
        total_loss = lambda_emb * loss_emb + lambda_vlm * loss_vlm
        if i==0 or ((i+1)/gradient_acc_steps)%print_every==0: 
            print(f"Iter {(i//gradient_acc_steps)+1:4d}/{n_gradient_steps}, RAM usage -> {get_memory_consumption(device):.2f} GB, Losses -> Embedding: {loss_emb.item():.8f}, VLM: {loss_vlm.item():.8f}, Total: {total_loss.item():.8f}, Lambdas -> Embedding: {lambda_emb:.2f}, VLM: {lambda_vlm:.2f}")

        # backpropagation
        grads += torch.autograd.grad(total_loss, raw_image)[0]

        if (i+1)%gradient_acc_steps == 0:
            # compute average gradient
            grads /= gradient_acc_steps

            # get learning rate from scheduler
            lr = lr_scheduler.get_lr(i//gradient_acc_steps)
            
            # optimization step
            with torch.no_grad():
                raw_image = attack_step_bim(raw_image, grads, lr, max_perturbation_pixels, initial_image)
            
            # zero the gradient
            grads = torch.zeros_like(raw_image)
            
        # -- stop indenting here and uncomment next line to profile timing issues --
        # print(prof.key_averages().table(sort_by="cpu_time_total"))
        
    return raw_image


def attack_step_bim(
        raw_image: torch.tensor, 
        grads: torch.tensor, 
        lr: float, 
        max_perturbation_pixels: int, 
        initial_image: torch.tensor):
    """
    Implement the basic iterative method attack (FGSM but iterative)
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