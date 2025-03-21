import torch
import torch.autograd.profiler as profiler
from utils.embedding import EmbeddingModel, compute_embedding_loss
from utils.vlm import VLM
from utils.scheduler import LearningRateScheduler
from utils.utils import get_memory_consumption
from utils.attack_config import AttackConfig



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

    initial_image = raw_image.clone()
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
            loss_emb = compute_embedding_loss(image_embedding, user_query_embedding_batch, emb_loss_type)
    

        if lambda_vlm > 0:
            # generation loss function
            out = vlm.forward(raw_image, mock_images, full_text_vlm_prompt_batch, overwrite=True)
            loss_vlm = vlm.compute_gen_loss(out, target_tokens)

        # total loss function
        total_loss = lambda_emb * loss_emb + lambda_vlm * loss_vlm
        if i==0 or ((i+1)/gradient_acc_steps)%print_every==0: 
            print(f"Iter {(i//gradient_acc_steps)+1:4d}/{n_gradient_steps}, RAM usage -> {get_memory_consumption(device):.2f} GB, Losses -> Embedding: {loss_emb.item():.8f}, VLM: {loss_vlm.item():.8f}, Total: {total_loss.item():.8f}")

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


# if __name__ == "__main__":
#     # imports
#     from transformers.image_utils import load_image
#     from utils import plot_images, get_device
#     import torchvision.transforms as T

#     emb_model_name = "google/siglip2-base-patch16-224" # "openai/clip-vit-base-patch16", "vidore/colSmol-256M", "google/siglip2-base-patch16-224", ""
#     vlm_model_name = "HuggingFaceTB/SmolVLM-256M-Instruct" # "HuggingFaceTB/SmolVLM-256M-Instruct", "naver-clova-ix/donut-base-finetuned-docvqa"
#     user_query = "They are eating fish. What type of fish are they eating?"
#     target_answer = "They are actually eating beef"
#     image = load_image("https://farm9.staticflickr.com/8096/8445896722_e28fb3f055_z.jpg")
#     device = get_device(prefer_mps=True)
#     max_perturbation = 0.05
#     n_gradient_steps = 100
#     gradient_acc_steps = 4
#     lr_scheduler = LearningRateScheduler(start_lr=255 * (5e-3), end_lr=255*(5e-4), n_iter=n_gradient_steps) # multiply by 255 since input is [0,255]
#     lambda_emb = 1
#     lambda_vlm = 0
#     print_every = 10
#     max_batch_size_per_iter = 10 # number of queries to optimize for simulataneously (actual batch size is min(this, len([user_query])))
#     emb_loss_type = "mse" # mse, l2, l2_nosqrt, cos
#     print("Initialized variables!")

#     embedder = EmbeddingModel(emb_model_name, device)
#     vlm = VLM(vlm_model_name, device)
#     print("Loaded models and processors!")

#     image_tensor = T.PILToTensor()(image)
#     initial_image = image_tensor.clone()
#     image_tensor = image_tensor.float()
#     image_tensor.requires_grad = True

#     image_adv = rag_attack(
#         raw_image=image_tensor,
#         emb_model_name=emb_model_name,
#         embedder=embedder,
#         vlm=vlm,
#         user_query=user_query,
#         target_answer=target_answer,
#         max_perturbation=max_perturbation,
#         n_gradient_steps=n_gradient_steps,
#         print_every=print_every,
#         lr_scheduler=lr_scheduler,
#         max_batch_size_per_iter=max_batch_size_per_iter,
#         gradient_acc_steps=gradient_acc_steps,
#         lambda_emb=lambda_emb,
#         lambda_vlm=lambda_vlm,
#         emb_loss_type=emb_loss_type,
#         device=device
#     )

#     print(f"MSE: {torch.nn.functional.mse_loss(initial_image, image_adv)}")
#     print(f"Linf: {(initial_image - image_adv).norm(p=float('inf'))}")
#     plot_images([T.ToPILImage()(image/255) for image in [initial_image, image_adv]], n_subplots=2)

#     # Test generation
#     out_init = vlm.generate(initial_image, user_query) # this is the image we started with
#     out_adv = vlm.generate(image_adv, user_query) # this is the image we optimized but after potentially being modified by the processor
#     out_adv_ow = vlm.generate(image_adv, user_query, overwrite=True) # this is the image we optimized
#     print(out_init, out_adv, out_adv_ow, sep="\n===\n")

#     # test retrieval
#     loss_emb_init = embedder.compare_embeddings(initial_image, user_query)
#     loss_emb_adv = embedder.compare_embeddings(image_adv.type(torch.int32), user_query)
#     loss_emb_adv_ow = embedder.compare_embeddings(image_adv.type(torch.int32), user_query, overwrite=True)
#     print(loss_emb_init, loss_emb_adv, loss_emb_adv_ow, sep="\n----\n")