import torch
from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoModelForZeroShotImageClassification, AutoModelForVision2Seq
from utils.image_utils import process_image
from utils.model_utils import compute_txt_embedding, compute_img_embedding, compute_embedding_loss, vlm_forward, compute_vlm_loss

def get_attack_prompt_vlm(
        processor: AutoProcessor, 
        user_query: str, 
        target_generation: str, 
        device: str):
    """
    builds the prompt skeleton for the VLM including the image placeholder, the user query, and the required response
    """
    target_tokens = processor(text=target_generation, return_tensors="pt").to(device)['input_ids'][0]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_query}
            ]
        },
        {
            "role": "assistant",
            "content": target_generation
        },
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=False)

    return prompt, target_tokens

def rag_attack(
        raw_image: torch.tensor, 
        emb_model_name: str,
        model_emb: AutoModel, 
        processor_emb: AutoProcessor,
        tokenizer_emb: AutoTokenizer, 
        vlm_model_name: str,
        model_vlm: AutoModel, 
        processor_vlm: AutoProcessor,
        user_query: list, # but maybe string also
        target_answer: str,
        max_perturbation: float,
        n_iter: int,
        lr: float,
        lambda_emb: float,
        lambda_vlm: float,
        emb_loss_type: str,
        device: str):
    """
    Simulates an attack against the full RAG pipeline.
    The input image is jointly optimized w.r.t. the retriever and the VLM outputs
    TODO: add option to optimize image w.r.t. only one objective
    """
    initial_image = raw_image.clone()
    max_perturbation_pixels = max_perturbation*255


    # retrieval processing
    if lambda_emb > 0:
        if type(user_query) == str: user_query = [user_query]
        user_query_embedding = compute_txt_embedding(user_query, emb_model_name, model_emb, tokenizer_emb, processor_emb, device)

    # VLM processing
    if lambda_vlm > 0:
        full_text_vlm_prompt, target_tokens = get_attack_prompt_vlm(processor_vlm, user_query, target_answer, device)

    # initial values for loss
    loss_emb, loss_vlm = torch.tensor([0]).to(device), torch.tensor([0]).to(device)

    # attack iterations
    for i in range(n_iter):

        # silly learning rate schedule
        if i==200: lr = 255*(3e-4)
        if i==600: lr = 255*(3e-5)

        # ensure that raw_image requires grad
        raw_image.requires_grad = True

        if lambda_emb > 0:
            # retrieval loss function
            image_embedding = compute_img_embedding(raw_image, initial_image, emb_model_name, model_emb, tokenizer_emb, processor_emb, device, overwrite=True)
            loss_emb = compute_embedding_loss(image_embedding, user_query_embedding, emb_loss_type)
      

        if lambda_vlm > 0:
            # generation loss function
            out = vlm_forward(raw_image, initial_image, full_text_vlm_prompt, vlm_model_name, model_vlm, processor_vlm, device, overwrite=True)
            loss_vlm = compute_vlm_loss(vlm_model_name, out, target_tokens)

        # total loss function
        total_loss = lambda_emb * loss_emb + lambda_vlm * loss_vlm
        if i==0 or i%10==9: print(f"Iter {i+1:4d}, Losses -> Embedding: {loss_emb.item():.8f}, VLM: {loss_vlm.item():.8f}, Total: {total_loss.item():.8f}")

        # backpropagation
        grads = torch.autograd.grad(total_loss, raw_image)

        # optimization step
        with torch.no_grad():
            # take step
            raw_image -= lr * torch.sign(grads[0])
    
            # clip according to attack budget
            torch.clip(raw_image, min=initial_image-max_perturbation_pixels, max=initial_image+max_perturbation_pixels, out=raw_image)
            # clip to make sure we stay within allowed RGB values
            torch.clip(raw_image, min=0, max=255, out=raw_image)
    
    return raw_image


if __name__ == "__main__":
    # imports
    from transformers.image_utils import load_image
    from utils.utils import plot_images, get_device
    from utils.model_utils import generate_from_vlm, test_embeddings_loss, load_emb_model, load_vlm_model
    import torchvision.transforms as T

    emb_model_name = "openai/clip-vit-base-patch16"
    vlm_model_name = "HuggingFaceTB/SmolVLM-256M-Instruct" # "HuggingFaceTB/SmolVLM-256M-Instruct" or "openai/clip-vit-base-patch16", 
    user_query = "They are eating fish. What type of fish are they eating?"
    target_answer = "They are actually eating beef"
    image = load_image("https://farm9.staticflickr.com/8096/8445896722_e28fb3f055_z.jpg")
    device = get_device(prefer_mps=True)
    max_perturbation = 0.05
    lr = 255 * (5e-3) # multiply by 255 since input is [0,255]
    lambda_emb = 1
    lambda_vlm = 0.5
    n_iter = 20
    emb_loss_type = "mse" # mse, l2, l2_nosqrt, cos
    print("Initialized variables!")

    
    model_emb, processor_emb, tokenizer_emb = load_emb_model(emb_model_name, device)
    model_vlm, processor_vlm = load_vlm_model(vlm_model_name, device)
    print("Loaded models and processors!")

    image_tensor = T.PILToTensor()(image)
    initial_image = image_tensor.clone()
    image_tensor = image_tensor.float()
    image_tensor.requires_grad = True

    image_adv = rag_attack(
        raw_image=image_tensor,
        emb_model_name=emb_model_name,
        model_emb=model_emb,
        processor_emb=processor_emb,
        tokenizer_emb=tokenizer_emb,
        vlm_model_name=vlm_model_name,
        model_vlm=model_vlm,
        processor_vlm=processor_vlm,
        user_query=user_query,
        target_answer=target_answer,
        max_perturbation=max_perturbation,
        n_iter=n_iter,
        lr=lr,
        lambda_emb=lambda_emb,
        lambda_vlm=lambda_vlm,
        emb_loss_type=emb_loss_type,
        device=device
    )

    print(f"MSE: {torch.nn.functional.mse_loss(initial_image, image_adv)}")
    print(f"Linf: {(initial_image - image_adv).norm(p=float('inf'))}")
    plot_images([T.ToPILImage()(image/255) for image in [initial_image, image_adv]], n_subplots=2)

    # Test generation
    out_init = generate_from_vlm(initial_image, user_query, model_vlm, processor_vlm, device) # this is the image we started with
    out_adv = generate_from_vlm(image_adv, user_query, model_vlm, processor_vlm, device) # this is the image we optimized but after potentially being modified by the processor
    out_adv_ow = generate_from_vlm(image_adv, user_query, model_vlm, processor_vlm, device, overwrite=True) # this is the image we optimized
    print(out_init, out_adv, out_adv_ow, sep="\n===\n")

    # test retrieval
    loss_emb_init = test_embeddings_loss(initial_image, user_query, emb_model_name, model_emb, processor_emb, tokenizer_emb, device)
    loss_emb_adv = test_embeddings_loss(image_adv.type(torch.int32), user_query, emb_model_name, model_emb, processor_emb, tokenizer_emb, device)
    loss_emb_adv_ow = test_embeddings_loss(image_adv.type(torch.int32), user_query, emb_model_name, model_emb, processor_emb, tokenizer_emb, device, overwrite=True)
    print(loss_emb_init, loss_emb_adv, loss_emb_adv_ow, sep="\n----\n")
    
