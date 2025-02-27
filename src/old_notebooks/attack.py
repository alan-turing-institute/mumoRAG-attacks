import torch
from utils import unnormalize_image, get_min_max_image, normalize_image
from transformers import AutoProcessor

# computes an adversarial example by perturbing the image so that its embeddgings are close to those of the prompt text
def adversarial_attack_fgsm_iterative(image_input, model, prompt, tokenizer, processor, n_iters, lr, initial_image, perturb_max):
    image_input['pixel_values'].requires_grad = True
    prompt_embedding = model.get_text_features(**tokenizer([prompt], return_tensors="pt", truncation=True))[0].detach()
    
    for i in range(n_iters):
        image_features = model.get_image_features(**image_input)[0]

        loss = torch.nn.functional.mse_loss(image_features, prompt_embedding)
        if i%10==0: print(i, loss.item())

        grads = torch.autograd.grad(loss, image_input['pixel_values'])

        with torch.no_grad():
            image_input['pixel_values'] -= lr * torch.sign(grads[0])
            # clip to make sure perturbation is not too high
            torch.clip(image_input['pixel_values'], min=initial_image-perturb_max, max=initial_image+perturb_max, out=image_input['pixel_values'])
            # clip to make sure we stay within allowed RGB values
            rgb_range = get_min_max_image(processor)
            torch.clip(image_input['pixel_values'], min=rgb_range[0], max=rgb_range[1], out=image_input['pixel_values'])

        
    return image_input


def adversarial_attack_vlm_fgsm_iterative(model, processor, query, image_tensor, image_tensor_ppd, target_generation, n_iters, lr, initial_image, perturb_max, device):
        
    init_norm = unnormalize_image(initial_image, processor)
    
    target_tokens = processor(text=target_generation, return_tensors="pt").to(device)['input_ids'][0]
    n_tokens_target = len(target_tokens)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": query}
            ]
        },
        {
            "role": "assistant",
            "content": target_generation
        },
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=False)

    for i in range(n_iters):
        image_tensor_ppd.requires_grad = True
        inputs = processor(text=prompt, images=[image_tensor], return_tensors="pt").to(device)
        inputs['pixel_values'][0][0] = image_tensor_ppd
        out = model(**inputs)

        # only look at the indices corresponding to generating the target generation tokens
        logits_to_optimize = out.logits[0,-n_tokens_target-2:-2,:]
        # logits_to_maximize = torch.gather(input=logits_to_optimize, dim=1, index=target_tokens.unsqueeze(1).to(device))
        loss = torch.nn.CrossEntropyLoss()(logits_to_optimize, target_tokens)
        if i%10==0: print(f"iter: {i}, loss: {loss.item():.8f}")

 
        grads = torch.autograd.grad(loss, image_tensor_ppd)

        if i>=100: lr=3e-4

        with torch.no_grad():
            image_tensor_ppd -= lr * torch.sign(grads[0])

            # stay within perturb_max
            image_norm = unnormalize_image(image_tensor_ppd, processor)
            torch.clip(image_norm, min=init_norm-perturb_max, max=init_norm+perturb_max, out=image_norm)
            image_tensor_ppd = normalize_image(image_norm, processor)

            # stay within allowed RGB values
            rgb_range = get_min_max_image(processor)
            torch.clip(image_tensor_ppd, rgb_range[0].to(device), rgb_range[1].to(device), out=image_tensor_ppd)
    
    return image_tensor_ppd

def get_text_prompt_vlm(processor: AutoProcessor, user_query: str, target_generation: str, device: str):
    """
    builds the prompt skeleton for the VLM including the image, the user query, and the required response
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