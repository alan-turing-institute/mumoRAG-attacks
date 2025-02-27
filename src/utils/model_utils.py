from .image_utils import process_image
import torch
import torchvision.transforms as T
from transformers import AutoModelForVision2Seq, AutoModel, AutoTokenizer, AutoProcessor, AutoModelForZeroShotImageClassification


def load_emb_model(model_name:str, device):
    if model_name == "openai/clip-vit-base-patch16":
        model_emb = AutoModelForZeroShotImageClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if device == "mps" else torch.bfloat16).to(device)
        processor_emb = AutoProcessor.from_pretrained(model_name)
        tokenizer_emb = AutoTokenizer.from_pretrained(model_name)
    return model_emb, processor_emb, tokenizer_emb

def load_vlm_model(model_name:str, device):
    if model_name == "HuggingFaceTB/SmolVLM-256M-Instruct":
        model_vlm = AutoModelForVision2Seq.from_pretrained(
            model_name, 
            torch_dtype=torch.float32 if device == "mps" else torch.bfloat16,
            _attn_implementation="flash_attention_2" if device == "cuda" else "eager",).to(device)
        processor_vlm = AutoProcessor.from_pretrained(model_name)
        processor_vlm.image_processor.do_image_splitting = False
    
    return model_vlm, processor_vlm

def test_vlm_prompt(user_query: str, processor):
    # Create input messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_query}
            ]
        },
    ]

    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    return prompt

def generate_from_vlm(image: torch.tensor, user_query: str, model_vlm, processor_vlm, device: str, overwrite: bool = False):
    model_vlm.eval()
    test_prompt = test_vlm_prompt(user_query, processor_vlm)
    inputs = processor_vlm(text=test_prompt, images=[T.ToPILImage()(image)], return_tensors="pt").to(device)
    
    if overwrite:
        image_ppd = process_image(image, processor_vlm)
        inputs['pixel_values'][0][0] = image_ppd
    
    generated_ids = model_vlm.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.5)
    generated_texts = processor_vlm.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    
    return generated_texts[0]

def vlm_forward(image, mock_image, prompt, model_name: str, model_vlm, processor_vlm, device, overwrite: bool = False):
    if model_name == "HuggingFaceTB/SmolVLM-256M-Instruct":
        if overwrite:
            inputs_vlm = processor_vlm(text=[prompt], images=[mock_image], return_tensors="pt").to(device) # here we feed the intiial image since we are overwriting it anyways
            image_ppd_vlm = process_image(image, processor_vlm)
            inputs_vlm['pixel_values'][0][0] = image_ppd_vlm
        else:
            inputs_vlm = processor_vlm(text=[prompt], images=[image], return_tensors="pt").to(device) # here we feed the intiial image since we are overwriting it anyways
        return model_vlm(**inputs_vlm)
    
    quit(f"Not supported model {model_name}!")
    
def compute_vlm_loss(model_name, vlm_output, target_tokens):
    # TODO: not sure if we need to pass logits to softmax first
    if model_name == "HuggingFaceTB/SmolVLM-256M-Instruct":
        logits_to_optimize = vlm_output.logits[0,-len(target_tokens)-2:-2,:]
        return torch.nn.CrossEntropyLoss()(logits_to_optimize, target_tokens)
    
    quit(f"Not supported model {model_name}!")

def test_embeddings_loss(image, user_query, model_name: str, model_emb, processor_emb, tokenizer_emb, device, overwrite=False, loss_type: str = "mse"):
    model_emb.eval()
    if type(user_query) == str: user_query = [user_query]

    user_query_embedding = compute_txt_embedding(user_query, model_name, model_emb, tokenizer_emb, processor_emb, device)        
    image_emebedding = compute_img_embedding(image, image, model_name, model_emb, tokenizer_emb, processor_emb, device, overwrite)

    return compute_embedding_loss(image_emebedding, user_query_embedding, loss_type).item()

    # if len(user_query) > 1:
    #     image_features = model_emb.get_image_features(**image_input_emb)
    #     return torch.nn.functional.mse_loss(image_features, user_query_embedding, reduction="none").mean(dim=1) # return losses for all queries
    #     # return torch.nn.CosineSimilarity()(image_features, user_query_embedding)


def compute_txt_embedding(user_query, model_name:str, model, tokenizer, processor, device):
    if model_name == "openai/clip-vit-base-patch16":
        user_query_embedding = model.get_text_features(**tokenizer(user_query, return_tensors="pt", truncation=True, padding=True).to(device)).detach() # it had [0].detach()
        return user_query_embedding
    
    quit(f"Not supported model {model_name}!")

def compute_img_embedding(image, mock_image, model_name:str, model, tokenizer, processor, device, overwrite=False):
    if model_name == "openai/clip-vit-base-patch16":
        if overwrite:
            image_input_emb = processor(images=[mock_image], return_tensors='pt').to(device)
            # we cannot process multiple images
            image_ppd_emb = process_image(image, processor)
            image_input_emb['pixel_values'][0] = image_ppd_emb
        else:
            if isinstance(image, list):
                image_input_emb = processor(images=image, return_tensors='pt').to(device)
            else:
                image_input_emb = processor(images=[image], return_tensors='pt').to(device)

        image_embedding = model.get_image_features(**image_input_emb)
        return image_embedding
    
    quit(f"Not supported model {model_name}!")

def compute_embedding_loss(image_embedding, text_embedding, loss_type: str):
    if loss_type == "mse":
        return torch.nn.functional.mse_loss(image_embedding, text_embedding)
    elif loss_type == "l2":
        return torch.nn.functional.pairwise_distance(image_embedding, text_embedding).mean()
    elif loss_type == "l2_nosqrt":
        return torch.nn.functional.pairwise_distance(image_embedding, text_embedding).pow(2).mean()
    elif loss_type == "cos":
        return -torch.nn.CosineSimilarity()(image_embedding, text_embedding).mean(dim=1)