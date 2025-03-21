from transformers import AutoModelForZeroShotImageClassification, AutoModel, AutoModelForImageTextToText, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
import torch
import torch.nn.functional as F
from utils.image_utils import process_image
from utils.utils import plot_images
import torchvision.transforms as T


# candidate models
MODEL_NAMES = [
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-large-patch14",
    "google/siglip2-base-patch16-224",
    "jinaai/jina-clip-v2",
    "vidore/colSmol-256M", # maybe not worth trying since it requires installing colpali
    # larger models: test later
    "royokong/e5-v",
    "nomic-ai/nomic-embed-vision-v1.5", # multimodal retrieval requires using this in conjunction with "nomic-ai/nomic-embed-text-v1.5"
    "vidore/colpali-v1.3-hf",
]

CLIP_LIKE_MODELS = ["openai/clip-vit-base-patch16", "google/siglip2-base-patch16-224", ]

class EmbeddingModel():

    def __init__(self, model_name, device, quantize=False):
        self.name = model_name
        self.device = device

        quantization_config = BitsAndBytesConfig(load_in_4bit=True) if quantize else None
        
        if model_name == "jinaai/jina-clip-v2":
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if device == "mps" else "auto",
                trust_remote_code=True).to(device)
            self.processor = None
            self.tokenizer = None

        if model_name in CLIP_LIKE_MODELS:
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if device == "mps" else "auto").to(device)
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
        if model_name == "royokong/e5-v":
            self.model = AutoModelForImageTextToText.from_pretrained(model_name, quantization_config=quantization_config).to(device)
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.tokenizer = None
            # reduce number of image patches
            self.processor.patch_size=16
            self.processor.image_processor.image_grid_pinpoints=[[336,336]]
            self.processor.image_processor.size['shortest_edge'] = 336

        
        self.model.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def compare_embeddings(self, image, user_query, overwrite=False, loss_type: str = "mse"):
        self.model.eval()
        if type(user_query) == str: user_query = [user_query]

        user_query_embedding = self.compute_txt_embedding(user_query)        
        image_emebedding = self.compute_img_embedding(image, image, overwrite)

        return compute_embedding_loss(image_emebedding, user_query_embedding, loss_type).item()

        # if len(user_query) > 1:
        #     image_features = model_emb.get_image_features(**image_input_emb)
        #     return torch.nn.functional.mse_loss(image_features, user_query_embedding, reduction="none").mean(dim=1) # return losses for all queries
        #     # return torch.nn.CosineSimilarity()(image_features, user_query_embedding)


    def compute_txt_embedding(self, user_query):
        
        if self.name == "jinaai/jina-clip-v2":
            return torch.tensor(self.model.encode_text(user_query)).to(self.device).type(self.model.dtype)

        if self.name in CLIP_LIKE_MODELS:
            user_query_embedding = self.model.get_text_features(**self.tokenizer(user_query, return_tensors="pt", truncation=True, padding=True).to(self.device)) # it had [0].detach()
            return user_query_embedding
        
        if self.name == "royokong/e5-v":
            if isinstance(user_query, str): user_query = [user_query]
            llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
            text_prompt = llama3_template.format('<sent>\nSummary above sentence in one word: ')
            text_inputs = self.processor([text_prompt.replace('<sent>', text) for text in user_query], return_tensors="pt", padding=True).to(self.device)
            user_query_embedding = self.model(**text_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :] 
            user_query_embedding = F.normalize(user_query_embedding, dim=-1)
            print(user_query_embedding.shape)
            return user_query_embedding
        
        quit(f"Not supported model {self.name}!")

    def compute_img_embedding(self, image, mock_image, overwrite=False):
        self.model.eval()

        if self.name == "jinaai/jina-clip-v2":
            # TODO: we should consider the processor
            if isinstance(image, list): 
                image = torch.cat(tuple([T.PILToTensor()(img.resize((512,512))).unsqueeze(0) for img in image]))
            else:
                image = image.unsqueeze(0)
            embeddings = self.model.get_image_features(image.to(self.device))
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings
        
        if self.name in CLIP_LIKE_MODELS:
            if overwrite:
                image_input_emb = self.processor(images=[mock_image], return_tensors='pt').to(self.device)
                # we cannot process multiple images
                image_ppd_emb = process_image(image, self)
                image_input_emb['pixel_values'][0] = image_ppd_emb
            else:
                if isinstance(image, list):
                    image_input_emb = self.processor(images=image, return_tensors='pt').to(self.device)
                else:
                    image_input_emb = self.processor(images=[image], return_tensors='pt').to(self.device)

            image_embedding = self.model.get_image_features(**image_input_emb)
            return image_embedding
        
        if self.name == "royokong/e5-v":
            llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
            img_prompt = llama3_template.format('<image>\nSummary above image in one word: ')
            
            if overwrite:
                img_inputs = self.processor(img_prompt, mock_image, return_tensors="pt", padding=True).to(self.device)
                image_ppd_emb = process_image(image)
                img_inputs['pixel_values'][0][0] = image_ppd_emb
                img_inputs['pixel_values'][0][1] = image_ppd_emb
            else:
                if not isinstance(image, list): image = [image]
                img_inputs = self.processor([img_prompt]*len(image), image, return_tensors="pt", padding=True).to(self.device)
            image_embedding = self.model(**img_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :] 
            image_embedding = F.normalize(image_embedding, dim=-1)
            print(image_embedding.shape)
            return image_embedding

        
        quit(f"Not supported model {self.name}!")


"""
Functions
"""
def compute_embedding_loss(image_embedding, text_embedding, loss_type: str):
    if loss_type == "mse":
        return torch.nn.functional.mse_loss(image_embedding, text_embedding)
    elif loss_type == "l2":
        return torch.nn.functional.pairwise_distance(image_embedding, text_embedding).mean()
    elif loss_type == "l2_nosqrt":
        return torch.nn.functional.pairwise_distance(image_embedding, text_embedding).pow(2).mean()
    elif loss_type == "cos":
        # return -(image_embedding @ text_embedding.transpose(0,1)).mean()
        return -torch.nn.CosineSimilarity()(image_embedding, text_embedding).mean()

# add an extra column to the dataset containing the embeddings of images
def add_img_embedding_column(ds, embedder: EmbeddingModel, existing_col_name="image", new_col_name="image_embeddings", device="cpu"):
    func = lambda example: {
        new_col_name: embedder.compute_img_embedding(image=[example[existing_col_name]], mock_image=None, overwrite=False)[0].cpu().detach().numpy()
    }
    return ds.map(func)

# add an extra column to the dataset containing the embeddings of texts (not used yet)
def add_txt_embedding_column(ds, embedder: EmbeddingModel, existing_col_name="text", new_col_name="text_embeddings"):
    func = lambda example: {
        new_col_name: embedder.compute_txt_embedding(user_query=example[existing_col_name])[0].float().cpu().detach().numpy()
    }
    return ds.map(func)

# find images close to the provided prompt in embedding space
def retrieve_images_by_prompt(prompt, ds_with_faiss, embedder: EmbeddingModel, topk, device="cpu", plot=True):
    prompt_embedding = embedder.compute_txt_embedding(prompt).float().cpu().detach().numpy()
    
    scores, retrieved_examples = ds_with_faiss.get_nearest_examples("image_embeddings", prompt_embedding, k=topk)
    if plot: plot_images(retrieved_examples["image"], topk)
    return scores, retrieved_examples
        