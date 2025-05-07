from typing import Optional

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from strenum import StrEnum
from transformers import AutoModel, AutoModelForImageTextToText, AutoTokenizer, AutoProcessor, BitsAndBytesConfig

from utils.image_utils import process_image
from utils.utils import plot_images


class EmbedderName(StrEnum):
    CLIP_BASE_PATCH16 = "openai/clip-vit-base-patch16"
    CLIP_LARGE_PATCH14 = "openai/clip-vit-large-patch14"
    SIGLIP2_BASE_PATCH16 = "google/siglip2-base-patch16-224"
    SIGLIP2_LARGE_PATCH16 = "google/siglip2-large-patch16-256"
    JINA_CLIP_2 = "jinaai/jina-clip-v2"
    E5_V = "royokong/e5-v"
    COLSMOL_500M = "vidore/colSmol-500M"
    COLSMOL_256M = "vidore/colSmol-256M"
    SMOLVLM_256M = "HuggingFaceTB/SmolVLM-256M-Instruct"  # just to test if it runs?
    COLPALI = "vidore/colpali-v1.3"
    QWEN2_GME_2B = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
    QWEN2_GME_7B = "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct"


CLIP_LIKE_MODELS = [
    EmbedderName.CLIP_BASE_PATCH16,
    EmbedderName.CLIP_LARGE_PATCH14,
    EmbedderName.SIGLIP2_BASE_PATCH16,
    EmbedderName.SIGLIP2_LARGE_PATCH16,
]

COLSMOL_MODELS = [
    EmbedderName.COLSMOL_500M,
    EmbedderName.COLSMOL_256M,
    EmbedderName.SMOLVLM_256M  # NOTE: we can use any VLM as if it was a colpali model
]

COLPALI_MODELS = COLSMOL_MODELS + [
    EmbedderName.COLPALI,
]


class EmbeddingLoss(StrEnum):
    MSE = "mse"
    COS = "cos"
    MAXSIM = "maxsim"
    AVGSIM = "avgsim"
    SOFTMAXSIM = "softmaxsim"
    COS_AVGEMB = "cos_avgemb"


COLPALI_LOSSES = [
    EmbeddingLoss.MAXSIM,
    EmbeddingLoss.AVGSIM,
    EmbeddingLoss.SOFTMAXSIM,
    EmbeddingLoss.COS_AVGEMB,
]
NON_COLPALI_LOSSES = [
    EmbeddingLoss.COS,
    EmbeddingLoss.MSE
]

QWEN2_MODELS = [
    EmbedderName.QWEN2_GME_2B,
    EmbedderName.QWEN2_GME_7B,
]


def is_loss_compatible(model_name_emb: EmbedderName, loss: EmbeddingLoss) -> bool:
    if (model_name_emb in COLPALI_MODELS and loss not in COLPALI_LOSSES) or (
            model_name_emb not in COLPALI_MODELS and loss in COLPALI_LOSSES):
        return False
    return True


# candidate models
MODEL_NAMES = [
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-large-patch14",
    "google/siglip2-base-patch16-224",
    "jinaai/jina-clip-v2",
    "vidore/colSmol-256M",  # maybe not worth trying since it requires installing colpali
    # larger models: test later
    "royokong/e5-v",
    "nomic-ai/nomic-embed-vision-v1.5",
    # multimodal retrieval requires using this in conjunction with "nomic-ai/nomic-embed-text-v1.5"
    "vidore/colpali-v1.3",
]


class EmbeddingModel:
    def __init__(self, model_name: EmbedderName, device, quantize=False, colpali_only_images=False):
        self.name = model_name
        self.device = device
        self.colpali_only_images = colpali_only_images

        quantization_config = BitsAndBytesConfig(load_in_4bit=True) if quantize else None

        if model_name == EmbedderName.JINA_CLIP_2:
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
            self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if model_name == EmbedderName.E5_V:
            self.model = AutoModelForImageTextToText.from_pretrained(model_name,
                                                                     quantization_config=quantization_config).to(device)
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.tokenizer = None
            # reduce number of image patches
            self.processor.patch_size = 16
            self.processor.image_processor.image_grid_pinpoints = [[336, 336]]
            self.processor.image_processor.size['shortest_edge'] = 336

        if model_name == EmbedderName.COLPALI:
            from colpali_engine.models import ColPali, ColPaliProcessor

            self.model = ColPali.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if device == "mps" else torch.bfloat16).to(device)
            self.processor = ColPaliProcessor.from_pretrained(model_name)
            self.tokenizer = None

        if model_name in COLSMOL_MODELS:
            from colpali_engine.models import ColIdefics3, ColIdefics3Processor
            self.model = ColIdefics3.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if device == "mps" else torch.bfloat16).to(device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.processor = ColIdefics3Processor.from_pretrained(model_name)
            self.processor.image_processor.do_image_splitting = False
        # if model_name in QWEN2_MODELS:
        #     self.model = AutoModel.from_pretrained("Alibaba-NLP/gme-Qwen2-VL-2B-Instruct", revision="refs/pr/10", trust_remote_code=True)
        #     self.processor = AutoProcessor.from_pretrained("Alibaba-NLP/gme-Qwen2-VL-2B-Instruct", revision="refs/pr/10", trust_remote_code=True)
        #     self.tokenizer = None

        self.model.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def compare_embeddings(self, image, user_query, overwrite=False, loss_type: EmbeddingLoss = EmbeddingLoss.MSE):
        self.model.eval()
        if type(user_query) == str: user_query = [user_query]

        user_query_embedding = self.compute_txt_embedding(user_query)
        image_embedding = self.compute_img_embedding(image, image, overwrite)

        return self.compute_embedding_loss(image_embedding, user_query_embedding, loss_type).item()

    def compute_txt_embedding(self, user_query):

        if self.name == EmbedderName.JINA_CLIP_2:
            return torch.tensor(self.model.encode_text(user_query)).to(self.device).type(self.model.dtype)

        if self.name in CLIP_LIKE_MODELS:
            user_query_embedding = self.model.get_text_features(
                **self.tokenizer(user_query, return_tensors="pt", truncation=True, padding=True).to(
                    self.device))  # it had [0].detach()
            return user_query_embedding

        if self.name == EmbedderName.E5_V:
            if isinstance(user_query, str): user_query = [user_query]
            llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
            text_prompt = llama3_template.format('<sent>\nSummary above sentence in one word: ')
            text_inputs = self.processor([text_prompt.replace('<sent>', text) for text in user_query],
                                         return_tensors="pt", padding=True).to(self.device)
            user_query_embedding = self.model(**text_inputs, output_hidden_states=True, return_dict=True).hidden_states[
                                       -1][:, -1, :]
            user_query_embedding = F.normalize(user_query_embedding, dim=-1)
            print(user_query_embedding.shape)
            return user_query_embedding

        if self.name == EmbedderName.COLPALI:
            batch_queries = self.processor.process_queries(user_query).to(self.device)
            user_query_embedding = self.model(**batch_queries)
            return user_query_embedding

        if self.name in COLSMOL_MODELS:
            batch_queries = self.processor.process_queries(user_query).to(self.device)
            user_query_embedding = self.model(**batch_queries)
            return user_query_embedding

        raise ValueError(f"Not supported model {self.name}!")

    def compute_img_embedding(self, image, mock_image, overwrite=False):
        self.model.eval()

        if self.name == EmbedderName.JINA_CLIP_2:
            # TODO: we should consider the processor
            if isinstance(image, list):
                image = torch.cat(tuple([T.PILToTensor()(img.resize((512, 512))).unsqueeze(0) for img in image]))
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
                if not isinstance(image, list): image = [image]
                image_input_emb = self.processor(images=image, return_tensors='pt').to(self.device)

            image_embedding = self.model.get_image_features(**image_input_emb)
            return image_embedding

        if self.name == EmbedderName.E5_V:
            llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
            img_prompt = llama3_template.format('<image>\nSummary above image in one word: ')

            if overwrite:
                img_inputs = self.processor(img_prompt, mock_image, return_tensors="pt", padding=True).to(self.device)
                image_ppd_emb = process_image(image)
                img_inputs['pixel_values'][0][0] = image_ppd_emb
                img_inputs['pixel_values'][0][1] = image_ppd_emb
            else:
                if not isinstance(image, list): image = [image]
                img_inputs = self.processor([img_prompt] * len(image), image, return_tensors="pt", padding=True).to(
                    self.device)
            image_embedding = self.model(**img_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:,
                              -1, :]
            image_embedding = F.normalize(image_embedding, dim=-1)
            print(image_embedding.shape)
            return image_embedding

        if self.name == EmbedderName.COLPALI or self.name in COLSMOL_MODELS:
            if overwrite:
                image_input_emb = self.processor.process_images([T.ToPILImage()(mock_image)]).to(self.device)

                # we cannot process multiple images
                image_ppd_emb = process_image(image, self)
                image_input_emb['pixel_values'][0] = image_ppd_emb
            else:
                if not isinstance(image, list): image = [image]
                image_input_emb = self.processor.process_images(image).to(self.device)

            if self.colpali_only_images:
                # remove non-image tokens
                # hard-coding indices works for now, but may not for future versions of colpali
                image_input_emb.input_ids = image_input_emb.input_ids[:, 9:-2]
                image_input_emb.attention_mask = image_input_emb.attention_mask[:, 9:-2]
                image_embedding = self.model(input_ids=image_input_emb.input_ids,
                                             attention_mask=image_input_emb.attention_mask,
                                             pixel_values=image_input_emb.pixel_values,
                                             pixel_attention_mask=image_input_emb.pixel_attention_mask)
            else:
                image_embedding = self.model(**image_input_emb)
            return image_embedding

        raise ValueError(f"Not supported model {self.name}!")

    
    def compute_embedding_loss(self, image_embedding, text_embedding, loss_type: EmbeddingLoss, is_targeted: bool, positive_idx: list[int]):
        if not is_targeted:
            return self._compute_embedding_loss(image_embedding, text_embedding, loss_type)
        
        # compute loss separately for in-target and out-of-target queries
        negative_idx = [i for i in range(text_embedding.shape[0]) if  i not in positive_idx]
        text_embedding_pos = text_embedding[positive_idx, :]
        text_embedding_neg = text_embedding[negative_idx, :]
        loss_pos, loss_neg = torch.tensor([0]).to(self.device), torch.tensor([0]).to(self.device)
        if text_embedding_pos.shape[0]>0: loss_pos = self._compute_embedding_loss(image_embedding, text_embedding_pos, loss_type)
        if text_embedding_neg.shape[0]>0: loss_neg = self._compute_embedding_loss(image_embedding, text_embedding_neg, loss_type)
        return loss_pos - loss_neg
        

    def _compute_embedding_loss(self, image_embedding, text_embedding, loss_type: EmbeddingLoss):
        # colpali has its own retrieval score (MaxSim)
        if (self.name in COLPALI_MODELS or self.name == EmbedderName.COLPALI) and loss_type != EmbeddingLoss.COS_AVGEMB:
            # my version of the scoring function (allowing different losses)
            return -1 * score_multi_vector_modified(text_embedding, image_embedding, device=self.device,
                                                    loss=loss_type).mean()

        match loss_type:
            case EmbeddingLoss.COS_AVGEMB:
                return 1 - torch.nn.CosineSimilarity()(image_embedding.mean(dim=1), text_embedding.mean(dim=1)).mean()
            case EmbeddingLoss.MSE:
                return torch.nn.functional.mse_loss(image_embedding, text_embedding)
            case "l2":
                return torch.nn.functional.pairwise_distance(image_embedding, text_embedding).mean()
            case "l2_nosqrt":
                return torch.nn.functional.pairwise_distance(image_embedding, text_embedding).pow(2).mean()
            case EmbeddingLoss.COS:
                # return -(image_embedding @ text_embedding.transpose(0,1)).mean()
                return 1 - torch.nn.CosineSimilarity()(image_embedding, text_embedding).mean()
            case _:
                raise ValueError(f"Unknown loss type {loss_type}!")

    def compare_embeddings(self, embeddings_1, embeddings_2, loss_type):
        if (self.name in COLPALI_MODELS or self.name == EmbedderName.COLPALI) and loss_type != EmbeddingLoss.COS_AVGEMB:
            # my version of the scoring function (allowing different losses)
            return score_multi_vector_modified(embeddings_1, embeddings_2, device=self.device,
                                                    loss=loss_type)
        match loss_type:
            case EmbeddingLoss.COS_AVGEMB:
                return torch.nn.CosineSimilarity()(embeddings_1.mean(dim=1), embeddings_2.mean(dim=1)).mean()
            case EmbeddingLoss.COS:
                return torch.nn.CosineSimilarity()(embeddings_1, embeddings_2)
            case _:
                raise ValueError(f"Unknown loss type {loss_type}!")

"""
Functions
"""


def score_multi_vector_modified(
        qs: torch.Tensor | list[torch.Tensor],
        ps: torch.Tensor | list[torch.Tensor],
        batch_size: int = 128,
        device: Optional[str | torch.device] = None,
        loss: EmbeddingLoss = EmbeddingLoss.MAXSIM,
) -> torch.Tensor:
    """
    NOTE: this is a modified version of Colpali's scoring function at https://github.com/illuin-tech/colpali/blob/main/colpali_engine/utils/processing_utils.py
    """

    if len(qs) == 0:
        raise ValueError("No queries provided")
    if len(ps) == 0:
        raise ValueError("No passages provided")

    scores_list: list[torch.Tensor] = []

    for i in range(0, len(qs), batch_size):
        scores_batch = []
        qs_batch = torch.nn.utils.rnn.pad_sequence(qs[i: i + batch_size], batch_first=True, padding_value=0).to(
            device
        )
        for j in range(0, len(ps), batch_size):
            ps_batch = torch.nn.utils.rnn.pad_sequence(
                ps[j: j + batch_size], batch_first=True, padding_value=0
            ).to(device)

            match loss:
                case EmbeddingLoss.MAXSIM:
                    scores_batch.append(torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2))
                case EmbeddingLoss.AVGSIM:
                    scores_batch.append(torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).mean(dim=3).sum(dim=2))
                case EmbeddingLoss.SOFTMAXSIM:
                    all_scores = torch.einsum("bnd,csd->bcns", qs_batch, ps_batch)
                    scores_batch.append((all_scores * all_scores.softmax(dim=3)).sum(dim=3).sum(dim=2))
                case EmbeddingLoss.COS_AVGEMB:
                    # take the average of embeddings over tokens then compute cosine similarity
                    print(qs_batch.shape, ps_batch.shape)
                    scores_batch.append(torch.nn.CosineSimilarity()(qs_batch.mean(dim=1), ps_batch.mean(dim=1)))
        scores_batch = torch.cat(scores_batch, dim=1).cpu()
        scores_list.append(scores_batch)

    scores = torch.cat(scores_list, dim=0)
    assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

    scores = scores.to(torch.float32)
    return scores


# add an extra column to the dataset containing the embeddings of images
def add_img_embedding_column(ds, embedder: EmbeddingModel, existing_col_name="image", new_col_name="image_embeddings",
                             device="cpu"):
    func = lambda example: {
        new_col_name:
            embedder.compute_img_embedding(image=[example[existing_col_name]], mock_image=None, overwrite=False)[
                0].cpu().detach().numpy()
    }
    return ds.map(func)


# add an extra column to the dataset containing the embeddings of texts (not used yet)
def add_txt_embedding_column(ds, embedder: EmbeddingModel, existing_col_name="text", new_col_name="text_embeddings"):
    func = lambda example: {
        new_col_name: embedder.compute_txt_embedding(user_query=example[existing_col_name])[
            0].float().cpu().detach().numpy()
    }
    return ds.map(func)


# find images close to the provided prompt in embedding space
def retrieve_images_by_prompt(prompt, ds_with_faiss, embedder: EmbeddingModel, topk, device="cpu", plot=True):
    prompt_embedding = embedder.compute_txt_embedding(prompt).float().cpu().detach().numpy()

    scores, retrieved_examples = ds_with_faiss.get_nearest_examples("image_embeddings", prompt_embedding, k=topk)
    if plot: plot_images(retrieved_examples["image"], topk)
    return scores, retrieved_examples
