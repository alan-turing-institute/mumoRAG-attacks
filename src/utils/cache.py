from functools import lru_cache
from typing import Tuple

from .text_embedding import TextEmbeddingModel, TextEmbedderName
from .vlm import VLM, VLMName
from .judge import JudgeVLM
from .embedding import EmbeddingModel, EmbedderName
from .dataset import ViDoReDataset, DatasetName
from .logger import logger


@lru_cache(maxsize=1)
def load_text_embedder(gen_text_embedder: TextEmbedderName, device: str) ->TextEmbeddingModel:
    return TextEmbeddingModel(gen_text_embedder, device=device)

@lru_cache(maxsize=1)
def load_vlm(model_name_vlm: VLMName, device: str) -> VLM:
    logger.info(f"VLM: loading {model_name_vlm}")
    return VLM(model_name_vlm, device)

@lru_cache(maxsize=1)
def load_judge(model_name_jdg: VLMName, device: str) -> VLM:
    logger.info(f"JudgeVLM: loading {model_name_jdg}")
    return JudgeVLM(model_name_jdg, device)

@lru_cache(maxsize=1)
def _load_embedder(model_name_emb: EmbedderName, quantize: bool, colpali_only_images: bool, device: str) -> EmbeddingModel:
    logger.info(f"Embedding: loading {model_name_emb}")
    return EmbeddingModel(model_name_emb, device, quantize=quantize, colpali_only_images=colpali_only_images)


@lru_cache(maxsize=1)
def load_embedder_and_dataset(
    ds_name: DatasetName, model_name_emb: EmbedderName, do_retrieval: bool, quantize: bool, colpali_only_images: bool, device: str
) -> Tuple[EmbeddingModel, ViDoReDataset]:
    logger.info(f"Dataset: loading {ds_name} with {model_name_emb}")
    embedder = _load_embedder(model_name_emb, quantize=quantize, colpali_only_images=colpali_only_images, device=device)
    dataset = ViDoReDataset(ds_name, do_retrieval=do_retrieval, embedder=embedder)
    return embedder, dataset
