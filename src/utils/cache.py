from functools import lru_cache
from typing import Tuple

from .text_embedding import TextEmbeddingModel, TextEmbedderName
from .vlm import VLM, VLMName
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
def _load_embedder(model_name_emb: EmbedderName, device: str) -> EmbeddingModel:
    logger.info(f"Embedding: loading {model_name_emb}")
    return EmbeddingModel(model_name_emb, device)


@lru_cache(maxsize=1)
def load_embedder_and_dataset(
    ds_name: DatasetName, model_name_emb: EmbedderName, do_retrieval: bool, device: str
) -> Tuple[EmbeddingModel, ViDoReDataset]:
    logger.info(f"Dataset: loading {ds_name} with {model_name_emb}")
    embedder = _load_embedder(model_name_emb, device)
    dataset = ViDoReDataset(ds_name, do_retrieval=do_retrieval, embedder=embedder)
    return embedder, dataset
