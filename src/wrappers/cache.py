from functools import lru_cache
from typing import Optional

from utils.logger import logger

from .dataset import DatasetName, Dataset, create_dataset
from .embedded_dataset import EmbeddedDataset
from .embedding import EmbeddingModel, EmbedderName
from .judge import JudgeVLM
from .text_embedding import TextEmbeddingModel, TextEmbedderName
from .vlm import VLM, VLMName


@lru_cache(maxsize=1)
def get_text_embedder(gen_text_embedder: TextEmbedderName, device: str) -> TextEmbeddingModel:
    return TextEmbeddingModel(gen_text_embedder, device=device)


@lru_cache(maxsize=1)
def get_vlm(model_name_vlm: VLMName, device: str) -> VLM:
    logger.info(f"VLM: loading {model_name_vlm}")
    return VLM(model_name_vlm, device)


@lru_cache(maxsize=1)
def get_judge(model_name_jdg: VLMName, device: str) -> JudgeVLM:
    logger.info(f"JudgeVLM: loading {model_name_jdg}")
    return JudgeVLM(model_name_jdg, device)


@lru_cache(maxsize=1)
def get_embedder(model_name_emb: EmbedderName, quantize: bool, colpali_only_images: bool, device: str) -> EmbeddingModel:
    logger.info(f"Embedding: loading {model_name_emb}")
    return EmbeddingModel(model_name_emb, device, quantize=quantize, colpali_only_images=colpali_only_images)


# @lru_cache(maxsize=1)  # Datasets are not immutable, caching can lead to odd behaviour
def get_dataset(ds_name: DatasetName, train_ratio: float = 0.8, num_images: Optional[int] = None) -> Dataset:
    logger.info(f"Dataset: loading {ds_name}")
    return create_dataset(ds_name=ds_name, train_ratio=train_ratio, num_images=num_images)


# @lru_cache(maxsize=1)  # Datasets are not immutable, caching can lead to odd behaviour
def get_embedded_dataset(
    dataset: Dataset, model_name_emb: EmbedderName, quantize: bool, colpali_only_images: bool, device: str
) -> EmbeddedDataset:
    logger.info(f"Dataset: loading {dataset.ds_name} with {model_name_emb}")
    embedder = get_embedder(model_name_emb, quantize=quantize, colpali_only_images=colpali_only_images, device=device)
    return EmbeddedDataset(dataset, embedder)
