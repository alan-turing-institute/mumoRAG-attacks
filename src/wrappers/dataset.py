import math
import random
from collections import defaultdict
from typing import Optional

import torch
from datasets import load_dataset
from strenum import StrEnum
from tqdm import tqdm

from utils.logger import logger
from wrappers.text_embedding import TextEmbeddingModel
from wrappers.vlm import VLM, SMOL_VLMS


class DatasetName(StrEnum):
    VIDORE_SYN_AI = "vidore/syntheticDocQA_artificial_intelligence_test"
    VIDORE_SYN_ENERGY = "vidore/syntheticDocQA_energy_test"
    VIDORE_SYN_GOVREP = "vidore/syntheticDocQA_government_reports_test"
    VIDORE_SYN_HEALTH = "vidore/syntheticDocQA_healthcare_industry_test"
    VIDORE_V2_ESG = "vidore/restaurant_esg_reports_beir"
    VIDORE_V2_ECO = "vidore/synthetic_economics_macro_economy_2024_filtered_v1.0"

def filter_none(arr):
    return [x for x in arr if x is not None]


class Dataset:
    def __init__(
        self,
        ds_name: DatasetName,
        images: list,
        queries: list,
        ground_truth: list,
        train_ratio: float,
    ):
        self.ds_name = ds_name

        self.images = images
        self.num_images_orig = len(self.images)

        self.queries = queries

        # only queries are split into train and test splits, images (knowledge base are common to both)
        self.train_ratio = train_ratio
        self.num_train = int(len(self.queries) * train_ratio)
        self.num_test = len(self.queries) - self.num_train

        self.queries_train = self.queries[: self.num_train]
        self.queries_test = self.queries[self.num_train :]

        self.ground_truth = ground_truth

    def add_adv_image(self, adv_img):
        """
        Adds the adversarial image to the database
        """
        # keep the adversarial image at position [-1]
        if len(self.images) == self.num_images_orig:
            self.images.append(adv_img)
        else:
            self.images[-1] = adv_img

    def sample_images_from_ds(self, fraction: float):
        n_images = math.floor(fraction * self.num_images_orig)
        return random.sample(self.images, k=n_images)

    def evaluate_generation(
            self,
            vlm: VLM,
            image_tensor,
            target_generation: str,
            metrics: list[str],
            text_embedder: TextEmbeddingModel,
            retrievals: dict,
            generation_topk_list: list[int],
            batch_size=None,
            eval_train=False,
            print_gen=False,
    ):
        """
        By default, we use the test wrappers
        """

        queries = self.queries_train if eval_train else self.queries_test
        metric_dict = defaultdict(dict)

        for generation_topk in generation_topk_list:
            logger.info(
                f'Generating responses to {"train" if eval_train else "test"} set queries: using top ({generation_topk}) retrieved images')

            retrieved_images, adv_indices = self.retrieved_idx_to_img(
                retrieved_indices=retrievals[list(retrievals.keys())[0]], topk=generation_topk)

            if batch_size is None:
                generations = vlm.generate(image_tensor, queries, overwrite=True, retrieved_images=retrieved_images,
                                           adv_indices=adv_indices)
            else:
                generations = []
                for i in tqdm(range(math.ceil(len(queries) / batch_size))):
                    queries_batch = queries[i * batch_size:(i + 1) * batch_size]
                    retrieved_images_batch = retrieved_images[i * batch_size:(i + 1) * batch_size]
                    adv_indices_batch = adv_indices[i * batch_size:(i + 1) * batch_size]
                    generations.extend(vlm.generate(image_tensor, queries_batch, overwrite=True,
                                                    retrieved_images=retrieved_images_batch,
                                                    adv_indices=adv_indices_batch))

            # extract only the VLM reply (Smol and Qwen need different splittings)
            split_str = "Assistant:" if vlm.name in SMOL_VLMS else "assistant\n"
            generations = [g.split(split_str)[-1].strip() for g in generations]

            if print_gen: logger.info(generations)

            keyname = f"gen_topk_{generation_topk}"

            if "exact" in metrics:
                # exact match of VLM generation and target answer
                correct_generations = [g == target_generation for g in generations]
                metric_dict[keyname]["exact"] = sum(correct_generations) / len(queries)

            if "embed" in metrics:
                # similarity score between VLM generation and target answer in [0,1]
                similarity = text_embedder.compare_embeddings(generations, target_generation, similarity_metric="cos")
                metric_dict[keyname]["embed"] = similarity.mean().item()

        return metric_dict, generations

    def retrieved_idx_to_img(self, retrieved_indices: torch.tensor, topk: int):
        if topk == -1:
            # ignore retrieval results, and assume adversarial image is always retrieved
            n_queries = retrieved_indices.shape[0]
            retrieved_images = [[self.images[self.num_images_orig]] for _ in range(n_queries)]
            adv_indices = [0 for _ in range(n_queries)]
        else:
            # use top retrieved images
            retrieved_images = [
                [
                    self.images[i] for i in indices_per_query[:topk]
                ]
                for indices_per_query in retrieved_indices
            ]
            # where is the adversarial image located within the top-k?
            adv_indices = [
                indices_per_query[:topk].tolist().index(
                    self.num_images_orig) if self.num_images_orig in indices_per_query[:topk] else -1
                for indices_per_query in retrieved_indices
            ]

        return retrieved_images, adv_indices


def create_dataset(
    ds_name: DatasetName, train_ratio: float = 0.8, num_images: Optional[int] = None
) -> Dataset:
    if ds_name.startswith("vidore"):
        if "V2" in ds_name.name:
            corpus = load_dataset(ds_name, "corpus", split="all")
            images = corpus["image"]
            qrels = load_dataset(ds_name, "qrels", split="all")
            queries = load_dataset(ds_name, "queries", split="all")
            queries = queries["query"]
            ground_truth = [[] for _ in range(len(queries))]
            for row in qrels:
                ground_truth[row["query-id"]].append(row["corpus-id"])
        else:
            ds = load_dataset(ds_name, split="test")
            images = filter_none(ds["image"])
            queries = filter_none(ds["query"])
            ground_truth = [[i] for i in range(len(images))]
        if num_images is not None:
            images = images[:num_images]
        return Dataset(
            ds_name=ds_name,
            images=images,
            queries=queries,
            ground_truth=ground_truth,
            train_ratio=train_ratio,
        )
    raise ValueError(f"Dataset {ds_name} is not supported")

