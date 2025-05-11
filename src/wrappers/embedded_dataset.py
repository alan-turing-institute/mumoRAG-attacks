import itertools
import math
import time

import torch
from tqdm import tqdm

from config import EMBEDDINGS_FOLDER
from wrappers.embedding import EmbeddingModel, EmbedderName, EmbeddingLoss, COLPALI_MODELS, score_multi_vector_modified
from utils.logger import logger
from .dataset import Dataset


def make_safe_filename(s):
    def safe_char(c):
        if c.isalnum():
            return c
        else:
            return "_"
    return "".join(safe_char(c) for c in s).rstrip("_")


class EmbeddedDataset:
    def __init__(self, dataset: Dataset, embedder: EmbeddingModel):
        self.dataset = dataset
        self.embedder = embedder

        self.embeddings_folder = EMBEDDINGS_FOLDER

        self.embedder = embedder
        self.query_embeddings = None
        self.image_embeddings = None
        try:
            self.attempt_load_embeddings()
        except FileNotFoundError:
            logger.info("Precomputing Embeddings ...")
            batch_size = 4 if self.embedder.name == EmbedderName.COLPALI else 16
            with torch.no_grad():
                self.compute_embeddings(batch_size=batch_size)

    def attempt_load_embeddings(self):
        loaded_obj = torch.load(self.embeddings_filename, weights_only=False)
        self.image_embeddings = loaded_obj['image_embeddings'].type(self.embedder.model.dtype)
        self.query_embeddings = loaded_obj['query_embeddings'].type(self.embedder.model.dtype)
        logger.info(f"Loaded precomputed embeddings from disk {self.embeddings_filename}")

    @property
    def embeddings_filename(self):
        emb_str = f"{self.dataset.ds_name}_{self.embedder.name}"
        if self.embedder.colpali_only_images: emb_str += f"_{self.embedder.colpali_only_images}"
        return self.embeddings_folder / f"embeds_{make_safe_filename(emb_str)}.pt"

    def compute_embeddings(self, batch_size=None, for_queries=True, for_images=True):
        if for_images:
            # batching yields faster results (batch_size=16 seems good)
            if batch_size is None: batch_size = len(self.dataset.images)
            t = time.time()
            img_embeds = [self.embedder.compute_img_embedding(self.dataset.images[i * batch_size:(i + 1) * batch_size], None)
                          for i in tqdm(range(math.ceil(len(self.dataset.images) / batch_size)))]
            self.image_embeddings = torch.cat(tuple(img_embeds), dim=0)
            logger.info(f"Computed {len(self.dataset.images)} image embeddings in {time.time() - t:.2f}s")
        if for_queries:
            t = time.time()
            self.query_embeddings = self.embedder.compute_txt_embedding(self.dataset.queries)
            logger.info(f"Computed {len(self.dataset.queries)} query embeddings in {time.time() - t:.2f}s")

        # save to file
        dict_to_save = {
            "image_embeddings": self.image_embeddings,
            "query_embeddings": self.query_embeddings,
            "wrappers": self.dataset.ds_name,
            "model_name": self.embedder.name
        }
        torch.save(dict_to_save, self.embeddings_filename)
        logger.info(f"Saved computed embeddings to disk {self.embeddings_filename}")

    def add_adv_image(self, adv_img):
        """
        Adds the adversarial image to the database along with its embedding
        """
        if len(self.dataset.images) == self.dataset.num_images_orig:
            self.image_embeddings = torch.cat((self.image_embeddings, self.embedder.compute_img_embedding([adv_img], None)), dim=0)
        else:
            self.image_embeddings[-1,:] = self.embedder.compute_img_embedding([adv_img], None)
        self.dataset.add_adv_image(adv_img)


    def create_retriever_score_table(self, loss_type: EmbeddingLoss):
        """
        creates a [num_queries x num_images] tensor of scores/losses
        """

        if self.embedder.name in COLPALI_MODELS and loss_type != EmbeddingLoss.COS_AVGEMB:
            return -1 * score_multi_vector_modified(qs=self.query_embeddings, ps=self.image_embeddings, loss=loss_type)

        if loss_type == EmbeddingLoss.COS_AVGEMB:
            img_emb_avg = self.image_embeddings.mean(dim=1)
            txt_emb_avg = self.query_embeddings.mean(dim=1)

            img_embs = img_emb_avg.unsqueeze(0).repeat(txt_emb_avg.shape[0], 1, 1)
            txt_embs = txt_emb_avg.unsqueeze(1).repeat(1, img_embs.shape[1], 1)

            return 1 - torch.nn.functional.cosine_similarity(img_embs, txt_embs, dim=-1)

        img_embs = self.image_embeddings.unsqueeze(0).repeat(self.query_embeddings.shape[0], 1, 1)
        txt_embs = self.query_embeddings.unsqueeze(1).repeat(1, self.image_embeddings.shape[1], 1)

        match loss_type:
            case EmbeddingLoss.MSE:
                losses = torch.nn.functional.mse_loss(txt_embs, img_embs, reduction="none")
                return torch.mean(losses, dim=-1)
            case EmbeddingLoss.COS:
                return 1 - torch.nn.functional.cosine_similarity(txt_embs, img_embs, dim=-1)
            case _:
                raise ValueError(f"Unknown loss type: {loss_type}")

    def evaluate_retrieval(self, ks: list[int], loss_types: list[EmbeddingLoss], is_targeted: bool, target_query_idx, include_adv=True):
        """
        Accuracy@k: whether the top-k retrieved images include the ground truth image
        """
        metric_dict = {}
        retrievals = {}
        ks = sorted(ks)  # sort ascendingly
        loss_types_and_topks = itertools.product(loss_types, ks)

        for loss_type, k in loss_types_and_topks:
            keyname = f"loss_{loss_type}_topk_{k}"
            losses = self.create_retriever_score_table(loss_type)
            if not include_adv:
                losses = losses[:, :self.dataset.num_images_orig]

            topk = torch.topk(losses, k=k, dim=-1, largest=False, sorted=True)

            correct_retrievals = [any(x in topk.indices[i] for x in self.dataset.ground_truth_retrievals[i]) for i in
                                  range(len(self.dataset.queries))]
            accuracy_train = sum(correct_retrievals[:self.dataset.num_train]) / self.dataset.num_train
            accuracy_test = sum(correct_retrievals[self.dataset.num_train:]) / self.dataset.num_test
            accuracy = accuracy_train * self.dataset.train_ratio + accuracy_test * (1 - self.dataset.train_ratio)

            # if include_adv=False, then will always be zero
            adversarial_retrievals = [self.dataset.num_images_orig in topk.indices[i] for i in range(len(self.dataset.queries))]
            asr_train = sum(adversarial_retrievals[:self.dataset.num_train]) / self.dataset.num_train
            asr_test = sum(adversarial_retrievals[self.dataset.num_train:]) / self.dataset.num_test

            metric_dict[keyname] = {"acc": accuracy, "asr_train": asr_train, "asr_test": asr_test}

            # targeted attack metrics
            if is_targeted:
                metric_dict[keyname].update(self.dataset.compute_targeted_metrics(adversarial_retrievals, target_query_idx))

            # keep only the retrievals for highest (lastest) k, should include those for small k
            retrievals[f"loss_{loss_type}"] = {"train": topk.indices[:self.dataset.num_train],
                                               "test": topk.indices[self.dataset.num_train:]}

        return metric_dict, retrievals
