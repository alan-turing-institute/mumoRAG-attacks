import hashlib
import itertools
import time
from pathlib import Path
from typing import Literal, Optional

import math
import torch
from datasets import load_dataset
from strenum import StrEnum
from tqdm import tqdm

from .embedding import EmbeddingModel, EmbedderName, EmbeddingLoss, COLPALI_MODELS, score_multi_vector_modified
from .vlm import VLM, SMOL_VLMS, QWEN_VLMS
from .text_embedding import TextEmbeddingModel, TextEmbedderName


class DatasetName(StrEnum):
    VIDORE_SYN_AI = "vidore/syntheticDocQA_artificial_intelligence_test"
    VIDORE_SYN_ENERGY = "vidore/syntheticDocQA_energy_test"
    VIDORE_SYN_GOVREP = "vidore/syntheticDocQA_government_reports_test"
    VIDORE_SYN_HEALTH = "vidore/syntheticDocQA_healthcare_industry_test"
    VIDORE_V2_ESG = "vidore/restaurant_esg_reports_beir"
    VIDORE_V2_ECO = "vidore/synthetic_economics_macro_economy_2024_filtered_v1.0"


def filter_none(arr):
    return [x for x in arr if x is not None]


class ViDoReDataset:
    def __init__(self, ds_name: DatasetName, do_retrieval: bool = True, embedder: EmbeddingModel=None, train_ratio: float = 0.8, num_images: Optional[int] = None):
        self.do_retrieval = do_retrieval
        self.train_ratio = train_ratio
        self.ds_name = ds_name

        self.embeddings_folder = Path(__file__).parents[2] / "data/embeddings/"

        if "V2" in ds_name.name:
            corpus = load_dataset(ds_name, "corpus", split='all')
            qrels = load_dataset(ds_name, "qrels", split='all')
            queries = load_dataset(ds_name, "queries", split='all')
            self.images = corpus['image']
            self.queries = queries['query']
            self.ground_truth = [[] for _ in range(len(queries))]
            for row in qrels:
                self.ground_truth[row['query-id']].append(row['corpus-id'])
        else:
            ds = load_dataset(ds_name, split='test')
            self.images = filter_none(ds['image'])
            self.queries = filter_none(ds['query'])
            self.ground_truth = [[i] for i in range(len(self.images))]

        if num_images is not None:
            self.images = self.images[:num_images]

        self.num_images_orig = len(self.images)

        # only queries are split into train and test splits, images (knowledge base are common to both)
        self.num_train = int(len(self.queries) * train_ratio)
        self.num_test = len(self.queries) - self.num_train

        self.queries_train = self.queries[:self.num_train]
        self.queries_test = self.queries[self.num_train:]

        if self.do_retrieval:
            self.embedder = embedder
            try:
                self.attempt_load_embeddings()
            except FileNotFoundError: 
                print("Precomputing Embeddings ...")
                batch_size = 4 if self.embedder.name == EmbedderName.COLPALI_HF else 16
                with torch.no_grad():
                    self.compute_embeddings(batch_size=batch_size)

    def add_adv_image(self, adv_img):
        """
        Adds the adversarial image to the database along with its embedding
        """
        # keep the adversarial image at position [-1]
        t = time.time()
        if len(self.images) == self.num_images_orig:
            self.images.append(adv_img)
            if self.do_retrieval: self.image_embeddings = torch.cat((self.image_embeddings, self.embedder.compute_img_embedding([adv_img], None)), dim=0)
        else:
            self.images[-1] = adv_img
            if self.do_retrieval: self.image_embeddings[-1,:] = self.embedder.compute_img_embedding([adv_img], None)
        # print(f"Added adversarial image embeddings in {time.time()-t:.2f}s")
    
   
    def attempt_load_embeddings(self,):
        loaded_obj = torch.load(self.embeddings_filename(), weights_only=False)
        self.image_embeddings = loaded_obj['image_embeddings'].type(self.embedder.model.dtype)
        self.query_embeddings = loaded_obj['query_embeddings'].type(self.embedder.model.dtype)  
        print("Loaded precomputed embeddings from disk.")

    def embeddings_filename(self,):
        emb_str = f"{self.ds_name}{self.embedder.name}"
        hash_str = hashlib.md5(emb_str.encode()).hexdigest()
        return self.embeddings_folder / f"embeds_{hash_str}.pt"


    def compute_embeddings(self, batch_size=None, for_queries=True, for_images=True):
        if for_images:
            # batching yields faster results (batch_size=16 seems good)
            if batch_size is None: batch_size = len(self.images)
            t = time.time()
            img_embeds = [self.embedder.compute_img_embedding(self.images[i*batch_size:(i+1)*batch_size], None) for i in tqdm(range(math.ceil(len(self.images)/batch_size)))]
            self.image_embeddings = torch.cat(tuple(img_embeds), dim=0)
            print(f"Computed {len(self.images)} image embeddings in {time.time()-t:.2f}s")
        if for_queries:
            t = time.time()
            self.query_embeddings = self.embedder.compute_txt_embedding(self.queries)
            print(f"Computed {len(self.queries)} query embeddings in {time.time()-t:.2f}s")
        
        # save to file
        dict_to_save = {
            "image_embeddings": self.image_embeddings, 
            "query_embeddings": self.query_embeddings,
            "dataset":          self.ds_name,
            "model_name":       self.embedder.name
        }
        torch.save(dict_to_save, self.embeddings_filename())
        print("Saved computed embeddings to disk.")
    
    
    def create_retriever_score_table(self, loss_type: EmbeddingLoss):
        """
        creates a [num_queries x num_images] tensor of scores/losses
        """
        if self.embedder.name in COLPALI_MODELS and loss_type != EmbeddingLoss.COS_AVGEMB:
            return -1*score_multi_vector_modified(qs=self.query_embeddings, ps=self.image_embeddings, loss=loss_type)

        if loss_type == EmbeddingLoss.COS_AVGEMB:
            img_embs = self.image_embeddings.mean(dim=1).unsqueeze(0).repeat(len(self.queries), 1, 1)
            txt_embs = self.query_embeddings.mean(dim=1).unsqueeze(1).repeat(1, len(self.images), 1)
            return 1 - torch.nn.functional.cosine_similarity(img_embs, txt_embs, dim=-1)

        img_embs = self.image_embeddings.unsqueeze(0).repeat(len(self.queries), 1, 1)
        txt_embs = self.query_embeddings.unsqueeze(1).repeat(1, len(self.images), 1)

        if loss_type == EmbeddingLoss.MSE:
            losses = torch.nn.functional.mse_loss(txt_embs, img_embs, reduction="none")
            losses = torch.mean(losses, dim=-1)
        elif loss_type == EmbeddingLoss.COS:
            losses = 1-torch.nn.functional.cosine_similarity(txt_embs, img_embs, dim=-1)
        return losses

    
    def evaluate_retrieval(self, ks: list[int], loss_types: list[str], include_adv=True):
        """
        Accuracy@k: whether the top-k retrieved images include the ground truth image
        """
        metric_dict = {}
        retrievals = {}
        ks = sorted(ks) # sort ascendingly
        loss_types_and_topks = itertools.product(loss_types, ks)
        
        for loss_type, k in loss_types_and_topks:
            losses = self.create_retriever_score_table(loss_type)
            if not include_adv:
                losses = losses[:,:self.num_images_orig]
            
            topk = torch.topk(losses, k=k, dim=-1, largest=False, sorted=True)

            correct_retrievals = [any(x in topk.indices[i] for x in self.ground_truth[i]) for i in range(len(self.queries))]
            accuracy_train = sum(correct_retrievals[:self.num_train]) / self.num_train
            accuracy_test = sum(correct_retrievals[self.num_train:]) / self.num_test
            accuracy = accuracy_train * self.train_ratio + accuracy_test * (1-self.train_ratio)

            # if include_adv=False, then will always be zero
            adversarial_retrievals = [self.num_images_orig in topk.indices[i] for i in range(len(self.queries))]
            asr_train = sum(adversarial_retrievals[:self.num_train]) / self.num_train
            asr_test = sum(adversarial_retrievals[self.num_train:]) / self.num_test

            metric_dict[f"loss_{loss_type}_topk_{k}"] = {"acc": accuracy, "asr_train": asr_train, "asr_test": asr_test}
            # keep only the retrievals for highest k, should include those for small k
            retrievals[f"loss_{loss_type}"] = topk.indices

        return metric_dict, retrievals
    
    
    def evaluate_generation(self, vlm: VLM, image_tensor, target_generation: str, metrics: list[str], text_embedder: TextEmbeddingModel, batch_size=None, eval_train=False, print_gen=False):
        """
        By default, we use the test dataset
        """
        t = time.time()

        metric_dict = {}

        queries = self.queries_train if eval_train else self.queries_test
        if batch_size is None:
            generations = vlm.generate(image_tensor, queries, overwrite=True)
        else:
            generations = []
            for i in range(math.ceil(len(queries) / batch_size)):
                queries_batch = queries[i*batch_size:(i+1)*batch_size]
                generations.extend(vlm.generate(image_tensor, queries_batch, overwrite=True))
        
        # extract only the VLM reply (Smol and Qwen need different splittings)
        split_str = "Assistant:" if vlm.name in SMOL_VLMS else "assistant\n"
        generations = [g.split(split_str)[-1].strip() for g in generations]

        if print_gen: print(generations)

        for metric in metrics:
            # exact match of VLM generation and target answer
            if metric == "exact":
                correct_generations = [g == target_generation for g in generations]
                metric_value = sum(correct_generations) / len(queries)
            
            # similarity score between VLM generation and target anser in [0,1]
            if metric == "embed":
                similarity = text_embedder.compare_embeddings(generations, target_generation, similarity_metric="cos")
                metric_value = similarity.mean().item()

            metric_dict[metric] = metric_value

        print(f"Evaluated {len(queries)} generations in {time.time()-t:.2f}s")

        return metric_dict, generations