from datasets import load_dataset
from .embedding import EmbeddingModel, EmbedderName
from .vlm import VLM
import time
import torch
import math
import hashlib
from typing import Literal
from strenum import StrEnum
import itertools
from tqdm import tqdm
from pathlib import Path


class DatasetName(StrEnum):
    VIDORE_SYN_AI = "vidore/syntheticDocQA_artificial_intelligence_test"
    VIDORE_SYN_ENERGY = "vidore/syntheticDocQA_energy_test"
    VIDORE_SYN_GOVREP = "vidore/syntheticDocQA_government_reports_test"
    VIDORE_SYN_HEALTH = "vidore/syntheticDocQA_healthcare_industry_test"


class ViDoReDataset:
    def __init__(self, ds_name: str, do_retrieval: bool = True, embedder: EmbeddingModel=None, train_ratio: float = 0.8, num_images: int = -1):
        self.do_retrieval = do_retrieval
        self.train_ratio = train_ratio
        self.ds_name = ds_name

        self.embeddings_folder = "../data/embeddings/"

        self.ds = load_dataset(ds_name, split='test')
        self.num_queries = 100
        self.num_images = len(self.ds) if num_images == -1 else num_images
        self.num_images_orig = self.num_images

        # only queries are split into train and test splits, images (knowledge base are common to both)
        self.num_train = int(self.num_queries * train_ratio)
        self.num_test = self.num_queries - self.num_train

        self.queries = self.ds[:self.num_queries]['query']
        self.images = self.ds[:self.num_images]['image']

        self.queries_train = self.queries[:self.num_train]
        self.queries_test = self.queries[self.num_train:]

        self.ground_truth = torch.arange(end=self.num_queries) # query (i)'s ground truth image is image (i)

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
            self.num_images = len(self.images)
            if self.do_retrieval: self.image_embeddings = torch.cat((self.image_embeddings, self.embedder.compute_img_embedding([adv_img], None)), dim=0)
        else:
            self.images[-1] = adv_img
            if self.do_retrieval: self.image_embeddings[-1,:] = self.embedder.compute_img_embedding([adv_img], None)
        print(f"Added adversarial image embeddings in {time.time()-t:.2f}s")
    
   
    def attempt_load_embeddings(self,):
        loaded_obj = torch.load(self.embeddings_filename(), weights_only=False)
        self.image_embeddings = loaded_obj['image_embeddings'].type(self.embedder.model.dtype)
        self.query_embeddings = loaded_obj['query_embeddings'].type(self.embedder.model.dtype)  
        print("Loaded precomputed embeddings from disk.")

    def embeddings_filename(self,):
        emb_str = f"{self.ds_name}{self.embedder.name}"
        hash_str = hashlib.md5(emb_str.encode()).hexdigest()
        return self.embeddings_folder + f"embeds_{hash_str}.pt"


    def compute_embeddings(self, batch_size=None, for_queries=True, for_images=True):
        if for_images:
            # batching yields faster results (batch_size=16 seems good)
            if batch_size is None: batch_size = self.num_images
            t = time.time()
            img_embeds = [self.embedder.compute_img_embedding(self.images[i*batch_size:(i+1)*batch_size], None) for i in tqdm(range(math.ceil(len(self.images)/batch_size)))]
            self.image_embeddings = torch.cat(tuple(img_embeds), dim=0)
            print(f"Computed {self.num_images} image embeddings in {time.time()-t:.2f}s")
        if for_queries:
            t = time.time()
            self.query_embeddings = self.embedder.compute_txt_embedding(self.queries)
            print(f"Computed {self.num_queries} query embeddings in {time.time()-t:.2f}s")
        
        # save to file
        dict_to_save = {
            "image_embeddings": self.image_embeddings, 
            "query_embeddings": self.query_embeddings,
            "dataset":          self.ds_name,
            "model_name":       self.embedder.name
        }
        torch.save(dict_to_save, self.embeddings_filename())
        print("Saved computed embeddings to disk.")
    
    
    def create_retriever_score_table(self, loss_type: Literal["mse"]|Literal["cos"]="mse"):
        """
        creates a [num_queries x num_images] tensor of scores/losses
        """
        img_embs = self.image_embeddings.unsqueeze(0).repeat(self.num_queries, 1, 1)
        txt_embs = self.query_embeddings.unsqueeze(1).repeat(1, self.num_images, 1)
        
        if loss_type == "mse":
            losses = torch.nn.functional.mse_loss(txt_embs, img_embs, reduction="none")
            losses = torch.mean(losses, dim=-1)
        elif loss_type == "cos":
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
            
            correct_retrievals = [self.ground_truth[i] in topk.indices[i] for i in range(self.num_queries)]
            accuracy_train = sum(correct_retrievals[:self.num_train]) / self.num_train
            accuracy_test = sum(correct_retrievals[self.num_train:]) / self.num_test

            # if include_adv=False, then will always be zero
            adversarial_retrievals = [self.num_images_orig in topk.indices[i] for i in range(self.num_queries)]
            asr_train = sum(adversarial_retrievals[:self.num_train]) / self.num_train
            asr_test = sum(adversarial_retrievals[self.num_train:]) / self.num_test

            metric_dict[f"loss_{loss_type}_topk_{k}"] = {"acc_train": accuracy_train, "acc_test": accuracy_test, "asr_train": asr_train, "asr_test": asr_test}
            # keep only the retrievals for highest k, should include thodse for small k
            retrievals[f"loss_{loss_type}"] = topk.indices

        return metric_dict, retrievals
    
    
    def evaluate_generation(self, vlm: VLM, image_tensor, target_generation: str, metric="exact", eval_train=False, print_gen=False):
        """
        By default, we use the test dataset
        """
        t = time.time()

        queries = self.queries_train if eval_train else self.queries_test
        generations = vlm.generate(image_tensor, queries, overwrite=True)
        # extract only the VLM reply
        generations = [g.split("Assistant:")[-1].strip() for g in generations]

        if print_gen: print(generations)

        if metric == "exact":
            correct_generations = [g == target_generation for g in generations]
        asr = sum(correct_generations) / len(queries)

        print(f"Evaluated {len(queries)} generations in {time.time()-t:.2f}s")

        return asr, generations