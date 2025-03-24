from datasets import load_dataset
from .embedding import EmbeddingModel
from .vlm import VLM
import time
import torch
import math

class ViDoReDataset:
    def __init__(self, ds_name: str, do_retrieval: bool = True, embedder=None, train_ratio: float = 0.8):
        self.do_retrieval = do_retrieval
        self.train_ratio = train_ratio

        self.ds = load_dataset(ds_name, split='test')
        self.num_queries = 100
        self.num_images = len(self.ds)
        self.num_images_orig = self.num_images

        # only queries are split into train and test splits, images (knowledge base are common to both)
        self.num_train = int(self.num_queries * train_ratio)
        self.num_test = self.num_queries - self.num_train

        # overrides (just for testing)
        # self.num_queries = 50
        # self.num_images = 200

        self.queries = self.ds[:self.num_queries]['query']
        self.images = self.ds[:self.num_images]['image']

        self.queries_train = self.queries[:self.num_train]
        self.queries_test = self.queries[self.num_train:]

        self.ground_truth = torch.arange(end=self.num_queries) # query (i)'s ground truth image is image (i)

        if self.do_retrieval:
            self.embedder = embedder
            self.compute_embeddings(batch_size=16)

    
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
    
   
    def compute_embeddings(self, batch_size=None, for_queries=True, for_images=True):
        if for_images:
            # batching yields faster results (batch_size=16 seems good)
            if batch_size is None: batch_size = self.num_images
            t = time.time()
            img_embeds = [self.embedder.compute_img_embedding(self.images[i*batch_size:(i+1)*batch_size], None) for i in range(math.ceil(len(self.images)/batch_size))]
            self.image_embeddings = torch.cat(tuple(img_embeds), dim=0)
            print(f"Computed {self.num_images} image embeddings in {time.time()-t:.2f}s")
        if for_queries:
            t = time.time()
            self.query_embeddings = self.embedder.compute_txt_embedding(self.queries)
            print(f"Computed {self.num_queries} query embeddings in {time.time()-t:.2f}s")
    
    
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
            losses = -torch.nn.functional.cosine_similarity(txt_embs, img_embs, dim=-1)
        return losses

    
    def evaluate_retrieval(self, k=1, loss_type="mse", include_adv=True):
        """
        Accuracy@k: whether the top-k retrieved images include the ground truth image
        """
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

        return {"acc_train": accuracy_train, "acc_test": accuracy_test, "asr_train": asr_train, "asr_test": asr_test}
    
    
    def evaluate_generation(self, vlm: VLM, image_tensor, target_generation: str, metric="exact", eval_train=False):
        """
        By default, we use the test dataset
        """
        t = time.time()

        queries = self.queries_train if eval_train else self.queries_test
        generations = vlm.generate(image_tensor, queries, overwrite=True)
        # extract only the VLM reply
        generations = [g.split("Assistant:")[-1].strip() for g in generations]

        if metric == "exact":
            correct_generations = [g == target_generation for g in generations]
        asr = sum(correct_generations) / len(queries)

        print(f"Evaluated {len(queries)} generations in {time.time()-t:.2f}s")

        return asr, generations