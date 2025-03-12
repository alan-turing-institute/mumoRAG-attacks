from datasets import load_dataset
from utils.embedding import EmbeddingModel
import time
import torch

class ViDoReDataset():
    def __init__(self, ds_name: str):
        self.ds = load_dataset(ds_name, split='test')
        self.num_queries = 100
        self.num_images = len(self.ds)
        self.num_images_orig = self.num_images
        
        # overides (just for testing)
        # self.num_queries = 50
        # self.num_images = 200

        self.queries = self.ds[:self.num_queries]['query']
        self.images = self.ds[:self.num_images]['image']

        self.ground_truth = torch.arange(end=self.num_queries) # query (i)'s ground truth image is image (i)
    
    def add_adv_images(self, adv_imgs):
        self.images.extend(adv_imgs)
        self.num_images = len(self.images)
    
    def compute_embeddings(self, embedder: EmbeddingModel, for_queries=True, for_images=True):
        if for_images:
            t = time.time()
            self.image_embeddings = embedder.compute_img_embedding(self.images, None)
            print(f"Computed {self.num_images} image embddings in {time.time()-t:.2f}s")
        if for_queries:
            t = time.time()
            self.query_embeddings = embedder.compute_txt_embedding(self.queries)
            print(f"Computed {self.num_queries} query embddings in {time.time()-t:.2f}s")
    
    def create_retriever_score_table(self, loss_type="mse"):
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

    def evaluate(self, k=1, loss_type="mse", include_adv=True):
        """
        Accuracy@k: whether the top-k retrieved images include the ground truth image
        """
        losses = self.create_retriever_score_table(loss_type)
        if not include_adv:
            losses = losses[:,:self.num_images_orig]
        
        topk = torch.topk(losses, k=k, dim=-1, largest=False, sorted=True)
        accuracy = sum([self.ground_truth[i] in topk.indices[i] for i in range(self.num_queries)]) / self.num_queries

        # top-k least losses, top-k most relevant images, accuracy
        return topk.values, topk.indices, accuracy            