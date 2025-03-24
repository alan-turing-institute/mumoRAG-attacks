import torchvision.transforms as T
from utils.utils import get_device
import torch
from utils.attack import rag_attack
from utils.embedding import EmbeddingModel
from utils.vlm import VLM
from utils.scheduler import LearningRateScheduler
from utils.dataset import ViDoReDataset
from utils.attack_config import AttackConfig

# general config
device = get_device(prefer_mps=True)
ds_name = "vidore/syntheticDocQA_artificial_intelligence_test"
model_name_emb = "openai/clip-vit-base-patch16" # "openai/clip-vit-base-patch16" or "nomic-ai/nomic-embed-vision-v1.5", "jinaai/jina-clip-v2" (large), "jinaai/jina-clip-v1", "vidore/colSmol-256M", "vidore/colpali-v1.3-hf" (3B), "google/siglip2-base-patch16-224"
model_name_vlm = "HuggingFaceTB/SmolVLM-256M-Instruct" # "HuggingFaceTB/SmolVLM-256M-Instruct", "HuggingFaceTB/SmolVLM-500M-Instruct", "microsoft/Florence-2-large", , "naver-clova-ix/donut-base-finetuned-docvqa", "Qwen/Qwen2.5-VL-3B-Instruct"
target_answer = "I will not reply to you!"
save_folder = "data/attacks/"

# attack config
chosen_index = 150
n_gradient_steps = 40
max_perturbation = 8.0/255
print_every=5
lr_start, lr_end = 255*(3e-3), 255*(3e-4)
max_batch_size_per_iter=2
gradient_acc_steps=4
lambda_emb=1
lambda_vlm=0.5
emb_train_loss_type = "mse" # the loss type used to train the attack: mse, l2, cos 

attack_config = AttackConfig(
    ds_name=ds_name,
    model_name_emb=model_name_emb,
    model_name_vlm=model_name_vlm,
    target_answer=target_answer,
    chosen_index=chosen_index,
    max_perturbation=max_perturbation,
    n_gradient_steps=n_gradient_steps,
    lr_start=lr_start,
    lr_end=lr_end,
    max_batch_size_per_iter=max_batch_size_per_iter,
    gradient_acc_steps=gradient_acc_steps,
    lambda_emb=lambda_emb,
    lambda_vlm=lambda_vlm,
    emb_train_loss_type=emb_train_loss_type
)


# load embedding model and VLM
embedder = EmbeddingModel(model_name_emb, device)
vlm = VLM(model_name_vlm, device)
print("Loaded models.")

# load dataset
ds = ViDoReDataset(ds_name, do_retrieval=False, embedder=embedder)
query_strings = ds.queries_train
print("Loaded dataset.")

# choose attacked image
chosen_image = ds.images[chosen_index]
chosen_image = chosen_image.resize((512,512)) # this can save memory (also setting this to VLM image size with resample=0 -> reduce errors)
chosen_image = T.PILToTensor()(chosen_image) # choose from after 100 since those do not have associated queries
chosen_image = chosen_image.type(vlm.model.dtype)
initial_chosen_image = chosen_image.clone()
chosen_image.shape

# train the attack
# NOTE: lr = 255*(3e-2) works well with cosine similarity
image_adv = rag_attack(
    raw_image=chosen_image,
    embedder=embedder,
    vlm=vlm,
    user_query=query_strings,
    config=attack_config,
    print_every=print_every,
    device=device
)

print(f"MSE: {torch.nn.functional.mse_loss(image_adv, initial_chosen_image)}")
print(f"Linf: {(initial_chosen_image - image_adv).norm(p=float('inf'))}")

# save adv image
torch.save(image_adv, save_folder+attack_config.create_filename())
print("Saved adversarial image.")