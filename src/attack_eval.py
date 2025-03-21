import torchvision.transforms as T
from utils.utils import get_device, plot_images, print_memory_consumption
import torch
from utils.attack import rag_attack
from utils.embedding import EmbeddingModel
from utils.vlm import VLM
from utils.scheduler import LearningRateScheduler
from utils.dataset import ViDoReDataset
from utils.attack_config import AttackConfig

device = get_device(prefer_mps=True)
ds_name = "vidore/syntheticDocQA_artificial_intelligence_test"
model_name_emb = "openai/clip-vit-base-patch16" # "openai/clip-vit-base-patch16" or "nomic-ai/nomic-embed-vision-v1.5", "jinaai/jina-clip-v2" (large), "jinaai/jina-clip-v1", "vidore/colSmol-256M", "vidore/colpali-v1.3-hf" (3B), "google/siglip2-base-patch16-224"
model_name_vlm = "HuggingFaceTB/SmolVLM-256M-Instruct" # "HuggingFaceTB/SmolVLM-256M-Instruct", "HuggingFaceTB/SmolVLM-500M-Instruct", "microsoft/Florence-2-large", , "naver-clova-ix/donut-base-finetuned-docvqa", "Qwen/Qwen2.5-VL-3B-Instruct"
emb_test_loss_type = "mse" # the loss type used by the retriever embedding model: mse, l2, cos
target_answer = "I will not reply to you!"
do_retrieval = True # whether to add adversarial image to dataset and test retrieval
do_generation = True
save_folder = "data/attacks/"


# attack config
chosen_index = 150
n_gradient_steps = 40
max_perturbation = 8.0/255
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

# eval config
topk = 1


# load adversarial image
filename = save_folder+attack_config.create_filename()
try:
    image_adv = torch.load(filename) 
except:
    print(f"Error! Could not find file: {filename}! You need to train an attack with this configuration first")
    quit()

# load embedding model and VLM
embedder = EmbeddingModel(model_name_emb, device)
vlm = VLM(model_name_vlm, device)
print("Loaded models.")

# load dataset
ds = ViDoReDataset(ds_name, do_retrieval=do_retrieval, embedder=embedder)
print("Loaded dataset.")



if do_retrieval:
    print("=== Evaluating retrieval ...")
    ds.add_adv_image(T.ToPILImage()(image_adv/255))
    metric_dict_before = ds.evaluate_retrieval(k=topk, loss_type=emb_test_loss_type, include_adv=False)
    print(f"Before attack: {metric_dict_before}")
    metric_dict_after = ds.evaluate_retrieval(k=topk, loss_type=emb_test_loss_type, include_adv=True)
    print(f"After attack: {metric_dict_after}")

if do_generation:
    print("=== Evaluating generation ...")
    asr_test, gs_test = ds.evaluate_generation(vlm, image_adv, target_answer, eval_train=False)
    print(f"Test ASR: {asr_test:.2f}")
    asr_train, gs_train = ds.evaluate_generation(vlm, image_adv, target_answer, eval_train=True)
    print(f"Train ASR: {asr_train:.2f}")