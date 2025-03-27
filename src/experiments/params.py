"""
This file includes different configurations for the parameters of the attack and the multimodal RAG system
"""

from utils.dataset import DatasetName
from utils.embedding import EmbedderName
from utils.vlm import VLMName
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentTrainConfig:
    dataset_list: list[str]
    embedder_list: list[str]
    vlm_list: list[str]
    max_perturbation_list: list[float]
    emb_train_loss_type_list: list[str]
    is_adaptive_list: list[bool]
    save_folder: str
    target_answer: str
    chosen_index: int
    n_gradient_steps: int
    print_every: int
    lr_start: float
    lr_end: float
    max_batch_size_per_iter: int
    gradient_acc_steps: int
    lambda_emb: float
    lambda_vlm: float
    lambda_constant: float

@dataclass
class ExperimentEvalConfig:
    emb_test_loss_type_list: list[str]
    topk_list: list[int]
    do_retrieval: bool
    do_generation: bool
    # following used to test transferability, [""] means white-box setting
    eval_emb_list: list[str] 
    eval_vlm_list: list[str]

# configuration 0: just for testing
exp_config_train = ExperimentTrainConfig(
    dataset_list=[DatasetName.VIDORE_SYN_AI],
    embedder_list=[EmbedderName.CLIP_BASE_PATCH16, EmbedderName.SIGLIP2_BASE_PATCH16],
    vlm_list=[VLMName.SMOLVLM_1_256M],
    max_perturbation_list=[x/255.0 for x in [8]], # [4, 8, 16, 32]
    emb_train_loss_type_list=["mse"], 
    is_adaptive_list=[True], 
    save_folder="../data/attacks/",
    target_answer="I will not reply to you!",
    chosen_index=150,
    n_gradient_steps=50,
    print_every=10,
    lr_start=255*(3e-3), 
    lr_end=255*(3e-4),
    max_batch_size_per_iter=2,
    gradient_acc_steps=4,
    lambda_emb=2,
    lambda_vlm=1,
    lambda_constant=0.2,
)
exp_config_eval = ExperimentEvalConfig(
    emb_test_loss_type_list=["mse", "cos"],
    topk_list=[1,5],
    do_retrieval=True,
    do_generation=True,
    eval_emb_list=[""],
    eval_vlm_list=[""],
)


# TODO: configuration 1 (small stuff that can run on laptop)


# TODO: configuration 2 (to be run on NVIDIA GPU)

