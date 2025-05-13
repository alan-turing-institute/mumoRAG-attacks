from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wrappers.attack_mask import AttackMask
from wrappers.dataset import DatasetName
from wrappers.embedding import EmbedderName, EmbeddingLoss
from wrappers.judge import JudgeMetric
from wrappers.vlm import VLMName
from . import ATTACKS_FOLDER


@dataclass
class ExperimentTrainConfig:
    dataset_list: list[DatasetName] = field(default_factory=lambda: [DatasetName.VIDORE_SYN_AI])
    # embedder_list: list[EmbedderName|list[EmbedderName]] but not supported by Hydra/OmegaConf
    embedder_list: list[Any] = field(default_factory=lambda: [EmbedderName.CLIP_LARGE_PATCH14])
    vlm_list: list[VLMName] = field(default_factory=lambda: [VLMName.SMOLVLM_1_2B])
    max_perturbation_list: list[float] = field(default_factory=lambda: [x / 255.0 for x in [8]])
    emb_train_loss_type_list: list[EmbeddingLoss] = field(default_factory=lambda: [EmbeddingLoss.DEFAULT])
    is_adaptive_list: list[bool] = field(default_factory=lambda: [False])
    save_folder: Path = ATTACKS_FOLDER
    target_answer_vlm: list[str] = field(default_factory=lambda: ["I will not reply to you!"])
    chosen_index_list: list[int] = field(default_factory=lambda: [150])
    n_gradient_steps: int = 500
    print_every: int = 5
    lr_start: float = 255 * 3e-3
    lr_end: float = 255 * 3e-4
    max_batch_size_per_iter: int = 2
    gradient_acc_steps: int = 4
    lambda_emb: float = 2
    lambda_vlm: float = 1
    lambda_constant: float = 0.2
    colpali_only_images: bool = False
    gen_topk_list: list[int] = field(default_factory=lambda: [1])
    kb_compromised_fraction: float = 0.1
    lambda_jdg: float = 0
    judge_list: list[VLMName] = field(default_factory=lambda: [VLMName.SMOLVLM_1_2B])
    target_answer_jdg: str = "YES"
    train_jdg_metric_list: list[JudgeMetric] = field(
        default_factory=lambda: [JudgeMetric.IMAGE_CONTEXT_RELEVANCY, JudgeMetric.IMAGE_FAITHFULNESS]
    )
    attack_mask_list: list[AttackMask] = field(default_factory=lambda: [AttackMask.Full])
    image_size: list[int] = field(default_factory=lambda: [512, 512])
    target_query_idx: list[int] = field(default_factory=lambda: [])
    is_targeted: bool = False
    n_knn_target_queries: int = 1  # itself
    optimize_nontargeted_queries_list: list[bool] = field(default_factory=lambda: [True])
