from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.scheduler import LearningRateConfig
from wrappers.attack_mask import AttackMask
from wrappers.dataset import DatasetName
from wrappers.embedding import EmbedderName, EmbeddingLoss
from wrappers.judge import JudgeMetric
from wrappers.vlm import VLMName
from . import ATTACKS_FOLDER

@dataclass
class VLMConfig:
    lambda_: float = 1
    # models: list[VLMName|list[VLMName]] but not supported by Hydra/OmegaConf
    models: list[Any] = field(default_factory=lambda: [VLMName.SMOLVLM_1_2B])
    target_answers: list[str] = field(default_factory=lambda: ["I will not reply to you!"])
    gen_topk_list: list[int] = field(default_factory=lambda: [1])

@dataclass
class JudgeConfig:
    lambda_: float = 0
    models: list[VLMName] = field(default_factory=lambda: [VLMName.SMOLVLM_1_2B])
    target_answer: str = "YES"
    metrics: list[JudgeMetric] = field(
        default_factory=lambda: [JudgeMetric.IMAGE_CONTEXT_RELEVANCY, JudgeMetric.IMAGE_FAITHFULNESS]
    )

@dataclass
class ExperimentTrainConfig:
    dataset_list: list[DatasetName] = field(default_factory=lambda: [DatasetName.VIDORE_SYN_AI])
    # embedder_list: list[EmbedderName|list[EmbedderName]] but not supported by Hydra/OmegaConf
    embedder_list: list[Any] = field(default_factory=lambda: [EmbedderName.CLIP_LARGE_PATCH14])
    max_perturbation_list: list[float] = field(default_factory=lambda: [x / 255.0 for x in [8]])
    emb_train_loss_type_list: list[EmbeddingLoss] = field(default_factory=lambda: [EmbeddingLoss.DEFAULT])

    vlm: VLMConfig | None = field(default_factory=VLMConfig)

    save_folder: Path = ATTACKS_FOLDER
    chosen_index_list: list[int] = field(default_factory=lambda: [150])
    is_adaptive_list: list[bool] = field(default_factory=lambda: [False])
    n_gradient_steps: int = 500
    max_batch_size_per_iter: int = 2
    lr: LearningRateConfig = field(default_factory=LearningRateConfig)
    gradient_acc_steps: int = 4
    print_every: int = 5
    lambda_emb: float = 2
    lambda_constant: float = 0.2
    colpali_only_images: bool = False
    kb_compromised_fraction: float = 0.1
    judge: JudgeConfig | None = None
    attack_mask_list: list[AttackMask] = field(default_factory=lambda: [AttackMask.Full])
    image_size: list[int] = field(default_factory=lambda: [512, 512])
    target_query_idx: list[int] = field(default_factory=lambda: [])
    is_targeted: bool = False
    n_knn_target_queries: int = 1  # itself
    optimize_nontargeted_queries_list: list[bool] = field(default_factory=lambda: [True])
