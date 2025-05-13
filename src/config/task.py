import hashlib
from dataclasses import dataclass, asdict
from typing import Literal
from itertools import product

from experiments.configstore import get_config_name
from utils.logger import logger
from utils.defence import DefenceName
from utils.scheduler import LearningRateConfig
from wrappers.attack_mask import AttackMask
from wrappers.dataset import DatasetName
from wrappers.embedding import EmbedderName, EmbeddingLoss, COLPALI_MODELS, is_loss_compatible, get_loss_with_default
from wrappers.judge import JudgeMetric
from wrappers.vlm import VLMName
from .experiment import ExperimentConfig


@dataclass
class TaskJudgeConfig:
    lambda_: float
    model_name: VLMName
    target_answer: str
    metrics: list[JudgeMetric]


@dataclass
class TaskConfig:
    """
    Class that includes information about the attack.
    A specific combination of values from ExperimentTrainConfig and ExperimentEvalConfig.
    Useful when saving and loading adversarial images.
    """

    ds_name: DatasetName
    model_name_embs: list[EmbedderName]
    model_name_vlm: VLMName
    target_answer_vlm: list[str]
    chosen_index: int
    max_perturbation: float
    n_gradient_steps: int
    max_batch_size_per_iter: int
    gradient_acc_steps: int
    lambda_emb: float
    lambda_vlm: float
    emb_train_loss_type: EmbeddingLoss
    is_adaptive: bool
    lambda_constant: float
    attack_mask: AttackMask
    image_size: list[int]
    target_query_idx: list[int]  # we assume the target queries are always from the training dataset
    is_targeted: bool
    n_knn_target_queries: int
    optimize_nontargeted_queries: bool
    judge: TaskJudgeConfig | None
    lr: LearningRateConfig
    colpali_only_images: bool = False
    gen_topk: int = 1
    kb_compromised_fraction: float = 0.1
    # eval
    eval_emb_name: EmbedderName | None = None
    eval_vlm_name: VLMName | None = None
    eval_jdg_name: VLMName | None = None
    defence: DefenceName = DefenceName.NONE

    def create_hash_string(self):
        target_str = (
            ",".join([str(query_id) for query_id in self.target_query_idx])
            + f"{self.n_knn_target_queries}"
            + f"{self.optimize_nontargeted_queries}"
            if self.is_targeted
            else ""
        )
        target_answer_str = ",".join(self.target_answer_vlm)

        model_name_embs = self.model_name_embs[0] if len(self.model_name_embs) == 1 else self.model_name_embs

        config_str = f"{model_name_embs}{self.model_name_vlm}{self.ds_name}{self.chosen_index}{target_answer_str}{self.max_perturbation}{self.emb_train_loss_type}{self.is_adaptive}{self.gen_topk}{self.kb_compromised_fraction}{target_str}{self.attack_mask}"

        if self.is_adaptive:
            config_str += f"{float(self.lambda_constant)}"
        else:
            config_str += f"{float(self.lambda_emb)}{float(self.lambda_vlm)}"

        if any([model_name in COLPALI_MODELS for model_name in self.model_name_embs]) and self.colpali_only_images:
            config_str += f"{self.colpali_only_images}"

        if self.judge:
            jdg_metric_str = "".join(self.judge.metrics)
            config_str += f"{self.judge.model_name}{self.judge.lambda_}{self.judge.target_answer}{jdg_metric_str}"

        hash_str = hashlib.md5(config_str.encode()).hexdigest()
        return hash_str

    def create_filename(self):
        hash_str = self.create_hash_string()
        return f"adv_img_{get_config_name()}_{hash_str}.pt"

    def to_dict(self):
        return asdict(self)


# standalone functions
def get_transferability_file_suffix(
        eval_emb_name: EmbedderName | Literal[""],
        eval_vlm_name: VLMName | Literal[""],
        eval_jdg_name: VLMName | Literal[""] = "",
) -> str:
    if eval_emb_name == "" and eval_vlm_name == "" and eval_jdg_name == "":
        return ""
    transfer_str = f"{eval_emb_name}{eval_vlm_name}{eval_jdg_name}"
    return f"_{hashlib.md5(transfer_str.encode()).hexdigest()}"

def get_defence_file_suffix(defence: DefenceName) -> str:
    if defence == DefenceName.NONE:
        return ""
    return f"_{defence.value}"

def generate_task_configs(exp_config: ExperimentConfig, include_eval: bool = False) -> list[TaskConfig]:
    parameter_collection = product(
        exp_config.train.dataset_list,
        exp_config.train.embedder_list,
        exp_config.train.vlm_list,
        exp_config.train.judge.models if exp_config.train.judge else [""],
        exp_config.train.max_perturbation_list,
        exp_config.train.emb_train_loss_type_list,
        exp_config.train.is_adaptive_list,
        exp_config.train.chosen_index_list,
        exp_config.train.gen_topk_list,
        exp_config.train.attack_mask_list,
        exp_config.train.optimize_nontargeted_queries_list,
        (exp_config.eval.eval_emb_list if include_eval else None) or [""],
        (exp_config.eval.eval_vlm_list if include_eval else None) or [""],
        (exp_config.eval.eval_jdg_list if include_eval else None) or [""],
        (exp_config.eval.defences_list if include_eval else None) or [DefenceName.NONE],
    )
    attack_configs = []
    for params in parameter_collection:
        (
            ds_name,
            model_name_embs,
            model_name_vlm,
            model_name_jdg,
            max_perturbation,
            emb_train_loss_type,
            is_adaptive,
            chosen_index,
            gen_topk,
            attack_mask,
            optimize_nontargeted_queries,
            eval_emb_name,
            eval_vlm_name,
            eval_jdg_name,
            defence,
        ) = params
        if len(exp_config.train.target_answer_vlm) > 1 and not exp_config.train.is_targeted:
            raise ValueError("Multiple target answers supported only for targeted attacks.")

        if isinstance(model_name_embs, list):
            if exp_config.train.is_targeted and exp_config.train.n_knn_target_queries > 1:
                raise ValueError("Multi-embedder tasks do not k-nearest neighbour targeted attacks.")
            if emb_train_loss_type != EmbeddingLoss.DEFAULT:
                raise ValueError("Multi-embedder tasks do not support choosing non-default loss type.")
        else:
            emb_train_loss_type = get_loss_with_default(model_name_embs, emb_train_loss_type)
            if not is_loss_compatible(model_name_embs, emb_train_loss_type):
                logger.warning(f"Loss {emb_train_loss_type} not compatible with {model_name_embs}; skipping task config")
                continue
            model_name_embs = [model_name_embs]

        if len(exp_config.train.target_answer_vlm) not in [1, len(exp_config.train.target_query_idx)]:
            raise ValueError(
                f"VLM target answers array has incompatible length ({len(exp_config.train.target_answer_vlm)}) with target queries ({len(exp_config.train.target_query_idx)})"
            )

        attack_configs.append(
            TaskConfig(
                ds_name=ds_name,
                model_name_embs=model_name_embs,
                model_name_vlm=model_name_vlm,
                target_answer_vlm=exp_config.train.target_answer_vlm,
                chosen_index=chosen_index,
                max_perturbation=max_perturbation,
                n_gradient_steps=exp_config.train.n_gradient_steps,
                lr=exp_config.train.lr,
                max_batch_size_per_iter=exp_config.train.max_batch_size_per_iter,
                gradient_acc_steps=exp_config.train.gradient_acc_steps,
                lambda_emb=exp_config.train.lambda_emb,
                lambda_vlm=exp_config.train.lambda_vlm,
                emb_train_loss_type=emb_train_loss_type,
                is_adaptive=is_adaptive,
                lambda_constant=exp_config.train.lambda_constant,
                eval_emb_name=eval_emb_name,
                eval_vlm_name=eval_vlm_name,
                eval_jdg_name=eval_jdg_name,
                colpali_only_images=exp_config.train.colpali_only_images,
                gen_topk=gen_topk,
                kb_compromised_fraction=exp_config.train.kb_compromised_fraction,
                judge=TaskJudgeConfig(
                    model_name=model_name_jdg,
                    lambda_=exp_config.train.judge.lambda_,
                    target_answer=exp_config.train.judge.target_answer,
                    metrics=exp_config.train.judge.metrics,
                ) if exp_config.train.judge else None,
                attack_mask=attack_mask,
                image_size=exp_config.train.image_size,
                target_query_idx=exp_config.train.target_query_idx,
                is_targeted=exp_config.train.is_targeted,
                n_knn_target_queries=exp_config.train.n_knn_target_queries,
                optimize_nontargeted_queries=optimize_nontargeted_queries,
                defence=defence,
            )
        )

    return attack_configs
