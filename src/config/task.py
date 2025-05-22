import hashlib
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Iterator

from experiments.configstore import get_config_name
from utils.defence import DefenceName
from utils.logger import logger
from utils.scheduler import LearningRateConfig
from wrappers.attack_mask import AttackMask
from wrappers.dataset import DatasetName
from wrappers.embedding import COLPALI_MODELS, EmbedderName, EmbeddingLoss, get_loss_with_default, is_loss_compatible
from wrappers.judge import JudgeMetric
from wrappers.vlm import VLMName

from .experiment import ExperimentConfig


@dataclass
class TaskVLMConfig:
    lambda_: float
    models: list[VLMName]
    target_answers: list[str]
    gen_topk: int


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
    vlm: TaskVLMConfig | None
    chosen_index: int
    max_perturbation: float
    n_gradient_steps: int
    max_batch_size_per_iter: int
    gradient_acc_steps: int
    lambda_emb: float
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
    kb_compromised_fraction: float = 0.1
    # eval
    eval_emb_name: EmbedderName | None = None
    eval_vlm_name: VLMName | None = None
    eval_jdg_name: VLMName | None = None
    defence: DefenceName = DefenceName.NONE

    def create_hash_string(self):
        target_str = ",".join([str(query_id) for query_id in self.target_query_idx]) + f"{self.n_knn_target_queries}" + f"{self.optimize_nontargeted_queries}" if self.is_targeted else ""
        target_answer_str = ",".join(self.vlm.target_answers) if self.vlm else ""

        model_name_embs = self.model_name_embs[0] if len(self.model_name_embs) == 1 else self.model_name_embs

        if self.vlm:
            model_name_vlms = self.vlm.models[0] if len(self.vlm.models) == 1 else self.vlm.models
            gen_topk = self.vlm.gen_topk
            vlm_float = float(self.vlm.lambda_)
        else:
            gen_topk = ""
            model_name_vlms = ""
            vlm_float = ""

        config_str = f"{model_name_embs}{model_name_vlms}{self.ds_name}{self.chosen_index}{target_answer_str}{self.max_perturbation}{self.emb_train_loss_type}{self.is_adaptive}{gen_topk}{self.kb_compromised_fraction}{target_str}{self.attack_mask}"

        if self.is_adaptive:
            config_str += f"{float(self.lambda_constant)}"
        else:
            config_str += f"{float(self.lambda_emb)}{vlm_float}"

        if any([model_name in COLPALI_MODELS for model_name in self.model_name_embs]) and self.colpali_only_images:
            config_str += f"{self.colpali_only_images}"

        if self.judge:
            jdg_metric_str = "".join(self.judge.metrics)
            config_str += f"{self.judge.model_name}{self.judge.lambda_}{self.judge.target_answer}{jdg_metric_str}"

        # if self.defence != DefenceName.NONE:
        #     defence_str = "".join(self.defence)
        #     config_str += f"{defence_str}"

        hash_str = hashlib.md5(config_str.encode()).hexdigest()
        return hash_str

    def create_filename(self):
        hash_str = self.create_hash_string()
        return f"adv_img_{get_config_name()}_{hash_str}.pt"

    def to_dict(self):
        return asdict(self)

    def get_result_filename(self, results_folder: Path) -> Path:
        hash_string = self.create_hash_string()
        transferability_suffix = get_transferability_file_suffix(self.eval_emb_name, self.eval_vlm_name, self.eval_jdg_name)
        return results_folder / f"metrics_{get_config_name()}_{hash_string}{transferability_suffix}.json"


# standalone functions
def get_transferability_file_suffix(
    eval_emb_name: EmbedderName | None,
    eval_vlm_name: VLMName | None,
    eval_jdg_name: VLMName | None = None,
) -> str:
    if not eval_emb_name and not eval_vlm_name and not eval_jdg_name:
        return ""
    transfer_str = f"{eval_emb_name or ''}{eval_vlm_name or ''}{eval_jdg_name or ''}"
    return f"_{hashlib.md5(transfer_str.encode()).hexdigest()}"


def get_defence_file_suffix(defence: DefenceName) -> str:
    if defence == DefenceName.NONE:
        return ""
    return f"_{defence.value}"


def generate_task_configs(exp_config: ExperimentConfig, include_eval: bool = False) -> list[TaskConfig]:
    if exp_config.train.is_targeted and exp_config.train.vlm is None:
        raise ValueError("Cannot run targeted experiment without specifying VLMs.")

    if include_eval and exp_config.train.vlm is None and exp_config.eval.do_generation and exp_config.eval.eval_vlm_list is None:
        raise ValueError("Cannot evaluate without specifying VLMs.")

    parameter_collection: Iterator[
        tuple[
            DatasetName,
            EmbedderName | list[EmbedderName],
            VLMName | list[VLMName],
            int,
            VLMName,
            float,
            EmbeddingLoss,
            bool,
            int,
            AttackMask,
            bool,
            EmbedderName | None,
            VLMName | None,
            VLMName | None,
            DefenceName,
        ]
    ] = product(  # type: ignore
        exp_config.train.dataset_list,
        exp_config.train.embedder_list,
        exp_config.train.vlm.models if exp_config.train.vlm else [None],
        exp_config.train.vlm.gen_topk_list if exp_config.train.vlm else [None],
        exp_config.train.judge.models if exp_config.train.judge else [None],
        exp_config.train.max_perturbation_list,
        exp_config.train.emb_train_loss_type_list,
        exp_config.train.is_adaptive_list,
        exp_config.train.chosen_index_list,
        exp_config.train.attack_mask_list,
        exp_config.train.optimize_nontargeted_queries_list,
        (exp_config.eval.eval_emb_list if include_eval else None) or [None],
        (exp_config.eval.eval_vlm_list if include_eval else None) or [None],
        (exp_config.eval.eval_jdg_list if include_eval else None) or [None],
        (exp_config.eval.defences_list if include_eval else None) or [DefenceName.NONE],
    )
    attack_configs = []
    for params in parameter_collection:
        (
            ds_name,
            model_name_embs,
            model_name_vlms,
            gen_topk,
            model_name_jdg,
            max_perturbation,
            emb_train_loss_type,
            is_adaptive,
            chosen_index,
            attack_mask,
            optimize_nontargeted_queries,
            eval_emb_name,
            eval_vlm_name,
            eval_jdg_name,
            defence,
        ) = params

        if exp_config.train.vlm and len(exp_config.train.vlm.target_answers) > 1 and not exp_config.train.is_targeted:
            raise ValueError("Multiple target answers supported only for targeted attacks.")

        if isinstance(model_name_embs, list):
            if include_eval and not eval_emb_name:
                raise ValueError("If trained with multiple embedders, evaluation config must specify a single embedder to test transferability")
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

        if isinstance(model_name_vlms, list):
            if include_eval and not eval_vlm_name:
                raise ValueError("If trained with multiple VLMs, evaluation config must specify a single VLM to test transferability")
        else:
            model_name_vlms = [model_name_vlms]

        if exp_config.train.vlm and len(exp_config.train.vlm.target_answers) not in [1, len(exp_config.train.target_query_idx)]:
            raise ValueError(f"VLM target answers array has incompatible length ({len(exp_config.train.vlm.target_answers)}) with target queries ({len(exp_config.train.target_query_idx)})")
        vlm_config = None
        if exp_config.train.vlm:
            vlm_config = TaskVLMConfig(
                models=model_name_vlms,
                target_answers=exp_config.train.vlm.target_answers,
                lambda_=exp_config.train.vlm.lambda_,
                gen_topk=gen_topk,
            )

        judge_config = None
        if exp_config.train.judge:
            judge_config = TaskJudgeConfig(
                model_name=model_name_jdg,
                lambda_=exp_config.train.judge.lambda_,
                target_answer=exp_config.train.judge.target_answer,
                metrics=exp_config.train.judge.metrics,
            )

        attack_configs.append(
            TaskConfig(
                ds_name=ds_name,
                model_name_embs=model_name_embs,
                vlm=vlm_config,
                chosen_index=chosen_index,
                max_perturbation=max_perturbation,
                n_gradient_steps=exp_config.train.n_gradient_steps,
                lr=exp_config.train.lr,
                max_batch_size_per_iter=exp_config.train.max_batch_size_per_iter,
                gradient_acc_steps=exp_config.train.gradient_acc_steps,
                lambda_emb=exp_config.train.lambda_emb,
                emb_train_loss_type=emb_train_loss_type,
                is_adaptive=is_adaptive,
                lambda_constant=exp_config.train.lambda_constant,
                eval_emb_name=eval_emb_name,
                eval_vlm_name=eval_vlm_name,
                eval_jdg_name=eval_jdg_name,
                colpali_only_images=exp_config.train.colpali_only_images,
                kb_compromised_fraction=exp_config.train.kb_compromised_fraction,
                judge=judge_config,
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
