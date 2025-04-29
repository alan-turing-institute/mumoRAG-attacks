import hashlib
from dataclasses import dataclass, asdict
from itertools import product
from typing import Optional

from wrappers.dataset import DatasetName
from wrappers.embedding import EmbedderName, EmbeddingLoss, COLPALI_MODELS
from wrappers.judge import JudgeMetric
from wrappers.vlm import VLMName

from .experiment import ExperimentConfig


@dataclass
class TaskConfig:
    """
    Class that includes information about the attack.
    A specific combination of values from ExperimentTrainConfig and ExperimentEvalConfig.
    Useful when saving and loading adversarial images.
    """

    ds_name: DatasetName
    model_name_emb: EmbedderName
    model_name_vlm: VLMName
    target_answer_vlm: list[str]
    chosen_index: int
    max_perturbation: float
    n_gradient_steps: int
    lr_start: float
    lr_end: float
    max_batch_size_per_iter: int
    gradient_acc_steps: int
    lambda_emb: float
    lambda_vlm: float
    emb_train_loss_type: EmbeddingLoss
    is_adaptive: bool
    lambda_constant: float
    model_name_jdg: VLMName
    lambda_jdg: float
    target_answer_jdg: str
    train_jdg_metric_list: list[JudgeMetric]
    target_query_idx: list[int]  # we assume the target queries are always from the training dataset
    eval_emb_name: Optional[EmbedderName] = None
    eval_vlm_name: Optional[VLMName] = None
    eval_jdg_name: Optional[VLMName] = None
    colpali_only_images: bool = False
    gen_topk: int = 1
    kb_compromised_fraction: float = 0.1

    def create_hash_string(self):
        target_str = ",".join([str(i) for i in self.target_query_idx])
        target_answer_str = ",".join(self.target_answer_vlm)

        config_str = f"{self.model_name_emb}{self.model_name_vlm}{self.ds_name}{self.chosen_index}{target_answer_str}{self.max_perturbation}{self.emb_train_loss_type}{self.is_adaptive}{self.gen_topk}{self.kb_compromised_fraction}{target_str}"

        if self.is_adaptive:
            config_str += f"{float(self.lambda_constant)}"
        else:
            config_str += f"{float(self.lambda_emb)}{float(self.lambda_vlm)}"

        if self.model_name_emb in COLPALI_MODELS and self.colpali_only_images:
            config_str += f"{self.colpali_only_images}"

        if self.lambda_jdg > 0:
            jdg_metric_str = "".join(self.train_jdg_metric_list)
            config_str += f"{self.model_name_jdg}{self.lambda_jdg}{self.target_answer_jdg}{jdg_metric_str}"

        hash_str = hashlib.md5(config_str.encode()).hexdigest()
        return hash_str

    def create_filename(self):
        hash_str = self.create_hash_string()
        return f"adv_img_{hash_str}.pt"

    def to_dict(self):
        return asdict(self)


# standalone function
def get_transferability_file_suffix(eval_emb_name: EmbedderName, eval_vlm_name: VLMName, eval_jdg_name: VLMName = ""):
    if eval_emb_name == "" and eval_vlm_name == "" and eval_jdg_name == "": return ""

    transfer_str = f"{eval_emb_name}{eval_vlm_name}{eval_jdg_name}"
    return f"_{hashlib.md5(transfer_str.encode()).hexdigest()}"


def generate_task_configs(exp_config: ExperimentConfig, include_eval: bool = False) -> list[TaskConfig]:
    parameter_collection = product(
        exp_config.train.dataset_list,
        exp_config.train.embedder_list,
        exp_config.train.vlm_list,
        exp_config.train.judge_list,
        exp_config.train.max_perturbation_list,
        exp_config.train.emb_train_loss_type_list,
        exp_config.train.is_adaptive_list,
        exp_config.train.chosen_index_list,
        exp_config.train.gen_topk_list,
        (exp_config.eval.eval_emb_list if include_eval else None) or [""],
        (exp_config.eval.eval_vlm_list if include_eval else None) or [""],
        (exp_config.eval.eval_jdg_list if include_eval else None) or [""],
    )
    attack_configs = []
    for params in parameter_collection:
        (
            ds_name,
            model_name_emb,
            model_name_vlm,
            model_name_jdg,
            max_perturbation,
            emb_train_loss_type,
            is_adaptive,
            chosen_index,
            gen_topk,
            eval_emb_name,
            eval_vlm_name,
            eval_jdg_name,
        ) = params

        correct_vlm_target = len(exp_config.train.target_answer_vlm) == 1 or len(exp_config.train.target_answer_vlm) == len(exp_config.train.target_query_idx)
        assert correct_vlm_target, f"VLM target answers array has incompatible length ({len(exp_config.train.target_answer_vlm)}) with target queries ({len(exp_config.train.target_query_idx)})"

        attack_configs.append(
            TaskConfig(
                ds_name=ds_name,
                model_name_emb=model_name_emb,
                model_name_vlm=model_name_vlm,
                target_answer_vlm=exp_config.train.target_answer_vlm,
                chosen_index=chosen_index,
                max_perturbation=max_perturbation,
                n_gradient_steps=exp_config.train.n_gradient_steps,
                lr_start=exp_config.train.lr_start,
                lr_end=exp_config.train.lr_end,
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
                model_name_jdg=model_name_jdg,
                lambda_jdg=exp_config.train.lambda_jdg,
                target_answer_jdg=exp_config.train.target_answer_jdg,
                train_jdg_metric_list=exp_config.train.train_jdg_metric_list,
                target_query_idx=exp_config.train.target_query_idx,
            )
        )

    return attack_configs
