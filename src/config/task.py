from dataclasses import dataclass, asdict
import hashlib
from itertools import product
from typing import Optional

from utils.dataset import DatasetName
from utils.embedding import EmbedderName, EmbeddingLoss, COLPALI_MODELS
from utils.vlm import VLMName
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
    target_answer: str
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
    eval_emb_name: Optional[EmbedderName]
    eval_vlm_name: Optional[VLMName]

    def create_hash_string(self):
        # this should include more info, but this suffices for now
        config_str = f"{self.model_name_emb}{self.model_name_vlm}{self.ds_name}{self.chosen_index}{self.target_answer}{self.max_perturbation}{self.emb_train_loss_type}{self.is_adaptive}"

        if self.is_adaptive:
            config_str += f"{self.lambda_constant}"
        else:
            # fixme: this is a hack to make old data work
            # the default values were ints, despite being typed as float; new config correctly loads it as float but generates different hash
            # config_str += f"{float(self.lambda_emb)}{float(self.lambda_vlm)}"
            config_str += f"{int(self.lambda_emb) if float(self.lambda_emb).is_integer() else self.lambda_emb}{int(self.lambda_vlm) if float(self.lambda_vlm).is_integer() else self.lambda_vlm}"

        hash_str = hashlib.md5(config_str.encode()).hexdigest()
        return hash_str

    def create_filename(self):
        hash_str = self.create_hash_string()
        return f"adv_img_{hash_str}.pt"

    def to_dict(self):
        return asdict(self)


# standalone function
def get_transferability_file_suffix(eval_emb_name: EmbedderName, eval_vlm_name: VLMName):
    if eval_emb_name == "" and eval_vlm_name == "": return ""

    transfer_str = f"{eval_emb_name}{eval_vlm_name}"
    return f"_{hashlib.md5(transfer_str.encode()).hexdigest()}"


def generate_task_configs(exp_config: ExperimentConfig, include_eval: bool = False) -> list[TaskConfig]:
    parameter_collection = product(
        exp_config.train.dataset_list,
        exp_config.train.embedder_list,
        exp_config.train.vlm_list,
        exp_config.train.max_perturbation_list,
        exp_config.train.emb_train_loss_type_list,
        exp_config.train.is_adaptive_list,
        exp_config.train.chosen_index_list,
        (exp_config.eval.eval_emb_list if include_eval else None) or [""],
        (exp_config.eval.eval_vlm_list if include_eval else None) or [""],
    )
    attack_configs = []
    for params in parameter_collection:
        (
            ds_name,
            model_name_emb,
            model_name_vlm,
            max_perturbation,
            emb_train_loss_type,
            is_adaptive,
            chosen_index,
            eval_emb_name,
            eval_vlm_name,
        ) = params

        attack_configs.append(
            TaskConfig(
                ds_name=ds_name,
                model_name_emb=model_name_emb,
                model_name_vlm=model_name_vlm,
                target_answer=exp_config.train.target_answer,
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
            )
        )

    return attack_configs
