from dataclasses import dataclass, asdict
import hashlib
from .embedding import COLPALI_MODELS

@dataclass
class AttackConfig:
    """
    Class that includes information about the attack. Useful when saving and loading adversarial images
    """
    ds_name: str
    model_name_emb: str
    model_name_vlm: str
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
    emb_train_loss_type: str
    is_adaptive: bool
    lambda_constant: float 
    colpali_only_images: bool


    def create_hash_string(self,):
        # this should include more info, but this suffices for now
        config_str = f"{self.model_name_emb}{self.model_name_vlm}{self.ds_name}{self.chosen_index}{self.target_answer}{self.max_perturbation}{self.emb_train_loss_type}{self.is_adaptive}"
        
        if self.is_adaptive: config_str += f"{self.lambda_constant}"
        else: config_str += f"{self.lambda_emb}{self.lambda_vlm}"

        if self.model_name_emb in COLPALI_MODELS and self.colpali_only_images: 
            config_str += f"{self.colpali_only_images}"
        
        hash_str = hashlib.md5(config_str.encode()).hexdigest()
        return hash_str

    def create_filename(self,):
        hash_str = self.create_hash_string()
        return f"adv_img_{hash_str}.pt"
    
    def to_dict(self,):
        return asdict(self)

# standalone function
def get_transferability_file_suffix(eval_emb_name, eval_vlm_name):
    if eval_emb_name == "" and eval_vlm_name == "": return ""
    
    transfer_str = f"{eval_emb_name}{eval_vlm_name}"
    return f"_{hashlib.md5(transfer_str.encode()).hexdigest()}"