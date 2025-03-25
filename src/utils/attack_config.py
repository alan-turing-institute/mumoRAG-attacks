from dataclasses import dataclass
import hashlib

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


    def create_filename(self,):
        # this should include more info, but this suffices for now
        config_str = f"{self.model_name_emb}{self.model_name_vlm}{self.ds_name}{self.chosen_index}{self.max_perturbation}{self.lambda_emb}{self.lambda_vlm}"
        hash_str = hashlib.md5(config_str.encode())
        return f"adv_img_{hash_str.hexdigest()}.pt"