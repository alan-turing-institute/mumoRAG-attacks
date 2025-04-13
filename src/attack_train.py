import hydra
import torch
import torchvision.transforms as T

from config.task import generate_task_configs
from config.experiment import ExperimentConfig
from experiments import DEFAULT_EXPERIMENT
from utils.attack import rag_attack
from utils.cache import load_vlm, load_embedder_and_dataset
from utils.embedding import is_loss_compatible
from utils.utils import get_device
from utils.logger import logger


@hydra.main(version_base=None, config_path="pkg://experiments", config_name=DEFAULT_EXPERIMENT)
def run(exp_config: ExperimentConfig):
    device = get_device(prefer_mps=True)
    task_configs = generate_task_configs(exp_config)
    n_evals = len(task_configs)

    for i, task_config in enumerate(task_configs):
        if not is_loss_compatible(task_config.model_name_emb, task_config.emb_train_loss_type): continue
        logger.info(f"{'+' * 20}\nTrain Attack {(i + 1):4d}/{n_evals}, task_config -> {task_config.to_dict()}")

        vlm = load_vlm(task_config.model_name_vlm, device)
        embedder, ds = load_embedder_and_dataset(
            task_config.ds_name,
            task_config.model_name_emb,
            do_retrieval=False,
            quantize=False,
            colpali_only_images=exp_config.train.colpali_only_images,
            device=device,
        )

        query_strings = ds.queries_train
        attack_images = ds.sample_attack_images(n_images=task_config.gen_topk) # images included by the attacker in the VLM context

        # choose attacked image
        chosen_image = ds.images[task_config.chosen_index]
        chosen_image = chosen_image.resize((512,512)) # this can save memory (also setting this to VLM image size with resample=0 -> reduce errors)
        chosen_image = T.PILToTensor()(chosen_image) # choose from after 100 since those do not have associated queries
        chosen_image = chosen_image.float()
        initial_chosen_image = chosen_image.clone()

        # train the attack
        image_adv = rag_attack(
            raw_image=chosen_image,
            embedder=embedder,
            vlm=vlm,
            user_query=query_strings,
            config=task_config,
            attack_images=attack_images,
            print_every=exp_config.train.print_every,
            device=device
        )

        logger.info(f"MSE: {torch.nn.functional.mse_loss(image_adv, initial_chosen_image)}")
        logger.info(f"Linf: {(initial_chosen_image - image_adv).norm(p=float('inf'))}")

        # save adv image
        attack_dict = task_config.to_dict()
        attack_dict["image_adv"] = image_adv.type(torch.uint8)
        filename = exp_config.train.save_folder / task_config.create_filename()
        torch.save(attack_dict, filename)
        logger.info(f"Saved adversarial image to {filename}.")

if __name__ == "__main__":
    run()