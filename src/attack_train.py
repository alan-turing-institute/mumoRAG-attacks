from pprint import pformat

import hydra
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf

from config.experiment import ExperimentConfig
from config.task import generate_task_configs
from experiments import DEFAULT_EXPERIMENT
from utils.attack import rag_attack
from utils.logger import logger
from utils.utils import get_device
from wrappers.cache import get_vlm, get_dataset, get_embedder, get_judge
from wrappers.embedding import is_loss_compatible


def run(exp_config: ExperimentConfig):
    device = get_device(prefer_mps=True)
    task_configs = generate_task_configs(exp_config)
    n_evals = len(task_configs)

    for i, task_config in enumerate(task_configs):
        if not is_loss_compatible(task_config.model_name_emb, task_config.emb_train_loss_type): continue
        logger.info(f"{'+' * 20}\nTrain Attack {(i + 1):4d}/{n_evals}, task_config -> {pformat(task_config.to_dict(), indent=4)}")

        vlm = get_vlm(task_config.model_name_vlm, device)
        ds = get_dataset(task_config.ds_name)
        embedder = get_embedder(
            task_config.model_name_emb,
            quantize=False,
            colpali_only_images=exp_config.train.colpali_only_images,
            device=device,
        )
        jdg = get_judge(task_config.model_name_jdg, device) if task_config.lambda_jdg > 0 else None

        query_strings = ds.queries_train
        attack_images = ds.sample_images_from_ds(fraction=task_config.kb_compromised_fraction)  # images included by the attacker in the VLM context (n-1 because the malicious image must be included)

        # choose attacked image
        chosen_image = ds.images[task_config.chosen_index]
        chosen_image = chosen_image.resize((512,512))  # this can save memory (also setting this to VLM image size with resample=0 -> reduce errors)
        chosen_image = T.PILToTensor()(chosen_image)  # choose from after 100 since those do not have associated queries
        chosen_image = chosen_image.float()
        initial_chosen_image = chosen_image.clone()

        # train the attack
        image_adv = rag_attack(
            raw_image=chosen_image,
            embedder=embedder,
            vlm=vlm,
            jdg=jdg,
            user_query=query_strings,
            config=task_config,
            attack_images=attack_images,
            print_every=exp_config.train.print_every,
            device=device,
        )

        logger.info(f"MSE: {torch.nn.functional.mse_loss(image_adv, initial_chosen_image)}")
        logger.info(f"Linf: {(initial_chosen_image - image_adv).norm(p=float('inf'))}")

        # save adv image
        attack_dict = task_config.to_dict()
        attack_dict["image_adv"] = image_adv.type(torch.uint8)
        filename = exp_config.train.save_folder / task_config.create_filename()
        torch.save(attack_dict, filename)
        logger.info(f"Saved adversarial image to {filename}.")


@hydra.main(version_base=None, config_path="pkg://experiments", config_name=DEFAULT_EXPERIMENT)
def main(exp_config: ExperimentConfig):
    run(OmegaConf.to_object(exp_config))


if __name__ == "__main__":
    main()
