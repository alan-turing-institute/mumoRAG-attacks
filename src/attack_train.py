import torchvision.transforms as T
from utils.utils import get_device
import torch
from utils.attack import rag_attack
from utils.embedding import EmbeddingModel, EmbedderName
from utils.vlm import VLM, VLMName
from utils.dataset import ViDoReDataset, DatasetName
from utils.attack_config import AttackConfig
from itertools import product
from experiments.params import exp_config_train, is_loss_comaptible
import gc


device = get_device(prefer_mps=True)
parameter_collection = product(
    exp_config_train.dataset_list, 
    exp_config_train.embedder_list, 
    exp_config_train.vlm_list, 
    exp_config_train.max_perturbation_list, 
    exp_config_train.emb_train_loss_type_list, 
    exp_config_train.is_adaptive_list,
    exp_config_train.chosen_index_list,
)

parameter_collection = [x for x in parameter_collection]
n_evals = len(parameter_collection)

for i, params in enumerate(parameter_collection):
    
    ds_name, model_name_emb, model_name_vlm, max_perturbation, emb_train_loss_type, is_adaptive, chosen_index = params

    if not is_loss_comaptible(model_name_emb, emb_train_loss_type): continue

    print("+"*20, f"\nTrain Attack {(i+1):4d}/{n_evals}, params -> {params}")

    attack_config = AttackConfig(
        ds_name=ds_name,
        model_name_emb=model_name_emb,
        model_name_vlm=model_name_vlm,
        target_answer=exp_config_train.target_answer,
        chosen_index=chosen_index,
        max_perturbation=max_perturbation,
        n_gradient_steps=exp_config_train.n_gradient_steps,
        lr_start=exp_config_train.lr_start,
        lr_end=exp_config_train.lr_end,
        max_batch_size_per_iter=exp_config_train.max_batch_size_per_iter,
        gradient_acc_steps=exp_config_train.gradient_acc_steps,
        lambda_emb=exp_config_train.lambda_emb,
        lambda_vlm=exp_config_train.lambda_vlm,
        emb_train_loss_type=emb_train_loss_type,
        is_adaptive=is_adaptive,
        lambda_constant=exp_config_train.lambda_constant
    )


    # TODO: we dont need to load the models and datasets every time if not changed
    # load embedding model and VLM
    embedder = EmbeddingModel(model_name_emb, device)
    vlm = VLM(model_name_vlm, device)
    print("Loaded models.")

    # load dataset
    ds = ViDoReDataset(ds_name, do_retrieval=False, embedder=embedder)
    query_strings = ds.queries_train
    print("Loaded dataset.")

    # choose attacked image
    chosen_image = ds.images[chosen_index]
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
        config=attack_config,
        print_every=exp_config_train.print_every,
        device=device
    )

    print(f"MSE: {torch.nn.functional.mse_loss(image_adv, initial_chosen_image)}")
    print(f"Linf: {(initial_chosen_image - image_adv).norm(p=float('inf'))}")

    # save adv image
    attack_dict = attack_config.to_dict()
    attack_dict["image_adv"] = image_adv.type(torch.uint8)
    filename = exp_config_train.save_folder / attack_config.create_filename()
    torch.save(attack_dict, filename)
    print(f"Saved adversarial image to {filename}.")


    del embedder, vlm, ds
    gc.collect()