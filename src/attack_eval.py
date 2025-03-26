import torchvision.transforms as T
from utils.utils import get_device, attempt_load_pt
import torch
from utils.embedding import EmbeddingModel
from utils.vlm import VLM
from utils.dataset import ViDoReDataset
from utils.attack_config import AttackConfig
from experiments.params import exp_config_train, exp_config_eval
from itertools import product


def load_adv_image(params) -> torch.tensor:
    ds_name, model_name_emb, model_name_vlm, max_perturbation, emb_train_loss_type, is_adaptive, _, _ = params
    lambda_constant = exp_config_train.lambda_constant

    attack_config = AttackConfig(
        ds_name=ds_name,
        model_name_emb=model_name_emb,
        model_name_vlm=model_name_vlm,
        target_answer=exp_config_train.target_answer,
        chosen_index=exp_config_train.chosen_index,
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
        lambda_constant=lambda_constant,
    )

    # load adversarial image
    filename = exp_config_train.save_folder+attack_config.create_filename()
    attack_info_dict = attempt_load_pt(filename)
    if attack_info_dict is None:
        print(f"Error! Could not find file: {filename}! You need to train an attack with this configuration first")
        raise ValueError
    
    return attack_info_dict



parameter_collection = product(
    exp_config_train.dataset_list, 
    exp_config_train.embedder_list, 
    exp_config_train.vlm_list, 
    exp_config_train.max_perturbation_list, 
    exp_config_train.emb_train_loss_type_list, 
    exp_config_train.is_adaptive_list,
    exp_config_eval.eval_emb_list,
    exp_config_eval.eval_vlm_list,
)
parameter_collection = [x for x in parameter_collection]
n_evals = len(parameter_collection)
device = get_device(prefer_mps=True)


# first we just make sure that all required files are on disk, so that we dont waste time
# this will raise an error if there are missing file(s)
for params in parameter_collection:
    _ = load_adv_image(params)


for i, params in enumerate(parameter_collection):

    print(f"++++++++++++++++++++++\nEval {(i+1):4d}/{n_evals}, params -> {params}")
    ds_name, model_name_emb, model_name_vlm, max_perturbation, emb_train_loss_type, is_adaptive, eval_emb_name, eval_vlm_name = params

    attack_info_dict = load_adv_image(params)
    image_adv = attack_info_dict['image_adv']
    

    # update model names in case we test transferability
    model_name_emb = model_name_emb if eval_emb_name=="" else eval_emb_name
    model_name_vlm = model_name_vlm if eval_vlm_name=="" else eval_vlm_name
    
    # TODO: we dont need to load the models and datasets every time if not changed
    # load embedding model and VLM
    embedder = EmbeddingModel(model_name_emb, device)
    vlm = VLM(model_name_vlm, device)
    print("Loaded models.")

    # load dataset
    ds = ViDoReDataset(ds_name, do_retrieval=exp_config_eval.do_retrieval, embedder=embedder)
    print("Loaded dataset.")


    # test retrieval
    if exp_config_eval.do_retrieval:
        print("=== Evaluating retrieval ...")
        ds.add_adv_image(T.ToPILImage()(image_adv/255))
        metric_dict_before, retrievals_before = ds.evaluate_retrieval(ks=exp_config_eval.topk_list, loss_types=exp_config_eval.emb_test_loss_type_list, include_adv=False)
        print(f"Before attack:\n{metric_dict_before}")
        metric_dict_after, retrievals_after = ds.evaluate_retrieval(ks=exp_config_eval.topk_list, loss_types=exp_config_eval.emb_test_loss_type_list, include_adv=True)
        print(f"After attack:\n{metric_dict_after}")
    
    # test generation
    if exp_config_eval.do_generation:
        print("=== Evaluating generation ...")
        asr_test, gs_test = ds.evaluate_generation(vlm, image_adv, exp_config_train.target_answer, eval_train=False)
        print(f"Test ASR: {asr_test:.2f}")
        asr_train, gs_train = ds.evaluate_generation(vlm, image_adv, exp_config_train.target_answer, eval_train=True)
        print(f"Train ASR: {asr_train:.2f}")