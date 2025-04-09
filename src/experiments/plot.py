# from .params import exp_config_train, exp_config_eval
from utils.dataset import ViDoReDataset
from utils.utils import plot_images
from utils.embedding import EmbedderName
from utils.vlm import VLMName
from attack_eval import extract_attack_config, get_transferability_file_suffix, load_adv_image
from itertools import product
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from enum import IntEnum
from torchvision import transforms as T


class MetricIdx(IntEnum):
    RETRIEVAL_RECALL1_BEFORE = 0
    RETRIEVAL_RECALL1_AFTER = 1 
    RETRIEVAL_ASR1_TRAIN = 2
    RETRIEVAL_ASR1_TEST = 3
    RETRIEVAL_RECALL5_BEFORE = 4
    RETRIEVAL_RECALL5_AFTER = 5 
    RETRIEVAL_ASR5_TRAIN = 6
    RETRIEVAL_ASR5_TEST = 7
    GENERATION_ASR_EXACT_TRAIN = 8
    GENERATION_ASR_EMBED_TRAIN = 9 
    GENERATION_ASR_EXACT_TEST = 10
    GENERATION_ASR_EMBED_TEST = 11

METRIC_NAMES  = [
    "Clean Retrieval Recall@1",
    "Poisoned Retrieval Recall@1", 
    "Retrieval ASR@1 (train)",
    "Retrieval ASR@1 (test)",
    "Clean Retrieval Recall@5",
    "Poisoned Retrieval Recall@5", 
    "Retrieval ASR@5 (train)",
    "Retrieval ASR@5 (test)",
    "Generation ASR - Exact (train)",
    "Generation ASR - Embedding (train)", 
    "Generation ASR - Exact (test)",
    "Generation ASR - Embedding (test)",
]


def create_params_collection(exp_config_train, exp_config_eval):
    parameter_collection = product(
        exp_config_train.dataset_list, 
        exp_config_train.embedder_list, 
        exp_config_train.vlm_list, 
        exp_config_train.max_perturbation_list, 
        exp_config_train.emb_train_loss_type_list, 
        exp_config_train.is_adaptive_list,
        exp_config_train.chosen_index_list,
        exp_config_eval.eval_emb_list,
        exp_config_eval.eval_vlm_list,
    )
    return  [x for x in parameter_collection]

def strip_hf_org(model_name):
    if isinstance(model_name, str): return model_name.split("/")[-1]
    if isinstance(model_name, list): 
        if isinstance(model_name[0], str):
            return [name.split("/")[-1] for name in model_name]
        if isinstance(model_name[0], tuple): 
            return [" , ".join([emb.split("/")[-1], vlm.split("/")[-1]]) for (emb, vlm) in model_name]
        
MODEL_NICKNAME_DICT = {
    EmbedderName.CLIP_BASE_PATCH16: "CLIP-B",
    EmbedderName.CLIP_LARGE_PATCH14: "CLIP-L",
    EmbedderName.JINA_CLIP_2: "Jina-CLIP",
    EmbedderName.COLSMOL_256M: "ColSmol-256M",
    EmbedderName.COLSMOL_500M: "ColSmol-500M",

    VLMName.SMOLVLM_1_256M: "SmolVLM-256M",
    VLMName.SMOLVLM_1_500M: "SmolVLM-500M",
    VLMName.SMOLVLM_1_2B: "SmolVLM",
    VLMName.QWEN_2p5_VL_3B: "Qwen2.5-VL-3B"
}

def shorten_model_name(model_name):
    if isinstance(model_name, str): return shorten_model_name_str(model_name)
    if isinstance(model_name, list): 
        if isinstance(model_name[0], str):
            return [shorten_model_name_str(name) for name in model_name]
        if isinstance(model_name[0], tuple): 
            return [" , ".join([shorten_model_name_str(emb), shorten_model_name_str(vlm)]) for (emb, vlm) in model_name]

def shorten_model_name_str(model_name: str):
    return MODEL_NICKNAME_DICT.get(model_name, -1) if MODEL_NICKNAME_DICT.get(model_name, -1) != -1 else model_name

# def create_heatmap_matrices(): 

def plot_heatmap(num_plots, metric_tables: np.array, xaxis, yaxis, xlabel: str, ylabel: str, titles):
    fig, axs = plt.subplots(1, num_plots, figsize=(12,5))
    for plt_idx in range(num_plots):
        values = metric_tables[plt_idx].T
        im = axs[plt_idx].imshow(values, cmap="autumn")

        # Show all ticks and label them with the respective list entries
        axs[plt_idx].set_xticks(range(len(xaxis)), labels=shorten_model_name(xaxis),
                                rotation=45, ha="right", rotation_mode="anchor")
        axs[plt_idx].set_yticks(range(len(yaxis)), labels=shorten_model_name(yaxis),
                                rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(yaxis)):
            for j in range(len(xaxis)):
                text = axs[plt_idx].text(j, i, f"{values[i, j]:.3f}",
                            ha="center", va="center", color="k")
        axs[plt_idx].set_title(titles[plt_idx])
        axs[plt_idx].set_ylabel(ylabel)
        axs[plt_idx].set_xlabel(xlabel)
        cbar = axs[plt_idx].figure.colorbar(im, ax=axs[plt_idx])
        # cbar.ax.set_ylabel("", rotation=-90, va="bottom")
    # fig.suptitle(f'DS: {ds_name}', fontsize=12)
    fig.tight_layout()
    plt.show()



def get_metrics(exp_config_eval, params: tuple, metrics_to_show):
    
    ds_name, model_name_emb, model_name_vlm, max_perturbation, emb_train_loss_type, is_adaptive, chosen_index, eval_emb_name, eval_vlm_name = params
    
    # print(exp_config_eval, params)
    filename = exp_config_eval.results_folder / f"metrics_{extract_attack_config(params).create_hash_string()}{get_transferability_file_suffix(eval_emb_name, eval_vlm_name)}.json"
    with open(filename, "r") as file: 
        metric_dict = json.loads(file.read())
    key1 = f"loss_{exp_config_eval.emb_test_loss_type_list[0]}_topk_{exp_config_eval.topk_list[0]}"
    key2 = f"loss_{exp_config_eval.emb_test_loss_type_list[0]}_topk_{exp_config_eval.topk_list[1]}"
    
    metrics_to_output = [
        metric_dict["retrieval"][key1]["recall_before"],
        metric_dict["retrieval"][key1]["recall_after"],
        metric_dict["retrieval"][key1]["asr_train"],
        metric_dict["retrieval"][key1]["asr_test"],
        metric_dict["retrieval"][key2]["recall_before"],
        metric_dict["retrieval"][key2]["recall_after"],
        metric_dict["retrieval"][key2]["asr_train"],
        metric_dict["retrieval"][key2]["asr_test"],
        metric_dict["generation"]["train"]["exact"],
        metric_dict["generation"]["train"]["embed"],
        metric_dict["generation"]["test"]["exact"],
        metric_dict["generation"]["test"]["embed"],
    ]

    return (
        [m for i,m in enumerate(metrics_to_output) if i in metrics_to_show], 
        [t for i,t in enumerate(METRIC_NAMES) if i in metrics_to_show]
    )

def plot_metrics_vs_perturbation(exp_config_train, exp_config_eval, metrics_to_show):
    """
    This function generates several plots, one plot per dataset
    Each curve corrsponds to one tuple (embedder, vlm)
    """
    if metrics_to_show is None: metrics_to_show = [i for i in range(len(METRIC_NAMES))]
    params_collection = create_params_collection(exp_config_train, exp_config_eval)
    num_plots = len(metrics_to_show)
    
    
    # populate x, y vectors for each plot, and each curve
    plot_dict_x = [defaultdict(list) for _ in range(num_plots)]
    plot_dict_y = [defaultdict(list) for _ in range(num_plots)]
    for params in params_collection:
        ds_name, model_name_emb, model_name_vlm, max_perturbation, emb_train_loss_type, is_adaptive, chosen_index, eval_emb_name, eval_vlm_name = params
        metrics, titles = get_metrics(exp_config_eval, params, metrics_to_show)
    
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            label = f"{shorten_model_name(model_name_emb)} + {shorten_model_name(model_name_vlm)}"
            plot_dict_x[i][label].append(max_perturbation*255)
            plot_dict_y[i][label].append(metric)
    
    markers = ["o", "v", "s", "x", "+"]
    fig, ax = plt.subplots(1,1, figsize=(6,3))
    for label in plot_dict_x[i].keys():
        for i in range(len(metrics_to_show)):
            ax.plot(plot_dict_x[i][label], plot_dict_y[i][label], label=titles[i], marker=markers[i])
            # axs[i].set_title(titles[i])
            ax.set_xlabel("Maximum Allowed Perturbation (/255)")
            ax.set_ylabel("Attack Success")
            ax.set_xscale("log")
    ax.legend(loc='lower right')
    ax.grid(visible=True)
    ax.set_xticks(plot_dict_x[0][label], labels=plot_dict_x[0][label])
    ax.set_xlim([0.9,260])
    ax.set_ylim([-0.05,1.05])


def plot_model_heatmap(exp_config_train, exp_config_eval, ds_idx, metrics_to_show):
    """
    This function generates one heatmaop per dataset 
    x-axis -> embedder models
    """
    if metrics_to_show is None: metrics_to_show = [i for i in range(len(METRIC_NAMES))]
    num_plots = len(metrics_to_show)
    params_collection = create_params_collection(exp_config_train, exp_config_eval)
    params = params_collection[0]
    ds_name, model_name_emb, model_name_vlm, max_perturbation, emb_train_loss_type, is_adaptive, chosen_index, eval_emb_name, eval_vlm_name = params
    embs = exp_config_train.embedder_list
    vlms = exp_config_train.vlm_list
    ds_name = exp_config_train.dataset_list[ds_idx]

    # populate matrices to be used as heatmap
    metric_tables = [np.zeros((len(embs), len(vlms))) for _ in range(num_plots)]
    for emb_idx, model_name_emb in enumerate(embs):
        for vlm_idx, model_name_vlm in enumerate(vlms):
            params = ds_name, model_name_emb, model_name_vlm, max_perturbation, emb_train_loss_type, is_adaptive, chosen_index, eval_emb_name, eval_vlm_name
            metrics, titles = get_metrics(exp_config_eval, params, metrics_to_show)

            for m_idx, metric in enumerate(metrics):
                metric_tables[m_idx][emb_idx][vlm_idx] = metric
            
    # show heatmaps
    plot_heatmap(num_plots=num_plots, metric_tables=metric_tables, xaxis=embs, yaxis=vlms, xlabel="Embedding Model (Retriever)", ylabel="VLM (Generator)", titles=titles)



def plot_transferability(exp_config_train, exp_config_eval, metrics_to_show):
    if metrics_to_show is None: metrics_to_show = [i for i in range(len(METRIC_NAMES))]
    num_plots = len(metrics_to_show)
    params_collection = create_params_collection(exp_config_train, exp_config_eval)
    params = params_collection[0]

    ds_name, model_name_emb, model_name_vlm, max_perturbation, emb_train_loss_type, is_adaptive, chosen_index, eval_emb_name, eval_vlm_name = params
    embs_train = exp_config_train.embedder_list
    vlms_train = exp_config_train.vlm_list
    embs_test = exp_config_eval.eval_emb_list
    vlms_test = exp_config_eval.eval_vlm_list

    train_axis_list = [x for x in product(embs_train, vlms_train)]
    test_axis_list = [x for x in product(embs_test, vlms_test)]

    metric_tables = [np.zeros((len(train_axis_list), len(test_axis_list))) for _ in range(num_plots)]
    for train_idx, (emb_train, vlm_train) in enumerate(train_axis_list):
        for test_idx, (emb_test, vlm_test) in enumerate(test_axis_list):
            params = ds_name, emb_train, vlm_train, max_perturbation, emb_train_loss_type, is_adaptive, chosen_index, emb_test, vlm_test
            metrics, titles = get_metrics(exp_config_eval, params, metrics_to_show)

            for m_idx, metric in enumerate(metrics):
                metric_tables[m_idx][train_idx][test_idx] = metric
    
    plot_heatmap(num_plots=num_plots, metric_tables=metric_tables, xaxis=train_axis_list, yaxis=test_axis_list, xlabel="Training Models", ylabel="Testing Models", titles=titles)


def plot_images_side_by_side(exp_config_train, exp_config_eval, metrics_to_show):
    if metrics_to_show is None: metrics_to_show = [i for i in range(len(METRIC_NAMES))]
    params_collection = create_params_collection(exp_config_train, exp_config_eval)
    params = params_collection[0]

    ds_name, model_name_emb, model_name_vlm, max_perturbation, emb_train_loss_type, is_adaptive, chosen_index, eval_emb_name, eval_vlm_name = params
    chosen_indices = exp_config_train.chosen_index_list
    ds_names = exp_config_train.dataset_list

    for ds_idx, ds_name in enumerate(ds_names):
        ds = ViDoReDataset(ds_name, do_retrieval=False)
        for img_idx, chosen_index in enumerate(chosen_indices):
            params = ds_name, model_name_emb, model_name_vlm, max_perturbation, emb_train_loss_type, is_adaptive, chosen_index, eval_emb_name, eval_vlm_name
            metrics, titles = get_metrics(exp_config_eval, params, metrics_to_show)

            image_clean = T.Resize((512,512))(ds.images[chosen_index])

            attack_info_dict = load_adv_image(params, exp_config_train)
            image_adv = T.ToPILImage()(attack_info_dict['image_adv'])

            images = [image_clean, image_adv]
            fig, axes = plt.subplots(1, len(images), figsize=(12, 6))

            for i, ax in enumerate(axes.flat):
                img = images[i]
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
            plt.tight_layout()
            
            metrics_str  = " , ".join([f"{t}: {m}" for m, t in zip(metrics, titles)])
            print(f"DS: {ds_name}, image: {chosen_index}\nEmbedder: {model_name_emb}\nVLM: {model_name_vlm}\n{metrics_str}")