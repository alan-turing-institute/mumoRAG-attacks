import json
from collections import defaultdict
from dataclasses import replace
from enum import IntEnum
from pprint import pprint

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms as T

from config.task import generate_task_configs, TaskConfig
from config.experiment import ExperimentConfig
from .image_utils import load_adv_image
from wrappers.embedding import EmbedderName, is_loss_compatible
from wrappers.vlm import VLMName
from .logger import logger
from wrappers.cache import get_dataset


class MetricIdx(IntEnum):
    RETRIEVAL_RECALL_BEFORE = 0
    RETRIEVAL_RECALL_AFTER = 1
    RETRIEVAL_ASR_TRAIN = 2
    RETRIEVAL_ASR_TEST = 3
    GENERATION_ASR_EXACT_TRAIN = 4
    GENERATION_ASR_EMBED_TRAIN = 5
    GENERATION_ASR_EXACT_TEST = 6
    GENERATION_ASR_EMBED_TEST = 7
    Generation_ACC_EMBED_GT_TRAIN = 8
    Generation_ACC_EMBED_GT_TEST = 9

METRIC_NAMES_DICT = {
    MetricIdx.RETRIEVAL_RECALL_BEFORE: "Clean Retrieval Recall",
    MetricIdx.RETRIEVAL_RECALL_AFTER: "Poisoned Retrieval Recall",
    MetricIdx.RETRIEVAL_ASR_TRAIN: "Retrieval ASR (train)",
    MetricIdx.RETRIEVAL_ASR_TEST: "Retrieval ASR (test)",
    MetricIdx.GENERATION_ASR_EXACT_TRAIN: "Generation ASR - Exact (train)",
    MetricIdx.GENERATION_ASR_EMBED_TRAIN: "Generation ASR - Embedding (train)",
    MetricIdx.Generation_ACC_EMBED_GT_TRAIN: "Generation Accuracy - Embedding (train)",
    MetricIdx.GENERATION_ASR_EXACT_TEST: "Generation ASR - Exact (test)",
    MetricIdx.GENERATION_ASR_EMBED_TEST: "Generation ASR - Embedding (test)",
    MetricIdx.Generation_ACC_EMBED_GT_TEST: "Generation Accuracy - Embedding(test)",
}

METRIC_NAMES  = [
    "Clean Retrieval Recall",
    "Poisoned Retrieval Recall", 
    "Retrieval ASR (train)",
    "Retrieval ASR (test)",
    "Clean Retrieval Recall",
    "Poisoned Retrieval Recall", 
    "Retrieval ASR (train)",
    "Retrieval ASR (test)",
    "Generation ASR - Exact (train)",
    "Generation ASR - Embedding (train)", 
    "Generation ASR - Exact (test)",
    "Generation ASR - Embedding (test)",
]


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
    EmbedderName.COLPALI: "ColPali",
    EmbedderName.SIGLIP2_LARGE_PATCH16: "SigLIP-L",
    EmbedderName.SIGLIP2_BASE_PATCH16: "SigLIP-B",

    VLMName.SMOLVLM_1_256M: "SmolVLM-256M",
    VLMName.SMOLVLM_1_500M: "SmolVLM-500M",
    VLMName.SMOLVLM_1_2B: "SmolVLM",
    VLMName.QWEN_2p5_VL_3B: "Qwen2.5-VL-3B",
    VLMName.INTERNVL_3_2B: "InternVL3-2B",
}

def shorten_model_name(model_name):
    if isinstance(model_name, str): return shorten_model_name_str(model_name)
    if isinstance(model_name, list):
        if len(model_name) == 1:
            return shorten_model_name_str(model_name[0])
        if isinstance(model_name[0], str):
            return [shorten_model_name_str(name) for name in model_name]
        if isinstance(model_name[0], tuple): 
            return [" , ".join([shorten_model_name_str(emb), shorten_model_name_str(vlm)]) for (emb, vlm) in model_name]
    raise NotImplementedError

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



def get_metrics(exp_config: ExperimentConfig, task_config: TaskConfig, metrics_to_show):
    filename = task_config.get_result_filename(exp_config.eval.results_folder)
    with open(filename, "r") as file:
        metric_dict = json.loads(file.read())
    metrics_to_output = []

    # retrieval metrics
    for topk in exp_config.eval.topk_list:
        key = f"loss_{exp_config.eval.emb_test_loss_type_list[0]}_topk_{topk}"
        metrics_to_output.extend([
            metric_dict["retrieval"][key]["recall_before"],
            metric_dict["retrieval"][key]["recall_after"],
            metric_dict["retrieval"][key]["asr_train"],
            metric_dict["retrieval"][key]["asr_test"],
        ])
    
    for gen_topk, split in product(exp_config.eval.gen_topk_list, ["train", "test"]):
        key = f"gen_topk_{gen_topk}"
        metrics_to_output.extend([
            metric_dict["generation"][split][key]["exact-asr"]["asr_universal"],
            metric_dict["generation"][split][key]["embed-adversarial"]["asr_universal"],
            metric_dict["generation"][split][key]["embed-ground-truth"]["accuracy"],
        ])

    return (
        [m for i,m in enumerate(metrics_to_output) if i in metrics_to_show], 
        [t for i,t in enumerate(METRIC_NAMES) if i in metrics_to_show]
    )

def get_metrics_2(exp_config: ExperimentConfig, task_config: TaskConfig, metrics_to_show, ret_topk_idx, gen_topk_idx, loss_idx):
    filename = task_config.get_result_filename(exp_config.eval.results_folder)
    with open(filename, "r") as file:
        metric_dict = json.loads(file.read())
    metrics_to_output = {}

    # retrieval metrics
    # for topk in exp_config.eval.topk_list:
    if task_config.eval_emb_name:
        model_name_emb = task_config.eval_emb_name
    else:
        model_name_emb = task_config.model_name_embs[0]
    emb_test_loss_type_list_compatible = [loss for loss in exp_config.eval.emb_test_loss_type_list if is_loss_compatible(model_name_emb, loss)]
    key = f"loss_{emb_test_loss_type_list_compatible[loss_idx]}_topk_{exp_config.eval.topk_list[ret_topk_idx]}"

    try:
        metrics_to_output[MetricIdx.RETRIEVAL_RECALL_BEFORE] = metric_dict["retrieval"][key]["recall_before"]
        metrics_to_output[MetricIdx.RETRIEVAL_RECALL_AFTER] = metric_dict["retrieval"][key]["recall_after"]
        metrics_to_output[MetricIdx.RETRIEVAL_ASR_TRAIN] = metric_dict["retrieval"][key]["asr_train"]
        metrics_to_output[MetricIdx.RETRIEVAL_ASR_TEST] = metric_dict["retrieval"][key]["asr_test"]
    except KeyError:
        pprint(exp_config)
        pprint(task_config)
        print(key)
        raise
        
    # for gen_topk in exp_config.eval.gen_topk_list:
    key = f"gen_topk_{exp_config.eval.gen_topk_list[gen_topk_idx]}"
    metrics_to_output[MetricIdx.GENERATION_ASR_EXACT_TRAIN] = metric_dict["generation"]["train"][key]["exact-asr"]["asr_universal"]
    metrics_to_output[MetricIdx.GENERATION_ASR_EMBED_TRAIN] = metric_dict["generation"]["train"][key]["embed-adversarial"]["asr_universal"] 
    metrics_to_output[MetricIdx.Generation_ACC_EMBED_GT_TRAIN] = metric_dict["generation"]["train"][key]["embed-ground-truth"]["accuracy"]
    metrics_to_output[MetricIdx.GENERATION_ASR_EXACT_TEST] = metric_dict["generation"]["test"][key]["exact-asr"]["asr_universal"]
    metrics_to_output[MetricIdx.GENERATION_ASR_EMBED_TEST] = metric_dict["generation"]["test"][key]["embed-adversarial"]["asr_universal"] 
    metrics_to_output[MetricIdx.Generation_ACC_EMBED_GT_TEST] = metric_dict["generation"]["test"][key]["embed-ground-truth"]["accuracy"] 

    metrics_filtered = {k: v for (k,v) in metrics_to_output.items() if k in metrics_to_show}
    metric_titles = [METRIC_NAMES_DICT[k] for k in metrics_to_show]
    return metrics_filtered, metric_titles

def plot_metrics_vs_perturbation(exp_config: ExperimentConfig, metrics_to_show, ret_topk_idx, gen_topk_idx, loss_idx=0):
    """
    This function generates several plots, one plot per wrappers
    Each curve corresponds to one tuple (embedder, vlm)
    """
    if metrics_to_show is None: metrics_to_show = [i for i in range(len(METRIC_NAMES))]
    task_configs = generate_task_configs(exp_config, include_eval=True)
    num_plots = len(metrics_to_show)

    # populate x, y vectors for each plot, and each curve
    plot_dict_x = [defaultdict(list) for _ in range(num_plots)]
    plot_dict_y = [defaultdict(list) for _ in range(num_plots)]
    for task_config in task_configs:
        metrics, titles = get_metrics_2(exp_config, task_config, metrics_to_show, ret_topk_idx, gen_topk_idx, loss_idx)

        for i, metric_name in enumerate(metrics_to_show):
        # for i, (metric, title) in enumerate(zip(metrics, titles)):
            metric = metrics[metric_name]
            label = f"{shorten_model_name(task_config.model_name_embs)} + {shorten_model_name(task_config.vlm.models) if task_config.vlm else ''}"
            plot_dict_x[i][label].append(task_config.max_perturbation*255)
            plot_dict_y[i][label].append(metric)
    markers = ["o", "v", "s", "x", "+", "^"]
    n_plots = len(plot_dict_x[0].keys())
    fig, ax = plt.subplots(1, n_plots, figsize=(15,3))
    for k, label in enumerate(plot_dict_x[0].keys()):
        for i in range(len(metrics_to_show)):
            ax[k].plot(plot_dict_x[i][label], plot_dict_y[i][label], label=titles[i], marker=markers[i])
            # axs[i].set_title(titles[i])
            ax[k].set_xlabel("Maximum Allowed Perturbation (/255)")
            ax[k].set_ylabel("Attack Success")
            ax[k].set_xscale("log")
        ax[k].legend(loc='lower right')
        ax[k].grid(visible=True)
        ax[k].set_xticks(plot_dict_x[0][label], labels=plot_dict_x[0][label])
        ax[k].set_xlim([0.9,260])
        ax[k].set_ylim([-0.05,1.05])
        ax[k].set_title(label)


def plot_colpali_ablations(exp_config: ExperimentConfig, metrics_to_show, ret_topk_idx=0, gen_topk_idx=0):
    """
    This function generates one heatmap per wrappers
    x-axis -> embedder models
    """
    if metrics_to_show is None: metrics_to_show = [i for i in range(len(METRIC_NAMES))]
    num_plots = len(metrics_to_show)
    task_configs = generate_task_configs(exp_config, include_eval=True)
    task_config = task_configs[0]
    losses_train = exp_config.train.emb_train_loss_type_list
    losses_eval = exp_config.eval.emb_test_loss_type_list

    # populate matrices to be used as heatmap
    metric_tables = [np.zeros((len(losses_train), len(losses_eval))) for _ in range(num_plots)]
    for loss_train_idx, loss_train in enumerate(losses_train):
        for loss_eval_idx, loss_eval in enumerate(losses_eval):
            task_config = replace(task_config,
                emb_train_loss_type=loss_train,
            )
            metrics, titles = get_metrics_2(exp_config, task_config, metrics_to_show, ret_topk_idx, gen_topk_idx, loss_eval_idx)
            # metrics, titles = get_metrics(exp_config, task_config, metrics_to_show)

            for m_idx, (metric_name, metric_value) in enumerate(metrics.items()):
                metric_tables[m_idx][loss_train_idx][loss_eval_idx] = metric_value

    # show heatmaps
    plot_heatmap(num_plots=num_plots, metric_tables=metric_tables, xaxis=losses_train, yaxis=losses_train, xlabel="Training Loss", ylabel="Evaluation Loss", titles=titles)


def plot_model_heatmap(exp_config: ExperimentConfig, ds_idx, metrics_to_show = None):
    """
    This function generates one heatmap per wrappers
    x-axis -> embedder models
    """
    if metrics_to_show is None:
        metrics_to_show = [i for i in range(len(METRIC_NAMES))]
    num_plots = len(metrics_to_show)
    task_configs = generate_task_configs(exp_config, include_eval=True)
    task_config = task_configs[0]
    embs = exp_config.train.embedder_list
    vlms = exp_config.train.vlm.models
    ds_name = exp_config.train.dataset_list[ds_idx]
    task_config = replace(task_config, ds_name=ds_name)

    # populate matrices to be used as heatmap
    metric_tables = [np.zeros((len(embs), len(vlms))) for _ in range(num_plots)]
    for emb_idx, model_name_emb in enumerate(embs):
        for vlm_idx, model_name_vlm in enumerate(vlms):
            task_config = replace(task_config,
                model_name_embs=model_name_emb,
                vlm=replace(task_config.vlm, models=model_name_vlm),
            )
            metrics, titles = get_metrics(exp_config, task_config, metrics_to_show)

            for m_idx, metric in enumerate(metrics):
                metric_tables[m_idx][emb_idx][vlm_idx] = metric

    # show heatmaps
    plot_heatmap(num_plots=num_plots, metric_tables=metric_tables, xaxis=embs, yaxis=vlms, xlabel="Embedding Model (Retriever)", ylabel="VLM (Generator)", titles=titles)



def plot_transferability(exp_config, metrics_to_show = None):
    if metrics_to_show is None:
        metrics_to_show = [i for i in range(len(METRIC_NAMES))]
    num_plots = len(metrics_to_show)
    task_configs = generate_task_configs(exp_config, include_eval=True)
    task_config = task_configs[0]

    embs_train = exp_config.train.embedder_list
    vlms_train = exp_config.train.vlm_list
    embs_test = exp_config.eval.eval_emb_list
    vlms_test = exp_config.eval.eval_vlm_list

    train_axis_list = [x for x in product(embs_train, vlms_train)]
    test_axis_list = [x for x in product(embs_test, vlms_test)]

    metric_tables = [np.zeros((len(train_axis_list), len(test_axis_list))) for _ in range(num_plots)]
    for train_idx, (emb_train, vlm_train) in enumerate(train_axis_list):
        for test_idx, (emb_test, vlm_test) in enumerate(test_axis_list):
            task_config = replace(task_config,
                model_name_emb=emb_train,
                model_name_vlm=vlm_train,
                eval_emb_name=emb_test,
                eval_vlm_name=vlm_test,
            )
            metrics, titles = get_metrics(exp_config, task_config, metrics_to_show)

            for m_idx, metric in enumerate(metrics):
                metric_tables[m_idx][train_idx][test_idx] = metric
    
    plot_heatmap(num_plots=num_plots, metric_tables=metric_tables, xaxis=train_axis_list, yaxis=test_axis_list, xlabel="Training Models", ylabel="Testing Models", titles=titles)


def plot_images_side_by_side(exp_config, metrics_to_show):
    if metrics_to_show is None:
        metrics_to_show = [i for i in range(len(METRIC_NAMES))]
    task_configs = generate_task_configs(exp_config, include_eval=True)
    task_config = task_configs[0]

    chosen_indices = exp_config.train.chosen_index_list
    ds_names = exp_config.train.dataset_list

    for ds_idx, ds_name in enumerate(ds_names):
        ds = get_dataset(ds_name)
        for img_idx, chosen_index in enumerate(chosen_indices):
            task_config = replace(task_config,
                ds_name=ds_name,
                chosen_index=chosen_index,
            )
            metrics, titles = get_metrics(exp_config, task_config, metrics_to_show)

            image_clean = T.Resize((512,512))(ds.images[chosen_index])

            image_adv = T.ToPILImage()(load_adv_image(task_config, exp_config.train))

            images = [image_clean, image_adv]
            fig, axes = plt.subplots(1, len(images), figsize=(12, 6))

            for i, ax in enumerate(axes.flat):
                img = images[i]
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
            plt.tight_layout()
            
            metrics_str  = " , ".join([f"{t}: {m}" for m, t in zip(metrics, titles)])
            logger.info(f"DS: {ds_name}, image: {chosen_index}\nEmbedder: {task_config.model_name_emb}\nVLM: {task_config.model_name_vlm}\n{metrics_str}")