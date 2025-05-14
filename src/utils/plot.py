import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace
from enum import IntEnum
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms as T

from config.task import get_transferability_file_suffix, generate_task_configs, TaskConfig
from config.experiment import ExperimentConfig
from .image_utils import load_adv_image
from wrappers.embedding import EmbedderName
from wrappers.vlm import VLMName
from .logger import logger
from wrappers.cache import get_dataset


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
    if isinstance(model_name, Iterable):
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
    key1 = f"loss_{exp_config.eval.emb_test_loss_type_list[0]}_topk_{exp_config.eval.topk_list[0]}"
    key2 = f"loss_{exp_config.eval.emb_test_loss_type_list[0]}_topk_{exp_config.eval.topk_list[1]}"
    
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

def plot_metrics_vs_perturbation(exp_config: ExperimentConfig, metrics_to_show):
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
        metrics, titles = get_metrics(exp_config, task_config, metrics_to_show)

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            label = f"{shorten_model_name(task_config.model_name_emb)} + {shorten_model_name(task_config.model_name_vlm)}"
            plot_dict_x[i][label].append(task_config.max_perturbation*255)
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


def plot_model_heatmap(exp_config, ds_idx, metrics_to_show = None):
    """
    This function generates one heatmap per wrappers
    x-axis -> embedder models
    """
    if metrics_to_show is None: metrics_to_show = [i for i in range(len(METRIC_NAMES))]
    num_plots = len(metrics_to_show)
    task_configs = generate_task_configs(exp_config, include_eval=True)
    task_config = task_configs[0]
    embs = exp_config.train.embedder_list
    vlms = exp_config.train.vlm_list
    ds_name = exp_config.train.dataset_list[ds_idx]
    task_config = replace(task_config, ds_name=ds_name)

    # populate matrices to be used as heatmap
    metric_tables = [np.zeros((len(embs), len(vlms))) for _ in range(num_plots)]
    for emb_idx, model_name_emb in enumerate(embs):
        for vlm_idx, model_name_vlm in enumerate(vlms):
            task_config = replace(task_config,
                model_name_emb=model_name_emb,
                model_name_vlm=model_name_vlm,
            )
            metrics, titles = get_metrics(exp_config, task_config, metrics_to_show)

            for m_idx, metric in enumerate(metrics):
                metric_tables[m_idx][emb_idx][vlm_idx] = metric

    # show heatmaps
    plot_heatmap(num_plots=num_plots, metric_tables=metric_tables, xaxis=embs, yaxis=vlms, xlabel="Embedding Model (Retriever)", ylabel="VLM (Generator)", titles=titles)



def plot_transferability(exp_config, metrics_to_show = None):
    if metrics_to_show is None: metrics_to_show = [i for i in range(len(METRIC_NAMES))]
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
    if metrics_to_show is None: metrics_to_show = [i for i in range(len(METRIC_NAMES))]
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