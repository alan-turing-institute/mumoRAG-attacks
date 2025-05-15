import json
from collections import defaultdict
from strenum import StrEnum
import matplotlib.pyplot as plt

from config.task import generate_task_configs, TaskConfig
from config.experiment import ExperimentConfig
from wrappers.embedding import EmbedderName, is_loss_compatible
from wrappers.vlm import VLMName


class Metric(StrEnum):
    RETRIEVAL_RECALL_BEFORE = "Clean Retrieval Recall"
    RETRIEVAL_RECALL_AFTER = "Poisoned Retrieval Recall"
    RETRIEVAL_ASR_TRAIN = "Retrieval ASR (train)"
    RETRIEVAL_ASR_TEST =  "Retrieval ASR (test)"
    GENERATION_ASR_EXACT_TRAIN = "Generation ASR - Exact (train)"
    GENERATION_ASR_EMBED_TRAIN = "Generation ASR - Embedding (train)"
    GENERATION_ASR_EXACT_TEST = "Generation Accuracy - Embedding (train)"
    GENERATION_ASR_EMBED_TEST = "Generation ASR - Exact (test)"
    GENERATION_ACC_EMBED_GT_TRAIN = "Generation ASR - Embedding (test)"
    GENERATION_ACC_EMBED_GT_TEST = "Generation Accuracy - Embedding (test)"
    JUDGE_IMAGE_CONTXT_REL_TRAIN = "Judge Image Content Relevancy (train)"
    JUDGE_IMAGE_CONTXT_REL_TEST = "Judge Image Content Relevancy (test)"
    JUDGE_IMAGE_FAITH_TRAIN = "Judge Image Faithfulness (train)"
    JUDGE_IMAGE_FAITH_TEST = "Judge Image Faithfulness (test)"
    JUDGE_ANS_REL_TRAIN = "Judge Answer Relevancy (train)"
    JUDGE_ANS_REL_TEST = "Judge Answer Relevancy (test)"
    


def strip_hf_org(model_name):
    if isinstance(model_name, str): return model_name.split("/")[-1]
    if isinstance(model_name, list): 
        if isinstance(model_name[0], str):
            return [name.split("/")[-1] for name in model_name]
        if isinstance(model_name[0], tuple): 
            return [" , ".join([emb.split("/")[-1], vlm.split("/")[-1]]) for (emb, vlm) in model_name]
    raise NotImplementedError(model_name)


MODEL_NICKNAME_DICT = {
    EmbedderName.CLIP_BASE_PATCH16: "CLIP-B",
    EmbedderName.CLIP_LARGE_PATCH14: "CLIP-L",
    EmbedderName.JINA_CLIP_2: "Jina-CLIP",
    EmbedderName.COLSMOL_256M: "ColSmol-256M",
    EmbedderName.COLSMOL_500M: "ColSmol-500M",
    EmbedderName.COLPALI: "ColPali",
    EmbedderName.SIGLIP2_LARGE_PATCH16: "SigLIP-L",
    EmbedderName.SIGLIP2_BASE_PATCH16: "SigLIP-B",
    EmbedderName.QWEN2_GME_2B: "GME-Qwen2-VL-2B",

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
    raise NotImplementedError(model_name)

def shorten_model_name_str(model_name: str):
    return MODEL_NICKNAME_DICT.get(model_name, -1) if MODEL_NICKNAME_DICT.get(model_name, -1) != -1 else model_name

def get_metrics(exp_config: ExperimentConfig, task_config: TaskConfig, metrics_to_show: list[Metric], ret_topk_idx, gen_topk_idx, loss_idx):
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

    metrics_to_output[Metric.RETRIEVAL_RECALL_BEFORE] = metric_dict["retrieval"][key]["recall_before"]
    metrics_to_output[Metric.RETRIEVAL_RECALL_AFTER] = metric_dict["retrieval"][key]["recall_after"]
    metrics_to_output[Metric.RETRIEVAL_ASR_TRAIN] = metric_dict["retrieval"][key]["asr_train"]
    metrics_to_output[Metric.RETRIEVAL_ASR_TEST] = metric_dict["retrieval"][key]["asr_test"]

    key = f"gen_topk_{exp_config.eval.gen_topk_list[gen_topk_idx]}"
    if metric_dict["generation"]:
        metrics_to_output[Metric.GENERATION_ASR_EXACT_TRAIN] = metric_dict["generation"]["train"][key]["exact-asr"]["asr_universal"]
        metrics_to_output[Metric.GENERATION_ASR_EMBED_TRAIN] = metric_dict["generation"]["train"][key]["embed-adversarial"]["asr_universal"]
        metrics_to_output[Metric.GENERATION_ACC_EMBED_GT_TRAIN] = metric_dict["generation"]["train"][key]["embed-ground-truth"]["accuracy"]
        metrics_to_output[Metric.GENERATION_ASR_EXACT_TEST] = metric_dict["generation"]["test"][key]["exact-asr"]["asr_universal"]
        metrics_to_output[Metric.GENERATION_ASR_EMBED_TEST] = metric_dict["generation"]["test"][key]["embed-adversarial"]["asr_universal"]
        metrics_to_output[Metric.GENERATION_ACC_EMBED_GT_TEST] = metric_dict["generation"]["test"][key]["embed-ground-truth"]["accuracy"]
        
    if metric_dict["judge"]:
        metrics_to_output[Metric.GENERATION_ASR_EXACT_TRAIN] = metric_dict["judge"]["train"][key]["image_context_relevancy"]["asr_universal"]
        metrics_to_output[Metric.GENERATION_ASR_EXACT_TEST] = metric_dict["judge"]["test"][key]["image_context_relevancy"]["asr_universal"]
        metrics_to_output[Metric.JUDGE_IMAGE_FAITH_TRAIN] = metric_dict["judge"]["train"][key]["image_faithfulness"]["asr_universal"]
        metrics_to_output[Metric.JUDGE_IMAGE_FAITH_TEST] = metric_dict["judge"]["test"][key]["image_faithfulness"]["asr_universal"]
        metrics_to_output[Metric.JUDGE_ANS_REL_TRAIN] = metric_dict["judge"]["train"][key]["answer_relevancy"]["asr_universal"]
        metrics_to_output[Metric.JUDGE_ANS_REL_TEST] = metric_dict["judge"]["test"][key]["answer_relevancy"]["asr_universal"]


    metrics_filtered = {k: v for (k,v) in metrics_to_output.items() if k in metrics_to_show}
    metric_titles = [k.value for k in metrics_to_show]
    return metrics_filtered, metric_titles


def plot_metrics_vs_perturbation(exp_config: ExperimentConfig, metrics_to_show: list[Metric], ret_topk_idx, gen_topk_idx, loss_idx=0):
    """
    This function generates several plots, one plot per wrappers
    Each curve corresponds to one tuple (embedder, vlm)
    """
    if metrics_to_show is None:
        metrics_to_show = [m for m in Metric]
    task_configs = generate_task_configs(exp_config, include_eval=True)
    num_plots = len(metrics_to_show)

    # populate x, y vectors for each plot, and each curve
    plot_dict_x = [defaultdict(list) for _ in range(num_plots)]
    plot_dict_y = [defaultdict(list) for _ in range(num_plots)]
    for task_config in task_configs:
        metrics, titles = get_metrics(exp_config, task_config, metrics_to_show, ret_topk_idx, gen_topk_idx, loss_idx)

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
