import json
from collections import defaultdict

import matplotlib.pyplot as plt
from strenum import StrEnum

from config.experiment import ExperimentConfig
from config.task import TaskConfig, generate_task_configs
from utils.image_utils import gpt_filename
from wrappers.embedded_dataset import make_safe_filename
from wrappers.embedding import EmbedderName, is_loss_compatible
from wrappers.judge import JudgeMetric
from wrappers.vlm import VLMEvaluationMetric, VLMName


class Metric(StrEnum):
    RETRIEVAL_RECALL_BEFORE = "Recall-B"
    RETRIEVAL_RECALL_AFTER = "Recall-A"
    RETRIEVAL_ASR_TRAIN = "ASR-R (train)"
    RETRIEVAL_ASR_TEST = "ASR-R (test)"
    RETRIEVAL_ASR_TARGETED = "ASR-R targeted"
    RETRIEVAL_FPR_TARGETED_TRAIN = "FPR-R targeted (train)"
    RETRIEVAL_FPR_TARGETED_TEST = "FPR-R targeted (test)"
    GENERATION_ASR_EXACT_TRAIN = "ASR-G-HARD (train)"
    GENERATION_ASR_EMBED_TRAIN = "SIM-G-ADV (train)"
    GENERATION_ASR_EXACT_TEST = "ASR-G-HARD (test)"
    GENERATION_ASR_EMBED_TEST = "SIM-G-ADV (test)"
    GENERATION_ACC_EMBED_GT_TRAIN = "SIM-G-GT (train)"
    GENERATION_ACC_EMBED_GT_TEST = "SIM-G-GT (test)"
    GENERATION_ASR_EXACT_TARGETED_TRAIN = "ASR-G-HARD targeted (train)"
    GENERATION_FPR_TARGETED_TRAIN = "FPR-G-HARD targeted (train)"
    GENERATION_ASR_EMBED_TARGETED_TRAIN = "SIM-G-ADV-POS targeted (train)"
    GENERATION_FPR_EMBED_TARGETED_TRAIN = "SIM-G-ADV-NEG targeted (train)"
    GENERATION_FPR_TARGETED_TEST = "FPR-G-HARD targeted (test)"
    GENERATION_FPR_EMBED_TARGETED_TEST = "SIM-G-ADV-NEG targeted (test)"
    JUDGE_IMAGE_CONTXT_REL_TRAIN = "Judge Image Content Relevancy (train)"
    JUDGE_IMAGE_CONTXT_REL_TEST = "Judge Image Content Relevancy (test)"
    JUDGE_IMAGE_FAITH_TRAIN = "Judge Image Faithfulness (train)"
    JUDGE_IMAGE_FAITH_TEST = "Judge Image Faithfulness (test)"
    JUDGE_ANS_REL_TRAIN = "Judge Answer Relevancy (train)"
    JUDGE_ANS_REL_TEST = "Judge Answer Relevancy (test)"


class PlotFilter:
    # condition lambdas (useful for selecting task configs from the tabulate object)
    CONDITION_SAME_MODELS = lambda row: row["eval vlm"] == row["vlm"] and row["eval emb"] == row["embedder"]

    # metric_lists
    METRICS_TEST = [
        Metric.RETRIEVAL_ASR_TEST,
        Metric.GENERATION_ASR_EXACT_TEST,
        Metric.GENERATION_ASR_EMBED_TEST,
        Metric.GENERATION_ACC_EMBED_GT_TEST,
        Metric.JUDGE_ANS_REL_TEST,
        Metric.JUDGE_IMAGE_FAITH_TEST,
        Metric.JUDGE_IMAGE_CONTXT_REL_TEST,
    ]
    METRICS_JUDGE = [
        Metric.JUDGE_ANS_REL_TEST,
        Metric.JUDGE_ANS_REL_TRAIN,
        Metric.JUDGE_IMAGE_CONTXT_REL_TEST,
        Metric.JUDGE_IMAGE_CONTXT_REL_TRAIN,
        Metric.JUDGE_IMAGE_FAITH_TEST,
        Metric.JUDGE_IMAGE_FAITH_TRAIN,
    ]

    METRICS_RECALL = [
        Metric.RETRIEVAL_RECALL_BEFORE,
        Metric.RETRIEVAL_RECALL_AFTER,
    ]

    ALL_METRICS_TARGETED = [
        Metric.RETRIEVAL_ASR_TARGETED,
        # Metric.RETRIEVAL_FPR_TARGETED_TRAIN,
        Metric.RETRIEVAL_FPR_TARGETED_TEST,
        # Metric.GENERATION_ASR_EXACT_TARGETED_TRAIN,
        # Metric.GENERATION_FPR_TARGETED_TRAIN,
        Metric.GENERATION_ASR_EMBED_TARGETED_TRAIN,
        # Metric.GENERATION_FPR_EMBED_TARGETED_TRAIN,
        # Metric.GENERATION_FPR_TARGETED_TEST,
        Metric.GENERATION_FPR_EMBED_TARGETED_TEST,
    ]

    METRICS_COLPALI = [
        Metric.RETRIEVAL_ASR_TEST,
    ]

    METRICS_TEST_JUDGE = list(set(METRICS_JUDGE) & set(METRICS_TEST))

    ALL_METRICS_UNTARGETED = METRICS_TEST + METRICS_RECALL

    METRICS_TOPK = [
        Metric.GENERATION_ASR_EXACT_TEST,
        Metric.GENERATION_ASR_EMBED_TEST,
        Metric.GENERATION_ACC_EMBED_GT_TEST,
    ]

    METRICS_TOPK_TARGETED = [
        Metric.GENERATION_ASR_EMBED_TARGETED_TRAIN,
        Metric.GENERATION_FPR_EMBED_TARGETED_TEST,
    ]


def strip_hf_org(model_name):
    if isinstance(model_name, str):
        return model_name.split("/")[-1]
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
    if isinstance(model_name, str):
        return shorten_model_name_str(model_name)
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


def get_metrics(
    exp_config: ExperimentConfig,
    task_config: TaskConfig,
    metrics_to_show: list[Metric],
    ret_topk_idx: int | None = None,
    gen_topk_idx: int | None = None,
    loss_idx: int | None = None,
):
    if task_config.eval_emb_name:
        model_name_emb = task_config.eval_emb_name
    else:
        model_name_emb = task_config.model_name_embs[0]

    if exp_config.eval.test_gpt_attack:
        if task_config.eval_vlm_name:
            model_name_vlm = task_config.eval_vlm_name
        else:
            model_name_vlm = task_config.vlm.models[0]

        filename = exp_config.eval.results_folder / f"metrics_{gpt_filename(task_config)}_{make_safe_filename(f'{model_name_emb}_{model_name_vlm}')}.json"
    else:
        filename = task_config.get_result_filename(exp_config.eval.results_folder)
    with open(filename, "r") as file:
        metric_dict = json.loads(file.read())
    metrics_to_output = {}

    emb_test_loss_type_list_compatible = [loss for loss in exp_config.eval.emb_test_loss_type_list if is_loss_compatible(model_name_emb, loss)]
    if loss_idx is not None:
        emb_test_loss_type_list_compatible = emb_test_loss_type_list_compatible[loss_idx : loss_idx + 1]

    if ret_topk_idx is not None:
        ret_topks = exp_config.eval.topk_list[ret_topk_idx : ret_topk_idx + 1]
    else:
        ret_topks = exp_config.eval.topk_list
    for loss in emb_test_loss_type_list_compatible:
        loss_tag = f"+{loss}" if len(emb_test_loss_type_list_compatible) > 1 else ""
        for ret_topk in ret_topks:
            key = f"loss_{loss}_topk_{ret_topk}"

            metrics_to_output[f"{Metric.RETRIEVAL_RECALL_BEFORE.value}{loss_tag}@{ret_topk}"] = metric_dict["retrieval"][key]["recall_before"]
            metrics_to_output[f"{Metric.RETRIEVAL_RECALL_AFTER.value}{loss_tag}@{ret_topk}"] = metric_dict["retrieval"][key]["recall_after"]
            metrics_to_output[f"{Metric.RETRIEVAL_ASR_TRAIN.value}{loss_tag}@{ret_topk}"] = metric_dict["retrieval"][key]["asr_train"]
            metrics_to_output[f"{Metric.RETRIEVAL_ASR_TEST.value}{loss_tag}@{ret_topk}"] = metric_dict["retrieval"][key]["asr_test"]
            if task_config.is_targeted:
                metrics_to_output[f"{Metric.RETRIEVAL_ASR_TARGETED.value}{loss_tag}@{ret_topk}"] = metric_dict["retrieval"][key]["asr_targeted"]
                metrics_to_output[f"{Metric.RETRIEVAL_FPR_TARGETED_TRAIN.value}{loss_tag}@{ret_topk}"] = metric_dict["retrieval"][key]["fpr_targeted_train"]
                metrics_to_output[f"{Metric.RETRIEVAL_FPR_TARGETED_TEST.value}{loss_tag}@{ret_topk}"] = metric_dict["retrieval"][key]["fpr_targeted_test"]

    if gen_topk_idx is not None:
        gen_topks = exp_config.eval.gen_topk_list[gen_topk_idx : gen_topk_idx + 1]
    else:
        gen_topks = exp_config.eval.gen_topk_list
    for gen_topk in gen_topks:
        key = f"gen_topk_{gen_topk}"
        if metric_dict["generation"]:
            metrics_to_output[f"{Metric.GENERATION_ASR_EXACT_TRAIN.value}@{gen_topk}"] = metric_dict["generation"]["train"][key][VLMEvaluationMetric.ASR_EXACT]["asr_universal"]
            metrics_to_output[f"{Metric.GENERATION_ASR_EMBED_TRAIN.value}@{gen_topk}"] = metric_dict["generation"]["train"][key][VLMEvaluationMetric.EMBED_ADV]["asr_universal"]
            metrics_to_output[f"{Metric.GENERATION_ACC_EMBED_GT_TRAIN.value}@{gen_topk}"] = metric_dict["generation"]["train"][key][VLMEvaluationMetric.EMBED_GT]["accuracy"]
            metrics_to_output[f"{Metric.GENERATION_ASR_EXACT_TEST.value}@{gen_topk}"] = metric_dict["generation"]["test"][key][VLMEvaluationMetric.ASR_EXACT]["asr_universal"]
            metrics_to_output[f"{Metric.GENERATION_ASR_EMBED_TEST.value}@{gen_topk}"] = metric_dict["generation"]["test"][key][VLMEvaluationMetric.EMBED_ADV]["asr_universal"]
            metrics_to_output[f"{Metric.GENERATION_ACC_EMBED_GT_TEST.value}@{gen_topk}"] = metric_dict["generation"]["test"][key][VLMEvaluationMetric.EMBED_GT]["accuracy"]
            if task_config.is_targeted:
                metrics_to_output[f"{Metric.GENERATION_ASR_EXACT_TARGETED_TRAIN.value}@{gen_topk}"] = metric_dict["generation"]["train"][key][VLMEvaluationMetric.ASR_EXACT]["asr_targeted"]
                metrics_to_output[f"{Metric.GENERATION_FPR_TARGETED_TRAIN.value}@{gen_topk}"] = metric_dict["generation"]["train"][key][VLMEvaluationMetric.ASR_EXACT]["fpr_targeted"]
                metrics_to_output[f"{Metric.GENERATION_ASR_EMBED_TARGETED_TRAIN.value}@{gen_topk}"] = metric_dict["generation"]["train"][key][VLMEvaluationMetric.EMBED_ADV]["asr_targeted"]
                metrics_to_output[f"{Metric.GENERATION_FPR_EMBED_TARGETED_TRAIN.value}@{gen_topk}"] = metric_dict["generation"]["train"][key][VLMEvaluationMetric.EMBED_ADV]["fpr_targeted"]
                metrics_to_output[f"{Metric.GENERATION_FPR_TARGETED_TEST.value}@{gen_topk}"] = metric_dict["generation"]["test"][key][VLMEvaluationMetric.ASR_EXACT]["fpr_targeted"]
                metrics_to_output[f"{Metric.GENERATION_FPR_EMBED_TARGETED_TEST.value}@{gen_topk}"] = metric_dict["generation"]["test"][key][VLMEvaluationMetric.EMBED_ADV]["fpr_targeted"]

        if metric_dict["judge"]:
            metrics_to_output[f"{Metric.JUDGE_IMAGE_CONTXT_REL_TRAIN.value}@{gen_topk}"] = metric_dict["judge"]["train"][key][JudgeMetric.IMAGE_CONTEXT_RELEVANCY]["asr_universal"]
            metrics_to_output[f"{Metric.JUDGE_IMAGE_CONTXT_REL_TEST.value}@{gen_topk}"] = metric_dict["judge"]["test"][key][JudgeMetric.IMAGE_CONTEXT_RELEVANCY]["asr_universal"]
            metrics_to_output[f"{Metric.JUDGE_IMAGE_FAITH_TRAIN.value}@{gen_topk}"] = metric_dict["judge"]["train"][key][JudgeMetric.IMAGE_FAITHFULNESS]["asr_universal"]
            metrics_to_output[f"{Metric.JUDGE_IMAGE_FAITH_TEST.value}@{gen_topk}"] = metric_dict["judge"]["test"][key][JudgeMetric.IMAGE_FAITHFULNESS]["asr_universal"]
            metrics_to_output[f"{Metric.JUDGE_ANS_REL_TRAIN.value}@{gen_topk}"] = metric_dict["judge"]["train"][key][JudgeMetric.ANSWER_RELEVANCY]["asr_universal"]
            metrics_to_output[f"{Metric.JUDGE_ANS_REL_TEST.value}@{gen_topk}"] = metric_dict["judge"]["test"][key][JudgeMetric.ANSWER_RELEVANCY]["asr_universal"]

    metric_titles = [m.value for m in metrics_to_show]
    metrics_filtered = {k: v for (k, v) in metrics_to_output.items() if k.split("@")[0].split("+")[0] in metric_titles}
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
            try:
                metric = metrics[f"{metric_name.value}@-1"]
            except KeyError:
                metric = metrics[f"{metric_name.value}@1"]
            label = f"{shorten_model_name(task_config.model_name_embs)} + {shorten_model_name(task_config.vlm.models) if task_config.vlm else ''}"
            plot_dict_x[i][label].append(task_config.max_perturbation * 255)
            plot_dict_y[i][label].append(metric)
    markers = ["o", "v", "s", "x", "+", "^"]
    n_plots = len(plot_dict_x[0].keys())
    n_rows = 2
    n_cols = n_plots // n_rows
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 5), sharex=True, sharey=True)
    for k, label in enumerate(plot_dict_x[0].keys()):
        axis = ax[k % n_rows][k // n_rows]
        for i in range(len(metrics_to_show)):
            axis.plot(plot_dict_x[i][label], plot_dict_y[i][label], label=titles[i], marker=markers[i])
            axis.set_xscale("log")
        axis.grid(visible=True)
        axis.set_xticks(plot_dict_x[0][label], labels=plot_dict_x[0][label])
        axis.set_xlim([0.9, 40])
        axis.set_ylim([-0.05, 1.05])
        axis.set_title(label)
    handles, labels = ax[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.25, 0.6), loc="center right")
    fig.supxlabel("Maximum Allowed Perturbation (/255)")
    fig.supylabel("Attack Success")
    fig.tight_layout()
    return fig

def cleanup_colnames_after_groupby(col_names):
    new_cols = []
    for col in col_names:
        if isinstance(col, tuple):
            col = f'{col[0]} {col[1]}'.strip()
        new_cols.append(col)
    return new_cols

def combine_mean_std_columns(df, aggregate_columns: list[str]):
    # --- Format mean and std columns into a single column ---
    for col in aggregate_columns:
        mean_col = f'{col} mean'
        std_col = f'{col} std'
        new_col = f'{col} (mean ± std)'
        
        # Fill NaN values in the 'std' column before formatting to avoid errors
        df[std_col] = df[std_col].fillna(0)
        
        # Create the new combined column
        df[new_col] = df.apply(
            lambda row: f'{row[mean_col]:.2f} ± {row[std_col]:.2f}',
            axis=1
        )
        
        # Drop the original mean and std columns
        df.drop(columns=[mean_col, std_col], inplace=True)

