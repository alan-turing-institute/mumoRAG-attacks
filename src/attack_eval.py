import json
from pprint import pformat
from typing import Any

import hydra
import torchvision.transforms.v2 as T
from omegaconf import OmegaConf

from config.eval import ExperimentEvalConfig
from config.experiment import ExperimentConfig
from config.task import get_transferability_file_suffix, generate_task_configs
from experiments import DEFAULT_EXPERIMENT
from experiments.configstore import get_config_name
from utils.attack import get_all_target_queries_and_answers
from utils.image_utils import load_adv_image
from utils.logger import logger
from utils.utils import get_device
from wrappers.cache import get_vlm, get_text_embedder, get_dataset, get_embedded_dataset, get_judge
from wrappers.embedding import is_loss_compatible
from wrappers.json_encoder import EnumEncoder
from wrappers.vlm import VLMEvaluationMetric


def get_retrieval_saved_info(exp_config_eval: ExperimentEvalConfig, metric_dict_before, metric_dict_after):
    retrieval_dict: dict[str, Any] = {
        "topk_list": exp_config_eval.topk_list,
        "eval_emb_loss": exp_config_eval.emb_test_loss_type_list,
    }

    for k in metric_dict_before.keys():
        retrieval_dict[k] = {
            "recall_before": metric_dict_before[k]["acc"],
            "recall_after": metric_dict_after[k]["acc"],
            "asr_train": metric_dict_after[k]["asr_train"],
            "asr_test": metric_dict_after[k]["asr_test"],
            "asr_targeted": metric_dict_after[k].get("asr_targeted", -1),
            "fpr_targeted_train": metric_dict_after[k].get("fpr_targeted_train", -1),
            "fpr_targeted_test": metric_dict_after[k].get("fpr_targeted_test", -1),
        }
    return retrieval_dict


def run(exp_config: ExperimentConfig):
    device = get_device(prefer_mps=True)
    task_configs = generate_task_configs(exp_config, include_eval=True)
    n_evals = len(task_configs)

    # first we just make sure that all required files are on disk, so that we don't waste time
    # this will raise an error if there are missing file(s)
    for task_config in task_configs:
        _ = load_adv_image(task_config, exp_config.train)

    # load text embedding model in case we need it for evaluation
    if VLMEvaluationMetric.EMBED_ADV in exp_config.eval.gen_metric_list or VLMEvaluationMetric.EMBED_GT in exp_config.eval.gen_text_embedder:
        text_embedder = get_text_embedder(exp_config.eval.gen_text_embedder, device=device)
    else:
        get_text_embedder.cache_clear()
        text_embedder = None

    for i, task_config in enumerate(task_configs):

        logger.info(f"Eval {(i + 1):4d}/{n_evals}, task_config -> {pformat(task_config.to_dict(), indent=4)}\n{'=' * 20}")
        image_adv = load_adv_image(task_config, exp_config.train)

        # update model names in case we test transferability
        model_name_emb = task_config.eval_emb_name if task_config.eval_emb_name else task_config.model_name_emb
        model_name_vlm = task_config.eval_vlm_name if task_config.eval_vlm_name else task_config.model_name_vlm
        model_name_jdg = task_config.eval_jdg_name if task_config.eval_jdg_name else task_config.model_name_jdg

        vlm = get_vlm(model_name_vlm, device)
        ds = get_dataset(task_config.ds_name)

        # update target queries and answers in case the attack is targeted
        all_target_query_idx, all_adv_target_answers_vlm, all_answers_vlm = get_all_target_queries_and_answers(
            exp_config.train.is_targeted,
            exp_config.train.target_query_idx,
            exp_config.train.target_answer_vlm,
            ds.queries,
            exp_config.train.n_knn_target_queries,
            ds.ground_truth_answers,
            model_name_emb,
            task_config.emb_train_loss_type,
            device,
        )

        retrievals_train, retrievals_test = None, None
        retrieval_metric_dict = None
        # test retrieval
        if exp_config.eval.do_retrieval:
            logger.info("=== Evaluating retrieval ...")
            embedded_ds = get_embedded_dataset(
                dataset=ds,
                model_name_emb=model_name_emb,
                quantize=False,
                colpali_only_images=exp_config.train.colpali_only_images,
                device=device,
            )
            embedded_ds.add_adv_image(T.ToPILImage()(image_adv / 255))
            # remove incompatible losses
            emb_test_loss_type_list_compatible = [loss for loss in exp_config.eval.emb_test_loss_type_list if is_loss_compatible(model_name_emb, loss)]
            metric_dict_before, retrievals_before = embedded_ds.evaluate_retrieval(
                ks=exp_config.eval.topk_list,
                loss_types=emb_test_loss_type_list_compatible,
                is_targeted=exp_config.train.is_targeted,
                target_query_idx=all_target_query_idx,
                include_adv=False,
            )
            metric_dict_after, retrievals_after = embedded_ds.evaluate_retrieval(
                ks=exp_config.eval.topk_list,
                loss_types=emb_test_loss_type_list_compatible,
                is_targeted=exp_config.train.is_targeted,
                target_query_idx=all_target_query_idx,
                include_adv=True,
            )
            retrieval_metric_dict = get_retrieval_saved_info(
                exp_config.eval,
                metric_dict_before,
                metric_dict_after,
            )
            retrievals_train = {k: v["train"] for k, v in retrievals_after.items()}
            retrievals_test = {k: v["test"] for k, v in retrievals_after.items()}

        generation_metric_dict = None
        judge_metric_dict = None

        # test generation
        if exp_config.eval.do_generation:
            logger.info("=== Evaluating generation ...")
            target_answer_vlm_train, target_answer_vlm_test = ds.split_train_test(all_answers_vlm)
            metric_vlm_dict_test, gs_vlm_dict_test = ds.evaluate_generation(
                vlm,
                image_adv,
                target_generation=target_answer_vlm_test,
                adv_target_generations=all_adv_target_answers_vlm,
                metrics=exp_config.eval.gen_metric_list,
                text_embedder=text_embedder,
                retrievals=retrievals_test,
                generation_topk_list=exp_config.eval.gen_topk_list,
                test_topk_order=exp_config.eval.test_topk_order,
                is_targeted=exp_config.train.is_targeted,
                target_query_idx=all_target_query_idx,
                batch_size=exp_config.eval.gen_batch_size,
                eval_train=False,
            )
            metric_vlm_dict_train, gs_vlm_dict_train = ds.evaluate_generation(
                vlm,
                image_adv,
                target_generation=target_answer_vlm_train,
                adv_target_generations=all_adv_target_answers_vlm,
                metrics=exp_config.eval.gen_metric_list,
                text_embedder=text_embedder,
                retrievals=retrievals_train,
                generation_topk_list=exp_config.eval.gen_topk_list,
                test_topk_order=exp_config.eval.test_topk_order,
                is_targeted=exp_config.train.is_targeted,
                target_query_idx=all_target_query_idx,
                batch_size=exp_config.eval.gen_batch_size,
                eval_train=True,
            )
            generation_metric_dict = {
                "train": metric_vlm_dict_train,
                "test": metric_vlm_dict_test,
            }

            if exp_config.eval.do_judge:
                judge = get_judge(model_name_jdg, device)
                logger.info("=== Evaluating using Judge ...")

                metric_jdg_dict_test, generation_jdg_dict_test = ds.evaluate_using_judge(
                    judge,
                    image_adv,
                    judge_metrics=exp_config.eval.eval_jdg_metric_list,
                    retrievals=retrievals_test,
                    generation_vlm_dict=gs_vlm_dict_test,
                    generation_topk_list=exp_config.eval.gen_topk_list,
                    test_topk_order=exp_config.eval.test_topk_order,
                    is_targeted=exp_config.train.is_targeted,
                    target_query_idx=all_target_query_idx,
                    batch_size=exp_config.eval.gen_batch_size,
                    eval_train=False,
                )
                metric_jdg_dict_train, generation_jdg_dict_train = ds.evaluate_using_judge(
                    judge,
                    image_adv,
                    judge_metrics=exp_config.eval.eval_jdg_metric_list,
                    retrievals=retrievals_train,
                    generation_vlm_dict=gs_vlm_dict_train,
                    generation_topk_list=exp_config.eval.gen_topk_list,
                    test_topk_order=exp_config.eval.test_topk_order,
                    is_targeted=exp_config.train.is_targeted,
                    target_query_idx=all_target_query_idx,
                    batch_size=exp_config.eval.gen_batch_size,
                    eval_train=True,
                )
                judge_metric_dict = {
                    "train": metric_jdg_dict_train,
                    "test": metric_jdg_dict_test,
                }

        # save results to JSON format
        metric_dict_full = {
            "retrieval": retrieval_metric_dict,
            "generation": generation_metric_dict,
            "judge": judge_metric_dict,
            "attack_config": task_config.to_dict(),
        }

        results_filename = (
            exp_config.eval.results_folder
            / f"metrics_{get_config_name()}_{task_config.create_hash_string()}{get_transferability_file_suffix(task_config.eval_emb_name, task_config.eval_vlm_name, task_config.eval_jdg_name)}.json"
        )

        with open(
            results_filename,
            "w",
        ) as file:
            json.dump(metric_dict_full, file, indent=4, cls=EnumEncoder)
        logger.info(f"Saved results to {results_filename}")


@hydra.main(version_base=None, config_path="pkg://experiments", config_name=DEFAULT_EXPERIMENT)
def main(exp_config: ExperimentConfig):
    run(OmegaConf.to_object(exp_config))


if __name__ == "__main__":
    main()
