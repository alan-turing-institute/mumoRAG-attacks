"""
This file includes different configurations for the parameters of the attack and the multimodal RAG system
"""

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from config.eval import ExperimentEvalConfig
from config.experiment import ExperimentConfig
from config.train import ExperimentTrainConfig, JudgeConfig, VLMConfig
from utils.defence import DefenceName
from wrappers.attack_mask import AttackMask
from wrappers.dataset import DatasetName
from wrappers.embedding import EmbedderName, EmbeddingLoss
from wrappers.judge import JudgeMetric
from wrappers.vlm import VLMName

DEFAULT_EXPERIMENT = "paper_non_targeted"

configstore = ConfigStore.instance()

datasets = [
    DatasetName.VIDORE_SYN_AI,
    # DatasetName.VIDORE_V2_ESG
]
embedders = [
    EmbedderName.CLIP_LARGE_PATCH14,
    EmbedderName.COLPALI,
    EmbedderName.QWEN2_GME_2B,
    # EmbedderName.SIGLIP2_LARGE_PATCH16
]

eval_embedders = [
    EmbedderName.CLIP_LARGE_PATCH14,
    EmbedderName.COLPALI,
    EmbedderName.QWEN2_GME_2B,
    # EmbedderName.SIGLIP2_LARGE_PATCH16
]

vlms = [
    VLMName.SMOLVLM_1_2B,
    VLMName.QWEN_2p5_VL_3B,
    VLMName.INTERNVL_3_2B
]

embedders_sets =[
    [EmbedderName.CLIP_LARGE_PATCH14, EmbedderName.COLPALI],
    [EmbedderName.COLPALI, EmbedderName.QWEN2_GME_2B],
    [EmbedderName.QWEN2_GME_2B, EmbedderName.CLIP_LARGE_PATCH14],
]

vlms_sets =[
    [VLMName.SMOLVLM_1_2B, VLMName.QWEN_2p5_VL_3B,],
    [VLMName.QWEN_2p5_VL_3B, VLMName.INTERNVL_3_2B],
    [VLMName.SMOLVLM_1_2B, VLMName.INTERNVL_3_2B]
]

chosen_indexes = [150,251,420,676,769]
gpt_chosen_indexes=[0,1,2,3,4]


eval_vlms = [
    VLMName.SMOLVLM_1_2B,
    VLMName.QWEN_2p5_VL_3B,
    VLMName.INTERNVL_3_2B
]


def load_config(name: str, config_path: str = "pkg://experiments"):
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=name, return_hydra_config=True)
        HydraConfig().set_config(cfg)
        cfg = compose(config_name=name)
    return OmegaConf.to_object(cfg)


def get_config_name() -> str:
    try:
        return HydraConfig.get().job.config_name
    except ValueError:
        return "testing"


"""
Development Configuration (default)
"""
configstore.store(
    name="dev",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            embedder_list=[EmbedderName.CLIP_BASE_PATCH16],
            vlm=VLMConfig(models=[VLMName.SMOLVLM_1_256M], gen_topk_list=[1]),
            print_every=2,
            n_gradient_steps=4,
        ),
        eval=ExperimentEvalConfig(
            gen_topk_list=[-1],
        ),
    ),
)

configstore.store(
    name="rebuttal_non_targeted",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=embedders,
            vlm=VLMConfig(models=vlms, lambda_=0),
        ),
        eval=ExperimentEvalConfig(
            eval_emb_list=embedders,
            eval_vlm_list=vlms,
        ),
    ),
)

configstore.store(
    name="rebuttal_targeted_attacks_oneQ_oneA",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=embedders,
            vlm=VLMConfig(
                models=vlms,
                target_answers=["Manually match each marker to a generic human template regardless of trial-specific subject calibration."],
                lambda_=0
            ),
            is_targeted=True,
            target_query_idx=[0],
            n_knn_target_queries=1,
        ),
        eval=ExperimentEvalConfig(
            eval_emb_list=embedders,
            eval_vlm_list=vlms,
        ),
    ),
)

configstore.store(
    name="paper_multi_transferability",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=[embedders],
            vlm=VLMConfig(models=[vlms]),
            chosen_index_list=chosen_indexes,
        ),
        eval=ExperimentEvalConfig(
            eval_emb_list=embedders,
            eval_vlm_list=vlms,
        ),
    ),
)

configstore.store(
    name="paper_leave_one_out_multi_transferability",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=embedders_sets,
            vlm=VLMConfig(models=vlms_sets),
            chosen_index_list=chosen_indexes,
        ),
        eval=ExperimentEvalConfig(
            eval_emb_list=embedders,
            eval_vlm_list=vlms,
        ),
    ),
)

"""
Generic DOS attack.
Produces:
- Table of main results: dataset x embedders x vlm -> metrics.
- Table of transferability of results: embedders x vlm x eval embedders x eval vlm -> subset of metrics.
"""
configstore.store(
    name="paper_multi_transferability_targeted",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=[embedders],
            vlm=VLMConfig(
                models=[vlms],
                target_answers=["Manually match each marker to a generic human template regardless of trial-specific subject calibration."],
            ),
            is_targeted=True,
            target_query_idx=[0],
            n_knn_target_queries=1,
            chosen_index_list=chosen_indexes,
        ),
        eval=ExperimentEvalConfig(
            eval_emb_list=embedders,
            eval_vlm_list=vlms,
        ),
    ),
)

configstore.store(
    name="paper_leave_one_out_multi_transferability_targeted",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=embedders_sets,
            vlm=VLMConfig(
                models=vlms_sets,
                target_answers=["Manually match each marker to a generic human template regardless of trial-specific subject calibration."],
            ),
            is_targeted=True,
            target_query_idx=[0],
            n_knn_target_queries=1,
            chosen_index_list=chosen_indexes,
        ),
        eval=ExperimentEvalConfig(
            eval_emb_list=embedders,
            eval_vlm_list=vlms,
        ),
    ),
)

configstore.store(
    name="paper_non_targeted",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=embedders,
            vlm=VLMConfig(models=vlms),
	    chosen_index_list=chosen_indexes,
        ),
        eval=ExperimentEvalConfig(
            eval_emb_list=eval_embedders,
            eval_vlm_list=vlms,
        ),
    ),
)

configstore.store(
    name="paper_universal_GT_baseline",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=embedders,
            vlm=VLMConfig(models=vlms),
	    chosen_index_list=chosen_indexes,
        n_gradient_steps=0,
        ),
    ),
)

configstore.store(
    name="paper_targeted_1-1_neg_baseline",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=embedders,
            vlm=VLMConfig(
                models=vlms,
                target_answers=[
                    "Manually match each marker to a generic human template regardless of trial-specific subject calibration."],
            ),
            is_targeted=True,
            target_query_idx=[0],
            chosen_index_list=chosen_indexes,
            n_knn_target_queries=1,
            n_gradient_steps=0,
        )
    ),
)

configstore.store(
    name="paper_targeted_m-1_neg_baseline",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=embedders,
            vlm=VLMConfig(models=vlms),
            is_targeted=True,
            target_query_idx=[0],
            n_knn_target_queries=5,
	        chosen_index_list=chosen_indexes,
            n_gradient_steps=0,
        ),
    ),
)

configstore.store(
    name="paper_targeted_m-m_neg_baseline",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=embedders,
            vlm=VLMConfig(
                models=vlms,
                target_answers=[
                    "Manually match each marker to a generic human template regardless of trial-specific subject calibration.",
                    "A micromort measures the number of accidents per million vehicles on the road and is used in transportation policy.",
                ],
            ),
            is_targeted=True,
            target_query_idx=[0, 1],
            n_knn_target_queries=1,
	        chosen_index_list=chosen_indexes,
            n_gradient_steps=0,
        ),
    ),
)

"""
Targeted attacks against a specific query.
Produces:
- Table of main results: dataset x embedders x vlm -> metrics.
- Table of transferability of results: embedders x vlm x eval embedders x eval vlm -> subset of metrics.
"""
configstore.store(
    name="paper_targeted_attacks_oneQ_oneA",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=embedders,
            vlm=VLMConfig(
                models=vlms,
                target_answers=["Manually match each marker to a generic human template regardless of trial-specific subject calibration."],
            ),
            is_targeted=True,
            target_query_idx=[0],
            chosen_index_list=chosen_indexes,
            n_knn_target_queries=1,
        ),
        eval=ExperimentEvalConfig(
            eval_emb_list=eval_embedders,
            eval_vlm_list=vlms,
        ),
    ),
)

"""
Targeted attacks against a subset of neighbouring queries.
Produces:
- Table of main results: dataset x embedders x vlm -> metrics.
"""
configstore.store(
    name="paper_targeted_attacks_multiQ_oneA",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=embedders,
            vlm=VLMConfig(models=vlms),
            is_targeted=True,
            target_query_idx=[0],
            n_knn_target_queries=5,
	        chosen_index_list=chosen_indexes,
        ),
    ),
)

"""
Targeted attacks against a subset of queries with specific answers.
Produces:
- Table of main results: dataset x embedders x vlm -> metrics.
"""
configstore.store(
    name="paper_targeted_attacks_multiQ_multiA",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=embedders,
            vlm=VLMConfig(
                models=vlms,
                target_answers=[
                    "Manually match each marker to a generic human template regardless of trial-specific subject calibration.",
                    "A micromort measures the number of accidents per million vehicles on the road and is used in transportation policy.",
                ],
            ),
            is_targeted=True,
            target_query_idx=[0, 1],
            n_knn_target_queries=1,
	        chosen_index_list=chosen_indexes,
        ),
    ),
)

"""
RAG evaluation and attack detection through VLM-as-a-judge.
Produces:
- Table of judge VLMS -> judge metrics.
"""
configstore.store(
    name="paper_judge_defence",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=embedders,
            vlm=VLMConfig(models=vlms),
            chosen_index_list=chosen_indexes,
        ),
        eval=ExperimentEvalConfig(
            do_judge=True,
            eval_jdg_list=eval_vlms,
            eval_jdg_metric_list=[
                JudgeMetric.IMAGE_CONTEXT_RELEVANCY,
                JudgeMetric.IMAGE_FAITHFULNESS,
                JudgeMetric.ANSWER_RELEVANCY,
            ],
        ),
    ),
)

"""
Adaptive attack with VLM-as-a-judge in objectives.
Produces:
- Table of judge VLMs (same for both train and eval)-> judge metrics.
"""
configstore.store(
    name="paper_judge_defence_adapt",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=embedders,
            vlm=VLMConfig(models=vlms),
            judge=JudgeConfig(
                lambda_=1,
                models=eval_vlms,
                metrics=[
                    JudgeMetric.IMAGE_CONTEXT_RELEVANCY,
                    JudgeMetric.IMAGE_FAITHFULNESS,
                    JudgeMetric.ANSWER_RELEVANCY,
                ],
            ),
            chosen_index_list=chosen_indexes,
        ),
        eval=ExperimentEvalConfig(
            do_judge=True,
            eval_jdg_list=eval_vlms,
            eval_jdg_metric_list=[
                JudgeMetric.IMAGE_CONTEXT_RELEVANCY,
                JudgeMetric.IMAGE_FAITHFULNESS,
                JudgeMetric.ANSWER_RELEVANCY,
            ],
        ),
    ),
)

"""
RAG evaluation and attack detection through VLM-as-a-judge with specific query.
Produces:
- Table of judge VLMs -> judge metrics.
"""
configstore.store(
    name="paper_judge_defence_targeted",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=embedders,
            vlm=VLMConfig(models=vlms),
            is_targeted=True,
            target_query_idx=[0],
            n_knn_target_queries=1,
            chosen_index_list=chosen_indexes,
        ),
        eval=ExperimentEvalConfig(
            do_judge=True,
            eval_jdg_list=eval_vlms,
            eval_jdg_metric_list=[
                JudgeMetric.IMAGE_CONTEXT_RELEVANCY,
                JudgeMetric.IMAGE_FAITHFULNESS,
                JudgeMetric.ANSWER_RELEVANCY,
            ],
        ),
    ),
)

"""
RAG evaluation and attack detection through VLM-as-a-judge with specific query adaptive.
Produces:
- Table of judge VLMs (same for eval and train) -> judge metrics.
"""
configstore.store(
    name="paper_judge_defence_targeted_adapt",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=embedders,
            vlm=VLMConfig(models=vlms),
            is_targeted=True,
            target_query_idx=[0],
            n_knn_target_queries=1,
            judge=JudgeConfig(
                lambda_=1,
                models=vlms,
                metrics=[
                    JudgeMetric.IMAGE_CONTEXT_RELEVANCY,
                    JudgeMetric.IMAGE_FAITHFULNESS,
                    JudgeMetric.ANSWER_RELEVANCY,
                ],
            ),
            chosen_index_list=chosen_indexes,
        ),
        eval=ExperimentEvalConfig(
            do_judge=True,
            eval_jdg_list=vlms,
            eval_jdg_metric_list=[
                JudgeMetric.IMAGE_CONTEXT_RELEVANCY,
                JudgeMetric.IMAGE_FAITHFULNESS,
                JudgeMetric.ANSWER_RELEVANCY,
            ],
        ),
    ),
)

"""
Varies max perturbation to measure effect on attack success.
Produces:
- Graphs for embedder x vlm with perturbation on x-axis
"""
configstore.store(
    name="paper_perturbation_plot",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=embedders,
            vlm=VLMConfig(models=vlms),
            max_perturbation_list=[x / 255.0 for x in [1, 2, 4, 8, 16, 32]],
        ),
    ),
)


"""
ColPali ablations (w/ colpali_only_images False)
Produces:
- (combined (w/ colpali_only_images True)) table emb_loss_train x emb_loss_eval x only_images
"""
configstore.store(
    name="paper_copali_ab",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=[EmbedderName.COLPALI],
            vlm=VLMConfig(models=[VLMName.SMOLVLM_1_2B]),
            emb_train_loss_type_list=[
                EmbeddingLoss.MAXSIM,
                EmbeddingLoss.AVGSIM,
                EmbeddingLoss.SOFTMAXSIM,
                EmbeddingLoss.COS_AVGEMB,
            ],
        ),
        eval=ExperimentEvalConfig(
            emb_test_loss_type_list=[
                EmbeddingLoss.MAXSIM,
                EmbeddingLoss.AVGSIM,
                EmbeddingLoss.SOFTMAXSIM,
                EmbeddingLoss.COS_AVGEMB,
            ],
        ),
    ),
)

"""
ColPali ablations (w/ colpali_only_images True)
"""
configstore.store(
    name="paper_copali_ab_cpoiT",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=[EmbedderName.COLPALI],
            vlm=VLMConfig(models=[VLMName.SMOLVLM_1_2B]),
            emb_train_loss_type_list=[
                EmbeddingLoss.MAXSIM,
                EmbeddingLoss.AVGSIM,
                EmbeddingLoss.SOFTMAXSIM,
                EmbeddingLoss.COS_AVGEMB,
            ],
            colpali_only_images=True,
        ),
        eval=ExperimentEvalConfig(
            emb_test_loss_type_list=[
                EmbeddingLoss.MAXSIM,
                EmbeddingLoss.AVGSIM,
                EmbeddingLoss.SOFTMAXSIM,
                EmbeddingLoss.COS_AVGEMB,
            ],
        ),
    ),
)

"""
Attack optimized when the malicious image is retrieved within top-k (not top-1)
Evaluation when image is retrieved within top-k (not top-1)
"""
configstore.store(
    name="paper_topk_context",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=embedders,
            vlm=VLMConfig(models=[VLMName.SMOLVLM_1_2B], gen_topk_list=[1, 5]),
            chosen_index_list=chosen_indexes,
        ),
        eval=ExperimentEvalConfig(
            gen_topk_list=[-1, 1, 5],
            test_topk_order=False,
        ),
    ),
)

"""
Attack optimized when the malicious image is retrieved within top-k (not top-1)
Evaluation when image is retrieved within top-k (not top-1)
"""
# configstore.store(
#     name="paper_topk_context_targeted",
#     node=ExperimentConfig(
#         train=ExperimentTrainConfig(
#             dataset_list=datasets,
#             embedder_list=[VLMName.SMOLVLM_1_2B],
#             vlm=VLMConfig(models=vlms, gen_topk_list=[
#                 1,
#                 5
#             ]),
#             is_targeted=True,
#             target_query_idx=[0],
#             n_knn_target_queries=1,
#             chosen_index_list=chosen_indexes,
#         ),
#         eval=ExperimentEvalConfig(
#             gen_topk_list=[-1, 1, 5],
#             test_topk_order=False,
#         ),
#     ),
# )

"""
RAG evaluation and attack detection through VLM-as-a-judge
"""
configstore.store(
    name="paper_combined_defence",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14],
            vlm=VLMConfig(models=[VLMName.SMOLVLM_1_2B], gen_topk_list=[1, 5]),
        ),
        eval=ExperimentEvalConfig(
            do_judge=True,
            eval_jdg_list=vlms,
            eval_jdg_metric_list=[
                JudgeMetric.IMAGE_CONTEXT_RELEVANCY,
                JudgeMetric.IMAGE_FAITHFULNESS,
                JudgeMetric.ANSWER_RELEVANCY,
            ],
            gen_topk_list=[-1, 1, 5],
            test_topk_order=False,
        ),
    ),
)
configstore.store(
    name="paper_combined_defence_adapt",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14],
            vlm=VLMConfig(models=[VLMName.SMOLVLM_1_2B], gen_topk_list=[1, 5]),
            judge=JudgeConfig(
                lambda_=1,
                models=vlms,
                metrics=[
                    JudgeMetric.IMAGE_CONTEXT_RELEVANCY,
                    JudgeMetric.IMAGE_FAITHFULNESS,
                    JudgeMetric.ANSWER_RELEVANCY,
                ],
            ),
        ),
        eval=ExperimentEvalConfig(
            do_judge=True,
            eval_jdg_list=vlms,
            eval_jdg_metric_list=[
                JudgeMetric.IMAGE_CONTEXT_RELEVANCY,
                JudgeMetric.IMAGE_FAITHFULNESS,
                JudgeMetric.ANSWER_RELEVANCY,
            ],
            gen_topk_list=[-1, 1, 5],
            test_topk_order=False,
        ),
    ),
)
configstore.store(
    name="paper_combined_defence_targeted",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14],
            vlm=VLMConfig(models=[VLMName.SMOLVLM_1_2B], gen_topk_list=[1, 5]),
            is_targeted=True,
            target_query_idx=[0],
            n_knn_target_queries=1,
            # optimize_nontargeted_queries_list=[True,False],
        ),
        eval=ExperimentEvalConfig(
            do_judge=True,
            eval_jdg_list=vlms,
            eval_jdg_metric_list=[
                JudgeMetric.IMAGE_CONTEXT_RELEVANCY,
                JudgeMetric.IMAGE_FAITHFULNESS,
                JudgeMetric.ANSWER_RELEVANCY,
            ],
            gen_topk_list=[-1, 1, 5],
            test_topk_order=False,
        ),
    ),
)
configstore.store(
    name="paper_combined_defence_targeted_adapt",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14],
            vlm=VLMConfig(models=[VLMName.SMOLVLM_1_2B], gen_topk_list=[1, 5]),
            is_targeted=True,
            target_query_idx=[0],
            n_knn_target_queries=1,
            # optimize_nontargeted_queries_list=[True,False],
            judge=JudgeConfig(
                lambda_=1,
                models=vlms,
                metrics=[
                    JudgeMetric.IMAGE_CONTEXT_RELEVANCY,
                    JudgeMetric.IMAGE_FAITHFULNESS,
                    JudgeMetric.ANSWER_RELEVANCY,
                ],
            ),
        ),
        eval=ExperimentEvalConfig(
            do_judge=True,
            eval_jdg_list=vlms,
            eval_jdg_metric_list=[
                JudgeMetric.IMAGE_CONTEXT_RELEVANCY,
                JudgeMetric.IMAGE_FAITHFULNESS,
                JudgeMetric.ANSWER_RELEVANCY,
            ],
            gen_topk_list=[-1, 1, 5],
            test_topk_order=False,
        ),
    ),
)

"""
Attack optimized when the malicious image is retrieved within top-k (not top-1)
Evaluation when image is retrieved within top-k (not top-1)
"""
configstore.store(
    name="paper_topk_context_targeted",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=embedders,
            vlm=VLMConfig(models=[VLMName.SMOLVLM_1_2B], gen_topk_list=[1, 5]),
            is_targeted=True,
            target_query_idx=[0],
            n_knn_target_queries=1,
            chosen_index_list=chosen_indexes,
        ),
        eval=ExperimentEvalConfig(
            gen_topk_list=[-1, 1, 5],
            test_topk_order=False,
        ),
    ),
)

"""
RAG evaluation and attack detection through VLM-as-a-judge
"""
configstore.store(
    name="paper_combined_defence",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14],
            vlm=VLMConfig(models=[VLMName.SMOLVLM_1_2B], gen_topk_list=[1, 5]),
        ),
        eval=ExperimentEvalConfig(
            do_judge=True,
            eval_jdg_list=vlms,
            eval_jdg_metric_list=[
                JudgeMetric.IMAGE_CONTEXT_RELEVANCY,
                JudgeMetric.IMAGE_FAITHFULNESS,
                JudgeMetric.ANSWER_RELEVANCY,
            ],
            gen_topk_list=[-1, 1, 5],
            test_topk_order=False,
        ),
    ),
)
configstore.store(
    name="paper_combined_defence_adapt",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14],
            vlm=VLMConfig(models=[VLMName.SMOLVLM_1_2B], gen_topk_list=[1, 5]),
            judge=JudgeConfig(
                lambda_=1,
                models=vlms,
                metrics=[
                    JudgeMetric.IMAGE_CONTEXT_RELEVANCY,
                    JudgeMetric.IMAGE_FAITHFULNESS,
                    JudgeMetric.ANSWER_RELEVANCY,
                ],
            ),
        ),
        eval=ExperimentEvalConfig(
            do_judge=True,
            eval_jdg_list=vlms,
            eval_jdg_metric_list=[
                JudgeMetric.IMAGE_CONTEXT_RELEVANCY,
                JudgeMetric.IMAGE_FAITHFULNESS,
                JudgeMetric.ANSWER_RELEVANCY,
            ],
            gen_topk_list=[-1, 1, 5],
            test_topk_order=False,
        ),
    ),
)
configstore.store(
    name="paper_combined_defence_targeted",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14],
            vlm=VLMConfig(models=[VLMName.SMOLVLM_1_2B], gen_topk_list=[1, 5]),
            is_targeted=True,
            target_query_idx=[0],
            n_knn_target_queries=1,
            # optimize_nontargeted_queries_list=[True,False],
        ),
        eval=ExperimentEvalConfig(
            do_judge=True,
            eval_jdg_list=vlms,
            eval_jdg_metric_list=[
                JudgeMetric.IMAGE_CONTEXT_RELEVANCY,
                JudgeMetric.IMAGE_FAITHFULNESS,
                JudgeMetric.ANSWER_RELEVANCY,
            ],
            gen_topk_list=[-1, 1, 5],
            test_topk_order=False,
        ),
    ),
)
configstore.store(
    name="paper_combined_defence_targeted_adapt",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14],
            vlm=VLMConfig(models=[VLMName.SMOLVLM_1_2B], gen_topk_list=[1, 5]),
            is_targeted=True,
            target_query_idx=[0],
            n_knn_target_queries=1,
            # optimize_nontargeted_queries_list=[True,False],
            judge=JudgeConfig(
                lambda_=1,
                models=vlms,
                metrics=[
                    JudgeMetric.IMAGE_CONTEXT_RELEVANCY,
                    JudgeMetric.IMAGE_FAITHFULNESS,
                    JudgeMetric.ANSWER_RELEVANCY,
                ],
            ),
        ),
        eval=ExperimentEvalConfig(
            do_judge=True,
            eval_jdg_list=vlms,
            eval_jdg_metric_list=[
                JudgeMetric.IMAGE_CONTEXT_RELEVANCY,
                JudgeMetric.IMAGE_FAITHFULNESS,
                JudgeMetric.ANSWER_RELEVANCY,
            ],
            gen_topk_list=[-1, 1, 5],
            test_topk_order=False,
        ),
    ),
)

configstore.store(
    name="paper_defences",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=embedders,
            vlm=VLMConfig(models=vlms),
            chosen_index_list=chosen_indexes,
        ),
        eval=ExperimentEvalConfig(defences_list=[DefenceName.PARAPHRASE]),
    ),
)

configstore.store(
    name="paper_targeted_defences",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=embedders,
            vlm=VLMConfig(models=vlms),
            is_targeted=True,
            target_query_idx=[0],
            n_knn_target_queries=1,
            chosen_index_list=chosen_indexes,
        ),
        eval=ExperimentEvalConfig(defences_list=[DefenceName.PARAPHRASE]),
    ),
)

"""
generates data for perturbation plot (full x-axis)
"""
configstore.store(
    name="perturbation_plot_targeted",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=embedders,
            vlm=VLMConfig(models=vlms),
            max_perturbation_list=[x / 255.0 for x in [1, 2, 4, 8, 16, 32, 64, 128, 256]],
            is_targeted=True,
            target_query_idx=[1],
            n_knn_target_queries=1,
        ),
    ),
)

"""
Performs attack constrained
"""
configstore.store(
    name="mask_attack",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=embedders,
            vlm=VLMConfig(models=vlms),
            attack_mask_list=[AttackMask.Figure, AttackMask.FirstQuadrant],
            n_gradient_steps=1000,
        ),
    ),
)

"""
Configuration  of figure 4
Big Table: transferability between models (12 x 12 table, diagonals are white-box attacks)
- should produce 8 x 8 images
- figure shows following metrics: recall_before, recall_after (avg train+test), retrieval_asr_train, retrieval_asr_test, generation_asr_train, generation_asr_test    
"""
configstore.store(
    name="transferability",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=[DatasetName.VIDORE_SYN_AI, DatasetName.VIDORE_V2_ESG],
            embedder_list=embedders + [EmbedderName.COLPALI],
            vlm=VLMConfig(models=vlms),
        ),
        eval=ExperimentEvalConfig(
            eval_emb_list=embedders + [EmbedderName.COLPALI],
            eval_vlm_list=eval_vlms,
        ),
    ),
)

"""
ChatGPT baselines for comparison
"""
# configstore.store(
#     name="paper_GPT_non_targeted",
#     node=ExperimentConfig(
#         train=ExperimentTrainConfig(
#             dataset_list=datasets,
#             embedder_list=[EmbedderName.CLIP_LARGE_PATCH14],
#             vlm=VLMConfig(models=[VLMName.SMOLVLM_1_2B]),
#         ),
#         eval=ExperimentEvalConfig(
#             gen_topk_list=[-1],
#             test_gpt_attack=True,
#         ),
#     ),
# )

configstore.store(
    name="paper_GPT_non_targeted",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14],
            vlm=VLMConfig(models=[VLMName.SMOLVLM_1_2B]),
            chosen_index_list=gpt_chosen_indexes,
        ),
        eval=ExperimentEvalConfig(
            eval_emb_list=embedders,
            eval_vlm_list=vlms,
            test_generative_attacks=[
                "gpt",
                # "gemini"
            ],
        ),
    ),
)

configstore.store(
    name="paper_GPT_targeted_attacks_oneQ_oneA",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14],
            vlm=VLMConfig(
                models=[VLMName.SMOLVLM_1_2B],
                target_answers=["Manually match each marker to a generic human template regardless of trial-specific subject calibration."],
            ),
            chosen_index_list=gpt_chosen_indexes,
            is_targeted=True,
            target_query_idx=[0],
            n_knn_target_queries=1,
        ),
        eval=ExperimentEvalConfig(
            eval_emb_list=embedders,
            eval_vlm_list=vlms,
            test_generative_attacks=[
                "gpt",
                # "gemini"
            ],
        ),
    ),
)

configstore.store(
    name="paper_GPT_targeted_attacks_multiQ_oneA",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14],
            vlm=VLMConfig(models=[VLMName.SMOLVLM_1_2B]),
            chosen_index_list=gpt_chosen_indexes,
            is_targeted=True,
            target_query_idx=[0],
            n_knn_target_queries=5,
        ),
        eval=ExperimentEvalConfig(
            eval_emb_list=embedders,
            eval_vlm_list=vlms,
            test_generative_attacks=[
                "gpt",
                # "gemini"
            ],

        ),
    ),
)

configstore.store(
    name="paper_GPT_targeted_attacks_multiQ_multiA",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=datasets,
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14],
            vlm=VLMConfig(
                models=[VLMName.SMOLVLM_1_2B],
                target_answers=[
                    "Manually match each marker to a generic human template regardless of trial-specific subject calibration.",
                    "A micromort measures the number of accidents per million vehicles on the road and is used in transportation policy.",
                ],
            ),
            chosen_index_list=gpt_chosen_indexes,
            is_targeted=True,
            target_query_idx=[0, 1],
            n_knn_target_queries=1,
        ),
        eval=ExperimentEvalConfig(
            eval_emb_list=embedders,
            eval_vlm_list=vlms,
            test_generative_attacks=[
                "gpt",
                # "gemini"
            ],
        ),
    ),
)
