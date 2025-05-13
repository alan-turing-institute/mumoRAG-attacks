"""
This file includes different configurations for the parameters of the attack and the multimodal RAG system
"""

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from utils.defence import DefenceName
from config.eval import ExperimentEvalConfig
from config.experiment import ExperimentConfig
from config.train import ExperimentTrainConfig
from wrappers.attack_mask import AttackMask
from wrappers.dataset import DatasetName
from wrappers.embedding import EmbedderName, EmbeddingLoss
from wrappers.judge import JudgeMetric
from wrappers.vlm import VLMName

DEFAULT_EXPERIMENT = "dev"

configstore = ConfigStore.instance()


def load_config(name, config_path="pkg://experiments"):
    with initialize(version_base=None, config_path=config_path):
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
            embedder_list=[EmbedderName.QWEN2_GME_2B],
            vlm_list=[VLMName.SMOLVLM_1_256M],
            gen_topk_list=[1],
        ),
        eval=ExperimentEvalConfig(
            gen_topk_list=[-1],
        ),
    ),
)


"""
Simple Defences
"""
configstore.store(
    name="simple_defences",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=[DatasetName.VIDORE_SYN_AI, DatasetName.VIDORE_V2_ESG],
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14, EmbedderName.SIGLIP2_LARGE_PATCH16, EmbedderName.JINA_CLIP_2, EmbedderName.COLPALI],
            vlm_list=[VLMName.SMOLVLM_1_2B, VLMName.QWEN_2p5_VL_3B, VLMName.INTERNVL_3_2B],
            gen_topk_list=[1],
        ),
        eval=ExperimentEvalConfig(
            gen_topk_list=[-1,1],
            defences_list=[DefenceName.NOISE, DefenceName.PARAPHRASE, DefenceName.NONE],
            noise_defence_level=8.0,
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
            dataset_list=[DatasetName.VIDORE_SYN_AI],
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14, EmbedderName.SIGLIP2_LARGE_PATCH16, EmbedderName.JINA_CLIP_2],
            vlm_list=[VLMName.SMOLVLM_1_2B, VLMName.QWEN_2p5_VL_3B, VLMName.INTERNVL_3_2B],
            attack_mask_list=[AttackMask.Figure, AttackMask.FirstQuadrant],
            n_gradient_steps=1000,
        ),
    ),
)

"""
Targeted attacks against a subset of queries
"""
configstore.store(
    name="targeted_attacks_oneQ_oneA",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=[DatasetName.VIDORE_SYN_AI],
            embedder_list=[
                EmbedderName.CLIP_LARGE_PATCH14,
                EmbedderName.SIGLIP2_LARGE_PATCH16,
                EmbedderName.JINA_CLIP_2,
                EmbedderName.COLPALI,
            ],
            vlm_list=[VLMName.SMOLVLM_1_2B, VLMName.QWEN_2p5_VL_3B, VLMName.INTERNVL_3_2B],
            is_targeted=True,
            target_query_idx=[0],
            n_knn_target_queries=1,
            optimize_nontargeted_queries_list=[True, False],
        ),
        eval=ExperimentEvalConfig(
            gen_topk_list=[-1, 1, 5],
        ),
    ),
)
configstore.store(
    name="targeted_attacks_multiQ_oneA",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=[DatasetName.VIDORE_SYN_AI],
            embedder_list=[
                EmbedderName.CLIP_LARGE_PATCH14,
                EmbedderName.SIGLIP2_LARGE_PATCH16,
                EmbedderName.JINA_CLIP_2,
                EmbedderName.COLPALI,
            ],
            vlm_list=[VLMName.SMOLVLM_1_2B, VLMName.QWEN_2p5_VL_3B, VLMName.INTERNVL_3_2B],
            is_targeted=True,
            target_query_idx=[0],
            n_knn_target_queries=5,
            optimize_nontargeted_queries_list=[True, False],
        ),
        eval=ExperimentEvalConfig(
            gen_topk_list=[-1, 1, 5],
        ),
    ),
)
configstore.store(
    name="targeted_attacks_multiQ_multiA",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=[DatasetName.VIDORE_SYN_AI],
            embedder_list=[
                EmbedderName.CLIP_LARGE_PATCH14,
                EmbedderName.SIGLIP2_LARGE_PATCH16,
                EmbedderName.JINA_CLIP_2,
                EmbedderName.COLPALI,
            ],
            vlm_list=[VLMName.SMOLVLM_1_2B, VLMName.QWEN_2p5_VL_3B, VLMName.INTERNVL_3_2B],
            is_targeted=True,
            target_query_idx=[0, 1],
            target_answer_vlm=[
                "Manually match each marker to a generic human template regardless of trial-specific subject calibration.",
                "A micromort measures the number of accidents per million vehicles on the road and is used in transportation policy.",
            ],
            n_knn_target_queries=1,
            optimize_nontargeted_queries_list=[True, False],
        ),
        eval=ExperimentEvalConfig(
            gen_topk_list=[-1, 1, 5],
        ),
    ),
)

"""
RAG evaluation and attack detection through VLM-as-a-judge
"""
configstore.store(
    name="judge_defence",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=[DatasetName.VIDORE_SYN_AI],
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14, EmbedderName.SIGLIP2_LARGE_PATCH16, EmbedderName.JINA_CLIP_2],
            vlm_list=[VLMName.SMOLVLM_1_2B, VLMName.QWEN_2p5_VL_3B, VLMName.INTERNVL_3_2B],
            lambda_jdg=1,
            judge_list=[VLMName.SMOLVLM_1_2B, VLMName.QWEN_2p5_VL_3B, VLMName.INTERNVL_3_2B],
            train_jdg_metric_list=[JudgeMetric.IMAGE_CONTEXT_RELEVANCY, JudgeMetric.IMAGE_FAITHFULNESS, JudgeMetric.ANSWER_RELEVANCY],
        ),
        eval=ExperimentEvalConfig(
            gen_topk_list=[-1, 1, 5],
            do_judge=True,
            eval_jdg_metric_list=[JudgeMetric.IMAGE_CONTEXT_RELEVANCY, JudgeMetric.IMAGE_FAITHFULNESS, JudgeMetric.ANSWER_RELEVANCY],
        ),
    ),
)

"""
Attack optimized when the malicious image is retrieved within top-k (not top-1)
Evaluation when image is retrieved within top-k (not top-1)
"""
configstore.store(
    name="topk_context",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=[DatasetName.VIDORE_SYN_AI],
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14, EmbedderName.SIGLIP2_LARGE_PATCH16, EmbedderName.JINA_CLIP_2],
            vlm_list=[VLMName.SMOLVLM_1_2B, VLMName.QWEN_2p5_VL_3B, VLMName.INTERNVL_3_2B],
            gen_topk_list=[1, 3, 5],
        ),
        eval=ExperimentEvalConfig(
            gen_topk_list=[-1, 1, 3, 5],
            test_topk_order=True,
            do_judge=True,
            eval_jdg_metric_list=[JudgeMetric.IMAGE_CONTEXT_RELEVANCY, JudgeMetric.IMAGE_FAITHFULNESS, JudgeMetric.ANSWER_RELEVANCY],
        ),
    ),
)

"""
generates data for perturbation plot (full x-axis)
"""
configstore.store(
    name="perturbation_plot",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=[DatasetName.VIDORE_SYN_AI],
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14, EmbedderName.SIGLIP2_LARGE_PATCH16, EmbedderName.JINA_CLIP_2],
            vlm_list=[VLMName.SMOLVLM_1_2B, VLMName.QWEN_2p5_VL_3B, VLMName.INTERNVL_3_2B],
            max_perturbation_list=[x / 255.0 for x in [1, 2, 4, 8, 16, 32, 64, 128, 256]],
        ),
    ),
)

"""
generates data for perturbation plot (full x-axis)
"""
configstore.store(
    name="perturbation_plot_targeted",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=[DatasetName.VIDORE_SYN_AI],
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14, EmbedderName.SIGLIP2_LARGE_PATCH16, EmbedderName.JINA_CLIP_2],
            vlm_list=[VLMName.SMOLVLM_1_2B, VLMName.QWEN_2p5_VL_3B, VLMName.INTERNVL_3_2B],
            max_perturbation_list=[x / 255.0 for x in [1, 2, 4, 8, 16, 32, 64, 128, 256]],
            is_targeted=True,
            target_query_idx=[1],
            n_knn_target_queries=1,
        ),
    ),
)


"""
ColPali ablations (w/ colpali_only_images False)
"""
configstore.store(
    name="copali_ab",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=[DatasetName.VIDORE_SYN_AI],
            embedder_list=[EmbedderName.COLPALI],
            vlm_list=[VLMName.SMOLVLM_1_2B, VLMName.QWEN_2p5_VL_3B, VLMName.INTERNVL_3_2B],
            emb_train_loss_type_list=[EmbeddingLoss.MAXSIM, EmbeddingLoss.AVGSIM, EmbeddingLoss.SOFTMAXSIM, EmbeddingLoss.COS_AVGEMB],
        ),
        eval=ExperimentEvalConfig(
            emb_test_loss_type_list=[EmbeddingLoss.MAXSIM, EmbeddingLoss.AVGSIM, EmbeddingLoss.SOFTMAXSIM, EmbeddingLoss.COS_AVGEMB],
        ),
    ),
)

"""
ColPali ablations (w/ colpali_only_images True)
"""
configstore.store(
    name="copali_ab_cpoiT",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=[DatasetName.VIDORE_SYN_AI],
            embedder_list=[EmbedderName.COLPALI],
            vlm_list=[VLMName.SMOLVLM_1_2B, VLMName.QWEN_2p5_VL_3B, VLMName.INTERNVL_3_2B],
            emb_train_loss_type_list=[EmbeddingLoss.MAXSIM, EmbeddingLoss.AVGSIM, EmbeddingLoss.SOFTMAXSIM, EmbeddingLoss.COS_AVGEMB],
            colpali_only_images=True,
        ),
        eval=ExperimentEvalConfig(
            emb_test_loss_type_list=[EmbeddingLoss.MAXSIM, EmbeddingLoss.AVGSIM, EmbeddingLoss.SOFTMAXSIM, EmbeddingLoss.COS_AVGEMB],
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
            embedder_list=[
                EmbedderName.CLIP_LARGE_PATCH14,
                EmbedderName.SIGLIP2_LARGE_PATCH16,
                EmbedderName.JINA_CLIP_2,
                EmbedderName.COLPALI,
            ],
            vlm_list=[VLMName.SMOLVLM_1_2B, VLMName.QWEN_2p5_VL_3B, VLMName.INTERNVL_3_2B],
        ),
        eval=ExperimentEvalConfig(
            eval_emb_list=[
                EmbedderName.CLIP_LARGE_PATCH14,
                EmbedderName.SIGLIP2_LARGE_PATCH16,
                EmbedderName.JINA_CLIP_2,
                EmbedderName.COLPALI,
            ],
            eval_vlm_list=[VLMName.SMOLVLM_1_2B, VLMName.QWEN_2p5_VL_3B, VLMName.INTERNVL_3_2B],
        ),
    ),
)
