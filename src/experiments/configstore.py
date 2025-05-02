"""
This file includes different configurations for the parameters of the attack and the multimodal RAG system
"""

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from config.eval import ExperimentEvalConfig
from config.experiment import ExperimentConfig
from config.train import ExperimentTrainConfig
from wrappers.embedding import EmbedderName, EmbeddingLoss
from wrappers.judge import JudgeMetric
from wrappers.vlm import VLMName
from wrappers.dataset import DatasetName


DEFAULT_EXPERIMENT = "testing"

configstore = ConfigStore.instance()


def load_config(name):
    with initialize(version_base=None, config_path="pkg://experiments"):
        cfg = compose(config_name=name)
    return OmegaConf.to_object(cfg)

"""
testing Configuration
"""
configstore.store(
    name="testing",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            embedder_list=[EmbedderName.CLIP_BASE_PATCH16],
            vlm_list=[VLMName.SMOLVLM_1_256M],
            gen_topk_list=[1],
            is_targeted=True,
            target_query_idx=[2],
            n_knn_target_queries=5,
            print_every=2,
            n_gradient_steps=25,
            optimize_nontargeted_queries_list=[True,False]
        ),
        eval=ExperimentEvalConfig(
            gen_topk_list=[-1],
        )
    ),
)


"""
Experiment to test the effect of the order of the malicious image on the generation ASR
"""
configstore.store(
    name="order_in_topk",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14],
            vlm_list=[VLMName.SMOLVLM_1_2B],
            gen_topk_list=[1],
        ),
        eval=ExperimentEvalConfig(
            gen_topk_list=[5],
            test_topk_order=True,
            eval_jdg_metric_list=[JudgeMetric.IMAGE_CONTEXT_RELEVANCY, JudgeMetric.IMAGE_FAITHFULNESS, JudgeMetric.ANSWER_RELEVANCY],
            do_judge=False,
        )
    ),
)

"""
Targeted attacks against a subset of queries
"""
configstore.store(
    name="targeted_attacks",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14, EmbedderName.JINA_CLIP_2, EmbedderName.COLPALI],
            vlm_list=[VLMName.SMOLVLM_1_2B, VLMName.QWEN_2p5_VL_3B],
            is_targeted=True,
            target_query_idx=[2,5],
            target_answer_vlm=["Question two me no likey", "Five is not my lucky number"],
            n_knn_target_queries=1,
        ),
        eval=ExperimentEvalConfig(
            gen_topk_list=[-1,1,5],
        )
    ),
)

"""
RAG evaluation and attack detection through VLM-as-a-judge
"""
configstore.store(
    name="judge_defence",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14, EmbedderName.JINA_CLIP_2, EmbedderName.COLPALI],
            vlm_list=[VLMName.SMOLVLM_1_2B, VLMName.QWEN_2p5_VL_3B],
        ),
        eval=ExperimentEvalConfig(
            gen_topk_list=[-1,1,5],
            do_judge=True,
            eval_jdg_metric_list=[JudgeMetric.IMAGE_CONTEXT_RELEVANCY, JudgeMetric.IMAGE_FAITHFULNESS, JudgeMetric.ANSWER_RELEVANCY],
        )
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
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14, EmbedderName.JINA_CLIP_2, EmbedderName.COLPALI],
            vlm_list=[VLMName.SMOLVLM_1_2B, VLMName.QWEN_2p5_VL_3B],
            gen_topk_list=[1,5],
        ),
        eval=ExperimentEvalConfig(
            gen_topk_list=[-1,1,5],
            eval_jdg_metric_list=[JudgeMetric.IMAGE_CONTEXT_RELEVANCY, JudgeMetric.IMAGE_FAITHFULNESS, JudgeMetric.ANSWER_RELEVANCY],
        )
    ),
)

"""
ColPali ablations
"""
configstore.store(
    name="topk_context",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            embedder_list=[EmbedderName.COLPALI],
            vlm_list=[VLMName.SMOLVLM_1_2B, VLMName.QWEN_2p5_VL_3B],
            emb_train_loss_type_list=[EmbeddingLoss.MAXSIM, EmbeddingLoss.AVGSIM, EmbeddingLoss.SOFTMAXSIM, EmbeddingLoss.COS_AVGEMB],
        ),
        eval=ExperimentEvalConfig(
            emb_test_loss_type_list=[EmbeddingLoss.MAXSIM, EmbeddingLoss.AVGSIM, EmbeddingLoss.SOFTMAXSIM, EmbeddingLoss.COS_AVGEMB],
        )
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
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14],
            vlm_list=[VLMName.SMOLVLM_1_2B],
            max_perturbation_list=[x / 255.0 for x in [1, 2, 4, 8, 16, 32, 64, 128, 256]],
        ),
    ),
)


"""
Configuration  of figure 3
Bar Chart / Big Table: comparing different models
- should produce 8 images
- figure shows following metrics: recall_before, recall_after (avg train+test), retrieval_asr_train, retrieval_asr_test, generation_asr_train, generation_asr_test
- based on results -> we can run it again with higher or smaller max_perturbation
"""
configstore.store(
    name="heatmap_plot",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=[DatasetName.VIDORE_SYN_AI, DatasetName.VIDORE_V2_ESG],
            embedder_list=[
                EmbedderName.CLIP_LARGE_PATCH14,
                EmbedderName.JINA_CLIP_2,
                EmbedderName.COLPALI,
            ],
            vlm_list=[VLMName.SMOLVLM_1_2B, VLMName.QWEN_2p5_VL_3B],
            max_perturbation_list=[x / 255.0 for x in [8]],
            n_gradient_steps=100,
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
                EmbedderName.JINA_CLIP_2,
                EmbedderName.COLPALI,
            ],
            vlm_list=[VLMName.SMOLVLM_1_2B, VLMName.QWEN_2p5_VL_3B],
        ),
        eval=ExperimentEvalConfig(
            eval_emb_list=[
                EmbedderName.CLIP_LARGE_PATCH14,
                EmbedderName.JINA_CLIP_2,
                EmbedderName.COLPALI,
            ],
            eval_vlm_list=[VLMName.SMOLVLM_1_2B, VLMName.QWEN_2p5_VL_3B],
        ),
    ),
)


"""
Configuration  to test showing images
- Qualitative demonstration showing non-attacked and attacked images on different base images
"""
configstore.store(
    name="display_images",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            dataset_list=[DatasetName.VIDORE_SYN_AI, DatasetName.VIDORE_V2_ESG],
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14],
            vlm_list=[VLMName.SMOLVLM_1_2B],
        ),
    ),
)
