"""
This file includes different configurations for the parameters of the attack and the multimodal RAG system
"""

from hydra.core.config_store import ConfigStore

from config.eval import ExperimentEvalConfig
from config.experiment import ExperimentConfig
from config.train import ExperimentTrainConfig
from utils.dataset import DatasetName
from utils.embedding import EmbedderName
from utils.vlm import VLMName

DEFAULT_EXPERIMENT = "testing"

configstore = ConfigStore.instance()

"""
testing Configuration
"""
configstore.store(
    name="testing",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14],
            vlm_list=[VLMName.SMOLVLM_1_2B],
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
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14],
            vlm_list=[VLMName.SMOLVLM_1_2B],
            max_perturbation_list=[
                x / 255.0 for x in [1, 2, 4, 8, 16, 32, 64, 128, 256]
            ],
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
Configuration to test transferability (6x6)
"""
configstore.store(
    name="transferability",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
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
    node = ExperimentConfig(
    train = ExperimentTrainConfig(
        dataset_list=[DatasetName.VIDORE_SYN_AI, DatasetName.VIDORE_V2_ESG],
        embedder_list=[EmbedderName.CLIP_LARGE_PATCH14],
        vlm_list=[VLMName.SMOLVLM_1_2B],
        ),
    ),
)
