"""
This file includes different configurations for the parameters of the attack and the multimodal RAG system
"""

from hydra.core.config_store import ConfigStore

from config.eval import ExperimentEvalConfig
from config.experiment import ExperimentConfig
from config.train import ExperimentTrainConfig, JudgeConfig, VLMConfig
from wrappers.embedding import EmbedderName
from wrappers.judge import JudgeMetric
from wrappers.vlm import VLMName

configstore = ConfigStore.instance()

"""
Testing Configuration (default)
"""
configstore.store(
    name="testing",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            embedder_list=[
                EmbedderName.SIGLIP2_BASE_PATCH16,
                EmbedderName.CLIP_BASE_PATCH16,
            ],
            vlm=VLMConfig(
                models=[VLMName.SMOLVLM_1_256M],
            ),
        ),
    ),
)


"""
Testing Configuration (no vlm)
"""
configstore.store(
    name="testing no vlm",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            embedder_list=[
                EmbedderName.CLIP_BASE_PATCH16,
            ],
            vlm=None,
        ),
        eval=ExperimentEvalConfig(
            do_generation=False,
        ),
    ),
)


"""
Testing Configuration (multi-embedder)
"""
configstore.store(
    name="testing multi-embedder",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            embedder_list=[
                [EmbedderName.SIGLIP2_BASE_PATCH16, EmbedderName.CLIP_BASE_PATCH16],
                EmbedderName.SIGLIP2_BASE_PATCH16,
                EmbedderName.CLIP_BASE_PATCH16,
            ],
            vlm=VLMConfig(
                models=[VLMName.SMOLVLM_1_256M],
            ),
        ),
        eval=ExperimentEvalConfig(
            eval_emb_list=[EmbedderName.SIGLIP2_BASE_PATCH16, EmbedderName.CLIP_BASE_PATCH16],
        ),
    ),
)

"""
Testing Configuration (multi-vlm)
"""
configstore.store(
    name="testing multi-vlm",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            embedder_list=[EmbedderName.CLIP_BASE_PATCH16],
            vlm=VLMConfig(
                models=[[VLMName.SMOLVLM_1_256M, VLMName.INTERNVL_3_1B], VLMName.SMOLVLM_1_256M, VLMName.INTERNVL_3_1B],
            ),
        ),
        eval=ExperimentEvalConfig(
            eval_vlm_list=[VLMName.SMOLVLM_1_256M, VLMName.INTERNVL_3_1B],
        ),
    ),
)

"""
RAG evaluation and attack detection through VLM-as-a-judge
"""
configstore.store(
    name="testing judge_defence",
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14],
            vlm=VLMConfig(
                models=[VLMName.SMOLVLM_1_2B],
            ),
            judge=JudgeConfig(
                lambda_=1,
                models=[VLMName.SMOLVLM_1_2B],
                metrics=[JudgeMetric.IMAGE_CONTEXT_RELEVANCY, JudgeMetric.IMAGE_FAITHFULNESS, JudgeMetric.ANSWER_RELEVANCY],
            ),
        ),
        eval=ExperimentEvalConfig(
            gen_topk_list=[-1, 1, 5],
            do_judge=True,
            eval_jdg_metric_list=[JudgeMetric.IMAGE_CONTEXT_RELEVANCY, JudgeMetric.IMAGE_FAITHFULNESS, JudgeMetric.ANSWER_RELEVANCY],
        ),
    ),
)
