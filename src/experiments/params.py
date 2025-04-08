"""
This file includes different configurations for the parameters of the attack and the multimodal RAG system
"""

from utils.dataset import DatasetName
from utils.embedding import EmbedderName
from utils.text_embedding import TextEmbedderName
from utils.vlm import VLMName
from dataclasses import dataclass, field
from pathlib import Path


# which experiment to run now?
EXPERIMENT_NUMBER = -2


"""
Default parameter values
"""
# train
dataset_list = [DatasetName.VIDORE_SYN_AI]
embedder_list = [EmbedderName.CLIP_LARGE_PATCH14]
vlm_list = [VLMName.SMOLVLM_1_2B]
max_perturbation_list = [x/255.0 for x in [8]]
emb_train_loss_type_list = ["cos"]
is_adaptive_list = [False]
save_folder = Path(__file__).parents[2] / "data/attacks/"
target_answer = "I will not reply to you!"
chosen_index_list = [150]
n_gradient_steps = 500
print_every = 5
lr_start = 255*(3e-3) 
lr_end = 255*(3e-4)
max_batch_size_per_iter = 2
gradient_acc_steps = 4
lambda_emb = 2
lambda_vlm = 1
lambda_constant = 0.2

# eval
results_folder = Path(__file__).parents[2] / "data/results/"
emb_test_loss_type_list = ["cos"]
topk_list = [1,5]
gen_metric_list = ["exact", "embed"]
gen_text_embedder = TextEmbedderName.JINA_TEXT_V3
do_retrieval = True
do_generation = True
eval_emb_list = [""]
eval_vlm_list = [""]

@dataclass
class ExperimentTrainConfig:
    dataset_list: list[str]             = field(default_factory= lambda: dataset_list)
    embedder_list: list[str]            = field(default_factory= lambda: embedder_list)
    vlm_list: list[str]                 = field(default_factory= lambda: vlm_list)
    max_perturbation_list: list[float]  = field(default_factory= lambda: max_perturbation_list)
    emb_train_loss_type_list: list[str] = field(default_factory= lambda: emb_train_loss_type_list)
    is_adaptive_list: list[bool]        = field(default_factory= lambda: is_adaptive_list)
    save_folder: Path                   = save_folder
    target_answer: str                  = target_answer
    chosen_index_list: list[int]        = field(default_factory= lambda: chosen_index_list)
    n_gradient_steps: int               = n_gradient_steps
    print_every: int                    = print_every
    lr_start: float                     = lr_start
    lr_end: float                       = lr_end
    max_batch_size_per_iter: int        = max_batch_size_per_iter
    gradient_acc_steps: int             = gradient_acc_steps
    lambda_emb: float                   = lambda_emb
    lambda_vlm: float                   = lambda_vlm
    lambda_constant: float              = lambda_constant

@dataclass
class ExperimentEvalConfig:
    results_folder: Path                = results_folder
    emb_test_loss_type_list: list[str]  = field(default_factory= lambda: emb_test_loss_type_list)
    topk_list: list[int]                = field(default_factory= lambda: topk_list)
    gen_metric_list: list[str]          = field(default_factory= lambda: gen_metric_list)
    gen_text_embedder: str              = gen_text_embedder
    gen_batch_size: int                 = 4
    do_retrieval: bool                  = do_retrieval
    do_generation: bool                 = do_generation
    # following used to test transferability, [""] means white-box setting
    eval_emb_list: list[str]            = field(default_factory= lambda: eval_emb_list)
    eval_vlm_list: list[str]            = field(default_factory= lambda: eval_vlm_list)

match EXPERIMENT_NUMBER:
    case -1:
        """
        testing Configuration (same to testing models, but without Qwen-3B)
        """
        exp_config_train = ExperimentTrainConfig(
            dataset_list=[DatasetName.VIDORE_SYN_AI, DatasetName.VIDORE_V2_ESG],
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14, EmbedderName.JINA_CLIP_2, EmbedderName.COLSMOL_500M],
            vlm_list=[VLMName.SMOLVLM_1_256M, VLMName.SMOLVLM_1_500M],
            max_perturbation_list=[x/255.0 for x in [8, 16]],
            save_folder = Path(__file__).parents[2] / "data/attacks/mac/",
            n_gradient_steps=100,
            print_every=5
        )
        exp_config_eval = ExperimentEvalConfig()

    case -2:
        """
        to test perturbation plot
        """
        exp_config_train = ExperimentTrainConfig(
            dataset_list=[DatasetName.VIDORE_SYN_AI],
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14, EmbedderName.JINA_CLIP_2, EmbedderName.COLSMOL_500M],
            vlm_list=[VLMName.SMOLVLM_1_256M, VLMName.SMOLVLM_1_500M],
            max_perturbation_list=[x/255.0 for x in [8, 16]],
            save_folder = Path(__file__).parents[2] / "data/attacks/mac/",
            n_gradient_steps=100,
        )
        exp_config_eval = ExperimentEvalConfig()
    case -22:
        """
        to test perturbation plot
        """
        exp_config_train = ExperimentTrainConfig(
            dataset_list=[DatasetName.VIDORE_SYN_AI],
            embedder_list=[EmbedderName.COLSMOL_500M],
            vlm_list=[VLMName.SMOLVLM_1_256M],
            max_perturbation_list=[x/255.0 for x in [1, 4, 8, 16, 32, 64, 256]],
            save_folder = Path(__file__).parents[2] / "data/attacks/mac/",
            n_gradient_steps=100,
        )
        exp_config_eval = ExperimentEvalConfig()

    case -3:
        """
        to test heatmap
        """
        exp_config_train = ExperimentTrainConfig(
            dataset_list=[DatasetName.VIDORE_SYN_AI],
            # dataset_list=[DatasetName.VIDORE_V2_ESG],
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14, EmbedderName.JINA_CLIP_2, EmbedderName.COLSMOL_500M],
            vlm_list=[VLMName.SMOLVLM_1_256M, VLMName.SMOLVLM_1_500M],
            max_perturbation_list=[x/255.0 for x in [8]],
            save_folder = Path(__file__).parents[2] / "data/attacks/mac/",
            n_gradient_steps=100,
        )
        exp_config_eval = ExperimentEvalConfig()

    case -4:
        """
        Configuration to test transferability (6x6)
        """
        exp_config_train = ExperimentTrainConfig(
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14, EmbedderName.JINA_CLIP_2, EmbedderName.COLSMOL_500M],
            vlm_list=[VLMName.SMOLVLM_1_256M, VLMName.SMOLVLM_1_500M],
            save_folder = Path(__file__).parents[2] / "data/attacks/mac/"
        )
        exp_config_eval = ExperimentEvalConfig(
            eval_emb_list=[EmbedderName.CLIP_LARGE_PATCH14, EmbedderName.JINA_CLIP_2, EmbedderName.COLSMOL_500M],
            eval_vlm_list=[VLMName.SMOLVLM_1_256M, VLMName.SMOLVLM_1_500M]
        )
    case -5:
        """
        Configuration  to test showing images
        - Qualitative demonstration showing non-attacked and attacked images on different base images
        """
        exp_config_train = ExperimentTrainConfig(
            dataset_list=[DatasetName.VIDORE_SYN_AI, DatasetName.VIDORE_V2_ESG],
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14],
            vlm_list=[VLMName.SMOLVLM_1_256M],
            save_folder = Path(__file__).parents[2] / "data/attacks/mac/"
        )
        exp_config_eval = ExperimentEvalConfig()
    case 0:
        """
        testing Configuration
        """
        exp_config_train = ExperimentTrainConfig(
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14],
            vlm_list=[VLMName.SMOLVLM_1_2B],
        )
        exp_config_eval = ExperimentEvalConfig()

    case 1:
        """
        Configuration  of figure 1
        - Qualitative demonstration showing attacked images on different base images
        - should produce 2x5 images
        """
        exp_config_train = ExperimentTrainConfig(
            dataset_list=[DatasetName.VIDORE_SYN_AI, DatasetName.VIDORE_V2_ESG],
            chosen_index_list=[120, 130, 140, 150, 160],
        )
        exp_config_eval = ExperimentEvalConfig()

    case 2:
        """
        Configuration  of figure 2
        Bar chart or line plot: ASR vs. perturbation budget
        - should produce 10 images
        - figure shows following curves: recall_before, recall_after (avg train+test), retrieval_asr_train, retrieval_asr_test, generation_asr_train, generation_asr_test    
        """
        exp_config_train = ExperimentTrainConfig(
            max_perturbation_list=[x/255.0 for x in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]], # [4, 8, 16, 32]
        )
        exp_config_eval = ExperimentEvalConfig()

    case 3:
        """
        Configuration  of figure 3
        Bar Chart / Big Table: comparing different models
        - should produce 8 images
        - figure shows following metrics: recall_before, recall_after (avg train+test), retrieval_asr_train, retrieval_asr_test, generation_asr_train, generation_asr_test
        - based on results -> we can run it again with higher or smaller max_perturbation    
        """
        exp_config_train = ExperimentTrainConfig(
            dataset_list=[DatasetName.VIDORE_SYN_AI, DatasetName.VIDORE_V2_ESG],
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14, EmbedderName.JINA_CLIP_2, EmbedderName.COLSMOL_256M],
            vlm_list=[VLMName.SMOLVLM_1_2B, VLMName.QWEN_2p5_VL_3B],
        )
        exp_config_eval = ExperimentEvalConfig()


    case 4:
        """
        Configuration  of figure 4
        Big Table: transferability between models (12 x 12 table, diagonals are white-box attacks) 
        - should produce 8 x 8 images
        - figure shows following metrics: recall_before, recall_after (avg train+test), retrieval_asr_train, retrieval_asr_test, generation_asr_train, generation_asr_test    
        """
        exp_config_train = ExperimentTrainConfig(
            dataset_list=[DatasetName.VIDORE_SYN_AI, DatasetName.VIDORE_V2_ESG],
            embedder_list=[EmbedderName.CLIP_LARGE_PATCH14, EmbedderName.JINA_CLIP_2, EmbedderName.COLSMOL_256M],
            vlm_list=[VLMName.SMOLVLM_1_2B, VLMName.QWEN_2p5_VL_3B]
        )
        exp_config_eval = ExperimentEvalConfig(
            eval_emb_list=[EmbedderName.CLIP_LARGE_PATCH14, EmbedderName.JINA_CLIP_2, EmbedderName.COLSMOL_256M],
            eval_vlm_list=[VLMName.SMOLVLM_1_2B, VLMName.QWEN_2p5_VL_3B]
        )
    case 5:
        exp_config_train = ExperimentTrainConfig(
            dataset_list=[DatasetName.VIDORE_V2_ESG],
            embedder_list=[EmbedderName.COLPALI],
            vlm_list=[VLMName.SMOLVLM_1_2B],
            max_perturbation_list=[128/255.0],
            n_gradient_steps = 1500,
        )
        exp_config_eval = ExperimentEvalConfig()
