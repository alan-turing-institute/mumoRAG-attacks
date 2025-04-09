from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from utils.text_embedding import TextEmbedderName
from utils.vlm import VLMName
from utils.embedding import EmbedderName
from . import RESULTS_FOLDER


@dataclass
class ExperimentEvalConfig:
    results_folder: Path = RESULTS_FOLDER
    emb_test_loss_type_list: list[str] = field(default_factory=lambda: ["cos"])
    topk_list: list[int] = field(default_factory=lambda: [1, 5])
    gen_metric_list: list[str] = field(default_factory=lambda: ["exact", "embed"])
    gen_text_embedder: TextEmbedderName = TextEmbedderName.JINA_TEXT_V3
    gen_batch_size: int = 4
    do_retrieval: bool = True
    do_generation: bool = True
    # following used to test transferability, None means white-box setting
    eval_emb_list: Optional[list[EmbedderName]] = None
    eval_vlm_list: Optional[list[VLMName]] = None
