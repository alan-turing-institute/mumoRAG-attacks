from dataclasses import dataclass, field

from .eval import ExperimentEvalConfig
from .train import ExperimentTrainConfig


@dataclass
class ExperimentConfig:
    train: ExperimentTrainConfig = field(default_factory=lambda: ExperimentTrainConfig())
    eval: ExperimentEvalConfig = field(default_factory=lambda: ExperimentEvalConfig())
