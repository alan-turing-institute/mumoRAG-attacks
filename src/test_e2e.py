import tempfile

from dataclasses import replace
from pathlib import Path

from config.experiment import ExperimentConfig
from attack_train import run as train_run
from attack_eval import run as eval_run
from experiments import load_config



def test_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        exp_config = load_config("testing", "pkg://experiments.testing")
        exp_config = ExperimentConfig(
            train=replace(exp_config.train, n_gradient_steps=2, save_folder = tmpdirname),
            eval=replace(exp_config.eval, results_folder = tmpdirname),
        )

        train_run(exp_config)
        eval_run(exp_config)

def test_end_to_end_multi_embedder():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        exp_config = load_config("testing multi-embedder", "pkg://experiments.testing")
        exp_config = ExperimentConfig(
            train=replace(exp_config.train, n_gradient_steps=2, save_folder = tmpdirname),
            eval=replace(exp_config.eval, results_folder = tmpdirname),
        )

        train_run(exp_config)
        eval_run(exp_config)

def test_end_to_end_judge():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        exp_config = load_config("testing judge_defence", "pkg://experiments.testing")
        exp_config = ExperimentConfig(
            train=replace(exp_config.train, n_gradient_steps=2, save_folder = tmpdirname),
            eval=replace(exp_config.eval, results_folder = tmpdirname),
        )

        train_run(exp_config)
        eval_run(exp_config)
