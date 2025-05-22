from dataclasses import replace
from unittest.mock import MagicMock

import torch

from attack_eval import run as eval_run
from attack_train import run as train_run
from config.experiment import ExperimentConfig
from experiments import load_config
from utils.utils import get_device
from wrappers.cache import get_dataset, get_embedded_dataset
from wrappers.dataset import DatasetName
from wrappers.embedding import EmbedderName


def test_embedding(mocker, tmp_path):
    ds = get_dataset(DatasetName.VIDORE_SYN_AI)

    save_spy: MagicMock = mocker.spy(torch, "save")
    load_spy: MagicMock = mocker.spy(torch, "load")
    # generate and save to disk
    get_embedded_dataset(
        dataset=ds,
        model_name_emb=EmbedderName.CLIP_BASE_PATCH16,
        quantize=False,
        colpali_only_images=False,
        device=get_device(True),
        embeddings_folder=tmp_path,
    )
    assert save_spy.call_count == 1
    assert isinstance(load_spy.spy_exception, FileNotFoundError)
    mocker.resetall()

    # reload from disk
    get_embedded_dataset(
        dataset=ds,
        model_name_emb=EmbedderName.CLIP_BASE_PATCH16,
        quantize=False,
        colpali_only_images=False,
        device=get_device(True),
        embeddings_folder=tmp_path,
    )
    assert save_spy.call_count == 0
    assert load_spy.spy_exception is None


def test_embedding_paraphrased(mocker, tmp_path):
    ds = get_dataset(
        DatasetName.VIDORE_SYN_AI,
        paraphrase_queries=True,
    )
    save_spy = mocker.spy(torch, "save")
    load_spy = mocker.spy(torch, "load")
    # generate and save to disk
    get_embedded_dataset(
        dataset=ds,
        model_name_emb=EmbedderName.CLIP_BASE_PATCH16,
        quantize=False,
        colpali_only_images=False,
        device=get_device(True),
        embeddings_folder=tmp_path,
    )
    assert save_spy.call_count == 1
    assert isinstance(load_spy.spy_exception, FileNotFoundError)
    mocker.resetall()

    # reload from disk
    get_embedded_dataset(
        dataset=ds,
        model_name_emb=EmbedderName.CLIP_BASE_PATCH16,
        quantize=False,
        colpali_only_images=False,
        device=get_device(True),
        embeddings_folder=tmp_path,
    )

    assert save_spy.call_count == 0
    assert load_spy.spy_exception is None


def test_end_to_end(tmp_path):
    exp_config = load_config("testing", "pkg://experiments.testing")
    exp_config = ExperimentConfig(
        train=replace(exp_config.train, n_gradient_steps=2, save_folder=tmp_path),
        eval=replace(exp_config.eval, results_folder=tmp_path),
    )

    train_run(exp_config)
    eval_run(exp_config)


def test_end_to_end_no_vlm(tmp_path):
    exp_config = load_config("testing no vlm", "pkg://experiments.testing")
    exp_config = ExperimentConfig(
        train=replace(exp_config.train, n_gradient_steps=2, save_folder=tmp_path),
        eval=replace(exp_config.eval, results_folder=tmp_path),
    )

    train_run(exp_config)
    eval_run(exp_config)


def test_end_to_end_multi_embedder(tmp_path):
    exp_config = load_config("testing multi-embedder", "pkg://experiments.testing")
    exp_config = ExperimentConfig(
        train=replace(exp_config.train, n_gradient_steps=2, save_folder=tmp_path),
        eval=replace(exp_config.eval, results_folder=tmp_path),
    )

    train_run(exp_config)
    eval_run(exp_config)


def test_end_to_end_multi_vlm(tmp_path):
    exp_config = load_config("testing multi-vlm", "pkg://experiments.testing")
    exp_config = ExperimentConfig(
        train=replace(exp_config.train, n_gradient_steps=2, save_folder=tmp_path),
        eval=replace(exp_config.eval, results_folder=tmp_path),
    )

    train_run(exp_config)
    eval_run(exp_config)


def test_end_to_end_judge(tmp_path):
    exp_config = load_config("testing judge_defence", "pkg://experiments.testing")
    exp_config = ExperimentConfig(
        train=replace(exp_config.train, n_gradient_steps=2, save_folder=tmp_path),
        eval=replace(exp_config.eval, results_folder=tmp_path),
    )

    train_run(exp_config)
    eval_run(exp_config)
