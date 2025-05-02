# mumoRAG-attacks
Adversarial and poisoning attacks against multimodal retrieval-augmented generation (RAG)

## Instructions
- Install poppler utils `brew install poppler`
- Install the project as editable `pip install -e .`

## Running

### Training
To train the attack:
```shell
python src/attack_train.py --config-name <name>
```

Experiments are defined in the [experiments module](src/experiments/configstore.py).

Fields of the config can be overridden, e.g.:

```shell
python src/attack_train.py --config-name <name> train.n_gradient_steps=10
```

### Evaluation

To train the attack:

```shell
python src/attack_eval.py --config-name <name>
```

## Experiment output

The experiment outputs are contained in the `data` directory and are split into subdirectories as follows.

### Attacks

Contains the output adversarial images.
File names contain the name of the experiment that generated them and a hash of the experiment parameters.

### Embeddings

Contains the cached embeddings of datasets using embedding models.
File names contain the name of the dataset and embedding model.

### Results

Contains the results json files produced from the evals.
File names contain the name of the experiment that generated them and a hash of the experiment parameters.

### Sharing data

By default, files are output into the `draft` subdirectories of `attacks` and `results`, which are ignored. 
To share work, move files into the parent directories.
You can then adjust the paths to point to the parent directories, either in the experiment configuration:

```python
configstore.store(
    ...,
    node=ExperimentConfig(
        train=ExperimentTrainConfig(
            ...,
            save_folder=ATTACKS_FOLDER.parent,
        ),
        test=ExperimentEvalConfig(
            ...,
            results_folder=RESULTS_FOLDER.parent,
        ),
    )
)
```

or via the CLI

```shell
python src/attack_train.py --config <config_name> train.save_folder="data/attacks" eval.results_folder="data/results"
```
