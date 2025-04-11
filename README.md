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

Experiments are defined in the [experiments module](./experiments/__init__.py).

Fields of the config can be overridden, e.g.:

```shell
python src/attack_train.py --config-name <name> train.n_gradient_steps=10
```

### Evaluation
To train the attack:
```shell
python src/attack_eval.py --config-name <name>
```
