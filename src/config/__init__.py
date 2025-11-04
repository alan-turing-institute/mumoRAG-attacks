from pathlib import Path

DATA_FOLDER = Path(__file__).parents[2] / "data"
OUTPUTS_FOLDER = Path(__file__).parents[2] / "outputs"
ATTACKS_FOLDER = DATA_FOLDER / "attacks" / "paper-multirun"
EMBEDDINGS_FOLDER = DATA_FOLDER / "embeddings"
PARAPHRASE_FOLDER = DATA_FOLDER / "paraphrased-queries"
RESULTS_FOLDER = DATA_FOLDER / "results" / "draft"
