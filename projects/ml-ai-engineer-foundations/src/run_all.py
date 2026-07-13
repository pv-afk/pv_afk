import json

from src.classification import run as run_classification
from src.dimensionality import run as run_dimensionality
from src.ensembles import run as run_ensembles
from src.regression import run as run_regression
from src.text_retrieval import run as run_text_retrieval


def run() -> dict[str, object]:
    return {
        "regression": run_regression(),
        "classification": run_classification(),
        "ensembles": run_ensembles(),
        "dimensionality": run_dimensionality(),
        "text_retrieval": run_text_retrieval(),
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))

