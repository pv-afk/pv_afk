from src.classification import run as run_classification
from src.dimensionality import run as run_dimensionality
from src.ensembles import run as run_ensembles
from src.regression import run as run_regression
from src.text_retrieval import search


def test_regression_metrics_are_finite():
    result = run_regression()
    assert result["rmse"] > 0
    assert -1 <= result["r2"] <= 1


def test_classifier_reaches_baseline_quality():
    result = run_classification()
    assert result["f1"] > 0.85
    assert result["roc_auc"] > 0.9


def test_ensemble_comparison_contains_three_models():
    result = run_ensembles()
    assert len(result) == 3
    assert result[0]["roc_auc_mean"] > 0.9


def test_pca_reduces_feature_count():
    result = run_dimensionality()
    assert result["selected_components"] < result["source_features"]
    assert result["explained_variance"] >= 0.95


def test_retrieval_returns_relevant_rag_document():
    result = search("поиск контекста для языковой модели", limit=3)
    assert len(result) == 3
    assert any(item["id"] == "rag" for item in result)

