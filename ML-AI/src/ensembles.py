from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier


def run(random_state: int = 42) -> list[dict[str, float | str]]:
    features, target = load_breast_cancer(return_X_y=True, as_frame=True)
    validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    models = {
        "decision_tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=5,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            n_jobs=1,
            random_state=random_state,
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            random_state=random_state,
        ),
    }

    results = []
    for name, model in models.items():
        scores = cross_validate(
            model,
            features,
            target,
            cv=validation,
            scoring={"f1": "f1", "roc_auc": "roc_auc"},
            n_jobs=1,
        )
        results.append(
            {
                "model": name,
                "f1_mean": float(scores["test_f1"].mean()),
                "f1_std": float(scores["test_f1"].std()),
                "roc_auc_mean": float(scores["test_roc_auc"].mean()),
            }
        )

    return sorted(results, key=lambda item: item["roc_auc_mean"], reverse=True)


if __name__ == "__main__":
    print(run())
