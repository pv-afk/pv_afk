from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run(random_state: int = 42) -> dict[str, float]:
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    target = (dataset.target == 0).astype(int)
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
        stratify=target,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(train_features, train_target)
    prediction = pipeline.predict(test_features)
    probability = pipeline.predict_proba(test_features)[:, 1]

    return {
        "accuracy": float(accuracy_score(test_target, prediction)),
        "precision": float(precision_score(test_target, prediction)),
        "recall": float(recall_score(test_target, prediction)),
        "f1": float(f1_score(test_target, prediction)),
        "roc_auc": float(roc_auc_score(test_target, probability)),
    }


if __name__ == "__main__":
    print(run())

