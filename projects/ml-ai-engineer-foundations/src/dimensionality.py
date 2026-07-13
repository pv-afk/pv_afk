from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run(random_state: int = 42) -> dict[str, float | int]:
    features, target = load_wine(return_X_y=True, as_frame=True)
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=random_state,
        stratify=target,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, svd_solver="full")),
            ("model", LogisticRegression(max_iter=3000, random_state=random_state)),
        ]
    )
    pipeline.fit(train_features, train_target)
    prediction = pipeline.predict(test_features)
    pca = pipeline.named_steps["pca"]

    return {
        "source_features": int(features.shape[1]),
        "selected_components": int(pca.n_components_),
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "accuracy": float(accuracy_score(test_target, prediction)),
    }


if __name__ == "__main__":
    print(run())

