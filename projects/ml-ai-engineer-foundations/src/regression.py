import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run(random_state: int = 42) -> dict[str, float]:
    features, target = load_diabetes(return_X_y=True, as_frame=True)
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 31))),
        ]
    )
    pipeline.fit(train_features, train_target)
    prediction = pipeline.predict(test_features)

    return {
        "mae": float(mean_absolute_error(test_target, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(test_target, prediction))),
        "r2": float(r2_score(test_target, prediction)),
        "alpha": float(pipeline.named_steps["model"].alpha_),
    }


if __name__ == "__main__":
    print(run())

