"""Улучшенная версия решения с дополнительными признаками истории.

Я сохраняю исходный ансамбль как один из независимых компонентов и добавляю
три модели на расширенном наборе признаков. Все события по-прежнему фильтруются
строгим условием event_ts < assignment_ts.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

# На macOS joblib иногда не может определить число физических ядер в песочнице.
# Я задаю безопасное значение заранее: это убирает служебное предупреждение и
# не меняет ни признаки, ни предсказания моделей.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from solution import (
    DATE_COLUMN,
    RANDOM_STATE,
    TARGET,
    apply_validated_rules_and_rerank,
    fit_predict_ensemble,
    make_tree_matrices,
    prepare_features,
    rank_within_day,
)


EXTRA_WINDOWS = (0.25, 0.5, 2, 5, 10, 21, 60)


def build_extra_event_features(
    leads: pd.DataFrame,
    events_log: pd.DataFrame,
) -> pd.DataFrame:
    """Я подробнее описываю давность, короткие окна и последние переходы."""
    lead_times = leads[["lead_id", "assignment_ts"]].copy()
    lead_times["assignment_ts"] = pd.to_datetime(
        lead_times["assignment_ts"], errors="raise"
    )

    lead_history = events_log.copy()
    lead_history["event_ts"] = pd.to_datetime(
        lead_history["event_ts"], errors="raise"
    )
    lead_history = lead_history.merge(
        lead_times, on="lead_id", validate="many_to_one"
    )
    lead_history = lead_history[
        lead_history["event_ts"] < lead_history["assignment_ts"]
    ].copy()
    assert (lead_history["event_ts"] < lead_history["assignment_ts"]).all()
    lead_history["age_days"] = (
        lead_history["assignment_ts"] - lead_history["event_ts"]
    ).dt.total_seconds() / 86_400
    lead_history = lead_history.sort_values(["lead_id", "event_ts"])

    lead_ids = leads["lead_id"].astype(str)
    extra_features = pd.DataFrame(index=lead_ids)
    action_types = sorted(lead_history["event_type"].dropna().unique())

    # Для каждого типа действия я отдельно считаю давность последнего события.
    action_recency = lead_history.pivot_table(
        index="lead_id",
        columns="event_type",
        values="age_days",
        aggfunc="min",
    )
    action_recency.columns = [
        f"extra_{action}_age_min" for action in action_recency.columns
    ]
    extra_features = extra_features.join(action_recency)

    action_mean_age = lead_history.pivot_table(
        index="lead_id",
        columns="event_type",
        values="age_days",
        aggfunc="mean",
    )
    action_mean_age.columns = [
        f"extra_{action}_age_mean" for action in action_mean_age.columns
    ]
    extra_features = extra_features.join(action_mean_age)

    source_recency = lead_history.pivot_table(
        index="lead_id",
        columns="src_slot",
        values="age_days",
        aggfunc="min",
    )
    source_recency.columns = [
        f"extra_src_{str(source).replace('.', '_')}_age_min"
        for source in source_recency.columns
    ]
    extra_features = extra_features.join(source_recency)

    # Сочетание контекста и действия оказалось полезнее двух отдельных счётчиков.
    lead_history["context_action"] = (
        lead_history["ctx_seq"].astype(str)
        + "__"
        + lead_history["event_type"].astype(str)
    )
    context_action_recency = lead_history.pivot_table(
        index="lead_id",
        columns="context_action",
        values="age_days",
        aggfunc="min",
    )
    context_action_recency.columns = [
        f"extra_ctx_action_{value}_age_min"
        for value in context_action_recency.columns
    ]
    extra_features = extra_features.join(context_action_recency)

    # Я добавляю промежуточные окна, которых не было в исходной таблице.
    for window in EXTRA_WINDOWS:
        window_name = str(window).replace(".", "_")
        recent_history = lead_history[lead_history["age_days"] <= window]
        extra_features[f"extra_event_count_{window_name}d"] = (
            recent_history.groupby("lead_id").size()
        )
        action_counts = recent_history.pivot_table(
            index="lead_id",
            columns="event_type",
            values="event_ts",
            aggfunc="size",
            fill_value=0,
        ).reindex(columns=action_types, fill_value=0)
        action_counts.columns = [
            f"extra_{action}_count_{window_name}d"
            for action in action_counts.columns
        ]
        extra_features = extra_features.join(action_counts)

    # Последние переходы дают модели короткий пользовательский сценарий целиком.
    last_events = {}
    for position in (1, 2, 3):
        event_at_position = (
            lead_history.groupby("lead_id", sort=False)
            .nth(-position)
            .set_index("lead_id")
        )
        last_events[position] = event_at_position
        extra_features[f"extra_last{position}_type_ctx"] = (
            event_at_position["event_type"].astype(str)
            + "__"
            + event_at_position["ctx_seq"].astype(str)
        )
    extra_features["extra_last_type_transition"] = (
        last_events[2]["event_type"].astype(str)
        + "__"
        + last_events[1]["event_type"].astype(str)
    )
    extra_features["extra_last_ctx_transition"] = (
        last_events[2]["ctx_seq"].astype(str)
        + "__"
        + last_events[1]["ctx_seq"].astype(str)
    )

    count_columns = [
        column for column in extra_features.columns if "count" in column
    ]
    extra_features[count_columns] = extra_features[count_columns].fillna(0.0)
    return extra_features.reindex(lead_ids).reset_index(drop=True)


def add_rounded_count_features(features: pd.DataFrame) -> pd.DataFrame:
    """Я восстанавливаю целочисленную структуру трёх зашумлённых счётчиков."""
    rounded_features = {}
    noisy_columns = (
        "seller_page_views_7d",
        "seller_page_views_14d",
        "seller_page_views_30d",
    )
    for column in noisy_columns:
        rounded = features[column].round()
        rounded_features[f"{column}_rounded"] = rounded
        rounded_features[f"{column}_rounding_error"] = features[column] - rounded
        rounded_features[f"{column}_rounding_error_abs"] = (
            features[column] - rounded
        ).abs()

    for short_window, long_window in zip(noisy_columns[:-1], noisy_columns[1:]):
        rounded_features[f"{short_window}_{long_window}_rounded_increment"] = (
            features[long_window].round() - features[short_window].round()
        )
    rounded_features["seller_page_views_30d_90d_rounded_increment"] = (
        features["seller_page_views_90d"]
        - features["seller_page_views_30d"].round()
    )
    return pd.concat(
        [features.reset_index(drop=True), pd.DataFrame(rounded_features)], axis=1
    )


def prepare_enhanced_features(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    events_log: pd.DataFrame,
):
    """Я создаю исходный и расширенный наборы признаков одним проходом."""
    base_train, base_test, base_num_columns, base_cat_columns = prepare_features(
        train_data, test_data, events_log
    )
    all_leads = pd.concat(
        [train_data.drop(columns=[TARGET]), test_data], ignore_index=True
    )
    extra_event_features = build_extra_event_features(all_leads, events_log)
    enhanced_features = pd.concat(
        [
            pd.concat([base_train, base_test], ignore_index=True),
            extra_event_features,
        ],
        axis=1,
    )
    enhanced_features = add_rounded_count_features(enhanced_features)
    enhanced_train = enhanced_features.iloc[: len(train_data)].copy()
    enhanced_test = enhanced_features.iloc[len(train_data) :].copy()
    enhanced_cat_columns = enhanced_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    return (
        base_train,
        base_test,
        base_num_columns,
        base_cat_columns,
        enhanced_train,
        enhanced_test,
        enhanced_cat_columns,
    )


def fit_enhanced_models(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    features_to_score: pd.DataFrame,
    dates_to_score: pd.Series,
    cat_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Я обучаю три устойчивые модели, выбранные на двух временных holdout."""
    train_matrix, score_matrix = make_tree_matrices(
        train_features, features_to_score, cat_columns
    )

    plain_lightgbm = LGBMClassifier(
        num_leaves=15,
        min_child_samples=60,
        learning_rate=0.02,
        n_estimators=1_800,
        reg_lambda=8.0,
        reg_alpha=0.2,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
    )
    plain_lightgbm.fit(train_matrix, train_target)
    plain_rank = rank_within_day(
        plain_lightgbm.predict_proba(score_matrix)[:, 1], dates_to_score
    )
    print("Обучена улучшенная LightGBM-модель")

    extra_trees_lightgbm = LGBMClassifier(
        num_leaves=31,
        min_child_samples=45,
        extra_trees=True,
        max_bin=127,
        learning_rate=0.02,
        n_estimators=600,
        reg_lambda=8.0,
        reg_alpha=0.2,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
    )
    extra_trees_lightgbm.fit(train_matrix, train_target)
    extra_trees_rank = rank_within_day(
        extra_trees_lightgbm.predict_proba(score_matrix)[:, 1], dates_to_score
    )
    print("Обучена Extra-Trees LightGBM-модель")

    catboost_train = train_features.reset_index(drop=True).copy()
    catboost_score_data = features_to_score.reset_index(drop=True).copy()
    for column in cat_columns:
        catboost_train[column] = (
            catboost_train[column].fillna("<пропуск>").astype(str)
        )
        catboost_score_data[column] = (
            catboost_score_data[column].fillna("<пропуск>").astype(str)
        )
    catboost_model = CatBoostClassifier(
        depth=8,
        learning_rate=0.03,
        iterations=750,
        l2_leaf_reg=8.0,
        random_strength=0.5,
        bagging_temperature=0.5,
        loss_function="Logloss",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    catboost_model.fit(
        catboost_train,
        train_target,
        cat_features=cat_columns,
        verbose=False,
    )
    catboost_rank = rank_within_day(
        catboost_model.predict_proba(catboost_score_data)[:, 1], dates_to_score
    )
    print("Обучена улучшенная CatBoost-модель")
    return plain_rank, extra_trees_rank, catboost_rank


def create_submission_v2(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    base_train: pd.DataFrame,
    base_test: pd.DataFrame,
    base_num_columns: list[str],
    base_cat_columns: list[str],
    enhanced_train: pd.DataFrame,
    enhanced_test: pd.DataFrame,
    enhanced_cat_columns: list[str],
) -> pd.DataFrame:
    """Я объединяю старый baseline с тремя новыми моделями."""
    baseline_rank, _, _ = fit_predict_ensemble(
        train_features=base_train,
        train_target=train_data[TARGET],
        features_to_score=base_test,
        dates_to_score=test_data[DATE_COLUMN],
        num_columns=base_num_columns,
        cat_columns=base_cat_columns,
    )
    plain_rank, extra_trees_rank, catboost_rank = fit_enhanced_models(
        enhanced_train,
        train_data[TARGET],
        enhanced_test,
        test_data[DATE_COLUMN],
        enhanced_cat_columns,
    )

    # Эти простые веса устойчиво выиграли на двух непересекающихся временных окнах.
    blended_rank = (
        0.20 * baseline_rank
        + 0.30 * plain_rank
        + 0.20 * extra_trees_rank
        + 0.30 * catboost_rank
    )
    final_scores = apply_validated_rules_and_rerank(
        blended_rank, base_test, test_data[DATE_COLUMN]
    )
    submission = pd.DataFrame(
        {
            "lead_id": test_data["lead_id"].astype(str),
            "score": final_scores.astype(float),
        }
    )
    assert list(submission.columns) == ["lead_id", "score"]
    assert submission["lead_id"].is_unique
    assert submission["score"].between(0, 1).all()
    assert len(submission) == len(test_data)
    return submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("submission_v2.csv"))
    args = parser.parse_args()

    started_at = time.time()
    print(
        f"pandas={pd.__version__}; numpy={np.__version__}; "
        f"sklearn={sklearn.__version__}"
    )
    train_data = pd.read_csv(args.data_dir / "train.csv")
    test_data = pd.read_csv(args.data_dir / "test.csv")
    events_log = pd.read_csv(args.data_dir / "events.csv")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        prepared = prepare_enhanced_features(train_data, test_data, events_log)
    (
        base_train,
        base_test,
        base_num_columns,
        base_cat_columns,
        enhanced_train,
        enhanced_test,
        enhanced_cat_columns,
    ) = prepared
    print(
        f"Признаков: baseline={base_train.shape[1]}, "
        f"улучшенная версия={enhanced_train.shape[1]}"
    )

    submission = create_submission_v2(
        train_data,
        test_data,
        base_train,
        base_test,
        base_num_columns,
        base_cat_columns,
        enhanced_train,
        enhanced_test,
        enhanced_cat_columns,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Файл {args.output} сохранён, строк: {len(submission):,}")
    print(f"Общее время работы: {time.time() - started_at:.1f} сек.")


if __name__ == "__main__":
    main()
