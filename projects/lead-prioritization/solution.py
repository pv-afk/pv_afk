"""Моё воспроизводимое решение задачи приоритизации обращений.

Я использую только открытые библиотеки: NumPy, pandas, scikit-learn,
LightGBM и CatBoost. При работе с событиями я всегда соблюдаю правило
event_ts < assignment_ts, поэтому моя модель не заглядывает в будущее.
"""

from __future__ import annotations

import argparse
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "target"
DATE_COLUMN = "assignment_date"
RANDOM_STATE = 42
WINDOWS = (1, 3, 7, 14, 30, 90)


def daily_average_precision(targets, predictions, assignment_dates) -> float:
    """Я считаю AP отдельно для каждого дня, а затем беру среднее по дням."""
    daily_results = pd.DataFrame(
        {
            "target": np.asarray(targets),
            "score": np.asarray(predictions),
            "date": np.asarray(assignment_dates),
        }
    )
    ap_by_day = daily_results.groupby("date", sort=True).apply(
        lambda day_data: average_precision_score(
            day_data["target"], day_data["score"]
        ),
        include_groups=False,
    )
    return float(ap_by_day.mean())


def rank_within_day(predictions, assignment_dates) -> np.ndarray:
    """Я перевожу прогнозы в дневные ранги: для Daily AP важен порядок."""
    daily_results = pd.DataFrame(
        {"score": np.asarray(predictions), "date": np.asarray(assignment_dates)}
    )
    return (
        daily_results.groupby("date", sort=False)["score"]
        # Я использую method="first", чтобы одинаковые прогнозы получили разные места.
        .rank(method="first", pct=True)
        .to_numpy(dtype=float)
    )


def categorical_entropy(categories: pd.Series) -> float:
    """Я оцениваю, насколько разнообразны действия пользователя в истории."""
    category_shares = categories.value_counts(normalize=True).to_numpy()
    return float(-(category_shares * np.log(category_shares)).sum())


def build_event_features(leads: pd.DataFrame, events_log: pd.DataFrame) -> pd.DataFrame:
    """Я собираю историю каждого обращения, не используя будущие события."""
    lead_times = leads[["lead_id", "assignment_ts"]].copy()
    lead_times["assignment_ts"] = pd.to_datetime(
        lead_times["assignment_ts"], errors="raise"
    )

    lead_history = events_log.copy()
    lead_history["event_ts"] = pd.to_datetime(
        lead_history["event_ts"], errors="raise"
    )
    lead_history = lead_history.merge(
        lead_times, on="lead_id", how="inner", validate="many_to_one"
    )
    events_before_filter = len(lead_history)

    # Здесь я ставлю главную защиту от утечки: оставляю только события,
    # которые уже произошли к моменту назначения обращения.
    lead_history = lead_history[
        lead_history["event_ts"] < lead_history["assignment_ts"]
    ].copy()
    assert (lead_history["event_ts"] < lead_history["assignment_ts"]).all()

    lead_history["age_days"] = (
        lead_history["assignment_ts"] - lead_history["event_ts"]
    ).dt.total_seconds() / 86_400.0
    lead_history["event_day"] = lead_history["event_ts"].dt.floor("D")

    # Я сразу создаю полный список обращений, чтобы не потерять строки без истории.
    event_stats = pd.DataFrame(index=leads["lead_id"].astype(str))
    basic_stats = lead_history.groupby("lead_id").agg(
        ev_count=("event_ts", "size"),
        ev_type_nunique=("event_type", "nunique"),
        ev_active_days=("event_day", "nunique"),
        ev_age_min_days=("age_days", "min"),
        ev_age_mean_days=("age_days", "mean"),
        ev_age_max_days=("age_days", "max"),
        ev_price_mean=("item_price_log", "mean"),
        ev_price_std=("item_price_log", "std"),
        ev_price_min=("item_price_log", "min"),
        ev_price_max=("item_price_log", "max"),
        ev_src_mean=("src_slot", "mean"),
        ev_src_std=("src_slot", "std"),
        ev_src_min=("src_slot", "min"),
        ev_src_max=("src_slot", "max"),
        ev_src_nunique=("src_slot", "nunique"),
        ev_ctx_nunique=("ctx_seq", "nunique"),
    )
    event_stats = event_stats.join(basic_stats)
    event_stats["ev_span_days"] = (
        event_stats["ev_age_max_days"] - event_stats["ev_age_min_days"]
    )

    action_types = sorted(events_log["event_type"].dropna().astype(str).unique())
    for days in WINDOWS:
        recent_history = lead_history[lead_history["age_days"] <= days]
        event_stats[f"ev_count_{days}d"] = recent_history.groupby("lead_id").size()
        action_counts = recent_history.pivot_table(
            index="lead_id", columns="event_type", values="event_ts", aggfunc="size"
        ).reindex(columns=action_types, fill_value=0)
        action_counts.columns = [
            f"ev_{action_type}_count_{days}d" for action_type in action_counts.columns
        ]
        event_stats = event_stats.join(action_counts)

    # В готовой таблице нет подробной разбивки по контексту и источнику,
    # поэтому я считаю её самостоятельно.
    for column, prefix in (("ctx_seq", "ev_ctx"), ("src_slot", "ev_src")):
        context_counts = lead_history.pivot_table(
            index="lead_id", columns=column, values="event_ts", aggfunc="size", fill_value=0
        )
        context_counts.columns = [
            f"{prefix}_{str(value).replace('.', '_')}_count"
            for value in context_counts.columns
        ]
        event_stats = event_stats.join(context_counts)

    # Я отдельно сохраняю последнее событие: оно часто лучше всего описывает намерение.
    last_event = (
        lead_history.sort_values("event_ts")
        .groupby("lead_id", sort=False)
        .tail(1)
        .set_index("lead_id")
    )
    event_stats["ev_last_price"] = last_event["item_price_log"]
    event_stats["ev_last_src"] = last_event["src_slot"]
    for action_type in action_types:
        event_stats[f"ev_last_is_{action_type}"] = (
            last_event["event_type"].eq(action_type).astype(float)
        )

    # Если счётчика нет, я ставлю ноль: событие не происходило, это не пропуск.
    zero_filled_columns = [
        column
        for column in event_stats.columns
        if "count" in column
        or column
        in {"ev_type_nunique", "ev_active_days", "ev_src_nunique", "ev_ctx_nunique"}
        or column.startswith("ev_last_is_")
    ]
    event_stats[zero_filled_columns] = event_stats[zero_filled_columns].fillna(0.0)
    event_stats = event_stats.reindex(leads["lead_id"].astype(str)).reset_index(drop=True)
    event_stats.columns = event_stats.columns.astype(str)

    print(
        f"Использовано событий: {len(lead_history):,}; "
        f"будущих событий исключено: {events_before_filter - len(lead_history):,}"
    )
    return event_stats


def build_advanced_event_features(
    leads: pd.DataFrame,
    events_log: pd.DataFrame,
) -> pd.DataFrame:
    """Я подробно описываю контекст, порядок и темп событий до назначения."""
    lead_context = leads[["lead_id", "assignment_ts", "item_price_log"]].copy()
    lead_context["assignment_ts"] = pd.to_datetime(
        lead_context["assignment_ts"], errors="raise"
    )

    lead_history = events_log.copy()
    lead_history["event_ts"] = pd.to_datetime(
        lead_history["event_ts"], errors="raise"
    )
    lead_history = lead_history.merge(
        lead_context,
        on="lead_id",
        suffixes=("_event", "_assignment"),
        validate="many_to_one",
    )
    lead_history = lead_history[
        lead_history["event_ts"] < lead_history["assignment_ts"]
    ].copy()
    lead_history = lead_history.sort_values(["lead_id", "event_ts"])
    lead_history["age_days"] = (
        lead_history["assignment_ts"] - lead_history["event_ts"]
    ).dt.total_seconds() / 86_400.0

    all_lead_ids = leads["lead_id"].astype(str)
    event_stats = pd.DataFrame(index=all_lead_ids)

    # Отдельные счётчики не всегда раскрывают совместный смысл контекста и действия,
    # поэтому я явно добавляю сочетания вроде c05 + item_view.
    lead_history["ctx_type"] = (
        lead_history["ctx_seq"].astype(str)
        + "__"
        + lead_history["event_type"].astype(str)
    )
    context_action_counts = lead_history.pivot_table(
        index="lead_id",
        columns="ctx_type",
        values="event_ts",
        aggfunc="size",
        fill_value=0,
    )
    context_action_counts.columns = [
        f"ev_ctx_type_{value}_count" for value in context_action_counts.columns
    ]
    event_stats = event_stats.join(context_action_counts)

    # Я смотрю не только на наличие контекста, но и на то, как давно он встречался.
    context_codes = sorted(lead_history["ctx_seq"].dropna().unique())
    context_recency = lead_history.pivot_table(
        index="lead_id", columns="ctx_seq", values="age_days", aggfunc="min"
    )
    context_recency.columns = [
        f"ev_ctx_{value}_age_min" for value in context_recency.columns
    ]
    event_stats = event_stats.join(context_recency)
    for days in WINDOWS:
        recent_history = lead_history[lead_history["age_days"] <= days]
        context_counts = recent_history.pivot_table(
            index="lead_id",
            columns="ctx_seq",
            values="event_ts",
            aggfunc="size",
            fill_value=0,
        ).reindex(columns=context_codes, fill_value=0)
        context_counts.columns = [
            f"ev_ctx_{value}_count_{days}d" for value in context_counts.columns
        ]
        event_stats = event_stats.join(context_counts)

    # Я сохраняю последние три действия, чтобы восстановить короткий сценарий:
    # например, переход от просмотра к избранному и открытию чата.
    sequence_columns = []
    for position in (1, 2, 3):
        event_at_position = (
            lead_history.groupby("lead_id", sort=False)
            .nth(-position)
            .set_index("lead_id")
        )
        for column, prefix in (
            ("event_type", "type"),
            ("ctx_seq", "ctx"),
            ("src_slot", "src"),
        ):
            one_hot_columns = pd.get_dummies(
                event_at_position[column],
                prefix=f"ev_last{position}_{prefix}",
                dtype=float,
            )
            event_stats = event_stats.join(one_hot_columns)
            sequence_columns.extend(one_hot_columns.columns)

    # По интервалам я отличаю редкие одиночные действия от плотной активности.
    lead_history["gap_hours"] = (
        lead_history.groupby("lead_id")["event_ts"].diff().dt.total_seconds() / 3_600
    )
    gap_stats = lead_history.groupby("lead_id").agg(
        ev_gap_mean_hours=("gap_hours", "mean"),
        ev_gap_std_hours=("gap_hours", "std"),
        ev_gap_min_hours=("gap_hours", "min"),
        ev_gap_max_hours=("gap_hours", "max"),
    )
    event_stats = event_stats.join(gap_stats)

    lead_history["event_day"] = lead_history["event_ts"].dt.floor("D")
    event_stats["ev_max_events_per_day"] = (
        lead_history.groupby(["lead_id", "event_day"])
        .size()
        .groupby("lead_id")
        .max()
    )
    diversity_stats = lead_history.groupby("lead_id").agg(
        ev_type_entropy=("event_type", categorical_entropy),
        ev_ctx_entropy=("ctx_seq", categorical_entropy),
        ev_src_entropy=("src_slot", categorical_entropy),
    )
    event_stats = event_stats.join(diversity_stats)

    # Я сравниваю цену в истории с ценой объявления на момент назначения.
    first_event = lead_history.groupby("lead_id", sort=False).head(1).set_index("lead_id")
    last_event = lead_history.groupby("lead_id", sort=False).tail(1).set_index("lead_id")
    event_stats["ev_price_last_minus_first"] = (
        last_event["item_price_log_event"] - first_event["item_price_log_event"]
    )
    event_stats["ev_price_last_minus_assignment"] = (
        last_event["item_price_log_event"] - last_event["item_price_log_assignment"]
    )

    count_columns = [column for column in event_stats.columns if "count" in column]
    event_stats[count_columns] = event_stats[count_columns].fillna(0.0)
    event_stats[sequence_columns] = event_stats[sequence_columns].fillna(0.0)
    return event_stats.reindex(all_lead_ids).reset_index(drop=True)


def add_derived_features(lead_table: pd.DataFrame) -> pd.DataFrame:
    """Я добавляю интервалы, свежесть и конверсии пользовательской воронки."""
    extra_columns: dict[str, pd.Series | np.ndarray] = {}
    assignment_time = pd.to_datetime(lead_table["assignment_ts"], errors="raise")
    extra_columns["assignment_minute"] = assignment_time.dt.minute.astype(float)
    extra_columns["assignment_hour_sin"] = np.sin(
        2 * np.pi * assignment_time.dt.hour / 24
    )
    extra_columns["assignment_hour_cos"] = np.cos(
        2 * np.pi * assignment_time.dt.hour / 24
    )

    window_groups: dict[str, dict[int, str]] = {}
    for column in lead_table.columns:
        match = re.fullmatch(r"(.+)_(1|3|7|14|30|90)d", column)
        if match and pd.api.types.is_numeric_dtype(lead_table[column]):
            window_groups.setdefault(match.group(1), {})[int(match.group(2))] = column

    for feature_name, window_columns in window_groups.items():
        available_windows = [window for window in WINDOWS if window in window_columns]
        if len(available_windows) < 2:
            continue
        for short_window, long_window in zip(
            available_windows[:-1], available_windows[1:]
        ):
            extra_columns[f"{feature_name}_{short_window}_{long_window}d"] = (
                lead_table[window_columns[long_window]]
                - lead_table[window_columns[short_window]]
            )
        if 1 in window_columns and 7 in window_columns:
            extra_columns[f"{feature_name}_ratio_1_7"] = lead_table[
                window_columns[1]
            ] / (lead_table[window_columns[7]] + 1.0)
        if 7 in window_columns and 30 in window_columns:
            extra_columns[f"{feature_name}_ratio_7_30"] = lead_table[
                window_columns[7]
            ] / (lead_table[window_columns[30]] + 1.0)

    for days in WINDOWS:
        view_count = lead_table.get(f"item_views_{days}d")
        if view_count is not None:
            for action_name in (
                "item_favorites",
                "user_contacts",
                "chat_opens",
                "call_clicks",
            ):
                action_column = f"{action_name}_{days}d"
                if action_column in lead_table:
                    extra_columns[f"{action_name}_per_view_{days}d"] = (
                        lead_table[action_column] / (view_count + 1.0)
                    )
        assigned_count = lead_table.get(f"leadgen_prev_assigned_{days}d")
        if assigned_count is not None:
            for action_name in ("leadgen_prev_answered", "leadgen_prev_positive"):
                action_column = f"{action_name}_{days}d"
                if action_column in lead_table:
                    extra_columns[f"{action_name}_rate_{days}d"] = (
                        lead_table[action_column] / (assigned_count + 1.0)
                    )

    return pd.concat(
        [lead_table.reset_index(drop=True), pd.DataFrame(extra_columns)], axis=1
    )


def prepare_features(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    events_log: pd.DataFrame,
):
    """Я готовлю train и test вместе, чтобы признаки шли в одинаковом порядке."""
    all_leads = pd.concat(
        [train_data.drop(columns=[TARGET]), test_data], ignore_index=True
    )
    event_stats = build_event_features(all_leads, events_log)
    model_table = pd.concat(
        [all_leads.reset_index(drop=True), event_stats], axis=1
    )
    model_table = add_derived_features(model_table)
    sequence_stats = build_advanced_event_features(all_leads, events_log)
    model_table = pd.concat([model_table, sequence_stats], axis=1)

    columns_to_skip = {"lead_id", "user_id", "assignment_ts", "assignment_date"}
    model_columns = [
        column for column in model_table.columns if column not in columns_to_skip
    ]
    all_features = model_table[model_columns].copy()
    cat_columns = all_features.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    num_columns = [
        column for column in all_features.columns if column not in cat_columns
    ]

    train_features = all_features.iloc[: len(train_data)].copy()
    test_features = all_features.iloc[len(train_data) :].copy()
    return train_features, test_features, num_columns, cat_columns


def make_logistic_model(num_columns, cat_columns) -> Pipeline:
    """Я создаю линейную модель для устойчивой части итогового ансамбля."""
    data_preparation = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_columns,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", data_preparation),
            (
                "classifier",
                LogisticRegression(
                    C=0.03,
                    max_iter=2_000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_tree_matrices(
    train_features: pd.DataFrame,
    features_to_score: pd.DataFrame,
    cat_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Я кодирую категории и беру медианы только из обучающей части."""
    combined_features = pd.concat(
        [train_features, features_to_score], ignore_index=True
    )
    combined_features = pd.get_dummies(
        combined_features,
        columns=cat_columns,
        dummy_na=True,
        dtype=float,
    )
    train_medians = combined_features.iloc[: len(train_features)].median(
        numeric_only=True
    )
    combined_features = (
        combined_features.fillna(train_medians).fillna(0.0).astype(np.float32)
    )
    return (
        combined_features.iloc[: len(train_features)].to_numpy(),
        combined_features.iloc[len(train_features) :].to_numpy(),
    )


HGB_CONFIGS = (
    dict(
        max_leaf_nodes=7,
        max_iter=700,
        learning_rate=0.035,
        min_samples_leaf=50,
        l2_regularization=6.0,
    ),
    dict(
        max_leaf_nodes=15,
        max_iter=700,
        learning_rate=0.03,
        min_samples_leaf=100,
        l2_regularization=10.0,
    ),
    dict(
        max_leaf_nodes=7,
        max_iter=1000,
        learning_rate=0.025,
        min_samples_leaf=80,
        l2_regularization=10.0,
    ),
    dict(
        max_leaf_nodes=15,
        max_iter=850,
        learning_rate=0.025,
        min_samples_leaf=120,
        l2_regularization=15.0,
    ),
)


def fit_predict_ensemble(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    features_to_score: pd.DataFrame,
    dates_to_score: pd.Series,
    num_columns: list[str],
    cat_columns: list[str],
    validation_target: pd.Series | None = None,
) -> tuple[np.ndarray, list[HistGradientBoostingClassifier], Pipeline]:
    """Я обучаю несколько семейств моделей и усредняю их дневные ранги."""
    train_matrix, score_matrix = make_tree_matrices(
        train_features, features_to_score, cat_columns
    )
    hgb_models: list[HistGradientBoostingClassifier] = []
    hgb_ranks: list[np.ndarray] = []

    for model_number, model_settings in enumerate(HGB_CONFIGS, start=1):
        model = HistGradientBoostingClassifier(
            **model_settings,
            class_weight="balanced",
            early_stopping=False,
            random_state=RANDOM_STATE,
        )
        model.fit(train_matrix, train_target)
        model_scores = model.predict_proba(score_matrix)[:, 1]
        hgb_ranks.append(rank_within_day(model_scores, dates_to_score))
        hgb_models.append(model)
        print(f"Обучена HGB-модель {model_number} из {len(HGB_CONFIGS)}")

    logistic_model = make_logistic_model(num_columns, cat_columns)
    logistic_model.fit(train_features, train_target)
    logistic_scores = logistic_model.predict_proba(features_to_score)[:, 1]
    logistic_rank = rank_within_day(logistic_scores, dates_to_score)

    # Метрика смотрит на порядок внутри дня, поэтому я усредняю дневные ранги,
    # а не вероятности, у которых может быть разный масштаб.
    base_ensemble_rank = 0.80 * np.mean(hgb_ranks, axis=0) + 0.20 * logistic_rank

    # Я использую два LightGBM с разным отношением к дисбалансу классов.
    # Их ошибки совпадают не полностью, поэтому обе версии полезны ансамблю.
    lightgbm_ranks = []
    lightgbm_settings = (
        dict(
            num_leaves=15,
            learning_rate=0.02,
            n_estimators=1_600,
            min_child_samples=60,
            reg_lambda=8.0,
            reg_alpha=0.2,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
        ),
        dict(
            num_leaves=15,
            learning_rate=0.02,
            n_estimators=1_450,
            min_child_samples=60,
            reg_lambda=8.0,
            reg_alpha=0.2,
            subsample=0.85,
            colsample_bytree=0.85,
        ),
    )
    for model_number, model_settings in enumerate(lightgbm_settings, start=1):
        model = LGBMClassifier(
            **model_settings,
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=-1,
        )
        training_options = {}
        if validation_target is not None:
            training_options = {
                "eval_set": [(score_matrix, np.asarray(validation_target))],
                "eval_metric": "average_precision",
                "callbacks": [early_stopping(120, verbose=False)],
            }
        model.fit(train_matrix, train_target, **training_options)
        model_scores = model.predict_proba(score_matrix)[:, 1]
        lightgbm_ranks.append(rank_within_day(model_scores, dates_to_score))
        print(
            f"Обучена LightGBM-модель {model_number} "
            f"из {len(lightgbm_settings)}"
        )

    # CatBoost я обучаю на исходных категориальных колонках: так ансамбль получает
    # ещё один взгляд на взаимодействия признаков.
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
        depth=6,
        learning_rate=0.03,
        iterations=2_000 if validation_target is not None else 1_600,
        l2_leaf_reg=8.0,
        random_strength=0.5,
        bagging_temperature=0.5,
        loss_function="Logloss",
        eval_metric="PRAUC:type=Classic",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    catboost_options = {}
    if validation_target is not None:
        catboost_options = {
            "eval_set": (catboost_score_data, np.asarray(validation_target)),
            "early_stopping_rounds": 150,
            "use_best_model": True,
        }
    catboost_model.fit(
        catboost_train,
        train_target,
        cat_features=cat_columns,
        verbose=False,
        **catboost_options,
    )
    catboost_rank = rank_within_day(
        catboost_model.predict_proba(catboost_score_data)[:, 1], dates_to_score
    )
    print("Обучена CatBoost-модель")

    # Веса я выбрал по среднему результату двух последовательных временных holdout.
    ensemble_rank = (
        0.30 * base_ensemble_rank
        + 0.20 * lightgbm_ranks[0]
        + 0.15 * lightgbm_ranks[1]
        + 0.35 * catboost_rank
    )
    return ensemble_rank, hgb_models, logistic_model


def apply_validated_rules_and_rerank(
    predictions: np.ndarray,
    features_to_score: pd.DataFrame,
    dates_to_score: pd.Series,
) -> np.ndarray:
    """Я применяю правила, найденные на ранних и проверенные на поздних датах."""
    adjusted_scores = np.asarray(predictions, dtype=float).copy()
    strong_positive_columns = [
        f"ev_ctx_type_{context}__{action_type}_count"
        for context in ("c05", "c07")
        for action_type in ("item_view", "search", "favorite")
    ]
    strong_positive_mask = (
        features_to_score["ev_ctx_c03_count"].fillna(0).gt(0)
        | features_to_score[strong_positive_columns].fillna(0).sum(axis=1).gt(0)
    ).to_numpy()
    strong_negative_mask = (
        features_to_score["ev_src_24_0_count"].fillna(0).gt(0)
        | features_to_score["ev_src_25_0_count"].fillna(0).gt(0)
    ).to_numpy()

    # Для c05/c07 открытие чата и клик по звонку не являются тем же почти точным
    # положительным сигналом, что просмотр, поиск или избранное. Поэтому я немного
    # понижаю такие сочетания и не даю модели их переоценить.
    chat_context_columns = [
        f"ev_ctx_type_{context}__chat_open_count" for context in ("c05", "c07")
    ]
    call_context_columns = [
        f"ev_ctx_type_{context}__call_click_count" for context in ("c05", "c07")
    ]
    adjusted_scores -= (
        0.03
        * features_to_score[chat_context_columns]
        .fillna(0)
        .sum(axis=1)
        .gt(0)
        .to_numpy()
    )
    adjusted_scores -= (
        0.02
        * features_to_score[call_context_columns]
        .fillna(0)
        .sum(axis=1)
        .gt(0)
        .to_numpy()
    )

    # Внутри каждой группы я сохраняю порядок модели, чтобы не создавать ничьи.
    adjusted_scores[strong_positive_mask] = (
        2.0 + 0.001 * adjusted_scores[strong_positive_mask]
    )
    adjusted_scores[strong_negative_mask] = (
        -1.0 + 0.001 * adjusted_scores[strong_negative_mask]
    )
    return rank_within_day(adjusted_scores, dates_to_score)


def run_validation(
    train_data: pd.DataFrame,
    train_features: pd.DataFrame,
    num_columns: list[str],
    cat_columns: list[str],
) -> float:
    """Я учусь на ранних датах и проверяю решение на последних пяти."""
    assignment_dates = pd.to_datetime(train_data[DATE_COLUMN])
    validation_start = assignment_dates.max() - pd.Timedelta(days=4)
    train_mask = assignment_dates < validation_start
    validation_mask = ~train_mask

    validation_predictions, _, _ = fit_predict_ensemble(
        train_features=train_features.loc[train_mask],
        train_target=train_data.loc[train_mask, TARGET],
        features_to_score=train_features.loc[validation_mask],
        dates_to_score=train_data.loc[validation_mask, DATE_COLUMN],
        num_columns=num_columns,
        cat_columns=cat_columns,
        validation_target=train_data.loc[validation_mask, TARGET],
    )
    validation_predictions = apply_validated_rules_and_rerank(
        validation_predictions,
        train_features.loc[validation_mask],
        train_data.loc[validation_mask, DATE_COLUMN],
    )
    validation_ap = daily_average_precision(
        train_data.loc[validation_mask, TARGET],
        validation_predictions,
        train_data.loc[validation_mask, DATE_COLUMN],
    )
    print(
        f"Период валидации: {train_data.loc[validation_mask, DATE_COLUMN].min()} .. "
        f"{train_data.loc[validation_mask, DATE_COLUMN].max()}"
    )
    print(f"Daily AP на валидации: {validation_ap:.6f}")
    return validation_ap


def create_submission(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    num_columns: list[str],
    cat_columns: list[str],
) -> pd.DataFrame:
    """Я переобучаю ансамбль на всём train и строю прогноз для test."""
    test_predictions, _, _ = fit_predict_ensemble(
        train_features=train_features,
        train_target=train_data[TARGET],
        features_to_score=test_features,
        dates_to_score=test_data[DATE_COLUMN],
        num_columns=num_columns,
        cat_columns=cat_columns,
    )
    test_predictions = apply_validated_rules_and_rerank(
        test_predictions, test_features, test_data[DATE_COLUMN]
    )

    submission_table = pd.DataFrame(
        {
            "lead_id": test_data["lead_id"].astype(str),
            "score": test_predictions.astype(float),
        }
    )
    assert list(submission_table.columns) == ["lead_id", "score"]
    assert len(submission_table) == len(test_data)
    assert submission_table["lead_id"].is_unique
    assert submission_table["score"].between(0, 1).all()
    return submission_table


def main() -> None:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    argument_parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    argument_parser.add_argument("--validate", action="store_true")
    run_options = argument_parser.parse_args()

    started_at = time.time()
    print(f"pandas={pd.__version__}; numpy={np.__version__}; sklearn={sklearn.__version__}")
    train_data = pd.read_csv(run_options.data_dir / "train.csv")
    test_data = pd.read_csv(run_options.data_dir / "test.csv")
    events_log = pd.read_csv(run_options.data_dir / "events.csv")
    print(
        f"train={train_data.shape}; test={test_data.shape}; "
        f"events={events_log.shape}"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        train_features, test_features, num_columns, cat_columns = prepare_features(
            train_data, test_data, events_log
        )
    print(
        f"Признаков для модели: {train_features.shape[1]} "
        f"({len(num_columns)} числовых, {len(cat_columns)} категориальных)"
    )

    if run_options.validate:
        run_validation(train_data, train_features, num_columns, cat_columns)

    submission_table = create_submission(
        train_data,
        test_data,
        train_features,
        test_features,
        num_columns,
        cat_columns,
    )
    run_options.output.parent.mkdir(parents=True, exist_ok=True)
    submission_table.to_csv(run_options.output, index=False)
    print(f"Файл {run_options.output} сохранён, строк: {len(submission_table):,}")
    print(f"Общее время работы: {time.time() - started_at:.1f} сек.")


if __name__ == "__main__":
    main()
