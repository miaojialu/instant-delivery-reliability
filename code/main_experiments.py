# -*- coding: utf-8 -*-
"""Reproducible LightGBM workflow for the Beijing and Baoding analyses.

The script expects the two cleaned order-level data files and
``lightgbm_utils.py`` in the project directory. Input and output paths
can be overridden with ``DELIVERY_BJ_DATA``, ``DELIVERY_BD_DATA``, and
``DELIVERY_OUTPUT_DIR``. 
"""

# In[]:
# =============================================================================
# 1. Model development and grouped validation
# =============================================================================

import importlib.metadata as gcv_metadata
import json as gcv_json
import os as gcv_os
import platform as gcv_platform
import random as gcv_random
from dataclasses import dataclass as gcv_dataclass
from datetime import datetime as gcv_datetime
from pathlib import Path as GCVPath

import lightgbm as gcv_lgb
import matplotlib.pyplot as gcv_plt
import numpy as gcv_np
import optuna as gcv_optuna
import pandas as gcv_pd
import sklearn as gcv_sklearn
from sklearn.metrics import mean_absolute_error as gcv_mae
from sklearn.metrics import mean_squared_error as gcv_mse
from sklearn.metrics import r2_score as gcv_r2
from sklearn.model_selection import GroupKFold as GCVGroupKFold
from sklearn.model_selection import GroupShuffleSplit as GCVGroupShuffleSplit


GCV_ROOT = (
    GCVPath(__file__).resolve().parent
    if "__file__" in globals()
    else GCVPath.cwd()
)


def gcv_env_path(name, default):
    value = gcv_os.environ.get(name)
    return GCVPath(value).expanduser().resolve() if value else GCVPath(default)


GCV_DATA_FILES = {
    "BJ": gcv_env_path("DELIVERY_BJ_DATA", GCV_ROOT / "北京订单数据20261019.csv"),
    "BD": gcv_env_path("DELIVERY_BD_DATA", GCV_ROOT / "保定订单数据20261019.csv"),
}
GCV_CITIES = ("BJ", "BD")
GCV_FEATURES = (
    "order_hourly",
    "is_weekend",
    "peak",
    "ride_distance_km",
    "merchant_preparation_time",
    "wave_duration_time",
    "loadorder_number",
    "courier_work_hours_now",
    "platform_assign_time",
    "demand_density",
    "supply_density",
    "road_length_km",
    "avg_node_degree",
    "prev_standardized_delay",
)
GCV_TARGET = "standardized_delay"
GCV_GROUP_COLUMNS = {
    "BJ": ("Date", "courier_id", "wave_index"),
    "BD": ("dt", "courier_id", "wave_id"),
}
GCV_ORDER_ID_COLUMNS = {"BJ": "source_row_id", "BD": "waybill_id"}
GCV_LAG_SOURCE = {"BJ": "prev_standardized_delay", "BD": "prior_order_delay"}

GCV_TEST_SIZE = 0.20
GCV_N_SPLITS = 5
GCV_OUTER_SPLIT_SEED = 42
GCV_PYTHON_RANDOM_SEED = 20260826
GCV_NUMPY_SEED = 20260826
GCV_OPTUNA_SEED = 20260826
GCV_LIGHTGBM_SEED = 20260826
GCV_BOOTSTRAP_SEEDS = {"BJ": 20260826, "BD": 20260827}
GCV_N_TRIALS = 50
GCV_MAX_BOOST_ROUNDS = 2000
GCV_EARLY_STOPPING_ROUNDS = 100
GCV_BOOTSTRAP_REPLICATES = 1000
GCV_NUM_THREADS = 8
GCV_FORCE_RETRAIN = False
GCV_REPEATABILITY_CHECK = True

GCV_OUTPUT_ROOT = gcv_env_path(
    "DELIVERY_OUTPUT_DIR",
    GCV_ROOT / "outputs" / "grouped_5fold_lightgbm",
)
GCV_MODEL_ROOT = GCV_OUTPUT_ROOT / "models"
GCV_RESULT_ROOT = GCV_OUTPUT_ROOT / "results"
GCV_METADATA_FILE = GCV_MODEL_ROOT / "model_metadata.json"
GCV_SPLIT_ROOT = GCV_OUTPUT_ROOT / "shared_sample_split"
GCV_ORDER_SPLIT_FILE = GCV_SPLIT_ROOT / "analysis_sample_split.csv"
GCV_WAVE_SPLIT_FILE = GCV_SPLIT_ROOT / "analysis_wave_split.csv"
GCV_SPLIT_METADATA_FILE = GCV_SPLIT_ROOT / "split_metadata.json"
GCV_FORCE_REBUILD_SHARED_SPLIT = False
GCV_PROTOCOL_VERSION = "coordinate_update_20261002_date_courier_wave_shared_split_v2"


@gcv_dataclass
class GCVSplit:
    """一个城市的 development/test 数据与分组信息。"""

    X_development: gcv_pd.DataFrame
    X_test: gcv_pd.DataFrame
    y_development: gcv_pd.Series
    y_test: gcv_pd.Series
    groups_development: gcv_pd.Series
    groups_test: gcv_pd.Series
    cv_fold_development: gcv_pd.Series
    test_source_index: gcv_pd.Index
    n_cleaned_total: int
    n_target_missing_removed: int
    group_key_description: str


def gcv_set_global_seeds():
    """设置 Python 和 NumPy 全局种子；其他程序使用显式局部种子。"""

    gcv_os.environ["PYTHONHASHSEED"] = str(GCV_PYTHON_RANDOM_SEED)
    gcv_random.seed(GCV_PYTHON_RANDOM_SEED)
    gcv_np.random.seed(GCV_NUMPY_SEED)


def gcv_atomic_to_csv(frame, path):
    """原子保存CSV，避免中断后留下不完整的共享划分表。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False, encoding="utf-8-sig")
    gcv_os.replace(temporary_path, path)


def gcv_atomic_write_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        gcv_json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    gcv_os.replace(temporary_path, path)


def gcv_canonical_component(series, city, column):
    """将分组字段转为稳定字符串；分组键不允许缺失。"""

    if series.isna().any():
        raise ValueError(f"{city}: 分组字段 {column} 存在缺失，不能生成 analysis_wave_id")
    if gcv_pd.api.types.is_numeric_dtype(series):
        numeric = gcv_pd.to_numeric(series, errors="raise")
        if gcv_np.all(gcv_np.isclose(numeric, gcv_np.round(numeric))):
            return gcv_pd.Series(
                gcv_np.round(numeric).astype(gcv_np.int64).astype(str),
                index=series.index,
                dtype="string",
            )
    return series.astype("string").str.strip()


def gcv_analysis_wave_id(frame, city):
    """生成跨城市唯一且可审计的 date-courier-wave 标识。"""

    parts = [
        gcv_canonical_component(frame[column], city, column)
        for column in GCV_GROUP_COLUMNS[city]
    ]
    result = gcv_pd.Series(city, index=frame.index, dtype="string")
    for part in parts:
        result = result + "::" + part
    return result.rename("analysis_wave_id")


def gcv_order_id(frame, city):
    column = GCV_ORDER_ID_COLUMNS[city]
    values = gcv_canonical_component(frame[column], city, column)
    if values.duplicated().any():
        raise ValueError(f"{city}: 稳定订单键 {column} 不唯一")
    return values.rename("order_id")


def gcv_split_signature():
    return {
        "protocol_version": GCV_PROTOCOL_VERSION,
        "group_columns": {
            city: list(GCV_GROUP_COLUMNS[city]) for city in GCV_CITIES
        },
        "order_id_columns": GCV_ORDER_ID_COLUMNS,
        "target": GCV_TARGET,
        "test_size": GCV_TEST_SIZE,
        "outer_split_seed": GCV_OUTER_SPLIT_SEED,
        "n_splits": GCV_N_SPLITS,
        "data": {
            city: {
                "path": str(GCV_DATA_FILES[city]),
                "size": GCV_DATA_FILES[city].stat().st_size,
                "mtime_ns": GCV_DATA_FILES[city].stat().st_mtime_ns,
            }
            for city in GCV_CITIES
        },
    }


def gcv_build_shared_split_tables():
    """只在此处生成外层样本划分和development五折编号。"""

    order_parts = []
    wave_parts = []
    for city in GCV_CITIES:
        group_columns = list(GCV_GROUP_COLUMNS[city])
        order_column = GCV_ORDER_ID_COLUMNS[city]
        required = [order_column, GCV_TARGET, *group_columns]
        frame = gcv_pd.read_csv(
            GCV_DATA_FILES[city], usecols=list(dict.fromkeys(required))
        )
        frame["source_row_number"] = frame.index.astype(gcv_np.int64)
        frame = frame.loc[frame[GCV_TARGET].notna()].copy()
        frame["order_id"] = gcv_order_id(frame, city)
        frame["analysis_wave_id"] = gcv_analysis_wave_id(frame, city)

        splitter = GCVGroupShuffleSplit(
            n_splits=1,
            test_size=GCV_TEST_SIZE,
            random_state=GCV_OUTER_SPLIT_SEED,
        )
        development_position, test_position = next(
            splitter.split(frame, groups=frame["analysis_wave_id"])
        )
        frame["sample_split"] = ""
        frame.iloc[
            development_position,
            frame.columns.get_loc("sample_split"),
        ] = "development"
        frame.iloc[
            test_position,
            frame.columns.get_loc("sample_split"),
        ] = "test"

        frame["cv_fold"] = gcv_pd.Series(
            gcv_pd.NA, index=frame.index, dtype="Int64"
        )
        development_frame = frame.iloc[development_position]
        cv_splitter = GCVGroupKFold(n_splits=GCV_N_SPLITS)
        for fold_number, (_, valid_local_position) in enumerate(
            cv_splitter.split(
                development_frame,
                groups=development_frame["analysis_wave_id"],
            ),
            start=1,
        ):
            valid_global_position = development_position[valid_local_position]
            frame.iloc[
                valid_global_position,
                frame.columns.get_loc("cv_fold"),
            ] = fold_number

        if frame.loc[frame["sample_split"] == "development", "cv_fold"].isna().any():
            raise AssertionError(f"{city}: 部分development订单未获得CV fold")
        if frame.loc[frame["sample_split"] == "test", "cv_fold"].notna().any():
            raise AssertionError(f"{city}: test订单不应获得CV fold")

        order_table = frame[
            [
                "order_id", "source_row_number", *group_columns,
                "analysis_wave_id", "sample_split", "cv_fold",
            ]
        ].copy()
        order_table.insert(0, "city", city)
        order_parts.append(order_table)

        consistency = frame.groupby("analysis_wave_id").agg(
            n_split=("sample_split", "nunique"),
            n_fold=("cv_fold", lambda value: value.dropna().nunique()),
        )
        if (consistency["n_split"] != 1).any() or (consistency["n_fold"] > 1).any():
            raise AssertionError(f"{city}: 同一analysis_wave_id出现多个划分结果")
        wave_table = frame[
            [*group_columns, "analysis_wave_id", "sample_split", "cv_fold"]
        ].drop_duplicates("analysis_wave_id")
        wave_counts = frame.groupby("analysis_wave_id").size().rename("n_orders")
        wave_table = wave_table.join(
            wave_counts, on="analysis_wave_id", validate="one_to_one"
        )
        wave_table.insert(0, "city", city)
        wave_parts.append(wave_table)

    order_split = gcv_pd.concat(order_parts, ignore_index=True)
    wave_split = gcv_pd.concat(wave_parts, ignore_index=True)
    if wave_split["analysis_wave_id"].duplicated().any():
        raise AssertionError("analysis_wave_id未做到跨城市唯一")
    if order_split["order_id"].duplicated().any():
        # Order identifiers are unique only within a city.
        duplicated_within_city = order_split.duplicated(["city", "order_id"])
        if duplicated_within_city.any():
            raise AssertionError("city + order_id在共享划分表中不唯一")

    gcv_atomic_to_csv(order_split, GCV_ORDER_SPLIT_FILE)
    gcv_atomic_to_csv(wave_split, GCV_WAVE_SPLIT_FILE)
    gcv_atomic_write_json(gcv_split_signature(), GCV_SPLIT_METADATA_FILE)
    print(f"共享订单划分表已保存：{GCV_ORDER_SPLIT_FILE}")
    print(f"共享波次划分表已保存：{GCV_WAVE_SPLIT_FILE}")


def gcv_ensure_shared_split_tables():
    global GCV_FORCE_REBUILD_SHARED_SPLIT
    required_files = (
        GCV_ORDER_SPLIT_FILE,
        GCV_WAVE_SPLIT_FILE,
        GCV_SPLIT_METADATA_FILE,
    )
    if GCV_FORCE_REBUILD_SHARED_SPLIT or not all(path.exists() for path in required_files):
        gcv_build_shared_split_tables()
        GCV_FORCE_REBUILD_SHARED_SPLIT = False
        return
    saved = gcv_json.loads(GCV_SPLIT_METADATA_FILE.read_text(encoding="utf-8"))
    current = gcv_split_signature()
    if saved != current:
        raise ValueError(
            "现有共享样本划分与当前数据或分组设置不一致。"
            "请人工核查后将 GCV_FORCE_REBUILD_SHARED_SPLIT 设为 True 重新生成。"
        )


def gcv_load_city_assignment(city):
    gcv_ensure_shared_split_tables()
    assignment = gcv_pd.read_csv(
        GCV_ORDER_SPLIT_FILE,
        dtype={
            "city": "string",
            "order_id": "string",
            "analysis_wave_id": "string",
            "sample_split": "string",
        },
    )
    assignment = assignment.loc[assignment["city"] == city].copy()
    if assignment.empty:
        raise ValueError(f"共享样本划分表中没有 {city}")
    if assignment["order_id"].duplicated().any():
        raise ValueError(f"共享样本划分表中 {city} order_id 不唯一")
    return assignment.set_index("order_id", verify_integrity=True)


def gcv_prepare_outer_split(city):
    """读取唯一共享表，不在训练函数中重新划分development/test。"""

    data_path = GCV_DATA_FILES[city]
    group_columns = list(GCV_GROUP_COLUMNS[city])
    order_column = GCV_ORDER_ID_COLUMNS[city]
    lag_source = GCV_LAG_SOURCE[city]
    source_features = [
        lag_source if feature == "prev_standardized_delay" else feature
        for feature in GCV_FEATURES
    ]
    required = [*source_features, GCV_TARGET, order_column, *group_columns]
    available = gcv_pd.read_csv(data_path, nrows=0).columns.tolist()
    missing = [column for column in required if column not in available]
    if missing:
        raise KeyError(f"{city} 数据缺少字段：{missing}")

    frame = gcv_pd.read_csv(data_path, usecols=required)
    frame["source_row_number"] = frame.index.astype(gcv_np.int64)
    if lag_source != "prev_standardized_delay":
        frame = frame.rename(columns={lag_source: "prev_standardized_delay"})
    target_missing = int(frame[GCV_TARGET].isna().sum())
    frame = frame.loc[frame[GCV_TARGET].notna()].copy()
    frame["order_id"] = gcv_order_id(frame, city)
    frame["analysis_wave_id"] = gcv_analysis_wave_id(frame, city)
    assignment = gcv_load_city_assignment(city)
    if set(frame["order_id"]) != set(assignment.index):
        raise ValueError(f"{city}: 当前分析订单与共享样本划分表不一致")
    mapped = assignment.reindex(frame["order_id"].to_numpy())
    mapped.index = frame.index
    if not mapped["analysis_wave_id"].equals(frame["analysis_wave_id"]):
        mismatch = int(
            (mapped["analysis_wave_id"] != frame["analysis_wave_id"]).sum()
        )
        raise ValueError(f"{city}: {mismatch}条订单的analysis_wave_id与共享表不一致")
    frame["sample_split"] = mapped["sample_split"]
    frame["cv_fold"] = mapped["cv_fold"].astype("Int64")
    X = frame.loc[:, list(GCV_FEATURES)].copy()
    y = frame[GCV_TARGET].copy()
    group_key = frame["analysis_wave_id"]
    development_position = gcv_np.flatnonzero(
        frame["sample_split"].eq("development").to_numpy()
    )
    test_position = gcv_np.flatnonzero(
        frame["sample_split"].eq("test").to_numpy()
    )
    split = GCVSplit(
        X_development=X.iloc[development_position].copy(),
        X_test=X.iloc[test_position].copy(),
        y_development=y.iloc[development_position].copy(),
        y_test=y.iloc[test_position].copy(),
        groups_development=group_key.iloc[development_position].copy(),
        groups_test=group_key.iloc[test_position].copy(),
        cv_fold_development=frame["cv_fold"].iloc[development_position].copy(),
        test_source_index=gcv_pd.Index(
            frame["source_row_number"].iloc[test_position].to_numpy()
        ),
        n_cleaned_total=len(frame),
        n_target_missing_removed=target_missing,
        group_key_description=" + ".join(group_columns),
    )

    repeated_groups = set(split.groups_development.unique()) & set(
        split.groups_test.unique()
    )
    if repeated_groups:
        raise AssertionError(
            f"{city}: development/test 存在 {len(repeated_groups)} 个重复组"
        )
    return split


def gcv_build_folds(city, split):
    """读取共享表中固定的五折编号并审计泄漏。"""

    fold_number_series = split.cv_fold_development.astype("Int64")
    observed_folds = sorted(fold_number_series.dropna().astype(int).unique())
    expected_folds = list(range(1, GCV_N_SPLITS + 1))
    if observed_folds != expected_folds:
        raise ValueError(
            f"{city}: 共享划分表中的CV folds为{observed_folds}，预期{expected_folds}"
        )
    positions = gcv_np.arange(len(fold_number_series))
    folds = [
        (
            positions[fold_number_series.to_numpy(dtype=int) != fold_number],
            positions[fold_number_series.to_numpy(dtype=int) == fold_number],
        )
        for fold_number in expected_folds
    ]
    audit_rows = []
    validation_group_occurrences = {}
    for fold_number, (train_position, valid_position) in enumerate(folds, start=1):
        train_groups = set(split.groups_development.iloc[train_position].unique())
        valid_groups = set(split.groups_development.iloc[valid_position].unique())
        overlap = train_groups & valid_groups
        if overlap:
            raise AssertionError(
                f"{city} fold {fold_number}: train/validation 存在重复组"
            )
        for group in valid_groups:
            validation_group_occurrences[group] = (
                validation_group_occurrences.get(group, 0) + 1
            )
        audit_rows.append(
            {
                "City": city,
                "Fold": fold_number,
                "CV_implementation": f"GroupKFold(n_splits={GCV_N_SPLITS})",
                "Grouping_key": split.group_key_description,
                "n_fold_train_orders": len(train_position),
                "n_fold_validation_orders": len(valid_position),
                "n_fold_train_groups": len(train_groups),
                "n_fold_validation_groups": len(valid_groups),
                "train_validation_repeated_groups": len(overlap),
            }
        )
    invalid_counts = {
        group: count
        for group, count in validation_group_occurrences.items()
        if count != 1
    }
    if invalid_counts:
        raise AssertionError(
            f"{city}: {len(invalid_counts)} 个 development 组未且仅进入一个 validation fold"
        )
    if len(validation_group_occurrences) != split.groups_development.nunique():
        raise AssertionError(f"{city}: 部分 development 组未被CV覆盖")
    return folds, gcv_pd.DataFrame(audit_rows)


def gcv_base_lightgbm_params():
    """返回显式设置的所有固定参数和复现种子。"""

    return {
        "objective": "regression_l2",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "feature_pre_filter": False,
        "seed": GCV_LIGHTGBM_SEED,
        "feature_fraction_seed": GCV_LIGHTGBM_SEED,
        "bagging_seed": GCV_LIGHTGBM_SEED,
        "data_random_seed": GCV_LIGHTGBM_SEED,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": GCV_NUM_THREADS,
    }


def gcv_trial_params(trial):
    """调优参数空间；所有城市使用同一搜索范围。"""

    max_depth = trial.suggest_int("max_depth", 4, 12)
    max_leaves = min(256, 2**max_depth)
    params = {
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.02, 0.12, log=True
        ),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "max_depth": max_depth,
        "num_leaves": trial.suggest_int("num_leaves", 16, max_leaves),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.60, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.60, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
    }
    params.update(gcv_base_lightgbm_params())
    return params


def gcv_train_one_fold(split, train_position, valid_position, params, record=False):
    """训练一个 grouped fold；不接触外层 test。"""

    train_data = gcv_lgb.Dataset(
        split.X_development.iloc[train_position],
        label=split.y_development.iloc[train_position],
        feature_name=list(GCV_FEATURES),
        free_raw_data=False,
    )
    valid_data = gcv_lgb.Dataset(
        split.X_development.iloc[valid_position],
        label=split.y_development.iloc[valid_position],
        feature_name=list(GCV_FEATURES),
        reference=train_data,
        free_raw_data=False,
    )
    history = {}
    callbacks = [
        gcv_lgb.early_stopping(GCV_EARLY_STOPPING_ROUNDS, verbose=False),
        gcv_lgb.log_evaluation(False),
    ]
    valid_sets = [valid_data]
    valid_names = ["validation"]
    if record:
        valid_sets = [train_data, valid_data]
        valid_names = ["train", "validation"]
        callbacks.append(gcv_lgb.record_evaluation(history))
    model = gcv_lgb.train(
        params,
        train_data,
        num_boost_round=GCV_MAX_BOOST_ROUNDS,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    prediction = model.predict(split.X_development.iloc[valid_position])
    rmse = float(
        gcv_np.sqrt(
            gcv_mse(split.y_development.iloc[valid_position], prediction)
        )
    )
    best_iteration = int(model.best_iteration or model.current_iteration())
    return model, rmse, best_iteration, history


def gcv_tune_city(city, split, folds):
    """通过 5-fold 平均 validation RMSE 调参。"""

    gcv_optuna.logging.set_verbosity(gcv_optuna.logging.WARNING)

    def objective(trial):
        params = gcv_trial_params(trial)
        fold_rmse = []
        fold_iterations = []
        for fold_number, (train_position, valid_position) in enumerate(
            folds, start=1
        ):
            _, rmse, best_iteration, _ = gcv_train_one_fold(
                split,
                train_position,
                valid_position,
                params,
                record=False,
            )
            fold_rmse.append(rmse)
            fold_iterations.append(best_iteration)
            trial.set_user_attr(f"fold_{fold_number}_RMSE", rmse)
            trial.set_user_attr(
                f"fold_{fold_number}_best_iteration", best_iteration
            )
        trial.set_user_attr("fold_RMSE_std", float(gcv_np.std(fold_rmse, ddof=1)))
        trial.set_user_attr(
            "median_best_iteration",
            int(gcv_np.rint(gcv_np.median(fold_iterations))),
        )
        return float(gcv_np.mean(fold_rmse))

    sampler = gcv_optuna.samplers.TPESampler(seed=GCV_OPTUNA_SEED)
    study = gcv_optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=f"grouped_5fold_{city}",
    )
    study.enqueue_trial(
        {
            "learning_rate": 0.05,
            "lambda_l1": 1.0,
            "lambda_l2": 1.0,
            "max_depth": 8,
            "num_leaves": 48,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.90,
            "bagging_freq": 1,
            "min_child_samples": 50,
            "min_gain_to_split": 0.0,
        }
    )
    study.optimize(objective, n_trials=GCV_N_TRIALS, n_jobs=1)
    best_params = dict(study.best_trial.params)
    best_params.update(gcv_base_lightgbm_params())
    return study, best_params


def gcv_refit_best_folds(city, split, folds, best_params):
    """Refit the selected parameters and save fold metrics and learning curves."""

    fold_rows = []
    curve_parts = []
    best_iterations = []
    for fold_number, (train_position, valid_position) in enumerate(folds, start=1):
        fold_model, rmse, best_iteration, history = gcv_train_one_fold(
            split,
            train_position,
            valid_position,
            best_params,
            record=True,
        )
        validation_prediction = fold_model.predict(
            split.X_development.iloc[valid_position]
        )
        fold_rows.append(
            {
                "City": city,
                "Fold": fold_number,
                "n_train_orders": len(train_position),
                "n_validation_orders": len(valid_position),
                "Validation_R2": float(
                    gcv_r2(
                        split.y_development.iloc[valid_position],
                        validation_prediction,
                    )
                ),
                "Validation_RMSE": rmse,
                "Validation_MAE": float(
                    gcv_mae(
                        split.y_development.iloc[valid_position],
                        validation_prediction,
                    )
                ),
                "best_iteration": best_iteration,
            }
        )
        best_iterations.append(best_iteration)
        curve_length = len(history["validation"]["rmse"])
        curve_parts.append(
            gcv_pd.DataFrame(
                {
                    "City": city,
                    "Fold": fold_number,
                    "round": gcv_np.arange(1, curve_length + 1),
                    "train_rmse": history["train"]["rmse"],
                    "validation_rmse": history["validation"]["rmse"],
                }
            )
        )
    final_iteration = int(gcv_np.rint(gcv_np.median(best_iterations)))
    return (
        gcv_pd.DataFrame(fold_rows),
        gcv_pd.concat(curve_parts, ignore_index=True),
        final_iteration,
    )


def gcv_plot_learning_curve(curve_table, best_iteration, save_path):
    """沿用原学习曲线的样式，展示5 folds均值。"""

    aggregate = (
        curve_table.groupby("round", as_index=False)
        .agg(
            train_rmse=("train_rmse", "mean"),
            validation_rmse=("validation_rmse", "mean"),
            n_folds_contributing=("Fold", "nunique"),
        )
    )
    figure, axis = gcv_plt.subplots(figsize=(10, 5))
    axis.plot(
        aggregate["round"],
        aggregate["train_rmse"],
        color="#4573b4",
        linewidth=1.5,
        label="Train RMSE",
    )
    axis.plot(
        aggregate["round"],
        aggregate["validation_rmse"],
        color="#d73221",
        linewidth=1.5,
        linestyle="--",
        label="Validation RMSE",
    )
    axis.axvline(
        x=best_iteration,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label=f"Median best round ({best_iteration})",
    )
    axis.set_xlabel("Boosting Round", fontsize=14)
    axis.set_ylabel("RMSE", fontsize=14)
    axis.set_title("LightGBM Learning Curve", fontsize=16)
    axis.legend(fontsize=12)
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    figure.savefig(save_path, bbox_inches="tight", dpi=300)
    gcv_plt.close(figure)
    return aggregate


def gcv_fit_final_model(split, best_params, best_iteration):
    """在完整 development 上使用锁定轮数拟合最终模型。"""

    development_data = gcv_lgb.Dataset(
        split.X_development,
        label=split.y_development,
        feature_name=list(GCV_FEATURES),
        free_raw_data=False,
    )
    return gcv_lgb.train(
        best_params,
        development_data,
        num_boost_round=best_iteration,
        callbacks=[gcv_lgb.log_evaluation(False)],
    )


def gcv_metric_values(y_true, prediction, weights=None):
    """计算 R2/RMSE/MAE；weights 用于分组bootstrap。"""

    y_true = gcv_np.asarray(y_true, dtype=float)
    prediction = gcv_np.asarray(prediction, dtype=float)
    if weights is None:
        return {
            "R2": float(gcv_r2(y_true, prediction)),
            "RMSE": float(gcv_np.sqrt(gcv_mse(y_true, prediction))),
            "MAE": float(gcv_mae(y_true, prediction)),
        }
    weights = gcv_np.asarray(weights, dtype=float)
    weight_sum = float(weights.sum())
    weighted_mean = float(gcv_np.sum(weights * y_true) / weight_sum)
    residual = y_true - prediction
    sse = float(gcv_np.sum(weights * residual**2))
    sst = float(gcv_np.sum(weights * (y_true - weighted_mean) ** 2))
    return {
        "R2": float(1.0 - sse / sst) if sst > 0 else gcv_np.nan,
        "RMSE": float(gcv_np.sqrt(sse / weight_sum)),
        "MAE": float(
            gcv_np.sum(weights * gcv_np.abs(residual)) / weight_sum
        ),
    }


def gcv_cluster_bootstrap_ci(city, split, prediction):
    """按test中的date-courier-wave组进行percentile bootstrap。"""

    y_true = split.y_test.to_numpy(dtype=float)
    group_codes, unique_groups = gcv_pd.factorize(split.groups_test, sort=True)
    n_groups = len(unique_groups)
    rng = gcv_np.random.default_rng(GCV_BOOTSTRAP_SEEDS[city])
    values = {"R2": [], "RMSE": [], "MAE": []}
    for _ in range(GCV_BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, n_groups, size=n_groups)
        group_weights = gcv_np.bincount(sampled, minlength=n_groups)
        order_weights = group_weights[group_codes]
        metrics = gcv_metric_values(y_true, prediction, weights=order_weights)
        for metric in values:
            values[metric].append(metrics[metric])
    result = {}
    for metric, metric_values in values.items():
        lower, upper = gcv_np.nanquantile(metric_values, [0.025, 0.975])
        result[f"{metric}_CI_low"] = float(lower)
        result[f"{metric}_CI_high"] = float(upper)
    return result, n_groups


def gcv_parameter_rows(city, best_params, best_iteration):
    """将调优、固定、派生和重要默认参数逐项展开。"""

    tuned = {
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "feature_fraction",
        "bagging_fraction",
        "bagging_freq",
        "lambda_l1",
        "lambda_l2",
        "min_gain_to_split",
    }
    explicit_order = [
        "objective",
        "boosting_type",
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "feature_fraction",
        "bagging_fraction",
        "bagging_freq",
        "lambda_l1",
        "lambda_l2",
        "min_gain_to_split",
        "metric",
        "feature_pre_filter",
        "deterministic",
        "force_col_wise",
        "num_threads",
        "seed",
        "feature_fraction_seed",
        "bagging_seed",
        "data_random_seed",
        "verbosity",
    ]
    rows = []
    for parameter in explicit_order:
        rows.append(
            {
                "City": city,
                "Parameter": parameter,
                "Value": best_params[parameter],
                "Source": "Optuna grouped 5-fold tuning"
                if parameter in tuned
                else "Explicit fixed setting",
            }
        )
    rows.extend(
        [
            {
                "City": city,
                "Parameter": "num_boost_round_max",
                "Value": GCV_MAX_BOOST_ROUNDS,
                "Source": "Training maximum",
            },
            {
                "City": city,
                "Parameter": "early_stopping_rounds",
                "Value": GCV_EARLY_STOPPING_ROUNDS,
                "Source": "Fold training setting",
            },
            {
                "City": city,
                "Parameter": "final_best_iteration",
                "Value": best_iteration,
                "Source": "Median of five fold-specific best iterations",
            },
            {
                "City": city,
                "Parameter": "max_bin",
                "Value": 255,
                "Source": "LightGBM 4.6.0 default; not tuned",
            },
            {
                "City": city,
                "Parameter": "min_sum_hessian_in_leaf",
                "Value": 0.001,
                "Source": "LightGBM 4.6.0 default; not tuned",
            },
            {
                "City": city,
                "Parameter": "max_delta_step",
                "Value": 0.0,
                "Source": "LightGBM 4.6.0 default; not tuned",
            },
        ]
    )
    return rows


def gcv_environment_table():
    """返回当前实际Python环境的可复现版本信息。"""

    packages = (
        "lightgbm",
        "shap",
        "PyALE",
        "alibi",
        "scikit-learn",
        "pandas",
        "numpy",
        "optuna",
        "matplotlib",
        "seaborn",
    )
    rows = [
        {"Software": "Python", "Version": gcv_platform.python_version()},
        {"Software": "Platform", "Version": gcv_platform.platform()},
    ]
    for package in packages:
        try:
            version = gcv_metadata.version(package)
        except gcv_metadata.PackageNotFoundError:
            version = "NOT_INSTALLED"
        rows.append({"Software": package, "Version": version})
    rows.append(
        {
            "Software": "ALE implementation",
            "Version": "PyALE.ale (1D/2D); low-cardinality 1D manual branch",
        }
    )
    return gcv_pd.DataFrame(rows)


def gcv_seed_inventory():
    """端到端记录所有随机/确定性过程。"""

    return gcv_pd.DataFrame(
        [
            {
                "Process": "development/test GroupShuffleSplit",
                "Seed": GCV_OUTER_SPLIT_SEED,
                "Notes": (
                    "Generated once and persisted; BJ Date+courier_id+wave_index; "
                    "BD dt+courier_id+wave_id"
                ),
            },
            {
                "Process": "development GroupKFold",
                "Seed": "None",
                "Notes": (
                    "Generated once by date-courier-wave and persisted as cv_fold"
                ),
            },
            {
                "Process": "Python random",
                "Seed": GCV_PYTHON_RANDOM_SEED,
                "Notes": "random.seed and PYTHONHASHSEED",
            },
            {
                "Process": "NumPy global RNG",
                "Seed": GCV_NUMPY_SEED,
                "Notes": "np.random.seed",
            },
            {
                "Process": "Optuna TPESampler",
                "Seed": GCV_OPTUNA_SEED,
                "Notes": "n_jobs=1",
            },
            {
                "Process": "LightGBM",
                "Seed": GCV_LIGHTGBM_SEED,
                "Notes": "seed, feature_fraction_seed, bagging_seed, data_random_seed",
            },
            {
                "Process": "BJ test cluster bootstrap",
                "Seed": GCV_BOOTSTRAP_SEEDS["BJ"],
                "Notes": f"B={GCV_BOOTSTRAP_REPLICATES}",
            },
            {
                "Process": "BD test cluster bootstrap",
                "Seed": GCV_BOOTSTRAP_SEEDS["BD"],
                "Notes": f"B={GCV_BOOTSTRAP_REPLICATES}",
            },
            {
                "Process": "SHAP interaction main sample",
                "Seed": 42,
                "Notes": "Defined in plotting cell; without replacement; test source",
            },
            {
                "Process": "SHAP interaction stability samples",
                "Seed": "42, 20260826, 20260827",
                "Notes": "Defined in plotting cell; fixed size 10000",
            },
            {
                "Process": "Discrete ALE jitter",
                "Seed": 0,
                "Notes": "Inherited from original plotting function; visual jitter only",
            },
        ]
    )


gcv_set_global_seeds()
gcv_splits = {city: gcv_prepare_outer_split(city) for city in GCV_CITIES}
gcv_folds = {}
gcv_fold_audit_parts = []
for gcv_city in GCV_CITIES:
    gcv_folds[gcv_city], gcv_fold_audit = gcv_build_folds(
        gcv_city, gcv_splits[gcv_city]
    )
    gcv_fold_audit_parts.append(gcv_fold_audit)
gcv_fold_audit_table = gcv_pd.concat(gcv_fold_audit_parts, ignore_index=True)

gcv_signature = {
    "protocol_version": GCV_PROTOCOL_VERSION,
    "features": list(GCV_FEATURES),
    "target": GCV_TARGET,
    "group_columns": {city: list(GCV_GROUP_COLUMNS[city]) for city in GCV_CITIES},
    "test_size": GCV_TEST_SIZE,
    "n_splits": GCV_N_SPLITS,
    "outer_split_seed": GCV_OUTER_SPLIT_SEED,
    "optuna_seed": GCV_OPTUNA_SEED,
    "lightgbm_seed": GCV_LIGHTGBM_SEED,
    "n_trials": GCV_N_TRIALS,
    "max_boost_rounds": GCV_MAX_BOOST_ROUNDS,
    "early_stopping_rounds": GCV_EARLY_STOPPING_ROUNDS,
    "shared_split": {
        "order_split_file": str(GCV_ORDER_SPLIT_FILE),
        "wave_split_file": str(GCV_WAVE_SPLIT_FILE),
        "split_metadata_file": str(GCV_SPLIT_METADATA_FILE),
    },
    "software": {
        "python": gcv_platform.python_version(),
        "lightgbm": gcv_lgb.__version__,
        "optuna": gcv_optuna.__version__,
        "sklearn": gcv_sklearn.__version__,
        "pandas": gcv_pd.__version__,
        "numpy": gcv_np.__version__,
    },
    "data": {
        city: {
            "path": str(GCV_DATA_FILES[city]),
            "size": GCV_DATA_FILES[city].stat().st_size,
            "mtime_ns": GCV_DATA_FILES[city].stat().st_mtime_ns,
            "n_cleaned_total": gcv_splits[city].n_cleaned_total,
            "n_development": len(gcv_splits[city].X_development),
            "n_test": len(gcv_splits[city].X_test),
        }
        for city in GCV_CITIES
    },
}

gcv_saved_ok = False
if not GCV_FORCE_RETRAIN and GCV_METADATA_FILE.exists():
    try:
        gcv_saved_metadata = gcv_json.loads(
            GCV_METADATA_FILE.read_text(encoding="utf-8")
        )
        gcv_saved_ok = gcv_saved_metadata.get("signature") == gcv_signature
        gcv_saved_ok &= all(
            (GCV_MODEL_ROOT / city / "model.txt").exists()
            for city in GCV_CITIES
        )
        gcv_saved_ok &= (GCV_RESULT_ROOT / "final_test_metrics.csv").exists()
    except (OSError, ValueError):
        gcv_saved_ok = False

if gcv_saved_ok:
    print("\nGrouped 5-fold 模型与当前数据/配置签名匹配，直接读取。")
    gcv_models = {
        city: gcv_lgb.Booster(
            model_str=(GCV_MODEL_ROOT / city / "model.txt").read_text(
                encoding="utf-8"
            )
        )
        for city in GCV_CITIES
    }
    gcv_final_metrics = gcv_pd.read_csv(
        GCV_RESULT_ROOT / "final_test_metrics.csv"
    )
    gcv_leakage_table = gcv_pd.read_csv(
        GCV_RESULT_ROOT / "development_test_leakage_check.csv"
    )
    gcv_parameter_table = gcv_pd.read_csv(
        GCV_RESULT_ROOT / "lightgbm_parameter_table.csv"
    )
else:
    GCV_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    GCV_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    gcv_models = {}
    gcv_metric_rows = []
    gcv_parameter_parts = []
    gcv_training_metadata = {}

    for gcv_city in GCV_CITIES:
        gcv_split = gcv_splits[gcv_city]
        gcv_city_model_dir = GCV_MODEL_ROOT / gcv_city
        gcv_city_result_dir = GCV_RESULT_ROOT / gcv_city
        gcv_city_model_dir.mkdir(parents=True, exist_ok=True)
        gcv_city_result_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n========== Grouped 5-fold LightGBM: {gcv_city} ==========")

        gcv_study, gcv_best_params = gcv_tune_city(
            gcv_city,
            gcv_split,
            gcv_folds[gcv_city],
        )
        (
            gcv_cv_metrics,
            gcv_long_curve,
            gcv_best_iteration,
        ) = gcv_refit_best_folds(
            gcv_city,
            gcv_split,
            gcv_folds[gcv_city],
            gcv_best_params,
        )
        gcv_model = gcv_fit_final_model(
            gcv_split,
            gcv_best_params,
            gcv_best_iteration,
        )

        # Evaluate the outer test set only after the model is locked.
        gcv_test_prediction = gcv_np.asarray(
            gcv_model.predict(gcv_split.X_test)
        )
        gcv_point_metrics = gcv_metric_values(
            gcv_split.y_test,
            gcv_test_prediction,
        )
        gcv_ci_metrics, gcv_n_test_groups = gcv_cluster_bootstrap_ci(
            gcv_city,
            gcv_split,
            gcv_test_prediction,
        )

        # Deterministic replay; the result does not alter model settings.
        gcv_repeat_max_abs_prediction_difference = gcv_np.nan
        if GCV_REPEATABILITY_CHECK:
            gcv_repeated_model = gcv_fit_final_model(
                gcv_split,
                gcv_best_params,
                gcv_best_iteration,
            )
            gcv_repeated_prediction = gcv_np.asarray(
                gcv_repeated_model.predict(gcv_split.X_test)
            )
            gcv_repeat_max_abs_prediction_difference = float(
                gcv_np.max(
                    gcv_np.abs(
                        gcv_test_prediction - gcv_repeated_prediction
                    )
                )
            )

        gcv_metric_rows.append(
            {
                "City": gcv_city,
                "n_cleaned_total": gcv_split.n_cleaned_total,
                "n_development": len(gcv_split.X_development),
                "n_test": len(gcv_split.X_test),
                "n_test_analysis_wave_groups": gcv_n_test_groups,
                "best_iteration": gcv_best_iteration,
                **gcv_point_metrics,
                **gcv_ci_metrics,
                "repeat_max_abs_prediction_difference": (
                    gcv_repeat_max_abs_prediction_difference
                ),
            }
        )
        gcv_parameter_parts.extend(
            gcv_parameter_rows(
                gcv_city,
                gcv_best_params,
                gcv_best_iteration,
            )
        )
        gcv_models[gcv_city] = gcv_model

        (gcv_city_model_dir / "model.txt").write_text(
            gcv_model.model_to_string(), encoding="utf-8"
        )
        (gcv_city_model_dir / "best_params.json").write_text(
            gcv_json.dumps(
                {
                    "best_params": gcv_best_params,
                    "best_iteration": gcv_best_iteration,
                    "best_mean_grouped_5fold_validation_RMSE": float(
                        gcv_study.best_value
                    ),
                    "fold_best_iterations": gcv_cv_metrics[
                        "best_iteration"
                    ].tolist(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        gcv_study.trials_dataframe().to_csv(
            gcv_city_result_dir / "optuna_grouped_5fold_trials.csv",
            index=False,
            encoding="utf-8-sig",
        )
        gcv_cv_metrics.to_csv(
            gcv_city_result_dir / "best_params_fold_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        gcv_long_curve.to_csv(
            gcv_city_result_dir / "grouped_5fold_learning_curve_long.csv",
            index=False,
            encoding="utf-8-sig",
        )
        gcv_aggregate_curve = gcv_plot_learning_curve(
            gcv_long_curve,
            gcv_best_iteration,
            gcv_city_result_dir / "LGBM_learning_curve.pdf",
        )
        gcv_aggregate_curve.to_csv(
            gcv_city_result_dir / "LGBM_training_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        gcv_pd.DataFrame(
            {
                "source_row_index": gcv_split.test_source_index,
                "analysis_wave_id": gcv_split.groups_test.to_numpy(),
                "y_true": gcv_split.y_test.to_numpy(),
                "prediction": gcv_test_prediction,
                "residual_y_minus_prediction": (
                    gcv_split.y_test.to_numpy() - gcv_test_prediction
                ),
            }
        ).to_csv(
            gcv_city_result_dir / "final_test_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        gcv_training_metadata[gcv_city] = {
            "best_trial_number": int(gcv_study.best_trial.number),
            "best_mean_grouped_5fold_validation_RMSE": float(
                gcv_study.best_value
            ),
            "best_iteration": gcv_best_iteration,
            "fold_best_iterations": gcv_cv_metrics[
                "best_iteration"
            ].tolist(),
            "n_complete_trials": int(
                sum(
                    trial.state == gcv_optuna.trial.TrialState.COMPLETE
                    for trial in gcv_study.trials
                )
            ),
        }

    gcv_final_metrics = gcv_pd.DataFrame(gcv_metric_rows)
    gcv_parameter_table = gcv_pd.DataFrame(gcv_parameter_parts)
    gcv_leakage_rows = []
    for gcv_city in GCV_CITIES:
        gcv_split = gcv_splits[gcv_city]
        repeated = set(gcv_split.groups_development.unique()) & set(
            gcv_split.groups_test.unique()
        )
        gcv_leakage_rows.append(
            {
                "City": gcv_city,
                "Total_cleaned_n": gcv_split.n_cleaned_total,
                "Development_n": len(gcv_split.X_development),
                "Development_percent": (
                    100 * len(gcv_split.X_development) / gcv_split.n_cleaned_total
                ),
                "Test_n": len(gcv_split.X_test),
                "Test_percent": (
                    100 * len(gcv_split.X_test) / gcv_split.n_cleaned_total
                ),
                "Grouping_key": gcv_split.group_key_description,
                "Cross_set_repeated_group_count": len(repeated),
                "Target_missing_removed": gcv_split.n_target_missing_removed,
            }
        )
    gcv_leakage_table = gcv_pd.DataFrame(gcv_leakage_rows)
    gcv_environment_versions = gcv_environment_table()
    gcv_random_seed_inventory = gcv_seed_inventory()

    gcv_final_metrics.to_csv(
        GCV_RESULT_ROOT / "final_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gcv_leakage_table.to_csv(
        GCV_RESULT_ROOT / "development_test_leakage_check.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gcv_fold_audit_table.to_csv(
        GCV_RESULT_ROOT / "grouped_5fold_split_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gcv_parameter_table.to_csv(
        GCV_RESULT_ROOT / "lightgbm_parameter_table.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gcv_environment_versions.to_csv(
        GCV_RESULT_ROOT / "software_environment.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gcv_random_seed_inventory.to_csv(
        GCV_RESULT_ROOT / "random_seed_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (GCV_RESULT_ROOT / "requirements.txt").write_text(
        "\n".join(
            f"{row.Software}=={row.Version}"
            for row in gcv_environment_versions.itertuples(index=False)
            if row.Software
            not in {"Python", "Platform", "ALE implementation"}
            and row.Version != "NOT_INSTALLED"
        )
        + "\n",
        encoding="utf-8",
    )
    GCV_METADATA_FILE.write_text(
        gcv_json.dumps(
            {
                "created_at": gcv_datetime.now().isoformat(timespec="seconds"),
                "design": {
                    "development_definition": (
                        "All cleaned orders not assigned to the outer grouped test; "
                        "contains all data used by grouped 5-fold CV and tuning"
                    ),
                    "outer_split": (
                        "Persisted GroupShuffleSplit by city-date-courier-wave"
                    ),
                    "inner_cv": (
                        "Persisted GroupKFold(n_splits=5) within development"
                    ),
                    "shared_order_split_file": str(GCV_ORDER_SPLIT_FILE),
                    "shared_wave_split_file": str(GCV_WAVE_SPLIT_FILE),
                    "optuna_objective": "mean of five validation-fold RMSE values",
                    "iteration_rule": "median of five fold-specific best iterations",
                    "final_fit": "complete development at locked iteration count",
                    "test_use": "one final evaluation after model selection",
                },
                "signature": gcv_signature,
                "training_results": gcv_training_metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

print("\n========== Development/test leakage check ==========")
print(gcv_leakage_table.round(5).to_string(index=False))
print("\n========== Grouped five-fold split audit ==========")
print(gcv_fold_audit_table.to_string(index=False))
print("\n========== Final untouched-test metrics ==========")
print(gcv_final_metrics.round(5).to_string(index=False))
print(f"\nSaved grouped 5-fold models: {GCV_MODEL_ROOT}")


# In[]:
# =============================================================================
# 2. SHAP, interaction, and ALE analysis
# =============================================================================

import gc as gfig_gc
import json as gfig_json

import lightgbm as gfig_lgb
import matplotlib.pyplot as gfig_plt
import numpy as gfig_np
import pandas as gfig_pd
import shap as gfig_shap
from PyALE._src.lib import quantile_ied as gfig_quantile_ied

try:
    import lightgbm_utils as gfig_original_plotting
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "lightgbm_utils.py must be available in the project directory."
    ) from exc


GFIG_ROOT = GCV_ROOT
GFIG_DATA_FILES = GCV_DATA_FILES
GFIG_CITIES = ("BJ", "BD")
GFIG_FEATURES = (
    "order_hourly",
    "is_weekend",
    "peak",
    "ride_distance_km",
    "merchant_preparation_time",
    "wave_duration_time",
    "loadorder_number",
    "courier_work_hours_now",
    "platform_assign_time",
    "demand_density",
    "supply_density",
    "road_length_km",
    "avg_node_degree",
    "prev_standardized_delay",
)
GFIG_TARGET = "standardized_delay"
GFIG_GROUP_COLUMNS = {
    "BJ": ("Date", "courier_id", "wave_index"),
    "BD": ("dt", "courier_id", "wave_id"),
}
GFIG_ORDER_ID_COLUMNS = {"BJ": "source_row_id", "BD": "waybill_id"}
GFIG_LAG_SOURCE = {"BJ": "prev_standardized_delay", "BD": "prior_order_delay"}
GFIG_OUTER_SPLIT_SEED = 42
GFIG_ALE_REQUESTED_GRID_SIZE = 20
GFIG_INTERACTION_MAIN_SAMPLE_SIZE = 10000
GFIG_INTERACTION_MAIN_SEED = 42
GFIG_STABILITY_SAMPLE_SIZES = (10000,)
GFIG_STABILITY_SEEDS = (42, 20260826, 20260827)
GFIG_STABILITY_TOP_K = 10

GFIG_EXPERIMENT_ROOT = GCV_OUTPUT_ROOT
GFIG_MODEL_ROOT = GFIG_EXPERIMENT_ROOT / "models"
GFIG_FIGURE_ROOT = GFIG_EXPERIMENT_ROOT / "figures"
GFIG_AUDIT_ROOT = GFIG_EXPERIMENT_ROOT / "results" / "figure_audit"
GFIG_METADATA_FILE = GFIG_MODEL_ROOT / "model_metadata.json"
GFIG_ORDER_SPLIT_FILE = (
    GFIG_EXPERIMENT_ROOT / "shared_sample_split" / "analysis_sample_split.csv"
)


def gfig_canonical_component(series, city, column):
    if series.isna().any():
        raise ValueError(f"{city}: 分组或订单键字段 {column} 存在缺失")
    if gfig_pd.api.types.is_numeric_dtype(series):
        numeric = gfig_pd.to_numeric(series, errors="raise")
        if gfig_np.all(gfig_np.isclose(numeric, gfig_np.round(numeric))):
            return gfig_pd.Series(
                gfig_np.round(numeric).astype(gfig_np.int64).astype(str),
                index=series.index,
                dtype="string",
            )
    return series.astype("string").str.strip()


def gfig_analysis_wave_id(frame, city):
    result = gfig_pd.Series(city, index=frame.index, dtype="string")
    for column in GFIG_GROUP_COLUMNS[city]:
        result = result + "::" + gfig_canonical_component(
            frame[column], city, column
        )
    return result


def gfig_prepare_outer_split(city):
    """从正式训练保存的唯一共享表读取development/test。"""

    group_columns = list(GFIG_GROUP_COLUMNS[city])
    order_column = GFIG_ORDER_ID_COLUMNS[city]
    lag_source = GFIG_LAG_SOURCE[city]
    source_features = [
        lag_source if feature == "prev_standardized_delay" else feature
        for feature in GFIG_FEATURES
    ]
    required = [*source_features, GFIG_TARGET, order_column, *group_columns]
    frame = gfig_pd.read_csv(GFIG_DATA_FILES[city], usecols=required)
    if lag_source != "prev_standardized_delay":
        frame = frame.rename(columns={lag_source: "prev_standardized_delay"})
    frame = frame.loc[frame[GFIG_TARGET].notna()].copy()
    frame["order_id"] = gfig_canonical_component(
        frame[order_column], city, order_column
    )
    if frame["order_id"].duplicated().any():
        raise ValueError(f"{city}: 稳定订单键 {order_column} 不唯一")
    frame["analysis_wave_id"] = gfig_analysis_wave_id(frame, city)
    if not GFIG_ORDER_SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"缺少唯一共享划分表：{GFIG_ORDER_SPLIT_FILE}\n"
            "请先运行正式训练单元生成划分表。"
        )
    assignment = gfig_pd.read_csv(
        GFIG_ORDER_SPLIT_FILE,
        dtype={
            "city": "string", "order_id": "string",
            "analysis_wave_id": "string", "sample_split": "string",
        },
    )
    assignment = assignment.loc[assignment["city"] == city].set_index(
        "order_id", verify_integrity=True
    )
    if set(frame["order_id"]) != set(assignment.index):
        raise ValueError(f"{city}: 图形数据与共享样本划分表的订单集合不一致")
    mapped = assignment.reindex(frame["order_id"].to_numpy())
    mapped.index = frame.index
    if not mapped["analysis_wave_id"].equals(frame["analysis_wave_id"]):
        raise ValueError(f"{city}: 图形数据analysis_wave_id与共享表不一致")
    frame["sample_split"] = mapped["sample_split"]
    X = frame.loc[:, list(GFIG_FEATURES)].copy()
    y = frame[GFIG_TARGET].copy()
    groups = frame["analysis_wave_id"]
    development_position = gfig_np.flatnonzero(
        frame["sample_split"].eq("development").to_numpy()
    )
    test_position = gfig_np.flatnonzero(
        frame["sample_split"].eq("test").to_numpy()
    )
    development_groups = set(groups.iloc[development_position].unique())
    test_groups = set(groups.iloc[test_position].unique())
    if development_groups & test_groups:
        raise AssertionError(f"{city}: figure split contains group leakage")
    return {
        "X_development": X.iloc[development_position].copy(),
        "X_test": X.iloc[test_position].copy(),
        "y_development": y.iloc[development_position].copy(),
        "y_test": y.iloc[test_position].copy(),
        "groups_development": groups.iloc[development_position].copy(),
        "groups_test": groups.iloc[test_position].copy(),
    }


def gfig_load_model(city):
    """只读取grouped 5-fold单元保存的最终模型。"""

    path = GFIG_MODEL_ROOT / city / "model.txt"
    params_path = GFIG_MODEL_ROOT / city / "best_params.json"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少 {city} grouped 5-fold 模型：{path}\n"
            "请先运行上一个 # In[]: 训练单元。"
        )
    if not params_path.exists():
        raise FileNotFoundError(f"缺少 {city} 训练参数：{params_path}")
    model = gfig_lgb.Booster(model_str=path.read_text(encoding="utf-8"))
    # Restore parameters required by SHAP when the model is loaded from text.
    saved_params = gfig_json.loads(
        params_path.read_text(encoding="utf-8")
    )["best_params"]
    model.params.update(saved_params)
    if model.feature_name() != list(GFIG_FEATURES):
        raise ValueError(f"{city} model feature order does not match figure code")
    return model


def gfig_validate_training_metadata(splits):
    """检查图形单元重建的数据规模与训练签名一致。"""

    if not GFIG_METADATA_FILE.exists():
        raise FileNotFoundError(f"缺少训练元数据：{GFIG_METADATA_FILE}")
    metadata = gfig_json.loads(GFIG_METADATA_FILE.read_text(encoding="utf-8"))
    signature = metadata.get("signature", {})
    if signature.get("features") != list(GFIG_FEATURES):
        raise ValueError("Training metadata feature list does not match")
    if signature.get("group_columns") != {
        city: list(GFIG_GROUP_COLUMNS[city]) for city in GFIG_CITIES
    }:
        raise ValueError("Training metadata group columns do not match")
    if signature.get("outer_split_seed") != GFIG_OUTER_SPLIT_SEED:
        raise ValueError("Training metadata outer split seed does not match")
    for city in GFIG_CITIES:
        saved = signature.get("data", {}).get(city, {})
        if saved.get("n_development") != len(splits[city]["X_development"]):
            raise ValueError(f"{city} development size does not match training")
        if saved.get("n_test") != len(splits[city]["X_test"]):
            raise ValueError(f"{city} test size does not match training")
    return metadata


def gfig_ale_bin_audit(city, X_development):
    """预先记录PyALE配置和重复分位点合并后的实际区间数。"""

    rows = []
    for feature in GFIG_FEATURES:
        values = gfig_pd.to_numeric(
            X_development[feature], errors="coerce"
        ).dropna()
        n_unique = int(values.nunique())
        if n_unique <= 10:
            rows.append(
                {
                    "City": city,
                    "Feature": feature,
                    "ALE_source": "complete development",
                    "ALE_method": "manual ordered discrete ALE",
                    "requested_grid_size": "not applicable",
                    "actual_intervals_or_levels": n_unique,
                    "binning": "all sorted unique values",
                    "missing_n": int(X_development[feature].isna().sum()),
                    "empty_interval_handling": "not applicable",
                }
            )
            continue
        quantiles = gfig_np.linspace(
            0,
            1,
            GFIG_ALE_REQUESTED_GRID_SIZE + 1,
            endpoint=True,
        )
        bin_values = [values.min()] + gfig_quantile_ied(
            values, quantiles
        ).to_list()
        unique_bins = gfig_np.unique(bin_values)
        rows.append(
            {
                "City": city,
                "Feature": feature,
                "ALE_source": "complete development",
                "ALE_method": "PyALE.ale continuous 1D",
                "requested_grid_size": GFIG_ALE_REQUESTED_GRID_SIZE,
                "actual_intervals_or_levels": max(len(unique_bins) - 1, 0),
                "binning": "equal-frequency empirical quantiles (type-1)",
                "missing_n": int(X_development[feature].isna().sum()),
                "empty_interval_handling": (
                    "duplicate quantile boundaries collapsed by np.unique; "
                    "PyALE groups only observed intervals"
                ),
            }
        )
    return gfig_pd.DataFrame(rows)


def gfig_interaction_pair_table(interaction_values, feature_names):
    """将不重复的非对角SHAP interaction强度转为排名表。"""

    values = gfig_np.asarray(interaction_values)
    if values.ndim == 4 and values.shape[0] == 1:
        values = values[0]
    mean_abs = gfig_np.abs(values).mean(axis=0)
    rows = []
    for first in range(len(feature_names)):
        for second in range(first + 1, len(feature_names)):
            feature_a = feature_names[first]
            feature_b = feature_names[second]
            rows.append(
                {
                    "Pair": " | ".join(sorted((feature_a, feature_b))),
                    "Feature_1": feature_a,
                    "Feature_2": feature_b,
                    "Mean_abs_SHAP_interaction": float(mean_abs[first, second]),
                }
            )
    table = gfig_pd.DataFrame(rows)
    table["Rank"] = (
        table["Mean_abs_SHAP_interaction"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return table.sort_values("Rank")


def gfig_interaction_stability(city, model, X_test):
    """比较样本量和种子变化时的interaction pair排名。"""

    explainer = gfig_shap.TreeExplainer(model)
    pair_tables = {}
    long_parts = []
    for sample_size in GFIG_STABILITY_SAMPLE_SIZES:
        actual_size = min(sample_size, len(X_test))
        for seed in GFIG_STABILITY_SEEDS:
            X_sample = X_test.sample(
                n=actual_size,
                replace=False,
                random_state=seed,
            )
            interaction_values = explainer.shap_interaction_values(X_sample)
            pair_table = gfig_interaction_pair_table(
                interaction_values,
                list(GFIG_FEATURES),
            )
            pair_table.insert(0, "City", city)
            pair_table.insert(1, "Requested_sample_size", sample_size)
            pair_table.insert(2, "Actual_sample_size", actual_size)
            pair_table.insert(3, "Seed", seed)
            pair_tables[(sample_size, seed)] = pair_table
            long_parts.append(pair_table)
            del interaction_values, X_sample
            gfig_gc.collect()

    reference_key = (
        GFIG_INTERACTION_MAIN_SAMPLE_SIZE,
        GFIG_INTERACTION_MAIN_SEED,
    )
    if reference_key not in pair_tables:
        raise KeyError("Interaction stability grid lacks the main reference sample")
    reference = pair_tables[reference_key].set_index("Pair")
    reference_top = set(
        reference.loc[reference["Rank"] <= GFIG_STABILITY_TOP_K].index
    )
    summary_rows = []
    for (sample_size, seed), table in pair_tables.items():
        candidate = table.set_index("Pair")
        aligned = reference[["Rank"]].join(
            candidate[["Rank"]],
            how="inner",
            lsuffix="_reference",
            rsuffix="_candidate",
        )
        candidate_top = set(
            candidate.loc[candidate["Rank"] <= GFIG_STABILITY_TOP_K].index
        )
        union = reference_top | candidate_top
        summary_rows.append(
            {
                "City": city,
                "Requested_sample_size": sample_size,
                "Actual_sample_size": int(table["Actual_sample_size"].iloc[0]),
                "Seed": seed,
                "Reference_sample_size": GFIG_INTERACTION_MAIN_SAMPLE_SIZE,
                "Reference_seed": GFIG_INTERACTION_MAIN_SEED,
                "All_pair_rank_Spearman": float(
                    aligned["Rank_reference"].corr(
                        aligned["Rank_candidate"], method="pearson"
                    )
                ),
                "Top10_overlap_n": len(reference_top & candidate_top),
                "Top10_Jaccard": (
                    len(reference_top & candidate_top) / len(union)
                    if union
                    else gfig_np.nan
                ),
            }
        )
    return (
        gfig_pd.concat(long_parts, ignore_index=True),
        gfig_pd.DataFrame(summary_rows),
    )


def gfig_plot_stability(summary_table, save_path):
    """额外审计图：使用原项目颜色/字体展示interaction排名稳定性。"""

    colors = {42: "#4573b4", 20260826: "#d73221", 20260827: "#B5AED5"}
    figure, axis = gfig_plt.subplots(figsize=(10, 5))
    for seed in GFIG_STABILITY_SEEDS:
        subset = summary_table.loc[summary_table["Seed"] == seed].sort_values(
            "Actual_sample_size"
        )
        axis.plot(
            subset["Actual_sample_size"],
            subset["All_pair_rank_Spearman"],
            marker="o",
            linewidth=1.5,
            color=colors[seed],
            label=f"Seed {seed}",
        )
    axis.set_xlabel("SHAP interaction sample size", fontsize=14)
    axis.set_ylabel("Rank correlation vs. 10,000-order reference", fontsize=14)
    axis.set_title("SHAP Interaction Ranking Stability", fontsize=16)
    axis.set_ylim(0, 1.02)
    axis.legend(fontsize=12)
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    figure.savefig(save_path, bbox_inches="tight", dpi=300)
    gfig_plt.close(figure)


gfig_splits = {city: gfig_prepare_outer_split(city) for city in GFIG_CITIES}
gfig_models = {city: gfig_load_model(city) for city in GFIG_CITIES}
gfig_training_metadata = gfig_validate_training_metadata(gfig_splits)
GFIG_FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
GFIG_AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

gfig_provenance_rows = []
gfig_ale_audit_parts = []
gfig_stability_pair_parts = []
gfig_stability_summary_parts = []

# Request 20 quantile intervals for continuous ALE; low-cardinality variables
# retain the discrete branch used by the original plotting module.
gfig_original_bin_function = gfig_original_plotting.calculate_optimal_bins
gfig_original_ale_function = gfig_original_plotting.ale


def gfig_ale_type_wrapper(*args, **kwargs):
    X_value = kwargs.get("X", args[0] if args else None)
    feature_value = kwargs.get("feature")
    if (
        X_value is not None
        and feature_value is not None
        and len(feature_value) == 1
        and X_value[feature_value[0]].nunique(dropna=True) > 10
    ):
        kwargs["feature_type"] = "continuous"
    elif (
        X_value is not None
        and feature_value is not None
        and len(feature_value) == 2
    ):
        # PyALE 1.2.0 requires complete observations for the selected pair.
        kwargs["X"] = X_value.dropna(subset=list(feature_value)).copy()
    return gfig_original_ale_function(*args, **kwargs)


gfig_original_plotting.calculate_optimal_bins = (
    lambda series: GFIG_ALE_REQUESTED_GRID_SIZE
)
gfig_original_plotting.ale = gfig_ale_type_wrapper
try:
    for gfig_city in GFIG_CITIES:
        gfig_split = gfig_splits[gfig_city]
        gfig_model = gfig_models[gfig_city]
        gfig_save_path = GFIG_FIGURE_ROOT / gfig_city
        gfig_save_path.mkdir(parents=True, exist_ok=True)
        feature_names = list(GFIG_FEATURES)
        depvar = GFIG_TARGET
        X_train = gfig_split["X_development"]
        X_test = gfig_split["X_test"]
        lightgbm_model = gfig_model
        save_path = str(gfig_save_path)

        shap_lightgbm = gfig_original_plotting.explain_shap_lightgbm(
            model=lightgbm_model,
            X_train=X_train,
            X_test=X_test,
            feature_names=feature_names,
            plot_type="bar",
            save_path=save_path,
            depvar=depvar,
        )

        gfig_ale_audit_parts.append(
            gfig_ale_bin_audit(gfig_city, X_train)
        )
        gfig_pair_table, gfig_stability_summary = gfig_interaction_stability(
            gfig_city,
            gfig_model,
            X_test,
        )
        gfig_stability_pair_parts.append(gfig_pair_table)
        gfig_stability_summary_parts.append(gfig_stability_summary)
        gfig_provenance_rows.extend(
            [
                {
                    "City": gfig_city,
                    "Output_family": "SHAP main effects/ranking/dependence",
                    "Model": f"{gfig_city} grouped-5-fold final local model",
                    "Data_source": "complete untouched outer test",
                    "n_orders": len(X_test),
                    "Sampling": "none",
                    "Seed": "not applicable",
                },
                {
                    "City": gfig_city,
                    "Output_family": "SHAP interaction main figures",
                    "Model": f"{gfig_city} grouped-5-fold final local model",
                    "Data_source": "untouched outer test",
                    "n_orders": min(
                        GFIG_INTERACTION_MAIN_SAMPLE_SIZE, len(X_test)
                    ),
                    "Sampling": "simple random without replacement; unstratified",
                    "Seed": GFIG_INTERACTION_MAIN_SEED,
                },
                {
                    "City": gfig_city,
                    "Output_family": "1D and 2D ALE",
                    "Model": f"{gfig_city} grouped-5-fold final local model",
                    "Data_source": (
                        "complete development for 1D; pairwise feature-complete "
                        "development for 2D"
                    ),
                    "n_orders": len(X_train),
                    "Sampling": "none",
                    "Seed": "discrete visual jitter seed=0 only",
                },
                {
                    "City": gfig_city,
                    "Output_family": "SHAP interaction stability audit",
                    "Model": f"{gfig_city} grouped-5-fold final local model",
                    "Data_source": "untouched outer test",
                    "n_orders": "10000",
                    "Sampling": "simple random without replacement; unstratified",
                    "Seed": "42/20260826/20260827",
                },
            ]
        )
finally:
    gfig_original_plotting.calculate_optimal_bins = gfig_original_bin_function
    gfig_original_plotting.ale = gfig_original_ale_function

gfig_figure_provenance = gfig_pd.DataFrame(gfig_provenance_rows)
gfig_ale_bin_settings = gfig_pd.concat(gfig_ale_audit_parts, ignore_index=True)
gfig_interaction_pair_stability = gfig_pd.concat(
    gfig_stability_pair_parts, ignore_index=True
)
gfig_interaction_stability_summary = gfig_pd.concat(
    gfig_stability_summary_parts, ignore_index=True
)

gfig_figure_provenance.to_csv(
    GFIG_AUDIT_ROOT / "figure_data_provenance.csv",
    index=False,
    encoding="utf-8-sig",
)
gfig_ale_bin_settings.to_csv(
    GFIG_AUDIT_ROOT / "ale_settings_by_feature.csv",
    index=False,
    encoding="utf-8-sig",
)
gfig_interaction_pair_stability.to_csv(
    GFIG_AUDIT_ROOT / "shap_interaction_pair_rankings_all_samples.csv",
    index=False,
    encoding="utf-8-sig",
)
gfig_interaction_stability_summary.to_csv(
    GFIG_AUDIT_ROOT / "shap_interaction_stability_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n========== Figure data provenance ==========")
print(gfig_figure_provenance.to_string(index=False))
print("\n========== SHAP interaction stability ==========")
print(gfig_interaction_stability_summary.round(5).to_string(index=False))
print(f"\nSaved grouped-5-fold figures: {GFIG_FIGURE_ROOT}")


# In[]:
# =============================================================================
# 3. Two-dimensional ALE
# =============================================================================

import json as g2d_json

import lightgbm as g2d_lgb
import numpy as g2d_np
import pandas as g2d_pd

g2d_original_plotting = gfig_original_plotting


G2D_ROOT = GCV_ROOT
G2D_EXPERIMENT_ROOT = GCV_OUTPUT_ROOT
G2D_MODEL_ROOT = G2D_EXPERIMENT_ROOT / "models"
G2D_FIGURE_ROOT = G2D_EXPERIMENT_ROOT / "figures"
G2D_AUDIT_ROOT = G2D_EXPERIMENT_ROOT / "results" / "figure_audit"
G2D_INTERACTION_FILE = (
    G2D_AUDIT_ROOT / "shap_interaction_pair_rankings_all_samples.csv"
)
G2D_DATA_FILES = GCV_DATA_FILES
G2D_GROUP_COLUMNS = {
    "BJ": ("Date", "courier_id", "wave_index"),
    "BD": ("dt", "courier_id", "wave_id"),
}
G2D_ORDER_ID_COLUMNS = {"BJ": "source_row_id", "BD": "waybill_id"}
G2D_LAG_SOURCE = {"BJ": "prev_standardized_delay", "BD": "prior_order_delay"}
G2D_FEATURES = (
    "order_hourly",
    "is_weekend",
    "peak",
    "ride_distance_km",
    "merchant_preparation_time",
    "wave_duration_time",
    "loadorder_number",
    "courier_work_hours_now",
    "platform_assign_time",
    "demand_density",
    "supply_density",
    "road_length_km",
    "avg_node_degree",
    "prev_standardized_delay",
)
G2D_TARGET = "standardized_delay"
G2D_INTERACTION_SAMPLE_SIZE = 10000
G2D_INTERACTION_SEED = 42
G2D_TOP_K = 5
G2D_GRID_SIZE = 20
G2D_ORDER_SPLIT_FILE = (
    G2D_EXPERIMENT_ROOT / "shared_sample_split" / "analysis_sample_split.csv"
)


def g2d_canonical_component(series, city, column):
    if series.isna().any():
        raise ValueError(f"{city}: 分组或订单键字段 {column} 存在缺失")
    if g2d_pd.api.types.is_numeric_dtype(series):
        numeric = g2d_pd.to_numeric(series, errors="raise")
        if g2d_np.all(g2d_np.isclose(numeric, g2d_np.round(numeric))):
            return g2d_pd.Series(
                g2d_np.round(numeric).astype(g2d_np.int64).astype(str),
                index=series.index,
                dtype="string",
            )
    return series.astype("string").str.strip()


def g2d_analysis_wave_id(frame, city):
    result = g2d_pd.Series(city, index=frame.index, dtype="string")
    for column in G2D_GROUP_COLUMNS[city]:
        result = result + "::" + g2d_canonical_component(
            frame[column], city, column
        )
    return result


def g2d_prepare_development(city):
    """读取共享划分表中的development，仅返回2D ALE所需特征。"""

    group_columns = list(G2D_GROUP_COLUMNS[city])
    order_column = G2D_ORDER_ID_COLUMNS[city]
    lag_source = G2D_LAG_SOURCE[city]
    source_features = [
        lag_source if feature == "prev_standardized_delay" else feature
        for feature in G2D_FEATURES
    ]
    required = [
        *source_features,
        G2D_TARGET,
        order_column,
        *group_columns,
    ]
    frame = g2d_pd.read_csv(G2D_DATA_FILES[city], usecols=required)
    if lag_source != "prev_standardized_delay":
        frame = frame.rename(columns={lag_source: "prev_standardized_delay"})
    frame = frame.loc[frame[G2D_TARGET].notna()].copy()
    frame["order_id"] = g2d_canonical_component(
        frame[order_column], city, order_column
    )
    if frame["order_id"].duplicated().any():
        raise ValueError(f"{city}: 稳定订单键 {order_column} 不唯一")
    frame["analysis_wave_id"] = g2d_analysis_wave_id(frame, city)
    if not G2D_ORDER_SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"缺少唯一共享划分表：{G2D_ORDER_SPLIT_FILE}\n"
            "请先运行正式训练单元生成划分表。"
        )
    assignment = g2d_pd.read_csv(
        G2D_ORDER_SPLIT_FILE,
        dtype={
            "city": "string", "order_id": "string",
            "analysis_wave_id": "string", "sample_split": "string",
        },
    )
    assignment = assignment.loc[assignment["city"] == city].set_index(
        "order_id", verify_integrity=True
    )
    if set(frame["order_id"]) != set(assignment.index):
        raise ValueError(f"{city}: 2D ALE数据与共享样本划分表不一致")
    mapped = assignment.reindex(frame["order_id"].to_numpy())
    mapped.index = frame.index
    if not mapped["analysis_wave_id"].equals(frame["analysis_wave_id"]):
        raise ValueError(f"{city}: 2D ALE的analysis_wave_id与共享表不一致")
    frame["sample_split"] = mapped["sample_split"]
    X = frame.loc[:, list(G2D_FEATURES)].copy()
    development_position = g2d_np.flatnonzero(
        frame["sample_split"].eq("development").to_numpy()
    )
    return X.iloc[development_position].copy()


def g2d_load_model(city):
    """读取已保存的城市最终模型和训练参数。"""

    model_dir = G2D_MODEL_ROOT / city
    model_path = model_dir / "model.txt"
    params_path = model_dir / "best_params.json"
    if not model_path.exists() or not params_path.exists():
        raise FileNotFoundError(f"{city}: 缺少第17单元保存的模型或参数")
    model = g2d_lgb.Booster(
        model_str=model_path.read_text(encoding="utf-8")
    )
    saved_params = g2d_json.loads(
        params_path.read_text(encoding="utf-8")
    )["best_params"]
    model.params.update(saved_params)
    if model.feature_name() != list(G2D_FEATURES):
        raise ValueError(f"{city}: 模型特征顺序与2D ALE单元不一致")
    return model


def g2d_interaction_matrix(city, ranking_table):
    """由已保存的seed=42、n=10,000排名表重建对称交互矩阵。"""

    city_table = ranking_table.loc[
        (ranking_table["City"] == city)
        & (
            ranking_table["Actual_sample_size"]
            == G2D_INTERACTION_SAMPLE_SIZE
        )
        & (ranking_table["Seed"] == G2D_INTERACTION_SEED)
    ].copy()
    expected_pairs = len(G2D_FEATURES) * (len(G2D_FEATURES) - 1) // 2
    if len(city_table) != expected_pairs:
        raise ValueError(
            f"{city}: seed=42、n=10000应有{expected_pairs}个特征对，"
            f"实际为{len(city_table)}"
        )
    matrix = g2d_pd.DataFrame(
        g2d_np.zeros((len(G2D_FEATURES), len(G2D_FEATURES))),
        index=G2D_FEATURES,
        columns=G2D_FEATURES,
    )
    for row in city_table.itertuples(index=False):
        matrix.loc[row.Feature_1, row.Feature_2] = (
            row.Mean_abs_SHAP_interaction
        )
        matrix.loc[row.Feature_2, row.Feature_1] = (
            row.Mean_abs_SHAP_interaction
        )
    return matrix


if not G2D_INTERACTION_FILE.exists():
    raise FileNotFoundError(
        f"缺少第18单元保存的interaction排名：{G2D_INTERACTION_FILE}"
    )
g2d_rankings = g2d_pd.read_csv(G2D_INTERACTION_FILE)
g2d_pairwise_audit_rows = []
g2d_original_ale_function = g2d_original_plotting.ale


def g2d_pairwise_complete_ale(*args, **kwargs):
    """仅对2D ALE参与变量排除缺失，并记录实际使用样本量。"""

    X_value = kwargs.get("X", args[0] if args else None)
    feature_value = kwargs.get("feature")
    if (
        X_value is not None
        and feature_value is not None
        and len(feature_value) == 2
    ):
        pair = list(feature_value)
        n_before = len(X_value)
        complete = X_value.dropna(subset=pair).copy()
        g2d_pairwise_audit_rows.append(
            {
                "Feature_1": pair[0],
                "Feature_2": pair[1],
                "Development_n_before": n_before,
                "Pairwise_missing_excluded_n": n_before - len(complete),
                "Development_n_used": len(complete),
                "Missing_rule": "drop only rows missing either plotted feature",
                "Grid_size_requested": G2D_GRID_SIZE,
            }
        )
        kwargs["X"] = complete
    return g2d_original_ale_function(*args, **kwargs)


g2d_original_plotting.ale = g2d_pairwise_complete_ale
try:
    for g2d_city in ("BJ", "BD"):
        g2d_X_development = g2d_prepare_development(g2d_city)
        g2d_model = g2d_load_model(g2d_city)
        g2d_matrix = g2d_interaction_matrix(g2d_city, g2d_rankings)
        g2d_save_path = G2D_FIGURE_ROOT / g2d_city
        g2d_save_path.mkdir(parents=True, exist_ok=True)
        g2d_audit_start = len(g2d_pairwise_audit_rows)
        g2d_original_plotting.plot_2d_ale_interactions(
            model=g2d_model,
            X_train=g2d_X_development,
            feature_names=list(G2D_FEATURES),
            save_path=str(g2d_save_path),
            depvar=G2D_TARGET,
            interaction_df=g2d_matrix,
            top_k=G2D_TOP_K,
            grid_size=G2D_GRID_SIZE,
        )
        for row in g2d_pairwise_audit_rows[g2d_audit_start:]:
            row["City"] = g2d_city
            row["Interaction_source"] = (
                "test SHAP interaction; n=10000; seed=42; without replacement"
            )
finally:
    g2d_original_plotting.ale = g2d_original_ale_function

g2d_pairwise_audit = g2d_pd.DataFrame(g2d_pairwise_audit_rows)
g2d_pairwise_audit = g2d_pairwise_audit.loc[
    :,
    [
        "City",
        "Feature_1",
        "Feature_2",
        "Development_n_before",
        "Pairwise_missing_excluded_n",
        "Development_n_used",
        "Missing_rule",
        "Grid_size_requested",
        "Interaction_source",
    ],
]
G2D_AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
g2d_pairwise_audit.to_csv(
    G2D_AUDIT_ROOT / "ale2d_pairwise_missing_audit.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n========== 2D ALE pairwise missing-data audit ==========")
print(g2d_pairwise_audit.to_string(index=False))
print(f"\n2D ALE figures refreshed: {G2D_FIGURE_ROOT}")


# In[]:
# =============================================================================
# 4. Fixed-horizon grouped learning curves
# =============================================================================

import json as glc_json

import lightgbm as glc_lgb
import matplotlib.pyplot as glc_plt
import numpy as glc_np
import pandas as glc_pd


GLC_ROOT = GCV_ROOT
GLC_EXPERIMENT_ROOT = GCV_OUTPUT_ROOT
GLC_MODEL_ROOT = GLC_EXPERIMENT_ROOT / "models"
GLC_RESULT_ROOT = GLC_EXPERIMENT_ROOT / "results"
GLC_DATA_FILES = {
    "BJ": GCV_DATA_FILES["BJ"],
    "BD": GCV_DATA_FILES["BD"],
}
GLC_GROUP_COLUMNS = {
    "BJ": ("Date", "courier_id", "wave_index"),
    "BD": ("dt", "courier_id", "wave_id"),
}
GLC_ORDER_ID_COLUMNS = {"BJ": "source_row_id", "BD": "waybill_id"}
GLC_LAG_SOURCE = {"BJ": "prev_standardized_delay", "BD": "prior_order_delay"}
GLC_FEATURES = (
    "order_hourly",
    "is_weekend",
    "peak",
    "ride_distance_km",
    "merchant_preparation_time",
    "wave_duration_time",
    "loadorder_number",
    "courier_work_hours_now",
    "platform_assign_time",
    "demand_density",
    "supply_density",
    "road_length_km",
    "avg_node_degree",
    "prev_standardized_delay",
)
GLC_TARGET = "standardized_delay"
GLC_N_SPLITS = 5
GLC_ORDER_SPLIT_FILE = (
    GLC_EXPERIMENT_ROOT / "shared_sample_split" / "analysis_sample_split.csv"
)


def glc_canonical_component(series, city, column):
    if series.isna().any():
        raise ValueError(f"{city}: 分组或订单键字段 {column} 存在缺失")
    if glc_pd.api.types.is_numeric_dtype(series):
        numeric = glc_pd.to_numeric(series, errors="raise")
        if glc_np.all(glc_np.isclose(numeric, glc_np.round(numeric))):
            return glc_pd.Series(
                glc_np.round(numeric).astype(glc_np.int64).astype(str),
                index=series.index,
                dtype="string",
            )
    return series.astype("string").str.strip()


def glc_analysis_wave_id(frame, city):
    result = glc_pd.Series(city, index=frame.index, dtype="string")
    for column in GLC_GROUP_COLUMNS[city]:
        result = result + "::" + glc_canonical_component(
            frame[column], city, column
        )
    return result


def glc_prepare_development(city):
    """读取共享表中的development、analysis_wave_id和固定CV fold。"""

    group_columns = list(GLC_GROUP_COLUMNS[city])
    order_column = GLC_ORDER_ID_COLUMNS[city]
    lag_source = GLC_LAG_SOURCE[city]
    source_features = [
        lag_source if feature == "prev_standardized_delay" else feature
        for feature in GLC_FEATURES
    ]
    required = [
        *source_features,
        GLC_TARGET,
        order_column,
        *group_columns,
    ]
    frame = glc_pd.read_csv(GLC_DATA_FILES[city], usecols=required)
    if lag_source != "prev_standardized_delay":
        frame = frame.rename(columns={lag_source: "prev_standardized_delay"})
    frame = frame.loc[frame[GLC_TARGET].notna()].copy()
    frame["order_id"] = glc_canonical_component(
        frame[order_column], city, order_column
    )
    if frame["order_id"].duplicated().any():
        raise ValueError(f"{city}: 稳定订单键 {order_column} 不唯一")
    frame["analysis_wave_id"] = glc_analysis_wave_id(frame, city)
    if not GLC_ORDER_SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"缺少唯一共享划分表：{GLC_ORDER_SPLIT_FILE}\n"
            "请先运行正式训练单元生成划分表。"
        )
    assignment = glc_pd.read_csv(
        GLC_ORDER_SPLIT_FILE,
        dtype={
            "city": "string", "order_id": "string",
            "analysis_wave_id": "string", "sample_split": "string",
        },
    )
    assignment = assignment.loc[assignment["city"] == city].set_index(
        "order_id", verify_integrity=True
    )
    if set(frame["order_id"]) != set(assignment.index):
        raise ValueError(f"{city}: 学习曲线数据与共享样本划分表不一致")
    mapped = assignment.reindex(frame["order_id"].to_numpy())
    mapped.index = frame.index
    if not mapped["analysis_wave_id"].equals(frame["analysis_wave_id"]):
        raise ValueError(f"{city}: 学习曲线analysis_wave_id与共享表不一致")
    frame["sample_split"] = mapped["sample_split"]
    frame["cv_fold"] = mapped["cv_fold"].astype("Int64")
    X = frame.loc[:, list(GLC_FEATURES)].copy()
    y = frame[GLC_TARGET].copy()
    groups = frame["analysis_wave_id"]
    development_position = glc_np.flatnonzero(
        frame["sample_split"].eq("development").to_numpy()
    )
    return (
        X.iloc[development_position].copy(),
        y.iloc[development_position].copy(),
        groups.iloc[development_position].copy(),
        frame["cv_fold"].iloc[development_position].copy(),
    )


def glc_fixed_horizon_curves(city):
    """用相同锁定轮数训练5 folds并返回逐折、逐轮RMSE。"""

    params_file = GLC_MODEL_ROOT / city / "best_params.json"
    if not params_file.exists():
        raise FileNotFoundError(f"{city}: 缺少训练参数 {params_file}")
    payload = glc_json.loads(params_file.read_text(encoding="utf-8"))
    params = payload["best_params"]
    final_iteration = int(payload["best_iteration"])
    X_development, y_development, groups_development, cv_fold_development = (
        glc_prepare_development(city)
    )
    observed_folds = sorted(
        cv_fold_development.dropna().astype(int).unique()
    )
    expected_folds = list(range(1, GLC_N_SPLITS + 1))
    if observed_folds != expected_folds:
        raise ValueError(
            f"{city}: 共享划分表CV folds为{observed_folds}，预期{expected_folds}"
        )
    positions = glc_np.arange(len(cv_fold_development))
    fold_array = cv_fold_development.to_numpy(dtype=int)
    folds = [
        (positions[fold_array != fold], positions[fold_array == fold])
        for fold in expected_folds
    ]
    parts = []
    for fold_number, (train_position, validation_position) in enumerate(
        folds, start=1
    ):
        train_data = glc_lgb.Dataset(
            X_development.iloc[train_position],
            label=y_development.iloc[train_position],
            feature_name=list(GLC_FEATURES),
            free_raw_data=False,
        )
        validation_data = glc_lgb.Dataset(
            X_development.iloc[validation_position],
            label=y_development.iloc[validation_position],
            feature_name=list(GLC_FEATURES),
            reference=train_data,
            free_raw_data=False,
        )
        history = {}
        glc_lgb.train(
            params,
            train_data,
            num_boost_round=final_iteration,
            valid_sets=[train_data, validation_data],
            valid_names=["train", "validation"],
            callbacks=[
                glc_lgb.log_evaluation(False),
                glc_lgb.record_evaluation(history),
            ],
        )
        if len(history["validation"]["rmse"]) != final_iteration:
            raise AssertionError(
                f"{city} fold {fold_number}: 学习曲线未达到锁定轮数"
            )
        parts.append(
            glc_pd.DataFrame(
                {
                    "City": city,
                    "Fold": fold_number,
                    "round": glc_np.arange(1, final_iteration + 1),
                    "train_rmse": history["train"]["rmse"],
                    "validation_rmse": history["validation"]["rmse"],
                }
            )
        )
    return glc_pd.concat(parts, ignore_index=True), final_iteration


def glc_plot_fixed_curve(city, long_table, final_iteration):
    """沿用原学习曲线风格绘制完整五折均值。"""

    aggregate = (
        long_table.groupby("round", as_index=False)
        .agg(
            train_rmse=("train_rmse", "mean"),
            validation_rmse=("validation_rmse", "mean"),
            n_folds_contributing=("Fold", "nunique"),
        )
    )
    if not (aggregate["n_folds_contributing"] == GLC_N_SPLITS).all():
        raise AssertionError(f"{city}: 学习曲线存在不足5折参与的轮次")
    figure, axis = glc_plt.subplots(figsize=(10, 5))
    axis.plot(
        aggregate["round"],
        aggregate["train_rmse"],
        color="#4573b4",
        linewidth=1.5,
        label="Train RMSE",
    )
    axis.plot(
        aggregate["round"],
        aggregate["validation_rmse"],
        color="#d73221",
        linewidth=1.5,
        linestyle="--",
        label="Validation RMSE",
    )
    axis.axvline(
        x=final_iteration,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label=f"Final round ({final_iteration})",
    )
    axis.set_xlabel("Boosting Round", fontsize=14)
    axis.set_ylabel("RMSE", fontsize=14)
    axis.set_title("LightGBM Learning Curve", fontsize=16)
    axis.legend(fontsize=12)
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    city_result_dir = GLC_RESULT_ROOT / city
    figure.savefig(
        city_result_dir / "LGBM_learning_curve.pdf",
        bbox_inches="tight",
        dpi=300,
    )
    glc_plt.close(figure)
    aggregate.to_csv(
        city_result_dir / "LGBM_training_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    long_table.to_csv(
        city_result_dir / "grouped_5fold_learning_curve_fixed_horizon_long.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return aggregate


glc_protocol_rows = []
for glc_city in ("BJ", "BD"):
    print(f"\nRecomputing fixed-horizon grouped learning curve: {glc_city}")
    glc_long_curve, glc_final_iteration = glc_fixed_horizon_curves(glc_city)
    glc_aggregate_curve = glc_plot_fixed_curve(
        glc_city,
        glc_long_curve,
        glc_final_iteration,
    )
    glc_protocol_rows.append(
        {
            "City": glc_city,
            "Curve_source": "development only",
            "CV": "Persisted GroupKFold(n_splits=5) by date-courier-wave",
            "Rounds_per_fold": glc_final_iteration,
            "Fold_contribution_each_round": 5,
            "Parameters": "saved best grouped-5-fold Optuna parameters",
            "Test_used": False,
        }
    )

glc_protocol = glc_pd.DataFrame(glc_protocol_rows)
glc_protocol.to_csv(
    GLC_RESULT_ROOT / "learning_curve_protocol.csv",
    index=False,
    encoding="utf-8-sig",
)
print("\n========== Fixed-horizon learning-curve protocol ==========")
print(glc_protocol.to_string(index=False))

# In[]:
# =============================================================================
# 5. Common-support transfer analysis
# =============================================================================

import json as cs2_json
from dataclasses import dataclass as cs2_dataclass

import lightgbm as cs2_lgb
import numpy as cs2_np
import pandas as cs2_pd
from sklearn.metrics import mean_absolute_error as cs2_mae
from sklearn.metrics import mean_squared_error as cs2_mse
from sklearn.metrics import r2_score as cs2_r2


CS2_PROJECT_ROOT = GCV_ROOT
CS2_EXPERIMENT_ROOT = GCV_OUTPUT_ROOT
CS2_ROOT = CS2_EXPERIMENT_ROOT / "common_support_transfer"
CS2_ROOT.mkdir(parents=True, exist_ok=True)
CS2_MODEL_ROOT = CS2_EXPERIMENT_ROOT / "models"
CS2_METADATA_FILE = CS2_MODEL_ROOT / "model_metadata.json"
CS2_ORDER_SPLIT_FILE = (
    CS2_EXPERIMENT_ROOT / "shared_sample_split" / "analysis_sample_split.csv"
)
CS2_DATA_FILES = GCV_DATA_FILES
CS2_CITIES = ("BJ", "BD")
CS2_FEATURES = (
    "order_hourly",
    "is_weekend",
    "peak",
    "ride_distance_km",
    "merchant_preparation_time",
    "wave_duration_time",
    "loadorder_number",
    "courier_work_hours_now",
    "platform_assign_time",
    "demand_density",
    "supply_density",
    "road_length_km",
    "avg_node_degree",
    "prev_standardized_delay",
)
CS2_TARGET = "standardized_delay"
CS2_GROUP_COLUMNS = {
    "BJ": ("Date", "courier_id", "wave_index"),
    "BD": ("dt", "courier_id", "wave_id"),
}
CS2_ORDER_ID_COLUMNS = {"BJ": "source_row_id", "BD": "waybill_id"}
CS2_LAG_SOURCE = {"BJ": "prev_standardized_delay", "BD": "prior_order_delay"}


@cs2_dataclass
class CS2Split:
    X_development: cs2_pd.DataFrame
    X_test: cs2_pd.DataFrame
    y_development: cs2_pd.Series
    y_test: cs2_pd.Series


def cs2_canonical_component(series, city, column):
    if series.isna().any():
        raise ValueError(f"{city}: 分组或订单键字段 {column} 存在缺失")
    if cs2_pd.api.types.is_numeric_dtype(series):
        numeric = cs2_pd.to_numeric(series, errors="raise")
        if cs2_np.all(cs2_np.isclose(numeric, cs2_np.round(numeric))):
            return cs2_pd.Series(
                cs2_np.round(numeric).astype(cs2_np.int64).astype(str),
                index=series.index,
                dtype="string",
            )
    return series.astype("string").str.strip()


def cs2_analysis_wave_id(frame, city):
    result = cs2_pd.Series(city, index=frame.index, dtype="string")
    for column in CS2_GROUP_COLUMNS[city]:
        result = result + "::" + cs2_canonical_component(
            frame[column], city, column
        )
    return result


def cs2_prepare_outer_split(city, assignments):
    order_column = CS2_ORDER_ID_COLUMNS[city]
    lag_source = CS2_LAG_SOURCE[city]
    source_features = [
        lag_source if feature == "prev_standardized_delay" else feature
        for feature in CS2_FEATURES
    ]
    required = list(dict.fromkeys([
        *source_features,
        CS2_TARGET,
        order_column,
        *CS2_GROUP_COLUMNS[city],
    ]))
    frame = cs2_pd.read_csv(CS2_DATA_FILES[city], usecols=required)
    if lag_source != "prev_standardized_delay":
        frame = frame.rename(columns={lag_source: "prev_standardized_delay"})
    frame = frame.loc[frame[CS2_TARGET].notna()].copy()
    frame["order_id"] = cs2_canonical_component(
        frame[order_column], city, order_column
    )
    if frame["order_id"].duplicated().any():
        raise ValueError(f"{city}: 稳定订单键 {order_column} 不唯一")
    frame["analysis_wave_id"] = cs2_analysis_wave_id(frame, city)

    assignment = assignments.loc[assignments["city"] == city].set_index(
        "order_id", verify_integrity=True
    )
    if set(frame["order_id"]) != set(assignment.index):
        raise ValueError(f"{city}: 数据与共享样本划分表的订单集合不一致")
    mapped = assignment.reindex(frame["order_id"].to_numpy())
    mapped.index = frame.index
    if not mapped["analysis_wave_id"].equals(frame["analysis_wave_id"]):
        raise ValueError(f"{city}: analysis_wave_id 与共享样本划分表不一致")
    frame["sample_split"] = mapped["sample_split"]
    development = frame["sample_split"].eq("development")
    test = frame["sample_split"].eq("test")
    return CS2Split(
        X_development=frame.loc[development, list(CS2_FEATURES)].copy(),
        X_test=frame.loc[test, list(CS2_FEATURES)].copy(),
        y_development=frame.loc[development, CS2_TARGET].copy(),
        y_test=frame.loc[test, CS2_TARGET].copy(),
    )


if not CS2_METADATA_FILE.exists():
    raise FileNotFoundError(f"缺少训练元数据：{CS2_METADATA_FILE}")
if not CS2_ORDER_SPLIT_FILE.exists():
    raise FileNotFoundError(f"缺少共享样本划分表：{CS2_ORDER_SPLIT_FILE}")
cs2_metadata = cs2_json.loads(CS2_METADATA_FILE.read_text(encoding="utf-8"))
cs2_signature = cs2_metadata.get("signature", {})
if cs2_signature.get("features") != list(CS2_FEATURES):
    raise ValueError("Common-support 特征顺序与训练元数据不一致")
if cs2_signature.get("group_columns") != {
    city: list(CS2_GROUP_COLUMNS[city]) for city in CS2_CITIES
}:
    raise ValueError("Common-support 分组字段与训练元数据不一致")
for cs2_city in CS2_CITIES:
    cs2_saved_data = cs2_signature.get("data", {}).get(cs2_city, {})
    cs2_stat = CS2_DATA_FILES[cs2_city].stat()
    if (
        cs2_saved_data.get("size") != cs2_stat.st_size
        or cs2_saved_data.get("mtime_ns") != cs2_stat.st_mtime_ns
    ):
        raise ValueError(f"{cs2_city}: 当前数据文件与训练时的数据签名不一致")

cs2_assignments = cs2_pd.read_csv(
    CS2_ORDER_SPLIT_FILE,
    dtype={
        "city": "string",
        "order_id": "string",
        "analysis_wave_id": "string",
        "sample_split": "string",
    },
)
cs2_splits = {
    city: cs2_prepare_outer_split(city, cs2_assignments)
    for city in CS2_CITIES
}
cs2_models = {}
for cs2_city in CS2_CITIES:
    cs2_model_path = CS2_MODEL_ROOT / cs2_city / "model.txt"
    if not cs2_model_path.exists():
        raise FileNotFoundError(f"缺少 {cs2_city} 正式模型：{cs2_model_path}")
    # Loading through model_str avoids intermittent path parsing on Windows.
    cs2_models[cs2_city] = cs2_lgb.Booster(
        model_str=cs2_model_path.read_text(encoding="utf-8")
    )
    if cs2_models[cs2_city].feature_name() != list(CS2_FEATURES):
        raise ValueError(f"{cs2_city}: 模型特征顺序与 Common-support 配置不一致")

CS2_CATEGORICAL = ("is_weekend", "peak")
CS2_CONTINUOUS = tuple(
    feature for feature in CS2_FEATURES if feature not in CS2_CATEGORICAL
)

cs2_bounds = {}
cs2_bound_rows = []
for cs2_feature in CS2_CONTINUOUS:
    cs2_city_ranges = {}
    for cs2_city in CS2_CITIES:
        cs2_values = cs2_pd.to_numeric(
            cs2_splits[cs2_city].X_development[cs2_feature], errors="coerce"
        ).dropna()
        cs2_city_ranges[cs2_city] = (
            float(cs2_values.min()), float(cs2_values.max())
        )
    cs2_lower = max(value[0] for value in cs2_city_ranges.values())
    cs2_upper = min(value[1] for value in cs2_city_ranges.values())
    if cs2_lower > cs2_upper:
        raise ValueError(f"{cs2_feature} 的两城 development 范围没有交集。")
    cs2_bounds[cs2_feature] = (cs2_lower, cs2_upper)
    cs2_bound_rows.append(
        {
            "Feature": cs2_feature,
            "BJ_development_min": cs2_city_ranges["BJ"][0],
            "BJ_development_max": cs2_city_ranges["BJ"][1],
            "BD_development_min": cs2_city_ranges["BD"][0],
            "BD_development_max": cs2_city_ranges["BD"][1],
            "Common_lower": cs2_lower,
            "Common_upper": cs2_upper,
        }
    )

cs2_shared_levels = {}
cs2_level_rows = []
for cs2_feature in CS2_CATEGORICAL:
    cs2_levels = {
        city: set(
            cs2_splits[city].X_development[cs2_feature].dropna().unique().tolist()
        )
        for city in CS2_CITIES
    }
    cs2_shared = sorted(cs2_levels["BJ"] & cs2_levels["BD"])
    if not cs2_shared:
        raise ValueError(f"{cs2_feature} 没有两城共同水平。")
    cs2_shared_levels[cs2_feature] = cs2_shared
    cs2_level_rows.append(
        {
            "Feature": cs2_feature,
            "BJ_levels": repr(sorted(cs2_levels["BJ"])),
            "BD_levels": repr(sorted(cs2_levels["BD"])),
            "Shared_levels": repr(cs2_shared),
        }
    )


def cs2_mask(frame):
    """要求全部模型特征非缺失且同时落入两城 development 共同支持域。"""

    keep = ~frame.loc[:, list(CS2_FEATURES)].isna().any(axis=1)
    for feature, (lower, upper) in cs2_bounds.items():
        values = cs2_pd.to_numeric(frame[feature], errors="coerce")
        keep &= values.between(lower, upper, inclusive="both")
    for feature, levels in cs2_shared_levels.items():
        keep &= frame[feature].isin(levels)
    return keep


cs2_coverage_rows = []
cs2_test_masks = {}
for cs2_city in CS2_CITIES:
    cs2_split = cs2_splits[cs2_city]
    cs2_keep = cs2_mask(cs2_split.X_test)
    cs2_test_masks[cs2_city] = cs2_keep
    cs2_coverage_rows.append(
        {
            "City": cs2_city,
            "Test_n": len(cs2_keep),
            "Common_support_n": int(cs2_keep.sum()),
            "Common_support_share": float(cs2_keep.mean()),
            "Excluded_n": int((~cs2_keep).sum()),
        }
    )


def cs2_metric_row(source_city, target_city, sample, calibration, y, prediction):
    return {
        "Training_city": source_city,
        "Evaluation_city": target_city,
        "Evaluation_sample": sample,
        "Calibration": calibration,
        "n": len(y),
        "R_squared": float(cs2_r2(y, prediction)),
        "RMSE": float(cs2_np.sqrt(cs2_mse(y, prediction))),
        "MAE": float(cs2_mae(y, prediction)),
    }


cs2_metric_rows = []
for cs2_source in CS2_CITIES:
    cs2_model = cs2_models[cs2_source]
    for cs2_target in CS2_CITIES:
        cs2_target_split = cs2_splits[cs2_target]
        cs2_full_prediction = cs2_model.predict(cs2_target_split.X_test)
        cs2_development_prediction = cs2_model.predict(
            cs2_target_split.X_development
        )
        # Estimate the calibration offset from the target development set only.
        cs2_bias = float(
            (cs2_target_split.y_development - cs2_development_prediction).mean()
        )
        for cs2_sample_name, cs2_keep in (
            ("Full test set", cs2_pd.Series(True, index=cs2_target_split.X_test.index)),
            ("Common-support subset", cs2_test_masks[cs2_target]),
        ):
            cs2_y = cs2_target_split.y_test.loc[cs2_keep]
            cs2_pred = cs2_full_prediction[cs2_keep.to_numpy()]
            cs2_metric_rows.append(
                cs2_metric_row(
                    cs2_source, cs2_target, cs2_sample_name,
                    "Raw", cs2_y, cs2_pred,
                )
            )
            cs2_metric_rows.append(
                cs2_metric_row(
                    cs2_source, cs2_target, cs2_sample_name,
                    "Development-set constant-bias calibration",
                    cs2_y, cs2_pred + cs2_bias,
                )
            )

cs2_pd.DataFrame(cs2_bound_rows).to_csv(
    CS2_ROOT / "continuous_support_bounds.csv", index=False, encoding="utf-8-sig"
)
cs2_pd.DataFrame(cs2_level_rows).to_csv(
    CS2_ROOT / "shared_categorical_levels.csv", index=False, encoding="utf-8-sig"
)
cs2_coverage = cs2_pd.DataFrame(cs2_coverage_rows)
cs2_coverage.to_csv(
    CS2_ROOT / "common_support_coverage.csv", index=False, encoding="utf-8-sig"
)
cs2_metrics = cs2_pd.DataFrame(cs2_metric_rows)
cs2_metrics.to_csv(
    CS2_ROOT / "transfer_metrics.csv", index=False, encoding="utf-8-sig"
)
print("\n========== Common-support coverage ==========")
print(cs2_coverage.round(5).to_string(index=False))
print("\n========== Local and cross-context metrics ==========")
print(cs2_metrics.round(5).to_string(index=False))


# In[]:
# =============================================================================
# 6. Strict temporal holdout sensitivity analysis
# =============================================================================
# The temporal sensitivity analysis retunes within expanding windows and holds
# out the final 6 Beijing days and final 2 Baoding days.

TS_ROOT = GCV_OUTPUT_ROOT / "temporal_holdout_sensitivity"
TS_MODEL_ROOT = TS_ROOT / "models"
TS_RESULT_ROOT = TS_ROOT / "results"
TS_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
TS_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
TS_DATE_COLUMNS = {"BJ": "Date", "BD": "dt"}
TS_TEST_DATE_SHARE = 0.20
TS_N_TRIALS = 30


def ts_prepare_split(city):
    data_path = GCV_DATA_FILES[city]
    group_columns = list(GCV_GROUP_COLUMNS[city])
    date_column = TS_DATE_COLUMNS[city]
    if group_columns[0] != date_column:
        raise ValueError(f"{city}: 时间字段与统一分组键不一致")
    lag_source = GCV_LAG_SOURCE[city]
    source_features = [
        lag_source if feature == "prev_standardized_delay" else feature
        for feature in GCV_FEATURES
    ]
    required = [
        *source_features, GCV_TARGET, *group_columns
    ]
    frame = gcv_pd.read_csv(data_path, usecols=list(dict.fromkeys(required)))
    if lag_source != "prev_standardized_delay":
        frame = frame.rename(columns={lag_source: "prev_standardized_delay"})
    frame = frame.loc[frame[GCV_TARGET].notna()].copy()
    dates = gcv_pd.to_numeric(frame[date_column], errors="raise")
    unique_dates = gcv_np.sort(dates.unique())
    n_test_dates = max(1, int(gcv_np.ceil(len(unique_dates) * TS_TEST_DATE_SHARE)))
    test_dates = unique_dates[-n_test_dates:]
    development_dates = unique_dates[:-n_test_dates]
    if len(development_dates) < 3:
        raise ValueError(f"{city} 时间开发期日期不足，无法构造 expanding-window CV。")

    development_mask = dates.isin(development_dates)
    test_mask = dates.isin(test_dates)
    X = frame.loc[:, list(GCV_FEATURES)]
    y = frame[GCV_TARGET]
    development_frame = frame.loc[development_mask]
    test_frame = frame.loc[test_mask]
    groups_development = gcv_analysis_wave_id(development_frame, city)
    groups_test = gcv_analysis_wave_id(test_frame, city)
    split = GCVSplit(
        X_development=X.loc[development_mask].copy(),
        X_test=X.loc[test_mask].copy(),
        y_development=y.loc[development_mask].copy(),
        y_test=y.loc[test_mask].copy(),
        groups_development=groups_development,
        groups_test=groups_test,
        cv_fold_development=gcv_pd.Series(
            gcv_pd.NA,
            index=frame.index[development_mask],
            dtype="Int64",
        ),
        test_source_index=frame.index[test_mask],
        n_cleaned_total=len(frame),
        n_target_missing_removed=0,
        group_key_description=" + ".join(group_columns),
    )
    development_date_series = dates.loc[development_mask].reset_index(drop=True)
    return split, development_date_series, development_dates, test_dates


def ts_expanding_folds(development_dates):
    unique_dates = gcv_np.sort(development_dates.unique())
    n_folds = min(5, len(unique_dates) - 2)
    validation_dates = unique_dates[-n_folds:]
    folds = []
    audit = []
    for fold_number, validation_date in enumerate(validation_dates, start=1):
        train_position = gcv_np.flatnonzero(
            development_dates.to_numpy() < validation_date
        )
        valid_position = gcv_np.flatnonzero(
            development_dates.to_numpy() == validation_date
        )
        if len(train_position) == 0 or len(valid_position) == 0:
            continue
        folds.append((train_position, valid_position))
        audit.append(
            {
                "Fold": fold_number,
                "Train_last_date": int(development_dates.iloc[train_position].max()),
                "Validation_date": int(validation_date),
                "n_train": len(train_position),
                "n_validation": len(valid_position),
                "Temporal_order_valid": bool(
                    development_dates.iloc[train_position].max() < validation_date
                ),
            }
        )
    return folds, gcv_pd.DataFrame(audit)


ts_metric_rows = []
ts_split_rows = []
ts_fold_audits = []
ts_original_trial_count = GCV_N_TRIALS
GCV_N_TRIALS = TS_N_TRIALS
try:
    for ts_city in GCV_CITIES:
        print(f"\nTemporal holdout retraining: {ts_city}")
        ts_split, ts_development_dates, ts_development_unique, ts_test_dates = (
            ts_prepare_split(ts_city)
        )
        ts_folds, ts_fold_audit = ts_expanding_folds(ts_development_dates)
        ts_fold_audit.insert(0, "City", ts_city)
        ts_fold_audits.append(ts_fold_audit)
        ts_study, ts_best_params = gcv_tune_city(ts_city, ts_split, ts_folds)
        ts_fold_metrics, ts_curve, ts_best_iteration = gcv_refit_best_folds(
            ts_city, ts_split, ts_folds, ts_best_params
        )
        ts_model = gcv_fit_final_model(
            ts_split, ts_best_params, ts_best_iteration
        )
        ts_prediction = ts_model.predict(ts_split.X_test)
        ts_values = gcv_metric_values(ts_split.y_test, ts_prediction)
        ts_metric_rows.append(
            {
                "City": ts_city,
                "n_test": len(ts_split.y_test),
                **ts_values,
                "Best_iteration": ts_best_iteration,
                "Optuna_trials": TS_N_TRIALS,
            }
        )
        ts_split_rows.append(
            {
                "City": ts_city,
                "Development_first_date": int(ts_development_unique.min()),
                "Development_last_date": int(ts_development_unique.max()),
                "Test_first_date": int(ts_test_dates.min()),
                "Test_last_date": int(ts_test_dates.max()),
                "n_development_dates": len(ts_development_unique),
                "n_test_dates": len(ts_test_dates),
                "n_development": len(ts_split.y_development),
                "n_test": len(ts_split.y_test),
                "test_share": len(ts_split.y_test) / ts_split.n_cleaned_total,
                "Future_to_past_leakage": False,
            }
        )
        ts_city_model_root = TS_MODEL_ROOT / ts_city
        ts_city_result_root = TS_RESULT_ROOT / ts_city
        ts_city_model_root.mkdir(parents=True, exist_ok=True)
        ts_city_result_root.mkdir(parents=True, exist_ok=True)
        (ts_city_model_root / "model.txt").write_text(
            ts_model.model_to_string(), encoding="utf-8"
        )
        (ts_city_model_root / "best_params.json").write_text(
            gcv_json.dumps(
                {
                    "best_params": ts_best_params,
                    "best_iteration": ts_best_iteration,
                    "best_mean_expanding_window_RMSE": ts_study.best_value,
                    "n_trials": TS_N_TRIALS,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        ts_fold_metrics.to_csv(
            ts_city_result_root / "fold_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        ts_curve.to_csv(
            ts_city_result_root / "learning_curve_long.csv",
            index=False,
            encoding="utf-8-sig",
        )
finally:
    GCV_N_TRIALS = ts_original_trial_count

ts_metrics = gcv_pd.DataFrame(ts_metric_rows)
ts_splits = gcv_pd.DataFrame(ts_split_rows)
ts_folds = gcv_pd.concat(ts_fold_audits, ignore_index=True)
ts_metrics.to_csv(
    TS_RESULT_ROOT / "temporal_test_metrics.csv", index=False, encoding="utf-8-sig"
)
ts_splits.to_csv(
    TS_RESULT_ROOT / "temporal_split_audit.csv", index=False, encoding="utf-8-sig"
)
ts_folds.to_csv(
    TS_RESULT_ROOT / "expanding_window_fold_audit.csv",
    index=False,
    encoding="utf-8-sig",
)
print("\n========== Strict temporal split audit ==========")
print(ts_splits.to_string(index=False))
print("\n========== Temporal holdout performance ==========")
print(ts_metrics.round(5).to_string(index=False))


# In[]:
# =============================================================================
# 7. Model-implied scenario analysis
# =============================================================================
# Each scenario changes only the named feature. Reported differences are
# model-implied associations rather than causal effects.

import matplotlib as sc2_mpl


SC2_ROOT = GCV_OUTPUT_ROOT / "model_implied_scenarios"
SC2_ROOT.mkdir(parents=True, exist_ok=True)
SC2_BOOTSTRAP_REPLICATES = 1000
SC2_SEEDS = {"BJ": 20261002, "BD": 20261003}
SC2_CITY_LABEL = {"BJ": "Beijing", "BD": "Baoding"}
SC2_CITY_COLOR = {"BJ": "#6A95C4", "BD": "#E64B35"}
SC2_FIGURE_WIDTH_MM = 213.36
SC2_FIGURE_WIDTH_IN = SC2_FIGURE_WIDTH_MM / 25.4
SC2_FIGURE_HEIGHT_IN = 2.6
SC2_FIGURE_DPI = 600
SC2_FONT_FAMILY = gcv_os.environ.get(
    "DELIVERY_FIGURE_FONT", "Times New Roman"
)


def sc2_cluster_mean_interval(values, groups, seed):
    values = gcv_np.asarray(values, dtype=float)
    groups = gcv_pd.Series(groups).astype(str).reset_index(drop=True)
    unique_groups, group_codes = gcv_np.unique(groups, return_inverse=True)
    group_sum = gcv_np.bincount(group_codes, weights=values)
    group_n = gcv_np.bincount(group_codes)
    rng = gcv_np.random.default_rng(seed)
    estimates = gcv_np.empty(SC2_BOOTSTRAP_REPLICATES, dtype=float)
    for replicate in range(SC2_BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        estimates[replicate] = group_sum[sampled].sum() / group_n[sampled].sum()
    return (
        float(values.mean()),
        float(gcv_np.quantile(estimates, 0.025)),
        float(gcv_np.quantile(estimates, 0.975)),
    )


def sc2_configure_matplotlib():
    sc2_mpl.rcParams.update(
        {
            "font.family": SC2_FONT_FAMILY,
            "font.sans-serif": ["Arial", "Liberation Sans", "sans-serif"],
            "font.size": 8.2,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "lines.linewidth": 1.45,
            "lines.markersize": 4.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def sc2_save_figure(figure, output_stem):
    figure.savefig(output_stem.with_suffix(".png"), dpi=SC2_FIGURE_DPI)
    figure.savefig(output_stem.with_suffix(".pdf"))
    figure.savefig(output_stem.with_suffix(".svg"))
    figure.savefig(
        output_stem.with_suffix(".tiff"),
        dpi=SC2_FIGURE_DPI,
        pil_kwargs={"compression": "tiff_lzw"},
    )


def sc2_style_axis(axis):
    axis.tick_params(direction="out", pad=2)
    for spine in axis.spines.values():
        spine.set_color("#444444")
        spine.set_linewidth(0.75)
    axis.grid(False)


def sc2_label_points(axis, x, y, y_offset_points=5, x_offset_points=0):
    x = gcv_np.asarray(x, dtype=float)
    y = gcv_np.asarray(y, dtype=float)
    x_offsets = gcv_np.broadcast_to(
        gcv_np.asarray(x_offset_points, dtype=float), x.shape
    )
    for x_value, y_value, x_offset in zip(x, y, x_offsets):
        axis.annotate(
            f"{y_value:.3f}",
            (x_value, y_value),
            xytext=(x_offset, y_offset_points),
            textcoords="offset points",
            ha="center",
            va="bottom" if y_offset_points >= 0 else "top",
            fontsize=6.7,
            color="#222222",
        )


def sc2_label_bars(
    axis,
    bars,
    x_offset_points=0,
    y_offset_points=3,
    label_zeros=False,
):
    for bar in bars:
        value = float(bar.get_height())
        if gcv_np.isclose(value, 0.0) and not label_zeros:
            continue
        axis.annotate(
            f"{value:.3f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(x_offset_points, y_offset_points),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.7,
            color="#222222",
        )


def sc2_subgroup_value(results, city, dimension, group_key, scenario_key):
    matched = results.loc[
        results["city"].eq(city)
        & results["dimension"].eq(dimension)
        & results["group_key"].eq(group_key)
        & results["scenario_key"].eq(scenario_key),
        "mean_model_implied_reduction",
    ]
    if len(matched) != 1:
        raise ValueError(
            "Expected one subgroup result for "
            f"{city}/{dimension}/{group_key}/{scenario_key}; found {len(matched)}"
        )
    return float(matched.iloc[0])


sc2_rows = []
sc2_subgroup_rows = []
sc2_sample_audit = {}
for sc2_city in GCV_CITIES:
    sc2_split = gcv_splits[sc2_city]
    sc2_model = gcv_models[sc2_city]
    sc2_X = sc2_split.X_test.copy()
    sc2_groups = sc2_split.groups_test.reset_index(drop=True)
    sc2_baseline = sc2_model.predict(
        sc2_X, num_threads=GCV_NUM_THREADS
    )
    sc2_sample_audit[sc2_city] = {
        "test_orders": int(len(sc2_X)),
        "test_waves": int(sc2_groups.nunique()),
        "missing_feature_values": {
            feature: int(count)
            for feature, count in sc2_X.isna().sum().items()
            if int(count) > 0
        },
    }

    sc2_scenarios = []
    for minutes in range(1, 6):
        changed = sc2_X.copy()
        changed["merchant_preparation_time"] = gcv_np.maximum(
            0.0, sc2_X["merchant_preparation_time"] - minutes
        )
        affected = changed["merchant_preparation_time"].ne(
            sc2_X["merchant_preparation_time"]
        )
        sc2_scenarios.append(
            (
                "prep_reduction",
                "Preparation-time reduction",
                minutes,
                changed,
                affected,
            )
        )

    changed = sc2_X.copy()
    changed["merchant_preparation_time"] = gcv_np.minimum(
        sc2_X["merchant_preparation_time"], 18.0
    )
    sc2_scenarios.append(
        (
            "prep_cap",
            "Preparation-time cap",
            18,
            changed,
            changed["merchant_preparation_time"].ne(
                sc2_X["merchant_preparation_time"]
            ),
        )
    )

    for cap in (40, 50, 60):
        changed = sc2_X.copy()
        changed["wave_duration_time"] = gcv_np.minimum(
            sc2_X["wave_duration_time"], float(cap)
        )
        wave_affected = changed["wave_duration_time"].ne(
            sc2_X["wave_duration_time"]
        )
        sc2_scenarios.append(
            ("wave_cap", "Wave-duration cap", cap, changed, wave_affected)
        )

        combined = changed.copy()
        combined["merchant_preparation_time"] = gcv_np.minimum(
            sc2_X["merchant_preparation_time"], 18.0
        )
        combined_affected = wave_affected | combined[
            "merchant_preparation_time"
        ].ne(sc2_X["merchant_preparation_time"])
        sc2_scenarios.append(
            (
                "combined_cap",
                "Combined preparation-18 and wave cap",
                cap,
                combined,
                combined_affected,
            )
        )

    for scenario_key, scenario_name, level, changed, affected in sc2_scenarios:
        scenario_prediction = sc2_model.predict(
            changed, num_threads=GCV_NUM_THREADS
        )
        reduction = sc2_baseline - scenario_prediction
        mean_value, lower, upper = sc2_cluster_mean_interval(
            reduction,
            sc2_groups,
            SC2_SEEDS[sc2_city] + int(level),
        )
        sc2_rows.append(
            {
                "city": sc2_city,
                "city_label": SC2_CITY_LABEL[sc2_city],
                "scenario_key": scenario_key,
                "scenario": scenario_name,
                "level_minutes": level,
                "test_orders": len(reduction),
                "test_waves": int(sc2_groups.nunique()),
                "affected_orders": int(affected.sum()),
                "affected_share": float(affected.mean()),
                "baseline_prediction_mean": float(sc2_baseline.mean()),
                "scenario_prediction_mean": float(scenario_prediction.mean()),
                "mean_model_implied_reduction": mean_value,
                "cluster_bootstrap_95ci_lower": lower,
                "cluster_bootstrap_95ci_upper": upper,
                "share_with_lower_predicted_delay": float((reduction > 0).mean()),
                "interpretation": "model-implied association; not a causal effect",
            }
        )

    prep_groups = (
        ("prep_le_18", "Prep ≤18", sc2_X["merchant_preparation_time"].le(18)),
        (
            "prep_18_30",
            "Prep 18–30",
            sc2_X["merchant_preparation_time"].gt(18)
            & sc2_X["merchant_preparation_time"].le(30),
        ),
        ("prep_gt_30", "Prep >30", sc2_X["merchant_preparation_time"].gt(30)),
    )
    prep_minus_5 = sc2_X.copy()
    prep_minus_5["merchant_preparation_time"] = gcv_np.maximum(
        0.0, sc2_X["merchant_preparation_time"] - 5.0
    )
    prep_cap_18 = sc2_X.copy()
    prep_cap_18["merchant_preparation_time"] = gcv_np.minimum(
        sc2_X["merchant_preparation_time"], 18.0
    )
    prep_scenarios = (
        ("prep_minus_5", "Prep -5 min", prep_minus_5),
        ("prep_cap_18", "Prep cap 18 min", prep_cap_18),
    )
    for group_key, group_label, mask in prep_groups:
        if not mask.any():
            continue
        mask_array = mask.to_numpy()
        for scenario_key, scenario_label, changed in prep_scenarios:
            scenario_prediction = sc2_model.predict(
                changed.loc[mask], num_threads=GCV_NUM_THREADS
            )
            reduction = sc2_baseline[mask_array] - scenario_prediction
            sc2_subgroup_rows.append(
                {
                    "city": sc2_city,
                    "city_label": SC2_CITY_LABEL[sc2_city],
                    "dimension": "merchant_preparation_time",
                    "group_key": group_key,
                    "group_label": group_label,
                    "scenario_key": scenario_key,
                    "scenario": scenario_label,
                    "orders": int(mask.sum()),
                    "waves": int(sc2_groups.loc[mask_array].nunique()),
                    "mean_model_implied_reduction": float(reduction.mean()),
                }
            )

    wave_cap_40 = sc2_X.copy()
    wave_cap_40["wave_duration_time"] = gcv_np.minimum(
        sc2_X["wave_duration_time"], 40.0
    )
    wave_groups = (
        (
            "wave_40_60",
            "Wave 40–60",
            sc2_X["wave_duration_time"].gt(40)
            & sc2_X["wave_duration_time"].le(60),
        ),
        ("wave_gt_60", "Wave >60", sc2_X["wave_duration_time"].gt(60)),
    )
    for group_key, group_label, mask in wave_groups:
        if not mask.any():
            continue
        mask_array = mask.to_numpy()
        scenario_prediction = sc2_model.predict(
            wave_cap_40.loc[mask], num_threads=GCV_NUM_THREADS
        )
        reduction = sc2_baseline[mask_array] - scenario_prediction
        sc2_subgroup_rows.append(
            {
                "city": sc2_city,
                "city_label": SC2_CITY_LABEL[sc2_city],
                "dimension": "wave_duration_time",
                "group_key": group_key,
                "group_label": group_label,
                "scenario_key": "wave_cap_40",
                "scenario": "Wave cap 40 min",
                "orders": int(mask.sum()),
                "waves": int(sc2_groups.loc[mask_array].nunique()),
                "mean_model_implied_reduction": float(reduction.mean()),
            }
        )

sc2_results = gcv_pd.DataFrame(sc2_rows)
sc2_subgroup_results = gcv_pd.DataFrame(sc2_subgroup_rows)
sc2_results.to_csv(
    SC2_ROOT / "scenario_results.csv", index=False, encoding="utf-8-sig"
)
sc2_subgroup_results.to_csv(
    SC2_ROOT / "scenario_subgroup_results.csv",
    index=False,
    encoding="utf-8-sig",
)


def sc2_draw_overall(results):
    figure, axes = gcv_plt.subplots(
        1,
        3,
        figsize=(SC2_FIGURE_WIDTH_IN, SC2_FIGURE_HEIGHT_IN),
        gridspec_kw={"width_ratios": [1.12, 0.90, 1.12]},
    )

    axis = axes[0]
    for city in GCV_CITIES:
        subset = results.loc[
            results["city"].eq(city)
            & results["scenario_key"].eq("prep_reduction")
        ].sort_values("level_minutes")
        x = subset["level_minutes"].to_numpy(dtype=float)
        y = subset["mean_model_implied_reduction"].to_numpy(dtype=float)
        axis.plot(x, y, marker="o", color=SC2_CITY_COLOR[city], label=SC2_CITY_LABEL[city])
        if city == "BJ":
            sc2_label_points(axis, x[:1], y[:1], 13, gcv_np.array([12]))
            sc2_label_points(axis, x[1:], y[1:], 5, gcv_np.zeros(len(x) - 1))
        else:
            sc2_label_points(axis, x, y, -8, gcv_np.array([12, 0, 0, 0, 0]))
    axis.set_xlabel("Reduction in merchant preparation time (min)")
    axis.set_ylabel(
        "Reduction in model-implied relative delivery delay",
        fontsize=7.0,
        labelpad=5,
    )
    axis.set_xticks([1, 2, 3, 4, 5])
    axis.set_xlim(0.55, 5.35)
    axis.set_ylim(
        0,
        results.loc[
            results["scenario_key"].eq("prep_reduction"),
            "mean_model_implied_reduction",
        ].max()
        * 1.23,
    )
    axis.legend(loc="upper left", frameon=False, handlelength=1.5, handletextpad=0.4)
    sc2_style_axis(axis)

    axis = axes[1]
    subset = results.loc[results["scenario_key"].eq("prep_cap")].set_index("city").loc[list(GCV_CITIES)]
    bars = axis.bar(
        [0, 1],
        subset["mean_model_implied_reduction"].to_numpy(dtype=float),
        width=0.68,
        color=[SC2_CITY_COLOR["BJ"], SC2_CITY_COLOR["BD"]],
    )
    axis.set_xticks([0, 1], ["Beijing", "Baoding"])
    axis.set_xlabel("Preparation-time cap: 18 min")
    axis.set_ylim(0, subset["mean_model_implied_reduction"].max() * 1.28)
    sc2_label_bars(axis, bars)
    sc2_style_axis(axis)

    axis = axes[2]
    for city in GCV_CITIES:
        subset = results.loc[
            results["city"].eq(city)
            & results["scenario_key"].eq("wave_cap")
        ].sort_values("level_minutes")
        x = subset["level_minutes"].to_numpy(dtype=float)
        y = subset["mean_model_implied_reduction"].to_numpy(dtype=float)
        axis.plot(x, y, marker="o", color=SC2_CITY_COLOR[city], label=SC2_CITY_LABEL[city])
        sc2_label_points(axis, x, y, 5, gcv_np.array([8, 0, -6]))
    axis.set_xlabel("Wave-duration cap (min)")
    axis.set_xticks([40, 50, 60])
    axis.set_xlim(39.0, 61.4)
    axis.set_ylim(
        0,
        results.loc[
            results["scenario_key"].eq("wave_cap"),
            "mean_model_implied_reduction",
        ].max()
        * 1.28,
    )
    axis.legend(loc="upper right", frameon=False, handlelength=1.5, handletextpad=0.4)
    sc2_style_axis(axis)

    figure.subplots_adjust(
        left=0.105, right=0.985, bottom=0.225, top=0.955, wspace=0.28
    )
    sc2_save_figure(figure, SC2_ROOT / "scenario_overall")
    gcv_plt.close(figure)


def sc2_draw_subgroups(results):
    figure, axes = gcv_plt.subplots(
        1,
        2,
        figsize=(SC2_FIGURE_WIDTH_IN, SC2_FIGURE_HEIGHT_IN),
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )

    axis = axes[0]
    prep_groups = ["prep_le_18", "prep_18_30", "prep_gt_30"]
    prep_labels = ["Prep ≤18", "Prep 18–30", "Prep >30"]
    series = (
        ("BJ", "prep_minus_5", "Beijing, Prep -5 min", "#5B88B5"),
        ("BD", "prep_minus_5", "Baoding, Prep -5 min", "#E64B35"),
        ("BJ", "prep_cap_18", "Beijing, Prep cap 18 min", "#B7DFE8"),
        ("BD", "prep_cap_18", "Baoding, Prep cap 18 min", "#E38A7C"),
    )
    centers = gcv_np.arange(len(prep_groups), dtype=float)
    width = 0.16
    offsets = gcv_np.array([-1.8, -0.6, 0.6, 1.8]) * width
    label_x_offsets = [-3.0, 3.0, -3.0, 3.0]
    label_y_offsets = [3.0, 9.0, 3.0, 3.0]
    max_value = 0.0
    for offset, x_text, y_text, (city, scenario_key, label, color) in zip(
        offsets, label_x_offsets, label_y_offsets, series
    ):
        values = gcv_np.array(
            [
                sc2_subgroup_value(
                    results,
                    city,
                    "merchant_preparation_time",
                    group_key,
                    scenario_key,
                )
                for group_key in prep_groups
            ]
        )
        bars = axis.bar(
            centers + offset, values, width=width, color=color, label=label
        )
        sc2_label_bars(axis, bars, x_text, y_text)
        max_value = max(max_value, float(values.max()))
    axis.set_xticks(centers, prep_labels)
    axis.set_ylabel(
        "Reduction in model-implied relative delivery delay",
        fontsize=7.0,
        labelpad=5,
    )
    axis.set_ylim(0, max_value * 1.28)
    axis.legend(loc="upper left", frameon=False, handlelength=1.8)
    sc2_style_axis(axis)

    axis = axes[1]
    wave_groups = ["wave_40_60", "wave_gt_60"]
    wave_labels = ["Wave 40–60", "Wave >60"]
    centers = gcv_np.arange(len(wave_groups), dtype=float)
    width = 0.28
    max_value = 0.0
    for offset, city in ((-width / 2, "BJ"), (width / 2, "BD")):
        values = gcv_np.array(
            [
                sc2_subgroup_value(
                    results,
                    city,
                    "wave_duration_time",
                    group_key,
                    "wave_cap_40",
                )
                for group_key in wave_groups
            ]
        )
        bars = axis.bar(
            centers + offset,
            values,
            width=width,
            color=SC2_CITY_COLOR[city],
            label=f"{SC2_CITY_LABEL[city]}, Wave cap 40 min",
        )
        sc2_label_bars(axis, bars)
        max_value = max(max_value, float(values.max()))
    axis.set_xticks(centers, wave_labels)
    axis.set_ylim(0, max_value * 1.30)
    axis.legend(loc="upper left", frameon=False, handlelength=1.8)
    sc2_style_axis(axis)

    figure.subplots_adjust(
        left=0.105, right=0.985, bottom=0.18, top=0.955, wspace=0.22
    )
    sc2_save_figure(figure, SC2_ROOT / "scenario_subgroups")
    gcv_plt.close(figure)


sc2_configure_matplotlib()
sc2_draw_overall(sc2_results)
sc2_draw_subgroups(sc2_subgroup_results)
gcv_atomic_write_json(
    {
        "created_at": gcv_datetime.now().astimezone().isoformat(),
        "interpretation": "model-implied association; not a causal effect",
        "bootstrap_replicates": SC2_BOOTSTRAP_REPLICATES,
        "bootstrap_seeds": SC2_SEEDS,
        "test_samples": sc2_sample_audit,
        "figure_width_inches": SC2_FIGURE_WIDTH_IN,
        "figure_width_mm": SC2_FIGURE_WIDTH_MM,
        "figure_height_inches": SC2_FIGURE_HEIGHT_IN,
        "figure_dpi": SC2_FIGURE_DPI,
        "font_family": SC2_FONT_FAMILY,
    },
    SC2_ROOT / "scenario_analysis_metadata.json",
)

print("\n========== Model-implied scenario results ==========")
print(sc2_results.round(5).to_string(index=False))
print("\n========== Scenario subgroup results ==========")
print(sc2_subgroup_results.round(5).to_string(index=False))

