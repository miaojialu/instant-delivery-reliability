# -*- coding: utf-8 -*-
"""Additional analyses for the Beijing and Baoding LightGBM study.

The script reuses the development/test assignment, five grouped folds, and
selected LightGBM recipe produced by the main experiment. It implements OLS
and additive-spline benchmarks, grouped bootstrap stability checks, robustness
analyses, and outcome description. 
"""

# In[]: 0. Shared configuration and persisted sample partition
# ============================================================================

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import lightgbm as lgb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else Path(default)


DATA_PATHS = {
    "BJ": env_path("DELIVERY_BJ_DATA", ROOT / "北京订单数据20261019.csv"),
    "BD": env_path("DELIVERY_BD_DATA", ROOT / "保定订单数据20261019.csv"),
}
TRAINING_ROOT = env_path(
    "DELIVERY_OUTPUT_DIR", ROOT / "outputs" / "grouped_5fold_lightgbm"
)
DEFAULT_OUTPUT_ROOT = env_path(
    "DELIVERY_ADDITIONAL_OUTPUT_DIR", ROOT / "outputs" / "additional_analyses"
)

TARGET = "standardized_delay"
DURATION_COLUMNS = ("finish_time", "estimate_finish_time")
FEATURES = (
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
BINARY_FEATURES = ("is_weekend", "peak")
CONTINUOUS_FEATURES = tuple(x for x in FEATURES if x not in BINARY_FEATURES)

GROUP_COLUMNS = {
    "BJ": ("Date", "courier_id", "wave_index"),
    "BD": ("dt", "courier_id", "wave_id"),
}
ORDER_ID_COLUMNS = {"BJ": "source_row_id", "BD": "waybill_id"}
LAG_SOURCE_COLUMN = {"BJ": "prev_standardized_delay", "BD": "prior_order_delay"}
ORDER_SPLIT_FILE = (
    TRAINING_ROOT / "shared_sample_split" / "analysis_sample_split.csv"
)
WAVE_SPLIT_FILE = (
    TRAINING_ROOT / "shared_sample_split" / "analysis_wave_split.csv"
)
SPLIT_METADATA_FILE = (
    TRAINING_ROOT / "shared_sample_split" / "split_metadata.json"
)

TEST_SIZE = 0.20
SPLIT_SEED = 42
MODEL_SEED = 20260826
RESAMPLE_SEED = 20261002
GAM_ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)
LINEAR_MODEL_NAME = "Linear model (ordinary least squares)"

ALE_FEATURES = (
    "ride_distance_km",
    "merchant_preparation_time",
    "wave_duration_time",
    "loadorder_number",
    "platform_assign_time",
    "prev_standardized_delay",
)
INTERACTION_PAIRS = (
    ("merchant_preparation_time", "wave_duration_time"),
    ("wave_duration_time", "loadorder_number"),
    ("wave_duration_time", "prev_standardized_delay"),
    ("ride_distance_km", "loadorder_number"),
)
FEATURE_LABELS = {
    "ride_distance_km": "Riding distance",
    "merchant_preparation_time": "Merchant preparation time",
    "wave_duration_time": "Wave duration",
    "loadorder_number": "Order load",
    "platform_assign_time": "Platform assignment time",
    "prev_standardized_delay": "Prior standardized delay",
}
EXTREME_FEATURE_UNITS = {
    "loadorder_number": "orders per wave",
    "wave_duration_time": "minutes",
}

FIGURE_WIDTH_MM = 183.0
FIGURE_WIDTH_IN = FIGURE_WIDTH_MM / 25.4
FIGURE_DPI = 600
FIGURE_FONT = os.environ.get("DELIVERY_FIGURE_FONT", "Times New Roman")
mpl.rcParams.update({
    "font.family": FIGURE_FONT,
    "font.sans-serif": ["Arial", "Liberation Sans", "sans-serif"],
    "font.size": 7.0,
    "axes.labelsize": 7.0,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


@dataclass
class CityData:
    city: str
    frame: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    cv_fold: pd.Series
    development_index: np.ndarray
    test_index: np.ndarray

    @property
    def X_development(self) -> pd.DataFrame:
        return self.X.iloc[self.development_index]

    @property
    def y_development(self) -> pd.Series:
        return self.y.iloc[self.development_index]

    @property
    def groups_development(self) -> pd.Series:
        return self.groups.iloc[self.development_index]

    @property
    def cv_fold_development(self) -> pd.Series:
        return self.cv_fold.iloc[self.development_index]

    @property
    def X_test(self) -> pd.DataFrame:
        return self.X.iloc[self.test_index]

    @property
    def y_test(self) -> pd.Series:
        return self.y.iloc[self.test_index]


def atomic_to_csv(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Write atomically so an interrupted run cannot leave a partial CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tmp", delete=False, dir=path.parent,
        encoding="utf-8-sig", newline="",
    ) as handle:
        temporary_path = Path(handle.name)
        frame.to_csv(handle, index=index)
    os.replace(temporary_path, path)


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tmp", delete=False, dir=path.parent,
        encoding="utf-8",
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def save_publication_figure(figure: mpl.figure.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".png"), dpi=FIGURE_DPI, bbox_inches="tight")
    figure.savefig(path.with_suffix(".tiff"), dpi=FIGURE_DPI, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight")


def save_booster(model: lgb.Booster, path: Path) -> None:
    """Write through a temporary UTF-8 file for reliable Windows paths."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tmp", delete=False, dir=path.parent, encoding="utf-8"
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(model.model_to_string())
    os.replace(temporary_path, path)


def metric_row(city: str, analysis: str, model: str, sample: str,
               y_true: pd.Series | np.ndarray,
               prediction: np.ndarray) -> dict:
    y_array = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    return {
        "city": city,
        "analysis": analysis,
        "model": model,
        "evaluation_sample": sample,
        "n": int(len(y_array)),
        "r2": float(r2_score(y_array, prediction)),
        "rmse": float(mean_squared_error(y_array, prediction, squared=False)),
        "mae": float(mean_absolute_error(y_array, prediction)),
    }


def canonical_component(series: pd.Series, city: str, column: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{city}: 分组或订单键字段 {column} 存在缺失")
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="raise")
        if np.all(np.isclose(numeric, np.round(numeric))):
            return pd.Series(
                np.round(numeric).astype(np.int64).astype(str),
                index=series.index,
                dtype="string",
            )
    return series.astype("string").str.strip()


def analysis_wave_id(frame: pd.DataFrame, city: str) -> pd.Series:
    result = pd.Series(city, index=frame.index, dtype="string")
    for column in GROUP_COLUMNS[city]:
        result = result + "::" + canonical_component(
            frame[column], city, column
        )
    return result.rename("analysis_wave_id")


def load_shared_assignment(city: str) -> pd.DataFrame:
    required_files = (ORDER_SPLIT_FILE, WAVE_SPLIT_FILE, SPLIT_METADATA_FILE)
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "缺少正式实验生成的唯一共享划分文件：\n- "
            + "\n- ".join(missing)
            + "\n请先运行完整实验脚本的正式训练单元。"
        )
    metadata = json.loads(SPLIT_METADATA_FILE.read_text(encoding="utf-8"))
    saved_groups = metadata.get("group_columns", {})
    if saved_groups.get(city) != list(GROUP_COLUMNS[city]):
        raise ValueError(
            f"{city}: 共享划分元数据中的分组键为{saved_groups.get(city)}，"
            f"当前脚本要求{list(GROUP_COLUMNS[city])}"
        )
    wave_table = pd.read_csv(
        WAVE_SPLIT_FILE,
        dtype={"city": "string", "analysis_wave_id": "string"},
    )
    if wave_table["analysis_wave_id"].duplicated().any():
        raise ValueError("共享波次划分表中的analysis_wave_id不唯一")
    assignment = pd.read_csv(
        ORDER_SPLIT_FILE,
        dtype={
            "city": "string", "order_id": "string",
            "analysis_wave_id": "string", "sample_split": "string",
        },
    )
    assignment = assignment.loc[assignment["city"] == city].copy()
    if assignment.empty:
        raise ValueError(f"共享样本划分表中没有 {city}")
    if assignment["order_id"].duplicated().any():
        raise ValueError(f"共享样本划分表中 {city} order_id 不唯一")
    valid_wave_ids = set(
        wave_table.loc[wave_table["city"] == city, "analysis_wave_id"]
    )
    if not set(assignment["analysis_wave_id"]).issubset(valid_wave_ids):
        raise ValueError(f"{city}: 订单划分表引用了波次表中不存在的analysis_wave_id")
    return assignment.set_index("order_id", verify_integrity=True)


def load_city_data(city: str, smoke_test: bool = False,
                   smoke_groups: int = 500) -> CityData:
    path = DATA_PATHS[city]
    if not path.exists():
        raise FileNotFoundError(path)

    header = pd.read_csv(path, nrows=0).columns.tolist()
    lag_source = LAG_SOURCE_COLUMN[city]
    order_column = ORDER_ID_COLUMNS[city]
    required = [
        *(x for x in FEATURES if x != "prev_standardized_delay"),
        lag_source, TARGET, order_column, *DURATION_COLUMNS, *GROUP_COLUMNS[city],
    ]
    missing = [column for column in required if column not in header]
    if missing:
        raise KeyError(f"{city} 缺少必要字段：{missing}")

    frame = pd.read_csv(path, usecols=list(dict.fromkeys(required)))
    frame["source_row_number"] = frame.index.astype(np.int64)
    if lag_source != "prev_standardized_delay":
        frame = frame.rename(columns={lag_source: "prev_standardized_delay"})

    numeric_columns = [*FEATURES, TARGET, *DURATION_COLUMNS]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[frame[TARGET].notna()].reset_index(drop=True)
    frame["order_id"] = canonical_component(
        frame[order_column], city, order_column
    )
    if frame["order_id"].duplicated().any():
        raise ValueError(f"{city}: 稳定订单键 {order_column} 不唯一")
    frame["analysis_wave_id"] = analysis_wave_id(frame, city)
    assignment = load_shared_assignment(city)
    if set(frame["order_id"]) != set(assignment.index):
        raise ValueError(f"{city}: 补充分析订单与共享样本划分表不一致")
    mapped = assignment.reindex(frame["order_id"].to_numpy())
    mapped.index = frame.index
    if not mapped["analysis_wave_id"].equals(frame["analysis_wave_id"]):
        raise ValueError(f"{city}: 补充分析analysis_wave_id与共享表不一致")
    frame["sample_split"] = mapped["sample_split"]
    frame["cv_fold"] = mapped["cv_fold"].astype("Int64")
    groups = frame["analysis_wave_id"].copy()

    if smoke_test:
        # Preserve complete analysis-wave groups in the smoke test.
        selected_groups = pd.Index(groups.drop_duplicates().iloc[:smoke_groups])
        keep = groups.isin(selected_groups)
        frame = frame.loc[keep].reset_index(drop=True)
        groups = groups.loc[keep].reset_index(drop=True)

    X = frame.loc[:, list(FEATURES)].copy()
    y = frame[TARGET].copy()
    cv_fold = frame["cv_fold"].copy()
    development_index = np.flatnonzero(
        frame["sample_split"].eq("development").to_numpy()
    )
    test_index = np.flatnonzero(frame["sample_split"].eq("test").to_numpy())
    if len(development_index) == 0 or len(test_index) == 0:
        raise ValueError(f"{city}: 共享划分在当前样本中缺少development或test")
    return CityData(
        city=city,
        frame=frame,
        X=X,
        y=y,
        groups=groups,
        cv_fold=cv_fold,
        development_index=np.asarray(development_index),
        test_index=np.asarray(test_index),
    )


def save_split_audit(data_by_city: dict[str, CityData], output_root: Path) -> None:
    rows = []
    for city, data in data_by_city.items():
        development_groups = set(data.groups_development.astype(str))
        test_groups = set(data.groups.iloc[data.test_index].astype(str))
        rows.append({
            "city": city,
            "n_total": len(data.frame),
            "n_development": len(data.development_index),
            "development_share": len(data.development_index) / len(data.frame),
            "n_test": len(data.test_index),
            "test_share": len(data.test_index) / len(data.frame),
            "n_groups_total": data.groups.nunique(),
            "n_groups_development": len(development_groups),
            "n_groups_test": len(test_groups),
            "overlapping_groups": len(development_groups & test_groups),
            "group_columns": " + ".join(GROUP_COLUMNS[city]),
            "split_seed": SPLIT_SEED,
            "shared_order_split_file": str(ORDER_SPLIT_FILE),
            "shared_wave_split_file": str(WAVE_SPLIT_FILE),
        })
    atomic_to_csv(pd.DataFrame(rows), output_root / "00_split_audit.csv")


def stored_cv_splits(data: CityData, requested_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read the persisted folds instead of repartitioning downstream."""

    fold_series = data.cv_fold_development.astype("Int64")
    observed = sorted(fold_series.dropna().astype(int).unique())
    if requested_folds != len(observed):
        raise ValueError(
            f"{data.city}: 请求{requested_folds}折，但共享划分表固定为{observed}"
        )
    expected = list(range(1, requested_folds + 1))
    if observed != expected:
        raise ValueError(f"{data.city}: 共享CV fold编号为{observed}，预期{expected}")
    fold_array = fold_series.to_numpy(dtype=int)
    positions = np.arange(len(fold_array))
    splits = []
    groups = data.groups_development.reset_index(drop=True)
    for fold in expected:
        train_position = positions[fold_array != fold]
        valid_position = positions[fold_array == fold]
        overlap = set(groups.iloc[train_position]) & set(groups.iloc[valid_position])
        if overlap:
            raise AssertionError(
                f"{data.city} fold {fold}: 共享划分存在analysis_wave_id泄漏"
            )
        splits.append((train_position, valid_position))
    return splits


def load_lightgbm_recipe(city: str, num_threads: int) -> tuple[dict, int]:
    metadata_path = TRAINING_ROOT / "models" / "model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"未找到正式训练元数据：{metadata_path}。请先运行完整实验脚本。"
        )
    training_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    signature = training_metadata.get("signature", {})
    expected_groups = {
        key: list(value) for key, value in GROUP_COLUMNS.items()
    }
    if signature.get("group_columns") != expected_groups:
        raise RuntimeError(
            "现有正式模型不是按当前 date-courier-wave 分组训练的。"
            "请先重新运行完整实验脚本，再运行需要LightGBM配方的补充分析。"
        )
    saved_split = signature.get("shared_split", {})
    if saved_split.get("order_split_file") != str(ORDER_SPLIT_FILE):
        raise RuntimeError(
            "正式模型元数据引用的样本划分表与当前共享划分表不一致。"
        )
    path = TRAINING_ROOT / "models" / city / "best_params.json"
    if not path.exists():
        raise FileNotFoundError(
            f"未找到已锁定的模型参数：{path}。脚本停止。"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = dict(payload["best_params"])
    params.update({
        "seed": MODEL_SEED,
        "feature_fraction_seed": MODEL_SEED,
        "bagging_seed": MODEL_SEED,
        "data_random_seed": MODEL_SEED,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": num_threads,
        "verbosity": -1,
    })
    return params, int(payload["best_iteration"])


def fit_lightgbm(X: pd.DataFrame, y: pd.Series, city: str,
                 num_threads: int, features: Sequence[str] = FEATURES) -> lgb.Booster:
    params, rounds = load_lightgbm_recipe(city, num_threads)
    train_set = lgb.Dataset(
        X.loc[:, list(features)], label=y,
        feature_name=list(features), free_raw_data=False,
    )
    return lgb.train(
        params, train_set, num_boost_round=rounds,
        callbacks=[lgb.log_evaluation(period=0)],
    )


def tree_shap_importance(model: lgb.Booster, X: pd.DataFrame,
                         city: str, label: str,
                         replicate: int | None = None) -> pd.DataFrame:
    contributions = np.asarray(model.predict(X, pred_contrib=True))
    if contributions.shape[1] == X.shape[1] + 1:
        contributions = contributions[:, :-1]
    importance = np.mean(np.abs(contributions), axis=0)
    result = pd.DataFrame({
        "city": city,
        "analysis": label,
        "feature": X.columns,
        "mean_abs_shap": importance,
        "n_shap": len(X),
    })
    result["rank"] = result["mean_abs_shap"].rank(
        method="min", ascending=False,
    ).astype(int)
    if replicate is not None:
        result["replicate"] = replicate
    return result.sort_values("rank")


def fixed_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.copy()
    return frame.sample(n=n, random_state=seed).copy()


def save_run_metadata(args: argparse.Namespace, output_root: Path) -> None:
    serialized_arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload = {
        "created_at": pd.Timestamp.now().isoformat(),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lgb.__version__,
        "data_paths": {key: str(value) for key, value in DATA_PATHS.items()},
        "training_recipe_root": str(TRAINING_ROOT),
        "shared_order_split_file": str(ORDER_SPLIT_FILE),
        "shared_wave_split_file": str(WAVE_SPLIT_FILE),
        "split_metadata_file": str(SPLIT_METADATA_FILE),
        "features": list(FEATURES),
        "target": TARGET,
        "group_columns": {key: list(value) for key, value in GROUP_COLUMNS.items()},
        "seeds": {
            "development_test_split": SPLIT_SEED,
            "lightgbm": MODEL_SEED,
            "resampling": RESAMPLE_SEED,
        },
        "arguments": serialized_arguments,
        "figure_settings": {
            "width_mm": FIGURE_WIDTH_MM,
            "dpi": FIGURE_DPI,
            "font_family": FIGURE_FONT,
            "formats": ["png", "tiff", "pdf", "svg"],
        },
        "benchmark_models": {
            "linear": {
                "name": LINEAR_MODEL_NAME,
                "estimator": "ordinary least squares",
                "regularization": "none",
                "continuous_processing": "median imputation and standardization",
                "categorical_processing": "most-frequent imputation and one-hot encoding",
            },
            "gam": {
                "name": "GAM (penalized B-spline additive model)",
                "penalty_alpha_grid": list(GAM_ALPHA_GRID),
                "spline_knots": 6,
                "spline_degree": 3,
            },
        },
    }
    atomic_write_json(payload, output_root / "00_run_metadata.json")


# In[]: 1. OLS, additive-spline, and LightGBM benchmarks
# ============================================================================

def make_linear_pipeline() -> Pipeline:
    """Ordinary least squares without splines or interaction terms."""

    continuous = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(
            drop="if_binary", handle_unknown="ignore", sparse_output=True,
        )),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", continuous, list(CONTINUOUS_FEATURES)),
            ("categorical", categorical, list(BINARY_FEATURES)),
        ],
        sparse_threshold=1.0,
    )
    return Pipeline([
        ("terms", preprocessor),
        ("model", LinearRegression()),
    ])


def fit_linear_baseline(data: CityData, grouped_cv_folds: int,
                        smoke_test: bool) -> tuple[Pipeline, float]:
    """Evaluate OLS on the persisted folds, then refit on development data."""

    linear = make_linear_pipeline()
    if smoke_test:
        linear.fit(data.X_development, data.y_development)
        return linear, np.nan

    fixed_cv = stored_cv_splits(data, grouped_cv_folds)
    fold_scores = cross_val_score(
        linear,
        data.X_development,
        data.y_development,
        scoring="neg_root_mean_squared_error",
        cv=fixed_cv,
        n_jobs=1,
    )
    linear.fit(data.X_development, data.y_development)
    return linear, float(-np.mean(fold_scores))


def linear_setting_row(city: str, cv_rmse: float,
                       grouped_cv_folds: int, smoke_test: bool) -> dict:
    return {
        "city": city,
        "model": LINEAR_MODEL_NAME,
        "estimator": "ordinary least squares",
        "regularization": "none",
        "grouped_cv_folds": 0 if smoke_test else grouped_cv_folds,
        "grouped_cv_rmse": cv_rmse,
        "continuous_processing": "median imputation; standardization",
        "categorical_processing": (
            "most-frequent imputation; one-hot encoding; "
            "performed inside the linear pipeline"
        ),
        "interaction_terms": "none",
    }

def make_gam_pipeline() -> Pipeline:
    """Penalized B-spline additive regression without interaction terms."""

    continuous = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("spline", SplineTransformer(
            n_knots=6,
            degree=3,
            knots="quantile",
            extrapolation="constant",
            include_bias=False,
            sparse_output=True,
        )),
    ])
    binary = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(
            drop="if_binary", handle_unknown="ignore", sparse_output=True,
        )),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("smooth_terms", continuous, list(CONTINUOUS_FEATURES)),
            ("binary_terms", binary, list(BINARY_FEATURES)),
        ],
        sparse_threshold=1.0,
    )
    return Pipeline([
        ("terms", preprocessor),
        ("model", Ridge(solver="lsqr", max_iter=10_000)),
    ])


def run_benchmark(data_by_city: dict[str, CityData], output_root: Path,
                  num_threads: int, gam_cv_folds: int,
                  smoke_test: bool) -> None:
    task_root = output_root / "01_benchmark_models"
    metrics_rows = []
    settings_rows = []
    linear_settings_rows = []

    for city, data in data_by_city.items():
        print(f"\n[{city}] 基准比较：拟合 LightGBM")
        lightgbm_model = fit_lightgbm(
            data.X_development, data.y_development, city, num_threads,
        )
        lightgbm_prediction = lightgbm_model.predict(data.X_test)
        metrics_rows.append(metric_row(
            city, "benchmark", "LightGBM", "held-out test",
            data.y_test, lightgbm_prediction,
        ))
        model_dir = task_root / "models" / city
        model_dir.mkdir(parents=True, exist_ok=True)
        save_booster(lightgbm_model, model_dir / "lightgbm.txt")

        print(f"[{city}] 基准比较：拟合普通最小二乘线性回归")
        linear, linear_cv_rmse = fit_linear_baseline(
            data, gam_cv_folds, smoke_test,
        )
        linear_prediction = linear.predict(data.X_test)
        metrics_rows.append(metric_row(
            city, "benchmark", LINEAR_MODEL_NAME,
            "held-out test", data.y_test, linear_prediction,
        ))
        linear_settings_rows.append(linear_setting_row(
            city, linear_cv_rmse, gam_cv_folds, smoke_test,
        ))

        print(f"[{city}] 基准比较：拟合 GAM")
        gam = make_gam_pipeline()
        if smoke_test:
            gam.set_params(model__alpha=1.0)
            gam.fit(data.X_development, data.y_development)
            best_alpha = 1.0
            cv_rmse = np.nan
        else:
            fixed_cv = stored_cv_splits(data, gam_cv_folds)
            search = GridSearchCV(
                gam,
                param_grid={"model__alpha": list(GAM_ALPHA_GRID)},
                scoring="neg_root_mean_squared_error",
                cv=fixed_cv,
                n_jobs=1,
                refit=True,
                return_train_score=False,
            )
            search.fit(data.X_development, data.y_development)
            gam = search.best_estimator_
            best_alpha = float(search.best_params_["model__alpha"])
            cv_rmse = float(-search.best_score_)

        gam_prediction = gam.predict(data.X_test)
        metrics_rows.append(metric_row(
            city, "benchmark", "GAM (penalized B-spline additive model)",
            "held-out test", data.y_test, gam_prediction,
        ))
        settings_rows.append({
            "city": city,
            "model": "GAM (penalized B-spline additive model)",
            "spline_knots": 6,
            "spline_degree": 3,
            "penalty_alpha": best_alpha,
            "grouped_cv_folds": 0 if smoke_test else gam_cv_folds,
            "grouped_cv_rmse": cv_rmse,
            "missing_value_handling": (
                "continuous median imputation; binary most-frequent imputation; "
                "performed inside the GAM pipeline"
            ),
        })

    atomic_to_csv(pd.DataFrame(metrics_rows), task_root / "model_performance.csv")
    atomic_to_csv(pd.DataFrame(settings_rows), task_root / "gam_settings.csv")
    atomic_to_csv(
        pd.DataFrame(linear_settings_rows), task_root / "linear_settings.csv"
    )
    print(f"基准模型结果已保存：{task_root}")


# In[]: 2. Grouped resampling stability
# ============================================================================

def group_bootstrap_indices(groups: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Sample analysis-wave groups with replacement and retain whole groups."""

    group_codes, unique_groups = pd.factorize(groups.astype(str), sort=False)
    members = [np.flatnonzero(group_codes == code) for code in range(len(unique_groups))]
    sampled_codes = rng.integers(0, len(unique_groups), size=len(unique_groups))
    return np.concatenate([members[code] for code in sampled_codes])


def make_ale_edges(X_development: pd.DataFrame, features: Iterable[str],
                   bins: int) -> dict[str, np.ndarray]:
    edges = {}
    probabilities = np.linspace(0.0, 1.0, bins + 1)
    for feature in features:
        values = pd.to_numeric(X_development[feature], errors="coerce").dropna()
        feature_edges = np.unique(values.quantile(probabilities).to_numpy(dtype=float))
        if len(feature_edges) >= 2:
            edges[feature] = feature_edges
    return edges


def one_dimensional_ale(model: lgb.Booster, X: pd.DataFrame,
                        edges_by_feature: dict[str, np.ndarray],
                        city: str, replicate: int) -> pd.DataFrame:
    rows = []
    for feature, edges in edges_by_feature.items():
        valid = X[feature].notna()
        sample = X.loc[valid].copy()
        if sample.empty:
            continue
        values = sample[feature].to_numpy(dtype=float)
        bin_index = np.searchsorted(edges[1:-1], values, side="right")
        n_bins = len(edges) - 1
        local_effect = np.full(n_bins, np.nan)
        counts = np.zeros(n_bins, dtype=int)

        for current_bin in range(n_bins):
            in_bin = bin_index == current_bin
            counts[current_bin] = int(in_bin.sum())
            if not in_bin.any():
                continue
            lower = float(edges[current_bin])
            upper = float(edges[current_bin + 1])
            X_low = sample.loc[in_bin].copy()
            X_high = sample.loc[in_bin].copy()
            X_low.loc[:, feature] = lower
            X_high.loc[:, feature] = upper
            local_effect[current_bin] = float(np.mean(
                model.predict(X_high) - model.predict(X_low)
            ))

        # Interpolate bins that are empty in the fixed test sample.
        effect_series = pd.Series(local_effect).interpolate(
            limit_direction="both"
        ).fillna(0.0)
        effect_array = effect_series.to_numpy(dtype=float)
        uncentered = np.cumsum(effect_array) - 0.5 * effect_array
        if counts.sum() > 0:
            centered = uncentered - np.average(uncentered, weights=counts)
        else:
            centered = uncentered

        for current_bin in range(n_bins):
            rows.append({
                "city": city,
                "replicate": replicate,
                "feature": feature,
                "bin": current_bin + 1,
                "lower": float(edges[current_bin]),
                "upper": float(edges[current_bin + 1]),
                "midpoint": float((edges[current_bin] + edges[current_bin + 1]) / 2),
                "n_test_in_bin": int(counts[current_bin]),
                "ale": float(centered[current_bin]),
            })
    return pd.DataFrame(rows)


def selected_shap_interactions(model: lgb.Booster, X: pd.DataFrame,
                               city: str, replicate: int) -> pd.DataFrame:
    import shap

    values = shap.TreeExplainer(model).shap_interaction_values(X)
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values)
    feature_position = {feature: i for i, feature in enumerate(X.columns)}
    rows = []
    for first, second in INTERACTION_PAIRS:
        i, j = feature_position[first], feature_position[second]
        rows.append({
            "city": city,
            "replicate": replicate,
            "feature_1": first,
            "feature_2": second,
            "mean_abs_shap_interaction": float(np.mean(np.abs(values[:, i, j]))),
            "n_interaction": len(X),
        })
    result = pd.DataFrame(rows)
    result["rank"] = result["mean_abs_shap_interaction"].rank(
        method="min", ascending=False,
    ).astype(int)
    return result


def interval_summary(frame: pd.DataFrame, group_columns: Sequence[str],
                     value_columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(list(group_columns), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        for value_column in value_columns:
            values = pd.to_numeric(group[value_column], errors="coerce").dropna()
            row.update({
                f"{value_column}_mean": values.mean(),
                f"{value_column}_sd": values.std(ddof=1),
                f"{value_column}_median": values.median(),
                f"{value_column}_p2_5": values.quantile(0.025),
                f"{value_column}_p97_5": values.quantile(0.975),
                f"{value_column}_min": values.min(),
                f"{value_column}_max": values.max(),
            })
        row["n_replicates"] = int(group["replicate"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_resampling(city_root: Path) -> None:
    replicate_dirs = sorted(city_root.glob("replicate_*"))
    if not replicate_dirs:
        return
    file_map = {
        "performance": "performance.csv",
        "shap": "shap_importance.csv",
        "ale": "ale.csv",
        "interaction": "interactions.csv",
    }
    combined = {}
    for label, filename in file_map.items():
        parts = [pd.read_csv(folder / filename) for folder in replicate_dirs
                 if (folder / filename).exists()]
        if parts:
            combined[label] = pd.concat(parts, ignore_index=True)

    summary_root = city_root / "summary"
    if "performance" in combined:
        atomic_to_csv(interval_summary(
            combined["performance"], ["city", "model"], ["r2", "rmse", "mae"]
        ), summary_root / "performance_stability.csv")
    if "shap" in combined:
        shap_summary = interval_summary(
            combined["shap"], ["city", "feature"], ["mean_abs_shap", "rank"]
        )
        top5 = (combined["shap"].assign(selected=lambda x: x["rank"] <= 5)
                .groupby(["city", "feature"])["selected"].mean()
                .rename("top5_selection_frequency").reset_index())
        shap_summary = shap_summary.merge(top5, on=["city", "feature"], how="left")
        atomic_to_csv(shap_summary, summary_root / "shap_rank_stability.csv")
    if "interaction" in combined:
        interaction_summary = interval_summary(
            combined["interaction"],
            ["city", "feature_1", "feature_2"],
            ["mean_abs_shap_interaction", "rank"],
        )
        atomic_to_csv(
            interaction_summary,
            summary_root / "interaction_stability.csv",
        )
    if "ale" in combined:
        ale_summary = interval_summary(
            combined["ale"],
            ["city", "feature", "bin", "lower", "upper", "midpoint"],
            ["ale"],
        )
        atomic_to_csv(ale_summary, summary_root / "ale_stability.csv")
        features = list(dict.fromkeys(ale_summary["feature"].tolist()))
        n_columns = 3
        n_rows = int(np.ceil(len(features) / n_columns))
        figure, axes = plt.subplots(
            n_rows, n_columns,
            figsize=(FIGURE_WIDTH_IN, 2.25 * n_rows),
            squeeze=False,
        )
        for axis, feature in zip(axes.flat, features):
            subset = ale_summary.loc[
                ale_summary["feature"] == feature
            ].sort_values("midpoint")
            x = subset["midpoint"].to_numpy(dtype=float)
            median = subset["ale_median"].to_numpy(dtype=float)
            lower = subset["ale_p2_5"].to_numpy(dtype=float)
            upper = subset["ale_p97_5"].to_numpy(dtype=float)
            axis.plot(x, median, color="#245B78", linewidth=1.5)
            axis.fill_between(x, lower, upper, color="#7FB3C8", alpha=0.35)
            axis.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
            axis.set_title(FEATURE_LABELS.get(feature, feature))
            axis.set_xlabel("Feature value")
            axis.set_ylabel("ALE")
        for axis in axes.flat[len(features):]:
            axis.axis("off")
        figure.tight_layout()
        save_publication_figure(figure, summary_root / "ale_stability_bands")
        plt.close(figure)


def run_resampling(data_by_city: dict[str, CityData], output_root: Path,
                   num_threads: int, n_resamples: int, shap_sample: int,
                   ale_sample: int, interaction_sample: int,
                   ale_bins: int) -> None:
    task_root = output_root / "02_grouped_resampling_stability"

    for city, data in data_by_city.items():
        city_root = task_root / city
        X_development = data.X_development.reset_index(drop=True)
        y_development = data.y_development.reset_index(drop=True)
        groups_development = data.groups_development.reset_index(drop=True)
        X_test = data.X_test.reset_index(drop=True)
        y_test = data.y_test.reset_index(drop=True)

        shap_X = fixed_sample(X_test, shap_sample, SPLIT_SEED + 101)
        ale_X = fixed_sample(X_test, ale_sample, SPLIT_SEED + 102)
        interaction_X = fixed_sample(X_test, interaction_sample, SPLIT_SEED + 103)
        ale_edges = make_ale_edges(X_development, ALE_FEATURES, ale_bins)

        for replicate in range(1, n_resamples + 1):
            replicate_root = city_root / f"replicate_{replicate:03d}"
            expected = [
                replicate_root / "performance.csv",
                replicate_root / "shap_importance.csv",
                replicate_root / "ale.csv",
                replicate_root / "interactions.csv",
            ]
            if all(path.exists() for path in expected):
                print(f"[{city}] 重抽样 {replicate}/{n_resamples} 已存在，跳过")
                continue

            print(f"[{city}] 重抽样 {replicate}/{n_resamples}")
            rng = np.random.default_rng(RESAMPLE_SEED + replicate)
            bootstrap_index = group_bootstrap_indices(groups_development, rng)
            model = fit_lightgbm(
                X_development.iloc[bootstrap_index],
                y_development.iloc[bootstrap_index],
                city,
                num_threads,
            )

            prediction = model.predict(X_test)
            performance = pd.DataFrame([metric_row(
                city, "group bootstrap refit", "LightGBM", "fixed held-out test",
                y_test, prediction,
            )])
            performance["replicate"] = replicate
            performance["n_sampled_development_rows"] = len(bootstrap_index)
            performance["n_unique_development_groups_sampled"] = int(
                groups_development.iloc[bootstrap_index].nunique()
            )

            shap_table = tree_shap_importance(
                model, shap_X, city, "group bootstrap refit", replicate,
            )
            ale_table = one_dimensional_ale(
                model, ale_X, ale_edges, city, replicate,
            )
            interaction_table = selected_shap_interactions(
                model, interaction_X, city, replicate,
            )

            atomic_to_csv(performance, replicate_root / "performance.csv")
            atomic_to_csv(shap_table, replicate_root / "shap_importance.csv")
            atomic_to_csv(ale_table, replicate_root / "ale.csv")
            atomic_to_csv(interaction_table, replicate_root / "interactions.csv")

        summarize_resampling(city_root)
    print(f"重抽样稳定性结果已保存：{task_root}")


# In[]: 3. Robustness analyses
# ============================================================================

def exact_top_n_mask(values: pd.Series, source_row_number: pd.Series,
                     n: int) -> tuple[pd.Series, pd.DataFrame]:
    """Select exactly n upper-tail rows with stable row-order tie breaking."""

    audit = pd.DataFrame({
        "value": pd.to_numeric(values, errors="coerce"),
        "source_row_number": source_row_number,
    }, index=values.index)
    ordered = audit.sort_values(
        ["value", "source_row_number"], ascending=[False, True],
        na_position="last", kind="mergesort",
    )
    removed_index = ordered.loc[ordered["value"].notna()].head(n).index
    keep = pd.Series(True, index=values.index)
    keep.loc[removed_index] = False
    return keep, ordered.loc[removed_index].copy()


def run_robustness(data_by_city: dict[str, CityData], output_root: Path,
                   num_threads: int, shap_sample: int,
                   tail200_variable: str | None) -> None:
    task_root = output_root / "03_robustness"
    performance_rows = []
    shap_parts = []
    audit_rows = []

    for city, data in data_by_city.items():
        X_development = data.X_development
        y_development = data.y_development
        X_test = data.X_test
        y_test = data.y_test

        print(f"\n[{city}] 稳健性：完整模型")
        base_model = fit_lightgbm(
            X_development, y_development, city, num_threads,
        )
        performance_rows.append(metric_row(
            city, "robustness", "Full specification", "full held-out test",
            y_test, base_model.predict(X_test),
        ))

        common_shap_index = fixed_sample(X_test, shap_sample, SPLIT_SEED + 201).index
        shap_parts.append(tree_shap_importance(
            base_model, X_test.loc[common_shap_index], city, "Full specification",
        ))

        print(f"[{city}] 稳健性：删除 wave_duration_time 特征")
        no_wave_features = tuple(
            feature for feature in FEATURES if feature != "wave_duration_time"
        )
        no_wave_model = fit_lightgbm(
            X_development.loc[:, no_wave_features], y_development,
            city, num_threads, no_wave_features,
        )
        performance_rows.append(metric_row(
            city, "robustness", "Without wave duration", "full held-out test",
            y_test, no_wave_model.predict(X_test.loc[:, no_wave_features]),
        ))
        shap_parts.append(tree_shap_importance(
            no_wave_model,
            X_test.loc[common_shap_index, no_wave_features],
            city,
            "Without wave duration",
        ))

        # Define both upper-tail thresholds from the development sample.
        thresholds = {
            feature: float(X_development[feature].quantile(0.99))
            for feature in EXTREME_FEATURE_UNITS
        }
        development_keep = pd.Series(True, index=X_development.index)
        test_keep = pd.Series(True, index=X_test.index)
        for feature, threshold in thresholds.items():
            development_keep &= X_development[feature].le(threshold)
            test_keep &= X_test[feature].le(threshold)
        audit_rows.append({
            "city": city,
            "analysis": "Original top-1% joint upper-tail trimming",
            "variable": "loadorder_number OR wave_duration_time",
            "unit": "orders per wave; minutes",
            "rule": "thresholds are development-set 99th percentiles; remove if either variable exceeds its threshold",
            "loadorder_number_threshold": thresholds["loadorder_number"],
            "wave_duration_time_threshold": thresholds["wave_duration_time"],
            "n_development_before": len(X_development),
            "n_development_removed": int((~development_keep).sum()),
            "n_development_retained": int(development_keep.sum()),
            "n_test_before": len(X_test),
            "n_test_removed": int((~test_keep).sum()),
            "n_test_retained": int(test_keep.sum()),
        })

        print(f"[{city}] 稳健性：development-q99 联合删尾")
        trimmed_model = fit_lightgbm(
            X_development.loc[development_keep],
            y_development.loc[development_keep],
            city,
            num_threads,
        )
        performance_rows.append(metric_row(
            city, "robustness", "Top-1%-trimmed refit",
            "test retained by development q99 thresholds",
            y_test.loc[test_keep], trimmed_model.predict(X_test.loc[test_keep]),
        ))
        performance_rows.append(metric_row(
            city, "robustness", "Full specification",
            "test retained by development q99 thresholds",
            y_test.loc[test_keep], base_model.predict(X_test.loc[test_keep]),
        ))
        trim_shap_index = common_shap_index.intersection(X_test.index[test_keep])
        shap_parts.append(tree_shap_importance(
            trimmed_model, X_test.loc[trim_shap_index], city,
            "Top-1%-trimmed refit",
        ))

        if tail200_variable is not None:
            print(f"[{city}] 稳健性：精确删除 {tail200_variable} 最大的 200 条")
            full_keep, removed = exact_top_n_mask(
                data.X[tail200_variable], data.frame["source_row_number"], 200,
            )
            development_keep_200 = full_keep.iloc[data.development_index]
            test_keep_200 = full_keep.iloc[data.test_index]
            development_keep_200.index = X_development.index
            test_keep_200.index = X_test.index

            removed_values = removed["value"].dropna()
            audit_rows.append({
                "city": city,
                "analysis": "Exact upper-tail 200 sensitivity",
                "variable": tail200_variable,
                "unit": EXTREME_FEATURE_UNITS[tail200_variable],
                "rule": (
                    "remove exactly 200 citywide observations with the largest values; "
                    "ties are resolved by ascending original source-row number"
                ),
                "removed_value_min": removed_values.min(),
                "removed_value_max": removed_values.max(),
                "n_development_before": len(X_development),
                "n_development_removed": int((~development_keep_200).sum()),
                "n_development_retained": int(development_keep_200.sum()),
                "n_test_before": len(X_test),
                "n_test_removed": int((~test_keep_200).sum()),
                "n_test_retained": int(test_keep_200.sum()),
            })
            tail_model = fit_lightgbm(
                X_development.loc[development_keep_200],
                y_development.loc[development_keep_200],
                city,
                num_threads,
            )
            performance_rows.append(metric_row(
                city, "robustness", f"Exact top-200 {tail200_variable} refit",
                f"test excluding exact citywide top 200 {tail200_variable}",
                y_test.loc[test_keep_200],
                tail_model.predict(X_test.loc[test_keep_200]),
            ))
            tail_shap_index = common_shap_index.intersection(
                X_test.index[test_keep_200]
            )
            shap_parts.append(tree_shap_importance(
                tail_model, X_test.loc[tail_shap_index], city,
                f"Exact top-200 {tail200_variable} refit",
            ))

    performance = pd.DataFrame(performance_rows)
    shap_table = pd.concat(shap_parts, ignore_index=True)
    rank_wide = shap_table.pivot_table(
        index=["city", "feature"], columns="analysis", values="rank",
    ).reset_index()
    atomic_to_csv(performance, task_root / "robustness_performance.csv")
    atomic_to_csv(pd.DataFrame(audit_rows), task_root / "filter_rule_audit.csv")
    atomic_to_csv(shap_table, task_root / "robustness_shap_rankings.csv")
    atomic_to_csv(rank_wide, task_root / "robustness_shap_rank_comparison.csv")

    clarification = {
        "verified_old_code_rule": (
            "For loadorder_number and wave_duration_time, compute each city's "
            "development-set 99th percentile and remove an observation when either "
            "variable exceeds its threshold. This is not a fixed-200 rule."
        ),
        "tail_200_status": (
            "not run because the variable was not specified"
            if tail200_variable is None
            else f"run for {tail200_variable} ({EXTREME_FEATURE_UNITS[tail200_variable]})"
        ),
    }
    atomic_write_json(clarification, task_root / "extreme_value_rule_clarification.json")
    print(f"稳健性结果已保存：{task_root}")


# In[]: 4. Outcome description
# ============================================================================

def run_outcome_description(data_by_city: dict[str, CityData],
                            output_root: Path) -> None:
    task_root = output_root / "04_outcome_description"
    rows = []
    duration_rows = []
    plot_payload = {}
    quantiles = {
        "p01": 0.01, "p05": 0.05, "p25": 0.25, "p50": 0.50,
        "p75": 0.75, "p95": 0.95, "p99": 0.99,
    }
    for city, data in data_by_city.items():
        values = pd.to_numeric(data.y, errors="coerce").dropna()
        row = {
            "city": city,
            "outcome": "relative delivery delay",
            "unit": "dimensionless ratio",
            "n": len(values),
            "mean": values.mean(),
            "sd": values.std(ddof=1),
            "minimum": values.min(),
            "maximum": values.max(),
            "on_time_n": int(values.le(0).sum()),
            "on_time_rate": values.le(0).mean(),
            "late_n": int(values.gt(0).sum()),
            "late_rate": values.gt(0).mean(),
            "on_time_definition": "relative delivery delay <= 0",
        }
        row.update({label: values.quantile(probability)
                    for label, probability in quantiles.items()})
        rows.append(row)
        plot_payload[city] = values

        duration_definitions = (
            ("actual fulfillment duration", "finish_time"),
            ("promised fulfillment duration", "estimate_finish_time"),
        )
        for measure, source_column in duration_definitions:
            all_values = pd.to_numeric(data.frame[source_column], errors="coerce")
            observed = all_values.dropna()
            duration_row = {
                "city": city,
                "measure": measure,
                "source_column": source_column,
                "unit": "minutes",
                "n_total_analysis_sample": len(all_values),
                "n_nonmissing": len(observed),
                "n_missing": int(all_values.isna().sum()),
                "missing_rate": float(all_values.isna().mean()),
                "mean": observed.mean(),
                "sd": observed.std(ddof=1),
                "minimum": observed.min(),
            }
            duration_row.update({
                label: observed.quantile(probability)
                for label, probability in quantiles.items()
            })
            duration_row["maximum"] = observed.max()
            duration_rows.append(duration_row)

    statistics = pd.DataFrame(rows)
    atomic_to_csv(statistics, task_root / "relative_delay_descriptive_statistics.csv")
    duration_statistics = pd.DataFrame(duration_rows)
    atomic_to_csv(
        duration_statistics,
        task_root / "fulfillment_duration_descriptive_statistics.csv",
    )

    figure, axes = plt.subplots(
        1, len(plot_payload), figsize=(FIGURE_WIDTH_IN, 3.0), sharey=False
    )
    if len(plot_payload) == 1:
        axes = [axes]
    for axis, (city, values) in zip(axes, plot_payload.items()):
        lower, upper = values.quantile([0.005, 0.995])
        visible = values.loc[values.between(lower, upper, inclusive="both")]
        axis.hist(
            visible, bins=80, density=True,
            color="#3B82A0", alpha=0.82, edgecolor="white", linewidth=0.25,
        )
        axis.axvline(0, color="#B33A3A", linestyle="--", linewidth=1.2,
                     label="On-time threshold")
        axis.set_title(f"{city} (n = {len(values):,})")
        axis.set_xlabel("Relative delivery delay")
        axis.set_ylabel("Density")
        axis.legend(loc="upper right", frameon=False)
        axis.text(
            0.02, 0.97,
            "Shown: P0.5–P99.5",
            transform=axis.transAxes, ha="left", va="top", fontsize=6.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
    figure.tight_layout()
    save_publication_figure(figure, task_root / "relative_delay_distribution")
    plt.close(figure)
    print(f"因变量描述统计和分布图已保存：{task_root}")


# In[]: 5. Command-line entry point
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行论文补充分析；在 Jupyter/IPython 中会忽略内核附加参数。"
    )
    parser.add_argument(
        "--task",
        choices=(
            "all", "benchmark", "resampling", "robustness", "outcome"
        ),
        default="all",
    )
    parser.add_argument("--cities", nargs="+", choices=("BJ", "BD"),
                        default=["BJ", "BD"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--gam-cv-folds", type=int, default=5)
    parser.add_argument("--n-resamples", type=int, default=50)
    parser.add_argument("--shap-sample", type=int, default=5_000)
    parser.add_argument("--ale-sample", type=int, default=5_000)
    parser.add_argument("--interaction-sample", type=int, default=500)
    parser.add_argument("--ale-bins", type=int, default=20)
    parser.add_argument(
        "--tail200-variable",
        choices=("loadorder_number", "wave_duration_time"),
        default=None,
        help=(
            "只有明确选择后才运行固定尾部 200 分析；单位分别为 orders per wave 和 minutes。"
        ),
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="仅按前 500 个组快速检查流程；结果不能用于论文。",
    )
    if "ipykernel" in sys.modules:
        args, unknown = parser.parse_known_args()
        if unknown:
            print(f"忽略 IPython/Jupyter 附加参数：{unknown}")
    else:
        args = parser.parse_args()
    if args.n_resamples < 1:
        parser.error("--n-resamples 必须至少为 1")
    if not 2 <= args.ale_bins <= 100:
        parser.error("--ale-bins 必须在 2 到 100 之间")
    return args


def main() -> None:
    args = parse_arguments()
    output_root = args.output_root
    if args.smoke_test:
        output_root = output_root / "smoke_test"
        args.n_resamples = min(args.n_resamples, 2)
        args.shap_sample = min(args.shap_sample, 500)
        args.ale_sample = min(args.ale_sample, 500)
        args.interaction_sample = min(args.interaction_sample, 100)

    output_root.mkdir(parents=True, exist_ok=True)
    save_run_metadata(args, output_root)
    data_by_city = {
        city: load_city_data(city, smoke_test=args.smoke_test)
        for city in args.cities
    }
    save_split_audit(data_by_city, output_root)

    if args.task in ("all", "benchmark"):
        run_benchmark(
            data_by_city, output_root, args.num_threads,
            args.gam_cv_folds, args.smoke_test,
        )
    if args.task in ("all", "resampling"):
        run_resampling(
            data_by_city, output_root, args.num_threads, args.n_resamples,
            args.shap_sample, args.ale_sample, args.interaction_sample,
            args.ale_bins,
        )
    if args.task in ("all", "robustness"):
        run_robustness(
            data_by_city, output_root, args.num_threads,
            args.shap_sample, args.tail200_variable,
        )
    if args.task in ("all", "outcome"):
        run_outcome_description(data_by_city, output_root)

    print(f"\n全部指定任务完成：{output_root}")


if __name__ == "__main__":
    main()
