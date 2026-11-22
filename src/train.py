# -*- coding: utf-8 -*-
"""Huấn luyện, đánh giá và tổng hợp kết quả các mô hình ML (TV3).

Chạy từ thư mục gốc repo:
    python src/train.py                      # E0 mặc định
    python src/train.py --experiments e0 e1 e2 s1 s2 s3
    python src/train.py --experiments all

Quy tắc không rò rỉ:
- load_xy() là nơi DUY NHẤT đọc dữ liệu.
- make_folds() tạo fold MỘT LẦN; mọi thí nghiệm (trừ E2) dùng lại.
- Mỗi fold mỗi mô hình dùng bản clone() mới của pipeline.
- Không fit, fit_transform ngoài pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)

# Thêm src/ vào sys.path khi chạy từ thư mục gốc repo
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import (
    EXPERIMENT_IDS,
    EXTRA_COLUMNS_S3,
    FEATURE_COLUMNS,
    INNER_SPLITS,
    N_REPEATS,
    N_SPLITS,
    POSITIVE_LABEL,
    REPO_ROOT,
    RESULTS_DIR,
    SEED,
    TABLES_DIR,
    TARGET_COLUMN,
)
from evaluation import compute_fold_metrics, get_scores, summarize, paired_differences
from pipelines import PARAM_GRIDS, get_models
from preprocessing import (
    RAW_PATH,
    clean_raw_data,
    get_feature_target_split,
)


# ---------------------------------------------------------------------------
# 1. Đọc dữ liệu
# ---------------------------------------------------------------------------

def load_xy(
    extra_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Đọc và làm sạch dữ liệu; tách X, y.

    Args:
        extra_columns: Cột bổ sung thêm vào X (chỉ dùng S3). Assert cột tồn tại.

    Returns:
        Tuple (X, y) — X có 6 cột chính (+ extra nếu có), y nhị phân (0/1).
    """
    raw = pd.read_csv(RAW_PATH, encoding="utf-8")
    cleaned = clean_raw_data(raw)
    X, y = get_feature_target_split(cleaned)

    if extra_columns:
        for col in extra_columns:
            assert col in cleaned.columns, (
                f"Cột '{col}' không tồn tại trong dữ liệu. "
                f"Các cột hiện có: {list(cleaned.columns)}"
            )
            X[col] = cleaned[col].values

    print(f"[load_xy] X.shape = {X.shape}, y.value_counts() =\n{y.value_counts().to_string()}")
    return X, y


# ---------------------------------------------------------------------------
# 2. Tạo fold
# ---------------------------------------------------------------------------

def make_folds(
    X: pd.DataFrame,
    y: pd.Series,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Tạo stratified k-fold một lần; lưu fold_assignments.csv.

    Args:
        X: Dữ liệu đặc trưng.
        y: Nhãn.

    Returns:
        Danh sách (train_indices, test_indices).
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))

    # Ghi fold_assignments.csv
    rows = []
    for fold_idx, (_, test_idx) in enumerate(folds):
        for ri in test_idx:
            rows.append({
                "row_index": X.index[ri],
                "fold": fold_idx,
                "y": int(y.iloc[ri]),
            })
    fold_df = pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)
    out_path = TABLES_DIR / "fold_assignments.csv"
    fold_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[make_folds] Đã lưu {out_path}")

    # In kích thước mỗi fold
    for i, (_, test_idx) in enumerate(folds):
        n_pos = int(y.iloc[test_idx].sum())
        print(f"  Fold {i}: test_size={len(test_idx)}, n_yes={n_pos}")

    return folds


# ---------------------------------------------------------------------------
# 3. Chạy CV experiment
# ---------------------------------------------------------------------------

def run_cv_experiment(
    experiment_id: str,
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple],
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chạy cross-validation cho nhiều mô hình trên các fold cho trước.

    Với mỗi (mô hình, fold): clone pipeline → fit train → predict test → tính metric.

    Args:
        experiment_id: ID thí nghiệm (ví dụ "E0_main").
        models: Dict tên -> Pipeline chưa fit.
        X: Dữ liệu đặc trưng (toàn bộ).
        y: Nhãn (toàn bộ).
        folds: Danh sách (train_idx, test_idx) từ make_folds().
        save: Có lưu kết quả vào CSV hay không.

    Returns:
        (metrics_per_fold_df, oof_predictions_df).
    """
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    metric_rows = []
    oof_rows = []

    for model_name, pipeline in models.items():
        print(f"  [{experiment_id}] Mô hình: {model_name}")
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            pipe = clone(pipeline)
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_score = get_scores(pipe, X_test)

            metrics = compute_fold_metrics(y_test.values, y_pred, y_score)
            metrics.update({
                "experiment": experiment_id,
                "model": model_name,
                "fold": fold_idx,
            })
            metric_rows.append(metrics)

            # OOF predictions
            for i, (ri, yt, yp, ys) in enumerate(
                zip(test_idx, y_test.values, y_pred, y_score)
            ):
                oof_rows.append({
                    "experiment": experiment_id,
                    "model": model_name,
                    "row_index": int(ri),
                    "fold": fold_idx,
                    "y_true": int(yt),
                    "y_score": float(ys),
                    "y_pred": int(yp),
                })

    metrics_df = pd.DataFrame(metric_rows)
    oof_df = pd.DataFrame(oof_rows)

    # Sắp xếp cột
    col_order = ["experiment", "model", "fold", "n_test", "n_pos",
                 "tp", "fp", "fn", "tn",
                 "accuracy", "precision_yes", "recall_yes", "f1_yes",
                 "roc_auc", "pr_auc"]
    metrics_df = metrics_df[[c for c in col_order if c in metrics_df.columns]]

    if save:
        _append_or_create(metrics_df, TABLES_DIR / "metrics_per_fold.csv")
        _append_or_create(oof_df, TABLES_DIR / "oof_predictions.csv")
        print(f"  → Đã lưu metrics và OOF predictions cho {experiment_id}")

    return metrics_df, oof_df


def _append_or_create(df: pd.DataFrame, path: Path) -> None:
    """Thêm vào CSV nếu đã có, tạo mới nếu chưa có."""
    if path.exists():
        existing = pd.read_csv(path, encoding="utf-8")
        # Xóa các dòng cùng experiment nếu có (để chạy lại không trùng)
        if "experiment" in existing.columns and "experiment" in df.columns:
            exps_to_remove = df["experiment"].unique()
            existing = existing[~existing["experiment"].isin(exps_to_remove)]
        combined = pd.concat([existing, df], ignore_index=True)
        combined.to_csv(path, index=False, encoding="utf-8")
    else:
        df.to_csv(path, index=False, encoding="utf-8")


# ---------------------------------------------------------------------------
# 4. Nested CV (E1)
# ---------------------------------------------------------------------------

def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nested CV: GridSearchCV trên train fold, đánh giá trên test fold.

    Args:
        X: Dữ liệu đặc trưng.
        y: Nhãn.
        folds: Outer folds (giống E0).

    Returns:
        (metrics_df, best_params_df).
    """
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    base_models = get_models()  # cấu hình E0 cơ bản
    scoring = make_scorer(f1_score, pos_label=POSITIVE_LABEL, zero_division=0)
    inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)

    metric_rows = []
    oof_rows = []
    param_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for model_name in ["LogisticRegression", "DecisionTree", "RandomForest", "SVM"]:
            pipeline = clone(base_models[model_name])
            param_grid = PARAM_GRIDS[model_name]

            gs = GridSearchCV(
                pipeline,
                param_grid,
                cv=inner_cv,
                scoring=scoring,
                refit=True,
                n_jobs=1,
            )
            gs.fit(X_train, y_train)

            y_pred = gs.predict(X_test)
            y_score = get_scores(gs.best_estimator_, X_test)

            metrics = compute_fold_metrics(y_test.values, y_pred, y_score)
            metrics.update({
                "experiment": "E1_nested",
                "model": model_name,
                "fold": fold_idx,
            })
            metric_rows.append(metrics)

            for ri, yt, yp, ys in zip(test_idx, y_test.values, y_pred, y_score):
                oof_rows.append({
                    "experiment": "E1_nested",
                    "model": model_name,
                    "row_index": int(ri),
                    "fold": fold_idx,
                    "y_true": int(yt),
                    "y_score": float(ys),
                    "y_pred": int(yp),
                })

            param_rows.append({
                "model": model_name,
                "outer_fold": fold_idx,
                "best_params": json.dumps(gs.best_params_),
                "inner_best_score": gs.best_score_,
            })

        # Dummy E1 dùng lại cấu hình E0
        # (ghi chú: không có lưới tuning cho Dummy)
        dummy_pipe = clone(base_models["Dummy"])
        dummy_pipe.fit(X_train, y_train)
        y_pred_d = dummy_pipe.predict(X_test)
        y_score_d = get_scores(dummy_pipe, X_test)
        metrics_d = compute_fold_metrics(y_test.values, y_pred_d, y_score_d)
        metrics_d.update({"experiment": "E1_nested", "model": "Dummy", "fold": fold_idx})
        metric_rows.append(metrics_d)
        for ri, yt, yp, ys in zip(test_idx, y_test.values, y_pred_d, y_score_d):
            oof_rows.append({
                "experiment": "E1_nested",
                "model": "Dummy",
                "row_index": int(ri),
                "fold": fold_idx,
                "y_true": int(yt),
                "y_score": float(ys),
                "y_pred": int(yp),
            })

        print(f"  [E1] Outer fold {fold_idx} xong")

    metrics_df = pd.DataFrame(metric_rows)
    oof_df = pd.DataFrame(oof_rows)
    params_df = pd.DataFrame(param_rows)

    col_order = ["experiment", "model", "fold", "n_test", "n_pos",
                 "tp", "fp", "fn", "tn",
                 "accuracy", "precision_yes", "recall_yes", "f1_yes",
                 "roc_auc", "pr_auc"]
    metrics_df = metrics_df[[c for c in col_order if c in metrics_df.columns]]

    _append_or_create(metrics_df, TABLES_DIR / "metrics_per_fold.csv")
    _append_or_create(oof_df, TABLES_DIR / "oof_predictions.csv")
    params_df.to_csv(TABLES_DIR / "best_params.csv", index=False, encoding="utf-8")
    print(f"  → Đã lưu E1 metrics, OOF và best_params.csv")

    return metrics_df, params_df


# ---------------------------------------------------------------------------
# 5. Repeated CV (E2)
# ---------------------------------------------------------------------------

def run_repeated_cv(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Repeated Stratified K-Fold CV (5×10 = 50 fold) với cấu hình E0.

    Args:
        models: Dict tên -> Pipeline (cấu hình E0).
        X: Dữ liệu đặc trưng.
        y: Nhãn.

    Returns:
        metrics_repeated DataFrame.
    """
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED
    )
    rows = []

    for model_name, pipeline in models.items():
        print(f"  [E2] Mô hình: {model_name}")
        for rep_fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
            repeat = rep_fold_idx // N_SPLITS
            fold = rep_fold_idx % N_SPLITS

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            pipe = clone(pipeline)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_score = get_scores(pipe, X_test)

            metrics = compute_fold_metrics(y_test.values, y_pred, y_score)
            metrics.update({
                "model": model_name,
                "repeat": repeat,
                "fold": fold,
            })
            rows.append(metrics)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(
        TABLES_DIR / "metrics_repeated.csv", index=False, encoding="utf-8"
    )

    # Summary E2
    summary_rows = []
    for model, group in metrics_df.groupby("model"):
        row = {"model": model}
        from config import METRIC_COLUMNS
        for col in METRIC_COLUMNS:
            vals = group[col].dropna()
            row[f"{col}_mean"] = vals.mean()
            row[f"{col}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
            # std của mean theo lần lặp
            repeat_means = group.groupby("repeat")[col].mean()
            row[f"{col}_std_of_repeat_mean"] = (
                repeat_means.std(ddof=1) if len(repeat_means) > 1 else 0.0
            )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        TABLES_DIR / "metrics_repeated_summary.csv", index=False, encoding="utf-8"
    )
    print(f"  → Đã lưu metrics_repeated.csv ({len(metrics_df)} dòng) và summary")

    return metrics_df


# ---------------------------------------------------------------------------
# 6. Ghi run_config.json
# ---------------------------------------------------------------------------

def write_run_config(experiments_run: list[str]) -> None:
    """Ghi thông tin cấu hình chạy vào run_config.json.

    Args:
        experiments_run: Danh sách experiment đã chạy.
    """
    import platform
    import subprocess

    import numpy
    import pandas
    import sklearn
    import matplotlib as mpl

    # Lấy commit hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=REPO_ROOT
        )
        commit_hash = result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        commit_hash = None

    # Tải lại dữ liệu để lấy kích thước
    raw = pd.read_csv(RAW_PATH, encoding="utf-8")
    cleaned = clean_raw_data(raw)
    X, y = get_feature_target_split(cleaned)

    from pipelines import get_models as _get_models
    base_models = _get_models()
    model_params = {}
    for name, pipe in base_models.items():
        clf = pipe.named_steps["clf"]
        model_params[name] = clf.get_params()

    config = {
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python_version": platform.python_version(),
        "package_versions": {
            "pandas": pandas.__version__,
            "numpy": numpy.__version__,
            "scikit-learn": sklearn.__version__,
            "matplotlib": mpl.__version__,
        },
        "commit_hash": commit_hash,
        "SEED": SEED,
        "N_SPLITS": N_SPLITS,
        "INNER_SPLITS": INNER_SPLITS,
        "N_REPEATS": N_REPEATS,
        "FEATURE_COLUMNS": FEATURE_COLUMNS,
        "TARGET_COLUMN": TARGET_COLUMN,
        "EXCLUDED_COLUMNS": [
            "Timestamp",
            "Do you have Anxiety?",
            "Do you have Panic attack?",
            "Did you seek any specialist for a treatment?",
        ],
        "data_shape": {"n_rows": int(X.shape[0]), "n_features": int(X.shape[1])},
        "label_distribution": y.value_counts().to_dict(),
        "model_params_E0": model_params,
        "PARAM_GRIDS_E1": PARAM_GRIDS,
        "experiments_run": experiments_run,
        "metric_definitions": {
            "positive_label": POSITIVE_LABEL,
            "zero_division": 0,
            "ddof": 1,
            "roc_auc": "from decision scores (predict_proba or decision_function)",
            "pr_auc": "average_precision_score",
        },
    }

    out_path = TABLES_DIR / "run_config.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=str)
    print(f"[write_run_config] Đã lưu {out_path}")


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Điểm vào CLI: python src/train.py --experiments all"""
    parser = argparse.ArgumentParser(
        description="Chạy thí nghiệm ML cho bài toán dự đoán trầm cảm sinh viên."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["e0"],
        help=(
            "Danh sách thí nghiệm cần chạy: e0 e1 e2 s1 s2 s3 hoặc all. "
            "Mặc định: e0."
        ),
    )
    args = parser.parse_args()

    exp_map = {
        "e0": ["E0_main"],
        "e1": ["E1_nested"],
        "e2": ["E2_repeated"],
        "s1": ["S1a_drop_course", "S1b_course_min_freq5"],
        "s2": ["S2_class_weight_balanced"],
        "s3": ["S3_plus_anxiety_panic"],
        "all": EXPERIMENT_IDS,
    }

    # Mở rộng danh sách
    exps_to_run: list[str] = []
    for key in args.experiments:
        key_lower = key.lower()
        if key_lower in exp_map:
            exps_to_run.extend(exp_map[key_lower])
        else:
            print(f"[CẢNH BÁO] Thí nghiệm '{key}' không nhận ra. Bỏ qua.")
    exps_to_run = list(dict.fromkeys(exps_to_run))  # loại trùng, giữ thứ tự

    print(f"\n=== Bắt đầu chạy thí nghiệm: {exps_to_run} ===\n")

    t_total = time.time()

    # Xóa CSV kết quả cũ nếu chạy all, để đảm bảo tái lập
    if exps_to_run == EXPERIMENT_IDS:
        for f in [
            TABLES_DIR / "metrics_per_fold.csv",
            TABLES_DIR / "oof_predictions.csv",
        ]:
            if f.exists():
                f.unlink()
                print(f"[main] Đã xóa {f} để tái lập")

    # --- Tải dữ liệu và tạo fold (dùng chung) ---
    X, y = load_xy()
    folds = make_folds(X, y)

    all_metrics = []
    experiments_done = []

    # E0: Kết quả chính
    if "E0_main" in exps_to_run:
        print("\n--- E0: Kết quả chính ---")
        t0 = time.time()
        models_e0 = get_models()
        m_df, _ = run_cv_experiment("E0_main", models_e0, X, y, folds)
        all_metrics.append(m_df)
        experiments_done.append("E0_main")
        print(f"  E0 xong trong {time.time() - t0:.1f}s")

    # S1a: Bỏ cột Course
    if "S1a_drop_course" in exps_to_run:
        print("\n--- S1a: Bỏ cột Course ---")
        t0 = time.time()
        models_s1a = get_models(course_mode="drop")
        m_df, _ = run_cv_experiment("S1a_drop_course", models_s1a, X, y, folds)
        all_metrics.append(m_df)
        experiments_done.append("S1a_drop_course")
        print(f"  S1a xong trong {time.time() - t0:.1f}s")

    # S1b: Course min_frequency
    if "S1b_course_min_freq5" in exps_to_run:
        print("\n--- S1b: Course min_frequency=5 ---")
        t0 = time.time()
        models_s1b = get_models(course_mode="min_freq")
        m_df, _ = run_cv_experiment("S1b_course_min_freq5", models_s1b, X, y, folds)
        all_metrics.append(m_df)
        experiments_done.append("S1b_course_min_freq5")
        print(f"  S1b xong trong {time.time() - t0:.1f}s")

    # S2: class_weight balanced
    if "S2_class_weight_balanced" in exps_to_run:
        print("\n--- S2: class_weight='balanced' ---")
        t0 = time.time()
        models_s2 = get_models(class_weight="balanced")
        m_df, _ = run_cv_experiment("S2_class_weight_balanced", models_s2, X, y, folds)
        all_metrics.append(m_df)
        experiments_done.append("S2_class_weight_balanced")
        print(f"  S2 xong trong {time.time() - t0:.1f}s")

    # E1: Nested CV
    if "E1_nested" in exps_to_run:
        print("\n--- E1: Nested CV ---")
        t0 = time.time()
        run_nested_cv(X, y, folds)
        experiments_done.append("E1_nested")
        print(f"  E1 xong trong {time.time() - t0:.1f}s")

    # E2: Repeated CV
    if "E2_repeated" in exps_to_run:
        print("\n--- E2: Repeated CV ---")
        t0 = time.time()
        models_e2 = get_models()  # cấu hình E0
        run_repeated_cv(models_e2, X, y)
        experiments_done.append("E2_repeated")
        print(f"  E2 xong trong {time.time() - t0:.1f}s")

    # S3: Thêm Anxiety + Panic attack (làm sau cùng)
    if "S3_plus_anxiety_panic" in exps_to_run:
        print("\n--- S3: Thêm Anxiety + Panic attack (phụ, không dùng để kết luận) ---")
        t0 = time.time()
        X_s3, y_s3 = load_xy(extra_columns=EXTRA_COLUMNS_S3)
        models_s3 = get_models(extra_categorical=EXTRA_COLUMNS_S3)
        m_df, _ = run_cv_experiment("S3_plus_anxiety_panic", models_s3, X_s3, y_s3, folds)
        all_metrics.append(m_df)
        experiments_done.append("S3_plus_anxiety_panic")
        print(f"  S3 xong trong {time.time() - t0:.1f}s")

    # Tổng hợp metrics_summary và paired_differences
    metrics_path = TABLES_DIR / "metrics_per_fold.csv"
    if metrics_path.exists():
        full_metrics = pd.read_csv(metrics_path, encoding="utf-8")

        summary = summarize(full_metrics)
        summary.to_csv(
            TABLES_DIR / "metrics_summary.csv", index=False, encoding="utf-8"
        )
        print("\n[main] Đã lưu metrics_summary.csv")

        diffs = paired_differences(full_metrics)
        diffs.to_csv(
            TABLES_DIR / "paired_differences.csv", index=False, encoding="utf-8"
        )
        print("[main] Đã lưu paired_differences.csv")

    # Ghi run_config.json
    write_run_config(experiments_done)

    print(f"\n=== Hoàn thành. Tổng thời gian: {time.time() - t_total:.1f}s ===")
    print(f"Kết quả tại: {TABLES_DIR}")


if __name__ == "__main__":
    main()
