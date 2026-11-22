# -*- coding: utf-8 -*-
"""Tính metric, tổng hợp, so sánh cặp, vẽ hình và helper error analysis (TV3).

Mọi hàm vẽ:
- Có tiêu đề tiếng Việt, nhãn trục, chú giải.
- dpi=200, bbox_inches="tight".
- Lưu vào FIGURES_DIR.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
)

from config import (
    FIGURES_DIR,
    METRIC_COLUMNS,
    MODEL_NAMES,
    POSITIVE_LABEL,
    SEED,
)

matplotlib.rcParams["axes.unicode_minus"] = False

# Thử thiết lập font hỗ trợ tiếng Việt
def _setup_font() -> None:
    """Thiết lập font matplotlib hỗ trợ tiếng Việt."""
    import matplotlib.font_manager as fm
    # Ưu tiên DejaVu Sans (mặc định, hỗ trợ Latin có dấu tốt)
    # Nếu hệ thống có font Noto Sans thì dùng
    available = {f.name for f in fm.fontManager.ttflist}
    for font in ["Noto Sans", "Arial Unicode MS", "DejaVu Sans"]:
        if font in available:
            matplotlib.rcParams["font.family"] = font
            break

_setup_font()


# ---------------------------------------------------------------------------
# 1. Tính metric
# ---------------------------------------------------------------------------

def get_scores(fitted_estimator, X: pd.DataFrame) -> np.ndarray:
    """Trả về điểm số xác suất lớp dương (nếu có predict_proba) hoặc decision_function.

    Args:
        fitted_estimator: Pipeline đã được fit.
        X: Dữ liệu đầu vào.

    Returns:
        Mảng 1-D điểm số cho lớp dương.
    """
    if hasattr(fitted_estimator, "predict_proba"):
        return fitted_estimator.predict_proba(X)[:, 1]
    elif hasattr(fitted_estimator, "decision_function"):
        return fitted_estimator.decision_function(X)
    else:
        raise AttributeError(
            "Mô hình không có predict_proba hoặc decision_function."
        )


def compute_fold_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_score: np.ndarray,
) -> dict:
    """Tính các metric trên một fold.

    Args:
        y_true: Nhãn thật.
        y_pred: Nhãn dự đoán (nhị phân).
        y_score: Điểm số xác suất lớp dương.

    Returns:
        Dict chứa các metric và thông tin confusion matrix.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # ROC-AUC và PR-AUC: cần có cả 2 lớp
    n_unique = len(np.unique(y_true))
    if n_unique < 2:
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        roc_auc = roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score, pos_label=POSITIVE_LABEL)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_yes": precision_score(
            y_true, y_pred, pos_label=POSITIVE_LABEL,
            average="binary", zero_division=0
        ),
        "recall_yes": recall_score(
            y_true, y_pred, pos_label=POSITIVE_LABEL,
            average="binary", zero_division=0
        ),
        "f1_yes": f1_score(
            y_true, y_pred, pos_label=POSITIVE_LABEL,
            average="binary", zero_division=0
        ),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "n_test": len(y_true),
        "n_pos": int(y_true.sum()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# ---------------------------------------------------------------------------
# 2. Tổng hợp và so sánh cặp
# ---------------------------------------------------------------------------

def summarize(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp mean và std theo (experiment, model) cho các metric.

    Args:
        metrics_df: DataFrame có cột experiment, model, fold và các metric.

    Returns:
        DataFrame dạng rộng với mean và std (ddof=1) cho từng metric.
    """
    agg_dict = {}
    for col in METRIC_COLUMNS:
        agg_dict[f"{col}_mean"] = (col, "mean")
        agg_dict[f"{col}_std"] = (col, lambda x: x.std(ddof=1))

    # Dùng groupby + agg
    result_rows = []
    for (exp, model), group in metrics_df.groupby(["experiment", "model"]):
        row: dict = {"experiment": exp, "model": model}
        for col in METRIC_COLUMNS:
            vals = group[col].dropna()
            row[f"{col}_mean"] = vals.mean()
            row[f"{col}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
        result_rows.append(row)

    summary_df = pd.DataFrame(result_rows)
    return summary_df


def paired_differences(
    metrics_df: pd.DataFrame,
    references: tuple[str, ...] = ("Dummy", "LogisticRegression"),
) -> pd.DataFrame:
    """Tính hiệu metric (model - reference) theo từng fold.

    Args:
        metrics_df: DataFrame có cột experiment, model, fold và các metric.
        references: Các mô hình tham chiếu để so sánh.

    Returns:
        DataFrame với mean_diff, std_diff, n_folds_better.
    """
    diff_metrics = ["f1_yes", "recall_yes", "roc_auc"]
    rows = []

    for exp, exp_group in metrics_df.groupby("experiment"):
        for ref in references:
            ref_data = exp_group[exp_group["model"] == ref].set_index("fold")
            if ref_data.empty:
                continue
            for model in exp_group["model"].unique():
                if model == ref:
                    continue
                model_data = exp_group[exp_group["model"] == model].set_index("fold")
                for metric in diff_metrics:
                    # Join theo fold
                    merged = ref_data[[metric]].join(
                        model_data[[metric]], lsuffix="_ref", rsuffix="_model"
                    ).dropna()
                    if merged.empty:
                        continue
                    diffs = merged[f"{metric}_model"] - merged[f"{metric}_ref"]
                    rows.append({
                        "experiment": exp,
                        "model": model,
                        "reference": ref,
                        "metric": metric,
                        "mean_diff": diffs.mean(),
                        "std_diff": diffs.std(ddof=1) if len(diffs) > 1 else 0.0,
                        "n_folds_better": int((diffs > 0).sum()),
                    })

    return pd.DataFrame(rows)


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Chuyển DataFrame thành bảng Markdown.

    Args:
        df: DataFrame cần chuyển đổi.
        floatfmt: Định dạng số thực.

    Returns:
        Chuỗi Markdown.
    """
    return df.to_markdown(index=False, floatfmt=floatfmt)


# ---------------------------------------------------------------------------
# 3. Hàm vẽ
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, filename: str) -> Path:
    """Lưu hình vào FIGURES_DIR."""
    out_path = FIGURES_DIR / filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_confusion_matrices(
    oof_predictions: pd.DataFrame,
    experiment: str = "E0_main",
    models: list[str] | None = None,
) -> Path:
    """Vẽ ma trận nhầm lẫn (gộp out-of-fold) cho các mô hình.

    Args:
        oof_predictions: DataFrame có cột experiment, model, y_true, y_pred.
        experiment: ID thí nghiệm cần vẽ.
        models: Danh sách mô hình cần vẽ; mặc định dùng MODEL_NAMES.

    Returns:
        Đường dẫn file hình.
    """
    if models is None:
        models = MODEL_NAMES

    exp_data = oof_predictions[oof_predictions["experiment"] == experiment]
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        mdata = exp_data[exp_data["model"] == model]
        if mdata.empty:
            ax.set_title(f"{model}\n(không có dữ liệu)")
            continue
        cm = confusion_matrix(mdata["y_true"], mdata["y_pred"], labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{model}", fontsize=10)
        ax.set_xlabel("Nhãn dự đoán")
        ax.set_ylabel("Nhãn thật")

    fig.suptitle(
        f"Ma trận nhầm lẫn — {experiment} (gộp out-of-fold)",
        fontsize=12, y=1.02
    )
    return _save_fig(fig, "01_confusion_matrices_e0.png")


def plot_roc_pr_curves(
    oof_predictions: pd.DataFrame,
    experiment: str = "E0_main",
    models: list[str] | None = None,
) -> Path:
    """Vẽ đường cong ROC và PR (out-of-fold gộp).

    Args:
        oof_predictions: DataFrame có cột experiment, model, y_true, y_score.
        experiment: ID thí nghiệm.
        models: Danh sách mô hình.

    Returns:
        Đường dẫn file hình.
    """
    if models is None:
        models = MODEL_NAMES

    exp_data = oof_predictions[oof_predictions["experiment"] == experiment]

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for model, color in zip(models, colors):
        mdata = exp_data[exp_data["model"] == model]
        if mdata.empty:
            continue
        y_true = mdata["y_true"].values
        y_score = mdata["y_score"].values

        if len(np.unique(y_true)) < 2:
            continue

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val = roc_auc_score(y_true, y_score)
        ax_roc.plot(fpr, tpr, label=f"{model} (AUC={auc_val:.3f})", color=color)

        # PR
        prec, rec, _ = precision_recall_curve(y_true, y_score, pos_label=POSITIVE_LABEL)
        ap_val = average_precision_score(y_true, y_score, pos_label=POSITIVE_LABEL)
        ax_pr.plot(rec, prec, label=f"{model} (AP={ap_val:.3f})", color=color)

    # Đường random
    ax_roc.plot([0, 1], [0, 1], "k--", label="Random")
    ax_roc.set_xlabel("Tỷ lệ Dương tính Giả (FPR)")
    ax_roc.set_ylabel("Tỷ lệ Dương tính Thật (Recall)")
    ax_roc.set_title("Đường cong ROC (out-of-fold gộp)")
    ax_roc.legend(fontsize=8)

    # Đường tỷ lệ positive
    pos_rate = exp_data["y_true"].mean() if not exp_data.empty else 0.35
    ax_pr.axhline(y=pos_rate, color="k", linestyle="--", label=f"Baseline ({pos_rate:.3f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Đường cong Precision-Recall (out-of-fold gộp)")
    ax_pr.legend(fontsize=8)

    fig.suptitle(f"Đường cong ROC và PR — {experiment}", fontsize=13)
    fig.tight_layout()
    return _save_fig(fig, "02_roc_pr_oof_e0.png")


def plot_metrics_by_fold(
    metrics_df: pd.DataFrame,
    experiment: str = "E0_main",
    metric: str = "f1_yes",
) -> Path:
    """Vẽ metric theo từng fold cho mỗi mô hình.

    Args:
        metrics_df: DataFrame có cột experiment, model, fold và metric.
        experiment: ID thí nghiệm.
        metric: Tên metric cần vẽ.

    Returns:
        Đường dẫn file hình.
    """
    exp_data = metrics_df[metrics_df["experiment"] == experiment]
    models = [m for m in MODEL_NAMES if m in exp_data["model"].unique()]
    folds = sorted(exp_data["fold"].unique())

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for model, color in zip(models, colors):
        mdata = exp_data[exp_data["model"] == model].sort_values("fold")
        ax.plot(mdata["fold"], mdata[metric], marker="o", label=model, color=color)

    ax.set_xlabel("Fold")
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title(f"{metric.upper()} theo từng fold — {experiment}")
    ax.legend(fontsize=9)
    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    fig.tight_layout()
    return _save_fig(fig, "03_metrics_by_fold_e0.png")


def plot_model_comparison(
    summary_df: pd.DataFrame,
    experiments: list[str] | None = None,
    metric: str = "f1_yes",
) -> Path:
    """Vẽ so sánh mô hình (mean ± std) giữa các thí nghiệm.

    Args:
        summary_df: DataFrame kết quả của summarize().
        experiments: Danh sách experiment cần vẽ.
        metric: Metric cần so sánh.

    Returns:
        Đường dẫn file hình.
    """
    if experiments is None:
        experiments = ["E0_main", "E1_nested"]

    data = summary_df[summary_df["experiment"].isin(experiments)]
    models = [m for m in MODEL_NAMES if m in data["model"].unique()]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.35
    n_exp = len(experiments)
    offsets = np.linspace(-(n_exp - 1) * width / 2, (n_exp - 1) * width / 2, n_exp)
    colors = plt.cm.tab10(np.linspace(0, 0.5, n_exp))

    for exp, offset, color in zip(experiments, offsets, colors):
        exp_data = data[data["experiment"] == exp]
        means, stds = [], []
        for m in models:
            row = exp_data[exp_data["model"] == m]
            if row.empty:
                means.append(0)
                stds.append(0)
            else:
                means.append(float(row[f"{metric}_mean"].values[0]))
                stds.append(float(row[f"{metric}_std"].values[0]))
        ax.bar(
            x + offset, means, width,
            yerr=stds, capsize=4,
            label=exp, color=color, alpha=0.85
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"So sánh mô hình theo {metric.upper()} (mean ± std CV)")
    ax.legend()
    fig.tight_layout()
    return _save_fig(fig, "04_model_comparison_e0_e1.png")


def plot_repeated_distribution(
    metrics_repeated: pd.DataFrame,
    metric: str = "f1_yes",
) -> Path:
    """Vẽ phân bố metric trên 50 fold (E2 — Repeated CV).

    Args:
        metrics_repeated: DataFrame có cột model, repeat, fold và metric.
        metric: Metric cần vẽ.

    Returns:
        Đường dẫn file hình.
    """
    import seaborn as sns

    models = [m for m in MODEL_NAMES if m in metrics_repeated["model"].unique()]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=metrics_repeated[metrics_repeated["model"].isin(models)],
        x="model",
        y=metric,
        order=models,
        ax=ax,
        palette="tab10",
    )
    ax.set_title(f"Phân bố {metric.upper()} trên 50 fold — E2 Repeated CV\n"
                 "(Lưu ý: các fold không độc lập, std chỉ tham khảo)")
    ax.set_xlabel("Mô hình")
    ax.set_ylabel(metric.upper())
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return _save_fig(fig, "05_repeated_cv_distribution.png")


def plot_sensitivity(
    summary_df: pd.DataFrame,
    metric: str = "f1_yes",
) -> Path:
    """Vẽ phân tích độ nhạy giữa các thí nghiệm phụ.

    Args:
        summary_df: DataFrame kết quả của summarize().
        metric: Metric cần vẽ.

    Returns:
        Đường dẫn file hình.
    """
    sensitivity_exps = [
        "E0_main",
        "S1a_drop_course",
        "S1b_course_min_freq5",
        "S2_class_weight_balanced",
    ]
    data = summary_df[summary_df["experiment"].isin(sensitivity_exps)]
    models = [m for m in MODEL_NAMES if m in data["model"].unique()]

    fig, ax = plt.subplots(figsize=(11, 5))
    n_exp = len(sensitivity_exps)
    x = np.arange(len(models))
    width = 0.18
    offsets = np.linspace(-(n_exp - 1) * width / 2, (n_exp - 1) * width / 2, n_exp)
    colors = plt.cm.tab10(np.linspace(0, 0.8, n_exp))

    for exp, offset, color in zip(sensitivity_exps, offsets, colors):
        exp_data = data[data["experiment"] == exp]
        means = []
        for m in models:
            row = exp_data[exp_data["model"] == m]
            means.append(float(row[f"{metric}_mean"].values[0]) if not row.empty else 0)
        ax.bar(x + offset, means, width, label=exp, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Phân tích độ nhạy — {metric.upper()} (phân tích phụ, không thay kết quả chính)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save_fig(fig, "06_sensitivity.png")


def plot_error_rate_by_group(error_rate_df: pd.DataFrame) -> Path:
    """Vẽ tỷ lệ sai theo nhóm.

    Args:
        error_rate_df: DataFrame kết quả của error_rate_by_group().

    Returns:
        Đường dẫn file hình.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    groups = error_rate_df["group_value"].astype(str)
    x = np.arange(len(groups))
    ax.bar(x, error_rate_df["error_rate"], color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel("Tỷ lệ sai")
    feature = error_rate_df["feature"].iloc[0] if not error_rate_df.empty else ""
    ax.set_title(f"Tỷ lệ sai theo nhóm — {feature}")
    # Ghi n lên cột
    for xi, (_, row) in zip(x, error_rate_df.iterrows()):
        if row["n"] < 10:
            ax.text(xi, row["error_rate"] + 0.01, f"n={row['n']}\n(!)", ha="center", fontsize=8, color="red")
        else:
            ax.text(xi, row["error_rate"] + 0.01, f"n={row['n']}", ha="center", fontsize=8)
    fig.tight_layout()
    return _save_fig(fig, "07_error_rate_by_group.png")


def plot_permutation_importance(perm_imp_df: pd.DataFrame) -> Path:
    """Vẽ permutation importance (mean ± std qua các fold).

    Args:
        perm_imp_df: DataFrame với cột feature, model, mean_importance, std_importance.

    Returns:
        Đường dẫn file hình.
    """
    models = perm_imp_df["model"].unique()
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)

    for ax, model in zip(axes[0], models):
        mdata = perm_imp_df[perm_imp_df["model"] == model].sort_values(
            "mean_importance", ascending=True
        )
        ax.barh(
            mdata["feature"],
            mdata["mean_importance"],
            xerr=mdata["std_importance"],
            color="teal", alpha=0.8, capsize=4
        )
        ax.set_title(f"Permutation Importance — {model}")
        ax.set_xlabel("Giảm ROC-AUC (trung bình qua các fold)")

    fig.suptitle("Tầm quan trọng feature theo Permutation Importance\n"
                 "(Lưu ý: không diễn giải nhân quả)", fontsize=11)
    fig.tight_layout()
    return _save_fig(fig, "08_permutation_importance.png")


def plot_lr_coefficients(lr_coef_df: pd.DataFrame) -> Path:
    """Vẽ hệ số Logistic Regression (fit trên toàn bộ dữ liệu, chỉ minh họa).

    Args:
        lr_coef_df: DataFrame với cột feature và coefficient.

    Returns:
        Đường dẫn file hình.
    """
    df = lr_coef_df.sort_values("coefficient", ascending=True)
    colors = ["red" if c > 0 else "steelblue" for c in df["coefficient"]]
    fig, ax = plt.subplots(figsize=(8, max(5, len(df) * 0.3)))
    ax.barh(df["feature"], df["coefficient"], color=colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Hệ số Logistic Regression (fit trên toàn bộ dữ liệu — chỉ minh họa)\n"
                 "Không diễn giải nhân quả")
    ax.set_xlabel("Hệ số")
    fig.tight_layout()
    return _save_fig(fig, "09_lr_coefficients.png")


# ---------------------------------------------------------------------------
# 4. Helper Error Analysis
# ---------------------------------------------------------------------------

def misclassified_table(
    oof_predictions: pd.DataFrame,
    X: pd.DataFrame,
    model: str,
    experiment: str = "E0_main",
    error_type: str = "both",
) -> pd.DataFrame:
    """Trả về bảng các dòng bị phân loại sai (FN hoặc FP).

    Args:
        oof_predictions: DataFrame có cột experiment, model, row_index, y_true, y_pred.
        X: Dữ liệu đặc trưng gốc (theo chỉ số dòng).
        model: Tên mô hình.
        experiment: ID thí nghiệm.
        error_type: "fn" (False Negative), "fp" (False Positive), hoặc "both".

    Returns:
        DataFrame các dòng bị sai kèm thông tin feature.
    """
    mdata = oof_predictions[
        (oof_predictions["experiment"] == experiment)
        & (oof_predictions["model"] == model)
    ].copy()

    if error_type == "fn":
        mask = (mdata["y_true"] == 1) & (mdata["y_pred"] == 0)
    elif error_type == "fp":
        mask = (mdata["y_true"] == 0) & (mdata["y_pred"] == 1)
    else:
        mask = mdata["y_true"] != mdata["y_pred"]

    errors = mdata[mask].set_index("row_index")
    return errors.join(X, how="left").reset_index()


def error_rate_by_group(
    oof_predictions: pd.DataFrame,
    X: pd.DataFrame,
    feature: str,
    model: str,
    experiment: str = "E0_main",
    age_bins: list | None = None,
) -> pd.DataFrame:
    """Tính tỷ lệ dự đoán sai theo nhóm của một feature.

    Args:
        oof_predictions: DataFrame OOF predictions.
        X: Dữ liệu đặc trưng.
        feature: Tên cột cần nhóm.
        model: Tên mô hình.
        experiment: ID thí nghiệm.
        age_bins: Nếu feature là "Age", truyền bins để nhóm.

    Returns:
        DataFrame với cột feature, group_value, n, n_errors, error_rate.
    """
    mdata = oof_predictions[
        (oof_predictions["experiment"] == experiment)
        & (oof_predictions["model"] == model)
    ].copy()
    mdata = mdata.set_index("row_index").join(X[[feature]], how="left").reset_index()
    mdata["is_error"] = (mdata["y_true"] != mdata["y_pred"]).astype(int)

    if feature == "Age" and age_bins is not None:
        labels = [f"{age_bins[i]}-{age_bins[i+1]-1}" for i in range(len(age_bins) - 1)]
        mdata["group_value"] = pd.cut(
            mdata[feature], bins=age_bins, labels=labels, right=False
        )
    else:
        mdata["group_value"] = mdata[feature]

    result = (
        mdata.groupby("group_value", observed=True)
        .agg(n=("is_error", "count"), n_errors=("is_error", "sum"))
        .reset_index()
    )
    result["error_rate"] = result["n_errors"] / result["n"]
    result.insert(0, "feature", feature)
    return result


def consistent_errors(
    oof_predictions: pd.DataFrame,
    experiment: str = "E0_main",
) -> pd.DataFrame:
    """Tìm các dòng bị mọi mô hình (trừ Dummy) phân loại sai.

    Args:
        oof_predictions: DataFrame OOF predictions.
        experiment: ID thí nghiệm.

    Returns:
        DataFrame các row_index bị tất cả mô hình sai.
    """
    exp_data = oof_predictions[oof_predictions["experiment"] == experiment].copy()
    exp_data["is_error"] = (exp_data["y_true"] != exp_data["y_pred"]).astype(int)
    non_dummy = exp_data[exp_data["model"] != "Dummy"]

    # Các dòng mà mọi mô hình (trừ Dummy) đều sai
    models_non_dummy = non_dummy["model"].unique()
    n_models = len(models_non_dummy)

    error_counts = (
        non_dummy.groupby("row_index")["is_error"].sum().reset_index()
    )
    consistent = error_counts[error_counts["is_error"] == n_models].copy()
    consistent.columns = ["row_index", "n_models_wrong"]

    # Lấy y_true từ bất kỳ mô hình nào
    y_true_map = (
        non_dummy.drop_duplicates("row_index")[["row_index", "y_true"]]
        .set_index("row_index")["y_true"]
    )
    consistent["y_true"] = consistent["row_index"].map(y_true_map)
    return consistent


def permutation_importance_cv(
    pipelines: dict[str, object],
    folds: list[tuple],
    X: pd.DataFrame,
    y: pd.Series,
    models_to_eval: list[str] | None = None,
    n_repeats: int = 30,
) -> pd.DataFrame:
    """Tính permutation importance trên test fold (mỗi fold một lần).

    Hoán vị cột gốc (trước one-hot) để Course được đánh giá như một cột.

    Args:
        pipelines: Dict tên -> Pipeline (đã được fit trên train fold tương ứng).
        folds: Danh sách (train_idx, test_idx).
        X: Dữ liệu đặc trưng gốc.
        y: Nhãn.
        models_to_eval: Mô hình cần đánh giá; mặc định ["LogisticRegression", "RandomForest"].
        n_repeats: Số lần hoán vị.

    Returns:
        DataFrame với cột model, feature, mean_importance, std_importance.
    """
    from sklearn.base import clone
    from sklearn.metrics import make_scorer

    if models_to_eval is None:
        models_to_eval = ["LogisticRegression", "RandomForest"]

    # sklearn 1.9+: dùng "roc_auc" string scorer thay vì make_scorer(roc_auc_score, needs_proba=True)
    scoring = "roc_auc"
    X_arr = X.reset_index(drop=True)
    y_arr = y.reset_index(drop=True)

    all_rows = []
    for model_name in models_to_eval:
        feature_importances: dict[str, list[float]] = {col: [] for col in X.columns}

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train = X_arr.iloc[train_idx]
            y_train = y_arr.iloc[train_idx]
            X_test = X_arr.iloc[test_idx]
            y_test = y_arr.iloc[test_idx]

            pipe = clone(pipelines[model_name])
            pipe.fit(X_train, y_train)

            result = permutation_importance(
                pipe, X_test, y_test,
                scoring=scoring,
                n_repeats=n_repeats,
                random_state=SEED + fold_idx,
                n_jobs=1,
            )
            for i, col in enumerate(X.columns):
                feature_importances[col].extend(result.importances[i].tolist())

        for feat, vals in feature_importances.items():
            vals_arr = np.array(vals)
            all_rows.append({
                "model": model_name,
                "feature": feat,
                "mean_importance": vals_arr.mean(),
                "std_importance": vals_arr.std(ddof=1) if len(vals_arr) > 1 else 0.0,
            })

    return pd.DataFrame(all_rows)


def lr_coefficients_table(
    pipeline: object,
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Fit LR trên toàn bộ dữ liệu và trả về bảng hệ số (chỉ để minh họa).

    Args:
        pipeline: Pipeline LogisticRegression chưa fit.
        X: Toàn bộ dữ liệu đặc trưng.
        y: Toàn bộ nhãn.

    Returns:
        DataFrame với cột feature và coefficient, sắp theo |coefficient| giảm dần.
    """
    from sklearn.base import clone

    pipe = clone(pipeline)
    pipe.fit(X, y)

    feature_names = pipe.named_steps["prep"].get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_[0]

    df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    df = df.reindex(df["coefficient"].abs().sort_values(ascending=False).index)
    return df.reset_index(drop=True)
