# -*- coding: utf-8 -*-
"""Hằng số cấu hình cho phần ML & Evaluation (TV3).

Module này chỉ chứa hằng số — không chứa logic.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Import từ TV1 (không sửa preprocessing.py)
# ---------------------------------------------------------------------------
import sys

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from preprocessing import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402

# ---------------------------------------------------------------------------
# Seed và tham số CV
# ---------------------------------------------------------------------------
SEED: int = 42
N_SPLITS: int = 5
INNER_SPLITS: int = 3
N_REPEATS: int = 10
POSITIVE_LABEL: int = 1

# ---------------------------------------------------------------------------
# Đường dẫn
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = REPO_ROOT / "results" / "ml"
TABLES_DIR: Path = RESULTS_DIR / "tables"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
REPORT_PATH: Path = REPO_ROOT / "report" / "experimental_results.md"

# Tạo thư mục nếu chưa có
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Cột feature
# ---------------------------------------------------------------------------
NUMERIC_COLUMNS: list[str] = ["Age"]
CATEGORICAL_COLUMNS: list[str] = [
    c for c in FEATURE_COLUMNS if c not in NUMERIC_COLUMNS
]

# Sanity check: NUMERIC + CATEGORICAL phải bằng FEATURE_COLUMNS
assert set(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS) == set(FEATURE_COLUMNS), (
    f"NUMERIC_COLUMNS + CATEGORICAL_COLUMNS không khớp FEATURE_COLUMNS.\n"
    f"FEATURE_COLUMNS: {FEATURE_COLUMNS}\n"
    f"NUMERIC_COLUMNS: {NUMERIC_COLUMNS}\n"
    f"CATEGORICAL_COLUMNS: {CATEGORICAL_COLUMNS}"
)
assert len(NUMERIC_COLUMNS) + len(CATEGORICAL_COLUMNS) == len(FEATURE_COLUMNS), (
    "Số cột không khớp (có thể có trùng lặp)"
)

# Cột Course (lấy từ FEATURE_COLUMNS, không gõ tay)
COURSE_COLUMN: str = [c for c in FEATURE_COLUMNS if "course" in c.lower()][0]
MIN_FREQUENCY_COURSE: int = 5

# Cột bị loại khỏi mô hình chính
EXCLUDED_COLUMNS: list[str] = [
    "Timestamp",
    "Do you have Anxiety?",
    "Do you have Panic attack?",
    "Did you seek any specialist for a treatment?",
]

# Cột thêm cho thí nghiệm phụ S3 (không bao gồm Treatment — quyết định D7)
EXTRA_COLUMNS_S3: list[str] = [
    "Do you have Anxiety?",
    "Do you have Panic attack?",
]

# ---------------------------------------------------------------------------
# Tên mô hình
# ---------------------------------------------------------------------------
MODEL_NAMES: list[str] = [
    "Dummy",
    "LogisticRegression",
    "DecisionTree",
    "RandomForest",
    "SVM",
]

# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------
PRIMARY_METRIC: str = "f1_yes"
CO_PRIMARY_METRIC: str = "recall_yes"
METRIC_COLUMNS: list[str] = [
    "accuracy",
    "precision_yes",
    "recall_yes",
    "f1_yes",
    "roc_auc",
    "pr_auc",
]

# ---------------------------------------------------------------------------
# ID thí nghiệm
# ---------------------------------------------------------------------------
EXPERIMENT_IDS: list[str] = [
    "E0_main",
    "E1_nested",
    "E2_repeated",
    "S1a_drop_course",
    "S1b_course_min_freq5",
    "S2_class_weight_balanced",
    "S3_plus_anxiety_panic",
]
