# -*- coding: utf-8 -*-
"""Xây dựng preprocessor và pipeline cho các mô hình ML (TV3).

Yêu cầu: scikit-learn >= 1.2 (dùng sparse_output, min_frequency).
"""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import (
    CATEGORICAL_COLUMNS,
    COURSE_COLUMN,
    MIN_FREQUENCY_COURSE,
    NUMERIC_COLUMNS,
    SEED,
)


def build_preprocessor(
    scale: bool,
    course_mode: str = "onehot",
    extra_categorical: list[str] | None = None,
) -> ColumnTransformer:
    """Xây dựng ColumnTransformer tiền xử lý dữ liệu.

    Args:
        scale: Có chuẩn hóa cột số (StandardScaler) hay không.
        course_mode: Cách mã hóa cột Course:
            - "onehot": one-hot bình thường (cấu hình chính).
            - "drop": bỏ hoàn toàn cột Course (S1a).
            - "min_freq": one-hot với min_frequency (S1b).
        extra_categorical: Các cột phân loại bổ sung (chỉ dùng S3).

    Returns:
        ColumnTransformer chưa được fit.
    """
    # Danh sách cột phân loại (không kể Course, sẽ xử lý riêng nếu cần)
    cat_cols_no_course = [c for c in CATEGORICAL_COLUMNS if c != COURSE_COLUMN]
    if extra_categorical:
        cat_cols_no_course = cat_cols_no_course + list(extra_categorical)

    # Transformer cho cột số
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    num_transformer = Pipeline(steps=num_steps)

    # Transformer cho cột phân loại thông thường
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    transformers: list = [
        ("num", num_transformer, NUMERIC_COLUMNS),
        ("cat", cat_transformer, cat_cols_no_course),
    ]

    if course_mode == "onehot":
        # Course nằm trong nhóm phân loại chung
        transformers = [
            ("num", num_transformer, NUMERIC_COLUMNS),
            (
                "cat",
                cat_transformer,
                cat_cols_no_course + [COURSE_COLUMN],
            ),
        ]
    elif course_mode == "drop":
        # S1a: bỏ hoàn toàn Course
        transformers = [
            ("num", num_transformer, NUMERIC_COLUMNS),
            ("cat", cat_transformer, cat_cols_no_course),
        ]
    elif course_mode == "min_freq":
        # S1b: Course có transformer riêng với min_frequency
        course_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        min_frequency=MIN_FREQUENCY_COURSE,
                        handle_unknown="infrequent_if_exist",
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers = [
            ("num", num_transformer, NUMERIC_COLUMNS),
            ("cat", cat_transformer, cat_cols_no_course),
            ("course", course_transformer, [COURSE_COLUMN]),
        ]
    else:
        raise ValueError(
            f"course_mode không hợp lệ: '{course_mode}'. "
            "Chọn 'onehot', 'drop' hoặc 'min_freq'."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_models(
    class_weight: str | None = None,
    course_mode: str = "onehot",
    extra_categorical: list[str] | None = None,
) -> dict[str, Pipeline]:
    """Trả về dict tên -> Pipeline (chưa fit) cho 5 mô hình.

    Args:
        class_weight: None (mặc định E0) hoặc "balanced" (S2).
        course_mode: Truyền vào build_preprocessor.
        extra_categorical: Truyền vào build_preprocessor (S3).

    Returns:
        Dict với khóa là tên mô hình (theo MODEL_NAMES).
    """
    # Preprocessor cho mô hình cần scale và không cần scale
    prep_scale = build_preprocessor(
        scale=True,
        course_mode=course_mode,
        extra_categorical=extra_categorical,
    )
    prep_no_scale = build_preprocessor(
        scale=False,
        course_mode=course_mode,
        extra_categorical=extra_categorical,
    )

    # class_weight áp dụng cho LR, DT, RF, SVM (không áp dụng Dummy)
    cw = class_weight  # None hoặc "balanced"

    models: dict[str, Pipeline] = {
        "Dummy": Pipeline(
            steps=[
                ("prep", prep_no_scale),
                ("clf", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "LogisticRegression": Pipeline(
            steps=[
                ("prep", prep_scale),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        # penalty='l2' deprecated ở sklearn 1.8+; dùng l1_ratio=0 thay thế
                        # Nhưng để tương thích rộng hơn, bỏ penalty (mặc định l2)
                        max_iter=1000,
                        random_state=SEED,
                        class_weight=cw,
                    ),
                ),
            ]
        ),
        "DecisionTree": Pipeline(
            steps=[
                ("prep", prep_no_scale),
                (
                    "clf",
                    DecisionTreeClassifier(
                        max_depth=3,
                        min_samples_leaf=3,
                        random_state=SEED,
                        class_weight=cw,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("prep", prep_no_scale),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=3,
                        min_samples_leaf=3,
                        random_state=SEED,
                        n_jobs=1,
                        class_weight=cw,
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            steps=[
                ("prep", prep_scale),
                (
                    "clf",
                    # SVC(probability=True) deprecated ở sklearn 1.9+
                    # Dùng CalibratedClassifierCV để lấy predict_proba
                    CalibratedClassifierCV(
                        SVC(
                            kernel="rbf",
                            C=1.0,
                            gamma="scale",
                            class_weight=cw,
                        ),
                        ensemble=False,
                    ),
                ),
            ]
        ),
    }
    return models


# ---------------------------------------------------------------------------
# Lưới siêu tham số cho E1 (Nested CV)
# Khóa dạng clf__<tham_số>
# ---------------------------------------------------------------------------
PARAM_GRIDS: dict[str, dict] = {
    "LogisticRegression": {
        "clf__C": [0.1, 1, 10],
    },
    "DecisionTree": {
        "clf__max_depth": [2, 3, 4],
        "clf__min_samples_leaf": [1, 3, 5],
    },
    "RandomForest": {
        "clf__max_depth": [3, None],
        "clf__min_samples_leaf": [1, 3],
    },
    "SVM": {
        # CalibratedClassifierCV(SVC(...)) → tham số qua estimator__C
        "clf__estimator__C": [0.1, 1, 10],
    },
}
# Dummy không có lưới tuning
