"""Quy trình làm sạch dữ liệu cho bộ Student Mental Health (Kaggle).

Module này chuẩn hóa dữ liệu ở mức **định dạng** (không thay đổi ngữ nghĩa,
không điền giá trị thiếu) nhằm bàn giao một quy trình nhất quán cho Member 3
(người xây dựng mô hình Machine Learning):

- Tuyệt đối **không** sửa file gốc ``data/raw/Student Mental health.csv``.
- ``clean_raw_data``: sao chép dữ liệu, cắt khoảng trắng thừa ở tên cột và
  giá trị các cột ``object``, chuẩn hóa chữ thường cột
  ``Your current year of Study``; **không** gộp tên ngành ở
  ``What is your course?``; **không** điền NaN ở ``Age`` (việc điền median
  do ``SimpleImputer`` trong Pipeline của Member 3 thực hiện trong quá trình
  cross-validation để tránh Data Leakage).
- ``get_feature_target_split``: tách 6 đặc trưng đầu vào và nhãn nhị phân
  (``Yes`` -> 1, ``No`` -> 0).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: Đường dẫn tới file gốc (chỉ đọc, không bao giờ ghi đè).
RAW_PATH: Path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "Student Mental health.csv"
)

#: Thư mục và đường dẫn lưu dữ liệu đã làm sạch.
PROCESSED_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "processed"
CLEANED_PATH: Path = PROCESSED_DIR / "student_mental_health_cleaned.csv"

#: 6 đặc trưng đầu vào được dùng trong mô hình chính.
FEATURE_COLUMNS: list[str] = [
    "Choose your gender",
    "Age",
    "What is your course?",
    "Your current year of Study",
    "What is your CGPA?",
    "Marital status",
]

#: Biến mục tiêu.
TARGET_COLUMN: str = "Do you have Depression?"


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch định dạng dữ liệu mà không thay đổi ngữ nghĩa gốc.

    Quy tắc áp dụng (không sửa file gốc — hàm luôn làm việc trên một bản sao):

    1. Cắt khoảng trắng thừa đầu/cuối ở tên tất cả các cột.
    2. Cắt khoảng trắng thừa đầu/cuối ở giá trị các cột kiểu ``object``.
    3. Chuẩn hóa chữ thường toàn bộ cột ``Your current year of Study``.
    4. Cột ``What is your course?`` chỉ được cắt khoảng trắng (đã thực hiện ở
       bước 2) — KHÔNG gộp các tên ngành viết tắt/gần giống nhau.
    5. Cột ``Age`` giữ nguyên giá trị thiếu (NaN) — không điền giá trị tại
       đây để tránh Data Leakage; median sẽ do ``SimpleImputer`` trong
       Pipeline của Member 3 xử lý bên trong cross-validation.

    Args:
        df: DataFrame gốc được đọc từ file CSV.

    Returns:
        DataFrame mới đã được làm sạch định dạng; ``df`` đầu vào không bị
        thay đổi.
    """
    cleaned = df.copy()

    # 1) Cắt khoảng trắng thừa ở tên cột.
    cleaned.columns = cleaned.columns.str.strip()

    # 2) Cắt khoảng trắng thừa ở giá trị các cột kiểu chuỗi (object/str).
    #    (.str.strip() giữ nguyên NaN, không làm mất giá trị thiếu.)
    string_cols = cleaned.select_dtypes(include=["object", "str"]).columns
    for col in string_cols:
        cleaned[col] = cleaned[col].str.strip()

    # 3) Chuẩn hóa chữ thường cho năm học (gộp "Year 1"/"year 1" -> "year 1").
    if "Your current year of Study" in cleaned.columns:
        cleaned["Your current year of Study"] = (
            cleaned["Your current year of Study"].str.lower()
        )

    # 4) "What is your course?" chỉ được strip (đã làm ở bước 2); không gộp nhóm.
    # 5) "Age" giữ nguyên NaN; việc điền median thuộc về Pipeline của Member 3.

    return cleaned


def get_feature_target_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Tách 6 đặc trưng đầu vào và nhãn nhị phân từ DataFrame đã làm sạch.

    Args:
        df: DataFrame đã được làm sạch (kết quả của :func:`clean_raw_data`).

    Returns:
        Tuple ``(X, y)`` trong đó:

        - ``X``: DataFrame gồm đúng 6 cột trong :data:`FEATURE_COLUMNS`.
        - ``y``: Series nhãn nhị phân ánh xạ ``Yes`` -> 1, ``No`` -> 0.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    if y.isnull().any():
        unexpected = sorted(df.loc[y.isnull(), TARGET_COLUMN].dropna().unique())
        raise ValueError(
            f"Cột mục tiêu chứa giá trị không thuộc {{Yes, No}}: {unexpected}"
        )

    return X, y


def _report(df: pd.DataFrame, title: str) -> None:
    """In báo cáo tóm tắt các chỉ số kiểm tra chất lượng dữ liệu."""
    print(f"\n=== {title} ===")
    print(f"Shape: {df.shape}")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("Giá trị thiếu: không có")
    else:
        print("Giá trị thiếu theo cột:")
        print(missing.to_string())

    year_col = "Your current year of Study"
    print(
        f"Unique '{year_col}': "
        f"{sorted(df[year_col].dropna().astype(str).unique())}"
    )
    print(
        f"Unique 'What is your CGPA?': "
        f"{sorted(df['What is your CGPA?'].dropna().astype(str).unique())}"
    )
    print(
        f"Số giá trị duy nhất 'What is your course?': "
        f"{df['What is your course?'].nunique()}"
    )


def main() -> None:
    """Chạy thử quy trình: đọc -> làm sạch -> tách -> lưu + in báo cáo."""
    raw = pd.read_csv(RAW_PATH, encoding="utf-8")
    _report(raw, "TRƯỚC khi làm sạch (raw)")

    cleaned = clean_raw_data(raw)
    _report(cleaned, "SAU khi làm sạch (cleaned)")

    X, y = get_feature_target_split(cleaned)
    print("\n=== Tách đặc trưng / nhãn ===")
    print(f"X shape: {X.shape}")
    print(f"Các cột của X: {list(X.columns)}")
    print("Phân bố nhãn y:")
    print(y.value_counts().to_string())

    # Lưu dữ liệu đã làm sạch định dạng (không đụng tới file gốc).
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEANED_PATH, index=False, encoding="utf-8")
    print(f"\nĐã lưu dữ liệu đã làm sạch vào: {CLEANED_PATH}")


if __name__ == "__main__":
    main()
