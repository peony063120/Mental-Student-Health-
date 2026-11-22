# Mental Student Health — Dự đoán Trầm cảm Tự khai báo của Sinh viên

> **Đề tài:** Research Track — Trí tuệ Nhân tạo  
> **Bộ dữ liệu:** Student Mental Health (Kaggle, n=101)  
> **Mục tiêu:** Dự đoán tình trạng trầm cảm tự khai báo bằng Machine Learning cổ điển.

---

## Cấu trúc Repo

```
data/
├── raw/Student Mental health.csv        # Dữ liệu gốc (chỉ đọc)
└── processed/student_mental_health_cleaned.csv
src/
├── preprocessing.py    # TV1 — làm sạch dữ liệu (KHÔNG SỬA)
├── config.py           # TV3 — hằng số, đường dẫn, tên mô hình
├── pipelines.py        # TV3 — preprocessor và pipeline các mô hình
├── evaluation.py       # TV3 — metric, hình, helper error analysis
└── train.py            # TV3 — huấn luyện, CLI
notebooks/
├── 01_data_inspection.ipynb    # TV1
├── 02_eda.ipynb                # TV1
├── 03_pipeline_leakage_check.ipynb   # TV3 — kiểm tra chống rò rỉ
├── 04_model_comparison.ipynb         # TV3 — bảng và hình so sánh
└── 05_error_analysis.ipynb           # TV3 — phân tích sai số
results/ml/
├── tables/   # CSV kết quả (do train.py sinh ra)
└── figures/  # Hình (9 hình PNG)
report/
├── EDA.md                      # TV1
└── experimental_results.md     # TV3
dataset_info.md                 # TV1
requirements.txt
```

---

## Phần ML & Evaluation (TV3)

### Cài đặt

```bash
pip install -r requirements.txt
```

Yêu cầu: Python ≥ 3.10, scikit-learn ≥ 1.9.1 (dùng `sparse_output`, `min_frequency`).  
Trên Windows: `set PYTHONUTF8=1` trước khi chạy.

### Chạy thí nghiệm

```bash
# Từ thư mục gốc repo:

# Chỉ chạy kết quả chính (E0)
python src/train.py --experiments e0

# Chạy tất cả thí nghiệm (E0, E1, E2, S1a, S1b, S2, S3)
python src/train.py --experiments all

# Chạy từng nhóm tùy chọn
python src/train.py --experiments e0 e1 s1 s2
```

| Mã | Thí nghiệm | Mô tả |
|----|-----------|-------|
| `e0` | E0_main | **Kết quả chính** — 5-fold CV, siêu tham số cố định |
| `e1` | E1_nested | Nested CV — tuning không rò rỉ |
| `e2` | E2_repeated | Repeated CV (5×10) — đo độ dao động |
| `s1` | S1a + S1b | Phân tích độ nhạy cho cột Course |
| `s2` | S2_class_weight_balanced | class_weight='balanced' |
| `s3` | S3_plus_anxiety_panic | +Anxiety +Panic (phụ, khác khung chính) |

### Thứ tự chạy đầy đủ

```bash
# 1. Chạy tất cả thí nghiệm
python src/train.py --experiments all

# 2. Kiểm tra chống rò rỉ (C1–C10)
python -m nbconvert --to notebook --execute --inplace notebooks/03_pipeline_leakage_check.ipynb

# 3. Sinh hình và bảng so sánh
python -m nbconvert --to notebook --execute --inplace notebooks/04_model_comparison.ipynb

# 4. Phân tích sai số
python -m nbconvert --to notebook --execute --inplace notebooks/05_error_analysis.ipynb
```

### Kết quả nằm ở đâu

| File | Nội dung |
|------|---------|
| `results/ml/tables/metrics_per_fold.csv` | Metric mỗi mô hình × mỗi fold × mỗi thí nghiệm |
| `results/ml/tables/metrics_summary.csv` | Mean ± std tổng hợp |
| `results/ml/tables/oof_predictions.csv` | Dự đoán out-of-fold |
| `results/ml/tables/best_params.csv` | Tham số tốt nhất (E1) |
| `results/ml/tables/metrics_repeated.csv` | Kết quả E2 (250 dòng) |
| `results/ml/tables/paired_differences.csv` | So sánh cặp mô hình |
| `results/ml/tables/run_config.json` | Cấu hình chạy, phiên bản thư viện |
| `results/ml/figures/01_*.png` … `09_*.png` | 9 hình PNG |
| `report/experimental_results.md` | Báo cáo thực nghiệm đầy đủ |

### Kết quả chính (E0 — 5-Fold CV)

| Mô hình | F1(Yes) mean ± std | Recall(Yes) mean | ROC-AUC mean |
|---------|-------------------|-----------------|-------------|
| Dummy | 0.000 ± 0.000 | 0.000 | 0.500 |
| LogisticRegression | **0.666 ± 0.111** | **0.543** | **0.806** |
| DecisionTree | 0.620 ± 0.117 | 0.457 | 0.760 |
| RandomForest | 0.620 ± 0.117 | 0.457 | 0.726 |
| SVM | 0.659 ± 0.120 | 0.543 | 0.776 |

*(Nguồn: `results/ml/tables/metrics_summary.csv` — không điền tay)*

> ⚠️ Nhãn là **tự khai báo**, n=101. Mô hình **không phải công cụ chẩn đoán**. Kết quả chỉ áp dụng trên bộ dữ liệu này.

### Tái lập

Seed = 42. Xóa `results/ml/` và chạy lại `python src/train.py --experiments all` → kết quả y hệt.  
Xem chi tiết cấu hình tại `results/ml/tables/run_config.json`.

---

## Phần Data & EDA (TV1)

Xem `dataset_info.md` và `report/EDA.md`.

## Phần Literature & Methodology (TV2)

*(TV2 bổ sung)*
