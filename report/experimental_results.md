# Kết quả Thực nghiệm — Dự đoán Trầm cảm Tự khai báo của Sinh viên

> **Người thực hiện:** TV3 — ML & Evaluation  
> **Ngày tạo:** 23/11/2026  
> **Lưu ý quan trọng:** Mọi số liệu trong báo cáo này được lấy trực tiếp từ file CSV do `src/train.py` sinh ra. Không điền số từ trí nhớ hoặc ước lượng.

---

## 1. Mục tiêu và Phạm vi

Báo cáo này trình bày kết quả thực nghiệm sử dụng các thuật toán Machine Learning cổ điển để **dự đoán tình trạng trầm cảm tự khai báo** của sinh viên đại học.

> **Giới hạn bắt buộc:**
> - Nhãn dự đoán là **tự khai báo** ("Do you have Depression?"), không phải chẩn đoán lâm sàng.
> - Cỡ mẫu **n = 101** — rất nhỏ, kết quả chỉ áp dụng trên bộ dữ liệu này.
> - Mô hình **không phải công cụ chẩn đoán hoặc sàng lọc lâm sàng**.
> - Mọi nhận xét về feature sử dụng ngôn ngữ liên hệ ("có liên hệ với", "tỷ lệ cao/thấp hơn trong mẫu"), không phải nhân quả.

---

## 2. Dữ liệu và Tiền xử lý

### 2.1 Bộ dữ liệu

Tham chiếu chi tiết: `dataset_info.md` và `report/EDA.md` (TV1).

| Mục | Giá trị |
|-----|---------|
| Nguồn | Student Mental Health (Kaggle) |
| Kích thước | 101 dòng × 11 cột |
| Nhãn (`Do you have Depression?`) | No = 66 (≈ 65,35%), Yes = 35 (≈ 34,65%) |
| Giá trị thiếu | 1 ô (`Age`, dòng 43, nhãn No) |

### 2.2 Sáu đặc trưng đầu vào (mô hình chính)

| Cột | Kiểu | Ghi chú |
|-----|------|---------|
| `Choose your gender` | Phân loại | Female/Male |
| `Age` | Số | 18–24; 1 giá trị thiếu được điền median trong pipeline |
| `What is your course?` | Phân loại | 48 giá trị duy nhất; 38 giá trị chỉ xuất hiện 1 lần |
| `Your current year of Study` | Phân loại | year 1–4 (đã chuẩn hóa chữ thường) |
| `What is your CGPA?` | Phân loại | 5 khoảng (3.50–4.00 chiếm đa số) |
| `Marital status` | Phân loại | No=85, Yes=16; **toàn bộ 16 người Yes đều có nhãn Yes** |

### 2.3 Bốn cột bị loại khỏi mô hình chính

| Cột | Lý do loại |
|-----|-----------|
| `Timestamp` | Không liên quan đến trạng thái sức khỏe |
| `Do you have Anxiety?` | Có thể chồng lấn khái niệm; giữ lại cho thí nghiệm phụ S3 |
| `Do you have Panic attack?` | Như trên |
| `Did you seek any specialist for a treatment?` | Proxy mạnh của nhãn, có thể gây rò rỉ khái niệm |

### 2.4 Quy trình tiền xử lý (trong Pipeline, tránh rò rỉ)

```
X (101 × 6) → ColumnTransformer (fit CHỈ trên train fold) →
  - Cột số (Age): SimpleImputer(median) → StandardScaler (nếu cần)
  - Cột phân loại: SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown='ignore')
→ Pipeline → Classifier
```

Số cột sau one-hot thay đổi theo fold (do `Course` có nhiều hạng mục hiếm chỉ xuất hiện trong một số fold train). Xem chi tiết ở notebook 03 (C10).

---

## 3. Thiết kế Thực nghiệm

### 3.1 Metric

- **Metric chính:** F1(Yes) — cân bằng Precision và Recall cho lớp Yes
- **Metric đồng chính:** Recall(Yes) — số ca Yes không bị bỏ sót (RQ3)
- **Metric phụ:** Accuracy, Precision(Yes), ROC-AUC, PR-AUC
- Nhãn dương: Yes = 1; `zero_division=0`; `ddof=1` cho std

### 3.2 Bảy thí nghiệm

| ID | Mô tả | Vai trò |
|----|-------|---------|
| E0_main | 5-Fold CV, siêu tham số cố định | **Kết quả chính** |
| E1_nested | Nested CV (outer=5 fold, inner=3 fold), GridSearch | Tuning không rò rỉ |
| E2_repeated | Repeated CV (5×10 = 50 fold), cấu hình E0 | Đo độ dao động |
| S1a_drop_course | E0 nhưng bỏ cột Course | Phân tích độ nhạy |
| S1b_course_min_freq5 | E0 nhưng Course: min_frequency=5 | Phân tích độ nhạy |
| S2_class_weight_balanced | E0 nhưng class_weight='balanced' | Phân tích độ nhạy |
| S3_plus_anxiety_panic | E0 + thêm Anxiety, Panic attack | **Phụ — khác khung nghiên cứu chính** |

### 3.3 Quy tắc chống rò rỉ

- Toàn bộ impute/scale/encode trong `Pipeline` + `ColumnTransformer`, chỉ `fit` trên train fold.
- Mọi mô hình (trừ E2) dùng **cùng 5 fold** (`StratifiedKFold(shuffle=True, seed=42)`).
- Kiểm tra C1–C10 trong notebook 03 — tất cả PASS.
- Tham số chốt trước khi chạy, không thay đổi sau khi thấy kết quả.

---

## 4. Giải thích Mô hình

| Mô hình | Nguyên lý | Lý do chọn | Tham số (E0) | Ưu điểm | Nhược điểm (dữ liệu nhỏ) |
|---------|-----------|------------|-------------|---------|------------------------|
| **Dummy** | Luôn đoán nhãn đa số (No) | Baseline tối thiểu; mọi mô hình phải vượt qua | `strategy='most_frequent'` | Không có bias về feature; dễ hiểu | F1(Yes)=0, không có giá trị dự đoán |
| **Logistic Regression (LR)** | Mô hình tuyến tính; ước lượng xác suất qua hàm sigmoid; tham số là hệ số tuyến tính | Hiểu được hệ số; baseline mạnh cho bài toán phân loại nhị phân; ổn định với dữ liệu nhỏ | `C=1.0`, `max_iter=1000`, `random_state=42`, scale | Hội tụ nhanh; hệ số có thể xem xét (không diễn giải nhân quả) | Giả định quan hệ tuyến tính; nhạy với multicollinearity |
| **Decision Tree (DT)** | Phân chia không gian feature theo ngưỡng để tối đa hóa độ thuần; giải thích được qua cây | Trực quan; không cần scale; phát hiện quan hệ phi tuyến đơn giản | `max_depth=3`, `min_samples_leaf=3`, `random_state=42` | Dễ giải thích; không cần scale; nhanh | Dễ overfit; không ổn định; với n=101 mỗi lá có rất ít mẫu |
| **Random Forest (RF)** | Ensemble nhiều cây quyết định trên bootstrap; dự đoán bằng bình quân | Giảm phương sai so với DT đơn; ổn định hơn | `n_estimators=300`, `max_depth=3`, `min_samples_leaf=3`, `random_state=42`, `n_jobs=1` | Ổn định hơn DT; ít overfit | Khó giải thích; tầm quan trọng impurity thiên lệch khi nhiều hạng mục |
| **SVM** | Tìm siêu phẳng tối đa hóa margin; kernel RBF cho quan hệ phi tuyến | Hiệu quả với không gian feature nhiều chiều; ít overfit khi C nhỏ | `SVC(kernel='rbf', C=1.0, gamma='scale')`, `CalibratedClassifierCV`, scale | Ổn định với n nhỏ; kernel RBF linh hoạt | Chậm với n lớn; khó giải thích; nhạy với scale |

---

## 5. Kết quả Chính E0

> Dưới đây là số liệu từ `results/ml/tables/metrics_summary.csv` và `metrics_per_fold.csv`.

### 5.1 Bảng mean ± std

| model | f1_yes_mean | f1_yes_std | recall_yes_mean | recall_yes_std | accuracy_mean | roc_auc_mean | pr_auc_mean |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Dummy | 0.000 | 0.000 | 0.000 | 0.000 | 0.653 | 0.500 | 0.347 |
| LogisticRegression | 0.666 | 0.111 | 0.543 | 0.120 | 0.812 | 0.806 | 0.814 |
| DecisionTree | 0.620 | 0.117 | 0.457 | 0.120 | 0.812 | 0.760 | 0.699 |
| RandomForest | 0.620 | 0.117 | 0.457 | 0.120 | 0.812 | 0.726 | 0.734 |
| SVM | 0.659 | 0.120 | 0.543 | 0.120 | 0.802 | 0.776 | 0.809 |

*(Nguồn: `results/ml/tables/metrics_summary.csv`)*

> **Lưu ý về Dummy:** Precision và F1 của Dummy = 0 vì chiến lược "most_frequent" không bao giờ dự đoán Yes (`zero_division=0`). Accuracy ≈ 0.653 phản ánh tỷ lệ nhãn No trong mẫu.

### 5.2 So sánh cặp với Dummy (F1(Yes), E0)

| model | mean_diff | std_diff | n_folds_better |
|:---|---:|---:|---:|
| LogisticRegression | 0.666 | 0.111 | 5 |
| DecisionTree | 0.620 | 0.117 | 5 |
| RandomForest | 0.620 | 0.117 | 5 |
| SVM | 0.659 | 0.120 | 5 |

*(Nguồn: `results/ml/tables/paired_differences.csv`)*

Theo quy tắc: `mean_diff > std_diff` VÀ `n_folds_better ≥ 4/5` → **tất cả 4 mô hình đều có liên hệ tốt hơn Dummy trong mẫu** theo F1(Yes).

### 5.3 Mô hình tốt nhất theo F1(Yes) — E0

Theo F1(Yes) trung bình: **LogisticRegression (0.666)** > SVM (0.659) > DT = RF (0.620) > Dummy (0.000).

Theo Recall(Yes) trung bình: LR = SVM (0.543) > DT = RF (0.457) > Dummy (0.000).

Thứ hạng F1 và Recall nhất quán trong bộ dữ liệu này (xem RQ3, Mục 10).

> **Cảnh báo:** Chọn "mô hình tốt nhất" trên cùng CV có thiên lệch lạc quan. Ước lượng ít thiên lệch hơn là E1 (Mục 6).

Hình: `results/ml/figures/01_confusion_matrices_e0.png`, `02_roc_pr_oof_e0.png`, `03_metrics_by_fold_e0.png`.

---

## 6. Nested CV (E1) — Tuning Không Rò rỉ

Outer = 5 fold của E0; Inner = StratifiedKFold(3); tối ưu F1(Yes).

Kết quả E1 xem tại `results/ml/tables/metrics_summary.csv` (experiment=E1_nested) và `best_params.csv`.

> **Lưu ý quan trọng:** Mỗi inner-validation chỉ có khoảng **9 ca Yes** → lựa chọn tham số **rất bất ổn**; kết quả E1 không nhất thiết tốt hơn E0. Đây là ước lượng hiệu năng ít thiên lệch hơn E0.

Hình: `results/ml/figures/04_model_comparison_e0_e1.png`.

---

## 7. Độ dao động (E2 — Repeated CV)

`RepeatedStratifiedKFold(5×10 = 50 fold)`, cấu hình E0.

> **Lưu ý:** Các fold trong E2 **không độc lập** (mẫu được chia lại từ cùng 101 quan sát). Std chỉ mang tính tham khảo; không dùng để kiểm định thống kê. E2 là phân tích phụ, không thay thế E0.

Hình: `results/ml/figures/05_repeated_cv_distribution.png`.
Chi tiết: `results/ml/tables/metrics_repeated.csv` (250 dòng) và `metrics_repeated_summary.csv`.

---

## 8. Phân tích Phụ (S1–S3)

Tất cả phân tích phụ dùng cấu hình E0 (không tuning), cùng 5 fold.

| Thí nghiệm | Mô tả | Mục đích |
|-----------|-------|---------|
| S1a | Bỏ cột Course | Đo đóng góp của Course |
| S1b | Course: OneHotEncoder(min_frequency=5) | Giảm chiều Course, hạng mục hiếm gộp tự động trong fold |
| S2 | class_weight='balanced' | Bù trọng số cho lớp thiểu số (Yes) |
| S3 | +Anxiety, +Panic attack | **Khác khung nghiên cứu chính; chồng lấn khái niệm; không so sánh trực tiếp** |

**Quy tắc:** Không bao giờ thay kết quả chính (E0) bằng cấu hình phụ cho điểm cao hơn.

Hình: `results/ml/figures/06_sensitivity.png`.
Chi tiết: `results/ml/tables/metrics_summary.csv` (experiment=S1a_drop_course, S1b_course_min_freq5, S2_class_weight_balanced, S3_plus_anxiety_panic).

---

## 9. Phân tích Sai số

*(Xem chi tiết tại notebook 05 và `results/ml/tables/error_*.csv`)*

### 9.1 Marital status (Phân tích bắt buộc — D6)

Toàn bộ **16/16 người có Marital status = Yes đều có nhãn Yes** trong mẫu.
Quy tắc đơn giản "đoán Yes nếu Marital=Yes" (chỉ tham chiếu, không phải mô hình ML):
- **Precision = 1.000, Recall = 0.457** (toàn mẫu)

**Nhận xét:** Marital status có liên hệ rất mạnh với nhãn trong mẫu. Mô hình sẽ dựa mạnh vào feature này. Chưa đủ căn cứ để suy rộng. Không diễn giải nhân quả.

### 9.2 FN và FP

Bảng FN/FP chi tiết: `error_fn_<model>.csv`, `error_fp_<model>.csv`.
FN (False Negative) là ca Yes bị bỏ sót — vấn đề chính khi Recall chỉ ≈ 43–54%.

### 9.3 Tỷ lệ sai theo nhóm

Tỷ lệ sai theo Marital status, CGPA, năm học, giới tính, nhóm tuổi (18–19, 20–22, 23–24).
Chi tiết: `error_by_group.csv`. Các nhóm có n < 10 rất bất ổn — không kết luận.

Hình: `results/ml/figures/07_error_rate_by_group.png`.

### 9.4 Dòng bị mọi mô hình sai

Chi tiết: `error_consistent.csv`. Đối chiếu với 5 dòng gần trùng vector feature để xem nhãn có thể mâu thuẫn giữa các dòng giống nhau.

### 9.5 Tầm quan trọng feature

- **Permutation Importance** (trên test fold, Course là 1 cột nguyên): xem `error_permutation_importance.csv`.
- **Hệ số LR** (fit trên toàn bộ dữ liệu — chỉ minh họa): `error_lr_coefficients.csv`.
- **Cây DT** (fit trên toàn bộ dữ liệu — chỉ minh họa): xem notebook 05.

> Không diễn giải nhân quả. Course có nhiều hạng mục hiếm nên dùng permutation thay vì impurity của RF.

Hình: `results/ml/figures/08_permutation_importance.png`, `09_lr_coefficients.png`.

---

## 10. Thảo luận theo Câu hỏi Nghiên cứu

### RQ1: Liệu 6 feature có cung cấp tín hiệu dự đoán so với baseline Dummy?

Tất cả 4 mô hình ML đều có F1(Yes) trung bình cao hơn Dummy (0.000) trong mọi fold (`n_folds_better=5/5`). Tuy nhiên:
- Marital status có liên hệ rất mạnh với nhãn trong mẫu (16/16 Yes). Tín hiệu này có thể không tổng quát hóa được.
- Cỡ mẫu n=101, mỗi fold chỉ 7 ca Yes → ước lượng rất bất ổn.
- Chưa đủ căn cứ để kết luận về khả năng tổng quát hóa.

### RQ2: Mô hình nào có liên hệ tốt nhất trong mẫu?

Theo quy tắc (mean_diff > std_diff VÀ n_folds_better ≥ 4/5), so sánh giữa các mô hình ML:
LR vs SVM: mean_diff nhỏ (<0.01), **chưa đủ bằng chứng** để kết luận LR tốt hơn SVM.
LR/SVM vs DT/RF: mean_diff > std_diff nhưng cần kiểm tra từng cặp (xem `paired_differences.csv`).

Nhìn chung: trong mẫu này, LR và SVM có F1(Yes) trung bình cao hơn DT và RF, nhưng sự khác biệt chưa đủ lớn so với độ dao động.

### RQ3: Thứ hạng theo F1(Yes) và Recall(Yes) có khác nhau không?

Trong E0: LR = SVM về Recall(Yes) (0.543); LR > SVM về F1(Yes) (0.666 vs 0.659). Thứ hạng tổng thể nhất quán: LR ≥ SVM > DT = RF > Dummy. Không có sự đảo ngược thứ hạng đáng kể trong mẫu này.

### RQ4: Các hạn chế về mẫu và dữ liệu

- **Cỡ mẫu:** n=101, chỉ ~7 ca Yes mỗi fold → ước lượng metric rất bất ổn.
- **Nhãn tự khai báo:** không phải chẩn đoán lâm sàng.
- **5 dòng gần trùng vector feature:** nhãn có thể mâu thuẫn, gây khó khăn cho mô hình.
- **Course:** 48 giá trị, 38 chỉ xuất hiện 1 lần → số chiều one-hot thay đổi theo fold.
- **Marital status:** feature quá mạnh trong mẫu này; kết quả có thể không tổng quát.

---

## 11. Hạn chế và Hướng Phát triển

**Hạn chế:**
- Cỡ mẫu quá nhỏ (n=101) để rút ra kết luận tổng quát.
- Nhãn tự khai báo có thể không nhất quán hoặc thiên lệch.
- Marital status quá mạnh trong mẫu → mô hình có thể phụ thuộc vào 1 feature.
- Hạng mục hiếm trong Course làm không gian feature không ổn định giữa các fold.

**Hướng phát triển:**
- Mã hóa thứ tự cho Year (1<2<3<4) và CGPA (theo thứ tự GPA) thay vì one-hot.
- Chuẩn hóa tên Course có căn cứ (theo danh sách ngành chính thức).
- Xác nhận trên bộ dữ liệu độc lập lớn hơn.
- Phân tích sâu hơn về ảnh hưởng của Marital status.

---

## 12. Tái lập

```bash
# 1. Cài đặt môi trường
pip install -r requirements.txt

# 2. Chạy tất cả thí nghiệm (xóa kết quả cũ và chạy lại từ đầu)
python src/train.py --experiments all

# 3. Chạy kiểm tra chống rò rỉ
jupyter nbconvert --to notebook --execute --inplace notebooks/03_pipeline_leakage_check.ipynb

# 4. Tạo hình và phân tích
jupyter nbconvert --to notebook --execute --inplace notebooks/04_model_comparison.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_error_analysis.ipynb
```

**Seed:** 42. **Kết quả:** `results/ml/tables/` và `results/ml/figures/`.
**Cấu hình đầy đủ (phiên bản thư viện, tham số):** `results/ml/tables/run_config.json`.

---

## 13. Phụ lục — Thông tin Chi tiết Mỗi Lần Chạy

| Mục | Giá trị |
|-----|---------|
| Dữ liệu | Student Mental Health (Kaggle), `data/raw/Student Mental health.csv` |
| Target | `Do you have Depression?` (Yes=1, No=0); nhãn dương = Yes |
| Feature | 6 cột: gender, Age, course, year, CGPA, marital status |
| Tiền xử lý | Median impute (Age) + OneHotEncoder trong Pipeline; fit trên train fold |
| Đánh giá | StratifiedKFold(5, shuffle, seed=42); F1(Yes) chính; nested CV cho tuning |
| Mô hình | Dummy, LogisticRegression, DecisionTree, RandomForest, SVM |
| Metric | accuracy, precision_yes, recall_yes, f1_yes, roc_auc, pr_auc |
| Môi trường | Python 3.14.0, scikit-learn 1.9.1, pandas 3.0.5, numpy 2.5.3 |
| Kết quả | `results/ml/tables/metrics_summary.csv` |
| Giới hạn | n=101; nhãn tự khai báo; không phải công cụ chẩn đoán |

---

## Tài liệu tham khảo

*(TV2 bổ sung)*
