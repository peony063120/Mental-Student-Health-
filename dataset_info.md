# Tài liệu mô tả bộ dữ liệu (Data Dictionary)

**Đề tài:** Dự đoán tự khai báo trầm cảm ở sinh viên bằng học máy cổ điển
*Student Depression Self-report Prediction Using Classical Machine Learning*

**Bộ dữ liệu:** Student Mental Health (Kaggle)

---

## 1. Thông tin chung

| Mục | Nội dung |
|---|---|
| Nguồn gốc | Kaggle — bộ dữ liệu khảo sát sức khỏe tâm thần của sinh viên, do người dùng **shariful07** công bố |
| URL | https://www.kaggle.com/datasets/shariful07/student-mental-health |
| Định dạng file | CSV, phân tách bằng dấu phẩy (`,`), có dòng tiêu đề |
| Quy mô | **101 dòng × 11 cột** |
| Biến mục tiêu | `Do you have Depression?` — **No: 66** (65,35%), **Yes: 35** (34,65%) |
| Dữ liệu thiếu | **1 ô thiếu** duy nhất tại cột `Age` (100/101 giá trị hợp lệ); các cột còn lại không thiếu |
| Dòng trùng lặp hoàn toàn | **0** dòng |

Nhận xét ban đầu: tập dữ liệu có quy mô nhỏ (101 quan sát) và biến mục tiêu mất cân bằng nhẹ (tỷ lệ No/Yes ≈ 1,89 : 1), cần lưu ý khi chia tập huấn luyện/kiểm định và lựa chọn chỉ số đánh giá (ưu tiên F1, ROC-AUC thay vì chỉ dựa vào Accuracy).

---

## 2. Bảng mô tả chi tiết 11 thuộc tính

| STT | Tên cột gốc | Kiểu dữ liệu | Ý nghĩa | Vai trò trong nghiên cứu | Ghi chú bất thường |
|---|---|---|---|---|---|
| 1 | `Timestamp` | `object` (chuỗi) | Thời điểm ghi nhận phiếu trả lời khảo sát | **Loại khỏi mô hình chính** | Không mang ý nghĩa dự báo; định dạng ngày–giờ không đồng nhất (ví dụ `8/7/2020 12:02` so với `13/07/2020 19:08:32`). |
| 2 | `Choose your gender` | `object` (chuỗi) | Giới tính của sinh viên | Biến đầu vào chính (Feature) | Không bất thường: `Female` 75, `Male` 26; không thiếu giá trị. |
| 3 | `Age` | `float64` | Tuổi của sinh viên | Biến đầu vào chính (Feature) | Có **1 giá trị thiếu** nên pandas đọc thành `float64` (bản chất là số nguyên); khoảng giá trị 18–24, trung bình 20,53. |
| 4 | `What is your course?` | `object` (chuỗi) | Ngành/chương trình học của sinh viên | Biến đầu vào chính (Feature) | **49 giá trị duy nhất**; nhiều biến thể viết hoa/thường, lỗi chính tả và khoảng trắng thừa (xem chi tiết mục 3.3). |
| 5 | `Your current year of Study` | `object` (chuỗi) | Năm học hiện tại (năm 1 – năm 4) | Biến đầu vào chính (Feature) | Viết hoa không nhất quán giữa `Year`/`year`; chỉ tồn tại dạng `year 4` (xem chi tiết mục 3.1). |
| 6 | `What is your CGPA?` | `object` (chuỗi) | Khoảng điểm trung bình tích lũy (CGPA) | Biến đầu vào chính (Feature) | Tồn tại giá trị thừa khoảng trắng cuối `3.50 - 4.00 `; quy ước viết khoảng điểm không thống nhất (xem chi tiết mục 3.2). |
| 7 | `Marital status` | `object` (chuỗi) | Tình trạng hôn nhân | Biến đầu vào chính (Feature) | Không bất thường: `No` 85, `Yes` 16; không thiếu giá trị. |
| 8 | `Do you have Depression?` | `object` (chuỗi) | Tự khai báo có bị trầm cảm hay không | **Biến mục tiêu (Target)** | `No` 66 (65,35%), `Yes` 35 (34,65%); mất cân bằng nhẹ. |
| 9 | `Do you have Anxiety?` | `object` (chuỗi) | Tự khai báo có bị lo âu hay không | **Loại khỏi mô hình chính** | Triệu chứng đồng diễn (co-morbid) với trầm cảm; nếu đưa vào làm đặc trưng sẽ gây rủi ro rò rỉ thông tin về nhãn. |
| 10 | `Do you have Panic attack?` | `object` (chuỗi) | Tự khai báo có bị cơn hoảng loạn hay không | **Loại khỏi mô hình chính** | Tương tự `Do you have Anxiety?` — triệu chứng đồng diễn, tiềm ẩn rò rỉ dữ liệu. |
| 11 | `Did you seek any specialist for a treatment?` | `object` (chuỗi) | Đã từng tìm chuyên gia để điều trị hay chưa | **Loại khỏi mô hình chính** | Thông tin phát sinh *sau* khi đã có vấn đề; là biến đại diện (proxy) rất mạnh của nhãn và gây **rò rỉ dữ liệu** (target leakage) nếu dùng để dự báo. |

---

## 3. Tóm tắt các lỗi định dạng phát hiện được

### 3.1. `Your current year of Study` — viết hoa không nhất quán

Cột có 7 giá trị duy nhất do cách viết hoa chữ cái đầu không thống nhất giữa `Year` và `year`:

| Giá trị gốc | Số lượng |
|---|---|
| `year 1` | 41 |
| `Year 3` | 19 |
| `Year 2` | 16 |
| `year 2` | 10 |
| `year 4` | 8 |
| `year 3` | 5 |
| `Year 1` | 2 |

Lưu ý: dạng chuẩn hóa dự kiến là `Year 1`, `Year 2`, `Year 3`, `Year 4`. Hiện tại **không tồn tại** dạng `Year 4` (chỉ có `year 4`), cho thấy giá trị được nhập liệu không tuân theo một quy ước duy nhất.

### 3.2. `What is your CGPA?` — khoảng trắng thừa và quy ước viết khoảng

Cột có 6 giá trị duy nhất thay vì 5 vì một giá trị thừa dấu cách ở cuối:

| Giá trị gốc | Số lượng | Ghi chú |
|---|---|---|
| `3.50 - 4.00` | 47 | |
| `3.00 - 3.49` | 43 | |
| `2.50 - 2.99` | 4 | |
| `0 - 1.99` | 4 | viết `0` thay vì `0.00` |
| `2.00 - 2.49` | 2 | |
| `3.50 - 4.00 ` | 1 | **thừa khoảng trắng cuối** |

Ngoài ra, khoảng điểm đầu tiên được viết là `0 - 1.99` (không có hai chữ số thập phân) trong khi các khoảng còn lại dùng hai chữ số (`2.00 - 2.49`, …), dẫn đến quy ước hiển thị không thống nhất.

### 3.3. `What is your course?` — biến thể chính tả, viết hoa và khoảng trắng

Cột có **49 giá trị duy nhất**, trong đó nhiều giá trị thực chất cùng một ngành học nhưng được ghi theo nhiều cách khác nhau:

- **Viết hoa/thường khác nhau:** `Engineering` / `Engine` / `engin`; `KIRKHS` / `Kirkhs` / `Koe` / `KOE` / `koe`; `BENL` / `Benl`; `Psychology` / `psychology`; `Laws` / `Law`.
- **Ngôn ngữ khác nhau cho cùng ngành:** `Islamic education` / `Islamic Education` / `Pendidikan islam` / `Pendidikan Islam` / `Pendidikan Islam `.
- **Lỗi chính tả:** `Mathemathics` (đúng: *Mathematics*), `Malcom` (nghi vấn lỗi gõ).
- **Thừa khoảng trắng cuối (7 giá trị):** `Nursing `, `Accounting `, `Communication `, `Human Sciences `, `Usuluddin `, `Fiqh fatwa `, `Pendidikan Islam `.
- **Viết tắt/viết tắt không thống nhất:** `BIT`, `BCS`, `IT`, `CTS`, `ENM`, `ALA`, `KOE`, `MHSC`, `TAASL`, `KENMS`, `DIPLOMA TESL` xen lẫn với tên đầy đủ như `Biomedical science`, `Business Administration`, `Diploma Nursing`, `Human Resources`, `Marine science`.

Các bất thường trên khiến một biến categorical có số nhãn (cardinality) phồng lên tới 49, cần được chuẩn hóa (gộp nhóm ngành, thống nhất chữ hoa/thường, cắt khoảng trắng, sửa chính tả) trước khi đưa vào mô hình.
