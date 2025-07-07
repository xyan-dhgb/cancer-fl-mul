# Giới thiệu quá trình thực hiện multimodal 
## Mô tả về dataset: 
### Tên bệnh: Ung thu da
### Nguồn: https://www.kaggle.com/datasets/mahdavi1202/skin-cancer


Giới thiệu quá trình thực hiện multimodal
Mô tả về dataset:
Tên bệnh: Ung thư da (Skin Cancer)
Nguồn: Kaggle - Skin Cancer: Malignant vs. Benign
Tham khảo từ: Mahdavi et al., Skin Cancer Image and Metadata Dataset
File images:
Bộ dữ liệu bao gồm ảnh da người có tổn thương được phân loại là lành tính (benign) hoặc ác tính (malignant). Cụ thể:

Hình ảnh tổn thương da với kích thước đồng nhất (ví dụ: 224x224 pixel)

Các loại ung thư da khác nhau được thể hiện trong ảnh, bao gồm:

Basal Cell Carcinoma

Squamous Cell Carcinoma

Melanoma

Benign Keratosis

Dermatofibroma

v.v.

Ảnh có định dạng .jpg trong thư mục images/

Ảnh được đặt tên bằng image_id và ánh xạ tới dữ liệu thuộc tính trong file .csv tương ứng

File CSV:
Tên file: metadata.csv hoặc tương tự

Mô tả các cột chính:
image_id: Tên file ảnh (không bao gồm đuôi .jpg)

label: Nhãn bệnh (Benign hoặc Malignant)

age: Tuổi bệnh nhân

sex: Giới tính (male / female)

anatom_site: Vị trí vùng da có tổn thương (head/neck, torso, lower extremity, v.v.)

diagnosis: Tên chẩn đoán cụ thể (Melanoma, Nevus, Seborrheic Keratosis, v.v.)

Phân tích dữ liệu từ file CSV:
Biểu đồ phân phối nhãn bệnh:
Hiển thị tỷ lệ ảnh có nhãn Benign và Malignant

Phân phối giới tính và độ tuổi bệnh nhân:
Histogram hoặc Boxplot tuổi bệnh nhân

Biểu đồ cột giới tính (male, female)

Vị trí giải phẫu thường gặp:
Biểu đồ thể hiện số lượng ca ung thư da theo từng vị trí (anatom_site)

Tương quan giữa diagnosis và label:
Biểu đồ trực quan hóa thể hiện mối quan hệ giữa diagnosis và nhãn bệnh (label)

Một số hình ảnh ví dụ từ dataset:
  Hình ảnh bệnh nhân với nhãn Malignant:
  Hình ảnh bệnh nhân với nhãn Benign:
(Chèn hình ảnh minh họa từ tập dữ liệu)

Training multimodal
Dữ liệu từ ảnh (image_id.jpg) sẽ được kết hợp với các đặc trưng phi hình ảnh (age, sex, anatom_site, ...) để xây dựng mô hình học đa phương thức (multimodal learning).

Quy trình huấn luyện bao gồm:

Tiền xử lý ảnh và đặc trưng văn bản

Xây dựng mô hình đa nhánh (CNN + MLP)

Đánh giá độ chính xác (accuracy, ROC AUC, v.v.)

 Link notebook demo: (bạn có thể cập nhật link Google Colab hoặc Drive nếu có)