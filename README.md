# Ứng dụng phương pháp học liên kết kết hợp với trí tuệ nhân tạo đa thể thức để chuẩn đoán bệnh ung thư trong y học

Sinh viên thực hiện:
- Đinh Huỳnh Gia Bảo – 22520101 (MMTT2022.1)
- Trần Gia Bảo – 22520117 (MMTT2022.1)


Đây là đồ án chuyên ngành (NT114.P21 - VN(ĐA)) với mục đích triển khai một hệ thống học liên kết (Federated Learning) kết hợp với trí tuệ nhân tạo đa phương thức (Multimodal AI) cho bài toán chẩn đoán ung thư trong y học.

## Giới thiệu

### Federated Learning (FL) là gì?

Federated Learning (FL) là một phương pháp học máy cho phép huấn luyện một mô hình chung trên nhiều thiết bị hoặc máy chủ chứa dữ liệu cục bộ mà không cần chia sẻ dữ liệu trực tiếp. 

Trong đồ án này, chúng tôi áp dụng FL để xây dựng một mô hình chẩn đoán ung thư sử dụng dữ liệu y tế đa phương thức từ Kaggle (một trang web chứa nhiều dataset uy tín) và các tổ chức y tế trên toàn thế giới, bảo vệ quyền riêng tư của bệnh nhân.

## Đặc điểm
- Triển khai Federated Learning với thư viện TensorFlow Federated (TFF)
- Mô hình đa phương thức kết hợp dữ liệu hình ảnh X-quang và dữ liệu bệnh nhân (tập tin CSV)
- Bảo vệ quyền riêng tư dữ liệu bệnh nhân

## Yêu cầu
- Ngôn ngữ lập trình: Python
- Môi trường: VSCode (hoặc Pycharm), Jupyter Notebook
- Thư viện:
    + Đối với Federated Learning:
        + flower >= 2.0.0
        + tensorflow-federated >= 2.0.0

## Cài đặt
```bash
pip install -r requirements.txt
```

## Sử dụng
```bash
python main.py
```

## Cấu trúc dự án
- `config/`: Cấu hình dự án
- `data/`: Quản lý và tiền xử lý dữ liệu
- `docs/`: Tài liệu, ghi chú liên quan đến đồ án
- `federated/`: Triển khai học liên bang
- `log/`: Lưu trũ lịch sử trong quá trình khởi chạy để tiện lợi trong việc debug
- `models/`: Định nghĩa mô hình
- `notebook/`: Triển khai code dưới dạng Jupyter Notebook
- `utils/`: Tiện ích, đánh giá và hiển thị
-  `evaluate.py`: Đánh giá mô hình
- `main.py`: Điểm khởi đầu chương trình
- `requirement.txt`: Thông tin các thư viện cần thiết của đồ án