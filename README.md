# Ứng dụng phương pháp học liên kết kết hợp với trí tuệ nhân tạo đa thể thức để chuẩn đoán bệnh ung thư trong y học

Sinh viên thực hiện:
- Đinh Huỳnh Gia Bảo – 22520101 (MMTT2022.1)
- Trần Gia Bảo – 22520117 (MMTT2022.1)

Cán bộ hướng dẫn: 
- ThS. Nguyễn Khánh Thuật 

Đây là đồ án chuyên ngành (NT114.P21 - VN(ĐA)) với mục đích triển khai một hệ thống học liên kết (Federated Learning) kết hợp với trí tuệ nhân tạo đa phương thức (Multimodal AI) cho bài toán chẩn đoán ung thư trong y học.

## Giới thiệu

### Federated Learning (FL) là gì?

Federated Learning (FL) là một phương pháp học máy cho phép huấn luyện một mô hình chung trên nhiều thiết bị hoặc máy chủ chứa dữ liệu cục bộ mà không cần chia sẻ dữ liệu trực tiếp. 

Trong đồ án này, chúng tôi áp dụng FL để xây dựng một mô hình chẩn đoán ung thư sử dụng dữ liệu y tế đa phương thức từ Kaggle (một trang web chứa nhiều dataset uy tín) và các tổ chức y tế trên toàn thế giới, bảo vệ quyền riêng tư của bệnh nhân.

### Multimoda là gì?

Là một dạng trí tuệ nhân tạo có khả năng xử lý đồng thời nhiều dạng dữ liệu và thực hiện các tác vụ đa phương thức khác nhau, chẳng hạn như hình ảnh, âm thanh và văn bản, …

Quá trình kết hợp các phương thức này bắt đầu bằng nhiều mô hình đơn phương thức (Unimodal).

## Mục tiêu

Mục tiêu chính mà chúng em hướng đến đó chính là xây dựng một hệ thống AI có khả năng chẩn đoán bệnh ung thư bằng cách kết hợp học liên kết và trí tuệ nhân tạo đa thể thức nhằm tối ưu hóa quá trình truyền tải dữ liệu giữa các cơ sở y tế. Cụ thể:

- Phát triển một mô hình Multimodal AI có khả năng phân tích đồng hình ảnh y tế và hồ sơ sức khỏe điện tử (EHR - Electronic Health Record) để chẩn đoán chính xác và kịp thời. 

- Triển khai hệ thống FL đào tạo các mô hình dựa trên dữ liệu từ nhiều bệnh viện, từ đó bảo vệ quyền riêng tư của bệnh nhân.

- Phân tích hiệu suất của hệ thống về độ chính xác, hiệu quả truyền dữ liệu và mức độ an ninh so với các phương pháp truyền thông tin truyền thống.

Đồ án này tập trung vào các loại ung thư phổ biến như **ung thư phổi, ung thư vú và ung thư gan**, bởi vì đây là những bệnh lý có tỷ lệ mắc cao và có dữ liệu y khoa phong phú để thử nghiệm

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
```bash
federated_cancer_diagnosis/
│
├── asset/                           # Hình ảnh minh họa
│   └── docs-img/
│       └── cancer-statistic.png
├── config/                          # Cấu hình tham số cho FL
│   └── fl_config.py
├── data/                            # Xử lý và nạp dữ liệu
│   ├── data_loader.py               # Tải và phân chia dữ liệu cho FL
│   ├── preprocessing.py             # Tiền xử lý dữ liệu
├── docs/                            # Tài liệu tham khảo
├── federated/                       # Thành phần chính của FL
│   ├── __init__.py                     
│   ├── client.py                    # Hàm huấn luyện local client
│   ├── server.py                    # Tổng hợp mô hình và điều phối vòng lặp FL
│   └── trainer.py                   # Vòng lặp huấn luyện liên kết (global loop)
├── models/                          # Kiến trúc mạng AI
├── utils/                           # Tiện ích, đánh giá và hiển thị
│   ├── __init__.py
│   ├── dataset.py                   # Chia tập train/test/val, xử lý multimodal
│   ├── metrics.py                   # Accuracy, precision, recall, F1-score,...
│   └── visualization.py             # Biểu đồ loss/accuracy, confusion matrix,...
├── logs/                            # Lưu kết quả huấn luyện
├── notebook/                        # Thử nghiệm, trình bày trong báo cáo
│   └── explore_multimodal.ipynb     # Khám phá, visualize ảnh và csv
├── evaluate.py                      # Đánh giá mô hình sau FL
├── main.py                          # Chạy toàn bộ hệ thống FL
├── requirements.txt                 # Thư viện cần cài đặt
├── .gitignore                       # Các file git bỏ qua
└── README.md                        # Giới thiệu tổng quan project

```