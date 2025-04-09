# Ứng dụng phương pháp học liên kết kết hợp với trí tuệ nhân tạo đa thể thức để chuẩn đoán bệnh ung thư trong y học

Sinh viên thực hiện:
- Đinh Huỳnh Gia Bảo – 22520101 (MMTT2022.1)
- Trần Gia Bảo – 22520117 (MMTT2022.1)

Cán bộ hướng dẫn: 
- ThS. Nguyễn Khánh Thuật 

Đây là đồ án chuyên ngành (NT114.P21 - VN(ĐA)) với mục đích triển khai một hệ thống học liên kết (Federated Learning) kết hợp với trí tuệ nhân tạo đa phương thức (Multimodal AI) cho bài toán chẩn đoán ung thư trong y học.

## Giới thiệu

### Tình hình

Ung thư là một trong những căn bệnh nguy hiểm nhất và là nguyên nhân gây tử vong hàng đầu trên thế giới. Theo Tổ chức Y tế Thế giới (WHO), đây là nguyên nhân tử vong đứng thứ hai toàn cầu, với khoảng 9,6 triệu ca tử vong vào năm 2018, tương đương cứ 6 người thì có 1 người tử vong vì ung thư. Tình hình ung thư tại Việt Nam đang ở mức báo động với số ca mắc và tử vong không ngừng gia tăng. Theo thống kê của GLOBOCAN (Global Cancer Observatory) năm 2022, Việt Nam ghi nhận 180.480 ca mắc mới và 120.184 ca tử vong do ung thư. Xét về tỷ lệ mắc mới, Việt Nam đứng thứ 20 tại châu Á và xếp thứ 101 trên toàn cầu, cho thấy gánh nặng bệnh tật ngày càng nghiêm trọng.

Ở nam giới, các loại ung thư phổ biến nhất gồm ung thư phổi, tuyến tiền liệt, đại trực tràng, dạ dày và gan, trong khi ở nữ giới, ung thư vú, đại trực tràng, phổi, cổ tử cung và tuyến giáp là những loại thường gặp nhất. Trong đó, tại Việt Nam hiện nay là, ở nam giới: Ung thư gan chiếm tỷ lệ cao nhất (19,7%), tiếp theo là ung thư phổi (17,7%) và ung thư dạ dày (11%); ở nữ giới: Ung thư vú chiếm tỷ lệ cao nhất (28,9%), ung thư phổi (8,7%) và ung thư đại trực tràng (8,7%).

![Thống kê ung thư](/asset/docs-img/cancer-statistic.png)

Tuy nhiên, hệ thống y tế vẫn còn đang gặp nhiều khó khăn trong việc chẩn đoán chính xác do thiếu hụt dữ liệu chất lượng, giới hạn trong chia sẻ thông tin giữa các bệnh viện và sự đa dạng trong triệu chứng của bệnh nhân.

Trong bối cảnh đó, trí tuệ nhân tạo (Artificial Intelligence - AI) đã được ứng dụng rộng rãi trong y học, đặc biệt là để hỗ trợ phát hiện và chẩn đoán ung thư. Điều đáng nói ở đây là các hệ thống AI chăm sóc sức khỏe truyền thống thường dựa vào kiến trúc AI tập trung được đặt tại đám mây hoặc trung tâm dữ liệu để phân tích và học hỏi từ dữ liệu y tế.

Giải pháp này tuy mang lại khả năng xử lý mạnh mẽ nhưng lại gặp nhiều hạn chế, như độ trễ cao do truyền tải dữ liệu thô và gặp khó khăn khi có nhu cầu mở rộng hệ thống. Hơn nữa, việc phụ thuộc vào máy chủ trung tâm hoặc bên thứ ba có thể gây ra những rủi ro nghiêm trọng về quyền riêng tư, chẳng hạn như rò rỉ thông tin người dùng và vi phạm dữ liệu. Đặc biệt, trong y tế điện tử, bảo vệ dữ liệu sức khỏe là yêu cầu cấp thiết theo các quy định như HIPAA (United States Health Insurance Portability and Accountability Act). Vì vậy, học liên kết (Federated Learning - FL) đang nổi lên như một giải pháp tiềm năng, giúp hiện thực hóa và triểm khai các ứng dụng chăm sóc sức khỏe thông minh với chi phí hiệu quả hơn, đồng thời tăng cường bảo vệ quyền riêng tư của bệnh nhân.

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