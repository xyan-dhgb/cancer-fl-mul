
# %% [markdown]
# Part 1
# Nhập các thư viện cần thiết:

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import cv2

# Deep learning libraries
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


# %%
!pip install gdown

# %%
!gdown 1MvbaK4KXWyLRJAQ8YuJQ8vrlK9bBVScA

# %%
!unzip archive.zip

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
np.random.seed(42)
tf.random.set_seed(42)

# %%
def load_csv_data(file_path):
    """
    Tải EHR data từ CSV file
    """
    print("Loading CSV data...")
    df = pd.read_csv(file_path)
    return df

# %%
#  Hiển thị thông tin dữ liệu CSV
df = load_csv_data('/content/mias_derived_info.csv')
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
df.head()

# %%
# 2.1 phân phối dữ liệu
"""
 phân tích:
Số lượng giá trị duy nhất trong mỗi cột
Kiểm tra giá trị thiếu trong mỗi cột
"""
print("\nUnique values in each column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

# %%
# 2.2  Kiểm tra các giá trị còn thiếu
print("\nMissing values in each column:")
print(df.isnull().sum())

# %%
# xem xét sự phân bố các lớp và mức độ nghiêm trọng
"""
vẽ hai biểu đồ cột:
Phân phối của các lớp chẩn đoán (NORM, CIRC, SPIC, vv.)
Phân phối mức độ nghiêm trọng (Normal, Benign, Malignant)
"""
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
class_counts = df['CLASS'].value_counts()
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title('Distribution of Classes')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
severity_counts = df['SEVERITY'].value_counts()
sns.barplot(x=severity_counts.index, y=severity_counts.values)
plt.title('Distribution of Severity')

plt.tight_layout()
plt.show()


# %%
# 2.4 trực quan hóa mật độ
"""
trực quan hóa:
Phân phối mật độ vú (A, B, C/D)
Mối quan hệ giữa lớp chẩn đoán và mức độ nghiêm trọng
"""
plt.figure(figsize=(8, 5))
density_counts = df['DENSITY'].value_counts()
sns.barplot(x=density_counts.index, y=density_counts.values)
plt.title('Distribution of Breast Density')
plt.show()

# %%
# 2.5 mối quan hệ giữa classclass và mức độ nghiêm trọng
plt.figure(figsize=(12, 6))
sns.countplot(x='CLASS', hue='SEVERITY', data=df)
plt.title('Class vs Severity')
plt.xticks(rotation=45)
plt.show()

# %%
# 3.1 Chức năng tiền xử lý dữ liệu CSV
"""
    Tiền xử lý dữ liệu: Hàm preprocess_csv_data tiền xử lý dữ liệu dạng bảng:
Tạo bản sao để tránh sửa đổi DataFrame gốc
Điền giá trị thiếu trong các cột tọa độ (X, Y, RADIUS) bằng -1
Chuyển đổi các biến phân loại thành dạng số:
Mô nền (BG): F→0, G→1, D→2
Lớp chẩn đoán (CLASS): NORM→0, CIRC→1, v.v.
Mức độ nghiêm trọng: Normal→0, Benign→1, Malignant→2
Mật độ vú: A→1, B→2, C/D→3
Thang điểm BI-RADS: 1-5
    """
def preprocess_csv_data(df):
    # Tạo một bản sao của dataframe để tránh sửa đổi bản gốc
    processed_df = df.copy()

    # Điền tọa độ còn thiếu bằng -1 (đối với trường hợp bình thường)
    for col in ['X', 'Y', 'RADIUS']:
        processed_df[col] = processed_df[col].fillna(-1)

    # Chuyển đổi các biến phân loại thành số
    # Background tissue
    bg_map = {'F': 0, 'G': 1, 'D': 2}
    processed_df['BG'] = processed_df['BG'].map(bg_map)

    # Class mapping
    class_map = {
        'NORM': 0,
        'CIRC': 1,
        'SPIC': 2,
        'ARCH': 3,
        'ASYM': 4,
        'CALC': 5,
        'MISC': 6
    }
    processed_df['CLASS_NUM'] = processed_df['CLASS'].map(class_map)

    # Severity mapping
    severity_map = {'Normal': 0, 'Benign': 1, 'Malignant': 2}
    processed_df['SEVERITY_NUM'] = processed_df['SEVERITY'].map(severity_map)

    # Density mapping
    density_map = {'A': 1, 'B': 2, 'C/D': 3}
    processed_df['DENSITY_NUM'] = processed_df['DENSITY'].map(density_map)

    # BI-RADS mapping
    birads_map = {
        'BI-RADS 1': 1,
        'BI-RADS 2': 2,
        'BI-RADS 3': 3,
        'BI-RADS 4': 4,
        'BI-RADS 5': 5
    }
    processed_df['BIRADS_NUM'] = processed_df['BI-RADS'].map(birads_map)

    return processed_df

# %%
# 3.2 Tiền xử lý data
"""
Đoạn này áp dụng hàm tiền xử lý và hiển thị kết quả
"""
processed_df = preprocess_csv_data(df)
print("Processed dataframe:")
processed_df.head()

# %%
# 3.3 Chức năng tải và xử lý trước hình ảnh
"""
Hàm load_image để xử lý hình ảnh mammogram:
Tải hình ảnh dưới dạng grayscale
Tạo hình ảnh trống nếu không tìm thấy
Thay đổi kích thước thành 224x224 pixel
Chuẩn hóa pixel (chia cho 255)
Thêm chiều kênh cho mô hình CNN
"""
def load_image(image_path, target_size=(224, 224)):
    """
    Tải và xử lý trước các X-ray image
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # giữ chỗ cho hình ảnh bị thiếu
        img = np.zeros(target_size)
    else:
        img = cv2.resize(img, target_size)

    # Chuẩn hóa hình ảnh
    img = img / 255.0

    # Thêm kích thước kênh
    img = np.expand_dims(img, axis=-1)

    return img

# %%
# 3.4 Thực hiện việcviệc chuẩn bị tập dữ liệu với cả dữ liệu hình ảnh và CSV
"""
Hàm prepare_multimodal_data chuẩn bị dữ liệu cho mô hình đa phương thức:
Trích xuất đặc trưng bảng
Chuẩn hóa dữ liệu bảng
Chuyển đổi biến mục tiêu sang dạng one-hot
Tải và xử lý tất cả hình ảnh
Chia dữ liệu thành tập huấn luyện và kiểm tra (80/20)
"""
def prepare_multimodal_data(df, image_dir, target_size=(224, 224)):
    """
    Chuẩn bị multimodal dataset để kết hợp images and CSV data
    """
    # Trích xuất các tính năng có liên quan từ CSV
    X_tabular = df[['BG', 'CLASS_NUM', 'X', 'Y', 'RADIUS', 'DENSITY_NUM', 'BIRADS_NUM']].values

    # Chuẩn hóa dữ liệu dạng bảng
    scaler = StandardScaler()
    X_tabular = scaler.fit_transform(X_tabular)

    # Biến mục tiêu
    y = df['SEVERITY_NUM'].values

    # Chuyển đổi sang danh mục
    y_cat = to_categorical(y)

    # Chuẩn bị dữ liệu hình ảnh
    X_images = []
    for ref_num in df['REFNUM'].values:
        image_path = os.path.join(image_dir, f"{ref_num}.pgm")
        img = load_image(image_path, target_size)
        X_images.append(img)

    X_images = np.array(X_images)

    # Chia tách tập dữ liệu
    X_tab_train, X_tab_test, X_img_train, X_img_test, y_train, y_test = train_test_split(
        X_tabular, X_images, y_cat, test_size=0.2, random_state=42, stratify=y_cat
    )

    return X_tab_train, X_tab_test, X_img_train, X_img_test, y_train, y_test

print("we'll assume these images would be available.")

# %%
# 4.1 Xây dựng mô hình đa phương thức (multimodal model)
"""
Hàm build_multimodal_model xây dựng một mô hình học sâu với hai nhánh:
Nhánh dữ liệu bảng:
Đầu vào: Các đặc trưng số (BG, CLASS_NUM, X, Y, v.v.)
Kiến trúc: Hai lớp Dense với dropout để tránh overfitting

Nhánh hình ảnh:
Đầu vào: Hình ảnh grayscale
Chuyển đổi từ grayscale sang RGB
Sử dụng ResNet50 đã được huấn luyện trước trên ImageNet
Đóng băng (freeze) các trọng số của mô hình gốc
Thêm các lớp Dense và Dropout sau đó

Kết hợp hai nhánh:
Ghép nối các đặc trưng từ cả hai nhánh
Thêm các lớp Dense và Dropout
Lớp đầu ra với activation softmax cho phân loại 3 lớp
"""
def build_multimodal_model(tabular_shape, image_shape, num_classes=3):
    """
    Xây dựng multimodal model data dạng bảng và images
    """
    # Thiết lập nnhánh dữ liệu dạng bảng
    tabular_input = Input(shape=(tabular_shape,), name='tabular_input')
    x_tab = Dense(64, activation='relu')(tabular_input)
    x_tab = Dropout(0.3)(x_tab)
    x_tab = Dense(32, activation='relu')(x_tab)
    x_tab = Model(inputs=tabular_input, outputs=x_tab)

    # Nhánh hình ảnh sử dụng mô hình được đào tạo trước
    image_input = Input(shape=image_shape, name='image_input')

    # Chuyển đổi greyscale sang RGB bằng cách lặp lại kênh
    x_img = tf.keras.layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1))(image_input)

    # Sử dụng ResNet được đào tạo trước với trọng số
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Đóng băng mô hình cơ sở
    base_model.trainable = False

    x_img = base_model(x_img)
    x_img = Flatten()(x_img)
    x_img = Dense(128, activation='relu')(x_img)
    x_img = Dropout(0.5)(x_img)
    x_img = Model(inputs=image_input, outputs=x_img)

    # Kết hợp tất cả các nhánh
    # Late fusion
    combined = concatenate([x_tab.output, x_img.output])

    # Các lớp phân loại cuối cùng
    # Classification
    x = Dense(64, activation='relu')(combined)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    # Tạo và biên dịch mô hình
    model = Model(inputs=[x_tab.input, x_img.input], outputs=output)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# %%
# 4.2 Thực hiện đào tạo mô hình
"""
Hàm train_model huấn luyện mô hình:
Sử dụng ImageDataGenerator cho data augmentation (xoay, dịch, zoom, lật)
Thêm early stopping để dừng sớm nếu mô hình không cải thiện
Huấn luyện mô hình với dữ liệu từ cả hai nhánh
"""
def train_model(model, X_tab_train, X_img_train, y_train, X_tab_val, X_img_val, y_val, epochs=20, batch_size=32):
    """
    Huấn luyện multimodal model
    """
    # Tăng cường dữ liệu cho hình ảnh
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    # Tạo lệnh callback để dừng sớm
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Huấn luyện model
    history = model.fit(
        [X_tab_train, X_img_train],
        y_train,
        validation_data=([X_tab_val, X_img_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )

    return model, history

# %%
# 4.3 Thực hiện đánh giá mô hình
"""
Hàm evaluate_model đánh giá hiệu suất mô hình:
Dự đoán trên tập kiểm tra
Tạo báo cáo phân loại (precision, recall, f1-score)
Vẽ ma trận nhầm lẫn
Vẽ đường cong ROC cho mỗi lớp và tính AUC
"""
def evaluate_model(model, X_tab_test, X_img_test, y_test):
    # nhận dự đoán
    y_pred_prob = model.predict([X_tab_test, X_img_test])
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # tính toán số liệu
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Benign', 'Malignant']))

    # xây dựng đồ thị ma trận kết hợp (confusion)
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Benign', 'Malignant'],
                yticklabels=['Normal', 'Benign', 'Malignant'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Vẽ đường cong ROC cho mỗi class
    plt.figure(figsize=(10, 8))
    class_names = ['Normal', 'Benign', 'Malignant']

    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.show()

    return y_pred, y_pred_prob

# %% [markdown]
# # Part 5: Huấn luyện và đánh giá mô hình

# %%
"""
Phần này thực hiện toàn bộ quy trình:
Chuẩn bị dữ liệu đa phương thức
Xây dựng mô hình
In tóm tắt cấu trúc mô hình
Huấn luyện mô hình
Đánh giá hiệu suất mô hình
"""
# Xác định thư mục hình ảnh
image_dir = '/content/MIAS'

# %%
# Chuẩn bị dữ liệu đa phương thức
X_tab_train, X_tab_test, X_img_train, X_img_test, y_train, y_test = prepare_multimodal_data(
    processed_df, image_dir
)

# %%
# xây dựng model
model = build_multimodal_model(
    tabular_shape=X_tab_train.shape[1],
    image_shape=X_img_train.shape[1:],
    num_classes=3
)

# %%
# In ra model tổng quát
model.summary()

# %%
# huấn luyện model
trained_model, history = train_model(
    model,
    X_tab_train, X_img_train, y_train,
    X_tab_test, X_img_test, y_test,
    epochs=20
)

# %%
# đánh giá model
y_pred, y_pred_prob = evaluate_model(trained_model, X_tab_test, X_img_test, y_test)

# %% [markdown]
# # 6. Xây dựng ví dụ cho khởi tạo bệnh nhân mới

# %%
"""
Hàm predict_cancer_diagnosis thực hiện dự đoán cho bệnh nhân mới:
Tiền xử lý dữ liệu bảng của bệnh nhân
Tải và tiền xử lý hình ảnh mammogram
Thực hiện dự đoán bằng mô hình đã huấn luyện
Trả về chẩn đoán (Normal, Benign, Malignant)
Cung cấp độ tin cậy và xác suất cho từng lớp
"""

def predict_cancer_diagnosis(model, csv_data, image_path):
    """
     Thực hiện dự đoán bằng cả dữ liệu CSV data and X-ray image
    """
    # Xử lý dữ liệu CSV
    processed_csv = preprocess_csv_data(pd.DataFrame([csv_data]))

    # Thêm cột 'SEVERITY_NUM' nếu nó không tồn tại (fix bug) và đặt thành 0 (Bình thường)
    if 'SEVERITY_NUM' not in processed_csv.columns:
        processed_csv['SEVERITY_NUM'] = 0

    X_tab = processed_csv[['BG', 'CLASS_NUM', 'X', 'Y', 'RADIUS', 'DENSITY_NUM', 'BIRADS_NUM']].values

    # Tải và xử lý hình ảnh
    X_img = np.expand_dims(load_image(image_path), axis=0)

    # thực hiện dự đoán
    prediction = model.predict([X_tab, X_img])
    pred_class = np.argmax(prediction, axis=1)[0]

    # Map back lại cái chuẩn đoán
    diagnosis_map = {0: 'Normal', 1: 'Benign', 2: 'Malignant'}
    diagnosis = diagnosis_map[pred_class]

    # Tính toán độ tin cậy
    confidence = prediction[0][pred_class] * 100

    return {
        'diagnosis': diagnosis,
        'confidence': confidence,
        'probabilities': {
            'Normal': float(prediction[0][0] * 100),
            'Benign': float(prediction[0][1] * 100),
            'Malignant': float(prediction[0][2] * 100)
        }
    }

# %%
# Xây dựng dữ liệu bệnh nhân mẫu
# Thêm cột SEVERITY
#minh họa cách sử dụng hàm dự đoán với một bệnh nhân mẫu
# và hiển thị kết quả.
new_patient = {
    'REFNUM': 'new_patient_001',
    'BG': 'G',
    'CLASS': 'CIRC',
    'X': 520,
    'Y': 380,
    'RADIUS': 45,
    'DENSITY': 'B',
    'BI-RADS': 'BI-RADS 3',
    'SEVERITY': 'Benign'  # Có thể cần điều chỉnh giá trị này dựa trên đầu vào dự kiến
}

# %%
# đường đi đến cái mammogram của bệnh nhân
patient_image_path = '/content/MIAS'

# %%
# thực hiện chuẩn đoán
diagnosis_result = predict_cancer_diagnosis(trained_model, new_patient, patient_image_path)
print(f"Diagnosis: {diagnosis_result['diagnosis']}")
print(f"Confidence: {diagnosis_result['confidence']:.2f}%")
print("Probabilities:")
for class_name, prob in diagnosis_result['probabilities'].items():
    print(f"  {class_name}: {prob:.2f}%")

# %% [markdown]
# #Part 7: Phân tích tầm quan trọng đặc trưng của bảng

# %%
!pip install eli5

# %%
"""
Phần này phân tích tầm quan trọng của các đặc trưng bảng:
Tạo lớp bọc để tương thích với API của scikit-learn
Sử dụng PermutationImportance để xác định đặc trưng nào quan trọng nhất
Hiển thị trọng số của từng đặc trưng
"""

# %%
# Phân tích tầm quan trọng của tính năng (đối với dữ liệu dạng bảng)
from eli5.sklearn import PermutationImportance

# %%
# Tạo một hàm bao bọc chỉ sử dụng dữ liệu dạng bảng
def predict_tabular_wrapper(X):
    # Tạo dữ liệu hình ảnh giả có hình dạng phù hợp
    dummy_img = np.zeros((X.shape[0], 224, 224, 1))
    return model.predict([X, dummy_img])

# %%
# Tính toán tầm quan trọng của hoán vị
# Gói mô hình trong một ước lượng giả
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score  # Đưa ra độ chuẩn xác (accuracy_score)

class ModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        return self

    def predict(self, X):
        # Tạo dữ liệu hình ảnh giả có hình dạng phù hợp
        dummy_img = np.zeros((X.shape[0], 224, 224, 1))
        return np.argmax(self.model.predict([X, dummy_img]), axis=1)

    def score(self, X, y):
         # Chuyển đổi y sang định dạng đa lớp nếu cần thiết
        if y.ndim == 2 and y.shape[1] > 1:  # Kiểm tra xem y có được mã hóa one-hot không
            y = np.argmax(y, axis=1)

        # Dự đoán bằng cách sử dụng wrapper
        y_pred = self.predict(X)

        # Tính toán độ chính xác
        return accuracy_score(y, y_pred)


# Bao bọc mô hình gốc
wrapped_model = ModelWrapper(model)

# Tính toán tầm quan trọng của hoán vị bằng cách sử dụng mô hình được gói
perm = PermutationImportance(wrapped_model, random_state=42).fit(
    X_tab_test, y_test
)

# %%
# Hiển thị tầm quan trọng của tính năng
import eli5
feature_names = ['BG', 'CLASS', 'X', 'Y', 'RADIUS', 'DENSITY', 'BI-RADS']
eli5.show_weights(perm, feature_names=feature_names)

# %% [markdown]
# # 8. Chức năng trực quan hóa dành cho mục đích sử dụng lâm sàng

# %%

# Trực quan hóa chẩn đoán
"""
Hàm visualize_diagnosis tạo trực quan hóa để hiển thị kết quả chẩn đoán:
Hiển thị hình ảnh mammogram gốc
Nếu có heatmap, hiển thị các vùng đáng ngờ với overlay màu
Thêm thông tin chẩn đoán và độ tin cậy
"""
def visualize_diagnosis(image_path, prediction_result, heatmap=None):
    # tải hình ảnh lên
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (512, 512))

    plt.figure(figsize=(14, 7))

    # hiển thị ảnh gốc lên
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Mammogram')
    plt.axis('off')

    # Hiển thị hình ảnh với các vùng được tô sáng (có bản đồ nhiệt)
    plt.subplot(1, 2, 2)
    if heatmap is not None:
        # Áp dụng bản đồ nhiệt
        heatmap = cv2.resize(heatmap, (512, 512))
        plt.imshow(img, cmap='gray')
        plt.imshow(heatmap, cmap='jet', alpha=0.4)
        plt.title('Suspicious Areas')
    else:
        plt.imshow(img, cmap='gray')
        plt.title('No Heatmap Available')
    plt.axis('off')

    # Thêm thông tin chẩn đoán
    diagnosis = prediction_result['diagnosis']
    confidence = prediction_result['confidence']
    plt.figtext(0.5, 0.01, f"Diagnosis: {diagnosis} (Confidence: {confidence:.2f}%)",
                ha="center", fontsize=14,
                bbox={"facecolor":"white", "alpha":0.8, "pad":5})

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()

# %%
 # hình ảnh thực tế đến ảnh chụp quang tuyến vú
"""
minh họa cách sử dụng hàm trực quan hóa:
Sử dụng một đường dẫn hình ảnh cụ thể
Tạo một từ điển kết quả dự đoán mẫu
Hiển thị trực quan hóa không có heatmap
Tạo một heatmap mẫu để mô phỏng vùng đáng ngờ
Hiển thị trực quan hóa với heatmap
"""
image_path = "/content/MIAS/mdb001.png"

# Tạo một dictionary kết quả dự đoán mẫu phù hợp với định dạng mong đợi
prediction_result = {
    'diagnosis': 'Benign',
    'confidence': 85.7,
    'probabilities': {
        'Normal': 10.2,
        'Benign': 85.7,
        'Malignant': 4.1
    }
}

# Gọi hàm (không có bản đồ nhiệt)
visualize_diagnosis(image_path, prediction_result)

# Ngoài ra, bản đồ nhiệt thường xuất phát từ phân tích mô hình
# Có thể tạo một bản đồ nhiệt đơn giản để trình diễn:
import numpy as np
heatmap = np.zeros((512, 512))
# Tạo một "điểm nóng" trong bản đồ nhiệt
heatmap[200:300, 200:300] = np.linspace(0, 1, 100)[:, np.newaxis] * np.linspace(0, 1, 100)[np.newaxis, :]

# Sau đó gọi với bản đồ nhiệt
visualize_diagnosis(image_path, prediction_result, heatmap)


