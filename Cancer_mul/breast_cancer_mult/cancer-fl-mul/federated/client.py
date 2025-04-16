# This is the client used for Federated Learning

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

class FederatedClient:
    def __init__(self, model, data_loader, client_id, lr=0.001, device='cuda'):
        """
        Lớp đại diện cho một client trong hệ thống học liên bang
        
        Args:
            model: Mô hình multimodal
            data_loader: DataLoader chứa dữ liệu của client
            client_id (str): ID của client
            lr (float): Learning rate cho quá trình huấn luyện
            device (str): Thiết bị thực hiện tính toán ('cuda' hoặc 'cpu')
        """
        self.client_id = client_id
        self.device = device
        self.model = model.to(device)
        self.data_loader = data_loader
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
        
    def train(self, epochs=1):
        """
        Huấn luyện cục bộ trên dữ liệu của client
        
        Args:
            epochs (int): Số epochs huấn luyện
            
        Returns:
            dict: Thông tin về quá trình huấn luyện
        """
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        print(f"Client {self.client_id} bắt đầu huấn luyện với {len(self.data_loader)} batches")
        
        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0
            
            for batch in self.data_loader:
                csv_features = batch['csv_features'].to(self.device)
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images, csv_features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                # Lưu kết quả dự đoán và nhãn để đánh giá hiệu suất
                predictions = outputs.detach().cpu().numpy()
                true_labels = labels.detach().cpu().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(true_labels)
                
                # In tiến độ sau mỗi 10 batches
                if batch_count % 10 == 0:
                    print(f"Client {self.client_id}, Epoch {epoch+1}/{epochs}, Batch {batch_count}/{len(self.data_loader)}, Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / len(self.data_loader)
            print(f"Client {self.client_id}, Epoch {epoch+1}/{epochs} hoàn thành, Loss: {avg_epoch_loss:.4f}")
            total_loss += avg_epoch_loss
        
        # Tính toán các metrics
        binary_predictions = np.array(all_predictions) >= 0.5
        metrics = {
            'loss': total_loss / epochs,
            'accuracy': accuracy_score(all_labels, binary_predictions),
            'precision': precision_score(all_labels, binary_predictions),
            'recall': recall_score(all_labels, binary_predictions),
            'f1': f1_score(all_labels, binary_predictions),
            'auc': roc_auc_score(all_labels, all_predictions)
        }
        
        print(f"Client {self.client_id} kết thúc huấn luyện, Metrics: {metrics}")
        return metrics
    
    def evaluate(self, test_loader):
        """
        Đánh giá mô hình trên tập kiểm tra
        
        Args:
            test_loader: DataLoader chứa dữ liệu đánh giá
            
        Returns:
            dict: Các metrics đánh giá
        """
        self.model.eval()
        all_outputs = []
        all_labels = []
        test_loss = 0
        
        with torch.no_grad():
            for batch in test_loader:
                csv_features = batch['csv_features'].to(self.device)
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images, csv_features)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                
                all_outputs.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Chuyển đổi xác suất thành nhãn dự đoán
        all_outputs = np.array(all_outputs)
        all_labels = np.array(all_labels)
        predicted_labels = (all_outputs >= 0.5).astype(int)
        
        # Tính toán các metrics
        metrics = {
            'loss': test_loss / len(test_loader),
            'accuracy': accuracy_score(all_labels, predicted_labels),
            'precision': precision_score(all_labels, predicted_labels),
            'recall': recall_score(all_labels, predicted_labels),
            'f1': f1_score(all_labels, predicted_labels),
            'auc': roc_auc_score(all_labels, all_outputs)
        }
        
        return metrics
    
    def get_model_params(self):
        """
        Trả về các tham số mô hình hiện tại
        
        Returns:
            dict: Tham số mô hình
        """
        return {k: v.cpu() for k, v in self.model.state_dict().items()}
    
    def set_model_params(self, model_params):
        """
        Cập nhật các tham số mô hình từ server
        
        Args:
            model_params (dict): Tham số mô hình mới
        """
        self.model.load_state_dict(model_params)
        self.model = self.model.to(self.device)  # Đảm bảo mô hình vẫn ở trên thiết bị đúng
