import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import copy

class FederatedServer:
    def __init__(self, global_model, device='cuda'):
        """
        Lớp đại diện cho server trong hệ thống học liên bang
        
        Args:
            global_model: Mô hình toàn cục
            device (str): Thiết bị thực hiện tính toán ('cuda' hoặc 'cpu')
        """
        self.global_model = global_model.to(device)
        self.device = device
        self.clients = []
        self.client_weights = []  # Trọng số cho mỗi client (dựa trên số lượng dữ liệu)
        self.training_history = {
            'global_loss': [],
            'global_accuracy': [],
            'global_auc': [],
            'round': []
        }
    
    def add_client(self, client, weight=1.0):
        """
        Thêm client vào server
        
        Args:
            client: Đối tượng FederatedClient
            weight (float): Trọng số của client trong phép trung bình có trọng số
        """
        self.clients.append(client)
        self.client_weights.append(weight)
        
        # Chuẩn hóa trọng số
        total_weight = sum(self.client_weights)
        self.client_weights = [w / total_weight for w in self.client_weights]
        
        print(f"Đã thêm client {client.client_id} với trọng số {self.client_weights[-1]:.4f}")
    
    def federated_averaging(self):
        """
        Thực hiện federated averaging từ các client để cập nhật mô hình toàn cục
        """
        print("Bắt đầu quá trình Federated Averaging...")
        
        # Lấy tham số mô hình từ tất cả clients
        client_params = [client.get_model_params() for client in self.clients]
        
        # Thực hiện phép trung bình có trọng số
        global_params = copy.deepcopy(self.global_model.state_dict())
        
        for param_name in global_params:
            # Khởi tạo tham số bằng 0
            global_params[param_name] = torch.zeros_like(global_params[param_name])
            
            # Tính trung bình có trọng số
            for i, client_p in enumerate(client_params):
                global_params[param_name] += self.client_weights[i] * client_p[param_name]
        
        # Cập nhật mô hình toàn cục
        self.global_model.load_state_dict(global_params)
        
        # Cập nhật mô hình cho tất cả clients
        print("Đang cập nhật mô hình cho các client...")
        for client in self.clients:
            client.set_model_params(global_params)
        
        print("Federated Averaging hoàn tất")
    
    def evaluate(self, test_loader):
        """
        Đánh giá mô hình toàn cục trên tập kiểm tra
        
        Args:
            test_loader: DataLoader chứa dữ liệu đánh giá
            
        Returns:
            dict: Các metrics đánh giá
        """
        self.global_model.eval()
        criterion = torch.nn.BCELoss()
        all_outputs = []
        all_labels = []
        test_loss = 0
        
        with torch.no_grad():
            for batch in test_loader:
                csv_features = batch['csv_features'].to(self.device)
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.global_model(images, csv_features)
                loss = criterion(outputs, labels)
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
        
        # Lưu vào lịch sử
        self.training_history['global_loss'].append(metrics['loss'])
        self.training_history['global_accuracy'].append(metrics['accuracy'])
        self.training_history['global_auc'].append(metrics['auc'])
        
        return metrics
        
    def train_federated(self, rounds, local_epochs, test_loader=None):
        """
        Huấn luyện mô hình theo phương pháp học liên bang
        
        Args:
            rounds (int): Số vòng huấn luyện liên bang
            local_epochs (int): Số epochs huấn luyện cục bộ trên mỗi client
            test_loader: DataLoader cho tập đánh giá toàn cục (có thể None)
            
        Returns:
            dict: Lịch sử huấn luyện
        """
        print(f"Bắt đầu quá trình huấn luyện liên bang với {rounds} vòng")
        
        for round_num in range(1, rounds + 1):
            print(f"\n--- Vòng Federated Learning {round_num}/{rounds} ---")
            
            # Lưu trữ metrics của từng client trong vòng hiện tại
            client_metrics = []
            
            # Huấn luyện từng client
            for client in self.clients:
                print(f"\nHuấn luyện client {client.client_id}...")
                metrics = client.train(epochs=local_epochs)
                client_metrics.append(metrics)
            
            # Thực hiện Federated Averaging để cập nhật mô hình toàn cục
            self.federated_averaging()
            
            # Đánh giá mô hình toàn cục (nếu có test_loader)
            if test_loader is not None:
                print("\nĐánh giá mô hình toàn cục...")
                global_metrics = self.evaluate(test_loader)
                
                # Lưu số vòng hiện tại
                self.training_history['round'].append(round_num)
                
                print(f"Vòng {round_num}/{rounds} hoàn thành. "
                      f"Global Loss: {global_metrics['loss']:.4f}, "
                      f"Accuracy: {global_metrics['accuracy']:.4f}, "
                      f"AUC: {global_metrics['auc']:.4f}")
        
        print("\nQuá trình huấn luyện liên bang hoàn tất!")
        return self.training_history