# Haven't written for this file yet
# Update later

from .client import FederatedClient
from .server import FederatedServer

class FederatedTrainer:
    def __init__(self, server: FederatedServer, clients: list, rounds: int = 10):
        """
        Lớp điều phối quá trình huấn luyện liên bang
        
        Args:
            server (FederatedServer): Server trung tâm để tổng hợp mô hình
            clients (list): Danh sách các client tham gia
            rounds (int): Số vòng huấn luyện liên bang
        """
        self.server = server
        self.clients = clients
        self.rounds = rounds

    def train(self):
        """
        Điều phối quá trình huấn luyện liên bang
        """
        for round_num in range(1, self.rounds + 1):
            print(f"--- Bắt đầu vòng {round_num}/{self.rounds} ---")
            
            # Gửi tham số mô hình từ server đến các client
            global_params = self.server.get_global_model_params()
            for client in self.clients:
                client.set_model_params(global_params)
            
            # Huấn luyện cục bộ trên từng client
            client_metrics = []
            for client in self.clients:
                print(f"Client {client.client_id} đang huấn luyện...")
                metrics = client.train(epochs=1)  # Huấn luyện 1 epoch mỗi vòng
                client_metrics.append(metrics)
            
            # Tổng hợp tham số mô hình từ các client
            client_params = [client.get_model_params() for client in self.clients]
            self.server.aggregate(client_params)
            
            # Đánh giá mô hình toàn cục (nếu cần)
            global_metrics = self.server.evaluate_global_model()
            print(f"--- Kết thúc vòng {round_num}/{self.rounds}, Global Metrics: {global_metrics} ---")
        
        print("Quá trình huấn luyện liên bang hoàn tất.")

    def evaluate(self, test_loader):
        """
        Đánh giá mô hình toàn cục trên tập kiểm tra
        
        Args:
            test_loader: DataLoader chứa dữ liệu kiểm tra
        
        Returns:
            dict: Các metrics đánh giá
        """
        print("Đang đánh giá mô hình toàn cục...")
        return self.server.evaluate_global_model(test_loader)