import flwr as fl

# Khởi tạo chiến lược FedAvg
strategy = fl.server.strategy.FedAvg()

# Chạy server
fl.server.start_server(
    server_address="127.0.0.1:8081",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy
)
