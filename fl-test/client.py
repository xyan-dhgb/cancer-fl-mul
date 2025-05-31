import flwr as fl
import torch
from model import Net
from utils import train, test
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
testset  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader  = torch.utils.data.DataLoader(testset, batch_size=32)

model = Net()

class FLClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in model.state_dict().values()]
    
    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(model, trainloader)
        return self.get_parameters(config), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc = test(model, testloader)
        return float(0.0), len(testloader.dataset), {"accuracy": acc}

fl.client.start_numpy_client(server_address="127.0.0.1:8081", client=FLClient())
