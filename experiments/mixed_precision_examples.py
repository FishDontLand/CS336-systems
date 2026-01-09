import torch
from torch import nn


def accumulation_example():
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    print(s)

    s = torch.tensor(0, dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(s)


    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(s)

    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    print(s)

def toy_model_example():
    class ToyModel(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.fc1 = nn.Linear(in_features, 10, bias=False)
            self.ln = nn.LayerNorm(10)
            self.fc2 = nn.Linear(10, out_features, bias=False)
            self.relu = nn.ReLU()

        def forward(self, x):
            print('initial: ', x.dtype)
            x = self.fc1(x)
            print('after fc1: ', x.dtype)
            x = self.relu(x)
            print('after relu ', x.dtype)
            x = self.ln(x)
            print('after ln ', x.dtype)
            x = self.fc2(x)
            print('after fc2 ', x.dtype)
            return x

    device = torch.device('cuda:0')
    toy_model = ToyModel(in_features=20, out_features=10)
    data = torch.randn(100, 20, device=device)
    toy_model.to(device)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        toy_model(data)


if __name__ == "__main__":
    toy_model_example()