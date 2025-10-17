import argparse

import torch
import torch.nn as nn

from deki_smpc import FedAvgClient


class LSTM500k(nn.Module):
    """
    LSTM with roughly 500k parameters.
    Config: vocab_size=1500, embed_dim=128, hidden_size=256, num_classes=10
    """

    def __init__(
        self,
        vocab_size: int = 1500,
        embed_dim: int = 128,
        hidden_size: int = 256,
        num_classes: int = 10,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embed(x)  # (batch, seq_len, embed_dim)
        out, (h_n, c_n) = self.lstm(x)
        last = h_n[-1]  # (batch, hidden_size)
        logits = self.fc(last)  # (batch, num_classes)
        return logits


class LSTMModel100k(nn.Module):
    def __init__(
        self,
        input_size=64,
        hidden_size=128,
        num_layers=1,
        output_size=10,
        bidirectional=False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        D = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * D, output_size)

    def forward(self, x):
        # x shape: [batch, seq_len, input_size]
        out, _ = self.lstm(x)  # out: [batch, seq_len, hidden]
        out = out[:, -1, :]  # take last time step
        return self.fc(out)  # [batch, output_size]


class LSTM818k(nn.Module):
    """
    Vocab 2033, embed 128, hidden 313, classes 10.
    Total params 818000.
    """

    def __init__(
        self,
        vocab_size: int = 2033,
        embed_dim: int = 128,
        hidden_size: int = 313,
        num_classes: int = 10,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len)
        x = self.embed(x)
        out, (h_n, c_n) = self.lstm(x)
        last = h_n[-1]  # (batch, hidden)
        logits = self.fc(last)
        return logits


class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),  # 32x32 -> 28x28
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),  # 14x14 -> 10x10
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 10x10 -> 5x5
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 5 * 5, 120)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.flatten(out)  # shape: (N, 16*5*5)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


class LeNet273k(nn.Module):
    """
    Input: 1x32x32
    Params: exactly 273,000 for num_classes=10
    Layout: conv5x5 -> relu -> pool, conv5x5 -> relu -> pool, flatten, fc -> relu, fc -> relu, fc
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        assert (
            num_classes == 10
        ), "This config has 273,000 params only when num_classes=10."

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=0),  # 32->28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28->14
            nn.Conv2d(8, 48, kernel_size=5, stride=1, padding=0),  # 14->10
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 10->5
        )
        self.flatten = nn.Flatten()
        # 48 * 5 * 5 = 1200
        self.classifier = nn.Sequential(
            nn.Linear(1200, 196),
            nn.ReLU(inplace=True),
            nn.Linear(196, 134),
            nn.ReLU(inplace=True),
            nn.Linear(134, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


# sanity check on parameter count
model = LSTM500k()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {total_params:,}")

parser = argparse.ArgumentParser(description="Federated Learning Client")
parser.add_argument(
    "--client_name",
    type=str,
    default="client_1",
    help="Name of the client",
)

parser.add_argument(
    "--aggregation_server_ip",
    type=str,
    default="127.0.0.1",
    help="IP address of the aggregation server",
)

parser.add_argument(
    "--aggregation_server_port",
    type=int,
    default=8080,
    help="Port of the aggregation server",
)

parser.add_argument(
    "--num_clients",
    type=int,
    default=4,
    help="Total number of clients participating in federated learning",
)

parser.add_argument(
    "--preshared_secret",
    type=str,
    default="my_secure_presHared_secret_123!",
    help="Preshared secret for secure communication",
)

client_name = parser.parse_args().client_name
aggregation_server_ip = parser.parse_args().aggregation_server_ip
aggregation_server_port = parser.parse_args().aggregation_server_port
num_clients = parser.parse_args().num_clients
preshared_secret = parser.parse_args().preshared_secret

client = FedAvgClient(
    aggregation_server_ip=aggregation_server_ip,
    aggregation_server_port=aggregation_server_port,
    num_clients=num_clients,
    preshared_secret=preshared_secret,
    client_name=client_name,  # For better logging at the server. MUST BE UNIQUE ACROSS ALL CLIENTS
    model=model,
)

updated_model = client.aggregate()
