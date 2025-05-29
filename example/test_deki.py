import argparse

import torch
import torch.nn as nn

from deki_smpc import FedAvgClient


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=10)
        with torch.no_grad():
            self.linear.weight.fill_(3.123134)
            self.linear.bias.fill_(1.123123)

    def forward(self, x):
        return self.linear(x)


model = LinearModel()

parser = argparse.ArgumentParser(description="Federated Learning Client")
parser.add_argument(
    # client
    "--client_name",
    type=str,
    default="client_20",
    help="Name of the client",
)

client_name = parser.parse_args().client_name

client = FedAvgClient(
    aggregation_server_ip="127.0.0.1",
    aggregation_server_port=8080,
    num_clients=4,
    preshared_secret="my_secure_presHared_secret_123!",
    client_name=client_name,  # For better logging at the server. MUST BE UNIQUE ACROSS ALL CLIENTS
    model=model,
)

updated_model = client.aggregate()
