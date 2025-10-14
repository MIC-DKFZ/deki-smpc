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
