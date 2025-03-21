import json
from hashlib import sha256

import requests
from torch.nn import Module

from deki_smpc.models import KeyClientRegistration


class FedAvgClient:

    def __init__(
        self,
        key_aggregation_server_ip: str,
        key_aggregation_server_port: int,
        fl_aggregation_server_ip: str,
        fl_aggregation_server_port: int,
        num_clients: int = None,
        preshared_secret: str = None,
        client_name: str = None,
    ):
        assert num_clients is not None, "Number of clients must be provided"
        assert num_clients >= 3, "Number of clients must be at least 3"

        assert preshared_secret is not None, "Preshared secret must be provided"
        # Validate that preshared secret is at least 16 characters long and contains at least one number, one uppercase letter, and one special character
        assert (
            len(preshared_secret) >= 16
        ), "Preshared secret must be at least 16 characters long"
        assert any(
            char.isdigit() for char in preshared_secret
        ), "Preshared secret must contain at least one number"
        assert any(
            char.isupper() for char in preshared_secret
        ), "Preshared secret must contain at least one uppercase letter"
        assert any(
            not char.isalnum() for char in preshared_secret
        ), "Preshared secret must contain at least one special character"

        self.key_aggregation_server_ip = key_aggregation_server_ip
        self.key_aggregation_server_port = key_aggregation_server_port
        self.fl_aggregation_server_ip = fl_aggregation_server_ip
        self.fl_aggregation_server_port = fl_aggregation_server_port
        self.num_clients = num_clients
        self.preshared_secret = sha256(preshared_secret.encode()).hexdigest()
        self.client_name = client_name
        self.public_facing_ip = requests.get(
            "https://api.ipify.org/?format=json"
        ).json()["ip"]
        self.__connect_to_key_aggregation_server()
        self.num_total_fl_rounds = self.__connect_to_fl_aggregation_server()
        self.current_fl_round = 0

        # start key aggregation routine
        self.__key_aggregation_routine()

    def __key_aggregation_routine(self):
        pass

    def __connect_to_key_aggregation_server(self):

        request_body = KeyClientRegistration(
            ip_address=self.public_facing_ip,
            client_name=self.client_name,
            preshared_secret=self.preshared_secret,
        )

        response = requests.post(
            url=f"http://{self.key_aggregation_server_ip}:{self.key_aggregation_server_port}/register",
            data=json.dumps(request_body.dict()),
        )

        if response.status_code != 200:
            raise ConnectionError(
                f"Failed to connect to key aggregation server: {response.text}"
            )

    def __connect_to_fl_aggregation_server(self):
        return 10  # TODO: replace with actual number of rounds

    def __average_model(self):
        pass

    def submit_model(self, model: Module):
        assert isinstance(model, Module), "Model must be a PyTorch module"
        # encrypt model
        # send model to fl server
        pass

    def receive_aggregated_model(self):
        # poll fl server for aggregated model
        # decrypt model
        self.__average_model()
        pass


if __name__ == "__main__":

    from torchvision.models import resnet18

    model = resnet18()

    client = FedAvgClient(
        key_aggregation_server_ip="127.0.0.1",
        key_aggregation_server_port=8080,
        fl_aggregation_server_ip="127.0.0.1",
        fl_aggregation_server_port=8081,
        num_clients=3,
        preshared_secret="my_secure_presHared_secret_123!",
        client_name="client_1",  # For better logging at the server‚
    )

    client.submit_model(model=model)
    client.receive_aggregated_model()
