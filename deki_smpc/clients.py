import gzip
import io
import json
import logging
import threading
from hashlib import sha256
from time import sleep

import requests
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from models import CheckForTaskRequest, KeyClientRegistration
from torch.nn import Module
from torchvision.models import resnet18
from utils import SecurityUtils

logging.basicConfig(level=logging.INFO)


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
        model: Module = None,
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
        self.model = model

        # start key aggregation routine
        self.__key_aggregation_routine()

    def __upload_key(self, model: Module, phase: int):
        # 1. Extract state_dict
        state_dict = model.state_dict()

        # 2. Serialize and compress
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb") as f:
            torch.save(state_dict, f)
        buffer.seek(0)

        response = requests.post(
            url=f"http://{self.key_aggregation_server_ip}:{self.key_aggregation_server_port}/aggregation/phase/{phase}/upload",
            files={
                "key": ("model.pt.gz", buffer, "application/octet-stream"),
                "client_name": (None, self.client_name),
            },
        )
        if response.status_code != 200:
            raise ConnectionError(
                f"Failed to upload key to key aggregation server: {response.text}"
            )

    def __download_key(self, phase: int) -> dict:
        response = requests.get(
            url=f"http://{self.key_aggregation_server_ip}:{self.key_aggregation_server_port}/aggregation/phase/{phase}/download",
            json={"client_name": self.client_name},
            stream=True,
        )

        if response.status_code != 200:
            raise ConnectionError(
                f"Failed to download key from key aggregation server: {response.text}"
            )

        # Decompress and load the state_dict
        buffer = io.BytesIO(response.content)
        with gzip.GzipFile(fileobj=buffer, mode="rb") as f:
            state_dict = torch.load(f, map_location="cpu")

        return state_dict

    def __phase_1_routine(self):
        # Phase 1: Group key generation

        phase = 1

        phase_1_tasks = {
            "upload": False,
            "download": False,
        }

        while True:
            response = requests.get(
                url=f"http://{self.key_aggregation_server_ip}:{self.key_aggregation_server_port}/aggregation/phase/{phase}/check_for_task",
                data=json.dumps(
                    {
                        "client_name": self.client_name,
                    }
                ),
            )
            if response.status_code == 204:
                logging.info(
                    f"No task available for {self.client_name}. Continuing to poll."
                )
                # No task available, continue polling
                sleep(1)
                continue

            if response.status_code != 200:
                raise ConnectionError(
                    f"Failed to connect to key aggregation server: {response.text}"
                )
            task = response.json()
            logging.info(f"Task received for {self.client_name}: {task}")

            if task["action"] == "upload":
                # upload key
                self.__upload_key(model=self.model, phase=phase)
                # Mark the task as completed
                phase_1_tasks["upload"] = True

            elif task["action"] == "download":
                # download key
                state_dict = self.__download_key(phase=phase)
                logging.info(f"Key downloaded for {self.client_name}: {state_dict}")
                # Mark the task as completed
                phase_1_tasks["download"] = True

            if all(phase_1_tasks.values()):
                # All tasks completed, exit the loop
                break
            sleep(1)  # Sleep for a while before checking again

    def __phase_2_routine(self):
        # Phase 2:

        phase = 2

        response = requests.get(
            url=f"http://{self.key_aggregation_server_ip}:{self.key_aggregation_server_port}/tasks/participants",
        )
        if response.status_code != 200:
            raise ConnectionError(
                f"Failed to connect to key aggregation server: {response.text}"
            )
        participants = response.json()
        if self.client_name not in participants["phase_2_clients"]:
            logging.info(
                f"{self.client_name} is not a participant in phase 2. Exiting key aggregation routine."
            )
            return

        while True:
            response = requests.get(
                url=f"http://{self.key_aggregation_server_ip}:{self.key_aggregation_server_port}/aggregation/phase/{phase}/check_for_task",
                data=json.dumps(
                    {
                        "client_name": self.client_name,
                    }
                ),
            )

            if response.status_code == 204:

                response = requests.get(
                    url=f"http://{self.key_aggregation_server_ip}:{self.key_aggregation_server_port}/aggregation/phase/{phase}/active_tasks",
                )

                if response.status_code != 200:
                    raise ConnectionError(
                        f"Failed to connect to key aggregation server: {response.text}"
                    )

                tasks = response.json()

                if "active" in tasks:
                    if len(tasks["active"]) == 0 and len(tasks["pending"]) == 0:
                        # No active tasks, exit the loop
                        logging.info(
                            f"All phase 2 tasks completed for {self.client_name}. Exiting key aggregation routine."
                        )
                        break
                logging.info(
                    f"No task available for {self.client_name}. Continuing to poll."
                )
                # No task available, continue polling
                sleep(1)
                continue

            if response.status_code != 200:
                raise ConnectionError(
                    f"Failed to connect to key aggregation server: {response.text}"
                )

            task = response.json()
            logging.info(f"Task received for {self.client_name}: {task}")

            if task["action"] == "upload":
                # upload key
                self.__upload_key(model=self.model, phase=phase)

            elif task["action"] == "download":
                # download key
                state_dict = self.__download_key(phase=phase)
                logging.info(f"Key downloaded for {self.client_name}: {state_dict}")

            sleep(1)  # Sleep for a while before checking again

    def __key_aggregation_routine(self):

        self.__phase_1_routine()

        # Wait for all clients to finish phase 1
        while True:
            response = requests.get(
                url=f"http://{self.key_aggregation_server_ip}:{self.key_aggregation_server_port}/aggregation/phase/1/active_tasks",
            )
            if response.status_code != 200:
                raise ConnectionError(
                    f"Failed to connect to key aggregation server: {response.text}"
                )
            tasks = response.json()
            if len(tasks["active"]) == 0 and len(tasks["pending"]) == 0:
                # No active tasks, exit the loop
                logging.info(
                    f"All phase 1 tasks completed for {self.client_name}. Exiting key aggregation routine."
                )
                break
            logging.info(
                f"Waiting for all clients to finish phase 1. Continuing to poll."
            )
            sleep(1)

        logging.info(
            f"All clients have finished phase 1. Proceeding to phase 2 for {self.client_name}."
        )

        self.__phase_2_routine()

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
    import argparse

    import torch
    import torch.nn as nn

    class LinearModel(nn.Module):
        def __init__(self):
            super(LinearModel, self).__init__()
            self.linear = nn.Linear(in_features=1, out_features=10)
            with torch.no_grad():
                self.linear.weight.fill_(1.0)
                self.linear.bias.fill_(0.0)

        def forward(self, x):
            return self.linear(x)

    model = LinearModel()

    # masked_dict = SecurityUtils.generate_secure_random_mask(model)
    # print(masked_dict)

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
        key_aggregation_server_ip="127.0.0.1",
        key_aggregation_server_port=8080,
        fl_aggregation_server_ip="127.0.0.1",
        fl_aggregation_server_port=8081,
        num_clients=3,
        preshared_secret="my_secure_presHared_secret_123!",
        client_name=client_name,  # For better logging at the server. MUST BE UNIQUE ACROSS ALL CLIENTS
        model=model,
    )

    client.submit_model(model=model)
    client.receive_aggregated_model()
