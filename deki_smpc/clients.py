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
        self.current_fl_round = 0
        self.model = model
        self.state_dict = model.state_dict() if model else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # start key aggregation routine
        self.__key_aggregation_routine()

    def __upload_key(self, state_dict: dict, phase: int):

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

    def __add_keys(self, downloaded_state_dict: dict):
        """
        Add the downloaded state_dict values to the existing state_dict values.
        This operation is performed element-wise for each tensor in the state_dict.
        """
        logging.info(f"Adding keys {self.state_dict} and {downloaded_state_dict}")

        if self.state_dict is None:
            self.state_dict = downloaded_state_dict
        else:
            for key in self.state_dict:
                if key in downloaded_state_dict:
                    self.state_dict[key] += downloaded_state_dict[key]

    def __phase_1_routine(self):
        # Phase 1: Group key generation

        phase = 1

        first_check = True
        first_in_group = False

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
                if first_check:
                    first_check = False
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
                if first_check:
                    first_in_group = True
                    first_check = False
                # upload key
                self.__upload_key(state_dict=self.state_dict, phase=phase)
                # Mark the task as completed
                phase_1_tasks["upload"] = True

            elif task["action"] == "download":
                # download key
                downloaded_state_dict = self.__download_key(phase=phase)
                logging.info(
                    f"Key downloaded for {self.client_name}: {downloaded_state_dict}"
                )
                if first_in_group:
                    self.state_dict = downloaded_state_dict
                else:
                    # Add the downloaded keys to the existing ones
                    self.__add_keys(downloaded_state_dict)
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
                self.__upload_key(state_dict=self.state_dict, phase=phase)

            elif task["action"] == "download":
                # download key
                state_dict = self.__download_key(phase=phase)
                logging.info(f"Key downloaded for {self.client_name}: {state_dict}")
                # Add the downloaded keys to the existing ones
                self.__add_keys(state_dict)

            sleep(1)  # Sleep for a while before checking again

    def __phase_3_routine(self):
        """
        Phase 3: Final sum upload and download.
        """
        phase = 3

        # Check if the client is the recipient of the final sum
        response = requests.get(
            url=f"http://{self.key_aggregation_server_ip}:{self.key_aggregation_server_port}/aggregation/final/recipient",
        )

        if response.status_code != 200:
            raise ConnectionError(
                f"Failed to connect to key aggregation server: {response.text}"
            )

        recipient_info = response.json()
        if recipient_info.get("recipient") == self.client_name:
            # Upload the final sum
            buffer = io.BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode="wb") as f:
                torch.save(self.state_dict, f)
            buffer.seek(0)

            upload_response = requests.post(
                url=f"http://{self.key_aggregation_server_ip}:{self.key_aggregation_server_port}/aggregation/final/upload",
                files={
                    "final_sum": (
                        "final_weights.pt.gz",
                        buffer,
                        "application/octet-stream",
                    ),
                    "client_name": (None, self.client_name),
                },
            )

            if upload_response.status_code != 200:
                raise ConnectionError(
                    f"Failed to upload final sum to key aggregation server: {upload_response.text}"
                )

            logging.info(f"Final sum uploaded by {self.client_name}.")
            return

        # Download the final sum
        while True:
            response = requests.get(
                url=f"http://{self.key_aggregation_server_ip}:{self.key_aggregation_server_port}/aggregation/final/download",
                stream=True,
            )

            if response.status_code == 200:
                # Decompress and load the final state_dict
                buffer = io.BytesIO(response.content)
                with gzip.GzipFile(fileobj=buffer, mode="rb") as f:
                    final_state_dict = torch.load(f, map_location="cpu")

                logging.info(
                    f"Final sum downloaded for {self.client_name}: {final_state_dict}"
                )

                # Update the local state_dict with the final sum
                self.state_dict = final_state_dict
                break

            elif response.status_code == 404:
                sleep(1)
            else:
                raise ConnectionError(
                    f"Failed to download final sum from key aggregation server: {response.text}"
                )

    def __key_aggregation_routine(self):
        logging.info("-- PHASE 1 --")
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
            sleep(1)

        logging.info(
            f"All clients have finished phase 1. Proceeding to phase 2 for {self.client_name}."
        )

        logging.info("-- PHASE 2 --")
        self.__phase_2_routine()
        logging.info("-- PHASE 3 --")
        self.__phase_3_routine()

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
