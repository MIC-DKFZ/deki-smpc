import io
import json
import logging
import os
import sys
import tempfile
import time
from copy import deepcopy
from hashlib import sha256
from time import sleep

import httpx
import lz4.frame
import requests
import torch
from requests.adapters import HTTPAdapter
from torch.nn import Module
from tqdm import tqdm
from urllib3.util import Retry

from .models import KeyClientRegistration
from .utils import FixedPointConverter, SecurityUtils


class FedAvgClient:

    def __init__(
        self,
        aggregation_server_ip: str,
        aggregation_server_port: int,
        num_clients: int = None,
        preshared_secret: str = None,
        client_name: str = None,
        model: Module = None,
        ignore_model_keys: list = [],
        logging_level: int = logging.INFO,
    ):
        assert num_clients is not None, "Number of clients must be provided"
        assert num_clients >= 3, "Number of clients must be at least 3"
        assert model is not None, "Torch model must be provided"
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
        logging.basicConfig(level=logging_level)

        self.key_aggregation_server_ip = aggregation_server_ip
        self.key_aggregation_server_port = aggregation_server_port
        self.num_clients = num_clients
        self.preshared_secret = sha256(preshared_secret.encode()).hexdigest()
        self.client_name = client_name
        self.public_facing_ip = requests.get(
            "https://api.ipify.org/?format=json"
        ).json()["ip"]
        self.url = f"http://{self.key_aggregation_server_ip}:{self.key_aggregation_server_port}"

        # Initialize a requests Session with retries
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[502, 503, 504],
            allowed_methods={"GET", "POST"},
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.__connect_to_key_aggregation_server()
        self.current_fl_round = 0
        self.model = model.float() if model else None
        self.state_dict = model.state_dict() if model else None
        self.ignore_model_keys = ignore_model_keys

        # Check for int tensors in the state_dict (e.g. num batches tracked)
        if self.state_dict is not None:
            if len(self.ignore_model_keys) == 0:
                for key, _ in self.state_dict.items():
                    if (
                        self.state_dict[key].dtype == torch.int64
                        or self.state_dict[key].dtype == torch.int32
                        or self.state_dict[key].dtype == torch.int16
                        or self.state_dict[key].dtype == torch.int8
                    ):
                        self.ignore_model_keys.append(key)

        if len(self.ignore_model_keys) > 0:
            logging.info(f"Ignoring model keys: {self.ignore_model_keys}")

        self.aggregated_state_dict = None
        self.public_key = (
            SecurityUtils.dummy_generate_secure_random_mask(self.state_dict)
            if self.state_dict
            else None
        )
        self.private_key = deepcopy(self.public_key) if self.public_key else None
        self.secure_random_mask = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fpe = FixedPointConverter(device=self.device)
        self.chunk_size = 1024 * 1024  # 1 MB

    @staticmethod
    def __measure_time(func):
        """
        Decorator to measure the execution time of a function.
        """

        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logging.info(
                f"Execution time for {func.__name__}: {end_time - start_time:.2f} seconds"
            )
            return result

        return wrapper

    def _iter_bytes(self, b: bytes, pbar: tqdm):
        mv = memoryview(b)
        total = len(b)
        i = 0
        while i < total:
            j = i + self.chunk_size
            chunk = mv[i:j]
            pbar.update(len(chunk))
            yield chunk
            i = j

    def _stream_upload(self, url: str, headers: dict, content_iterable):
        # httpx >= 0.28 streaming upload path
        with httpx.Client(timeout=None) as client:
            with client.stream(
                "PUT", url, headers=headers, content=content_iterable
            ) as resp:
                resp.raise_for_status()

    def __measure_request_time(self, request_func, *args, **kwargs):
        """
        Measure the time taken for a request.
        """
        url = kwargs.get("url", "Unknown URL")
        start_time = time.time()
        response = request_func(self.session, *args, **kwargs)
        end_time = time.time()
        logging.info(f"Request to {url} took {end_time - start_time:.2f} seconds")
        return response

    @__measure_time
    def __upload_model(self, state_dict: dict):

        # Serialize and compress before uploading
        buffer = self.__serialize_and_compress(state_dict)

        upload_response = self.__measure_request_time(
            lambda session, **kwargs: session.post(**kwargs),
            url=f"{self.url}/secure-fl/upload",
            files={
                "model": (
                    "model.pt.gz",
                    buffer,
                    "application/octet-stream",
                ),
                "client_name": (None, self.client_name),
            },
        )

        if upload_response.status_code != 200:
            raise ConnectionError(
                f"Failed to upload model to fl server: {upload_response.text}"
            )

        logging.info(f"Model uploaded by {self.client_name}.")

    @__measure_time
    def __download_model(self) -> dict:
        # Download the final sum
        while True:
            response = self.__measure_request_time(
                lambda session, **kwargs: session.get(**kwargs),
                url=f"{self.url}/secure-fl/download",
                stream=True,
            )

            if response.status_code == 200:
                # Decompress and load the final state_dict after downloading
                buffer = io.BytesIO(response.content)
                state_dict = self.__decompress_and_load(buffer.getvalue())

                logging.info("Model downloaded")
                break

            elif response.status_code == 404:
                sleep(1)
            else:
                raise ConnectionError(
                    f"Failed to download model from fl server: {response.text}"
                )

        return state_dict

    @__measure_time
    def __upload_key(self, state_dict: dict, phase: int):

        bio = io.BytesIO()
        torch.save(state_dict, bio, _use_new_zipfile_serialization=True)
        data = bio.getvalue()

        url = f"{self.url}/key-aggregation/aggregation/upload"
        headers = {
            "Content-Type": "application/octet-stream",
            "X-Filename": f"{phase}_{self.client_name}_state_dict.pth",
            "X-Client-Name": self.client_name,
            "X-Phase": str(phase),
        }
        with tqdm(
            total=len(data),
            unit="B",
            unit_scale=True,
            desc=f"Uploading: {phase}_{self.client_name}_state_dict.pth",
            ascii=True,
        ) as pbar:
            self._stream_upload(url, headers, self._iter_bytes(data, pbar))
        logging.info(
            f"Uploaded id='{phase}_{self.client_name}_state_dict.pth' as {headers['X-Filename']} ({len(data)} bytes)."
        )

    @__measure_time
    def __download_key(self, phase: int) -> dict:
        headers = {
            "X-Client-Name": self.client_name,
            "X-Phase": str(phase),
        }
        url = f"{self.url}/key-aggregation/aggregation/download"
        with tempfile.TemporaryDirectory(prefix=f"{phase}") as tmpdir:
            out_path = os.path.join(
                tmpdir, f"{phase}_{self.client_name}_state_dict.pth"
            )
            with httpx.Client(timeout=None) as client:
                with client.stream("GET", url, headers=headers) as resp:
                    if resp.status_code == 404:
                        logging.error(
                            f"No artifact for id='{phase}_{self.client_name}_state_dict.pth'."
                        )
                        sys.exit(1)
                    resp.raise_for_status()
                    total = int(resp.headers.get("content-length") or 0)
                    with open(out_path, "wb") as f, tqdm(
                        total=total if total > 0 else None,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading: {phase}_{self.client_name}_state_dict.pth",
                        ascii=True,
                    ) as pbar:
                        for chunk in resp.iter_bytes(self.chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
            logging.info(
                f"Saved id='{phase}_{self.client_name}_state_dict.pth' to {out_path}"
            )
            with open(out_path, "rb") as f:
                data = f.read()
            state_dict = torch.load(io.BytesIO(data), map_location="cpu")
            # try:
            #     model = torchvision.models.resnet18(weights=None)
            # except TypeError:
            #     model = torchvision.models.resnet18(pretrained=False)
            # missing, unexpected = model.load_state_dict(state_dict, strict=False)

        logging.info(f"Key downloaded for {self.client_name}")

        return state_dict

    @__measure_time
    def __add_keys(self, downloaded_state_dict: dict):
        """
        Add the downloaded state_dict values to the existing state_dict values.
        This operation is performed element-wise for each tensor in the state_dict.
        """
        if self.public_key is None:
            self.public_key = downloaded_state_dict
        else:
            for key in self.public_key:
                if key in downloaded_state_dict:
                    if FixedPointConverter.is_float_tensor(self.public_key[key]):
                        # Convert float tensors to int64 for addition
                        self.public_key[key] = self.fpe.encode(self.public_key[key])

                    if FixedPointConverter.is_float_tensor(downloaded_state_dict[key]):
                        # Convert float tensors to int64 for addition
                        downloaded_state_dict[key] = self.fpe.encode(
                            downloaded_state_dict[key]
                        )
                    # Move tensors to GPU for addition
                    self.public_key[key] = self.public_key[key].to(self.device)
                    downloaded_state_dict[key] = downloaded_state_dict[key].to(
                        self.device
                    )
                    self.public_key[key] += downloaded_state_dict[key]
                    # Offload tensors back to CPU
                    self.public_key[key] = self.public_key[key].cpu()

    @__measure_time
    def __convert_state_dict_to_int(self, state_dict: dict) -> dict:
        for key, val in state_dict.items():
            if key in self.ignore_model_keys:
                state_dict[key] = val
                continue
            state_dict[key] = self.fpe.encode(val)

        return state_dict

    @__measure_time
    def __convert_int_to_state_dict(self, int_state_dict: dict) -> dict:
        for key, val in int_state_dict.items():
            if key in self.ignore_model_keys:
                int_state_dict[key] = val
                continue
            # Convert JSON-loaded lists or scalars into a LongTensor on the correct device
            if not torch.is_tensor(val):
                int_state_dict[key] = torch.tensor(
                    val, dtype=torch.long, device=self.device
                )
            else:
                int_state_dict[key] = val.to(dtype=torch.long, device=self.device)
            # Decode the integer tensor back to the original state tensor
            int_state_dict[key] = self.fpe.decode(int_state_dict[key])

        return int_state_dict

    @__measure_time
    def __shield_key(self, state_dict: dict, secure_random_mask: dict = None):
        """
        Shield the state_dict by applying a random mask to each tensor.
        The mask is generated using the SecurityUtils class.
        """
        state_dict = self.__convert_state_dict_to_int(state_dict)

        if secure_random_mask is None:
            secure_random_mask = SecurityUtils.generate_secure_random_mask(state_dict)

        for key, _ in state_dict.items():
            state_dict[key] = (state_dict[key] + secure_random_mask[key]).to(
                self.device
            )

        return state_dict, secure_random_mask

    @__measure_time
    def __unshield_key(
        self, shielded_state_dict: dict, secure_random_mask: dict = None
    ):
        """
        Unshield the state_dict by removing the random mask from each tensor.
        The mask is generated using the SecurityUtils class.
        """

        for key, _ in shielded_state_dict.items():
            # Ensure both tensors are on the same device
            shielded_state_dict[key] = shielded_state_dict[key].to(self.device)
            secure_random_mask[key] = secure_random_mask[key].to(self.device)

            shielded_state_dict[key] = (
                shielded_state_dict[key] - secure_random_mask[key]
            )

        return self.__convert_int_to_state_dict(shielded_state_dict)

    @__measure_time
    def __phase_1_routine(self):
        # Phase 1: Group key generation

        phase = 1
        first_senders = []
        while len(first_senders) == 0:
            response = self.__measure_request_time(
                lambda session, **kwargs: session.get(**kwargs),
                url=f"{self.url}/key-aggregation/aggregation/phase/{phase}/first_senders",
            )
            if response.status_code != 200:
                raise ConnectionError(
                    f"Failed to connect to key aggregation server: {response.text}"
                )

            first_senders = response.json()["first_senders"]

        logging.info(first_senders)
        # Check if the client is the first in the group
        first_in_group = False
        if self.client_name in first_senders:
            first_in_group = True
            logging.info(f"{self.client_name} is the first sender in the group.")
        else:
            logging.info(f"{self.client_name} is not the first sender in the group.")

        phase_1_tasks = {
            "upload": False,
            "download": False,
        }

        while True:
            response = self.__measure_request_time(
                lambda session, **kwargs: session.get(**kwargs),
                url=f"{self.url}/key-aggregation/aggregation/phase/{phase}/check_for_task",
                data=json.dumps(
                    {
                        "client_name": self.client_name,
                    }
                ),
            )
            if response.status_code == 204:
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

                if first_in_group:
                    # Shield the state_dict before uploading
                    self.public_key, self.secure_random_mask = self.__shield_key(
                        self.public_key
                    )
                # upload key
                self.__upload_key(state_dict=self.public_key, phase=phase)
                # Mark the task as completed
                phase_1_tasks["upload"] = True

            elif task["action"] == "download":
                downloaded_state_dict = self.__download_key(phase=phase)
                if first_in_group:
                    self.public_key = downloaded_state_dict
                    # Unshield the state_dict after downloading
                    self.public_key = self.__unshield_key(
                        self.public_key, self.secure_random_mask
                    )
                else:
                    # Add the downloaded keys to the existing ones
                    self.__add_keys(downloaded_state_dict)
                # Mark the task as completed
                phase_1_tasks["download"] = True

            if all(phase_1_tasks.values()):
                # All tasks completed, exit the loop
                break
            sleep(1)  # Sleep for a while before checking again

    @__measure_time
    def __phase_2_routine(self):
        # Phase 2:

        phase = 2

        response = self.__measure_request_time(
            lambda session, **kwargs: session.get(**kwargs),
            url=f"{self.url}/key-aggregation/tasks/participants",
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
            response = self.__measure_request_time(
                lambda session, **kwargs: session.get(**kwargs),
                url=f"{self.url}/key-aggregation/aggregation/phase/{phase}/check_for_task",
                data=json.dumps(
                    {
                        "client_name": self.client_name,
                    }
                ),
            )

            if response.status_code == 204:

                response = self.__measure_request_time(
                    lambda session, **kwargs: session.get(**kwargs),
                    url=f"{self.url}/key-aggregation/aggregation/phase/{phase}/active_tasks",
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
                self.__upload_key(state_dict=self.public_key, phase=phase)

            elif task["action"] == "download":
                # download key
                state_dict = self.__download_key(phase=phase)
                # Add the downloaded keys to the existing ones
                self.__add_keys(state_dict)

            sleep(1)  # Sleep for a while before checking again

    @__measure_time
    def __phase_3_routine(self):
        """
        Phase 3: Final sum upload and download.
        """

        # Check if the client is the recipient of the final sum
        response = self.__measure_request_time(
            lambda session, **kwargs: session.get(**kwargs),
            url=f"{self.url}/key-aggregation/aggregation/final/recipient",
        )

        if response.status_code != 200:
            raise ConnectionError(
                f"Failed to connect to key aggregation server: {response.text}"
            )

        recipient_info = response.json()
        if recipient_info.get("recipient") == self.client_name:
            # Serialize and compress the state_dict before uploading
            buffer = self.__serialize_and_compress(self.public_key)

            upload_response = self.__measure_request_time(
                lambda session, **kwargs: session.post(**kwargs),
                url=f"{self.url}/key-aggregation/aggregation/final/upload",
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
            response = self.__measure_request_time(
                lambda session, **kwargs: session.get(**kwargs),
                url=f"{self.url}/key-aggregation/aggregation/final/download",
                stream=True,
            )

            if response.status_code == 200:
                # Decompress and load the final state_dict after downloading
                buffer = io.BytesIO(response.content)
                self.public_key = self.__decompress_and_load(buffer.getvalue())
                logging.info(f"Final sum downloaded for {self.client_name}")
                break

            elif response.status_code == 404:
                sleep(1)
            else:
                raise ConnectionError(
                    f"Failed to download final sum from key aggregation server: {response.text}"
                )

    @__measure_time
    def __key_aggregation_routine(self):
        logging.info("-- PHASE 1 --")
        self.__phase_1_routine()

        # Wait for all clients to finish phase 1
        while True:
            response = self.__measure_request_time(
                lambda session, **kwargs: session.get(**kwargs),
                url=f"{self.url}/key-aggregation/aggregation/phase/1/active_tasks",
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

    @__measure_time
    def __connect_to_key_aggregation_server(self):

        request_body = KeyClientRegistration(
            ip_address=self.public_facing_ip,
            client_name=self.client_name,
            preshared_secret=self.preshared_secret,
        )

        response = self.__measure_request_time(
            lambda session, **kwargs: session.post(**kwargs),
            url=f"{self.url}/key-aggregation/register",
            data=json.dumps(request_body.dict()),
        )

        if response.status_code != 200:
            raise ConnectionError(
                f"Failed to connect to key aggregation server: {response.text}"
            )

    @__measure_time
    def __serialize_and_compress(self, state_dict: dict) -> io.BytesIO:
        """
        Serialize and compress the state_dict.
        """
        buffer = io.BytesIO()
        with lz4.frame.open(buffer, mode="wb") as f:
            torch.save(state_dict, f)
        buffer.seek(0)
        return buffer

    @__measure_time
    def __decompress_and_load(self, compressed_data: bytes) -> dict:
        """
        Decompress and load the state_dict.
        """
        buffer = io.BytesIO(compressed_data)
        with lz4.frame.open(buffer, mode="rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        return state_dict

    @__measure_time
    def prepare_transfer(self):
        self.__key_aggregation_routine()

    @__measure_time
    def aggregate(self) -> Module:

        self.prepare_transfer()

        shielded_model, _ = self.__shield_key(self.state_dict, self.private_key)

        self.__upload_model(shielded_model)

        model_data = self.__unshield_key(self.__download_model(), self.public_key)

        response = self.__measure_request_time(
            lambda session, **kwargs: session.post(**kwargs),
            url=f"{self.url}/key-aggregation/aggregation/finished",
            data=json.dumps(
                {
                    "client_name": self.client_name,
                }
            ),
        )
        if response.status_code != 200:
            raise ConnectionError(
                f"Failed to notify server of completion: {response.text}"
            )

        for k in model_data.keys():
            model_data[k] = model_data[k] / self.num_clients

        self.aggregated_state_dict = model_data
        return self.aggregated_state_dict
