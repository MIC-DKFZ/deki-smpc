import inspect
import io
import json
import logging
import os
import time
from hashlib import sha256

import httpx
import requests
import torch
from openfhe import (
    BINARY,
    DeserializeCiphertext,
    DeserializePrivateKey,
    DeserializePublicKey,
    Serialize,
)
from torch.nn import Module
from tqdm import tqdm

from deki_smpc.config import MAX_LENGTH, cc
from deki_smpc.utils import (
    _iter_bytes,
    _stream_upload,
    chunk_list,
    flatten_state_dict,
    pack_chunks,
    reconstruct_state_dict,
    unchunk_list,
    unpack_chunks,
)

logging.basicConfig(level=logging.ERROR)

# Unique run ID for this client run
run_id = sha256(f"{time.time()}_{os.urandom(16)}".encode()).hexdigest()[:8]
logging.info(f"Client run ID: {run_id}")

time_measurement_data = {}
time_measurement_log_path = os.path.join(
    "./logs",
    f"{run_id}_client_time_measurements.json",
)
os.makedirs(os.path.dirname(time_measurement_log_path), exist_ok=True)
logging.info(f"Time measurement log path: {time_measurement_log_path}")

transmitted_bytes_data = {}
transmitted_bytes_log_path = os.path.join(
    "./logs",
    f"{run_id}_client_transmitted_bytes.json",
)
os.makedirs(os.path.dirname(transmitted_bytes_log_path), exist_ok=True)


def measure_time(time_measurement_data=None, time_measurement_log_path=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logging.info(
                f"Execution time for {func.__name__}: {end_time - start_time:.2f} seconds"
            )
            unique_id = sha256(f"{func.__name__}_{start_time}".encode()).hexdigest()[:8]
            time_measurement_data[f"{func.__name__}_{unique_id}"] = {
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
            }
            with open(time_measurement_log_path, "w") as f:
                json.dump(time_measurement_data, f, indent=4)
            return result

        return wrapper

    return decorator


class CkksClient:

    def __init__(
        self,
        aggregation_server_ip: str,
        aggregation_server_port: int,
        key_server_ip: str = None,
        key_server_port: int = None,
        num_clients: int = None,
        client_name: str = None,
        model: Module = None,
        ignore_model_keys: list = [],
        preshared_secret: str = None,
    ):
        assert num_clients is not None, "Number of clients must be provided"
        assert num_clients >= 2, "Number of clients must be at least 2"
        assert model is not None, "Torch model must be provided"
        assert preshared_secret is not None, "Preshared secret must be provided"

        self.key_aggregation_server_ip = aggregation_server_ip
        self.key_aggregation_server_port = aggregation_server_port
        self.key_server_ip = key_server_ip
        self.key_server_port = key_server_port
        self.num_clients = num_clients
        self.client_name = client_name
        self.preshared_secret = preshared_secret

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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk_size = 1024 * 1024  # 1 MB

        MAX_RETRIES = 3
        RETRY_DELAY = 2  # seconds

        @measure_time(
            time_measurement_data=time_measurement_data,
            time_measurement_log_path=time_measurement_log_path,
        )
        def download_file(url, payload, filename):
            for attempt in range(MAX_RETRIES):
                response = requests.get(url, json=payload)

                if response.status_code == 200 and response.content:
                    with open(filename, "wb") as f:
                        f.write(response.content)

                    logging.info(f"Downloaded {filename} successfully.")
                    return filename

                elif response.status_code == 204:
                    logging.warning(
                        f"Attempt {attempt + 1}: No content available for {filename}. Retrying in {RETRY_DELAY} seconds..."
                    )
                    time.sleep(RETRY_DELAY)
                    continue

                else:
                    raise RuntimeError(
                        f"Failed to download {filename}. "
                        f"Status code: {response.status_code}, Response: {response.text}"
                    )

            raise RuntimeError(
                f"Failed to download {filename} after {MAX_RETRIES} retries (got 204)."
            )

        # Usage
        url_pub = f"http://{self.key_server_ip}:{self.key_server_port}/key-distribution/download/public_key"
        pub_file = download_file(
            url_pub,
            {"preshared_secret": self.preshared_secret},
            f"{self.client_name}_public.key",
        )
        self.pubkey, _ = DeserializePublicKey(pub_file, BINARY)

        url_sec = f"http://{self.key_server_ip}:{self.key_server_port}/key-distribution/download/secret_key"
        sec_file = download_file(
            url_sec,
            {"preshared_secret": self.preshared_secret},
            f"{self.client_name}_secret.key",
        )
        self.seckey, _ = DeserializePrivateKey(sec_file, BINARY)

        # register the client at the aggregation server

        headers = {"X-Client-Name": self.client_name}

        url = f"http://{self.key_aggregation_server_ip}:{self.key_aggregation_server_port}/secure-fl/register-client"
        response = requests.post(url, headers=headers)

        if response.status_code != 200:
            raise Exception(
                f"Failed to register client at aggregation server: {response.text}"
            )

        logging.info(
            f"Client {self.client_name} registered at aggregation server {self.key_aggregation_server_ip}:{self.key_aggregation_server_port}"
        )

        while True:
            # Wait till all clients are registered
            url = f"http://{self.key_aggregation_server_ip}:{self.key_aggregation_server_port}/secure-fl/all-clients-registered"
            response = requests.get(url, headers=headers)

            body = response.json()
            if body.get("all_clients_registered"):
                logging.info("All clients registered. Proceeding...")
                break
            time.sleep(2)
            logging.info("Waiting for all clients to register...")

    @measure_time(
        time_measurement_data=time_measurement_data,
        time_measurement_log_path=time_measurement_log_path,
    )
    def __create_encrypted_payload(self) -> tuple[bytes, dict, dict, int]:
        x, mapping_dict, ignored_dict = flatten_state_dict(
            self.state_dict, self.ignore_model_keys
        )
        chunks = chunk_list(x, max_length=MAX_LENGTH)

        for i, chunk in enumerate(chunks):
            ptxt = cc.MakeCKKSPackedPlaintext(chunk)
            c = cc.Encrypt(self.pubkey, ptxt)
            # Serialize and deserialize the ciphertext
            chunks[i] = Serialize(c, BINARY)

        del c
        payload = pack_chunks(chunks)
        return payload, mapping_dict, ignored_dict, len(chunks)

    @measure_time(
        time_measurement_data=time_measurement_data,
        time_measurement_log_path=time_measurement_log_path,
    )
    def __load_encrypted_payload(
        self, payload: bytes, mapping_dict, ignored_dict
    ) -> list:

        chunks = unpack_chunks(payload)

        for i, chunk in enumerate(chunks):
            # write to file for deserialization
            with open(f"{self.client_name}_ciphertext_aggregated.txt", "wb") as f:
                f.write(chunk)

            chunk, _ = DeserializeCiphertext(
                f"{self.client_name}_ciphertext_aggregated.txt", BINARY
            )
            cc = chunk.GetCryptoContext()
            cc.EvalMultKeyGen(self.seckey)
            # Decrypt each chunk
            chunk = cc.Decrypt(chunk, self.seckey)
            chunks[i] = chunk.GetCKKSPackedValue()
            # keep only the real part
            for j, number in enumerate(chunks[i]):
                chunks[i][j] = number.real

        del cc
        del chunk

        # Rebuild the state_dict
        reconstructed_state_dict = reconstruct_state_dict(
            unchunk_list(chunks), mapping_dict, ignored_dict
        )
        return reconstructed_state_dict

    @measure_time(
        time_measurement_data=time_measurement_data,
        time_measurement_log_path=time_measurement_log_path,
    )
    def upload_model(self, num_chunks: int, payload: bytes):
        # Send the payload to the aggregation server
        url = f"http://{self.key_aggregation_server_ip}:{self.key_aggregation_server_port}/secure-fl/upload"

        headers = {
            "Content-Type": "application/octet-stream",
            "X-Client-Name": f"{self.client_name}",
            "X-Chunk-Total": str(num_chunks),
        }

        size = len(payload)
        with tqdm(
            total=size,
            unit="B",
            unit_scale=True,
            desc=f"Uploading: ciphertext.txt for client {self.client_name}",
            ascii=True,
        ) as pbar:
            _stream_upload(url, headers, _iter_bytes(payload, pbar))

        logging.info(f"Uploaded shielded model of {self.client_name}: ({size} bytes).")
        unique_id = sha256(
            f"{inspect.stack()[0][3]}_{time.time()}".encode()
        ).hexdigest()[:8]
        with open(transmitted_bytes_log_path, "w") as f:
            transmitted_bytes_data[f"shielded_model_upload_{unique_id}"] = size
            json.dump(transmitted_bytes_data, f, indent=4)

    @measure_time(
        time_measurement_data=time_measurement_data,
        time_measurement_log_path=time_measurement_log_path,
    )
    def download_model(self, num_chunks: int) -> io.BytesIO:
        # Download the aggregated encrypted model from the aggregation server
        url = f"http://{self.key_aggregation_server_ip}:{self.key_aggregation_server_port}/secure-fl/download-aggregate"

        headers = {
            "Content-Type": "application/octet-stream",
            "X-Client-Name": f"{self.client_name}",
            "X-Chunk-Total": str(num_chunks),
        }

        with httpx.Client(timeout=None) as client:
            with client.stream("GET", url, headers=headers) as resp:

                total = int(resp.headers.get("content-length") or 0)
                buf = io.BytesIO()

                with tqdm(
                    total=total if total > 0 else None,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading: aggregated ciphertext",
                    ascii=True,
                ) as pbar:
                    for chunk in resp.iter_bytes(self.chunk_size):
                        if chunk:
                            buf.write(chunk)
                            pbar.update(len(chunk))

        buf.seek(0)
        size = buf.getbuffer().nbytes
        unique_id = sha256(
            f"{inspect.stack()[0][3]}_{time.time()}".encode()
        ).hexdigest()[:8]
        with open(transmitted_bytes_log_path, "w") as f:
            transmitted_bytes_data[f"shielded_model_upload_{unique_id}"] = size
            json.dump(transmitted_bytes_data, f, indent=4)
        logging.info(f"Model downloaded to RAM (size={size} bytes)")
        return buf

    @measure_time(
        time_measurement_data=time_measurement_data,
        time_measurement_log_path=time_measurement_log_path,
    )
    def aggregate(self) -> Module:
        payload, mapping_dict, ignored_dict, num_chunks = (
            self.__create_encrypted_payload()
        )

        # Upload the encrypted model to the aggregation server
        self.upload_model(num_chunks, payload)

        del payload

        # Wait for the aggregation server to finish aggregation
        while True:
            url = f"http://{self.key_aggregation_server_ip}:{self.key_aggregation_server_port}/secure-fl/model-aggregation-completed"

            response = requests.get(
                url,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Client-Name": f"{self.client_name}",
                    "X-Chunk-Total": str(num_chunks),
                },
            )

            body = response.json()
            if body.get("model_aggregation_completed"):
                logging.info("Aggregation done. Proceeding to download...")
                break
            time.sleep(2)
            logging.info("Waiting for aggregation to be done...")

        # Download the aggregated encrypted model from the aggregation server
        buf = self.download_model(num_chunks)

        aggregated_state_dict = self.__load_encrypted_payload(
            buf.read(), mapping_dict, ignored_dict
        )

        model = self.model.load_state_dict(aggregated_state_dict)

        url = f"http://{self.key_aggregation_server_ip}:{self.key_aggregation_server_port}/secure-fl/mark-aggregation-download-complete"

        response = requests.post(url, headers={"X-Client-Name": f"{self.client_name}"})

        return model
