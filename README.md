# deki-smpc

**deki-smpc** is a lightweight client for a Secure Multi-Party Computation (SMPC)-based Federated Learning (FL) framework. Unlike traditional FL frameworks (e.g., Flower) that force you into rigid training loops, **deki-smpc** lets you **seamlessly integrate privacy-preserving aggregation** into **your own training workflow** — whether you're using custom frameworks like **nnU-Net** or any other deep learning stack.

## ✨ Key Features

- **Minimal Invasion**: No need to refactor your training loop! Just plug **deki-smpc** into your existing code.
- **Dual-Server Architecture**:
  - **Key Exchange Server**: Fast, RESTful key generation (built with FastAPI).
  - **FL Aggregation Server**: Secure model aggregation (leveraging FastAPI (used to be Flower) + PyTorch-based SMPC).
- **Efficient Key Generation**: Key generation happens **in parallel** with model training, avoiding extra overhead.
- **Security First**: Uses a preshared secret and SMPC protocols to guarantee data privacy across participants.
- **Flexible and Lightweight**: Focused, extensible, and easy to integrate.

## 📦 System Components

- **Key Exchange Server**  
  REST API using **FastAPI** for efficient and secure multi-party key generation.

- **FL Aggregation Server**  
  Built on **FastAPI** with **PyTorch-based Secure Aggregation** — ensuring privacy without sacrificing flexibility (used to be Flower).

## 🚀 Quick Usage Example

##### 1. Create a virtual environment:

deki-smpc supports Python 3.10+ and works with Conda, pip, or any other virtual environment. Here’s an example using Conda:

```
conda create -n deki-smpc python=3.10
conda activate deki-smpc
```

##### 2. Install this repository

Clone and install this repository:

```bash
git clone https://github.com/MIC-DKFZ/deki-smpc
cd deki-smpc
pip install -e .
```

##### 3. Getting started with the server

First you need to startup a [deki-smpc](https://github.com/MIC-DKFZ/deki-smpc-server) aggregation server.

##### 4. Getting started with the client

```python
from deki_smpc import FedAvgClient

# Initialize the Deki SMPC client
client = FedAvgClient(
    aggregation_server_ip="127.0.0.1",
    aggregation_server_port=8080,
    num_clients=4,
    preshared_secret="my_secure_presHared_secret_123!",
    client_name=client_name,  # For better logging at the server. MUST BE UNIQUE ACROSS ALL CLIENTS
    model=local_model, # PyTorch model
)

# Use it to securely aggregate your model
aggregated_model = client.update_model(local_model)
```

✅ That's it! Integrate it wherever you train your models.

## 🔥 Why deki-smpc?

Most FL frameworks dictate your workflow.
deki-smpc empowers you to keep your framework (nnU-Net, MONAI, custom PyTorch loops, etc.) while adding secure federated aggregation with minimal changes.

You focus on training great models.
We handle secure aggregation.

## 📚 Documentation

Coming soon! (Stay tuned.)

## ⚠️ Work in Progress

This is currently a work in progress to transfer the implementation used for the experiments in the paper into a clean and fully usable implementation.

## Cite

> Hamm, B., Kirchhoff, Y., Rokuss, M., Schader, P., Neher, P., Parampottupadam, S., Floca, R., Maier-Hein, K. (2025). Efficient Privacy-Preserving Medical Cross-Silo Federated Learning. https://doi.org/10.36227/techrxiv.174650601.13181048/v1

Link: [![TechRxiv](https://d197for5662m48.cloudfront.net/images/institution/26407/whitelabel_logo_image/5cfcfb7bb5078907655a652ac0444b47.png)](https://doi.org/10.36227/techrxiv.174650601.13181048/v1)