# deki-smpc

**deki-smpc** is a lightweight client for a Secure Multi-Party Computation (SMPC)-based Federated Learning (FL) framework. Unlike traditional FL frameworks (e.g., Flower) that force you into rigid training loops, **deki-smpc** lets you **seamlessly integrate privacy-preserving aggregation** into **your own training workflow** — whether you're using custom frameworks like **nnU-Net** or any other deep learning stack.

## ✨ Key Features

- **Minimal Invasion**: No need to refactor your training loop! Just plug **deki-smpc** into your existing code.
- **Dual-Server Architecture**:
  - **Key Exchange Server**: Fast, RESTful key generation (built with FastAPI).
  - **FL Aggregation Server**: Secure model aggregation (leveraging Flower + PyTorch-based SMPC).
- **Efficient Key Generation**: Key generation happens **in parallel** with model training, avoiding extra overhead.
- **Security First**: Uses a preshared secret and SMPC protocols to guarantee data privacy across participants.
- **Flexible and Lightweight**: Focused, extensible, and easy to integrate.

## 📦 System Components

- **Key Exchange Server**  
  REST API using **FastAPI** for efficient and secure multi-party key generation.

- **FL Aggregation Server**  
  Built on **Flower** with **PyTorch-based Secure Aggregation** — ensuring privacy without sacrificing flexibility.

## 🚀 Quick Usage Example

First you need to startup a [deki-smpc](https://github.com/MIC-DKFZ/deki-smpc-server) aggregation server.

```python
from deki_smpc import FedAvgClient

# Initialize the Deki SMPC client
client = FedAvgClient(
    key_aggregation_server_ip="127.0.0.1",
    key_aggregation_server_port=5000,
    fl_aggregation_server_ip="127.0.0.1",
    fl_aggregation_server_port=5001,
    num_clients=3,
    preshared_secret="my_secure_presHared_secret_123!",
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