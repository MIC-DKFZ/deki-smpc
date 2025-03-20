# deki-smpc

## Usage

Should be as easy as this:

```
client = FedAvgClient(
    key_aggregation_server_ip="127.0.0.1",
    key_aggregation_server_port=5000,
    fl_aggregation_server_ip="127.0.0.1",
    fl_aggregation_server_port=5001,
    num_clients=3,
    preshared_secret="my_secure_presHared_secret_123!",
)

client.submit_model()
client.receive_aggregated_model()
```