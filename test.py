import requests
import json
# Send request to the key aggregation server get /tasks/active

key_aggregation_server_ip = "127.0.0.1"
key_aggregation_server_port = 8080
client_name = "client_18"

response = requests.get(
    url=f"http://{key_aggregation_server_ip}:{key_aggregation_server_port}/tasks/active"
)

print(response.json())

response = requests.get(
    url=f"http://{key_aggregation_server_ip}:{key_aggregation_server_port}/tasks/check_for_task",
    data=json.dumps(
        {
            "client_name": client_name,
        }
    ),
)

print(response.json())