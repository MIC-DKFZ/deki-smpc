import json
import os

PATH_TO_DATA = "/home/b556m/projects/deki-smpc/example"
folders = ["50_100k", "50_100k_cpu", "50_100k_cpu_no_logs"]
min = None
for folder in folders:
    path = os.path.join(PATH_TO_DATA, folder)
    for json_file in os.listdir(path):
        if json_file.endswith(".json"):
            full_path = os.path.join(path, json_file)
            with open(full_path, "r") as f:
                data = json.load(f)

            # find key starting with "aggregate_"
            aggregate_keys = [
                key for key in data.keys() if key.startswith("aggregate_")
            ]
            for key in aggregate_keys:
                if min is None or data[key]["duration"] < min:
                    min = data[key]["duration"]

print(f"Minimum aggregation time: {min:.2f} seconds")
