import subprocess
import sys


def start_clients(n):
    for i in range(n):
        client_name = f"client_{i}"
        command = f"{sys.executable} ./deki_smpc/clients.py --client_name {client_name}"
        print(f"Starting {client_name} in a new terminal...")

        # Launch in a new gnome-terminal window, keeping it open after execution
        subprocess.Popen(
            ["gnome-terminal", "--", "bash", "-c", f"{command}; exec bash"]
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n", type=int, required=True, help="Number of clients to start"
    )
    args = parser.parse_args()

    start_clients(args.n)
