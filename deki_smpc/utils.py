import secrets
import string
from dataclasses import dataclass
import subprocess
import torch
import itertools


@dataclass
class KeyPair:
    round: int = 0
    # This is a tensor of same shape as the model parameters
    private_encryption_key: dict = None
    # This is a tensor of same shape as the model parameters
    shared_decryption_key: dict = None


# Only precompute the keys for the next round
@dataclass
class KeyQueue:
    this_round: KeyPair = None
    next_round: KeyPair = None


class SecurityUtils:

    def __init__(self):
        self.key_queue = KeyQueue()

    @staticmethod
    def generate_preshared_secret(length: int = 32) -> str:
        """Generate a cryptographically secure preshared secret.

        - Must be at least `length` characters long (default: 32)
        - Contains at least one uppercase letter, one digit, and one special character.
        - Uses `secrets` for true randomness.
        """
        if length < 16:
            raise ValueError("Secret length must be at least 16 characters")

        # Securely select one character from each required category
        uppercase = secrets.choice(string.ascii_uppercase)
        digit = secrets.choice(string.digits)
        special = secrets.choice(string.punctuation)

        # Generate the remaining characters securely
        all_characters = string.ascii_letters + string.digits + string.punctuation
        remaining_chars = "".join(
            secrets.choice(all_characters) for _ in range(length - 3)
        )

        # Combine and shuffle securely
        secret = list(uppercase + digit + special + remaining_chars)
        secrets.SystemRandom().shuffle(secret)

        return "".join(secret)

    @staticmethod
    def generate_secure_random_mask(model: torch.nn.Module) -> dict:
        """
        Generates a dictionary of secure random tensors with the same shape
        as the parameters of a given PyTorch model using the secrets module.
        """
        mask = {}
        for name, param in model.named_parameters():
            num_elements = param.numel()
            num_bytes = num_elements * 4  # 4 bytes per 32-bit int
            random_bytes = bytes([secrets.randbelow(256) for _ in range(num_bytes)])
            random_tensor = torch.frombuffer(random_bytes, dtype=torch.int32)
            mask[name] = random_tensor.view(param.shape)
        return mask
