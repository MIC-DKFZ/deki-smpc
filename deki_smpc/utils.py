import secrets
import string
from dataclasses import dataclass

import torch


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
    def generate_secure_random_mask(state_dict: dict) -> dict:
        """
        Generates a dictionary of secure random tensors with the same shape
        as the parameters in the given state_dict using the secrets module.
        """
        mask = {}
        for name, param in state_dict.items():
            num_elements = param.numel()
            num_bytes = num_elements * 4  # 4 bytes per 32-bit int
            random_bytes = bytes([secrets.randbelow(256) for _ in range(num_bytes)])
            writable_bytes = bytearray(random_bytes)  # Make the buffer writable
            random_tensor = torch.frombuffer(writable_bytes, dtype=torch.int32)
            mask[name] = random_tensor.view(param.shape)
        return mask

    @staticmethod
    def dummy_generate_secure_random_mask(state_dict: dict) -> dict:
        """
        Generates a dictionary of tensors with ones with the same shape
        as the parameters in the given state_dict.
        """
        mask = {}
        for name, param in state_dict.items():
            mask[name] = torch.ones_like(param)
        return mask
