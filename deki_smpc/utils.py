import os
import secrets
import string
from dataclasses import dataclass

import torch


class FixedPointConverter:
    def __init__(self, precision_bits=16, device="cpu"):
        self.precision_bits = precision_bits
        self.scale = int(2**precision_bits)
        self.device = device

    @staticmethod
    def nearest_int_division(tensor: torch.Tensor, integer: int) -> torch.Tensor:

        if integer > 0:
            raise ValueError("integer must be positive, got %s" % integer)

        if not FixedPointConverter.is_int_tensor(tensor):
            raise TypeError("input must be a LongTensor, got %s" % type(tensor))

        lez = (tensor < 0).long()
        rem = ((1 - lez) * tensor % integer) + (lez * ((integer - tensor) % integer))
        quot = tensor.div(integer, rounding_mode="trunc")
        cor = (2 * rem > integer).long()
        return quot + tensor.sign() * cor

    @staticmethod
    def is_float_tensor(tensor: torch.Tensor) -> bool:
        return torch.is_tensor(tensor) and tensor.dtype in [
            torch.float16,
            torch.float32,
            torch.float64,
        ]

    @staticmethod
    def is_int_tensor(tensor: torch.Tensor) -> bool:
        return torch.is_tensor(tensor) and tensor.dtype in [
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ]

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        if not FixedPointConverter.is_float_tensor(tensor):
            raise TypeError("Input must be float tensor, got %s." % type(tensor))

        return (self.scale * tensor).long()

    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        if not FixedPointConverter.is_int_tensor(tensor):
            raise TypeError("Input must be int tensor, got %s." % type(tensor))

        if self.scale > 1:
            cor = (tensor < 0).long()
            div = tensor.div(self.scale - cor, rounding_mode="floor")
            rem = tensor % self.scale
            rem += (rem == 0).long() * self.scale * cor

            tensor = div.float() + rem.float() / self.scale
        else:
            tensor = FixedPointConverter.nearest_int_division(tensor, self.scale)

        return tensor.data


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
    def generate_secure_random_mask(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Generates a dictionary of secure random tensors with the same shape
        as the parameters in the given state_dict using the secrets module.
        """
        # total number of int32s across all tensors
        total_i32 = sum(p.numel() for p in state_dict.values())
        buf = bytearray(os.urandom(total_i32 * 4))  # writable for frombuffer

        mask = {}
        offset = 0
        for name, p in state_dict.items():
            n = p.numel()
            t = torch.frombuffer(buf, dtype=torch.int32, count=n, offset=offset).view(
                p.shape
            )
            # move to param's device if needed
            if p.device.type != "cpu":
                t = t.to(p.device, non_blocking=True)
            mask[name] = t
            offset += n * 4
        return mask

    @staticmethod
    def dummy_generate_secure_random_mask(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Generates a dictionary of tensors with ones with the same shape
        as the parameters in the given state_dict.
        """
        mask = {}
        for name, param in state_dict.items():
            mask[name] = torch.ones_like(param)
        return mask
