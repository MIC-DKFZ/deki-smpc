"""Utility helpers for fixed-point encoding and key mask generation."""

import math
import os
import secrets
import string
from dataclasses import dataclass

import torch

TensorStateDict = dict[str, torch.Tensor]


class FixedPointConverter:
    """Encode/decode tensors between floating-point and fixed-point integer forms."""

    def __init__(
        self,
        precision_bits: int = 24,
        device: str | torch.device = "cpu",
        max_aggregation_terms: int = 1,
    ) -> None:
        """Create a converter with fixed-point precision and aggregation headroom.

        ``max_aggregation_terms`` reserves enough signed ``int64`` range for the
        sum of that many encoded tensors. Masking still operates in the int64
        ring, where intermediate wraparound is expected and cancels on unmasking.
        """
        if (
            isinstance(precision_bits, bool)
            or not isinstance(precision_bits, int)
            or not 0 <= precision_bits <= 62
        ):
            raise ValueError(
                "precision_bits must be an integer between 0 and 62, got %r"
                % precision_bits
            )
        int64_max = torch.iinfo(torch.int64).max
        if (
            isinstance(max_aggregation_terms, bool)
            or not isinstance(max_aggregation_terms, int)
            or max_aggregation_terms <= 0
            or max_aggregation_terms > int64_max
        ):
            raise ValueError(
                "max_aggregation_terms must be an integer between 1 and %d, got %r"
                % (int64_max, max_aggregation_terms)
            )

        self.precision_bits = precision_bits
        self.scale = int(2**precision_bits)
        self.device = device
        self.max_aggregation_terms = max_aggregation_terms
        self.max_encoded_magnitude = int64_max // max_aggregation_terms

        # Comparing int64 bounds in float64 can accidentally round the bound up.
        # If that happened, move down to the nearest representable safe value.
        float64_encoded_magnitude = float(self.max_encoded_magnitude)
        if float64_encoded_magnitude > self.max_encoded_magnitude:
            float64_encoded_magnitude = math.nextafter(
                float64_encoded_magnitude, -math.inf
            )
        self._safe_float64_encoded_magnitude = float64_encoded_magnitude

    @staticmethod
    def nearest_int_division(tensor: torch.Tensor, integer: int) -> torch.Tensor:
        """Divide an integer tensor with nearest rounding and sign correction."""

        if integer <= 0:
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
        """Return True when tensor has a floating-point dtype."""
        return torch.is_tensor(tensor) and tensor.dtype in [
            torch.float16,
            torch.float32,
            torch.float64,
        ]

    @staticmethod
    def is_int_tensor(tensor: torch.Tensor) -> bool:
        """Return True when tensor has an integer dtype."""
        return torch.is_tensor(tensor) and tensor.dtype in [
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ]

    def encode(
        self, tensor: torch.Tensor, *, tensor_name: str | None = None
    ) -> torch.Tensor:
        """Convert a float tensor to fixed-point integer representation."""
        if not FixedPointConverter.is_float_tensor(tensor):
            raise TypeError("Input must be float tensor, got %s." % type(tensor))

        label = "tensor" if tensor_name is None else f"tensor {tensor_name!r}"
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"Cannot encode {label}: values must all be finite")

        rounded = (tensor.double() * self.scale).round()
        if not bool(torch.isfinite(rounded).all().item()):
            raise OverflowError(
                f"Cannot encode {label}: scaling by 2**{self.precision_bits} "
                "produced a non-finite value"
            )

        observed_magnitude = (
            float(rounded.abs().max().item()) if rounded.numel() else 0.0
        )
        if observed_magnitude > self._safe_float64_encoded_magnitude:
            max_value_magnitude = self._safe_float64_encoded_magnitude / self.scale
            raise OverflowError(
                f"Cannot encode {label} safely: maximum absolute value exceeds "
                f"{max_value_magnitude:.17g} at {self.precision_bits}-bit precision "
                f"with {self.max_aggregation_terms} aggregation terms"
            )

        return rounded.long()

    def normalize_integer(
        self, tensor: torch.Tensor, *, tensor_name: str | None = None
    ) -> torch.Tensor:
        """Validate an integer tensor's aggregation range and return int64."""
        if not FixedPointConverter.is_int_tensor(tensor):
            raise TypeError("Input must be int tensor, got %s." % type(tensor))

        normalized = tensor.to(dtype=torch.int64)
        if not normalized.numel():
            return normalized

        label = "tensor" if tensor_name is None else f"tensor {tensor_name!r}"
        exceeds_bound = (normalized > self.max_encoded_magnitude) | (
            normalized < -self.max_encoded_magnitude
        )
        if bool(exceeds_bound.any().item()):
            raise OverflowError(
                f"Cannot aggregate {label} safely: absolute integer value exceeds "
                f"{self.max_encoded_magnitude} with "
                f"{self.max_aggregation_terms} aggregation terms"
            )
        return normalized

    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert a fixed-point integer tensor back to floating-point."""
        if not FixedPointConverter.is_int_tensor(tensor):
            raise TypeError("Input must be int tensor, got %s." % type(tensor))

        return (tensor.double() / self.scale).float()


@dataclass
class KeyPair:
    """Container for private and shared key tensors for a given round."""

    round: int = 0
    # This is a tensor of same shape as the model parameters
    private_encryption_key: TensorStateDict | None = None
    # This is a tensor of same shape as the model parameters
    shared_decryption_key: TensorStateDict | None = None


# Only precompute the keys for the next round
@dataclass
class KeyQueue:
    """Two-slot queue holding current and next key pairs."""

    this_round: KeyPair | None = None
    next_round: KeyPair | None = None


class SecurityUtils:
    """Helper functions for generating secrets and masking tensors."""

    def __init__(self) -> None:
        """Initialize an empty key queue."""
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
        state_dict: TensorStateDict,
    ) -> TensorStateDict:
        """
        Generates a dictionary of secure random tensors with the same shape
        as the parameters in the given state_dict using the secrets module.
        """
        # total number of int32s across all tensors
        total_i32 = sum(p.numel() for p in state_dict.values())
        buf = bytearray(os.urandom(total_i32 * 4))  # writable for frombuffer

        mask: TensorStateDict = {}
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
        state_dict: TensorStateDict,
    ) -> TensorStateDict:
        """
        Generates a dictionary of tensors with ones with the same shape
        as the parameters in the given state_dict.
        """
        mask: TensorStateDict = {}
        for name, param in state_dict.items():
            mask[name] = torch.ones_like(param)
        return mask
