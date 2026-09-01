"""Utility helpers for fixed-point encoding and key mask generation."""

import math

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
        if isinstance(precision_bits, bool) or not isinstance(precision_bits, int) or not 0 <= precision_bits <= 62:
            raise ValueError(f"precision_bits must be an integer between 0 and 62, got {precision_bits!r}")
        int64_max = torch.iinfo(torch.int64).max
        if (
            isinstance(max_aggregation_terms, bool)
            or not isinstance(max_aggregation_terms, int)
            or max_aggregation_terms <= 0
            or max_aggregation_terms > int64_max
        ):
            raise ValueError(
                f"max_aggregation_terms must be an integer between 1 and {int64_max}, got {max_aggregation_terms!r}"
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
            float64_encoded_magnitude = math.nextafter(float64_encoded_magnitude, -math.inf)
        self._safe_float64_encoded_magnitude = float64_encoded_magnitude

    @staticmethod
    def nearest_int_division(tensor: torch.Tensor, integer: int) -> torch.Tensor:
        """Divide an integer tensor with nearest rounding and sign correction."""

        if integer <= 0:
            raise ValueError(f"integer must be positive, got {integer}")

        if not FixedPointConverter.is_int_tensor(tensor):
            raise TypeError(f"input must be a LongTensor, got {type(tensor)}")

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
            torch.bfloat16,
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

    def encode(self, tensor: torch.Tensor, *, tensor_name: str | None = None) -> torch.Tensor:
        """Convert a float tensor to fixed-point integer representation."""
        if not FixedPointConverter.is_float_tensor(tensor):
            raise TypeError(f"Input must be float tensor, got {type(tensor)}.")

        label = "tensor" if tensor_name is None else f"tensor {tensor_name!r}"
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"Cannot encode {label}: values must all be finite")

        rounded = (tensor.double() * self.scale).round()
        if not bool(torch.isfinite(rounded).all().item()):
            raise OverflowError(
                f"Cannot encode {label}: scaling by 2**{self.precision_bits} produced a non-finite value"
            )

        observed_magnitude = float(rounded.abs().max().item()) if rounded.numel() else 0.0
        if observed_magnitude > self._safe_float64_encoded_magnitude:
            max_value_magnitude = self._safe_float64_encoded_magnitude / self.scale
            raise OverflowError(
                f"Cannot encode {label} safely: maximum absolute value exceeds "
                f"{max_value_magnitude:.17g} at {self.precision_bits}-bit precision "
                f"with {self.max_aggregation_terms} aggregation terms"
            )

        return rounded.long()

    def normalize_integer(self, tensor: torch.Tensor, *, tensor_name: str | None = None) -> torch.Tensor:
        """Validate an integer tensor's aggregation range and return int64."""
        if not FixedPointConverter.is_int_tensor(tensor):
            raise TypeError(f"Input must be int tensor, got {type(tensor)}.")

        normalized = tensor.to(dtype=torch.int64)
        if not normalized.numel():
            return normalized

        label = "tensor" if tensor_name is None else f"tensor {tensor_name!r}"
        exceeds_bound = (normalized > self.max_encoded_magnitude) | (normalized < -self.max_encoded_magnitude)
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
            raise TypeError(f"Input must be int tensor, got {type(tensor)}.")

        return (tensor.double() / self.scale).float()
