import io

import pytest
import torch

from deki_smpc.utils import FixedPointConverter


def _legacy_encode(tensor: torch.Tensor, scale: int) -> torch.Tensor:
    return (scale * tensor).long()


def _legacy_decode(tensor: torch.Tensor, scale: int) -> torch.Tensor:
    tensor = tensor.clone()
    correction = (tensor < 0).long()
    quotient = tensor.div(scale - correction, rounding_mode="floor")
    remainder = tensor % scale
    remainder += (remainder == 0).long() * scale * correction
    return quotient.float() + remainder.float() / scale


def _serialized_size(tensor: torch.Tensor) -> int:
    buffer = io.BytesIO()
    torch.save({"weight": tensor}, buffer, _use_new_zipfile_serialization=True)
    return buffer.getbuffer().nbytes


def test_default_precision_is_24_bits() -> None:
    converter = FixedPointConverter()

    assert converter.precision_bits == 24
    assert converter.scale == 16_777_216


def test_encode_rounds_to_nearest_in_double_precision() -> None:
    converter = FixedPointConverter()
    scaled_values = torch.tensor(
        [-1.5, -0.51, -0.5, -0.49, 0.49, 0.5, 0.51, 1.5],
        dtype=torch.float64,
    )
    values = scaled_values / converter.scale

    encoded = converter.encode(values)

    assert torch.equal(
        encoded,
        torch.tensor([-2, -1, 0, 0, 0, 0, 1, 2], dtype=torch.int64),
    )


def test_encode_uses_double_precision_before_scaling() -> None:
    converter = FixedPointConverter()
    values = torch.tensor([1.0, -1.0, 1_000.0], dtype=torch.float16)

    encoded = converter.encode(values)

    assert torch.equal(
        encoded,
        torch.tensor(
            [converter.scale, -converter.scale, 1_000 * converter.scale],
            dtype=torch.int64,
        ),
    )


def test_round_trip_precision_improves_by_at_least_500x() -> None:
    weights = torch.randn(
        200_000, generator=torch.Generator().manual_seed(7), dtype=torch.float32
    )
    converter = FixedPointConverter()

    legacy = (_legacy_encode(weights, 2**16).double() / (2**16)).float()
    improved = converter.decode(converter.encode(weights))
    legacy_error = (legacy - weights).abs()
    improved_error = (improved - weights).abs()

    assert legacy_error.max() / improved_error.max() >= 500
    assert improved_error.max() <= 2**-25


def test_double_divide_matches_legacy_reconstruction_at_16_bits() -> None:
    scale = 2**16
    encoded = torch.tensor(
        [
            -2 * scale,
            -scale,
            -12_345,
            -1,
            0,
            1,
            12_345,
            scale,
            2 * scale,
            2**31,
        ],
        dtype=torch.int64,
    )
    converter = FixedPointConverter(precision_bits=16)

    assert torch.equal(converter.decode(encoded), _legacy_decode(encoded, scale))


def test_encoding_preserves_wire_dtype_shape_and_serialized_size() -> None:
    weights = torch.randn(
        (17, 19), generator=torch.Generator().manual_seed(11), dtype=torch.float32
    )
    converter = FixedPointConverter()

    legacy = _legacy_encode(weights, 2**16)
    improved = converter.encode(weights)

    assert improved.dtype == legacy.dtype == torch.int64
    assert improved.shape == legacy.shape
    assert improved.numel() == legacy.numel()
    assert improved.untyped_storage().nbytes() == legacy.untyped_storage().nbytes()
    assert _serialized_size(improved) == _serialized_size(legacy)


def test_decode_returns_float32_with_unchanged_shape() -> None:
    encoded = torch.tensor([[-2, -1, 0], [1, 2, 3]], dtype=torch.int64)

    decoded = FixedPointConverter().decode(encoded)

    assert decoded.dtype == torch.float32
    assert decoded.shape == encoded.shape


@pytest.mark.parametrize("integer", [0, -1, -17])
def test_nearest_int_division_rejects_nonpositive_integer(integer: int) -> None:
    with pytest.raises(ValueError, match=f"integer must be positive, got {integer}"):
        FixedPointConverter.nearest_int_division(
            torch.tensor([1], dtype=torch.int64), integer
        )


def test_nearest_int_division_accepts_positive_integer() -> None:
    tensor = torch.tensor([-7, -5, -4, 0, 4, 5, 7], dtype=torch.int64)

    divided = FixedPointConverter.nearest_int_division(tensor, 2)

    assert torch.equal(divided, torch.tensor([-3, -2, -2, 0, 2, 2, 3]))


@pytest.mark.parametrize(
    ("method", "value", "message"),
    [
        ("encode", torch.tensor([1], dtype=torch.int64), "Input must be float tensor"),
        ("decode", torch.tensor([1.0]), "Input must be int tensor"),
    ],
)
def test_converter_type_checks_are_preserved(
    method: str, value: torch.Tensor, message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        getattr(FixedPointConverter(), method)(value)
