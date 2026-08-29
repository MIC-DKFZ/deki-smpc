import io

import pytest
import torch

from deki_smpc.clients import FedAvgClient
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
    assert converter.max_aggregation_terms == 1


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


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_encode_rejects_nonfinite_values(value: float) -> None:
    converter = FixedPointConverter()

    with pytest.raises(ValueError, match="values must all be finite"):
        converter.encode(torch.tensor([value]), tensor_name="weight")


def test_encode_reserves_headroom_for_the_aggregated_sum() -> None:
    converter = FixedPointConverter(
        precision_bits=60,
        max_aggregation_terms=4,
    )

    assert torch.equal(
        converter.encode(torch.tensor([1.0], dtype=torch.float64)),
        torch.tensor([2**60], dtype=torch.int64),
    )

    with pytest.raises(
        OverflowError,
        match="60-bit precision with 4 aggregation terms",
    ):
        converter.encode(
            torch.tensor([3.0], dtype=torch.float64), tensor_name="large_weight"
        )


def test_encode_rejects_values_that_would_overflow_during_cast() -> None:
    converter = FixedPointConverter()

    with pytest.raises(OverflowError, match="Cannot encode tensor safely"):
        converter.encode(torch.tensor([1e300], dtype=torch.float64))


def test_integer_normalization_reserves_aggregation_headroom() -> None:
    converter = FixedPointConverter(max_aggregation_terms=4)

    normalized = converter.normalize_integer(torch.tensor([17], dtype=torch.int32))

    assert normalized.dtype == torch.int64
    assert torch.equal(normalized, torch.tensor([17], dtype=torch.int64))
    with pytest.raises(OverflowError, match="Cannot aggregate tensor 'counter'"):
        converter.normalize_integer(
            torch.tensor([torch.iinfo(torch.int64).max]), tensor_name="counter"
        )


def test_int64_mask_wraparound_cancels_during_unmasking() -> None:
    encoded = torch.tensor([7], dtype=torch.int64)
    mask = torch.tensor([torch.iinfo(torch.int64).max], dtype=torch.int64)

    shielded = encoded + mask
    unshielded = shielded - mask

    assert torch.equal(unshielded, encoded)


def test_client_shielding_keeps_float_masks_in_the_int64_ring() -> None:
    client = object.__new__(FedAvgClient)
    client.device = torch.device("cpu")
    client.fpe = FixedPointConverter(max_aggregation_terms=4)
    client.ignore_model_keys = []
    state_dict = {"weight": torch.tensor([1.25], dtype=torch.float32)}
    mask = {"weight": torch.tensor([1.0], dtype=torch.float32)}

    shielded, encoded_mask = client._FedAvgClient__shield_key(state_dict, mask)

    assert shielded["weight"].dtype == torch.int64
    assert encoded_mask["weight"].dtype == torch.int64
    assert torch.equal(
        shielded["weight"],
        client.fpe.encode(state_dict["weight"]) + client.fpe.encode(mask["weight"]),
    )
    assert state_dict["weight"].dtype == torch.float32
    assert mask["weight"].dtype == torch.float32


def test_client_unshielding_normalizes_float_masks_before_subtraction() -> None:
    client = object.__new__(FedAvgClient)
    client.device = torch.device("cpu")
    client.fpe = FixedPointConverter(max_aggregation_terms=4)
    client.ignore_model_keys = []
    encoded_value = client.fpe.encode(torch.tensor([1.25]))
    float_mask = {"weight": torch.tensor([1.0])}
    shielded = {"weight": encoded_value + client.fpe.encode(float_mask["weight"])}

    unshielded = client._FedAvgClient__unshield_key(shielded, float_mask)

    assert torch.equal(unshielded["weight"], torch.tensor([1.25]))


def test_masked_multi_client_sum_stays_correct_in_int64_ring() -> None:
    num_clients = 4
    client = object.__new__(FedAvgClient)
    client.device = torch.device("cpu")
    client.fpe = FixedPointConverter(max_aggregation_terms=num_clients)
    client.ignore_model_keys = []
    client_weights = [
        torch.tensor([1.25, -2.5]),
        torch.tensor([0.5, 4.0]),
        torch.tensor([-1.0, 0.25]),
        torch.tensor([2.0, -0.75]),
    ]
    private_mask = {"weight": torch.ones(2)}

    shielded_models = [
        client._FedAvgClient__shield_key({"weight": weights}, private_mask)[0]
        for weights in client_weights
    ]
    server_sum = {
        "weight": sum(
            (model["weight"] for model in shielded_models),
            torch.zeros(2, dtype=torch.int64),
        )
    }
    public_mask = {"weight": torch.full((2,), float(num_clients))}

    assert server_sum["weight"].dtype == torch.int64
    decoded_sum = client._FedAvgClient__unshield_key(server_sum, public_mask)
    decoded_average = decoded_sum["weight"] / num_clients

    assert torch.equal(decoded_average, torch.stack(client_weights).mean(dim=0))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"precision_bits": -1}, "precision_bits"),
        ({"precision_bits": 63}, "precision_bits"),
        ({"precision_bits": True}, "precision_bits"),
        ({"max_aggregation_terms": 0}, "max_aggregation_terms"),
        ({"max_aggregation_terms": True}, "max_aggregation_terms"),
    ],
)
def test_converter_rejects_invalid_range_configuration(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        FixedPointConverter(**kwargs)


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
