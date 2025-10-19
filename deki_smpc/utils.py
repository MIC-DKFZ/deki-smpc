import logging
import struct
import time

import httpx
import torch
from openfhe import *
from tqdm import tqdm

CHUNK_SIZE = 1024 * 1024  # 1MB

_MAGIC = b"CHNK"  # 4 bytes
_VERSION = 1  # 1 byte
_HDR_FMT = ">4sB"  # magic, version
_CNT_FMT = ">I"  # uint32: number of chunks
_LEN_FMT = ">Q"  # uint64: per-chunk length


class FixedPointConverter:
    def __init__(self, precision_bits=16):
        self.precision_bits = precision_bits
        self.scale = 2**precision_bits

    def encode(self, value: float) -> int:
        """Convert a float to fixed-point integer."""
        if not isinstance(value, (float, int)):
            raise TypeError(f"Input must be float or int, got {type(value)}")
        return int(round(value * self.scale))

    def decode(self, value: int) -> float:
        """Convert fixed-point integer back to float."""
        if not isinstance(value, int):
            raise TypeError(f"Input must be int, got {type(value)}")
        return value / self.scale


def measure_time(func):
    """
    Decorator to measure the execution time of a function.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(
            f"Execution time for {func.__name__}: {end_time - start_time:.2f} seconds"
        )
        return result

    return wrapper


def pack_chunks(chunks):
    """
    Pack an iterable of bytes-like objects into a single bytes payload.

    Wire format:
      [MAGIC=CHNK][VERSION=1][COUNT:u32][ (LEN:u64)(DATA) ... repeated COUNT times ]
    """
    # Normalize to bytes and validate types early
    bchunks = []
    for idx, ch in enumerate(chunks):
        if not isinstance(ch, (bytes, bytearray, memoryview)):
            raise TypeError(f"Chunk {idx} is not bytes-like")
        bchunks.append(bytes(ch))

    parts = []
    parts.append(struct.pack(_HDR_FMT, _MAGIC, _VERSION))
    parts.append(struct.pack(_CNT_FMT, len(bchunks)))
    for ch in bchunks:
        parts.append(struct.pack(_LEN_FMT, len(ch)))
        parts.append(ch)
    return b"".join(parts)


def unpack_chunks(blob):
    """
    Unpack a bytes payload produced by pack_chunks back into a list of bytes.
    Raises ValueError if the blob is malformed.
    """
    mv = memoryview(blob)
    offset = 0

    # Header
    if len(mv) < struct.calcsize(_HDR_FMT) + struct.calcsize(_CNT_FMT):
        raise ValueError("Blob too small")

    magic, version = struct.unpack_from(_HDR_FMT, mv, offset)
    offset += struct.calcsize(_HDR_FMT)

    if magic != _MAGIC:
        raise ValueError("Bad magic header")
    if version != _VERSION:
        raise ValueError(f"Unsupported version {version}")

    (count,) = struct.unpack_from(_CNT_FMT, mv, offset)
    offset += struct.calcsize(_CNT_FMT)

    chunks = []
    for i in range(count):
        if offset + struct.calcsize(_LEN_FMT) > len(mv):
            raise ValueError(f"Truncated before chunk {i} length")
        (length,) = struct.unpack_from(_LEN_FMT, mv, offset)
        offset += struct.calcsize(_LEN_FMT)

        end = offset + length
        if end > len(mv):
            raise ValueError(f"Truncated in chunk {i} data")
        chunks.append(bytes(mv[offset:end]))
        offset = end

    if offset != len(mv):
        raise ValueError("Trailing bytes after last chunk")

    return chunks


def chunk_list(flat_list, max_length=8192):
    """
    Splits a long list into chunks of size max_length.

    Args:
        flat_list (list[float]): the input list
        max_length (int): maximum length of each chunk

    Returns:
        list[list[float]]: list of chunks
    """
    return [flat_list[i : i + max_length] for i in range(0, len(flat_list), max_length)]


def unchunk_list(chunks):
    """
    Flattens a list of lists back into a single list.

    Args:
        chunks (list[list[float]]): list of chunks

    Returns:
        list[float]: flattened list
    """
    return [x for chunk in chunks for x in chunk]


def flatten_state_dict(
    state_dict: dict[str, torch.Tensor], ignore_keys: list[str] = None
):
    if ignore_keys is None:
        ignore_keys = []

    flat_list = []
    mapping_dict = {}
    ignored_dict = {}

    offset = 0
    for key, tensor in state_dict.items():
        if key in ignore_keys:
            ignored_dict[key] = tensor.clone()
            continue

        values = tensor.flatten().tolist()
        n = len(values)

        mapping_dict[key] = {
            "shape": tensor.shape,
            "start": offset,
            "end": offset + n,
        }

        flat_list.extend(values)
        offset += n

    return flat_list, mapping_dict, ignored_dict


def list_of_float_to_int(flat_list: list[float]) -> list[int]:

    fpc = FixedPointConverter()

    for i, entry in enumerate(flat_list):
        flat_list[i] = fpc.encode(entry)

    return flat_list


def list_of_int_to_float(flat_list: list[int]) -> list[float]:

    fpc = FixedPointConverter()

    for i, entry in enumerate(flat_list):
        flat_list[i] = fpc.decode(entry)

    return flat_list


def reconstruct_state_dict(
    flat_list: list[float],
    mapping_dict: dict[str, dict],
    ignored_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    new_state_dict = {}

    for key, meta in mapping_dict.items():
        start, end = meta["start"], meta["end"]
        shape = meta["shape"]

        tensor_values = flat_list[start:end]
        tensor = torch.tensor(tensor_values, dtype=torch.float32).reshape(shape)

        new_state_dict[key] = tensor

    # restore ignored values
    for key, tensor in ignored_dict.items():
        new_state_dict[key] = tensor.clone()

    return new_state_dict


def _iter_bytes(b: bytes, pbar: tqdm):
    mv = memoryview(b)
    total = len(b)
    i = 0
    while i < total:
        j = i + CHUNK_SIZE
        chunk = mv[i:j]
        pbar.update(len(chunk))
        yield chunk
        i = j


def _stream_upload(url: str, headers: dict, content_iterable):
    # httpx >= 0.28 streaming upload path
    with httpx.Client(timeout=None) as client:
        with client.stream(
            "PUT", url, headers=headers, content=content_iterable
        ) as resp:
            resp.raise_for_status()
