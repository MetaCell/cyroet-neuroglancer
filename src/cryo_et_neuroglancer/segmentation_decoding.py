import struct
from ctypes import LittleEndianStructure, c_uint64
from math import ceil

import numpy as np

from .chunk import Chunk
from .utils import get_grid_size_from_block_shape

# The units for the offsets are in 32-bit words
OFFSET_BYTES = 4
BYTES_PER_DATA_VALUE = 4
LEAST_SIGNIFICANT_24_BITS = 0x00FFFFFF
ALLOWED_ENCODED_BITS = (0, 1, 2, 4, 8, 16, 32)


class BlockHeader(LittleEndianStructure):
    _fields_ = [
        ("lookup_table_offset", c_uint64, 24),
        ("encoded_bits", c_uint64, 8),
        ("encoded_values_offset", c_uint64, 32),
    ]


# def _verify_bits(encoded_bits: int) -> int:
#     if encoded_bits not in ALLOWED_ENCODED_BITS:
#         raise ValueError(
#             f"The encoded bits must one of {ALLOWED_ENCODED_BITS} but got {encoded_bits}"
#         )
#     return encoded_bits


# def _verify_encoded_values_offset(
#     encoded_values_offset: int, encoded_bits: int, chunk_size: int
# ) -> int:
#     if encoded_bits != 0 and encoded_values_offset > chunk_size:
#         raise ValueError(
#             f"The encoded values offset must be less than the chunk length but got {encoded_values_offset} and {chunk_size}"
#         )
#     return encoded_values_offset


# def _verify_lookup_table_offset(lookup_table_offset: int, chunk_size: int) -> int:
#     if lookup_table_offset > chunk_size:
#         raise ValueError(
#             f"The lookup table offset must be less than the chunk length but got {lookup_table_offset} and {chunk_size}"
#         )
#     return lookup_table_offset


def _unpack_encoded_values(
    packed_values: bytes | bytearray, bits: int, nb_values: int
) -> np.ndarray:
    assert bits > 0, "Cannot decode packed values encoded using 0 bits by values"
    assert 32 % bits == 0

    values_per_word = 32 // bits
    mask = (1 << bits) - 1

    values = (r[0] for r in struct.iter_unpack("<I", packed_values))
    res = []
    for intval in values:
        res.extend(
            (intval >> (shift * bits)) & mask for shift in range(values_per_word)
        )
    return np.array(res, dtype="I")


# def ceil_div(a, b):
#     """Ceil integer division (``ceil(a / b)`` using integer arithmetic)."""
#     return (a - 1) // b + 1


# def _unpack_encoded_values(packed_values, bits, num_values):
#     assert bits > 0
#     assert 32 % bits == 0
#     bitmask = (1 << bits) - 1
#     values_per_32bit = 32 // bits
#     padded_values = np.empty(
#         values_per_32bit * ceil_div(num_values, values_per_32bit), dtype="I"
#     )
#     packed_values = np.frombuffer(packed_values, dtype="<I")
#     for shift in range(values_per_32bit):
#         padded_values[shift::values_per_32bit] = (
#             packed_values >> (shift * bits)
#         ) & bitmask
#     return padded_values[:num_values]


def _unpad_block(
    z: int,
    y: int,
    x: int,
    block: np.ndarray,
    block_shape: tuple[int, int, int],
    chunk_shape: tuple[int, int, int],
) -> np.ndarray:
    """Unpad the block to the given block shape"""
    zmax = min(block_shape[0], chunk_shape[0] - z * block_shape[0])
    ymax = min(block_shape[1], chunk_shape[1] - y * block_shape[1])
    xmax = min(block_shape[2], chunk_shape[2] - x * block_shape[2])
    return block[:zmax, :ymax, :xmax]


def _decode_block_header(block: bytearray, block_offset: int) -> BlockHeader:
    return BlockHeader.from_buffer(block, block_offset)


def _decode_lookup_table(
    block: bytearray, lookup_table_offset: int, encoded_bits: int
) -> np.ndarray:
    lookup_table_start = lookup_table_offset * BYTES_PER_DATA_VALUE
    lookup_table_end = lookup_table_start + (2**encoded_bits) * BYTES_PER_DATA_VALUE
    dtype = np.dtype(np.uint32).newbyteorder("<")
    return np.frombuffer(block[lookup_table_start:lookup_table_end], dtype=dtype)


def _extract_encoded_values(
    block: bytearray, encoded_values_offset: int, encoded_bits: int, nb_values: int
) -> np.ndarray:
    block_size = int(np.prod(nb_values))
    values_per_word = 32 // encoded_bits
    values_per_block = ceil(block_size / values_per_word)
    encoded_values_end = encoded_values_offset + BYTES_PER_DATA_VALUE * values_per_block
    packed_values = block[encoded_values_offset:encoded_values_end]
    unpacked_encoded_values = _unpack_encoded_values(
        packed_values, encoded_bits, nb_values
    )
    return unpacked_encoded_values


def _decode_encoded_values(
    block: bytearray,
    encoded_values_offset: int,
    encoded_bits: int,
    lookup_table: np.ndarray,
    block_shape: tuple[int, int, int],
) -> np.ndarray:
    if encoded_bits == 0:
        return np.full(block_shape, lookup_table[0], dtype=np.uint32)

    block_size = int(np.prod(block_shape))
    encoded_values = _extract_encoded_values(
        block, encoded_values_offset, encoded_bits, nb_values=block_size
    )
    return lookup_table[encoded_values].reshape(block_shape)


def _decode_block(
    block: bytearray, block_offset: int, block_shape: tuple[int, int, int]
) -> np.ndarray:
    header: BlockHeader = _decode_block_header(block, block_offset)
    lookup_table = _decode_lookup_table(
        block, header.lookup_table_offset, header.encoded_bits
    )
    decoded_values = _decode_encoded_values(
        block,
        header.encoded_values_offset * OFFSET_BYTES,
        header.encoded_bits,
        lookup_table,
        block_shape,
    )
    return decoded_values


def _decode_chunk_header(bytearray_chunk: bytearray):
    return struct.unpack("<I", bytearray_chunk[:4])[0]


def decode_chunk(chunk: Chunk, block_size: tuple[int, int, int]) -> np.ndarray:
    """Decode the given chunk

    Parameters
    ----------
    chunk : np.ndarray
        The chunk to decode
    encoded_bits : int
        The number of bits used to encode the chunk

    Returns
    -------
    np.ndarray
        The decoded chunk
    """
    chunk_shape = chunk.shape

    all_decoded_values = np.zeros(chunk_shape, dtype=np.uint32)
    gz, gy, gx = get_grid_size_from_block_shape(chunk_shape, block_size)

    nb_channels = _decode_chunk_header(chunk.buffer)
    chunk_array = chunk.buffer[nb_channels * BYTES_PER_DATA_VALUE :]
    for z, y, x in np.ndindex(gz, gy, gx):
        block_offset = 8 * (x + gx * (y + gy * z))
        decoded_values = _decode_block(chunk_array, block_offset, block_size)
        decoded_values = _unpad_block(z, y, x, decoded_values, block_size, chunk_shape)
        all_decoded_values[
            z * block_size[0] : (z + 1) * block_size[0],
            y * block_size[1] : (y + 1) * block_size[1],
            x * block_size[2] : (x + 1) * block_size[2],
        ] = decoded_values

    return all_decoded_values
