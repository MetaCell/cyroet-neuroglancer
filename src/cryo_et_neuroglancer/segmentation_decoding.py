from math import ceil
import struct
import numpy as np

from cryo_et_neuroglancer.utils import get_grid_size_from_block_shape
from cryo_et_neuroglancer.chunk import Chunk

# The units for the offsets are in 32-bit words
OFFSET_BYTES = 4
BYTES_PER_DATA_VALUE = 4
LEAST_SIGNIFICANT_24_BITS = 0x00FFFFFF
ALLOWED_ENCODED_BITS = (0, 1, 2, 4, 8, 16, 32)


def _verify_bits(encoded_bits: int) -> int:
    if encoded_bits not in ALLOWED_ENCODED_BITS:
        raise ValueError(
            f"The encoded bits must one of {ALLOWED_ENCODED_BITS} but got {encoded_bits}"
        )
    return encoded_bits


def _verify_encoded_values_offset(
    encoded_values_offset: int, encoded_bits: int, chunk_size: int
) -> int:
    if encoded_bits != 0 and encoded_values_offset > chunk_size:
        raise ValueError(
            f"The encoded values offset must be less than the chunk length but got {encoded_values_offset} and {chunk_size}"
        )
    return encoded_values_offset


def _verify_lookup_table_offset(lookup_table_offset: int, chunk_size: int) -> int:
    if lookup_table_offset > chunk_size:
        raise ValueError(
            f"The lookup table offset must be less than the chunk length but got {lookup_table_offset} and {chunk_size}"
        )
    return lookup_table_offset


def _unpack_encoded_values(
    packed_values: np.ndarray, encoded_bits: int, block_size: int
) -> np.ndarray:
    # TODO this might not work just yet, only an idea
    values_per_word = 32 // encoded_bits
    return packed_values.reshape(-1, values_per_word)[:, :block_size].flatten()


def unpad_block(
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


def _decode_block_header(
    header: tuple[int, int], chunk_size: int
) -> tuple[int, int, int]:
    first_int, second_int = header
    # Pull the lookup offset from the least significant 24 bits
    lookup_table_offset = _verify_lookup_table_offset(
        OFFSET_BYTES * (first_int & LEAST_SIGNIFICANT_24_BITS), chunk_size
    )

    # Pull the encoded bits from the most significant 8 bits
    encoded_bits = _verify_bits(first_int >> 24)

    # Pull the encoded values offset from the second integer
    encoded_values_offset = _verify_encoded_values_offset(
        OFFSET_BYTES * second_int, encoded_bits, chunk_size
    )

    return lookup_table_offset, encoded_bits, encoded_values_offset


def _decode_lookup_table(
    chunk: Chunk, lookup_table_offset: int, encoded_bits: int
) -> np.ndarray:
    lookup_table_end = lookup_table_offset + (2**encoded_bits) * BYTES_PER_DATA_VALUE
    dtype = np.uint32
    dtype = dtype.newbyteorder("<")
    return np.frombuffer(
        chunk.buffer[lookup_table_offset:lookup_table_end], dtype=dtype
    )


def _decode_encoded_values(
    chunk: Chunk,
    encoded_values_offset: int,
    encoded_bits: int,
    lookup_table: np.ndarray,
    block_shape: tuple[int, int, int],
) -> np.ndarray:
    if encoded_bits == 0:
        encoded_values = np.full(block_shape, lookup_table[0], dtype=np.uint32)
    else:
        block_size = np.prod(block_shape)
        values_per_word = 32 // encoded_bits
        values_per_block = ceil(block_size / values_per_word)
        encoded_values_end = (
            encoded_values_offset + BYTES_PER_DATA_VALUE * values_per_block
        )
        dtype = np.uint32
        dtype = dtype.newbyteorder("<")
        # TODO ensure that the little endian is setup for other frombuffer calls
        packed_values = np.frombuffer(
            chunk.buffer[encoded_values_offset:encoded_values_end], dtype=dtype
        )
        encoded_values = _unpack_encoded_values(packed_values, encoded_bits, block_size)
        decoded_values = lookup_table[encoded_values].reshape(block_shape)
    return decoded_values


def _decode_block(
    chunk: Chunk, block_offset: int, chunk_size: int, block_shape: tuple[int, int, int]
) -> np.ndarray:
    header_values = struct.unpack_from("<II", chunk.buffer, block_offset)
    header: tuple[int, int] = (header_values[0], header_values[1])
    lookup_table_offset, encoded_bits, encoded_values_offset = _decode_block_header(
        header, chunk_size
    )

    lookup_table = _decode_lookup_table(chunk, lookup_table_offset, encoded_bits)
    decoded_values = _decode_encoded_values(
        chunk, encoded_values_offset, encoded_bits, lookup_table, block_shape
    )
    return decoded_values


def decode_chunk(chunk: Chunk, block_shape: tuple[int, int, int]) -> np.ndarray:
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
    # TODO this doesn't store the chunk values correctly yet
    all_decoded_values = np.zeros(chunk.shape, dtype=np.uint32)
    gz, gy, gx = get_grid_size_from_block_shape(chunk.shape, block_shape)

    for z, y, x in np.ndindex(gz, gy, gx):
        block_offset = 8 * (x + gx * (y + gy * z))
        decoded_values = _decode_block(chunk, block_offset, chunk.size, block_shape)
        decoded_values = unpad_block(z, y, x, decoded_values, block_shape, chunk.shape)
        all_decoded_values[
            z * block_shape[0] : (z + 1) * block_shape[0],
            y * block_shape[1] : (y + 1) * block_shape[1],
            x * block_shape[2] : (x + 1) * block_shape[2],
        ] = decoded_values

    return all_decoded_values
