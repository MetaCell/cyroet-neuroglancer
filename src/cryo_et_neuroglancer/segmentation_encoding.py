import struct

import numpy as np
import operator

import functools

from .utils import pad_block, get_grid_size_from_block_shape, number_of_encoding_bits
from cryo_et_neuroglancer.chunk import Chunk


def _get_buffer_position(buffer: bytearray) -> int:
    """Return the current position in the buffer"""
    assert len(buffer) % 4 == 0, "Buffer length must be a multiple of 4"
    return len(buffer) // 4


def _pack_encoded_values(values: np.ndarray, bits: int) -> bytes:
    """
    Pack the encoded values into 32bit unsigned integers

    To view the packed values as a numpy array, use the following:
    np.frombuffer(packed_values, dtype=np.uint32).view(f"u{encoded_bits}")

    Parameters
    ----------
    values : np.ndarray
        The values to encode
    bits : int
        The number of bits used to encode the values

    Returns
    -------
    packed_values : bytes
        The packed values

    Details
    -------

    Values are packed in a little endian 32bits, from LSB to MSB.
    Consequently, the first value of the array will be stored first from the LSB
    then, shifted to the left from the nb of bits necessary to encode the value,
    the next value from the array is considered.
    Each small encoded value are reduced in a huge 32bits using a simple bits | operator

    Here is an example for an array [1, 0, 2, 2, 1].
    There is only 3 different values here, so we need to encode each value on 2bits
    The result would then be:
    values    1   2   2   0   1
    encoded  01  10  10  00  01
    result 0b110100001 packed in a unsigned 32bit little endian
    """
    if bits == 0:
        return bytes()
    assert 32 % bits == 0
    assert np.array_equal(values, values & ((1 << bits) - 1))
    values_per_32bit = 32 // bits
    padded_values = np.pad(
        values.astype("<I", casting="unsafe"),
        [(0, -len(values) % values_per_32bit)],
        mode="constant",
        constant_values=0,
    )
    assert len(padded_values) % values_per_32bit == 0
    # packed_values: np.ndarray = functools.reduce(
    #     np.bitwise_or,
    #     (
    #         padded_values[shift::values_per_32bit] << (shift * bits)
    #         for shift in range(values_per_32bit)
    #     ),
    # )
    # return packed_values.tobytes()

    packed_values: int = functools.reduce(
        operator.or_,
        (value << (shift * bits) for shift, value in enumerate(padded_values)),
    )
    return struct.pack("<I", packed_values)


def get_back_values_from_buffer(bytes_: bytes) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the values from the given buffer

    This is for the encoded values in the neuroglancer segmentation format, so are in uint32.

    Parameters
    ----------
    bytes_ : bytes
        The buffer to get the values from

    Returns
    -------
    values : np.ndarray
        The values
    binary_representation : np.ndarray
        The binary representation of the values
    """
    values = np.frombuffer(bytes_, dtype=np.uint32).view(np.uint32)
    binary_representation = np.unpackbits(values.view(np.uint8), bitorder="little")
    return values, binary_representation


def _create_block_header(
    buffer: bytearray,
    lookup_table_offset: int,
    encoded_bits: int,
    encoded_values_offset: int,
    block_offset: int,
) -> None:
    """
    Create a block header (64-bit)

    First 24 bits are the lookup table offset (little endian)
    Next 8 bits are the number of bits used to encode the values
    Last 32 bits are the offset to the encoded values (little endian)
    All values are unsigned integers

    Parameters
    ----------
    buffer : bytearray
        The buffer to write the block header to
    lookup_table_offset : int
        The offset in the buffer to the lookup table for this block
    encoded_bits : int
        The number of bits used to encode the values
    encoded_values_offset : int
        The offset in the buffer to the encoded values for this block
    block_offset : int
        The offset in the buffer to the block header
    """
    struct.pack_into(
        "<II",
        buffer,
        block_offset,
        lookup_table_offset | (encoded_bits << 24),
        encoded_values_offset,
    )


def _create_lookup_table(
    buffer: bytearray,
    stored_lookup_tables: dict[bytes, tuple[int, int]],
    unique_values: np.ndarray,
) -> tuple[int, int]:
    """
    Create a lookup table for the given values

    Parameters
    ----------
    buffer : bytearray
        The buffer to write the lookup table to
    stored_lookup_tables : dict[bytes, int]
        A dictionary mapping values to their offset in the buffer
    unique_values : np.ndarray
        The values to write to the buffer
        Must be uint32 or uint64

    Returns
    -------
    lookup_table_offset : int
        The offset in the buffer to the lookup table for the given values
    encoded_bits : int
        The number of bits used to encode the values
    """
    unique_values = unique_values.astype(np.uint32)
    values_in_bytes = unique_values.tobytes()
    if values_in_bytes not in stored_lookup_tables:
        lookup_table_offset = _get_buffer_position(buffer)
        encoded_bits = number_of_encoding_bits(len(unique_values))
        stored_lookup_tables[values_in_bytes] = (
            lookup_table_offset,
            encoded_bits,
        )
        buffer += values_in_bytes
    else:
        lookup_table_offset, encoded_bits = stored_lookup_tables[values_in_bytes]
    return lookup_table_offset, encoded_bits


def _create_encoded_values(
    buffer: bytearray, positions: np.ndarray, encoded_bits: int
) -> int:
    """Create the encoded values for the given values

    Parameters
    ----------
    buffer: bytearray
        The buffer to write the encoded values to
    positions: da.Array
        The values to encode (positions in the lookup table)
    encoded_bits: int
        The number of bits used to encode the values

    Returns
    -------
    encoded_values_offset: int
        The offset in the buffer to the encoded values
    """
    encoded_values_offset = _get_buffer_position(buffer)
    buffer += _pack_encoded_values(positions, encoded_bits)
    return encoded_values_offset


def _create_file_chunk_header(number_channels: int = 1) -> bytearray:
    buf = bytearray(4 * number_channels)
    for offset in range(number_channels):
        struct.pack_into("<I", buf, offset * 4, len(buf) // 4)
    return buf


def create_segmentation_chunk(
    data: np.ndarray,
    dimensions: tuple[tuple[int, int, int], tuple[int, int, int]],
    block_size: tuple[int, int, int] = (8, 8, 8),
) -> Chunk:
    """Convert data in a dask array to a neuroglancer segmentation chunk"""
    bz, by, bx = block_size
    gz, gy, gx = get_grid_size_from_block_shape(data.shape, block_size)
    stored_lookup_tables: dict[bytes, tuple[int, int]] = {}
    # big enough to hold the 64-bit starting block headers
    buffer = bytearray(gx * gy * gz * 8)

    # data = np.moveaxis(data, (0, 1, 2), (2, 1, 0))
    for z, y, x in np.ndindex((gz, gy, gx)):
        block = data[
            z * bz : (z + 1) * bz, y * by : (y + 1) * by, x * bx : (x + 1) * bx
        ]
        unique_values, encoded_values = np.unique(block, return_inverse=True)
        if block.shape != block_size:
            block = pad_block(block, block_size)

        lookup_table_offset, encoded_bits = _create_lookup_table(
            buffer, stored_lookup_tables, unique_values
        )
        encoded_values_offset = _create_encoded_values(
            buffer, encoded_values, encoded_bits
        )
        block_offset = 8 * (x + gx * (y + gy * z))
        _create_block_header(
            buffer,
            lookup_table_offset,
            encoded_bits,
            encoded_values_offset,
            block_offset,
        )

    return Chunk(_create_file_chunk_header() + buffer, dimensions)
