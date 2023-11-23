import struct

import numpy as np
import pytest

from cryo_et_neuroglancer.chunk import Chunk
from cryo_et_neuroglancer.segmentation_decoding import (
    _decode_block_header,
    _decode_chunk_header,
    _decode_encoded_values,
    _decode_lookup_table,
    _extract_encoded_values,
    _unpack_encoded_values,
    decode_chunk,
)
from cryo_et_neuroglancer.segmentation_encoding import (
    _create_block_header,
    _create_encoded_values,
    _create_file_chunk_header,
    _create_lookup_table,
    create_segmentation_chunk,
)


@pytest.mark.parametrize(
    "packed_values, nb_bits, nb_values, expected",
    [
        (0b10_00_01, 2, 3, [1, 0, 2]),
        (0b0100_0011_0010_0000_0001, 4, 5, [1, 0, 2, 3, 4]),
    ],
)
def test__unpack_encoded_values(packed_values, nb_bits, nb_values, expected):
    result = _unpack_encoded_values(
        struct.pack("<I", packed_values), nb_bits, nb_values
    )
    assert len(result) == 32 // nb_bits
    for value, expected_value in zip(result, expected):
        assert value == expected_value


def test__decode_block_header():
    offset = 0
    buffer = bytearray(64 // 8)
    _create_block_header(
        buffer,
        lookup_table_offset=0xBEEF00,
        encoded_bits=0xAB,
        encoded_values_offset=0xDEADBEEF,
        block_offset=offset,
    )
    result = _decode_block_header(buffer, offset)

    assert result.lookup_table_offset == 0xBEEF00
    assert result.encoded_bits == 0xAB
    assert result.encoded_values_offset == 0xDEADBEEF


def test__decode_lookup_table():
    buffer = bytearray(32)
    global_lut = {}
    lut_offset, nb_bits = _create_lookup_table(buffer, global_lut, np.array([1, 0, 3]))

    lut = _decode_lookup_table(buffer, lut_offset, nb_bits)

    assert len(lut) == 3
    assert np.all(lut == np.array([1, 0, 3]))


def test__extract_encoded_values():
    buffer = bytearray()  # will start in 0
    offset = _create_encoded_values(buffer, np.array([1, 0, 2]), 2)
    encoded_values = _extract_encoded_values(buffer, offset, 2, 3)
    assert np.all(encoded_values == np.array([1, 0, 2] + [0] * ((32 // 2) - 3)))


def test__decode_encoded_values():
    buffer = bytearray(32)
    global_lut = {}
    lut_offset, nb_bits = _create_lookup_table(
        buffer, global_lut, np.array([42, 22, 55])
    )
    lut = _decode_lookup_table(buffer, lut_offset, nb_bits)

    buffer = bytearray()  # will start in 0
    offset = _create_encoded_values(buffer, np.array([1, 0, 2, 0]), 2)

    result = _decode_encoded_values(buffer, offset, 2, lut, (1, 4, 4))

    assert np.all(result[0, 0] == np.array([22, 42, 55, 42]))

    # Here are the pad values that were not removed
    assert np.all(result[0, 1] == np.array([42, 42, 42, 42]))
    assert np.all(result[0, 2] == np.array([42, 42, 42, 42]))
    assert np.all(result[0, 3] == np.array([42, 42, 42, 42]))


def test__decode_chunk_header():
    header = _create_file_chunk_header()
    nb_channels = _decode_chunk_header(header)
    assert nb_channels == 1


def test__decode_block():
    ...


def test__decode_chunk():
    # We take a small 8x8 cube
    array = np.array(
        [
            [
                [0, 2, 2, 0],
                [0, 2, 2, 0],
                [0, 2, 2, 0],
                [2, 2, 2, 2],
                [2, 2, 2, 2],
                [0, 2, 2, 0],
                [0, 2, 2, 0],
                [0, 2, 2, 0],
            ],
            [
                [3, 0, 0, 0],
                [1, 3, 1, 1],
                [0, 0, 3, 0],
                [1, 1, 1, 3],
                [0, 0, 3, 0],
                [1, 3, 1, 1],
                [3, 0, 0, 0],
                [1, 3, 1, 1],
            ],
            [
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
            ],
            [
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
            ],
            [
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
            ],
            [
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
            ],
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
                [18, 19, 20, 21],
                [22, 23, 24, 25],
                [26, 27, 28, 29],
                [30, 31, 32, 33],
            ],
            [
                [1, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 1, 1],
            ],
        ],
        dtype=np.uint32,
    )
    shape = array.shape
    chunk: Chunk = create_segmentation_chunk(
        array,
        dimensions=((0, 0, 0), shape),  # type: ignore
        block_size=shape,  # type: ignore
    )

    result = decode_chunk(chunk, block_size=shape)  # type: ignore
    assert result.shape == array.shape  # type: ignore
    assert np.all(result == array)

    # Forcing all values >= 1 to 1
    chunk: Chunk = create_segmentation_chunk(
        array,
        dimensions=((0, 0, 0), shape),  # type: ignore
        block_size=shape,  # type: ignore
        convert_non_zero_to=1,
    )

    result = decode_chunk(chunk, block_size=shape)  # type: ignore
    assert result.shape == array.shape  # type: ignore
    assert np.all(result[0, 0] == np.array([0, 1, 1, 0]))
    assert np.all(result[0, 3] == np.array([1, 1, 1, 1]))
    assert np.all(result[1, 1] == np.array([1, 1, 1, 1]))

    # Forcing all values >= 1 to 5
    chunk: Chunk = create_segmentation_chunk(
        array,
        dimensions=((0, 0, 0), shape),  # type: ignore
        block_size=shape,  # type: ignore
        convert_non_zero_to=5,
    )

    result = decode_chunk(chunk, block_size=shape)  # type: ignore
    assert result.shape == array.shape  # type: ignore
    assert np.all(result[0, 0] == np.array([0, 5, 5, 0]))
    assert np.all(result[0, 3] == np.array([5, 5, 5, 5]))
    assert np.all(result[1, 1] == np.array([5, 5, 5, 5]))
