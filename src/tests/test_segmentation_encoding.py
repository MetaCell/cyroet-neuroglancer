import struct
import pytest
from cryo_et_neuroglancer.segmentation_encoding import (
    _get_buffer_position,
    _create_block_header,
    _create_lookup_table,
    _pack_encoded_values,
)
from ctypes import c_uint64, LittleEndianStructure
import numpy as np

from cryo_et_neuroglancer.utils import number_of_encoding_bits


# Used for decoding the header
class BlockHeader(LittleEndianStructure):
    _fields_ = [
        ("lookup_table_offset", c_uint64, 24),
        ("encoded_bits", c_uint64, 8),
        ("encoded_values_offset", c_uint64, 32),
    ]


def test__get_buffer_position():
    assert _get_buffer_position(bytearray(8)) == 2
    assert _get_buffer_position(bytearray(16)) == 4

    with pytest.raises(AssertionError):
        _get_buffer_position(bytearray(5))


def test__create_block_header__without_offset():
    buffer = bytearray(64 // 8)
    _create_block_header(
        buffer,
        lookup_table_offset=0xBEEF00,
        encoded_bits=0xAB,
        encoded_values_offset=0xDEADBEEF,
        block_offset=0,
    )
    result = BlockHeader.from_buffer(buffer, 0)

    assert result.lookup_table_offset == 0xBEEF00
    assert result.encoded_bits == 0xAB
    assert result.encoded_values_offset == 0xDEADBEEF


def test__create_block_header__with_offset():
    offset = 0x4
    buffer = bytearray(64 // 8 + offset)
    _create_block_header(
        buffer,
        lookup_table_offset=0xBEEF00,
        encoded_bits=0xAB,
        encoded_values_offset=0xDEADBEEF,
        block_offset=offset,
    )
    result = BlockHeader.from_buffer(buffer, offset)

    assert result.lookup_table_offset == 0xBEEF00
    assert result.encoded_bits == 0xAB
    assert result.encoded_values_offset == 0xDEADBEEF


## Need to go back again there
def test__create_lookup_table__fill_global_lut():
    buffer = bytearray.fromhex("DEADBEEF")
    global_lut = {}

    lut_offset, nb_bits = _create_lookup_table(buffer, global_lut, np.array([1, 0, 3]))

    assert nb_bits == 2
    assert len(global_lut) == 1
    assert len(buffer) == 16  # ?
    assert lut_offset == 1  # ?


@pytest.mark.parametrize(
    "array, nb_bits, expected",
    [
        ([1, 0, 2], 2, 0b10_00_01),
        ([1, 0, 2, 3, 4], 4, 0b0100_0011_0010_0000_0001),
    ],
)
def test__pack_encoded_values(array, nb_bits, expected):
    encoded = _pack_encoded_values(np.array(array), nb_bits)
    assert encoded == struct.pack("<I", expected)


def test__pack_encoded_values__needs_32multiple():
    array = [1, 0, 2, 3, 4]
    nb_bits = 3
    with pytest.raises(AssertionError):
        _pack_encoded_values(np.array(array), nb_bits)
