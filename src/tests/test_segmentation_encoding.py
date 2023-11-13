import pytest
from cryo_et_neuroglancer.segmentation_encoding import (
    _get_buffer_position,
    _create_block_header,
)
from ctypes import c_uint64, LittleEndianStructure


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
