from functools import lru_cache
from math import ceil
from pathlib import Path
from typing import Iterator, Optional

import dask.array as da
import numpy as np

from .io import load_omezarr_data


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get


def parse_color(input_color: list[str]) -> tuple[int, int, int, int]:
    """
    Parse the color from a list of strings to a list of integers

    Parameters
    ----------
    color : list[str]
        The color as a list of strings

    Returns
    -------
    list[int]
        The color as a list of integers
    """
    if len(input_color) == 1:
        color = input_color[0]
        red = int(color[1:3], 16)
        green = int(color[3:5], 16)
        blue = int(color[5:7], 16)
        output_color = (red, green, blue)
    elif len(input_color) != 3:
        raise ValueError(f"Color must be a list of 3 values, provided: {input_color}")
    else:
        output_color = tuple(int(x) for x in input_color)  # type: ignore
    return output_color + (255,)


def compute_contrast_limits(
    zarr_path: Path,
) -> tuple[tuple[float, float], tuple[int, int, int]]:
    """Compute the contrast limits for the given ZARR file"""
    data = load_omezarr_data(zarr_path)
    middle_z_slice = data.shape[0] // 2
    middle_y_slice = data.shape[1] // 2
    middle_x_slice = data.shape[2] // 2
    z_start = max(middle_z_slice - 2, 0)
    z_end = min(middle_z_slice + 2, data.shape[0])
    sample_data = get_random_samples(data[z_start:z_end], 1500)
    limits = np.percentile(sample_data, (5.0, 95.0))
    return np.round(limits, 2), (middle_z_slice, middle_y_slice, middle_x_slice)


def get_random_samples(dask_array: da.Array, size: int) -> np.ndarray:
    shape = dask_array.shape

    random_indices = [np.random.choice(dim, size=size) for dim in shape]

    random_samples = dask_array.compute()[
        random_indices[0], random_indices[1], random_indices[2]
    ]

    return random_samples


def get_resolution(
    resolution: Optional[tuple[float, float, float] | list[float] | float]
) -> tuple[float, float, float]:
    if resolution is None:
        resolution = [
            1.348,
        ]
    if not isinstance(resolution, (tuple, list)):
        resolution = [resolution]
    if len(resolution) == 1:
        resolution = (resolution[0],) * 3  # type: ignore
    if len(resolution) != 3:
        raise ValueError("Resolution tuple must have 3 values")
    if any(x <= 0 for x in resolution):
        raise ValueError("Resolution component has to be > 0")
    return resolution  # type: ignore


def pad_block(block: np.ndarray, block_size: tuple[int, int, int]) -> np.ndarray:
    """Pad the block to the given block size with zeros"""
    return np.pad(
        block,
        (
            (0, block_size[0] - block.shape[0]),
            (0, block_size[1] - block.shape[1]),
            (0, block_size[2] - block.shape[2]),
        ),
        # mode='edge'
    )


def iterate_chunks(
    dask_data: da.Array,
) -> Iterator[tuple[da.Array, tuple[tuple[int, int, int], tuple[int, int, int]]]]:
    """Iterate over the chunks in the dask array"""
    chunk_layout = dask_data.chunks

    for zi, z in enumerate(chunk_layout[0]):
        for yi, y in enumerate(chunk_layout[1]):
            for xi, x in enumerate(chunk_layout[2]):
                chunk = dask_data.blocks[zi, yi, xi]

                # Calculate the chunk dimensions
                start = (
                    sum(chunk_layout[0][:zi]),
                    sum(chunk_layout[1][:yi]),
                    sum(chunk_layout[2][:xi]),
                )
                end = (start[0] + z, start[1] + y, start[2] + x)
                dimensions = (start, end)
                yield chunk, dimensions


def make_transform(input_dict: dict, dim: str, resolution: float):
    input_dict[dim] = [resolution * 10e-10, "m"]


def get_grid_size_from_block_shape(
    data_shape: tuple[int, int, int], block_shape: tuple[int, int, int]
) -> tuple[int, int, int]:
    """
    Calculate the grid size from the block shape and data shape

    Both the data shape and block size should be in z, y, x order

    Parameters
    ----------
    data_shape : tuple[int, int, int]
        The shape of the data
    block_shape : tuple[int, int, int]
        The block shape

    Returns
    -------
    tuple[int, int, int]
        The grid size as gz, gy, gx
    """
    gz = ceil(data_shape[0] / block_shape[0])
    gy = ceil(data_shape[1] / block_shape[1])
    gx = ceil(data_shape[2] / block_shape[2])
    return gz, gy, gx


@lru_cache()
def number_of_encoding_bits(nb_values: int) -> int:
    """
    Return the number of bits needed to encode a number of values

    Parameters
    ----------
    nb_values : int
        The number of values that needs to be encoded

    Returns
    -------
    int between (0, 1, 2, 4, 8, 16, 32)
        The number of bits necessary
    """
    for nb_bits in (0, 1, 2, 4, 8, 16, 32):
        if (1 << nb_bits) >= nb_values:
            return nb_bits
    raise ValueError("Too many unique values in block")
