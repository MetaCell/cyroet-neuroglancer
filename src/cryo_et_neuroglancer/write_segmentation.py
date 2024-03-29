import shutil
import sys
from pathlib import Path
from typing import Any, Iterator, Optional

import dask.array as da
import numpy as np
from tqdm import tqdm

from .chunk import Chunk
from .io import load_omezarr_data, write_metadata
from .segmentation_encoding import create_segmentation_chunk
from .utils import iterate_chunks


def _create_metadata(
    chunk_size: tuple[int, int, int],
    block_size: tuple[int, int, int],
    data_size: tuple[int, int, int],
    data_directory: str,
    resolution: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, Any]:
    """Create the metadata for the segmentation"""
    metadata = {
        "@type": "neuroglancer_multiscale_volume",
        "data_type": "uint32",
        "num_channels": 1,
        "scales": [
            {
                "chunk_sizes": [chunk_size[::-1]],
                "encoding": "compressed_segmentation",
                "compressed_segmentation_block_size": block_size[::-1],
                "resolution": resolution,
                "key": data_directory,
                "size": data_size[
                    ::-1
                ],  # reverse the data size to pass from Z-Y-X to X-Y-Z
            }
        ],
        "type": "segmentation",
    }
    return metadata


def create_segmentation(
    dask_data: da.Array,
    block_size: tuple[int, int, int],
    convert_non_zero_to: Optional[int] = 0,
) -> Iterator[Chunk]:
    """Yield the neuroglancer segmentation format chunks"""
    to_iterate = iterate_chunks(dask_data)
    num_iters = np.prod(dask_data.numblocks)
    for chunk, dimensions in tqdm(
        to_iterate, desc="Processing chunks", total=num_iters
    ):
        yield create_segmentation_chunk(
            chunk.compute(),
            dimensions,
            block_size,
            convert_non_zero_to=convert_non_zero_to,
        )


def main(
    filename: Path,
    block_size: tuple[int, int, int] = (64, 64, 64),
    data_directory: str = "data",
    delete_existing_output_directory: bool = False,
    output_path: Optional[Path] = None,
    resolution: tuple[float, float, float] = (1.0, 1.0, 1.0),
    convert_non_zero_to: Optional[int] = 0,
) -> None:
    """Convert the given OME-Zarr file to neuroglancer segmentation format with the given block size"""
    print(f"Converting {filename} to neuroglancer compressed segmentation format")
    dask_data = load_omezarr_data(filename)
    remove_ending = filename.stem.endswith(".zarr") or filename.stem.endswith("_zarr")
    output_name = filename.stem[:-5] if remove_ending else filename.stem
    output_directory = output_path or filename.parent / f"precomputed-{output_name}"
    if delete_existing_output_directory and output_directory.exists():
        contents = list(output_directory.iterdir())
        content_names = sorted([c.name for c in contents])
        if content_names and content_names != ["data", "info"]:
            print(
                f"The output directory {output_directory!s} exists and contains non-conversion related files, not deleting it"
            )
            sys.exit(1)
        else:
            print(
                f"The output directory {output_directory!s} exists from a previous run, deleting before starting the conversion"
            )
            shutil.rmtree(output_directory)
    elif not delete_existing_output_directory and output_directory.exists():
        print(f"The output directory {output_directory!s} already exists")
        sys.exit(1)
    output_directory.mkdir(parents=True, exist_ok=True)
    for c in create_segmentation(
        dask_data, block_size, convert_non_zero_to=convert_non_zero_to
    ):
        c.write_to_directory(output_directory / data_directory)

    if len(dask_data.chunksize) != 3:
        raise ValueError(f"Expected 3 chunk dimensions, got {len(dask_data.chunksize)}")
    metadata = _create_metadata(
        dask_data.chunksize,
        block_size,
        dask_data.shape,
        data_directory,
        resolution,  # type: ignore
    )
    write_metadata(metadata, output_directory)
    print(f"Wrote segmentation to {output_directory}")
