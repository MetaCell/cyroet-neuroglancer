from pathlib import Path
from typing import Any, Iterator, Optional
from tqdm import tqdm
import numpy as np
import dask.array as da
from cryo_et_neuroglancer.chunk import Chunk
import sys
import shutil

from cryo_et_neuroglancer.utils import iterate_chunks
from cryo_et_neuroglancer.segmentation_encoding import (
    create_segmentation_chunk,
)
from cryo_et_neuroglancer.io import load_omezarr_data, write_metadata


def _create_metadata(
    chunk_size: tuple[int, int, int],
    block_size: tuple[int, int, int],
    data_size: tuple[int, int, int],
    data_directory: str,
    resolution: tuple[int, int, int] = (1, 1, 1),
) -> dict[str, Any]:
    """Create the metadata for the segmentation"""
    metadata = {
        "@type": "neuroglancer_multiscale_volume",
        "data_type": "uint32",
        "num_channels": 1,
        "scales": [
            {
                "chunk_sizes": [chunk_size],
                "encoding": "compressed_segmentation",
                "compressed_segmentation_block_size": block_size,
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
    dask_data: da.Array, block_size: tuple[int, int, int]
) -> Iterator[Chunk]:
    """Yield the neuroglancer segmentation format chunks"""
    to_iterate = iterate_chunks(dask_data)
    num_iters = np.prod(dask_data.numblocks)
    for chunk, dimensions in tqdm(
        to_iterate, desc="Processing chunks", total=num_iters
    ):
        yield create_segmentation_chunk(chunk.compute(), dimensions, block_size)


def main(
    filename: Path,
    block_size: tuple[int, int, int] = (64, 64, 64),
    data_directory: str = "data",
    delete_existing_output_directory: bool = False,
    output_path: Optional[Path] = None,
    resolution: tuple[int, int, int] = (1, 1, 1),
) -> None:
    """Convert the given OME-Zarr file to neuroglancer segmentation format with the given block size"""
    print(f"Converting {filename} to neuroglancer compressed segmentation format")
    dask_data = load_omezarr_data(filename)
    output_directory = (
        output_path or filename.parent / f"precomputed-{filename.stem[:-5]}"
    )
    if delete_existing_output_directory and output_directory.exists():
        contents = list(output_directory.iterdir())
        content_names = sorted([c.name for c in contents])
        if content_names != ["data", "info"]:
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
    for c in create_segmentation(dask_data, block_size):
        c.write_to_directory(output_directory / data_directory)

    metadata = _create_metadata(
        dask_data.chunksize, block_size, dask_data.shape, data_directory, resolution
    )
    write_metadata(metadata, output_directory)
    print(f"Wrote segmentation to {output_directory}")
