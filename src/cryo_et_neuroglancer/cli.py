import argparse
import sys
from pathlib import Path
from typing import Optional

import neuroglancer.cli

from .url_creation import load_jsonstate_to_browser, viewer_to_url
from .write_segmentation import main as segmentation_encode


def handle_json_load(path: str, **kwargs):
    json_path = Path(path)
    if not json_path.exists():
        print(f"JSON file {json_path.absolute()} does not exit")
        return -1
    return load_jsonstate_to_browser(json_path, **kwargs)


def encode_segmentation(
    zarr_path: str,
    skip_existing: bool,
    output: str,
    block_size: int,
    resolution: Optional[tuple[float, float, float] | list[float]],
):
    file_path = Path(zarr_path)
    if not file_path.exists():
        print(f"The input ZARR folder {file_path!s} doesn't exist")
        return 1
    if resolution is None:
        print("No resolution provided, using default value of 1.348nm")
        resolution = [
            1.348,
        ]
    if len(resolution) == 1:
        resolution = (resolution[0],) * 3  # type: ignore
    if len(resolution) != 3:
        print("Resolution tuple must have 3 values")
        return 2
    if any(x <= 0 for x in resolution):
        print("Resolution component has to be > 0")
        return 3
    block_size = int(block_size)
    block_shape = (block_size, block_size, block_size)
    output_path = Path(output) if output else None
    segmentation_encode(
        file_path,
        block_shape,
        delete_existing_output_directory=not skip_existing,
        output_path=output_path,
        resolution=resolution,  # type: ignore
    )
    return 0


def parse_args(args):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Segmentation encoding
    subcommand = subparsers.add_parser(
        "encode-segmentation", help="Encode segmentation file"
    )
    subcommand.add_argument(
        "zarr_path", help="Path towards your segmentation ZARR folder"
    )
    subcommand.add_argument(
        "--skip-existing",
        default=False,
        action="store_true",
        help="Skips already existing target folders",
    )
    subcommand.add_argument(
        "-o", "--output", required=False, help="Output folder to produce"
    )
    subcommand.add_argument(
        "-b", "--block-size", required=False, default=64, help="Block size"
    )
    subcommand.add_argument(
        "-r",
        "--resolution",
        nargs="+",
        type=float,
        help="Resolution, must be either 3 values for X Y Z separated by spaces, or a single value that will be set for X Y and Z",
    )
    subcommand.set_defaults(func=encode_segmentation)

    # URL creation
    subcommand = subparsers.add_parser(
        "create-url",
        help="Open a neuroglancer viewer and creates a URL and JSON state on-demand",
    )
    neuroglancer.cli.add_server_arguments(subcommand)
    subcommand.set_defaults(func=viewer_to_url)

    # JSON loading
    subcommand = subparsers.add_parser(
        "load-state",
        help="Load a neuroglancer JSON state file in a neuroglancer viewer",
    )
    subcommand.add_argument("path", help="JSON state file to load")
    neuroglancer.cli.add_server_arguments(subcommand)
    subcommand.set_defaults(func=handle_json_load)

    return parser.parse_args(args)


def main(argv=sys.argv[1:]):
    """Console script for cryo_et_neuroglancer converter."""
    args = parse_args(argv)
    kwargs = {k: v for k, v in vars(args).items() if k != "func"}
    func = getattr(args, "func", None)
    if not func:
        print("Missing arguments, type --help for more information")
        exit(-1)
    status_code = args.func(**kwargs)
    exit(status_code)
