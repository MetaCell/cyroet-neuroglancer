import argparse
import sys
from pathlib import Path
from typing import Optional

import neuroglancer.cli

from .state_generation import create_annotation, create_image, create_segmentation
from .url_creation import combine_json_layers, load_jsonstate_to_browser, viewer_to_url
from .utils import get_resolution
from .write_annotations import main as annotations_encode
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
    convert_non_zero: int,
    resolution: Optional[tuple[float, float, float] | list[float]],
):
    file_path = Path(zarr_path)
    if not file_path.exists():
        print(f"The input ZARR folder {file_path!s} doesn't exist")
        return 1
    resolution = get_resolution(resolution)

    block_size = int(block_size)
    block_shape = (block_size, block_size, block_size)
    output_path = Path(output) if output else None
    segmentation_encode(
        file_path,
        block_shape,
        delete_existing_output_directory=not skip_existing,
        output_path=output_path,
        resolution=resolution,  # type: ignore
        convert_non_zero_to=convert_non_zero,
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
        "-o",
        "--output",
        required=False,
        help="Output folder to produce in precomputed format. If not provided, the output will be precomputed_<zarr_path> with the last 5 characters removed (_zarr)",
    )
    subcommand.add_argument(
        "-b",
        "--block-size",
        required=False,
        default=64,
        help="Block size (default: 64)",
    )
    subcommand.add_argument(
        "-r",
        "--resolution",
        nargs="+",
        type=float,
        help="Resolution in nm, must be either 3 values for X Y Z separated by spaces, or a single value that will be set for X Y and Z (default: 1.348)",
    )
    subcommand.add_argument(
        "--convert-non-zero",
        required=False,
        type=int,
        nargs="?",
        default=0,
        const=1,
        help="Force all values >= 1 to an integer. If the option is used without arguments, 1 is considered.",
    )
    subcommand.set_defaults(func=encode_segmentation)

    # Annotation encoding
    subcommand = subparsers.add_parser(
        "encode-annotation", help="Encode annotations file"
    )
    subcommand.add_argument(
        "json_path",
        help="Path towards the JSON file containing the annotations metadata",
        type=Path,
    )
    subcommand.add_argument(
        "-o", "--output", required=False, help="Output folder to produce", type=Path
    )
    subcommand.add_argument(
        "-r", "--resolution", required=False, help="Resolution", type=float
    )
    subcommand.add_argument(
        "-c",
        "--color",
        required=False,
        nargs=4,
        type=int,
        help="Color of the points as 0-255 RGBA",
    )
    subcommand.set_defaults(func=annotations_encode)

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

    # Image JSON creation
    subcommand = subparsers.add_parser(
        "create-image",
        help="Create a JSON file for a given image",
    )
    # TODO should this maybe support remote ZARR files?
    subcommand.add_argument(
        "source",
        help="Path towards the remote ZARR file",
    )
    subcommand.add_argument(
        "-z",
        "--zarr-path",
        required=False,
        help="Path towards the local ZARR file",
    )
    subcommand.add_argument(
        "-o", "--output", required=False, help="Output json to produce", type=Path
    )
    subcommand.add_argument(
        "-n",
        "--name",
        required=False,
        help="Name of the layer (default: source filename)",
    )
    subcommand.add_argument(
        "-u",
        "--url",
        required=False,
        help="URL of the zarr server (by default, the source path is used)",
    )
    subcommand.add_argument(
        "-r",
        "--resolution",
        nargs="+",
        type=float,
        help="Resolution in nm, must be either 3 values for X Y Z separated by spaces, or a single value that will be set for X Y and Z (default: 1.348)",
        required=False,
    )
    subcommand.set_defaults(func=create_image)

    # Annotation JSON creation
    subcommand = subparsers.add_parser(
        "create-annotation",
        help="Create a JSON file for a given annotation",
    )
    subcommand.add_argument(
        "source",
        help="Path towards the remote ZARR file",
    )
    subcommand.add_argument(
        "-z",
        "--zarr-path",
        required=False,
        help="Path towards the local ZARR file",
    )
    subcommand.add_argument(
        "-o", "--output", required=False, help="Output json to produce", type=Path
    )
    subcommand.add_argument(
        "-n",
        "--name",
        required=False,
        help="Name of the layer (default: source filename)",
    )
    subcommand.add_argument(
        "-u",
        "--url",
        required=False,
        help="URL of the zarr server (by default, the source path is used)",
    )
    subcommand.add_argument(
        "-c",
        "--color",
        required=False,
        type=str,
        help="A hex string followed the name of the color e.g. #FF0000 red",
    )
    subcommand.add_argument(
        "-s",
        "--point-size-multiplier",
        required=False,
        type=float,
        help="The point size multiplier to use for the annotation",
    )
    subcommand.set_defaults(func=create_annotation)

    # Segmentation JSON creation
    subcommand = subparsers.add_parser(
        "create-segmentation",
        help="Create a JSON file for a given segmentation",
    )
    subcommand.add_argument(
        "source",
        help="Path towards the remote ZARR file",
    )
    subcommand.add_argument(
        "-z",
        "--zarr-path",
        required=False,
        help="Path towards the local ZARR file",
    )
    subcommand.add_argument(
        "-o", "--output", required=False, help="Output json to produce", type=Path
    )
    subcommand.add_argument(
        "-n",
        "--name",
        required=False,
        help="Name of the layer (default: source filename)",
    )
    subcommand.add_argument(
        "-u",
        "--url",
        required=False,
        help="URL of the zarr server (by default, the source path is used)",
    )
    subcommand.add_argument(
        "-c",
        "--color",
        required=False,
        type=str,
        help="A hex string followed the name of the color e.g. #FF0000 red",
    )
    subcommand.set_defaults(func=create_segmentation)

    # JSON combination
    subcommand = subparsers.add_parser(
        "combine-json",
        help="Combine multiple layer JSON files into a single JSON file to render",
    )
    subcommand.add_argument(
        "json_paths",
        nargs="+",
        help="JSON files to combine",
        type=Path,
    )
    subcommand.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output json to produce",
        type=Path,
    )
    subcommand.add_argument(
        "-r",
        "--resolution",
        nargs="+",
        type=float,
        help="Resolution in nm, must be either 3 values for X Y Z separated by spaces, or a single value that will be set for X Y and Z (default: 1.348)",
        required=False,
    )
    subcommand.set_defaults(func=combine_json_layers)

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
