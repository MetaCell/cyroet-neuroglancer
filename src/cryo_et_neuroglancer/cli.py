import argparse
import sys
from .write_segmentation import main as segmentation_encode
from pathlib import Path


def encode_segmentation(
    zarr_path: str, skip_existing: bool, output: str, block_size: int
):
    file_path = Path(zarr_path)
    if not file_path.exists():
        print(f"The input ZARR folder {file_path!s} doesn't exist")
        return 1
    block_shape = (block_size, block_size, block_size)
    output_path = Path(output)
    segmentation_encode(
        file_path,
        block_shape,
        delete_existing_output_directory=not skip_existing,
        output_path=output_path,
    )
    return 0


def parse_args(args):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # metadata check
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
        "-b", "--block-size", required=False, default=64, help="Bloc size"
    )
    subcommand.set_defaults(func=encode_segmentation)

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
