import json
import webbrowser
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import neuroglancer
import neuroglancer.cli
from neuroglancer.url_state import to_json_dump, to_url

from .utils import get_resolution, make_transform


def launch_nglancer(server_kwargs) -> neuroglancer.Viewer:
    neuroglancer.cli.handle_server_arguments(SimpleNamespace(**server_kwargs))
    viewer = neuroglancer.Viewer()
    return viewer


def open_browser(viewer: neuroglancer.Viewer, hang: bool = False):
    print(viewer)
    webbrowser.open_new(viewer.get_viewer_url())
    if hang:
        input("Press Enter to continue...")


def dump_url_and_state(viewer: neuroglancer.Viewer, output_path: Optional[Path] = None):
    with viewer.txn() as s:
        url = to_url(s)
        json_state = to_json_dump(s, indent=2)
    if output_path:
        output_path.write_text(f"{url}\n{json_state}")
    else:
        print(url, json_state)


def loop_json_and_url(viewer: neuroglancer.Viewer, output_path: Optional[Path] = None):
    while input("Press Enter to print url and json or q to quit...") != "q":
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if output_path:
            output_path = output_path.parent / f"{output_path.stem}_{current_time}.txt"
        else:
            output_path = Path(f"nglancer_url_and_json_{current_time}.txt")
        print(f"Writing url and json to {output_path}")
        dump_url_and_state(viewer, output_path=output_path)


def viewer_to_url(**server_kwargs):
    viewer = launch_nglancer(server_kwargs)
    open_browser(viewer)
    loop_json_and_url(viewer)
    return 0


def load_jsonstate_to_browser(path: Path, **server_kwargs):
    json_content = path.read_text(encoding="utf-8")
    state = neuroglancer.viewer_state.ViewerState(json.loads(json_content))

    viewer = launch_nglancer(server_kwargs)
    viewer.set_state(state)

    open_browser(viewer)
    loop_json_and_url(viewer, output_path=path)
    return 0


def combine_json_layers(
    json_paths: list[Path],
    output: Path,
    resolution: Optional[tuple[float, float, float] | list[float]] = None,
):
    layers = [json.load(open(p, "r")) for p in json_paths]
    image_layers = [layer for layer in layers if layer["type"] == "image"]
    if len(image_layers) != 0:
        first_image = image_layers[0]
        middle_z = first_image["_non_neuroglancer_middlez"]
    else:
        middle_z = 480
    resolution = get_resolution(resolution)
    dimensions: dict = {}
    for dim, res in zip("zyx", resolution[::-1]):
        make_transform(dimensions, dim, res)

    combined_json = {
        "dimensions": dimensions,
        "layers": layers,
        "selectedLayer": {
            "visible": True,
            "layer": layers[0]["name"],
        },
        "position": [float(middle_z) + 0.5, 463.5, 430.5],
        "crossSectionScale": 1.8,
        "projectionOrientation": [
            0.0,
            0.655,
            0.0,
            -0.756,
        ],
    }
    json.dump(combined_json, open(output, "w"), indent=2)
