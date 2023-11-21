import webbrowser
from pathlib import Path
from typing import Optional
from datetime import datetime

import neuroglancer
import neuroglancer.cli
from neuroglancer.url_state import to_url, to_json_dump

from .utils import DotDict


def launch_nglancer(server_kwargs) -> neuroglancer.Viewer:
    neuroglancer.cli.handle_server_arguments(DotDict(server_kwargs))
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
        json_state = to_json_dump(s)
    if output_path:
        with open(output_path, "w") as f:
            f.write(f"{url}\n{json_state}")
    else:
        print(url, json_state)


def loop_json_and_url(viewer: neuroglancer.Viewer):
    finished = False
    while not finished:
        pressed = input("Press Enter to print url and json or q to quit...")
        if pressed == "q":
            finished = True
        else:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = Path(f"nglancer_url_and_json_{current_time}.txt")
            print(f"Writing url and json to {output_path}")
            dump_url_and_state(viewer, output_path=output_path)


def viewer_to_url(**server_kwargs):
    viewer = launch_nglancer(server_kwargs)
    open_browser(viewer)
    loop_json_and_url(viewer)
