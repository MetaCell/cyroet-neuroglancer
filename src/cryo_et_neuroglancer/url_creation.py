import json
import webbrowser
from pathlib import Path
from typing import Optional
from datetime import datetime

import neuroglancer
import neuroglancer.cli
from neuroglancer.url_state import json_to_url_safe, to_url, to_json_dump

from types import SimpleNamespace


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


def load_jsonstate_to_browser(path: str, **server_kwargs):
    json_path = Path(path)
    json_content = json_path.read_text(encoding="utf-8")
    state = neuroglancer.viewer_state.ViewerState(json.loads(json_content))

    viewer = launch_nglancer(server_kwargs)
    viewer.set_state(state)

    open_browser(viewer)
    loop_json_and_url(viewer, output_path=json_path)
    return 0
