import json
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from .utils import compute_contrast_limits, get_resolution, make_transform


class RenderingTypes(Enum):
    """Types of rendering for Neuroglancer."""

    SEGMENTATION = auto()
    IMAGE = auto()
    ANNOTATION = auto()

    def __str__(self):
        return self.name.lower()


@dataclass
class RenderingJSONGenerator:
    """Generates a JSON file for Neuroglancer to read."""

    source: str
    name: str

    @property
    def layer_type(self) -> str:
        """Returns the layer type for Neuroglancer."""
        try:
            return str(self._type)  # type: ignore
        except AttributeError:
            raise ValueError(f"Unknown rendering type {self._type}")  # type: ignore

    def to_json(self, output: Path) -> dict:
        """Writes the JSON to a file."""
        json_info = self.generate_json()
        json.dump(json_info, output.open("w"), indent=2)
        return json_info

    @abstractmethod
    def generate_json(self) -> dict:
        """Generates the JSON for Neuroglancer."""
        raise NotImplementedError


@dataclass
class ImageJSONGenerator(RenderingJSONGenerator):
    """Generates a JSON file for Neuroglancer to read."""

    resolution: tuple[float, float, float]
    contrast_limits: tuple[float, float] = (-64, 64)
    middle_slices: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self):
        self._type = RenderingTypes.IMAGE

    def create_shader(self):
        distance = self.contrast_limits[1] - self.contrast_limits[0]
        window_start = self.contrast_limits[0] - (distance / 10)
        window_end = self.contrast_limits[1] + (distance / 10)
        return f"#uicontrol invlerp contrast(range=[{self.contrast_limits[0]}, {self.contrast_limits[1]}], window=[{window_start}, {window_end}])\nvoid main() {{\n  emitGrayscale(contrast());\n}}"

    def generate_json(self) -> dict:
        transform: dict = {}
        for dim, resolution in zip("zyx", self.resolution[::-1]):
            make_transform(transform, dim, resolution)

        original: dict = {}
        for dim, resolution in zip("zyx", self.resolution[::-1]):
            make_transform(original, dim, resolution)

        source = {
            "url": f"zarr://{self.source}",
            "transform": {
                "outputDimensions": transform,
                "inputDimensions": original,
            },
        }

        return {
            "type": self.layer_type,
            "name": self.name,
            "source": source,
            "shader": self.create_shader(),
            "tab": "rendering",
            "_non_neuroglancer_middle": self.middle_slices,
        }


@dataclass
class SegmentationJSONGenerator(RenderingJSONGenerator):
    """Generates a JSON file for Neuroglancer to read."""

    color: tuple[str, str]

    def __post_init__(self):
        self._type = RenderingTypes.SEGMENTATION

    def generate_json(self) -> dict:
        return {
            "type": self.layer_type,
            "name": f"{self.name} ({self.color[1]})",
            "source": f"precomputed://{self.source}",
            "tab": "rendering",
            "selectedAlpha": 1,
            "hoverHighlight": False,
            "segments": [
                1,
            ],
            "segmentDefaultColor": self.color[0],
        }


@dataclass
class AnnotationJSONGenerator(RenderingJSONGenerator):
    """Generates a JSON file for Neuroglancer to read."""

    color: tuple[str, str]
    point_size_multiplier: float = 1.0

    def __post_init__(self):
        self._type = RenderingTypes.ANNOTATION

    def generate_json(self) -> dict:
        return {
            "type": self.layer_type,
            "name": f"{self.name} ({self.color[1]})",
            "source": f"precomputed://{self.source}",
            "tab": "rendering",
            "shader": f"#uicontrol float pointScale slider(min=0.01, max=2.0, default={self.point_size_multiplier}, step=0.01)\n"
            "void main() {\n  "
            + "setColor(prop_point_color());\n  "
            + "setPointMarkerSize(pointScale * prop_diameter());\n"
            + "}",
        }


def setup_creation(
    source: str,
    name: Optional[str],
    url: Optional[str],
    output: Optional[Path],
    zarr_path: Optional[str],
    resolution: Optional[float | tuple[float, float, float]],
) -> tuple[str, str, str, Path, str, tuple[float, float, float]]:
    name = Path(source).stem if name is None else name
    url = url if url is not None else ""
    zarr_path = zarr_path if zarr_path is not None else source
    output = output if output is not None else Path(f"{name}.json")
    resolution = get_resolution(resolution)
    sep = "/" if url else ""
    source = f"{url}{sep}{source}"
    return source, name, url, output, zarr_path, resolution


def process_color(color: Optional[str]) -> tuple[str, str]:
    if color is None:
        return ("#ff0000", "red")
    else:
        color_parts = color.split(" ")
        if len(color_parts) == 1:
            raise ValueError(
                "Color must be a hex string followed by a name e.g. #FF0000 red"
            )
        return (color_parts[0], " ".join(color_parts[1:]))


def create_image(
    source: str,
    zarr_path: Optional[str],
    name: Optional[str],
    resolution: Optional[float | tuple[float, float, float]],
    url: Optional[str],
    output: Optional[Path],
) -> int:
    source, name, url, output, zarr_path, resolution = setup_creation(
        source, name, url, output, zarr_path, resolution
    )
    contrast_limits, middles = compute_contrast_limits(Path(zarr_path))
    json_generator = ImageJSONGenerator(
        source=source,
        name=name,
        resolution=resolution,
        contrast_limits=contrast_limits,
        middle_slices=middles,
    )
    json_generator.to_json(output)
    return 0


def create_annotation(
    source: str,
    zarr_path: Optional[str],
    name: Optional[str],
    url: Optional[str],
    output: Optional[Path],
    color: Optional[str],
    point_size_multiplier: Optional[float],
) -> int:
    source, name, url, output, zarr_path, _ = setup_creation(
        source, name, url, output, zarr_path, None
    )
    new_color = process_color(color)
    point_size_multiplier = (
        1.0 if point_size_multiplier is None else point_size_multiplier
    )
    json_generator = AnnotationJSONGenerator(
        source=source,
        name=name,
        color=new_color,
        point_size_multiplier=point_size_multiplier,
    )
    json_generator.to_json(output)
    return 0


def create_segmentation(
    source: str,
    zarr_path: Optional[str],
    name: Optional[str],
    url: Optional[str],
    output: Optional[Path],
    color: Optional[str],
) -> int:
    source, name, url, output, zarr_path, _ = setup_creation(
        source, name, url, output, zarr_path, None
    )
    color_tuple = process_color(color)
    json_generator = SegmentationJSONGenerator(
        source=source, name=name, color=color_tuple
    )
    json_generator.to_json(output)
    return 0
