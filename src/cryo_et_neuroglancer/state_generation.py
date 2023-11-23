import json
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Union

from .utils import compute_contrast_limits, get_resolution, make_transform


class RenderingTypes(Enum):
    """Types of rendering for Neuroglancer."""

    SEGMENTATION = auto()
    IMAGE = auto()
    ANNOTATION = auto()


@dataclass
class RenderingJSONGenerator:
    """Generates a JSON file for Neuroglancer to read."""

    source: str
    name: str

    @property
    def layer_type(self) -> str:
        """Returns the layer type for Neuroglancer."""
        if self._type == RenderingTypes.SEGMENTATION:  # type: ignore
            return "segmentation"
        elif self._type == RenderingTypes.IMAGE:  # type: ignore
            return "image"
        elif self._type == RenderingTypes.ANNOTATION:  # type: ignore
            return "annotation"
        else:
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

    def __post_init__(self):
        self._type = RenderingTypes.IMAGE

    def create_shader(self):
        distance = self.contrast_limits[1] - self.contrast_limits[0]
        window_start = self.contrast_limits[0] - (distance / 10)
        window_end = self.contrast_limits[1] + (distance / 10)
        return f"#uicontrol invlerp normalized(range=[{self.contrast_limits[0]}, {self.contrast_limits[1]}], window=[{window_start}, {window_end}])\nvoid main() {{\n  emitGrayscale(normalized());\n}}"

    def generate_json(self) -> dict:
        transform: dict = {}
        for dim, resolution in zip("xyz", self.resolution):
            make_transform(transform, dim, resolution)

        original: dict = {}
        for dim, resolution in zip("zyx", self.resolution[::-1]):
            make_transform(original, dim, resolution)

        source = {
            "url": self.source,
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
        }


# TODO proper json state
@dataclass
class SegmentationJSONGenerator(RenderingJSONGenerator):
    """Generates a JSON file for Neuroglancer to read."""

    color: Union[str, tuple[str, str]]

    def __post_init__(self):
        self._type = RenderingTypes.SEGMENTATION

    def generate_json(self) -> dict:
        return {
            "type": self.layer_type,
            "name": self.name,
            "source": self.source,
            "tab": "rendering",
            "selectedAlpha": 0,
            "notSelectedAlpha": 0,
            "hideSegmentZero": True,
            "color": self.color,
        }


# TODO proper json state
@dataclass
class AnnotationJSONGenerator(RenderingJSONGenerator):
    """Generates a JSON file for Neuroglancer to read."""

    color: Union[str, tuple[str, str]]
    point_size_multiplier: float = 1.0

    def __post_init__(self):
        self._type = RenderingTypes.ANNOTATION

    def generate_json(self) -> dict:
        return {
            "type": self.layer_type,
            "name": self.name,
            "source": self.source,
            "tab": "rendering",
            "pointSize": self.point_size_multiplier,
            "pointRenderMode": "screenFacing",
            "lineWidth": 1,
            "lineWidthUnits": "pixels",
            "pointColor": self.color,
            "lineColor": self.color,
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
    source = f"zarr://{url}{sep}{source}"
    return source, name, url, output, zarr_path, resolution


def create_image(
    source: str,
    zarr_path: Optional[str],
    name: Optional[str],
    resolution: Optional[float | tuple[float, float, float]],
    url: Optional[str],
    output: Optional[Path],
) -> dict:
    source, name, url, output, zarr_path, resolution = setup_creation(
        source, name, url, output, zarr_path, resolution
    )
    contrast_limits = compute_contrast_limits(Path(zarr_path))
    json_generator = ImageJSONGenerator(
        source=source, name=name, resolution=resolution, contrast_limits=contrast_limits
    )
    return json_generator.to_json(output)


def create_annotation(
    source: str,
    zarr_path: Optional[str],
    name: Optional[str],
    url: Optional[str],
    output: Optional[Path],
    color: Optional[str],
    point_size_multiplier: Optional[float],
) -> dict:
    source, name, url, output, zarr_path, _ = setup_creation(
        source, name, url, output, zarr_path, None
    )
    if color is None:
        # TODO proper color parsing
        color = "red"
    point_size_multiplier = (
        1.0 if point_size_multiplier is None else point_size_multiplier
    )
    json_generator = AnnotationJSONGenerator(
        source=source,
        name=name,
        color=color,
        point_size_multiplier=point_size_multiplier,
    )
    return json_generator.to_json(output)


def create_segmentation(
    source: str,
    zarr_path: Optional[str],
    name: Optional[str],
    url: Optional[str],
    output: Optional[Path],
    color: Optional[str],
) -> dict:
    source, name, url, output, zarr_path, _ = setup_creation(
        source, name, url, output, zarr_path, None
    )
    if color is None:
        # TODO proper color parsing
        color = "red"
    json_generator = SegmentationJSONGenerator(source=source, name=name, color=color)
    return json_generator.to_json(output)
