# cyro-et-neuroglancer

## User guide

### Installation

```bash
git clone https://github.com/MetaCell/cyroet-neuroglancer
cd cyroet-neuroglancer
pip install cyro-et-neuroglancer
```

### Usage

```bash
cryoet-converter --help
```

See the examples folder for bash scripts that create all the necessary files for a neuroglancer viewer from a cryo-ET dataset.

### Annotations

./examples/convert_and_view_image_and_annotations.sh

![Annotations](examples/annotation.png)

### Segmentation

./examples/convert_and_view_image_and_segmentation.sh

![Segmentation](examples/segmentation.png)

## Development

### Developer installation

```bash
git clone https://github.com/MetaCell/cyroet-neuroglancer
cd cyroet-neuroglancer
pip install -e ".[dev]"
pre-commit install
```

### Pre-commit hooks

To run all pre-commit hooks on all files run:

```bash
pre-commit run --all-files
```

This will lint via `ruff`, format via `ruff` (essentially `black`), and check mypy types with the arguments specified in `.pre-commit-config.yaml`.

### Testing

```bash
pytest
```

### Mypy

Manual type checking with mypy:

```bash
mypy .
```
