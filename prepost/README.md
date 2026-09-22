# `prepost`

Source utilities for the fluidised-bed case. They are intended to run from the repository root against the case's CFD and DEM output.

## Choose a document

| Need | Document |
| --- | --- |
| Generate plots and a Bond-number summary | [Quick start](docs/QUICK_START.md) |
| Understand input data, units, and analysis assumptions | [Library guide](docs/LIBRARY_DOCUMENTATION.md) |
| Look up Python classes and functions | [API reference](docs/API_REFERENCE.md) |
| Adapt one of the repository's runnable scripts | [Examples](docs/EXAMPLES.md) |

## Public Python API

```python
from prepost import FlBedPlot, LIGGGHTSTemplatePopulator, ModelAnalysis, liggghts2vtk, msq_displ
```

The package is source-tree based at present. Create an environment and install the runtime dependencies, then invoke code from the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r prepost/requirements.txt uncertainties
mkdir -p plots pyoutputs
python plot_fluidn_curves.py
```

`prepost/pyproject.toml` declares the future package metadata, but `pip install ./prepost` is not currently supported because the build expects a `src/prepost/` layout that this checkout does not contain.

## Included tools

- `FlBedPlot`: pressure and void-fraction plots from OpenFOAM VTK cutting planes.
- `ModelAnalysis`: overshoot, DHR, and hysteresis estimates with propagated standard errors.
- `LIGGGHTSTemplatePopulator`: render JKR or SJKR inputs from `templates/`.
- `liggghts2vtk()` and `msq_displ()`: convert particle dumps and calculate per-particle mean-squared displacement.

The Julia source in `src/` is an experimental parallel implementation. The repository scripts and examples use the Python API as the supported path.
