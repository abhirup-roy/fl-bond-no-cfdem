# Quick start: analyse a completed case

Run these commands from the repository root. This tutorial assumes the CFD-DEM simulation has finished and wrote the default files named below.

## 1. Create the environment

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r prepost/requirements.txt uncertainties
mkdir -p plots pyoutputs
```

The additional `uncertainties` dependency is used by `ModelAnalysis` but is not listed in the legacy `requirements.txt` file.

## 2. Check the inputs

```bash
test -d CFD/postProcessing/cuttingPlane
test -f DEM/post/collisions.csv
test -f prepost/velcfg.txt
```

Each numeric time directory under `CFD/postProcessing/cuttingPlane/` needs `p_zNormal0.vtk` through `p_zNormal4.vtk`. Void-fraction analysis uses `voidfraction_yNormal.vtk` in the same directories. `prepost/velcfg.txt` maps simulation time to the imposed vertical gas velocity.

## 3. Plot fluidisation curves

```bash
python plot_fluidn_curves.py
```

This reads five z-normal pressure planes and the y-normal void-fraction plane. It produces pressure versus velocity/time and void fraction versus velocity plots in `plots/`. The velocity plot also writes `plots/probe0_plot_P.npy`; its columns are upward velocity, upward pressure, upward standard error, downward velocity, downward pressure, and downward standard error.

To control the plot from Python:

```python
from prepost import FlBedPlot

plot = FlBedPlot(
    pressure_path="CFD/postProcessing/cuttingPlane/",
    nprobes=5,
    velcfg_path="prepost/velcfg.txt",
    dump2csv=False,
    plots_dir="plots/",
)
plot.plot_pressure(x_var="velocity", slice_dirn="z", png_name="pressure")
plot.plot_voidfrac(slice_dirn="y", x_var="velocity", png_name="void_fraction")
```

`plots/` must already exist. `rho_f` defaults to `1.28 kg/m³`; set it to the fluid density used by the OpenFOAM case when converting kinematic pressure to Pa.

## 4. Calculate Bond numbers

```bash
python find_bondno.py
```

The script uses a particle diameter of `150e-6 m`, density of `2700 kg/m³`, and coarse-graining factor `2.44`. Change those values in the script for another case. It writes `pyoutputs/model_summary.json` with `(value, standard_error)` pairs for `Overshoot`, `DHR`, and `Hysteresis`.

Equivalent driver code:

```python
from prepost import ModelAnalysis

model = ModelAnalysis(
    pressure_path="CFD/postProcessing/cuttingPlane/",
    nprobes=5,
    velcfg_path="prepost/velcfg.txt",
    dump2csv=False,
    plots_dir="plots/",
)
model.define_params(diameter=150e-6, rho_p=2700, cg_factor=2.44)
print(model.model_summary())
```

`ModelAnalysis` loads pressure, contacts, and void fraction during construction; construct it only after all required output exists.

Next: [inputs and model assumptions](LIBRARY_DOCUMENTATION.md) or [the API reference](API_REFERENCE.md).
