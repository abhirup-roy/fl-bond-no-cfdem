# Quick Start Guide - Fluidised Bed Analysis Library

## Python Quick Start

### 1. Basic Setup

```bash
cd prepost
pip install -r requirements.txt
```

### 2. Simple Pressure Plot

```python
from prepost.plotting import FlBedPlot

plot = FlBedPlot(
    pressure_path="CFD/postProcessing/cuttingPlane/",
    nprobes=5,
    velcfg_path="velcfg.txt"
)

# Plot pressure vs time
plot.plot_pressure(slice_dirn="z", x_var="time")
```

### 3. Calculate Bond Numbers

```python
from prepost.model_analysis import ModelAnalysis

model = ModelAnalysis(nprobes=5)
model.define_params(diameter=1e-3, rho_p=2600, bed_mass=0.5)

# Get all bond numbers
results = model.model_summary()
print(results)
# Output: {'Overshoot': 0.024, 'DHR': 0.031, 'Hysteresis': 0.029}
```

### 4. Plot Void Fraction

```python
plot.plot_voidfrac(slice_dirn="z", x_var="velocity")
```

### 5. Contact Area Analysis

```python
plot.plot_contactarea(csv_path="DEM/post/collisions.csv")
```

---

## Julia Quick Start

### 1. Setup

```bash
cd prepost
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

### 2. Basic Usage

```julia
using FluidisedBedAnalysis

flbed = FluidisedBed(
    presure_path="CFD/postProcessing/cuttingPlane/",
    n_probes=5,
    dump2csv=false,
    velcfg_path="velcfg.txt",
    plots_dir="plots/"
)

# Plot pressure
plot_pressure(flbed, x_var="time", slice_dirn='z')
plot_pressure(flbed, x_var="velocity", slice_dirn='z')
```

### 3. Bond Number Calculations

```julia
sim_params(
    flbed,
    p_diameter=1e-3,
    rho_p=2600.0,
    cg_factor=1.0,
    poisson_ratio=0.25,
    youngs_modulus=70e9,
    ced=0.5
)

results = model_summary(flbed)
for (key, val) in results
    println("$key: $val")
end
```

---

## Expected Directory Structure

```
fl-bond-no-cfdem/
├── CFD/
│   └── postProcessing/
│       └── cuttingPlane/
│           ├── 0.001/
│           │   ├── p_zNormal0.vtk
│           │   ├── p_zNormal1.vtk
│           │   └── ...
│           ├── 0.002/
│           └── ...
├── DEM/
│   └── post/
│       ├── collisions.csv
│       ├── [dump files]
│       └── vtk/
├── plots/
├── velcfg.txt
└── prepost/
```

---

## Common Tasks

### Plot Pressure vs Velocity with Hysteresis

```python
plot = FlBedPlot(...)
plot.plot_pressure(
    x_var="velocity",
    slice_dirn="z",
    use_slices=True,
    png_name="pressure_velocity_hysteresis"
)
```

### Calculate DHR Model Only

```python
model = ModelAnalysis(nprobes=5)
model.define_params(diameter=1e-3, rho_p=2600, bed_mass=0.5)
dhr_value = model.dhr_model()
print(f"DHR: {dhr_value}")
```

### Export Probe 0 Data

```python
plot.plot_pressure(
    x_var="velocity",
    slice_dirn="z",
    dump_probe0=True
)
# Saves to: plots/probe0_plot_voidfrac.npy
```

### Convert LIGGGHTS to VTK

```python
from prepost.xtra_utils import liggghts2vtk

liggghts2vtk(
    timestep=5e-6,
    dump_every=5
)
# Outputs: DEM/post/vtk/*.vtk
```

### Calculate Mean Squared Displacement

```python
from prepost.xtra_utils import msq_displ

msd = msq_displ(
    dump_dir="DEM/post/",
    plot=True,
    dump=True
)
```

---

## Parameters Reference

### Particle Properties
- **Diameter:** 1e-3 to 1e-4 m (typical: 500 μm)
- **Density:** 2600 kg/m³ (silica), 1600 kg/m³ (coal), 3000+ (minerals)
- **Poisson's Ratio:** 0.20-0.30 (typical: 0.25)
- **Young's Modulus:** 50-100 GPa (typical: 70 GPa)

### Aggregation Methods
- **"cdf_median"** - CDF-based median (recommended)
- **"mean"** - Arithmetic mean
- **"median"** - Standard median

### Slice Directions
- **'z'** - Multiple probe points (z-normal slices)
- **'y'** - Single aggregated value (y-normal slices)

---

## Troubleshooting Quick Reference

| Error | Solution |
|-------|----------|
| `FileNotFoundError` | Check file paths exist |
| `LoadError: VTK file not found` | Verify simulation completed |
| `ValueError: Invalid aggregation` | Use valid agg method: `"cdf_median"` |
| `ArgumentError: parameters not set` | Call `define_params()` first |
| `Division by zero` | Check n_atoms > 0 in CSV |

---

## Output Files

| File | Description |
|------|-------------|
| `plots/pressure_*.png` | Pressure plots |
| `plots/voidfrac_*.png` | Void fraction plots |
| `plots/probe0_plot_*.npy` | Probe 0 numpy arrays |
| `DEM/post/vtk/*.vtk` | Converted LIGGGHTS data |
| `pyoutputs/msd.npy` | Mean squared displacement |

