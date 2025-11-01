# Fluidised Bed Analysis Library Documentation

## Overview

This library provides a comprehensive suite of tools for pre/post-processing simulations of fluidised bed systems using CFDEM coupling between OpenFOAM (CFD) and LIGGGHTS (DEM). The library includes both **Python** and **Julia** implementations with overlapping functionality for data analysis, visualization, and model calculations.

**Author:** Abhirup Roy  
**Email:** axr154@bham.ac.uk  
**License:** MIT  
**Version:** 0.1  
**Status:** Development

---

## Table of Contents

1. [Python Library](#python-library)
   - [Installation](#python-installation)
   - [Modules](#python-modules)
   - [Classes & Functions](#python-classes--functions)
   - [Usage Examples](#python-usage-examples)

2. [Julia Library](#julia-library)
   - [Installation](#julia-installation)
   - [Modules](#julia-modules)
   - [Functions](#julia-functions)
   - [Usage Examples](#julia-usage-examples)

3. [Shared Concepts](#shared-concepts)
4. [Data Formats](#data-formats)
5. [Troubleshooting](#troubleshooting)

---

# Python Library

## Python Installation

### Requirements

The library requires Python 3.7+ with the following dependencies:

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `pandas >= 1.1.5` - Data manipulation and analysis
- `numpy >= 1.19.5` - Numerical computing
- `matplotlib >= 3.3.4` - Plotting and visualization
- `pyvista >= 0.33.3` - 3D visualization (VTK support)
- `pyevtk >= 1.6.0` - VTK file generation
- `Jinja2 >= 3.1.6` - Template processing

### Setup

```bash
cd prepost
python -m pip install -e .
```

---

## Python Modules

### 1. `plotting.py` - Core Visualization Module

The main module for plotting pressure and void fraction data from CFDEM simulations.

#### Class: `FlBedPlot`

**Purpose:** Main interface for reading and visualizing fluidised bed simulation data.

**Constructor:**
```python
FlBedPlot(
    pressure_path: str,
    nprobes: int,
    velcfg_path: str,
    dump2csv: bool = True,
    plots_dir: str = "plots/"
)
```

**Parameters:**
- `pressure_path` (str): Path to pressure data. For slices, point to `CFD/postProcessing/cuttingPlane/`
- `nprobes` (int): Number of pressure probes in simulation
- `velcfg_path` (str): Path to velocity configuration file
- `dump2csv` (bool): Save probe data to CSV files
- `plots_dir` (str): Directory to save generated plots

**Raises:**
- `FileNotFoundError`: If pressure_path, velcfg_path, or plots_dir doesn't exist

---

### `FlBedPlot` Methods

#### `plot_pressure()`

Plot pressure data from simulation against time or velocity.

**Signature:**
```python
def plot_pressure(
    self,
    x_var: str,
    png_name: Optional[str] = None,
    use_slices: Optional[bool] = True,
    slice_dirn: Optional[str] = None,
    y_agg: Optional[str] = None,
    dump_probe0: Optional[bool] = True
) -> None
```

**Parameters:**
- `x_var` (str): Variable to plot against. Options: `"time"` or `"velocity"`
- `png_name` (str, optional): Custom output filename. Auto-generated if None
- `use_slices` (bool): Read from VTK slices (True) or text probes (False)
- `slice_dirn` (str, optional): Slice direction. `"z"` (z-normal) or `"y"` (y-normal)
- `y_agg` (str, optional): Aggregation for y-slices. Options: `"cdf_median"`, `"mean"`, `"median"`
- `dump_probe0` (bool): Export probe 0 data to numpy file

**Example:**
```python
plot = FlBedPlot(pressure_path="CFD/postProcessing/cuttingPlane/",
                 nprobes=5, velcfg_path="velcfg.txt")

# Plot pressure vs time
plot.plot_pressure(slice_dirn="z", x_var="time")

# Plot pressure vs velocity with hysteresis
plot.plot_pressure(slice_dirn="z", x_var="velocity")
```

**Output:**
- PNG file saved to `plots_dir`
- Optional numpy array of probe 0 data

---

#### `plot_voidfrac()`

Plot void fraction data against time or velocity.

**Signature:**
```python
def plot_voidfrac(
    self,
    slice_dirn: str,
    x_var: str,
    post_dir: str = "CFD/postProcessing/cuttingPlane/",
    png_name: Optional[str] = None,
    dump_probe0: bool = True
) -> None
```

**Parameters:**
- `slice_dirn` (str): Slice direction. `"z"` or `"y"`
- `x_var` (str): X-axis variable. `"time"` or `"velocity"`
- `post_dir` (str): Path to postprocessing directory with void fraction VTK files
- `png_name` (str, optional): Custom filename
- `dump_probe0` (bool): Export probe 0 data to numpy file

**Example:**
```python
# Plot void fraction vs velocity
plot.plot_voidfrac(slice_dirn="z", x_var="velocity", png_name="voidfrac_velocity")

# Plot void fraction vs time
plot.plot_voidfrac(slice_dirn="y", x_var="time")
```

---

#### `plot_contactarea()`

Plot contact area per particle from DEM collision data.

**Signature:**
```python
def plot_contactarea(
    self,
    csv_path: str = "DEM/post/collisions.csv",
    png_name: Optional[str] = None
) -> None
```

**Parameters:**
- `csv_path` (str): Path to collision CSV file
- `png_name` (str, optional): Custom output filename

**Example:**
```python
plot.plot_contactarea(csv_path="DEM/post/collisions.csv", png_name="contact_area")
```

**Expected CSV format:**
```
time    n_atoms    a_contact    n_contact
0.001   1000       0.0052       2145
0.002   1000       0.0051       2143
...
```

---

#### Helper Methods

**`find_cdfmedian(arr: np.ndarray) -> float`**

Calculate the median value using cumulative distribution function.

```python
arr = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
median = plot.find_cdfmedian(arr)  # Returns 2.0
```

---

### 2. `model_analysis.py` - Bond Number Models

Module for calculating bond numbers using different models based on fluidisation hysteresis.

#### Class: `ModelAnalysis`

Inherits from `FlBedPlot` and adds bond number calculation capabilities.

**Constructor:**
```python
ModelAnalysis(**kwargs)
```

**Parameters (all optional):**
- `pressure_path` (str): Default: `"CFD/postProcessing/cuttingPlane/"`
- `nprobes` (int): Default: `5`
- `velcfg_path` (str): Default: `"prepost/velcfg.txt"`
- `dump2csv` (bool): Default: `False`
- `plots_dir` (str): Default: `"plots/"`

**Example:**
```python
model = ModelAnalysis(nprobes=5)
```

---

#### `ModelAnalysis` Methods

**`define_params(diameter, rho_p, bed_mass, cg_factor=None)`**

Define parameters for bond number calculations.

**Parameters:**
- `diameter` (float): Particle diameter in meters
- `rho_p` (float): Particle density in kg/m³
- `bed_mass` (float): Total bed mass in kg
- `cg_factor` (float, optional): Coarse-graining factor for scaled simulations

**Example:**
```python
model.define_params(
    diameter=1e-3,        # 1 mm particles
    rho_p=2600,          # Silica density
    bed_mass=0.5,
    cg_factor=2.0        # Coarse-grained by factor of 2
)
```

**Note:** If `cg_factor` is provided, particle properties are scaled:
- Effective density: `rho_p / cg_factor`
- Effective diameter: `diameter * cg_factor`

---

**`overshoot_model() -> float`**

Calculate bond number using pressure overshoot model (Hsu, Huang and Kuo 2018).

**Formula:**
$$Bo = \frac{6 \Delta P}{\bar{N}_c^2 (1-\epsilon) d_p \rho_p g}$$

where:
- $\Delta P$ = pressure overshoot (max pressure - steady-state)
- $\bar{N}_c$ = average contact number per particle
- $\epsilon$ = void fraction
- $d_p$ = particle diameter
- $\rho_p$ = particle density
- $g$ = gravitational acceleration

**Example:**
```python
bo_overshoot = model.overshoot_model()
print(f"Bond number (overshoot): {bo_overshoot:.4f}")
```

---

**`dhr_model() -> float`**

Calculate expansion parameter using Davidson-Harrison-Richardson model (Soleimani et al. 2021).

**Formula:**
$$n = \left(\frac{\epsilon_2}{\epsilon_1}\right)^3 \frac{1-\epsilon_1}{1-\epsilon_2} - 1$$

where $\epsilon_1$, $\epsilon_2$ are void fractions at minimum fluidization during up and down flow.

**Example:**
```python
dhr = model.dhr_model()
print(f"DHR model parameter: {dhr:.4f}")
```

---

**`hyst_model() -> float`**

Calculate hysteresis parameter (Affleck et al. 2023).

**Formula:**
$$Bo = \frac{P_1 - P_2}{P_{ss} \Delta N_c}$$

where:
- $P_1$ = max pressure during fluidisation
- $P_2$ = pressure at $u_{mf}$ during defluidisation
- $P_{ss}$ = steady-state pressure
- $\Delta N_c$ = contact number difference

**Example:**
```python
bo_hyst = model.hyst_model()
print(f"Bond number (hysteresis): {bo_hyst:.4f}")
```

---

**`model_summary() -> dict`**

Calculate bond numbers using all three models.

**Returns:** Dictionary with keys:
- `"Overshoot"`: Overshoot model result
- `"DHR"`: DHR model result
- `"Hysteresis"`: Hysteresis model result

**Example:**
```python
results = model.model_summary()
print(results)
# Output: {'Overshoot': 0.0245, 'DHR': 0.0312, 'Hysteresis': 0.0289}
```

---

### 3. `xtra_utils.py` - Utility Functions

Utility functions for processing LIGGGHTS dump files and analyzing particle dynamics.

#### `liggghts2vtk()`

Convert LIGGGHTS dump files to VTK format for 3D visualization.

**Signature:**
```python
def liggghts2vtk(
    timestep: float = 5e-6,
    vtk_dir: Optional[str] = None,
    dump_every: Optional[int] = None,
    liggghts_dump_dir: Optional[str] = None,
    file_suffix: str = ".liggghts_run"
) -> None
```

**Parameters:**
- `timestep` (float): Simulation timestep in seconds (default: 5e-6)
- `vtk_dir` (str): Output directory for VTK files (default: `"DEM/post/vtk/"`)
- `dump_every` (int, optional): Process every nth file (default: all files)
- `liggghts_dump_dir` (str): Directory containing LIGGGHTS dump files (default: `"DEM/post/"`)
- `file_suffix` (str): Suffix of dump files (default: `".liggghts_run"`)

**Generated VTK Data:**
- Position (x, y, z)
- Velocity magnitude and components (vx, vy, vz)
- Particle radius
- Simulation time

**Example:**
```python
from prepost.xtra_utils import liggghts2vtk

liggghts2vtk(
    timestep=5e-6,
    vtk_dir="DEM/post/vtk/",
    dump_every=10  # Process every 10th dump
)
```

---

#### `msq_displ()`

Calculate mean squared displacement of particles.

**Signature:**
```python
def msq_displ(
    time_rng: Optional[tuple[float, float]] = None,
    dump_dir: str = "DEM/post",
    dump: bool = True,
    plot: bool = True,
    timestep: float = 5e-6,
    direction: Optional[str] = None
) -> pd.Series
```

**Parameters:**
- `time_rng` (tuple, optional): (start_time, end_time) for analysis range
- `dump_dir` (str): Directory with LIGGGHTS dump files
- `dump` (bool): Save results to numpy file
- `plot` (bool): Generate histogram plot
- `timestep` (float): Simulation timestep in seconds
- `direction` (str, optional): Displacement direction ('x', 'y', 'z') (default: 'z')

**Returns:** `pd.Series` with particle IDs as index and MSD as values

**Output Files:**
- `pyoutputs/msd.npy` - 2D array (particle_ids, msd_values)
- `pyoutputs/msd_histogram.png` - Histogram plot

**Example:**
```python
from prepost.xtra_utils import msq_displ

# Calculate MSD for full simulation
msd = msq_displ(dump_dir="DEM/post/", plot=True, dump=True)

# Calculate MSD for specific time range
msd_subset = msq_displ(
    time_rng=(0.1, 0.5),  # 100-500 ms
    dump_dir="DEM/post/"
)
```

---

## Python Usage Examples

### Complete Analysis Workflow

```python
from prepost.plotting import FlBedPlot
from prepost.model_analysis import ModelAnalysis
import matplotlib.pyplot as plt

# 1. Initialize pressure plotting
pressure_plot = FlBedPlot(
    pressure_path="CFD/postProcessing/cuttingPlane/",
    nprobes=5,
    velcfg_path="velcfg.txt",
    plots_dir="plots/"
)

# 2. Plot pressure data
pressure_plot.plot_pressure(
    slice_dirn="z",
    x_var="time",
    png_name="pressure_time_z"
)

pressure_plot.plot_pressure(
    slice_dirn="z",
    x_var="velocity",
    png_name="pressure_velocity_z"
)

# 3. Plot void fraction
pressure_plot.plot_voidfrac(
    slice_dirn="z",
    x_var="velocity",
    png_name="voidfrac_velocity_z"
)

# 4. Calculate bond numbers
model = ModelAnalysis(nprobes=5)
model.define_params(
    diameter=1e-3,
    rho_p=2600,
    bed_mass=0.5
)

results = model.model_summary()
print("Bond Number Results:")
for model_name, value in results.items():
    print(f"  {model_name}: {value:.6f}")

# 5. Plot contact area
model.plot_contactarea(
    csv_path="DEM/post/collisions.csv",
    png_name="contact_area"
)

plt.show()
```

---

# Julia Library

## Julia Installation

### Requirements

Julia 1.6+ with the following packages (see `Project.toml`):

```julia
] add CSV DataFrames Plots Statistics StatsBase VTKDataIO WriteVTK
```

### Setup

```bash
cd prepost
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

---

## Julia Modules

### Module Structure

```
FluidisedBedAnalysis.jl (main module)
├── CurvePlots.jl       (plotting functionality)
└── BoModels.jl         (bond number models)
```

---

## Julia Functions

### 1. `CurvePlots` Module - Plotting

#### Type: `FluidisedBed`

Mutable struct representing a fluidised bed simulation.

**Definition:**
```julia
@kwdef mutable struct FluidisedBed
    presure_path::String
    n_probes::Int
    dump2csv::Bool
    velcfg_path::String
    plots_dir::String
    
    # Optional velocity data
    t::Vector{Float32} = Vector{Float32}()
    v_z::Vector{Float32} = Vector{Float32}()
    void_frac_path::Union{String, Nothing} = nothing
    
    # Internal state
    _timeser_df::Union{DataFrame, Nothing} = nothing
    _df_store::Union{Array, Nothing} = nothing
    
    # Physical parameters
    p_diameter::Union{Real, Nothing} = nothing
    𝜌_p::Union{Real, Nothing} = nothing
    poisson_ratio::Union{Real, Nothing} = nothing
    youngs_modulus::Union{Real, Nothing} = nothing
    ced::Union{Real, Nothing} = nothing
    
    # Model storage
    _model_store = Dict{String, Array{Float32}}()
end
```

**Constructor:**
```julia
flbed = FluidisedBed(
    presure_path="CFD/postProcessing/cuttingPlane/",
    n_probes=5,
    dump2csv=false,
    velcfg_path="velcfg.txt",
    plots_dir="plots/"
)
```

---

#### `plot_pressure()`

Generate pressure plots from fluidised bed simulation.

**Signature:**
```julia
function plot_pressure(
    flbed::FluidisedBed;
    x_var::String="velocity",
    png_name=nothing,
    use_slices::Bool=true,
    slice_dirn::Char='z',
    y_agg=nothing
)
```

**Parameters:**
- `flbed::FluidisedBed`: The fluidised bed object
- `x_var::String`: Plot against `"time"` or `"velocity"`
- `png_name`: Custom filename (auto-generated if `nothing`)
- `use_slices::Bool`: Use VTK slices or text probes
- `slice_dirn::Char`: Slice direction `'z'` or `'y'`
- `y_agg`: Aggregation method for y-slices (`"cdf_median"`, `"mean"`, `"median"`)

**Example:**
```julia
plot_pressure(flbed, x_var="velocity", slice_dirn='z')
```

---

#### `plot_voidfrac()`

Generate void fraction plots.

**Signature:**
```julia
function plot_voidfrac(
    flbed::FluidisedBed;
    slice_dirn::Char='y',
    x_var::String="velocity",
    png_name=nothing
)
```

**Example:**
```julia
plot_voidfrac(flbed, slice_dirn='z', x_var="time")
```

---

#### Helper Functions

**`_find_cdfmedian(x::Vector{Float32}) -> Float32`**

Calculate CDF-based median value with linear interpolation if exact median doesn't exist.

**`_read_probetxt(flbed::FluidisedBed)`**

Read velocity configuration from text file.

**`_calc_vel!(flbed::FluidisedBed, time_df::DataFrame)`**

Calculate and add velocity and direction columns to DataFrame.

**`_probe2df(flbed, use_slices, slice_dirn, y_agg) -> DataFrame`**

Convert probe data to DataFrame, with optional VTK slice reading.

**`_read_vf(flbed, post_dir, slice_dirn) -> DataFrame`**

Read void fraction data from VTK files.

---

### 2. `BoModels` Module - Bond Number Calculations

#### `sim_params()`

Set simulation parameters for bond number calculations.

**Signature:**
```julia
function sim_params(
    flbed::FluidisedBed;
    p_diameter::Float64,
    rho_p::Real,
    cg_factor::Real,
    poisson_ratio::Real,
    youngs_modulus::Real,
    ced::Union{Real, Nothing} = nothing
)
```

**Parameters:**
- `p_diameter`: Particle diameter (meters)
- `rho_p`: Particle density (kg/m³)
- `cg_factor`: Coarse-graining factor
- `poisson_ratio`: Poisson's ratio of particle material
- `youngs_modulus`: Young's modulus (Pa)
- `ced`: Cohesive energy density (optional)

**Example:**
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
```

---

#### `overshoot_model()`

Calculate bond number from pressure overshoot.

**Signature:**
```julia
function overshoot_model(flbed::FluidisedBed)::Float64
```

**Returns:** Bond number from overshoot model

**Example:**
```julia
bo = overshoot_model(flbed)
```

---

#### `dhr_model()`

Calculate expansion parameter using DHR model.

**Signature:**
```julia
function dhr_model(flbed::FluidisedBed)
```

---

#### `hyst_model()`

Calculate hysteresis parameter.

**Signature:**
```julia
function hyst_model(flbed::FluidisedBed)::Float64
```

---

#### `intrinsic_bond_num()`

Calculate intrinsic bond number.

**Signature:**
```julia
function intrinsic_bond_num(flbed::FluidisedBed)
```

**Formula:**
$$Bo_n = \frac{27\pi^3 (ced)^3 R_{eq}^2}{64 m_p g (E_{eq})^2}$$

where:
- $E_{eq} = \frac{E}{2(1-\nu)^2}$
- $R_{eq} = \frac{d_p}{4}$
- $m_p = \frac{\rho_p d_p^3}{6}$

---

#### `model_summary()`

Calculate all bond number models.

**Signature:**
```julia
function model_summary(flbed::FluidisedBed)::Dict
```

**Returns:** Dictionary with keys:
- `"overshoot_model"`
- `"dhr_model"`
- `"hyst_model"`
- `"intrinsic_bo"`

**Example:**
```julia
results = model_summary(flbed)
println(results)
```

---

## Julia Usage Examples

### Complete Analysis Workflow

```julia
using FluidisedBedAnalysis

# 1. Create FluidisedBed object
flbed = FluidisedBed(
    presure_path="CFD/postProcessing/cuttingPlane/",
    n_probes=5,
    dump2csv=false,
    velcfg_path="velcfg.txt",
    plots_dir="plots/"
)

# 2. Plot pressure data
plot_pressure(flbed, x_var="time", slice_dirn='z')
plot_pressure(flbed, x_var="velocity", slice_dirn='z')

# 3. Plot void fraction
plot_voidfrac(flbed, slice_dirn='z', x_var="velocity")

# 4. Set simulation parameters
sim_params(
    flbed,
    p_diameter=1e-3,
    rho_p=2600.0,
    cg_factor=1.0,
    poisson_ratio=0.25,
    youngs_modulus=70e9,
    ced=0.5
)

# 5. Calculate bond numbers
results = model_summary(flbed)
println("Bond Number Results:")
for (key, val) in results
    println("  $key: $val")
end
```

---

# Shared Concepts

## Fluidisation Process

All models analyze the fluidisation-defluidisation hysteresis cycle:

1. **Fluidisation Phase (Up):** Velocity increases from rest to maximum
2. **Maximum Phase (Max):** Velocity held at maximum value
3. **Defluidisation Phase (Down):** Velocity decreases back to rest

This creates hysteresis in pressure and void fraction measurements due to:
- Contact network rebuilding
- Settling of particles
- Drag force variations

## Slice Directions

- **Z-direction (`'z'`):** Slices perpendicular to z-axis (multiple probe points)
- **Y-direction (`'y'`):** Slices perpendicular to y-axis (single aggregated value)

## Velocity Configuration File Format

Expected format (`velcfg.txt`):

```
(0.00000000e+00 0.00000000e+00)
(1.00000000e-02 0.10000000e+00)
(2.00000000e-02 0.20000000e+00)
(3.00000000e-02 0.20000000e+00)
(4.00000000e-02 0.10000000e+00)
(5.00000000e-02 0.00000000e+00)
```

Format: `(time_seconds fluidizing_velocity_m/s)`

---

# Data Formats

## Collision CSV Format

Expected format for `DEM/post/collisions.csv`:

```csv
time,n_atoms,a_contact,n_contact
0.001,1000,0.0052,2145
0.002,1000,0.0051,2143
0.003,1000,0.0050,2140
```

**Columns:**
- `time`: Simulation time (seconds)
- `n_atoms`: Number of particles
- `a_contact`: Total contact area (m²)
- `n_contact`: Total number of contacts

---

## VTK File Structure

CFDEM outputs VTK format files in:
- **Pressure slices:** `CFD/postProcessing/cuttingPlane/[time]/p_[direction][index].vtk`
- **Void fraction slices:** `CFD/postProcessing/cuttingPlane/[time]/voidfraction_[direction][index].vtk`
- **LIGGGHTS dumps:** `DEM/post/[name].liggghts_run`

---

# Troubleshooting

## Common Issues

### Issue: "FileNotFoundError: Pressure data at ... does not exist"

**Solution:** Verify paths are correct and simulations have completed:
```python
import os
pressure_path = "CFD/postProcessing/cuttingPlane/"
assert os.path.exists(pressure_path), f"Path does not exist: {pressure_path}"
```

### Issue: "LoadError: VTK file not found"

**Solution:** Check VTK files are present and named correctly:
```bash
ls CFD/postProcessing/cuttingPlane/0/p_zNormal*.vtk
```

### Issue: "ValueError: Invalid aggregation method"

**Solution:** Use only valid aggregation methods:
- `"cdf_median"` - CDF-based median
- `"mean"` - Arithmetic mean
- `"median"` - Standard median

### Issue: "ArgumentError: FluidisedBed parameters not set"

**Solution:** Call `define_params()` or `sim_params()` before calculating bond numbers:
```python
model.define_params(diameter=1e-3, rho_p=2600, bed_mass=0.5)
```

### Issue: "Division by zero" in `_read_collisions()`

**Solution:** Check that `n_atoms` is non-zero in collision CSV:
```python
df = pd.read_csv("DEM/post/collisions.csv")
assert (df['n_atoms'] > 0).all(), "n_atoms contains zero values"
```

## Debugging Tips

### Enable Detailed Output

**Python:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Julia:**
```julia
using Logging
Logging.disable_logging(Logging.LogLevel.Debug)  # Remove to enable debug logs
```

### Check Data Integrity

**Python:**
```python
plot = FlBedPlot(...)
df = plot._probe2df(use_slices=True, slice_dirn='z', y_agg=None)
print(df.describe())
print(df.isna().sum())  # Check for missing values
```

**Julia:**
```julia
plot_pressure(flbed)  # Functions print intermediate values
```

---

## Performance Considerations

### Large Datasets

For large simulations with many timesteps:

1. **Python:** Use `dump_every` parameter to subsample:
```python
liggghts2vtk(dump_every=10)  # Process every 10th file
```

2. **Julia:** Process data incrementally
```julia
# Implement streaming for large VTK files
```

### Memory Usage

- Void fraction for 1000 timesteps × 5 probes: ~20 KB
- VTK files are not loaded entirely into memory (streaming approach)
- Contact data: depends on number of collisions

---

## References

### Publications Referenced

1. **Overshoot Model:** Hsu, Huang and Kuo (2018) - Pressure overshoot during fluidization
2. **DHR Model:** Soleimani et al. (2021) - Dimensionless Height Ratio expansion parameter
3. **Hysteresis Model:** Affleck et al. (2023) - Hysteresis-based bond number estimation

---

## Support and Contributing

**Author:** Abhirup Roy  
**Email:** axr154@bham.ac.uk

For issues or contributions, please contact the author or submit a pull request to the repository.

---

## License

MIT License - See LICENSE file for details

