# API Reference - Fluidised Bed Analysis Library

## Python API Reference

### `prepost.plotting` Module

#### Class: `FlBedPlot`

Main class for visualization and analysis of fluidised bed data.

**Methods:**

| Method | Purpose | Returns |
|--------|---------|---------|
| `plot_pressure()` | Plot pressure vs time or velocity | None |
| `plot_voidfrac()` | Plot void fraction vs time or velocity | None |
| `plot_contactarea()` | Plot contact area from DEM data | None |
| `find_cdfmedian()` | Calculate CDF-based median | float |
| `_probe2df()` | Read probe data to DataFrame | pd.DataFrame |
| `_read_probetxt()` | Parse velocity config file | None |
| `_calc_vel()` | Add velocity columns to DataFrame | None |
| `_read_voidfrac()` | Read void fraction VTK data | pd.DataFrame |
| `_read_collisions()` | Read collision CSV data | pd.DataFrame |

---

### `prepost.model_analysis` Module

#### Class: `ModelAnalysis(FlBedPlot)`

Extends `FlBedPlot` with bond number calculation methods.

**Methods:**

| Method | Purpose | Returns |
|--------|---------|---------|
| `define_params()` | Set particle and material properties | None |
| `overshoot_model()` | Calculate overshoot bond number | float |
| `dhr_model()` | Calculate DHR expansion parameter | float |
| `hyst_model()` | Calculate hysteresis parameter | float |
| `model_summary()` | Calculate all models | dict |
| `_access_pressures()` | Load and process pressure data | tuple[pd.Series, pd.Series] |
| `_access_voidfrac()` | Load and process void fraction | tuple[pd.Series, pd.Series] |
| `_access_contactn()` | Load and process contact data | tuple[pd.Series, pd.Series] |
| `_store_data()` | Cache data for repeated access | None |

---

### `prepost.xtra_utils` Module

#### Function: `liggghts2vtk()`

Convert LIGGGHTS dump files to VTK format.

**Signature:**
```python
liggghts2vtk(
    timestep: float = 5e-6,
    vtk_dir: Optional[str] = None,
    dump_every: Optional[int] = None,
    liggghts_dump_dir: Optional[str] = None,
    file_suffix: str = ".liggghts_run"
) -> None
```

**VTK Output Data:**
- `x`, `y`, `z` - Particle positions
- `vx`, `vy`, `vz` - Velocity components
- `velocity` - Velocity magnitude
- `radius` - Particle radius
- `time` - Simulation time

---

#### Function: `msq_displ()`

Calculate mean squared displacement of particles.

**Signature:**
```python
msq_displ(
    time_rng: Optional[tuple[float, float]] = None,
    dump_dir: str = "DEM/post",
    dump: bool = True,
    plot: bool = True,
    timestep: float = 5e-6,
    direction: Optional[str] = None
) -> pd.Series
```

**Returns:** Series with particle ID index and MSD values

**Output Files:**
- `pyoutputs/msd.npy` - 2D array of (particle_id, msd)
- `pyoutputs/msd_histogram.png` - Histogram plot

---

## Julia API Reference

### `FluidisedBedAnalysis` Module

**Exports:**
- `FluidisedBed` - Main struct
- `plot_pressure()` - Plot pressure data
- `plot_voidfrac()` - Plot void fraction
- `sim_params()` - Set parameters
- `overshoot_model()` - Overshoot model
- `dhr_model()` - DHR model
- `hyst_model()` - Hysteresis model
- `intrinsic_bond_num()` - Intrinsic bond number
- `model_summary()` - All models

---

### `CurvePlots` Module

#### Type: `FluidisedBed`

```julia
@kwdef mutable struct FluidisedBed
    presure_path::String
    n_probes::Int
    dump2csv::Bool
    velcfg_path::String
    plots_dir::String
    
    t::Vector{Float32} = Vector{Float32}()
    v_z::Vector{Float32} = Vector{Float32}()
    void_frac_path::Union{String, Nothing} = nothing
    
    _timeser_df::Union{DataFrame, Nothing} = nothing
    _df_store::Union{Array, Nothing} = nothing
    p_diameter::Union{Real, Nothing} = nothing
    𝜌_p::Union{Real, Nothing} = nothing
    poisson_ratio::Union{Real, Nothing} = nothing
    youngs_modulus::Union{Real, Nothing} = nothing
    ced::Union{Real, Nothing} = nothing
    _model_store::Dict{String, Array{Float32}} = Dict()
end
```

---

#### Function: `plot_pressure()`

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
- `x_var`: `"time"` or `"velocity"`
- `slice_dirn`: `'z'` or `'y'`
- `y_agg`: `"cdf_median"`, `"mean"`, or `"median"`

---

#### Function: `plot_voidfrac()`

**Signature:**
```julia
function plot_voidfrac(
    flbed::FluidisedBed;
    slice_dirn::Char='y',
    x_var::String="velocity",
    png_name=nothing
)
```

---

#### Helper Functions

**`_find_cdfmedian(x::Vector{Float32}) -> Float32`**

CDF-based median with linear interpolation.

**`_read_probetxt(flbed::FluidisedBed) -> Nothing`**

Parse velocity configuration file.

**`_calc_vel!(flbed::FluidisedBed, time_df::DataFrame) -> Nothing`**

Add velocity and direction columns to DataFrame.

**`_probe2df(flbed, use_slices, slice_dirn, y_agg) -> DataFrame`**

Convert probe data to DataFrame.

**`_read_vf(flbed, post_dir, slice_dirn) -> DataFrame`**

Read void fraction from VTK files.

---

### `BoModels` Module

#### Function: `sim_params()`

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

---

#### Function: `overshoot_model()`

**Signature:**
```julia
function overshoot_model(flbed::FluidisedBed) -> Float64
```

**Formula:**
$$Bo = \frac{6(P_{max} - P_{ss})}{\bar{N}_c^2(1-\epsilon)d_p\rho_p g}$$

---

#### Function: `dhr_model()`

**Signature:**
```julia
function dhr_model(flbed::FluidisedBed) -> Float64
```

**Formula:**
$$n = \left(\frac{\epsilon_{down}}{\epsilon_{up}}\right)^3 \frac{1-\epsilon_{up}}{1-\epsilon_{down}} - 1$$

---

#### Function: `hyst_model()`

**Signature:**
```julia
function hyst_model(flbed::FluidisedBed) -> Float64
```

**Formula:**
$$Bo = \frac{P_{max} - P_{down}}{P_{ss} \Delta N_c}$$

---

#### Function: `intrinsic_bond_num()`

**Signature:**
```julia
function intrinsic_bond_num(flbed::FluidisedBed) -> Float64
```

**Formula:**
$$Bo_n = \frac{27\pi^3(ced)^3 R_{eq}^2}{64 m_p g (E_{eq})^2}$$

where:
- $E_{eq} = \frac{E}{2(1-\nu)^2}$ (equivalent Young's modulus)
- $R_{eq} = \frac{d_p}{4}$ (equivalent radius)
- $m_p = \frac{\rho_p d_p^3}{6}$ (particle mass)

---

#### Function: `model_summary()`

**Signature:**
```julia
function model_summary(flbed::FluidisedBed) -> Dict
```

**Returns:** Dictionary with keys:
- `"overshoot_model"` -> Float64
- `"dhr_model"` -> Float64
- `"hyst_model"` -> Float64
- `"intrinsic_bo"` -> Float64

---

## Data Structures

### Python DataFrames

#### Pressure DataFrame

| Column | Type | Description |
|--------|------|-------------|
| `time` | float | Simulation time (s) |
| `probe_0` | float | Pressure at probe 0 (Pa) |
| `probe_1` | float | Pressure at probe 1 (Pa) |
| ... | ... | ... |
| `v_z` | float | Vertical velocity (m/s) |
| `direction` | str | `"up"`, `"max"`, or `"down"` |

#### Void Fraction DataFrame

| Column | Type | Description |
|--------|------|-------------|
| `time` | float | Simulation time (s) |
| `probe_0` | float | Void fraction at probe 0 (-) |
| `probe_1` | float | Void fraction at probe 1 (-) |
| ... | ... | ... |
| `v_z` | float | Vertical velocity (m/s) |
| `direction` | str | Flow direction |

#### Contact DataFrame

| Column | Type | Description |
|--------|------|-------------|
| `time` | float | Simulation time (s) |
| `n_atoms` | int | Number of particles (-) |
| `a_contact` | float | Total contact area (m²) |
| `n_contact` | int | Total contacts (-) |
| `contactn` | float | Contacts per particle (-) |
| `v_z` | float | Vertical velocity (m/s) |
| `direction` | str | Flow direction |

---

### Julia DataFrames

Same structure as Python, with Float32 precision for numeric columns.

---

## Constants

### Physical Parameters

```python
# Gravitational acceleration
g = 9.81  # m/s²

# Pi
π = 3.14159...
```

### Material Properties (Examples)

| Material | Density (kg/m³) | Young's Modulus (GPa) | Poisson's Ratio |
|----------|------------------|----------------------|-----------------|
| Silica (SiO₂) | 2600 | 70-90 | 0.17-0.23 |
| Coal | 1600 | 5-8 | 0.25-0.35 |
| Aluminum | 2700 | 70 | 0.33 |
| Steel | 7850 | 200 | 0.30 |

---

## Environment Variables

None required. All paths configured via function parameters.

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity |
|-----------|------------|
| Reading single VTK file | O(n) - n = data points |
| Calculating CDF median | O(n log n) |
| Grouping by velocity/direction | O(n) |
| All model calculations | O(n) |

### Space Complexity

| Data | Size (approx) |
|------|----------------|
| 1000 timesteps, 5 probes pressure | ~40 KB |
| 1000 timesteps, 5 probes void fraction | ~40 KB |
| 10000 particles, 100 timesteps | ~4 MB |
| Single large VTK file (10M points) | 500+ MB |

---

## Error Handling

### Python Exceptions

| Exception | Condition | Solution |
|-----------|-----------|----------|
| `FileNotFoundError` | Missing data files | Verify paths and run simulation |
| `ValueError` | Invalid parameter | Check allowed values |
| `LoadError` | VTK file not found | Check file existence/naming |
| `AttributeError` | Parameters not set | Call `define_params()` |
| `TypeError` | Wrong data type | Check input types |

### Julia Exceptions

| Exception | Condition | Solution |
|-----------|-----------|----------|
| `LoadError` | File not found | Verify path exists |
| `ArgumentError` | Invalid argument | Check parameter values |
| `ErrorException` | Generic error | Check message details |

---

## Version Information

- **Python:** 3.7+
- **Julia:** 1.6+
- **Library Version:** 0.1 (Development)

