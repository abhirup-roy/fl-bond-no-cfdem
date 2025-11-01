# Fluidised Bed Analysis Library - Documentation Index

Complete documentation for the Python and Julia libraries for CFDEM fluidised bed simulation analysis.

Generated with the help of Claude Haiku 4.5

**Author:** Abhirup Roy  
**Version:** 0.1 (Development)  
**License:** MIT

---

## Documentation Files

### 📖 Main Documentation

1. **[LIBRARY_DOCUMENTATION.md](docs/LIBRARY_DOCUMENTATION.md)** - Complete Reference
   - Comprehensive API documentation
   - Class and function definitions
   - Installation instructions
   - Usage patterns and examples
   - **Start here for detailed information**

2. **[QUICK_START.md](docs/QUICK_START.md)** - Getting Started
   - Basic setup and installation
   - Simple code examples
   - Common use cases
   - Parameters reference
   - Troubleshooting quick reference
   - **Start here if you're new**

3. **[API_REFERENCE.md](docs/API_REFERENCE.md)** - Function Reference
   - Detailed method signatures
   - Parameter specifications
   - Return types and values
   - Data structure specifications
   - Performance characteristics
   - **Use this for quick API lookup**

4. **[EXAMPLES.md](docs/EXAMPLES.md)** - Code Examples
   - Complete analysis pipelines
   - Parametric studies
   - Particle tracking workflows
   - Interactive notebooks
   - Testing scripts
   - **Copy-paste ready code**

---

## Quick Navigation

### By Use Case

**I want to plot pressure data:**
→ [QUICK_START.md - Plot Pressure](docs/QUICK_START.md#plot-pressure-vs-velocity-with-hysteresis) | [LIBRARY_DOCUMENTATION.md - plot_pressure()](LIBRARY_DOCUMENTATION.md#plot_pressure)

**I want to calculate bond numbers:**
→ [QUICK_START.md - Bond Numbers](docs/QUICK_START.md#calculate-bond-numbers) | [LIBRARY_DOCUMENTATION.md - Model Analysis](LIBRARY_DOCUMENTATION.md#2-model_analysispy---bond-number-models)

**I want to analyze particle motion:**
→ [QUICK_START.md - Particle Motion](docs/QUICK_START.md#calculate-mean-squared-displacement) | [EXAMPLES.md - Particle Tracking](EXAMPLES.md#example-3-particle-tracking-analysis)

**I need the complete API:**
→ [API_REFERENCE.md](docs/API_REFERENCE.md)

**I want working code examples:**
→ [EXAMPLES.md](docs/EXAMPLES.md)

---

## By Language

### Python

| Task | Reference |
|------|-----------|
| Setup & Install | [QUICK_START.md](docs/QUICK_START.md#python-quick-start) |
| Plotting | [LIBRARY_DOCUMENTATION.md](docs/LIBRARY_DOCUMENTATION.md#1-plottingpy---core-visualization-module) |
| Bond Numbers | [LIBRARY_DOCUMENTATION.md](docs/LIBRARY_DOCUMENTATION.md#2-model_analysispy---bond-number-models) |
| Utilities | [LIBRARY_DOCUMENTATION.md](docs/LIBRARY_DOCUMENTATION.md#3-xtra_utilspy---utility-functions) |
| Examples | [EXAMPLES.md](docs/EXAMPLES.md#python-examples) |
| API Reference | [API_REFERENCE.md](docs/API_REFERENCE.md#python-api-reference) |

### Julia

| Task | Reference |
|------|-----------|
| Setup & Install | [QUICK_START.md](docs/QUICK_START.md#julia-quick-start) |
| Plotting | [LIBRARY_DOCUMENTATION.md](docs/LIBRARY_DOCUMENTATION.md#1-curveplot-module---plotting) |
| Bond Numbers | [LIBRARY_DOCUMENTATION.md](docs/LIBRARY_DOCUMENTATION.md#2-bomodels-module---bond-number-calculations) |
| Examples | [EXAMPLES.md](docs/EXAMPLES.md#julia-examples) |
| API Reference | [API_REFERENCE.md](docs/API_REFERENCE.md#julia-api-reference) |

---

## Key Concepts

### Three Bond Number Models

1. **Overshoot Model** (Hsu, Huang and Kuo 2018)
   - Based on pressure overshoot during fluidization
   - Formula: $Bo = \frac{6\Delta P}{\bar{N}_c^2(1-\epsilon)d_p\rho_p g}$
   - Reference: [LIBRARY_DOCUMENTATION.md](docs/LIBRARY_DOCUMENTATION.md#overshoot_model)

2. **DHR Model** (Soleimani et al. 2021)
   - Dimensionless Height Ratio
   - Formula: $n = \left(\frac{\epsilon_2}{\epsilon_1}\right)^3\frac{1-\epsilon_1}{1-\epsilon_2}-1$
   - Reference: [LIBRARY_DOCUMENTATION.md](docs/LIBRARY_DOCUMENTATION.md#dhr_model)

3. **Hysteresis Model** (Affleck et al. 2023)
   - Based on fluidization hysteresis
   - Formula: $Bo = \frac{P_1-P_2}{P_{ss}\Delta N_c}$
   - Reference: [LIBRARY_DOCUMENTATION.md](docs/LIBRARY_DOCUMENTATION.md#hyst_model)

---

## Data Formats

### Input Files Required

```
CFD/postProcessing/cuttingPlane/[time]/
├── p_zNormal0.vtk          # Pressure at different heights
├── p_zNormal1.vtk
├── ...
└── voidfraction_zNormal0.vtk  # Void fraction data

DEM/post/
├── collisions.csv          # Collision data (CSV format)
└── *.liggghts_run         # DEM dump files

velcfg.txt                 # Velocity configuration
```

### CSV Format (collisions.csv)

```
time,n_atoms,a_contact,n_contact
0.001,1000,0.0052,2145
0.002,1000,0.0051,2143
```

See [LIBRARY_DOCUMENTATION.md - Data Formats](LIBRARY_DOCUMENTATION.md#data-formats) for details.

---

## Installation

### Python

```bash
cd prepost
pip install -r requirements.txt
```

### Julia

```bash
cd prepost
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

Full instructions: [QUICK_START.md](docs/QUICK_START.md)

---

## Common Tasks

### 1. Create Basic Plot

**Python:**
```python
from prepost.plotting import FlBedPlot

plot = FlBedPlot(pressure_path="CFD/postProcessing/cuttingPlane/",
                 nprobes=5, velcfg_path="velcfg.txt")
plot.plot_pressure(slice_dirn="z", x_var="time")
```

**Julia:**
```julia
using FluidisedBedAnalysis

flbed = FluidisedBed(presure_path="CFD/postProcessing/cuttingPlane/",
                     n_probes=5, velcfg_path="velcfg.txt", plots_dir="plots/")
plot_pressure(flbed, x_var="time", slice_dirn='z')
```

### 2. Calculate Bond Numbers

**Python:**
```python
from prepost.model_analysis import ModelAnalysis

model = ModelAnalysis(nprobes=5)
model.define_params(diameter=1e-3, rho_p=2600, bed_mass=0.5)
results = model.model_summary()
```

**Julia:**
```julia
using FluidisedBedAnalysis

flbed = FluidisedBed(...)
sim_params(flbed, p_diameter=1e-3, rho_p=2600.0, cg_factor=1.0,
           poisson_ratio=0.25, youngs_modulus=70e9, ced=0.5)
results = model_summary(flbed)
```

More examples: [EXAMPLES.md](docs/EXAMPLES.md)

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `FileNotFoundError` | Check paths exist and simulation completed |
| `LoadError: VTK not found` | Verify VTK files present with correct naming |
| `ValueError: Invalid parameter` | Use allowed values (e.g., `"cdf_median"` for y_agg) |
| `AttributeError: parameters not set` | Call `define_params()` before model calculations |
| `Division by zero` | Check n_atoms > 0 in collision CSV |

See [LIBRARY_DOCUMENTATION.md - Troubleshooting](docs/LIBRARY_DOCUMENTATION.md#troubleshooting) for more.

---

## Directory Structure

```
prepost/
├── LIBRARY_DOCUMENTATION.md    ← Main reference
├── QUICK_START.md              ← Getting started
├── API_REFERENCE.md            ← API lookup
├── EXAMPLES.md                 ← Code examples
├── README.md                   ← This file
│
├── plotting.py                 # Python plotting module
├── model_analysis.py           # Python model analysis module
├── xtra_utils.py               # Python utilities
│
├── src/
│   ├── FluidisedBedAnalysis.jl # Main Julia module
│   ├── CurvePlots.jl           # Julia plotting
│   └── model_analysis.jl       # Julia models
│
├── Project.toml                # Julia project file
├── requirements.txt            # Python dependencies
│
└── examples/                   # Example scripts (reference)
    ├── full_analysis.py
    ├── parametric_study.py
    ├── particle_tracking.py
    ├── basic_plotting.jl
    ├── bond_number_calc.jl
    ├── comparative_analysis.jl
    └── validate_data.py
```

---

## Version History

### v0.1 (Current - Development)

- Initial release
- Python: FlBedPlot, ModelAnalysis, utilities
- Julia: FluidisedBedAnalysis, CurvePlots, BoModels
- Three bond number models implemented

### Future Features

- Real-time visualization
- Clustering and classification algorithms
- Machine learning predictions
- Parallel processing optimization

---

## Contact & Support

**Author:** Abhirup Roy  
**Email:** axr154@bham.ac.uk  
**Status:** Development

For issues or feature requests, please contact the author.

---

## References

1. **Hsu, Huang and Kuo (2018)** - Hsu, W.Y., Huang, A.N., Kuo, H.P., 2018. Analysis of interparticle forces
and particle-wall interactions by powder bed pressure drops at incipient
fluidization. Powder Technology 325, 64–68
2. **Soleimani et al. (2021)** - Soleimani, I., Elahipanah, N., Shabanian, J., Chaouki, J., 2021. In-situ quantification of the magnitude of interparticle forces and its temperature variation in a gas-solid fluidized bed. Chemical Engineering Science 232, 116349
3. **Affleck et al. (2023)** - Affleck, S., Thomas, A., Routh, A., Vriend, N., 2023. Novel protocol for quantifying powder cohesivity through fluidisation tests. Powder Technology 415, 118147.

---

## License

MIT License - See LICENSE file for details

---

## Getting Help

1. **New user?** → Start with [QUICK_START.md](docs/QUICK_START.md)
2. **Need specific function?** → Check [API_REFERENCE.md](docs/API_REFERENCE.md)
3. **Want to see code?** → Browse [EXAMPLES.md](docs/EXAMPLES.md)
4. **Detailed reference?** → Read [LIBRARY_DOCUMENTATION.md](docs/LIBRARY_DOCUMENTATION.md)
5. **Can't find answer?** → Contact axr154@bham.ac.uk

---

**Last Updated:** 1 November 2025  
**Documentation Version:** 1.0

