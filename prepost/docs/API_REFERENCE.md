# Python API reference

Import public symbols with:

```python
from prepost import FlBedPlot, LIGGGHTSTemplatePopulator, ModelAnalysis, liggghts2vtk, msq_displ
```

## `FlBedPlot`

```python
FlBedPlot(pressure_path, nprobes, velcfg_path, dump2csv=True, plots_dir="plots/", rho_f=1.28)
```

`pressure_path`, `velcfg_path`, and `plots_dir` must exist. `rho_f` converts OpenFOAM kinematic pressure to Pa.

| Method | Signature | Result |
| --- | --- | --- |
| Pressure plot | `plot_pressure(x_var, png_name=None, use_slices=True, slice_dirn=None, y_agg=None, dump_probe0=True, nprocs=None)` | PNG in `plots_dir`; with a z-slice velocity plot, optionally `probe0_plot_P.npy`. |
| Void-fraction plot | `plot_voidfrac(slice_dirn, x_var, post_dir="CFD/postProcessing/cuttingPlane/", png_name=None, dump_probe0=True, nprocs=None)` | PNG in `plots_dir`; a z-slice velocity plot can write `probe0_plot_voidfrac.npy`. |

`x_var` is `"time"` or `"velocity"`. Slice directions are `"z"` and `"y"`. For pressure y-slices, `y_agg` can be `"cdf_median"`, `"mean"`, or `"median"`. `nprocs` sets the worker count used to parse time directories.

## `ModelAnalysis`

```python
ModelAnalysis(
    pressure_path="CFD/postProcessing/cuttingPlane/",
    nprobes=5,
    velcfg_path="prepost/velcfg.txt",
    dump2csv=False,
    plots_dir="plots/",
    rho_f=1.28,
)
```

It inherits `FlBedPlot` and immediately reads the input data.

| Method | Signature | Return |
| --- | --- | --- |
| Define material values | `define_params(diameter, rho_p, cg_factor=None)` | `None` |
| Pressure-overshoot model | `overshoot_model()` | `(value, standard_error)` |
| DHR model | `dhr_model()` | `(value, standard_error)` |
| Hysteresis model | `hyst_model()` | `(value, standard_error)` |
| All models | `model_summary()` | `{"Overshoot": ..., "DHR": ..., "Hysteresis": ...}` |

Call `define_params()` before any model method. `diameter` is metres and `rho_p` is kg/m³.

## LIGGGHTS template generation

```python
LIGGGHTSTemplatePopulator(write_dir, template_dir, auto_cg, **kwargs)
```

`template_dir` must contain `jkr/` and/or `sjkr/`; `write_dir` must exist. Optional constructor values are `radius`, `density`, `bed_mass`, `contact_dumpstep`, and, when `auto_cg=True`, required `cg_factor`.

| Method | Required arguments | Notes |
| --- | --- | --- |
| `populate_sjkr_template` | `ced`, `dump_params` | Renders SJKR input files. Optional `dump_filename` and `dump_filetype` (`"json"` or `"txt"`). |
| `populate_jkr_template` | `autocomp_workofadhesion`, `dump_params`, `young_mod`, `poisson_ratio`, `contact_dumpstep` | Provide `surface_energy` when automatic adhesion is enabled, otherwise optionally `workofadhesion`. |
| `set_timestep` | `timestep`, `kind` | `kind` is `"init"`, `"run"`, or `"all"`. |
| `set_init_time` | `time` | Converts settling time to initialisation steps. |

Both population methods overwrite `DEM/in.liggghts_init` and `DEM/in.liggghts_run` when `write_dir="DEM"`. With `dump_params=True`, create `pyoutputs/` first.

## Particle-dump utilities

```python
liggghts2vtk(timestep=5e-6, vtk_dir=None, dump_every=None,
             liggghts_dump_dir=None, file_suffix=".liggghts_run")
msq_displ(time_rng=None, dump_dir="DEM/post", dump=True, plot=True,
          timestep=5e-6, direction=None)
```

`liggghts2vtk()` emits VTK point data containing position, radius, time, and velocity components. `msq_displ()` returns a pandas Series indexed by particle ID, and can write `pyoutputs/msd.npy` and `pyoutputs/msd_histogram.png`. Both assume LIGGGHTS dump files have the expected nine-line header and at least two time steps.
