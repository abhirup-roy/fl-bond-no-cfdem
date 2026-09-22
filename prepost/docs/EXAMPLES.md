# Examples

All examples run from the repository root after creating `plots/` and `pyoutputs/` and installing the Python dependencies.

## Render the default SJKR inputs

`run_templating.py` is the case configuration, not a generic command-line tool. Edit its constructor arguments and `ced`, then run:

```bash
python run_templating.py
```

It renders `templates/sjkr/` into `DEM/in.liggghts_init` and `DEM/in.liggghts_run`, and saves `pyoutputs/params.json`.

For a JKR input instead:

```python
from prepost import LIGGGHTSTemplatePopulator

renderer = LIGGGHTSTemplatePopulator(
    write_dir="DEM", template_dir="templates", auto_cg=False,
    radius=183e-6, density=1109, bed_mass=0.0849,
)
renderer.populate_jkr_template(
    autocomp_workofadhesion=True,
    surface_energy=0.057,
    young_mod=5.4e6,
    poisson_ratio=0.25,
    contact_dumpstep=2645,
    dump_params=True,
)
```

## Produce the repository plots

```bash
python plot_fluidn_curves.py
```

The script calls `plot_pressure()` for pressure versus velocity and time, and `plot_voidfrac()` for void fraction versus velocity. Edit `pressure_path`, `velcfg_path`, `nprobes`, or the plot calls at the top of the script for another case.

## Write a Bond-number summary

```bash
python find_bondno.py
cat pyoutputs/model_summary.json
```

The JSON contains a two-element array for each model: calculated value followed by propagated standard error. This compact driver is the preferred starting point for a parameter sweep: vary `diameter`, `rho_p`, and `cg_factor`, then write each `model_summary()` result under a distinct output name.

## Convert particle dumps to VTK

```python
from prepost import liggghts2vtk

liggghts2vtk(
    liggghts_dump_dir="DEM/post",
    vtk_dir="DEM/post/vtk",
    dump_every=10,
    timestep=5e-6,
)
```

The dump directory needs at least two files ending in `.liggghts_run` and the standard LIGGGHTS item header. The function creates `vtk_dir` when needed.

## Calculate mean-squared displacement

```python
from prepost import msq_displ

msd = msq_displ(
    dump_dir="DEM/post",
    timestep=5e-6,
    time_rng=(0.0, 0.1),
    dump=True,
    plot=True,
)
print(msd.describe())
```

The current implementation calculates squared displacement from successive `z` positions. It requires an existing `pyoutputs/` directory when writing results.
