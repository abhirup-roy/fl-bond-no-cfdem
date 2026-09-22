# Fluidisation-based Bond Number case

This repository contains a CFDEM fluidised-bed case and the Python tools used to prepare its LIGGGHTS inputs and analyse its output. The analysis estimates Bond number with pressure overshoot, DHR, and hysteresis models.

## What is here

| Path                      | Purpose                                                                                |
| ------------------------- | -------------------------------------------------------------------------------------- |
| `CFD/`                  | OpenFOAM case, including mesh and coupling configuration.                              |
| `DEM/`                  | LIGGGHTS input scripts, meshes, and run output.                                        |
| `templates/`            | Jinja templates for JKR and SJKR LIGGGHTS inputs.                                      |
| `prepost/`              | Python source for templating, plotting, conversion, and Bond-number analysis.          |
| `run_templating.py`     | Writes the configured SJKR inputs to`DEM/`.                                          |
| `4core.bbrun.sh`        | BlueBEAR Slurm job that runs the coupled case. Despite its name, it requests 16 tasks. |
| `plot_fluidn_curves.py` | Produces pressure and void-fraction plots from completed output.                       |
| `find_bondno.py`        | Calculates all three Bond-number estimates.                                            |

## Run the supplied case

The solver job is site-specific: it loads BlueBEAR modules, OpenFOAM 5, and a CFDEM-JKR installation. Run these commands from the repository root only after adapting module paths, CFDEM paths, and Slurm resources to your environment.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r prepost/requirements.txt uncertainties
mkdir -p plots pyoutputs

python run_templating.py
sbatch 4core.bbrun.sh
```

Wait for the solver job to finish, then run the analysis locally or submit the SLURM plotting job:

```bash
python plot_fluidn_curves.py
python find_bondno.py
# or: sbatch bbPlotCurves.sh
```

The default analysis expects:

```text
CFD/postProcessing/cuttingPlane/<time>/
  p_zNormal0.vtk ... p_zNormal4.vtk
  voidfraction_yNormal.vtk
DEM/post/collisions.csv
prepost/velcfg.txt
```

It writes plots to `plots/`, the probe-0 pressure array to `plots/probe0_plot_P.npy`, and the Bond-number summary to `pyoutputs/model_summary.json`.

`cleanCase.sh` removes generated case output. Inspect and adapt it before use; it is intentionally not part of the normal workflow.

## Python analysis

`prepost` is currently used directly from this checkout; its package build configuration does not match its flat source layout. Run scripts from the repository root, or add the repository root to `PYTHONPATH` in your own driver.

Start with [the analysis quick start](prepost/docs/QUICK_START.md), then use the [API reference](prepost/docs/API_REFERENCE.md) for exact call signatures.

## Model context

Framework from Roy et al., "A CFD-DEM validation of fluidisation-based models for the granular bond number", *Powder Technology,* 482 (2027), [doi.org/10.1016/j.powtec.2026.123077.](https://doi.org/10.1016/j.powtec.2026.123077.)

## Acknowledgements

This project was made with the support of:

* Abhirup Roy (University of Birmingham)
* Hanqiao Che (Guangxi University)
* Kit Windows-Yule (University of Birmingham)
* Amalia Thomas (Freeman Technology)
