# Analysis guide

## Data flow

```text
OpenFOAM cutting-plane VTK + velocity schedule ──> pressure / void-fraction curves
LIGGGHTS collisions.csv + velocity schedule ────> contacts per particle
all three signals ──────────────────────────────> Bond-number estimates
```

`FlBedPlot` reads VTK arrays with PyVista, averages each plane at each time, then groups samples by the velocity plateaux defined in `prepost/velcfg.txt`. It labels data before the maximum-velocity plateau as `up`, the plateau as `max`, and later data as `down`.

## Inputs and units

| Input | Used for | Requirement |
| --- | --- | --- |
| `p_zNormal<i>.vtk` | Pressure | `i = 0…nprobes-1`; array name `p`. |
| `voidfraction_yNormal.vtk` | Void fraction | Array name `voidfraction`; the Python workflow uses its CDF median. |
| `DEM/post/collisions.csv` | Contacts | Columns including `time`, `n_atoms`, and `n_contact`. |
| `prepost/velcfg.txt` | Velocity bins | Lines of time and velocity-vector data; the final component is vertical velocity. |

OpenFOAM pressure is treated as kinematic pressure and multiplied by `rho_f` (default `1.28 kg/m³`) to yield Pa. Pass the density used by your case to `FlBedPlot` or `ModelAnalysis` when it differs.

## Bond-number calculations

`ModelAnalysis.define_params(diameter, rho_p, cg_factor=None)` applies the case's coarse-graining convention: when `cg_factor` is supplied, it scales diameter by the factor and particle density inversely. It then reports:

- **Overshoot:** pressure overshoot relative to the final upward-pressure point, normalised by average contacts, void fraction, diameter, density, and gravity.
- **DHR:** expansion contrast at the velocity corresponding to the maximum upward pressure.
- **Hysteresis:** pressure difference at that velocity, normalised by the difference between upward and downward contact counts.

Each Python method returns `(nominal_value, standard_error)`. Standard errors come from the standard error of grouped measurements and are propagated with `uncertainties`.

## Constraints

- The Python model currently expects contacts at `DEM/post/collisions.csv` and void-fraction VTK files below `CFD/postProcessing/cuttingPlane/`, even if a different pressure path is supplied.
- Time directories must be parseable as floats. Remove unrelated entries from the cutting-plane directory before analysis.
- A velocity schedule needs repeated time/velocity entries to delimit each plateau. Samples in the ramps become `NaN` velocity bins and are excluded by pandas grouping.
- Pressure y-slice aggregation is supported but warns because it can be less representative than the z-normal probe planes.

For calls and return values, see [API_REFERENCE.md](API_REFERENCE.md).
