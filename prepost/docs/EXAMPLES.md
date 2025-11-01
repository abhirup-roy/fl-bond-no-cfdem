# Example Scripts and Workflows

## Python Examples

### Example 1: Complete Analysis Pipeline

**File: `examples/full_analysis.py`**

```python
#!/usr/bin/env python3
"""
Complete fluidised bed analysis pipeline combining pressure, void fraction,
and bond number calculations.
"""

from prepost.plotting import FlBedPlot
from prepost.model_analysis import ModelAnalysis
import matplotlib.pyplot as plt

def main():
    # Initialize plotting
    plot = FlBedPlot(
        pressure_path="CFD/postProcessing/cuttingPlane/",
        nprobes=5,
        velcfg_path="velcfg.txt",
        plots_dir="plots/"
    )
    
    print("=" * 50)
    print("FLUIDISED BED ANALYSIS PIPELINE")
    print("=" * 50)
    
    # 1. Pressure Analysis
    print("\n1. Generating pressure plots...")
    plot.plot_pressure(
        slice_dirn="z",
        x_var="time",
        png_name="01_pressure_time_z",
        use_slices=True
    )
    plot.plot_pressure(
        slice_dirn="z",
        x_var="velocity",
        png_name="02_pressure_velocity_z",
        use_slices=True
    )
    print("   ✓ Pressure plots saved")
    
    # 2. Void Fraction Analysis
    print("\n2. Generating void fraction plots...")
    plot.plot_voidfrac(
        slice_dirn="z",
        x_var="time",
        png_name="03_voidfrac_time_z"
    )
    plot.plot_voidfrac(
        slice_dirn="z",
        x_var="velocity",
        png_name="04_voidfrac_velocity_z"
    )
    print("   ✓ Void fraction plots saved")
    
    # 3. Contact Area Analysis
    print("\n3. Generating contact area plot...")
    plot.plot_contactarea(
        csv_path="DEM/post/collisions.csv",
        png_name="05_contact_area"
    )
    print("   ✓ Contact area plot saved")
    
    # 4. Bond Number Calculations
    print("\n4. Calculating bond numbers...")
    model = ModelAnalysis(nprobes=5)
    
    # Define particle parameters
    model.define_params(
        diameter=1e-3,      # 1 mm particles
        rho_p=2600,         # Silica density
        bed_mass=0.5
    )
    print("   - Particle diameter: 1.0 mm")
    print("   - Particle density: 2600 kg/m³")
    print("   - Bed mass: 0.5 kg")
    
    # Calculate all models
    results = model.model_summary()
    
    print("\n   Bond Number Results:")
    print("   " + "-" * 40)
    print(f"   Overshoot Model:   {results['Overshoot']:.6f}")
    print(f"   DHR Model:         {results['DHR']:.6f}")
    print(f"   Hysteresis Model:  {results['Hysteresis']:.6f}")
    print("   " + "-" * 40)
    
    # 5. Generate Summary
    print("\n5. Analysis Summary:")
    print(f"   Output directory: {plot.plots_dir}")
    print("   Generated files:")
    print("   - 01_pressure_time_z.png")
    print("   - 02_pressure_velocity_z.png")
    print("   - 03_voidfrac_time_z.png")
    print("   - 04_voidfrac_velocity_z.png")
    print("   - 05_contact_area.png")
    print("   - probe_pressure.csv")
    
    print("\n" + "=" * 50)
    print("Analysis Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
python examples/full_analysis.py
```

---

### Example 2: Parametric Study with Coarse-Graining

**File: `examples/parametric_study.py`**

```python
#!/usr/bin/env python3
"""
Parametric study examining effect of coarse-graining on bond numbers.
"""

from prepost.model_analysis import ModelAnalysis
import pandas as pd

def main():
    # Base parameters
    particle_diameter = 1e-3  # 1 mm
    particle_density = 2600   # Silica
    bed_mass = 0.5
    
    # Coarse-graining factors to test
    cg_factors = [1.0, 2.0, 4.0, 8.0]
    
    results_data = []
    
    print("Parametric Study: Effect of Coarse-Graining")
    print("=" * 60)
    print(f"{'CG Factor':<12} {'Overshoot':<15} {'DHR':<15} {'Hysteresis':<15}")
    print("-" * 60)
    
    for cg in cg_factors:
        model = ModelAnalysis(nprobes=5)
        
        model.define_params(
            diameter=particle_diameter,
            rho_p=particle_density,
            bed_mass=bed_mass,
            cg_factor=cg
        )
        
        results = model.model_summary()
        
        print(f"{cg:<12.1f} {results['Overshoot']:<15.6f} "
              f"{results['DHR']:<15.6f} {results['Hysteresis']:<15.6f}")
        
        results_data.append({
            'CG_Factor': cg,
            'Overshoot': results['Overshoot'],
            'DHR': results['DHR'],
            'Hysteresis': results['Hysteresis']
        })
    
    # Save results
    df = pd.DataFrame(results_data)
    df.to_csv("results/parametric_study.csv", index=False)
    
    print("=" * 60)
    print(f"Results saved to: results/parametric_study.csv")

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
mkdir -p results
python examples/parametric_study.py
```

---

### Example 3: Particle Tracking Analysis

**File: `examples/particle_tracking.py`**

```python
#!/usr/bin/env python3
"""
Analyze particle motion using mean squared displacement.
"""

from prepost.xtra_utils import liggghts2vtk, msq_displ
import numpy as np

def main():
    print("Particle Tracking Analysis")
    print("=" * 50)
    
    # 1. Convert LIGGGHTS dumps to VTK
    print("\n1. Converting LIGGGHTS dumps to VTK...")
    liggghts2vtk(
        timestep=5e-6,
        dump_every=10,
        vtk_dir="DEM/post/vtk/",
        liggghts_dump_dir="DEM/post/"
    )
    print("   ✓ VTK files created")
    
    # 2. Calculate MSD for entire simulation
    print("\n2. Computing mean squared displacement...")
    msd_full = msq_displ(
        dump_dir="DEM/post/",
        dump=True,
        plot=True,
        timestep=5e-6
    )
    print(f"   ✓ MSD computed for {len(msd_full)} particles")
    
    # 3. Statistics
    print("\n3. MSD Statistics:")
    print(f"   Mean MSD:   {msd_full.mean():.6e} m²")
    print(f"   Median MSD: {msd_full.median():.6e} m²")
    print(f"   Max MSD:    {msd_full.max():.6e} m²")
    print(f"   Min MSD:    {msd_full.min():.6e} m²")
    
    # 4. Time-windowed analysis
    print("\n4. Computing MSD for specific time window...")
    msd_window = msq_displ(
        time_rng=(0.1, 0.5),
        dump_dir="DEM/post/",
        dump=False,
        plot=False
    )
    print(f"   ✓ MSD for t∈[0.1, 0.5]s computed")
    
    print("\n" + "=" * 50)
    print("Particle tracking analysis complete!")

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
python examples/particle_tracking.py
```

---

## Julia Examples

### Example 1: Basic Plotting

**File: `examples/basic_plotting.jl`**

```julia
#!/usr/bin/env julia

using FluidisedBedAnalysis

function main()
    # Create FluidisedBed object
    flbed = FluidisedBed(
        presure_path="CFD/postProcessing/cuttingPlane/",
        n_probes=5,
        dump2csv=false,
        velcfg_path="velcfg.txt",
        plots_dir="plots/"
    )
    
    println("=" ^ 50)
    println("FLUIDISED BED PLOTTING")
    println("=" ^ 50)
    
    # Plot pressure vs time
    println("\n1. Plotting pressure vs time...")
    plot_pressure(flbed, x_var="time", slice_dirn='z')
    println("   ✓ Saved")
    
    # Plot pressure vs velocity
    println("\n2. Plotting pressure vs velocity...")
    plot_pressure(flbed, x_var="velocity", slice_dirn='z')
    println("   ✓ Saved")
    
    # Plot void fraction vs time
    println("\n3. Plotting void fraction vs time...")
    plot_voidfrac(flbed, slice_dirn='z', x_var="time")
    println("   ✓ Saved")
    
    # Plot void fraction vs velocity
    println("\n4. Plotting void fraction vs velocity...")
    plot_voidfrac(flbed, slice_dirn='z', x_var="velocity")
    println("   ✓ Saved")
    
    println("\n" * "=" ^ 50)
    println("Plotting complete!")
end

main()
```

**Usage:**
```bash
julia examples/basic_plotting.jl
```

---

### Example 2: Bond Number Calculation

**File: `examples/bond_number_calc.jl`**

```julia
#!/usr/bin/env julia

using FluidisedBedAnalysis

function main()
    # Initialize
    flbed = FluidisedBed(
        presure_path="CFD/postProcessing/cuttingPlane/",
        n_probes=5,
        dump2csv=false,
        velcfg_path="velcfg.txt",
        plots_dir="plots/"
    )
    
    println("=" ^ 50)
    println("BOND NUMBER CALCULATION")
    println("=" ^ 50)
    
    # Set parameters
    println("\nSetting simulation parameters...")
    sim_params(
        flbed,
        p_diameter=1e-3,
        rho_p=2600.0,
        cg_factor=1.0,
        poisson_ratio=0.25,
        youngs_modulus=70e9,
        ced=0.5
    )
    println("✓ Parameters set")
    
    # Calculate models
    println("\nCalculating bond numbers...")
    results = model_summary(flbed)
    
    println("\nResults:")
    println("-" ^ 50)
    for (key, val) in results
        println("$key: $val")
    end
    println("-" ^ 50)
end

main()
```

**Usage:**
```bash
julia examples/bond_number_calc.jl
```

---

### Example 3: Comparative Analysis

**File: `examples/comparative_analysis.jl`**

```julia
#!/usr/bin/env julia

using FluidisedBedAnalysis
using DataFrames
using CSV

function main()
    # Test different particle densities
    densities = [1600, 2600, 3000]  # coal, silica, mineral
    
    results_table = DataFrame(
        density=Float64[],
        overshoot=Float64[],
        dhr=Float64[],
        hyst=Float64[]
    )
    
    println("=" ^ 60)
    println("COMPARATIVE ANALYSIS: Effect of Particle Density")
    println("=" ^ 60)
    
    for ρ in densities
        flbed = FluidisedBed(
            presure_path="CFD/postProcessing/cuttingPlane/",
            n_probes=5,
            dump2csv=false,
            velcfg_path="velcfg.txt",
            plots_dir="plots/"
        )
        
        sim_params(
            flbed,
            p_diameter=1e-3,
            rho_p=Float64(ρ),
            cg_factor=1.0,
            poisson_ratio=0.25,
            youngs_modulus=70e9,
            ced=0.5
        )
        
        results = model_summary(flbed)
        
        push!(results_table, (
            density=ρ,
            overshoot=results["overshoot_model"],
            dhr=results["dhr_model"],
            hyst=results["hyst_model"]
        ))
        
        println("\nρ = $ρ kg/m³")
        println("  Overshoot: $(results["overshoot_model"])")
        println("  DHR:       $(results["dhr_model"])")
        println("  Hysteresis: $(results["hyst_model"])")
    end
    
    # Save results
    CSV.write("results/comparative_density.csv", results_table)
    println("\n" * "=" ^ 60)
    println("Results saved to: results/comparative_density.csv")
end

main()
```

**Usage:**
```bash
mkdir -p results
julia examples/comparative_analysis.jl
```

---

## Interactive Notebooks (Jupyter)

### Python Notebook Template

**File: `notebooks/analysis_template.ipynb`**

```
Cell 1: Setup
%matplotlib inline
from prepost.plotting import FlBedPlot
from prepost.model_analysis import ModelAnalysis
import matplotlib.pyplot as plt

Cell 2: Initialize
plot = FlBedPlot(
    pressure_path="CFD/postProcessing/cuttingPlane/",
    nprobes=5,
    velcfg_path="velcfg.txt"
)

Cell 3: Pressure Plots
plot.plot_pressure(slice_dirn="z", x_var="time")
plt.show()

Cell 4: Bond Numbers
model = ModelAnalysis(nprobes=5)
model.define_params(diameter=1e-3, rho_p=2600, bed_mass=0.5)
results = model.model_summary()
print(results)

Cell 5: Export Results
import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## Testing Scripts

### Validation Script

**File: `examples/validate_data.py`**

```python
#!/usr/bin/env python3
"""
Validate input data integrity and format.
"""

import os
import pandas as pd

def validate_environment():
    """Check required directories and files."""
    required_dirs = [
        "CFD/postProcessing/cuttingPlane/",
        "DEM/post/",
        "plots/"
    ]
    
    required_files = [
        "velcfg.txt",
        "DEM/post/collisions.csv"
    ]
    
    print("Validating environment...")
    
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} (MISSING)")
    
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (MISSING)")

def validate_csv():
    """Check collision CSV format."""
    try:
        df = pd.read_csv("DEM/post/collisions.csv")
        required_cols = ["time", "n_atoms", "a_contact", "n_contact"]
        
        missing = set(required_cols) - set(df.columns)
        if missing:
            print(f"✗ Missing columns: {missing}")
        else:
            print(f"✓ CSV format valid")
            print(f"  Rows: {len(df)}")
            print(f"  Time range: {df['time'].min():.4f} - {df['time'].max():.4f}")
    except Exception as e:
        print(f"✗ CSV validation failed: {e}")

def main():
    print("=" * 50)
    print("DATA VALIDATION")
    print("=" * 50)
    validate_environment()
    validate_csv()

if __name__ == "__main__":
    main()
```

