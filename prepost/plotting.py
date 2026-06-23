#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plotting utils for CFDEM fluidised bed simulations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import pyvista as pv
from warnings import warn
from typing import Optional
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def _parse_slice_dir(
    dir_name: str,
    base_path: str,
    variable: str,
    slice_dirn: str,
    nprobes: int,
    y_agg: Optional[str] = None,
) -> tuple[float, list[float] | float]:
    """
    Parses VTK slice data for a single timestep directory.
    For use in multiprocessing"""
    if slice_dirn == "z":
        val_lst = []
        for i in range(nprobes):
            data = pv.read(f"{base_path}/{dir_name}/{variable}_zNormal{i}.vtk")
            val_lst.append(data.get_array(variable).mean().item())
        return float(dir_name), val_lst

    elif slice_dirn == "y":
        data = pv.read(f"{base_path}/{dir_name}/{variable}_yNormal.vtk")
        arr = data.get_array(variable)

        if y_agg == "cdf_median":
            val = find_cdfmedian(arr)
        elif y_agg == "mean":
            val = arr.mean().item()
        elif y_agg == "median":
            val = np.median(arr).item()
        else:
            val = arr.mean().item()

        return float(dir_name), val

    raise ValueError("Invalid slice direction. Choose 'z' or 'y'")


def find_cdfmedian(arr: np.ndarray) -> float:
    """
    Finds the median of the CDF of the given array.

    Args:
      arr:
        A numpy array of values for which the median of the CDF is to be calculated.

    Returns:
        The median of the CDF of the given array as a float.
    """

    x, counts = np.unique(arr, return_counts=True)
    cusum = np.cumsum(counts)
    cdf = cusum / cusum[-1]

    median_idx = cdf.tolist().index(np.percentile(cdf, 50, method="nearest"))
    return x[median_idx].item()


class FlBedPlot:
    def __init__(
        self,
        pressure_path: str,
        nprobes: int,
        velcfg_path: str,
        dump2csv: bool = True,
        plots_dir: str = "plots/",
    ):
        """
        Initialise the FlBedPlot class for plotting pressure and void fraction data from fluidised bed simulations

        Args:
          pressure_path:
            Path to the pressure data. If using slices, point to cuttingPlane directory.
          nprobes:
            Number of probes in simulation
          velcfg_path:
            Path to the vel_cfg file
          dump2csv:
            Save the probe data to a csv file
          plots_dir:
            Directory to save the plots

        Raises:
          FileNotFoundError: If the pressure_path, velcfg_path or plots_dir does not exist
        """
        # Check if the paths exist
        if not os.path.exists(pressure_path):
            raise FileNotFoundError(f"Pressure data at {pressure_path} does not exist")

        if not os.path.exists(velcfg_path):
            raise FileNotFoundError(f"Velocity config at {velcfg_path} does not exist")
        if not os.path.isdir(plots_dir):
            raise FileNotFoundError(f" Plots directory at {plots_dir} does not exist")

        self.pressure_path = pressure_path
        self.nprobes = nprobes
        self.dump2csv = dump2csv
        self.velcfg_path = velcfg_path
        self.plots_dir = plots_dir
        self.data_cache = dict()

        rcParams.update({"font.size": 20})

    def _read_slices_parallel(
        self,
        base_path: str,
        variable: str,
        slice_dirn: str,
        y_agg: Optional[str] = None,
        nprocs: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generalised function to read slice data in parallel using ProcessPoolExecutor.
        Caches the read data to avoid redundant parses.
        """
        cache_key = f"{base_path}_{variable}_{slice_dirn}_{y_agg}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key].copy()

        times = os.listdir(base_path)

        parse_func = partial(
            _parse_slice_dir,
            base_path=base_path,
            variable=variable,
            slice_dirn=slice_dirn,
            nprobes=self.nprobes,
            y_agg=y_agg,
        )

        with ProcessPoolExecutor(max_workers=nprocs) as executor:
            results = executor.map(parse_func, times)

        data_dict = dict(results)

        if slice_dirn == "z":
            columns = [f"Probe {i}" for i in range(self.nprobes)]
        else:
            col_name = "pressure" if variable == "p" else "void_frac"
            columns = [col_name]

        df = pd.DataFrame.from_dict(
            data_dict, orient="index", columns=columns
        ).sort_index()

        self.data_cache[cache_key] = df
        return df.copy()

    def _probe2df(
        self,
        use_slices: bool | None,
        slice_dirn: str | None,
        y_agg: str | None,
        nprocs: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Helper function to map the probe data to a pandas dataframe, with data indexed by time.
        Dumps the data to a csv file if dump2csv attribute of class is True.

        Args:
          use_slices:
            Whether to use slices or probes. If True, reads the pressure data from the cuttingPlane directory.
          slice_dirn:
            Direction of the slice. "z" for z-normal slices, "y" for y-normal slices.
          y_agg:
            Aggregation method for y-normal slices. "cdf_median" for median of CDF, "mean" for mean, "median" for median.

        Returns:
          A pandas DataFrame with the pressure data indexed by time.

        Raises:
            ValueError: If use_slices is True and slice_dirn is not "z" or "y".
            ValueError: If y_agg is not "cdf_median", "mean" or "median".
        """
        pressure_df = pd.DataFrame()

        if use_slices:
            if y_agg not in [None, "cdf_median", "mean", "median"]:
                raise ValueError(
                    "Invalid aggregation method. Choose 'cdf_median', 'mean' or 'median'"
                )

            pressure_df = self._read_slices_parallel(
                base_path=self.pressure_path,
                variable="p",
                slice_dirn=slice_dirn,
                y_agg=y_agg,
                nprocs=nprocs,
            )

            if slice_dirn == "y" and self.dump2csv:
                pressure_df.to_csv("probe_pressure.csv")

        else:
            # Make df from the probe data
            headers = ["Probe Time"]
            for i in range(self.nprobes):
                headers.append(f"Probe {i}")

            pressure_df = pd.read_csv(
                self.pressure_path,
                sep=r"\s+",
                comment="#",
                names=headers,
                header=None,
            ).set_index("Probe Time")

        # Dump to csv if specified
        if self.dump2csv:
            pressure_df.to_csv("probe_pressure.csv")

        return pressure_df

    def _read_probetxt(self):
        """
        Helper function to read the velocity config file and extract the time and corresponding fluid velocities.
        """
        with open(self.velcfg_path, "r") as f:
            probe_text = f.read().splitlines(False)

            # read plot times and corresponding fluid velocities
            self.t = []
            self.v_z = []

            for line in probe_text:
                line_splt = line.replace("(", "").replace(")", "").split()
                self.t.append(float(line_splt[0]))
                self.v_z.append(float(line_splt[-1]))
            print("Selected times: ", self.t)
            print("Corresponding vel: ", self.v_z)

    def _calc_vel(self, df: pd.DataFrame) -> None:
        """
        Helper function to map the velocity to pressure. Reads the velocity config file and maps the velocity to the time-series data.
        This function assumes that the DataFrame is indexed by time and contains pressure data. It adds a new column "V_z" to the DataFrame
        with the corresponding fluid velocities and a "direction" column indicating whether the velocity is increasing, decreasing or at max.

        Args:
          df:
            A pandas DataFrame with the pressure data indexed by time.
        """
        # Initialise bounds and velocity
        bounds = []
        vel = []

        self._read_probetxt()
        # Find the bounds for velocity
        for i in range(len(self.t) - 1):
            if self.v_z[i] == self.v_z[i + 1]:
                bounds.append([self.t[i], self.t[i + 1]])
                vel.append(self.v_z[i])

                if self.v_z[i] == max(self.v_z):
                    max_vel_t1, max_vel_t2 = self.t[i], self.t[i + 1]

            else:
                pass

        # Lower and upper time bounds for each velocity
        lb = [b[0] for b in bounds]
        ub = [b[1] for b in bounds]
        vz_arr = np.zeros_like(
            df.index.to_numpy(), dtype=float
        )  # Initialize as float array

        # Map the velocity to the pressure data
        for i in range(len(bounds)):
            mask = (df.index.to_numpy() > lb[i]) & (df.index.to_numpy() < ub[i])
            vz_arr[mask] = vel[i]

            if i < len(bounds) - 1:
                gap_mask = (df.index.to_numpy() >= ub[i]) & (
                    df.index.to_numpy() <= lb[i + 1]
                )
                vz_arr[gap_mask] = np.nan
        df["V_z"] = vz_arr

        conditions = [
            df.index < max_vel_t1,
            (df.index >= max_vel_t1) & (df.index <= max_vel_t2),
        ]
        choices = ["up", "max"]
        df["direction"] = np.select(conditions, choices, default="down")

    def plot_pressure(
        self,
        x_var: str,
        png_name: Optional[str] = None,
        use_slices: Optional[bool] = True,
        slice_dirn: Optional[str] = None,
        y_agg: Optional[str] = None,
        dump_probe0: Optional[bool] = True,
        nprocs: Optional[int] = None,
    ):
        """
        Plot the pressure data from simulation

        Args:
          x_var:
            Variable to plot against. "time" for time, "velocity" for velocity.
          png_name:
            (OPTIONAL) Name of the png file to save the plot. If not specified, the filename is selected automatically
          use_slices:
            (OPTIONAL) Whether to use slices or probes. If True, reads the pressure data from the cuttingPlane directory.
            Default is True.
          slice_dirn:
            (OPTIONAL) Direction of the slice. "z" for z-normal slices, "y" for y-normal slices. If use_slices is False, this argument is ignored.
          y_agg:
            (OPTIONAL) Aggregation method for y-normal slices. "cdf_median" for median of CDF, "mean" for mean, "median" for median.
            If not specified, defaults to None.
          dump_probe0:
            (OPTIONAL) Whether to dump the probe0 data to a numpy file for further analysis. Default is True.

        Raises:
          ValueError: If use_slices is True and slice_dirn is not "z" or "y".
          ValueError: If y_agg is not "cdf_median", "mean" or "median".
        """
        if slice_dirn == "y":
            warn(
                "Aggregating pressure data using y-slices can yield inaccurate results. Use with caution"
            )

        plot_suffix = "slices" if use_slices else "probes"
        pressure_df = self._probe2df(
            use_slices=use_slices, slice_dirn=slice_dirn, y_agg=y_agg, nprocs=nprocs
        )

        if x_var == "time":
            pressure_df.plot(
                xlabel="Time (s)",
                figsize=(20, 10),
                fontsize=20,
                ylabel="Pressure (Pa)",
                title=f"Pressure at {plot_suffix}",
            )
            plt.savefig(
                self.plots_dir + png_name + ".png"
            ) if png_name else plt.savefig(self.plots_dir + "probe_pressure.png")

        elif x_var == "velocity":
            plt.figure(figsize=[20, 10])
            self._calc_vel(df=pressure_df)

            vel_plot_df = pressure_df.groupby(["direction", "V_z"]).mean()
            vel_plot_std = pressure_df.groupby(["direction", "V_z"]).std()

            # Sort the data for plotting
            vel_up = (
                vel_plot_df[
                    vel_plot_df.index.get_level_values(level="direction").isin(
                        ["up", "max"]
                    )
                ]
                .reset_index("direction", drop=True)
                .sort_index()
            )

            vel_up_std = (
                vel_plot_std[
                    vel_plot_std.index.get_level_values(level="direction").isin(
                        ["up", "max"]
                    )
                ]
                .reset_index("direction", drop=True)
                .sort_index()
            )

            vel_down = (
                vel_plot_df[
                    vel_plot_df.index.get_level_values(level="direction").isin(
                        ["down", "max"]
                    )
                ]
                .reset_index("direction", drop=True)
                .sort_index()
            )

            vel_down_std = (
                vel_plot_std[
                    vel_plot_std.index.get_level_values(level="direction").isin(
                        ["down", "max"]
                    )
                ]
                .reset_index("direction", drop=True)
                .sort_index()
            )

            if slice_dirn == "z":
                for i in range(self.nprobes):
                    plt.errorbar(
                        vel_up.index,
                        vel_up[pressure_df.columns[i]],
                        yerr=vel_up_std[pressure_df.columns[i]],
                        label=f"Probe {i} (Up)",
                        color=f"C{i}",
                        marker="o",
                    )
                    plt.errorbar(
                        vel_down.index,
                        vel_down[pressure_df.columns[i]],
                        yerr=vel_down_std[pressure_df.columns[i]],
                        label=f"Probe {i} (Down)",
                        color=f"C{i}",
                        marker="o",
                        linestyle="dashed",
                    )
                if dump_probe0:
                    probe0_up_v = vel_up.index.to_numpy()
                    probe0_up_p = vel_up[pressure_df.columns[0]].to_numpy()
                    probe0_down_v = vel_down.index.to_numpy()
                    probe0_up_p_err = vel_up_std[pressure_df.columns[0]].to_numpy()
                    probe0_down_p = vel_down[pressure_df.columns[0]].to_numpy()
                    probe0_down_p_err = vel_down_std[pressure_df.columns[0]].to_numpy()
                    probe0_2d = np.vstack(
                        (
                            probe0_up_v,
                            probe0_up_p,
                            probe0_up_p_err,
                            probe0_down_v,
                            probe0_down_p,
                            probe0_down_p_err,
                        )
                    ).T
                    np.save(
                        os.path.join(self.plots_dir, "probe0_plot_voidfrac.npy"),
                        probe0_2d,
                    )

            else:
                plt.errorbar(
                    vel_up.index,
                    vel_up["pressure"],
                    yerr=vel_up_std["pressure"],
                    label=r"$V_z$ Increasing",
                    color="C0",
                    marker="o",
                )
                plt.errorbar(
                    vel_down.index,
                    vel_down["pressure"],
                    yerr=vel_down_std["pressure"],
                    label=r"$V_z$ Increasing",
                    color="C0",
                    marker="o",
                    linestyle="dashed",
                )
            plt.xlabel("Velocity (m/s)")
            plt.ylabel("Pressure (Pa)")

        plt.legend()
        plt.title(f"Pressure vs Velocity for {plot_suffix}")

        plt.savefig(self.plots_dir + f"{png_name}.png") if png_name else plt.savefig(
            self.plots_dir + f"pressure_vel_plot_{plot_suffix}.png"
        )

    def _read_voidfrac(
        self, post_dir: str, slice_dirn: str, nprocs: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Helper function to read the void fraction data using `pyvista` module.
        Args:
          post_dir:
            Path to the postprocessing directory containing the void fraction data.
          slice_dirn:
            Direction of the slice. "z" for z-normal slices, "y" for y-normal slices.
          nproc:
            Number of processes to use for parallel processing.

        Returns:
          A pandas DataFrame with the void fraction data indexed by time.

        Raises:
          ValueError: If slice_dirn is not "z" or "y".
        """

        if slice_dirn not in ["z", "y"]:
            raise ValueError("Invalid slice direction. Choose 'z' or 'y'")

        y_agg = "cdf_median" if slice_dirn == "y" else None

        return self._read_slices_parallel(
            base_path=post_dir,
            variable="void_frac",
            slice_dirn=slice_dirn,
            y_agg=y_agg,
            nprocs=nprocs,
        )

    def plot_voidfrac(
        self,
        slice_dirn: str,
        x_var: str,
        post_dir: str = "CFD/postProcessing/cuttingPlane/",
        png_name: Optional[str] = None,
        dump_probe0: bool = True,
        nprocs: Optional[int] = None,
    ):
        """
        Plots the void fraction data against time or velocity, depending on the x_var argument.
        If x_var is "time", the plot will show void fraction vs time. If x_var is "velocity",
        the plot will show void fraction vs velocity. The velocity is calculated from the velocity config file.

        Args:
          slice_dirn:
            Direction of the slice. "z" for z-normal slices, "y" for y-normal slices.
          x_var:
            Variable to plot against. "time" for time, "velocity" for velocity.
          post_dir:
            (OPTIONAL) Path to the postprocessing directory containing the void fraction data. Default is "CFD/postProcessing/cuttingPlane/".
          png_name:
            (OPTIONAL) Name of the png file to save the plot. If not specified, the filename is selected automatically
          dump_probe0:
            (OPTIONAL) Whether to dump the probe0 data to a numpy file for further analysis. Default is True.

        Raises:
          ValueError: If slice_dirn is not "z" or "y".
          ValueError: If x_var is not "time" or "velocity".
        """
        plt.figure(figsize=[20, 10])
        voidfrac_df = self._read_voidfrac(
            slice_dirn=slice_dirn, post_dir=post_dir, nprocs=nprocs
        )

        if x_var == "time":
            voidfrac_df.plot(
                xlabel="Time (s)",
                ylabel="Void Fraction (-)",
                title="Void Fraction vs Time",
            )
            plt.savefig(
                self.plots_dir + f"{png_name}.png"
            ) if png_name else plt.savefig(
                self.plots_dir + f"voidfrac_time_plot_{slice_dirn}.png"
            )
            plt.xlabel("Velocity (m/s)")

        elif x_var == "velocity":
            self._calc_vel(df=voidfrac_df)

            vel_plot_df = voidfrac_df.groupby(["direction", "V_z"]).mean()
            vel_plot_std = voidfrac_df.groupby(["direction", "V_z"]).std()

            # Sort the data for plotting
            vel_up = (
                vel_plot_df[
                    vel_plot_df.index.get_level_values(level="direction").isin(
                        ["up", "max"]
                    )
                ]
                .reset_index("direction", drop=True)
                .sort_index()
            )
            # print("Vel up", vel_up)
            vel_up_std = (
                vel_plot_std[
                    vel_plot_std.index.get_level_values(level="direction").isin(
                        ["up", "max"]
                    )
                ]
                .reset_index("direction", drop=True)
                .sort_index()
            )

            vel_down = (
                vel_plot_df[
                    vel_plot_df.index.get_level_values(level="direction").isin(
                        ["down", "max"]
                    )
                ]
                .reset_index("direction", drop=True)
                .sort_index()
            )

            vel_down_std = (
                vel_plot_std[
                    vel_plot_std.index.get_level_values(level="direction").isin(
                        ["down", "max"]
                    )
                ]
                .reset_index("direction", drop=True)
                .sort_index()
            )

            if slice_dirn == "z":
                for i in range(self.nprobes):
                    plt.errorbar(
                        vel_up.index,
                        vel_up[voidfrac_df.columns[i]],
                        yerr=vel_up_std[voidfrac_df.columns[i]],
                        label=f"Probe {i} (Up)",
                        color=f"C{i}",
                        marker="o",
                    )
                    plt.errorbar(
                        vel_down.index,
                        vel_down[voidfrac_df.columns[i]],
                        yerr=vel_down_std[voidfrac_df.columns[i]],
                        label=f"Probe {i} (Down)",
                        color=f"C{i}",
                        marker="o",
                        linestyle="dashed",
                    )
            else:
                plt.errorbar(
                    vel_up.index,
                    vel_up["void_frac"],
                    yerr=vel_up_std["void_frac"],
                    label=r"$V_z$ Increasing",
                    color="C0",
                    marker="o",
                )
                plt.errorbar(
                    vel_down.index,
                    vel_down["void_frac"],
                    yerr=vel_down_std["void_frac"],
                    label=r"$V_z$ Decreasing",
                    color="C0",
                    marker="o",
                    linestyle="dashed",
                )
                if dump_probe0:
                    probe0_up_v = vel_up.index.to_numpy()
                    probe0_up_p = vel_up["void_frac"].to_numpy()
                    probe0_up_p_err = vel_up_std["void_frac"].to_numpy()
                    probe0_down_v = vel_down.index.to_numpy()
                    probe0_down_p = vel_down["void_frac"].to_numpy()
                    probe0_down_p_err = vel_down_std["void_frac"].to_numpy()
                    probe0_2d = np.vstack(
                        (
                            probe0_up_v,
                            probe0_up_p,
                            probe0_up_p_err,
                            probe0_down_v,
                            probe0_down_p,
                            probe0_down_p_err,
                        )
                    ).T
                    np.save(
                        os.path.join(self.plots_dir, "probe0_plot_P.npy"), probe0_2d
                    )

            plt.xlabel("Velocity (m/s)")
            plt.ylabel("Void Fraction (-)")

        plt.legend()
        plt.title("Void Fraction vs Velocity")

        plt.savefig(self.plots_dir + f"{png_name}.png") if png_name else plt.savefig(
            self.plots_dir + f"voidfrac_vel_plot_{slice_dirn}.png"
        )

    def _read_collisions(self, csv_path: str, calltype: str) -> pd.DataFrame:
        """
        Helper function to read the collision data from the DEM simulation

        Args:
          csv_path:
            Path to the csv file containing the collision data.
          calltype:
            Type of data to return. "contactarea" for contact area per atom, "contactn" for contact number per atom.

        Returns:
          A pandas DataFrame with the collision data. If calltype is "contactarea", the DataFrame will contain a_contact_peratom column.
          If calltype is "contactn", the DataFrame will contain a contactn column.
        """
        try:
            df = pd.read_csv(csv_path, sep="\\s+")
        except Exception as e:
            raise Exception(f"Error reading collision data from {csv_path}: {e}")

        if calltype == "contactarea":
            df["a_contact_peratom"] = df.a_contact / df.n_atoms
            return df.drop(columns=["n_atoms"])
        elif calltype == "contactn":
            df["contactn"] = df.n_contact / df.n_atoms * 2
            return df
        else:
            raise ValueError(
                f"Invalid calltype '{calltype}'. Must be 'contactarea' or 'contactn'."
            )

    def plot_contactarea(
        self, csv_path: str = "DEM/post/collisions.csv", png_name: Optional[str] = None
    ) -> None:
        """
        Plot the contact area data from the DEM simulations

        Args:
          csv_path:
            Path to the csv file containing the collision data. Default is 'DEM/post/collisions.csv'.
          png_name:
            (OPTIONAL) Name of the png file to save the plot. If not specified, the filename is selected automatically.
        """
        fig = plt.figure(figsize=[20, 10])
        ax = fig.gca()

        contact_df = self._read_collisions(csv_path=csv_path, calltype="contactarea")
        ax.plot(
            contact_df.time,
            contact_df.a_contact_peratom,
            label="Contact Area per Atom",
            color="C0",
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"Contact Area per Particle ($m^2$)")
        ax.set_title("Contact Area vs Time")
        if png_name:
            plt.savefig(self.plots_dir + f"{png_name}.png")
        else:
            plt.savefig(self.plots_dir + "contactarea_time_plot.png")


if __name__ == "__main__":
    # Example usage - usable as a standalone script for default paths

    pressure_path = "CFD/postProcessing/cuttingPlane/"
    velcfg_path = "velcfg.txt"

    probe_cfdem_slices = FlBedPlot(
        pressure_path=pressure_path, nprobes=5, velcfg_path=velcfg_path, dump2csv=False
    )

    """
    Z-Normal Slices vs Time
    """
    # probe_cfdem_slices.plot_pressure(slice_dirn="z", x_var="time", png_name="pressure_time_plot_z", use_slices=True)
    # probe_cfdem_slices.plot_voidfrac(slice_dirn="z", x_var="time", png_name="voidfrac_time_plot_z")

    """
    Y-Normal Slices vs Velocity
    """
    # probe_cfdem_slices.plot_pressure(slice_dirn="y", x_var="velocity", png_name="pressure_vel_plot_y", use_slices=True, y_agg='median')
    # probe_cfdem_slices.plot_voidfrac(slice_dirn="y", x_var="velocity", png_name="voidfrac_vel_plot_y")

    """
    Y-Normal Slices vs Time
    """
    # probe_cfdem_slices.plot_pressure(slice_dirn="y", x_var="time", png_name="pressure_time_plot_y", use_slices=True, y_agg='median')
    # probe_cfdem_slices.plot_voidfrac(slice_dirn="y", x_var="time", png_name="voidfrac_time_plot_y")

    """
    Z-normal Slices vs Velocity
    """
    # probe_cfdem_slices.plot_pressure(slice_dirn="z", x_var="velocity", png_name="pressure_vel_plot_z", use_slices=True)
    # probe_cfdem_slices.plot_voidfrac(slice_dirn="z", x_var="velocity", png_name="voidfrac_vel_plot_z")

    probe_cfdem_slices.plot_pressure(
        slice_dirn="z",
        x_var="velocity",
        png_name="pressure_vel_plot_z",
        use_slices=True,
    )

    probe_cfdem_slices.plot_voidfrac(
        slice_dirn="y", x_var="velocity", png_name="voidfrac_time_plot_y"
    )

    probe_cfdem_slices.plot_pressure(
        slice_dirn="z", x_var="time", png_name="pressure_time_plot_z", use_slices=True
    )
