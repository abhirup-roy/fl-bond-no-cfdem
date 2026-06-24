#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculates bond number for the simulation data using different models
"""

from .plotting import FlBedPlot
import pandas as pd
import numpy as np
from typing import Optional
import uncertainties

__author__ = "Abhirup Roy"
__credits__ = ["Abhirup Roy"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Abhirup Roy"
__email__ = "axr154@bham.ac.uk"
__status__ = "Development"


class ModelAnalysis(FlBedPlot):
    def __init__(
        self,
        pressure_path: str = "CFD/postProcessing/cuttingPlane/",
        nprobes: int = 5,
        velcfg_path: str = "prepost/velcfg.txt",
        dump2csv: bool = False,
        plots_dir: str = "plots/",
    ):
        """
        Initialise object to calculate the bond number using different models

        Args:
          pressure_path:
            Path to the pressure data. If using slices, point to CFD cuttingPlane directory.
          probes:
            Number of probes in simulation. Default is 5
          velcfg_path:
            Path to the vel_cfg file
          dump2csv:
            Whether or not to save the probe data to a csv file
          plots_dir:
            Directory to save the plots
        """

        super().__init__(
            pressure_path=pressure_path,
            nprobes=nprobes,
            velcfg_path=velcfg_path,
            dump2csv=dump2csv,
            plots_dir=plots_dir,
        )

        self._store_data()

    def _access_pressures(self) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Helper function to load the pressure data and divide into aerated and non-aerated regions

        Returns:
            A tuple of pd.Series of pressure data with increasing and decreasing velocity
            for probe 0
        """

        pressure_df: pd.DataFrame = super()._probe2df(
            use_slices=True, slice_dirn="z", y_agg=None
        )

        super()._calc_vel(df=pressure_df)
        vel_plot_df = pressure_df.groupby(["direction", "V_z"]).mean()
        vel_plot_err = pressure_df.groupby(["direction", "V_z"]).sem()

        vel_up = (
            vel_plot_df[
                vel_plot_df.index.get_level_values(level="direction").isin(
                    ["up", "max"]
                )
            ]
            .reset_index("direction", drop=True)
            .sort_index()
        )

        vel_up_err = (
            vel_plot_err[
                vel_plot_err.index.get_level_values(level="direction").isin(
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

        vel_down_err = (
            vel_plot_err[
                vel_plot_err.index.get_level_values(level="direction").isin(
                    ["down", "max"]
                )
            ]
            .reset_index("direction", drop=True)
            .sort_index()
        )

        return (
            vel_up["Probe 0"],
            vel_down["Probe 0"],
            vel_up_err["Probe 0"],
            vel_down_err["Probe 0"],
        )

    def _access_voidfrac(self) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Helper function to load void fraction data and divide into
        increasing and decreasing velocity regions

        Returns:
          A tuple of pd.Series of void fraction data with increasing and decreasing velocity
        """

        voidfrac_df = super()._read_voidfrac(
            slice_dirn="y", post_dir="CFD/postProcessing/cuttingPlane/"
        )

        super()._calc_vel(df=voidfrac_df)

        vel_plot_df = voidfrac_df.groupby(["direction", "V_z"]).mean()
        vel_plot_err = voidfrac_df.groupby(["direction", "V_z"]).sem()

        vel_up = (
            vel_plot_df[
                vel_plot_df.index.get_level_values(level="direction").isin(
                    ["up", "max"]
                )
            ]
            .reset_index("direction", drop=True)
            .sort_index()
        )

        vel_up_err = (
            vel_plot_err[
                vel_plot_err.index.get_level_values(level="direction").isin(
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

        vel_down_err = (
            vel_plot_err[
                vel_plot_err.index.get_level_values(level="direction").isin(
                    ["down", "max"]
                )
            ]
            .reset_index("direction", drop=True)
            .sort_index()
        )

        squeezed_up = vel_up.squeeze()
        squeezed_down = vel_down.squeeze()
        squeezed_up_err = vel_up_err.squeeze()
        squeezed_down_err = vel_down_err.squeeze()

        # Ensure we return Series objects
        if not isinstance(squeezed_up, pd.Series):
            raise TypeError("squeezed_up is not a pd.Series")
        if not isinstance(squeezed_down, pd.Series):
            raise TypeError("squeezed_down is not a pd.Series")

        return squeezed_up, squeezed_down, squeezed_up_err, squeezed_down_err

    def _access_contactn(
        self, contact_csv_path="DEM/post/collisions.csv"
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Helper function to read the contact data and divide into increasing and decreasing
        velocity regions

        Args:
          contact_csv_path: Path to the contact data csv file. Default is 'DEM/post/collisions.csv'

        Returns:
          A tuple of pd.Series of contact number data with increasing and decreasing velocity
        """
        contact_df = super()._read_collisions(contact_csv_path, calltype="contactn")
        contact_df.set_index("time", inplace=True)
        contact_df.index -= contact_df.index.min()

        super()._calc_vel(df=contact_df)

        contact_plot_df = contact_df.groupby(["direction", "V_z"]).mean()
        contact_plot_err = contact_df.groupby(["direction", "V_z"]).sem()

        vel_up = (
            contact_plot_df[
                contact_plot_df.index.get_level_values(level="direction").isin(
                    ["up", "max"]
                )
            ]
            .reset_index("direction", drop=True)
            .sort_index()
        )

        vel_up_err = (
            contact_plot_err[
                contact_plot_err.index.get_level_values(level="direction").isin(
                    ["up", "max"]
                )
            ]
            .reset_index("direction", drop=True)
            .sort_index()
        )

        vel_down = (
            contact_plot_df[
                contact_plot_df.index.get_level_values(level="direction").isin(
                    ["down", "max"]
                )
            ]
            .reset_index("direction", drop=True)
            .sort_index()
        )

        vel_down_err = (
            contact_plot_err[
                contact_plot_err.index.get_level_values(level="direction").isin(
                    ["down", "max"]
                )
            ]
            .reset_index("direction", drop=True)
            .sort_index()
        )

        return (
            vel_up["contactn"],
            vel_down["contactn"],
            vel_up_err["contactn"],
            vel_down_err["contactn"],
        )

    def _store_data(self):
        """
        Store the data in the class. Called in __init__ to prevent repeated calculations
        """
        (
            self.pressure_up,
            self.pressure_down,
            self.pressure_up_err,
            self.pressure_down_err,
        ) = self._access_pressures()
        (
            self.contactn_up,
            self.contactn_down,
            self.contactn_up_err,
            self.contactn_down_err,
        ) = self._access_contactn()
        (
            self.voidfrac_up,
            self.voidfrac_down,
            self.voidfrac_up_err,
            self.voidfrac_down_err,
        ) = self._access_voidfrac()

        self.u_mf = self.pressure_up.idxmax()

    def define_params(
        self,
        diameter: float,
        rho_p: float,
        cg_factor: Optional[float] = None,
    ):
        """
        Define the parameters for the model. Must be called before calculating the Bond number.

        Args:
          diameter:
            Diameter of the particles (in m)
          rho_p:
            Density of the particles (in kg/m^3)
          bed_mass:
            Mass of the bed (in kg)
          cg_factor:
            Coarse-graining factor. If not provided, no coarse-graining is applied.
        """

        if cg_factor:
            self.rho_p = rho_p / cg_factor
            self.diameter = diameter * cg_factor
        else:
            self.rho_p = rho_p
            self.diameter = diameter

    def overshoot_model(self) -> tuple[float, float]:
        """
        Calculate the Bond number overshoot model from Hsu, Huang and Kuo (2018)
        """

        if not hasattr(self, "rho_p"):
            raise AttributeError("Define the parameters first using `define_params`")
        elif not hasattr(self, "diameter"):
            raise AttributeError("Define the parameters first using `define_params`")

        idx_max = self.pressure_up.idxmax()
        p_1 = uncertainties.ufloat(
            self.pressure_up.max(), self.pressure_up_err.loc[idx_max]
        )

        p_ss = uncertainties.ufloat(
            self.pressure_up.iloc[-1], self.pressure_up_err.iloc[-1]
        )
        p_over = p_1 - p_ss

        all_contact = pd.concat([self.contactn_up, self.contactn_down])
        avg_contactn = uncertainties.ufloat(all_contact.mean(), all_contact.sem())

        all_voidfrac = pd.concat([self.voidfrac_up, self.voidfrac_down])
        avg_voidfrac = uncertainties.ufloat(all_voidfrac.mean(), all_voidfrac.sem())

        bond_no = (6 * p_over) / (
            avg_contactn**2 * (1 - avg_voidfrac) * self.diameter * self.rho_p * 9.81
        )
        return bond_no.nominal_value, bond_no.std_dev

    def dhr_model(self) -> tuple[float, float]:
        """ "
        Calculate the Bond number usin DHR model from Soleimani et al. (2021)"
        """
        vf1_val = self.voidfrac_up.loc[self.u_mf]
        vf1_err = self.voidfrac_up_err.loc[self.u_mf]
        voidfrac1 = uncertainties.ufloat(vf1_val, vf1_err)

        vf2_val = self.voidfrac_down.loc[self.u_mf]
        vf2_err = self.voidfrac_down_err.loc[self.u_mf]
        voidfrac2 = uncertainties.ufloat(vf2_val, vf2_err)

        bond_no = (voidfrac2 / voidfrac1) ** 3 * (1 - voidfrac1) / (1 - voidfrac2) - 1

        return bond_no.nominal_value, bond_no.std_dev

    def hyst_model(self) -> tuple[float, float]:
        """
        Calculate the Bond number using hysteresis model from Affleck et al. (2023)
        """
        idx_max = self.pressure_up.idxmax()
        p_1 = uncertainties.ufloat(
            self.pressure_up.max(), self.pressure_up_err.loc[idx_max]
        )

        p_ss = uncertainties.ufloat(
            self.pressure_up.iloc[-1], self.pressure_up_err.iloc[-1]
        )

        p_2 = uncertainties.ufloat(
            self.pressure_down.loc[self.u_mf], self.pressure_down_err.loc[self.u_mf]
        )

        k_up = uncertainties.ufloat(
            self.contactn_up.loc[self.u_mf], self.contactn_up_err.loc[self.u_mf]
        )
        k_down = uncertainties.ufloat(
            self.contactn_down.loc[self.u_mf], self.contactn_down_err.loc[self.u_mf]
        )
        delta_k = abs(k_up - k_down)

        bond_no = (p_1 - p_2) / (p_ss * delta_k)

        return bond_no.nominal_value, bond_no.std_dev

    def model_summary(self) -> dict:
        """
        Return a summary Bond number calculated using different models
        """
        overshoot = self.overshoot_model()
        dhr = self.dhr_model()
        hyst = self.hyst_model()

        return {"Overshoot": overshoot, "DHR": dhr, "Hysteresis": hyst}
