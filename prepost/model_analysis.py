#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculates bond number for the simulation data using different models
"""

import pandas as pd
import numpy as np
from typing import Optional
import uncertainties
from .plotting import FlBedPlot, _calc_fluctuation_err, _calc_fluctuation_mean


class ModelAnalysis(FlBedPlot):
    def __init__(
        self,
        pressure_path: str = "CFD/postProcessing/cuttingPlane/",
        nprobes: int = 5,
        velcfg_path: str = "prepost/velcfg.txt",
        dump2csv: bool = False,
        plots_dir: str = "plots/",
        sample_frac: float = 0.5,
        error_kind: str = "std",
        average_type: str = "mean",
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
          sample_frac:
            Fraction of the sample to use for analysis
        """

        super().__init__(
            pressure_path=pressure_path,
            nprobes=nprobes,
            velcfg_path=velcfg_path,
            dump2csv=dump2csv,
            plots_dir=plots_dir,
            sample_frac=sample_frac,
            error_kind=error_kind,
            average_type=average_type,
        )

        self._store_data()

    def _access_pressures(self) -> tuple[pd.Series]:
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

        num_cols = pressure_df.select_dtypes(include=[np.number]).columns
        grouped_df = pressure_df.groupby(["direction", "V_z"])
        vel_plot_df = grouped_df[num_cols].agg(
            _calc_fluctuation_mean, valid_split=self.valid_split, average_type="mean"
        )
        vel_plot_std = grouped_df[num_cols].agg(
            _calc_fluctuation_err,
            valid_split=self.valid_split,
            error_kind=self.error_kind,
            average_type="mean",
        )

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

        up_max = vel_up.max().max()
        outlier_mask = (vel_down > up_max).any(axis=1)
        vel_down = vel_down[~outlier_mask]
        vel_down_std = vel_down_std.loc[vel_down.index]

        return (
            vel_up["Probe 0"],
            vel_down["Probe 0"],
            vel_up_std["Probe 0"],
            vel_down_std["Probe 0"],
        )

    def _access_voidfrac(self) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Helper function to load void fraction data and divide into
        increasing and decreasing velocity regions

        Returns:
          A tuple of pd.Series of void fraction data with increasing and decreasing velocity,
          followed by their standard deviations.
        """

        voidfrac_df = super()._read_voidfrac(
            slice_dirn="y", post_dir="CFD/postProcessing/cuttingPlane/"
        )

        super()._calc_vel(df=voidfrac_df)

        num_cols = voidfrac_df.select_dtypes(include=[np.number]).columns
        grouped_df = voidfrac_df.groupby(["direction", "V_z"])
        vel_plot_df = grouped_df[num_cols].agg(
            _calc_fluctuation_mean, valid_split=self.valid_split, average_type="mean"
        )
        vel_plot_std = grouped_df[num_cols].agg(
            _calc_fluctuation_err,
            valid_split=self.valid_split,
            error_kind=self.error_kind,
            average_type="mean",
        )

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

        return (
            vel_up["void_frac"],
            vel_down["void_frac"],
            vel_up_std["void_frac"],
            vel_down_std["void_frac"],
        )

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
          followed by their standard deviations.
        """
        contact_df = super()._read_collisions(contact_csv_path, calltype="contactn")
        contact_df.set_index("time", inplace=True)

        super()._calc_vel(df=contact_df)

        numeric_cols = contact_df.select_dtypes(include=[np.number]).columns
        grouped_df = contact_df.groupby(["direction", "V_z"])
        contact_plot_df = grouped_df[numeric_cols].agg(
            _calc_fluctuation_mean, valid_split=self.valid_split, average_type="mean"
        )
        contact_plot_std = grouped_df[numeric_cols].agg(
            _calc_fluctuation_err,
            valid_split=self.valid_split,
            error_kind=self.error_kind,
            average_type="mean",
        )

        vel_up = (
            contact_plot_df[
                contact_plot_df.index.get_level_values(level="direction").isin(
                    ["up", "max"]
                )
            ]
            .reset_index("direction", drop=True)
            .sort_index()
        )
        vel_up_std = (
            contact_plot_std[
                contact_plot_std.index.get_level_values(level="direction").isin(
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
        vel_down_std = (
            contact_plot_std[
                contact_plot_std.index.get_level_values(level="direction").isin(
                    ["down", "max"]
                )
            ]
            .reset_index("direction", drop=True)
            .sort_index()
        )

        return (
            vel_up["contactn"],
            vel_down["contactn"],
            vel_up_std["contactn"],
            vel_down_std["contactn"],
        )

    def _store_data(self):
        """
        Store the data in the class. Called in __init__ to prevent repeated calculations
        """
        (
            self.pressure_up,
            self.pressure_down,
            self.pressure_up_std,
            self.pressure_down_std,
        ) = self._access_pressures()

        (
            self.contactn_up,
            self.contactn_down,
            self.contactn_up_std,
            self.contactn_down_std,
        ) = self._access_contactn()

        (
            self.voidfrac_up,
            self.voidfrac_down,
            self.voidfrac_up_std,
            self.voidfrac_down_std,
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
          cg_factor:
            Coarse-graining factor. If not provided, no coarse-graining is applied.
        """

        cg_factor = float(cg_factor) if cg_factor else 1.0
        self.cg_factor = cg_factor
        self.rho_p = rho_p / cg_factor
        self.diameter = diameter * cg_factor

    def overshoot_model(self) -> tuple[float, float]:
        """
        Calculate the Bond number overshoot model from Hsu, Huang and Kuo (2018)
        Returns: (nominal_value, standard_deviation)
        """
        idx_max = self.pressure_up.idxmax()
        p_1 = uncertainties.ufloat(
            self.pressure_up.max(), self.pressure_up_std.loc[idx_max]
        )
        p_ss = uncertainties.ufloat(
            self.pressure_up.iloc[-1], self.pressure_up_std.iloc[-1]
        )
        p_over = p_1 - p_ss

        concat_contact = pd.concat([self.contactn_up, self.contactn_down])
        concat_void = pd.concat([self.voidfrac_up, self.voidfrac_down])
        avg_contactn = uncertainties.ufloat(
            concat_contact.mean(),
            pd.concat([self.contactn_up_std, self.contactn_down_std]).mean(),
        )
        avg_voidfrac = uncertainties.ufloat(
            concat_void.mean(),
            pd.concat([self.voidfrac_up_std, self.voidfrac_down_std]).mean(),
        )
        bond_no = (6 * p_over) / (
            avg_contactn**2 * (1 - avg_voidfrac) * self.diameter * self.rho_p * 9.81
        )
        return bond_no.nominal_value, bond_no.std_dev

    def dhr_model(self) -> tuple[float, float]:
        """
        Calculate the Bond number using DHR model from Soleimani et al. (2021)
        Returns: (nominal_value, standard_deviation)
        """

        v1_val = self.voidfrac_up.loc[self.u_mf]
        v1_std = self.voidfrac_up_std.loc[self.u_mf]
        voidfrac1 = uncertainties.ufloat(v1_val, v1_std)

        v2_val = self.voidfrac_down.loc[self.u_mf]
        v2_std = self.voidfrac_down_std.loc[self.u_mf]
        voidfrac2 = uncertainties.ufloat(v2_val, v2_std)

        bond_no = (voidfrac2 / voidfrac1) ** 3 * (1 - voidfrac1) / (1 - voidfrac2) - 1

        return bond_no.nominal_value, bond_no.std_dev

    def hyst_model(self) -> tuple[float, float]:
        """
        Calculate the Bond number using hysteresis model from Affleck et al. (2023)
        Returns: (nominal_value, standard_deviation)
        """

        idx_max = self.pressure_up.idxmax()
        p_1 = uncertainties.ufloat(
            self.pressure_up.max(), self.pressure_up_std.loc[idx_max]
        )

        p_ss = uncertainties.ufloat(
            self.pressure_up.iloc[-1], self.pressure_up_std.iloc[-1]
        )

        p_2 = uncertainties.ufloat(
            self.pressure_down.loc[self.u_mf], self.pressure_down_std.loc[self.u_mf]
        )

        k_up = uncertainties.ufloat(
            self.contactn_up.loc[self.u_mf], self.contactn_up_std.loc[self.u_mf]
        )
        k_down = uncertainties.ufloat(
            self.contactn_down.loc[self.u_mf], self.contactn_down_std.loc[self.u_mf]
        )
        delta_k = abs(k_up - k_down)

        try:
            bond_no = (p_1 - p_2) / (p_ss * delta_k)
        except ZeroDivisionError:
            raise ZeroDivisionError(
                "Cannot divide by zero in hysteresis model calculation."
                f"p_ss: {p_ss}, delta_k: {delta_k}"
            )

        return bond_no.nominal_value, bond_no.std_dev

    def model_summary(self) -> dict:
        """
        Return a summary Bond number calculated using different models
        """
        overshoot = self.overshoot_model()
        dhr = self.dhr_model()
        hyst = self.hyst_model()

        return {"Overshoot": overshoot, "DHR": dhr, "Hysteresis": hyst}
