"""Gridded Plume Analysis Tool (GPAT).

Simulate aircraft trajectories, estimate aircraft performance, fuel burn and emissions.

Plot associated aircraft exhaust plumes, subject to Gaussian dispersion and advection.
Aggregate plumes to an Eulerian grid for photochemical and microphysical processing.
"""

import argparse
import os
import random
import pathlib
import shutil
import subprocess
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from turtle import ht
from typing import Literal, Optional

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
import ipywidgets as widgets
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod, Transformer
import json

from pycontrails.core import Flight, GeoVectorDataset, MetDataset, models
from pycontrails.core.models import Model
from pycontrails.models.dry_advection import DryAdvection
from pycontrails.models.emissions import Emissions
from pycontrails.models.ps_model import PSFlight
from pycontrails.physics import constants, geo, thermo, units

### GPAT Model Parameters ###
@dataclass
class SimParams:
    """Default simulation parameters."""
    # Temporal domain
    t_sim: tuple[pd.Timestamp, pd.Timedelta, pd.Timedelta] = field(
        default_factory=lambda: (
            pd.to_datetime("2022-01-20 12:00:00"),
            pd.Timedelta(seconds=20),
            pd.Timedelta(hours=120),
        )
    )  # (start time, time step, run time)

    t_out: tuple[pd.Timestamp, pd.Timedelta, pd.Timedelta] = field(
        default_factory=lambda: (
            pd.to_datetime("2022-01-20 12:00:00"),
            pd.Timedelta(minutes=5),
            pd.Timedelta(hours=4),
        )
    )  # (start time, time step, run time)

    #  spatial domain
    lat_bounds: tuple[float, float] = (0.0, 1.0)  # lat bounds [deg]
    lon_bounds: tuple[float, float] = (0.0, 1.0)  # lon bounds [deg]
    alt_bounds: tuple[float, float] = (12000, 13000)  # alt bounds [m]
    domain_margin_deg: float = 0.0  # extra horizontal buffer [deg] added to auto bounds for plume advection
    hres_sim_c: float = 0.01  # horizontal resolution [deg]
    vres_sim_c: float = 500  # vertical resolution [m]
    hres_sim_f: float = 0.001  # horizontal resolution [deg]
    vres_sim_f: float = 100  # vertical resolution [m]

    run_path: Optional[str] = None  # path to run GPAT from
    data_path: Optional[str] = None  # path to data directory
    job_id: Optional[str] = None  # job ID

@dataclass
class MetParams:
    """Default meteorological parameters."""
    eastward_wind: float | None = 0.0  # m/s
    northward_wind: float | None = 0.0  # m/s
    lagrangian_tendency_of_air_pressure: float | None = 0.0  # Pa/s

@dataclass
class ChemParams:
    """Default chemistry parameters."""

    run_chem: bool = True  # whether to run chemistry model

    species_emi: tuple[str, ...] = ("NO",)

    species_out: tuple[str, ...] = (
    "O3", "NO2", "NO", "NO3", "N2O5", "HNO3",
    "HONO", "HO2", "OH", "H2O2",
    "CO", "CH4", "CH3O2",
    "HO2NO2", "PAN", "SO2"
    )


class GPATStatic(Model):
    """Gridded Plume Analysis Tool (GPAT).

    Simulate aircraft trajectories, estimate aircraft performance, fuel burn and emissions. Then 
    aggregates emissions, bg chemistry and meteorology to an Eulerian grid for photochemical and
    microphysical processing.

    Parameters
    ----------
    sim_params : SimParams
        Simulation parameters.
    met_params : MetParams
        Meteorological parameters.
    chem_params : ChemParams
        Chemistry parameters.
    """

    name = "GPATStatic"
    long_name = "Gridded Plume Analysis Tool for static Eulerian photochemical processing."
    # default_params = (FlParams, PlParams, SimParams)

    
    def __init__(self, 
                 sim_params: SimParams, 
                 met_params: MetParams, 
                 chem_params: ChemParams):
        super().__init__()

        # Build spatial grid from current bounds
        self._build_grid(sim_params)

        # Generate time vectors
        self.times_sim = pd.date_range(
            start=sim_params.t_sim[0],
            end=sim_params.t_sim[0] + sim_params.t_sim[2],
            freq=sim_params.t_sim[1],
        )

        self.times_out = pd.date_range(
            start=sim_params.t_out[0],
            end=sim_params.t_out[0] + sim_params.t_out[2],
            freq=sim_params.t_out[1],
        )

        # Set up paths and job ID
        self.run_path = sim_params.run_path
        self.data_path = sim_params.data_path
        self.job_id = sim_params.job_id

        sim_params.date_created = pd.Timestamp.now()

        # Grab species numbers from the input files for later use in indexing model outputs
        chem_params.species_emi_num = grab_species_num(self.run_path, chem_params.species_emi)
        chem_params.species_out_num = grab_species_num(self.run_path, chem_params.species_out)
        chem_params.species_boxm_num = grab_species_num_boxm(self.run_path)

        # Validate species and time hierarchies
        validate_species_hierarchy(chem_params)
        validate_time_hierarchy(sim_params)

        # Set up input and output directories for the job
        self.inputs_job = self.data_path + "inputs/" + self.job_id + "/"
        self.inputs_glob = self.data_path + "inputs/glob/"
        self.outputs_job = self.data_path + "outputs/" + self.job_id + "/"  

        # If job dirs exist, clear and recreate
        if os.path.exists(self.inputs_job):
            shutil.rmtree(self.inputs_job)

        if os.path.exists(self.outputs_job):
            shutil.rmtree(self.outputs_job)

        os.makedirs(self.inputs_job)
        os.makedirs(self.outputs_job)      

        all_params = {
            "sim_params": sim_params,
            "met_params": met_params,
            "chem_params": chem_params,
        }

        # Set the model parameters
        self.sim_params = sim_params
        self.met_params = met_params
        self.chem_params = chem_params
        self.all_params = all_params

        # Set up the model classes for preprocessing and running GPAT
        self.setup = GPATStaticSetup(self)
        self.run = GPATStaticRun(self)

    def _build_grid(self, sim_params):
        """Build coarse grid vectors and meter axes from sim_params bounds.

        Also builds padded met grid arrays (``met_lons``, ``met_lats``,
        ``met_levels``) that extend one cell beyond the BOXM domain on
        each side so that DryAdvection can interpolate at flight
        waypoints near the domain boundary.
        """
        hres = sim_params.hres_sim_c
        vres = sim_params.vres_sim_c

        self.lats = np.arange(
            sim_params.lat_bounds[0] + hres / 2,
            sim_params.lat_bounds[1],
            hres,
        )
        self.lons = np.arange(
            sim_params.lon_bounds[0] + hres / 2,
            sim_params.lon_bounds[1],
            hres,
        )
        self.alts = np.arange(
            sim_params.alt_bounds[0] + vres / 2,
            sim_params.alt_bounds[1],
            vres,
        )

        # Padded met grid: one extra cell on each side for DryAdvection
        self.met_lats = np.arange(
            sim_params.lat_bounds[0] - hres / 2,
            sim_params.lat_bounds[1] + hres,
            hres,
        )
        self.met_lons = np.arange(
            sim_params.lon_bounds[0] - hres / 2,
            sim_params.lon_bounds[1] + hres,
            hres,
        )
        self.met_alts = np.arange(
            sim_params.alt_bounds[0] - vres / 2,
            sim_params.alt_bounds[1] + vres,
            vres,
        )
        self.met_levels = units.m_to_pl(self.met_alts)

        self.levels = units.m_to_pl(self.alts)

        # Convert 1D lon/lat axes to meter axes
        ref_lon = float(self.lons.min())
        ref_lat = float(self.lats.min())
        self.lons_m, _ = lonlat_to_m(
            self.lons,
            np.full_like(self.lons, ref_lat, dtype=float),
            ref_lon,
            ref_lat,
        )
        _, self.lats_m = lonlat_to_m(
            np.full_like(self.lats, ref_lon, dtype=float),
            self.lats,
            ref_lon,
            ref_lat,
        )

    def preprocess_gpat_static(self):
        """Preprocess inputs for GPAT FORTRAN implementation (BOXM and CONTRAIL in future)."""
        # Generate meteorological data
        self.met = self.setup.gen_met()

        # Generate background chemistry data
        self.bg_chem = self.setup.gen_bg_chem()

        # Generate input NetCDF datasets and output templates for FORTRAN
        self.setup.gen_inputs()
        self.setup.gen_outputs()

    def eval(self):
        """Run the GPAT FORTRAN implementation (BOXM and CONTRAIL in future)."""
        # Run BOXM model
        self.run.run_boxm_static()


class GPATStaticSetup:
    """Setup the GPAT model, ready for FORTRAN computation."""

    def __init__(self, gpat):
        self.gpat = gpat

    # Setup methods
    def gen_met(self) -> MetDataset:
        """Generate meteorology data.

        Uses the padded met grid (``met_lons``, ``met_lats``,
        ``met_levels``) so that DryAdvection can interpolate at flight
        waypoints near the BOXM domain boundary.
        """
        met_params = self.gpat.met_params
        met_lons = self.gpat.met_lons
        met_lats = self.gpat.met_lats
        met_levels = self.gpat.met_levels

        # Step 1: Create with STANDARD names for MetDataset validation
        met_standard = xr.Dataset(
            data_vars={
                "eastward_wind": (
                    ("time", "level", "latitude", "longitude"),
                    np.full((len(self.gpat.times_sim), len(met_levels), len(met_lats), len(met_lons)), met_params.eastward_wind),
                ),
                "northward_wind": (
                    ("time", "level", "latitude", "longitude"),
                    np.full((len(self.gpat.times_sim), len(met_levels), len(met_lats), len(met_lons)), met_params.northward_wind),
                ),
                "lagrangian_tendency_of_air_pressure": (
                    ("time", "level", "latitude", "longitude"),
                    np.full((len(self.gpat.times_sim), len(met_levels), len(met_lats), len(met_lons)), met_params.lagrangian_tendency_of_air_pressure),
                ),
            },
            coords={
                "longitude": met_lons,
                "latitude": met_lats,
                "level": met_levels,
                "time": pd.to_datetime(self.gpat.times_sim.values).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )

        # Step 2: Initialize MetDataset (validates standard names)
        met = MetDataset(met_standard)

        month = self.gpat.times_sim[0].month

        # Step 3: Load and interpolate climatology with standard names
        air_temperature = (
            xr.open_dataarray(self.gpat.inputs_glob + "air_temperature.nc", engine="netcdf4")
            .sel(month=month - 1)
            .interp(
                longitude=met_lons,
                latitude=met_lats,
                level=met_levels,
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
            .broadcast_like(met.data["eastward_wind"])
        )

        h2o_concs = (
            xr.open_dataarray(self.gpat.inputs_glob + "h2o_concs.nc", engine="netcdf4")
            .sel(month=month - 1)
            .interp(
                longitude=met_lons,
                latitude=met_lats,
                level=met_levels,
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
            .broadcast_like(met.data["eastward_wind"])
        )
        N_A = 6.022e23  # Avogadro's number
        
        # Add temp and H2O to met dataset
        met.data["air_temperature"] = air_temperature
        met.data["H2O"] = h2o_concs.transpose("latitude", "longitude", "level", "time")

        # Calculate specific humidity and relative humidity
        rho_d = met["air_pressure"].data / (constants.R_d * met["air_temperature"].data)
        met.data["specific_humidity"] = met.data["H2O"] * constants.M_d / (N_A * rho_d * 1e-6)
        met.data["relative_humidity"] = thermo.rhi(
            met.data["specific_humidity"], met.data["air_temperature"], met.data["air_pressure"]
        )

        # Calculate number density of air (M) to feed into box model calcs
        met.data["M"] = (N_A / constants.M_d) * rho_d * 1e-6  # [molecules / cm^3]
        met.data["M"] = met.data["M"].transpose("latitude", "longitude", "level", "time")

        # Calculate O2 and N2 number concs based on M
        met.data["O2"] = 2.079e-01 * met.data["M"]
        met.data["N2"] = 7.809e-01 * met.data["M"]

        # calculate solar zenith angle
        met.data["sza"] = (
            ("latitude", "longitude", "time"),
            self.calc_sza(
                met["latitude"].data.values, met["longitude"].data.values, met["time"].data.values
            ),
        )

        # # add time relative to simulation start time
        # time_rel_s = (met["time"].values - np.datetime64(self.gpat.times_sim[0])).astype("timedelta64[s]").astype(float)
        # met.data = met.data.assign_coords({
        #     "time_rel_s": (("time"), time_rel_s)  
        # })

        return met

    def gen_bg_chem(self) -> xr.Dataset:
        """Generate background chemistry data."""
        month = self.gpat.times_sim[0].month

        # Load background chemistry dataset
        bg_chem = (
            xr.open_dataset(self.gpat.inputs_glob + "species.nc", engine="netcdf4")
            .sel(month=month - 1)
            .rename({"species": "species_boxm"})
        )

        # List of species indices (1-based) that are initialized in boxm_orig
        initialized_species = [
            4,  # NO2
            8,  # NO
            6,  # O3
            11, # CO
            21, # CH4
            39, # HCHO
            42, # CH3CHO
            73, # CH3COCH3
            23, # C2H6
            30, # C2H4
            25, # C3H8
            32, # C3H6
            59, # C2H2
            28, # NC4H10
            34, # TBUT2ENE
            61, # BENZENE
            64, # TOLUENE
            67, # OXYL
            43, # C5H8
            12, # H2O2
            14, # HNO3
            71, # C2H5CHO
            76, # CH3OH
            101,# MEK
            144,# CH3OOH
            198,# PAN
            202 # MPAN
        ]

        # Zero out all species except those initialized in boxm_orig
        n_species = bg_chem.bg_chem.shape[-1]
        for s in range(1, n_species + 1):
            if s not in initialized_species:
                bg_chem.bg_chem[:, :, :, s - 1] = 0

        bg_chem = bg_chem * 1e09  # convert mixing ratio to ppb

        # downselect and interpolate bg_chem to the simulation grid
        return bg_chem.interp(longitude=self.gpat.lons, latitude=self.gpat.lats, level=self.gpat.levels)

    def ac_perf(self) -> list[Flight]:
        """Calculate aircraft performance using PS Model."""
        met = self.gpat.met
        fl = self.gpat.fl

        ps_model = PSFlight()

        for i, fli in enumerate(fl):

            print(f"Before {i}: {fli.dataframe}")

            # mask flight to simulation domain
            fli = self.mask_flight_to_sim_domain(fli)

            print(f"After {i}: {fli.dataframe}")

            # downselect met data to the flight trajectory
            fli.downselect_met(met)
            fli["air_temperature"] = models.interpolate_met(met, fli, "air_temperature")
            fli["specific_humidity"] = models.interpolate_met(met, fli, "specific_humidity")
            fli["true_airspeed"] = fli.segment_groundspeed()
            
            print(f"flight {i} done")

            # get ac performance data using Poll-Schumann Model
            fl[i] = ps_model.eval(fli)

        return fl

    def emissions(self) -> list[Flight]:
        """Estimate emissions using Pycontrails Emissions Model."""
        # met = self.met
        fl = self.gpat.fl

        emi_model = Emissions()

        for i, _fli in enumerate(fl):
            
            # get emissions data
            fl[i] = emi_model.eval(fl[i])

            # Iterate over the columns in the DataFrame
            for column in fl[i].dataframe.columns:
                # Replace NaN values in the column with the value from the previous row
                fl[i].dataframe[column] = fl[i].dataframe[column].ffill()

            # emission indices
            eis = {
                # primary combustion products
                "CO2": 3.16,
                "H2O": 1.23,
                "SO2": 0.00084,
                # secondary combustion products
                "nvPM": fl[i]["nvpm_ei_m"],
                "NO": 0.95 * fl[i]["nox_ei"],
                "NO2": 0.05 * fl[i]["nox_ei"],
                "CO": fl[i]["co_ei"],
                # hydrocarbon speciation
                "HCHO": 0.12 * fl[i]["hc_ei"],  # formaldehyde
                "CH3CHO": 0.04 * fl[i]["hc_ei"],  # acetaldehyde
                "C2H4": 0.15 * fl[i]["hc_ei"],  # ethylene
                "C3H6": 0.04 * fl[i]["hc_ei"],  # propene
                "C2H2": 0.04 * fl[i]["hc_ei"],  # acetylene
                "BENZENE": 0.02 * fl[i]["hc_ei"],  # benzene
            }

            # calculate emission mass total at each waypoint
            for species, ei in eis.items():
                fl[i][species] = ei * fl[i]["fuel_burn"] # [kg]

        return fl

    def gen_inputs(self):
        """Generate BOXM inputs."""
        # Initialise parameters dataset
        self.init_params()
        # Initialize the box model dataset
        self.init_boxm_ds_nc()

    def gen_outputs(self):
        """Generate BOXM output templates."""
        # Initialise boxm coarse output dataset
        self.init_boxm_out_nc()

    # --- Input dataset initialization methods ---
    def init_params(self):
        """Save the simulation parameters to a JSON file for record-keeping."""
        params_path = pathlib.Path(f"{self.gpat.inputs_job}/params.json")
        with open(params_path, "w") as f:
            json.dump(self.gpat.all_params, f, default=str, indent=4)
        print(f"Saved simulation parameters to {params_path}")

    def init_boxm_ds_nc(self):
        """Initialize the box model dataset (met + background chem on coarse grid)."""
        sim_params = self.gpat.sim_params
        fl_params = self.gpat.fl_params
        chem_params = self.gpat.chem_params

        # --- Merge meteorology and background chemistry fields ---
        # Met lives on the padded grid (met_lons/met_lats/met_levels) for
        # DryAdvection, but BOXM only needs the coarse grid.  Downselect met
        # to the BOXM cell centres before merging with bg_chem.
        met_boxm = self.gpat.met.data.sel(
            longitude=self.gpat.lons,
            latitude=self.gpat.lats,
            level=self.gpat.levels,
        )
        self.gpat.boxm_ds = xr.merge([met_boxm, self.gpat.bg_chem])

        # add altitude coordinate
        self.gpat.boxm_ds = self.gpat.boxm_ds.assign_coords(
            time_rel_s = ("time", (self.gpat.boxm_ds["time"].values - np.datetime64(self.gpat.sim_params.t_sim[0])).astype("timedelta64[s]").astype(int)),
            time_idx = ("time", ((self.gpat.boxm_ds["time"].values - np.datetime64(self.gpat.sim_params.t_sim[0])).astype("timedelta64[s]").astype(int) // int(self.gpat.sim_params.t_sim[1].total_seconds())) + 1),
            species_boxm_num = ("species_boxm", chem_params.species_boxm_num)
        )
        self.gpat.boxm_ds["time"] = self.gpat.boxm_ds["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # --- Drop unneeded diagnostic fields ---
        drop_vars = [
            "specific_humidity",
            "relative_humidity",
            "lagrangian_tendency_of_air_pressure",
            "month",
        ]
        self.gpat.boxm_ds = self.gpat.boxm_ds.drop_vars([v for v in drop_vars if v in self.gpat.boxm_ds.variables])

        # --- Assign useful metadata ---
        self.gpat.boxm_ds = self.gpat.boxm_ds.assign_attrs(
            #domain params
            ts_fl=sim_params.t_fl[1].total_seconds(),
            ts_pl=sim_params.t_pl[1].total_seconds(),
            ts_sim=sim_params.t_sim[1].total_seconds(),
            ts_out=sim_params.t_out[1].total_seconds(),

            hres_sim_c=sim_params.hres_sim_c,
            vres_sim_c=sim_params.vres_sim_c,
            hres_sim_f=sim_params.hres_sim_f,
            vres_sim_f=sim_params.vres_sim_f,

            # chem model params
            photol_params=57,
            photol_coeffs=96,
            therm_coeffs=512,
            flux_species=130,

            run_chem=int(chem_params.run_chem),
            n_ac=fl_params.n_ac,

            description="BOXM coarse-grid meteorology and background chemistry fields",
            note="Emissions and plume segments handled separately via PL_DS.NC and FL_DS.NC",
        )

        # Flatten spatial dimensions for easy Fortran indexing
        # (Fortran expects a 1D cell index)
        self.gpat.boxm_ds_stacked = self.gpat.boxm_ds.stack(
            {"cell": ["level", "longitude", "latitude"]}
        ).reset_index("cell")

        # Change to latitude_c, longitude_c, level_c for consistency with BOXM_C_OUT.NC
        self.gpat.boxm_ds_stacked = self.gpat.boxm_ds_stacked.rename(
            {"level": "level_c", "altitude": "altitude_c", "longitude": "longitude_c", "latitude": "latitude_c"}
        )

        # Add coarse cell centres in metres
        # Convert the latitude/longitude coordinates (in degrees) to meters
        longitude_c_m, latitude_c_m = lonlat_to_m(
            self.gpat.boxm_ds_stacked.longitude_c.values,
            self.gpat.boxm_ds_stacked.latitude_c.values,
            self.gpat.lons.min(),
            self.gpat.lats.min()
        )
        
        self.gpat.boxm_ds_stacked = self.gpat.boxm_ds_stacked.assign_coords(
            longitude_c_m=("cell", longitude_c_m),
            latitude_c_m=("cell", latitude_c_m)
        )

        # Set bg_chem to y_bg_c for consistency with FORTRAN naming
        self.gpat.boxm_ds_stacked = self.gpat.boxm_ds_stacked.rename({"bg_chem": "Y_bg_c"})

        # Transpose so that cell is the fastest-varying dimension
        self.gpat.boxm_ds_stacked = self.gpat.boxm_ds_stacked.transpose("cell", "species_boxm", "time")

        # Add mol_mass_c as coordinate for chemistry calculations.
        # Use explicit mechanism aliases plus formula parsing for standard species.
        species_boxm = self.gpat.boxm_ds_stacked["species_boxm"].values

        atom_mass = {
            "H": 1.00794,
            "C": 12.0107,
            "N": 14.0067,
            "O": 15.9994,
            "S": 32.065,
        }
        molar_mass_alias_g_per_mol = {
            "O1D": atom_mass["O"],
            "PAN": 121.05,        # C2H3NO5
            "SA": 98.079,         # H2SO4 proxy
            "NA": 63.01284,       # HNO3 proxy
            "APINENE": 136.24,    # C10H16
            "BPINENE": 136.24,    # C10H16
            "TOLUENE": 92.14,     # C7H8
            "BENZENE": 78.11,     # C6H6
            "OXYL": 106.16,       # C8H10
        }

        def formula_mass_g_per_mol(species_name: str) -> float:
            matches = re.findall(r"([A-Z][a-z]?)([0-9]*)", species_name)
            if not matches:
                return np.nan

            rebuilt = "".join(elem + count for elem, count in matches)
            if rebuilt != species_name:
                return np.nan

            total = 0.0
            for elem, count in matches:
                if elem not in atom_mass:
                    return np.nan
                n = int(count) if count else 1
                total += atom_mass[elem] * n
            return total

        mol_mass_c = np.full(len(species_boxm), np.nan, dtype=float)
        for i, s in enumerate(species_boxm):
            s_key = (s.decode() if isinstance(s, bytes) else str(s)).strip().upper()
            if s_key in molar_mass_alias_g_per_mol:
                mol_mass_c[i] = molar_mass_alias_g_per_mol[s_key] * 1.0e-3
            else:
                mm = formula_mass_g_per_mol(s_key)
                if np.isfinite(mm):
                    mol_mass_c[i] = mm * 1.0e-3

        self.gpat.boxm_ds_stacked = self.gpat.boxm_ds_stacked.assign_coords(mol_mass_c=("species_boxm", mol_mass_c))

        # Delete any existing NetCDF
        nc_path = pathlib.Path(f"{self.gpat.inputs_job}/boxm_ds.nc")
        if nc_path.exists():
            print("Deleting existing boxm_ds.nc")
            nc_path.unlink()

        # Save to NetCDF file
        self.gpat.boxm_ds_stacked.to_netcdf(nc_path, mode="w")
        print(f"Saved {nc_path}")

    # --- Output dataset initialization methods ---
    def init_boxm_out_nc(self):
        """Initialize the box model coarse output dataset (BOXM_C_OUT.NC)."""
        species_out = np.array(self.gpat.chem_params.species_out, dtype="U10")
        chem_params = self.gpat.chem_params

        self.gpat.boxm_out = xr.Dataset(
            data_vars={
                "Y_bg_c": (
                    ("time", "level_c", "longitude_c", "latitude_c", "species_out"),
                    da.zeros(
                        (
                            len(self.gpat.times_out), 
                            len(self.gpat.levels), 
                            len(self.gpat.lons), 
                            len(self.gpat.lats),
                            len(species_out)
                        ),
                        dtype=float
                    ),
                    {"units": "ppb"},
                ),
                "Y_del_c": (
                    ("time", "level_c", "longitude_c", "latitude_c", "species_out"),
                    da.zeros(
                        (
                            len(self.gpat.times_out), 
                            len(self.gpat.levels), 
                            len(self.gpat.lons), 
                            len(self.gpat.lats),
                            len(species_out)
                        ),
                        dtype=float
                    ),
                    {"units": "ppb"},
                ),
                "active_flag": (
                    ("time", "level_c", "longitude_c", "latitude_c"),
                    da.zeros(
                        (
                            len(self.gpat.times_out), 
                            len(self.gpat.levels), 
                            len(self.gpat.lons), 
                            len(self.gpat.lats)
                        ),
                        dtype=bool
                    )
                ),
            },
            coords={
                "time": pd.to_datetime(self.gpat.times_out.values).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "time_rel_s": ("time", (self.gpat.times_out.values - np.datetime64(self.gpat.sim_params.t_sim[0])).astype("timedelta64[s]").astype(int)),
                "time_idx": ("time", ((self.gpat.times_out.values - np.datetime64(self.gpat.sim_params.t_sim[0])).astype("timedelta64[s]").astype(int) // int(self.gpat.sim_params.t_sim[1].total_seconds())) + 1),
                "level_c": self.gpat.levels,
                "altitude_c": ("level_c", units.pl_to_m(self.gpat.levels)),
                "longitude_c": self.gpat.lons,
                "latitude_c": self.gpat.lats,
                "species_out": species_out,
                "species_out_num": ("species_out", chem_params.species_out_num),
            }
        )

        # Flatten spatial dimensions for easy Fortran indexing
        # (Fortran expects a 1D cell index)
        self.gpat.boxm_out_stacked = self.gpat.boxm_out.stack(
            {"cell": ["level_c", "longitude_c", "latitude_c"]}
        ).reset_index("cell")

        # Transpose so that cell is the fastest-varying dimension
        self.gpat.boxm_out_stacked = self.gpat.boxm_out_stacked.transpose("cell", "species_out", "time")

        # --- Assign useful metadata ---
        self.gpat.boxm_out_stacked = self.gpat.boxm_out_stacked.assign_attrs(
            description="BOXM output coarse-grid chemistry fields",
        )

        # Delete any existing NetCDF
        nc_path = pathlib.Path(f"{self.gpat.outputs_job}/boxm_out.nc")
        if nc_path.exists():
            print("Deleting existing boxm_out.nc")
            nc_path.unlink()

        # Save to NetCDF file
        self.gpat.boxm_out_stacked.to_netcdf(nc_path, mode="w")
        print(f"Saved {nc_path}")

    # Helper functions used in GPAT Setup
    def calc_sza(self, latitudes, longitudes, timesteps):
        """Calculate szas for each cell at all timesteps.

        Parameters
        ----------
        latitudes : np.array
            Array of latitudes.
        longitudes : np.array
            Array of longitudes.
        timesteps : np.array
            Array of timesteps.

        Returns
        -------
        np.array
            Array of szas for each cell at all timesteps.
        """

        sza = np.zeros((len(latitudes), len(longitudes), len(timesteps)))

        for lon, lonval in enumerate(longitudes):
            for lat, latval in enumerate(latitudes):
                theta_rad = geo.orbital_position(timesteps)

                sza[lat, lon, :] = np.arccos(
                    geo.cosine_solar_zenith_angle(lonval, latval, timesteps, theta_rad)
                )
        return sza

class GPATStaticRun:
    """Run the GPAT model."""
   
    def __init__(self, gpat):
        self.gpat = gpat

    # Run methods
    def run_boxm_static(self) -> xr.Dataset:
        """Run BOXM."""
        # Run the box model
        subprocess.call(
            [self.gpat.run_path + "boxm_static", self.gpat.job_id, self.gpat.data_path],
        )


# Helper functions for validation and setup
def lonlat_to_m(lon, lat, global_ref_lon, global_ref_lat):
    transformer = Transformer.from_crs("epsg:4326", f"+proj=tmerc +lat_0={global_ref_lat} +lon_0={global_ref_lon} +k=1 +x_0=0 +y_0=0", always_xy=True)
    lon_m, lat_m = transformer.transform(lon, lat)
    return lon_m, lat_m

def m_to_lonlat(lon_m, lat_m, global_ref_lon, global_ref_lat):
    transformer = Transformer.from_crs("epsg:4326", f"+proj=tmerc +lat_0={global_ref_lat} +lon_0={global_ref_lon} +k=1 +x_0=0 +y_0=0", always_xy=True)
    lon, lat = transformer.transform(lon_m, lat_m, direction="INVERSE")
    return lon, lat

def validate_species_hierarchy(chem_params):
    """Validate that species_emi, species_plume, species_out all belong to species_boxm."""
    species_boxm = set(chem_params.species_boxm_num)
    
    # Check species_emi
    for s in chem_params.species_emi_num:
        if s not in species_boxm:
            raise ValueError(f"species_emi {s} not found in species_boxm")
    
    # Check species_plume
    for s in chem_params.species_pl_num:
        if s not in species_boxm:
            raise ValueError(f"species_plume {s} not found in species_boxm")
    
    # Check species_out
    for s in chem_params.species_out_num:
        if s not in species_boxm:
            raise ValueError(f"species_out {s} not found in species_boxm")

def validate_time_hierarchy(sim_params):
    """Validate that time_fl, time_pl, time_out all belong within time_sim (boxm)."""
    t_sim_start = sim_params.t_sim[0]
    t_sim_end = sim_params.t_sim[0] + sim_params.t_sim[2]
    
    # Check time_fl
    t_fl_start = sim_params.t_fl[0]
    t_fl_end = sim_params.t_fl[0] + sim_params.t_fl[2]
    if t_fl_start < t_sim_start or t_fl_end > t_sim_end:
        raise ValueError(f"time_fl [{t_fl_start}, {t_fl_end}] outside time_sim [{t_sim_start}, {t_sim_end}]")
    
    # Check time_pl
    t_pl_start = sim_params.t_pl[0]
    t_pl_end = sim_params.t_pl[0] + sim_params.t_pl[2]
    if t_pl_start < t_sim_start or t_pl_end > t_sim_end:
        raise ValueError(f"time_pl [{t_pl_start}, {t_pl_end}] outside time_sim [{t_sim_start}, {t_sim_end}]")
    
    # Check time_out
    t_out_start = sim_params.t_out[0]
    t_out_end = sim_params.t_out[0] + sim_params.t_out[2]
    if t_out_start < t_sim_start or t_out_end > t_sim_end:
        raise ValueError(f"time_out [{t_out_start}, {t_out_end}] outside time_sim [{t_sim_start}, {t_sim_end}]")

def grab_species_num(run_path, species_out: np.array) -> np.array:
    """Grab the species numbers for the species of interest in output."""
    # Read species names from the file into a list, skip empty lines
    with open(f"{run_path}species_num.txt") as file:
        species_list = [line.strip() for line in file if line.strip()]

    # Create a dictionary mapping species names to their line numbers
    species_dict = {species: index for index, species in enumerate(species_list)}

    return np.array([species_dict[species] + 1 for species in species_out])

def grab_species_num_boxm(run_path) -> np.array:
    """Grab all species numbers from the species_num.txt file for BOXM."""
    # Read species names from the file into a list, skip empty lines
    with open(f"{run_path}species_num.txt") as file:
        species_list = [line.strip() for line in file if line.strip()]

    # Return all species numbers (1-indexed for FORTRAN)
    return np.arange(1, len(species_list) + 1)

def dict_to_dataclass(cls, dict_obj):
    """Convert a dictionary to a dataclass instance.

    Parameters
    ----------
    cls : dataclass
        Dataclass.
    dict_obj : dict
        Dictionary.

    Returns
    -------
    dataclass
        Dataclass instance.
    """
    return cls(**dict_obj)

def filter_inherited_params(instance, base_class):
    """Filter out inherited parameters from a dataclass instance.

    Parameters
    ----------
    instance : dataclass
        Dataclass instance.
    base_class : dataclass
        Base class.

    Returns
    -------
    dict
        Dictionary containing only the parameters unique to the instance.
    """
    base_fields = {f.name for f in fields(base_class)}
    instance_dict = asdict(instance)
    return {k: v for k, v in instance_dict.items() if k not in base_fields}