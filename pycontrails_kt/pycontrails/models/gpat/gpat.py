"""Gridded Plume Analysis Tool (GPAT).

Simulate aircraft trajectories, estimate aircraft performance, fuel burn and emissions.

Plot associated aircraft exhaust plumes, subject to Gaussian dispersion and advection.
Aggregate plumes to an Eulerian grid for photochemical and microphysical processing.
"""

import argparse
import os
import pathlib
import pickle
import random
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Literal, Optional

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod
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
    # flight time
    t_fl: tuple[pd.Timestamp, pd.Timedelta, pd.Timedelta] = field(
        default_factory=lambda: (
            pd.to_datetime("2022-01-20 13:00:00"),
            pd.Timedelta(minutes=2),
            pd.Timedelta(hours=1),
        )
    )  # (start time, time step, run time)

    # plume time
    t_pl: tuple[pd.Timestamp, pd.Timedelta, pd.Timedelta] = field(
        default_factory=lambda: (
            pd.to_datetime("2022-01-20 13:00:00"),
            pd.Timedelta(minutes=2),
            pd.Timedelta(hours=2),
        )
    )  # (start time, time step, max age)

    # simulation time
    t_sim: tuple[pd.Timestamp, pd.Timedelta, pd.Timedelta] = field(
        default_factory=lambda: (
            pd.to_datetime("2022-01-20 12:00:00"),
            pd.Timedelta(seconds=20),
            pd.Timedelta(hours=120),
        )
    )  # (start time, time step, run time)

    #  spatial domain
    lat_bounds: tuple[float, float] = (0.0, 1.0)  # lat bounds [deg]
    lon_bounds: tuple[float, float] = (0.0, 1.0)  # lon bounds [deg]
    alt_bounds: tuple[float, float] = (12000, 13000)  # alt bounds [m]
    hres_sim_c: float = 0.01  # horizontal resolution [deg]
    vres_sim_c: float = 500  # vertical resolution [m]
    hres_sim_f: float = 0.001  # horizontal resolution [deg]
    vres_sim_f: float = 100  # vertical resolution [m]

    run_path: Optional[str] = None  # path to run GPAT from
    data_path: Optional[str] = None  # path to data directory
    job_id: Optional[str] = None  # job ID
    
    # run_chem: bool = True  # whether to run chemistry model
    # wind_effects: bool = True  # whether to include wind effects in plume dispersion

@dataclass
class FlParams:
    """Default flight/fleet parameters."""
    mode: Literal["direct", "synthetic"] = "direct"
    file: Optional[str] = None

    ac_type: Optional[str] = "A320"  # aircraft type
    fl0_speed: Optional[float] = 100.0  # m/s
    fl0_heading: Optional[float] = 0.0  # deg
    fl0_coords0: Optional[tuple[float, float, float]] = (0.1, 0.125, 12500)  # lat, lon, alt [deg, deg, m]
    sep_dist: Optional[tuple[float, float, float]] = (5000, 2000, 0)  # dx, dy, dz [m]
    n_ac: Optional[int] = 1  # number of aircraft

@dataclass
class PlumeParams:
    """Default plume dispersion parameters."""
    depth: float = 50.0  # initial plume depth, [m]
    width: float = 50.0  # initial plume width, [m]
    verbose_outputs: bool = False  # print verbose outputs
    n_slices: int = 5  # number of slices in the plume
    shear: float = 0.01  # shear [m/s]

@dataclass
class MetParams:
    """Default meteorological parameters."""
    eastward_wind: Optional[float] = 0.0  # m/s
    northward_wind: Optional[float] = 0.0  # m/s
    lagrangian_tendency_of_air_pressure: Optional[float] = 0.0  # Pa/s

@dataclass
class ChemParams:
    """Default chemistry parameters."""
    run_chem: bool = True  # whether to run chemistry model

    species_emi: tuple = ("NO",)
    
    species_plume: tuple = (
    # Core NOx–O3 photochemistry memory
    "NO", "NO2", "O3", "NO3", "N2O5",
    "HNO3", "HONO", "HO2NO2",  # HO2NO2 = pernitric acid

    # Key NOx reservoir / transport species (strongly affects UT NOx lifetime)
    "PAN", "CH3O2NO2",         # peroxyacetyl nitrate family / proxy

    # HOx reservoirs (carry HOx memory better than OH/HO2)
    "H2O2", "CH3OOH",          # methyl hydroperoxide if present

    # Reactivity / oxidation capacity tracers
    "CO", "CH4", "HCHO",       # HCHO is a great “VOC oxidation state” marker

    # Sulfur / aerosol coupling (keep if aerosols matter)
    "SO2", "SA"                # SA = sulfuric acid proxy in your mechanism list
    )
    
    species_out: tuple = (
    "O3", "NO2", "NO", "NO3", "N2O5", "HNO3",
    "HONO", "HO2", "OH", "H2O2",
    "CO", "CH4", "CH3O2",
    "HO2NO2", "PAN", "SO2"
    )


# @dataclass
# class ContrailParams:


class GPAT(Model):
    """Gridded Plume Analysis Tool (GPAT).

    Simulate aircraft trajectories, estimate aircraft performance, fuel burn and emissions. Then
    aggregates emissions, bg chemistry and meteorology to an Eulerian grid for photochemical and
    microphysical processing.

    Parameters
    ----------
    sim_params : SimParams
        Simulation parameters.
    fl_params : FlParams
        Flight parameters.
    plume_params : PlumeParams
        Plume dispersion parameters.
    met_params : MetParams
        Meteorological parameters.
    chem_params : ChemParams
        Chemistry parameters.
    contrail_params : ContrailParams
        Contrail parameters.
    """

    name = "GPAT"
    long_name = "Gridded Plume Analysis Tool"
    # default_params = (FlParams, PlumeParams, SimParams)

    
    def __init__(self, sim_params: SimParams, 
                 fl_params: FlParams,
                 plume_params: PlumeParams, 
                 met_params: MetParams, 
                 chem_params: ChemParams):
        super().__init__()

        # Generate the coarse grid vectors
        self.lats = np.arange(
            sim_params.lat_bounds[0] + sim_params.hres_sim_c / 2,
            sim_params.lat_bounds[1],
            sim_params.hres_sim_c,
        )
        self.lons = np.arange(
            sim_params.lon_bounds[0] + sim_params.hres_sim_c / 2,
            sim_params.lon_bounds[1],
            sim_params.hres_sim_c,
        )
        self.alts = np.arange(
            sim_params.alt_bounds[0] + sim_params.vres_sim_c / 2,
            sim_params.alt_bounds[1],
            sim_params.vres_sim_c,
        )

        self.levels = units.m_to_pl(self.alts)

        # Generate time vectors
        self.times_fl = pd.date_range(
            start=sim_params.t_fl[0],
            end=sim_params.t_fl[0] + sim_params.t_fl[2],
            freq=sim_params.t_fl[1],
        )

        self.times_pl = pd.date_range(
            start=sim_params.t_pl[0],
            end=sim_params.t_pl[0] + sim_params.t_pl[2],
            freq=sim_params.t_pl[1],
        )

        self.times_sim = pd.date_range(
            start=sim_params.t_sim[0],
            end=sim_params.t_sim[0] + sim_params.t_sim[2],
            freq=sim_params.t_sim[1],
        )
        
        if sim_params.run_path is None:
            self.run_path = os.environ["PYCONTRAILSDIR"] + "models/gpat/"

        else:
            self.run_path = sim_params.run_path

        if sim_params.data_path is None:
            self.data_path = "/projects/Impact_of_aviation_on_climate/Kieran2024/"

        else:
            self.data_path = sim_params.data_path

        if sim_params.job_id is None:
            try:
                self.job_id = os.environ["SLURM_JOB_ID"]
            except KeyError:
                # If SLURM_JOB_ID is not found, generate a random number as job ID
                self.job_id = str(random.randint(100000, 999999))

        else:
            self.job_id = sim_params.job_id

        sim_params.date_created = pd.Timestamp.now()

        chem_params.species_emi_num = grab_species_num(self.run_path, chem_params.species_emi)
        chem_params.species_plume_num = grab_species_num(self.run_path, chem_params.species_plume)
        chem_params.species_out_num = grab_species_num(self.run_path, chem_params.species_out)

        self.inputs_job = self.data_path + "inputs/" + self.job_id + "/"
        self.inputs_glob = self.data_path + "inputs/glob/"
        self.outputs_job = self.data_path + "outputs/" + self.job_id + "/"  

        if os.path.exists(self.inputs_job):
            shutil.rmtree(self.inputs_job)

        if os.path.exists(self.outputs_job):
            shutil.rmtree(self.outputs_job)

        os.makedirs(self.inputs_job)
        os.makedirs(self.outputs_job)      

        all_params = {
            "sim_params": sim_params,
            "fl_params": fl_params,
            "plume_params": plume_params,
            "met_params": met_params,
            "chem_params": chem_params,
        }

        # Set the model parameters
        self.sim_params = sim_params
        self.fl_params = fl_params
        self.plume_params = plume_params
        self.met_params = met_params
        self.chem_params = chem_params
        self.all_params = all_params

        self.setup = GPATSetup(self)
        self.run = GPATRun(self)
        self.analysis = GPATAnalysis(self)

    def preprocess_gpat(self):
        """Preprocess inputs for GPAT FORTRAN implementation (BOXM and CONTRAIL in future)."""
        # Generate flight trajectory points
        self.fl = self.setup.traj_gen()

        # Generate meteorological data
        self.met = self.setup.gen_met()

        # Generate background chemistry data
        self.bg_chem = self.setup.gen_bg_chem()

        # Calculate aircraft performance using PS Model
        self.fl = self.setup.ac_perf()

        # Estimate emissions using Pycontrails Emissions Model
        self.fl = self.setup.emissions()

        # Simulate plume dispersion/advection using Pycontrails Dry Advection Model
        self.fl, self.pl = self.setup.sim_plumes()

        # Generate input NetCDF datasets and output templates for FORTRAN
        self.setup.gen_inputs()
        self.setup.gen_outputs()

    def eval(self):
        """Run the GPAT FORTRAN implementation (BOXM and CONTRAIL in future)."""
        # Run BOXM model
        self.run.run_boxm()
        # self.run.run_contrails

    def postprocess_gpat(self):
        """Postprocess outputs from GPAT FORTRAN implementation."""
        # Load outputs
        self.analysis.load_output_datasets()
        # self.analysis.load_patch_table()
        # self.analysis.load_plume_out()
        # self.analysis.plume_to_grid()
        
class GPATSetup:
    """Setup the GPAT model, ready for FORTRAN computation."""

    def __init__(self, gpat):
        self.gpat = gpat

    # Setup methods
    def traj_gen(self) -> list[Flight]:
        """Generate flight trajectory points."""
        fl_params = self.gpat.fl_params
        sim_params = self.gpat.sim_params

        if fl_params.mode == "direct":
            # provide flight file - csv to convert to Flight object (Pd df)
            if fl_params.file is None:
                raise ValueError("Flight file must be provided for direct mode.")
            fl = Flight.read_csv(fl_params.file)
            fl.attrs = {"flight_id": 0, "aircraft_type": fl_params.ac_type}
            mask = (
                (fl["latitude"] > sim_params.lat_bounds[0] + 0.01)
                & (fl["latitude"] < sim_params.lat_bounds[1] - 0.01)
                & (fl["longitude"] > sim_params.lon_bounds[0] + 0.01)
                & (fl["longitude"] < sim_params.lon_bounds[1] - 0.01)
                & (fl["altitude"] > sim_params.alt_bounds[0])
                & (fl["altitude"] < sim_params.alt_bounds[1])
            )
            fl = fl.filter(mask)
            fl = [fl]
            return fl

        # generate synthetic formation flight
        if fl_params.mode == "synthetic":
            if fl_params.n_ac < 1:
                raise ValueError("Number of aircraft must be at least 1.")
            if fl_params.ac_type is None:
                raise ValueError("Aircraft type must be provided for synthetic mode.")
            if fl_params.fl0_coords0 is None:
                raise ValueError("Initial coordinates must be provided for synthetic mode.")
            if fl_params.fl0_speed is None:
                raise ValueError("Flight speed must be provided for synthetic mode.")
            if fl_params.fl0_heading is None:
                raise ValueError("Flight heading must be provided for synthetic mode.")
            if fl_params.sep_dist is None:
                raise ValueError("Separation distances must be provided for synthetic mode.")
            
            fl = []

            lat0, lon0, alt0 = fl_params.fl0_coords0
            heading = fl_params.fl0_heading
            dist = fl_params.fl0_speed * sim_params.t_fl[2].total_seconds()

            # calculate the final coordinates
            geod = Geod(ellps="WGS84")
            lon1, lat1, _ = geod.fwd(lon0, lat0, heading, dist)

            # create flight object for leader flight and resample points according to ts_fl
            df = pd.DataFrame()
            df["longitude"] = [lon0, lon1]
            df["latitude"] = [lat0, lat1]
            df["altitude"] = [alt0, alt0]
            df["time"] = [sim_params.t_fl[0], (sim_params.t_fl[0] + sim_params.t_fl[2])]

            ts_fl_min = int(sim_params.t_fl[1].total_seconds() / 60)

            fl0 = Flight(df).resample_and_fill(freq=f"{ts_fl_min}min")
            fl0.attrs = {"flight_id": 0, "aircraft_type": fl_params.ac_type}
            mask = (
                (fl0["latitude"] > sim_params.lat_bounds[0] + 0.01)
                & (fl0["latitude"] < sim_params.lat_bounds[1] - 0.01)
                & (fl0["longitude"] > sim_params.lon_bounds[0] + 0.01)
                & (fl0["longitude"] < sim_params.lon_bounds[1] - 0.01)
                & (fl0["altitude"] > sim_params.alt_bounds[0])
                & (fl0["altitude"] < sim_params.alt_bounds[1])
            )

            fl0 = fl0.filter(mask)
            fl.append(fl0)

            fli = fl0

            if fl_params.n_ac > 1:
                # create follower flight trajectories
                for i in range(1, fl_params.n_ac):
                    fli = fli.copy()

                    # calculate new coords for follower flight
                    lon_dx, lat_dx, _ = geod.fwd(lon0, lat0, heading, fl_params.sep_dist[0])
                    lon_dx_dy, lat_dx_dy, _ = geod.fwd(
                        lon_dx, lat_dx, heading + 90, fl_params.sep_dist[1]
                    )
                    alt_dx_dy = alt0 + fl_params.sep_dist[2]

                    # Calculate the differences in lat, lon, alt
                    dlat = lat_dx_dy - lat0
                    dlon = lon_dx_dy - lon0
                    dalt = alt_dx_dy - alt0

                    # Update the latitude and longitude of each point in the flight path
                    fli["latitude"] += dlat
                    fli["longitude"] += dlon
                    fli["altitude"] += dalt
                    fli.attrs = {"flight_id": int(i), "aircraft_type": fl_params.ac_type}

                    mask = (
                        (fli["latitude"] > sim_params.lat_bounds[0] + 0.01)
                        & (fli["latitude"] < sim_params.lat_bounds[1] - 0.01)
                        & (fli["longitude"] > sim_params.lon_bounds[0] + 0.01)
                        & (fli["longitude"] < sim_params.lon_bounds[1] - 0.01)
                        & (fli["altitude"] > sim_params.alt_bounds[0])
                        & (fli["altitude"] < sim_params.alt_bounds[1])
                    )
                    fli = fli.filter(mask)
                    fl.append(fli)

                    # Update starting coordinates for next flight
                    lon0, lat0, alt0 = lon_dx_dy, lat_dx_dy, alt_dx_dy

            return fl

    def gen_met(self) -> MetDataset:
        """Generate meteorology data."""
        met_params = self.gpat.met_params

        # Step 1: Create with STANDARD names for MetDataset validation
        met_standard = xr.Dataset(
            data_vars={
                "eastward_wind": (
                    ("time", "level", "latitude", "longitude"),
                    np.full((len(self.gpat.times_sim), len(self.gpat.levels), len(self.gpat.lats), len(self.gpat.lons)), met_params.eastward_wind),
                ),
                "northward_wind": (
                    ("time", "level", "latitude", "longitude"),
                    np.full((len(self.gpat.times_sim), len(self.gpat.levels), len(self.gpat.lats), len(self.gpat.lons)), met_params.northward_wind),
                ),
                "lagrangian_tendency_of_air_pressure": (
                    ("time", "level", "latitude", "longitude"),
                    np.full((len(self.gpat.times_sim), len(self.gpat.levels), len(self.gpat.lats), len(self.gpat.lons)), met_params.lagrangian_tendency_of_air_pressure),
                ),
            },
            coords={
                "longitude": self.gpat.lons,
                "latitude": self.gpat.lats,
                "level": self.gpat.levels,
                "time": self.gpat.times_sim,
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
                longitude=self.gpat.lons,
                latitude=self.gpat.lats,
                level=self.gpat.levels,
                method="linear"
            )
            .broadcast_like(met.data["eastward_wind"])
        )

        h2o_concs = (
            xr.open_dataarray(self.gpat.inputs_glob + "h2o_concs.nc", engine="netcdf4")
            .sel(month=month - 1)
            .interp(
                longitude=self.gpat.lons,
                latitude=self.gpat.lats,
                level=self.gpat.levels,
                method="linear"
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

        return met

    def gen_bg_chem(self) -> xr.Dataset:
        """Generate background chemistry data."""
        month = self.gpat.times_sim[0].month

        bg_chem = (
            xr.open_dataset(self.gpat.inputs_glob + "species.nc", engine="netcdf4")
            .sel(month=month - 1)
        )

        for s in [1,2,3,5,7,9,10,13,15,16,17,18,19,20,22,24,26,27,29,31,33,35,36,37,38,40,
                  41,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,60,62,63,65,66,68,69,70,
                  72,74,75,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,
                  98,99,100,102,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,
                  119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,
                  137,138,139,140,141,142,143,145,146,147,148,149,150,151,152,153,154,155,
                  156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,
                  174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,
                  192,193,194,195,196,197,199,200,201,203,204,205,206,207,208,209,210,211,
                  212,213,214,215,216,217,218,219]:
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
            # downselect met data to the flight trajectory
            fli.downselect_met(met)
            fl[i]["air_temperature"] = models.interpolate_met(met, fli, "air_temperature")
            fl[i]["specific_humidity"] = models.interpolate_met(met, fli, "specific_humidity")
            fl[i]["true_airspeed"] = fli.segment_groundspeed()
            print(f"flight {i} done")

            # get ac performance data using Poll-Schumann Model
            fl[i] = ps_model.eval(fl[i])

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
                fl[i].dataframe[column] = fl[i].dataframe[column].fillna(method="ffill")

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

    def sim_plumes(self) -> list[pd.DataFrame]:
        """Simulate plume dispersion/advection using Pycontrails Dry Advection Model."""
        plume_params = self.gpat.plume_params
        sim_params = self.gpat.sim_params
        
        met = self.gpat.met
        fl = self.gpat.fl

        dry_adv = DryAdvection(
            met,
            max_age=sim_params.t_pl[2],
            dt_integration=sim_params.t_pl[1],
            shear=plume_params.shear,
        )

        pl = []

        for i, fli in enumerate(fl):
            pli = dry_adv.eval(fli)
            pl.append(pli)

            # convert both flights and plumes to dataframes
            fl[i] = fl[i].dataframe
            for column in fl[i].columns:
                # Replace NaN values in the column with the value from the previous row
                fl[i][column] = fl[i][column].fillna(method="ffill")

            pl[i] = pl[i].dataframe

            # calc plume heading
            pl[i] = self.calc_heading(pl[i])
            pl[i]["flight_id"] = fl[i]["flight_id"][0]
            fl[i]["waypoint"] = fl[i].index

        # concatenate all flights and plumes into single dfs
        fl_df = pd.concat(fl)
        pl_df = pd.concat(pl)

        # merge the two dataframes
        fl = fl_df
        pl = pd.merge(
            fl_df[
                [
                    "flight_id",
                    "waypoint",
                    "fuel_flow",
                    "fuel_burn",
                    "true_airspeed",
                    "CO2",
                    "H2O",
                    "SO2",
                    "NO",
                    "NO2",
                    "CO",
                    "HCHO",
                    "CH3CHO",
                    "C2H4",
                    "C3H6",
                    "C2H2",
                    "BENZENE",
                    "nvPM",
                ]
            ],
            pl_df[
                [
                    "flight_id",
                    "waypoint",
                    "time",
                    "age",
                    "longitude",
                    "latitude",
                    "level",
                    "width",
                    "depth",
                    "heading",
                    "sigma_yy",
                    "sigma_yz",
                    "sigma_zz",
                ]
            ],
            on=["flight_id", "waypoint"],
        ).sort_values(by=["time", "flight_id", "waypoint"])

        pl["sin_a"] = np.sin(np.radians(pl["heading"]))
        pl["cos_a"] = np.cos(np.radians(pl["heading"]))
        pl["altitude"] = units.pl_to_m(pl["level"])
        pl["time"] = pl["time"] - sim_params.t_pl[1]

        return fl, pl

    def gen_inputs(self):
        """Generate BOXM inputs."""
        # Initialize the box model dataset
        self.init_boxm_ds_nc()
        # Initialise flight dataset
        self.init_fl_ds_nc()
        #Initialise plume dataset
        self.init_pl_ds_nc()
        
    def gen_outputs(self):
        """Generate BOXM output templates."""
        # Initialise boxm coarse output dataset
        self.init_boxm_out_nc()
        # Initialise patch table output dataset
        self.init_patch_table_nc()
        # Initialise plume output dataset
        self.init_pl_out_nc()  

    # Initialize input datasets
    def init_boxm_ds_nc(self):
        """Initialize the box model dataset (met + background chem on coarse grid)."""
        sim_params = self.gpat.sim_params
        chem_params = self.gpat.chem_params

        # --- Merge meteorology and background chemistry fields ---
        self.gpat.boxm_ds = xr.merge([self.gpat.met.data, self.gpat.bg_chem])

        # --- Drop unneeded diagnostic fields ---
        drop_vars = [
            "specific_humidity",
            "relative_humidity",
            "eastward_wind",
            "northward_wind",
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
            hres_sim_c=sim_params.hres_sim_c,
            vres_sim_c=sim_params.vres_sim_c,
            hres_sim_f=sim_params.hres_sim_f,
            vres_sim_f=sim_params.vres_sim_f,
            # species definitions
            species_emi=chem_params.species_emi,
            species_plume=chem_params.species_plume,
            species_out=chem_params.species_out,
            # species nums
            species_emi_num=chem_params.species_emi_num,
            species_plume_num=chem_params.species_plume_num,
            species_out_num=chem_params.species_out_num,

            description="BOXM coarse-grid meteorology and background chemistry fields",
            note="Emissions and plume segments handled separately via PL_DS.NC and FL_DS.NC",
        )

        # Flatten spatial dimensions for easy Fortran indexing
        # (Fortran expects a 1D cell index)
        self.gpat.boxm_ds_stacked = self.gpat.boxm_ds.stack(
            {"cell": ["level", "longitude", "latitude"]}
        ).reset_index("cell")

        # Delete any existing NetCDF
        nc_path = pathlib.Path(f"{self.gpat.inputs_job}/boxm_ds.nc")
        if nc_path.exists():
            print("Deleting existing boxm_ds.nc")
            nc_path.unlink()

        # Save to NetCDF file
        self.gpat.boxm_ds_stacked.to_netcdf(nc_path, mode="w")
        print(f"Saved {nc_path}")

    def init_fl_ds_nc(self):
        """Initialize the flight dataset for BOXM."""
        df = self.gpat.fl.copy()

        # Set multi-index (flight_id, waypoint) and convert to xarray
        fl_ds = df.set_index(["flight_id", "waypoint"]).to_xarray()

        # Drop individual species variables
        fl_ds = fl_ds.drop_vars(["air_temperature", "specific_humidity", 
                                 "nox_ei", "co_ei", "hc_ei", "nvpm_ei_m",
                                 "nvpm_ei_n", "co2", "h2o", "so2", 
                                 "sulphates", "oc", "nox", "co", "hc", 
                                 "nvpm_mass", "nvpm_number", "nvPM"])
        
        # --- Assign useful metadata ---
        fl_ds = fl_ds.assign_attrs(
            description="Flight trajectory and emissions data for BOXM",
        )

        # Save to NetCDF
        nc_path = pathlib.Path(f"{self.gpat.inputs_job}/fl_ds.nc")
        if nc_path.exists():
            nc_path.unlink()
        fl_ds.to_netcdf(nc_path, mode="w")
        print(f"Saved {nc_path}")
    
    def init_pl_ds_nc(self):
        """Initialize the plume dataset for the box model (PL_DS.NC)."""
        df = self.gpat.pl.copy()
        sim_params = self.gpat.sim_params
        chem_params = self.gpat.chem_params

        species_cols = self.gpat.chem_params.species_emi
        all_species_cols = ['CO2', 'H2O', 'SO2', 'NO', 
                            'NO2', 'CO', 'HCHO', 'CH3CHO', 
                            'C2H4', 'C3H6', 'C2H2', 'BENZENE', 'nvPM']
        # Set multi-index (flight_id, waypoint, time)
        pl_ds = df.set_index(["flight_id", "waypoint", "time"]).to_xarray()

        # Stack species into a single 4D DataArray (flight_id, waypoint, time, species)
        species_data = []
        for col in species_cols:
            species_data.append(pl_ds[col])
        
        pl_ds["emi_species_mass"] = xr.concat(
            species_data,
            dim=pd.Index(species_cols, name="species")
        ).transpose("flight_id", "waypoint", "time", "species")  # <-- ensure correct order
        
        # Drop ALL emission species variables (both stacked and unstacked)
        vars_to_drop = [col for col in all_species_cols if col in pl_ds.data_vars]
        if vars_to_drop:
            pl_ds = pl_ds.drop_vars(vars_to_drop)

        # drop remaining unneeded variables
        # pl_ds = pl_ds.drop_vars(["sin_a", "cos_a", "

        # Assign useful metadata
        pl_ds = pl_ds.assign_attrs(
            ts_fl=sim_params.t_fl[1].total_seconds(),
            ts_pl=sim_params.t_pl[1].total_seconds(),
            ts_sim=sim_params.t_sim[1].total_seconds(),

            species_emi=chem_params.species_emi,
            species_plume=chem_params.species_plume,
        )

        # Save to NetCDF
        nc_path = pathlib.Path(f"{self.gpat.inputs_job}/pl_ds.nc")
        if nc_path.exists():
            print("Deleting existing pl_ds.nc")
            nc_path.unlink()
        pl_ds.to_netcdf(nc_path, mode="w")
        print(f"Saved {nc_path}")

    # Initialize output datasets
    def init_boxm_out_nc(self):
        """Initialize the box model coarse output dataset (BOXM_C_OUT.NC)."""
        species_out = np.array(self.gpat.chem_params.species_out, dtype="U10")
        sim_params = self.gpat.sim_params
        chem_params = self.gpat.chem_params

        self.gpat.boxm_out = xr.Dataset(
            data_vars={
                "Y_bg_c": (
                    ("time", "level", "longitude", "latitude", "species_out"),
                    da.zeros(
                        (
                            len(self.gpat.times_sim), 
                            len(self.gpat.levels), 
                            len(self.gpat.lons), 
                            len(self.gpat.lats),
                            len(species_out)
                        ),
                        dtype=float
                    ),
                    {"units": "mol_cm3"},
                ),
                "Y_del_c": (
                    ("time", "level", "longitude", "latitude", "species_out"),
                    da.zeros(
                        (
                            len(self.gpat.times_sim), 
                            len(self.gpat.levels), 
                            len(self.gpat.lons), 
                            len(self.gpat.lats),
                            len(species_out)
                        ),
                        dtype=float
                    ),
                    {"units": "mol_cm3"},
                ),
                "active_flag": (
                    ("time", "level", "longitude", "latitude"),
                    da.zeros(
                        (
                            len(self.gpat.times_sim), 
                            len(self.gpat.levels), 
                            len(self.gpat.lons), 
                            len(self.gpat.lats)
                        ),
                        dtype=bool
                    )
                ),
            },
            coords={
                "time": self.gpat.times_sim,
                "level": self.gpat.levels,
                "longitude": self.gpat.lons,
                "latitude": self.gpat.lats,
                "species_out": species_out,
            }
        )

        # Flatten spatial dimensions for easy Fortran indexing
        # (Fortran expects a 1D cell index)
        self.gpat.boxm_out_stacked = self.gpat.boxm_out.stack(
            {"cell": ["level", "longitude", "latitude"]}
        ).reset_index("cell")

        # --- Assign useful metadata ---
        self.gpat.boxm_out = self.gpat.boxm_out.assign_attrs(
            #domain params
            ts_fl=sim_params.t_fl[1].total_seconds(),
            ts_pl=sim_params.t_pl[1].total_seconds(),
            ts_sim=sim_params.t_sim[1].total_seconds(),
            hres_sim_c=sim_params.hres_sim_c,
            vres_sim_c=sim_params.vres_sim_c,
            hres_sim_f=sim_params.hres_sim_f,
            vres_sim_f=sim_params.vres_sim_f,
            # species definitions
            species_emi=chem_params.species_emi,
            species_plume=chem_params.species_plume,
            species_out=chem_params.species_out,
            # species nums
            species_emi_num=chem_params.species_emi_num,
            species_plume_num=chem_params.species_plume_num,
            species_out_num=chem_params.species_out_num,

            description="BOXM coarse-grid meteorology and background chemistry fields",
            note="Emissions and plume segments handled separately via PL_DS.NC and FL_DS.NC",
        )

        # Delete any existing NetCDF
        nc_path = pathlib.Path(f"{self.gpat.outputs_job}/boxm_out.nc")
        if nc_path.exists():
            print("Deleting existing boxm_out.nc")
            nc_path.unlink()

        # Save to NetCDF file
        self.gpat.boxm_out_stacked.to_netcdf(nc_path, mode="w")
        print(f"Saved {nc_path}")
        
    def init_patch_table_nc(self):
        """Initialize the patch output dataset (PATCH_OUT.NC)."""
        species_plume = np.array(self.gpat.chem_params.species_plume, dtype="U10")
        sim_params = self.gpat.sim_params
        chem_params = self.gpat.chem_params

        self.gpat.patch_table = xr.Dataset(
            data_vars={
                "Y_del_f": (
                    ("row", "species_plume"), 
                    da.zeros(
                        (
                            0, 
                            len(species_plume)
                        ),
                        dtype=float,
                    ),  
                    {"units": "mol_cm3"},
                ),
            },
            coords={
            "row": ("row", np.array([], dtype=int)),   # 0-length row dimension
            "time": ("row", np.array([], dtype="datetime64[ns]")),
            "patch_id": ("row", np.array([], dtype=int)),

            "latitude": ("row", np.array([], dtype=float)),
            "longitude": ("row", np.array([], dtype=float)),
            "level": ("row", np.array([], dtype=int)),

            "latitude_f": ("row", np.array([], dtype=float)),
            "longitude_f": ("row", np.array([], dtype=float)),
            "level_f": ("row", np.array([], dtype=int)),

            "species_plume": ("species_plume", species_plume),
            }
        )

        # --- Assign useful metadata ---
        self.gpat.boxm_ds = self.gpat.boxm_ds.assign_attrs(
            #domain params
            ts_fl=sim_params.t_fl[1].total_seconds(),
            ts_pl=sim_params.t_pl[1].total_seconds(),
            ts_sim=sim_params.t_sim[1].total_seconds(),
            hres_sim_c=sim_params.hres_sim_c,
            vres_sim_c=sim_params.vres_sim_c,
            hres_sim_f=sim_params.hres_sim_f,
            vres_sim_f=sim_params.vres_sim_f,
            # species definitions
            species_emi=chem_params.species_emi,
            species_plume=chem_params.species_plume,
            species_out=chem_params.species_out,
            # species nums
            species_emi_num=chem_params.species_emi_num,
            species_plume_num=chem_params.species_plume_num,
            species_out_num=chem_params.species_out_num,

            description="BOXM coarse-grid meteorology and background chemistry fields",
            note="Emissions and plume segments handled separately via PL_DS.NC and FL_DS.NC",
        )

        # Delete any existing NetCDF
        nc_path = pathlib.Path(f"{self.gpat.outputs_job}/patch_table.nc")
        if nc_path.exists():
            print("Deleting existing patch_table.nc")
            nc_path.unlink()

        # Save to NetCDF file
        self.gpat.patch_table.to_netcdf(nc_path, mode="w")
        print(f"Saved {nc_path}")
        
    def init_pl_out_nc(self):
        """Initialize the plume output dataset (PL_OUT.NC)."""
        # Load the input plume dataset
        pl_ds = xr.open_dataset(f"{self.gpat.inputs_job}/pl_ds.nc")

        sim_params = self.gpat.sim_params
        chem_params = self.gpat.chem_params
        
        # Create output dataset with same structure
        self.gpat.pl_out = pl_ds.copy(deep=True)
        
        # Rename and repurpose the species mass variable
        # Store DELTA mass (change from initial emissions due to chemistry)
        self.gpat.pl_out = self.gpat.pl_out.rename({"emi_species_mass": "delta_species_mass"})
        
        # Zero out delta mass (initially, no chemistry has occurred)
        self.gpat.pl_out["delta_species_mass"].values[:] = 0.0
        
        # If output species differ from input species, create new variable
        if set(self.gpat.chem_params.species_emi) != set(self.gpat.chem_params.species_out):
            self.gpat.pl_out = self.gpat.pl_out.drop_vars("species")
            
            species_plume = np.array(self.gpat.chem_params.species_plume, dtype="U10")
            
            # Create delta_species_mass for output species
            self.gpat.pl_out["delta_species_mass"] = (
                ("flight_id", "waypoint", "time", "species_plume"),
                np.zeros((
                    len(self.gpat.pl_out.flight_id),
                    len(self.gpat.pl_out.waypoint),
                    len(self.gpat.pl_out.time),
                    len(species_plume)
                ), dtype=float),
                {
                    "units": "kg",
                    "long_name": "Change in species mass due to chemistry",
                    "note": "Add to initial emission mass (from fl_ds.nc) to get total mass"
                }
            )
            self.gpat.pl_out["species_plume"] = species_plume
        else:
            # Just update metadata
            self.gpat.pl_out["delta_species_mass"].attrs = {
                "units": "kg",
                "long_name": "Change in species mass due to chemistry",
                "note": "Add to initial emission mass (from fl_ds.nc) to get total mass"
            }

        # Add aspect ratio column
        self.gpat.pl_out["aspect_ratio"] = (
            self.gpat.pl_out["width"] / self.gpat.pl_out["depth"]
        )
        
        # Add metadata
        self.gpat.pl_out.assign_attrs(
            ts_fl=sim_params.t_fl[1].total_seconds(),
            ts_pl=sim_params.t_pl[1].total_seconds(),
            ts_sim=sim_params.t_sim[1].total_seconds(),

            species_emi=chem_params.species_emi,
            species_plume=chem_params.species_plume,
        )
        
        # Save
        nc_path = pathlib.Path(f"{self.gpat.outputs_job}/pl_out.nc")
        if nc_path.exists():
            print("Deleting existing pl_out.nc")
            nc_path.unlink()
        self.gpat.pl_out.to_netcdf(nc_path, mode="w")
        print(f"Saved {nc_path}")

    # Helper functions used in GPAT Setup
    def calc_heading(self, pl_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate heading for each plume.

        Parameters
        ----------
        pl_df : pd.DataFrame
            DataFrame containing plume data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing plume data with heading.
        """
        # Sort the dataframe by time and waypoint
        pl_df = pl_df.sort_values(by=["time", "waypoint"])

        # Group the dataframe by the timestep and apply the function
        pl_df["heading"] = pl_df.groupby("time").apply(self.calculate_heading_g).reset_index(drop=True)

        return pl_df

    def calculate_heading_g(self, group):
        """Calculate heading for each timestep.

        Parameters
        ----------
        group : pd.DataFrame
            DataFrame containing plume data for a single timestep.

        Returns
        -------
        pd.Series
            Series containing heading for each plume in the timestep.
        """

        g = Geod(ellps="WGS84")

        startlat = group["latitude"].values[:-1]
        startlon = group["longitude"].values[:-1]
        endlat = group["latitude"].values[1:]
        endlon = group["longitude"].values[1:]
        az12, az21, dist = g.inv(startlon, startlat, endlon, endlat)

        heading = (90 - az12) % 360

        return pd.Series(
            np.concatenate([[heading[0]], heading]) if len(heading) > 0 else [np.nan], index=group.index
        )

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

class GPATRun:
    """Run the GPAT model."""
   
    def __init__(self, gpat):
        self.gpat = gpat

    # Run methods
    def run_boxm(self) -> xr.Dataset:
        """Run BOXM."""
        # Run the box model
        subprocess.call(
            [self.run_path + "boxm", self.job_id],
        )

    # def run_contrails(self) -> xr.Dataset:
    #     # Run the contrail model
    #     # subprocess.call(
    #     #     [self.run_path + "contrails", self.job_id],
    #     # )
    #     pass

class GPATAnalysis:
    """Analyze GPAT model outputs."""

    def __init__(self, gpat):
        self.gpat = gpat

    # Loading data methods
    def load_output_datasets(self):
        self.boxm_out = xr.open_dataset(f"{self.gpat.inputs_job}/boxm_out.nc")
        self.patch_table = xr.open_dataset(f"{self.gpat.outputs_job}/patch_table.nc")
        self.pl_out = xr.open_dataset(f"{self.gpat.outputs_job}/pl_out.nc")

    def unstack_boxm_out(self):
        """Unstack the box model coarse output dataset."""
        print("Chunking the dataset")
        # Convert the dataset to a Dask dataset
        self.boxm_ds = self.boxm_ds.chunk({"cell": 100})  # Adjust chunk size based on your mem

        print("Set coords")
        # Convert 'level', 'lat', and 'lon' to coordinates
        self.boxm_ds_unstacked = self.boxm_ds.set_coords(["level", "longitude", "latitude"])

        print("Set index")
        # Create a multi-index for the 'cell' dimension
        self.boxm_ds_unstacked = self.boxm_ds_unstacked.set_index(
            cell=["level", "longitude", "latitude"]
        )

        print("Unstack the dataset")
        # Unstack the dataset
        self.boxm_ds_unstacked = self.boxm_ds_unstacked.unstack("cell")

        print("Compute the result")
        # # Compute the result to trigger the lazy evaluation
        # self.boxm_ds_unstacked = self.boxm_ds_unstacked.compute()

    # Plotting methods

    # Validation methods
    def mc_test(params, fl_df, pl_df, chem_ds):
        """Check if mass is conserved in the box model.

        Parameters
        ----------
        params : pd.Series
            Series containing simulation parameters.
        fl_df : pd.DataFrame
            DataFrame containing flight data.
        pl_df : pd.DataFrame
            DataFrame containing plume data.
        chem_ds : xr.Dataset
            Dataset containing chemical data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing mass conservation data.
        """

        # Initialize the dictionary
        vecmass = {emi_species: [] for emi_species in chem_ds["emi_species"].values.tolist()}
        gridmass = {emi_species: [] for emi_species in chem_ds["emi_species"].values.tolist()}
        mc = {emi_species: [] for emi_species in chem_ds["emi_species"].values.tolist()}

        # Constants
        mm = [30.01, 46.01, 28.01, 30.03, 44.05, 28.05, 42.08, 26.04, 78.11]  # g/mol
        NA = 6.022e23  # Avogadro's number

        for s, emi_species in enumerate(chem_ds["emi_species"].values):
            max_fl_time = fl_df["time"].max()

            for ts, t in enumerate(pl_df["time"].unique()[:-1]):
                if ts == 0:
                    total_vector_mass = 0
                    total_grid_mass = 0
                    percent_mass_conserved = 0
                    vecmass[emi_species].append(total_vector_mass)
                    gridmass[emi_species].append(total_grid_mass)
                    mc[emi_species].append(percent_mass_conserved)
                    continue

                previous_time = pl_df["time"].unique()[ts - 1]
                fl_snapshot = fl_df[fl_df["time"] == previous_time]

                if t <= max_fl_time:
                    # Accumulate vector mass for all flights
                    vector_mass = fl_snapshot[emi_species]

                    total_vector_mass += vector_mass.sum()

                # Grab plume mass from grid data
                grid_concs = chem_ds["emi"].sel(emi_species=emi_species, time=t)

                if (grid_concs == 0).all():
                    pass
                else:
                    # Compute the boolean indexer first
                    grid_concs_over_zero = grid_concs > 0
                    grid_concs_over_zero = grid_concs.where(grid_concs_over_zero, drop=True)

                    grid_mass = (
                        grid_concs_over_zero
                        * chem_ds["M"].sel(time=t)
                        * 1e-9
                        * (mm[s] / NA)
                        * params.loc["vres_sim"]
                        * units.latitude_distance_to_m(params.loc["hres_sim"])
                        * units.longitude_distance_to_m(
                            params.loc["hres_sim"],
                            (params.loc["lat_bounds"][0] + params.loc["lat_bounds"][1]) / 2,
                        )
                        * 1e03
                    )  # convert to kg/m^3

                    total_grid_mass = grid_mass.sum().item()

                    percent_mass_conserved = total_grid_mass / total_vector_mass * 100

                # Append the percentage to the list in the dictionary
                vecmass[emi_species].append(total_vector_mass)
                gridmass[emi_species].append(total_grid_mass)
                mc[emi_species].append(percent_mass_conserved)

        # convert the dictionary to a DataFrame
        vecmass = pd.DataFrame(
            vecmass, index=pl_df["time"].unique()[:-1], 
            columns=chem_ds["emi_species"].values.tolist()
        )
        gridmass = pd.DataFrame(
            gridmass, index=pl_df["time"].unique()[:-1], 
            columns=chem_ds["emi_species"].values.tolist()
        )
        mc = pd.DataFrame(
            mc, index=pl_df["time"].unique()[:-1], 
            columns=chem_ds["emi_species"].values.tolist()
        )

        return vecmass, gridmass, mc

    def boxm_test(run_path, data_path, job_id, chem_ds):
        """Run the box model for selected cells and job_id.

        Parameters
        ----------
        run_path : str
            Path to the box model executable.
        data_path : str
            Path to the data directory.
        job_id : str
            Job ID.
        chem_ds : xr.Dataset
            Dataset containing chemical data.

        Returns
        -------
        xr.Dataset
            Dataset containing chemical data for the selected cells.
        """

        # create input file for original boxm
        gen_boxm_orig_input(data_path, chem_ds, job_id)

        gen_zen_file(data_path, chem_ds, job_id)

        gen_emi_file(data_path, chem_ds, job_id)

        # # calls fortran with input file and generates .OUT files
        subprocess.call(
            [run_path + "boxm_orig", data_path, job_id],
        )

        return update_chem_ds(data_path, chem_ds, job_id)


# Helper functions for GPAT
def grab_species_num(run_path, species_out: np.array) -> np.array:
    """Grab the species numbers for the species of interest in output."""
    # Read species names from the file into a list
    with open(f"{run_path}species_num.txt") as file:
        species_list = [line.strip() for line in file]

    # Create a dictionary mapping species names to their line numbers
    species_dict = {species: index for index, species in enumerate(species_list)}

    return np.array([species_dict[species] + 1 for species in species_out])

# Function to convert dictionary to dataclass instance
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

# Function to filter out inherited params from ModelParams
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
