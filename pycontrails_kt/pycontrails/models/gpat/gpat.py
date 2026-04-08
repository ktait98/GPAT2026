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
    domain_mode: Literal["fixed", "auto"] = "fixed"  # "auto" derives bounds from flight start/end points
    domain_margin_deg: float = 0.0  # extra horizontal buffer [deg] added to auto bounds for plume advection
    hres_sim_c: float = 0.01  # horizontal resolution [deg]
    vres_sim_c: float = 500  # vertical resolution [m]
    hres_sim_f: float = 0.001  # horizontal resolution [deg]
    vres_sim_f: float = 100  # vertical resolution [m]

    run_path: Optional[str] = None  # path to run GPAT from
    data_path: Optional[str] = None  # path to data directory
    job_id: Optional[str] = None  # job ID

@dataclass
class FlParams:
    """Default flight/fleet parameters."""
    mode: Literal["direct", "synthetic"] = "direct"
    file: Optional[str] = None
    ac_type: Optional[str] = "A320"  # aircraft type
    fl0_speed: Optional[float] = 100.0  # m/s
    fl0_rocd: Optional[float] = 0.0  # m/s
    target_altitude: Optional[float] = None  # m, overrides fl0_rocd when set
    fl0_heading: Optional[float] = 0.0  # deg
    fl0_coords0: Optional[tuple[float, float, float]] = (0.1, 0.125, 12500)  # lat, lon, alt [deg, deg, m]
    # Optional control points for a tactile route definition.
    # Each tuple is (lat, lon, alt_m); times are spread uniformly over t_fl.
    control_waypoints: Optional[list[tuple[float, float, float]]] = None
    domain_margin_deg: float = 0.01
    sep_dist: Optional[tuple[float, float, float]] = (5000, 2000, 0)  # dx, dy, dz [m]
    n_ac: Optional[int] = 1  # number of aircraft

@dataclass
class PlParams:
    """Default plume dispersion parameters."""
    depth: float = 50.0  # initial plume depth, [m]
    width: float = 50.0  # initial plume width, [m]
    verbose_outputs: bool = False  # print verbose outputs
    shear: float = 0.01  # shear [m/s]
    n_slices: int = 3  # number of slices in the plume
    f_max: float = 0.99  # maximum fraction of total emissions in any slice
    output_pl_slices: bool = True  # output plume slices to netCDF
    n_points: int = 32

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
    delta_chem_mode: int = 0  # 0=fine only, 1=fine+coarse delta chemistry

    species_emi: tuple[str, ...] = ("NO",)
    
    species_pl: tuple[str, ...] = (
    # Core NOx-O3 photochemistry memory
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
    
    species_out: tuple[str, ...] = (
    "O3", "NO2", "NO", "NO3", "N2O5", "HNO3",
    "HONO", "HO2", "OH", "H2O2",
    "CO", "CH4", "CH3O2",
    "HO2NO2", "PAN", "SO2"
    )


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
    pl_params : PlParams
        Plume dispersion parameters.
    met_params : MetParams
        Meteorological parameters.
    chem_params : ChemParams
        Chemistry parameters.
    """

    name = "GPAT"
    long_name = "Gridded Plume Analysis Tool"
    # default_params = (FlParams, PlParams, SimParams)

    
    def __init__(self, 
                 sim_params: SimParams, 
                 fl_params: FlParams,
                 pl_params: PlParams, 
                 met_params: MetParams, 
                 chem_params: ChemParams):
        super().__init__()

        # Build spatial grid from current bounds
        self._build_grid(sim_params)

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

        self.times_out = pd.date_range(
            start=sim_params.t_out[0],
            end=sim_params.t_out[0] + sim_params.t_out[2],
            freq=sim_params.t_out[1],
        )

        # Set up paths and job ID
        self.run_path = sim_params.run_path
        self.data_path = sim_params.data_path
        self.job_id = sim_params.job_id

        # If running in direct mode with a specified flight file, derive job_id from the filename if not already set.
        if fl_params.mode == "direct" and fl_params.file is not None:
            self.job_id = os.path.splitext(os.path.basename(fl_params.file))[0]

        sim_params.date_created = pd.Timestamp.now()

        # Grab species numbers from the input files for later use in indexing model outputs
        chem_params.species_emi_num = grab_species_num(self.run_path, chem_params.species_emi)
        chem_params.species_pl_num = grab_species_num(self.run_path, chem_params.species_pl)
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
            "fl_params": fl_params,
            "pl_params": pl_params,
            "met_params": met_params,
            "chem_params": chem_params,
        }

        # Set the model parameters
        self.sim_params = sim_params
        self.fl_params = fl_params
        self.pl_params = pl_params
        self.met_params = met_params
        self.chem_params = chem_params
        self.all_params = all_params

        # Set up the model classes for preprocessing and running GPAT
        self.setup = GPATSetup(self)
        self.run = GPATRun(self)

    def _build_grid(self, sim_params):
        """Build coarse grid vectors and meter axes from sim_params bounds."""
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

    def _auto_bounds_from_flights(self, flights):
        """Derive domain bounds from flight start/end waypoints.

        Sets ``sim_params.*_bounds`` to the bounding box of all first and
        last waypoints, plus a configurable horizontal margin, snapped
        outward to the nearest coarse grid cell boundary.

        The margin (``sim_params.domain_margin_deg``) should account for
        plume advection and dispersion so that plumes blown by wind are
        not clipped at the domain edge.
        """
        lats, lons, alts = [], [], []
        for fl in flights:
            df = fl.dataframe
            for idx in [0, -1]:  # first and last waypoint
                lats.append(float(df["latitude"].iloc[idx]))
                lons.append(float(df["longitude"].iloc[idx]))
                alts.append(float(df["altitude"].iloc[idx]))

        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)
        alt_min, alt_max = min(alts), max(alts)

        # Apply horizontal margin for plume advection/dispersion
        margin = self.sim_params.domain_margin_deg
        lat_min -= margin
        lat_max += margin
        lon_min -= margin
        lon_max += margin

        # Snap bounds outward to coarse grid resolution
        hres = self.sim_params.hres_sim_c
        lat_min = np.floor(lat_min / hres) * hres
        lat_max = np.ceil(lat_max / hres) * hres
        lon_min = np.floor(lon_min / hres) * hres
        lon_max = np.ceil(lon_max / hres) * hres

        # Pad altitude and snap to vertical resolution
        vres = self.sim_params.vres_sim_c
        alt_min = np.floor(alt_min / vres) * vres
        alt_max = np.ceil(alt_max / vres) * vres

        self.sim_params.lat_bounds = (lat_min, lat_max)
        self.sim_params.lon_bounds = (lon_min, lon_max)
        self.sim_params.alt_bounds = (alt_min, alt_max)

        print(
            f"Auto domain bounds (margin={margin:.4f} deg, snapped to hres={hres}, vres={vres}):\n"
            f"  lat_bounds = ({lat_min:.6f}, {lat_max:.6f})\n"
            f"  lon_bounds = ({lon_min:.6f}, {lon_max:.6f})\n"
            f"  alt_bounds = ({alt_min:.1f}, {alt_max:.1f})"
        )

    def preprocess_gpat(self):
        """Preprocess inputs for GPAT FORTRAN implementation (BOXM and CONTRAIL in future)."""
        # Generate flight trajectory points
        if self.fl_params.n_ac > 0:
            self.fl = self.setup.traj_gen()

        # Auto-derive domain bounds from flight endpoints if requested
        if self.sim_params.domain_mode == "auto" and self.fl_params.n_ac > 0:
            self._auto_bounds_from_flights(self.fl)
            self._build_grid(self.sim_params)

        # Generate meteorological data
        self.met = self.setup.gen_met()

        # Generate background chemistry data
        self.bg_chem = self.setup.gen_bg_chem()

        # Calculate aircraft performance using PS Model
        if self.fl_params.n_ac > 0:
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


class GPATSetup:
    """Setup the GPAT model, ready for FORTRAN computation."""

    def __init__(self, gpat):
        self.gpat = gpat

    # Setup methods
    def traj_gen(self) -> list[Flight]:
        """Generate flight trajectory points. Supports loading and plotting all test flights if requested."""
        fl_params = self.gpat.fl_params
        sim_params = self.gpat.sim_params

        if fl_params.mode == "direct":
            if fl_params.file is None:
                raise ValueError("Flight file must be provided for direct mode.")

            df = pd.read_csv(fl_params.file)
            fl = []

            for flight_id in sorted(df["flight_id"].dropna().unique()):
                df_flt = df[df["flight_id"] == flight_id].copy()

                # Ensure time is datetime and sort
                df_flt["time"] = pd.to_datetime(df_flt["time"])
                df_flt = df_flt.sort_values("time").reset_index(drop=True)

                # Skip degenerate flights
                if len(df_flt) < 2:
                    print(f"Skipping flight_id={flight_id}: fewer than 2 valid points")
                    continue

                fli = Flight(data=df_flt)
                fli.attrs = {"flight_id": int(flight_id), "aircraft_type": fl_params.ac_type}

                fli["waypoint"] = np.arange(len(fli))
                fl.append(fli)

            if not fl:
                raise ValueError("No valid flights were loaded from the direct-mode CSV.")

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
            flight_duration_s = sim_params.t_fl[2].total_seconds()
            dist = fl_params.fl0_speed * flight_duration_s
            if fl_params.target_altitude is not None:
                alt1 = float(fl_params.target_altitude)
                rocd_used = (alt1 - alt0) / flight_duration_s
            else:
                rocd_used = float(fl_params.fl0_rocd)
                alt1 = alt0 + rocd_used * flight_duration_s

            # Guard against synthetic trajectories that immediately leave the
            # vertical simulation domain and collapse to 0-1 retained waypoints.
            rocd_min = (sim_params.alt_bounds[0] - alt0) / flight_duration_s
            rocd_max = (sim_params.alt_bounds[1] - alt0) / flight_duration_s
            if not (rocd_min <= rocd_used <= rocd_max):
                raise ValueError(
                    "Synthetic trajectory exits alt_bounds over t_fl duration. "
                    f"Computed alt1={alt1:.1f} m from alt0={alt0:.1f} m, "
                    f"rocd={rocd_used:.4f} m/s, duration={flight_duration_s:.1f} s; "
                    f"alt_bounds={sim_params.alt_bounds}. "
                    f"Choose fl0_rocd within [{rocd_min:.4f}, {rocd_max:.4f}] m/s, "
                    "or widen alt_bounds."
                )

            # calculate the final coordinates
            geod = Geod(ellps="WGS84")
            lon1, lat1, _ = geod.fwd(lon0, lat0, heading, dist)

            # Build leader route from either control waypoints or start/end pair.
            control_waypoints = getattr(fl_params, "control_waypoints", None)
            if control_waypoints is not None and len(control_waypoints) >= 2:
                lat_points = [float(wp[0]) for wp in control_waypoints]
                lon_points = [float(wp[1]) for wp in control_waypoints]
                alt_points = [float(wp[2]) for wp in control_waypoints]
            else:
                lon_points = [lon0, lon1]
                lat_points = [lat0, lat1]
                alt_points = [alt0, alt1]

            times = pd.date_range(
                start=sim_params.t_fl[0],
                end=sim_params.t_fl[0] + sim_params.t_fl[2],
                periods=len(lat_points),
            )

            # create Flight object for leader flight and resample points according to ts_fl
            df = pd.DataFrame()
            df["longitude"] = lon_points
            df["latitude"] = lat_points
            df["altitude"] = alt_points
            df["time"] = times
            ts_fl_sec = int(sim_params.t_fl[1].total_seconds())
            if ts_fl_sec < 60:
                freq_str = f"{ts_fl_sec}s"
            else:
                freq_str = f"{int(ts_fl_sec // 60)}min"

            fl0 = Flight(df).resample_and_fill(freq=freq_str)
            fl0.attrs = {"flight_id": int(0), "aircraft_type": fl_params.ac_type}
            fl0["waypoint"] = np.arange(len(fl0))  # add waypoint index for tracking in BOXM and PL outputs
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

        # # add time relative to simulation start time
        # time_rel_s = (met["time"].values - np.datetime64(self.gpat.times_sim[0])).astype("timedelta64[s]").astype(float)
        # met.data = met.data.assign_coords({
        #     "time_rel_s": (("time"), time_rel_s)  
        # })

        return met

    def gen_bg_chem(self) -> xr.Dataset:
        """Generate background chemistry data."""
        month = self.gpat.times_sim[0].month

        bg_chem = (
            xr.open_dataset(self.gpat.inputs_glob + "species.nc", engine="netcdf4")
            .sel(month=month - 1)
            .rename({"species": "species_boxm"})
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
    
    def sim_plumes(self) -> list[pd.DataFrame]:
        """Simulate plume dispersion/advection using Pycontrails Dry Advection Model."""
        pl_params = self.gpat.pl_params
        sim_params = self.gpat.sim_params
        
        met = self.gpat.met
        fl = self.gpat.fl

        dry_adv = DryAdvection(
            met,
            max_age=sim_params.t_pl[2],
            dt_integration=sim_params.t_pl[1],
            shear=pl_params.shear,
        )

        pl_list = []
        fl_list = []

        # simulate plumes for each flight and store results in list of dataframes
        for i, fli in enumerate(fl):
            # print(f"Flight {i}: type={type(fli)}, hasattr(dataframe)={hasattr(fli, 'dataframe')}, hasattr(data)={hasattr(fli, 'data')}, len={len(fli) if hasattr(fli, '__len__') else 'N/A'}")
            # if hasattr(fli, 'dataframe'):
            #     print(f"  dataframe shape: {fli.dataframe.shape}")
            #     print(f"  dataframe columns: {fli.dataframe.columns}")
            #     print(f"  dataframe head:\n{fli.dataframe.head()}")
            #     print(f"  dataframe NaNs:\n{fli.dataframe.isna().sum()}")
            #     # print(f"  dataframe Nan rows:\n{fli.dataframe[fli.dataframe.isna().any(axis=1)]}")
            # elif hasattr(fli, 'data'):
            #     print(f"  data shape: {fli.data.shape}")

            met = self.gpat.met

            if len(fli.dataframe) < 2:
                print(f"Skipping flight {i}: fewer than 2 points after met domain masking (pre-advection)")
                continue

            pli = dry_adv.eval(fli)

            # Attach altitude to plume for later masking and analysis
            pli["altitude"] = units.pl_to_m(pli["level"])

            # Mask plume to simulation domain
            pli = self.mask_plume_to_sim_domain(pli)

            print(fli)
            print(pli)
      
            met = self.gpat.met

            # convert both flights and plumes to dataframes
            fli_df = fli.dataframe
            # Add level to flight dataframe right after converting
            fli_df["level"] = units.m_to_pl(fli_df["altitude"].values)
            # Reorder columns to have level after latitude
            var_list = list(fli_df)
            var_list.remove("level")
            var_list.insert(var_list.index("latitude") + 1, "level")
            fli_df = fli_df[var_list]
            for column in fli_df.columns:
                # Replace NaN values in the column with the value from the previous row
                fli_df[column] = fli_df[column].ffill()

            pli_df = pli.dataframe
            # Only calculate heading if there are at least 2 rows
            if len(pli_df) < 2:
                print(f"Skipping flight {i}: plume has fewer than 2 points after masking.")
                continue
            pli_df = self.calc_heading(pli_df)
            if len(fli_df) == 0 or len(pli_df) == 0:
                print(f"Skipping flight {i}: all points advected outside met domain after DryAdvection.")
                continue
            pli_df["flight_id"] = fli_df["flight_id"].iloc[0]
            fl_list.append(fli_df)
            pl_list.append(pli_df)

        # concatenate all successfully processed flights and plumes into single dfs
        if not fl_list or not pl_list:
            print("No valid flights or plumes to process.")
            return None, None
        fl_df = pd.concat(fl_list)
        pl_df = pd.concat(pl_list)

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

        pl["longitude_m"], pl["latitude_m"] = lonlat_to_m(
                                                pl["longitude"].values,
                                                pl["latitude"].values,
                                                self.gpat.lons.min(),
                                                self.gpat.lats.min(),
                                            )
 
        pl["sin_a"] = np.sin(np.radians(pl["heading"]))
        pl["cos_a"] = np.cos(np.radians(pl["heading"]))
        pl["altitude"] = units.pl_to_m(pl["level"])
        pl["time"] = pl["time"] - sim_params.t_pl[1]
        pl["age"] = pl["age"] - sim_params.t_pl[1]

        return fl, pl
    
    def gen_inputs(self):
        """Generate BOXM inputs."""
        # Initialise parameters dataset
        self.init_params()
        if self.gpat.fl_params.n_ac > 0:
            # Initialise flight dataset
            self.init_fl_ds_nc()
            #Initialise plume dataset
            self.init_pl_ds_nc()
        # Initialize the box model dataset
        self.init_boxm_ds_nc()

    def gen_outputs(self):
        """Generate BOXM output templates."""
        # Initialise boxm coarse output dataset
        self.init_boxm_out_nc()
        
        if self.gpat.fl_params.n_ac > 0:
            # Initialise patch table output dataset
            self.init_patch_table_nc()
            # Initialise plume output dataset
            self.init_pl_out_nc()  

    # --- Input dataset initialization methods ---
    def init_params(self):
        """Save the simulation parameters to a JSON file for record-keeping."""
        params_path = pathlib.Path(f"{self.gpat.inputs_job}/params.json")
        with open(params_path, "w") as f:
            json.dump(self.gpat.all_params, f, default=str, indent=4)
        print(f"Saved simulation parameters to {params_path}")

    def init_fl_ds_nc(self):
        """Initialize the flight dataset for BOXM."""
        df = self.gpat.fl.copy()

        # Ensure flight_id is numeric; if not, create numeric flight_id and store original as flight_name
        if not pd.api.types.is_numeric_dtype(df["flight_id"]):
            df["flight_name"] = df["flight_id"]
            df["flight_id"] = pd.factorize(df["flight_id"])[0] + 1  # start from 1
        else:
            df["flight_id"] = df["flight_id"].astype(int) + 1  # start flight_id from 1 (FORTRAN)
        df["waypoint"] = df["waypoint"].astype(int) + 1

        # pl_keys = self.gpat.pl[["flight_id", "waypoint"]].drop_duplicates()
        # # Do NOT increment pl_keys here!
        # df = df.merge(pl_keys, on=["flight_id", "waypoint"], how="inner")

        # Sort by flight_id, waypoint to establish the seg_id ordering
        df = df.sort_values(by=["flight_id", "waypoint"]).reset_index(drop=True)
        df["seg_id"] = df.groupby(["flight_id", "waypoint"]).ngroup() + 1

        # Set multi-index (flight_id, waypoint) and convert to xarray
        self.gpat.fl_ds = df.set_index(["seg_id"]).to_xarray()

        # Set flight id and waypoint as coordinates
        self.gpat.fl_ds = self.gpat.fl_ds.assign_coords(
            flight_id = ("seg_id", df["flight_id"].values),
            waypoint = ("seg_id", df["waypoint"].values)
        )

        self.gpat.fl_ds["time_rel_s"] = (self.gpat.fl_ds["time"] - np.datetime64(self.gpat.sim_params.t_sim[0])).astype("timedelta64[s]").astype(int)
        self.gpat.fl_ds["time_idx"] = (self.gpat.fl_ds["time"] - np.datetime64(self.gpat.sim_params.t_sim[0])).astype("timedelta64[s]").astype(int) // int(self.gpat.sim_params.t_sim[1].total_seconds()) + 1
        self.gpat.fl_ds["time"] = self.gpat.fl_ds["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Drop individual species variables
        self.gpat.fl_ds = self.gpat.fl_ds.drop_vars(["air_temperature", "specific_humidity", 
                                 "nox_ei", "co_ei", "hc_ei", "nvpm_ei_m",
                                 "nvpm_ei_n", "co2", "h2o", "so2", 
                                 "sulphates", "oc", "nox", "co", "hc", 
                                 "nvpm_mass", "nvpm_number", "nvPM",
                                 "CO2", "H2O", "SO2", "NO", "NO2",
                                 "CO", "HCHO", "CH3CHO", "C2H4", "C3H6", "C2H2", "BENZENE"])
        
        # --- Assign useful metadata ---
        self.gpat.fl_ds = self.gpat.fl_ds.assign_attrs(
            description="Flight trajectory and emissions data for BOXM",
        )

        # Save to NetCDF
        nc_path = pathlib.Path(f"{self.gpat.inputs_job}/fl_ds.nc")
        if nc_path.exists():
            nc_path.unlink()
        self.gpat.fl_ds.to_netcdf(nc_path, mode="w")
        print(f"Saved {nc_path}")

    def init_pl_ds_nc(self):
        """Initialize the plume dataset for the box model (PL_DS.NC)."""
        df = self.gpat.pl.copy()
        sim_params = self.gpat.sim_params
        pl_params = self.gpat.pl_params
        chem_params = self.gpat.chem_params

        # Sort by flight_id, waypoint to establish the seg_id ordering
        df = df.sort_values(by=["flight_id", "waypoint"]).reset_index(drop=True)

        # Ensure flight_id is numeric; if not, create numeric flight_id and store original as flight_name
        if not pd.api.types.is_numeric_dtype(df["flight_id"]):
            df["flight_name"] = df["flight_id"]
            df["flight_id"] = pd.factorize(df["flight_id"])[0] + 1  # start from 1
        else:
            df["flight_id"] = df["flight_id"].astype(int) + 1  # start flight_id from 1 (FORTRAN)
        df["waypoint"] = df["waypoint"].astype(int) + 1
        
        # Create segment index: assign same seg_id to all rows with same (flight_id, waypoint)
        df["seg_id"] = df.groupby(["flight_id", "waypoint"]).ngroup() + 1  # start seg_id from 1 (FORTRAN)

        # Calculate total number of unique segments
        nseg = df["seg_id"].max()

        # Get unique flight_id and waypoint for each seg_id
        seg_unique = df.drop_duplicates(subset=["seg_id"])[["seg_id", "flight_id", "waypoint"]].set_index("seg_id").sort_index()
        
        # Set multi-index (seg_id, time)
        self.gpat.pl_ds = df.set_index(["seg_id", "time"]).to_xarray()

        # Set flight id and waypoint as coordinates using unique values per seg_id
        self.gpat.pl_ds = self.gpat.pl_ds.assign_coords(
            flight_id = ("seg_id", seg_unique["flight_id"].values),
            waypoint = ("seg_id", seg_unique["waypoint"].values),
            species_emi_num = ("species_emi", chem_params.species_emi_num)
        )

        species_cols = list(self.gpat.chem_params.species_emi)

        # Extract species emission mass directly from plume dataframe (once per waypoint)
        # Use the dataframe before conversion to xarray to ensure species columns are available
        species_data = [df[col] for col in species_cols if col in df.columns]

        # Stack species into single variable: (seg_id, species_emi)
        # Concatenate along species dimension
        if species_data:
            seg_id_vals = df["seg_id"].unique()
            seg_id_vals.sort()
            emi_pl_mass_seg = xr.concat(
                [xr.DataArray(data.groupby(df["seg_id"]).first().values, dims="seg_id", coords={"seg_id": seg_id_vals}) for data in species_data],
                dim=pd.Index([col for col in species_cols if col in df.columns], name="species_emi")
            ).transpose("seg_id", "species_emi")

            # Expand to time dimension and only populate first timestep per segment
            emi_pl_mass = xr.DataArray(
                np.zeros((len(self.gpat.pl_ds.seg_id), len(emi_pl_mass_seg.species_emi)), dtype=float),
                coords={
                    "seg_id": self.gpat.pl_ds.seg_id,
                    "species_emi": emi_pl_mass_seg.species_emi,
                },
                dims=("seg_id", "species_emi"),
            )

            # Determine first time for each segment
            first_time = df.groupby("seg_id")["time"].min()
            for seg_id, t0 in first_time.items():
                if t0 in self.gpat.pl_ds.time.values:
                    emi_pl_mass.loc[dict(seg_id=seg_id)] = emi_pl_mass_seg.sel(seg_id=seg_id)

            # Add to plume dataset
            self.gpat.pl_ds["emi_pl_mass"] = emi_pl_mass
        
        # Drop all individual species columns
        all_species_cols = ['CO2', 'H2O', 'SO2', 'NO', 
                            'NO2', 'CO', 'HCHO', 'CH3CHO', 
                            'C2H4', 'C3H6', 'C2H2', 'BENZENE', 'nvPM']

        self.gpat.pl_ds = self.gpat.pl_ds.drop_vars((*all_species_cols, "fuel_flow", "fuel_burn", "true_airspeed", "sin_a", "cos_a"))

        # Convert timedelta to total seconds, handling NaT values before casting to int.
        age_values = self.gpat.pl_ds["age"].values
        age_clean = np.where(np.isnat(age_values), np.timedelta64(0, "s"), age_values)
        age_seconds = age_clean.astype("timedelta64[s]").astype(np.int64)
        self.gpat.pl_ds["age_s"] = (("seg_id", "time"), age_seconds)

        df = self.gpat.pl_ds["age_s"].to_dataframe().reset_index()
        # first_time_dict = df.groupby("seg_id")["time"].min().to_dict()

        # # Create a boolean mask for (seg_id, time) pairs that are the first for that seg_id
        # first_time_mask = np.zeros(self.gpat.pl_ds["age_s"].shape, dtype=bool)
        # for i, seg_id in enumerate(self.gpat.pl_ds["age_s"].seg_id.values):
        #     first_time = first_time_dict[seg_id]
        #     time_idx = np.where(self.gpat.pl_ds["age_s"].time.values == first_time)[0]
        #     if len(time_idx) > 0:
        #         first_time_mask[i, time_idx[0]] = True

        # Now build the active mask
        active_mask = (self.gpat.pl_ds["age_s"].values > 0) & (self.gpat.pl_ds["age_s"].values <= sim_params.t_pl[2].total_seconds())
        active_mask = active_mask.astype(int)
        self.gpat.pl_ds = self.gpat.pl_ds.assign_coords(active_seg_flag=(("seg_id", "time"), active_mask))

        # for seg_id in self.gpat.pl_ds.seg_id.values:
        #     print(f"Seg = {seg_id} Active seg flag = {self.gpat.pl_ds['active_seg_flag'].sel(seg_id=seg_id).values}")

        self.gpat.pl_ds = self.gpat.pl_ds.assign_coords(
            time_rel_s = ("time", (self.gpat.pl_ds["time"].values - np.datetime64(self.gpat.sim_params.t_sim[0])).astype("timedelta64[s]").astype(int)),
            time_idx = ("time", ((self.gpat.pl_ds["time"].values - np.datetime64(self.gpat.sim_params.t_sim[0])).astype("timedelta64[s]").astype(int) // int(self.gpat.sim_params.t_sim[1].total_seconds())) + 1),
            species_emi_num = ("species_emi", chem_params.species_emi_num)
        )

        self.gpat.pl_ds["time"] = self.gpat.pl_ds["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        self.gpat.pl_ds["age"].values = self.gpat.pl_ds["age"].values.astype(str)

        self.gpat.pl_ds = self.gpat.pl_ds.assign_attrs(
            nseg=nseg,
            ts_fl=sim_params.t_fl[1].total_seconds(),
            ts_pl=sim_params.t_pl[1].total_seconds(),
            ts_sim=sim_params.t_sim[1].total_seconds(),
            ts_out=sim_params.t_out[1].total_seconds(),
            # species definitions
            species_emi=chem_params.species_emi,
            species_pl=chem_params.species_pl,
            # species nums
            species_emi_num=chem_params.species_emi_num,
            species_pl_num=chem_params.species_pl_num,
            # boxm attrs
            n_slices=pl_params.n_slices,
            f_max=pl_params.f_max,
            output_pl_slices=int(pl_params.output_pl_slices),
            n_points=pl_params.n_points,
            max_age_s=int(sim_params.t_pl[2].total_seconds()),

            description="Emission species mass in plume segments",
        )

        # Save to NetCDF
        nc_path = pathlib.Path(f"{self.gpat.inputs_job}/pl_ds.nc")
        if nc_path.exists():
            print("Deleting existing pl_ds.nc")
            nc_path.unlink()
        self.gpat.pl_ds.to_netcdf(nc_path, mode="w")
        print(f"Saved {nc_path}")

    def init_boxm_ds_nc(self):
        """Initialize the box model dataset (met + background chem on coarse grid)."""
        sim_params = self.gpat.sim_params
        fl_params = self.gpat.fl_params
        chem_params = self.gpat.chem_params

        # --- Merge meteorology and background chemistry fields ---
        self.gpat.boxm_ds = xr.merge([self.gpat.met.data, self.gpat.bg_chem])

        #self.gpat.boxm_ds["time_rel_s"] = (self.gpat.boxm_ds["time"] - np.datetime64(self.gpat.sim_params.t_sim[0])).astype("timedelta64[s]").astype(float)
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
            delta_chem_mode=int(chem_params.delta_chem_mode),
            n_ac=fl_params.n_ac,

            # grid topology for advection
            nlon=len(self.gpat.boxm_ds.longitude),
            nlat=len(self.gpat.boxm_ds.latitude),
            nlev=len(self.gpat.boxm_ds.level),

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
    def init_pl_out_nc(self):
        """Initialize the plume output dataset (PL_OUT.NC)."""
        # Load the input plume dataset
        pl_ds = xr.open_dataset(f"{self.gpat.inputs_job}/pl_ds.nc")
        chem_params = self.gpat.chem_params
        
        species_pl = np.array(chem_params.species_pl, dtype="U10")
        times_out = pd.to_datetime(self.gpat.times_out.values).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Build a fresh output dataset on the output time grid
        self.gpat.pl_out = xr.Dataset(
            coords={
                "seg_id": pl_ds.seg_id,
                "time": times_out,
                "species_pl": species_pl,
            }
        )

        # build and attach empty pl_slices type
        if self.gpat.pl_params.output_pl_slices:
            pl_slices = xr.Dataset(
                {
                "y_half": (("seg_id", "slice_id", "time"), np.zeros((len(pl_ds.seg_id), self.gpat.pl_params.n_slices, len(times_out)))),
                "z_half": (("seg_id", "slice_id", "time"), np.zeros((len(pl_ds.seg_id), self.gpat.pl_params.n_slices, len(times_out)))),
                "m_frac": ("slice_id", np.zeros(self.gpat.pl_params.n_slices)),
                "w_slice": ("slice_id", np.zeros(self.gpat.pl_params.n_slices)),
                "ellipses_m": (("seg_id", "pt_id", "coord", "time"), np.zeros((len(pl_ds.seg_id), self.gpat.pl_params.n_points, 3, len(times_out)))),
                "slice_polys_m": (("seg_id", "slice_id", "corner_id", "coord", "time"), np.zeros((len(pl_ds.seg_id), self.gpat.pl_params.n_slices, 4, 3, len(times_out)))),
                "slice_boxes_m": (("face", "slice_id", "corner_id", "coord", "time"), np.zeros((2, self.gpat.pl_params.n_slices, 4, 3, len(times_out)))),
                },
                coords={
                    "seg_id": self.gpat.pl_out.seg_id,
                    "species_pl": self.gpat.pl_out.species_pl,
                    "time": self.gpat.pl_out.time,
                    "pt_id": np.arange(1, self.gpat.pl_params.n_points + 1),
                    "slice_id": np.arange(1, self.gpat.pl_params.n_slices + 1),
                    "corner_id": xr.DataArray(np.array(["BL", "TL", "TR", "BR"], dtype="U2"), dims=("corner_id",)),
                    "coord": xr.DataArray(np.array(["lon_m", "lat_m", "alt_m"], dtype="U5"), dims=("coord",)),
                    "face": xr.DataArray(np.array(["back", "front"], dtype="U6"), dims=("face",)),
                }
            )
            for var in pl_slices.data_vars:
                self.gpat.pl_out[var] = pl_slices[var]

        # Carry segment-level identifiers
        if "flight_id" in pl_ds.coords:
            self.gpat.pl_out = self.gpat.pl_out.assign_coords(
                flight_id=("seg_id", pl_ds["flight_id"].values)
            )
        if "waypoint" in pl_ds.coords:
            self.gpat.pl_out = self.gpat.pl_out.assign_coords(
                waypoint=("seg_id", pl_ds["waypoint"].values)
            )

        # Time-related coordinates for output cadence
        self.gpat.pl_out = self.gpat.pl_out.assign_coords(
            species_pl_num=("species_pl", chem_params.species_pl_num),
            time_rel_s=("time", (self.gpat.times_out.values - np.datetime64(self.gpat.sim_params.t_sim[0])).astype("timedelta64[s]").astype(int)),
            time_idx=("time", ((self.gpat.times_out.values - np.datetime64(self.gpat.sim_params.t_sim[0])).astype("timedelta64[s]").astype(int) // int(self.gpat.sim_params.t_sim[1].total_seconds())) + 1),
        )

        pl_mass = xr.DataArray(
            np.zeros((
                len(self.gpat.pl_out.seg_id),
                len(species_pl),
                len(times_out),
            ), dtype=float),
            coords={
                "seg_id": self.gpat.pl_out.seg_id,
                "species_pl": species_pl,
                "species_pl_num": ("species_pl", chem_params.species_pl_num),
                "time": times_out,
                "time_rel_s": ("time", self.gpat.pl_out["time_rel_s"].values),
                "time_idx": ("time", self.gpat.pl_out["time_idx"].values),
            },
            dims=("seg_id", "species_pl", "time"),
            attrs={
                "units": "kg",
                "long_name": "Change in species mass due to chemistry",
                "note": "Add to initial emission mass (from fl_ds.nc) to get total mass"
            }
        )
        
        self.gpat.pl_out["pl_mass"] = pl_mass
        
        # Add metadata
        self.gpat.pl_out = self.gpat.pl_out.assign_attrs(
            description="Plume segment output for BOXM",
        )
        
        # Save
        nc_path = pathlib.Path(f"{self.gpat.outputs_job}/pl_out.nc")
        if nc_path.exists():
            print("Deleting existing pl_out.nc")
            nc_path.unlink()
        self.gpat.pl_out.to_netcdf(nc_path, mode="w")
        print(f"Saved {nc_path}")
    
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
        
    def init_patch_table_nc(self):
        """Initialize the patch output dataset (PATCH_OUT.NC)."""
        species_out = self.gpat.chem_params.species_out
        chem_params = self.gpat.chem_params

        self.gpat.patch_table = xr.Dataset(
            data_vars={
                "Y_del_f": (
                    ("row", "species_out"), 
                    da.zeros((0, len(species_out)), dtype=float),
                    {"units": "ppb"},
                ),
                
            },
            coords={
            
            "row": np.array([], dtype=int),   # 0-length row dimension
            "species_out": np.array(species_out),
            "species_out_num": ("species_out", chem_params.species_out_num),
            "time":   ("row", np.array([], dtype="object")),
            "time_rel_s": ("row", np.array([], dtype="int64")),
            "time_idx": ("row", np.array([], dtype="int64")),
            "row_cell_c": ("row", np.array([], dtype="int64")),
            "row_cell_f": ("row", np.array([], dtype="int64")),
            "latitude_f":  ("row", np.array([], dtype="float64")),
            "longitude_f": ("row", np.array([], dtype="float64")),
            "altitude_f":  ("row", np.array([], dtype="float64")),
            "level_f":     ("row", np.array([], dtype="float64")),
            }
        )

        # --- Assign useful metadata ---
        self.gpat.patch_table = self.gpat.patch_table.assign_attrs(
            description="Fine grid plume patch output for BOXM",
        )

        # Delete any existing NetCDF
        nc_path = pathlib.Path(f"{self.gpat.outputs_job}/patch_table.nc")
        if nc_path.exists():
            print("Deleting existing patch_table.nc")
            nc_path.unlink()

        # Save to NetCDF file
        self.gpat.patch_table.to_netcdf(nc_path, mode="w", unlimited_dims=["row"])
        print(f"Saved {nc_path}")

    # Helper functions used in GPAT Setup
    def calc_heading(self, pl_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate heading for each plume at each timestep."""
        pl_df = pl_df.sort_values(by=["time", "waypoint"]).copy()

        heading = (
            pl_df.groupby("time", group_keys=False)
            .apply(self.calculate_heading_g)
        )

        pl_df["heading"] = heading.reindex(pl_df.index)
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
    
    def mask_flight_to_sim_domain(self, flight: Flight) -> Flight:
        
        sim_params = self.gpat.sim_params
        
        lon_min, lon_max = sim_params.lon_bounds
        lat_min, lat_max = sim_params.lat_bounds
        alt_min, alt_max = sim_params.alt_bounds

        mask = (
            (flight.dataframe["longitude"] >= lon_min)
            & (flight.dataframe["longitude"] <= lon_max)
            & (flight.dataframe["latitude"] >= lat_min)
            & (flight.dataframe["latitude"] <= lat_max)
            & (flight.dataframe["altitude"] >= alt_min)
            & (flight.dataframe["altitude"] <= alt_max)
        )

        masked_df = flight.dataframe.loc[mask].copy()
        new_flight = Flight(data=masked_df)
        if hasattr(flight, "attrs"):
            new_flight.attrs = getattr(flight, "attrs", {})
        return new_flight
    
    def mask_plume_to_sim_domain(self, plume: GeoVectorDataset) -> GeoVectorDataset:

        sim_params = self.gpat.sim_params

        lon_min, lon_max = sim_params.lon_bounds
        lat_min, lat_max = sim_params.lat_bounds
        alt_min, alt_max = sim_params.alt_bounds

        mask = (
            (plume.dataframe["longitude"] >= lon_min)
            & (plume.dataframe["longitude"] <= lon_max)
            & (plume.dataframe["latitude"] >= lat_min)
            & (plume.dataframe["latitude"] <= lat_max)
            & (plume.dataframe["altitude"] >= alt_min)
            & (plume.dataframe["altitude"] <= alt_max)
        )

        masked_df = plume.dataframe.loc[mask].copy()
        new_plume = GeoVectorDataset(data=masked_df)
        if hasattr(plume, "attrs"):
            new_plume.attrs = getattr(plume, "attrs", {})
        return new_plume


class GPATRun:
    """Run the GPAT model."""
   
    def __init__(self, gpat):
        self.gpat = gpat

    # Run methods
    def run_boxm(self) -> xr.Dataset:
        """Run BOXM."""
        # Run the box model
        subprocess.call(
            [self.gpat.run_path + "boxm", self.gpat.job_id, self.gpat.data_path],
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