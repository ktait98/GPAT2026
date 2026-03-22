import numpy as np
import pandas as pd
import xarray as xr
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pycontrails.models.gpat.gpat import GPAT, SimParams, FlParams, PlParams, MetParams, ChemParams, dict_to_dataclass
import os
import holoviews as hv

# global simulation parameters
sim_params = {
    "t_fl": (pd.to_datetime("2022-01-20 13:00:00"), pd.Timedelta(minutes=1), pd.Timedelta(hours=1)),# (start time, time step, run time)
    "t_pl": (pd.to_datetime("2022-01-20 13:00:00"), pd.Timedelta(minutes=1), pd.Timedelta(hours=2)),# (start time, time step, max age)
    "t_sim": (pd.to_datetime("2022-01-20 12:00:00"), pd.Timedelta(seconds=20), pd.Timedelta(hours=4)),# (start time, time step, run time)
    "t_out": (pd.to_datetime("2022-01-20 12:00:00"), pd.Timedelta(minutes=1), pd.Timedelta(hours=4)),# (start time, time step, run time)
    "lat_bounds": (0, 1),  # lat bounds [deg]
    "lon_bounds": (0, 1),  # lon bounds [deg]
    "alt_bounds": (10000, 13000),  # alt bounds [m]
    "hres_sim_c": 0.5,  # coarse horizontal resolution [deg]
    "vres_sim_c": 500,  # coarse vertical resolution [m]
    "hres_sim_f": 0.01,  # fine horizontal resolution [deg]
    "vres_sim_f": 500,  # fine vertical resolution [m]

    "run_path": "/home/ktait98/GPAT2026/pycontrails_kt/pycontrails/models/gpat/",
    "data_path": "/home/ktait98/GPAT2026/pycontrails_kt/pycontrails/models/gpat/data/", # "/projects/Impact_of_aviation_on_climate
    "job_id": "test",
}

#flight trajectory parameters
fl_params = {
    "mode": "synthetic",
    "file": None,  # flight trajectory file

    "ac_type": "A320",  # aircraft type
    "fl0_speed": 150.0,  # m/s
    "fl0_rocd": 0.1,  # m/s (ignored when target_altitude is set)
    "fl0_heading": 45.0,  # deg
    "fl0_coords0": (0.1, 0.1, 10500),  # lat, lon, alt [deg, deg, m]
    "target_altitude": 11950,  # m, reached at end of t_fl
    "control_waypoints": [
        (0.10, 0.10, 10500),
        (0.25, 0.25, 10850),
        (0.45, 0.45, 11200),
        (0.65, 0.65, 11600),
        (0.95, 0.95, 11950),
        (1.0, 1.0, 12000)
    ],
    "clip_to_domain": True,
    "domain_margin_deg": 0.005,
    "sep_dist": (1000, 0, 0),  # dx, dy, dz [m]
    "n_ac": 2,  # number of aircraft
}

# plume dispersion parameters
pl_params = {
    "depth": 50.0,  # initial plume depth, [m]
    "width": 50.0,  # initial plume width, [m]
    "verbose_outputs": False,  # print verbose outputs
    "shear": 0.01,  # shear [m/s]
    "n_slices": 3,  # number of slices in the plume
    "f_max": 0.99,  # maximum fraction of total emissions in any slice
    "output_pl_slices": True,  # output plume slices to netCDF
    }

# meteorology parameters
met_params = {
    "eastward_wind": 0.0,  # m/s
    "northward_wind": 0.0,  # m/s
    "lagrangian_tendency_of_air_pressure": 0.0,  # m/s
}

# chemistry parameters
chem_params = {
    "run_chem": False,
    "species_emi": ("NO", "CO", "SO2"),
    # "species_pl": ("NO", "CO", "SO2"),
    "species_pl": ("NO", "NO2", "O3", "NO3", "N2O5",
                      "HNO3", "HONO", "HO2NO2","PAN", 
                      "CH3O2NO2","H2O2", "CH3OOH",
                      "CO", "CH4", "HCHO", "SO2", "SA"),
    "species_out": ("O3", "NO2", "NO", "NO3", "N2O5", 
                    "HNO3", "HONO", "HO2", "OH", "H2O2",
                    "CO", "CH4", "CH3O2","HO2NO2", "PAN", "SO2" )
}

sim_params = SimParams(**sim_params)
fl_params = FlParams(**fl_params)
pl_params = PlParams(**pl_params)
met_params = MetParams(**met_params)
chem_params = ChemParams(**chem_params)

gpat = GPAT(sim_params, fl_params, pl_params, met_params, chem_params)

gpat.preprocess_gpat()
gpat.eval()