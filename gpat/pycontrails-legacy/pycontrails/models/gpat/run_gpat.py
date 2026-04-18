import pandas as pd
from pathlib import Path

from pycontrails.models.gpat.gpat import (
    GPAT,
    SimParams,
    FlParams,
    PlParams,
    MetParams,
    ChemParams,
)

sim_params = SimParams(
    t_fl=(
        pd.to_datetime("2022-01-20 13:00:00"),
        pd.Timedelta(seconds=60),
        pd.Timedelta(minutes=60),
    ),
    t_pl=(
        pd.to_datetime("2022-01-20 13:00:00"),
        pd.Timedelta(seconds=60),
        pd.Timedelta(hours=6),
    ),
    t_sim=(
        pd.to_datetime("2022-01-20 12:00:00"),
        pd.Timedelta(seconds=20),
        pd.Timedelta(hours=24),
    ),
    t_out=(
        pd.to_datetime("2022-01-20 12:00:00"),
        pd.Timedelta(seconds=60),
        pd.Timedelta(hours=24),
    ),
    lat_bounds=(47.0, 48.0),
    lon_bounds=(-33.0, -32.0),
    alt_bounds=(12000.0, 13000.0),
    hres_sim_c=0.2,
    vres_sim_c=500.0,
    hres_sim_f=0.1,
    vres_sim_f=250.0,
    run_path=str(Path("~/GPAT2026/pycontrails_kt/pycontrails/models/gpat/").expanduser()) + "/",
    data_path=str(Path("~/GPAT2026/pycontrails_kt/pycontrails/models/gpat/data/").expanduser()) + "/",
    job_id="opensky_test",
)

fl_params = FlParams(
    mode="opensky",
    file=None,                      # use live query + cache
    ac_type="A320",                 # fallback type for PSFlight
    callsigns=None,                 # or e.g. ["BAW123", "DAL45"]
    icao24=None,                    # or explicit ICAO24 filters
    opensky_username=None,          # fill if needed
    opensky_password=None,          # fill if needed
    use_cache=True,
    force_refresh=False,
    cache_dir=str(Path("~/GPAT2026/pycontrails_kt/pycontrails/models/gpat/data/opensky_cache").expanduser()),
    min_points=2,
    n_ac=1,                         # keep >0 so preprocess_gpat enters traj_gen()
)

pl_params = PlParams(
    depth=50.0,
    width=50.0,
    verbose_outputs=False,
    shear=0.05,
    n_slices=3,
    f_max=0.99,
    output_pl_slices=True,
)

met_params = MetParams(
    eastward_wind=0.0,
    northward_wind=0.0,
    lagrangian_tendency_of_air_pressure=0.0,
)

chem_params = ChemParams(
    run_chem=True,
    species_emi=("NO", "CO", "SO2"),
    species_pl=(
        "NO", "NO2", "O3", "NO3", "N2O5",
        "HNO3", "HONO", "HO2NO2", "PAN",
        "CH3O2NO2", "H2O2", "CH3OOH",
        "CO", "CH4", "HCHO", "SO2", "SA"
    ),
    species_out=(
        "O3", "NO2", "NO", "NO3", "N2O5",
        "HNO3", "HONO", "HO2", "OH", "H2O2",
        "CO", "CH4", "CH3O2", "HO2NO2", "PAN", "SO2"
    ),
)

gpat = GPAT(sim_params, fl_params, pl_params, met_params, chem_params)
gpat.preprocess_gpat()
# gpat.eval()