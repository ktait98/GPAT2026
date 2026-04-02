# GPAT: Gridded Plume Analysis Tool
GPAT (Gridded Plume Analysis Tool) is a Python framework for simulating aircraft trajectories, estimating aircraft performance, fuel burn, and emissions, and modeling the dispersion and chemistry of aircraft exhaust plumes. It aggregates plume data to an Eulerian grid for photochemical processing.

## Features
- **Trajectory Simulation:** Supports both direct (from file) and synthetic flight trajectory generation.
- **Meteorology and Chemistry:** Loads and interpolates meteorological and background chemistry data.
- **Aircraft Performance:** Integrates with PSFlight for aircraft performance modeling.
- **Emissions Modeling:** Uses Pycontrails Emissions Model for estimating emissions.
- **Plume Dispersion:** Simulates plume dispersion and advection using Gaussian models and dry advection.
- **NetCDF I/O:** Prepares input and output datasets in NetCDF format for compatibility with BOXM.F90. This program handles plume projection to grid, chemistry integration, backprojection to the plume See BOXM_README.md for more info.

## Main Classes and Structure
**SimParams, FlParams, PlParams, MetParams, ChemParams** (Dataclasses)
----------------------------------------------------------------------
Purpose: Hold all simulation, flight, plume, meteorology, and chemistry parameters in a structured, type-safe way.
Usage: Passed to the main GPAT class for configuration.
- SimParams: time grids, spatial bounds, resolution, paths, job ID.
- FlParams: flight mode (direct/synthetic), aircraft type, initial position, number of aircraft.
- PlParams: plume initial width/depth, number of slices, wind shear, output options.
- MetParams: wind components, air pressure tendency.
- ChemParams: which species to track for emissions, plume, and output, chemistry toggles.

**GPAT** (Main Model Class)
---------------------------
- **Purpose:** 
  - Orchestrates the entire workflow to populate input and output files: setup, preprocessing and running.

- **Key attributes:**
  - Stores all parameter dataclasses.
  - Sets up paths, time grids, and coordinates.
  - Instantiates helper classes: GPATSetup, GPATRun.

- **Key methods:**
  - **init():**
    - Generates coarse grid vectors (lats, lons, alts and levels).
    - Converts lon/lat axes from degrees to metres (lons_m, lats_m) for adherence with plume dimensions.
    - Generates time vectors (times_fl, times_pl, times_sim, times_out).
    - Sets run and data paths and job ID for simulation.
    - Grabs species numbers for species_emi, species_pl, species_out, so that a common index can be used to identify species in BOXM.f90.
    - Validate species and time hierarchies.
    - Setup input and output directories.
    - Resets file paths for job ID.
    - Sets model parameters and subclasses to GPAT class instance.

  - **preprocess_gpat():**
    - Generates flight trajectories, meteorology and background chemistry data for simulation.
    - Calculates aircraft performance using Poll-Schumann Model [ref] and estimates emission indices and masses for each flight using Pycontrails.
    - Simulates plume advection and dispersion using Dry Advection module in Pycontrails [ref].
    - Generates input files and initialises zero-filled output files to be manipulated in BOXM.f90 and updated once run is complete.
    
  - **eval():** 
    - Runs BOXM.f90, the FORTRAN projection/backprojection and photochemical box model.

**GPATSetup** (Setup Class)
---------------------------
- **Purpose:** 
  - Handles all input generation, parameter validation, and NetCDF dataset creation.

- **traj_gen():** 
  - Generates flight trajectories (from file or synthetic) with flight and simulation params as input.
  - Mode = direct: This reads csv data (flight_id, time, latitude, longitude, altitude) and turns each flight into a Pycontrails Flight object.
  - Mode = synthetic: Creates formation flight trajectories according to leader aircraft initial state (coords of first waypoint, speed, heading angle, rate of climb/descent), number of aircraft and separation minima.
  - Inputs: self, FlParams, SimParams.
  - Outputs: FlightDataset (1st gen - trajectory points only).

- **gen_met():** 
  - Generates meteorology data from MetParams input config, as well as air temperature and water vapour concs from files in data/inputs/glob.
  - Interpolates meteorology data from files to model resolution and domain bounds.
  - Calculates additional variables - air pressure, specific humidity, relative humidity, M, O2, N2, Solar Zenith Angle.
  - Inputs: self, meteorology params
  - Outputs: Met dataset

- **gen_bg_chem():** 
  - Loads and interpolates background chemistry from species.nc file in data/inputs/glob.
  - Converts to ppb from mixing ratio.
  - Only sets species that are initialised at background.
  - Inputs: self, species NetCDF file.
  - Outputs: Background chemistry data to assign to BOXM dataset (Y_bg_c).

- **ac_perf():** 
  - Downselection to interpolate temperature and specific humidity to flight data.
  - Calculates true airspeed for each flight waypoint.
  - Calculates aircraft performance using Poll-Schumann methodology, assigning aircraft mass, fuel flow, fuel burn, etc. 
  - Inputs: self, meteorology dataset, flight dataset.
  - Outputs: Flight dataset (2nd gen - aircraft performance added: aircraft mass, fuel flow, fuel burn etc.).

- **emissions():**
  - Estimates emissions for each flight segment using aircraft performance data and emission indices.
  - Inputs: self, flight dataset, emission parameters.
  - Outputs: Emissions dataset (per-flight, per-segment).

- **sim_plumes():**
  - Simulates plume dispersion and advection using meteorological data and plume parameters.
  - Inputs: self, emissions dataset, meteorology dataset, plume parameters.
  - Outputs: Plume dataset (spatially and temporally resolved).

- **gen_inputs():**
  - Writes all required input NetCDF files for the box model, including flight, plume, and background chemistry datasets.
  - Inputs: self, processed datasets.
  - Outputs: NetCDF input files (fl_ds.nc, pl_ds.nc, boxm_ds.nc).

- **gen_outputs():**
  - Initializes and writes all output NetCDF templates for the simulation, ensuring correct structure for downstream processing.
  - Inputs: self, simulation parameters.
  - Outputs: Zero-filled NetCDF output files (pl_out.nc, boxm_out.nc, patch_table.nc).

**GPATRun** (Run Class)
---------------------------
- **Purpose:**
  - Runs GPAT fortran executable, BOXM.F90.

# Input Files
- **fl_ds:**
  - Dimensions: ['seg_id']
  - Coordinates: ['waypoint', 'flight_id', 'seg_id']
  - Variables: ['longitude', 'latitude', 'level', 'altitude', 'time', 'true_airspeed', 'aircraft_mass', 'engine_efficiency', 'fuel_flow', 'fuel_burn', 'thrust', 'rocd', 'fuel_flow_per_engine', 'thrust_setting', 'time_rel_s', 'time_idx']

- **pl_ds:**
  - Dimensions: ['seg_id', 'time', 'species_emi']
  - Coordinates: ['flight_id', 'waypoint', 'seg_id', 'time', 'species_emi_num', 'species_emi', 'active_seg_flag', 'time_rel_s', 'time_idx']
  - Variables: ['age', 'longitude', 'latitude', 'level', 'width', 'depth', 'heading', 'sigma_yy', 'sigma_yz', 'sigma_zz', 'longitude_m', 'latitude_m', 'altitude', 'emi_pl_mass', 'age_s']

- **boxm_ds:**
  - Dimensions: ['time', 'cell', 'species_boxm']
  - Coordinates: ['time', 'air_pressure', 'altitude_c', 'species_boxm', 'time_rel_s', 'time_idx', 'species_boxm_num', 'level_c', 'longitude_c', 'latitude_c', 'longitude_c_m', 'latitude_c_m', 'mol_mass_c']
  - Variables: ['air_temperature', 'H2O', 'M', 'O2', 'N2', 'sza', 'Y_bg_c']

# Output Files
- **boxm_out:**
  - Dimensions: ['level_c', 'longitude_c', 'latitude_c', 'time', 'species_out']
  - Coordinates: ['level_c', 'longitude_c', 'latitude_c', 'time', 'species_out', 'time_rel_s', 'time_idx', 'altitude_c', 'species_out_num']
  - Variables: ['Y_bg_c', 'Y_del_c', 'active_flag']

- **pl_out:**
  - Dimensions: ['seg_id', 'time', 'species_pl', 'slice_id', 'pt_id', 'coord', 'corner_id', 'face']
  - Coordinates: ['flight_id', 'waypoint', 'seg_id', 'time', 'species_pl', 'slice_id', 'pt_id', 'coord', 'corner_id', 'face', 'species_pl_num', 'time_rel_s', 'time_idx']
  - Variables: ['y_half', 'z_half', 'm_frac', 'w_slice', 'ellipses_m', 'slice_polys_m', 'slice_boxes_m', 'pl_mass']

- **patch_table:**
  - Dimensions: ['row', 'species_out']
  - Coordinates: ['row', 'species_out', 'species_out_num', 'time', 'time_rel_s', 'time_idx', 'row_cell_c', 'row_cell_f', 'latitude_f', 'longitude_f', 'altitude_f', 'level_f']
  - Variables: ['Y_del_f']

## Parameterization
Parameters are managed via Python dataclasses. For automated or batch runs, use config files (YAML/JSON) and load them into the dataclasses.

## Extending and Customizing
- Add new chemistry or microphysics models by extending the relevant classes.
- Modify parameter dataclasses to add new simulation options.
- Use the analysis tools for custom validation and visualization.

## Requirements
- Python 3.8+
- numpy, pandas, xarray, pyproj, matplotlib, dask, netCDF4, and other dependencies from pycontrails
- FORTRAN box model and contrail model executables (for full workflow)