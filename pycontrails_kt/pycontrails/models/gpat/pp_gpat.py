"""GPATPostProcessor class for post-processing GPAT model outputs."""
import numpy as np
import pandas as pd
import xarray as xr
import pyvista as pv
from pyproj import Transformer
import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass

class GPATPostProcessor:
    """Post-process GPAT model outputs."""

    def __init__(self, data_path, criteria):
        
        self.params_dict = {}
        self.fl_ds_dict = {}
        self.pl_ds_dict = {}
        self.boxm_ds_dict = {}
        self.boxm_out_dict = {}
        self.patch_table_dict = {}
        self.pl_out_dict = {}
        
        self.inputs = data_path + "inputs/"
        self.outputs = data_path + "outputs/"
        self.criteria = criteria
        self.jobs_df = self.create_jobs_df()
        self.filtered_df = self.filter_jobs_df()
        self.job_ids = self.filtered_df.index.tolist()
        
        self.analysis = GPATAnalysis(self)
        self.plotting = GPATPlotting(self)
        self.validation = GPATValidation(self)

        self.load_data()

    def create_jobs_df(self):
        """Create a DataFrame containing the job parameters for each simulation run.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the job parameters for each simulation run.
        """

        outputs_dir = self.outputs
        jobs = []

        for job_id in os.listdir(outputs_dir):
            job_dir = os.path.join(outputs_dir, job_id)
            if os.path.isdir(job_dir):
                # Check if the params file exists in the subdirectory
                params_file = os.path.join(self.inputs, job_id, "params.json")
                if os.path.isfile(params_file):
                    params = json.load(open(params_file))

                    # Flatten the dictionary
                    data_dict = params.copy()
                    data_dict["job_id"] = job_id

                    # Check if n_ac > 0 and verify the existence of fl and pl files
                    if data_dict.get("n_ac", 0) > 0:
                        expected_files = [
                            "params.json",
                            "fl_ds.nc",
                            "pl_ds.nc",
                            "boxm_ds.nc",
                        ]
                    else:
                        expected_files = ["params.json", "boxm_ds.nc"]

                    # Check in inputs directory
                    input_dir = os.path.join(self.inputs, job_id)
                    if all(os.path.isfile(os.path.join(input_dir, file)) for file in expected_files):
                        # Also check for outputs
                        if data_dict.get("n_ac", 0) > 0:
                            output_files = ["boxm_out.nc", "patch_table.nc", "pl_out.nc"]
                        else:
                            output_files = ["boxm_out.nc", "patch_table.nc"]
                        output_dir = os.path.join(self.outputs, job_id)
                        if all(os.path.isfile(os.path.join(output_dir, file)) for file in output_files):
                            jobs.append(data_dict)

        jobs_df = pd.DataFrame(jobs)

        return jobs_df.set_index("job_id")

    def filter_jobs_df(self):
        """Filter the jobs DataFrame based on the criteria provided.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the job parameters after applying the filtering criteria.
        """

        jobs_df = self.jobs_df
        criteria = self.criteria

        filtered_df = jobs_df.copy()

        for key, value in criteria.items():
            if isinstance(value, tuple) and len(value) == 2:
                # Range filter
                filtered_df = filtered_df[
                    (filtered_df[key] >= value[0]) & (filtered_df[key] <= value[1])
                ]
            elif key == "job_id":
                if isinstance(value, list):
                    # Combine the list of strings into a single regex pattern
                    pattern = "|".join(value)
                    filtered_df = filtered_df[filtered_df.index.str.contains(pattern)]
                else:
                    filtered_df = filtered_df[filtered_df.index.str.contains(value)]
            else:
                # Exact match filter
                filtered_df = filtered_df[filtered_df[key] == value]

        return filtered_df

    def load_data(self):
        for job_id in self.job_ids:
            self.analysis.load_input_datasets(job_id)
            self.analysis.load_output_datasets(job_id)
            self.analysis.unstack_boxm_out(job_id)

class GPATAnalysis:
    """Analyze GPAT model outputs."""

    def __init__(self, pp_gpat):
        self.pp_gpat = pp_gpat
    
    # Loading data methods
    def load_input_datasets(self, job_id):
        self.pp_gpat.params_dict[job_id] = json.load(open(f"{self.pp_gpat.inputs}{job_id}/params.json"))
        self.pp_gpat.fl_ds_dict[job_id] = xr.open_dataset(f"{self.pp_gpat.inputs}{job_id}/fl_ds.nc")
        self.pp_gpat.pl_ds_dict[job_id] = xr.open_dataset(f"{self.pp_gpat.inputs}{job_id}/pl_ds.nc")
        self.pp_gpat.boxm_ds_dict[job_id] = xr.open_dataset(f"{self.pp_gpat.inputs}{job_id}/boxm_ds.nc")

    def load_output_datasets(self, job_id):
        self.pp_gpat.boxm_out_dict[job_id] = xr.open_dataset(f"{self.pp_gpat.outputs}{job_id}/boxm_out.nc")
        self.pp_gpat.patch_table_dict[job_id] = xr.open_dataset(f"{self.pp_gpat.outputs}{job_id}/patch_table.nc")
        self.pp_gpat.pl_out_dict[job_id] = xr.open_dataset(f"{self.pp_gpat.outputs}{job_id}/pl_out.nc")

    def unstack_boxm_out(self, job_id):
        """Unstack the box model coarse output dataset."""
        boxm_out = self.pp_gpat.boxm_out_dict[job_id]
        
        print("Chunking the dataset")
        # Convert the dataset to a Dask dataset
        boxm_out = boxm_out.chunk({"cell": 100})  # Adjust chunk size based on your mem

        print("Set coords")
        # Convert 'level', 'lat', and 'lon' to coordinates
        boxm_out_unstacked = boxm_out.set_coords(["level_c", "longitude_c", "latitude_c"])

        print("Set index")
        # Create a multi-index for the 'cell' dimension
        boxm_out_unstacked = boxm_out_unstacked.set_index(
            cell=["level_c", "longitude_c", "latitude_c"]
        )

        print("Unstack the dataset")
        # Unstack the dataset
        boxm_out_unstacked = boxm_out_unstacked.unstack("cell")

        print("Compute the result")
        # # Compute the result to trigger the lazy evaluation
        # boxm_out_unstacked = boxm_out_unstacked.compute()
        
        self.pp_gpat.boxm_out_dict[job_id] = boxm_out_unstacked

class GPATPlotting:
    """Plot GPAT model outputs."""

    def __init__(self, pp_gpat):
        self.pp_gpat = pp_gpat

    def plot_plumes_3d_mpl(self, job_id, time_idx=184):
        # %matplotlib widget
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        pl_ds = self.pp_gpat.pl_ds_dict[job_id][["time_idx", "longitude", "longitude_m", "latitude", "latitude_m", "altitude", "width", "depth", "heading", "flight_id", "seg_id"]].to_dataframe().reset_index().dropna()
        pl_out = self.pp_gpat.pl_out_dict[job_id][["time_idx", "ellipses_m", "slice_polys_m"]].to_dataframe().reset_index().dropna()


        # Select the ellipse and all slices for a given seg_id and time
        ellipse = pl_out.ellipses_m.isel(seg_id=0, time=81)  # shape: (pt_id, coord)
        slice_polys = pl_out.slice_polys_m.isel(seg_id=0, time=81)  # shape: (slice_id, corner_id, coord)

        # Extract ellipse coordinates
        lon = ellipse.sel(coord="lon_m").values
        lat = ellipse.sel(coord="lat_m").values
        alt = ellipse.sel(coord="alt_m").values

        # Optionally, close the ellipse by repeating the first point
        lon = np.append(lon, lon[0])
        lat = np.append(lat, lat[0])
        alt = np.append(alt, alt[0])

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the ellipse
        ax.plot(lon, lat, alt, marker='o', label='Ellipse')

        # Plot each slice polygon
        for slice_id in range(slice_polys.sizes['slice_id']):
            poly = slice_polys.isel(slice_id=slice_id)
            poly_lon = poly.sel(coord="lon_m").values
            poly_lat = poly.sel(coord="lat_m").values
            poly_alt = poly.sel(coord="alt_m").values
            # Optionally, close the polygon
            poly_lon = np.append(poly_lon, poly_lon[0])
            poly_lat = np.append(poly_lat, poly_lat[0])
            poly_alt = np.append(poly_alt, poly_alt[0])
            ax.plot(poly_lon, poly_lat, poly_alt, marker='x', label=f'Slice {slice_id}')

        ax.set_xlabel("Longitude (m)")
        ax.set_ylabel("Latitude (m)")
        ax.set_zlabel("Altitude (m)")
        ax.set_title("Plume Ellipse and Slice Polygons (3D)")
        ax.legend()

        # Make axes equal scale
        max_range = np.array([lon.max() - lon.min(), 
                            lat.max() - lat.min(), 
                            alt.max() - alt.min()]).max() / 2.0

        mid_x = (lon.max() + lon.min()) * 0.5
        mid_y = (lat.max() + lat.min()) * 0.5
        mid_z = (alt.max() + alt.min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        plt.show()

    def plot_plumes_3d_pv(self, job_id, time_idx=184):
        
        pl_ds = self.pp_gpat.pl_ds_dict[job_id]
        pl_out = self.pp_gpat.pl_out_dict[job_id]
        
        # --- Plotting with PyVista (all in meters) ---
        flight_ids_t = (
            pl_ds
            .where(pl_ds["time_idx"] == time_idx, drop=True)["flight_id"]
            .values
        )
        flight_ids_t = np.unique(flight_ids_t)

        plotter = pv.Plotter()
        plotter.add_axes()
        plotter.show_grid()
        # Disabled for performance: plotter.enable_anti_aliasing()
        
        for fid in flight_ids_t:
            centres_m = []
            ellipses_m = []
            slice_polys_m = []
            seg_ids_f = (
                pl_ds
                .where((pl_ds["time_idx"] == time_idx) & (pl_ds["flight_id"] == fid), drop=True)["seg_id"]
                .values
            )
            seg_ids_f = np.unique(seg_ids_f)
            
            # For each segment in this flight and timestep, get the plume parameters and corresponding output shapes for plotting
            for seg_id_py, seg_id in enumerate(seg_ids_f):
                mask_seg = (
                    (pl_ds["time_idx"] == time_idx)
                    & (pl_ds["seg_id"] == seg_id)
                    & (pl_ds["flight_id"] == fid)
                )
                df_seg = pl_ds.where(mask_seg, drop=True)

                mask_out_seg = (
                    (pl_out["time_idx"] == time_idx)
                    & (pl_out["seg_id"] == seg_id)
                    & (pl_out["flight_id"] == fid)
                )
                df_out_seg = pl_out.where(mask_out_seg, drop=True)
                
                if df_seg.sizes.get("seg_id", 0) == 0 or df_out_seg.sizes.get("seg_id", 0) == 0:
                    continue
                row = df_seg.isel(seg_id=0).squeeze(drop=True)
                row_out = df_out_seg.isel(seg_id=0).squeeze(drop=True)
                
                # Only append valid ellipses (not None and have data)
                if row_out["ellipses_m"] is not None and hasattr(row_out["ellipses_m"], '__len__') and len(row_out["ellipses_m"]) > 0:
                    # Extract center coordinates (meters)
                    lon_m = row["longitude_m"].item()
                    lat_m = row["latitude_m"].item()
                    alt_m = row["altitude"].item()
                    
                    # Only append centers if they're finite
                    if np.isfinite([lon_m, lat_m, alt_m]).all():
                        centres_m.append([lon_m, lat_m, alt_m])
                        
                        ellipses_m.append(row_out["ellipses_m"].values)  # shape: (n_points, 3)
                        slice_polys_m.append(row_out["slice_polys_m"].values)  # shape: (n_slices, 4, 3)

            centres_m = np.array(centres_m)
            ellipses_m = np.array(ellipses_m)
            slice_polys_m = np.array(slice_polys_m)

            if ellipses_m.ndim == 3 and ellipses_m.shape[1] == 3:
                ellipses_m = np.transpose(ellipses_m, (0, 2, 1))

            if slice_polys_m.ndim == 4 and slice_polys_m.shape[2] == 3:
                slice_polys_m = np.transpose(slice_polys_m, (0, 1, 3, 2))

            for i in range(len(ellipses_m) - 1):
                e1 = ellipses_m[i]
                e2 = ellipses_m[i+1]
                
                # Filter out NaN/inf values
                if not (np.all(np.isfinite(e1)) and np.all(np.isfinite(e2))):
                    continue
                    
                n = e1.shape[0]
                points = np.vstack([e1, e2])
                faces = []
                for j in range(n):
                    j2 = (j + 1) % n
                    faces.extend([4, j, j2, n + j2, n + j])
                faces = np.array(faces)
                mesh = pv.PolyData(points, faces)
                plotter.add_mesh(mesh, color="red", opacity=0.5)


            if len(centres_m) > 1:
                trajectory = np.array(centres_m)
                # Filter out any remaining invalid points
                valid_mask = np.all(np.isfinite(trajectory), axis=1)
                trajectory = trajectory[valid_mask]
                
                if trajectory.shape[0] > 1:
                    plotter.add_lines(trajectory, color="blue", width=3, connected=True)
            
            # Create meshes connecting slice polygons between consecutive segments
            if slice_polys_m.ndim == 4 and slice_polys_m.shape[0] > 1:
                num_segments = slice_polys_m.shape[0]
                num_slices = slice_polys_m.shape[1]
                
                # Verify we have matching number of segments between ellipses and slices
                if len(ellipses_m) == num_segments:
                    for slice_idx in range(num_slices):
                        for seg_idx in range(num_segments - 1):
                            poly1 = slice_polys_m[seg_idx, slice_idx]
                            poly2 = slice_polys_m[seg_idx + 1, slice_idx]
                            
                            # Check if both polygons are valid and have proper shape
                            if (poly1.shape[0] >= 3 and poly2.shape[0] >= 3 and 
                                np.all(np.isfinite(poly1)) and np.all(np.isfinite(poly2))):
                                n_corners = poly1.shape[0]
                                points = np.vstack([poly1, poly2])
                                faces = []
                                for corner_idx in range(n_corners):
                                    next_corner = (corner_idx + 1) % n_corners
                                    faces.extend([4, corner_idx, next_corner, n_corners + next_corner, n_corners + corner_idx])
                                faces = np.array(faces)
                                mesh = pv.PolyData(points, faces)
                                plotter.add_mesh(mesh, color="green", opacity=0.3)

        # Set equal aspect ratio for better visualization
        plotter.enable_parallel_projection()
        
        # Add camera position and rendering enhancements
        plotter.camera_position = 'iso'
        plotter.add_text(
            f"Flight Plumes at time_idx={time_idx} (coordinates in meters)\\nRed: Ellipse meshes | Blue: Trajectories | Green: Slice meshes",
            position='upper_left',
            font_size=10,
            color='black'
        )
        
        # Show with interactive controls
        # Use full_screen=False and auto_close=False for maximum interactivity
        plotter.show(
            interactive=True,
            auto_close=False,
            full_screen=False
        )

class GPATValidation:
    """Validate GPAT model outputs."""

    def __init__(self, pp_gpat):
        self.pp_gpat = pp_gpat

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


# Helper functions for PP GPAT
def lonlat_to_m(lon, lat, global_ref_lon, global_ref_lat):
    transformer = Transformer.from_crs(
        "epsg:4326",
        f"+proj=tmerc +lat_0={global_ref_lat} +lon_0={global_ref_lon} +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs",
        always_xy=True,
    )

    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    valid = np.isfinite(lon_arr) & np.isfinite(lat_arr)

    lon_m = np.full_like(lon_arr, np.nan, dtype=float)
    lat_m = np.full_like(lat_arr, np.nan, dtype=float)

    if np.any(valid):
        lon_m_valid, lat_m_valid = transformer.transform(lon_arr[valid], lat_arr[valid])
        lon_m[valid] = lon_m_valid
        lat_m[valid] = lat_m_valid

    return lon_m, lat_m

def m_to_lonlat(lon_m, lat_m, global_ref_lon, global_ref_lat):
    transformer = Transformer.from_crs(
        f"+proj=tmerc +lat_0={global_ref_lat} +lon_0={global_ref_lon} +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs",
        "epsg:4326",
        always_xy=True,
    )

    lon_m_arr = np.asarray(lon_m, dtype=float)
    lat_m_arr = np.asarray(lat_m, dtype=float)
    valid = np.isfinite(lon_m_arr) & np.isfinite(lat_m_arr)

    lon = np.full_like(lon_m_arr, np.nan, dtype=float)
    lat = np.full_like(lat_m_arr, np.nan, dtype=float)

    if np.any(valid):
        lon_valid, lat_valid = transformer.transform(lon_m_arr[valid], lat_m_arr[valid])
        lon[valid] = lon_valid
        lat[valid] = lat_valid

    return lon, lat


