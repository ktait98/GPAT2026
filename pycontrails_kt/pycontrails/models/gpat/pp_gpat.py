"""GPATPostProcessor class for post-processing GPAT model outputs."""
import numpy as np
import pandas as pd
import xarray as xr
import pyvista as pv
import matplotlib.pyplot as plt
from pyproj import Transformer
import ast
import json
import os
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
import ipywidgets as widgets
from IPython.display import display, clear_output

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
                    params["job_id"] = job_id
                    self.params_dict[job_id] = params

                    # Flatten the dictionary
                    fl_params_str = params.get("fl_params", "")
                    match = re.search(r"n_ac\s*=\s*([0-9]+)", fl_params_str)
                    n_ac = int(match.group(1)) if match else 0
                    print(n_ac)
                    # Check if n_ac > 0 and verify the existence of fl and pl files
                    if n_ac > 0:
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
                        if n_ac > 0:
                            output_files = ["boxm_out.nc", "patch_table.nc", "pl_out.nc"]
                        else:
                            output_files = ["boxm_out.nc"]
                        output_dir = os.path.join(self.outputs, job_id)
                        if all(os.path.isfile(os.path.join(output_dir, file)) for file in output_files):
                            params["job_id"] = job_id
                            self.params_dict[job_id] = params
                            jobs.append(params)

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
        fl_params_str = self.pp_gpat.params_dict[job_id].get("fl_params", "")
        match = re.search(r"n_ac\s*=\s*([0-9]+)", fl_params_str)
        n_ac = int(match.group(1)) if match else 0
        self.pp_gpat.params_dict[job_id] = json.load(open(f"{self.pp_gpat.inputs}{job_id}/params.json"))
        if n_ac > 0:
            self.pp_gpat.fl_ds_dict[job_id] = xr.open_dataset(f"{self.pp_gpat.inputs}{job_id}/fl_ds.nc")
            self.pp_gpat.pl_ds_dict[job_id] = xr.open_dataset(f"{self.pp_gpat.inputs}{job_id}/pl_ds.nc")
        self.pp_gpat.boxm_ds_dict[job_id] = xr.open_dataset(f"{self.pp_gpat.inputs}{job_id}/boxm_ds.nc")

    def load_output_datasets(self, job_id):
        self.pp_gpat.boxm_out_dict[job_id] = xr.open_dataset(f"{self.pp_gpat.outputs}{job_id}/boxm_out.nc")
        fl_params_str = self.pp_gpat.params_dict[job_id].get("fl_params", "")
        match = re.search(r"n_ac\s*=\s*([0-9]+)", fl_params_str)
        n_ac = int(match.group(1)) if match else 0
        if n_ac > 0:
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

        print("Normalize dimension order")
        boxm_out_unstacked = boxm_out_unstacked.transpose(
            "time", "level_c", "longitude_c", "latitude_c", "species_out", missing_dims="ignore"
        )

        print("Collapse broadcast altitude coordinate")
        if "altitude_c" in boxm_out_unstacked.coords and set(boxm_out_unstacked["altitude_c"].dims) == {
            "level_c", "longitude_c", "latitude_c"
        }:
            boxm_out_unstacked = boxm_out_unstacked.assign_coords(
                altitude_c=boxm_out_unstacked["altitude_c"].isel(longitude_c=0, latitude_c=0, drop=True)
            )

        print("Compute the result")
        # # Compute the result to trigger the lazy evaluation
        # boxm_out_unstacked = boxm_out_unstacked.compute()
        
        self.pp_gpat.boxm_out_dict[job_id] = boxm_out_unstacked

    def print_input_ds(self, job_id, ds_type):
        if ds_type == "fl":
            print(self.pp_gpat.fl_ds_dict[job_id])
        elif ds_type == "pl":
            print(self.pp_gpat.pl_ds_dict[job_id])
        elif ds_type == "boxm":
            print(self.pp_gpat.boxm_ds_dict[job_id])
        else:
            print("Invalid dataset type. Please choose from 'fl', 'pl', or 'boxm'.")

    def print_output_ds(self, job_id, ds_type):
        if ds_type == "boxm_out":
            print(self.pp_gpat.boxm_out_dict[job_id])
        elif ds_type == "patch_table":
            print(self.pp_gpat.patch_table_dict[job_id])
        elif ds_type == "pl_out":
            print(self.pp_gpat.pl_out_dict[job_id])
        else:
            print("Invalid dataset type. Please choose from 'boxm_out', 'patch_table', or 'pl_out'.")

    def print_output_da(self, job_id, ds_type, da_name):
        ds_dict_map = {
            "boxm_out": self.pp_gpat.boxm_out_dict,
            "patch_table": self.pp_gpat.patch_table_dict,
            "pl_out": self.pp_gpat.pl_out_dict,
        }

        if ds_type not in ds_dict_map:
            print("Invalid dataset type. Please choose from 'boxm_out', 'patch_table', or 'pl_out'.")
            return

        ds = ds_dict_map[ds_type].get(job_id)
        if ds is None:
            print(f"No loaded dataset '{ds_type}' for job_id {job_id}.")
            return

        if da_name in ds:
            print(ds[da_name])
            return

        print(f"DataArray '{da_name}' not found in {ds_type} for job_id {job_id}.")

class GPATPlotting:
    """Plot GPAT model outputs."""

    def __init__(self, pp_gpat):
        self.pp_gpat = pp_gpat
        self.patch_plot_min_value = None
        self.patch_plot_max_log10_span = 3.0

    # Time series line plotting
    def plot_time_series(self, job_id, species="NO", level=None, lat=None, lon=None, data_var="Y_bg_c", avg_over_domain=False):
        boxm_out = self.pp_gpat.boxm_out_dict[job_id]
        da = boxm_out[data_var]
        # Sort relevant coordinates to ensure monotonicity for .sel(method="nearest")
        for coord in ["species_out", "level_c", "latitude_c", "longitude_c"]:
            if coord in da.coords:
                da = da.sortby(coord)
        # Select species_out directly (no method), others with method="nearest"
        da = da.sel(species_out=species)
        da = da.sel(level_c=level, latitude_c=lat, longitude_c=lon, method="nearest")
        fig, ax = plt.subplots()
        da.plot.line(x="time", ax=ax)
        ax.set_facecolor('white')
        # Reduce x-tick density
        import matplotlib.ticker as ticker
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
        # Rotate x-tick labels for readability
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
        ax.set_title(f"Time series of {data_var} for {species} at level={level}, lat={lat}, lon={lon}")
        ax.set_xlabel("Time")
        ax.set_ylabel(f"{data_var} (mol/m^3)")
        ax.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_boxm_background_slider(self, job_id, species="NO", level=None, time_indices=None):                
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        boxm_out = self.pp_gpat.boxm_out_dict[job_id]
        all_time_indices = np.asarray(boxm_out["time_idx"].values, dtype=int)
        if time_indices is None:
            time_indices = all_time_indices
        else:
            time_indices = np.asarray(time_indices, dtype=int)

        # Print available time indices and levels
        available_time_indices = np.asarray(boxm_out["time_idx"].values, dtype=int)
        print(f"Available time indices: {available_time_indices}")
        if "level_c" in boxm_out.dims:
            available_levels = np.asarray(boxm_out["level_c"].values)
            print(f"Available levels in level_c: {available_levels}")
        elif "altitude_c" in boxm_out.dims:
            available_levels = np.asarray(boxm_out["altitude_c"].values)
            print(f"Available altitudes in altitude_c: {available_levels}")
        else:
            available_levels = None

        # Restrict slider to valid time indices
        if time_indices is None:
            time_indices = available_time_indices
        else:
            time_indices = np.asarray(time_indices, dtype=int)
            time_indices = np.intersect1d(time_indices, available_time_indices)
            print(f"Using intersected time indices: {time_indices}")

        # Restrict level to valid value
        if level is not None and available_levels is not None:
            # Use np.isclose to find the closest available level
            idx = np.argmin(np.abs(available_levels - float(level)))
            level = available_levels[idx]
            print(f"Using closest available level: {level}")

        vmin = float(boxm_out["Y_bg_c"].sel(species_out=species, level_c=level).min().compute().item())
        vmax = float(boxm_out["Y_bg_c"].sel(species_out=species, level_c=level).max().compute().item())

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        slider_ax = fig.add_axes([0.15, 0.05, 0.7, 0.05])
        slider = Slider(slider_ax, 'Time Index', int(time_indices.min()), int(time_indices.max()), valinit=int(time_indices[0]), valstep=1)

        def plot_for_time(t_idx):
            idx = np.argmin(np.abs(time_indices - t_idx))
            t_idx_actual = time_indices[idx]
            print(f"Requested t_idx={t_idx}, using closest available t_idx={t_idx_actual}")
            _t_sel = np.flatnonzero(all_time_indices == t_idx_actual)
            if _t_sel.size == 0:
                print(f"No matching time index for t_idx_actual={t_idx_actual}")
                return None
            _b_t = boxm_out.isel(time=_t_sel).squeeze("time")
            # Slice by altitude/level if provided
            if level is not None:
                if "level_c" in _b_t.dims:
                    level_arr = np.asarray(_b_t["level_c"].values)
                    idx = np.argmin(np.abs(level_arr - float(level)))
                    level_actual = level_arr[idx]
                    print(f"Requested level={level}, using closest available level={level_actual}")
                    _b_t = _b_t.isel(level_c=idx)
                elif "altitude_c" in _b_t.dims:
                    alt_arr = np.asarray(_b_t["altitude_c"].values)
                    idx = np.argmin(np.abs(alt_arr - float(level)))
                    alt_actual = alt_arr[idx]
                    print(f"Requested altitude={level}, using closest available altitude={alt_actual}")
                    _b_t = _b_t.isel(altitude_c=idx)
            vals = np.asarray(_b_t["Y_bg_c"].sel(species_out=species).values)
            lat = np.asarray(_b_t["latitude_c"].values)
            lon = np.asarray(_b_t["longitude_c"].values)
            if vals.size > 0:
                print(f"Frame max: {np.nanmax(vals)}, min: {np.nanmin(vals)}")
            
            if vals.size == 0 or np.all(~np.isfinite(vals)):
                ax.set_title("No valid data to plot")
                print("No valid data to plot for this slice.")
                return None
            if vals.ndim == 1 and lat.ndim == 1 and lon.ndim == 1:
                print("Plotting as scatter for diagnostic.")
                sc = ax.scatter(lon, lat, c=vals, cmap="viridis", vmin=vmin, vmax=vmax)
                return sc
            elif vals.ndim == 2 and lat.ndim == 1 and lon.ndim == 1:
                print("Plotting as pcolormesh.")
                mesh = ax.pcolormesh(lon, lat, vals, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
                return mesh
            else:
                ax.set_title("Unsupported array shape for plotting")
                print("Unsupported array shape for plotting.")
                return None

        mesh = plot_for_time(int(slider.val))
        cbar = fig.colorbar(mesh, ax=ax)
        ax.set_title(f"Y_bg_c {species}, time_idx={slider.val}, level={level}")

        def slider_update(val):
            # Only update the data array, not axes or colorbar
            mesh = plot_for_time(int(val))
            if mesh is not None:
                # Update mesh data
                if hasattr(mesh, 'set_array'):
                    mesh.set_array(mesh.get_array())
                # Update title
                ax.set_title(f"Y_bg_c {species}, time_idx={val}, level={level}")
            fig.canvas.draw_idle()

        slider.on_changed(slider_update)
        plt.show()

    def plot_boxm_patch_slider(
        self,
        job_id,
        species="NO",
        mode="total",                    # "background", "coarse_delta", "patch_delta", "total", "total_with_patch"
        level=None,                      # required for vertical_mode="single"
        time_indices=None,
        vertical_mode="single",          # "single", "column", "topk"
        top_k=3,
        level_tol=None,
        use_nearest_level=True,
        dynamic_colorbar=False,
        cmap="viridis",
    ):
        """
        Plot GPAT chemistry fields on a 2D lat-lon map with a time slider.

        Data sources
        ------------
        boxm_out   = self.pp_gpat.boxm_out_dict[job_id]
        pl_out     = self.pp_gpat.pl_out_dict[job_id]      # loaded but not used here yet
        patch_table = self.pp_gpat.patch_table_dict[job_id]

        Modes
        -----
        background       : Y_bg_c
        coarse_delta     : Y_del_c
        patch_delta      : sparse patch_table Y_del_f accumulated to coarse map
        total            : Y_bg_c + Y_del_c
        total_with_patch : Y_bg_c + patch-accumulated Y_del_f

        Vertical modes
        --------------
        single : one level
        column : sum over all levels
        topk   : sum top-k levels per lat-lon column

        Notes
        -----
        - boxm_out after unstack_boxm_out has dims:
            (time, level_c, longitude_c, latitude_c, species_out)
        - For plotting with pcolormesh(lon, lat, Z), Z must be shaped (lat, lon),
        so fields are transposed before plotting.
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        allowed_modes = {"background", "coarse_delta", "patch_delta", "total", "total_with_patch"}
        if mode not in allowed_modes:
            raise ValueError(f"mode must be one of {allowed_modes}, got {mode!r}")

        allowed_vertical_modes = {"single", "column", "topk"}
        if vertical_mode not in allowed_vertical_modes:
            raise ValueError(f"vertical_mode must be one of {allowed_vertical_modes}, got {vertical_mode!r}")

        if vertical_mode == "single" and level is None:
            raise ValueError("level must be provided when vertical_mode='single'")

        if vertical_mode == "topk" and int(top_k) < 1:
            raise ValueError("top_k must be >= 1 when vertical_mode='topk'")

        boxm_out = self.pp_gpat.boxm_out_dict[job_id]
        patch_table = self.pp_gpat.patch_table_dict[job_id]
        pl_out = self.pp_gpat.pl_out_dict.get(job_id, None)  # not used yet, but kept for consistency

        all_time_indices = np.asarray(boxm_out["time_idx"].values, dtype=int)
        if time_indices is None:
            time_indices = all_time_indices
        else:
            time_indices = np.intersect1d(np.asarray(time_indices, dtype=int), all_time_indices)

        if time_indices.size == 0:
            raise ValueError("No valid time_indices remain after intersecting with boxm_out time_idx.")

        print(f"Using time indices: {time_indices}")

        lon_bg = np.asarray(boxm_out["longitude_c"].values, dtype=float)
        lat_bg = np.asarray(boxm_out["latitude_c"].values, dtype=float)
        level_bg_all = np.asarray(boxm_out["level_c"].values, dtype=float)

        if "level_f" in patch_table:
            patch_levels_all = np.unique(np.asarray(patch_table["level_f"].values, dtype=float))
        else:
            patch_levels_all = np.array([], dtype=float)

        if vertical_mode == "single":
            bg_level_req = float(level_bg_all[np.argmin(np.abs(level_bg_all - float(level)))])
            print(f"Requested level={level}, using background level={bg_level_req}")

            if patch_levels_all.size > 0:
                patch_level_req = float(patch_levels_all[np.argmin(np.abs(patch_levels_all - float(level)))])
                print(f"Requested level={level}, using patch level={patch_level_req}")
            else:
                patch_level_req = None
        else:
            bg_level_req = None
            patch_level_req = None

        def _snap_time_index(t_idx_req):
            idx = int(np.argmin(np.abs(time_indices - int(t_idx_req))))
            return int(time_indices[idx])

        def _get_boxm_time_slice(t_idx_actual):
            sel = np.flatnonzero(all_time_indices == t_idx_actual)
            if sel.size == 0:
                raise ValueError(f"No matching time_idx={t_idx_actual}")
            return boxm_out.isel(time=sel).squeeze("time")

        def _get_2d_from_coarse_da(da3):
            """
            da3 is expected after species selection, with dims either:
            - (level_c, longitude_c, latitude_c)
            - (longitude_c, latitude_c)
            Returns array shaped (lat, lon) for plotting.
            """
            vals = np.asarray(da3.values, dtype=float)

            if vertical_mode == "single":
                level_arr = np.asarray(da3["level_c"].values, dtype=float)
                idx = int(np.argmin(np.abs(level_arr - bg_level_req)))
                vals2 = vals[idx, :, :]   # (lon, lat)

            elif vertical_mode == "column":
                if vals.ndim == 3:
                    vals2 = np.nansum(vals, axis=0)   # (lon, lat)
                else:
                    vals2 = vals

            elif vertical_mode == "topk":
                if vals.ndim == 3:
                    nz = vals.shape[0]
                    k = min(int(top_k), nz)
                    work = np.where(np.isfinite(vals), vals, -np.inf)
                    part = np.partition(work, kth=nz - k, axis=0)
                    topk_vals = part[-k:, :, :]
                    topk_vals = np.where(np.isfinite(topk_vals), topk_vals, 0.0)
                    vals2 = np.sum(topk_vals, axis=0)  # (lon, lat)
                else:
                    vals2 = vals
            else:
                raise ValueError(vertical_mode)

            # convert (lon, lat) -> (lat, lon) for pcolormesh(lon, lat, Z)
            return np.asarray(vals2, dtype=float).T

        def _nearest_index_1d(arr, values):
            arr = np.asarray(arr, dtype=float)
            values = np.asarray(values, dtype=float)
            return np.abs(arr[None, :] - values[:, None]).argmin(axis=1)

        def _patch_rows_dataframe(t_idx_actual):
            mask_t = np.asarray(patch_table["time_idx"].values, dtype=int) == int(t_idx_actual)
            if not np.any(mask_t):
                return pd.DataFrame(columns=["latitude_f", "longitude_f", "level_f", "y_del"])

            ds_p = patch_table.isel(row=np.flatnonzero(mask_t))
            y_del = np.asarray(ds_p["Y_del_f"].sel(species_out=species).values, dtype=float)
            lat_f = np.asarray(ds_p["latitude_f"].values, dtype=float)
            lon_f = np.asarray(ds_p["longitude_f"].values, dtype=float)
            lev_f = np.asarray(ds_p["level_f"].values, dtype=float) if "level_f" in ds_p else np.full_like(lat_f, np.nan)

            valid = np.isfinite(y_del) & np.isfinite(lat_f) & np.isfinite(lon_f)
            if not np.any(valid):
                return pd.DataFrame(columns=["latitude_f", "longitude_f", "level_f", "y_del"])

            return pd.DataFrame({
                "latitude_f": lat_f[valid],
                "longitude_f": lon_f[valid],
                "level_f": lev_f[valid],
                "y_del": y_del[valid],
            })

        def _build_patch_delta_2d(t_idx_actual):
            """
            Build a 2D field on the coarse lon/lat map from sparse patch_table rows.
            Returned shape is (lat, lon).
            """
            df = _patch_rows_dataframe(t_idx_actual)
            if df.empty:
                return np.zeros((lat_bg.size, lon_bg.size), dtype=float)

            if vertical_mode == "single":
                levs = df["level_f"].values
                if np.all(~np.isfinite(levs)) or patch_level_req is None:
                    return np.zeros((lat_bg.size, lon_bg.size), dtype=float)

                if use_nearest_level:
                    unique_levs = np.unique(levs[np.isfinite(levs)])
                    if unique_levs.size == 0:
                        return np.zeros((lat_bg.size, lon_bg.size), dtype=float)
                    lev_use = float(unique_levs[np.argmin(np.abs(unique_levs - patch_level_req))])
                    mask = np.isclose(levs, lev_use)
                else:
                    if level_tol is None:
                        mask = np.isclose(levs, patch_level_req)
                    else:
                        mask = np.abs(levs - patch_level_req) <= float(level_tol)

                df = df.loc[mask]
                if df.empty:
                    return np.zeros((lat_bg.size, lon_bg.size), dtype=float)

                # grid stored for plotting as (lat, lon)
                delta = np.zeros((lat_bg.size, lon_bg.size), dtype=float)
                i_lat = _nearest_index_1d(lat_bg, df["latitude_f"].values)
                i_lon = _nearest_index_1d(lon_bg, df["longitude_f"].values)

                for ii, jj, vv in zip(i_lat, i_lon, df["y_del"].values):
                    delta[ii, jj] += vv

                return delta

            elif vertical_mode == "column":
                delta = np.zeros((lat_bg.size, lon_bg.size), dtype=float)
                i_lat = _nearest_index_1d(lat_bg, df["latitude_f"].values)
                i_lon = _nearest_index_1d(lon_bg, df["longitude_f"].values)

                for ii, jj, vv in zip(i_lat, i_lon, df["y_del"].values):
                    delta[ii, jj] += vv

                return delta

            elif vertical_mode == "topk":
                df = df[np.isfinite(df["level_f"].values)]
                if df.empty:
                    return np.zeros((lat_bg.size, lon_bg.size), dtype=float)

                i_lat = _nearest_index_1d(lat_bg, df["latitude_f"].values)
                i_lon = _nearest_index_1d(lon_bg, df["longitude_f"].values)
                levs = df["level_f"].values
                vals = df["y_del"].values

                unique_levs = np.unique(levs)
                level_to_idx = {lev: i for i, lev in enumerate(unique_levs)}
                nz = len(unique_levs)

                # shape (z, lat, lon)
                delta_3d = np.zeros((nz, lat_bg.size, lon_bg.size), dtype=float)
                for lev, ii, jj, vv in zip(levs, i_lat, i_lon, vals):
                    kk = level_to_idx[lev]
                    delta_3d[kk, ii, jj] += vv

                k = min(int(top_k), nz)
                work = np.where(np.isfinite(delta_3d), delta_3d, -np.inf)
                part = np.partition(work, kth=nz - k, axis=0)
                topk_vals = part[-k:, :, :]
                topk_vals = np.where(np.isfinite(topk_vals), topk_vals, 0.0)
                return np.sum(topk_vals, axis=0)

            else:
                raise ValueError(vertical_mode)

        def _build_fields(t_idx_actual):
            ds_t = _get_boxm_time_slice(t_idx_actual)

            bg_2d = _get_2d_from_coarse_da(ds_t["Y_bg_c"].sel(species_out=species))
            coarse_del_2d = _get_2d_from_coarse_da(ds_t["Y_del_c"].sel(species_out=species))
            patch_del_2d = _build_patch_delta_2d(t_idx_actual)

            if mode == "background":
                plot_2d = bg_2d
            elif mode == "coarse_delta":
                plot_2d = coarse_del_2d
            elif mode == "patch_delta":
                plot_2d = patch_del_2d
            elif mode == "total":
                plot_2d = bg_2d + coarse_del_2d
            elif mode == "total_with_patch":
                plot_2d = bg_2d + patch_del_2d
            else:
                raise ValueError(mode)

            return plot_2d, bg_2d, coarse_del_2d, patch_del_2d

        if dynamic_colorbar:
            vmin = vmax = None
        else:
            mins = []
            maxs = []
            print("Precomputing fixed colour scale...")
            for t_idx in time_indices:
                vals, _, _, _ = _build_fields(int(t_idx))
                finite = np.isfinite(vals)
                if np.any(finite):
                    mins.append(np.nanmin(vals))
                    maxs.append(np.nanmax(vals))
            if not mins:
                raise ValueError("No finite values found across selected frames.")
            vmin = float(np.min(mins))
            vmax = float(np.max(maxs))
            print(f"Fixed colour scale: vmin={vmin}, vmax={vmax}")

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.20)

        slider_ax = fig.add_axes([0.15, 0.06, 0.70, 0.05])
        slider = Slider(
            slider_ax,
            "Time Index",
            int(time_indices.min()),
            int(time_indices.max()),
            valinit=int(time_indices[0]),
            valstep=1,
        )

        cbar = None

        def _vertical_label():
            if vertical_mode == "single":
                return f"single level={level}"
            if vertical_mode == "column":
                return "full column"
            if vertical_mode == "topk":
                return f"top-{top_k} levels"
            return vertical_mode

        def draw_frame(t_idx_req):
            nonlocal cbar
            ax.clear()

            t_idx_actual = _snap_time_index(t_idx_req)
            vals, bg_2d, coarse_del_2d, patch_del_2d = _build_fields(t_idx_actual)

            finite = np.isfinite(vals)
            if not np.any(finite):
                ax.set_title(f"No valid data for time_idx={t_idx_actual}")
                fig.canvas.draw_idle()
                return

            if dynamic_colorbar:
                vmin_loc = float(np.nanmin(vals))
                vmax_loc = float(np.nanmax(vals))
            else:
                vmin_loc = vmin
                vmax_loc = vmax

            artist = ax.pcolormesh(
                lon_bg,
                lat_bg,
                vals,
                shading="auto",
                cmap=cmap,
                vmin=vmin_loc,
                vmax=vmax_loc,
            )

            if cbar is not None:
                cbar.remove()
            cbar = fig.colorbar(artist, ax=ax)

            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title(
                f"{mode.upper()} | {species} | time_idx={t_idx_actual} | {_vertical_label()}\n"
                f"plot min/max={np.nanmin(vals):.3e}/{np.nanmax(vals):.3e} | "
                f"bg min/max={np.nanmin(bg_2d):.3e}/{np.nanmax(bg_2d):.3e} | "
                f"coarse Δ min/max={np.nanmin(coarse_del_2d):.3e}/{np.nanmax(coarse_del_2d):.3e} | "
                f"patch Δ sum={np.nansum(patch_del_2d):.3e}"
            )

            print(
                f"[FRAME] mode={mode}, species={species}, time_idx={t_idx_actual}, vertical={_vertical_label()} | "
                f"plot min/max={np.nanmin(vals):.6e}/{np.nanmax(vals):.6e} | "
                f"bg min/max={np.nanmin(bg_2d):.6e}/{np.nanmax(bg_2d):.6e} | "
                f"coarse delta min/max={np.nanmin(coarse_del_2d):.6e}/{np.nanmax(coarse_del_2d):.6e} | "
                f"patch delta sum={np.nansum(patch_del_2d):.6e}"
            )

            fig.canvas.draw_idle()

        def slider_update(val):
            draw_frame(int(val))

        slider.on_changed(slider_update)
        draw_frame(int(time_indices[0]))
        plt.show()

    def animate_plumes_3d_plotly(
        self,
        job_id,
        time_idx_start=None,
        time_idx_end=None,
        frame_stride=3,
        output_path="plumes_animation.html",
        overlay_patch=False,
        patch_species="NO",
        patch_cmap="Plasma_r",
        patch_clim=None,
        z_exaggeration=1.0,
        flight_ids=None,
        overlay_ellipses=True,
        overlay_slices=True,
    ):
        """Interactive 3D animation rendered in the browser via Plotly/WebGL.

        Generates a self-contained HTML file with play/pause controls and a
        time slider.  The scene can be orbited, zoomed and panned interactively
        while the animation is paused.  Much faster to generate than the PyVista
        GIF/MP4 version because no off-screen rendering is required.

        Parameters
        ----------
        job_id : str
            GPAT job identifier.
        time_idx_start, time_idx_end : int, optional
            First / last time-step indices (inclusive) to include.
        frame_stride : int
            Skip every *frame_stride*-th available time step (default 3).
        output_path : str
            Output HTML file path.
        overlay_patch : bool
            Render the fine-grid concentration overlay as translucent
            fine-grid voxels (log10-scaled colour axis).
        patch_species : str
            Species name for the patch overlay.
        patch_cmap : str
            Plotly colorscale name (e.g. ``"Plasma_r"``, ``"Viridis"``).
        patch_clim : tuple[float, float] or None
            Fixed ``(vmin, vmax)`` in mol/m³ for the patch colour scale.
            Applied on a log10 axis.  If *None*, derived from all available
            non-zero patch values so the colour scale is consistent across
            frames.
        z_exaggeration : float
            Vertical exaggeration factor (1.0 = true scale).
        flight_ids : array-like or None
            Restrict to these flight IDs.
        overlay_ellipses : bool
            Render the ellipses (ribbon) mesh for each flight.
        overlay_slices : bool
            Render the slice mesh for each flight.
        """
        import plotly.graph_objects as go

        pl_ds = self.pp_gpat.pl_ds_dict[job_id]
        pl_out = self.pp_gpat.pl_out_dict[job_id]
        boxm_ds = self.pp_gpat.boxm_ds_dict[job_id]
        patch_table = self.pp_gpat.patch_table_dict[job_id]
        params = self.pp_gpat.params_dict.get(job_id, {})

        def _grid_attr(name):
            if name in boxm_ds.attrs:
                return float(boxm_ds.attrs[name])
            return float(params[name])

        hres_sim_c = _grid_attr("hres_sim_c")
        hres_sim_f = _grid_attr("hres_sim_f")
        vres_sim_c = _grid_attr("vres_sim_c")
        vres_sim_f = _grid_attr("vres_sim_f")

        nx_f = max(1, int(round(hres_sim_c / hres_sim_f)))
        nz_f = max(1, int(round(vres_sim_c / vres_sim_f)))
        hres_f_deg = hres_sim_c / nx_f
        vres_f_m = vres_sim_c / nz_f

        proj_ref_lon = float(boxm_ds["longitude_c"].min())
        proj_ref_lat = float(boxm_ds["latitude_c"].min())

        def _to_plot_m(lon_deg, lat_deg):
            return lonlat_to_m(lon_deg, lat_deg, proj_ref_lon, proj_ref_lat)

        def _fine_lengths_m(lat_deg):
            lat_arr = np.asarray(lat_deg, dtype=float)
            earth_radius_m = 6371000.0
            dx_m = earth_radius_m * np.cos(np.deg2rad(lat_arr)) * np.deg2rad(hres_f_deg)
            dy_m = np.full_like(lat_arr, earth_radius_m * np.deg2rad(hres_f_deg), dtype=float)
            dz_m = np.full_like(lat_arr, vres_f_m, dtype=float)
            return dx_m, dy_m, dz_m

        def _ellipse_center_m(ellipse_pts, row):
            ea = np.asarray(ellipse_pts, dtype=float)
            if ea.ndim == 2 and ea.shape[0] == 3:
                ea = ea.T
            if ea.ndim == 2 and ea.shape[1] == 3:
                fin = np.all(np.isfinite(ea), axis=1)
                if np.any(fin):
                    return np.nanmean(ea[fin], axis=0)
            return np.array([row["longitude_m"].item(), row["latitude_m"].item(), row["altitude"].item()], dtype=float)

        # --- Index arrays ---
        pl_time_rows     = np.asarray(pl_ds["time_idx"].values,     dtype=int)
        pl_fid_rows      = np.asarray(pl_ds["flight_id"].values)
        pl_seg_rows      = np.asarray(pl_ds["seg_id"].values,       dtype=int)
        pl_out_time_rows = np.asarray(pl_out["time_idx"].values,    dtype=int)
        pl_out_fid_rows  = np.asarray(pl_out["flight_id"].values)
        pl_out_seg_rows  = np.asarray(pl_out["seg_id"].values,      dtype=int)
        patch_time_rows  = np.asarray(patch_table["time_idx"].values, dtype=int)

        def _patch_subset(t_idx):
            row_sel = np.flatnonzero(patch_time_rows == int(t_idx))
            return patch_table.isel(row=row_sel) if row_sel.size else patch_table.isel(row=slice(0, 0))

        # --- Time steps ---
        common_times = np.intersect1d(np.unique(pl_time_rows), np.unique(pl_out_time_rows))
        if time_idx_start is not None:
            common_times = common_times[common_times >= int(time_idx_start)]
        if time_idx_end is not None:
            common_times = common_times[common_times <= int(time_idx_end)]
        common_times = common_times[::max(1, int(frame_stride))]
        if common_times.size == 0:
            raise ValueError(
                "No time steps available for the given range. "
                f"pl_ds time_idx range: {int(pl_time_rows.min())}–{int(pl_time_rows.max())}."
            )
        print(f"Building {common_times.size} Plotly frames: time_idx {common_times[0]}–{common_times[-1]}")

        # --- Flight list ---
        flight_ids_all = np.unique(pl_fid_rows)
        if flight_ids is not None:
            filt = np.atleast_1d(flight_ids)
            flight_ids_all = np.array([f for f in flight_ids_all if f in filt])

        flight_colors = ["firebrick", "darkorange", "royalblue", "seagreen",
                         "purple", "crimson", "teal", "brown"]

        _zs = float(z_exaggeration) if z_exaggeration and float(z_exaggeration) > 0.0 else 1.0

        # --- Stable global bounds for cleaner axes across all frames ---
        # Prefer simulation-domain bounds from params; fallback to box-grid
        # coordinates if params are unavailable.
        sim_lat_bounds = params.get("lat_bounds", None)
        sim_lon_bounds = params.get("lon_bounds", None)
        sim_alt_bounds = params.get("alt_bounds", None)

        if sim_lat_bounds is None and "latitude_c" in boxm_ds:
            sim_lat_bounds = (
                float(np.nanmin(boxm_ds["latitude_c"].values)),
                float(np.nanmax(boxm_ds["latitude_c"].values)),
            )
        if sim_lon_bounds is None and "longitude_c" in boxm_ds:
            sim_lon_bounds = (
                float(np.nanmin(boxm_ds["longitude_c"].values)),
                float(np.nanmax(boxm_ds["longitude_c"].values)),
            )
        if sim_alt_bounds is None and "altitude_c" in boxm_ds:
            sim_alt_bounds = (
                float(np.nanmin(boxm_ds["altitude_c"].values)),
                float(np.nanmax(boxm_ds["altitude_c"].values)),
            )

        use_sim_bounds = (
            sim_lat_bounds is not None
            and sim_lon_bounds is not None
            and sim_alt_bounds is not None
            and len(sim_lat_bounds) == 2
            and len(sim_lon_bounds) == 2
            and len(sim_alt_bounds) == 2
        )

        if use_sim_bounds:
            lat0, lat1 = [float(v) for v in sim_lat_bounds]
            lon0, lon1 = [float(v) for v in sim_lon_bounds]
            alt0, alt1 = [float(v) for v in sim_alt_bounds]

            lon_corners = np.array([lon0, lon1, lon0, lon1], dtype=float)
            lat_corners = np.array([lat0, lat0, lat1, lat1], dtype=float)
            x_corners, y_corners = _to_plot_m(lon_corners, lat_corners)

            x_min = float(np.nanmin(x_corners))
            x_max = float(np.nanmax(x_corners))
            y_min = float(np.nanmin(y_corners))
            y_max = float(np.nanmax(y_corners))
            z_min = float(min(alt0, alt1) * _zs)
            z_max = float(max(alt0, alt1) * _zs)
            print(f"Using simulation bounds for axes: X[{x_min:.0f},{x_max:.0f}] Y[{y_min:.0f},{y_max:.0f}] Z[{z_min:.0f},{z_max:.0f}]")
        else:
            lon_all = np.asarray(pl_ds["longitude_m"].values, dtype=float)
            lat_all = np.asarray(pl_ds["latitude_m"].values, dtype=float)
            alt_all = np.asarray(pl_ds["altitude"].values, dtype=float) * _zs
            finite = np.isfinite(lon_all) & np.isfinite(lat_all) & np.isfinite(alt_all)
            if not np.any(finite):
                raise ValueError("No finite plume coordinates available to build scene axes.")

            x_min = float(np.nanmin(lon_all[finite]))
            x_max = float(np.nanmax(lon_all[finite]))
            y_min = float(np.nanmin(lat_all[finite]))
            y_max = float(np.nanmax(lat_all[finite]))
            z_min = float(np.nanmin(alt_all[finite]))
            z_max = float(np.nanmax(alt_all[finite]))

        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, 1.0)
        z_span = max(z_max - z_min, 1.0)

        x_pad = 0.05 * x_span
        y_pad = 0.05 * y_span
        z_pad = max(0.10 * z_span, 300.0 * _zs)

        x_range = [x_min - x_pad, x_max + x_pad]
        y_range = [y_min - y_pad, y_max + y_pad]
        z_range = [z_min - z_pad, z_max + z_pad]

        # Preserve overall proportions but avoid a visually collapsed z-axis.
        max_span = max(x_span, y_span)
        aspect_z = min(max(z_span / max_span, 0.15), 2.0)

        # --- Global patch clim (log10 scale) ---
        log_clim = None
        patch_available = False
        req_vars = {"longitude_f", "latitude_f", "altitude_f", "Y_del_f"}
        if overlay_patch and req_vars.issubset(set(patch_table.variables) | set(patch_table.coords)):
            if patch_species in patch_table["species_out"].values:
                patch_available = True
                all_pv = np.asarray(
                    patch_table["Y_del_f"].sel(species_out=patch_species).values, dtype=float
                )
                pos = all_pv[self._patch_value_mask(all_pv)[0]]
                if pos.size > 0:
                    if patch_clim is not None:
                        log_clim = (np.log10(float(patch_clim[0])), np.log10(float(patch_clim[1])))
                    else:
                        log_clim = (float(np.log10(pos.min())), float(np.log10(pos.max())))
                    print(f"Patch log10 clim for {patch_species}: {log_clim[0]:.2f} – {log_clim[1]:.2f}")

        # --- Mesh helpers ---
        def _ribbon_data(ellipses_arr):
            """(n_segs, n_pts, 3) → dict with x/y/z/i/j/k for go.Mesh3d."""
            if len(ellipses_arr) < 2:
                return dict(x=[], y=[], z=[], i=[], j=[], k=[])
            xs, ys, zs, ii, jj, kk = [], [], [], [], [], []
            offset = 0
            for si in range(len(ellipses_arr) - 1):
                e1, e2 = ellipses_arr[si], ellipses_arr[si + 1]
                if not (np.all(np.isfinite(e1)) and np.all(np.isfinite(e2))):
                    continue
                n = e1.shape[0]
                pts = np.vstack([e1, e2])
                xs += pts[:, 0].tolist(); ys += pts[:, 1].tolist(); zs += pts[:, 2].tolist()
                for j in range(n):
                    j2 = (j + 1) % n
                    ii += [offset + j,  offset + j2]
                    jj += [offset + j2, offset + n + j2]
                    kk += [offset + n + j, offset + n + j]
                offset += 2 * n
            return dict(x=xs, y=ys, z=zs, i=ii, j=jj, k=kk)

        def _slice_data(slice_polys_arr, n_ellipses):
            """(n_segs, n_slices, n_corners, 3) → dict for go.Mesh3d."""
            if slice_polys_arr.ndim != 4 or slice_polys_arr.shape[0] < 2 or slice_polys_arr.shape[0] != n_ellipses:
                return dict(x=[], y=[], z=[], i=[], j=[], k=[])
            n_segs, n_slices = slice_polys_arr.shape[0], slice_polys_arr.shape[1]
            xs, ys, zs, ii, jj, kk = [], [], [], [], [], []
            offset = 0
            for sl in range(n_slices):
                for sg in range(n_segs - 1):
                    p1, p2 = slice_polys_arr[sg, sl], slice_polys_arr[sg + 1, sl]
                    if p1.shape[0] < 3 or not np.all(np.isfinite(p1)) or not np.all(np.isfinite(p2)):
                        continue
                    if np.linalg.norm(p1) < 1.0 or np.linalg.norm(p2) < 1.0:
                        continue
                    nc = p1.shape[0]
                    pts = np.vstack([p1, p2])
                    xs += pts[:, 0].tolist(); ys += pts[:, 1].tolist(); zs += pts[:, 2].tolist()
                    for ci in range(nc):
                        nci = (ci + 1) % nc
                        ii += [offset + ci,  offset + nci]
                        jj += [offset + nci, offset + nc + nci]
                        kk += [offset + nc + ci, offset + nc + ci]
                    offset += 2 * nc
            return dict(x=xs, y=ys, z=zs, i=ii, j=jj, k=kk)

        def _voxel_mesh_data(xc, yc, zc, dx, dy, dz, values):
            """Build triangular mesh data for axis-aligned voxel cells."""
            if len(xc) == 0:
                return dict(x=[], y=[], z=[], i=[], j=[], k=[], intensity=[])

            offsets = np.array([
                [-1.0, -1.0, -1.0], [ 1.0, -1.0, -1.0],
                [ 1.0,  1.0, -1.0], [-1.0,  1.0, -1.0],
                [-1.0, -1.0,  1.0], [ 1.0, -1.0,  1.0],
                [ 1.0,  1.0,  1.0], [-1.0,  1.0,  1.0],
            ], dtype=float)
            tris = np.array([
                [0, 1, 2], [0, 2, 3],
                [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4],
                [1, 2, 6], [1, 6, 5],
                [2, 3, 7], [2, 7, 6],
                [3, 0, 4], [3, 4, 7],
            ], dtype=int)

            xs, ys, zs = [], [], []
            ii, jj, kk = [], [], []
            intensity = []
            v_off = 0

            for cx, cy, cz, hx, hy, hz, val in zip(
                xc,
                yc,
                zc,
                0.5 * np.asarray(dx, dtype=float),
                0.5 * np.asarray(dy, dtype=float),
                0.5 * np.asarray(dz, dtype=float),
                np.asarray(values, dtype=float),
            ):
                verts = np.column_stack([
                    cx + offsets[:, 0] * hx,
                    cy + offsets[:, 1] * hy,
                    cz + offsets[:, 2] * hz,
                ])
                xs.extend(verts[:, 0].tolist())
                ys.extend(verts[:, 1].tolist())
                zs.extend(verts[:, 2].tolist())
                intensity.extend([val] * 8)

                ii.extend((v_off + tris[:, 0]).tolist())
                jj.extend((v_off + tris[:, 1]).tolist())
                kk.extend((v_off + tris[:, 2]).tolist())
                v_off += 8

            return dict(x=xs, y=ys, z=zs, i=ii, j=jj, k=kk, intensity=intensity)

        # --- Per-frame geometry collector ---
        def _collect_frame(t_idx):
            """Return (traj_data, ribbon_data, slice_data, patch_data) per flight + patch."""
            tsel_pl  = np.flatnonzero(pl_time_rows     == int(t_idx))
            tsel_out = np.flatnonzero(pl_out_time_rows == int(t_idx))
            empty_s  = dict(x=[], y=[], z=[])
            empty_m  = dict(x=[], y=[], z=[], i=[], j=[], k=[])
            empty_p  = dict(x=[], y=[], z=[], i=[], j=[], k=[], intensity=[])

            if tsel_pl.size == 0 or tsel_out.size == 0:
                trajs    = [empty_s] * len(flight_ids_all)
                ribbons  = [empty_m] * len(flight_ids_all)
                slices   = [empty_m] * len(flight_ids_all)
                return trajs, ribbons, slices, empty_p

            tp_pl  = int(tsel_pl[0])
            tp_out = int(tsel_out[0])
            trajs, ribbons, slices = [], [], []

            for fid in flight_ids_all:
                centres, ell_list, spl_list = [], [], []
                for seg_id in np.unique(pl_seg_rows[pl_fid_rows == fid]):
                    ss  = np.flatnonzero((pl_seg_rows     == int(seg_id)) & (pl_fid_rows     == fid))
                    sso = np.flatnonzero((pl_out_seg_rows == int(seg_id)) & (pl_out_fid_rows == fid))
                    if ss.size == 0 or sso.size == 0:
                        continue
                    row     = pl_ds.isel(seg_id=int(ss[0]),  time=tp_pl).squeeze(drop=True)
                    row_out = pl_out.isel(seg_id=int(sso[0]), time=tp_out).squeeze(drop=True)
                    rc = np.array([row["longitude_m"].item(), row["latitude_m"].item(), row["altitude"].item()], dtype=float)
                    if not np.all(np.isfinite(rc)):
                        continue
                    if (row_out["ellipses_m"] is not None
                            and hasattr(row_out["ellipses_m"], "__len__")
                            and len(row_out["ellipses_m"]) > 0):
                        ev = row_out["ellipses_m"].values
                        cm = _ellipse_center_m(ev, row)
                        if np.all(np.isfinite(cm)):
                            centres.append(list(cm))
                            ell_list.append(ev)
                            spl_list.append(row_out["slice_polys_m"].values)

                ea = np.array(ell_list)
                spa = np.array(spl_list)
                if ea.ndim == 3 and ea.shape[1] == 3:
                    ea = np.transpose(ea, (0, 2, 1))
                if spa.ndim == 4 and spa.shape[2] == 3:
                    spa = np.transpose(spa, (0, 1, 3, 2))
                if _zs != 1.0:
                    if len(centres) > 0:
                        ca = np.array(centres); ca[:, 2] *= _zs; centres = ca.tolist()
                    if ea.ndim == 3 and ea.shape[2] == 3:
                        ea[:, :, 2] *= _zs
                    if spa.ndim == 4 and spa.shape[3] == 3:
                        spa[:, :, :, 2] *= _zs

                if len(centres) > 1:
                    traj = np.array(centres)
                    traj = traj[np.all(np.isfinite(traj), axis=1)]
                    trajs.append(dict(x=traj[:, 0].tolist(), y=traj[:, 1].tolist(), z=traj[:, 2].tolist()) if traj.shape[0] > 1 else empty_s)
                else:
                    trajs.append(empty_s)

                # Only append ribbons and slices if enabled
                if overlay_ellipses:
                    ribbons.append(_ribbon_data(ea) if len(ea) >= 2 else empty_m)
                else:
                    ribbons.append(empty_m)
                if overlay_slices:
                    slices.append(_slice_data(spa, len(ea)))
                else:
                    slices.append(empty_m)

            # Patch overlay
            patch_d = empty_p
            if overlay_patch and patch_available and log_clim is not None:
                pt = _patch_subset(int(t_idx))
                if pt.sizes.get("row", 0) > 0:
                    pv_vals = pt["Y_del_f"].sel(species_out=patch_species).values
                    lf = pt["longitude_f"].values; ltf = pt["latitude_f"].values; af = pt["altitude_f"].values
                    value_mask, plot_floor = self._patch_value_mask(pv_vals)
                    valid = np.isfinite(pv_vals) & np.isfinite(lf) & np.isfinite(ltf) & np.isfinite(af) & value_mask
                    if np.any(valid):
                        lm, latm = _to_plot_m(lf[valid], ltf[valid])
                        dx_f, dy_f, dz_f = _fine_lengths_m(ltf[valid])
                        patch_d = _voxel_mesh_data(
                            lm,
                            latm,
                            af[valid] * _zs,
                            dx_f,
                            dy_f,
                            dz_f * _zs,
                            np.log10(pv_vals[valid]),
                        )
            return trajs, ribbons, slices, patch_d

        # --- Build initial traces (frame 0) ---
        trajs0, ribbons0, slices0, patch0 = _collect_frame(common_times[0])
        n_flt = len(flight_ids_all)
        initial_traces = []

        for fi, fid in enumerate(flight_ids_all):
            col = flight_colors[fi % len(flight_colors)]
            t0 = trajs0[fi]
            initial_traces.append(go.Scatter3d(
                x=t0["x"], y=t0["y"], z=t0["z"],
                mode="lines",
                line=dict(color=col, width=4),
                name=f"{fid} trajectory",
                showlegend=True,
            ))
            if overlay_ellipses:
                r0 = ribbons0[fi]
                initial_traces.append(go.Mesh3d(
                    x=r0["x"], y=r0["y"], z=r0["z"],
                    i=r0["i"], j=r0["j"], k=r0["k"],
                    color=col, opacity=0.5,
                    name=f"{fid} ellipses",
                    showlegend=False,
                ))
            if overlay_slices:
                s0 = slices0[fi]
                initial_traces.append(go.Mesh3d(
                    x=s0["x"], y=s0["y"], z=s0["z"],
                    i=s0["i"], j=s0["j"], k=s0["k"],
                    color=col, opacity=0.25,
                    name=f"{fid} slices",
                    showlegend=False,
                ))

        if overlay_patch and patch_available:
            p0 = patch0
            initial_traces.append(go.Mesh3d(
                x=p0["x"], y=p0["y"], z=p0["z"],
                i=p0["i"], j=p0["j"], k=p0["k"],
                intensity=p0["intensity"] if p0["intensity"] else [],
                colorscale=patch_cmap,
                cmin=log_clim[0] if log_clim else None,
                cmax=log_clim[1] if log_clim else None,
                colorbar=dict(
                    title=(
                        f"log10 Y_del_f<br>{patch_species} (ppb)"
                        if self._resolve_patch_plot_floor(
                            patch_table["Y_del_f"].sel(species_out=patch_species).values
                        ) is None
                        else (
                            "log10 Y_del_f<br>"
                            f"{patch_species} (ppb)<br>"
                            f">= {self._resolve_patch_plot_floor(patch_table['Y_del_f'].sel(species_out=patch_species).values):.1e}"
                        )
                    )
                ),
                opacity=0.22,
                name=f"patch {patch_species}",
                showlegend=True,
            ))

        # --- Build frames ---
        frames = []
        # Calculate number of animated traces per frame
        n_traces_per_flight = 1 + int(overlay_ellipses) + int(overlay_slices)
        N_animated = n_traces_per_flight * n_flt + (1 if (overlay_patch and patch_available) else 0)
        for frame_num, t_idx in enumerate(common_times):
            trajs, ribbons, slices, patch_d = _collect_frame(t_idx)
            fd = []
            for fi in range(n_flt):
                t = trajs[fi]
                fd.append(go.Scatter3d(x=t["x"], y=t["y"], z=t["z"]))
                if overlay_ellipses:
                    r = ribbons[fi]
                    fd.append(go.Mesh3d(x=r["x"], y=r["y"], z=r["z"], i=r["i"], j=r["j"], k=r["k"]))
                if overlay_slices:
                    s = slices[fi]
                    fd.append(go.Mesh3d(x=s["x"], y=s["y"], z=s["z"], i=s["i"], j=s["j"], k=s["k"]))
            if overlay_patch and patch_available:
                p = patch_d
                fd.append(go.Mesh3d(
                    x=p["x"], y=p["y"], z=p["z"],
                    i=p["i"], j=p["j"], k=p["k"],
                    intensity=p["intensity"] if p["intensity"] else [],
                ))
            frames.append(go.Frame(
                data=fd,
                traces=list(range(N_animated)),
                name=str(t_idx),
            ))
            if (frame_num + 1) % 10 == 0 or frame_num == common_times.size - 1:
                print(f"  Built frame {frame_num + 1}/{common_times.size}: time_idx={t_idx}")

        ztitle = "Altitude (m)" if np.isclose(_zs, 1.0) else f"Altitude ×{_zs:.0f} (m)"
        flight_note = "all flights" if flight_ids is None else f"flight_ids={list(np.atleast_1d(flight_ids))}"

        layout = go.Layout(
            title=f"GPAT Plumes — {flight_note}",
            scene=dict(
                xaxis=dict(
                    title="X (m)",
                    range=x_range,
                    autorange=False,
                    nticks=6,
                    tickformat=".0f",
                    showspikes=False,
                    showbackground=True,
                    backgroundcolor="rgb(245,245,245)",
                    gridcolor="rgb(210,210,210)",
                    zeroline=False,
                ),
                yaxis=dict(
                    title="Y (m)",
                    range=y_range,
                    autorange=False,
                    nticks=6,
                    tickformat=".0f",
                    showspikes=False,
                    showbackground=True,
                    backgroundcolor="rgb(245,245,245)",
                    gridcolor="rgb(210,210,210)",
                    zeroline=False,
                ),
                zaxis=dict(
                    title=ztitle,
                    range=z_range,
                    autorange=False,
                    nticks=4 if np.isclose(_zs, 1.0) else 6,
                    tickformat=".0f",
                    showspikes=False,
                    showbackground=True,
                    backgroundcolor="rgb(250,250,250)",
                    gridcolor="rgb(220,220,220)",
                    zeroline=False,
                ),
                aspectmode="manual",
                aspectratio=dict(x=1.0, y=y_span / x_span if x_span > 0 else 1.0, z=aspect_z),
                camera=dict(
                    up=dict(x=0.0, y=0.0, z=1.0),
                    eye=dict(x=1.65, y=1.65, z=0.85),
                ),
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=1.05, x=0.0, xanchor="left",
                buttons=[
                    dict(label="▶ Play",
                         method="animate",
                         args=[None, dict(frame=dict(duration=int(1000 / max(1, fps := 5)), redraw=True),
                                          fromcurrent=True, mode="immediate")]),
                    dict(label="⏸ Pause",
                         method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode="immediate")]),
                ],
            )],
            sliders=[dict(
                active=0,
                currentvalue=dict(prefix="time_idx: ", visible=True, xanchor="center"),
                pad=dict(t=50),
                steps=[dict(
                    method="animate",
                    args=[[str(t)], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                    label=str(t),
                ) for t in common_times],
            )],
        )

        fig = go.Figure(data=initial_traces, layout=layout, frames=frames)
        fig.write_html(output_path, include_plotlyjs="cdn")
        print(f"Plotly animation saved → {output_path}")


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


