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

    def plot_patch_heatmap_2d(
        self,
        time_idx,
        job_id=None,
        species="NO",
        shared_color_scale=True,
        color_scale=None,
        overlay_plume_centers=False,
        overlay_trajectories=False,
        flight_ids=None,
        plume_marker_size=20,
        plume_marker_color="white",
        plume_marker_edge_color="black",
        plume_marker_alpha=0.9,
        trajectory_color="black",
        trajectory_linewidth=1.2,
        trajectory_alpha=0.8,
        subplot_ncols=4,
        subplot_figsize=None,
        _ax=None,
        _show=True,
    ):
        if job_id is None:
            if not self.pp_gpat.job_ids:
                raise ValueError("No GPAT jobs are loaded.")
            job_id = self.pp_gpat.job_ids[0]

        if not np.isscalar(time_idx):
            time_indices = [int(t) for t in time_idx]
            if len(time_indices) == 0:
                raise ValueError("time_idx iterable is empty.")

            color_scale_use = color_scale
            if shared_color_scale and color_scale is None:
                # Build a shared scale from timestep-local rows to avoid loading all rows at once.
                patch_table = self.pp_gpat.patch_table_dict[job_id]
                time_rows_all = np.asarray(patch_table["time_idx"].values, dtype=int)
                vmax = 0.0
                for t_idx in time_indices:
                    row_sel = np.flatnonzero(time_rows_all == int(t_idx))
                    if row_sel.size == 0:
                        continue
                    vals_t = np.asarray(
                        patch_table["Y_del_f"].isel(row=row_sel).sel(species_out=species).values
                    )
                    vals_t = vals_t[np.isfinite(vals_t) & (vals_t != 0.0)]
                    if vals_t.size > 0:
                        vmax = max(vmax, float(np.nanmax(vals_t)))

                if vmax > 0.0:
                    color_scale_use = (0.0, vmax)

            ncols = max(1, min(int(subplot_ncols), len(time_indices)))
            nrows = int(np.ceil(len(time_indices) / ncols))

            if subplot_figsize is None:
                subplot_figsize = (5 * ncols, 4 * nrows)

            fig, axes = plt.subplots(nrows, ncols, figsize=subplot_figsize, squeeze=False)
            da_by_time = {}

            for i, t_idx in enumerate(time_indices):
                ax_i = axes.flat[i]
                try:
                    da_by_time[t_idx] = self.plot_patch_heatmap_2d(
                        t_idx,
                        job_id=job_id,
                        species=species,
                        shared_color_scale=shared_color_scale,
                        color_scale=color_scale_use,
                        overlay_plume_centers=overlay_plume_centers,
                        overlay_trajectories=overlay_trajectories,
                        flight_ids=flight_ids,
                        plume_marker_size=plume_marker_size,
                        plume_marker_color=plume_marker_color,
                        plume_marker_edge_color=plume_marker_edge_color,
                        plume_marker_alpha=plume_marker_alpha,
                        trajectory_color=trajectory_color,
                        trajectory_linewidth=trajectory_linewidth,
                        trajectory_alpha=trajectory_alpha,
                        subplot_ncols=subplot_ncols,
                        subplot_figsize=subplot_figsize,
                        _ax=ax_i,
                        _show=False,
                    )
                except ValueError as exc:
                    ax_i.set_title(f"time_idx={t_idx}\n{exc}")
                    ax_i.set_axis_off()

            for j in range(len(time_indices), nrows * ncols):
                axes.flat[j].set_axis_off()

            fig.tight_layout()
            if _show:
                plt.show()

            return da_by_time

        patch_table = self.pp_gpat.patch_table_dict[job_id]
        boxm_ds = self.pp_gpat.boxm_ds_dict[job_id]
        params = self.pp_gpat.params_dict.get(job_id, {})
        sim_raw = params.get("sim_params", params)

        def _extract_sim_value(sim_obj, key):
            if is_dataclass(sim_obj):
                sim_obj = asdict(sim_obj)

            if isinstance(sim_obj, dict):
                if key in sim_obj:
                    return float(sim_obj[key])
                if "sim_params" in sim_obj:
                    return _extract_sim_value(sim_obj["sim_params"], key)

            if isinstance(sim_obj, str):
                try:
                    obj = json.loads(sim_obj)
                    if isinstance(obj, dict) and key in obj:
                        return float(obj[key])
                except Exception:
                    pass

                try:
                    obj = ast.literal_eval(sim_obj)
                    if isinstance(obj, dict) and key in obj:
                        return float(obj[key])
                except Exception:
                    pass

                match = re.search(
                    rf"{key}\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                    sim_obj,
                )
                if match:
                    return float(match.group(1))

            raise ValueError(f"Could not extract '{key}' from sim_params")

        def _extract_sim_bounds(sim_obj, key):
            if is_dataclass(sim_obj):
                sim_obj = asdict(sim_obj)

            if isinstance(sim_obj, dict):
                if key in sim_obj:
                    bounds = sim_obj[key]
                    if len(bounds) != 2:
                        raise ValueError(f"Expected '{key}' to have length 2.")
                    return float(bounds[0]), float(bounds[1])
                if "sim_params" in sim_obj:
                    return _extract_sim_bounds(sim_obj["sim_params"], key)

            if isinstance(sim_obj, str):
                try:
                    obj = json.loads(sim_obj)
                    if isinstance(obj, dict) and key in obj:
                        bounds = obj[key]
                        return float(bounds[0]), float(bounds[1])
                except Exception:
                    pass

                try:
                    obj = ast.literal_eval(sim_obj)
                    if isinstance(obj, dict) and key in obj:
                        bounds = obj[key]
                        return float(bounds[0]), float(bounds[1])
                except Exception:
                    pass

                match = re.search(
                    rf"{key}\s*=\s*\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)",
                    sim_obj,
                )
                if match:
                    return float(match.group(1)), float(match.group(2))

            raise ValueError(f"Could not extract '{key}' from sim_params")

        hres_sim_c = _extract_sim_value(sim_raw, "hres_sim_c")
        hres_sim_f = _extract_sim_value(sim_raw, "hres_sim_f")
        vres_sim_c = _extract_sim_value(sim_raw, "vres_sim_c")
        vres_sim_f = _extract_sim_value(sim_raw, "vres_sim_f")

        time_rows = np.asarray(patch_table["time_idx"].values, dtype=int)
        row_sel = np.flatnonzero(time_rows == int(time_idx))
        if row_sel.size == 0:
            raise ValueError(
                f"No patch_table rows found for time_idx={time_idx}."
            )

        # Slice to the requested timestep first to avoid loading all rows from disk.
        patch_t = patch_table.isel(row=row_sel)

        vals = np.asarray(patch_t["Y_del_f"].sel(species_out=species).values)
        mask = np.isfinite(vals) & (vals != 0.0)

        if not np.any(mask):
            raise ValueError(
                f"No nonzero patch_table values found for species={species} at time_idx={time_idx}."
            )

        if {"latitude_f", "longitude_f"}.issubset(set(patch_t.variables) | set(patch_t.coords)):
            lat_plot = np.asarray(patch_t["latitude_f"].values)[mask]
            lon_plot = np.asarray(patch_t["longitude_f"].values)[mask]
        else:
            nx_f = max(1, int(np.rint(hres_sim_c / hres_sim_f)))
            ny_f = nx_f
            nz_f = max(1, int(np.rint(vres_sim_c / vres_sim_f)))
            hres_f_deg = hres_sim_c / nx_f

            row_cell_c = np.asarray(patch_t["row_cell_c"].values, dtype=int)
            row_cell_f = np.asarray(patch_t["row_cell_f"].values, dtype=int)

            rem_f = (row_cell_f - 1) % (nx_f * ny_f)
            iy_f = rem_f // nx_f + 1
            ix_f = rem_f % nx_f + 1

            lat_c = boxm_ds["latitude_c"].values[row_cell_c - 1]
            lon_c = boxm_ds["longitude_c"].values[row_cell_c - 1]

            lat_all = lat_c - 0.5 * hres_sim_c + (iy_f - 0.5) * hres_f_deg
            lon_all = lon_c - 0.5 * hres_sim_c + (ix_f - 0.5) * hres_f_deg

            lat_plot = lat_all[mask]
            lon_plot = lon_all[mask]

        val_plot = vals[mask]

        try:
            lat_min, lat_max = _extract_sim_bounds(sim_raw, "lat_bounds")
            lon_min, lon_max = _extract_sim_bounds(sim_raw, "lon_bounds")
        except ValueError:
            lat_min = float(np.nanmin(lat_plot) - 0.5 * hres_sim_f)
            lat_max = float(np.nanmax(lat_plot) + 0.5 * hres_sim_f)
            lon_min = float(np.nanmin(lon_plot) - 0.5 * hres_sim_f)
            lon_max = float(np.nanmax(lon_plot) + 0.5 * hres_sim_f)

        n_lat = max(1, int(np.rint((lat_max - lat_min) / hres_sim_f)))
        n_lon = max(1, int(np.rint((lon_max - lon_min) / hres_sim_f)))
        lat_centers = lat_min + (np.arange(n_lat) + 0.5) * hres_sim_f
        lon_centers = lon_min + (np.arange(n_lon) + 0.5) * hres_sim_f

        j = np.floor((lat_plot - lat_min) / hres_sim_f).astype(int)
        i = np.floor((lon_plot - lon_min) / hres_sim_f).astype(int)

        inside = (j >= 0) & (j < lat_centers.size) & (i >= 0) & (i < lon_centers.size)
        if not np.any(inside):
            raise ValueError("No patch cells fall within the fine-grid plotting bounds.")

        grid = np.zeros((lat_centers.size, lon_centers.size), dtype=float)
        np.add.at(grid, (j[inside], i[inside]), val_plot[inside])

        da_plot = xr.DataArray(
            grid,
            coords={"latitude_f": lat_centers, "longitude_f": lon_centers},
            dims=("latitude_f", "longitude_f"),
            name=f"Y_del_f_{species}",
        )

        lat_edges = lat_min + np.arange(n_lat + 1) * hres_sim_f
        lon_edges = lon_min + np.arange(n_lon + 1) * hres_sim_f

        if _ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            ax = _ax
            fig = ax.figure

        if color_scale is None:
            vmin, vmax = None, None
        else:
            vmin, vmax = float(color_scale[0]), float(color_scale[1])

        mesh = ax.pcolormesh(
            lon_edges,
            lat_edges,
            da_plot.values,
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(mesh, ax=ax, label=f"{species} Y_del_f")

        if overlay_plume_centers or overlay_trajectories:
            pl_ds = self.pp_gpat.pl_ds_dict.get(job_id)

            if pl_ds is None:
                print(f"No pl_ds available for job_id={job_id}; skipping plume overlay.")
            elif not {"time_idx", "flight_id", "longitude", "latitude"}.issubset(set(pl_ds.variables) | set(pl_ds.coords)):
                print("pl_ds is missing required variables (time_idx, flight_id, longitude, latitude); skipping plume overlay.")
            else:
                pl_t = pl_ds.where(pl_ds["time_idx"] == int(time_idx), drop=True)

                if pl_t.sizes.get("seg_id", 0) == 0:
                    print(f"No plume segments found for time_idx={time_idx}; skipping plume overlay.")
                else:
                    if flight_ids is not None:
                        flight_ids_filter = np.atleast_1d(flight_ids)
                        pl_t = pl_t.where(np.isin(pl_t["flight_id"], flight_ids_filter), drop=True)

                    if pl_t.sizes.get("seg_id", 0) == 0:
                        print("No plume segments left after flight_id filtering; skipping plume overlay.")
                    else:
                        unique_fids = np.unique(pl_t["flight_id"].values)
                        for fid in unique_fids:
                            pl_f = pl_t.where(pl_t["flight_id"] == fid, drop=True)

                            lon = np.asarray(pl_f["longitude"].values).ravel()
                            lat = np.asarray(pl_f["latitude"].values).ravel()

                            if "seg_id" in pl_f:
                                seg = np.asarray(pl_f["seg_id"].values).ravel()
                                order = np.argsort(seg)
                            else:
                                order = np.arange(lon.size)

                            lon = lon[order]
                            lat = lat[order]
                            valid = np.isfinite(lon) & np.isfinite(lat)
                            lon = lon[valid]
                            lat = lat[valid]

                            if lon.size == 0:
                                continue

                            if overlay_trajectories and lon.size > 1:
                                ax.plot(
                                    lon,
                                    lat,
                                    color=trajectory_color,
                                    linewidth=trajectory_linewidth,
                                    alpha=trajectory_alpha,
                                    zorder=4,
                                )

                            if overlay_plume_centers:
                                ax.scatter(
                                    lon,
                                    lat,
                                    s=plume_marker_size,
                                    c=plume_marker_color,
                                    edgecolors=plume_marker_edge_color,
                                    alpha=plume_marker_alpha,
                                    linewidths=0.5,
                                    zorder=5,
                                )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Whole-domain fine-grid patch plot: {species}, time_idx={time_idx}")
        if _show and _ax is None:
            fig.tight_layout()
            plt.show()

        return da_plot

    def plot_plumes_3d_pv(
        self,
        job_id,
        time_idx=184,
        overlay_patch=False,
        patch_species="NO",
        patch_opacity=0.18,
        flight_ids=None,
    ):
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

        # Match the existing projected meter frame used by the GPAT plume geometry.
        proj_ref_lon = float(boxm_ds["longitude_c"].min())
        proj_ref_lat = float(boxm_ds["latitude_c"].min())

        def _to_plot_m(lon_deg, lat_deg):
            lon_m, lat_m = lonlat_to_m(lon_deg, lat_deg, proj_ref_lon, proj_ref_lat)
            return lon_m, lat_m

        def _fine_lengths_m(lat_deg):
            lat_arr = np.asarray(lat_deg, dtype=float)
            earth_radius_m = 6371000.0
            dx_m = earth_radius_m * np.cos(np.deg2rad(lat_arr)) * np.deg2rad(hres_f_deg)
            dy_m = np.full_like(lat_arr, earth_radius_m * np.deg2rad(hres_f_deg), dtype=float)
            dz_m = np.full_like(lat_arr, vres_f_m, dtype=float)
            return dx_m, dy_m, dz_m

        def _build_hex_grid(xc, yc, zc, dx, dy, dz):
            centers = np.column_stack([xc, yc, zc])
            if centers.size == 0:
                return None

            half = 0.5 * np.column_stack([dx, dy, dz])
            offsets = np.array([
                [-1.0, -1.0, -1.0],
                [ 1.0, -1.0, -1.0],
                [ 1.0,  1.0, -1.0],
                [-1.0,  1.0, -1.0],
                [-1.0, -1.0,  1.0],
                [ 1.0, -1.0,  1.0],
                [ 1.0,  1.0,  1.0],
                [-1.0,  1.0,  1.0],
            ])

            points = (centers[:, None, :] + offsets[None, :, :] * half[:, None, :]).reshape(-1, 3)
            n_cells = centers.shape[0]
            cells = np.hstack([
                np.full((n_cells, 1), 8, dtype=np.int64),
                np.arange(n_cells * 8, dtype=np.int64).reshape(n_cells, 8),
            ]).ravel()
            celltypes = np.full(n_cells, pv.CellType.HEXAHEDRON, dtype=np.uint8)
            return pv.UnstructuredGrid(cells, celltypes, points)

        def _ellipse_center_m(ellipse_pts, row):
            ellipse_arr = np.asarray(ellipse_pts, dtype=float)
            if ellipse_arr.ndim == 2 and ellipse_arr.shape[0] == 3:
                ellipse_arr = ellipse_arr.T
            if ellipse_arr.ndim == 2 and ellipse_arr.shape[1] == 3:
                finite = np.all(np.isfinite(ellipse_arr), axis=1)
                if np.any(finite):
                    return np.nanmean(ellipse_arr[finite], axis=0)

            lon_m = row["longitude_m"].item()
            lat_m = row["latitude_m"].item()
            alt_m = row["altitude"].item()
            return np.array([lon_m, lat_m, alt_m], dtype=float)

        # --- Plotting with PyVista (all in meters) ---
        flight_ids_t = np.unique(
            pl_ds.where(pl_ds["time_idx"] == time_idx, drop=True)["flight_id"].values
        )
        if flight_ids is not None:
            flight_ids_filter = np.atleast_1d(flight_ids)
            flight_ids_t = np.array([fid for fid in flight_ids_t if fid in flight_ids_filter])

        flight_styles = [
            {"ellipse": "firebrick", "trajectory": "navy", "slice": "seagreen"},
            {"ellipse": "darkorange", "trajectory": "purple", "slice": "teal"},
            {"ellipse": "crimson", "trajectory": "royalblue", "slice": "olive"},
            {"ellipse": "brown", "trajectory": "black", "slice": "limegreen"},
        ]

        plotter = pv.Plotter()
        plotter.add_axes()
        plotter.show_bounds(
            grid=True,
            location="outer",
            xtitle="X (m)",
            ytitle="Y (m)",
            ztitle="Altitude (m)",
            n_xlabels=5,
            n_ylabels=5,
            n_zlabels=5,
            fmt="%.0f",
            minor_ticks=False,
        )

        if overlay_patch:
            required_vars = {"longitude_f", "latitude_f", "altitude_f", "Y_del_f"}
            if required_vars.issubset(set(patch_table.variables) | set(patch_table.coords)):
                if patch_species in patch_table["species_out"].values:
                    patch_time_used = int(time_idx)
                    patch_t = patch_table.where(patch_table["time_idx"] == patch_time_used, drop=True)

                    if patch_t.sizes.get("row", 0) > 0:
                        patch_vals = patch_t["Y_del_f"].sel(species_out=patch_species).values
                        lon_f = patch_t["longitude_f"].values
                        lat_f = patch_t["latitude_f"].values
                        alt_f = patch_t["altitude_f"].values

                        valid = (
                            np.isfinite(patch_vals)
                            & np.isfinite(lon_f)
                            & np.isfinite(lat_f)
                            & np.isfinite(alt_f)
                            & (patch_vals > 0.0)
                        )
                    else:
                        patch_vals = np.array([], dtype=float)
                        lon_f = np.array([], dtype=float)
                        lat_f = np.array([], dtype=float)
                        alt_f = np.array([], dtype=float)
                        valid = np.array([], dtype=bool)

                    if not np.any(valid):
                        rows_time = np.asarray(patch_table["time_idx"].values)
                        vals_all = np.asarray(
                            patch_table["Y_del_f"].sel(species_out=patch_species).values,
                            dtype=float,
                        )
                        rows_with_data = rows_time[np.isfinite(vals_all) & (vals_all > 0.0)]

                        if rows_with_data.size > 0:
                            candidate_times = np.unique(rows_with_data.astype(int))
                            patch_time_used = int(
                                candidate_times[np.argmin(np.abs(candidate_times - int(time_idx)))]
                            )
                            patch_t = patch_table.where(
                                patch_table["time_idx"] == patch_time_used,
                                drop=True,
                            )
                            patch_vals = patch_t["Y_del_f"].sel(species_out=patch_species).values
                            lon_f = patch_t["longitude_f"].values
                            lat_f = patch_t["latitude_f"].values
                            alt_f = patch_t["altitude_f"].values
                            valid = (
                                np.isfinite(patch_vals)
                                & np.isfinite(lon_f)
                                & np.isfinite(lat_f)
                                & np.isfinite(alt_f)
                                & (patch_vals > 0.0)
                            )
                            print(
                                f"No nonzero patch_table values for {patch_species} at time_idx={time_idx}; "
                                f"using nearest available patch timestep time_idx={patch_time_used}."
                            )

                    if np.any(valid):
                        lon_m_f, lat_m_f = _to_plot_m(lon_f[valid], lat_f[valid])
                        dx_f, dy_f, dz_f = _fine_lengths_m(lat_f[valid])
                        cube_grid = _build_hex_grid(lon_m_f, lat_m_f, alt_f[valid], dx_f, dy_f, dz_f)
                        if cube_grid is not None:
                            patch_log10 = np.log10(np.maximum(patch_vals[valid], 1.0e-30))
                            cube_grid.cell_data["patch_log10"] = patch_log10
                            clim = [float(np.nanmin(patch_log10)), float(np.nanmax(patch_log10))]
                            if np.isclose(clim[0], clim[1]):
                                clim[1] = clim[0] + 1.0
                            plotter.add_mesh(
                                cube_grid,
                                scalars="patch_log10",
                                cmap="Greys",
                                opacity=patch_opacity,
                                clim=clim,
                                show_scalar_bar=True,
                                scalar_bar_args={"title": f"log10(Y_del_f {patch_species})"},
                            )
                    else:
                        print(
                            f"No nonzero patch_table values found for {patch_species} in this dataset."
                        )
                else:
                    print(f"Species {patch_species} is unavailable in patch_table species_out.")
            else:
                print("patch_table is missing fine-grid coordinate variables. Re-run GPAT after rebuilding boxm.f90 to populate latitude_f/longitude_f/altitude_f.")

        for flight_idx, fid in enumerate(flight_ids_t):
            style = flight_styles[flight_idx % len(flight_styles)]
            centres_m = []
            ellipses_m = []
            slice_polys_m = []
            seg_ids_f = (
                pl_ds
                .where((pl_ds["time_idx"] == time_idx) & (pl_ds["flight_id"] == fid), drop=True)["seg_id"]
                .values
            )
            seg_ids_f = np.unique(seg_ids_f)

            for seg_id in seg_ids_f:
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

                row_center = np.array(
                    [
                        row["longitude_m"].item(),
                        row["latitude_m"].item(),
                        row["altitude"].item(),
                    ],
                    dtype=float,
                )
                # Some terminal segments carry plume geometry but invalid source-center metadata.
                # Skip them so ribbons/trajectories end on the last physically consistent segment.
                if not np.all(np.isfinite(row_center)):
                    continue

                if row_out["ellipses_m"] is not None and hasattr(row_out["ellipses_m"], "__len__") and len(row_out["ellipses_m"]) > 0:
                    ellipse_vals = row_out["ellipses_m"].values
                    centre_m = _ellipse_center_m(ellipse_vals, row)

                    if np.all(np.isfinite(centre_m)):
                        centres_m.append(centre_m.tolist())
                        ellipses_m.append(ellipse_vals)
                        slice_polys_m.append(row_out["slice_polys_m"].values)

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
                plotter.add_mesh(mesh, color=style["ellipse"], opacity=0.5)

            if len(centres_m) > 1:
                trajectory = np.array(centres_m)
                valid_mask = np.all(np.isfinite(trajectory), axis=1)
                trajectory = trajectory[valid_mask]

                if trajectory.shape[0] > 1:
                    plotter.add_lines(trajectory, color=style["trajectory"], width=3, connected=True)

                if slice_polys_m.ndim == 4 and slice_polys_m.shape[0] > 1:
                    num_segments = slice_polys_m.shape[0]
                    num_slices = slice_polys_m.shape[1]

                    if len(ellipses_m) == num_segments:
                        for slice_idx in range(num_slices):
                            for seg_idx in range(num_segments - 1):
                                poly1 = slice_polys_m[seg_idx, slice_idx]
                                poly2 = slice_polys_m[seg_idx + 1, slice_idx]

                                if (
                                    poly1.shape[0] >= 3
                                    and poly2.shape[0] >= 3
                                    and np.all(np.isfinite(poly1))
                                    and np.all(np.isfinite(poly2))
                                    and np.linalg.norm(poly1) > 1.0
                                    and np.linalg.norm(poly2) > 1.0
                                ):
                                    n_corners = poly1.shape[0]
                                    points = np.vstack([poly1, poly2])
                                    faces = []
                                    for corner_idx in range(n_corners):
                                        next_corner = (corner_idx + 1) % n_corners
                                        faces.extend([4, corner_idx, next_corner, n_corners + next_corner, n_corners + corner_idx])
                                    faces = np.array(faces)
                                    mesh = pv.PolyData(points, faces)
                                    plotter.add_mesh(mesh, color=style["slice"], opacity=0.3)

            plotter.enable_parallel_projection()
            plotter.camera_position = "iso"
        flight_note = "all flights" if flight_ids is None else f"flight_ids={list(np.atleast_1d(flight_ids))}"
        plotter.add_text(
            f"Flight Plumes at time_idx={time_idx} ({flight_note}, projected meters)\\nPer-flight colors: ellipse / trajectory / slice",
            position="upper_left",
            font_size=10,
            color="black",
        )

        plotter.show(
            interactive=True,
            auto_close=False,
            full_screen=False,
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


