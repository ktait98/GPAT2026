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
        self.patch_plot_min_value = None
        self.patch_plot_max_log10_span = 3.0

    def set_patch_plot_policy(self, min_value=None, max_log10_span=3.0):
        """Configure the default fine-grid patch filtering used across plots.

        Parameters
        ----------
        min_value : float or None, optional
            Absolute lower bound in mol/m^3 for rendering patch cells.
            ``None`` disables the absolute floor.
        max_log10_span : float or None, default 3.0
            Maximum number of log10 decades below the species peak to render.
            For example, ``3.0`` keeps values down to ``peak * 1e-3``.
            ``None`` disables the relative floor.
        """
        if min_value is not None:
            min_value = float(min_value)
            if min_value <= 0.0:
                raise ValueError("min_value must be positive when provided.")

        if max_log10_span is not None:
            max_log10_span = float(max_log10_span)
            if max_log10_span <= 0.0:
                raise ValueError("max_log10_span must be positive when provided.")

        self.patch_plot_min_value = min_value
        self.patch_plot_max_log10_span = max_log10_span

    def _resolve_patch_plot_floor(self, values, min_value=None, max_log10_span=None):
        """Return the effective patch rendering floor for the provided values."""
        vals = np.asarray(values, dtype=float)
        pos = vals[np.isfinite(vals) & (vals > 0.0)]
        if pos.size == 0:
            return None

        floor_candidates = []

        min_value_use = self.patch_plot_min_value if min_value is None else min_value
        if min_value_use is not None:
            min_value_use = float(min_value_use)
            if min_value_use > 0.0:
                floor_candidates.append(min_value_use)

        max_log10_span_use = (
            self.patch_plot_max_log10_span if max_log10_span is None else max_log10_span
        )
        if max_log10_span_use is not None:
            max_log10_span_use = float(max_log10_span_use)
            if max_log10_span_use > 0.0:
                floor_candidates.append(float(np.nanmax(pos)) * 10.0 ** (-max_log10_span_use))

        if not floor_candidates:
            return None

        return max(floor_candidates)

    def _patch_value_mask(self, values, min_value=None, max_log10_span=None):
        """Return a finite positive mask filtered by the active patch plotting floor."""
        vals = np.asarray(values, dtype=float)
        mask = np.isfinite(vals) & (vals > 0.0)
        floor = self._resolve_patch_plot_floor(vals, min_value=min_value, max_log10_span=max_log10_span)
        if floor is not None:
            mask &= vals >= floor
        return mask, floor

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
                    vals_t = vals_t[self._patch_value_mask(vals_t)[0]]
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
        mask, plot_floor = self._patch_value_mask(vals)

        if not np.any(mask):
            raise ValueError(
                f"No patch_table values found for species={species} at time_idx={time_idx} above the active plotting threshold."
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
        cbar_label = f"{species} Y_del_f"
        if plot_floor is not None:
            cbar_label += f"\n(filtered: >= {plot_floor:.1e} mol/m^3)"
        fig.colorbar(mesh, ax=ax, label=cbar_label)

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

    def plot_patch_heatmap_3d(
        self,
        time_idx,
        job_id=None,
        species="NO",
        color_scale=None,
        cmap="viridis",
        opacity=0.35,
        show_edges=False,
        orbit=True,
        orbit_n_points=72,
        orbit_factor=2.5,
        off_screen=None,
        screenshot_path=None,
        window_size=(1400, 900),
        _show=True,
    ):
        """Plot fine-grid patch values as a 3D PyVista grid.

        Parameters
        ----------
        time_idx : int
            Simulation timestep index.
        job_id : str, optional
            GPAT job ID. Defaults to the first loaded job.
        species : str, default "NO"
            Species in ``species_out`` to visualize.
        color_scale : tuple[float, float], optional
            Explicit ``(vmin, vmax)`` color scale.
        cmap : str, default "viridis"
            Colormap for rendering.
        opacity : float, default 0.35
            Patch opacity.
        show_edges : bool, default False
            Whether to render cell edges.
        orbit : bool, default True
            Enable camera orbit animation.
        orbit_n_points : int, default 72
            Number of points in the orbital camera path.
        orbit_factor : float, default 2.5
            Radius factor for orbital camera path.
        off_screen : bool, optional
            Force PyVista off-screen rendering. If ``None``, auto-detected from
            the environment (headless sessions use off-screen mode).
        screenshot_path : str, optional
            Output path for a screenshot in off-screen mode.
        window_size : tuple[int, int], default (1400, 900)
            Plotter window size.
        _show : bool, default True
            Whether to display the figure immediately.

        Returns
        -------
        pyvista.UnstructuredGrid
            3D cell grid with scalar data ``Y_del_f_{species}``.
        """
        if job_id is None:
            if not self.pp_gpat.job_ids:
                raise ValueError("No GPAT jobs are loaded.")
            job_id = self.pp_gpat.job_ids[0]

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

        hres_sim_c = _extract_sim_value(sim_raw, "hres_sim_c")
        hres_sim_f = _extract_sim_value(sim_raw, "hres_sim_f")
        vres_sim_c = _extract_sim_value(sim_raw, "vres_sim_c")
        vres_sim_f = _extract_sim_value(sim_raw, "vres_sim_f")

        time_rows = np.asarray(patch_table["time_idx"].values, dtype=int)
        row_sel = np.flatnonzero(time_rows == int(time_idx))
        if row_sel.size == 0:
            raise ValueError(f"No patch_table rows found for time_idx={time_idx}.")

        patch_t = patch_table.isel(row=row_sel)
        vals = np.asarray(patch_t["Y_del_f"].sel(species_out=species).values, dtype=float)
        mask, plot_floor = self._patch_value_mask(vals)
        if not np.any(mask):
            raise ValueError(
                f"No patch_table values found for species={species} at time_idx={time_idx} above the active plotting threshold."
            )

        required_fine = {"latitude_f", "longitude_f", "altitude_f"}
        if required_fine.issubset(set(patch_t.variables) | set(patch_t.coords)):
            lat_all = np.asarray(patch_t["latitude_f"].values, dtype=float)
            lon_all = np.asarray(patch_t["longitude_f"].values, dtype=float)
            alt_all = np.asarray(patch_t["altitude_f"].values, dtype=float)
        else:
            nx_f = max(1, int(np.rint(hres_sim_c / hres_sim_f)))
            ny_f = nx_f
            nz_f = max(1, int(np.rint(vres_sim_c / vres_sim_f)))

            row_cell_c = np.asarray(patch_t["row_cell_c"].values, dtype=int)
            row_cell_f = np.asarray(patch_t["row_cell_f"].values, dtype=int)

            rem_xy = (row_cell_f - 1) % (nx_f * ny_f)
            ix_f = rem_xy % nx_f + 1
            iy_f = rem_xy // nx_f + 1
            iz_f = ((row_cell_f - 1) // (nx_f * ny_f)) % nz_f + 1

            hres_f_deg = hres_sim_c / nx_f
            vres_f_m = vres_sim_c / nz_f

            lat_c = np.asarray(boxm_ds["latitude_c"].values, dtype=float)[row_cell_c - 1]
            lon_c = np.asarray(boxm_ds["longitude_c"].values, dtype=float)[row_cell_c - 1]
            alt_c = np.asarray(boxm_ds["altitude_c"].values, dtype=float)[row_cell_c - 1]

            lat_all = lat_c - 0.5 * hres_sim_c + (iy_f - 0.5) * hres_f_deg
            lon_all = lon_c - 0.5 * hres_sim_c + (ix_f - 0.5) * hres_f_deg
            alt_all = alt_c - 0.5 * vres_sim_c + (iz_f - 0.5) * vres_f_m

        lat_plot = lat_all[mask]
        lon_plot = lon_all[mask]
        alt_plot = alt_all[mask]
        val_plot = vals[mask]

        valid_xyz = (
            np.isfinite(lat_plot)
            & np.isfinite(lon_plot)
            & np.isfinite(alt_plot)
            & np.isfinite(val_plot)
        )
        if not np.any(valid_xyz):
            raise ValueError("No finite patch coordinates/values available for plotting.")

        lat_plot = lat_plot[valid_xyz]
        lon_plot = lon_plot[valid_xyz]
        alt_plot = alt_plot[valid_xyz]
        val_plot = val_plot[valid_xyz]

        if color_scale is None:
            vmin = float(np.nanmin(val_plot))
            vmax = float(np.nanmax(val_plot))
        else:
            vmin, vmax = float(color_scale[0]), float(color_scale[1])

        nx_f = max(1, int(np.rint(hres_sim_c / hres_sim_f)))
        nz_f = max(1, int(np.rint(vres_sim_c / vres_sim_f)))
        hres_f_deg = hres_sim_c / nx_f
        vres_f_m = vres_sim_c / nz_f

        ref_lon = float(np.nanmin(np.asarray(boxm_ds["longitude_c"].values, dtype=float)))
        ref_lat = float(np.nanmin(np.asarray(boxm_ds["latitude_c"].values, dtype=float)))

        x_m, y_m = lonlat_to_m(lon_plot, lat_plot, ref_lon, ref_lat)

        earth_radius_m = 6371000.0
        dx_m = earth_radius_m * np.cos(np.deg2rad(lat_plot)) * np.deg2rad(hres_f_deg)
        dy_m = np.full_like(dx_m, earth_radius_m * np.deg2rad(hres_f_deg), dtype=float)
        dz_m = np.full_like(dx_m, vres_f_m, dtype=float)

        centers = np.column_stack([x_m, y_m, alt_plot])
        half = 0.5 * np.column_stack([dx_m, dy_m, dz_m])
        offsets = np.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ]
        )

        points = (centers[:, None, :] + offsets[None, :, :] * half[:, None, :]).reshape(-1, 3)
        n_cells = centers.shape[0]
        cells = np.hstack(
            [
                np.full((n_cells, 1), 8, dtype=np.int64),
                np.arange(n_cells * 8, dtype=np.int64).reshape(n_cells, 8),
            ]
        ).ravel()
        celltypes = np.full(n_cells, pv.CellType.HEXAHEDRON, dtype=np.uint8)
        patch_grid = pv.UnstructuredGrid(cells, celltypes, points)
        patch_grid.cell_data[f"Y_del_f_{species}"] = val_plot

        if _show:
            env_off_screen = (
                str(os.environ.get("PYVISTA_OFF_SCREEN", "")).strip().lower()
                in {"1", "true", "yes", "on"}
            )
            no_display = os.environ.get("DISPLAY") is None and os.environ.get("WAYLAND_DISPLAY") is None
            off_screen_use = bool(off_screen) if off_screen is not None else (env_off_screen or no_display)

            if off_screen_use:
                pv.OFF_SCREEN = True

            plotter = pv.Plotter(off_screen=off_screen_use, window_size=window_size)
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
            plotter.add_mesh(
                patch_grid,
                scalars=f"Y_del_f_{species}",
                cmap=cmap,
                opacity=opacity,
                clim=[vmin, vmax],
                show_edges=show_edges,
                show_scalar_bar=True,
                scalar_bar_args={
                    "title": (
                        f"{species} Y_del_f"
                        if plot_floor is None
                        else f"{species} Y_del_f\n>= {plot_floor:.1e} mol/m^3"
                    )
                },
            )
            plotter.camera_position = "iso"

            if orbit and not off_screen_use:
                try:
                    path = plotter.generate_orbital_path(
                        n_points=max(12, int(orbit_n_points)),
                        factor=float(orbit_factor),
                        viewup=[0.0, 0.0, 1.0],
                    )
                except TypeError:
                    path = plotter.generate_orbital_path(
                        n_points=max(12, int(orbit_n_points)),
                        factor=float(orbit_factor),
                    )

                try:
                    plotter.orbit_on_path(path, write_frames=False, threaded=True)
                except TypeError:
                    plotter.orbit_on_path(path, write_frames=False)

            plotter.add_text(
                f"Patch 3D: species={species}, time_idx={time_idx}",
                position="upper_left",
                font_size=10,
                color="black",
            )

            if off_screen_use:
                if screenshot_path is None:
                    screenshot_path = (
                        f"patch_3d_{job_id}_{species}_time{int(time_idx)}.png"
                    )
                plotter.show(screenshot=screenshot_path, auto_close=True)
                print(f"Saved off-screen 3D patch image to: {screenshot_path}")
            else:
                plotter.show(interactive=True, auto_close=False, full_screen=False)

        return patch_grid

    def plot_plumes_3d_pv(
        self,
        job_id,
        time_idx=184,
        overlay_patch=False,
        patch_species="NO",
        patch_opacity=0.18,
        patch_cmap="plasma_r",
        z_exaggeration=1.0,
        axis_font_size=13,
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

        def _build_hex_grid(xc, yc, zc, dx, dy, dz, z_scale=1.0):
            centers = np.column_stack([xc, yc, np.asarray(zc) * z_scale])
            if centers.size == 0:
                return None

            half = 0.5 * np.column_stack([dx, dy, np.asarray(dz) * z_scale])
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
        pl_time_rows = np.asarray(pl_ds["time_idx"].values, dtype=int)
        pl_fid_rows = np.asarray(pl_ds["flight_id"].values)
        pl_seg_rows = np.asarray(pl_ds["seg_id"].values, dtype=int)

        pl_out_time_rows = np.asarray(pl_out["time_idx"].values, dtype=int)
        pl_out_fid_rows = np.asarray(pl_out["flight_id"].values)
        pl_out_seg_rows = np.asarray(pl_out["seg_id"].values, dtype=int)

        patch_time_rows = np.asarray(patch_table["time_idx"].values, dtype=int)

        def _patch_subset_for_time(t_idx: int):
            row_sel = np.flatnonzero(patch_time_rows == int(t_idx))
            if row_sel.size == 0:
                return patch_table.isel(row=slice(0, 0))
            return patch_table.isel(row=row_sel)

        time_sel_pl = np.flatnonzero(pl_time_rows == int(time_idx))
        time_sel_out = np.flatnonzero(pl_out_time_rows == int(time_idx))
        if time_sel_pl.size == 0 or time_sel_out.size == 0:
            raise ValueError(
                f"time_idx={time_idx} is not available in plume outputs. "
                f"pl_ds time_idx range is {int(pl_time_rows.min())}-{int(pl_time_rows.max())}; "
                f"pl_out time_idx range is {int(pl_out_time_rows.min())}-{int(pl_out_time_rows.max())}."
            )

        time_pos_pl = int(time_sel_pl[0])
        time_pos_out = int(time_sel_out[0])
        flight_ids_t = np.unique(pl_fid_rows)
        if flight_ids is not None:
            flight_ids_filter = np.atleast_1d(flight_ids)
            flight_ids_t = np.array([fid for fid in flight_ids_t if fid in flight_ids_filter])

        flight_styles = [
            {"ellipse": "firebrick", "trajectory": "navy", "slice": "seagreen"},
            {"ellipse": "darkorange", "trajectory": "purple", "slice": "teal"},
            {"ellipse": "crimson", "trajectory": "royalblue", "slice": "olive"},
            {"ellipse": "brown", "trajectory": "black", "slice": "limegreen"},
        ]

        _zs = float(z_exaggeration) if (z_exaggeration and float(z_exaggeration) > 0.0) else 1.0
        ztitle = "Altitude (m)" if np.isclose(_zs, 1.0) else f"Altitude (m) [×{_zs:.0f} vert. exag.]"
        zlabel_count = 3 if np.isclose(_zs, 1.0) else 5

        plotter = pv.Plotter(window_size=(1600, 1000))
        plotter.add_axes()
        plotter.show_bounds(
            grid=True,
            location="outer",
            xtitle="X (m)",
            ytitle="Y (m)",
            ztitle=ztitle,
            n_xlabels=4,
            n_ylabels=4,
            n_zlabels=zlabel_count,
            font_size=int(axis_font_size),
            fmt="%.0f",
            ticks="outside",
            minor_ticks=False,
        )

        if overlay_patch:
            required_vars = {"longitude_f", "latitude_f", "altitude_f", "Y_del_f"}
            if required_vars.issubset(set(patch_table.variables) | set(patch_table.coords)):
                if patch_species in patch_table["species_out"].values:
                    patch_time_used = int(time_idx)
                    patch_t = _patch_subset_for_time(patch_time_used)

                    if patch_t.sizes.get("row", 0) > 0:
                        patch_vals = patch_t["Y_del_f"].sel(species_out=patch_species).values
                        lon_f = patch_t["longitude_f"].values
                        lat_f = patch_t["latitude_f"].values
                        alt_f = patch_t["altitude_f"].values

                        value_mask, plot_floor = self._patch_value_mask(patch_vals)
                        valid = (
                            np.isfinite(patch_vals)
                            & np.isfinite(lon_f)
                            & np.isfinite(lat_f)
                            & np.isfinite(alt_f)
                            & value_mask
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
                            patch_t = _patch_subset_for_time(patch_time_used)
                            patch_vals = patch_t["Y_del_f"].sel(species_out=patch_species).values
                            lon_f = patch_t["longitude_f"].values
                            lat_f = patch_t["latitude_f"].values
                            alt_f = patch_t["altitude_f"].values
                            value_mask, plot_floor = self._patch_value_mask(patch_vals)
                            valid = (
                                np.isfinite(patch_vals)
                                & np.isfinite(lon_f)
                                & np.isfinite(lat_f)
                                & np.isfinite(alt_f)
                                & value_mask
                            )
                            print(
                                f"No patch_table values for {patch_species} at time_idx={time_idx} above the active plotting threshold; "
                                f"using nearest available patch timestep time_idx={patch_time_used}."
                            )

                    if np.any(valid):
                        lon_m_f, lat_m_f = _to_plot_m(lon_f[valid], lat_f[valid])
                        dx_f, dy_f, dz_f = _fine_lengths_m(lat_f[valid])
                        cube_grid = _build_hex_grid(lon_m_f, lat_m_f, alt_f[valid], dx_f, dy_f, dz_f, z_scale=_zs)
                        if cube_grid is not None:
                            patch_vals_plot = patch_vals[valid].astype(float)
                            cube_grid.cell_data["patch_vals"] = patch_vals_plot
                            clim = [float(np.nanmin(patch_vals_plot)), float(np.nanmax(patch_vals_plot))]
                            if clim[1] <= clim[0]:
                                pad = max(abs(clim[0]) * 0.05, 1.0e-30)
                                clim = [max(0.0, clim[0] - pad), clim[1] + pad]
                            plotter.add_mesh(
                                cube_grid,
                                scalars="patch_vals",
                                cmap=patch_cmap,
                                opacity=patch_opacity,
                                clim=clim,
                                show_scalar_bar=True,
                                scalar_bar_args={
                                    "title": (
                                        f"Y_del_f {patch_species}"
                                        if plot_floor is None
                                        else f"Y_del_f {patch_species}\n>= {plot_floor:.1e} mol/m^3"
                                    ),
                                    "fmt": "%.1e",
                                    "n_labels": 5,
                                },
                            )
                    else:
                        print(
                            f"No patch_table values found for {patch_species} in this dataset above the active plotting threshold."
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
            seg_ids_f = pl_seg_rows[pl_fid_rows == fid]
            seg_ids_f = np.unique(seg_ids_f)

            for seg_id in seg_ids_f:
                seg_sel = np.flatnonzero((pl_seg_rows == int(seg_id)) & (pl_fid_rows == fid))
                seg_out_sel = np.flatnonzero((pl_out_seg_rows == int(seg_id)) & (pl_out_fid_rows == fid))

                if seg_sel.size == 0 or seg_out_sel.size == 0:
                    continue

                row = pl_ds.isel(seg_id=int(seg_sel[0]), time=time_pos_pl).squeeze(drop=True)
                row_out = pl_out.isel(seg_id=int(seg_out_sel[0]), time=time_pos_out).squeeze(drop=True)

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

            if _zs != 1.0:
                if centres_m.ndim == 2 and centres_m.shape[1] == 3:
                    centres_m[:, 2] *= _zs
                if ellipses_m.ndim == 3 and ellipses_m.shape[2] == 3:
                    ellipses_m[:, :, 2] *= _zs
                if slice_polys_m.ndim == 4 and slice_polys_m.shape[3] == 3:
                    slice_polys_m[:, :, :, 2] *= _zs

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
            f"Flight Plumes at time_idx={time_idx} ({flight_note}, projected meters)\nPer-flight colors: ellipse / trajectory / slice",
            position="upper_left",
            font_size=10,
            color="black",
        )

        plotter.show(
            interactive=True,
            auto_close=False,
            full_screen=False,
        )

    def animate_plumes_3d_pv(
        self,
        job_id,
        time_idx_start=None,
        time_idx_end=None,
        frame_stride=1,
        output_path="plumes_animation.gif",
        fps=5,
        overlay_patch=False,
        patch_species="NO",
        patch_opacity=0.18,
        patch_cmap="plasma_r",
        patch_clim=None,
        z_exaggeration=1.0,
        axis_font_size=13,
        flight_ids=None,
    ):
        """Render a 3D animation of flight plumes evolving over time.

        Saves a GIF (default) or MP4 (inferred from *output_path* extension)
        that shows plume ellipses, slice polyhedra and, optionally, the
        fine-grid patch concentration overlay for each animated time step.

        Parameters
        ----------
        job_id : str
            GPAT job identifier.
        time_idx_start : int, optional
            First time-step index to include.  Defaults to the earliest
            available index in *pl_ds*.
        time_idx_end : int, optional
            Last time-step index to include (inclusive).  Defaults to the
            latest available index in *pl_ds*.
        frame_stride : int
            Step through the sorted list of available time indices.
            ``frame_stride=1`` (default) animates every available step;
            ``frame_stride=2`` uses every second step, etc.
        output_path : str
            Output file path.  Use ``.gif`` for a looping GIF or ``.mp4``
            for an MP4 video.
        fps : int
            Frames per second in the output file.
        overlay_patch : bool
            Whether to render the fine-grid concentration overlay.
        patch_species : str
            Species name for the patch overlay (must exist in
            ``species_out``).
        patch_opacity : float
            Opacity of patch voxels (0–1).
        patch_cmap : str
            Matplotlib colormap name for the patch overlay.
        patch_clim : tuple[float, float] or None
            Fixed ``(vmin, vmax)`` for the patch colorbar.  If *None*, the
            range is computed from the full patch dataset so the colour
            scale is consistent across all frames.
        z_exaggeration : float
            Vertical exaggeration factor (1.0 = true scale).
        axis_font_size : int
            Font size for axis tick labels.
        flight_ids : array-like or None
            Restrict the animation to these flight IDs.  *None* animates
            all flights.
        """
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
            lon_m, lat_m = lonlat_to_m(lon_deg, lat_deg, proj_ref_lon, proj_ref_lat)
            return lon_m, lat_m

        def _fine_lengths_m(lat_deg):
            lat_arr = np.asarray(lat_deg, dtype=float)
            earth_radius_m = 6371000.0
            dx_m = earth_radius_m * np.cos(np.deg2rad(lat_arr)) * np.deg2rad(hres_f_deg)
            dy_m = np.full_like(lat_arr, earth_radius_m * np.deg2rad(hres_f_deg), dtype=float)
            dz_m = np.full_like(lat_arr, vres_f_m, dtype=float)
            return dx_m, dy_m, dz_m

        def _build_hex_grid(xc, yc, zc, dx, dy, dz, z_scale=1.0):
            centers = np.column_stack([xc, yc, np.asarray(zc) * z_scale])
            if centers.size == 0:
                return None
            half = 0.5 * np.column_stack([dx, dy, np.asarray(dz) * z_scale])
            offsets = np.array([
                [-1.0, -1.0, -1.0], [ 1.0, -1.0, -1.0],
                [ 1.0,  1.0, -1.0], [-1.0,  1.0, -1.0],
                [-1.0, -1.0,  1.0], [ 1.0, -1.0,  1.0],
                [ 1.0,  1.0,  1.0], [-1.0,  1.0,  1.0],
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

        # --- Index arrays ---
        pl_time_rows = np.asarray(pl_ds["time_idx"].values, dtype=int)
        pl_fid_rows = np.asarray(pl_ds["flight_id"].values)
        pl_seg_rows = np.asarray(pl_ds["seg_id"].values, dtype=int)

        pl_out_time_rows = np.asarray(pl_out["time_idx"].values, dtype=int)
        pl_out_fid_rows = np.asarray(pl_out["flight_id"].values)
        pl_out_seg_rows = np.asarray(pl_out["seg_id"].values, dtype=int)

        patch_time_rows = np.asarray(patch_table["time_idx"].values, dtype=int)

        def _patch_subset_for_time(t_idx):
            row_sel = np.flatnonzero(patch_time_rows == int(t_idx))
            if row_sel.size == 0:
                return patch_table.isel(row=slice(0, 0))
            return patch_table.isel(row=row_sel)

        # --- Determine time steps to animate ---
        # Only animate timesteps present in BOTH pl_ds and pl_out
        common_times = np.intersect1d(np.unique(pl_time_rows), np.unique(pl_out_time_rows))
        if time_idx_start is not None:
            common_times = common_times[common_times >= int(time_idx_start)]
        if time_idx_end is not None:
            common_times = common_times[common_times <= int(time_idx_end)]
        common_times = common_times[::max(1, int(frame_stride))]

        if common_times.size == 0:
            raise ValueError(
                "No time steps available for animation with the given parameters. "
                f"pl_ds time_idx range: {int(pl_time_rows.min())}-{int(pl_time_rows.max())}."
            )

        print(f"Animating {common_times.size} frames: time_idx {common_times[0]}–{common_times[-1]}")

        # --- Global flight-id list ---
        flight_ids_all = np.unique(pl_fid_rows)
        if flight_ids is not None:
            flight_ids_filter = np.atleast_1d(flight_ids)
            flight_ids_all = np.array([fid for fid in flight_ids_all if fid in flight_ids_filter])

        flight_styles = [
            {"ellipse": "firebrick",   "trajectory": "navy",      "slice": "seagreen"},
            {"ellipse": "darkorange",  "trajectory": "purple",    "slice": "teal"},
            {"ellipse": "crimson",     "trajectory": "royalblue", "slice": "olive"},
            {"ellipse": "brown",       "trajectory": "black",     "slice": "limegreen"},
        ]

        _zs = float(z_exaggeration) if (z_exaggeration and float(z_exaggeration) > 0.0) else 1.0
        ztitle = "Altitude (m)" if np.isclose(_zs, 1.0) else f"Altitude (m) [×{_zs:.0f} vert. exag.]"
        zlabel_count = 3 if np.isclose(_zs, 1.0) else 5
        flight_note = "all flights" if flight_ids is None else f"flight_ids={list(np.atleast_1d(flight_ids))}"

        # --- Pre-compute global patch clim for colour consistency ---
        global_patch_clim = patch_clim  # may be None or user-supplied tuple
        if overlay_patch and global_patch_clim is None:
            required_vars = {"longitude_f", "latitude_f", "altitude_f", "Y_del_f"}
            if (required_vars.issubset(set(patch_table.variables) | set(patch_table.coords))
                    and patch_species in patch_table["species_out"].values):
                all_patch_vals = np.asarray(
                    patch_table["Y_del_f"].sel(species_out=patch_species).values,
                    dtype=float,
                )
                finite_pos = all_patch_vals[self._patch_value_mask(all_patch_vals)[0]]
                if finite_pos.size > 0:
                    global_patch_clim = (float(finite_pos.min()), float(finite_pos.max()))
                    print(f"Global patch clim for {patch_species}: {global_patch_clim[0]:.3e} – {global_patch_clim[1]:.3e}")

        # --- Pre-compute scene bounds for stable axes across frames ---
        # Use all plume centroid positions across every timestep and flight.
        lon_m_all, lat_m_all, alt_all = [], [], []
        for t_idx in common_times:
            t_pos = int(np.flatnonzero(pl_time_rows == int(t_idx))[0])
            for fid in flight_ids_all:
                seg_ids_f = np.unique(pl_seg_rows[pl_fid_rows == fid])
                for seg_id in seg_ids_f:
                    seg_sel = np.flatnonzero((pl_seg_rows == int(seg_id)) & (pl_fid_rows == fid))
                    if seg_sel.size == 0:
                        continue
                    row = pl_ds.isel(seg_id=int(seg_sel[0]), time=t_pos).squeeze(drop=True)
                    lon_m = float(row["longitude_m"].item()) if np.isfinite(row["longitude_m"].item()) else np.nan
                    lat_m = float(row["latitude_m"].item()) if np.isfinite(row["latitude_m"].item()) else np.nan
                    alt   = float(row["altitude"].item())   if np.isfinite(row["altitude"].item())   else np.nan
                    lon_m_all.append(lon_m)
                    lat_m_all.append(lat_m)
                    alt_all.append(alt * _zs)

        lon_m_all = np.array(lon_m_all, dtype=float)
        lat_m_all = np.array(lat_m_all, dtype=float)
        alt_all   = np.array(alt_all,   dtype=float)

        finite_mask = np.isfinite(lon_m_all) & np.isfinite(lat_m_all) & np.isfinite(alt_all)
        if not np.any(finite_mask):
            raise ValueError("No finite plume positions found for the requested time range.")

        pad_xy = (np.ptp(lon_m_all[finite_mask]) + np.ptp(lat_m_all[finite_mask])) * 0.05
        pad_z  = max(np.ptp(alt_all[finite_mask]) * 0.1, 500.0 * _zs)
        scene_bounds = (
            float(lon_m_all[finite_mask].min()) - pad_xy,
            float(lon_m_all[finite_mask].max()) + pad_xy,
            float(lat_m_all[finite_mask].min()) - pad_xy,
            float(lat_m_all[finite_mask].max()) + pad_xy,
            float(alt_all[finite_mask].min())   - pad_z,
            float(alt_all[finite_mask].max())   + pad_z,
        )

        # Also extend bounds to cover patch overlay cells if present
        if overlay_patch and global_patch_clim is not None:
            required_vars = {"longitude_f", "latitude_f", "altitude_f", "Y_del_f"}
            if required_vars.issubset(set(patch_table.variables) | set(patch_table.coords)):
                lon_f_all = np.asarray(patch_table["longitude_f"].values, dtype=float)
                lat_f_all = np.asarray(patch_table["latitude_f"].values, dtype=float)
                alt_f_all = np.asarray(patch_table["altitude_f"].values, dtype=float)
                fv = np.isfinite(lon_f_all) & np.isfinite(lat_f_all) & np.isfinite(alt_f_all)
                if np.any(fv):
                    lm, latm = _to_plot_m(lon_f_all[fv], lat_f_all[fv])
                    scene_bounds = (
                        min(scene_bounds[0], float(lm.min())   - pad_xy),
                        max(scene_bounds[1], float(lm.max())   + pad_xy),
                        min(scene_bounds[2], float(latm.min()) - pad_xy),
                        max(scene_bounds[3], float(latm.max()) + pad_xy),
                        min(scene_bounds[4], float(alt_f_all[fv].min()) * _zs - pad_z),
                        max(scene_bounds[5], float(alt_f_all[fv].max()) * _zs + pad_z),
                    )

        # --- Set up plotter ---
        ext = os.path.splitext(output_path)[1].lower()
        plotter = pv.Plotter(off_screen=True, window_size=(1600, 1000))
        if ext == ".mp4":
            plotter.open_movie(output_path, framerate=fps)
        else:
            plotter.open_gif(output_path, fps=fps)

        camera_set = False

        # --- Frame loop ---
        for frame_num, t_idx in enumerate(common_times):
            plotter.clear_actors()

            time_sel_pl  = np.flatnonzero(pl_time_rows     == int(t_idx))
            time_sel_out = np.flatnonzero(pl_out_time_rows == int(t_idx))
            if time_sel_pl.size == 0 or time_sel_out.size == 0:
                continue

            time_pos_pl  = int(time_sel_pl[0])
            time_pos_out = int(time_sel_out[0])

            # ---- Patch overlay ----
            if overlay_patch:
                required_vars = {"longitude_f", "latitude_f", "altitude_f", "Y_del_f"}
                if (required_vars.issubset(set(patch_table.variables) | set(patch_table.coords))
                        and patch_species in patch_table["species_out"].values):
                    patch_t = _patch_subset_for_time(int(t_idx))
                    if patch_t.sizes.get("row", 0) > 0:
                        patch_vals = patch_t["Y_del_f"].sel(species_out=patch_species).values
                        lon_f  = patch_t["longitude_f"].values
                        lat_f  = patch_t["latitude_f"].values
                        alt_f  = patch_t["altitude_f"].values
                        value_mask, plot_floor = self._patch_value_mask(patch_vals)
                        valid  = (
                            np.isfinite(patch_vals) & np.isfinite(lon_f)
                            & np.isfinite(lat_f) & np.isfinite(alt_f)
                            & value_mask
                        )
                        if np.any(valid):
                            lon_m_f, lat_m_f = _to_plot_m(lon_f[valid], lat_f[valid])
                            dx_f, dy_f, dz_f = _fine_lengths_m(lat_f[valid])
                            cube_grid = _build_hex_grid(
                                lon_m_f, lat_m_f, alt_f[valid], dx_f, dy_f, dz_f, z_scale=_zs
                            )
                            if cube_grid is not None:
                                patch_vals_plot = patch_vals[valid].astype(float)
                                cube_grid.cell_data["patch_vals"] = patch_vals_plot
                                if global_patch_clim is not None:
                                    clim = list(global_patch_clim)
                                else:
                                    clim = [float(np.nanmin(patch_vals_plot)), float(np.nanmax(patch_vals_plot))]
                                    if clim[1] <= clim[0]:
                                        pad = max(abs(clim[0]) * 0.05, 1.0e-30)
                                        clim = [max(0.0, clim[0] - pad), clim[1] + pad]
                                plotter.add_mesh(
                                    cube_grid,
                                    scalars="patch_vals",
                                    cmap=patch_cmap,
                                    opacity=patch_opacity,
                                    clim=clim,
                                    show_scalar_bar=True,
                                    scalar_bar_args={
                                        "title": (
                                            f"Y_del_f {patch_species}"
                                            if plot_floor is None
                                            else f"Y_del_f {patch_species}\n>= {plot_floor:.1e} mol/m^3"
                                        ),
                                        "fmt": "%.1e",
                                        "n_labels": 5,
                                    },
                                )

            # ---- Flight plumes ----
            for flight_idx, fid in enumerate(flight_ids_all):
                style = flight_styles[flight_idx % len(flight_styles)]
                centres_m    = []
                ellipses_m   = []
                slice_polys_m = []
                seg_ids_f = np.unique(pl_seg_rows[pl_fid_rows == fid])

                for seg_id in seg_ids_f:
                    seg_sel     = np.flatnonzero((pl_seg_rows     == int(seg_id)) & (pl_fid_rows     == fid))
                    seg_out_sel = np.flatnonzero((pl_out_seg_rows == int(seg_id)) & (pl_out_fid_rows == fid))
                    if seg_sel.size == 0 or seg_out_sel.size == 0:
                        continue

                    row     = pl_ds.isel(seg_id=int(seg_sel[0]),     time=time_pos_pl).squeeze(drop=True)
                    row_out = pl_out.isel(seg_id=int(seg_out_sel[0]), time=time_pos_out).squeeze(drop=True)

                    row_center = np.array(
                        [row["longitude_m"].item(), row["latitude_m"].item(), row["altitude"].item()],
                        dtype=float,
                    )
                    if not np.all(np.isfinite(row_center)):
                        continue

                    if (row_out["ellipses_m"] is not None
                            and hasattr(row_out["ellipses_m"], "__len__")
                            and len(row_out["ellipses_m"]) > 0):
                        ellipse_vals = row_out["ellipses_m"].values
                        centre_m = _ellipse_center_m(ellipse_vals, row)
                        if np.all(np.isfinite(centre_m)):
                            centres_m.append(centre_m.tolist())
                            ellipses_m.append(ellipse_vals)
                            slice_polys_m.append(row_out["slice_polys_m"].values)

                centres_m     = np.array(centres_m)
                ellipses_m    = np.array(ellipses_m)
                slice_polys_m = np.array(slice_polys_m)

                if ellipses_m.ndim == 3 and ellipses_m.shape[1] == 3:
                    ellipses_m = np.transpose(ellipses_m, (0, 2, 1))
                if slice_polys_m.ndim == 4 and slice_polys_m.shape[2] == 3:
                    slice_polys_m = np.transpose(slice_polys_m, (0, 1, 3, 2))

                if _zs != 1.0:
                    if centres_m.ndim == 2 and centres_m.shape[1] == 3:
                        centres_m[:, 2] *= _zs
                    if ellipses_m.ndim == 3 and ellipses_m.shape[2] == 3:
                        ellipses_m[:, :, 2] *= _zs
                    if slice_polys_m.ndim == 4 and slice_polys_m.shape[3] == 3:
                        slice_polys_m[:, :, :, 2] *= _zs

                # Ellipse ribbons
                for i in range(len(ellipses_m) - 1):
                    e1, e2 = ellipses_m[i], ellipses_m[i + 1]
                    if not (np.all(np.isfinite(e1)) and np.all(np.isfinite(e2))):
                        continue
                    n = e1.shape[0]
                    points = np.vstack([e1, e2])
                    faces = []
                    for j in range(n):
                        j2 = (j + 1) % n
                        faces.extend([4, j, j2, n + j2, n + j])
                    plotter.add_mesh(pv.PolyData(points, np.array(faces)), color=style["ellipse"], opacity=0.5)

                # Trajectory line
                if len(centres_m) > 1:
                    traj = centres_m
                    valid_mask = np.all(np.isfinite(traj), axis=1)
                    traj = traj[valid_mask]
                    if traj.shape[0] > 1:
                        plotter.add_lines(traj, color=style["trajectory"], width=3, connected=True)

                # Slice polyhedra
                if slice_polys_m.ndim == 4 and slice_polys_m.shape[0] > 1:
                    num_segments = slice_polys_m.shape[0]
                    num_slices   = slice_polys_m.shape[1]
                    if len(ellipses_m) == num_segments:
                        for slice_idx in range(num_slices):
                            for seg_idx in range(num_segments - 1):
                                poly1 = slice_polys_m[seg_idx,     slice_idx]
                                poly2 = slice_polys_m[seg_idx + 1, slice_idx]
                                if (
                                    poly1.shape[0] >= 3 and poly2.shape[0] >= 3
                                    and np.all(np.isfinite(poly1)) and np.all(np.isfinite(poly2))
                                    and np.linalg.norm(poly1) > 1.0 and np.linalg.norm(poly2) > 1.0
                                ):
                                    n_c = poly1.shape[0]
                                    pts = np.vstack([poly1, poly2])
                                    fc  = []
                                    for ci in range(n_c):
                                        nc2 = (ci + 1) % n_c
                                        fc.extend([4, ci, nc2, n_c + nc2, n_c + ci])
                                    plotter.add_mesh(
                                        pv.PolyData(pts, np.array(fc)),
                                        color=style["slice"], opacity=0.3,
                                    )

            # ---- Bounds / axes (added last so they cover the full scene) ----
            plotter.add_axes()
            plotter.show_bounds(
                bounds=scene_bounds,
                grid=True,
                location="outer",
                xtitle="X (m)",
                ytitle="Y (m)",
                ztitle=ztitle,
                n_xlabels=4,
                n_ylabels=4,
                n_zlabels=zlabel_count,
                font_size=int(axis_font_size),
                fmt="%.0f",
                ticks="outside",
                minor_ticks=False,
            )

            # Time label
            plotter.add_text(
                f"Flight Plumes — time_idx={t_idx}  ({flight_note})\n"
                f"Frame {frame_num + 1}/{common_times.size}",
                position="upper_left",
                font_size=10,
                color="black",
            )

            # Fix camera on first frame so all frames share the same viewpoint
            if not camera_set:
                plotter.camera_position = "iso"
                plotter.reset_camera()
                camera_set = True

            plotter.write_frame()
            print(f"  Frame {frame_num + 1}/{common_times.size}: time_idx={t_idx}")

        plotter.close()
        print(f"Animation saved → {output_path}")

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

                ribbons.append(_ribbon_data(ea) if len(ea) >= 2 else empty_m)
                slices.append(_slice_data(spa, len(ea)))

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
            r0 = ribbons0[fi]
            initial_traces.append(go.Mesh3d(
                x=r0["x"], y=r0["y"], z=r0["z"],
                i=r0["i"], j=r0["j"], k=r0["k"],
                color=col, opacity=0.5,
                name=f"{fid} ellipses",
                showlegend=False,
            ))
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
                        f"log10 Y_del_f<br>{patch_species} (mol/m^3)"
                        if self._resolve_patch_plot_floor(
                            patch_table["Y_del_f"].sel(species_out=patch_species).values
                        ) is None
                        else (
                            "log10 Y_del_f<br>"
                            f"{patch_species} (mol/m^3)<br>"
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
        N_animated = 3 * n_flt + (1 if (overlay_patch and patch_available) else 0)
        for frame_num, t_idx in enumerate(common_times):
            trajs, ribbons, slices, patch_d = _collect_frame(t_idx)
            fd = []
            for fi in range(n_flt):
                t = trajs[fi]
                fd.append(go.Scatter3d(x=t["x"], y=t["y"], z=t["z"]))
                r = ribbons[fi]
                fd.append(go.Mesh3d(x=r["x"], y=r["y"], z=r["z"], i=r["i"], j=r["j"], k=r["k"]))
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


