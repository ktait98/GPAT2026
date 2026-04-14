
"""GPATPostProcessor class for post-processing GPAT model outputs."""
import pathlib
import subprocess
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import pyvista as pv
import matplotlib.pyplot as plt
from pyproj import Transformer
import ast
import json
from pycontrails.physics import constants, geo, thermo, units
import os
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
import ipywidgets as widgets
from IPython.display import display, clear_output

class GPATPostProcessor:
    """Post-process GPAT model outputs."""

    def __init__(self, run_path, data_path, criteria):
        self.run_path = run_path
        self.data_path = data_path
        
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
            print(self.pp_gpat.fl_ds_dict[job_id], "\n")
        elif ds_type == "pl":
            print(self.pp_gpat.pl_ds_dict[job_id], "\n")
        elif ds_type == "boxm":
            print(self.pp_gpat.boxm_ds_dict[job_id], "\n")
        else:
            print("Invalid dataset type. Please choose from 'fl', 'pl', or 'boxm'.")

    def print_output_ds(self, job_id, ds_type):
        if ds_type == "boxm_out":
            print(self.pp_gpat.boxm_out_dict[job_id], "\n")
        elif ds_type == "patch_table":
            print(self.pp_gpat.patch_table_dict[job_id], "\n")
        elif ds_type == "pl_out":
            print(self.pp_gpat.pl_out_dict[job_id], "\n")
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

    def summarize_pl_out_species(self, job_id, species="NO", filter_active=False):
        """Return time-indexed summary stats for plume mass and delta mass for one species.
        
        Parameters
        ----------
        job_id : str
            Job ID
        species : str
            Species name (default "NO")
        filter_active : bool
            If True, only return timesteps where max_pl_mass > 0 (polluted timesteps)
        """
        pl_out = self.pp_gpat.pl_out_dict[job_id]
        pl_mass = pl_out["pl_mass"].sel(species_pl=species)

        summary = pd.DataFrame(
            {
                "time_idx": np.asarray(pl_out["time_idx"].values, dtype=int),
                "max_pl_mass": pl_mass.max(dim="seg_id", skipna=True).values,
                "min_pl_mass": pl_mass.min(dim="seg_id", skipna=True).values,
            }
        )
        
        if filter_active:
            summary = summary[summary["max_pl_mass"] > 0].reset_index(drop=True)
        
        return summary

    def summarize_boxm_species(self, job_id, species="NO", filter_active=False, activity_threshold=0.0):
        """Return time-indexed summary stats for coarse bg/delta/total for one species."""
        boxm_out = self.pp_gpat.boxm_out_dict[job_id]

        y_bg = boxm_out["Y_bg_c"].sel(species_out=species)
        y_del = boxm_out["Y_del_c"].sel(species_out=species)
        # Treat missing delta values as zero when forming total.
        y_tot = y_bg + xr.where(np.isfinite(y_del), y_del, 0.0)

        reduce_dims = [d for d in y_bg.dims if d != "time"]
        neg_count = (y_tot < 0).sum(dim=reduce_dims)
        y_del_finite_count = np.isfinite(y_del).sum(dim=reduce_dims)
        y_del_all_nan = (y_del_finite_count == 0)

        summary = pd.DataFrame(
            {
                "time_idx": np.asarray(boxm_out["time_idx"].values, dtype=int),
                "bg_min": y_bg.min(dim=reduce_dims, skipna=True).values,
                "bg_max": y_bg.max(dim=reduce_dims, skipna=True).values,
                "del_min": y_del.min(dim=reduce_dims, skipna=True).values,
                "del_max": y_del.max(dim=reduce_dims, skipna=True).values,
                "tot_min": y_tot.min(dim=reduce_dims, skipna=True).values,
                "tot_max": y_tot.max(dim=reduce_dims, skipna=True).values,
                "tot_neg_count": np.asarray(neg_count.values, dtype=int),
                "del_finite_count": np.asarray(y_del_finite_count.values, dtype=int),
                "del_all_nan": np.asarray(y_del_all_nan.values, dtype=bool),
            }
        )

        if filter_active:
            thr = float(activity_threshold)
            active_mask = (
                (~summary["del_all_nan"]) &
                ((summary["del_max"] > thr) | (summary["del_min"] < -thr))
            )
            summary = summary[active_mask].reset_index(drop=True)
        return summary

    def find_negative_total_cells(self, job_id, species="NO", time_idx=None, max_rows=20):
        """Return rows where total concentration (Y_bg_c + Y_del_c) is negative."""
        boxm_out = self.pp_gpat.boxm_out_dict[job_id]
        y_bg = boxm_out["Y_bg_c"].sel(species_out=species)
        y_del = boxm_out["Y_del_c"].sel(species_out=species)
        y_tot = y_bg + y_del

        if time_idx is not None and "time_idx" in boxm_out.coords:
            all_tidx = np.asarray(boxm_out["time_idx"].values, dtype=int)
            t_sel = int(all_tidx[np.argmin(np.abs(all_tidx - int(time_idx)))])
            t_loc = np.flatnonzero(all_tidx == t_sel)
            if t_loc.size == 0:
                return pd.DataFrame()
            y_bg = y_bg.isel(time=t_loc).squeeze("time")
            y_del = y_del.isel(time=t_loc).squeeze("time")
            y_tot = y_tot.isel(time=t_loc).squeeze("time")

        neg_mask = xr.apply_ufunc(np.isfinite, y_tot, dask="parallelized") & (y_tot < 0)
        neg_mask = neg_mask.compute()
        neg_ds = xr.Dataset({"y_tot": y_tot, "y_bg": y_bg, "y_del": y_del}).where(neg_mask, drop=True)
        if neg_ds.sizes.get("time", 0) == 0 and "time" in neg_ds.sizes:
            return pd.DataFrame()

        out = neg_ds.to_dataframe().reset_index()
        out = out[np.isfinite(out["y_tot"])].copy()
        out = out.sort_values("y_tot")
        if max_rows is not None:
            out = out.head(int(max_rows))
        return out

    def summarize_patch_species(self, job_id, species="NO", threshold=1.0e-20, chunk_rows=200000):
        """Return patch-table summary by time_idx for one species.

        Uses chunked aggregation over rows to avoid memory spikes on large patch tables.
        """
        patch_table = self.pp_gpat.patch_table_dict.get(job_id)
        if patch_table is None:
            return pd.DataFrame()

        y_del = patch_table["Y_del_f"].sel(species_out=species)
        if "row" not in y_del.dims or "time_idx" not in patch_table:
            return pd.DataFrame()

        n_rows_total = int(y_del.sizes["row"])
        if n_rows_total <= 0:
            return pd.DataFrame()

        thr = float(threshold)
        chunk_rows = max(1, int(chunk_rows))

        nrows_by_t = {}
        nactive_by_t = {}
        neg_by_t = {}
        sum_active_by_t = {}
        max_by_t = {}
        min_by_t = {}

        for i0 in range(0, n_rows_total, chunk_rows):
            i1 = min(n_rows_total, i0 + chunk_rows)
            t_chunk = np.asarray(patch_table["time_idx"].isel(row=slice(i0, i1)).values, dtype=int)
            y_chunk = np.asarray(y_del.isel(row=slice(i0, i1)).values, dtype=float)

            if t_chunk.size == 0:
                continue

            unique_t = np.unique(t_chunk)
            for t in unique_t:
                mask_t = (t_chunk == t)
                vals = y_chunk[mask_t]
                if vals.size == 0:
                    continue

                finite = np.isfinite(vals)
                if not np.any(finite):
                    nrows_inc = 0
                    nactive_inc = 0
                    neg_inc = 0
                    sum_active_inc = 0.0
                    max_inc = -np.inf
                    min_inc = np.inf
                else:
                    vals_f = vals[finite]
                    nrows_inc = int(vals_f.size)

                    active = np.abs(vals_f) > thr
                    nactive_inc = int(np.sum(active))
                    neg_inc = int(np.sum(vals_f < 0.0))
                    sum_active_inc = float(np.sum(vals_f[active])) if nactive_inc > 0 else 0.0

                    max_inc = float(np.max(vals_f))
                    min_inc = float(np.min(vals_f))

                nrows_by_t[t] = nrows_by_t.get(t, 0) + nrows_inc
                nactive_by_t[t] = nactive_by_t.get(t, 0) + nactive_inc
                neg_by_t[t] = neg_by_t.get(t, 0) + neg_inc
                sum_active_by_t[t] = sum_active_by_t.get(t, 0.0) + sum_active_inc

                if t in max_by_t:
                    max_by_t[t] = max(max_by_t[t], max_inc)
                    min_by_t[t] = min(min_by_t[t], min_inc)
                else:
                    max_by_t[t] = max_inc
                    min_by_t[t] = min_inc

        if not nrows_by_t:
            return pd.DataFrame()

        times = np.array(sorted(nrows_by_t.keys()), dtype=int)
        nrows = np.array([nrows_by_t[t] for t in times], dtype=int)
        n_active = np.array([nactive_by_t.get(t, 0) for t in times], dtype=int)
        neg_count = np.array([neg_by_t.get(t, 0) for t in times], dtype=int)
        max_val = np.array([max_by_t.get(t, np.nan) for t in times], dtype=float)
        min_val = np.array([min_by_t.get(t, np.nan) for t in times], dtype=float)
        sum_active = np.array([sum_active_by_t.get(t, 0.0) for t in times], dtype=float)

        mean_active = np.zeros_like(sum_active, dtype=float)
        nz = n_active > 0
        mean_active[nz] = sum_active[nz] / n_active[nz]

        out = pd.DataFrame(
            {
                "time_idx": times,
                "nrows": nrows,
                "n_active": n_active,
                "max_val": max_val,
                "min_val": min_val,
                "mean_active": mean_active,
                "neg_count": neg_count,
            }
        )

        return out.reset_index(drop=True)

    def rank_fine_cells(self, job_id, species="NO", time_indices=None, n=5):
        """Rank fine cells by |Y_del_f| into high/mid/low/background tiers.

        Parameters
        ----------
        job_id : str
        species : str
        time_indices : list[int] or None
            Specific time indices to consider.  None = all timesteps.
        n : int
            Number of cells to show per tier.
        """
        pt = self.pp_gpat.patch_table_dict[job_id]
        t_all = pt["time_idx"].values.astype(int)

        if time_indices is not None:
            mask = np.isin(t_all, time_indices)
            pt = pt.isel(row=np.flatnonzero(mask))

        y = pt["Y_del_f"].sel(species_out=species).values.astype(float).ravel()
        lat = pt["latitude_f"].values.astype(float).ravel()
        lon = pt["longitude_f"].values.astype(float).ravel()
        alt = pt["altitude_f"].values.astype(float).ravel()
        cc = pt["row_cell_c"].values.astype(int).ravel()
        cf = pt["row_cell_f"].values.astype(int).ravel()
        ti = pt["time_idx"].values.astype(int).ravel()

        df = pd.DataFrame({
            "time_idx": ti, "cell_c": cc, "cell_f": cf,
            "lat": lat, "lon": lon, "alt": alt, "Y_del_f": y,
        })

        grp = df.groupby(["cell_c", "cell_f", "lat", "lon", "alt"]).agg(
            mean_abs=("Y_del_f", lambda x: np.abs(x).mean()),
            peak=("Y_del_f", "max"),
            trough=("Y_del_f", "min"),
            n_times=("Y_del_f", "count"),
        ).reset_index().sort_values("mean_abs", ascending=False)

        nonzero = grp[grp["mean_abs"] > 0]
        bg = grp[grp["mean_abs"] == 0]

        tiers = {}
        tiers["high"] = grp.head(n)
        if len(nonzero) > n:
            mid_start = len(nonzero) // 2
            tiers["mid"] = nonzero.iloc[mid_start : mid_start + n]
        tiers["low"] = nonzero.tail(n) if len(nonzero) > 0 else pd.DataFrame()
        tiers["background"] = bg.head(n) if len(bg) > 0 else pd.DataFrame()

        label = f"time_indices={time_indices}" if time_indices else "all timesteps"
        print(f"\n=== Fine cell ranking for {species} ({label}) ===")
        print(f"Total unique fine cells: {len(grp)}, nonzero: {len(nonzero)}, pure bg: {len(bg)}")
        for name, tier_df in tiers.items():
            print(f"\n--- {name.upper()} (n={len(tier_df)}) ---")
            if len(tier_df) > 0:
                print(tier_df.to_string(index=False))
            else:
                print("  (none)")

        return tiers

class GPATPlotting:
    """Plot GPAT model outputs."""

    def __init__(self, pp_gpat):
        self.pp_gpat = pp_gpat
        self.patch_plot_min_value = None
        self.patch_plot_max_log10_span = 3.0

    def plot_species_budget_timeseries(self, job_id, species="NO", level=None, metric="mean"):
        """Plot bg, delta and total coarse fields through time for one species."""
        boxm_out = self.pp_gpat.boxm_out_dict[job_id]
        y_bg = boxm_out["Y_bg_c"].sel(species_out=species)
        y_del = boxm_out["Y_del_c"].sel(species_out=species)

        if level is not None and "level_c" in y_bg.dims:
            lev = np.asarray(y_bg["level_c"].values, dtype=float)
            k = int(np.argmin(np.abs(lev - float(level))))
            y_bg = y_bg.isel(level_c=k)
            y_del = y_del.isel(level_c=k)
            print(f"Using nearest level_c={lev[k]} for requested level={level}")

        y_tot = y_bg + y_del
        reduce_dims = [d for d in y_bg.dims if d != "time"]
        if metric == "sum":
            bg_s = y_bg.sum(dim=reduce_dims)
            del_s = y_del.sum(dim=reduce_dims)
            tot_s = y_tot.sum(dim=reduce_dims)
            y_label = "domain sum"
        else:
            bg_s = y_bg.mean(dim=reduce_dims)
            del_s = y_del.mean(dim=reduce_dims)
            tot_s = y_tot.mean(dim=reduce_dims)
            y_label = "domain mean"

        x = np.asarray(boxm_out["time_idx"].values, dtype=int)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(x, bg_s.values, label="Y_bg_c")
        ax.plot(x, del_s.values, label="Y_del_c")
        ax.plot(x, tot_s.values, label="Y_total = Y_bg_c + Y_del_c")
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.6)
        ax.set_xlabel("time_idx")
        ax.set_ylabel(y_label)
        ax.set_title(f"Coarse species budget: {species}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_negative_total_map(self, job_id, species="NO", time_idx=None, level=None):
        """Plot total coarse field and overlay cells where total < 0."""
        boxm_out = self.pp_gpat.boxm_out_dict[job_id]
        y_bg = boxm_out["Y_bg_c"].sel(species_out=species)
        y_del = boxm_out["Y_del_c"].sel(species_out=species)
        y_tot = y_bg + y_del

        all_tidx = np.asarray(boxm_out["time_idx"].values, dtype=int)
        if time_idx is None:
            neg_any = ((y_tot < 0).sum(dim=[d for d in y_tot.dims if d != "time"]) > 0).values
            if np.any(neg_any):
                t_sel = int(all_tidx[np.flatnonzero(neg_any)[0]])
            else:
                t_sel = int(all_tidx[0])
        else:
            t_sel = int(all_tidx[np.argmin(np.abs(all_tidx - int(time_idx)))])

        t_loc = np.flatnonzero(all_tidx == t_sel)
        ds_t = y_tot.isel(time=t_loc).squeeze("time")
        if level is not None and "level_c" in ds_t.dims:
            lev = np.asarray(ds_t["level_c"].values, dtype=float)
            k = int(np.argmin(np.abs(lev - float(level))))
            ds_t = ds_t.isel(level_c=k)
            print(f"Using nearest level_c={lev[k]} for requested level={level}")
        elif "level_c" in ds_t.dims:
            ds_t = ds_t.isel(level_c=0)

        vals = np.asarray(ds_t.values, dtype=float)
        if vals.ndim == 2:
            vals = vals.T  # (lat, lon)
        lon = np.asarray(boxm_out["longitude_c"].values, dtype=float)
        lat = np.asarray(boxm_out["latitude_c"].values, dtype=float)

        neg_mask = np.isfinite(vals) & (vals < 0)
        n_neg = int(np.sum(neg_mask))
        print(f"time_idx={t_sel}: negative total cells={n_neg}")

        fig, ax = plt.subplots(figsize=(8, 6))
        mesh = ax.pcolormesh(lon, lat, vals, shading="auto", cmap="RdBu_r")
        fig.colorbar(mesh, ax=ax, label="Y_total")
        if n_neg > 0:
            iy, ix = np.where(neg_mask)
            ax.scatter(lon[ix], lat[iy], s=12, c="k", alpha=0.6, label="total < 0")
            ax.legend(loc="best")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Total coarse field for {species} at time_idx={t_sel}")
        plt.tight_layout()
        plt.show()

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
    
    def plot_cell_timeseries(
        self,
        job_id,
        cell_c,
        cell_f=None,
        compare_boxm_orig=False,
        species=("NO", "NO2", "O3", "OH", "HO2", "CO", "CH4", "HNO3"),
        show_del=True,
        time_idx_max=None,
    ):
        """Plot 2×4 time series for a single coarse cell, with optional fine-cell and
        boxm_orig comparisons.  All concentrations are converted to ppb.

        Cell is selected by coarse cell index (cell_c, 1-based Fortran convention) from
        boxm_ds.  An optional fine sub-cell (cell_f, also 1-based) can be overlaid from
        patch_table to show Y_del_f alongside Y_del_c — useful for validating the plume
        → coarse-grid conversion and advection after max plume age.

        A vertical marker is drawn at the last time index where any fine-cell patch
        exists for the selected cell, indicating the transition from fine-cell to
        coarse-cell representation of the plume perturbation.

        Parameters
        ----------
        job_id : str
            Job identifier in boxm_out_dict / boxm_ds_dict.
        cell_c : int
            1-based coarse cell index (Fortran convention) into boxm_ds.
        cell_f : int or None
            1-based fine cell index within the coarse cell (from patch_table).  None
            means no fine-cell overlay; ``show_del`` still shows Y_del_c.
        compare_boxm_orig : bool
            If True, read Y.OUT from the most recent boxm_orig run in
            ``outputs/<job_id>/Y.OUT`` and overlay per-species.  The file must already
            exist (run ``validation.boxm_test`` first).
        species : tuple[str]
            Up to 8 species to plot across the 2×4 grid.
        show_del : bool
            If True, also draw Y_del_c (coarse delta) and Y_bg_c + Y_del_c (total),
            and Y_del_f if cell_f is given.
        time_idx_max : int or None
            If provided, only plot output samples with ``time_idx <= time_idx_max``.
            Useful for Tier 1 no-emission chemistry comparisons on jobs that also
            include flights later in the simulation.
        """
        import pathlib as _pl

        boxm_ds  = self.pp_gpat.boxm_ds_dict[job_id]
        boxm_out = self.pp_gpat.boxm_out_dict[job_id]

        # ── Resolve coarse cell lat / lon / level from boxm_ds ──────────────────
        if "cell" not in boxm_ds.dims:
            raise ValueError("boxm_ds does not have a 'cell' dimension; cannot select by cell_c index")
        ds_cell = boxm_ds.isel(cell=int(cell_c) - 1)
        lat_val = float(ds_cell.coords.get("latitude_c",  ds_cell.coords.get("latitude")))
        lon_val = float(ds_cell.coords.get("longitude_c", ds_cell.coords.get("longitude")))
        lev_val = float(ds_cell.coords.get("level_c",     ds_cell.coords.get("level")))
        M_val   = float(np.asarray(ds_cell["M"].values).ravel()[0])

        # ── Select matching coarse output cell ───────────────────────────────────
        if "cell" in boxm_out.dims:
            out_cell = boxm_out.isel(cell=int(cell_c) - 1)
        else:
            # unstacked (level_c × longitude_c × latitude_c)
            out_cell = boxm_out.sel(latitude_c=lat_val, longitude_c=lon_val, method="nearest")
            lev_arr  = np.asarray(out_cell["level_c"].values, dtype=float)
            out_cell = out_cell.isel(level_c=int(np.argmin(np.abs(lev_arr - lev_val))))

        # ── Time axis ────────────────────────────────────────────────────────────
        time_idx_arr = np.asarray(boxm_out["time_idx"].values, dtype=int)
        try:
            time_axis = pd.to_datetime(boxm_out["time"].values)
            use_datetime = True
        except Exception:
            time_axis = time_idx_arr.astype(float)
            use_datetime = False

        time_mask = np.ones_like(time_idx_arr, dtype=bool)
        if time_idx_max is not None:
            time_mask = time_idx_arr <= int(time_idx_max)
            if not np.any(time_mask):
                raise ValueError(f"No samples satisfy time_idx <= {time_idx_max}")

        # ── Fine-cell (Y_del_f) time series from patch_table ────────────────────
        # Reconstruct a per-output-timestep time series by pivoting patch_table rows
        # onto the coarse output time grid for the selected (cell_c, cell_f) pair.
        y_del_f_series = {}     # species → ndarray[float], ppb, aligned to time_idx_arr
        last_fine_time  = None  # for end-of-plume marker
        if cell_f is not None:
            pt = self.pp_gpat.patch_table_dict.get(job_id)
            if pt is not None:
                cc_all = np.asarray(pt["row_cell_c"].values, dtype=int)
                cf_all = np.asarray(pt["row_cell_f"].values, dtype=int)
                ti_all = np.asarray(pt["time_idx"].values,   dtype=int)
                mask   = (cc_all == int(cell_c)) & (cf_all == int(cell_f))
                if mask.sum() > 0:
                    rows_sel = np.flatnonzero(mask)
                    pt_filt  = pt.isel(row=rows_sel)
                    ti_filt  = ti_all[mask]

                    if len(ti_filt) > 0:
                        last_ti   = int(ti_filt.max())
                        idx_in_ts = np.flatnonzero(time_idx_arr == last_ti)
                        if idx_in_ts.size > 0:
                            last_fine_time = time_axis[idx_in_ts[0]]

                    sp_out = pt["species_out"].values if "species_out" in pt.dims else []
                    for sp in species:
                        if sp not in sp_out:
                            continue
                        raw = np.asarray(pt_filt["Y_del_f"].sel(species_out=sp).values, dtype=float)
                        ts  = np.full(len(time_idx_arr), np.nan)
                        for j, tidx in enumerate(ti_filt):
                            k = np.flatnonzero(time_idx_arr == tidx)
                            if k.size:
                                ts[k[0]] = raw[j]  # Y_del_f already in ppb
                        y_del_f_series[sp] = ts
                else:
                    print(f"Warning: no patch_table rows for cell_c={cell_c}, cell_f={cell_f}")

        # ── boxm_orig Y.OUT ──────────────────────────────────────────────────────
        # Y.OUT has 16 columns: TIME + species_out (in ppb). Use native header and species_out_num.
        orig_y_ppb = {}  # species → ndarray[float] interpolated onto output time grid
        if compare_boxm_orig:
            yout_path = _pl.Path(self.pp_gpat.data_path) / "outputs" / job_id / "Y.OUT"
            if yout_path.exists():
                _y_df = pd.read_csv(yout_path, dtype=np.float64)  # native header: TIME, Y1..Y219
                n_new = len(time_idx_arr)
                if "time_rel_s" in boxm_out:
                    new_t = np.asarray(boxm_out["time_rel_s"].values, dtype=float)
                else:
                    new_t = np.arange(n_new, dtype=float)
                if "TIME" in _y_df.columns:
                    orig_t = _y_df["TIME"].values.astype(float)
                else:
                    orig_t = np.linspace(new_t.min(), new_t.max(), len(_y_df))
                for sp in species:
                    if sp not in out_cell.coords.get("species_out", []):
                        continue
                    sp_num = int(out_cell["species_out_num"].sel(species_out=sp).values.item())
                    col = f"Y{sp_num}"
                    if col in _y_df.columns:
                        orig_y_ppb[sp] = np.interp(new_t, orig_t, _y_df[col].values)
            else:
                print(f"Warning: {yout_path} not found; run validation.boxm_test first")

        # ── End-of-plume marker from active_flag (coarse cell) ──────────────────
        # If no fine cell is provided, fall back to the last active coarse output time.
        coarse_last_active_time = None
        if last_fine_time is None and "active_flag" in boxm_out:
            try:
                af_cell = out_cell["active_flag"].values.astype(bool)
                active_rows = np.flatnonzero(af_cell)
                if active_rows.size:
                    coarse_last_active_time = time_axis[active_rows[-1]]
            except Exception:
                pass

        marker_time  = last_fine_time if last_fine_time is not None else coarse_last_active_time
        marker_label = "end of fine patches" if last_fine_time is not None else "last coarse active"

        # Prefer explicit handoff time if available
        explicit_handoff_time = None
        if "handoff_time_idx" in boxm_out:
            try:
                hidx = int(np.asarray(boxm_out["handoff_time_idx"].values).ravel()[0])
                k = np.flatnonzero(time_idx_arr == hidx)
                if k.size:
                    explicit_handoff_time = time_axis[k[0]]
            except Exception:
                pass
        elif "handoff_time_idx" in boxm_out.attrs:
            try:
                hidx = int(boxm_out.attrs["handoff_time_idx"])
                k = np.flatnonzero(time_idx_arr == hidx)
                if k.size:
                    explicit_handoff_time = time_axis[k[0]]
            except Exception:
                pass

        if explicit_handoff_time is not None:
            marker_time = explicit_handoff_time
            marker_label = "global handoff"

        # ── Subplots ─────────────────────────────────────────────────────────────
        plot_species = list(species)[:8]
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True)

        for i, sp in enumerate(plot_species):
            ax = axes[i // 4][i % 4]
            sp_in_out = "species_out" in out_cell.dims and sp in out_cell["species_out"].values

            if sp_in_out:
                y_bg_ppb = np.asarray(out_cell["Y_bg_c"].sel(species_out=sp).values, dtype=float)
                ax.plot(time_axis[time_mask], y_bg_ppb[time_mask], color="tab:blue", linewidth=1.2, label="$Y_{bg,c}$")

                if show_del and "Y_del_c" in out_cell:
                    y_del_ppb = np.asarray(out_cell["Y_del_c"].sel(species_out=sp).values, dtype=float)
                    ax.plot(time_axis[time_mask], (y_bg_ppb + y_del_ppb)[time_mask],
                            color="tab:orange", linestyle="--", linewidth=1.0,
                            label="$Y_{bg,c} + Y_{del,c}$")
                    ax.plot(time_axis[time_mask], y_del_ppb[time_mask],
                            color="tab:red", linestyle=":", linewidth=0.9,
                            label="$Y_{del,c}$")

            if show_del and sp in y_del_f_series:
                ax.plot(time_axis[time_mask], y_del_f_series[sp][time_mask],
                        color="tab:green", linestyle="-.", linewidth=1.1,
                        label=f"$Y_{{del,f}}$ (cell_f={cell_f})")

            if sp in orig_y_ppb:
                if sp == "CH4" and sp in orig_y_ppb:
                    print("CH4 orig_y_ppb:", orig_y_ppb[sp])
                    print("orig_t:", orig_t)
                    print("new_t:", new_t)
                    print("Y.OUT CH4 values:", _y_df[col].values)
                ax.plot(time_axis[time_mask], orig_y_ppb[sp][time_mask],
                        color="k", linestyle="--", linewidth=0.8,
                        label="boxm_orig Y.OUT")

            if marker_time is not None:
                ax.axvline(x=marker_time, color="gray", linestyle=":", linewidth=1.0,
                           alpha=0.8, label=marker_label)

            ax.set_title(sp, fontsize=10)
            ax.set_ylabel("ppb", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            if i >= 4:
                ax.set_xlabel("Time")

        for j in range(len(plot_species), 8):
            axes[j // 4][j % 4].set_visible(False)

        # Deduplicated legend from first subplot, placed inside the figure
        handles, labels = axes[0][0].get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
        if seen:
            fig.legend(
                list(seen.values()), list(seen.keys()),
                loc="lower center",
                ncol=min(len(seen), 6),
                fontsize=9,
                bbox_to_anchor=(0.5, 0.01),
                framealpha=0.9,
            )

        title = (
            f"Cell chemistry — job={job_id}, cell_c={cell_c}"
            + (f", cell_f={cell_f}" if cell_f is not None else "")
            + f"\n(lat={lat_val:.3f}, lon={lon_val:.3f}, {lev_val:.1f} hPa)"
        )
        fig.suptitle(title, fontsize=11)
        plt.tight_layout(rect=[0, 0.09, 1, 0.96])
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
        mode="total",                    # "background", "coarse_delta", "patch_delta", "patch_delta_fine", "total", "total_with_patch"
        level=None,                      # required for vertical_mode="single"
        time_indices=None,
        vertical_mode="single",          # "single", "column", "topk"
        top_k=3,
        level_tol=None,
        use_nearest_level=True,
        dynamic_colorbar=False,
        cmap="viridis",
        overlay_flights=False,
        overlay_slices=False,
        slice_id=None,                   # None = outermost slice, or 1-based index
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

        allowed_modes = {"background", "coarse_delta", "patch_delta", "patch_delta_fine", "total", "total_with_patch", "total_with_patch_fine"}
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
        pl_out = self.pp_gpat.pl_out_dict.get(job_id, None)
        fl_ds = self.pp_gpat.fl_ds_dict.get(job_id, None)
        boxm_ds = self.pp_gpat.boxm_ds_dict.get(job_id, None)

        # Projection reference for m -> lonlat conversion (overlay)
        if (overlay_flights or overlay_slices) and boxm_ds is not None:
            _proj_ref_lon = float(boxm_ds["longitude_c"].min())
            _proj_ref_lat = float(boxm_ds["latitude_c"].min())
        else:
            _proj_ref_lon = _proj_ref_lat = None

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
                i_lat = _nearest_index_1d(lon_bg, df["longitude_f"].values)
                i_lon = _nearest_index_1d(lon_bg, df["latitude_f"].values)

                for ii, jj, vv in zip(i_lat, i_lon, df["y_del"].values):
                    delta[ii, jj] += vv

                return delta

            elif vertical_mode == "column":
                delta = np.zeros((lat_bg.size, lon_bg.size), dtype=float)
                i_lat = _nearest_index_1d(lon_bg, df["latitude_f"].values)
                i_lon = _nearest_index_1d(lon_bg, df["longitude_f"].values)

                for ii, jj, vv in zip(i_lat, i_lon, df["y_del"].values):
                    delta[ii, jj] += vv

                return delta

            elif vertical_mode == "topk":
                df = df[np.isfinite(df["level_f"].values)]
                if df.empty:
                    return np.zeros((lat_bg.size, lon_bg.size), dtype=float)

                i_lat = _nearest_index_1d(lon_bg, df["latitude_f"].values)
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
            elif mode in ("patch_delta_fine", "total_with_patch_fine"):
                plot_2d = None  # handled separately in draw_frame
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
            if mode == "patch_delta_fine":
                # Precompute from raw fine-cell delta values
                mins = []
                maxs = []
                print("Precomputing fixed colour scale (fine cells)...")
                for t_idx in time_indices:
                    df_p = _patch_rows_dataframe(int(t_idx))
                    if not df_p.empty:
                        if vertical_mode == "single" and patch_level_req is not None:
                            levs = df_p["level_f"].values
                            df_p = df_p[np.isclose(levs, patch_level_req)]
                        if not df_p.empty:
                            mins.append(float(df_p["y_del"].min()))
                            maxs.append(float(df_p["y_del"].max()))
                if not mins:
                    raise ValueError("No finite fine-cell values across selected frames.")
                vmin = float(np.min(mins))
                vmax = float(np.max(maxs))
            elif mode == "total_with_patch_fine":
                # Precompute from bg + delta (total) values
                mins = []
                maxs = []
                print("Precomputing fixed colour scale (total fine cells)...")
                for t_idx in time_indices:
                    ds_t = _get_boxm_time_slice(int(t_idx))
                    bg_2d_t = _get_2d_from_coarse_da(ds_t["Y_bg_c"].sel(species_out=species))
                    mins.append(float(np.nanmin(bg_2d_t)))
                    maxs.append(float(np.nanmax(bg_2d_t)))
                    df_p = _patch_rows_dataframe(int(t_idx))
                    if not df_p.empty:
                        if vertical_mode == "single" and patch_level_req is not None:
                            levs = df_p["level_f"].values
                            df_p = df_p[np.isclose(levs, patch_level_req)]
                        if not df_p.empty:
                            # Peak total = max bg + max delta
                            maxs.append(float(np.nanmax(bg_2d_t)) + float(df_p["y_del"].max()))
                if not mins:
                    raise ValueError("No finite values across selected frames.")
                vmin = float(np.min(mins))
                vmax = float(np.max(maxs))
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
        plt.subplots_adjust(bottom=0.20, right=0.85)

        slider_ax = fig.add_axes([0.15, 0.06, 0.55, 0.05])
        slider = Slider(
            slider_ax,
            "Time Index",
            int(time_indices.min()),
            int(time_indices.max()),
            valinit=int(time_indices[0]),
            valstep=1,
        )

        cbar_ax = fig.add_axes([0.87, 0.20, 0.03, 0.68])

        def _vertical_label():
            if vertical_mode == "single":
                return f"single level={level}"
            if vertical_mode == "column":
                return "full column"
            if vertical_mode == "topk":
                return f"top-{top_k} levels"
            return vertical_mode

        def draw_frame(t_idx_req):
            ax.clear()

            t_idx_actual = _snap_time_index(t_idx_req)
            vals, bg_2d, coarse_del_2d, patch_del_2d = _build_fields(t_idx_actual)

            if mode in ("patch_delta_fine", "total_with_patch_fine"):
                # vals is None for these modes; skip the coarse-grid path
                pass
            else:
                finite = np.isfinite(vals)
                if not np.any(finite):
                    ax.set_title(f"No valid data for time_idx={t_idx_actual}")
                    fig.canvas.draw_idle()
                    return

            if dynamic_colorbar:
                if vals is not None:
                    vmin_loc = float(np.nanmin(vals))
                    vmax_loc = float(np.nanmax(vals))
                else:
                    vmin_loc = None
                    vmax_loc = None
            else:
                vmin_loc = vmin
                vmax_loc = vmax

            if mode in ("patch_delta_fine", "total_with_patch_fine"):
                # pcolormesh on fine-resolution grid
                _boxm_ds = boxm_ds if boxm_ds is not None else self.pp_gpat.boxm_ds_dict[job_id]
                hres_f = float(_boxm_ds.attrs["hres_sim_f"])

                df_p = _patch_rows_dataframe(t_idx_actual)
                if vertical_mode == "single" and patch_level_req is not None and not df_p.empty:
                    levs = df_p["level_f"].values
                    df_p = df_p[np.isclose(levs, patch_level_req)]

                # Build regular fine grid from domain edges
                hres_c = float(lon_bg[1] - lon_bg[0]) if lon_bg.size > 1 else float(_boxm_ds.attrs.get("hres_sim_c", hres_f))
                vres_c_lat = float(lat_bg[1] - lat_bg[0]) if lat_bg.size > 1 else hres_c
                lon_min = float(lon_bg.min()) - hres_c / 2 + hres_f / 2
                lon_max = float(lon_bg.max()) + hres_c / 2 - hres_f / 2
                lat_min = float(lat_bg.min()) - vres_c_lat / 2 + hres_f / 2
                lat_max = float(lat_bg.max()) + vres_c_lat / 2 - hres_f / 2

                lon_fine = np.arange(lon_min, lon_max + hres_f * 0.5, hres_f)
                lat_fine = np.arange(lat_min, lat_max + hres_f * 0.5, hres_f)
                lon_edges = np.concatenate([lon_fine - hres_f / 2, [lon_fine[-1] + hres_f / 2]])
                lat_edges = np.concatenate([lat_fine - hres_f / 2, [lat_fine[-1] + hres_f / 2]])

                if mode == "total_with_patch_fine":
                    # Layer 1: coarse background everywhere
                    ax.pcolormesh(
                        lon_bg, lat_bg, bg_2d,
                        shading="auto", cmap=cmap, vmin=vmin_loc, vmax=vmax_loc,
                    )

                    # Layer 2: fine cells (bg + delta) only where plume is active
                    fine_grid = np.full((lat_fine.size, lon_fine.size), np.nan, dtype=float)

                    if not df_p.empty:
                        lat_vals = df_p["latitude_f"].values
                        lon_vals = df_p["longitude_f"].values
                        y_vals = df_p["y_del"].values

                        i_lat = np.round((lat_vals - lat_fine[0]) / hres_f).astype(int)
                        i_lon = np.round((lon_vals - lon_fine[0]) / hres_f).astype(int)
                        i_lat = np.clip(i_lat, 0, lat_fine.size - 1)
                        i_lon = np.clip(i_lon, 0, lon_fine.size - 1)

                        # Accumulate deltas into the fine grid
                        for ii, jj, vv in zip(i_lat, i_lon, y_vals):
                            if np.isnan(fine_grid[ii, jj]):
                                fine_grid[ii, jj] = vv
                            else:
                                fine_grid[ii, jj] += vv

                        # Add background to active fine cells only
                        active = np.isfinite(fine_grid)
                        fi, fj = np.where(active)
                        bg_i = _nearest_index_1d(lon_bg, lat_fine[fi])
                        bg_j = _nearest_index_1d(lon_bg, lon_fine[fj])
                        for k in range(len(fi)):
                            fine_grid[fi[k], fj[k]] += bg_2d[bg_i[k], bg_j[k]]

                    artist = ax.pcolormesh(
                        lon_edges, lat_edges, fine_grid,
                        shading="flat", cmap=cmap, vmin=vmin_loc, vmax=vmax_loc,
                    )
                    ax.set_xlim(lon_edges[0], lon_edges[-1])
                    ax.set_ylim(lat_edges[0], lat_edges[-1])

                    cbar_ax.clear()
                    fig.colorbar(artist, cax=cbar_ax)

                    sc_vals = fine_grid[np.isfinite(fine_grid)]
                    n_active = int(np.sum(np.isfinite(fine_grid)))
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")
                    ax.set_title(
                        f"TOTAL WITH PATCH FINE | {species} | time_idx={t_idx_actual} | {_vertical_label()}\n"
                        f"bg min/max={np.nanmin(bg_2d):.3e}/{np.nanmax(bg_2d):.3e} | "
                        f"n_active_fine={n_active} | fine min/max="
                        + (f"{np.nanmin(sc_vals):.3e}/{np.nanmax(sc_vals):.3e}" if sc_vals.size > 0 else "N/A")
                    )
                    print(
                        f"[FRAME] total_with_patch_fine, {species}, time_idx={t_idx_actual}, "
                        f"bg_min={np.nanmin(bg_2d):.6e}, bg_max={np.nanmax(bg_2d):.6e}, "
                        f"n_active_fine={n_active}"
                        + (f", fine_min={np.nanmin(sc_vals):.6e}, fine_max={np.nanmax(sc_vals):.6e}" if sc_vals.size > 0 else "")
                    )

                else:
                    # patch_delta_fine: only fine deltas, no background
                    fine_grid = np.full((lat_fine.size, lon_fine.size), np.nan, dtype=float)

                    if df_p.empty:
                        ax.set_title(f"No fine cells at time_idx={t_idx_actual}")
                        fig.canvas.draw_idle()
                        return

                    lat_vals = df_p["latitude_f"].values
                    lon_vals = df_p["longitude_f"].values
                    y_vals = df_p["y_del"].values

                    i_lat = np.round((lat_vals - lat_fine[0]) / hres_f).astype(int)
                    i_lon = np.round((lon_vals - lon_fine[0]) / hres_f).astype(int)
                    i_lat = np.clip(i_lat, 0, lat_fine.size - 1)
                    i_lon = np.clip(i_lon, 0, lon_fine.size - 1)

                    for ii, jj, vv in zip(i_lat, i_lon, y_vals):
                        if np.isnan(fine_grid[ii, jj]):
                            fine_grid[ii, jj] = vv
                        else:
                            fine_grid[ii, jj] += vv

                    sc_vals = fine_grid[np.isfinite(fine_grid)]
                    if sc_vals.size == 0:
                        ax.set_title(f"No finite fine cells at time_idx={t_idx_actual}")
                        fig.canvas.draw_idle()
                        return

                    if dynamic_colorbar:
                        vmin_loc = float(np.nanmin(sc_vals))
                        vmax_loc = float(np.nanmax(sc_vals))

                    artist = ax.pcolormesh(
                        lon_edges, lat_edges, fine_grid,
                        shading="flat", cmap=cmap, vmin=vmin_loc, vmax=vmax_loc,
                    )
                    ax.set_xlim(lon_edges[0], lon_edges[-1])
                    ax.set_ylim(lat_edges[0], lat_edges[-1])

                    cbar_ax.clear()
                    fig.colorbar(artist, cax=cbar_ax)

                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")
                    n_filled = int(np.sum(np.isfinite(fine_grid)))
                    ax.set_title(
                        f"PATCH DELTA FINE | {species} | time_idx={t_idx_actual} | {_vertical_label()}\n"
                        f"n_cells={n_filled} | min/max={np.nanmin(sc_vals):.3e}/{np.nanmax(sc_vals):.3e}"
                    )
                    print(
                        f"[FRAME] patch_delta_fine, {species}, time_idx={t_idx_actual}, "
                        f"n_cells={n_filled}, min={np.nanmin(sc_vals):.6e}, max={np.nanmax(sc_vals):.6e}"
                    )
            else:
                artist = ax.pcolormesh(
                    lon_bg,
                    lat_bg,
                    vals,
                    shading="auto",
                    cmap=cmap,
                    vmin=vmin_loc,
                    vmax=vmax_loc,
                )

                cbar_ax.clear()
                fig.colorbar(artist, cax=cbar_ax)

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

            # --- Overlays ---------------------------------------------------
            if overlay_flights and fl_ds is not None:
                flight_colors_2d = ["white", "cyan", "lime", "yellow", "magenta"]
                for ci, fid in enumerate(np.unique(fl_ds["flight_id"].values)):
                    mask = fl_ds["flight_id"].values == fid
                    flon = np.asarray(fl_ds["longitude"].values[mask], dtype=float)
                    flat = np.asarray(fl_ds["latitude"].values[mask], dtype=float)
                    col = flight_colors_2d[ci % len(flight_colors_2d)]
                    ax.plot(flon, flat, "-", color=col, linewidth=1.5, alpha=0.85,
                            label=f"flight {fid}")
                    ax.plot(flon, flat, "o", color=col, markersize=2, alpha=0.6)
                ax.legend(loc="upper right", fontsize=7, framealpha=0.5)

            if overlay_slices and pl_out is not None and _proj_ref_lon is not None:
                from matplotlib.patches import Polygon as MplPolygon
                from matplotlib.collections import PatchCollection

                pl_out_tidx = np.asarray(pl_out["time_idx"].values, dtype=int)
                # time_idx may be 2-D (seg_id, time); take first row if so
                if pl_out_tidx.ndim == 2:
                    pl_out_tidx = pl_out_tidx[0]
                t_pos = np.flatnonzero(pl_out_tidx == t_idx_actual)
                if t_pos.size > 0:
                    tp = int(t_pos[0])
                    n_slices = int(pl_out.sizes["slice_id"])
                    if slice_id is not None:
                        slice_idx = int(slice_id) - 1
                    else:
                        slice_idx = n_slices - 1  # outermost

                    polygons = []
                    for seg_i in range(int(pl_out.sizes["seg_id"])):
                        corners = pl_out["slice_polys_m"].isel(
                            seg_id=seg_i, slice_id=slice_idx, time=tp
                        ).values  # (corner_id=4, coord=3)
                        if not np.all(np.isfinite(corners)):
                            continue
                        if not np.any(np.abs(corners) > 1.0):
                            continue  # skip degenerate (zero-extent) polygons
                        x_m = corners[:, 0]
                        y_m = corners[:, 1]
                        lons_deg, lats_deg = m_to_lonlat(
                            x_m, y_m, _proj_ref_lon, _proj_ref_lat
                        )
                        poly = MplPolygon(
                            np.column_stack([lons_deg, lats_deg]),
                            closed=True,
                        )
                        polygons.append(poly)

                    if polygons:
                        pc = PatchCollection(
                            polygons,
                            facecolor="none",
                            edgecolor="red",
                            linewidth=0.8,
                            alpha=0.8,
                        )
                        ax.add_collection(pc)
            # ---------------------------------------------------------------

            fig.canvas.draw_idle()

        def slider_update(val):
            draw_frame(int(val))

        slider.on_changed(slider_update)
        draw_frame(int(time_indices[0]))
        plt.show()

    def _patch_value_mask(self, arr):
        """Return (mask, min_finite_positive_value) for valid patch values (finite and > 0)."""
        arr = np.asarray(arr, dtype=float)
        mask = np.isfinite(arr) & (arr > 0)
        plot_floor = np.nanmin(arr[mask]) if np.any(mask) else None
        return mask, plot_floor
    
    def _resolve_patch_plot_floor(self, arr):
        """Return the minimum finite, positive value in arr, or None if not found."""
        arr = np.asarray(arr, dtype=float)
        mask = np.isfinite(arr) & (arr > 0)
        if np.any(mask):
            return np.nanmin(arr[mask])
        return None

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

    # # EMI species mapping: (EMI index 0-based, Y index 1-based, species name)
    EMI_SPECIES_MAP = [
        (0, 8,  "NO"),
        (1, 4,  "NO2"),
        (2, 11, "CO"),
        (3, 39, "HCHO"),
        (4, 42, "CH3CHO"),
        (5, 30, "C2H4"),
        (6, 32, "C3H6"),
        (7, 59, "C2H2"),
        (8, 61, "BENZENE"),
    ]

    def boxm_test(
        self,
        job_id,
        cell,
        cell_f=None,
        plot_validation=False,
        plot_handoff=False,
        handoff_species=("NO", "NO2", "O3", "OH", "HO2", "HNO3"),
    ):
        """Two-tier validation of the new box model against boxm_orig.

        Tier 1 — Chemistry kernel: run boxm_orig with zero emissions for a
        single cell and compare its Y time series against boxm_out.nc Y_bg_c.
        Both codes call the identical CHEMCO/PHOTOL/DERIV routines, so the
        results should match to floating-point precision.

        Tier 2 — Emission response: extract Y_del_c from the new model for
        the selected cell, feed it as EMI to boxm_orig (with the EMI injection
        lines now uncommented), and compare the resulting Y_total time series
        against boxm_out.nc (Y_bg_c + Y_del_c).

        Parameters
        ----------
        job_id : str
            Job ID.
        cell : int or dict
            Cell index (int) or dict with latitude, longitude, level keys.
        cell_f : int or dict, optional
            Fine cell index (int) or dict with latitude, longitude, level keys.
        plot_validation : bool, optional
            Whether to plot validation results.
        plot_handoff : bool, optional
            Whether to plot handoff results.
        handoff_species : tuple of str, optional
            Species to include in handoff plots.


        Returns
        -------
        dict
            ``{"tier1_chemistry": {...}, "tier2_emissions": {...}}``
        """
        # If cell is an int, convert to dict using boxm_ds
        if isinstance(cell, int):
            boxm_ds = self.pp_gpat.boxm_ds_dict[job_id]
            boxm_ds_cell = boxm_ds.isel(cell=cell) if "cell" in boxm_ds.dims else None
            if boxm_ds_cell is not None:
                cell = {
                    "latitude": float(boxm_ds_cell.coords["latitude_c"]),
                    "longitude": float(boxm_ds_cell.coords["longitude_c"]),
                    "level": float(boxm_ds_cell.coords["level_c"]),
                }

        boxm_ds_cell = self._compat_boxm_ds_cell(
            self._find_boxm_ds_cell(self.pp_gpat.boxm_ds_dict[job_id], cell)
        )

        species = boxm_ds_cell["species"].values
        species_num = np.asarray(boxm_ds_cell["species_boxm_num"].values, dtype=int)
        y0 = np.asarray(boxm_ds_cell["Y_bg_c"].values, dtype=float)

        print("\nGPAT initial state:")
        for n, s, v in sorted(zip(species_num, species, y0), key=lambda x: x[0]):
            if str(s) in {"NO", "NO2", "O3", "OH", "HO2", "CO", "CH4", "HNO3"}:
                print(f"Y{n:3d} {str(s):>8s} {v:.6e}")

        boxm_out_cell = self._compat_boxm_ds_cell(
            self._find_boxm_ds_cell(self.pp_gpat.boxm_out_dict[job_id], cell)
        )

        # Normalise cell dict to compat names used by helpers
        cell = {
            "latitude": float(boxm_out_cell.coords.get("latitude_c", boxm_out_cell.coords.get("latitude"))),
            "longitude": float(boxm_out_cell.coords.get("longitude_c", boxm_out_cell.coords.get("longitude"))),
            "level": float(boxm_out_cell.coords.get("level_c", boxm_out_cell.coords.get("level"))),
        }

        # ── Tier 1: Chemistry kernel (zero emissions) ──
        self.gen_boxm_orig_input(boxm_ds_cell, job_id)
        self.gen_zen_file(boxm_ds_cell, job_id)
        self.gen_emi_file(boxm_ds_cell, cell, job_id, with_emissions=False)

        subprocess.call(
            [self.pp_gpat.run_path + "boxm_orig", self.pp_gpat.data_path, job_id],
        )

        print(boxm_out_cell)

        self.update_boxm_out_cell(boxm_out_cell, job_id)

        print(boxm_out_cell["Y_bg_c"].sel(species_out="CO").values)

        tier1 = self.compare_chemistry_kernel(job_id, boxm_ds_cell, boxm_out_cell, cell)

        # plot time series comparison for Tier 1 chemistry kernel validation
        _cell_c_int = self._cell_dict_to_index(self.pp_gpat.boxm_ds_dict[job_id], cell)

        # Tier 1 compares no-emission chemistry; for jobs with flights, only
        # use the pre-emission window to avoid plume-response contamination.
        tier1_time_idx_max = None
        fl_ds = self.pp_gpat.fl_ds_dict.get(job_id)
        if fl_ds is not None and "time_idx" in fl_ds:
            try:
                first_emi_time_idx = int(np.nanmin(np.asarray(fl_ds["time_idx"].values, dtype=int)))
                tier1_time_idx_max = first_emi_time_idx - 1
            except Exception:
                tier1_time_idx_max = None
        
        if plot_validation == True:
            self.pp_gpat.plotting.plot_cell_timeseries(
                job_id,
                cell_c=_cell_c_int,
                cell_f=None,
                compare_boxm_orig=True,
                show_del=False,
                time_idx_max=tier1_time_idx_max,
            )
        
        self.gen_emi_file(boxm_ds_cell, cell, job_id, with_emissions=True)

        subprocess.call(
            [self.pp_gpat.run_path + "boxm_orig", self.pp_gpat.data_path, job_id],
        )

        tier2 = self.compare_emission_response(job_id, boxm_ds_cell, boxm_out_cell, cell)

        # plot time series comparison for Tier 2 emission response validation
        if plot_validation == True:
            self.pp_gpat.plotting.plot_cell_timeseries(
                job_id,
                cell_c=_cell_c_int,
                cell_f=None,
                compare_boxm_orig=True,
                species=("NO", "NO2", "O3", "OH", "HO2", "CO", "CH4", "HNO3"),
                show_del=True,
            )

        # ── Handoff-aware diagnostic plot ──────────────────────────────────────
        handoff_info = {}
        if plot_handoff:
            _cell_c_int = self._cell_dict_to_index(self.pp_gpat.boxm_ds_dict[job_id], cell)
            handoff_time_idx = self.get_handoff_time_idx(job_id, cell_c=_cell_c_int, cell_f=cell_f)

            print("\n" + "=" * 60)
            print("Handoff-aware GPAT diagnostic")
            print("=" * 60)
            print(f"cell_c={_cell_c_int}, cell_f={cell_f}, handoff_time_idx={handoff_time_idx}")

            self.pp_gpat.plotting.plot_cell_timeseries(
                job_id,
                cell_c=_cell_c_int,
                cell_f=cell_f,
                compare_boxm_orig=False,
                species=handoff_species,
                show_del=True,
            )

            handoff_info = {
                "cell_c": _cell_c_int,
                "cell_f": cell_f,
                "handoff_time_idx": handoff_time_idx,
            }

        return {"tier1_chemistry": tier1, "tier2_emissions": tier2, "handoff_info": handoff_info}

    def gen_boxm_orig_input(self, boxm_ds_cell, job_id):
        """Generate the input file for the original box model."""

        # Write where boxm_orig.for expects: outputs/<job_id>/boxm_input.txt
        output_dir = pathlib.Path(self.pp_gpat.data_path) / "outputs" / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        boxm_orig_path = output_dir / "boxm_input.txt"
        if boxm_orig_path.exists():
            boxm_orig_path.unlink()

        # open file using a context manager
        with open(boxm_orig_path, "w") as boxm_input:
            start_time = pd.to_datetime(boxm_ds_cell["time"].values[0])
            end_time = pd.to_datetime(boxm_ds_cell["time"].values[-1])
            runtime = int((end_time - start_time) / np.timedelta64(1, "D")) % 365
            day = start_time.day
            month = start_time.month
            year = start_time.year
            altitude = boxm_ds_cell["altitude"].item()
            plevel = boxm_ds_cell["level"].item()
            level = get_pressure_level(altitude)
            longitude = boxm_ds_cell["longitude"].item()
            longbox = longitude_to_longbox(longitude)
            latitude = boxm_ds_cell["latitude"].item()
            latbox = latitude_to_latbox(latitude)
            M = boxm_ds_cell["M"].values[0]
            # P = boxm_ds_cell["air_pressure"].item()
            H2O = boxm_ds_cell["H2O"].values[0]
            temp = boxm_ds_cell["air_temperature"].values[0]

            boxm_input.write(
                f"{day}\n{month}\n{year}\n{level}\n{longbox}\n{latbox}\n{runtime}\n{M}\n{plevel}"
                f"\n{H2O}\n{temp}\n"
            )
            for s in [
                "NO2",
                "NO",
                "O3",
                "CO",
                "CH4",
                "HCHO",
                "CH3CHO",
                "CH3COCH3",
                "C2H6",
                "C2H4",
                "C3H8",
                "C3H6",
                "C2H2",
                "NC4H10",
                "TBUT2ENE",
                "BENZENE",
                "TOLUENE",
                "OXYL",
                "C5H8",
                "H2O2",
                "HNO3",
                "C2H5CHO",
                "CH3OH",
                "MEK",
                "CH3OOH",
                "PAN",
                "MPAN",
            ]:
                boxm_input.write(f"{boxm_ds_cell['Y_bg_c'].sel(species=s).item()}\n")

    def gen_zen_file(self, boxm_ds_cell, job_id):
        """Generate the ZEN file for the original box model."""

        # Write where boxm_orig.for expects: outputs/<job_id>/zen.csv
        output_dir = pathlib.Path(self.pp_gpat.data_path) / "outputs" / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        zen_file_path = output_dir / "zen.csv"
        if zen_file_path.exists():
            zen_file_path.unlink()

        # Extract the sza data and convert it to a DataFrame
        sza_data = boxm_ds_cell["sza"].values
        sza_df = pd.DataFrame(sza_data, columns=["sza"])

        # Write the DataFrame to a CSV file
        sza_df.to_csv(zen_file_path, index=False, header=False)

    def gen_emi_file(self, boxm_ds_cell, cell, job_id, with_emissions=False):
        """Generate the EMI file for boxm_orig.

        Parameters
        ----------
        boxm_ds_cell : xr.Dataset
            Input dataset sliced to the cell under test.
        cell : dict
            Cell selector with latitude, longitude, altitude.
        job_id : str
            Job ID.
        with_emissions : bool
            If False, write zeros (Tier 1 bg-only run).  If True, extract
            Y_del_c from boxm_out.nc for the 9 EMI species and write those
            as the emission perturbation (molec/cm³) per timestep.
        """
        # Write where boxm_orig.for expects: outputs/<job_id>/emi.csv
        output_dir = pathlib.Path(self.pp_gpat.data_path) / "outputs" / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        emi_file_path = output_dir / "emi.csv"
        if emi_file_path.exists():
            emi_file_path.unlink()

        n_timesteps = len(boxm_ds_cell["time"])

        if not with_emissions:
            zeros = np.zeros((n_timesteps, 9))
            pd.DataFrame(zeros).to_csv(emi_file_path, index=False, header=False)
            return

        # Extract Y_del_c for this cell from boxm_out
        boxm_out = self.pp_gpat.boxm_out_dict[job_id]
        out_cell = self._find_out_cell(boxm_out, cell)

        emi_array = np.zeros((len(out_cell["time"]), 9))
        for emi_idx, _y_idx, sp_name in self.EMI_SPECIES_MAP:
            if sp_name in out_cell["species_out"].values:
                emi_array[:, emi_idx] = out_cell["Y_del_c"].sel(species_out=sp_name).values

        # boxm_orig may run at finer timestep (DTS=20s) than boxm_out;
        # interpolate to match the number of timesteps boxm_orig expects.
        if len(out_cell["time"]) != n_timesteps:
            from scipy.interpolate import interp1d
            x_out = np.linspace(0, 1, len(out_cell["time"]))
            x_in = np.linspace(0, 1, n_timesteps)
            interp_fn = interp1d(x_out, emi_array, axis=0, kind="linear", fill_value="extrapolate")
            emi_array = interp_fn(x_in)

        pd.DataFrame(emi_array).to_csv(emi_file_path, index=False, header=False)

    def update_boxm_out_cell(self, boxm_out_cell, job_id):
        """Update the chemical dataset with the output from the original box model.

        Parameters
        ----------
        boxm_out_cell : xr.Dataset
            An xarray Dataset containing the chemical data for the cell.
        job_id : str
            The job ID for the simulation.

        Returns
        -------
        xr.Dataset
            An xarray Dataset containing the updated chemical data.
        """

        outputs_dir = pathlib.Path(f"{self.pp_gpat.data_path}outputs/{job_id}")
        sza_df = pd.read_csv(
            outputs_dir / "ZEN.OUT", header=0, names=["TIME", "ZEN"], dtype=np.float64
        )

        J_df = pd.read_csv(
            outputs_dir / "J.OUT",
            header=0,
            names=[
                "TIME",
                "J1",
                "J2",
                "J3",
                "J4",
                "J5",
                "J6",
                "J7",
                "J8",
                "J9",
                "J10",
                "J11",
                "J12",
                "J13",
                "J14",
                "J15",
                "J16",
                "J17",
                "J18",
                "J19",
                "J20",
                "J21",
                "J22",
                "J23",
                "J24",
                "J25",
                "J26",
                "J27",
                "J28",
                "J29",
                "J30",
                "J31",
                "J32",
                "J33",
                "J34",
                "J35",
                "J36",
                "J37",
                "J38",
                "J39",
                "J40",
                "J41",
                "J42",
                "J43",
                "J44",
                "J45",
                "J46",
                "J47",
                "J48",
                "J49",
                "J50",
            ],
            dtype=np.float64,
        )

        DJ_df = pd.read_csv(
            outputs_dir / "DJ.OUT",
            header=0,
            names=[
                "TIME",
                "DJ1",
                "DJ2",
                "DJ3",
                "DJ4",
                "DJ5",
                "DJ6",
                "DJ7",
                "DJ8",
                "DJ9",
                "DJ10",
                "DJ11",
                "DJ12",
                "DJ13",
                "DJ14",
                "DJ15",
                "DJ16",
                "DJ17",
                "DJ18",
                "DJ19",
                "DJ20",
                "DJ21",
                "DJ22",
                "DJ23",
                "DJ24",
                "DJ25",
                "DJ26",
                "DJ27",
                "DJ28",
                "DJ29",
                "DJ30",
                "DJ31",
                "DJ32",
                "DJ33",
                "DJ34",
                "DJ35",
                "DJ36",
                "DJ37",
                "DJ38",
                "DJ39",
                "DJ40",
                "DJ41",
                "DJ42",
                "DJ43",
                "DJ44",
                "DJ45",
                "DJ46",
                "DJ47",
                "DJ48",
                "DJ49",
                "DJ50",
            ],
            dtype=np.float64,
        )

        RC_df = pd.read_csv(
            outputs_dir / "RC.OUT",
            header=0,
            names=[
                "TIME",
                "RC1",
                "RC2",
                "RC3",
                "RC4",
                "RC5",
                "RC6",
                "RC7",
                "RC8",
                "RC9",
                "RC10",
                "RC11",
                "RC12",
                "RC13",
                "RC14",
                "RC15",
                "RC16",
                "RC17",
                "RC18",
                "RC19",
                "RC20",
                "RC21",
                "RC22",
                "RC23",
                "RC24",
                "RC25",
                "RC26",
                "RC27",
                "RC28",
                "RC29",
                "RC30",
                "RC31",
                "RC32",
                "RC33",
                "RC34",
                "RC35",
                "RC36",
                "RC37",
                "RC38",
                "RC39",
                "RC40",
                "RC41",
                "RC42",
                "RC43",
                "RC44",
                "RC45",
                "RC46",
                "RC47",
                "RC48",
                "RC49",
                "RC50",
            ],
            dtype=np.float64,
        )

        # get species names — Y.OUT has native header TIME,Y1,...,Y219
        Y_df = pd.read_csv(outputs_dir / "Y.OUT", dtype=np.float64)

        # # Update the chem_ds_stacked with the new data
        # Update zen data
        boxm_out_cell["sza_orig"] = ("time", da.zeros(boxm_out_cell.sizes["time"]))
        zen_vals = sza_df["ZEN"].values * np.pi / 180
        target_len = boxm_out_cell["sza_orig"].shape[0]
        if len(zen_vals) != target_len:
            # Interpolate or truncate to match the target length
            x_old = np.linspace(0, 1, len(zen_vals))
            x_new = np.linspace(0, 1, target_len)
            zen_vals = np.interp(x_new, x_old, zen_vals)
        boxm_out_cell["sza_orig"].loc[:] = zen_vals

        boxm_out_cell["J_orig"] = xr.DataArray(
            da.zeros((boxm_out_cell.sizes["time"], 5)),
            dims=("time", "photol_params")
        )
        for pp, photol_params in enumerate(J_df.columns[1:6]):
            j_vals = J_df[photol_params].values
            target_len = boxm_out_cell["J_orig"].shape[0]
            if len(j_vals) != target_len:
                x_old = np.linspace(0, 1, len(j_vals))
                x_new = np.linspace(0, 1, target_len)
                j_vals = np.interp(x_new, x_old, j_vals)
            boxm_out_cell["J_orig"].loc[:, pp] = j_vals

        boxm_out_cell["DJ_orig"] = xr.DataArray(
            da.zeros((boxm_out_cell.sizes["time"], 5)),
            dims=("time", "photol_coeffs")
        )
        for pc, photol_coeffs in enumerate(DJ_df.columns[1:6]):
            dj_vals = DJ_df[photol_coeffs].values
            target_len = boxm_out_cell["DJ_orig"].shape[0]
            if len(dj_vals) != target_len:
                x_old = np.linspace(0, 1, len(dj_vals))
                x_new = np.linspace(0, 1, target_len)
                dj_vals = np.interp(x_new, x_old, dj_vals)
            boxm_out_cell["DJ_orig"].loc[:, pc] = dj_vals

        boxm_out_cell["RC_orig"] = xr.DataArray(
            da.zeros((boxm_out_cell.sizes["time"], 5)),
            dims=("time", "therm_coeffs")
        )
        for tc, therm_coeffs in enumerate(RC_df.columns[1:6]):
            rc_vals = RC_df[therm_coeffs].values
            target_len = boxm_out_cell["RC_orig"].shape[0]
            if len(rc_vals) != target_len:
                x_old = np.linspace(0, 1, len(rc_vals))
                x_new = np.linspace(0, 1, target_len)
                rc_vals = np.interp(x_new, x_old, rc_vals)
            boxm_out_cell["RC_orig"].loc[:, tc] = rc_vals

        boxm_out_cell["Y_orig"] = xr.DataArray(
            da.zeros((boxm_out_cell.sizes["time"], boxm_out_cell.sizes["species_out"])),
            dims=("time", "species_out")
        )
        target_len = boxm_out_cell["Y_orig"].shape[0]
        for _s, species in enumerate(boxm_out_cell["species_out"].values):
            # Use Fortran 1-based index to select the correct column from Y.OUT
            sp_num = int(boxm_out_cell["species_out_num"].sel(species_out=species).values.item())
            y_vals = Y_df[f"Y{sp_num}"].values
            if len(y_vals) != target_len:
                x_old = np.linspace(0, 1, len(y_vals))
                x_new = np.linspace(0, 1, target_len)
                y_vals = np.interp(x_new, x_old, y_vals)
            boxm_out_cell["Y_orig"].loc[:, species] = y_vals

        return boxm_out_cell

    @staticmethod
    def _cell_dict_to_index(boxm_ds, cell):
        """Return 1-based coarse cell index matching a cell dict in boxm_ds.

        Parameters
        ----------
        boxm_ds : xr.Dataset
            Stacked input dataset with a 'cell' dimension.
        cell : dict
            Dict with 'latitude', 'longitude', 'level' keys.

        Returns
        -------
        int
            1-based cell index (Fortran convention).
        """
        if "cell" not in boxm_ds.dims:
            return 1  # unstacked; caller will use lat/lon/level directly
        lat_key = "latitude_c" if "latitude_c" in boxm_ds else "latitude"
        lon_key = "longitude_c" if "longitude_c" in boxm_ds else "longitude"
        lev_key = "level_c"     if "level_c"     in boxm_ds else "level"
        lat_vals = np.asarray(boxm_ds[lat_key].values, dtype=float)
        lon_vals = np.asarray(boxm_ds[lon_key].values, dtype=float)
        lev_vals = np.asarray(boxm_ds[lev_key].values, dtype=float)
        dist = (
            (lat_vals - float(cell["latitude"]))  ** 2
            + (lon_vals - float(cell["longitude"])) ** 2
            + ((lev_vals - float(cell["level"])) / 1000.0) ** 2
        )
        return int(np.argmin(dist)) + 1  # convert 0-based → 1-based

    def r_sq(y_true, y_pred):
        """Calculate the R-squared value for a model."""
        y_true_mean = np.mean(y_true)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true_mean) ** 2)

        return 1 - (ss_res / ss_tot)

    def compare_chemistry_kernel(self, job_id, boxm_ds_cell, boxm_out_cell, cell):
        """Tier 1: Compare boxm_orig Y.OUT against boxm_out.nc Y_bg_c.

        Both codes run the same CHEMCO/PHOTOL/DERIV routines on the same
        initial state with no emissions, so the time series should match
        to floating-point precision.

        Parameters
        ----------
        job_id : str
            Job ID.
        boxm_ds_cell : xr.Dataset
            Input dataset sliced to the cell under test.
        cell : dict
            Cell selector with latitude, longitude, altitude.

        Returns
        -------
        dict
            Per-species metrics: NRMSE, correlation, RMSE.
        """
        # Read Y.OUT — has a header row "TIME,Y1,...,Y219", then CSV data.
        # Y.OUT contains ALL 219 Fortran Y-array species in ppb.
        outputs_dir = pathlib.Path(f"{self.pp_gpat.data_path}outputs/{job_id}")

        # Use the provided output cell directly.
        new_cell = boxm_out_cell
        new_species = list(new_cell["species_out"].values)

        # For jobs with flights, restrict Tier 1 metrics to the no-emission
        # window before first emission time index.
        tier1_mask = None
        fl_ds = self.pp_gpat.fl_ds_dict.get(job_id)
        if fl_ds is not None and "time_idx" in fl_ds and "time_idx" in new_cell.coords:
            try:
                first_emi_time_idx = int(np.nanmin(np.asarray(fl_ds["time_idx"].values, dtype=int)))
                out_time_idx = np.asarray(new_cell["time_idx"].values, dtype=int)
                tier1_mask = out_time_idx < first_emi_time_idx
            except Exception:
                tier1_mask = None

        # Read Y.OUT using native header (TIME, Y1, ..., Y219); select columns by Fortran index
        y_df = pd.read_csv(outputs_dir / "Y.OUT", dtype=np.float64)
        orig_time = y_df["TIME"].values  # seconds

        print(f"\n{'='*60}")
        print(f"Tier 1: Chemistry kernel comparison (cell lat={cell['latitude']:.2f}, "
              f"lon={cell['longitude']:.2f}, alt={cell['level']:.0f})")
        print(f"{'='*60}")
        print(f"  boxm_orig timesteps: {len(orig_time)}")
        print(f"  boxm_out  timesteps: {len(new_cell['time'])}")
        print(f"  {'Species':>10s}  {'NRMSE':>12s}  {'Correlation':>12s}  {'Max Abs Err':>12s}")
        print(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}")

        results = {}
        for sp in new_species:
            # Get the Fortran 1-based Y-array index for this species
            sp_num = int(new_cell["species_out_num"].sel(species_out=sp).values.item())
            col = f"Y{sp_num}"
            if col not in y_df.columns:
                continue
            orig_vals = y_df[col].values  # ppb at correct Fortran column

            # Y_bg_c is in ppb in boxm_out (units attribute confirmed)
            new_vals_raw = new_cell["Y_bg_c"].sel(species_out=sp).values
            new_vals = new_vals_raw  # already ppb — no conversion needed

            # Interpolate original (Y.OUT TIME in seconds) onto new model times.
            if "time_rel_s" in new_cell.coords and "TIME" in y_df.columns:
                new_times = np.asarray(new_cell["time_rel_s"].values, dtype=float)
                orig_times = np.asarray(y_df["TIME"].values, dtype=float)
                orig_interp = np.interp(new_times, orig_times, orig_vals)
            else:
                new_times = np.arange(len(new_vals), dtype=float)
                orig_times_norm = np.linspace(0, len(new_vals) - 1, len(orig_vals))
                orig_interp = np.interp(new_times, orig_times_norm, orig_vals)

            if tier1_mask is not None:
                if np.any(tier1_mask):
                    orig_interp = orig_interp[tier1_mask]
                    new_vals = new_vals[tier1_mask]
                else:
                    continue

            diff = orig_interp - new_vals
            rmse = np.sqrt(np.mean(diff**2))
            mean_val = np.mean(np.abs(orig_interp)) + 1e-30
            nrmse = rmse / mean_val
            max_err = np.max(np.abs(diff))
            corr = np.corrcoef(orig_interp, new_vals)[0, 1] if len(new_vals) > 1 else np.nan

            results[sp] = {"nrmse": nrmse, "correlation": corr, "rmse": rmse, "max_abs_error": max_err}
            print(f"  {sp:>10s}  {nrmse:12.4e}  {corr:12.6f}  {max_err:12.4e}")

        return results

    def compare_emission_response(self, job_id, boxm_ds_cell, boxm_out_cell, cell):
        """Tier 2: Compare boxm_orig Y.OUT (with EMI) against boxm_out.nc Y_total.

        boxm_orig was run with Y_del_c fed as the EMI perturbation, so its
        Y output = bg + emission response.  The new model's equivalent is
        Y_bg_c + Y_del_c.  We compare the two for the 9 EMI species.

        Parameters
        ----------
        job_id : str
            Job ID.
        boxm_ds_cell : xr.Dataset
            Input dataset sliced to the cell under test.
        boxm_out_cell : xr.Dataset
            Output dataset sliced to the cell under test.
        cell : dict
            Cell selector with latitude, longitude, altitude.

        Returns
        -------
        dict
            Per-species comparison metrics and physical-consistency checks.
        """
        # Read boxm_orig Y.OUT (now with emissions applied)
        outputs_dir = pathlib.Path(f"{self.pp_gpat.data_path}outputs/{job_id}")

        # Read boxm_out.nc Y_total = Y_bg_c + Y_del_c for the same cell
        boxm_out = self.pp_gpat.boxm_out_dict[job_id]
        out_cell = self._find_out_cell(boxm_out, cell)
        new_species = list(out_cell["species_out"].values)

        # Read Y.OUT using native header (TIME, Y1, ..., Y219)
        y_df = pd.read_csv(outputs_dir / "Y.OUT", dtype=np.float64)
        orig_time = y_df["TIME"].values

        print(f"\n{'='*60}")
        print(f"Tier 2: Emission response comparison (same cell)")
        print(f"{'='*60}")
        print(f"  {'Species':>10s}  {'NRMSE':>12s}  {'Correlation':>12s}  {'Max Abs Err':>12s}")
        print(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}")

        results = {}

        for emi_idx, _y_idx, sp in self.EMI_SPECIES_MAP:
            if sp not in new_species:
                continue
            # Get Fortran 1-based index for this species
            sp_num = int(out_cell["species_out_num"].sel(species_out=sp).values.item())
            col = f"Y{sp_num}"
            if col not in y_df.columns:
                continue
            orig_vals = y_df[col].values  # ppb at correct Fortran column

            # New model total = Y_bg_c + Y_del_c — both in ppb, no conversion needed
            y_bg = out_cell["Y_bg_c"].sel(species_out=sp).values
            y_del = out_cell["Y_del_c"].sel(species_out=sp).values if "Y_del_c" in out_cell else np.zeros_like(y_bg)
            new_total_ppb = y_bg + y_del

            # Interpolate original onto new model times (seconds).
            if "time_rel_s" in out_cell.coords and "TIME" in y_df.columns:
                new_times = np.asarray(out_cell["time_rel_s"].values, dtype=float)
                orig_times = np.asarray(y_df["TIME"].values, dtype=float)
                orig_interp = np.interp(new_times, orig_times, orig_vals)
            else:
                new_times = np.arange(len(new_total_ppb), dtype=float)
                orig_times_norm = np.linspace(0, len(new_total_ppb) - 1, len(orig_vals))
                orig_interp = np.interp(new_times, orig_times_norm, orig_vals)

            diff = orig_interp - new_total_ppb
            rmse = np.sqrt(np.mean(diff**2))
            mean_val = np.mean(np.abs(orig_interp)) + 1e-30
            nrmse = rmse / mean_val
            max_err = np.max(np.abs(diff))
            corr = np.corrcoef(orig_interp, new_total_ppb)[0, 1] if len(new_total_ppb) > 1 else np.nan

            results[sp] = {"nrmse": nrmse, "correlation": corr, "rmse": rmse, "max_abs_error": max_err}
            print(f"  {sp:>10s}  {nrmse:12.4e}  {corr:12.6f}  {max_err:12.4e}")

        # Physical consistency checks on new model
        if "Y_del_c" in out_cell:
            y_del_all = np.asarray(out_cell["Y_del_c"].values)
            y_bg_all = np.asarray(out_cell["Y_bg_c"].values)
            y_total_all = y_bg_all + y_del_all
            n_negative = int((y_total_all < -1e-20).sum())
            results["n_negative_total"] = n_negative
            if n_negative > 0:
                worst = float(y_total_all.min())
                print(f"  WARNING: {n_negative} negative Y_total values, worst = {worst:.4e}")
            else:
                print(f"  All Y_total values non-negative: PASS")

        return results

    def boxm_test_fine(self, job_id, cell_c=None, cell_f=None, n_cells=3):
        """Tier 3: Fine-cell chemistry validation against boxm_orig.

        Pick fine (patch) cells from patch_table.nc with varying emission
        magnitudes, extract their Y_del_f time series, feed as EMI to
        boxm_orig, and compare the resulting Y_total against
        Y_bg_c(parent coarse cell) + Y_del_f.

        Fine cells share meteorology with their parent coarse cell, so
        boxm_orig is initialised identically to the coarse-cell Tier 2
        test but with fine-cell perturbations as EMI.

        Parameters
        ----------
        job_id : str
            Job ID.
        cell_c : int, optional
            1-based coarse cell index (Fortran convention).  If None,
            auto-select cells.
        cell_f : int, optional
            1-based fine sub-cell index within coarse cell.  If None,
            auto-select cells.
        n_cells : int
            Number of cells to test when auto-selecting.

        Returns
        -------
        dict
            ``{cell_key: {species: metrics, ...}, ...}``
        """
        pt = self.pp_gpat.patch_table_dict.get(job_id)
        if pt is None:
            print(f"No patch_table loaded for job {job_id}")
            return {}

        boxm_ds = self.pp_gpat.boxm_ds_dict[job_id]

        if cell_c is not None and cell_f is not None:
            cells = [(int(cell_c), int(cell_f))]
        else:
            cells = self._select_fine_cells(job_id, n_cells)

        if not cells:
            print("No fine cells available for testing.")
            return {}

        results = {}
        for cc, cf in cells:
            key = f"cell_c={cc}_cell_f={cf}"
            parent_idx = cc - 1  # Fortran 1-based → Python 0-based

            boxm_ds_cell = self._compat_boxm_ds_cell(boxm_ds.isel(cell=parent_idx))
            cell = {
                "latitude": boxm_ds_cell["latitude"],
                "longitude": boxm_ds_cell["longitude"],
                "level": boxm_ds_cell["level"],
            }

            # Reuse existing helpers for initial conditions and SZA
            self.gen_boxm_orig_input(boxm_ds_cell, job_id)
            self.gen_zen_file(boxm_ds_cell, job_id)
            self.gen_emi_file_fine(boxm_ds_cell, job_id, cc, cf)

            subprocess.call(
                [self.pp_gpat.run_path + "boxm_orig", self.pp_gpat.data_path, job_id],
            )

            results[key] = self.compare_fine_emission_response(
                job_id, boxm_ds_cell, cell, cc, cf
            )

        return results

    def plot_boxm_validation_timeseries(self, boxm_ds_cell, boxm_out_cell, job_id, tier="chemistry_kernel"):
            """Plot time series comparison for boxm_orig vs new model for Tier 1 or Tier 2.

            One subplot per species, with R² annotated on each panel.

            Parameters
            ----------
            boxm_ds_cell : xr.Dataset
                Input dataset sliced to the cell under test.
            boxm_out_cell : xr.Dataset
                Output dataset sliced to the cell under test.
            job_id : str
                Job ID.
            tier : str
                "chemistry_kernel" for Tier 1, "emission_response" for Tier 2.
            """
            import math

            # New model species available in this cell
            new_species = list(boxm_out_cell["species_out"].values)

            # Read Y.OUT from boxm_orig with native header (TIME, Y1, ..., Y219; ppb)
            outputs_dir = pathlib.Path(f"{self.pp_gpat.data_path}outputs/{job_id}")
            y_df = pd.read_csv(outputs_dir / "Y.OUT", dtype=np.float64)

            # All species_out are always available via Fortran Y-index
            plot_species = list(new_species)
            if not plot_species:
                print("No matching species to plot.")
                return

            if tier == "chemistry_kernel":
                label_new = "New model ($Y_{bg}$)"
                label_orig = "boxm_orig (no EMI)"
            else:
                label_new = "New model ($Y_{bg} + Y_{del}$)"
                label_orig = "boxm_orig (with EMI)"

            n_species = len(plot_species)
            n_cols = 3
            n_rows = math.ceil(n_species / n_cols)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False)

            for i, sp in enumerate(plot_species):
                ax = axes[i // n_cols][i % n_cols]
                # Get correct Fortran Y-array column for this species
                sp_num = int(boxm_out_cell["species_out_num"].sel(species_out=sp).values.item())
                orig_vals = y_df[f"Y{sp_num}"].values  # ppb

                if tier == "chemistry_kernel":
                    new_vals = boxm_out_cell["Y_bg_c"].sel(species_out=sp).values  # already ppb
                else:
                    y_bg = boxm_out_cell["Y_bg_c"].sel(species_out=sp).values
                    y_del = (
                        boxm_out_cell["Y_del_c"].sel(species_out=sp).values
                        if "Y_del_c" in boxm_out_cell
                        else np.zeros_like(y_bg)
                    )
                    new_vals = y_bg + y_del  # both in ppb — no conversion needed

                # Interpolate orig (fine timestep) onto new model time grid
                new_times = np.arange(len(new_vals), dtype=float)
                orig_times_norm = np.linspace(0, len(new_vals) - 1, len(orig_vals))
                orig_interp = np.interp(new_times, orig_times_norm, orig_vals)

                r2 = self.r_sq(orig_interp, new_vals)

                ax.plot(new_times, orig_interp, color="tab:blue", linestyle="--", linewidth=1.0, label=label_orig)
                ax.plot(new_times, new_vals, color="tab:orange", linewidth=1.0, label=label_new)
                ax.set_title(sp, fontsize=10)
                ax.set_xlabel("Time index", fontsize=8)
                ax.set_ylabel("ppb", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
                ax.text(0.97, 0.05, f"$R^2$ = {r2:.3f}", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=8,
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

            # Add shared legend from first subplot, hide unused axes
            handles, labels = axes[0][0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9,
                       bbox_to_anchor=(0.5, -0.02))
            for j in range(n_species, n_rows * n_cols):
                axes[j // n_cols][j % n_cols].set_visible(False)

            fig.suptitle(
                f"Box model validation — {tier.replace('_', ' ').title()} ({job_id})",
                fontsize=12, y=1.01,
            )
            plt.tight_layout()
            plt.savefig(
                pathlib.Path(f"{self.pp_gpat.data_path}outputs/{job_id}") / f"boxm_validation_{tier}.png",
                dpi=150, bbox_inches="tight",
            )
            plt.show()

    # ------------------------------------------------------------------
    # Helpers for boxm_test_fine
    # ------------------------------------------------------------------

    @staticmethod
    def _find_out_cell(boxm_out, cell):
        """Find the nearest cell in a stacked output dataset."""
        if "cell" in boxm_out.dims:
            lat_key = "latitude_c" if "latitude_c" in boxm_out else "latitude"
            lon_key = "longitude_c" if "longitude_c" in boxm_out else "longitude"
            lev_key = "level_c" if "level_c" in boxm_out else "level"
            lat_vals = np.asarray(boxm_out[lat_key].values, dtype=float)
            lon_vals = np.asarray(boxm_out[lon_key].values, dtype=float)
            lev_vals = np.asarray(boxm_out[lev_key].values, dtype=float)
            dist = (
                (lat_vals - float(cell["latitude"])) ** 2
                + (lon_vals - float(cell["longitude"])) ** 2
                + ((lev_vals - float(cell["level"])) / 1000.0) ** 2
            )
            idx = int(np.argmin(dist))
            return boxm_out.isel(cell=idx)
        result = boxm_out.sel(
            latitude_c=float(cell["latitude"]),
            longitude_c=float(cell["longitude"]),
            method="nearest",
        )
        # Also select altitude/level if present
        if "level_c" in result.dims and "level" in cell:
            lev_vals = np.asarray(result["level_c"].values, dtype=float)
            level_idx = int(np.argmin(np.abs(lev_vals - float(cell["level"]))))
            result = result.isel(level_c=level_idx)
        return result

    @staticmethod
    def _find_boxm_ds_cell(boxm_ds, cell):
        """Find the nearest cell in the stacked boxm_ds matching lat/lon/level.

        Works with both stacked (cell dim) and unstacked datasets.
        """
        if "cell" in boxm_ds.dims:
            lat_key = "latitude_c" if "latitude_c" in boxm_ds else "latitude"
            lon_key = "longitude_c" if "longitude_c" in boxm_ds else "longitude"
            lev_key = "level_c" if "level_c" in boxm_ds else "level"
            lat_vals = np.asarray(boxm_ds[lat_key].values, dtype=float)
            lon_vals = np.asarray(boxm_ds[lon_key].values, dtype=float)
            lev_vals = np.asarray(boxm_ds[lev_key].values, dtype=float)
            dist = (
                (lat_vals - float(cell["latitude"])) ** 2
                + (lon_vals - float(cell["longitude"])) ** 2
                + ((lev_vals - float(cell["level"])) / 1000.0) ** 2
            )
            idx = int(np.argmin(dist))
            return boxm_ds.isel(cell=idx)
        # Unstacked: use available coordinate names
        lat_key = "latitude_c" if "latitude_c" in boxm_ds else "latitude"
        lon_key = "longitude_c" if "longitude_c" in boxm_ds else "longitude"
        lev_key = "level_c" if "level_c" in boxm_ds else "level"
        return boxm_ds.sel(
            **{
                lat_key: float(cell["latitude"]),
                lon_key: float(cell["longitude"]),
                lev_key: float(cell["level"]),
            },
            method="nearest",
        )

    @staticmethod
    def _compat_boxm_ds_cell(ds_cell):
        """Rename stacked boxm_ds coordinates to match gen_boxm_orig_input."""
        rename_map = {}
        for src, dst in [
            ("altitude_c", "altitude"),
            ("level_c", "level"),
            ("longitude_c", "longitude"),
            ("latitude_c", "latitude"),
            ("species_boxm", "species"),
        ]:
            if src in ds_cell and dst not in ds_cell:
                rename_map[src] = dst
        return ds_cell.rename(rename_map) if rename_map else ds_cell

    def get_handoff_time_idx(self, job_id, cell_c=None, cell_f=None):
        """Return handoff time_idx for a selected coarse/fine cell.

        Priority:
        1. explicit boxm_out variable/attr if present
        2. last fine patch time for the selected (cell_c, cell_f)
        3. last active coarse time for cell_c
        """
        boxm_out = self.pp_gpat.boxm_out_dict[job_id]
        pt = self.pp_gpat.patch_table_dict.get(job_id)

        # 1) explicit global handoff, if written by BOXM
        if "handoff_time_idx" in boxm_out:
            vals = np.asarray(boxm_out["handoff_time_idx"].values).ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size:
                return int(vals[0])

        if "handoff_time_idx" in boxm_out.attrs:
            try:
                return int(boxm_out.attrs["handoff_time_idx"])
            except Exception:
                pass

        # 2) last fine patch time for specific fine cell
        if pt is not None and cell_c is not None and cell_f is not None:
            cc = np.asarray(pt["row_cell_c"].values, dtype=int)
            cf = np.asarray(pt["row_cell_f"].values, dtype=int)
            ti = np.asarray(pt["time_idx"].values, dtype=int)
            mask = (cc == int(cell_c)) & (cf == int(cell_f))
            if np.any(mask):
                return int(np.max(ti[mask]))

        # 3) last active coarse time for selected coarse cell
        if cell_c is not None and "active_flag" in boxm_out:
            try:
                if "cell" in boxm_out.dims:
                    af = np.asarray(boxm_out["active_flag"].isel(cell=int(cell_c) - 1).values, dtype=bool)
                else:
                    return None
                tids = np.asarray(boxm_out["time_idx"].values, dtype=int)
                idx = np.flatnonzero(af)
                if idx.size:
                    return int(tids[idx[-1]])
            except Exception:
                pass

        return None

    def _select_fine_cells(self, job_id, n_cells=3, min_timesteps=5):
        """Auto-select fine cells spanning low/mid/high Y_del_f magnitude.

        Returns
        -------
        list[tuple[int, int]]
            (cell_c, cell_f) pairs, 1-based (Fortran convention).
        """
        pt = self.pp_gpat.patch_table_dict[job_id]
        cc_vals = np.asarray(pt["row_cell_c"].values, dtype=int)
        cf_vals = np.asarray(pt["row_cell_f"].values, dtype=int)
        y_del = np.asarray(pt["Y_del_f"].values, dtype=float)  # (row, species_out)

        row_mag = np.nansum(np.abs(y_del), axis=1)

        # Group by (cell_c, cell_f)
        pair_keys = np.stack([cc_vals, cf_vals], axis=1)
        unique_pairs = {(int(r[0]), int(r[1])) for r in pair_keys}

        pair_stats = []
        for cc, cf in unique_pairs:
            mask = (cc_vals == cc) & (cf_vals == cf)
            n_ts = int(mask.sum())
            if n_ts < min_timesteps:
                continue
            mean_mag = float(np.nanmean(row_mag[mask]))
            pair_stats.append((cc, cf, mean_mag, n_ts))

        if not pair_stats:
            return []

        pair_stats.sort(key=lambda x: x[2])
        n = min(n_cells, len(pair_stats))
        indices = np.linspace(0, len(pair_stats) - 1, n, dtype=int)
        selected = [(pair_stats[i][0], pair_stats[i][1]) for i in indices]

        print(f"Selected {len(selected)} fine cells for Tier 3:")
        for cc, cf in selected:
            stat = next(s for s in pair_stats if s[0] == cc and s[1] == cf)
            print(f"  cell_c={cc}, cell_f={cf}, "
                  f"mean |Y_del_f|={stat[2]:.4e} ppb, timesteps={stat[3]}")

        return selected

    def gen_emi_file_fine(self, boxm_ds_cell, job_id, cell_c, cell_f):
        """Generate EMI file from Y_del_f for a specific fine cell.

        Y_del_f in patch_table is in ppb; boxm_orig expects EMI in the
        same internal units as Y (number density = M * ppb / 1e9).

        Parameters
        ----------
        boxm_ds_cell : xr.Dataset
            Input dataset for the parent coarse cell (with compat names).
        job_id : str
        cell_c, cell_f : int
            1-based cell indices (Fortran convention).
        """
        input_dir = pathlib.Path(self.pp_gpat.inputs) / job_id
        emi_file_path = input_dir / "emi.csv"
        if emi_file_path.exists():
            emi_file_path.unlink()

        pt = self.pp_gpat.patch_table_dict[job_id]
        n_timesteps = len(boxm_ds_cell["time"])

        cc_vals = np.asarray(pt["row_cell_c"].values, dtype=int)
        cf_vals = np.asarray(pt["row_cell_f"].values, dtype=int)
        mask = (cc_vals == cell_c) & (cf_vals == cell_f)
        fine_rows = pt.isel(row=mask)

        emi_array = np.zeros((n_timesteps, 9))

        if len(fine_rows["row"]) == 0:
            pd.DataFrame(emi_array).to_csv(emi_file_path, index=False, header=False)
            return

        M_vals = boxm_ds_cell["M"].values  # shape (n_timesteps,)
        boxm_times = pd.to_datetime(boxm_ds_cell["time"].values)
        fine_times = pd.to_datetime(fine_rows["time"].values)

        for emi_idx, _y_idx, sp_name in self.EMI_SPECIES_MAP:
            if sp_name not in pt["species_out"].values:
                continue

            y_del_ppb = fine_rows["Y_del_f"].sel(species_out=sp_name).values

            for i, ft in enumerate(fine_times):
                t_idx = int(np.argmin(np.abs(boxm_times - ft)))
                if 0 <= t_idx < n_timesteps:
                    # ppb → number density (same units as boxm_orig Y)
                    emi_array[t_idx, emi_idx] = y_del_ppb[i] * M_vals[t_idx] / 1.0e9

        pd.DataFrame(emi_array).to_csv(emi_file_path, index=False, header=False)

    def compare_fine_emission_response(self, job_id, boxm_ds_cell, cell, cell_c, cell_f):
        """Tier 3: Compare boxm_orig Y.OUT against Y_bg_c(parent) + Y_del_f.

        Comparison is performed only at timesteps where the fine cell has
        data in patch_table.

        Parameters
        ----------
        job_id : str
        boxm_ds_cell : xr.Dataset
            Parent coarse cell input dataset.
        cell : dict
            Cell selector with latitude, longitude, altitude.
        cell_c, cell_f : int
            1-based cell indices (Fortran convention).

        Returns
        -------
        dict
            Per-species metrics and mean perturbation size.
        """
        outputs_dir = pathlib.Path(f"{self.pp_gpat.data_path}outputs/{job_id}")
        # Y.OUT has native header TIME,Y1..Y219. Use species_out_num from out_cell.
        y_df = pd.read_csv(outputs_dir / "Y.OUT", dtype=np.float64)

        pt = self.pp_gpat.patch_table_dict[job_id]
        cc_vals = np.asarray(pt["row_cell_c"].values, dtype=int)
        cf_vals = np.asarray(pt["row_cell_f"].values, dtype=int)
        mask = (cc_vals == cell_c) & (cf_vals == cell_f)
        fine_rows = pt.isel(row=mask)
        fine_times = pd.to_datetime(fine_rows["time"].values)

        # Parent coarse cell background from boxm_out
        boxm_out = self.pp_gpat.boxm_out_dict[job_id]
        parent_idx = cell_c - 1
        out_cell = boxm_out.isel(cell=parent_idx)

        M_val = boxm_ds_cell["M"].values[0]
        boxm_out_times = pd.to_datetime(out_cell["time"].values)

        n_fine = len(fine_rows["row"])
        n_orig = len(orig_y)

        print(f"\n{'='*60}")
        print(f"Tier 3: Fine-cell emission response "
              f"(cell_c={cell_c}, cell_f={cell_f})")
        print(f"  Parent: lat={cell['latitude'].item():.2f}, "
              f"lon={cell['longitude'].item():.2f}, "
              f"alt={cell['altitude'].item():.0f}")
        print(f"  Fine cell data points: {n_fine}")
        print(f"  boxm_orig timesteps:   {n_orig}")
        print(f"{'='*60}")
        print(f"  {'Species':>10s}  {'NRMSE':>12s}  {'Correlation':>12s}  "
              f"{'Max Abs Err':>12s}  {'Mean |del|':>12s}")
        print(f"  {'-'*10}  {'-'*12}  {'-'*12}  "
              f"{'-'*12}  {'-'*12}")

        results = {}
        out_species = out_cell["species_out"].values if "species_out" in out_cell else []

        # Fractional time axis for interpolating boxm_orig onto fine cell times
        if len(boxm_out_times) > 1:
            t_span = (boxm_out_times[-1] - boxm_out_times[0]).total_seconds()
        else:
            t_span = 1.0
        origin = boxm_out_times[0]

        for emi_idx, y_idx, sp in self.EMI_SPECIES_MAP:
            if sp not in pt["species_out"].values:
                continue
            if sp not in out_species:
                continue

            y_del_f_ppb = fine_rows["Y_del_f"].sel(species_out=sp).values
            mean_del = float(np.nanmean(np.abs(y_del_f_ppb)))

            # Build new-model total at each fine cell timestep
            new_total_ppb = np.zeros(n_fine)
            for i, ft in enumerate(fine_times):
                # Nearest boxm_out time for Y_bg_c (already ppb)
                t_out_idx = int(np.argmin(np.abs(boxm_out_times - ft)))
                y_bg_ppb = float(out_cell["Y_bg_c"].sel(species_out=sp).isel(time=t_out_idx).values)
                new_total_ppb[i] = y_bg_ppb + y_del_f_ppb[i]

            # Interpolate boxm_orig onto the fine cell times
            sp_num = int(out_cell["species_out_num"].sel(species_out=sp).values.item())
            col = f"Y{sp_num}"
            if col not in y_df.columns:
                continue
            orig_vals_all = y_df[col].values  # ppb at correct Fortran column
            orig_frac = np.linspace(0, 1, n_orig)
            fine_frac = np.array([
                (ft - origin).total_seconds() / max(t_span, 1.0)
                for ft in fine_times
            ])
            fine_frac = np.clip(fine_frac, 0, 1)
            orig_interp = np.interp(fine_frac, orig_frac, orig_vals_all)

            diff = orig_interp - new_total_ppb
            rmse = float(np.sqrt(np.mean(diff**2)))
            mean_abs = float(np.mean(np.abs(orig_interp))) + 1e-30
            nrmse = rmse / mean_abs
            max_err = float(np.max(np.abs(diff)))
            if n_fine > 1:
                corr = float(np.corrcoef(orig_interp, new_total_ppb)[0, 1])
            else:
                corr = np.nan

            results[sp] = {
                "nrmse": nrmse,
                "correlation": corr,
                "rmse": rmse,
                "max_abs_error": max_err,
                "mean_del_ppb": mean_del,
            }
            print(f"  {sp:>10s}  {nrmse:12.4e}  {corr:12.6f}  "
                  f"{max_err:12.4e}  {mean_del:12.4e}")

        return results


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

def latitude_to_latbox(latitude):
    """Convert latitude to latbox for the original box model.

    Parameters
    ----------
    latitude : float
        The latitude value.

    Returns
    -------
    int
        The latbox value.
    """

    # Map the latitude to the range 0-1
    normalized_latitude = (latitude + 87.5) / 180

    # Map the normalized latitude to the range 1-72
    latbox = normalized_latitude * 36 + 1

    # Round to the nearest integer and return
    return round(latbox)

def longitude_to_longbox(longitude):
    """Convert longitude to longbox for the original box model.

    Parameters
    ----------
    longitude : float
        The longitude value.

    Returns
    -------
    int
        The longbox value.
    """

    # Map the longitude to the range 0-1
    normalized_longitude = (longitude + 177.5) / 360

    # Map the normalized longitude to the range 1-144
    longbox = normalized_longitude * 72 + 1

    # Round to the nearest integer and return
    return round(longbox)

def get_pressure_level(alt):
    """Get the pressure level for the original box model.

    Parameters
    ----------
    alt : float
        The altitude value.

    Returns
    -------
    int
        The pressure level.
    """

    # Convert alt to pressure level (hPa)``
    chem_pressure_levels = np.array([962, 861, 759, 658, 556, 454, 353, 251, 150.5])

    # Convert altitude to pressure using a standard atmosphere model
    pressure = units.m_to_pl(alt)

    # Find the index of the closest value in the array
    return (np.abs(chem_pressure_levels - pressure)).argmin()

