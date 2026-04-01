import numpy as np
import pandas as pd
import xarray as xr
from dataclasses import asdict
from pycontrails.models.gpat.pp_gpat import GPATPostProcessor
import os
import json
import pyvista as pv
np.set_printoptions(threshold=np.inf, linewidth=500)

data_path = f"{os.getcwd()}/data/"

criteria = {
    "job_id": ["test", "2_flights_origin"]
}

pp_gpat = GPATPostProcessor(data_path=data_path, criteria=criteria)

boxm_ds = pp_gpat.boxm_ds_dict[pp_gpat.job_ids[0]]
fl_ds = pp_gpat.fl_ds_dict[pp_gpat.job_ids[0]]
pl_ds = pp_gpat.pl_ds_dict[pp_gpat.job_ids[0]]

boxm_out = pp_gpat.boxm_out_dict[pp_gpat.job_ids[0]]
pl_out = pp_gpat.pl_out_dict[pp_gpat.job_ids[0]]
patch_table = pp_gpat.patch_table_dict[pp_gpat.job_ids[0]]

for t in range(60, 180, 1):
    print(pl_out["pl_mass"].sel(species_pl="NO", seg_id=5).isel(time=t).values)

print(patch_table["Y_del_f"].sel(species_out="NO", row=slice(0,30)).values)
print(pl_out["pl_mass"].sel(species_pl="NO").isel(time=60+60).values)

pl_ds = pp_gpat.pl_ds_dict[pp_gpat.job_ids[0]]

# print(pl_ds["emi_pl_mass"].sel(species_emi="NO").values)
# print(pl_ds["age_s"].sel(seg_id=31).values)
# print(pl_ds["heading"].sel(time='2022-01-20T13:30:00Z').values)
# boxm_out["Y_bg_c"] = boxm_out["Y_bg_c"] * 1e9 / 5.9e+18  # Convert from kg/kg to ppb

# pp_gpat.plotting.plot_time_series(pp_gpat.job_ids[0], species="NO", level=217, lat=0.5, lon=0.5, data_var="Y_del_c")

# pp_gpat.plotting.plot_boxm_background_slider(pp_gpat.job_ids[1], species="NO", level=boxm_ds["level_c"].values[2], time_indices=None)

# pp_gpat.plotting.plot_boxm_patch_slider(
#     job_id=pp_gpat.job_ids[1],
#     species="NO",
#     mode="total_with_patch",
#     vertical_mode="topk",
#     top_k=3,
# )

job_id = pp_gpat.job_ids[0]
patch_table = pp_gpat.patch_table_dict[job_id]

t_idx = 1852  # or whatever frame the slider is on

mask_t = np.asarray(patch_table["time_idx"].values, dtype=int) == t_idx
print("rows at t_idx:", mask_t.sum())

if mask_t.sum() > 0:
    ds_t = patch_table.isel(row=np.flatnonzero(mask_t))
    y = np.asarray(ds_t["Y_del_f"].sel(species_out="NO").values, dtype=float)
    print("NO min/max at t_idx:", np.nanmin(y), np.nanmax(y))
    print("nonzero count:", np.count_nonzero(np.isfinite(y) & (y != 0.0)))
    print("unique level_f sample:", np.unique(np.asarray(ds_t["level_f"].values, dtype=float))[:20])
    print("row_cell_c min/max:", np.nanmin(ds_t["row_cell_c"].values), np.nanmax(ds_t["row_cell_c"].values))

# pp_gpat.plotting.plot_boxm_patch_slider(
#     job_id=pp_gpat.job_ids[0],
#     species="NO",
#     mode="patch_delta",
#     vertical_mode="single",
#     level=230.0,
#     dynamic_colorbar=True,
# )

# pp_gpat.plotting.plot_patch_heatmap_2d_with_slider(
#     time_indices=patch_time_idx,
#     level=11000,  # Use middle altitude
#     job_id=pp_gpat.job_ids[0],
#     species="NO", 
#     data_vars=["Y_bg_c"],
#     overlay_trajectories=True,
#     color_scale=color_scale
# )

# pp_gpat.plotting.plot_heatmap_slider(job_id=pp_gpat.job_ids[0], species="NO", data_var="Y_bg_c", level=11000)

pp_gpat.plotting.animate_plumes_3d_plotly(
    job_id=pp_gpat.job_ids[1],
    patch_species="NO",
    overlay_ellipses=False,
    overlay_slices=False,
    overlay_patch=True,
    time_idx_start=181,
    time_idx_end=181+3*180,
    output_path="plume.html"
)

# print(f"boxm_ds: {boxm_ds['Y_bg_c'].sel(species_boxm='NO').isel(cell=0).values}")
# #print(f"boxm_out: {boxm_out['Y_bg_c'].sel(species_out='NO').isel(latitude_c=1, longitude_c=1, level_c=1).values}")
# pl_time_idx = np.asarray(pp_gpat.pl_ds_dict[pp_gpat.job_ids[0]]["time_idx"].values, dtype=int)
# boxm_time_idx = np.asarray(pp_gpat.boxm_ds_dict[pp_gpat.job_ids[0]]["time_idx"].values, dtype=int)
# # Check available time indices
# patch_time_idx = np.asarray(patch_table["time_idx"].values, dtype=int)
# print(f"Available time_idx in patch_table: {np.min(patch_time_idx)} to {np.max(patch_time_idx)}")
# print(f"Available time_idx in pl_ds: {np.min(pl_time_idx)} to {np.max(pl_time_idx)}")
# print(f"Available time_idx in boxm_ds: {np.min(boxm_time_idx)} to {np.max(boxm_time_idx)}")
