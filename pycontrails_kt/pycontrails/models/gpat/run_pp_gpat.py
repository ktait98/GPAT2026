import numpy as np
import pandas as pd
import xarray as xr
from dataclasses import asdict
from pycontrails.models.gpat.pp_gpat import GPATPostProcessor
import os
import pyvista as pv
np.set_printoptions(threshold=np.inf, linewidth=500)

data_path = f"{os.getcwd()}/data/"

criteria = {
    "job_id": "GPAT_Feb_2026_test_2_ac",
}

pp_gpat = GPATPostProcessor(data_path=data_path, criteria=criteria)


boxm_ds = pp_gpat.boxm_ds_dict[pp_gpat.job_ids[0]]
if pp_gpat.params_dict[pp_gpat.job_ids[0]].get("n_ac", 0) > 0:
    fl_ds = pp_gpat.fl_ds_dict[pp_gpat.job_ids[0]]
    pl_ds = pp_gpat.pl_ds_dict[pp_gpat.job_ids[0]]

boxm_out = pp_gpat.boxm_out_dict[pp_gpat.job_ids[0]]
if pp_gpat.params_dict[pp_gpat.job_ids[0]].get("n_ac", 0) > 0:
    pl_out = pp_gpat.pl_out_dict[pp_gpat.job_ids[0]]
    patch_table = pp_gpat.patch_table_dict[pp_gpat.job_ids[0]]

# print(f"boxm_ds: {boxm_ds['Y_bg_c'].sel(species_boxm='NO').isel(cell=0).values}")
# #print(f"boxm_out: {boxm_out['Y_bg_c'].sel(species_out='NO').isel(latitude_c=1, longitude_c=1, level_c=1).values}")
# pl_time_idx = np.asarray(pp_gpat.pl_ds_dict[pp_gpat.job_ids[0]]["time_idx"].values, dtype=int)
# boxm_time_idx = np.asarray(pp_gpat.boxm_ds_dict[pp_gpat.job_ids[0]]["time_idx"].values, dtype=int)
# # Check available time indices
# patch_time_idx = np.asarray(patch_table["time_idx"].values, dtype=int)
# print(f"Available time_idx in patch_table: {np.min(patch_time_idx)} to {np.max(patch_time_idx)}")
# print(f"Available time_idx in pl_ds: {np.min(pl_time_idx)} to {np.max(pl_time_idx)}")
# print(f"Available time_idx in boxm_ds: {np.min(boxm_time_idx)} to {np.max(boxm_time_idx)}")


# Plot specific variables with trajectories
# pp_gpat.plotting.plot_patch_heatmap_2d(
#     time_idx=184+3*60,
#     level=11000,  # Use middle altitude
#     job_id=pp_gpat.job_ids[0],
#     species="NO", 
#     data_vars=["Y_bg_c"],
#     overlay_trajectories=True
# )

color_scale = (boxm_out["Y_bg_c"].sel(species_out="NO").min(), boxm_out["Y_bg_c"].sel(species_out="NO").max())

pp_gpat.plotting.plot_time_series(pp_gpat.job_ids[0], species="NO", level=217, lat=0.5, lon=0.5, data_var="Y_bg_c")

pp_gpat.plotting.plot_boxm_background_slider(pp_gpat.job_ids[0], species="NO", level=217.56859635, time_indices=None)

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

# pp_gpat.plotting.animate_plumes_3d_plotly(
#     job_id=pp_gpat.job_ids[0],
#     patch_species="NO",
#     overlay_patch=True,
#     time_idx_start=184,
#     time_idx_end=184+3*180,
# )