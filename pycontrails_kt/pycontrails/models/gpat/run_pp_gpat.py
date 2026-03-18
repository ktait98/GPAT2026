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

fl_ds = pp_gpat.fl_ds_dict[pp_gpat.job_ids[0]]
pl_ds = pp_gpat.pl_out_dict[pp_gpat.job_ids[0]]
boxm_ds = pp_gpat.boxm_out_dict[pp_gpat.job_ids[0]]

pl_out = pp_gpat.pl_out_dict[pp_gpat.job_ids[0]]
boxm_out = pp_gpat.boxm_out_dict[pp_gpat.job_ids[0]]
patch_table = pp_gpat.patch_table_dict[pp_gpat.job_ids[0]]

print(boxm_out["Y_bg_c"].sel(species_out="NO").values)

# da_nonzero = patch_table["Y_del_f"].where(patch_table["Y_del_f"] != 0, drop=True)
# da_clean = patch_table["Y_del_f"].where(np.isfinite(patch_table["Y_del_f"]) & (patch_table["Y_del_f"] != 0), drop=True)
# # print(da_nonzero.sel(species_out="NO").values)
# print(da_clean.sel(species_out="NO").values)


pl_time_idx = np.asarray(pp_gpat.pl_ds_dict[pp_gpat.job_ids[0]]["time_idx"].values, dtype=int)

# Check available time indices
patch_time_idx = np.asarray(patch_table["time_idx"].values, dtype=int)
print(f"Available time_idx in patch_table: {np.min(patch_time_idx)} to {np.max(patch_time_idx)}")
print(f"Available time_idx in pl_ds: {np.min(pl_time_idx)} to {np.max(pl_time_idx)}")

# Plot specific variables with trajectories
# pp_gpat.plotting.plot_patch_heatmap_2d(
#     time_idx=184+3*60,
#     level=11000,  # Use middle altitude
#     job_id=pp_gpat.job_ids[0],
#     species="NO", 
#     data_vars=["Y_bg_c"],
#     overlay_trajectories=True
# )

# pp_gpat.plotting.plot_patch_heatmap_2d_with_slider(
#     time_indices=patch_time_idx,
#     level=11000,  # Use middle altitude
#     job_id=pp_gpat.job_ids[0],
#     species="NO", 
#     data_vars=["Y_bg_c"],
#     overlay_trajectories=True,
# )

# pp_gpat.plotting.animate_plumes_3d_plotly(
#     job_id=pp_gpat.job_ids[0],
#     patch_species="NO",
#     overlay_patch=True,
#     time_idx_start=184,
#     time_idx_end=184+3*180,
# )