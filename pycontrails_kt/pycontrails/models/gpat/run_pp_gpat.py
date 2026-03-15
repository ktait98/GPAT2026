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

print(boxm_out)

# da_nonzero = patch_table["Y_del_f"].where(patch_table["Y_del_f"] != 0, drop=True)
# da_clean = patch_table["Y_del_f"].where(np.isfinite(patch_table["Y_del_f"]) & (patch_table["Y_del_f"] != 0), drop=True)
# # print(da_nonzero.sel(species_out="NO").values)
# print(da_clean.sel(species_out="NO").values)

# pp_gpat.plotting.plot_plumes_3d_pv(job_id=pp_gpat.job_ids[0], 
#                                     time_idx=184+3*15, overlay_patch=True)   

pl_time_idx = np.asarray(pp_gpat.pl_ds_dict[pp_gpat.job_ids[0]]["time_idx"].values, dtype=int)

pp_gpat.plotting.animate_plumes_3d_plotly(
    job_id=pp_gpat.job_ids[0],
    time_idx_start=int(pl_time_idx.min()),
    time_idx_end=int(pl_time_idx.max()),
    frame_stride=3,           # every 3rd step — adjust for speed vs smoothness
    output_path="plumes.html",
    overlay_patch=True,
    patch_species="NO",
)

# pp_gpat.plotting.plot_patch_heatmap_2d(
#     range(184+3*15, 184+3*20, 3),
#     job_id=None,
#     species="NO",
#     overlay_plume_centers=True,
#     overlay_trajectories=True,
# )

# pp_gpat.plotting.plot_patch_heatmap_3d(
#     time_idx=184,
#     job_id=pp_gpat.job_ids[0],
#     species="NO",
#     opacity=0.4,
#     show_edges=True,
#     off_screen=True,
#     screenshot_path="patch_3d_NO_time184.png",
#     )
