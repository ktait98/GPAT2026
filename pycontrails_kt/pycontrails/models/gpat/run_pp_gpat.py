import numpy as np
import pandas as pd
import xarray as xr
from dataclasses import asdict
from pycontrails.models.gpat.pp_gpat import GPATPostProcessor
import os
import pyvista as pv

data_path = f"{os.getcwd()}/data/"

criteria = {
    "job_id": "GPAT_Feb_2026_test_2_ac",
}

pp_gpat = GPATPostProcessor(data_path=data_path, criteria=criteria)



print(pp_gpat.job_ids)

pp_gpat.plotting.plot_plumes_3d_pv(job_id=pp_gpat.job_ids[0], time_idx=280)              