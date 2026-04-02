# BOXM: Grid Projection, Chemistry, and Backprojection Engine

BOXM is the FORTRAN core used by GPAT to step plume chemistry forward in time while coupling a moving Lagrangian plume representation to an Eulerian chemistry grid. In practical terms, it is the executable that:

- reads the GPAT input NetCDF files,
- reconstructs plume geometry at a given timestep,
- converts plume mass into sparse fine-grid concentrations,
- runs chemistry on the background coarse grid and on plume-affected fine-grid patches,
- aggregates plume chemistry back to the coarse grid,
- backprojects the updated plume perturbation back into plume segments for the next timestep, and
- writes grid, plume, and sparse patch outputs back to NetCDF.

This file accompanies the higher-level GPAT overview in `GPAT_README.md` and focuses specifically on the structure and responsibilities of `boxm.f90`.

## High-Level Role in GPAT

Within the overall GPAT workflow, Python handles setup, preprocessing, input generation, and output templating. BOXM then performs the timestep-resolved coupled plume/grid update.

Conceptually, each BOXM timestep follows this sequence:

1. Load meteorology and chemistry state for the current simulation time.
2. Reconstruct active plume segment geometry from `pl_ds`.
3. Convert plume segment mass into fine-grid perturbation concentrations using sparse projection weights.
4. Run coarse-grid background chemistry.
5. Run fine-grid chemistry inside plume-affected patches.
6. Re-average fine-grid perturbations back to the coarse grid.
7. Backproject updated fine-grid perturbations into plume segment masses.
8. Write outputs for the current output time.

This makes BOXM the coupling layer between the Lagrangian plume representation and the Eulerian chemistry representation.

## File Structure of `boxm.f90`

`boxm.f90` is organized as a set of modules followed by a top-level driver program:

1. `HELPERS`
2. `RUN_CHEM_UTILS`
3. `DEFINE_INPUT_TYPES`
4. `DEFINE_STATE_TYPES`
5. `DEFINE_OUTPUT_TYPES`
6. `BOXM_RUN_UTILS`
7. `VALIDATION_UTILS`
8. `PROGRAM BOXM_RUN`

That layout is sensible: the file begins with low-level utility functions, then chemistry routines, then NetCDF-backed input/output types, then in-memory state objects, then orchestration logic, and finally the executable driver.

---

## 1. `HELPERS` module

Purpose: shared low-level numerical and geometry utilities.

This module provides:

- `NC_CHECK` for NetCDF error handling.
- `ERFINV` for inverse error-function calculations used when converting Gaussian plume variance into slice half-widths/half-depths.
- `OVERLAP_1D` for interval overlap checks.
- `DEG_TO_M_DX` and `DEG_TO_M_DY` for degree-to-metre conversion.
- `RECT_SLICE_RAW_OVERLAP` and related polygon clipping/intersection helpers for estimating overlap between plume slice geometry and grid cells.

Why it matters:

- It contains the geometry primitives used by projection and activation logic.
- It isolates NetCDF error handling from the rest of the code.
- It supports both simple and more general intersection calculations for plume slices and cell footprints.

---

## 2. `RUN_CHEM_UTILS` module

Purpose: chemistry kernel and chemistry workspace allocation.

This is the chemistry-heavy part of the file. It contains:

- allocation/setup routines such as `CHEM_ALLOC`,
- kinetic and photolysis support routines such as `CHEMCO`, `CALC_J`, and `PHOTOL`,
- the ODE/right-hand-side chemistry machinery in `DERIV`, and
- `RUN_LEGACY_CHEM`, which is the main chemistry entry point used elsewhere in BOXM.

Interpretation:

- This module encapsulates the photochemical box model.
- Other BOXM modules do not implement chemistry directly; they prepare concentrations and environmental fields, then call `RUN_LEGACY_CHEM`.
- Both coarse-grid background chemistry and fine-grid plume chemistry are ultimately routed through the same chemistry core.

---

## 3. `DEFINE_INPUT_TYPES` module

Purpose: define NetCDF-backed input dataset classes and their read/init/close methods.

This module defines three main input types:

### `FL_DS_TYPE`
Represents the flight/segment input dataset.

Key contents:
- segment identity and indexing,
- waypoint and flight IDs,
- static segment geometry and timing,
- aircraft-performance-related fields such as fuel burn, thrust, and true airspeed.

Main methods:
- `INIT`
- `READ_STATIC`
- `SUMMARY`
- `CLOSE`

### `PL_DS_TYPE`
Represents the plume input dataset.

Key contents:
- plume geometry over time,
- active segment flags,
- age, width, depth, heading,
- covariance terms (`sigma_yy`, `sigma_yz`, `sigma_zz`),
- emitted plume mass by emitted species,
- plume slicing attributes such as `NSLICES`, `FMAX`, and `NPOINTS`.

Main methods:
- `INIT`
- `READ_STATIC`
- `SUMMARY`
- `CLOSE`

### `BOXM_DS_TYPE`
Represents the coarse-grid box-model input dataset.

Key contents:
- coarse-cell coordinates and metrics,
- time coordinates,
- species numbering and molecular masses,
- meteorological parameters (`TEMP`, `H2O`, `M`, `O2`, `N2`, `SZA`),
- coarse-grid background chemistry (`Y_BG_C`),
- model attributes such as timestep sizes and coarse/fine resolutions,
- switches like `RUN_CHEM` and `N_AC`.

Main methods:
- `INIT`
- `READ_STATIC`
- `SUMMARY`
- `CLOSE`

Why this module matters:

- It is the BOXM input boundary.
- It provides a clear translation from NetCDF files to FORTRAN arrays.
- It keeps file I/O separate from timestep logic.

---

## 4. `DEFINE_STATE_TYPES` module

Purpose: define the in-memory evolving model state used during the simulation.

This is the most important structural module for understanding BOXM.

It defines three coupled state types:

### `PL_STATE_TYPE`
Represents the current Lagrangian plume-segment state.

Key responsibilities:
- hold the current plume geometry for all segments at the active time,
- hold carried plume mass by segment and species,
- build slice geometry for projection,
- store the sparse mapping from plume slices to fine-grid cells.

Key fields:
- current segment position and dimensions,
- `PL_MASS(NSEG, NSPL)` for species mass carried in plume segments,
- `Y_HALF`, `Z_HALF`, `W_SLICE` for slice geometry/weights,
- `ELLIPSES_M` and `SLICE_POLYS_M` for geometric diagnostics and mapping,
- sparse map arrays: `MAP_SEG`, `MAP_SLICE`, `MAP_CELL_C`, `MAP_CELL_F`, `MAP_W`.

Key methods:
- `INIT_FROM_PL_DS`
- `INIT_FROM_PL_OUT`
- `BUILD_ELLIPSES_M`
- `BUILD_SLICE_POLYS_M`
- `ADVANCE_GEOM`
- `BUILD_ACTIVE`
- `EMI_TO_PLUMES`
- `PROJECT_TO_GRID`
- `BACKPROJECT_FROM_GRID`

### `BOXM_STATE_TYPE`
Represents the current coarse-grid Eulerian state.

Key responsibilities:
- hold the active timestep meteorology,
- hold coarse-grid background concentrations,
- hold coarse-grid plume perturbation concentrations,
- flag active coarse cells impacted by plume geometry.

Key fields:
- meteorological arrays (`TEMP`, `H2O`, `M`, `O2`, `N2`, `SZA`),
- `Y_BG_C` for background chemistry,
- `Y_DEL_C` for coarse-grid plume perturbations,
- `ACTIVE_FLAG` for active coarse cells,
- cell geometry metrics `DX_C_M`, `DY_C_M`.

Key methods:
- `INIT_FROM_BOXM_DS`
- `ADVANCE_MET`
- `RUN_COARSE_BG_CHEM`
- `RUN_COARSE_DELTA_CHEM`

### `PATCH_STATE_TYPE`
Represents the sparse fine-grid patch chemistry state.

Key responsibilities:
- store only those fine-grid cells that are actually intersected by plume projection,
- avoid allocating and solving chemistry on a full global fine grid,
- provide the bridge between plume projection and coarse-grid averaging.

Key fields:
- row indexing arrays linking sparse rows to `(coarse cell, fine subcell)` pairs,
- `Y_DEL_F(NROWS, NSBOXM)` for fine-grid perturbation concentrations.

Key methods:
- `INIT_FROM_BOXM_DS`
- `BUILD_ROWS_FROM_W`
- `ACCUM_DELTAS_FROM_W`
- `RUN_FINE_DELTA_CHEM`

### Why this module is central

This module contains the actual plume/grid coupling logic. In particular:

- `PL_STATE_ADVANCE_GEOM` rebuilds plume geometry at the current timestep.
- `PL_STATE_BUILD_ACTIVE` determines which coarse cells need attention.
- `PL_STATE_EMI_TO_PLUMES` injects new emitted mass into the plume state and removes expired plume mass.
- `PL_STATE_PROJECT_TO_GRID` builds the sparse projection map from plume slices to fine-grid cells.
- `PATCH_STATE_ACCUM_DELTAS_FROM_W` converts plume mass to fine-grid concentration perturbations.
- `PATCH_STATE_RUN_FINE_DELTA_CHEM` performs patch-level chemistry.
- `BOXM_STATE_RUN_COARSE_DELTA_CHEM` re-averages fine-grid plume perturbations back to the coarse grid and advances coarse plume perturbation chemistry.
- `PL_STATE_BACKPROJECT_FROM_GRID` converts fine-grid chemical perturbations back into plume-segment masses for the next timestep.

That is the heart of BOXM.

---

## 5. `DEFINE_OUTPUT_TYPES` module

Purpose: define NetCDF-backed output dataset classes and their write/init/close methods.

This module manages the three BOXM output products:

### `PL_OUT_TYPE`
Stores plume-resolved output diagnostics, including:
- slice half-widths/half-depths,
- slice geometry,
- plume mass by segment/species,
- plume polygon diagnostics.

### `BOXM_OUT_TYPE`
Stores coarse-grid chemistry output, including:
- `Y_BG_C`,
- `Y_DEL_C`,
- `ACTIVE_FLAG`.

### `PATCH_TABLE_TYPE`
Stores sparse fine-grid patch output, including:
- row-to-cell mapping,
- row metadata,
- `Y_DEL_F` for each sparse fine-grid row.

Why this module matters:

- It preserves both Eulerian and Lagrangian diagnostics.
- It makes debugging possible because you can inspect plume, coarse-grid, and fine-grid sparse states separately.
- The patch table is especially useful for validating projection/backprojection and sparse chemistry behaviour.

---

## 6. `BOXM_RUN_UTILS` module

Purpose: orchestrate the BOXM timestep workflow.

This module is the operational controller of the model.

### `BOXM_RUN_INIT`
Initializes datasets, states, outputs, and chemistry allocation.

Responsibilities:
- open and read input datasets,
- initialize plume, grid, and patch states,
- initialize output files,
- align plume species with the output species list,
- allocate chemistry workspace if chemistry is enabled.

### `PROJECT_PLUMES_TO_GRID`
Coordinates the plume-to-grid step.

Responsibilities:
- advance plume geometry,
- determine active cells,
- inject emissions into plume mass,
- project plume mass to sparse fine-grid rows,
- build sparse patch rows,
- accumulate fine-grid perturbations from projection weights.

### `ADVANCE_MET`
Loads the meteorological state for the current timestep from `BOXM_DS` into `BOXM_STATE`.

### `RUN_CHEM`
Coordinates chemistry execution.

Responsibilities:
- run coarse-grid background chemistry,
- run fine-grid plume patch chemistry,
- run coarse-grid plume-perturbation chemistry after re-averaging.

### `BACKPROJECT_GRID_TO_PLUMES`
Calls the plume backprojection routine so that chemically updated plume perturbations are returned to segment mass form.

### `WRITE_OUTPUTS`
Writes plume, coarse-grid, and patch outputs at output times.

### `RESET_STATES`
Clears timestep-local arrays while preserving the carried plume mass that must survive between timesteps.

This routine is important conceptually: not everything is reset. The plume mass is intentionally retained because it is the memory of the plume between timesteps.

### `CLOSE_DATASETS`
Closes all NetCDF inputs and outputs cleanly.

---

## 7. `VALIDATION_UTILS` module

Purpose: built-in validation and debugging helpers.

This module includes:

- `CHECK_SEGMENT_MASS_RECOVERY`
- `PRINT_TOTAL_PLUME_MASS`
- `PRINT_TOTAL_PATCH_MASS`

These routines are especially useful when checking:

- whether projection and backprojection are conservative,
- whether sparse row mappings are complete,
- whether any mass is being lost due to missing row matches or clipping,
- whether patch mass and plume mass remain consistent.

For development and debugging, this is one of the most practically valuable modules in the file.

---

## 8. `PROGRAM BOXM_RUN`

Purpose: executable entry point.

This is the driver that stitches everything together.

### Initialization
It reads:
- `JOB_ID` from the command line,
- `DATA_PATH` from the command line,
- then calls `BOXM_RUN_INIT`.

### Main loop
It loops over `TIME_IDX = 1 : NTBOXM` and performs, in order:

1. progress reporting,
2. `RESET_STATES`,
3. `ADVANCE_MET`,
4. plume-to-grid projection at output-aligned times,
5. chemistry update if enabled,
6. grid-to-plume backprojection at output-aligned times,
7. output writing at output-aligned times.

### Finalization
At the end it calls `CLOSE_DATASETS`.

This makes `PROGRAM BOXM_RUN` thin by design, which is good: the high-level control flow is visible immediately, while the implementation details remain in the modules.

## Core Algorithmic Flow

A useful way to understand BOXM is by the three internal representations it keeps in sync.

### 1. Plume representation: `PL_STATE`
This is Lagrangian and segment-based.

- Species are carried as mass in plume segments.
- Geometry evolves according to precomputed plume fields from `pl_ds`.
- Each segment is decomposed into slices for mapping.

### 2. Patch representation: `PATCH_STATE`
This is sparse fine-grid and only exists where plume mass maps.

- Each sparse row corresponds to one unique `(coarse cell, fine subcell)` pair.
- Species are stored as concentration perturbations `Y_DEL_F`.
- Chemistry is run on these fine cells using coarse-cell meteorology replicated onto the fine subdivision.

### 3. Coarse-grid representation: `BOXM_STATE`
This is Eulerian and cell-based.

- `Y_BG_C` stores background chemistry.
- `Y_DEL_C` stores plume perturbation chemistry on the coarse grid.
- `ACTIVE_FLAG` marks cells potentially impacted by the plume geometry.

The BOXM algorithm is essentially a controlled transfer between these three representations.

## Projection and Backprojection Logic

### Projection
Projection proceeds broadly as:

1. Rebuild segment slice geometry.
2. Determine which coarse cells are active.
3. For each segment slice, compute overlap with fine subcells.
4. Store only nonzero mappings in sparse arrays.
5. Convert plume mass into fine-grid concentration perturbations using the overlap weights and fine-cell volume.

This avoids solving chemistry on an everywhere-refined domain.

### Fine-grid chemistry
For each active coarse cell:

1. Reconstruct a local fine-grid chemistry array.
2. Populate background values from the coarse background state.
3. Populate perturbations from sparse patch rows.
4. Form total concentrations as background plus perturbation.
5. Run chemistry.
6. Recover perturbation as `updated total - background`.

### Coarse-grid aggregation
After fine-grid chemistry, patch concentrations are volume-averaged back into `BOXM_STATE%Y_DEL_C`.

### Backprojection
Finally, the fine-grid perturbation field is mapped back to segment species mass so that the plume carries forward a chemically updated perturbation to the next timestep.

This is what gives BOXM its coupled plume/grid character.

## Main Design Ideas in BOXM

A few design choices define the structure of the code:

### Sparse fine-grid storage
The code does not store a full refined grid. It stores only touched fine cells through sparse row maps. This is essential for tractability.

### Separation of concerns
- NetCDF I/O lives in dataset/output modules.
- Evolving concentrations and geometry live in state modules.
- Chemistry lives in `RUN_CHEM_UTILS`.
- Orchestration lives in `BOXM_RUN_UTILS`.

That makes the code much easier to reason about.

### Coarse background plus plume perturbation split
The model distinguishes between:
- background chemistry on the coarse grid, and
- plume-induced perturbation chemistry on fine patches and aggregated coarse fields.

This is a useful conceptual split for validation and diagnostics.

### Geometry-driven activity filtering
Only coarse cells touched by plume envelopes are activated. That reduces unnecessary fine-grid chemistry work.

## Practical Reading Guide for `boxm.f90`

If you want to understand the code quickly, read it in this order:

1. `PROGRAM BOXM_RUN`
2. `BOXM_RUN_UTILS`
3. `DEFINE_STATE_TYPES`
4. `DEFINE_INPUT_TYPES` and `DEFINE_OUTPUT_TYPES`
5. `RUN_CHEM_UTILS`
6. `HELPERS`
7. `VALIDATION_UTILS`

That order gives you the workflow first, then the data structures, then the detailed implementations.

## What to Focus on When Modifying BOXM

If you plan to extend or debug the code, the most important subroutines are:

- `PL_STATE_ADVANCE_GEOM`
- `PL_STATE_BUILD_ACTIVE`
- `PL_STATE_EMI_TO_PLUMES`
- `PL_STATE_PROJECT_TO_GRID`
- `PATCH_STATE_BUILD_ROWS_FROM_W`
- `PATCH_STATE_ACCUM_DELTAS_FROM_W`
- `PATCH_STATE_RUN_FINE_DELTA_CHEM`
- `BOXM_STATE_RUN_COARSE_DELTA_CHEM`
- `PL_STATE_BACKPROJECT_FROM_GRID`
- `WRITE_OUTPUTS`

Together, these define the full plume-grid-plume transfer loop.

## Suggested Validation Questions for BOXM

When validating BOXM specifically, useful questions are:

- Does total emitted plume mass project conservatively to the patch representation?
- Does backprojection recover segment mass consistently from the sparse fine-grid state?
- Are active coarse cells correctly identified, without clipping or excessive activation?
- Do fine-grid patch rows map uniquely and completely to coarse/fine cell pairs?
- Does `Y_DEL_C` equal the volume-weighted coarse average of `Y_DEL_F`?
- When chemistry is off, does the projection/backprojection loop remain mass-conservative?
- When chemistry is on, are changes attributable to chemistry rather than mapping loss?
- Are plume segments correctly zeroed after exceeding `MAX_AGE_S`?

The built-in validation utilities already support several of these checks.

## Summary

BOXM is the numerical coupling engine inside GPAT.

Its structure is built around three ideas:

- NetCDF-backed input/output dataset classes,
- in-memory plume, coarse-grid, and sparse patch state objects,
- a timestep driver that performs projection, chemistry, aggregation, and backprojection.

In one sentence: `boxm.f90` takes plume-segment mass, maps it to a sparse fine-grid chemistry problem, evolves that chemistry, and maps the result back to plume segments and coarse-grid diagnostics.
