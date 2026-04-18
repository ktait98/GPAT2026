# GPAT Projection Method Notes

Current projection behavior is implemented in:
- `pycontrails/models/gpat/boxm.f90`

## 1. Runtime position

`PROJECT_PLUMES_TO_GRID` calls:
1. `PL_STATE%ADVANCE_GEOM(...)`
2. `PL_STATE%ADVANCE_MASS(...)`
3. `PL_STATE%BUILD_ACTIVE(BOXM_DS, BOXM_STATE)`
4. `PL_STATE%PROJECT_TO_GRID(...)`
5. `PATCH_STATE%BUILD_ROWS_FROM_W(...)`
6. `PATCH_STATE%ACCUM_DELTAS_FROM_W(...)`

## 2. What `PL_STATE_BUILD_ACTIVE` now does

`PL_STATE_BUILD_ACTIVE` uses one 3D bounding box per included segment (not per waypoint center):
- Segment envelope = min/max over all `SLICE_POLYS_M(SEG_ID, :, :, :)`
- Padding is applied in horizontal and vertical directions
- A coarse cell is active if its coarse-cell box overlaps this segment envelope in x/y/z

Nearest-cell center fallback has been removed.

## 3. Which segments are included

Projection-side segment inclusion uses:
- included if `ACTIVE_SEG_FLAG(SEG_ID) == 1`
- or included if segment plume mass is nonzero (`sum(abs(PL_MASS(seg,:))) > MASS_EPS`)

This keeps passive/empty segments out while retaining physically relevant segments even when flags lag.

## 4. Candidate coarse cells in projection

`PL_STATE_PROJECT_TO_GRID` builds candidate coarse-cell ids once:
- if `USE_ACTIVE_MASK_IN_PROJECTION = .TRUE.`: use only `ACTIVE_FLAG`
- else: use all coarse cells (`NCELL`)

Current default is `FALSE` to avoid active-mask staircase clipping.

## 5. Segment-level culling envelope inside projection

Inside projection, each slice of a segment uses the same precomputed segment envelope:
- `SEG_X_MIN/MAX(SEG_ID)`
- `SEG_Y_MIN/MAX(SEG_ID)`
- `SEG_Z_MIN/MAX(SEG_ID)`

This avoids waypoint-local bounding behavior during culling.

## 6. Bridge logic

Neighbor bridges are determined by flight/waypoint continuity, not `SEG_ID +/- 1`:
- previous bridge: same `FL_ID`, waypoint `WP-1`
- next bridge: same `FL_ID`, waypoint `WP+1`

Weight blend per slice:
- both bridges: `W_PREV=0.25`, `W_CUR=0.50`, `W_NEXT=0.25`
- prev only: `0.50, 0.50, 0.00`
- next only: `0.00, 0.50, 0.50`
- none: `0.00, 1.00, 0.00`

So the current slice footprint always contributes.

## 7. Overlap evaluation and sparse map construction

Two-pass sparse map build remains:
1. pass-1 counts nonzero overlaps (`NNZ`)
2. pass-2 computes `RAW_SUM`, normalizes, and writes rows

Stored rows:
- `MAP_SEG`
- `MAP_SLICE`
- `MAP_CELL_C`
- `MAP_CELL_F`
- `MAP_W`

Weight formula:
- `WEIGHT = W_SLICE(SLICE_ID) * RAW / RAW_SUM`

`RAW` comes from weighted blend of `RAW_PREV`, `RAW_CUR`, `RAW_NEXT`, each evaluated via `RECT_SLICE_RAW_OVERLAP`.

## 8. Current continuity-first toggles

Top-level toggles in `HELPERS` module:
- `USE_FULL_SLICE_FOOTPRINT = .TRUE.`
- `USE_SLICE_BRIDGES = .TRUE.`
- `USE_ACTIVE_MASK_IN_PROJECTION = .FALSE.`
- `USE_BBOX_PREFILTER_IN_PROJECTION = .FALSE.`
- `ENABLE_BACKPROJECTION = .FALSE.`

Notes:
- `USE_ACTIVE_MASK_IN_PROJECTION = .FALSE.` avoids active-mask clipping artifacts.
- `USE_BBOX_PREFILTER_IN_PROJECTION = .FALSE.` avoids axis-aligned prefilter staircase artifacts.
- `ENABLE_BACKPROJECTION = .FALSE.` prevents nonphysical reinjection while fine-grid chemistry update remains placeholder.

## 9. Diagnostics

When `DEBUG_PROJECTION = .TRUE.`, projection prints:
- included segment count
- candidate coarse-cell count
- active slice count
- nonzero slice count
- sparse-map nnz
- active-mask and bbox-prefilter toggle states

## 10. Downstream consumers

The sparse map is used by:
- `PATCH_STATE_BUILD_ROWS_FROM_W`
- `PATCH_STATE_ACCUM_DELTAS_FROM_W`

Backprojection exists in `PL_STATE_BACKPROJECT_FROM_GRID`, but is currently guarded off by default.
