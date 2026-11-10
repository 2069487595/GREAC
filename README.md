# GREAC code

Code that produced **GREAC v1.0**, a global dataset of elevation–area–capacity curves and 30 m bathymetry for 28,481 reservoirs derived from digital elevation models.

- **Dataset:** https://doi.org/10.5281/zenodo.24097990 (CC BY-NC-SA 4.0)
- **Code archive:** all versions at https://doi.org/10.5281/zenodo.24110168 (v1.0: https://doi.org/10.5281/zenodo.24110169)
- **Changes:** v1.0.1 adds the attribute `flag_empty_extent` to the data package (`greac_package_v1.py`).
- **Paper:** GREAC: global reservoir elevation–area–capacity curves and 30 m bathymetry for 28,481 reservoirs from digital elevation models (submitted to *Earth System Science Data*).

The scripts are the versions that were run, in the layout of the original project. Paths are built from the repository root (`ROOT`), so they work unchanged once the input data are placed as described below. Only absolute paths of the original machine were replaced ([PATH_CHANGES.md](PATH_CHANGES.md)); [FILES.tsv](FILES.tsv) gives the SHA-256 of each source file.

## Workflow

Each script starts with a docstring that states what it does, its subcommands, inputs and outputs.

| Step | Scripts (`04_Code/scripts/`) |
|---|---|
| 1. FABDEM tiles and run list | `route1_global_tiles_v1.py` |
| 2. Regions $F$, $W_0$, $R$, water levels $H_\mathrm{flat}$, $H_\mathrm{seed}$, $H_\mathrm{top}$; TC-cap beds and curves (core line, 5,168 reservoirs) | `route1_global_run_v1.py`, `route1_global_summary_v1.py` |
| 3. Dam-height reading and TC-dam | `r2_global_run_v1.py`, `r2_global_summary_v1.py`, `r2_global_batch.sh` |
| 4. Further reservoirs of GDW: extension, north of 60° N, re-runs | `gdw_coverage_v1.py`, `gdw_ext_v1.py`, `highlat_v1.py`, `remnant_fix_v1.py`, `gdw_ext_batch.sh` |
| 5. Floors recorded by SRTM | `srtm_bed_v1.py`, `srtm_bed_v3.py`, `srtm_bed_pre2000_flag_v1.py`, `srtm_bed_batch.sh`, `srtm_bed_v3_batch.sh` |
| 6. Small-reservoir tier and HydroLAKES v1.1 polygons | `dataset_coverage_v3.py`, `small_tier_v1.py`, `hl11_small_v1.py`, `hl11_small_b_v1.py`, `small_tier_batch.sh`, `hl11_small_batch.sh`, `hl11_b_batch.sh` |
| 7. Seed-gap indicator | `seed_gap_flag_v1.py` |
| 8. Inventory and data package | `greac_inventory_v1.py`, `greac_package_v1.py` |

The other modules in `04_Code/scripts/` are imported by these scripts, for shared functions such as the region and curve routines (`aev_registry_v1.py`, `aev_observation_share_v1.py`), the topographic-continuity steps (`route1_tao_flat_v1.py`, `route1_seed_level_v1.py`, `route1_hybrid_v1.py`) and the dam-height reading (`r2_dev_v1.py`).

**TC-Tao kernel.** `04_Code/versions/` holds the run version of the topographic-continuity method of Tao et al. (2026):
- `m0_m1_m2_runner_v4/geometry_adapter.py`;
- the frozen kernel in `Tao-original/FROZEN-20260921/project/04_Code/scripts/`, adapted from the code released by Tao et al. (2026, CC BY 4.0) to run on any grid and mask.

The released code itself is not included; see [TAO_CODE.md](TAO_CODE.md).

## Input data

All paths are relative to the repository root.

| Path | Content | Source |
|---|---|---|
| `02_Data_Raw/FABDEM_V1-2_all/`, `02_Data_Raw/FABDEM_tiles_extracted/` | FABDEM V1-2 tiles | https://doi.org/10.5523/bris.s5hqmjcdj8yo2ibzi9b4ew3sn |
| `02_Data_Raw/NASADEM_Extracted/HGT`, `.../SWB` | NASADEM elevation and SRTM water body data | https://doi.org/10.5067/MEaSUREs/NASADEM/NASADEM_HGT.001, https://doi.org/10.5067/MEaSUREs/SRTM/SRTMSWBD.003 |
| `02_Data_Raw/SWO_GSW_v1_5_2024/occurrence_v1_5_2024`, `02_Data_Raw/GSW_yearly2000_20261027/` | JRC Global Surface Water v1.5: occurrence, yearly classification of 2000 | https://global-surface-water.appspot.com/download |
| `02_Data_Raw/input_mead/HydroLAKES_polys_v10_wgs84.gpkg` | HydroLAKES v1.0 polygons | https://www.hydrosheds.org/products/hydrolakes |
| `02_Data_Raw/GDW_v1_0_20261027/extracted/GDW_v1_0_shp/` | GDW v1.0 reservoirs and barriers | https://doi.org/10.6084/m9.figshare.25988293 |
| `03_Data_Processed/` | Per-reservoir input grids written by the run scripts | generated |
| `05_Results/by_date/<date>/<name>/` | Results, in the dated folders of the runs that produced GREAC v1.0 | generated |

The exact file names each script reads are given at its top (`ROOT / '...'`).

## Running

Run the steps in the order of the table. For example:

```
python -I 04_Code/scripts/route1_global_run_v1.py --help
python -I 04_Code/scripts/greac_package_v1.py attributes
```

The batch scripts (`*.sh`) split the runs into parts that run in parallel. Run scripts that reuse earlier outputs check that their inputs match the locked runs (SHA-256 in the per-reservoir `*_run.json` records).

## Environment

- Python 3.14.6; packages in [requirements.txt](requirements.txt).
- **TC-Tao kernel:** compiled with Numba. The original runs loaded Numba 0.67.0 with NumPy 2.5.3 from an isolated folder, `03_Data_Processed/tibet_three_lakes_v1/compute_runtime`, which the kernel puts first on `sys.path` when it exists. With the packages of `requirements.txt` installed, that folder is not needed.

## Licence and citation

- **Code:** MIT licence ([LICENSE](LICENSE)).
- **Kernel files under `04_Code/versions/Tao-original/`:** adapted from the code of Tao et al. (2026), released under CC BY 4.0. Cite:
  - Tao, F.: Integrating topographic continuity and lake recession dynamics for improved bathymetry mapping from DEMs, Zenodo [code], https://doi.org/10.5281/zenodo.19881484, 2026.
  - Tao, F., Wang, Y., Jing, Y., She, X., Lu, S., and Li, Y.: Integrating topographic continuity and lake recession dynamics for improved bathymetry mapping from DEMs, Hydrol. Earth Syst. Sci., 30, 2741–2758, https://doi.org/10.5194/hess-30-2741-2026, 2026.
- **When using GREAC or this code,** please cite the dataset: Ning, C., et al.: GREAC v1.0: global reservoir elevation–area–capacity curves and 30 m bathymetry for 28,481 reservoirs from digital elevation models, Zenodo [data set], https://doi.org/10.5281/zenodo.24097990, 2026.
