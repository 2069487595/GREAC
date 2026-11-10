# Code of Tao et al. (2026)

The TC-Tao kernel in `04_Code/versions/Tao-original/FROZEN-20260921/project/04_Code/scripts/` imports functions from the code released by Tao et al. (2026), which is not included here.

**Source:** Tao, F.: Integrating topographic continuity and lake recession dynamics for improved bathymetry mapping from DEMs, Zenodo [code], https://doi.org/10.5281/zenodo.19881484, 2026 (CC BY 4.0).

**Placement:** put the `scripts` folder of the release, which contains the package `lake_bathymetry`, at:

```
04_Code/versions/Tao-original/FROZEN-20260921/project/10_References/Tao_source_code/scripts/
```

Alternatively, set the environment variable `GREAC_TAO_CODE` to that folder. It is read by `tao_compat.py`; `mead_full2d_alpha_masking_v1.py` uses the path above.

**Files used** (SHA-256 of the copy used for GREAC):

| File | SHA-256 |
|---|---|
| `lake_bathymetry/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `lake_bathymetry/analysis.py` | `6d375cd42cd70904dbfaa92b35d141af32bb083f0c3f93162007a584ada04af1` |
| `lake_bathymetry/core.py` | `97a3aac324aa02df779bee7407783e546cc069bd52c4eb9878f95cfd0bc76a68` |
| `lake_bathymetry/geometry_utils.py` | `8b8ace7950f581bfb7be3747ce6b45ee8b01a760a6c3bed606f691e2a22a69ba` |
| `lake_bathymetry/io_utils.py` | `eff2c17f82f2b275d2532b14c40307c5394c0e09f8cadeba8600ef8dbdac688f` |
| `lake_bathymetry/volume.py` | `1d0018cd32681a9d75c8d416de7146806978511ea7f0d65aa583899d02e38013` |

`tao_compat.py` only adapts the calls to NumPy ≥ 2 (float copies of integer contour points); the functions of the release are not changed.
