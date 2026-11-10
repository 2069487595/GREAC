"""R1-C-mDEM, the pre-registered multi-DEM variant (expansion protocol 446b182e1, section 2).

Rule (no tunable parameter):
  1. area(F) <= area(W0): FABDEM already sees at least as much bed as NASADEM -> mDEM = R1-C (no separate run).
  2. otherwise (FABDEM-era water higher than in Feb 2000):
     X      = F & ~W0, land in 2000: terrain = NASADEM + offset
     offset = median(FABDEM - NASADEM) over stable land: outside the SWO>=1% region, within 600 m of it, SWO
              occurrence 0, both DEMs finite (absorbs EGM96 vs EGM2008 and the DEM-to-DEM bias)
     unknown F2 = F & W0, holes < 10 cells filled (as in route1_tao_flat_v1.flattened_region)
     water level: rule C (route1_seed_level_v1.rule_C, unchanged) on the merged terrain with F2; the rest as R1-C.
Details pinned here before the implementation check ran (the protocol text leaves them open):
  a. Floor of the seed level. R1-C uses max(rule C, H_flat), H_flat = plateau of the DEM whose flat water defines
     the unknown region. In case 2 that region is the Feb-2000 water, so the floor is the NASADEM plateau in W0
     (most frequent value, aev_observation_share_v1.water_plateau) + offset. Keeping the FABDEM plateau would force
     the level up to the 2011-15 water surface, which the variant exists to avoid.
  b. NASADEM voids inside X keep their FABDEM value.
  c. If F2 has fewer than 100 cells (admission rule 6 threshold), mDEM falls back to R1-C and says so.
Output: <key>_mDEM_tao_F.tif covers F like the R1-C raster (Tao prediction on F2, NASADEM + offset on X), so the
surface is built the same way for both (DEM outside F, raster inside F); <key>_mDEM_F2_tao_F.tif is the raw Tao
output on F2; <key>_mDEM.json records case, offset, levels and cell counts.
Implementation check (protocol section 2.3): route1_mdem_check_v1.py on the 12 used reservoirs, code behaviour only.
"""
from pathlib import Path
import sys, json

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import rasterio
from scipy import ndimage as ndi
import aev_registry_v1 as reg
import aev_observation_share_v1 as aev
import route1_tao_flat_v1 as r1
import route1_seed_level_v1 as sl

STABLE_RING_M = 600.0
MIN_F2_CELLS = 100


def fill_small_holes(M, n_max=10):
    lab, n = ndi.label(ndi.binary_fill_holes(M) & ~M)
    if n:
        sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        M = M | np.isin(lab, np.arange(1, n + 1)[sizes < n_max])
    return M


def dem_offset(dem, nas, swo1, occ, cell_m=30.0):
    """median(FABDEM - NASADEM) on stable land next to the reservoir, with the cell count and IQR."""
    dist = ndi.distance_transform_edt(~swo1) * cell_m
    stable = ~swo1 & (dist <= STABLE_RING_M) & (occ == 0) & np.isfinite(dem) & np.isfinite(nas)
    d = dem[stable] - nas[stable]
    if d.size < 100:
        raise RuntimeError(f'stable land for the DEM offset has {d.size} cells (< 100)')
    q1, q3 = np.quantile(d, [0.25, 0.75])
    return float(np.median(d)), dict(n_stable=int(d.size), iqr_m=float(q3 - q1))


def setup(g, F, H_flat):
    """Case, merged terrain, unknown region and floor. g needs dem, W0, swo1, occ, nasadem."""
    dem, W0, nas = g['dem'], g['W0'], g['nasadem']
    info = dict(F_cells=int(F.sum()), W0_cells=int(W0.sum()), H_flat_fabdem_m=H_flat)
    if F.sum() <= W0.sum():
        return dict(case=1, dem=dem, F2=F, floor=H_flat, X=np.zeros_like(F), info=dict(info, note='F <= W0: mDEM = R1-C'))
    off, oinfo = dem_offset(dem, nas, g['swo1'], g['occ'])
    X = F & ~W0
    dem_m = dem.copy()
    ok = X & np.isfinite(nas)
    dem_m[ok] = nas[ok] + off
    F2 = fill_small_holes(F & W0)
    H_nas, share = aev.water_plateau(nas, W0)
    info.update(offset_m=off, **oinfo, X_cells=int(X.sum()), X_nasadem_void=int((X & ~np.isfinite(nas)).sum()),
                F2_cells=int(F2.sum()), H_plateau_nasadem_m=H_nas, nasadem_plateau_share_of_W0=share,
                floor_m=H_nas + off)
    if F2.sum() < MIN_F2_CELLS:
        return dict(case=1, dem=dem, F2=F, floor=H_flat, X=np.zeros_like(F), info=dict(info, note='F2 < 100 cells: fallback to R1-C'))
    return dict(case=2, dem=dem_m, F2=F2, floor=H_nas + off, X=X, info=info)


def write(folder, name, arr, g, crs):
    t = g['transform']
    with rasterio.open(folder / name, 'w', driver='GTiff', height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype='float32', nodata=np.nan, crs=f'EPSG:{crs}', compress='deflate',
                       transform=rasterio.Affine(t[0], t[1], t[2], t[3], t[4], t[5])) as d:
        d.write(arr.astype('float32'), 1)


def run(key, g, folder):
    """R1-C-mDEM for one reservoir. key must be in aev_registry_v1.RESERVOIRS (output CRS)."""
    F, H_flat, _ = r1.flattened_region(g['dem'], g['W0'], g['swo1'])
    s = setup(g, F, H_flat)
    rec = dict(key=key, case=s['case'], **s['info'])
    if s['case'] == 1:
        (folder / f'{key}_mDEM.json').write_text(json.dumps(rec, indent=2, default=float), encoding='utf-8')
        return rec
    H_C, infoC = sl.rule_C(s['dem'], s['F2'], g['occ'], g['swo1'])
    H_seed = max(H_C, s['floor'])
    gm = dict(g, dem=s['dem'])
    pred, meta = sl.run_tao(key, gm, s['F2'], H_seed, folder, 'mDEM_F2')
    out = np.full(F.shape, np.nan)
    out[s['F2']] = pred[s['F2']]
    out[s['X']] = s['dem'][s['X']]
    write(folder, f'{key}_mDEM_tao_F.tif', out, g, reg.RESERVOIRS[key]['crs'])
    rec.update(H_rule_C_m=H_C, H_seed_m=H_seed, floor_applied=bool(H_C < s['floor']), rule_C=infoC, alpha=meta['alpha'],
               iterations=meta['iterations'], F2_predicted_finite=int(np.isfinite(pred[s['F2']]).sum()),
               raster_nan_in_F=int((~np.isfinite(out[F])).sum()))
    (folder / f'{key}_mDEM.json').write_text(json.dumps(rec, indent=2, default=float), encoding='utf-8')
    return rec
