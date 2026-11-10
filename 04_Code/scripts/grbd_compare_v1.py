"""GRBD (Li et al. 2020) vs Route 1 on the two covered reservoirs (D06 Sam Rayburn, T02 Cascade), in GRBD's own range.

GRBD rasters (doi:10.18738/T8/TO5HJG, EGM2008, downloaded by the user into 02_Data_Raw/baseline_products_20260929/GRBD)
hold only the remotely sensed band: every pixel below the band carries the band minimum (Sam Rayburn 69 %,
Cascade 73 % of pixels), the projected part described in the paper is not in the release. GRBD therefore has no
full-pool volume, and the comparison is limited to its band [H_min, min(H_max, normal pool)]:
  * A-E: area vs the official curve at official levels inside the band (RMSE, bias, NRMSE = RMSE / official
    area at the band top)
  * band volume V(H_top) - V(H_min) vs the official curve
GRBD is reprojected (nearest) onto our 30 m grid; the other models' curves are the ones written by
globathy_compare_v1 (SWO>=1% extent). Reads no survey points.
"""
from pathlib import Path
import sys, json, warnings

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import aev_registry_v1 as reg
import usbr_test_registry_v1 as tr
import route1_ablation_v1 as ab

OUT = reg.ROOT / '05_Results/by_date/2026-09-29/baseline_products_v1'
GRBD = reg.ROOT / '02_Data_Raw/baseline_products_20260929/GRBD'
CASES = {'D06': 839, 'T02': 8978}
MODELS = ['GLOBathy', 'R1-C', 'T-flat', 'T-SWBD']


def grbd_on_grid(hid, g, epsg):
    t = g['transform']
    dst = np.full(g['dem'].shape, np.nan)
    with rasterio.open(GRBD / f'{hid}.tif') as s:
        src = s.read(1).astype(float); src[src <= 0] = np.nan
        reproject(src, dst, src_transform=s.transform, src_crs=s.crs, dst_transform=rasterio.Affine(t[0], t[1], t[2], t[3], t[4], t[5]),
                  dst_crs=f'EPSG:{epsg}', src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest)
    return dst


def band_scores(A_fun, V_fun, cur, lo, top):
    sel = (cur.H >= lo) & (cur.H <= top)
    Hs = cur.H[sel].values
    e = np.array([A_fun(h) for h in Hs]) - cur.area_km2[sel].values
    A_top = float(np.interp(top, cur.H, cur.area_km2))
    Vb_off = float(np.interp(top, cur.H, cur.volume_km3) - np.interp(lo, cur.H, cur.volume_km3))
    Vb = V_fun(top) - V_fun(lo)
    return dict(n_levels=int(sel.sum()), AE_RMSE_km2=float(np.sqrt(np.mean(e * e))), AE_bias_km2=float(e.mean()),
                AE_NRMSE_pct=float(np.sqrt(np.mean(e * e)) / A_top * 100), band_V_official_km3=Vb_off, band_V_km3=Vb,
                band_V_PE_pct=(Vb / Vb_off - 1) * 100)


def main():
    warnings.filterwarnings('ignore')
    rows, info = [], {}
    for key, hid in CASES.items():
        g, F, H_flat, _, cur, H_pool, _ = ab.setup(key)
        epsg = tr.TEST[key]['epsg'] if key.startswith('T') else reg.RESERVOIRS[key]['crs']
        z = grbd_on_grid(hid, g, epsg)
        lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
        top = min(hi, H_pool)
        A_g = lambda h: float(np.sum(z <= h)) * reg.CELL_AREA / 1e6
        V_g = lambda h: float(np.nansum(np.clip(h - z, 0, None))) * reg.CELL_AREA / 1e9
        info[key] = dict(Hylak_id=hid, GRBD_min_m=lo, GRBD_max_m=hi, band_top_m=top, H_pool_m=H_pool, H_flat_m=H_flat,
                         GRBD_area_km2=float(np.isfinite(z).sum() * 9e-4), floor_share=float(np.mean(np.abs(z[np.isfinite(z)] - lo) < 1e-3)))
        rows.append(dict(id=key, model='GRBD', **band_scores(A_g, V_g, cur, lo, top)))
        for m in MODELS:
            tab = pd.read_csv(OUT / f'{key}_{m}_curves.csv')
            rows.append(dict(id=key, model=m, **band_scores(lambda h: float(np.interp(h, tab.H, tab.A_swo1)),
                                                            lambda h: float(np.interp(h, tab.H, tab.V_swo1)), cur, lo, top)))
        for r in rows[-5:]:
            print(key, r['model'], {q: round(w, 3) for q, w in r.items() if isinstance(w, float)}, flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'grbd_compare.csv', index=False, encoding='utf-8-sig')
    (OUT / 'grbd_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
