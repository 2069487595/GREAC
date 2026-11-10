"""GLOBathy (Khazaei et al. 2022) vs Route 1 on the 12 reservoirs, same grid, same extent, same metrics.

GLOBathy is rebuilt exactly as its released script Generate_Bathymetry_Rasters.py does (figshare c.5243309,
02_Data_Raw/baseline_products_20260929/GLOBathy/scripts): the HydroLAKES polygon is rasterized on the global
1-arc-second WGS84 lattice (origin -180/-90, envelope snapped as in getgrid, ALL_TOUCHED=FALSE), the distance
of every lake pixel to the nearest non-lake pixel is taken in pixel units (GDAL ComputeProximity VALUES=0,
DISTUNITS=PIXEL == Euclidean distance transform), and depth = distance * Dmax_use_m / max(distance).
Depth is then sampled (nearest) onto our 30 m UTM grid. GLOBathy has no elevation: its water surface is set
to the median DEM on the ring just outside the polygon (the H_W0 definition of the dev diagnostics).
Surfaces compared (all keep FABDEM outside their own unknown region):
  GLOBathy : H_surf - depth inside the HydroLAKES polygon
  R1-C, T-flat : Tao inside the FABDEM plateau F (route1_seed_level_v1 / route1_tao_flat_v1 / route1_test_v1)
  T-SWBD   : frozen M0 inside SWBD W0
Metrics, per reservoir: volume below H_surf (GLOBathy's own domain) vs the official curve; full-pool volume,
A-E and V-E NRMSE up to the normal pool inside the SWO>=1% extent; survey/sonar cell RMSE inside W0.
Caveat recorded in the output: for the 6 TWDB dev reservoirs GLOBathy uses observed Dmax (Dmax_use_m differs
from its model estimate), for the 6 USBR test reservoirs it uses its model estimate.
Usage: .venv/Scripts/python.exe 04_Code/scripts/globathy_compare_v1.py
"""
from pathlib import Path
import sys, json, warnings

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import shapely
from pyogrio.raw import read
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage as ndi
import aev_registry_v1 as reg
import aev_observation_share_v1 as aev
import usbr_test_registry_v1 as tr
import route1_ablation_v1 as ab

BY = reg.ROOT / '05_Results/by_date/2026-09-29'
OUT = BY / 'baseline_products_v1'
GLOB = reg.ROOT / '02_Data_Raw/baseline_products_20260929/GLOBathy/GLOBathy_basic_parameters'
CS = 1 / 3600


def dmax_table(ids):
    df = pd.concat([pd.read_csv(p) for p in sorted(GLOB.glob('GLOBathy_basic_parameters(*K*LAKES).csv'))])
    df = df[df.Hylak_id.isin(ids)].set_index('Hylak_id')
    df['uses_observed_Dmax'] = (df.Dmax_use_m - df.Dmax_est_PAVEW_m).abs() > 1e-6
    return df


def globathy_native(poly, dmax):
    xmin, ymin, xmax, ymax = poly.bounds
    i0 = int((xmin + 180) / CS // 1); j0 = int((ymax + 90) / CS // 1) + 1
    i1 = int((xmax + 180) / CS // 1) + 1; j1 = int((ymin + 90) / CS // 1)
    cols, rows = i1 - i0, j0 - j1
    x0, y1 = i0 * CS - 180, j0 * CS - 90
    mask = rasterize([(poly, 1)], out_shape=(rows, cols), transform=from_origin(x0, y1, CS, CS), all_touched=False, dtype='uint8')
    dist = ndi.distance_transform_edt(mask)
    return dist * dmax / dist.max(), (x0, y1)


def to_grid(depth, origin, g, epsg):
    t = g['transform']; rr, cc = np.indices(g['dem'].shape)
    lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(t[2] + (cc + .5) * t[0], t[5] + (rr + .5) * t[4])
    r = np.floor((origin[1] - lat) / CS).astype(int); c = np.floor((lon - origin[0]) / CS).astype(int)
    ok = (r >= 0) & (r < depth.shape[0]) & (c >= 0) & (c < depth.shape[1])
    out = np.full(g['dem'].shape, np.nan)
    out[ok] = depth[r[ok], c[ok]]
    out[out <= 0] = np.nan
    return out


def surfaces(key, g, F):
    """Model surfaces (DEM outside each model's own region) for R1-C, T-flat, T-SWBD."""
    if key.startswith('T'):
        p = {'R1-C': BY / f'route1_test_v1/{key}/{key}_C_tao_F.tif', 'T-flat': BY / f'route1_test_v1/{key}/{key}_route1_tao_F.tif',
             'T-SWBD': BY / f'route1_test_v1/{key}/{key}_M0.tif'}
    else:
        m0 = reg.RESERVOIRS[key]['m0'] or aev.OUT / f'{key}_M0.tif'
        p = {'R1-C': BY / f'route1_seed_level_v1/{key}_C_tao_F.tif', 'T-flat': BY / f'route1_tao_flat_v1/{key}_route1_tao_F.tif',
             'T-SWBD': m0}
    out = {}
    for name, path in p.items():
        region = g['W0'] if name == 'T-SWBD' else F
        pred = reg.load_m0(path, g)
        z = g['dem'].copy(); z[region] = pred[region]
        out[name] = (z, region)
    return out


def metrics(z, region, g, cur, H_pool, H_surf, cells):
    lo = np.floor(min(np.nanmin(z[region]), cur.H.min()) / aev.STEP) * aev.STEP
    tab = aev.curve_table(z, g['W0'] | region, {'swo1': g['swo1']}, np.arange(lo, H_pool + 2 + aev.STEP, aev.STEP))
    at = lambda H, c: float(np.interp(H, tab.H, tab[c]))
    off = lambda H, c: float(np.interp(H, cur.H, cur[c]))
    sel = (cur.H <= H_pool + 1e-6) & (cur.H >= tab.H.min())
    ae = aev.fit_stats(np.interp(cur.H[sel], tab.H, tab.A_swo1), cur.area_km2[sel].values)
    ve = aev.fit_stats(np.interp(cur.H[sel], tab.H, tab.V_swo1), cur.volume_km3[sel].values)
    rr, cc, zr = cells; inW = g['W0'][rr, cc]
    s = ab.r1.err_stats(z[rr, cc][inW], zr[inW])
    return dict(V_below_Hsurf_PE_pct=(at(H_surf, 'V_swo1') / off(H_surf, 'volume_km3') - 1) * 100,
                Vmax_PE_pct=(at(H_pool, 'V_swo1') / off(H_pool, 'volume_km3') - 1) * 100,
                AE_NRMSE_pct=ae.get('NRMSE_pct'), VE_NRMSE_pct=ve.get('NRMSE_pct'), AE_r=ae.get('r'),
                cells_W0_RMSE=s.get('RMSE'), cells_W0_bias=s.get('bias')), tab


def main():
    warnings.filterwarnings('ignore')
    for k, v in tr.TEST.items():
        reg.RESERVOIRS.setdefault(k, dict(name=v['name'], crs=v['epsg']))
    ids = json.loads((OUT / 'hydrolakes_ids.json').read_text(encoding='utf-8'))
    dm = dmax_table([v['Hylak_id'] for v in ids.values()])
    rows, info = [], {}
    for key, rec in ids.items():
        hid = rec['Hylak_id']
        g, F, H_flat, _, cur, H_pool, cells = ab.setup(key)
        epsg = tr.TEST[key]['epsg'] if key.startswith('T') else reg.RESERVOIRS[key]['crs']
        meta, _, geoms, _ = read(reg.ROOT / '02_Data_Raw/input_mead/HydroLAKES_polys_v10_wgs84.gpkg', where=f'Hylak_id = {hid}', columns=['Hylak_id'])
        poly = shapely.make_valid(shapely.from_wkb(geoms[0]))
        dmax = float(dm.loc[hid, 'Dmax_use_m'])
        depth, origin = globathy_native(poly, dmax)
        d = to_grid(depth, origin, g, epsg)
        P = np.isfinite(d)
        H_surf = float(np.median(g['dem'][ndi.binary_dilation(P, np.ones((3, 3))) & ~P]))
        zG = g['dem'].copy(); zG[P] = H_surf - d[P]
        surf = surfaces(key, g, F); surf = {'GLOBathy': (zG, P), **surf}
        V_own_official = float(np.interp(H_surf, cur.H, cur.volume_km3))
        info[key] = dict(name=rec['name'], Hylak_id=hid, Dmax_use_m=dmax, uses_observed_Dmax=bool(dm.loc[hid, 'uses_observed_Dmax']),
                         Dmax_est_PAVEW_m=float(dm.loc[hid, 'Dmax_est_PAVEW_m']), H_surf_m=H_surf, polygon_km2_on_grid=float(P.sum() * 9e-4),
                         V_GLOBathy_own_km3=float(np.nansum(d) * 900 / 1e9), V_official_below_Hsurf_km3=V_own_official,
                         official_max_depth_below_Hsurf_m=float(H_surf - cur.H.min()))
        for name, (z, region) in surf.items():
            m, tab = metrics(z, region, g, cur, H_pool, H_surf, cells)
            tab.to_csv(OUT / f'{key}_{name}_curves.csv', index=False)
            rows.append(dict(id=key, set='test' if key.startswith('T') else 'dev', name=rec['name'], model=name, **m))
            print(key, name, {q: round(w, 2) for q, w in m.items() if isinstance(w, float)}, flush=True)
        pd.DataFrame(rows).to_csv(OUT / 'globathy_compare.csv', index=False, encoding='utf-8-sig')
    (OUT / 'globathy_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    s = pd.DataFrame(rows)
    agg = s.assign(abs_vmax=s.Vmax_PE_pct.abs(), abs_own=s.V_below_Hsurf_PE_pct.abs()).groupby(['set', 'model']).agg(
        median_abs_Vmax_PE=('abs_vmax', 'median'), mean_abs_Vmax_PE=('abs_vmax', 'mean'), median_abs_own_PE=('abs_own', 'median'),
        median_AE_NRMSE=('AE_NRMSE_pct', 'median'), median_VE_NRMSE=('VE_NRMSE_pct', 'median'), median_cells_RMSE=('cells_W0_RMSE', 'median'))
    agg.to_csv(OUT / 'globathy_compare_aggregate.csv', encoding='utf-8-sig'); print(agg.round(2), flush=True)


if __name__ == '__main__':
    main()
