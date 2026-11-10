"""Route 1 test: Tao reconstructs only the DEM's own flattened water surface; all DEM terrain is kept.

Single change against Tao-original M0 (frozen v4 runs): the region Tao reconstructs.
  M0      : unknown = SWBD W0 (Feb 2000), start level = median DEM in W0
  Route 1 : unknown = F, the FABDEM flattened plateau, start level = plateau elevation H_flat
Everything else is the frozen Tao code path used for M0 (v3 bank-aligned initialize + recession via the
v4 geometry adapter): buffer rule, slope stretching, 5-degree alpha rule, recession schedule.

F (fixed before any run, see protocol.json):
  cells whose DEM equals the W0 plateau value (|dz| <= 0.005 m), inside SWO>=1% region or W0,
  8-connected to the plateau inside W0; holes smaller than 10 cells filled (flattening noise),
  larger holes kept as known terrain (islands).
Scoring (official TWDB curves and survey cells are read only here, never for any choice):
  * Tao part  : volume below H_flat inside F vs TWDB V(H_flat)
  * full curve: Route-1 surface, extent masks swo1 / bound / dem_fill, Vmax error at conservation pool
  * survey cells inside W0: elevation error of Route 1 vs M0 (same cells), split into F and W0 minus F
v2 (--seed-cap, added after v1 and written to route1_tao_flat_v2/): the v1 run showed the known cells
bordering F sit 2-9 m (median) above the plateau, so Tao descended from heights above the water surface
and left the bed above it in canyon reservoirs. v2 adds one physical constraint and nothing else: seed
heights used by the propagation are capped at the water level, min(DEM, H_flat). Slopes are still fitted
on the uncapped DEM; the output surface outside F stays the uncapped DEM.
Usage: .venv/Scripts/python.exe 04_Code/scripts/route1_tao_flat_v1.py [--keys D01 ...] [--seed-cap]
"""
from pathlib import Path
import sys, json, argparse, datetime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi
import aev_registry_v1 as reg
import aev_observation_share_v1 as aev
import shore_slope_check_v1 as ssc

OUT = reg.ROOT / '05_Results/by_date/2026-09-29/route1_tao_flat_v1'
AEV = aev.OUT
PROTOCOL = dict(
    change='unknown region F = FABDEM flattened plateau instead of SWBD W0; start level = plateau elevation',
    F_rule='|dem - plateau| <= 0.005 m, within (SWO>=1% region | W0), 8-connected to plateau cells in W0; '
           'holes < 10 cells filled, larger holes kept as known terrain',
    plateau='most frequent DEM value in W0 (rounded 0.01 m), as in aev_observation_share_v1',
    tao_code='04_Code/versions/m0_m1_m2_runner_v4/geometry_adapter.py: initialize(dem, F) + tao_run.recession(inp, F, alpha_rule, H_flat)',
    unchanged=['buffer 100-600 m turning-point rule', 'slope stretching branches', 'alpha 5-degree rule',
               'recession schedule', 'frozen kernel'],
    evaluation=['TWDB curve (EGM2008) for volumes', 'survey reference cells for elevations', 'no parameter chosen from either'],
)


def flattened_region(dem, W0, swo1):
    H_flat, share = aev.water_plateau(dem, W0)
    flat = np.isfinite(dem) & (np.abs(dem - H_flat) <= 0.005) & (swo1 | W0)
    F = reg.connected_to(flat, flat & W0)
    lab, n = ndi.label(ndi.binary_fill_holes(F) & ~F)
    if n:
        sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        F |= np.isin(lab, np.arange(1, n + 1)[sizes < 10])
    return F, H_flat, share


def run_tao(key, g, F, H_flat, folder, seed_cap=False):
    sys.path.insert(0, str(reg.ROOT / '04_Code/versions/m0_m1_m2_runner_v4'))
    import geometry_adapter as ga
    dem = g['dem']
    init = ga.initialize(dem, F)
    inp = dem.copy(); inp[F] = np.nan
    original = ga.tao_run.initialize
    if seed_cap:
        def capped(z, unknown):
            d, known, slope, info = original(z, unknown)
            return np.where(known, np.minimum(d, H_flat), d), known, slope, dict(info, seed_cap_m=H_flat)
        ga.tao_run.initialize = capped
    try:
        pred, stat = ga.tao_run.recession(inp, F, init['alpha_rule'], H_flat, key + '/route1')
    finally:
        ga.tao_run.initialize = original
    if not stat['complete']:
        raise RuntimeError(f'{key}: incomplete, {stat["remaining_unknown"]} cells left')
    pred[~F] = np.nan
    t = g['transform']
    with rasterio.open(folder / f'{key}_route1_tao_F.tif', 'w', driver='GTiff', height=pred.shape[0],
                       width=pred.shape[1], count=1, dtype='float32', nodata=np.nan,
                       crs=f'EPSG:{reg.RESERVOIRS[key]["crs"]}', compress='deflate',
                       transform=rasterio.Affine(t[0], t[1], t[2], t[3], t[4], t[5])) as d:
        d.write(pred.astype('float32'), 1)
    info = {k: v for k, v in init['info'].items() if not isinstance(v, (np.ndarray,))}
    return pred, dict(alpha=init['alpha_rule'], iterations=stat['iterations'], init=info)


def survey_cells(key, g):
    """All survey reference cells on the analysis grid (EGM2008, converted by Codex)."""
    ref_path, grid_path = ssc.REFERENCES[key]
    with np.load(ref_path) as d:
        rr, cc, z = d['row'].astype(int), d['col'].astype(int), d['elevation_EGM2008_m'].astype(float)
    if grid_path is not None:
        with np.load(grid_path) as d:
            t = d['transform']
        rr = rr + int(round((g['transform'][5] - t[5]) / 30)); cc = cc + int(round((t[2] - g['transform'][2]) / 30))
    ok = (rr >= 0) & (rr < g['dem'].shape[0]) & (cc >= 0) & (cc < g['dem'].shape[1]) & np.isfinite(z)
    return rr[ok], cc[ok], z[ok]


def err_stats(pred, ref):
    e = pred - ref
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return dict(n=0)
    return dict(n=int(len(e)), RMSE=float(np.sqrt(np.mean(e * e))), MAE=float(np.mean(abs(e))), bias=float(e.mean()))


def analyse(key, folder, seed_cap=False):
    rec = reg.RESERVOIRS[key]
    g = reg.load_reservoir(key)
    dem, W0 = g['dem'], g['W0']
    F, H_flat, share = flattened_region(dem, W0, g['swo1'])
    if F[:3].any() or F[-3:].any() or F[:, :3].any() or F[:, -3:].any():
        raise RuntimeError(f'{key}: F touches the grid border')
    tao, meta = run_tao(key, g, F, H_flat, folder, seed_cap)
    z = dem.copy(); z[F] = tao[F]
    off = reg.egm_offset_m(key, g); cur = reg.official_curve(key, off)
    H_pool = float(reg.to_egm(key, rec['pool_ft'], off))
    masks = {'swo1': g['swo1'], 'dem_fill': None}
    if 'official' in g:
        masks['bound'] = g['official']
    lo = np.floor(min(np.nanmin(z[F]), cur.H.min()) / aev.STEP) * aev.STEP
    tab = aev.curve_table(z, W0 | F, masks, np.arange(lo, H_pool + 2 + aev.STEP, aev.STEP))
    tab.to_csv(folder / f'{key}_route1_curves.csv', index=False)
    off_at = lambda H, c: float(np.interp(H, cur.H, cur[c]))
    at = lambda H, c: float(np.interp(H, tab.H, tab[c]))
    Vmax = off_at(H_pool, 'volume_km3')
    v_tao = float(np.sum(np.clip(H_flat - z[F], 0, None))) * reg.CELL_AREA / 1e9
    prev = pd.read_csv(AEV / 'summary_reservoirs.csv').set_index('id').loc[key]
    row = dict(id=key, name=rec['name'], H_flat_m=H_flat, F_km2=F.sum() * reg.CELL_AREA / 1e6,
               W0_km2=W0.sum() * reg.CELL_AREA / 1e6, A_TWDB_at_H_flat=off_at(H_flat, 'area_km2'),
               alpha_route1=meta['alpha'], alpha_M0=prev['alpha_M0'],
               buffer_slope_deg_route1=meta['init']['buffer_slope_deg'],
               share_below_H_flat_pct=off_at(H_flat, 'volume_km3') / Vmax * 100,
               V_route1_tao_km3=v_tao, V_TWDB_below_H_flat_km3=off_at(H_flat, 'volume_km3'),
               Vmax_TWDB_km3=Vmax)
    row['route1_tao_part_PE_pct'] = (v_tao / row['V_TWDB_below_H_flat_km3'] - 1) * 100
    row['M0_own_domain_PE_pct'] = prev['M0_own_domain_PE_pct']
    for name in masks:
        row[f'Vmax_route1_{name}_PE_pct'] = (at(H_pool, f'V_{name}') / Vmax - 1) * 100
        row[f'Vmax_M0_{name}_PE_pct'] = prev.get(f'Vmax_TaoW0_{name}_PE_pct')
        sel = (cur.H <= H_pool) & (cur.H >= tab.H.min())
        mA = np.interp(cur.H[sel], tab.H, tab[f'A_{name}'])
        row[f'AE_NRMSE_route1_{name}_pct'] = aev.fit_stats(mA, cur.area_km2[sel].values).get('NRMSE_pct')
        row[f'AE_NRMSE_M0_{name}_pct'] = prev.get(f'AE_NRMSE_TaoW0_{name}_pct')
    # survey cells inside W0: same cells for both surfaces
    m0 = reg.load_m0(rec['m0'] if rec['m0'] is not None else AEV / f'{key}_M0.tif', g)
    zm0 = dem.copy(); zm0[W0] = m0[W0]
    rr, cc, zr = survey_cells(key, g)
    inW = W0[rr, cc]
    parts = dict(W0=inW, F=inW & F[rr, cc], W0_not_F=inW & ~F[rr, cc])
    cells = {}
    for pname, msk in parts.items():
        cells[pname] = dict(M0=err_stats(zm0[rr, cc][msk], zr[msk]), route1=err_stats(z[rr, cc][msk], zr[msk]))
        for model in ['M0', 'route1']:
            s = cells[pname][model]
            row[f'cells_{pname}_{model}_RMSE'] = s.get('RMSE'); row[f'cells_{pname}_{model}_bias'] = s.get('bias')
        row[f'cells_{pname}_n'] = cells[pname]['M0'].get('n', 0)
    (folder / f'{key}_route1.json').write_text(json.dumps(dict(meta=meta, cells=cells), indent=2, default=float),
                                               encoding='utf-8')
    return row


def main():
    p = argparse.ArgumentParser(); p.add_argument('--keys', nargs='*', default=list(reg.RESERVOIRS))
    p.add_argument('--seed-cap', action='store_true')
    a = p.parse_args()
    out = OUT.parent / 'route1_tao_flat_v2' if a.seed_cap else OUT
    out.mkdir(parents=True, exist_ok=True)
    proto = out / 'protocol.json'
    if not proto.exists():  # written once, before the first prediction
        extra = dict(seed_cap='known seed heights capped at H_flat (min(DEM, H_flat)); added after v1 diagnosis',
                     v1='route1_tao_flat_v1 kept unchanged') if a.seed_cap else {}
        proto.write_text(json.dumps(dict(PROTOCOL, **extra, written_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),
                                    indent=2), encoding='utf-8')
    rows = []
    for key in a.keys:
        print('RESERVOIR', key, reg.RESERVOIRS[key]['name'], flush=True)
        try:
            rows.append(analyse(key, out, a.seed_cap))
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
        except Exception as exc:
            import traceback; traceback.print_exc()
            rows.append(dict(id=key, name=reg.RESERVOIRS[key]['name'], error=repr(exc)))
        pd.DataFrame(rows).to_csv(out / 'summary_route1.csv', index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
