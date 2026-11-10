"""Does the underwater slope resemble the drawdown-band slope, the upland slope, or Tao's slope field?

For every surveyed cell inside SWBD W0 with SWO occurrence >= 95% (permanent water):
  s_uw     = (z_shore - z_ref) / d mean underwater slope from the nearest W0 shore cell (its own DEM
             elevation z_shore, i.e. the SWBD-epoch shoreline) down to the surveyed bed
  s_band   = DEM slope sampled 0-480 m outward from that shore point, cells in the SWO>=1% region outside W0
  s_upland = same profile, cells outside the SWO>=1% region
  s_tao    = Tao's initialized (stretched) slope at the nearest non-zero shoreline position
             (frozen Tao v3 initialize via the v4 geometry adapter, run on W0 exactly like M0).
Depth statistics use H_W0 = median DEM on the ring just outside W0 (the shoreline Tao starts from).
Slopes are dimensionless (tan).
Scoring only; nothing is fitted.
Usage: .venv/Scripts/python.exe 04_Code/scripts/shore_slope_check_v1.py [--keys D01 ...]
"""
from pathlib import Path
import sys, json, argparse, datetime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import spearmanr
import aev_registry_v1 as reg

OUT = reg.ROOT / '05_Results/by_date/2026-09-29/shore_slope_check_v1'
PROFILE_CELLS = 16  # 0..480 m outward, the longest Tao buffer

# Surveyed reference cells (EGM2008 elevations, already datum-converted by Codex) and the grid they index.
REFERENCES = {
    'D01': (reg.V4 / 'inputs/D01_reference_supplementary.npz', reg.V4 / 'inputs/D01_inputs.npz'),
    'D02': (reg.AUDIT / 'TWDB_MEDINA_reference_cells.npz',
            reg.CODEX / '2026-09-17/twdb_fabdem_intersection_screen/TWDB_MEDINA_inputs.npz'),
    'D03': (reg.CODEX / '2026-09-23/dev_admission_closure_20260923_v1/reference_current_grid/D03_reference_cells_current_grid.npz', None),
    'D04': (reg.CODEX / '2026-09-23/dev_admission_closure_20260923_v1/reference_current_grid/D04_reference_cells_current_grid.npz', None),
    'D05': (reg.AUDIT / 'TWDB_MEREDITH_reference_cells.npz',
            reg.CODEX / '2026-09-17/twdb_fabdem_intersection_screen/TWDB_MEREDITH_inputs.npz'),
    'D06': (reg.CODEX / '2026-09-23/dev_admission_closure_20260923_v1/reference_current_grid/D06_reference_cells_current_grid.npz', None),
}


def reference_cells(key, g):
    ref_path, grid_path = REFERENCES[key]
    with np.load(ref_path) as d:
        rr, cc, z = d['row'].astype(int), d['col'].astype(int), d['elevation_EGM2008_m'].astype(float)
    if grid_path is not None:
        with np.load(grid_path) as d:
            t = d['transform']
        rr = rr + int(round((g['transform'][5] - t[5]) / 30))
        cc = cc + int(round((t[2] - g['transform'][2]) / 30))
    ok = (rr >= 0) & (rr < g['dem'].shape[0]) & (cc >= 0) & (cc < g['dem'].shape[1]) & np.isfinite(z)
    rr, cc, z = rr[ok], cc[ok], z[ok]
    keep = g['W0'][rr, cc] & (g['occ'][rr, cc] >= 95)
    return rr[keep], cc[keep], z[keep]


def tao_slope_field(g):
    sys.path.insert(0, str(reg.ROOT / '04_Code/versions/m0_m1_m2_runner_v4'))
    import geometry_adapter as ga
    init = ga.initialize(g['dem'], g['W0'])
    return np.asarray(init['slope'], float), init['info'], init['alpha_rule']


def ls_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    vx = np.var(x)
    return np.nan if vx == 0 else float(np.mean((x - x.mean()) * (y - y.mean())) / vx)


def analyse(key):
    g = reg.load_reservoir(key)
    rr, cc, zref = reference_cells(key, g)
    dem, W0 = g['dem'], g['W0']
    band = g['swo1'] & ~W0
    upland = ~g['swo1']
    H0 = float(np.median(dem[ndi.binary_dilation(W0, np.ones((3, 3))) & ~W0]))
    tao, info, alpha = tao_slope_field(g)
    dist, idx = ndi.distance_transform_edt(W0, return_indices=True)
    _, tidx = ndi.distance_transform_edt(~(tao > 0), return_indices=True)
    ny, nx = dem.shape
    recs = []
    for r, c, zr in zip(rr, cc, zref):
        pr, pc = idx[0, r, c], idx[1, r, c]
        d = dist[r, c] * 30.0
        v = np.array([pr - r, pc - c], float)
        n = np.hypot(*v)
        if n == 0:
            continue
        v /= n
        pts = {'band': ([], []), 'upland': ([], [])}
        for k in range(PROFILE_CELLS + 1):
            sr, sc = int(round(pr + v[0] * k)), int(round(pc + v[1] * k))
            if not (0 <= sr < ny and 0 <= sc < nx) or W0[sr, sc] or not np.isfinite(dem[sr, sc]):
                continue
            cls = 'band' if band[sr, sc] else ('upland' if upland[sr, sc] else None)
            if cls:
                pts[cls][0].append(k * 30.0); pts[cls][1].append(dem[sr, sc])
        rec = dict(row=r, col=c, depth=H0 - zr, d=d, z_shore=dem[pr, pc], s_uw=(dem[pr, pc] - zr) / d,
                   s_tao=tao[tidx[0, pr, pc], tidx[1, pr, pc]])
        for cls, (x, y) in pts.items():
            rec['s_' + cls] = ls_slope(x, y) if len(x) >= 3 else np.nan
        recs.append(rec)
    df = pd.DataFrame(recs)
    deg = lambda s: float(np.degrees(np.arctan(s)))
    out = dict(id=key, name=reg.RESERVOIRS[key]['name'], n_cells=len(df), H_W0_m=H0, alpha_rule=alpha,
               tao_buffer_slope_deg=info['buffer_slope_deg'],
               depth_median_m=float(df.depth.median()), depth_p90_m=float(df.depth.quantile(.9)),
               shore_distance_median_m=float(df.d.median()))
    for k in ['s_uw', 's_band', 's_upland', 's_tao']:
        s = df[k].dropna()
        out[f'{k}_median_deg'] = deg(s.median()) if len(s) else None
        out[f'{k}_n'] = int(len(s))
    for k in ['s_band', 's_upland', 's_tao']:
        ok = df[['s_uw', k]].dropna()
        out[f'rho_s_uw_{k}'] = float(spearmanr(ok.s_uw, ok[k])[0]) if len(ok) > 10 else None
    out['rho_depth_distance'] = float(spearmanr(df.depth, df.d)[0])
    deep = df.depth >= df.depth.quantile(.75)
    for k in ['s_uw', 's_band', 's_upland', 's_tao']:
        s = df.loc[deep, k].dropna()
        out[f'deep_{k}_median_deg'] = deg(s.median()) if len(s) else None
    out['tao_over_uw_ratio'] = float(df.s_tao.median() / df.s_uw.median())
    out['band_over_uw_ratio'] = float(df.s_band.median() / df.s_uw.median()) if df.s_band.notna().any() else None
    df.to_csv(OUT / f'{key}_cells.csv', index=False)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--keys', nargs='*', default=list(REFERENCES))
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in a.keys:
        print('RESERVOIR', key, flush=True)
        rows.append(analyse(key))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
        pd.DataFrame(rows).to_csv(OUT / 'summary.csv', index=False, encoding='utf-8-sig')
    (OUT / 'run_info.json').write_text(json.dumps(dict(
        script='04_Code/scripts/shore_slope_check_v1.py', run_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        keys=a.keys, profile_m=PROFILE_CELLS * 30, permanent_rule='W0 & SWO occurrence >= 95'), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
