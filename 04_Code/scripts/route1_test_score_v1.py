"""Route-1 test set, step 4 of docs/plans/2026-09-29-route1-test-protocol.md: score the locked outputs.

Run only after route1_test_v1/lock_manifest.json is committed. Hashes of all locked rasters are verified
first; any mismatch stops the scoring. Then, per reservoir and model (R1-C, T-flat, T-SWBD):
  * A-E-V inside the SWO>=1% extent (surface = DEM, with the model inside its own unknown region)
  * full-pool volume error at the report's normal pool; datum sensitivity as % of Vmax per metre
  * A-E and V-E r / NRMSE (range-normalised) / bias over official levels up to the pool
  * sonar cells inside W0 (median of points per 30 m cell), split into W0 & F and W0 minus F;
    Cascade has no separate sonar layer: its mixed bathymetric + aerial layer is scored as secondary
Official elevations and survey Z (project feet) go to EGM2008 through usbr_test_registry_v1.egm_offset_m.
Usage: .venv/Scripts/python.exe 04_Code/scripts/route1_test_score_v1.py [--keys T01 ...]
"""
from pathlib import Path
import sys, json, argparse, hashlib, warnings

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import pyproj
from pyproj import Transformer
from pyogrio.raw import read
import aev_registry_v1 as reg
import aev_observation_share_v1 as aev
import route1_tao_flat_v1 as r1
import usbr_test_registry_v1 as tr
import usbr_official_curves_v1 as oc

RUN = reg.ROOT / '05_Results/by_date/2026-09-29/route1_test_v1'
MODELS = {'R1-C': '{k}_C_tao_F.tif', 'T-flat': '{k}_route1_tao_F.tif', 'T-SWBD': '{k}_M0.tif'}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_lock():
    lock = json.loads((RUN / 'lock_manifest.json').read_text(encoding='utf-8'))
    for key, run in lock['runs'].items():
        for name, h in run['outputs'].items():
            if sha(RUN / key / name) != h:
                raise RuntimeError(f'locked output changed: {key}/{name}')
    return lock


def sonar_cells(key, g, off):
    rec = tr.TEST[key]
    layer, secondary = rec['sonar_layer'], False
    if layer is None:
        layer, secondary = 'BathyAerialPoints', True
    path = f'/vsizip/{tr.SEALED / rec["archive"]}'
    import pyogrio, zipfile
    gdb = sorted({n.split('/')[0] for n in zipfile.ZipFile(tr.SEALED / rec['archive']).namelist() if '.gdb' in n.split('/')[0]})[0]
    meta, _, _, fields = read(f'{path}/{gdb}', layer=layer, columns=['POINT_X', 'POINT_Y', 'POINT_Z'], read_geometry=False)
    cols = dict(zip(meta['fields'], fields))
    x, y = Transformer.from_crs(pyproj.CRS.from_user_input(meta['crs']), rec['epsg'], always_xy=True).transform(
        cols['POINT_X'].astype(float), cols['POINT_Y'].astype(float))
    z = (cols['POINT_Z'].astype(float) + off['feet']) * reg.FT + off['metres']
    t = g['transform']
    rr = np.floor((t[5] - y) / 30).astype(int); cc = np.floor((x - t[2]) / 30).astype(int)
    ok = (rr >= 0) & (rr < g['dem'].shape[0]) & (cc >= 0) & (cc < g['dem'].shape[1]) & np.isfinite(z)
    df = pd.DataFrame(dict(r=rr[ok], c=cc[ok], z=z[ok])).groupby(['r', 'c']).z.agg(['median', 'count']).reset_index()
    return df.r.values, df.c.values, df['median'].values, dict(layer=layer, secondary=secondary, points=int(ok.sum()),
                                                                 points_outside_grid=int((~ok).sum()), cells=len(df),
                                                                 z_ft_range=[float(np.nanmin(cols['POINT_Z'])), float(np.nanmax(cols['POINT_Z']))])


def score(key, lock):
    g = tr.load_test(key)
    dem, W0, swo1 = g['dem'], g['W0'], g['swo1']
    F, H_flat, _ = r1.flattened_region(dem, W0, swo1)
    run = json.loads((RUN / key / f'{key}_run.json').read_text(encoding='utf-8'))
    off = tr.egm_offset_m(key)
    cur = tr.official_curve(key, off)
    pool_ft = oc.POOL[key][0]
    H_pool = tr.to_egm(pool_ft, off)
    Vmax = float(np.interp(H_pool, cur.H, cur.volume_km3))
    area_match = float(np.interp(F.sum() * 9e-4, np.maximum.accumulate(cur.area_km2.values), cur.H.values))
    rr, cc, zr, pinfo = sonar_cells(key, g, off)
    inW = W0[rr, cc]
    rows, curves = [], {}
    for model, pattern in MODELS.items():
        region = W0 if model == 'T-SWBD' else F
        pred = reg.load_m0(RUN / key / pattern.format(k=key), g)
        z = dem.copy(); z[region] = pred[region]
        lo = np.floor(min(np.nanmin(z[region]), cur.H.min()) / aev.STEP) * aev.STEP
        tab = aev.curve_table(z, W0 | F, {'swo1': swo1}, np.arange(lo, H_pool + 2 + aev.STEP, aev.STEP))
        tab.to_csv(RUN / key / f'{key}_{model}_curves.csv', index=False)
        at = lambda H, c: float(np.interp(H, tab.H, tab[c]))
        sel = (cur.H <= H_pool + 1e-6) & (cur.H >= tab.H.min())
        ae = aev.fit_stats(np.interp(cur.H[sel], tab.H, tab.A_swo1), cur.area_km2[sel].values)
        ve = aev.fit_stats(np.interp(cur.H[sel], tab.H, tab.V_swo1), cur.volume_km3[sel].values)
        row = dict(id=key, name=tr.TEST[key]['name'], model=model, Vmax_official_km3=Vmax, H_pool_m=H_pool,
                   Vmax_model_km3=at(H_pool, 'V_swo1'), Vmax_PE_pct=(at(H_pool, 'V_swo1') / Vmax - 1) * 100,
                   dV_per_m_pct=(at(H_pool + 1, 'V_swo1') - at(H_pool - 1, 'V_swo1')) / 2 / Vmax * 100,
                   A_pool_model_km2=at(H_pool, 'A_swo1'), A_pool_official_km2=float(np.interp(H_pool, cur.H, cur.area_km2)),
                   AE_r=ae.get('r'), AE_NRMSE_pct=ae.get('NRMSE_pct'), AE_bias_km2=ae.get('bias'),
                   VE_r=ve.get('r'), VE_NRMSE_pct=ve.get('NRMSE_pct'), VE_bias_km3=ve.get('bias'))
        for pname, msk in dict(W0=inW, W0_F=inW & F[rr, cc], W0_notF=inW & ~F[rr, cc]).items():
            s = r1.err_stats(z[rr, cc][msk], zr[msk])
            row[f'cells_{pname}_n'] = s.get('n', 0); row[f'cells_{pname}_RMSE'] = s.get('RMSE'); row[f'cells_{pname}_bias'] = s.get('bias')
        rows.append(row)
    info = dict(key=key, offset=off, pool_ft=pool_ft, pool_zone=oc.POOL[key][1], H_pool_m=H_pool, Vmax_km3=Vmax,
                H_flat_m=H_flat, H_seed_m=run['H_seed_m'], H_area_matched_official_m=area_match,
                seed_minus_area_matched_m=run['H_seed_m'] - area_match, flat_minus_area_matched_m=H_flat - area_match,
                share_below_H_flat_pct=float(np.interp(H_flat, cur.H, cur.volume_km3)) / Vmax * 100,
                share_below_H_W0_pct=float(np.interp(aev.outer_ring_level(dem, W0, np.zeros_like(W0))[0], cur.H, cur.volume_km3)) / Vmax * 100,
                sonar=pinfo, cells_in_W0=int(inW.sum()))
    (RUN / key / f'{key}_score.json').write_text(json.dumps(info, indent=2, default=float), encoding='utf-8')
    return rows, info


def main():
    warnings.filterwarnings('ignore')
    p = argparse.ArgumentParser(); p.add_argument('--keys', nargs='*', default=list(tr.TEST))
    a = p.parse_args()
    lock = verify_lock()
    rows, infos, errors = [], [], {}
    for key in a.keys:
        try:
            r, i = score(key, lock); rows += r; infos.append(i)
            print(key, [(x['model'], round(x['Vmax_PE_pct'], 1), round(x['cells_W0_RMSE'] or np.nan, 2)) for x in r], flush=True)
        except Exception as exc:
            import traceback; traceback.print_exc(); errors[key] = repr(exc)
    s = pd.DataFrame(rows); s.to_csv(RUN / 'summary_test.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([{k: v for k, v in i.items() if not isinstance(v, dict)} for i in infos]).to_csv(
        RUN / 'levels_test.csv', index=False, encoding='utf-8-sig')
    if len(s):
        agg = s.assign(abs_pe=s.Vmax_PE_pct.abs()).groupby('model').agg(
            n=('abs_pe', 'count'), median_abs_Vmax_PE=('abs_pe', 'median'), mean_abs_Vmax_PE=('abs_pe', 'mean'),
            max_abs_Vmax_PE=('abs_pe', 'max'), median_AE_NRMSE=('AE_NRMSE_pct', 'median'), median_VE_NRMSE=('VE_NRMSE_pct', 'median'),
            median_cells_W0_RMSE=('cells_W0_RMSE', 'median'))
        agg.to_csv(RUN / 'aggregate_test.csv', encoding='utf-8-sig'); print(agg.round(2), flush=True)
    (RUN / 'score_errors.json').write_text(json.dumps(errors, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
