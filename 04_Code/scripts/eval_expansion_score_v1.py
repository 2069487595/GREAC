"""Step 7b (expansion protocol 446b182e1, section 6): score the locked runs on the 101, tests H1-H3, strata, QA test.

Order: verify every locked output hash (route1_expansion_run_v1/lock_manifest.json; any mismatch stops); strata come
from strata.csv (committed before this script ran). Per reservoir and model, as route1_test_score_v1:
  surface = DEM with the model inside its own region: R1-C / T-flat / mDEM on F (mDEM case 1 = the R1-C raster),
  T-SWBD on W0, GLOBathy on its polygon cells; A-E-V inside SWO>=1% (0.25 m levels, wet cells connected to W0 | F |
  region; arrays cropped to the bounding box of those masks, which leaves the result unchanged);
  primary: Vmax error at the report's normal pool; secondary: A-E / V-E r and NRMSE over official levels up to the
  pool; datum sensitivity: official area at the pool x 1 m as % of Vmax.
Official curves: eval_expansion_registry_v1.official (TWDB table_id from twdb_download_list; USBR curves/<key>).
Sonar (USBR A with a downloaded package, gis_download_manifest.csv): point layers whose name matches 'bath' or
  'thymetric' (case-insensitive; Tom Steed has two such layers, both used; aerial and RTK shore-point layers are not
  sonar); Z from the point geometry, taken as feet in the datum of the report tables and converted with the same
  offset as the official curve. Unit check fixed here: the 5th-95th percentile of Z must lie within
  [curve bottom - 50 ft, pool + 50 ft], else the reservoir gets no sonar score. Cells = median of the points per 30 m
  cell; RMSE over cells inside W0, also split W0 & F and W0 minus F.
Tests (unit = reservoir): H1 |Vmax error| R1-C < T-flat, one-sided paired Wilcoxon, TWDB new + USBR A, plus wins and
  a bootstrap 95 % interval of the median paired difference (10,000 resamples, seed 20260930); H2 R1-C vs GLOBathy,
  two-sided, USBR A with GLOBathy's estimated Dmax (the 6 test reservoirs reported separately from
  baseline_products_v1); H3 mDEM vs R1-C, two-sided, case-2 reservoirs, plus the descriptive result without the
  forest-flagged ones (mdem_forest_flags.csv). QA test: Spearman of R1-C |Vmax error| with the invisible share.
Outputs in 05_Results/by_date/2026-09-30/route1_expansion_score_v1/.
Usage: .venv/Scripts/python.exe 04_Code/scripts/eval_expansion_score_v1.py [--keys ...] [--part i/n] [--tests]
"""
from pathlib import Path
import sys, json, argparse, hashlib, warnings, re, zipfile, os

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import pyproj
import pyogrio
from pyogrio.raw import read as ogr_read
import shapely
from pyproj import Transformer
from scipy import stats
import aev_registry_v1 as reg
import aev_observation_share_v1 as aev
import route1_tao_flat_v1 as r1
import eval_expansion_registry_v1 as er
import eval_expansion_run_v1 as run

OUT = reg.ROOT / '05_Results/by_date/2026-09-30/route1_expansion_score_v1'
BASE = reg.ROOT / '05_Results/by_date/2026-09-29/baseline_products_v1'
MODELS = ['R1-C', 'T-flat', 'T-SWBD', 'mDEM', 'GLOBathy']
SEED = 20260930
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_lock():
    lock = json.loads((run.OUT / 'lock_manifest.json').read_text(encoding='utf-8'))
    for key, r in lock['runs'].items():
        for name, h in r['outputs'].items():
            if sha(run.OUT / key / name) != h:
                raise RuntimeError(f'locked output changed: {key}/{name}')
    return lock


def official(batch, key):
    if batch == 'TWDB':
        f = pd.read_csv(er.RES / 'twdb_facts.csv').set_index('key').loc[key].to_dict()
        tid = int(pd.read_csv(er.RES / 'twdb_download_list.csv').set_index('entity_id').loc[key, 'table_id'])
        return er.official(key, f, tid) + (f,)
    f = pd.read_csv(er.RES / 'usbr_facts.csv').set_index('key').loc[key].to_dict()
    if pd.isna(f.get('feet_offset')):
        f['feet_offset'] = 0.0
    return er.official(key, f) + (f,)


def sonar_cells(key, g, off, cur, pool_ft, manifest='gis_download_manifest.csv', exclude=None):
    """manifest / exclude (regex on layer names) were added for step 2 (group B); the defaults keep step 1 unchanged."""
    m = pd.read_csv(er.RES / manifest)
    m = m[(m.key == key) & m.bytes.notna()]
    if m.empty:
        return None, dict(status='no package')
    f = reg.ROOT / m.path.iloc[0]
    gdb = sorted({n.split('/')[0] for n in zipfile.ZipFile(f).namelist() if '.gdb' in n.split('/')[0]})[0]
    path = '/vsizip/' + os.path.abspath(f).replace('\\', '/') + '/' + gdb
    layers = [n for n, geom in pyogrio.list_layers(path) if 'Point' in str(geom) and re.search(r'(?i)bath|thymetric', n)
              and not (exclude and re.search(exclude, n))]
    if not layers:
        return None, dict(status='no bathymetric point layer')
    xs, ys, zs = [], [], []
    for lyr in layers:
        meta, _, geoms, _ = ogr_read(path, layer=lyr, columns=[])
        c = shapely.get_coordinates(shapely.from_wkb(geoms), include_z=True)
        x, y = Transformer.from_crs(pyproj.CRS.from_user_input(meta['crs']), g['meta']['epsg'], always_xy=True).transform(c[:, 0], c[:, 1])
        xs.append(x); ys.append(y); zs.append(c[:, 2])
    x, y, zft = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
    p5, p95 = np.nanpercentile(zft, [5, 95])
    lo_ok, hi_ok = cur.elevation_ft.min() - 50, pool_ft + 50
    info = dict(layers=layers, points=int(len(zft)), z_ft_p5=float(p5), z_ft_p95=float(p95), unit_window_ft=[float(lo_ok), float(hi_ok)])
    if not (lo_ok <= p5 and p95 <= hi_ok):
        return None, dict(info, status='unit check failed')
    z = (zft + off['feet']) * er.FT + off['metres']
    t = g['transform']
    rr = np.floor((t[5] - y) / 30).astype(int); cc = np.floor((x - t[2]) / 30).astype(int)
    ok = (rr >= 0) & (rr < g['dem'].shape[0]) & (cc >= 0) & (cc < g['dem'].shape[1]) & np.isfinite(z)
    df = pd.DataFrame(dict(r=rr[ok], c=cc[ok], z=z[ok])).groupby(['r', 'c']).z.median().reset_index()
    return (df.r.values, df.c.values, df.z.values), dict(info, status='ok', points_on_grid=int(ok.sum()), cells=len(df))


def crop(masks):
    rows = np.flatnonzero(np.any(np.logical_or.reduce(masks), axis=1)); cols = np.flatnonzero(np.any(np.logical_or.reduce(masks), axis=0))
    return slice(max(rows[0] - 1, 0), rows[-1] + 2), slice(max(cols[0] - 1, 0), cols[-1] + 2)


def surface(model, key, g, F, rec):
    W0 = g['W0']
    if model == 'T-SWBD':
        region, name = W0, f'{key}_M0.tif'
    elif model == 'GLOBathy':
        name = f'{key}_GLOBathy.tif'; region = None
    elif model == 'mDEM' and rec.get('mDEM_case') == 2:
        region, name = F, f'{key}_mDEM_tao_F.tif'
    elif model in ('R1-C', 'mDEM'):
        region, name = F, f'{key}_C_tao_F.tif'
    else:
        region, name = F, f'{key}_route1_tao_F.tif'
    pred = reg.load_m0(run.OUT / key / name, g)
    if region is None:
        region = np.isfinite(pred)
    z = g['dem'].copy(); z[region] = pred[region]
    return z, region


def score(batch, key):
    g = er.load(key)
    dem, W0, swo1 = g['dem'], g['W0'], g['swo1']
    F, H_flat, _ = r1.flattened_region(dem, W0, swo1)
    rec = json.loads((run.OUT / key / f'{key}_run.json').read_text(encoding='utf-8'))
    cur, H_pool, off, f = official(batch, key)
    Vmax = float(np.interp(H_pool, cur.H, cur.volume_km3)); A_pool = float(np.interp(H_pool, cur.H, cur.area_km2))
    area_match = float(np.interp(F.sum() * 9e-4, np.maximum.accumulate(cur.area_km2.values), cur.H.values))
    son, sinfo = sonar_cells(key, g, off, cur, f['pool_ft']) if batch == 'USBR-A' else (None, dict(status='TWDB: not scored'))
    (OUT / key).mkdir(parents=True, exist_ok=True)
    rows = []
    for model in MODELS:
        z, region = surface(model, key, g, F, rec)
        sl = crop([swo1, W0, F, region])
        lo = np.floor(min(np.nanmin(z[region]), cur.H.min()) / aev.STEP) * aev.STEP
        tab = aev.curve_table(z[sl], (W0 | F | region)[sl], {'swo1': swo1[sl]}, np.arange(lo, H_pool + 2 + aev.STEP, aev.STEP))
        tab.to_csv(OUT / key / f'{key}_{model}_curves.csv', index=False)
        at = lambda H, c: float(np.interp(H, tab.H, tab[c]))
        sel = (cur.H <= H_pool + 1e-6) & (cur.H >= tab.H.min())
        ae = aev.fit_stats(np.interp(cur.H[sel], tab.H, tab.A_swo1), cur.area_km2[sel].values)
        ve = aev.fit_stats(np.interp(cur.H[sel], tab.H, tab.V_swo1), cur.volume_km3[sel].values)
        row = dict(batch=batch, key=key, model=model, Vmax_official_km3=Vmax, A_pool_official_km2=A_pool, H_pool_m=H_pool,
                   Vmax_model_km3=at(H_pool, 'V_swo1'), Vmax_PE_pct=(at(H_pool, 'V_swo1') / Vmax - 1) * 100,
                   A_pool_model_km2=at(H_pool, 'A_swo1'), AE_r=ae.get('r'), AE_NRMSE_pct=ae.get('NRMSE_pct'),
                   VE_r=ve.get('r'), VE_NRMSE_pct=ve.get('NRMSE_pct'), n_levels=ae.get('n'))
        if son is not None:
            rr, cc, zr = son
            inW = W0[rr, cc]
            for pname, msk in dict(W0=inW, W0_F=inW & F[rr, cc], W0_notF=inW & ~F[rr, cc]).items():
                s = r1.err_stats(z[rr, cc][msk], zr[msk])
                row[f'cells_{pname}_n'] = s.get('n', 0); row[f'cells_{pname}_RMSE'] = s.get('RMSE'); row[f'cells_{pname}_bias'] = s.get('bias')
        rows.append(row)
    gl = rec.get('GLOBathy', {})
    info = dict(batch=batch, key=key, pool_ft=f['pool_ft'], datum=f['datum'], datum_uncertain=bool(f.get('datum_uncertain')) if pd.notna(f.get('datum_uncertain')) else False,
                offset=off, H_pool_m=H_pool, Vmax_km3=Vmax, A_pool_km2=A_pool, dV_per_m_official_pct=A_pool * 1e-3 / Vmax * 100,
                H_flat_m=H_flat, H_seed_m=rec['H_seed_m'], rule_C_fallback=rec['rule_C_fallback'],
                H_area_matched_official_m=area_match, seed_minus_area_matched_m=rec['H_seed_m'] - area_match,
                flat_minus_area_matched_m=H_flat - area_match,
                official_share_below_H_flat=float(np.interp(H_flat, cur.H, cur.volume_km3)) / Vmax,
                curve_bottom_m=float(cur.H.min()), mDEM_case=rec.get('mDEM_case'),
                GLOBathy_uses_observed_Dmax=gl.get('uses_observed_Dmax'), GLOBathy_Hylak_id=gl.get('Hylak_id'), sonar=sinfo)
    (OUT / key / f'{key}_score.json').write_text(json.dumps(info, indent=2, default=float), encoding='utf-8')
    return rows, info


def boot_median(d, n=10000):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = np.median(d[idx], axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def paired(s, a, b, keys, alternative):
    x = s[s.model == a].set_index('key').loc[keys, 'Vmax_PE_pct'].abs().values
    y = s[s.model == b].set_index('key').loc[keys, 'Vmax_PE_pct'].abs().values
    d = x - y
    w = stats.wilcoxon(x, y, alternative=alternative)
    return dict(n=len(keys), a=a, b=b, alternative=alternative, statistic=float(w.statistic), p=float(w.pvalue),
                wins_a=int((x < y).sum()), wins_b=int((x > y).sum()), ties=int((x == y).sum()),
                median_abs_a=float(np.median(x)), median_abs_b=float(np.median(y)),
                median_diff_abs_pct=float(np.median(d)), median_diff_ci95=boot_median(d))


def tests(s, levels):
    strata = pd.read_csv(OUT / 'strata.csv').set_index('key')
    flags = pd.read_csv(er.RES / 'mdem_forest_flags.csv').set_index('key')
    allk = list(levels.key)
    res = dict(H1=paired(s, 'R1-C', 'T-flat', allk, 'less'))
    res['H1_by_batch'] = {b: paired(s, 'R1-C', 'T-flat', list(levels[levels.batch == b].key), 'less') for b in ['TWDB', 'USBR-A']}
    h2 = list(levels[(levels.batch == 'USBR-A') & (levels.GLOBathy_uses_observed_Dmax == False)].key)
    res['H2'] = paired(s, 'R1-C', 'GLOBathy', h2, 'two-sided')
    t6 = pd.read_csv(BASE / 'globathy_compare.csv')
    t6 = t6[t6.set == 'test'].pivot(index='id', columns='model', values='Vmax_PE_pct')
    res['H2_test6_separate'] = dict(n=len(t6), median_abs_R1C=float(t6['R1-C'].abs().median()), median_abs_GLOBathy=float(t6['GLOBathy'].abs().median()),
                                    wins_R1C=int((t6['R1-C'].abs() < t6['GLOBathy'].abs()).sum()))
    c2 = list(levels[levels.mDEM_case == 2].key)
    res['H3'] = paired(s, 'mDEM', 'R1-C', c2, 'two-sided')
    c2n = [k for k in c2 if not bool(flags.loc[k, 'forest_flag'])]
    res['H3_without_forest_flag_descriptive'] = paired(s, 'mDEM', 'R1-C', c2n, 'two-sided')
    r = s[s.model == 'R1-C'].set_index('key')
    rho = stats.spearmanr(r.loc[allk, 'Vmax_PE_pct'].abs(), strata.loc[allk, 'invisible_share'])
    res['QA_spearman_absPE_vs_invisible_share'] = dict(n=len(allk), rho=float(rho.statistic), p=float(rho.pvalue))
    for b in ['TWDB', 'USBR-A']:
        kk = list(levels[levels.batch == b].key)
        rb = stats.spearmanr(r.loc[kk, 'Vmax_PE_pct'].abs(), strata.loc[kk, 'invisible_share'])
        res[f'QA_spearman_{b}'] = dict(n=len(kk), rho=float(rb.statistic), p=float(rb.pvalue))
    # strata: median |Vmax error| per model and R1-C wins over T-flat
    piv = s.pivot(index='key', columns='model', values='Vmax_PE_pct').abs()
    st = strata.loc[allk].copy(); st['batch_'] = levels.set_index('key').loc[allk, 'batch']
    rows = []
    for var in ['batch_', 'F_larger_than_W0', 'slope_tertile', 'invisible_tertile', 'survey_gap_tertile']:
        for lev, grp in st.groupby(var):
            p = piv.loc[grp.index]
            rows.append(dict(stratum=var.rstrip('_'), level=str(lev), n=len(p), **{f'median_abs_{m}': float(p[m].median()) for m in MODELS},
                             R1C_beats_Tflat=int((p['R1-C'] < p['T-flat']).sum())))
    pd.DataFrame(rows).to_csv(OUT / 'strata_report.csv', index=False)
    agg = s.assign(abs_pe=s.Vmax_PE_pct.abs()).groupby(['batch', 'model']).agg(
        n=('abs_pe', 'count'), median_abs_Vmax_PE=('abs_pe', 'median'), mean_abs_Vmax_PE=('abs_pe', 'mean'),
        median_Vmax_PE=('Vmax_PE_pct', 'median'), median_AE_NRMSE=('AE_NRMSE_pct', 'median'),
        median_VE_NRMSE=('VE_NRMSE_pct', 'median'), median_AE_r=('AE_r', 'median'))
    if 'cells_W0_RMSE' in s:
        agg = agg.join(s.groupby(['batch', 'model']).agg(sonar_n=('cells_W0_RMSE', 'count'), median_sonar_W0_RMSE=('cells_W0_RMSE', 'median')))
    agg.to_csv(OUT / 'aggregate.csv')
    (OUT / 'tests.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')
    return res, agg


def main():
    warnings.filterwarnings('ignore')
    p = argparse.ArgumentParser(); p.add_argument('--keys', nargs='*'); p.add_argument('--part'); p.add_argument('--tests', action='store_true')
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.tests:
        s = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob('*/*_rows.csv'))])
        levels = pd.DataFrame([{k: v for k, v in json.loads(f.read_text(encoding='utf-8')).items() if not isinstance(v, dict)}
                               for f in sorted(OUT.glob('*/*_score.json'))])
        s.to_csv(OUT / 'summary.csv', index=False); levels.to_csv(OUT / 'levels.csv', index=False)
        res, agg = tests(s, levels)
        print(json.dumps(res, indent=1, default=float)); print(agg.round(2).to_string())
        return
    verify_lock()
    jobs = run.sample_keys()
    if a.keys:
        jobs = [j for j in jobs if j[1] in a.keys]
    if a.part:
        i, n = map(int, a.part.split('/')); jobs = jobs[i - 1::n]
    for batch, key in jobs:
        if (OUT / key / f'{key}_rows.csv').exists():
            continue
        try:
            rows, info = score(batch, key)
            pd.DataFrame(rows).to_csv(OUT / key / f'{key}_rows.csv', index=False)
            print(key, batch, [(r['model'], round(r['Vmax_PE_pct'], 1)) for r in rows], info['sonar'].get('status'), flush=True)
        except Exception as exc:
            import traceback
            (OUT / key).mkdir(parents=True, exist_ok=True)
            (OUT / key / f'{key}_score_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
            print(key, batch, 'FAILED', repr(exc)[:200], flush=True)


if __name__ == '__main__':
    main()
