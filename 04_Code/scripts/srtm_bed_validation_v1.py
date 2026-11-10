"""Validation against SRTM-visible reservoir beds (protocol docs/plans/2026-10-08-route1-srtm-visible-bed-protocol.md,
frozen afdf3fd). Uses the locked global run (route1_global_run_v1) only.

Reference region X = F minus (W0 dilated 1 cell), minus cells within 300 m of the HydroLAKES pour point, minus NASADEM
voids; reference terrain = NASADEM + route1_mdem_v1.dem_offset; reference level = H_flat.
Steps (each its own call, in this order):
  --sample [--size lo:hi] [--part i/n]   admission rules A1-A6 from inputs only -> sample_parts/*.csv
  --merge                                 -> admission.csv
  --tflat [--part i/n]                    frozen T-flat on the admitted reservoirs -> tflat/<id>/
  --lock                                  hashes of the T-flat outputs -> tflat_lock.json
  --score                                 metrics in X below H_flat, H6 / H6b -> scores.csv, tests.json, aggregate.csv
Outputs: 05_Results/by_date/2026-10-08/srtm_bed_validation_v1/
"""
from pathlib import Path
import sys, json, argparse, hashlib, datetime, warnings, traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy import ndimage as ndi
from scipy import stats
import aev_registry_v1 as reg
import route1_tao_flat_v1 as r1
import route1_mdem_v1 as md
import route1_global_run_v1 as gr

OUT = reg.ROOT / '05_Results/by_date/2026-10-08/srtm_bed_validation_v1'
PROTOCOL = reg.ROOT / 'docs/plans/2026-10-08-route1-srtm-visible-bed-protocol.md'
EIGHT = np.ones((3, 3), bool)
SHARE_MIN, SHARE_SUBSET, X_MIN_KM2 = 0.70, 0.90, 1.0
P_IN_MIN, P_OUT_MIN, RING_TOL_M, DEPTH_MIN_M, POUR_M = 0.90, 0.80, 3.0, 5.0, 300.0
FOREST_M, FOREST_SHARE = -3.0, 0.3
SEED, STEP = 20261008, 0.25
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(hid):
    g = gr.load(hid)
    with np.load(gr.INPUTS / f'{hid}_inputs.npz') as d:
        g['nasadem'] = d['nasadem'].astype(float)
    return g


def pour_buffer(g, lon, lat):
    """Cells within POUR_M of the pour point (computed in a small window around it)."""
    t = g['transform']; shape = g['dem'].shape
    x, y = Transformer.from_crs(4326, int(g['meta']['epsg']), always_xy=True).transform(lon, lat)
    c0, r0 = (x - t[2]) / t[0], (y - t[5]) / t[4]
    k = int(np.ceil(POUR_M / 30.0)) + 2
    out = np.zeros(shape, bool)
    rs, re = max(int(r0) - k, 0), min(int(r0) + k + 1, shape[0]); cs, ce = max(int(c0) - k, 0), min(int(c0) + k + 1, shape[1])
    if re > rs and ce > cs:
        rr, cc = np.mgrid[rs:re, cs:ce]
        out[rs:re, cs:ce] = np.hypot(t[2] + (cc + .5) * t[0] - x, t[5] + (rr + .5) * t[4] - y) <= POUR_M
    return out


def regions(g, lon, lat):
    """F, H_flat, reference region X, shore ring, pour buffer."""
    F, H_flat, _ = r1.flattened_region(g['dem'], g['W0'], g['swo1'])
    W0d = ndi.binary_dilation(g['W0'], EIGHT)
    pour = pour_buffer(g, lon, lat)
    X = F & ~W0d & ~pour & np.isfinite(g['nasadem'])
    ring = ndi.binary_dilation(F, EIGHT, iterations=3) & ~F & ~W0d & ~pour & np.isfinite(g['dem']) & np.isfinite(g['nasadem'])
    return F, H_flat, X, ring


def admit(hid, lon, lat):
    g = load(hid)
    F, H_flat, X, ring = regions(g, lon, lat)
    rec = dict(Hylak_id=hid, F_km2=float(F.sum() * 9e-4), X_km2=float(X.sum() * 9e-4), X_share_of_F=float(X.sum() / F.sum()), H_flat_m=H_flat)
    fails = []
    if rec['X_share_of_F'] < SHARE_MIN:
        return dict(rec, status='excluded', rule='A2: X < 70 % of F')  # nothing else is computed for these
    if rec['X_km2'] < X_MIN_KM2:
        fails.append('A3: X < 1 km2')
    try:
        off, oinfo = md.dem_offset(g['dem'], g['nasadem'], g['swo1'], g['occ'])
    except RuntimeError as exc:
        return dict(rec, status='excluded', rule=(fails + ['A4: ' + str(exc)])[0])
    dist = ndi.distance_transform_edt(~g['swo1']) * 30.0
    st = ~g['swo1'] & (dist <= md.STABLE_RING_M) & (g['occ'] == 0) & np.isfinite(g['dem']) & np.isfinite(g['nasadem'])
    forest_share = float(((g['dem'][st] - g['nasadem'][st]) < FOREST_M).mean())
    zref = g['nasadem'] + off
    depth = np.clip(H_flat - zref[X], 0, None)
    rec.update(offset_m=off, offset_iqr_m=oinfo['iqr_m'], n_stable=oinfo['n_stable'], forest_share=forest_share,
               forest_flag=bool(forest_share >= FOREST_SHARE), p_in=float((zref[X] < H_flat).mean()),
               p_out=float((zref[ring] >= H_flat - RING_TOL_M).mean()) if ring.any() else np.nan, n_ring=int(ring.sum()),
               mean_ref_depth_m=float(depth.mean()), max_ref_depth_m=float(depth.max()), V_ref_km3=float(depth.sum() * 9e-7),
               subset_90=bool(rec['X_share_of_F'] >= SHARE_SUBSET))
    if not (rec['p_in'] >= P_IN_MIN and rec['p_out'] >= P_OUT_MIN):
        fails.append('A5: consistency check')
    if rec['mean_ref_depth_m'] < DEPTH_MIN_M:
        fails.append('A6: mean reference depth < 5 m')
    return dict(rec, status='excluded' if fails else 'admitted', rule=fails[0] if fails else '', all_failed=';'.join(fails))


def lst():
    return pd.read_csv(gr.OUT / 'global_list.csv').set_index('Hylak_id')


def sample(size, part):
    L = lst()
    ok = set(pd.read_csv(gr.OUT / 'global_summary.csv').Hylak_id)  # A1: locked global run has a product
    ids = [i for i in gr.jobs(size) if i in ok]
    tag = size.replace(':', '-')
    if part:
        i, n = map(int, part.split('/')); ids = ids[i - 1::n]; tag += f'_p{i}of{n}'
    (OUT / 'sample_parts').mkdir(parents=True, exist_ok=True)
    rows = []
    for k, hid in enumerate(ids):
        try:
            rows.append(admit(hid, float(L.loc[hid, 'Pour_long']), float(L.loc[hid, 'Pour_lat'])))
        except Exception as exc:
            rows.append(dict(Hylak_id=hid, status='excluded', rule='error: ' + repr(exc)[:150]))
        if rows[-1]['status'] == 'admitted' or k % 200 == 0:
            print(k, len(ids), hid, rows[-1]['status'], rows[-1].get('rule', ''), flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'sample_parts' / f'sample_{tag}.csv', index=False)
    print('PART_DONE', tag, len(rows), flush=True)


def merge():
    d = pd.concat([pd.read_csv(f) for f in sorted((OUT / 'sample_parts').glob('sample_*.csv'))]).drop_duplicates('Hylak_id')
    L = lst()
    d = d.join(L[['Lake_name', 'Country', 'Continent', 'Poly_src', 'Grand_id']], on='Hylak_id').sort_values('Hylak_id')
    d.to_csv(OUT / 'admission.csv', index=False, float_format='%.6g')
    a = d[d.status == 'admitted']
    print('checked', len(d), 'of', len(pd.read_csv(gr.OUT / 'global_summary.csv')), '| admitted', len(a), '| subset >= 90 %', int(a.subset_90.sum()),
          '| outside US', int((a.Country != 'United States of America').sum()), '| forest flag', int(a.forest_flag.sum()))
    print(d.rule.fillna('').value_counts().to_string())
    print(a.Continent.value_counts().to_dict())


def admitted():
    d = pd.read_csv(OUT / 'admission.csv')
    return d[d.status == 'admitted'].sort_values('F_km2')


def tflat(part):
    a = admitted(); ids = list(a.Hylak_id)
    if part:
        i, n = map(int, part.split('/')); ids = ids[i - 1::n]
    for hid in ids:
        folder = OUT / 'tflat' / str(hid)
        if (folder / f'{hid}_tflat.json').exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        try:
            g = gr.load(hid); key = str(hid)
            reg.RESERVOIRS.setdefault(key, dict(name=key, crs=int(g['meta']['epsg'])))
            F, H_flat, _ = r1.flattened_region(g['dem'], g['W0'], g['swo1'])
            _, info = r1.run_tao(key, g, F, H_flat, folder, seed_cap=False)
            (folder / f'{hid}_tflat.json').write_text(json.dumps(dict(Hylak_id=hid, H_flat_m=H_flat, alpha=info['alpha'], iterations=info['iterations'],
                                                                      inputs_sha256=sha(gr.INPUTS / f'{hid}_inputs.npz')), default=float), encoding='utf-8')
            print(hid, 'ok', flush=True)
        except Exception as exc:
            (folder / f'{hid}_error.json').write_text(json.dumps(dict(error=repr(exc), trace=traceback.format_exc())), encoding='utf-8')
            print(hid, 'FAILED', repr(exc)[:150], flush=True)


def lock():
    runs, missing = {}, []
    glock = json.loads((gr.OUT / 'lock_manifest.json').read_text(encoding='utf-8'))['runs']
    for hid in admitted().Hylak_id:
        f = OUT / 'tflat' / str(hid)
        if not (f / f'{hid}_tflat.json').exists():
            missing.append(int(hid)); continue
        runs[int(hid)] = dict(tflat={p.name: sha(p) for p in sorted(f.glob('*')) if p.suffix in ('.tif', '.json')},
                              global_locked={n: h for n, h in glock[str(hid)]['outputs'].items() if n.endswith(('_C_tao_F.tif', '_GLOBathy_est.tif'))})
    m = dict(protocol_sha256=sha(PROTOCOL), admission_sha256=sha(OUT / 'admission.csv'), script_sha256=sha(HERE / 'srtm_bed_validation_v1.py'),
             n=len(runs), missing=missing, runs=runs, locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (OUT / 'tflat_lock.json').write_text(json.dumps(m, indent=0), encoding='utf-8')
    print('LOCKED', len(runs), 'missing', missing, flush=True)


def metrics(z, zref, X, H_flat):
    zx, rx = z[X], zref[X]
    V = float(np.clip(H_flat - zx, 0, None).sum() * 9e-7); Vr = float(np.clip(H_flat - rx, 0, None).sum() * 9e-7)
    e = zx - rx
    levels = np.arange(np.floor(rx.min() / STEP) * STEP, H_flat + 1e-9, STEP)
    A = np.searchsorted(np.sort(zx), levels, side='right'); Ar = np.searchsorted(np.sort(rx), levels, side='right')
    return dict(V_km3=V, V_ref_km3=Vr, V_PE_pct=(V / Vr - 1) * 100, bed_RMSE_m=float(np.sqrt(np.mean(e * e))), bed_bias_m=float(e.mean()),
                AE_NRMSE_pct=float(np.sqrt(np.mean((A - Ar) ** 2.0)) / X.sum() * 100))


def paired(p, a, b, keys, rng):
    d = (p.loc[keys, a] - p.loc[keys, b]).dropna()
    nz = d[d != 0]
    w = stats.wilcoxon(nz, alternative='less') if len(nz) else None
    boot = np.median(rng.choice(d.values, (10000, len(d))), axis=1)
    return dict(n=int(len(d)), wins_a=int((d < 0).sum()), wins_b=int((d > 0).sum()), ties=int((d == 0).sum()),
                median_a=float(p.loc[d.index, a].median()), median_b=float(p.loc[d.index, b].median()),
                median_diff=float(d.median()), ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                p_one_sided=float(w.pvalue) if w else None)


def score():
    lk = json.loads((OUT / 'tflat_lock.json').read_text(encoding='utf-8'))
    assert lk['admission_sha256'] == sha(OUT / 'admission.csv'), 'admission table changed after the lock'
    a = admitted(); L = lst(); rows = []
    for r in a.itertuples():
        hid = int(r.Hylak_id)
        f = OUT / 'tflat' / str(hid)
        if not (f / f'{hid}_tflat.json').exists():
            rows.append(dict(Hylak_id=hid, model='T-flat', error='T-flat run failed')); continue
        for n, h in lk['runs'][str(hid)]['tflat'].items():
            assert sha(f / n) == h, f'{hid}: {n} changed after the lock'
        g = load(hid)
        F, H_flat, X, _ = regions(g, float(L.loc[hid, 'Pour_long']), float(L.loc[hid, 'Pour_lat']))
        zref = g['nasadem'] + r.offset_m
        r1c = reg.load_m0(gr.RUN / str(hid) / f'{hid}_C_tao_F.tif', g)
        tfl = reg.load_m0(f / f'{hid}_route1_tao_F.tif', g)
        ge = reg.load_m0(gr.RUN / str(hid) / f'{hid}_GLOBathy_est.tif', g)
        zG = g['dem'].copy(); P = np.isfinite(ge); zG[P] = ge[P]
        for model, z in [('R1-C', r1c), ('T-flat', tfl), ('GLOBathy-est', zG)]:
            rows.append(dict(Hylak_id=hid, model=model, H_flat_m=H_flat, X_km2=float(X.sum() * 9e-4), **metrics(z, zref, X, H_flat)))
        print(hid, {x['model']: round(x['V_PE_pct'], 1) for x in rows[-3:]}, flush=True)
    s = pd.DataFrame(rows).merge(a[['Hylak_id', 'Lake_name', 'Country', 'Continent', 'Poly_src', 'X_share_of_F', 'subset_90', 'forest_flag',
                                    'mean_ref_depth_m', 'offset_m', 'p_in', 'p_out']], on='Hylak_id')
    s.to_csv(OUT / 'scores.csv', index=False, float_format='%.6g')
    s = s[s.get('error').isna()] if 'error' in s else s
    s['absPE'] = s.V_PE_pct.abs()
    pv = s.pivot(index='Hylak_id', columns='model', values='absPE'); pr = s.pivot(index='Hylak_id', columns='model', values='bed_RMSE_m')
    meta = a.set_index('Hylak_id')
    rng = np.random.default_rng(SEED)
    allk = list(pv.dropna(subset=['R1-C', 'T-flat']).index)
    nonus = [k for k in allk if meta.loc[k, 'Country'] != 'United States of America']
    t = dict(H6=paired(pv, 'R1-C', 'T-flat', allk, rng), H6b=paired(pr, 'R1-C', 'T-flat', allk, rng),
             descriptive=dict(H6_outside_US=paired(pv, 'R1-C', 'T-flat', nonus, rng),
                              H6_subset_90=paired(pv, 'R1-C', 'T-flat', [k for k in allk if meta.loc[k, 'subset_90']], rng),
                              H6_without_forest=paired(pv, 'R1-C', 'T-flat', [k for k in allk if not meta.loc[k, 'forest_flag']], rng)))
    (OUT / 'tests.json').write_text(json.dumps(t, indent=1), encoding='utf-8')
    agg = s.groupby('model').agg(n=('Hylak_id', 'count'), median_abs_PE=('absPE', 'median'), mean_abs_PE=('absPE', 'mean'), median_PE=('V_PE_pct', 'median'),
                                 median_RMSE_m=('bed_RMSE_m', 'median'), median_bias_m=('bed_bias_m', 'median'), median_AE_NRMSE=('AE_NRMSE_pct', 'median'))
    agg.to_csv(OUT / 'aggregate.csv')
    tert = pd.qcut(meta.loc[allk, 'mean_ref_depth_m'], 3, labels=['shallow', 'mid', 'deep'])
    strata = pd.concat([s.assign(stratum='continent: ' + s.Continent), s.assign(stratum=np.where(s.forest_flag, 'forest flag', 'no forest flag')),
                        s.assign(stratum='depth: ' + s.Hylak_id.map(tert).astype(str)), s.assign(stratum=np.where(s.subset_90, 'X >= 90 %', 'X 70-90 %'))])
    strata.groupby(['stratum', 'model']).agg(n=('Hylak_id', 'count'), median_abs_PE=('absPE', 'median'), median_PE=('V_PE_pct', 'median'),
                                             median_RMSE_m=('bed_RMSE_m', 'median')).to_csv(OUT / 'strata.csv')
    print(agg.round(2).to_string()); print(json.dumps(t, indent=1))


def main():
    warnings.filterwarnings('ignore')
    p = argparse.ArgumentParser()
    for f in ['sample', 'merge', 'tflat', 'lock', 'score']:
        p.add_argument('--' + f, action='store_true')
    p.add_argument('--size', default='all'); p.add_argument('--part')
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.sample:
        sample(a.size, a.part)
    elif a.merge:
        merge()
    elif a.tflat:
        tflat(a.part)
    elif a.lock:
        lock()
    elif a.score:
        score()


if __name__ == '__main__':
    main()
