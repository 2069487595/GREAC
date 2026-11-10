"""Step 4 of the frozen plan docs/plans/2026-10-22-new-data-independent-test-plan.md (sections 4, 5 and 13): score the
locked global products on the admitted new reservoirs ONCE.

  --selftest  scores a few evaluated reservoirs with the same functions and their evaluation references
              (eval_expansion_score_v1.official) and must reproduce the global-pipeline errors already published in
              check_evaluated_R2.csv (TC-dam) and route1_global_v1/check_evaluated.csv (TC-cap, switching). Touches no new data.
  --score     verifies the lock (newdata_lock_v1/lock_manifest.json: every product file, reference curve, admission table,
              plan and script hash; the lock and this script must be committed), then for each admitted reservoir and method
              puts the model bed into its own region (TC-dam: <id>_bed_R2.tif on F, TC-cap on F, GLOBathy-est on its polygon,
              switching = route1_hybrid_v1.surfaces 'C2') and integrates inside R up to the official normal pool in EGM2008
              (route1_global_run_v1.curves), as the global reproduction check did. Writes per_reservoir.csv, tests.json,
              descriptive.json.
Tests (plan section 4): N1 |error| TC-dam < TC-cap, one-sided paired Wilcoxon, all admitted; N1-US and N1-BR the same,
Holm-adjusted together; N2 TC-dam vs GLOBathy-est, two-sided, without reservoirs where GLOBathy uses an observed
maximum depth; N3 TC-dam vs switching, two-sided. Wins, losses, ties and a bootstrap 95 % interval of the median paired
difference (10,000 resamples, seed 20261022). Zero differences (no dam-height reading: TC-dam = TC-cap) stay in the
sample and are dropped by the Wilcoxon statistic, as in the final test.
Usage: .venv/Scripts/python.exe -I 04_Code/scripts/newdata_score_v1.py --selftest | --score
"""
from pathlib import Path
import sys, json, argparse, warnings, hashlib, subprocess, datetime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from scipy import stats
import aev_registry_v1 as reg
import aev_observation_share_v1 as aev
import route1_tao_flat_v1 as r1
import route1_hybrid_v1 as hy
import route1_global_run_v1 as gr

ROOT = HERE.parents[1]
R1 = ROOT / '05_Results/by_date/2026-10-07/route1_global_v1'
R2 = ROOT / '05_Results/by_date/2026-10-10/r2_global_v1'
LOCK = ROOT / '05_Results/by_date/2026-10-23/newdata_lock_v1/lock_manifest.json'
REF = ROOT / '05_Results/by_date/2026-10-23/newdata_reference_v1'
ADM = ROOT / '05_Results/by_date/2026-10-22/newdata_admission_v2/admission.csv'
EVAL70 = ROOT / '05_Results/by_date/2026-10-11/size_strata_posthoc_v1/per_reservoir.csv'
OUT = ROOT / '05_Results/by_date/2026-10-23/newdata_score_v1'
SEED = 20261022
METHODS = ['TC-dam', 'TC-cap', 'GLOBathy-est', 'switching']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def surfaces(hid):
    """The four locked product surfaces of one reservoir on its global grid."""
    run = json.loads((R1 / 'run' / str(hid) / f'{hid}_run.json').read_text(encoding='utf-8'))
    g = gr.load(hid)
    F, H_flat, _ = r1.flattened_region(g['dem'], g['W0'], g['swo1'])
    r1c = reg.load_m0(R1 / 'run' / str(hid) / f'{hid}_C_tao_F.tif', g)
    gest = reg.load_m0(R1 / 'run' / str(hid) / f'{hid}_GLOBathy_est.tif', g)
    surf, sinfo = hy.surfaces(g, F, run['H_seed_m'], r1c, gest, run['GLOBathy_est'])
    p = R2 / 'run' / str(hid) / f'{hid}_bed_R2.tif'
    b = reg.load_m0(p if p.exists() else R1 / 'run' / str(hid) / f'{hid}_C_tao_F.tif', g)
    zD = g['dem'].copy(); zD[F] = b[F]
    out = {'TC-dam': (zD, F), 'TC-cap': surf['R1-C'], 'GLOBathy-est': surf['GLOBathy-est'], 'switching': surf['C2']}
    return g, F, H_flat, out, dict(F_le_W0=bool(sinfo['F_le_W0']), H_seed_m=run['H_seed_m'], H_top_m=run['H_top_m'],
                                   GLOBathy_uses_observed_Dmax=bool(run['GLOBathy_est'].get('uses_observed_Dmax')))


def score_one(hid, cur, H_pool, full_curve):
    """cur: official curve with H (EGM2008 m), area_km2, volume_km3. Returns one row per method."""
    g, F, H_flat, surf, info = surfaces(hid)
    Vmax = float(np.interp(H_pool, cur.H, cur.volume_km3)); A_pool = float(np.interp(H_pool, cur.H, cur.area_km2))
    rows = []
    for name in METHODS:
        z, region = surf[name]
        tab = gr.curves(z, region, g, F, H_pool)
        t = tab.iloc[-1]
        row = dict(method=name, Vmax_official_km3=Vmax, A_pool_official_km2=A_pool, Vmax_model_km3=float(t.V_swo1), A_pool_model_km2=float(t.A_swo1),
                   Vmax_PE_pct=(float(t.V_swo1) / Vmax - 1) * 100, H_pool_m=H_pool, H_flat_m=H_flat, **info)
        if full_curve:
            sel = (cur.H <= H_pool + 1e-6) & (cur.H >= tab.H.min())
            ae = aev.fit_stats(np.interp(cur.H[sel], tab.H, tab.A_swo1), cur.area_km2[sel].values)
            ve = aev.fit_stats(np.interp(cur.H[sel], tab.H, tab.V_swo1), cur.volume_km3[sel].values)
            row.update(AE_NRMSE_pct=ae.get('NRMSE_pct'), VE_NRMSE_pct=ve.get('NRMSE_pct'))
            if cur.H.min() <= H_flat < H_pool:
                Vb = float(np.interp(H_flat, cur.H, cur.volume_km3))
                row['below_PE_pct'] = (float(np.interp(H_flat, tab.H, tab.V_swo1)) / Vb - 1) * 100 if Vb > 0 else np.nan
                row['share_below_flat_official'] = Vb / Vmax
        rows.append(row)
    return rows


# ---------------------------------------------------------------- self-test on evaluated reservoirs (no new data)
def selftest():
    import eval_expansion_score_v1 as sc
    ev = pd.read_csv(R2 / 'check_evaluated_R2.csv').dropna(subset=['PE_R2_global'])
    old = pd.read_csv(R1 / 'check_evaluated.csv').set_index('key')
    picks = pd.concat([ev[ev.batch == b].head(2) for b in ['TWDB', 'USBR-A', 'USBR-B']])
    worst = 0.0
    for r in picks.itertuples():
        cur, H_pool, _, _ = sc.official('USBR-B' if r.batch == 'USBR-B' else r.batch, r.key)
        rows = {x['method']: x['Vmax_PE_pct'] for x in score_one(int(r.Hylak_id), cur, H_pool, True)}
        d = [abs(rows['TC-dam'] - r.PE_R2_global), abs(rows['TC-cap'] - old.loc[r.key, 'PE_R1-C_global']), abs(rows['switching'] - old.loc[r.key, 'PE_C2_global'])]
        worst = max(worst, *d)
        print(r.key, 'TC-dam', round(rows['TC-dam'], 4), 'vs', round(r.PE_R2_global, 4), '| TC-cap', round(rows['TC-cap'], 4), 'vs',
              round(old.loc[r.key, 'PE_R1-C_global'], 4), '| switching', round(rows['switching'], 4), 'vs', round(old.loc[r.key, 'PE_C2_global'], 4))
    print('SELFTEST max abs difference (percentage points):', worst)
    if worst > 1e-3:
        raise SystemExit('SELFTEST FAILED')


# ---------------------------------------------------------------- statistics
def boot(dd, n=10000):
    rng = np.random.default_rng(SEED); idx = rng.integers(0, len(dd), (n, len(dd)))
    m = np.median(dd[idx], axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def paired(x, y, alternative):
    x, y = np.abs(np.asarray(x, float)), np.abs(np.asarray(y, float)); dd = x - y
    w = stats.wilcoxon(x, y, alternative=alternative) if np.any(dd != 0) else None
    return dict(n=int(len(dd)), alternative=alternative, statistic=float(w.statistic) if w else None, p=float(w.pvalue) if w else None,
                wins=int((dd < 0).sum()), losses=int((dd > 0).sum()), ties=int((dd == 0).sum()), median_abs_a=float(np.median(x)),
                median_abs_b=float(np.median(y)), median_diff=float(np.median(dd)), median_diff_ci95=boot(dd))


def describe(s):
    a = s.Vmax_PE_pct
    return dict(n=int(len(s)), median_abs=float(a.abs().median()), mean_abs=float(a.abs().mean()), median_signed=float(a.median()),
                share_over50=float((a > 50).mean()), share_under50=float((a < -50).mean()), share_within20=float((a.abs() <= 20).mean()))


# ---------------------------------------------------------------- scoring
def verify_lock():
    for p in [LOCK, Path(__file__)]:
        rel = str(p.relative_to(ROOT)).replace(chr(92), '/')
        if subprocess.run(['git', 'ls-files', '--error-unmatch', rel], cwd=ROOT, capture_output=True).returncode != 0:
            raise SystemExit(f'{rel} is not committed')
        if subprocess.run(['git', 'status', '--porcelain', '--', rel], cwd=ROOT, capture_output=True, text=True).stdout.strip():
            raise SystemExit(f'{rel} differs from the committed version')
    man = json.loads(LOCK.read_text(encoding='utf-8'))
    for cand, r in man['runs'].items():
        for rel, h in r['files'].items():
            base = R1 if rel.startswith('route1_global_v1') else R2
            if sha(base / rel.split('/', 1)[1]) != h:
                raise SystemExit(f'locked product file changed: {rel}')
        if sha(gr.INPUTS / f"{r['Hylak_id']}_inputs.npz") != r['inputs_sha256']:
            raise SystemExit(f'input grids changed: {cand}')
    for cand, c in man['reference_curves'].items():
        if sha(REF / 'curves' / c['file']) != c['sha256']:
            raise SystemExit(f'reference curve changed: {cand}')
    if sha(ADM) != man['admission_sha256'] or sha(REF / 'reference_index.csv') != man['reference_index_sha256']:
        raise SystemExit('admission table or reference index changed')
    if sha(ROOT / man['plan']) != man['plan_sha256']:
        raise SystemExit('plan changed after the lock')
    return man


def score():
    man = verify_lock(); print('lock verified:', man['n'], 'reservoirs')
    OUT.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(REF / 'reference_index.csv').set_index('candidate')
    gs = pd.read_csv(R2 / 'global_summary_R2.csv').set_index('Hylak_id')
    rows = []
    for cand, r in man['runs'].items():
        x = idx.loc[cand]
        c = pd.read_csv(REF / 'curves' / man['reference_curves'][cand]['file'])
        cur = pd.DataFrame(dict(H=c.H_egm2008_m, area_km2=c.area_km2, volume_km3=c.volume_km3))
        full = x.source != 'OWRB'
        if not full:      # OWRB: the transcribed normal-pool point only
            cur = pd.DataFrame(dict(H=[x.H_pool_egm2008_m], area_km2=[x.A_pool_official_km2], volume_km3=[x.Vmax_official_km3]))
            cur = pd.concat([cur.assign(H=cur.H - 1e-6), cur])      # a two-point stub so that interpolation returns the point
        hid = int(r['Hylak_id']); s = gs.loc[hid]
        for row in score_one(hid, cur, float(x.H_pool_egm2008_m), full):
            rows.append(dict(candidate=cand, source=x.source, region='Brazil' if x.source == 'Brazil ANA' else 'US', Hylak_id=hid,
                             R2_status=r['R2_status'], HL_area_km2=s.HL_Lake_area_km2, D_dam_m=s.D_dam_m, admission_flags=x.flags,
                             large=bool(s.flag_area_beyond_validated) or (x.A_pool_official_km2 > 172), flag_area_beyond_validated=bool(s.flag_area_beyond_validated),
                             flag_poly_not_swbd=bool(s.flag_poly_not_swbd), flag_F_W0_beyond_validated=bool(s.flag_F_W0_beyond_validated),
                             flag_R_W0_beyond_validated=bool(s.flag_R_W0_beyond_validated), flag_D_dam_beyond_validated=bool(s.flag_D_dam_beyond_validated),
                             flag_no_dam_reading=r['R2_status'] != 'rescaled', **row))
        print(cand, ' '.join(f"{q['method']} {q['Vmax_PE_pct']:+.1f}" for q in rows[-4:]), flush=True)
    s = pd.DataFrame(rows); s.to_csv(OUT / 'per_reservoir.csv', index=False, float_format='%.6g', encoding='utf-8-sig')
    pv = s.pivot(index='candidate', columns='method', values='Vmax_PE_pct'); meta = s[s.method == 'TC-dam'].set_index('candidate')
    us, br = meta.index[meta.region == 'US'], meta.index[meta.region == 'Brazil']
    t = dict(N1=paired(pv['TC-dam'], pv['TC-cap'], 'less'), N1_US=paired(pv.loc[us, 'TC-dam'], pv.loc[us, 'TC-cap'], 'less'),
             N1_BR=paired(pv.loc[br, 'TC-dam'], pv.loc[br, 'TC-cap'], 'less'))
    ps = [t['N1_US']['p'], t['N1_BR']['p']]; order = np.argsort(ps)
    holm = [None, None]
    for k, i in enumerate(order):
        holm[i] = min(1.0, max(ps[i] * (2 - k), holm[order[k - 1]] if k else 0.0))
    t['N1_US']['p_holm'], t['N1_BR']['p_holm'] = holm
    n2 = meta.index[~meta.GLOBathy_uses_observed_Dmax]
    t['N2'] = paired(pv.loc[n2, 'TC-dam'], pv.loc[n2, 'GLOBathy-est'], 'two-sided')
    t['N3'] = paired(pv['TC-dam'], pv['switching'], 'two-sided')
    t['alpha'] = 0.05; t['scored_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (OUT / 'tests.json').write_text(json.dumps(t, indent=1), encoding='utf-8')
    # descriptive (plan section 4)
    d = {'overall': {m: describe(s[s.method == m]) for m in METHODS},
         'by_region': {reg_: {m: describe(s[(s.method == m) & (s.region == reg_)]) for m in METHODS} for reg_ in ['US', 'Brazil']},
         'by_source': {src: {m: describe(s[(s.method == m) & (s.source == src)]) for m in METHODS} for src in sorted(s.source.unique())}}
    e70 = pd.read_csv(EVAL70); e70 = e70[e70.set.isin(['validation', 'test'])]
    d['generalisation'] = dict(new_TC_dam_median_abs=float(pv['TC-dam'].abs().median()), new_ci95=boot(pv['TC-dam'].abs().values),
                               evaluated70_TC_dam_median_abs=float(e70.R2.abs().median()), evaluated70_n=int(len(e70)))
    big = meta.index[meta.large]
    d['large'] = dict(n=int(len(big)), **{m: describe(s[(s.method == m) & s.candidate.isin(big)]) for m in METHODS},
                      TC_dam_vs_TC_cap=paired(pv.loc[big, 'TC-dam'], pv.loc[big, 'TC-cap'], 'two-sided') if len(big) > 2 else None,
                      not_large={m: describe(s[(s.method == m) & ~s.candidate.isin(big)]) for m in METHODS})
    flags = ['flag_area_beyond_validated', 'flag_poly_not_swbd', 'flag_F_W0_beyond_validated', 'flag_R_W0_beyond_validated', 'flag_D_dam_beyond_validated', 'flag_no_dam_reading']
    d['flags'] = {f: {str(v): dict(n=int(len(k)), median_abs_TC_dam=float(pv.loc[k, 'TC-dam'].abs().median()), median_abs_TC_cap=float(pv.loc[k, 'TC-cap'].abs().median()))
                      for v, k in meta.groupby(f).groups.items()} for f in flags}
    d['F_le_W0'] = {str(v): dict(n=int(len(k)), **{m: float(pv.loc[k, m].abs().median()) for m in METHODS}, TC_dam_wins_over_TC_cap=int((pv.loc[k, 'TC-dam'].abs() < pv.loc[k, 'TC-cap'].abs()).sum()))
                    for v, k in meta.groupby('F_le_W0').groups.items()}
    fc = s[s.source != 'OWRB']
    d['curve_metrics_without_OWRB'] = {m: dict(n=int((fc.method == m).sum()), below_PE_median=float(fc[fc.method == m].below_PE_pct.median()),
                                               AE_NRMSE_median=float(fc[fc.method == m].AE_NRMSE_pct.median()), VE_NRMSE_median=float(fc[fc.method == m].VE_NRMSE_pct.median()))
                                       for m in METHODS}
    (OUT / 'descriptive.json').write_text(json.dumps(d, indent=1), encoding='utf-8')
    print(json.dumps(t, indent=1)); print(json.dumps(d['overall'], indent=1))


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    p = argparse.ArgumentParser(); p.add_argument('--selftest', action='store_true'); p.add_argument('--score', action='store_true'); a = p.parse_args()
    if a.selftest:
        selftest()
    elif a.score:
        score()
