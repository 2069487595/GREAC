"""Summary and descriptive checks of the global R2 run (spec docs/plans/2026-10-10-r2-global-run-spec.md, sections 5 and 7).
Run AFTER the lock; nothing here changes an output of the run.

  --summary  global_summary_R2.csv (one row per reservoir: HydroLAKES attributes, levels, R2 and R1-C at H_top, dam-height
             reading, status and flags), global_aev_R2.csv.gz (curves of R2 and R1-C on common levels; below a method's
             lowest bed level its area and volume are zero; for fallbacks R2 = R1-C), global_failed_R2.csv
  --check    check_report_R2.json and check_evaluated_R2.csv:
             1 status counts against the input-only coverage table (r2_global_coverage_v1/coverage.csv)
             2 evaluated reservoirs in the global list: R2 of the global run at the official normal pool against the R2
               of the evaluation (r2_posthoc_diagnosis_v1/combined.csv). All of these results were seen before; this
               only checks whether the changed input rules of the global pipeline alter the accuracy
             3 V_top(R2) / V_top(R1-C) by water-level regime
             4 V_top / GRanD capacity for R2 and R1-C
             5 distribution of the R1-C recheck differences
Flags use input and output geometry only: flag_R2_equals_R1C (no positive dam-height reading), flag_R2_failed,
flag_D_dam_beyond_validated (reading above the largest reading among the evaluated reservoirs of the global list) and
flag_area_beyond_validated (HydroLAKES area above the largest HydroLAKES area among the evaluated reservoirs of the global
list; added 2026-10-13 at the user's decision. The error statistics of the paper come from smaller reservoirs; the flag
says nothing about the accuracy of the flagged reservoirs).
"""
from pathlib import Path
import sys, json, argparse, warnings

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd

ROOT = HERE.parents[1]
OUT = ROOT / '05_Results/by_date/2026-10-10/r2_global_v1'
RUN = OUT / 'run'
G_OUT = ROOT / '05_Results/by_date/2026-10-07/route1_global_v1'
COVER = ROOT / '05_Results/by_date/2026-10-08/r2_global_coverage_v1/coverage.csv'
EVAL = ROOT / '05_Results/by_date/2026-10-09/r2_posthoc_diagnosis_v1/combined.csv'
KEEP = ['Hylak_id', 'Lake_name', 'Country', 'Continent', 'Grand_id', 'Poly_src', 'HL_Lake_area_km2', 'HL_Vol_total_mcm', 'HL_Depth_avg_m', 'Pour_long',
        'Pour_lat', 'crosses_60N', 'epsg', 'W0_km2', 'polygon_km2', 'F_km2', 'R_km2', 'H_flat_m', 'H_seed_m', 'H_top_m', 'regime', 'V_top_R1C_km3',
        'A_top_R1C_km2', 'max_depth_R1C_m', 'share_below_plateau_R1C', 'alpha_R1C', 'flag_rule_C_fallback', 'flag_floor_applied', 'flag_H_top_below_seed',
        'H_top_rule', 'flag_R_touches_edge', 'flag_fabdem_gap', 'flag_W0_lt_half_polygon', 'missing_tiles', 'validated_region', 'F_over_W0', 'R_over_W0',
        'flag_F_W0_beyond_validated', 'flag_R_W0_beyond_validated', 'flag_poly_not_swbd']
q = lambda x: [float(v) for v in np.nanpercentile(np.asarray(x, float), [25, 50, 75])]


def evaluated_ids():
    import route1_global_check_v1 as chk
    ev = chk.evaluated().dropna(subset=['Hylak_id']); ev['Hylak_id'] = ev.Hylak_id.astype(int)
    return ev


def summary():
    assert (OUT / 'lock_manifest.json').exists(), 'run --lock first'
    s = pd.read_csv(G_OUT / 'global_summary.csv')[KEEP]
    rows, curves, fails = [], [], []
    for k, hid in enumerate(s.Hylak_id):
        f = RUN / str(hid); j = f / f'{hid}_R2_run.json'
        a0 = pd.read_csv(G_OUT / 'run' / str(hid) / f'{hid}_aev.csv')[['H_m', 'A_R1C_km2', 'V_R1C_km3']].dropna()
        same = True
        if j.exists():
            r = json.loads(j.read_text(encoding='utf-8')); m = r['R2']; same = bool(r['R2_equals_R1C'])
            rows.append(dict(Hylak_id=hid, R2_status=r['status'], B1000_m=r['B_R_m'], B2000_m=r['B_2R_m'], D_dam_m=r['D_dam_m'], V_top_R2_km3=m['V_top_km3'],
                             A_top_R2_km2=m['A_top_km2'], max_depth_R2_m=m['max_depth_m'], r1c_recheck_max_abs_diff_m=r.get('r1c_recheck_max_abs_diff_m'),
                             runtime_R2_s=r['runtime_s']['total']))
        else:
            e = json.loads((f / f'{hid}_R2_error.json').read_text(encoding='utf-8'))['error'] if (f / f'{hid}_R2_error.json').exists() else 'no output'
            fails.append(dict(Hylak_id=hid, error=e[:300])); rows.append(dict(Hylak_id=hid, R2_status='failed'))
        a = a0.rename(columns={'A_R1C_km2': 'A_R2_km2', 'V_R1C_km3': 'V_R2_km3'}) if same else pd.read_csv(f / f'{hid}_aev_R2.csv')
        c = a.merge(a0, on='H_m', how='outer').sort_values('H_m').fillna(0.0); c.insert(0, 'Hylak_id', hid); curves.append(c)
        if k % 500 == 0:
            print('read', k, flush=True)
    s = s.merge(pd.DataFrame(rows), on='Hylak_id')
    failed = s.R2_status.eq('failed')
    for col, unit in [('V_top', 'km3'), ('A_top', 'km2'), ('max_depth', 'm')]:      # failed: the product falls back to R1-C
        s.loc[failed, f'{col}_R2_{unit}'] = s.loc[failed, f'{col}_R1C_{unit}']
    s['flag_R2_failed'] = failed
    s['flag_R2_equals_R1C'] = s.R2_status.isin(['fallback_undefined', 'fallback_nonpositive', 'failed'])
    s['dam_reading_correction_m'] = (s.B2000_m - s.B1000_m).clip(lower=0)
    ev = evaluated_ids(); ok = s[s.Hylak_id.isin(ev.Hylak_id) & s.R2_status.eq('rescaled')]
    vr = dict(n_evaluated_rescaled=int(len(ok)), D_dam_min_m=float(ok.D_dam_m.min()), D_dam_max_m=float(ok.D_dam_m.max()))
    in_list = s[s.Hylak_id.isin(ev.Hylak_id)]
    vr.update(n_evaluated_in_list=int(len(in_list)), HL_area_min_km2=float(in_list.HL_Lake_area_km2.min()), HL_area_max_km2=float(in_list.HL_Lake_area_km2.max()))
    (OUT / 'validated_range_R2.json').write_text(json.dumps(vr, indent=1), encoding='utf-8')
    s['flag_D_dam_beyond_validated'] = s.D_dam_m > vr['D_dam_max_m']
    s['flag_area_beyond_validated'] = s.HL_Lake_area_km2 > vr['HL_area_max_km2']
    s['recommended'] = 'R2'
    s.to_csv(OUT / 'global_summary_R2.csv', index=False, float_format='%.6g')
    pd.DataFrame(fails, columns=['Hylak_id', 'error']).to_csv(OUT / 'global_failed_R2.csv', index=False)
    pd.concat(curves).to_csv(OUT / 'global_aev_R2.csv.gz', index=False, float_format='%.6g', compression='gzip')
    print('reservoirs', len(s)); print(s.R2_status.value_counts().to_string()); print(vr)
    print(s.filter(like='flag_').sum().to_string())


def product_pe(batch, key, hid):
    import aev_registry_v1 as reg, route1_tao_flat_v1 as r1, eval_expansion_score_v1 as sc, route1_global_run_v1 as gr
    g = gr.load(hid); F, _, _ = r1.flattened_region(g['dem'], g['W0'], g['swo1'])
    p = RUN / str(hid) / f'{hid}_bed_R2.tif'
    b = reg.load_m0(p if p.exists() else gr.RUN / str(hid) / f'{hid}_C_tao_F.tif', g)
    z = g['dem'].copy(); z[F] = b[F]
    cur, H_pool, _, _ = sc.official(batch, key)
    Vmax = float(np.interp(H_pool, cur.H, cur.volume_km3))
    return dict(H_pool_m=H_pool, Vmax_official_km3=Vmax, PE_R2_global=(float(gr.curves(z, F, g, F, H_pool).iloc[-1].V_swo1) / Vmax - 1) * 100)


def check():
    s = pd.read_csv(OUT / 'global_summary_R2.csv'); rep = dict(n=int(len(s)), status={k: int(v) for k, v in s.R2_status.value_counts().items()})
    c = pd.read_csv(COVER)[['Hylak_id', 'status', 'D_dam_m']].rename(columns={'status': 'cov_status', 'D_dam_m': 'cov_D'})
    m = s.merge(c, on='Hylak_id', how='left')
    expect = m.R2_status.map({'rescaled': 'ok', 'not_rescaled': 'ok', 'fallback_undefined': 'undefined', 'fallback_nonpositive': 'nonpositive'})
    both = m.D_dam_m.notna() & m.cov_D.notna()
    rep['coverage_table'] = dict(status_agrees=int((expect == m.cov_status).sum()), status_differs=int(((expect != m.cov_status) & m.R2_status.ne('failed')).sum()),
                                 max_abs_D_dam_difference_m=float((m.D_dam_m - m.cov_D)[both].abs().max()))
    ev = evaluated_ids(); ev = ev[ev.Hylak_id.isin(s.Hylak_id)]
    e = pd.read_csv(EVAL).set_index('key'); old = pd.read_csv(G_OUT / 'check_evaluated.csv').set_index('key')
    st = s.set_index('Hylak_id').R2_status; rows = []
    for r in ev.itertuples():
        try:
            o = dict(batch=r.batch, key=r.key, Hylak_id=r.Hylak_id, R2_status=st[r.Hylak_id], **product_pe(r.batch, r.key, r.Hylak_id))
            o['PE_R2_evaluation'] = float(e.loc[r.key, 'Vmax_PE_R2']) if r.key in e.index else np.nan
            o['PE_R1C_evaluation'] = float(e.loc[r.key, 'Vmax_PE_R1C']) if r.key in e.index else np.nan
            o['PE_R1C_global'] = float(old.loc[r.key, 'PE_R1-C_global']) if r.key in old.index else np.nan
            rows.append(o)
        except Exception as exc:
            rows.append(dict(batch=r.batch, key=r.key, Hylak_id=r.Hylak_id, error=repr(exc)[:200]))
        print(r.key, 'done', flush=True)
    t = pd.DataFrame(rows); t.to_csv(OUT / 'check_evaluated_R2.csv', index=False, float_format='%.6g')
    ok = t.dropna(subset=['PE_R2_global', 'PE_R2_evaluation']); ch = (ok.PE_R2_global - ok.PE_R2_evaluation).abs()
    rep['evaluated'] = dict(n_in_list=int(len(ev)), n_scored=int(len(ok)), median_abs_PE_R2_global=float(ok.PE_R2_global.abs().median()),
                            median_abs_PE_R2_evaluation=float(ok.PE_R2_evaluation.abs().median()), median_abs_change_pp=float(ch.median()),
                            n_change_gt_5pp=int((ch > 5).sum()), median_abs_PE_R1C_global=float(ok.PE_R1C_global.abs().median()),
                            median_abs_PE_R1C_evaluation=float(ok.PE_R1C_evaluation.abs().median()),
                            R2_better_than_R1C_global=int((ok.PE_R2_global.abs() < ok.PE_R1C_global.abs()).sum()),
                            R2_worse_than_R1C_global=int((ok.PE_R2_global.abs() > ok.PE_R1C_global.abs()).sum()))
    s['ratio'] = s.V_top_R2_km3 / s.V_top_R1C_km3; res = s[s.R2_status.eq('rescaled') & (s.V_top_R1C_km3 > 0)]
    rep['V_R2_over_V_R1C_rescaled'] = dict(all=dict(n=int(len(res)), q25_50_75=q(res.ratio), share_R2_larger=float((res.ratio > 1).mean())),
                                           **{k: dict(n=int(len(v)), q25_50_75=q(v.ratio), share_R2_larger=float((v.ratio > 1).mean())) for k, v in res.groupby('regime')})
    cap = s.HL_Vol_total_mcm / 1e3; inside = ~(s.flag_F_W0_beyond_validated | s.flag_R_W0_beyond_validated)
    rep['V_top_over_GRanD'] = {n: dict(all=q((s[f'V_top_{n}_km3'] / cap)), inside_evaluated_geometry=q((s[f'V_top_{n}_km3'] / cap)[inside]),
                                       **{k: q((v[f'V_top_{n}_km3'] / (v.HL_Vol_total_mcm / 1e3))) for k, v in s.groupby('regime')}) for n in ['R2', 'R1C']}
    rep['total_volume_inside_evaluated_geometry_km3'] = dict(n=int(inside.sum()), R2=float(s.V_top_R2_km3[inside].sum()), R1C=float(s.V_top_R1C_km3[inside].sum()),
                                                             GRanD=float(cap[inside].sum()))
    d = s.r1c_recheck_max_abs_diff_m.dropna()
    rep['r1c_recheck'] = dict(n=int(len(d)), max_m=float(d.max()), p99_m=float(d.quantile(.99)), n_above_1mm=int((d > 1e-3).sum()))
    ok2 = s[s.R2_status.eq('rescaled')]
    rep['reading'] = dict(D_dam_m_q05_25_50_75_95=[float(v) for v in ok2.D_dam_m.quantile([.05, .25, .5, .75, .95])], n_correction_applied=int((ok2.dam_reading_correction_m > 0).sum()),
                          n_beyond_validated=int(s.flag_D_dam_beyond_validated.sum()), max_depth_R2_over_R1C_median=float((ok2.max_depth_R2_m / ok2.max_depth_R1C_m).median()))
    rep['fallback_by_continent'] = {k: dict(n=int(len(v)), fallback=int(v.flag_R2_equals_R1C.sum())) for k, v in s.groupby('Continent')}
    rep['flags'] = {k: int(v) for k, v in s.filter(like='flag_').sum().items()}
    (OUT / 'check_report_R2.json').write_text(json.dumps(rep, indent=1), encoding='utf-8'); print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    p = argparse.ArgumentParser(); p.add_argument('--summary', action='store_true'); p.add_argument('--check', action='store_true'); a = p.parse_args()
    if a.summary:
        summary()
    if a.check:
        check()
