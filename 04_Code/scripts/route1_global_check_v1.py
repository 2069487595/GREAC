"""Step 3 checks (spec section 6, descriptive, run after the lock): global products vs the evaluated reservoirs and GRanD.

1. Evaluated reservoirs in the global list (Hylak_id as recorded by the locked step-1 / step-2 runs: GLOBathy Hylak_id in
   route1_expansion_run_v1/<key>/<key>_run.json, GLOBathy_est Hylak_id in route1_hybrid_v1/test/<key>/<key>_hybrid.json):
   R1-C and C2 surfaces of the global run (rebuilt from the run's own rasters with route1_hybrid_v1.surfaces) give V at the
   official normal pool (eval_expansion_score_v1.official), compared with the locked step-2 scores (dev/summary_dev.csv,
   test_score/summary.csv) of the same reservoirs.
2. V(H_top) / HydroLAKES Vol_total (GRanD capacity, mcm -> km3) and A(H_top) / Lake_area, by regime and continent.
Writes check_evaluated.csv, check_grand.csv, check_report.json and fig_global.png in the run folder.
Usage: .venv/Scripts/python.exe 04_Code/scripts/route1_global_check_v1.py
"""
from pathlib import Path
import sys, json, warnings

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import aev_registry_v1 as reg
import aev_observation_share_v1 as aev
import route1_tao_flat_v1 as r1
import route1_hybrid_v1 as hy
import eval_expansion_score_v1 as sc
import route1_global_run_v1 as gr

OUT = gr.OUT
STEP1 = reg.ROOT / '05_Results/by_date/2026-09-30/route1_expansion_run_v1'
HYB = reg.ROOT / '05_Results/by_date/2026-10-01/route1_hybrid_v1'


def evaluated():
    rows = []
    for f in STEP1.glob('*/*_run.json'):
        r = json.loads(f.read_text(encoding='utf-8'))
        rows.append(dict(batch=r['batch'], key=r['key'], Hylak_id=(r.get('GLOBathy') or {}).get('Hylak_id')))
    for f in (HYB / 'test').glob('*/*_hybrid.json'):
        r = json.loads(f.read_text(encoding='utf-8'))
        rows.append(dict(batch='USBR-B', key=f.parent.name, Hylak_id=r['GLOBathy_est']['Hylak_id']))
    return pd.DataFrame(rows)


def locked_pe():
    d = pd.read_csv(HYB / 'dev/summary_dev.csv')[['key', 'model', 'Vmax_PE_pct']]
    t = pd.read_csv(HYB / 'test_score/summary.csv')[['key', 'model', 'Vmax_PE_pct']]
    return pd.concat([d, t]).pivot_table(index='key', columns='model', values='Vmax_PE_pct')


def product_pe(batch, key, hid):
    run = json.loads((gr.RUN / str(hid) / f'{hid}_run.json').read_text(encoding='utf-8'))
    g = gr.load(hid)
    F, H_flat, _ = r1.flattened_region(g['dem'], g['W0'], g['swo1'])
    r1c = reg.load_m0(gr.RUN / str(hid) / f'{hid}_C_tao_F.tif', g)
    gest = reg.load_m0(gr.RUN / str(hid) / f'{hid}_GLOBathy_est.tif', g)
    surf, sinfo = hy.surfaces(g, F, run['H_seed_m'], r1c, gest, run['GLOBathy_est'])
    cur, H_pool, off, f = sc.official('USBR-B' if batch == 'USBR-B' else batch, key)
    Vmax = float(np.interp(H_pool, cur.H, cur.volume_km3))
    out = dict(batch=batch, key=key, Hylak_id=hid, F_le_W0_global=sinfo['F_le_W0'], H_pool_m=H_pool, H_top_m=run['H_top_m'],
               Vmax_official_km3=Vmax)
    for name in ['R1-C', 'C2']:
        z, region = surf[name]
        t = gr.curves(z, region, g, F, H_pool).iloc[-1]
        out[f'PE_{name}_global'] = (float(t.V_swo1) / Vmax - 1) * 100
    out['Vtop_over_Vpool_C2'] = run['methods']['C2']['V_top_km3'] / Vmax
    return out


def main():
    warnings.filterwarnings('ignore')
    s = pd.read_csv(OUT / 'global_summary.csv')
    ev = evaluated().dropna(subset=['Hylak_id'])
    ev['Hylak_id'] = ev.Hylak_id.astype(int)
    ev = ev[ev.Hylak_id.isin(s.Hylak_id)]
    lk = locked_pe()
    rows = []
    for r in ev.itertuples():
        try:
            o = product_pe(r.batch, r.key, r.Hylak_id)
            for m in ['R1-C', 'C2']:
                o[f'PE_{m}_locked'] = lk.loc[r.key, m] if r.key in lk.index else np.nan
            rows.append(o)
        except Exception as exc:
            rows.append(dict(batch=r.batch, key=r.key, Hylak_id=r.Hylak_id, error=repr(exc)[:200]))
    c = pd.DataFrame(rows)
    c.to_csv(OUT / 'check_evaluated.csv', index=False, float_format='%.6g')
    rep = dict(n_evaluated_in_list=int(len(ev)), n_scored=int(c.get('PE_C2_global', pd.Series(dtype=float)).notna().sum()))
    ok = c.dropna(subset=['PE_C2_global', 'PE_C2_locked'])
    for m in ['R1-C', 'C2']:
        rep[m] = dict(median_abs_PE_global=float(ok[f'PE_{m}_global'].abs().median()), median_abs_PE_locked=float(ok[f'PE_{m}_locked'].abs().median()),
                      median_abs_change_pp=float((ok[f'PE_{m}_global'] - ok[f'PE_{m}_locked']).abs().median()),
                      n_change_gt_5pp=int(((ok[f'PE_{m}_global'] - ok[f'PE_{m}_locked']).abs() > 5).sum()))
    rep['median_Vtop_over_Vpool_C2'] = float(ok.Vtop_over_Vpool_C2.median())
    s['Vtop_over_GRanD_C2'] = s.V_top_C2_km3 / (s.HL_Vol_total_mcm / 1e3)
    s['Vtop_over_GRanD_R1C'] = s.V_top_R1C_km3 / (s.HL_Vol_total_mcm / 1e3)
    s['Atop_over_HL_area'] = s.A_top_C2_km2 / s.HL_Lake_area_km2
    q = lambda x: [float(v) for v in np.nanpercentile(x, [25, 50, 75])]
    rep['GRanD_ratio_C2_q25_50_75'] = {k: dict(n=int(len(v)), q=q(v.Vtop_over_GRanD_C2)) for k, v in s.groupby('regime')}
    rep['GRanD_ratio_C2_by_continent'] = {k: dict(n=int(len(v)), q=q(v.Vtop_over_GRanD_C2)) for k, v in s.groupby('Continent')}
    inside = ~(s.flag_F_W0_beyond_validated | s.flag_R_W0_beyond_validated)
    rep['GRanD_ratio_inside_validated_range'] = {m: {k: dict(n=int(len(v)), q=q(v[f'Vtop_over_GRanD_{m}'])) for k, v in s[inside].groupby('regime')}
                                                 for m in ['R1C', 'C2']}
    rep['GRanD_ratio_beyond_validated_range'] = {m: dict(n=int((~inside).sum()), q=q(s.loc[~inside, f'Vtop_over_GRanD_{m}'])) for m in ['R1C', 'C2']}
    dom = ~s.C2_outside_validated_domain
    rep['C2_over_R1C_F_gt_W0'] = dict(inside_domain=dict(n=int((dom & (s.regime == 'F>W0')).sum()),
                                                         q=q((s.V_top_C2_km3 / s.V_top_R1C_km3)[dom & (s.regime == 'F>W0')])),
                                      outside_domain=dict(n=int((~dom).sum()), q=q((s.V_top_C2_km3 / s.V_top_R1C_km3)[~dom])))
    rep['recommended'] = dict(counts={k: int(v) for k, v in s.recommended.value_counts().items()},
                              GRanD_ratio_q25_50_75=q(s.V_top_recommended_km3 / (s.HL_Vol_total_mcm / 1e3)),
                              GRanD_ratio_by_regime={k: q(v.V_top_recommended_km3 / (v.HL_Vol_total_mcm / 1e3)) for k, v in s.groupby('regime')})
    rep['area_ratio_q25_50_75'] = q(s.Atop_over_HL_area)
    rep['area_ratio_inside_validated_range_q25_50_75'] = q(s.loc[inside, 'Atop_over_HL_area'])
    s[['Hylak_id', 'Continent', 'regime', 'HL_Vol_total_mcm', 'V_top_R1C_km3', 'V_top_C2_km3', 'Vtop_over_GRanD_R1C', 'Vtop_over_GRanD_C2',
       'HL_Lake_area_km2', 'A_top_C2_km2', 'Atop_over_HL_area']].to_csv(OUT / 'check_grand.csv', index=False, float_format='%.6g')
    (OUT / 'check_report.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(json.dumps(rep, indent=1))
    plot(s, ok)


def plot(s, ok):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for reg_, col in [('F<=W0', '#1f77b4'), ('F>W0', '#ff7f0e')]:
        k = s[s.regime == reg_]
        ax[0].scatter(k.Pour_long, k.Pour_lat, s=np.clip(np.sqrt(k.HL_Lake_area_km2), 1, 40), c=col, alpha=.5, lw=0, label=f'{reg_} (n={len(k)})')
    ax[0].set(xlim=(-180, 180), ylim=(-57, 61), title='Global run: reservoirs by FABDEM-era water level', xlabel='lon', ylabel='lat')
    ax[0].legend(loc='lower left', fontsize=8)
    for reg_, col in [('F<=W0', '#1f77b4'), ('F>W0', '#ff7f0e')]:
        v = s.loc[s.regime == reg_, 'Vtop_over_GRanD_C2'].replace([np.inf, -np.inf], np.nan).dropna()
        ax[1].hist(np.log10(v[v > 0]), bins=60, alpha=.6, color=col, label=f'{reg_} median {v.median():.2f}')
    ax[1].axvline(0, color='k', lw=.8)
    ax[1].set(xlabel='log10( V(H_top) C2 / GRanD capacity )', ylabel='reservoirs', title='Storage at SWO maximum level vs GRanD capacity')
    ax[1].legend(fontsize=8)
    for m, col in [('R1-C', '#1f77b4'), ('C2', '#2ca02c')]:
        ax[2].scatter(ok[f'PE_{m}_locked'], ok[f'PE_{m}_global'], s=14, alpha=.7, color=col, label=m)
    lim = [-100, 200]
    ax[2].plot(lim, lim, 'k--', lw=.8)
    ax[2].set(xlim=lim, ylim=lim, xlabel='Vmax error, locked evaluation run (%)', ylabel='Vmax error, global run (%)',
              title=f'Evaluated reservoirs in the global run (n={len(ok)})')
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_global.png', dpi=150)


if __name__ == '__main__':
    main()
