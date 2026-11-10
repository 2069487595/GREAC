"""Step 3: collect the global run into one table per reservoir plus all curves (spec section 5).

Reads run/<id>/<id>_run.json and _aev.csv (and _error.json for failures); joins HydroLAKES attributes from
global_list.csv. Writes global_summary.csv, global_failed.csv and global_aev.csv.gz in the run folder.
Usage: .venv/Scripts/python.exe 04_Code/scripts/route1_global_summary_v1.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '05_Results/by_date/2026-10-07/route1_global_v1'
RUN = OUT / 'run'


def row(r):
    i, m = r['inputs'], r['methods']
    out = dict(Hylak_id=r['Hylak_id'], epsg=r['epsg'], W0_km2=r['W0_km2'], polygon_km2=i['polygon_km2_on_grid'], F_km2=r['F_km2'],
               R_km2=r['R_km2'], H_flat_m=r['H_flat_m'], H_seed_m=r['H_seed_m'], H_top_m=r['H_top_m'],
               regime='F<=W0' if r['F_le_W0'] else 'F>W0', recommended=r['recommended'])
    for k in ['R1C', 'C2', 'Gest']:
        out.update({f'V_top_{k}_km3': m[k]['V_top_km3'], f'A_top_{k}_km2': m[k]['A_top_km2'], f'max_depth_{k}_m': m[k]['max_depth_m']})
    out.update(share_below_plateau_R1C=m['R1C'].get('share_below_plateau'), Dmax_est_GLOBathy_m=r['GLOBathy_est']['Dmax_est_PAVEW_m'],
               H_surf_GLOBathy_m=r['GLOBathy_est']['H_surf_m'], alpha_R1C=r['alpha_R1C'],
               flag_rule_C_fallback=r['rule_C_fallback'] is not None, flag_floor_applied=r['floor_applied'],
               flag_H_top_below_seed=r['H_top_below_seed'], H_top_rule=r['H_top']['rule'],
               flag_R_touches_edge=i['R_touches_edge'],
               flag_fabdem_gap=(i['fabdem_finite_in_R'] is not None and i['fabdem_finite_in_R'] < 0.99),
               flag_W0_lt_half_polygon=r['W0_km2'] < 0.5 * i['polygon_km2_on_grid'],
               missing_tiles=';'.join(i['missing_tiles']), runtime_s=r['runtime_s']['total'])
    return out


def validated_range():
    """Max F/W0 and R/W0 over the 136 evaluated reservoirs (locked step-1 runs and group-B runs)."""
    v = []
    for f in list((ROOT / '05_Results/by_date/2026-09-30/route1_expansion_run_v1').glob('*/*_run.json')) +              list((ROOT / '05_Results/by_date/2026-10-01/route1_hybrid_v1/test').glob('*/*_run.json')):
        j = json.loads(f.read_text(encoding='utf-8'))
        v.append((j['F_km2'] / j['W0_km2'], j['swo1_km2'] / j['W0_km2']))
    v = np.array(v)
    return dict(n=len(v), F_over_W0_max=float(v[:, 0].max()), R_over_W0_max=float(v[:, 1].max()))


def failure_class(hid, err):
    j = ROOT / '03_Data_Processed/global_route1_20261007/inputs' / f'{hid}_inputs.json'
    m = json.loads(j.read_text(encoding='utf-8')) if j.exists() else {}
    if 'no SWBD water' in err:
        return 'no SWBD (Feb 2000) water inside the polygon'
    if m.get('fabdem_finite_in_R') == 0:
        return 'no FABDEM (Copernicus GLO-30 gap: Armenia / Azerbaijan)'
    if 'grid border' in err:
        return f"F reaches the grid border at the {m.get('margin_m', 0) / 1e3:.0f} km margin cap"
    return 'other'


def main():
    lst = pd.read_csv(OUT / 'global_list.csv')
    rows, fails, curves = [], [], []
    for hid in lst.Hylak_id:
        f = RUN / str(hid)
        if (f / f'{hid}_run.json').exists():
            rows.append(row(json.loads((f / f'{hid}_run.json').read_text(encoding='utf-8'))))
            a = pd.read_csv(f / f'{hid}_aev.csv'); a.insert(0, 'Hylak_id', hid); curves.append(a)
        elif (f / f'{hid}_error.json').exists():
            e = json.loads((f / f'{hid}_error.json').read_text(encoding='utf-8'))['error']
            fails.append(dict(Hylak_id=hid, failure_class=failure_class(hid, e), error=e[:300]))
    s = lst[['Hylak_id', 'Lake_name', 'Country', 'Continent', 'Grand_id', 'Poly_src', 'Lake_area', 'Vol_total', 'Depth_avg',
             'Pour_long', 'Pour_lat', 'crosses_60N']].rename(columns={'Lake_area': 'HL_Lake_area_km2', 'Vol_total': 'HL_Vol_total_mcm',
                                                                       'Depth_avg': 'HL_Depth_avg_m'})
    s = s.merge(pd.DataFrame(rows), on='Hylak_id')
    s['validated_region'] = s.Country.eq('United States of America')
    vr = validated_range()  # QA flags added after the run (spec section 8): outside the evaluated F/W0 or R/W0 range
    s['F_over_W0'] = s.F_km2 / s.W0_km2; s['R_over_W0'] = s.R_km2 / s.W0_km2
    s['flag_F_W0_beyond_validated'] = s.F_over_W0 > vr['F_over_W0_max']
    s['flag_R_W0_beyond_validated'] = s.R_over_W0 > vr['R_over_W0_max']
    (OUT / 'validated_range.json').write_text(json.dumps(vr, indent=1), encoding='utf-8')
    s['flag_poly_not_swbd'] = s.Poly_src.ne('SWBD')
    # C2 switches to GLOBathy-est only when F > W0; that switch was validated only on SWBD polygons (all 129 evaluated
    # reservoirs in the list) with F/W0 <= the evaluated maximum. Outside that domain the README points to R1-C.
    s['C2_outside_validated_domain'] = s.regime.eq('F>W0') & (s.flag_poly_not_swbd | s.flag_F_W0_beyond_validated)
    # Recommended product (user decision 2026-10-08): C2, except R1-C where C2 is outside its validated domain.
    # The locked run.json files keep 'C2' for every reservoir; the recommendation is defined here, in the summary.
    use_r1c = s.C2_outside_validated_domain
    s['recommended'] = np.where(use_r1c, 'R1-C', 'C2')
    for col in ['V_top', 'A_top', 'max_depth']:
        unit = {'V_top': 'km3', 'A_top': 'km2', 'max_depth': 'm'}[col]
        s[f'{col}_recommended_{unit}'] = np.where(use_r1c, s[f'{col}_R1C_{unit}'], s[f'{col}_C2_{unit}'])
    s.to_csv(OUT / 'global_summary.csv', index=False, float_format='%.6g')
    print('C2 outside validated domain', int(s.C2_outside_validated_domain.sum()))
    fl = pd.DataFrame(fails, columns=['Hylak_id', 'failure_class', 'error'])
    fl = fl.merge(lst[['Hylak_id', 'Lake_name', 'Country', 'Poly_src', 'Lake_area']], on='Hylak_id', how='left')
    fl.to_csv(OUT / 'global_failed.csv', index=False)
    print(fl.failure_class.value_counts().to_string())
    if curves:
        pd.concat(curves).to_csv(OUT / 'global_aev.csv.gz', index=False, float_format='%.6g', compression='gzip')
    print('ok', len(s), 'failed', len(fails), 'of', len(lst))
    print(s.regime.value_counts().to_string())
    print(s.filter(like='flag_').sum().to_string())


if __name__ == '__main__':
    main()
