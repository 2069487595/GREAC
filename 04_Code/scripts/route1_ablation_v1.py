"""Post-hoc ablation of Route 1 (R1-C): which part of the seed-level rule carries the gain. NOT a method change.

Written after the test set was scored (route1_test_v1, commit 4e36635aa); the frozen method stays R1-C.
Same F, same kept DEM, same frozen Tao code (route1_seed_level_v1.run_tao); only the cap level H changes:
  cap_C      : rule C (SWO occurrence-elevation), = R1-C          (test: route1_test_v1; dev: route1_seed_level_v1)
  cap_A      : rule A (GEV of shoreline heights), floor H_flat    (dev: route1_seed_level_v1)
  cap_flat   : H_flat, the FABDEM plateau itself                  (dev: route1_tao_flat_v2)
  no_cap     : shoreline seeds uncapped, h0 = H_flat = T-flat     (test: route1_test_v1; dev: route1_tao_flat_v1)
  cap_oracle : official level whose area equals the area of F (reads the official curve, diagnostic ceiling:
               what a perfect water level would give with the same Tao)
Runs only what does not exist yet: test cap_A, cap_flat, cap_oracle; dev cap_oracle. Existing runs are re-scored.
Usage: .venv/Scripts/python.exe 04_Code/scripts/route1_ablation_v1.py
"""
from pathlib import Path
import sys, json, datetime, warnings

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import aev_registry_v1 as reg
import aev_observation_share_v1 as aev
import route1_tao_flat_v1 as r1
import route1_seed_level_v1 as sl
import usbr_test_registry_v1 as tr
import usbr_official_curves_v1 as oc
import route1_test_score_v1 as ts

OUT = reg.ROOT / '05_Results/by_date/2026-09-29/route1_ablation_v1'
BY = reg.ROOT / '05_Results/by_date/2026-09-29'
EXISTING = {  # variant -> (dev path pattern, test path pattern)
    'cap_C': (BY / 'route1_seed_level_v1/{k}_C_tao_F.tif', BY / 'route1_test_v1/{k}/{k}_C_tao_F.tif'),
    'cap_A': (BY / 'route1_seed_level_v1/{k}_A_tao_F.tif', None),
    'cap_flat': (BY / 'route1_tao_flat_v2/{k}_route1_tao_F.tif', None),
    'no_cap': (BY / 'route1_tao_flat_v1/{k}_route1_tao_F.tif', BY / 'route1_test_v1/{k}/{k}_route1_tao_F.tif'),
}


def setup(key):
    """Grid, F, H_flat, official curve (EGM2008), normal pool and survey cells for a dev or test key."""
    if key.startswith('T'):
        g = tr.load_test(key); off = tr.egm_offset_m(key); cur = tr.official_curve(key, off)
        H_pool = tr.to_egm(oc.POOL[key][0], off)
        rr, cc, zr, _ = ts.sonar_cells(key, g, off)
    else:
        g = reg.load_reservoir(key); off = reg.egm_offset_m(key, g); cur = reg.official_curve(key, off)
        H_pool = float(reg.to_egm(key, reg.RESERVOIRS[key]['pool_ft'], off))
        rr, cc, zr = r1.survey_cells(key, g)
    F, H_flat, _ = r1.flattened_region(g['dem'], g['W0'], g['swo1'])
    H_oracle = float(np.interp(F.sum() * 9e-4, np.maximum.accumulate(cur.area_km2.values), cur.H.values))
    return g, F, H_flat, H_oracle, cur, H_pool, (rr, cc, zr)


def score(key, g, F, cur, H_pool, cells, pred):
    z = g['dem'].copy(); z[F] = pred[F]
    lo = np.floor(min(np.nanmin(z[F]), cur.H.min()) / aev.STEP) * aev.STEP
    tab = aev.curve_table(z, g['W0'] | F, {'swo1': g['swo1']}, np.arange(lo, H_pool + 2 + aev.STEP, aev.STEP))
    Vmax = float(np.interp(H_pool, cur.H, cur.volume_km3))
    rr, cc, zr = cells
    inW = g['W0'][rr, cc]
    s = r1.err_stats(z[rr, cc][inW], zr[inW]); sF = r1.err_stats(z[rr, cc][inW & F[rr, cc]], zr[inW & F[rr, cc]])
    return dict(Vmax_PE_pct=(float(np.interp(H_pool, tab.H, tab.V_swo1)) / Vmax - 1) * 100,
                cells_W0_RMSE=s.get('RMSE'), cells_F_bias=sF.get('bias'))


def main():
    warnings.filterwarnings('ignore')
    OUT.mkdir(parents=True, exist_ok=True)
    proto = OUT / 'protocol.json'
    if not proto.exists():
        proto.write_text(json.dumps(dict(status='post-hoc ablation after test scoring; frozen method unchanged (R1-C)',
                                         variants=['cap_C', 'cap_A', 'cap_flat', 'no_cap', 'cap_oracle'],
                                         written_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=2),
                         encoding='utf-8')
    for k, v in tr.TEST.items():
        reg.RESERVOIRS.setdefault(k, dict(name=v['name'], crs=v['epsg']))
    rows = []
    for key in list(tr.TEST) + [k for k in reg.RESERVOIRS if k.startswith('D')]:
        print('RESERVOIR', key, flush=True)
        g, F, H_flat, H_oracle, cur, H_pool, cells = setup(key)
        folder = OUT / key; folder.mkdir(exist_ok=True)
        levels = dict(cap_flat=H_flat, cap_oracle=max(H_oracle, H_flat))
        levels['cap_A'] = max(sl.rule_A(g['dem'], F)[0], H_flat)
        for variant in ['cap_C', 'cap_A', 'cap_flat', 'no_cap', 'cap_oracle']:
            dev_path, test_path = EXISTING.get(variant, (None, None))
            path = (test_path if key.startswith('T') else dev_path)
            if path is not None:
                pred = reg.load_m0(Path(str(path).format(k=key)), g); H = None
            else:
                H = levels[variant]
                pred, _ = sl.run_tao(key, g, F, H, folder, variant)
            row = dict(id=key, set='test' if key.startswith('T') else 'dev', variant=variant, H_cap_m=H,
                       H_flat_m=H_flat, H_oracle_m=H_oracle, **score(key, g, F, cur, H_pool, cells, pred))
            rows.append(row); print(' ', variant, {q: (round(w, 2) if isinstance(w, float) else w) for q, w in row.items() if q in ('H_cap_m', 'Vmax_PE_pct', 'cells_W0_RMSE')}, flush=True)
            pd.DataFrame(rows).to_csv(OUT / 'ablation.csv', index=False, encoding='utf-8-sig')
    s = pd.DataFrame(rows).assign(abs_pe=lambda d: d.Vmax_PE_pct.abs())
    agg = s.groupby(['set', 'variant']).agg(median_abs_PE=('abs_pe', 'median'), mean_abs_PE=('abs_pe', 'mean'),
                                            max_abs_PE=('abs_pe', 'max'), median_W0_RMSE=('cells_W0_RMSE', 'median'))
    agg.to_csv(OUT / 'ablation_aggregate.csv', encoding='utf-8-sig'); print(agg.round(2), flush=True)


if __name__ == '__main__':
    main()
