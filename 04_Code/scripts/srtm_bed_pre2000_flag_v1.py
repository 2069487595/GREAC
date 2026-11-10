"""Directly read floors built before 2000 (user decision 2026-11-08, recorded in docs/plans/2026-10-27-srtm-bed-v3-spec.md,
section 6): a reservoir whose GDW dam year is before 2000 gets flag_built_before_2000; if, in addition, SWBD water of
February 2000 covers at least 5 % of R (flag_w2000_ge_5pct), the floor read from SRTM is partly the water surface of
2000 and the reservoir is no longer recommended. Nothing in the locked runs changes; only the summary columns.
Input:  05_Results/by_date/2026-10-27/srtm_bed_v3/srtm_bed_summary.csv (locked v3 summary; year_dam from GDW)
Output: 05_Results/by_date/2026-11-08/srtm_bed_pre2000_v1/{srtm_bed_summary_v3_1.csv, summary.json}
Usage:  .venv/Scripts/python.exe -I 04_Code/scripts/srtm_bed_pre2000_flag_v1.py
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / '05_Results/by_date/2026-10-27/srtm_bed_v3/srtm_bed_summary.csv'
OUT = ROOT / '05_Results/by_date/2026-11-08/srtm_bed_pre2000_v1'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(SRC)
    y = s.year_dam.where(s.year_dam > 0)
    s['flag_built_before_2000'] = (y < 2000).fillna(False)
    s['recommended_v3'] = s.recommended
    s['recommended'] = s.recommended_v3 & ~(s.flag_built_before_2000 & s.flag_w2000_ge_5pct)
    s.to_csv(OUT / 'srtm_bed_summary_v3_1.csv', index=False, float_format='%.10g')
    rec0, rec1 = s.recommended_v3, s.recommended
    drop = rec0 & ~rec1
    out = dict(n=int(len(s)), recommended_v3=int(rec0.sum()), recommended_v3_1=int(rec1.sum()), no_longer_recommended=int(drop.sum()),
               flag_built_before_2000_recommended_v3_1=int((rec1 & s.flag_built_before_2000).sum()),
               year_known_recommended_v3=int((rec0 & y.notna()).sum()),
               V_top_km3_recommended_v3=round(float(s.V_top_km3[rec0].sum()), 2), V_top_km3_recommended_v3_1=round(float(s.V_top_km3[rec1].sum()), 2),
               dropped=s[drop][['key', 'country', 'poly_area_km2', 'year_dam', 'w2000_share_R', 'V_top_km3']].round(4).to_dict('records'))
    (OUT / 'summary.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'dropped'}, indent=1))
    print(pd.DataFrame(out['dropped']).to_string())


if __name__ == '__main__':
    main()
