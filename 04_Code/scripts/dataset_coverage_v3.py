"""Coverage of the combined dataset in GDW v1.0, v3: dataset_coverage_v2 on the SRTM-bed v3 curves, plus the number of
countries (GDW COUNTRY of the covered records, one spelling for all parts) and the storage at the curve top of each part
(TC-dam for the reconstructed parts, V_top for the SRTM bed).
Usage: .venv/Scripts/python.exe -I 04_Code/scripts/dataset_coverage_v3.py
"""
from pathlib import Path
import sys, json

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import dataset_coverage_v2 as c2

B = c2.B
OUT = B / '2026-10-27/dataset_coverage_v3'
PARTS = {'global': B / '2026-10-10/r2_global_v1/global_summary_R2.csv',
         'gdw_ext': B / '2026-10-27/gdw_ext_v1/ext_summary.csv',
         'highlat': B / '2026-10-27/highlat_v1/hl_summary.csv',
         'remnant': B / '2026-10-27/remnant_fix_v1/fix_summary.csv'}


def main():
    c2.main('srtm_bed_v3', OUT)
    out = json.loads((OUT / 'coverage.json').read_text(encoding='utf-8'))
    gp = pd.read_csv(OUT / 'gdw_part.csv')
    cty = pd.read_csv(B / '2026-10-27/gdw_coverage_v1/gdw_reservoirs_category.csv', usecols=['GDW_ID', 'COUNTRY'])
    gp = gp.merge(cty, on='GDW_ID', how='left')
    cov = gp[gp.part != 'none']
    parts = {}
    for k, p in PARTS.items():
        d = pd.read_csv(p)
        parts[k] = dict(n=int(len(d)), V_top_TC_dam_km3=round(float(d.V_top_R2_km3.sum()), 1),
                        dam_reading_share=round(float((d.R2_status == 'rescaled').mean()), 3))
    s = pd.read_csv(B / '2026-10-27/srtm_bed_v3/srtm_bed_summary.csv'); s = s[s.recommended]
    parts['srtm_bed'] = dict(n=int(len(s)), V_top_km3=round(float(s.V_top_km3.sum()), 1))
    for k in parts:
        parts[k]['countries_gdw'] = int(cov[cov.part == k].COUNTRY.nunique())
    out['products'] = dict(parts=parts, reconstructed_n=sum(parts[k]['n'] for k in PARTS),
                           reconstructed_countries_gdw=int(cov[cov.part.isin(list(PARTS))].COUNTRY.nunique()),
                           countries_gdw_all=int(cov.COUNTRY.nunique()))
    (OUT / 'coverage.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps(out['products'], indent=1))


if __name__ == '__main__':
    main()
