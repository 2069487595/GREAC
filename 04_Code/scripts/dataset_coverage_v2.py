"""Coverage of the combined dataset in GDW v1.0, v2: dataset_coverage_v1 with the SRTM-bed v2 curves, plus what is left
(by GDW category and polygon size) and how concentrated GDW storage is (share held by the largest N reservoirs).
Shares of the GDW reservoir area and storage without the controlled natural lakes, as in gdw_coverage_v1.
Usage: .venv/Scripts/python.exe -I 04_Code/scripts/dataset_coverage_v2.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / '05_Results/by_date'
OUT = B / '2026-10-27/dataset_coverage_v2'
SIZE = [0, 0.1, 1, 10, 100, np.inf]
SIZE_LABELS = ['<0.1', '0.1-1', '1-10', '10-100', '>100']


def shares(v, tot_a, tot_c):
    return dict(polygons=int(len(v)), area_share=round(float(v.AREA_POLY.clip(lower=0).sum() / tot_a), 4),
                cap_share=round(float(v.CAP_MCM.clip(lower=0).sum() / tot_c), 4))


def main(srtm='srtm_bed_v2', out_dir=OUT):
    """srtm: the SRTM-bed run folder (2026-10-27); out_dir: the output folder (defaults reproduce v2)."""
    g = pd.read_csv(B / '2026-10-27/gdw_coverage_v1/gdw_reservoirs_category.csv', low_memory=False)
    g = g[g.category != 'hl3_lake_control'].copy()
    tot_a, tot_c = g.AREA_POLY.clip(lower=0).sum(), g.CAP_MCM.clip(lower=0).sum()
    hl = {'global': set(pd.read_csv(B / '2026-10-07/route1_global_v1/global_summary.csv').Hylak_id),
          'gdw_ext': set(pd.read_csv(B / '2026-10-27/gdw_ext_v1/ext_summary.csv').Hylak_id),
          'remnant': set(pd.read_csv(B / '2026-10-27/remnant_fix_v1/fix_summary.csv').Hylak_id),
          'highlat': set(pd.read_csv(B / '2026-10-27/highlat_v1/hl_summary.csv').Hylak_id)}
    s = pd.read_csv(B / '2026-10-27' / srtm / 'srtm_bed_summary.csv')
    s_rec = s[s.recommended]
    part = pd.Series('none', index=g.index)
    part[g.category.eq('product')] = 'global'
    for name in ['gdw_ext', 'remnant', 'highlat']:
        part[part.eq('none') & g.HYLAK_ID.isin(hl[name])] = name
    in_srtm = lambda d: g.GDW_ID.isin(set(d.GDW_ID.dropna().astype(int))) | g.HYLAK_ID.isin(set(d.Hylak_id.dropna().astype(int)))
    part[part.eq('none') & in_srtm(s_rec)] = 'srtm_bed'
    g['part'] = part
    g['srtm_bed_not_recommended'] = part.eq('none') & in_srtm(s[~s.recommended])
    g['size'] = pd.cut(g.AREA_POLY, SIZE, labels=SIZE_LABELS, right=False)
    out = dict(parts={k: shares(v, tot_a, tot_c) for k, v in g.groupby('part')})
    cov = g[g.part != 'none']
    out['combined'] = shares(cov, tot_a, tot_c)
    out['reservoirs'] = dict(global_product=len(hl['global']), gdw_ext=len(hl['gdw_ext']), remnant=len(hl['remnant']), highlat=len(hl['highlat']),
                             srtm_bed_recommended=int(len(s_rec)))
    out['reservoirs']['total'] = sum(out['reservoirs'].values())
    rest = g[g.part == 'none']
    out['remaining'] = dict(all=shares(rest, tot_a, tot_c),
                            by_size={str(k): shares(v, tot_a, tot_c) for k, v in rest.groupby('size', observed=True)},
                            ge_1km2_by_category={k: shares(v, tot_a, tot_c) for k, v in rest[rest.AREA_POLY >= 1].groupby('category')},
                            ge_1km2_srtm_bed_not_recommended=shares(rest[(rest.AREA_POLY >= 1) & rest.srtm_bed_not_recommended], tot_a, tot_c))
    out['gdw_by_size'] = {str(k): dict(shares(v, tot_a, tot_c), covered=shares(v[v.part != 'none'], tot_a, tot_c))
                          for k, v in g.groupby('size', observed=True)}
    c = np.sort(g.CAP_MCM.clip(lower=0).values)[::-1]; cs = np.cumsum(c) / c.sum()
    out['gdw_storage_held_by_largest'] = {str(n): round(float(cs[n - 1]), 4) for n in (100, 500, 1000, 2000, 5000, 10000)}
    out['gdw_totals'] = dict(polygons=int(len(g)), area_km2=round(float(tot_a)), cap_km3=round(float(tot_c) / 1e3))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'coverage.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    g[['GDW_ID', 'HYLAK_ID', 'category', 'part', 'size', 'AREA_POLY', 'CAP_MCM', 'srtm_bed_not_recommended']].to_csv(out_dir / 'gdw_part.csv', index=False)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
