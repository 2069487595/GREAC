"""Inventory of GREAC v1.0 after the HydroLAKES v1.1 small reservoirs (option B) and the SRTM floor v3.1 (user request
2026-11-09: update the text, Figure 4 and Figure 9c to the new dataset). One row per reservoir with its product line,
outlet, storage at the curve top, quality class, dam-height status and seed gap, and every number the draft quotes about
the dataset. Read only (locked summaries); nothing is recomputed from rasters.

Product lines
  reconstructed  global (5,168), gdw_ext (3,563), high_lat (213), remnant (16)          TC-dam storage at H_top (k = 0)
  direct_floor   SRTM floor v3.1 recommended (1,604) + HydroLAKES v1.1 floors without water in 2000, recommended (420)
  small_tier     HydroLAKES v1.0 tier (15,457) + v1.1 polygons with SWBD water (219) + with JRC 2000 water (1,821), k = 1
Coverage: GDW v1.0 without the 112 regulated lakes (35,183 records), parts from hl11_small_v1/b/gdw_part_v1_0.csv.
Check: the old dataset (small_tier_v1/gdw_part_with_tier.csv) must give 26,072 reservoirs, 161 countries and regions,
98.9 % of the area and 98.5 % of the capacity, so the country count and the shares are computed as before.
Outputs: 05_Results/by_date/2026-11-09/greac_inventory_v1/{greac_reservoirs_v1_0.csv, numbers.json}
Usage: .venv/Scripts/python.exe -I 04_Code/scripts/greac_inventory_v1.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from pyogrio.raw import read

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / '05_Results/by_date'
GDW = ROOT / '02_Data_Raw/GDW_v1_0_20261027/extracted/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp'
HL = ROOT / '02_Data_Raw/input_mead/HydroLAKES_polys_v10_wgs84.gpkg'
OUT = R / '2026-11-09/greac_inventory_v1'
PART_OLD = R / '2026-10-28/small_tier_v1/gdw_part_with_tier.csv'
PART_NEW = R / '2026-11-08/hl11_small_v1/b/gdw_part_v1_0.csv'
LINE_OF_PART = {'global': 'reconstructed', 'gdw_ext': 'reconstructed', 'highlat': 'reconstructed', 'remnant': 'reconstructed',
                'srtm_bed': 'direct_floor', 'hl11_floor': 'direct_floor',
                'small_tier': 'small_tier', 'hl11_tier_swbd': 'small_tier', 'hl11_tier_jrc2000': 'small_tier'}


def attrs(path, cols):
    meta, _, _, f = read(path, columns=cols, read_geometry=False)
    return pd.DataFrame(dict(zip(meta['fields'], f)))


def reservoirs():
    sg = pd.read_csv(R / '2026-11-04/seed_gap_flag_v1/seed_gap_per_reservoir.csv.gz')
    rows = []
    for sub, rel in [('global', '2026-10-10/r2_global_v1/global_summary_R2.csv'), ('gdw_ext', '2026-10-27/gdw_ext_v1/ext_summary.csv'),
                     ('high_lat', '2026-10-27/highlat_v1/hl_summary.csv'), ('remnant', '2026-10-27/remnant_fix_v1/fix_summary.csv')]:
        d = pd.read_csv(R / rel, usecols=['Hylak_id', 'Pour_long', 'Pour_lat', 'V_top_R2_km3', 'A_top_R2_km2', 'HL_Lake_area_km2', 'D_dam_m'])
        d = d.merge(sg[sg.product_line == sub][['Hylak_id', 'R2_status', 'seed_gap_m', 'flag_seed_gap']], on='Hylak_id', how='left', validate='one_to_one')
        rows.append(pd.DataFrame(dict(line='reconstructed', sub_line=sub, hylak_id=d.Hylak_id, lon=d.Pour_long, lat=d.Pour_lat,
                                      area_km2=d.HL_Lake_area_km2, V_top_km3=d.V_top_R2_km3, A_top_km2=d.A_top_R2_km2, D_dam_m=d.D_dam_m,
                                      R2_status=d.R2_status, seed_gap_m=d.seed_gap_m, flag_seed_gap=d.flag_seed_gap, quality_class=np.nan)))
    s = pd.read_csv(R / '2026-10-28/small_tier_v1/small_tier_summary.csv')
    s = s.merge(sg[sg.product_line == 'small_tier'][['Hylak_id', 'seed_gap_m', 'flag_seed_gap']], on='Hylak_id', how='left', validate='one_to_one')
    for sub, d in [('small_hl10', s), ('small_hl11_swbd', pd.read_csv(R / '2026-11-08/hl11_small_v1/hl11_tier_summary.csv')),
                   ('small_hl11_jrc2000', pd.read_csv(R / '2026-11-08/hl11_small_v1/b/b_tier_summary.csv'))]:
        rows.append(pd.DataFrame(dict(line='small_tier', sub_line=sub, hylak_id=d.Hylak_id, lon=d.Pour_long, lat=d.Pour_lat,
                                      area_km2=d.HL_Lake_area_km2, V_top_km3=d.V_top_km3, A_top_km2=d.A_top_km2, D_dam_m=d.D_dam_m,
                                      R2_status=d.R2_status, seed_gap_m=d.seed_gap_m, flag_seed_gap=d.flag_seed_gap.astype(bool),
                                      quality_class=d.quality_class)))
    g = attrs(GDW, ['GDW_ID', 'LONG_RIV', 'LAT_RIV']).set_index('GDW_ID')
    hl = attrs(HL, ['Hylak_id', 'Pour_long', 'Pour_lat']).set_index('Hylak_id')
    f1 = pd.read_csv(R / '2026-11-08/srtm_bed_pre2000_v1/srtm_bed_summary_v3_1.csv'); f1 = f1[f1.recommended]
    f2 = pd.read_csv(R / '2026-11-08/hl11_small_v1/b/b_floor_summary.csv'); f2 = f2[f2.recommended]
    for sub, d in [('floor_v3_1', f1), ('floor_hl11', f2)]:
        lon, lat = [], []
        for r in d.itertuples():
            if pd.notna(r.GDW_ID) and int(r.GDW_ID) in g.index:
                lon.append(g.at[int(r.GDW_ID), 'LONG_RIV']); lat.append(g.at[int(r.GDW_ID), 'LAT_RIV'])
            else:
                lon.append(hl.at[int(r.Hylak_id), 'Pour_long']); lat.append(hl.at[int(r.Hylak_id), 'Pour_lat'])
        rows.append(pd.DataFrame(dict(line='direct_floor', sub_line=sub, hylak_id=d.Hylak_id.values, lon=lon, lat=lat,
                                      area_km2=d.poly_area_km2.values, V_top_km3=d.V_top_km3.values, A_top_km2=d.A_top_km2.values,
                                      D_dam_m=np.nan, R2_status='direct', seed_gap_m=np.nan, flag_seed_gap=False, quality_class=np.nan,
                                      flag_built_before_2000=d.flag_built_before_2000.astype(bool).values)))
    t = pd.concat(rows, ignore_index=True)
    t['flag_built_before_2000'] = t.flag_built_before_2000.fillna(False).astype(bool)
    t['flag_seed_gap'] = t.flag_seed_gap.fillna(False).astype(bool)
    t.insert(0, 'greac_id', np.arange(1, len(t) + 1))
    return t


def coverage(path, gdw):
    p = pd.read_csv(path).merge(gdw[['GDW_ID', 'COUNTRY']], on='GDW_ID', how='left')
    assert len(p) == 35183
    p['line'] = p.part.map(LINE_OF_PART).fillna('none')
    a, c = p.AREA_POLY.clip(lower=0), p.CAP_MCM.clip(lower=0); A, V = a.sum(), c.sum()

    def sh(m):
        return dict(records=int(m.sum()), area_pct=round(100 * float(a[m].sum() / A), 2), capacity_pct=round(100 * float(c[m].sum() / V), 2))
    cov = p.line.ne('none')
    big = p.AREA_POLY >= 1
    out = dict(total=sh(cov), countries=int(p.COUNTRY[cov].nunique()), no_product=sh(~cov),
               by_part={k: sh(p.part.eq(k)) for k in sorted(p.part.unique())}, by_line={k: sh(p.line.eq(k)) for k in sorted(p.line.unique())},
               recon_plus_floor=sh(p.line.isin(['reconstructed', 'direct_floor'])),
               ge1km2=dict(records=int(big.sum()), with_product=int((cov & big).sum()),
                           area_pct_of_class=round(100 * float(a[cov & big].sum() / a[big].sum()), 2),
                           capacity_pct_of_class=round(100 * float(c[cov & big].sum() / c[big].sum()), 2)),
               countries_by_line={k: int(p.COUNTRY[p.line.eq(k)].nunique()) for k in ['reconstructed', 'direct_floor', 'small_tier']})
    return p, out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gdw = attrs(GDW, ['GDW_ID', 'COUNTRY'])
    _, old = coverage(PART_OLD, gdw)
    assert old['countries'] == 161 and abs(old['total']['area_pct'] - 98.87) < 0.01 and abs(old['total']['capacity_pct'] - 98.51) < 0.01, old
    p, new = coverage(PART_NEW, gdw)
    t = reservoirs()
    vc = t.sub_line.value_counts()
    assert len(t) == 28481 and vc['global'] == 5168 and vc['floor_v3_1'] == 1604 and vc['floor_hl11'] == 420 and vc['small_hl11_jrc2000'] == 1821
    t.to_csv(OUT / 'greac_reservoirs_v1_0.csv', index=False, float_format='%.8g')
    rec, fl, sm = t[t.line == 'reconstructed'], t[t.line == 'direct_floor'], t[t.line == 'small_tier']
    smt = sm.copy(); smt['ratio5'] = smt.A_top_km2 / smt.area_km2 > 5
    ab = smt.quality_class.isin(['A', 'B'])
    pos = smt.R2_status.eq('rescaled')
    recon_lines = t[t.line.isin(['reconstructed', 'small_tier'])]
    nores = p[p.line.eq('none')]
    route = pd.read_csv(R / '2026-11-08/hl11_small_v1/b/route.csv')
    num = dict(
        n_total=int(len(t)), n_by_line={k: int(v) for k, v in t.line.value_counts().items()}, n_by_sub_line={k: int(v) for k, v in vc.items()},
        coverage=new, coverage_old_check=dict(countries=old['countries'], area_pct=old['total']['area_pct'], capacity_pct=old['total']['capacity_pct']),
        direct_floor=dict(n=int(len(fl)), V_top_km3=round(float(fl.V_top_km3.sum()), 1), countries=new['countries_by_line']['direct_floor'],
                          built_before_2000_recommended=int(fl.flag_built_before_2000.sum()),
                          with_curves=int(1731 + len(pd.read_csv(R / '2026-11-08/hl11_small_v1/b/b_floor_summary.csv')))),
        small_tier=dict(n=int(len(sm)), countries=new['countries_by_line']['small_tier'],
                        area_median_km2=round(float(sm.area_km2.median()), 2), area_iqr_km2=[round(float(sm.area_km2.quantile(q)), 2) for q in (.25, .75)],
                        positive_dam_reading_pct=round(100 * float(pos.mean()), 1), D_dam_median_m=round(float(sm.D_dam_m[pos].median()), 1),
                        classes={k: int(v) for k, v in sm.quality_class.value_counts().sort_index().items()},
                        V_top_total_km3=round(float(sm.V_top_km3.sum()), 1),
                        AB=dict(n=int(ab.sum()), V_top_km3=round(float(sm.V_top_km3[ab].sum()), 1), A_top_km2=round(float(sm.A_top_km2[ab].sum()), 0)),
                        extent_over_5x=dict(n=int(smt.ratio5.sum()), all_class_C=bool((smt.quality_class[smt.ratio5] == 'C').all()),
                                            V_top_km3=round(float(smt.V_top_km3[smt.ratio5].sum()), 1))),
        seed_gap=dict(n_reconstruction_lines=int(len(recon_lines)), flagged=int(recon_lines.flag_seed_gap.sum()),
                      flagged_pct=round(100 * float(recon_lines.flag_seed_gap.mean()), 1),
                      pct_by_line={k: round(100 * float(g.flag_seed_gap.mean()), 1) for k, g in recon_lines.groupby('sub_line')},
                      small_tier_pct=round(100 * float(sm.flag_seed_gap.mean()), 1)),
        no_product=dict(records=int(len(nores)), area_pct=new['no_product']['area_pct'], capacity_pct=new['no_product']['capacity_pct'],
                        by_category={k: int(v) for k, v in nores.category.value_counts().items()},
                        hl11_undecided_reservoirs=int((route.route == 'none').sum())),
    )
    (OUT / 'numbers.json').write_text(json.dumps(num, indent=1), encoding='utf-8', newline='\n')
    print(json.dumps(num, indent=1))


if __name__ == '__main__':
    main()
