"""How much of the Global Dam Watch database (GDW v1.0; Lehner et al. 2024) the global product covers, and why the
other GDW reservoirs have no product. Descriptive; nothing here changes the product.

Each of the 35,295 GDW reservoir polygons gets exactly one category, in this order:
  product            matched to one of the 5,168 product reservoirs (same HydroLAKES id, or same GRanD id)
  listed_failed      in the 5,277-reservoir run list but the run failed (global_failed.csv)
  hl2_small          HydroLAKES Lake_type 2, excluded because Lake_area < 1 km2
  hl2_outside_lat    HydroLAKES Lake_type 2, excluded because the pour point is outside 56 S - 60 N
  hl3_lake_control   HydroLAKES Lake_type 3 (controlled natural lake), not run
  small              not a HydroLAKES reservoir; GDW polygon area < 1 km2
  outside_lat        not a HydroLAKES reservoir; dam point (LAT_RIV, complete for all records) outside 56 S - 60 N
  candidate_hl10     not a HydroLAKES reservoir, >= 1 km2, inside 56 S - 60 N, polygon in HydroLAKES v1.0 (Lake_type 1)
  candidate_hl11     as above, but the GDW HYLAK_ID is beyond the v1.0 id range (polygon only in HydroLAKES v1.1;
                     mostly reservoirs filled after 2000, which the 2000 water mask of the product does not contain)
  candidate_no_hl    as above but without a HydroLAKES polygon
The candidate classes are the reservoirs the method could be run on without any change of rules; candidate_hl10 can
use the same input path as the product (HydroLAKES v1.0 polygon and pour point).
Shares are given against all GDW polygons and against GDW without the controlled natural lakes (hl3_lake_control:
Victoria, Baikal, Winnipeg, Ontario, ...), whose polygon area is mostly natural lake.
Inputs: 02_Data_Raw/GDW_v1_0_20261027 (Figshare 10.6084/m9.figshare.25988293, MD5 checked),
        02_Data_Raw/input_mead/HydroLAKES_polys_v10_wgs84.gpkg (attributes only),
        05_Results/by_date/2026-10-07/route1_global_v1/{global_summary,global_list,global_excluded,global_failed}.csv
Outputs: 05_Results/by_date/2026-10-27/gdw_coverage_v1/{gdw_reservoirs_category.csv, summary.json}
Usage: .venv/Scripts/python.exe -I 04_Code/scripts/gdw_coverage_v1.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from pyogrio.raw import read

ROOT = Path(__file__).resolve().parents[2]
GDW = ROOT / '02_Data_Raw/GDW_v1_0_20261027/extracted/GDW_v1_0_shp'
HL = ROOT / '02_Data_Raw/input_mead/HydroLAKES_polys_v10_wgs84.gpkg'
RUN = ROOT / '05_Results/by_date/2026-10-07/route1_global_v1'
OUT = ROOT / '05_Results/by_date/2026-10-27/gdw_coverage_v1'
COLS = ['GDW_ID', 'RES_NAME', 'DAM_NAME', 'COUNTRY', 'LAKE_CTRL', 'TIMELINE', 'YEAR_DAM', 'PRE_YEAR', 'DAM_HGT_M',
        'AREA_SKM', 'AREA_POLY', 'CAP_MCM', 'CAP_REP', 'CAP_MAX', 'LAT_RIV', 'LONG_RIV', 'LAT_DAM', 'LONG_DAM', 'ORIG_SRC', 'POLY_SRC',
        'GRAND_ID', 'HYLAK_ID', 'QUALITY']
ORDER = ['product', 'listed_failed', 'hl2_small', 'hl2_outside_lat', 'hl3_lake_control', 'small', 'outside_lat',
         'candidate_hl10', 'candidate_hl11', 'candidate_no_hl']


def attrs(path, columns, where=None):
    meta, _, _, f = read(path, columns=columns, read_geometry=False, where=where)
    return pd.DataFrame(dict(zip(meta['fields'], f)))


def stats(d, tot_area, tot_cap, nlc_area, nlc_cap):
    a, c = d.AREA_POLY.clip(lower=0), d.CAP_MCM.clip(lower=0)
    return dict(n=int(len(d)), area_km2=round(float(a.sum()), 1), cap_km3=round(float(c.sum()) / 1000, 1),
                area_share=round(float(a.sum() / tot_area), 4), cap_share=round(float(c.sum() / tot_cap), 4),
                area_share_no_lake_control=round(float(a.sum() / nlc_area), 4),
                cap_share_no_lake_control=round(float(c.sum() / nlc_cap), 4),
                area_median_km2=round(float(a.median()), 2) if len(d) else None,
                dam_height_recorded=int((d.DAM_HGT_M > 0).sum()), cap_reported=int((d.CAP_REP > 0).sum()),
                built_2000_or_later=int(((d.YEAR_DAM >= 2000) | (d.PRE_YEAR > 2000)).sum()))


def main():
    g = attrs(GDW / 'GDW_reservoirs_v1_0.shp', COLS)
    b = attrs(GDW / 'GDW_barriers_v1_0.shp', ['GDW_ID', 'DAM_HGT_M'])
    hl = attrs(HL, ['Hylak_id', 'Lake_type', 'Lake_area'])
    summ = pd.read_csv(RUN / 'global_summary.csv', usecols=['Hylak_id', 'Grand_id'])
    lst = pd.read_csv(RUN / 'global_list.csv', usecols=['Hylak_id', 'Grand_id'])
    exc = pd.read_csv(RUN / 'global_excluded.csv', usecols=['Hylak_id', 'Grand_id', 'Lake_type', 'excluded'])
    print(f'GDW reservoirs {len(g):,}, barriers {len(b):,}, HydroLAKES {len(hl):,}', flush=True)

    prod_hl, prod_gr = set(summ.Hylak_id), set(summ.Grand_id[summ.Grand_id > 0])
    failed_hl = set(lst.Hylak_id) - prod_hl
    reason = {}
    for _, r in exc.iterrows():
        k = ('hl3_lake_control' if r.Lake_type == 3 else 'hl2_small' if r.excluded.startswith('Lake_area')
             else 'hl2_outside_lat')
        reason[('hl', int(r.Hylak_id))] = k
        if r.Grand_id > 0:
            reason[('gr', int(r.Grand_id))] = k
    for i, gr in zip(lst.Hylak_id, lst.Grand_id):
        if i in failed_hl:
            reason[('hl', int(i))] = 'listed_failed'
            if gr > 0:
                reason[('gr', int(gr))] = 'listed_failed'

    hl_type = dict(zip(hl.Hylak_id.astype(int), hl.Lake_type.astype(int)))
    g['HL_Lake_type'] = [hl_type.get(int(i), 0) if i > 0 else 0 for i in g.HYLAK_ID]     # 0: not in HydroLAKES v1.0
    hl_max = int(hl.Hylak_id.max())
    in_lat = (g.LAT_RIV >= -56) & (g.LAT_RIV < 60)     # LAT_DAM is filled for about 6,000 records only (missing = 0)
    cat = []
    for r, lat_ok in zip(g.itertuples(), in_lat):
        h, gr = int(r.HYLAK_ID), int(r.GRAND_ID)
        if h in prod_hl or gr in prod_gr:
            cat.append('product'); continue
        k = reason.get(('hl', h)) or reason.get(('gr', gr))
        if k:
            cat.append(k); continue
        area = r.AREA_POLY if r.AREA_POLY > 0 else r.AREA_SKM
        cat.append('small' if area < 1 else 'outside_lat' if not lat_ok else
                   'candidate_no_hl' if h <= 0 else 'candidate_hl11' if h > hl_max else 'candidate_hl10')
    g['category'] = cat
    cand = g.category.str.startswith('candidate')

    # consistency of the two keys on the product reservoirs: GDW HYLAK_ID against the product's Hylak_id
    pg = g[(g.category == 'product') & (g.GRAND_ID > 0)].merge(summ, left_on='GRAND_ID', right_on='Grand_id', how='inner')
    key_agree = float((pg.HYLAK_ID == pg.Hylak_id).mean())
    covered = len(prod_hl & set(g.HYLAK_ID[g.HYLAK_ID > 0])) + len(
        set(summ.Hylak_id[summ.Grand_id.isin(set(g.GRAND_ID[g.GRAND_ID > 0]))]) - set(g.HYLAK_ID))

    tot_area, tot_cap = float(g.AREA_POLY.clip(lower=0).sum()), float(g.CAP_MCM.clip(lower=0).sum())
    nlc = g.category != 'hl3_lake_control'
    nlc_area, nlc_cap = float(g.AREA_POLY[nlc].clip(lower=0).sum()), float(g.CAP_MCM[nlc].clip(lower=0).sum())
    out = dict(
        source='GDW v1.0, Figshare 10.6084/m9.figshare.25988293 (GDW_v1_0_shp.zip, MD5 5064cf2315ef6159d9133b03596e761a)',
        gdw_reservoir_polygons=int(len(g)), gdw_barriers=int(len(b)),
        gdw_total_area_poly_km2=round(tot_area, 1), gdw_total_area_skm_km2=round(float(g.AREA_SKM.clip(lower=0).sum()), 1),
        gdw_total_cap_km3=round(tot_cap / 1000, 1),
        gdw_no_lake_control_area_poly_km2=round(nlc_area, 1), gdw_no_lake_control_cap_km3=round(nlc_cap / 1000, 1),
        hydrolakes_v10_max_id=hl_max,
        barriers_with_dam_height=int((b.DAM_HGT_M > 0).sum()),
        reservoirs_with_dam_height=int((g.DAM_HGT_M > 0).sum()), reservoirs_with_cap_reported=int((g.CAP_REP > 0).sum()),
        product_reservoirs_found_in_gdw=int(covered), product_reservoirs=int(len(summ)),
        product_key_agreement_hylak_vs_grand=round(key_agree, 4),
        categories={k: stats(g[g.category == k], tot_area, tot_cap, nlc_area, nlc_cap) for k in ORDER},
        candidates_all=stats(g[cand], tot_area, tot_cap, nlc_area, nlc_cap),
        candidate_hl10_lake_type={str(k): int(v) for k, v in
                                  g[g.category == 'candidate_hl10'].HL_Lake_type.value_counts().items()},
        candidate_area_quantiles_km2={str(q): round(float(v), 2) for q, v in
                                      g[cand].AREA_POLY.quantile([.1, .25, .5, .75, .9]).items()},
        candidates_top_countries={k: int(v) for k, v in g[cand].COUNTRY.value_counts().head(12).items()},
        candidate_orig_src={k: int(v) for k, v in g[cand].ORIG_SRC.value_counts().items()},
        candidate_year_known={k: int(((g.category == k) & ((g.YEAR_DAM > 0) | (g.PRE_YEAR > 0))).sum())
                              for k in ('candidate_hl10', 'candidate_hl11')},
        small_area_quantiles_km2={str(q): round(float(v), 3) for q, v in
                                  g[g.category == 'small'].AREA_POLY.quantile([.25, .5, .75]).items()},
    )
    OUT.mkdir(parents=True, exist_ok=True)
    g.sort_values(['category', 'GDW_ID']).to_csv(OUT / 'gdw_reservoirs_category.csv', index=False, encoding='utf-8')
    (OUT / 'summary.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
