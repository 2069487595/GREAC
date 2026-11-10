"""GREAC v1.0 data package for Zenodo (plan docs/plans/2026-11-09-greac-package-plan.md; user request 2026-11-09:
"你把这些准备好，我去创建一个账号，然后创建一个仓库获取DOI给你"). Rearranges the locked products listed by the inventory
(greac_inventory_v1); no product is recomputed. The bed rasters are the locked recommended surfaces masked to the
integration extent of their curves: R for the reconstructed line and the directly read floors, R widened by one cell
(k = 1) for the small-reservoir tier, where cells of the ring outside the locked raster take the FABDEM elevation as in
the k = 1 run (small_tier_v1 widen: z = FABDEM with F replaced by the bed). Cells below H_top that the curves leave out
(not 8-connected to the water surface W0 | F, or to the seed of a floor) are NoData, so summing (H_top - z) over the valid
cells reproduces V_top. H_flat, H_seed and H_top are the exact levels of the run records (the summaries round them to 0.01 m).

Subcommands
  attributes  upload/greac_attributes.csv; checks against the inventory, numbers.json and Table 2 of the draft
  curves      upload/curves_<line>.csv.gz; last row = V_top, A_top, H_top; TC-cap last row = V_top_R1C_km3
  beds        beds/<line>/<greac_id>_bed.tif [--part i/n]; check/beds_<i>_of_<n>.csv (storage at H_top from the raster)
  gpkg        upload/greac_reservoirs.gpkg (layer reservoirs: polygon + attributes; layer outlets: point + greac_id)
  finish      upload/beds_<line>.zip, upload/MANIFEST.sha256, check/package_check.json
Outputs: 05_Results/by_date/2026-11-09/greac_v1_0/{upload, beds, check}
Usage: .venv/Scripts/python.exe -I 04_Code/scripts/greac_package_v1.py attributes|curves|beds|gpkg|finish [--part i/n]
"""
from pathlib import Path
import argparse, hashlib, json, pickle, time, zipfile
import numpy as np
import pandas as pd
import rasterio
import shapely
from pyogrio.raw import read, write
from scipy import ndimage as ndi
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import aev_registry_v1 as reg          # noqa: E402  connected_to: the connectivity rule of the curves
import route1_tao_flat_v1 as r1        # noqa: E402  flattened_region: F of the locked runs

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / '05_Results/by_date'
D = ROOT / '03_Data_Processed'
OUT = R / '2026-11-09/greac_v1_0'
UP, BEDS, CHK = OUT / 'upload', OUT / 'beds', OUT / 'check'
INV = R / '2026-11-09/greac_inventory_v1'
GDW = ROOT / '02_Data_Raw/GDW_v1_0_20261027/extracted/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp'
HL = ROOT / '02_Data_Raw/input_mead/HydroLAKES_polys_v10_wgs84.gpkg'
PART = R / '2026-11-08/hl11_small_v1/b/gdw_part_v1_0.csv'
VR = json.loads((R / '2026-10-07/route1_global_v1/validated_range.json').read_text(encoding='utf-8'))
VR2 = json.loads((R / '2026-10-10/r2_global_v1/validated_range_R2.json').read_text(encoding='utf-8'))
EIGHT = np.ones((3, 3), bool)
CELL = 900.0
ZIP_MAX = 300_000_000         # bytes per bed archive
LINES = ['reconstructed', 'direct_floor', 'small_tier']

# sub_line -> locked sources. rec: reconstructed line (k = 0); small: tier (k = 1); floor: directly read floors (recommended)
SUB = {
    'global': dict(kind='rec', summary='2026-10-10/r2_global_v1/global_summary_R2.csv', r1='2026-10-07/route1_global_v1/run',
                   r2='2026-10-10/r2_global_v1/run', inputs='global_route1_20261007/inputs', part='global', hl='1.0'),
    'gdw_ext': dict(kind='rec', summary='2026-10-27/gdw_ext_v1/ext_summary.csv', r1='2026-10-27/gdw_ext_v1/run',
                    r2='2026-10-27/gdw_ext_v1/run', inputs='gdw_ext_20261027/inputs', part='gdw_ext', hl='1.0'),
    'high_lat': dict(kind='rec', summary='2026-10-27/highlat_v1/hl_summary.csv', r1='2026-10-27/highlat_v1/run',
                     r2='2026-10-27/highlat_v1/run', inputs='highlat_20261027/inputs', part='highlat', hl='1.0'),
    'remnant': dict(kind='rec', summary='2026-10-27/remnant_fix_v1/fix_summary.csv', r1='2026-10-27/remnant_fix_v1/run',
                    r2='2026-10-27/remnant_fix_v1/run', inputs='remnant_fix_20261027/inputs', part='remnant', hl='1.0'),
    'small_hl10': dict(kind='small', summary='2026-10-28/small_tier_v1/small_tier_summary.csv', r1='2026-10-28/small_tier_v1/run',
                       r2='2026-10-28/small_tier_v1/run', aev='2026-10-28/small_tier_v1/small_aev.csv.gz',
                       inputs='small_tier_20261028/inputs', part='small_tier', hl='1.0'),
    'small_hl11_swbd': dict(kind='small', summary='2026-11-08/hl11_small_v1/hl11_tier_summary.csv', r1='2026-11-08/hl11_small_v1/run',
                            r2='2026-11-08/hl11_small_v1/run', aev='2026-11-08/hl11_small_v1/small_aev.csv.gz',
                            inputs='hl11_small_20261108/inputs', part='hl11_tier_swbd', hl='1.1',
                            polygons='hl11_small_20261108/polygons.pkl'),
    'small_hl11_jrc2000': dict(kind='small', summary='2026-11-08/hl11_small_v1/b/b_tier_summary.csv', r1='2026-11-08/hl11_small_v1/b/tier/run',
                               r2='2026-11-08/hl11_small_v1/b/tier/run', aev='2026-11-08/hl11_small_v1/b/tier/small_aev.csv.gz',
                               inputs='hl11_small_20261108/b_tier/inputs', part='hl11_tier_jrc2000', hl='1.1',
                               polygons='hl11_small_20261108/b_tier/polygons.pkl'),
    'floor_v3_1': dict(kind='floor', summary='2026-11-08/srtm_bed_pre2000_v1/srtm_bed_summary_v3_1.csv', run='2026-10-27/srtm_bed_v3/run',
                       part='srtm_bed', hl='1.0'),
    'floor_hl11': dict(kind='floor', summary='2026-11-08/hl11_small_v1/b/b_floor_summary.csv', run='2026-11-08/hl11_small_v1/floor/run',
                       part='hl11_floor', hl='1.1'),
}
LINE = {'rec': 'reconstructed', 'small': 'small_tier', 'floor': 'direct_floor'}
FLAGS = ['flag_seed_gap', 'flag_no_dam_height', 'flag_area_beyond', 'flag_FW0_beyond', 'flag_RW0_beyond', 'flag_Ddam_beyond',
         'flag_poly_not_swbd', 'flag_tiny', 'flag_built_before_2000']
COLS = ['greac_id', 'product_line', 'sub_line', 'hylak_id', 'hydrolakes_version', 'gdw_ids', 'country', 'lon', 'lat', 'area_km2',
        'H_flat_m', 'H_seed_m', 'H_top_m', 'seed_gap_m', 'D_dam_m', 'regime', 'F_km2', 'W0_km2', 'R_km2', 'A_top_km2', 'V_top_km3',
        'V_top_tccap_km3'] + FLAGS + ['quality_class']


def attrs(path, cols):
    meta, _, _, f = read(path, columns=cols, read_geometry=False)
    return pd.DataFrame(dict(zip(meta['fields'], f)))


def same(a, b, rel=1e-7, ab=1e-12):  # the inventory CSV is written with 8 significant digits
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.all((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= np.maximum(ab, rel * np.abs(b)))))


def flagged(x):
    return pd.array(np.asarray(x, bool).astype('int8'), dtype='Int8')


def na_int8(n):
    return pd.array([pd.NA] * n, dtype='Int8')


# ---------------------------------------------------------------- attributes
def exact_levels(t, s, runs, keys, sub):
    """Levels of the per-reservoir run records: the summaries round them to 0.01 m, while V_top is at the exact H_top."""
    rec = [json.loads(p.read_text(encoding='utf-8')) for p in runs]
    for k in keys:
        x = np.array([np.nan if j.get(k) is None else float(j[k]) for j in rec])
        y = s[k].values.astype(float)     # floors without a FABDEM water plateau have no H_flat in either record
        assert np.all((np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= 0.0051)), (sub, k, float(np.nanmax(np.abs(x - y))))
        t[k] = x


def build_attributes():
    inv = pd.read_csv(INV / 'greac_reservoirs_v1_0.csv')
    part = pd.read_csv(PART).merge(attrs(GDW, ['GDW_ID', 'COUNTRY', 'GRAND_ID']), on='GDW_ID', how='left', validate='many_to_one')
    out, used = [], []
    for sub, c in SUB.items():
        i = inv[inv.sub_line == sub].reset_index(drop=True)
        s = pd.read_csv(R / c['summary'])
        if c['kind'] == 'floor':
            s = s[s.recommended].reset_index(drop=True)
        assert len(s) == len(i) and (i.line == LINE[c['kind']]).all(), sub
        n = len(s)
        t = pd.DataFrame(dict(greac_id=i.greac_id, product_line=i.line, sub_line=sub, lon=i.lon, lat=i.lat, A_top_km2=i.A_top_km2,
                              V_top_km3=i.V_top_km3))
        if c['kind'] in ('rec', 'small'):
            assert (s.Hylak_id.values == i.hylak_id.values).all(), sub
            t['hylak_id'] = s.Hylak_id.astype('Int64'); t['hydrolakes_version'] = c['hl']
            t['area_km2'] = s.HL_Lake_area_km2
            for k in ['H_flat_m', 'H_seed_m', 'H_top_m', 'F_km2', 'W0_km2', 'R_km2', 'D_dam_m']:
                t[k] = s[k].values
            exact_levels(t, s, [R / c['r1'] / str(h) / f'{h}_run.json' for h in s.Hylak_id.astype(int)], ['H_flat_m', 'H_seed_m', 'H_top_m'], sub)
            assert same(t.D_dam_m, i.D_dam_m) and (s.R2_status.values == i.R2_status.values).all(), sub
            assert set(s.regime) <= {'F>W0', 'F<=W0'} and ((s.regime == 'F>W0') == (s.F_km2 > s.W0_km2)).all(), sub
            t['regime'] = np.where(s.regime == 'F>W0', 'high', 'low')
            t['seed_gap_m'] = i.seed_gap_m.values
            # the maxima are values of evaluated reservoirs; ratios recomputed from the CSV differ from them by rounding (1e-16)
            FW, RW, tol = s.F_km2 / s.W0_km2, s.R_km2 / s.W0_km2, 1 + 1e-9
            f = dict(flag_seed_gap=i.flag_seed_gap.astype(bool).values, flag_no_dam_height=(s.R2_status != 'rescaled').values,
                     flag_area_beyond=(s.HL_Lake_area_km2 > VR2['HL_area_max_km2'] * tol).values,
                     flag_FW0_beyond=(FW > VR['F_over_W0_max'] * tol).values, flag_RW0_beyond=(RW > VR['R_over_W0_max'] * tol).values,
                     flag_Ddam_beyond=(s.D_dam_m > VR2['D_dam_max_m'] * tol).fillna(False).values,
                     flag_poly_not_swbd=(s.Poly_src != 'SWBD').values)
            assert (f['flag_seed_gap'] == (i.seed_gap_m >= 5).values).all(), sub
            # flags already in the locked summaries must agree with the definitions above
            for k, col in [('flag_FW0_beyond', 'flag_F_W0_beyond_validated'), ('flag_RW0_beyond', 'flag_R_W0_beyond_validated'),
                           ('flag_Ddam_beyond', 'flag_D_dam_beyond_validated'), ('flag_area_beyond', 'flag_area_beyond_validated'),
                           ('flag_poly_not_swbd', 'flag_poly_not_swbd'), ('flag_no_dam_height', 'flag_no_dam_height')]:
                if col in s:
                    assert (s[col].fillna(False).astype(bool).values == f[k]).all(), (sub, k)
            if 'flag_beyond_validated' in s:
                assert (s.flag_beyond_validated.astype(bool).values == (f['flag_FW0_beyond'] | f['flag_RW0_beyond'] | f['flag_Ddam_beyond'])).all(), sub
            for k, v in f.items():
                t[k] = flagged(v)
            if c['kind'] == 'rec':
                assert same(s.V_top_R2_km3, i.V_top_km3) and same(s.A_top_R2_km2, i.A_top_km2), sub
                t['V_top_tccap_km3'] = s.V_top_R1C_km3.values
                t['flag_tiny'] = na_int8(n); t['quality_class'] = pd.NA
            else:
                assert same(s.V_top_km3, i.V_top_km3) and same(s.A_top_km2, i.A_top_km2), sub
                assert (s.quality_class.values == i.quality_class.values).all(), sub
                t['V_top_tccap_km3'] = np.nan
                t['flag_tiny'] = flagged(s.flag_tiny.astype(bool)); t['quality_class'] = s.quality_class.values
            t['flag_built_before_2000'] = na_int8(n)
            p = part[part.part == c['part']]
            keys = s.Hylak_id.astype(int).values
            # GDW record -> reservoir: same HydroLAKES id, else same GRanD id (gdw_coverage_v1, category 'product')
            grand = {int(gr): int(h) for h, gr in zip(s.Hylak_id, s.Grand_id) if gr > 0} if 'Grand_id' in s else {}
            key_of = pd.Series([int(h) if int(h) in set(keys) else grand.get(int(gr), -1) for h, gr in zip(p.HYLAK_ID, p.GRAND_ID)],
                               index=p.index)
            assert (key_of > 0).all(), (sub, int((key_of <= 0).sum()))
        else:
            assert same(s.V_top_km3, i.V_top_km3) and same(s.A_top_km2, i.A_top_km2), sub
            t['hylak_id'] = s.Hylak_id.astype('Int64'); t['hydrolakes_version'] = np.where(s.Hylak_id.notna(), c['hl'], None)
            t['area_km2'] = s.poly_area_km2.values
            for k in ['H_flat_m', 'H_top_m', 'F_km2', 'R_km2']:
                t[k] = s[k].values
            exact_levels(t, s, [R / c['run'] / k / f'{k}_run.json' for k in s.key], ['H_flat_m', 'H_top_m'], sub)
            for k in ['H_seed_m', 'seed_gap_m', 'D_dam_m', 'W0_km2', 'V_top_tccap_km3']:
                t[k] = np.nan
            t['regime'] = None
            for k in FLAGS[:-1]:
                t[k] = na_int8(n)
            t['flag_built_before_2000'] = flagged(s.flag_built_before_2000.astype(bool))
            assert (s.flag_built_before_2000.astype(bool).values == i.flag_built_before_2000.astype(bool).values).all(), sub
            t['quality_class'] = pd.NA
            p = part[part.part == c['part']]
            # key: the GDW record of the floor, or its HydroLAKES polygon when the floor was built on one
            key_of = np.where(p.GDW_ID.isin(s.GDW_ID.dropna()), 'g' + p.GDW_ID.astype(int).astype(str), 'h' + p.HYLAK_ID.astype('Int64').astype(str))
            key_of = pd.Series(key_of, index=p.index)
            keys = np.where(s.GDW_ID.notna(), 'g' + s.GDW_ID.astype('Int64').astype(str), 'h' + s.Hylak_id.astype('Int64').astype(str))
        g = p.assign(key=key_of.values).sort_values(['key', 'AREA_POLY'], ascending=[True, False]).groupby('key', sort=False)
        ids = g.GDW_ID.apply(lambda x: ';'.join(str(int(v)) for v in x))
        ctry = g.COUNTRY.apply(lambda x: '; '.join(pd.unique(x.dropna())))
        assert set(ids.index) == set(keys) and len(keys) == len(set(keys)), sub
        t['gdw_ids'] = ids.loc[keys].values; t['country'] = ctry.loc[keys].values
        used.append(p.GDW_ID)
        out.append(t[COLS])
    a = pd.concat(out, ignore_index=True)
    used = pd.concat(used)
    assert used.is_unique, 'a GDW record is assigned to two reservoirs'
    return a, part[part.GDW_ID.isin(used)]


def cmd_attributes():
    UP.mkdir(parents=True, exist_ok=True); CHK.mkdir(parents=True, exist_ok=True)
    a, covered = build_attributes()
    num = json.loads((INV / 'numbers.json').read_text(encoding='utf-8'))
    core, recon = a[a.sub_line == 'global'], a[a.product_line.isin(['reconstructed', 'small_tier'])]
    sm, fl = a[a.product_line == 'small_tier'], a[a.product_line == 'direct_floor']
    chk = dict(
        n=len(a), n_by_sub_line=a.sub_line.value_counts().to_dict(),
        gdw_records=int(len(covered)), countries=int(covered.COUNTRY.nunique()),
        table2_core=dict(high=int((core.regime == 'high').sum()), low=int((core.regime == 'low').sum()),
                         **{k: int(core[k].sum()) for k in FLAGS[:7]}),
        seed_gap_reconstruction_lines=[int(recon.flag_seed_gap.sum()), int(len(recon))],
        classes=sm.quality_class.value_counts().sort_index().to_dict(),
        built_before_2000=[int(fl.flag_built_before_2000.sum()), int(len(fl))])
    assert chk['n'] == num['n_total'] == 28481 and chk['n_by_sub_line'] == num['n_by_sub_line']
    assert chk['gdw_records'] == num['coverage']['total']['records'] and chk['countries'] == num['coverage']['countries'] == 163
    assert chk['table2_core'] == dict(high=3485, low=1683, flag_seed_gap=1389, flag_no_dam_height=496, flag_area_beyond=267,
                                      flag_FW0_beyond=219, flag_RW0_beyond=262, flag_Ddam_beyond=72, flag_poly_not_swbd=897), chk['table2_core']
    assert chk['seed_gap_reconstruction_lines'] == [4505, 26457] and chk['classes'] == {'A': 4847, 'B': 7503, 'C': 5147}
    assert chk['built_before_2000'] == [144, 2024]
    assert a.greac_id.tolist() == list(range(1, len(a) + 1)) and a.gdw_ids.str.len().gt(0).all() and a.country.str.len().gt(0).all()
    # levels at full precision: H_top is the elevation of DEM water cells, and rounding it would leave those cells dry
    out = a.copy()
    for k in ['H_flat_m', 'H_seed_m', 'H_top_m']:
        out[k] = a[k].map(lambda v: '' if pd.isna(v) else repr(float(v)))
    out.to_csv(UP / 'greac_attributes.csv', index=False, float_format='%.8g', lineterminator='\n')
    b = pd.read_csv(UP / 'greac_attributes.csv', low_memory=False, float_precision='round_trip')
    assert all(((b[k] == a[k]) | (b[k].isna() & a[k].isna())).all() for k in ['H_flat_m', 'H_seed_m', 'H_top_m'])
    (CHK / 'attributes_check.json').write_text(json.dumps(chk, indent=1), encoding='utf-8', newline='\n')
    print(json.dumps(chk, indent=1))


# ---------------------------------------------------------------- curves
def rec_curves(c, hid, status):
    r1 = pd.read_csv(R / c['r1'] / str(hid) / f'{hid}_aev.csv', usecols=['H_m', 'A_R1C_km2', 'V_R1C_km3'])
    r1 = r1.rename(columns={'A_R1C_km2': 'A_tccap_km2', 'V_R1C_km3': 'V_tccap_km3'})
    if status == 'rescaled':
        r2 = pd.read_csv(R / c['r2'] / str(hid) / f'{hid}_aev_R2.csv').rename(columns={'A_R2_km2': 'A_km2', 'V_R2_km3': 'V_km3'})
    else:
        r2 = r1.rename(columns={'A_tccap_km2': 'A_km2', 'V_tccap_km3': 'V_km3'})
    t = r2.merge(r1, on='H_m', how='outer').sort_values('H_m')
    return t[t[['A_km2', 'V_km3', 'A_tccap_km2', 'V_tccap_km3']].notna().any(axis=1)]


def cmd_curves():
    a = pd.read_csv(UP / 'greac_attributes.csv', low_memory=False, float_precision='round_trip')
    res = {}
    for line in LINES:
        frames = []
        for sub, c in SUB.items():
            if LINE[c['kind']] != line:
                continue
            x = a[a.sub_line == sub]
            if c['kind'] == 'rec':
                status = pd.read_csv(R / c['summary'], usecols=['Hylak_id', 'R2_status']).set_index('Hylak_id').R2_status
                for k, r in enumerate(x.itertuples()):
                    t = rec_curves(c, int(r.hylak_id), status.loc[int(r.hylak_id)])
                    frames.append(t.assign(greac_id=r.greac_id))
                    if k % 1000 == 0:
                        print(line, sub, k, len(x), flush=True)
            elif c['kind'] == 'small':
                s = pd.read_csv(R / c['aev'], usecols=['Hylak_id', 'H_m', 'A_k1_km2', 'V_k1_km3'])
                s = s[s.A_k1_km2.notna() | s.V_k1_km3.notna()]
                gid = x.set_index('hylak_id').greac_id
                assert set(s.Hylak_id) == set(gid.index), sub
                frames.append(pd.DataFrame(dict(greac_id=gid.loc[s.Hylak_id].values, H_m=s.H_m.values, A_km2=s.A_k1_km2.values,
                                                V_km3=s.V_k1_km3.values)))
            else:
                ss = pd.read_csv(R / c['summary']); ss = ss[ss.recommended].reset_index(drop=True)
                for r, key in zip(x.itertuples(), ss.key):
                    frames.append(pd.read_csv(R / c['run'] / key / f'{key}_aev.csv').assign(greac_id=r.greac_id))
            print(line, sub, 'read', flush=True)
        cv = pd.concat(frames, ignore_index=True)
        cols = ['greac_id', 'H_m', 'A_km2', 'V_km3'] + (['A_tccap_km2', 'V_tccap_km3'] if line == 'reconstructed' else [])
        cv = cv[cols].sort_values(['greac_id', 'H_m'], kind='stable')
        top = cv.groupby('greac_id').tail(1).set_index('greac_id')
        x = a[a.product_line == line].set_index('greac_id')
        assert set(top.index) == set(x.index), line
        top = top.loc[x.index]
        dv, da, dh = (top.V_km3 - x.V_top_km3).abs(), (top.A_km2 - x.A_top_km2).abs(), (top.H_m - x.H_top_m).abs()
        ok_v = dv <= np.maximum(1e-9, 1e-5 * x.V_top_km3.abs()); ok_a = da <= np.maximum(1e-6, 1e-5 * x.A_top_km2.abs())
        r = dict(rows=int(len(cv)), reservoirs=int(len(x)), top_V_mismatch=int((~ok_v).sum()), top_A_mismatch=int((~ok_a).sum()),
                 top_H_max_abs_diff_m=float(dh.max()), rows_per_reservoir_median=float(cv.groupby('greac_id').size().median()))
        if line == 'reconstructed':
            tc = cv.dropna(subset=['V_tccap_km3']).groupby('greac_id').tail(1).set_index('greac_id').loc[x.index]
            ok_t = (tc.V_tccap_km3 - x.V_top_tccap_km3).abs() <= np.maximum(1e-9, 1e-5 * x.V_top_tccap_km3.abs())
            r['tccap_top_V_mismatch'] = int((~ok_t).sum())
        res[line] = r
        print(line, r, flush=True)
        assert r['top_V_mismatch'] == 0 and r['top_A_mismatch'] == 0 and r['top_H_max_abs_diff_m'] <= 0.006 and r.get('tccap_top_V_mismatch', 0) == 0
        # the curve files of the runs round H to 6 significant digits; the last row is at the exact H_top of the run record
        last = ~cv.greac_id.duplicated(keep='last')
        cv.loc[last, 'H_m'] = x.H_top_m.loc[cv.greac_id[last]].values
        assert cv.groupby('greac_id').H_m.apply(lambda h: bool(h.is_monotonic_increasing)).all(), line
        cv['H_m'] = cv.H_m.map('{:.3f}'.format)
        cv.to_csv(UP / f'curves_{line}.csv.gz', index=False, float_format='%.6g', lineterminator='\n',
                  compression=dict(method='gzip', compresslevel=9, mtime=0))
    (CHK / 'curves_check.json').write_text(json.dumps(res, indent=1), encoding='utf-8', newline='\n')


# ---------------------------------------------------------------- beds
def place(src, src_t, shape, dst_t):
    """Paste a grid-aligned raster into another grid by its origin (aev_registry_v1.place)."""
    res = dst_t[0]
    r0, c0 = (dst_t[5] - src_t[5]) / res, (src_t[2] - dst_t[2]) / res
    assert abs(r0 - round(r0)) < 1e-6 and abs(c0 - round(c0)) < 1e-6 and abs(src_t[0] - res) < 1e-9
    r0, c0 = int(round(r0)), int(round(c0))
    out = np.full(shape, np.nan)
    rs, cs, re, ce = max(r0, 0), max(c0, 0), min(r0 + src.shape[0], shape[0]), min(c0 + src.shape[1], shape[1])
    out[rs:re, cs:ce] = src[rs - r0:re - r0, cs - c0:ce - c0]
    return out


def bed_surface(c, hid, status):
    """Locked recommended surface on the input grid and the integration extent."""
    with np.load(D / c['inputs'] / f'{hid}_inputs.npz') as d:
        dem, Rm, W0, t = d['fabdem'].astype(float), d['R'].astype(bool), d['W0'].astype(bool), np.asarray(d['transform'], float)
    epsg = int(json.loads((D / c['inputs'] / f'{hid}_inputs.json').read_text(encoding='utf-8'))['epsg'])
    F = r1.flattened_region(dem, W0, Rm)[0]
    p = R / c['r2'] / str(hid) / f'{hid}_bed_R2.tif' if status == 'rescaled' else R / c['r1'] / str(hid) / f'{hid}_bed_R1C.tif'
    with rasterio.open(p) as s:
        b = s.read(1).astype(float); st = s.transform
        assert s.crs.to_epsg() == epsg
    b = place(b, [st.a, st.b, st.c, st.d, st.e, st.f], dem.shape, t)
    z = dem.copy(); v = np.isfinite(b); z[v] = b[v]
    assert not (Rm & ~v & np.isfinite(dem)).any(), (hid, 'locked raster does not cover R')
    E = ndi.binary_dilation(Rm, EIGHT, iterations=1) if c['kind'] == 'small' else Rm
    # the curves keep the wet cells connected to W0 | F (route1_global_run_v1.curves with region = F)
    return z, E, W0 | F, t, epsg, float(F.sum() * 9e-4)


def floor_surface(c, key):
    """Floor surface, R and the seed of its curve (srtm_bed_v1: curve_table(np.where(R, z, inf), seed, {'R': R}))."""
    with np.load(R / c['run'] / key / f'{key}_grid.npz') as d:
        return (d['bed'].astype(float), d['R'].astype(bool), d['seed'].astype(bool), np.asarray(d['transform'], float), int(d['epsg']),
                None)


def write_bed(path, z, E, t, epsg, tags):
    rows, cols = np.flatnonzero(E.any(1)), np.flatnonzero(E.any(0))
    r0, r1, c0, c1 = max(rows[0] - 1, 0), rows[-1] + 2, max(cols[0] - 1, 0), cols[-1] + 2
    a = np.where(E, z, np.nan)[r0:r1, c0:c1].astype('float32')
    with rasterio.open(path, 'w', driver='GTiff', height=a.shape[0], width=a.shape[1], count=1, dtype='float32', nodata=np.nan,
                       crs=f'EPSG:{epsg}', compress='deflate', predictor=3, tiled=True,
                       transform=rasterio.Affine(t[0], 0, t[2] + c0 * t[0], 0, t[4], t[5] + r0 * t[4])) as d:
        d.write(a, 1); d.update_tags(**tags); d.set_band_description(1, 'bed elevation (m above EGM2008)')
    return a


def cmd_beds(part):
    i, n = (int(v) for v in part.split('/'))
    a = pd.read_csv(UP / 'greac_attributes.csv', low_memory=False, float_precision='round_trip')
    status = {sub: pd.read_csv(R / c['summary'], usecols=['Hylak_id', 'R2_status']).set_index('Hylak_id').R2_status
              for sub, c in SUB.items() if c['kind'] != 'floor'}
    fkeys = {}
    for sub, c in SUB.items():
        if c['kind'] == 'floor':
            ss = pd.read_csv(R / c['summary']); ss = ss[ss.recommended].reset_index(drop=True)
            fkeys.update(dict(zip(a[a.sub_line == sub].greac_id, ss.key)))
    x = a.iloc[i - 1::n]
    rows, t0 = [], time.time()
    for k, r in enumerate(x.itertuples()):
        c = SUB[r.sub_line]; folder = BEDS / r.product_line; folder.mkdir(parents=True, exist_ok=True)
        f = folder / f'{r.greac_id}_bed.tif'
        if f.exists():
            with rasterio.open(f) as s:
                z = s.read(1).astype(float)
            vals = z[np.isfinite(z)]; n_ext = n_drop = None
        else:
            if c['kind'] == 'floor':
                zz, E, seed, t, epsg, F_km2 = floor_surface(c, fkeys[r.greac_id])
            else:
                zz, E, seed, t, epsg, F_km2 = bed_surface(c, int(r.hylak_id), status[r.sub_line].loc[int(r.hylak_id)])
                assert abs(F_km2 - r.F_km2) <= max(1e-6, 1e-5 * r.F_km2), (r.greac_id, F_km2, r.F_km2)   # summary: 6 digits
            # cells below H_top that the curve leaves out (not connected to the water surface, e.g. the river below the dam in
            # the widened extent) are set to NoData, so that summing over the valid cells reproduces the curve at H_top
            wet = E & np.isfinite(zz) & (zz <= float(r.H_top_m))
            drop = wet & ~reg.connected_to(wet, seed)
            n_ext, n_drop = int(E.sum()), int(drop.sum())
            E = E & ~drop
            if not E.any():
                # empty maximum extent R (7 reservoirs, zero area and storage at H_top): no bed raster
                assert r.A_top_km2 == 0 and r.V_top_km3 == 0, r.greac_id
                rows.append(dict(greac_id=r.greac_id, product_line=r.product_line, sub_line=r.sub_line, cells_extent=0, cells_dropped=n_drop,
                                 cells_valid=0, V_raster_km3=0.0, A_raster_km2=0.0, V_top_km3=r.V_top_km3, A_top_km2=r.A_top_km2, bytes=0))
                continue
            tags = dict(greac_id=str(r.greac_id), product_line=r.product_line, sub_line=r.sub_line,
                        hylak_id='' if pd.isna(r.hylak_id) else str(int(r.hylak_id)), gdw_ids=r.gdw_ids, H_top_m=f'{r.H_top_m:.3f}',
                        vertical_datum='EGM2008', units='m', dataset='GREAC v1.0')
            z = write_bed(f, zz, E, t, epsg, tags).astype(float)
            vals = z[np.isfinite(z)]
        H = float(r.H_top_m); wet = vals[vals <= H]
        V, A = float(np.sum(H - wet)) * CELL / 1e9, wet.size * CELL / 1e6
        rows.append(dict(greac_id=r.greac_id, product_line=r.product_line, sub_line=r.sub_line, cells_extent=n_ext, cells_dropped=n_drop,
                         cells_valid=int(vals.size),
                         V_raster_km3=V, A_raster_km2=A, V_top_km3=r.V_top_km3, A_top_km2=r.A_top_km2, bytes=f.stat().st_size))
        if k % 500 == 0:
            print(f'part {i}/{n}: {k}/{len(x)} {time.time() - t0:.0f}s', flush=True)
    CHK.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(CHK / f'beds_{i}_of_{n}.csv', index=False, float_format='%.8g', lineterminator='\n')
    print(f'part {i}/{n} done: {len(x)} in {time.time() - t0:.0f}s', flush=True)


# ---------------------------------------------------------------- GeoPackage
def polygons(a):
    """WKB polygons (EPSG:4326) by greac_id: HydroLAKES v1.0, GDW (floors) or the HydroLAKES v1.1 polygons of the runs."""
    geom = {}
    hl10 = a[(a.hydrolakes_version == '1.0') & a.hylak_id.notna()]
    assert hl10.hylak_id.is_unique
    meta, fids, _, f = read(HL, columns=['Hylak_id'], read_geometry=False, return_fids=True)
    assert meta['crs'] in ('EPSG:4326', 'OGC:CRS84'), meta['crs']
    fid_of = pd.Series(fids, index=f[0].astype(int))
    want = hl10.hylak_id.astype(int)
    _, _, g, ff = read(HL, columns=['Hylak_id'], fids=fid_of.loc[want].values)
    geom.update(zip(hl10.set_index(hl10.hylak_id.astype(int)).loc[ff[0].astype(int)].greac_id, g))
    fl = a[(a.product_line == 'direct_floor') & ~a.greac_id.isin(geom)]
    meta, fids, _, f = read(GDW, columns=['GDW_ID'], read_geometry=False, return_fids=True)
    assert meta['crs'] in ('EPSG:4326', 'OGC:CRS84'), meta['crs']
    fid_of = pd.Series(fids, index=f[0].astype(int))
    first = fl.gdw_ids.str.split(';').str[0].astype(int)
    assert fl.gdw_ids.str.contains(';').sum() == 0, 'a floor with several GDW records'
    _, _, g, ff = read(GDW, columns=['GDW_ID'], fids=fid_of.loc[first].values)
    geom.update(zip(fl.set_index(first.values).loc[ff[0].astype(int)].greac_id, g))
    for sub in ['small_hl11_swbd', 'small_hl11_jrc2000']:
        with open(D / SUB[sub]['polygons'], 'rb') as fh:
            pk = pickle.load(fh)
        x = a[a.sub_line == sub]
        geom.update({gid: pk[int(h)] for gid, h in zip(x.greac_id, x.hylak_id)})
    assert set(geom) == set(a.greac_id), len(set(a.greac_id) - set(geom))
    shp = shapely.from_wkb([geom[g] for g in a.greac_id])
    b = shapely.bounds(shp)
    assert (b[:, 0] >= -180).all() and (b[:, 2] <= 180).all() and (b[:, 1] >= -90).all() and (b[:, 3] <= 90).all()
    shp = np.array([s if s.geom_type == 'MultiPolygon' else shapely.MultiPolygon([s]) for s in shp])
    inside = shapely.contains_xy(shapely.buffer(shp, 0.02), a.lon.values, a.lat.values)
    return shp, inside


def cmd_gpkg():
    a = pd.read_csv(UP / 'greac_attributes.csv', low_memory=False, float_precision='round_trip', dtype={'hydrolakes_version': str})
    shp, inside = polygons(a)
    p = UP / 'greac_reservoirs.gpkg'
    if p.exists():
        p.unlink()
    fields, data, mask = [], [], []
    for k in a.columns:
        v = a[k]
        if k in FLAGS or k in ('greac_id', 'hylak_id'):
            m = v.isna().values; data.append(v.fillna(0).astype('int64').values)
        elif not pd.api.types.is_numeric_dtype(v):          # pandas 3 reads text as 'str', not object
            m = v.isna().values; data.append(v.fillna('').astype(str).values.astype(object))
        else:
            m = v.isna().values; data.append(v.astype(float).values)
        fields.append(k); mask.append(m)
    write(str(p), shapely.to_wkb(shp), data, fields, field_mask=mask, layer='reservoirs', driver='GPKG', crs='EPSG:4326',
          geometry_type='MultiPolygon')
    pts = shapely.to_wkb(shapely.points(a.lon.values, a.lat.values))
    write(str(p), pts, [a.greac_id.values.astype('int64')], ['greac_id'], layer='outlets', driver='GPKG', crs='EPSG:4326',
          geometry_type='Point', append=True)
    r = dict(reservoirs=int(len(a)), outlet_within_0p02deg_of_polygon=int(inside.sum()),
             outlet_far_from_polygon=a.loc[~inside, ['greac_id', 'sub_line']].astype(str).values.tolist()[:50], bytes=p.stat().st_size)
    (CHK / 'gpkg_check.json').write_text(json.dumps(r, indent=1), encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in r.items() if k != 'outlet_far_from_polygon'}), flush=True)


# ---------------------------------------------------------------- finish
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def cmd_finish():
    a = pd.read_csv(UP / 'greac_attributes.csv', low_memory=False, float_precision='round_trip')
    chk = pd.concat([pd.read_csv(f) for f in sorted(CHK.glob('beds_*_of_*.csv'))], ignore_index=True)
    assert chk.greac_id.is_unique and set(chk.greac_id) == set(a.greac_id), 'bed check incomplete'
    rel = (chk.V_raster_km3 - chk.V_top_km3) / chk.V_top_km3.where(chk.V_top_km3 > 0)
    beds = dict(n=int(len(chk)), V_rel_diff_median=float(rel.abs().median()),
                V_rel_diff_over_1pct=int((rel.abs() > 0.01).sum()), V_rel_diff_over_1pct_by_line=chk[rel.abs() > 0.01].product_line.value_counts().to_dict(),
                V_raster_below_curve=int((rel < -1e-5).sum()), zero_storage=int((chk.V_top_km3 <= 0).sum()),
                no_raster_empty_extent=chk[chk.cells_extent == 0].greac_id.astype(int).tolist())
    beds['V_rel_diff_max'] = float(rel.abs().max())
    assert beds['V_rel_diff_median'] < 1e-5 and beds['V_rel_diff_max'] < 1e-3, beds
    empty = set(beds['no_raster_empty_extent'])
    # archives of at most ZIP_MAX bytes (user 2026-11-10: a 1.08 GB file may break off in the browser upload),
    # beds_<line>_<k>.zip in greac_id order
    for old in UP.glob('beds_*.zip'):
        old.unlink()
    zips = {}
    for line in LINES:
        files = sorted((BEDS / line).glob('*_bed.tif'), key=lambda f: int(f.name.split('_')[0]))
        assert len(files) == int(((a.product_line == line) & ~a.greac_id.isin(empty)).sum()), line
        groups, cur, size = [], [], 0
        for f in files:
            s = f.stat().st_size
            if cur and size + s > ZIP_MAX:
                groups.append(cur); cur, size = [], 0
            cur.append(f); size += s
        groups.append(cur)
        zips[line] = []
        for k, g in enumerate(groups, 1):
            z = UP / f'beds_{line}_{k}.zip'
            with zipfile.ZipFile(z, 'w', compression=zipfile.ZIP_STORED) as zf:
                for f in g:
                    zf.write(f, f'beds/{line}/{f.name}')
            zips[line].append(dict(name=z.name, files=len(g), greac_id_first=int(g[0].name.split('_')[0]),
                                   greac_id_last=int(g[-1].name.split('_')[0]), bytes=z.stat().st_size))
            print('zipped', zips[line][-1], flush=True)
    up = sorted(f for f in UP.iterdir() if f.is_file() and f.name != 'MANIFEST.sha256')
    (UP / 'MANIFEST.sha256').write_text(''.join(f'{sha(f)}  {f.name}\n' for f in up), encoding='utf-8', newline='\n')
    res = dict(files={f.name: f.stat().st_size for f in sorted(UP.iterdir())}, total_bytes=int(sum(f.stat().st_size for f in UP.iterdir())),
               beds=beds, zips=zips, attributes=json.loads((CHK / 'attributes_check.json').read_text(encoding='utf-8')),
               curves=json.loads((CHK / 'curves_check.json').read_text(encoding='utf-8')),
               gpkg={k: v for k, v in json.loads((CHK / 'gpkg_check.json').read_text(encoding='utf-8')).items() if k != 'outlet_far_from_polygon'})
    (CHK / 'package_check.json').write_text(json.dumps(res, indent=1), encoding='utf-8', newline='\n')
    print(json.dumps({k: res[k] for k in ['files', 'total_bytes', 'beds']}, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['attributes', 'curves', 'beds', 'gpkg', 'finish'])
    p.add_argument('--part', default='1/1')
    a = p.parse_args()
    dict(attributes=cmd_attributes, curves=cmd_curves, beds=lambda: cmd_beds(a.part), gpkg=cmd_gpkg, finish=cmd_finish)[a.cmd]()


if __name__ == '__main__':
    main()
