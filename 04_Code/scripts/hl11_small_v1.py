"""Small GDW reservoirs whose polygons exist only in HydroLAKES v1.1 (plan docs/plans/2026-11-08-hl11-small-plan.md,
user approval 2026-11-08). The 4,696 records excluded from the small-reservoir tier as "HydroLAKES v1.1 only" are run
with the frozen code and rules of the tier; only the inputs change:
  polygon   the GDW v1.0 reservoir polygon (union of the GDW records that share the HYLAK_ID)
  outlet    the GDW river-snapped dam point (LONG_RIV, LAT_RIV), the pixel of highest flow accumulation in the polygon
  others    HydroLAKES v1.0 polygons as in the tier, plus the other GDW polygons in the window (overlap with the own
            polygon < 50 % of either area, as srtm_bed_v1.Polys.others)
  GLOBathy  no parameters for these polygons: GLOBathy-est (comparison only) is replaced by a flat placeholder at the
            ring elevation, written as {hid}_GLOBathy_est.tif so that the frozen readers work; TC-cap and TC-dam do not
            use it (checked by `repro` on 20 locked tier reservoirs); GLOBathy-est and switching columns are blanked
Routing: SWBD water inside the polygon -> small tier (TC-cap, TC-dam, R widened by k = 1, labels and classes);
         none -> SRTM-recorded floor (srtm_bed_v3.run_one on the GDW polygon), flagged flag_small_direct.
Subcommands:
  list                 list, polygons, FABDEM tiles                 repro [--n 20]   GLOBathy placeholder reproduces the lock
  run [--part i/n]     tier runs (gr.run_one, r2.run_one)           retry-edge | lock | widen [--part] | widen-lock
  tier-summary         tier summary with labels, G and classes      floor-list | floor-run [--part] | floor-lock | floor-summary
  coverage             GDW coverage with this batch (base: small_tier_v1 gdw_part_with_tier.csv)
Outputs: 05_Results/by_date/2026-11-08/hl11_small_v1/  (inputs in 03_Data_Processed/hl11_small_20261108/)
Usage: .venv/Scripts/python.exe -I 04_Code/scripts/hl11_small_v1.py <cmd> [...]
"""
from pathlib import Path
import sys, json, argparse, datetime, warnings, traceback, time, pickle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import shapely, shapely.ops
from shapely.strtree import STRtree
from pyogrio.raw import read
from pyproj import Transformer
from rasterio.features import rasterize
from scipy import ndimage as ndi
import small_tier_v1 as st          # sets the frozen modules' paths to the tier; redirected below
import route1_global_run_v1 as gr
import r2_global_run_v1 as r2
import srtm_bed_v1 as sb
import srtm_bed_v3 as s3

ROOT = gr.ROOT
PLAN = ROOT / 'docs/plans/2026-11-08-hl11-small-plan.md'
TIER = ROOT / '05_Results/by_date/2026-10-28/small_tier_v1'
TIER_DATA = ROOT / '03_Data_Processed/small_tier_20261028'
OUT = ROOT / '05_Results/by_date/2026-11-08/hl11_small_v1'
DATA = ROOT / '03_Data_Processed/hl11_small_20261108'
GDW_SHP = ROOT / '02_Data_Raw/GDW_v1_0_20261027/extracted/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp'
REASON = 'HydroLAKES v1.1 only (no GLOBathy parameters)'
EIGHT = np.ones((3, 3), bool)
GEST_COLS = ['V_top_Gest_km3', 'A_top_Gest_km2', 'max_depth_Gest_m', 'V_top_C2_km3', 'A_top_C2_km2', 'max_depth_C2_m',
             'Dmax_est_GLOBathy_m', 'H_surf_GLOBathy_m']


def redirect(out, data):
    st.OUT, st.DATA, st.WID, st.SPEC = out, data, out / 'widen', PLAN
    gr.OUT, gr.RUN, gr.DATA, gr.INPUTS = out, out / 'run', data, data / 'inputs'


redirect(OUT, DATA)
st.GUARDED = st.GUARDED + [HERE / 'hl11_small_v1.py', PLAN]
s3.OUT = OUT / 'floor'


# ---------------------------------------------------------------- GLOBathy placeholder and GDW others
def no_globathy(hid, poly_wgs, g, epsg, folder, dm):
    """Flat placeholder at the ring elevation of the own polygon; not a product, never reported."""
    P = g['P'].astype(bool)
    ring = ndi.binary_dilation(P, EIGHT) & ~P & np.isfinite(g['dem'])
    H = float(np.nanmedian(g['dem'][ring])) if ring.any() else float(np.nanmedian(g['dem'][P]))
    z = np.full(P.shape, np.nan); z[P] = H
    gr.erun.md.write(folder, f'{hid}_GLOBathy_est.tif', z, g, epsg)
    return z, dict(Hylak_id=hid, Dmax_est_PAVEW_m=float('nan'), Dmax_use_m=float('nan'), uses_observed_Dmax=False, H_surf_m=H,
                   polygon_km2_on_grid=float(P.sum() * 9e-4), placeholder='no GLOBathy parameters (HydroLAKES v1.1 polygon)')


gr.globathy_est = no_globathy
_ORIG_MASKS = gr.masks_on_grid
_GDW = {}


def gdw_index():
    if not _GDW:
        meta, _, geoms, f = read(GDW_SHP, columns=['GDW_ID'])
        _GDW['poly'] = shapely.make_valid(shapely.from_wkb(geoms))
        _GDW['tree'] = STRtree(_GDW['poly'])
    return _GDW


def masks_with_gdw(hid, P, epsg, tr, shape, lon_bounds):
    """route1_global_run_v1.masks_on_grid plus the other GDW polygons in the window."""
    Pm, other = _ORIG_MASKS(hid, P, epsg, tr, shape, lon_bounds)
    ix = gdw_index()
    to = Transformer.from_crs(4326, epsg, always_xy=True).transform
    oth = []
    for p in ix['poly'][ix['tree'].query(shapely.box(*lon_bounds))]:
        q = shapely.make_valid(shapely.ops.transform(to, p))
        inter = q.intersection(P).area if q.intersects(P) else 0.0
        if inter < 0.5 * min(q.area, P.area):
            oth.append((q, 1))
    if oth:
        other = other | (rasterize(oth, out_shape=shape, transform=tr, dtype='uint8').astype(bool) & ~Pm)
    return Pm, other


# ---------------------------------------------------------------- list
def cmd_list():
    ex = pd.read_csv(TIER / 'small_excluded.csv')
    ex = ex[ex.excluded.eq(REASON)]
    cols = ['GDW_ID', 'RES_NAME', 'DAM_NAME', 'COUNTRY', 'YEAR_DAM', 'DAM_HGT_M', 'AREA_POLY', 'CAP_MCM', 'CAP_REP', 'CAP_MAX',
            'QUALITY', 'ORIG_SRC', 'POLY_SRC', 'LONG_RIV', 'LAT_RIV', 'HYLAK_ID']
    meta, _, geoms, f = read(GDW_SHP, columns=cols)
    g = pd.DataFrame(dict(zip(meta['fields'], f))); g['wkb'] = list(geoms)
    g = g[g.GDW_ID.isin(ex.GDW_ID)].merge(ex[['GDW_ID', 'category']], on='GDW_ID').sort_values('GDW_ID')
    assert len(g) == len(ex) == 4696, (len(g), len(ex))
    rows, polys = [], {}
    for hid, d in g.groupby('HYLAK_ID'):
        P = shapely.make_valid(shapely.union_all(shapely.make_valid(shapely.from_wkb(list(d.wkb)))))
        r = d.iloc[0]
        b = P.bounds
        name = next((x for x in (r.RES_NAME, r.DAM_NAME) if isinstance(x, str) and x.strip()), '')
        rows.append(dict(Hylak_id=int(hid), Lake_name=name, Country=r.COUNTRY, Continent=np.nan, Poly_src=r.POLY_SRC,
                         Lake_type=np.nan, Grand_id=0, Lake_area=float(d.AREA_POLY.sum()), Vol_total=np.nan, Vol_res=np.nan,
                         Vol_src=np.nan, Depth_avg=np.nan, Elevation=np.nan, Pour_long=float(r.LONG_RIV), Pour_lat=float(r.LAT_RIV),
                         lon_min=b[0], lat_min=b[1], lon_max=b[2], lat_max=b[3], n_gdw_records=int(len(d)),
                         GDW_IDs=';'.join(str(int(i)) for i in d.GDW_ID),
                         **{'GDW_' + k: r[k] for k in st.GDW if k in r.index}))
        rows[-1]['GDW_AREA_POLY'] = float(d.AREA_POLY.sum())     # same area as Lake_area: the HL/GDW mismatch label does not apply
        polys[int(hid)] = shapely.to_wkb(P)
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    out_lat = (t.Pour_lat < -56) | (t.Pour_lat >= 60)
    t.loc[out_lat, ['Hylak_id']].assign(reason='outlet outside SWBD coverage (56 S - 60 N)').to_csv(OUT / 'excluded.csv', index=False)
    t = t[~out_lat]
    t.to_csv(OUT / 'small_list.csv', index=False, float_format='%.10g')
    with open(DATA / 'polygons.pkl', 'wb') as fh:
        pickle.dump({h: polys[h] for h in t.Hylak_id}, fh)
    print('GDW records', len(g), '| HYLAK_IDs', len(rows), '| outside 56S-60N', int(out_lat.sum()), '| listed', len(t), flush=True)
    st.extract_tiles(t)


# ---------------------------------------------------------------- reproduction check
def cmd_repro(a):
    """Run n locked tier reservoirs with the placeholder (no GDW others) in a separate folder; compare with the lock."""
    head = st.committed()
    rep, rdata = OUT / 'repro', DATA / 'repro'
    redirect(rep, rdata); st.DATA = TIER_DATA                     # polygons from the tier, inputs rebuilt in rdata
    gr.INPUTS = rdata / 'inputs'
    lst = pd.read_csv(TIER / 'small_list.csv')
    lat = (lst.lat_min + lst.lat_max) / 2
    lst['est'] = ((lst.lon_max - lst.lon_min) * 111.32 * np.cos(np.radians(lat)) + 30) * ((lst.lat_max - lst.lat_min) * 111.32 + 30)
    ok = [int(p.name) for p in (TIER / 'run').iterdir() if (p / f'{p.name}_R2_run.json').exists()]
    d = lst[lst.Hylak_id.isin(ok)].sort_values(['est', 'Hylak_id']).head(a.n)
    with open(TIER_DATA / 'polygons.pkl', 'rb') as fh:
        polys = pickle.load(fh)
    dm = gr.erun.dmax_all()
    rows = []
    for r in d.itertuples():
        hid = int(r.Hylak_id)
        rec = gr.run_one(hid, shapely.from_wkb(polys[hid]), dm, 15000.0)
        rec2 = r2.run_one(r, gr.RUN, head, recheck=False)
        L = json.loads((TIER / 'run' / str(hid) / f'{hid}_run.json').read_text(encoding='utf-8'))
        L2 = json.loads((TIER / 'run' / str(hid) / f'{hid}_R2_run.json').read_text(encoding='utf-8'))
        row = dict(Hylak_id=hid)
        for k, a_, b_ in [('H_flat', rec['H_flat_m'], L['H_flat_m']), ('H_seed', rec['H_seed_m'], L['H_seed_m']),
                          ('H_top', rec['H_top_m'], L['H_top_m']), ('F_km2', rec['F_km2'], L['F_km2']),
                          ('V_TCcap', rec['methods']['R1C']['V_top_km3'], L['methods']['R1C']['V_top_km3']),
                          ('D_dam', rec2['D_dam_m'], L2['D_dam_m']), ('V_TCdam', rec2['R2']['V_top_km3'], L2['R2']['V_top_km3'])]:
            row[k] = a_; row[k + '_locked'] = b_
            row[k + '_same'] = bool((np.isnan(a_) and np.isnan(b_)) or abs(a_ - b_) <= 1e-9 * max(1.0, abs(b_)))
        row['inputs_same'] = rec['inputs_sha256'] == L['inputs_sha256']
        rows.append(row)
        print(hid, 'repro', all(v for k, v in row.items() if k.endswith('_same')), 'inputs', row['inputs_same'], flush=True)
    t = pd.DataFrame(rows); t.to_csv(OUT / 'repro_check.csv', index=False, float_format='%.10g')
    same = t.filter(like='_same')
    res = dict(n=int(len(t)), all_identical=bool(same.all().all()), mismatches={c: int((~same[c]).sum()) for c in same.columns},
               inputs_identical=int(t.inputs_same.sum()), git_head=head, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (OUT / 'repro_check.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1), flush=True)


# ---------------------------------------------------------------- tier part
def production():
    assert (OUT / 'repro_check.json').exists() and json.loads((OUT / 'repro_check.json').read_text(encoding='utf-8'))['all_identical'], \
        'reproduction check missing or failed'
    gr.masks_on_grid = masks_with_gdw


def cmd_tier_summary(a):
    st.coverage = lambda s: None                     # coverage of this batch: `coverage`
    st.cmd_summary(a)
    s = pd.read_csv(OUT / 'small_tier_summary.csv')
    for c in GEST_COLS:
        if c in s.columns:
            s[c] = np.nan
    s['tier'] = 'small (< 1 km2), HydroLAKES v1.1 polygon from GDW'
    s['seed_gap_m'] = s.H_seed_m - s.H_flat_m
    s['flag_seed_gap'] = s.seed_gap_m >= 5
    s.to_csv(OUT / 'hl11_tier_summary.csv', index=False, float_format='%.6g')
    (OUT / 'small_tier_summary.csv').unlink()
    print('seed gap >= 5 m', int(s.flag_seed_gap.sum()), 'of', len(s), flush=True)


# ---------------------------------------------------------------- floor part
def cmd_floor_list():
    lst = pd.read_csv(OUT / 'small_list.csv')
    rows = []
    for r in lst.itertuples():
        e = gr.RUN / str(r.Hylak_id) / f'{r.Hylak_id}_error.json'
        if e.exists() and 'no SWBD water' in json.loads(e.read_text(encoding='utf-8'))['error']:
            rows.append(dict(key=f'g{int(r.GDW_GDW_ID)}', source='GDW', GDW_ID=int(r.GDW_GDW_ID), Hylak_id=int(r.Hylak_id), Grand_id=0,
                             country=r.Country, poly_area_km2=r.Lake_area, year_dam=r.GDW_YEAR_DAM, validation=''))
    t = pd.DataFrame(rows)
    s3.OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(s3.OUT / 'target_list.csv', index=False)
    print('floor targets (no SWBD water in the polygon)', len(t), flush=True)


def cmd_floor_run(a):
    warnings.filterwarnings('ignore')
    st.committed()
    t = pd.read_csv(s3.OUT / 'target_list.csv').sort_values('poly_area_km2')
    if a.part:
        i, n = map(int, a.part.split('/')); t = t.iloc[i - 1::n]
    polys, bars = sb.Polys(), s3.Barriers()
    for key in t.key:
        f = s3.OUT / 'run' / key
        if (f / f'{key}_run.json').exists() or (f / f'{key}_error.json').exists():
            continue
        t0 = time.time()
        try:
            r = s3.run_one(key, polys, bars)
            print(key, 'floor ok', f"R {r['R_km2']:.2f} F {r['F_km2']:.2f} V {r['V_top_km3']:.4g}", f'{time.time() - t0:.0f}s', flush=True)
        except Exception as exc:
            if isinstance(exc, MemoryError) or 'allocat' in repr(exc):
                print(key, 'RESOURCE', repr(exc)[:120], flush=True); continue
            f.mkdir(parents=True, exist_ok=True)
            (f / f'{key}_error.json').write_text(json.dumps(dict(error=repr(exc), trace=traceback.format_exc())), encoding='utf-8')
            print(key, 'FAILED floor', repr(exc)[:160], flush=True)


def cmd_floor_lock():
    head = st.committed(); t = pd.read_csv(s3.OUT / 'target_list.csv')
    runs, failed, missing = {}, {}, []
    for key in t.key:
        f = s3.OUT / 'run' / key
        if (f / f'{key}_run.json').exists():
            runs[key] = {p.name: gr.sha(p) for p in sorted(f.glob('*')) if p.suffix in ('.tif', '.csv', '.json', '.npz')}
        elif (f / f'{key}_error.json').exists():
            failed[key] = json.loads((f / f'{key}_error.json').read_text(encoding='utf-8'))['error'][:300]
        else:
            missing.append(key)
    m = dict(plan=str(PLAN.relative_to(ROOT)).replace(chr(92), '/'), plan_sha256=gr.sha(PLAN), git_head=head,
             scripts={n: gr.sha(HERE / n) for n in ['hl11_small_v1.py', 'srtm_bed_v1.py', 'srtm_bed_v3.py']},
             list_sha256=gr.sha(s3.OUT / 'target_list.csv'), n_listed=int(len(t)), n_ok=len(runs), n_failed=len(failed), missing=missing,
             failed=failed, runs=runs, locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (s3.OUT / 'lock_manifest.json').write_text(json.dumps(m, indent=0), encoding='utf-8')
    print('FLOOR LOCKED ok', len(runs), 'failed', len(failed), 'missing', len(missing), flush=True)


def cmd_floor_summary():
    t = pd.read_csv(s3.OUT / 'target_list.csv'); rows = []
    for key in t.key:
        p = s3.OUT / 'run' / key / f'{key}_run.json'
        if p.exists():
            r = json.loads(p.read_text(encoding='utf-8')); r.pop('missing_tiles', None); r.pop('offset_iqr_m', None); rows.append(r)
    s = t.merge(pd.DataFrame(rows), on='key')
    # srtm_bed_v3.cmd_summary rules, unchanged
    s['R_over_polygon'] = s.R_km2 / s.polygon_km2
    s['flag_R_beyond_polygon'] = s.R_over_polygon > 3.31
    s['flag_w2000_ge_5pct'] = s.w2000_share_R >= 0.05
    s['flag_curve_top_from_nasadem'] = s.ring_nasadem_share.fillna(0) >= 0.5
    s['A_top_over_R'] = s.A_top_km2 / s.R_km2
    s['flag_top_area_below_half_R'] = s.A_top_over_R < 0.5
    s['recommended'] = ~s.flag_R_beyond_polygon & ~s.flag_top_area_below_half_R
    s['flag_small_direct'] = True
    s.to_csv(OUT / 'hl11_floor_summary.csv', index=False, float_format='%.10g')
    print('floor targets', len(t), '| with curves', len(s), '| recommended', int(s.recommended.sum()),
          '| flags', s.filter(like='flag_').sum().to_dict(), flush=True)


# ---------------------------------------------------------------- coverage
def cmd_coverage():
    p = pd.read_csv(TIER / 'gdw_part_with_tier.csv')
    lst = pd.read_csv(OUT / 'small_list.csv')
    ids = {int(h): [int(x) for x in str(s).split(';')] for h, s in zip(lst.Hylak_id, lst.GDW_IDs)}
    tier = pd.read_csv(OUT / 'hl11_tier_summary.csv')
    fl = pd.read_csv(OUT / 'hl11_floor_summary.csv')
    tier_ids = {g for h in tier.Hylak_id for g in ids[int(h)]}
    floor_ids = {g for h in fl.Hylak_id[fl.recommended] for g in ids[int(h)]}
    cap = p.CAP_MCM.clip(lower=0); A, V = p.AREA_POLY.sum(), cap.sum()
    before = p.part.ne('none')
    assert not p.loc[p.GDW_ID.isin(tier_ids | floor_ids), 'part'].ne('none').any(), 'a record of this batch already has a product'
    p.loc[p.GDW_ID.isin(tier_ids), 'part'] = 'hl11_small_tier'
    p.loc[p.GDW_ID.isin(floor_ids), 'part'] = 'hl11_direct_floor'
    cov = p.part.ne('none')
    out = dict(n_records=int(len(p)), before=dict(n=int(before.sum()), area_share=float(p.AREA_POLY[before].sum() / A),
                                                   storage_share=float(cap[before].sum() / V)),
               batch=dict(tier_reservoirs=int(len(tier)), tier_records=len(tier_ids), floor_reservoirs_recommended=int(fl.recommended.sum()),
                          floor_records=len(floor_ids)),
               after=dict(n=int(cov.sum()), area_share=float(p.AREA_POLY[cov].sum() / A), storage_share=float(cap[cov].sum() / V)),
               no_product=dict(n=int((~cov).sum()), area_share=float(p.AREA_POLY[~cov].sum() / A), storage_share=float(cap[~cov].sum() / V)),
               finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    p.to_csv(OUT / 'gdw_part_with_hl11.csv', index=False)
    (OUT / 'coverage.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps(out, indent=1), flush=True)


def main():
    warnings.filterwarnings('ignore')
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['list', 'repro', 'run', 'retry-edge', 'lock', 'widen', 'widen-lock', 'tier-summary',
                                   'floor-list', 'floor-run', 'floor-lock', 'floor-summary', 'coverage'])
    p.add_argument('--part'); p.add_argument('--ids', nargs='*', type=int); p.add_argument('--k', type=int, default=st.K_PRODUCT)
    p.add_argument('--n', type=int, default=20)
    a = p.parse_args()
    if a.cmd in ('run', 'retry-edge'):
        production()
    {'list': cmd_list, 'repro': lambda: cmd_repro(a), 'run': lambda: st.cmd_run(a), 'retry-edge': lambda: st.cmd_retry_edge(a),
     'lock': st.cmd_lock, 'widen': lambda: st.cmd_widen(a), 'widen-lock': st.cmd_widen_lock, 'tier-summary': lambda: cmd_tier_summary(a),
     'floor-list': cmd_floor_list, 'floor-run': lambda: cmd_floor_run(a), 'floor-lock': cmd_floor_lock,
     'floor-summary': cmd_floor_summary, 'coverage': cmd_coverage}[a.cmd]()


if __name__ == '__main__':
    main()
