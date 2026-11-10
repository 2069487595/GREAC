"""Small-reservoir tier (spec docs/plans/2026-10-28-small-tier-spec.md): the GDW v1.0 reservoirs under 1 km2 without a
product, run on their HydroLAKES v1.0 polygon with the frozen global code (as gdw_ext_v1: only the module paths of
route1_global_run_v1 are redirected); after the lock, R widened by k = 0-3 cells for the TC-dam volume integral (product
k = 1, small_res_v1 fixtest); labels; checks on the 63 official references (small_res_v1) and on US NID normal capacities.

Subcommands:
  list        list and excluded records; FABDEM tiles extracted from the local archive (nothing is downloaded)
  run         route1_global_run_v1.run_one then r2_global_run_v1.run_one          [--part i/n] [--ids ...]
  retry-edge  failures because F reached the grid border, inputs rebuilt with the 45 km margin cap, run again
  lock        lock_manifest.json (SHA-256 of every run output)
  widen       TC-dam curves with R widened by k = 0..3 cells (k = 0 must reproduce the lock)    [--part i/n]
  widen-lock  widen_per_reservoir.csv, small_aev.csv.gz, widen_manifest.json
  summary     small_tier_summary.csv (product k = 1, or --k), labels, small_failed.csv, GDW coverage with the tier
  check       check_reference_63.csv, check_nid.csv, check_summary.json (spec section 4 decision)
Usage: .venv/Scripts/python.exe -I 04_Code/scripts/small_tier_v1.py <cmd> [...]
"""
from pathlib import Path
import sys, json, argparse, datetime, warnings, traceback, time, pickle, math, shutil, zipfile, subprocess

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import shapely
from pyogrio.raw import read
from scipy import ndimage as ndi
import route1_global_run_v1 as gr
import r2_global_run_v1 as r2
import route1_global_summary_v1 as gs
import route1_global_tiles_v1 as gt

ROOT = gr.ROOT
SPEC = ROOT / 'docs/plans/2026-10-28-small-tier-spec.md'
CAT = ROOT / '05_Results/by_date/2026-10-27/gdw_coverage_v1/gdw_reservoirs_category.csv'
PART = ROOT / '05_Results/by_date/2026-10-27/dataset_coverage_v3/gdw_part.csv'
COV3 = ROOT / '05_Results/by_date/2026-10-27/dataset_coverage_v3/coverage.json'
SR = ROOT / '05_Results/by_date/2026-10-27/small_res_v1'
OUT = ROOT / '05_Results/by_date/2026-10-28/small_tier_v1'
DATA = ROOT / '03_Data_Processed/small_tier_20261028'
WID = OUT / 'widen'
VR = ROOT / '05_Results/by_date/2026-10-07/route1_global_v1/validated_range.json'
VR2 = ROOT / '05_Results/by_date/2026-10-10/r2_global_v1/validated_range_R2.json'
COLS = ['Hylak_id', 'Lake_name', 'Country', 'Continent', 'Poly_src', 'Lake_type', 'Grand_id', 'Lake_area', 'Vol_total',
        'Vol_res', 'Vol_src', 'Depth_avg', 'Elevation', 'Pour_long', 'Pour_lat']
GDW = ['GDW_ID', 'RES_NAME', 'DAM_NAME', 'COUNTRY', 'YEAR_DAM', 'DAM_HGT_M', 'AREA_POLY', 'CAP_MCM', 'CAP_REP', 'CAP_MAX',
       'QUALITY', 'ORIG_SRC', 'POLY_SRC', 'LONG_RIV', 'LAT_RIV', 'category']
KS = (0, 1, 2, 3)
K_PRODUCT = 1
EIGHT = np.ones((3, 3), bool)
NO_DAM = ('fallback_undefined', 'fallback_nonpositive', 'failed')
GUARDED = [HERE / 'small_tier_v1.py', SPEC]
sha = gr.sha

# the frozen modules read these paths when their functions run
gr.OUT, gr.RUN, gr.DATA, gr.INPUTS = OUT, OUT / 'run', DATA, DATA / 'inputs'


def committed():
    head = r2.committed()              # the frozen global code and the R2 spec, tracked and unchanged
    for f in GUARDED:
        rel = str(f.relative_to(ROOT)).replace(chr(92), '/')
        if subprocess.run(['git', 'status', '--porcelain', '--', rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() or \
                subprocess.run(['git', 'ls-files', '--error-unmatch', rel], cwd=ROOT, capture_output=True).returncode != 0:
            raise SystemExit(f'{rel} is not committed')
    return head


def extract_tiles(d):
    """gdw_ext_v1.extract_tiles: 1-degree FABDEM tiles within 0.3 degrees of a polygon, from the local archive."""
    need = {(la, lo) for r in d.itertuples()
            for la in range(math.floor(r.lat_min - gt.PAD), math.floor(r.lat_max + gt.PAD) + 1)
            for lo in range(math.floor(r.lon_min - gt.PAD), math.floor(r.lon_max + gt.PAD) + 1)}
    have = {p.name[:7] for p in gt.CACHE.glob('*_FABDEM_V1-2.tif')}
    rows = []
    for la, lo in sorted(need):
        name = f'{gt.tag(la, lo)}_FABDEM_V1-2.tif'
        if gt.tag(la, lo) in have:
            rows.append(dict(tile=gt.tag(la, lo), status='cached')); continue
        z = gt.archive(la, lo)
        if not z.exists():
            rows.append(dict(tile=gt.tag(la, lo), status='no archive (no land)')); continue
        with zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist() if Path(n).name == name]
            if not names:
                rows.append(dict(tile=gt.tag(la, lo), status='not in archive (no land)')); continue
            tmp = gt.CACHE / (name + '.part')
            with zf.open(names[0]) as s, tmp.open('wb') as o:
                shutil.copyfileobj(s, o, 8 * 1024 * 1024)
            tmp.replace(gt.CACHE / name)
        rows.append(dict(tile=gt.tag(la, lo), status='extracted'))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'fabdem_tiles.csv', index=False)
    print(t.status.value_counts().to_string(), flush=True)


def products():
    ids = set()
    for p in ['2026-10-07/route1_global_v1/global_summary.csv', '2026-10-27/gdw_ext_v1/ext_summary.csv',
              '2026-10-27/highlat_v1/hl_list.csv', '2026-10-27/remnant_fix_v1/fix_list.csv']:
        ids |= set(pd.read_csv(ROOT / '05_Results/by_date' / p, usecols=['Hylak_id']).Hylak_id.astype(int))
    return ids


def cmd_list():
    c = pd.read_csv(CAT, low_memory=False)
    p = pd.read_csv(PART)
    rec = c[c.GDW_ID.isin(p.GDW_ID[p.part.eq('none')]) & ((c.AREA_POLY < 1) | c.category.eq('hl2_small'))].copy()
    ids = sorted(rec.HYLAK_ID[rec.HYLAK_ID > 0].astype(int).unique())
    meta, _, geoms, f = read(gr.GPKG, columns=COLS, where=f"Hylak_id IN ({','.join(map(str, ids))})")
    d = pd.DataFrame(dict(zip(meta['fields'], f)))
    wkb = dict(zip(d.Hylak_id.astype(int), geoms))
    d[['lon_min', 'lat_min', 'lon_max', 'lat_max']] = shapely.bounds(shapely.from_wkb(geoms))
    d['excluded'] = np.where((d.Pour_lat < -56) | (d.Pour_lat >= 60), 'outside SWBD coverage (56 S - 60 N)', '')
    inprod = d.Hylak_id.isin(products())
    d.loc[inprod, 'excluded'] = 'HydroLAKES polygon already in a product'
    rec['excluded'] = np.where(rec.HYLAK_ID <= 0, 'no HydroLAKES polygon',
                      np.where(~rec.HYLAK_ID.isin(d.Hylak_id), 'HydroLAKES v1.1 only (no GLOBathy parameters)', ''))
    ex = d.set_index('Hylak_id').excluded
    m = rec.excluded.eq('') & rec.HYLAK_ID.map(ex).ne('')
    rec.loc[m, 'excluded'] = rec.HYLAK_ID[m].map(ex)
    g = rec[rec.excluded.eq('')].sort_values('GDW_ID')
    n_gdw = g.groupby('HYLAK_ID').size().rename('n_gdw_records')
    g = g.drop_duplicates('HYLAK_ID')[['HYLAK_ID'] + GDW].rename(columns={'HYLAK_ID': 'Hylak_id', **{k: 'GDW_' + k for k in GDW}})
    keep = d[d.excluded.eq('')].drop(columns='excluded').merge(g, on='Hylak_id', how='left').merge(
        n_gdw.rename_axis('Hylak_id').reset_index(), on='Hylak_id', how='left').sort_values('Hylak_id')
    OUT.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    keep.to_csv(OUT / 'small_list.csv', index=False, float_format='%.10g')
    rec[rec.excluded.ne('')][['GDW_ID', 'HYLAK_ID', 'COUNTRY', 'AREA_POLY', 'CAP_MCM', 'category', 'excluded']].to_csv(
        OUT / 'small_excluded.csv', index=False, float_format='%.10g')
    with open(DATA / 'polygons.pkl', 'wb') as fh:
        pickle.dump({int(i): wkb[int(i)] for i in keep.Hylak_id}, fh)
    print('GDW records', len(rec), '| HydroLAKES v1.0 polygons', len(d), '| listed', len(keep),
          '| excluded records', rec.excluded[rec.excluded.ne('')].value_counts().to_dict(), flush=True)
    extract_tiles(keep)


def order():
    d = pd.read_csv(OUT / 'small_list.csv')
    lat = (d.lat_min + d.lat_max) / 2       # grid-size estimate of route1_global_run_v1.jobs
    w = (d.lon_max - d.lon_min) * 111.32 * np.cos(np.radians(lat)) + 30; h = (d.lat_max - d.lat_min) * 111.32 + 30
    d['est_Mcells'] = w * h / 0.0009 / 1e6
    return d.sort_values(['est_Mcells', 'Hylak_id'])


def part(d, a):
    if getattr(a, 'ids', None):
        d = d[d.Hylak_id.isin(a.ids)]
    if a.part:
        i, n = map(int, a.part.split('/')); d = d.iloc[i - 1::n]
    return d


def resource(exc):
    return isinstance(exc, MemoryError) or 'allocat' in repr(exc)


def run_list(d, max_margin_km):
    """gdw_ext_v1.run_list."""
    head = committed()
    with open(DATA / 'polygons.pkl', 'rb') as fh:
        polys = pickle.load(fh)
    dm = gr.erun.dmax_all()
    gr.RUN.mkdir(parents=True, exist_ok=True)
    for r in d.itertuples():
        hid = int(r.Hylak_id); f = gr.RUN / str(hid); t0 = time.time()
        if not (f / f'{hid}_run.json').exists() and not (f / f'{hid}_error.json').exists():
            try:
                rec = gr.run_one(hid, shapely.from_wkb(polys[hid]), dm, max_margin_km * 1000)
                print(hid, 'tc ok', rec['regime'][:5], f"W0 {rec['W0_km2']:.2f} F {rec['F_km2']:.2f}",
                      f"V_R1C {rec['methods']['R1C']['V_top_km3']:.4g}", f'{time.time() - t0:.0f}s', flush=True)
            except Exception as exc:
                if resource(exc):
                    print(hid, 'RESOURCE tc', repr(exc)[:120], flush=True); continue
                f.mkdir(parents=True, exist_ok=True)
                (f / f'{hid}_error.json').write_text(json.dumps(dict(error=repr(exc), trace=traceback.format_exc())), encoding='utf-8')
                print(hid, 'FAILED tc', repr(exc)[:200], f'{time.time() - t0:.0f}s', flush=True); continue
        if (f / f'{hid}_run.json').exists() and not (f / f'{hid}_R2_run.json').exists() and not (f / f'{hid}_R2_error.json').exists():
            t1 = time.time()
            try:
                rec = r2.run_one(r, gr.RUN, head, recheck=True)
                print(hid, 'dam', rec['status'], f"D {rec['D_dam_m']:.1f}", f"V_R2 {rec['R2']['V_top_km3']:.4g}", f'{time.time() - t1:.0f}s', flush=True)
            except Exception as exc:
                if resource(exc):
                    print(hid, 'RESOURCE dam', repr(exc)[:120], flush=True); continue
                (f / f'{hid}_R2_error.json').write_text(json.dumps(dict(error=repr(exc), trace=traceback.format_exc())), encoding='utf-8')
                print(hid, 'FAILED dam', repr(exc)[:200], flush=True)


def cmd_run(a):
    run_list(part(order(), a), 15.0)


def cmd_retry_edge(a):
    d = order()
    ids = []
    for hid in d.Hylak_id:
        e = gr.RUN / str(hid) / f'{hid}_error.json'
        if e.exists() and 'grid border' in json.loads(e.read_text(encoding='utf-8'))['error']:
            e.unlink()
            for p in (gr.INPUTS / f'{hid}_inputs.npz', gr.INPUTS / f'{hid}_inputs.json'):
                p.unlink(missing_ok=True)
            ids.append(int(hid))
    print('edge retries', len(ids), flush=True)
    run_list(d[d.Hylak_id.isin(ids)], 45.0)


def cmd_lock():
    head = committed()
    d = pd.read_csv(OUT / 'small_list.csv')
    runs, failed, missing = {}, {}, []
    for hid in d.Hylak_id:
        f = gr.RUN / str(hid)
        if (f / f'{hid}_run.json').exists():
            runs[int(hid)] = {p.name: sha(p) for p in sorted(f.glob('*')) if p.suffix in ('.tif', '.csv', '.json')}
        elif (f / f'{hid}_error.json').exists():
            failed[int(hid)] = json.loads((f / f'{hid}_error.json').read_text(encoding='utf-8'))['error'][:300]
        else:
            missing.append(int(hid))
    scripts = gr.SCRIPTS + ['r2_global_run_v1.py', 'r2_dev_v1.py', 'r2_candidates_round1.json', 'small_tier_v1.py']
    m = dict(spec=str(SPEC.relative_to(ROOT)).replace(chr(92), '/'), spec_sha256=sha(SPEC), git_head=head,
             list_sha256=sha(OUT / 'small_list.csv'), scripts={n: sha(HERE / n) for n in scripts}, n_listed=int(len(d)),
             n_ok=len(runs), n_failed=len(failed), missing=missing, failed=failed, runs=runs,
             locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (OUT / 'lock_manifest.json').write_text(json.dumps(m, indent=0), encoding='utf-8')
    print('LOCKED ok', len(runs), 'failed', len(failed), 'missing', len(missing), flush=True)


# ---------------------------------------------------------------- widening (after the lock)
def widened(R0, k):
    return ndi.binary_dilation(R0, EIGHT, iterations=k) if k else R0


def locked_tc_dam(run, rec2):
    """TC-dam V_top and A_top of the lock: the R2 record, or the R1-C record when TC-dam failed."""
    if rec2 is not None:
        return rec2['R2']['V_top_km3'], rec2['R2']['A_top_km2'], rec2['status']
    m = run['methods']['R1C']
    return m['V_top_km3'], m['A_top_km2'], 'failed'


def cmd_widen(a):
    import newdata_score_v1 as ns
    committed()
    assert (OUT / 'lock_manifest.json').exists(), 'run lock first'
    ns.R1 = ns.R2 = OUT
    lock = json.loads((OUT / 'lock_manifest.json').read_text(encoding='utf-8'))
    d = part(order(), a)
    d = d[d.Hylak_id.astype(str).isin(lock['runs'])]
    WID.mkdir(parents=True, exist_ok=True)
    for r in d.itertuples():
        hid = int(r.Hylak_id); jf = WID / f'{hid}.json'
        if jf.exists():
            continue
        t0 = time.time()
        run = json.loads((gr.RUN / str(hid) / f'{hid}_run.json').read_text(encoding='utf-8'))
        p2 = gr.RUN / str(hid) / f'{hid}_R2_run.json'
        rec2 = json.loads(p2.read_text(encoding='utf-8')) if p2.exists() else None
        g, F, H_flat, surf, info = ns.surfaces(hid)
        z, region = surf['TC-dam']; top = float(info['H_top_m']); R0 = g['swo1']
        out, tabs = dict(Hylak_id=hid, H_top_m=top), {}
        for k in KS:
            tab = gr.curves(z, region, dict(g, swo1=widened(R0, k)), F, top)
            t = tab.iloc[-1]
            out[f'V_top_k{k}_km3'], out[f'A_top_k{k}_km2'] = float(t.V_swo1), float(t.A_swo1)
            if k in (0, K_PRODUCT):
                tabs[k] = pd.DataFrame({'H_m': tab.H, f'A_k{k}_km2': tab.A_swo1, f'V_k{k}_km3': tab.V_swo1})
        V0, A0, status = locked_tc_dam(run, rec2)
        out.update(V_top_locked_km3=V0, A_top_locked_km2=A0, R2_status=status,
                   k0_reproduces=bool(abs(out['V_top_k0_km3'] - V0) <= max(1e-9, 1e-5 * abs(V0))), runtime_s=round(time.time() - t0, 2))
        tabs[0].merge(tabs[K_PRODUCT], on='H_m', how='outer').sort_values('H_m').to_csv(WID / f'{hid}_aev.csv', index=False, float_format='%.6g')
        jf.write_text(json.dumps(out), encoding='utf-8')
        print(hid, 'widen', ' '.join(f"{out[f'V_top_k{k}_km3']:.4g}" for k in KS), 'k0 ok' if out['k0_reproduces'] else 'K0 MISMATCH',
              f"{out['runtime_s']:.1f}s", flush=True)


def cmd_widen_lock():
    lock = json.loads((OUT / 'lock_manifest.json').read_text(encoding='utf-8'))
    rows, curves, missing = [], [], []
    for hid in sorted(map(int, lock['runs'])):
        jf = WID / f'{hid}.json'
        if not jf.exists():
            missing.append(hid); continue
        rows.append(json.loads(jf.read_text(encoding='utf-8')))
        c = pd.read_csv(WID / f'{hid}_aev.csv'); c.insert(0, 'Hylak_id', hid); curves.append(c)
    w = pd.DataFrame(rows)
    w.to_csv(OUT / 'widen_per_reservoir.csv', index=False, float_format='%.8g')
    pd.concat(curves).to_csv(OUT / 'small_aev.csv.gz', index=False, float_format='%.6g', compression='gzip')
    m = dict(lock_sha256=sha(OUT / 'lock_manifest.json'), script_sha256=sha(HERE / 'small_tier_v1.py'), n=len(w), missing=missing,
             n_k0_mismatch=int((~w.k0_reproduces).sum()), k0_mismatch=w.Hylak_id[~w.k0_reproduces].astype(int).tolist(),
             widen_per_reservoir_sha256=sha(OUT / 'widen_per_reservoir.csv'), small_aev_sha256=sha(OUT / 'small_aev.csv.gz'),
             locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (OUT / 'widen_manifest.json').write_text(json.dumps(m, indent=1), encoding='utf-8')
    print('widened', len(w), '| missing', len(missing), '| k0 mismatch', m['n_k0_mismatch'], flush=True)


# ---------------------------------------------------------------- labels (spec section 5)
_VR = {}


def ranges():
    if not _VR:
        _VR.update(json.loads(VR.read_text(encoding='utf-8'))); _VR.update(json.loads(VR2.read_text(encoding='utf-8')))
    return _VR


def labels(run, rec2, hl_area, gdw_area, year_dam, A_top_product):
    v = ranges()
    status = rec2['status'] if rec2 is not None else 'failed'
    D = rec2['D_dam_m'] if rec2 is not None else np.nan
    FW, RW = run['F_km2'] / run['W0_km2'], run['R_km2'] / run['W0_km2']
    ratio = hl_area / gdw_area if (gdw_area is not None and np.isfinite(gdw_area) and gdw_area > 0) else np.nan
    f = dict(flag_no_dam_height=status in NO_DAM, flag_F_gt_W0=not run['F_le_W0'],
             flag_beyond_validated=bool(FW > v['F_over_W0_max'] or RW > v['R_over_W0_max'] or (np.isfinite(D) and D > v['D_dam_max_m'])),
             flag_hl_gdw_mismatch=bool(np.isfinite(ratio) and not 0.5 <= ratio <= 2),
             flag_post2000=bool(year_dam is not None and np.isfinite(year_dam) and year_dam >= 2000),
             flag_tiny=bool(A_top_product < 0.1))
    f['quality_class'] = ('C' if f['flag_no_dam_height'] or f['flag_beyond_validated'] or f['flag_hl_gdw_mismatch']
                          else 'B' if f['flag_F_gt_W0'] else 'A')
    return dict(f, F_over_W0=FW, R_over_W0=RW, HL_over_GDW_area=ratio, D_dam_m=D, R2_status=status)


def failure_class(hid, err):
    """gdw_ext_v1.failure_class."""
    j = gr.INPUTS / f'{hid}_inputs.json'
    m = json.loads(j.read_text(encoding='utf-8')) if j.exists() else {}
    if 'no SWBD water' in err:
        return 'no SWBD (Feb 2000) water inside the polygon'
    if m.get('fabdem_finite_in_R') == 0:
        return 'no FABDEM'
    if 'grid border' in err:
        return f"F reaches the grid border at the {m.get('margin_m', 0) / 1e3:.0f} km margin cap"
    return 'other'


def size_class(a):
    return pd.cut(a, [0, 0.3, 1, np.inf], right=False, labels=['<0.3', '0.3-1', '>=1'])


def cmd_summary(a):
    assert (OUT / 'widen_manifest.json').exists(), 'run widen-lock first'
    k = a.k
    lst = pd.read_csv(OUT / 'small_list.csv')
    w = pd.read_csv(OUT / 'widen_per_reservoir.csv').set_index('Hylak_id')
    rows, fails = [], []
    for r in lst.itertuples():
        hid = int(r.Hylak_id); f = gr.RUN / str(hid)
        if not (f / f'{hid}_run.json').exists():
            e = json.loads((f / f'{hid}_error.json').read_text(encoding='utf-8'))['error'] if (f / f'{hid}_error.json').exists() else 'no output'
            fails.append(dict(Hylak_id=hid, GDW_ID=r.GDW_GDW_ID, failure_class=failure_class(hid, e), error=e[:300])); continue
        run = json.loads((f / f'{hid}_run.json').read_text(encoding='utf-8'))
        p2 = f / f'{hid}_R2_run.json'
        rec2 = json.loads(p2.read_text(encoding='utf-8')) if p2.exists() else None
        x = w.loc[hid]
        row = gs.row(run); row.pop('recommended')     # the global C2 label; the tier's product is product_method
        row.update(product_method='TC-cap (no dam height read)' if (rec2 is None or rec2['status'] in NO_DAM) else 'TC-dam',
                   widen_cells=k, V_top_km3=x[f'V_top_k{k}_km3'], A_top_km2=x[f'A_top_k{k}_km2'],
                   **{f'V_top_k{j}_km3': x[f'V_top_k{j}_km3'] for j in KS if j != k}, **{f'A_top_k{j}_km2': x[f'A_top_k{j}_km2'] for j in KS if j != k},
                   B1000_m=rec2['B_R_m'] if rec2 else np.nan, B2000_m=rec2['B_2R_m'] if rec2 else np.nan)
        row.update(labels(run, rec2, r.Lake_area, r.GDW_AREA_POLY, r.GDW_YEAR_DAM, x[f'A_top_k{k}_km2']))
        rows.append(row)
    keep = ['Hylak_id', 'Lake_name', 'Country', 'Continent', 'Lake_type', 'Poly_src', 'Lake_area', 'Vol_total', 'Depth_avg', 'Pour_long',
            'Pour_lat', 'n_gdw_records'] + [c for c in lst.columns if c.startswith('GDW_')]
    s = lst[keep].rename(columns={'Lake_type': 'HL_Lake_type', 'Lake_area': 'HL_Lake_area_km2', 'Vol_total': 'HL_Vol_total_mcm',
                                  'Depth_avg': 'HL_Depth_avg_m', 'GDW_GDW_ID': 'GDW_ID', 'GDW_category': 'GDW_coverage_category'}).merge(
        pd.DataFrame(rows), on='Hylak_id')
    s.insert(1, 'tier', 'small (< 1 km2)')
    s.to_csv(OUT / 'small_tier_summary.csv', index=False, float_format='%.6g')
    pd.DataFrame(fails, columns=['Hylak_id', 'GDW_ID', 'failure_class', 'error']).to_csv(OUT / 'small_failed.csv', index=False)
    print('listed', len(lst), '| with product', len(s), '| failed', len(fails), flush=True)
    print(pd.DataFrame(fails).failure_class.value_counts().to_string() if fails else '', flush=True)
    print(s.product_method.value_counts().to_string()); print(s.quality_class.value_counts().sort_index().to_string())
    print(s.filter(like='flag_').sum().to_string(), flush=True)
    coverage(s)


def coverage(s):
    """GDW coverage with the tier, as dataset_coverage_v3 (denominator: GDW without the controlled lakes)."""
    p = pd.read_csv(PART)
    cap = p.CAP_MCM.clip(lower=0); A, V = p.AREA_POLY.sum(), cap.sum()
    c3 = json.loads(COV3.read_text(encoding='utf-8'))
    before = p.part.ne('none')
    p.loc[p.GDW_ID.isin(s.GDW_ID) & p.part.eq('none'), 'part'] = 'small_tier'
    cov = p.part.ne('none'); lt1 = p.AREA_POLY < 1
    out = dict(n_records=int(len(p)), area_km2=float(A), storage_km3=float(V / 1e3),
               v3_check=dict(n=int(before.sum()), area_share=float(p.AREA_POLY[before].sum() / A), storage_share=float(cap[before].sum() / V)),
               tier_records=int(p.part.eq('small_tier').sum()), tier_area_km2=float(p.AREA_POLY[p.part.eq('small_tier')].sum()),
               tier_storage_km3=float(cap[p.part.eq('small_tier')].sum() / 1e3),
               with_tier=dict(n=int(cov.sum()), area_share=float(p.AREA_POLY[cov].sum() / A), storage_share=float(cap[cov].sum() / V)),
               lt1=dict(n=int(lt1.sum()), covered=int((cov & lt1).sum()), area_share_of_class=float(p.AREA_POLY[cov & lt1].sum() / p.AREA_POLY[lt1].sum()),
                        storage_share_of_class=float(cap[cov & lt1].sum() / cap[lt1].sum())),
               by_size={b: dict(n=int((p['size'] == b).sum()), covered=int((cov & (p['size'] == b)).sum()),
                                storage_share_covered=float(cap[cov & (p['size'] == b)].sum() / V)) for b in p['size'].unique()},
               coverage_v3_file=str(COV3.relative_to(ROOT)).replace(chr(92), '/'), coverage_v3_combined=c3['combined'])
    p.to_csv(OUT / 'gdw_part_with_tier.csv', index=False)
    (OUT / 'coverage.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ['v3_check', 'coverage_v3_combined', 'tier_records', 'tier_storage_km3', 'with_tier', 'lt1']}, indent=1), flush=True)


# ---------------------------------------------------------------- checks (spec section 4)
def gdw_by_hylak():
    c = pd.read_csv(CAT, low_memory=False, usecols=['GDW_ID', 'HYLAK_ID', 'AREA_POLY', 'YEAR_DAM'])
    return c[c.HYLAK_ID > 0].sort_values('GDW_ID').drop_duplicates('HYLAK_ID').set_index('HYLAK_ID')


def reference_63():
    """The 63 scored reservoirs of small_res_v1 with TC-dam at the official pool level for k = 0..3 (fixtest procedure),
    k = 0..2 checked against fixtest_per_reservoir.csv, and the spec section 5 labels at the product k."""
    import small_res_v1 as sr
    import newdata_score_v1 as ns
    s = pd.read_csv(SR / 'score_per_reservoir.csv')
    s = s[s.status.eq('scored') & s.identity_ok & s.method.eq('TC-dam')].drop_duplicates('key')
    ft = pd.read_csv(SR / 'fixtest_per_reservoir.csv'); ft = ft[ft.method.eq('TC-dam')]
    half = ft.drop_duplicates('key').set_index('key').half
    fpe = ft.set_index(['key', 'k']).PE
    gl = gdw_by_hylak()
    rows = []
    for r in s.itertuples():
        hid = int(r.Hylak_id)
        R1, R2 = {'global': (sr.G1, sr.G2), 'gdw_ext': (sr.EXT, sr.EXT), 'study': (SR, SR)}[r.product]
        ns.R1, ns.R2 = R1, R2; gr.INPUTS = sr.INPUTS[r.product]
        g, F, H_flat, surf, info = ns.surfaces(hid)
        z, region = surf['TC-dam']; R0 = g['swo1']
        run = json.loads((R1 / 'run' / str(hid) / f'{hid}_run.json').read_text(encoding='utf-8'))
        p2 = R2 / 'run' / str(hid) / f'{hid}_R2_run.json'
        rec2 = json.loads(p2.read_text(encoding='utf-8')) if p2.exists() else None
        row = dict(key=r.key, source=r.source, product=r.product, Hylak_id=hid, half=half.get(r.key), HL_area_km2=r.HL_area_km2,
                   A_pool_official_km2=r.A_pool_official_km2, Vmax_official_km3=r.Vmax_official_km3)
        for k in KS:
            tab = gr.curves(z, region, dict(g, swo1=widened(R0, k)), F, r.H_pool_m)
            t = tab.iloc[-1]
            row[f'PE_k{k}'] = (float(t.V_swo1) / r.Vmax_official_km3 - 1) * 100
            row[f'A_ratio_k{k}'] = float(t.A_swo1) / r.A_pool_official_km2
            if (r.key, k) in fpe.index:
                row[f'fixtest_ok_k{k}'] = bool(abs(row[f'PE_k{k}'] - fpe[(r.key, k)]) <= max(1e-3, 1e-4 * abs(fpe[(r.key, k)])))
        top = gr.curves(z, region, dict(g, swo1=widened(R0, K_PRODUCT)), F, run['H_top_m']).iloc[-1]
        gd = gl.loc[hid] if hid in gl.index else None
        row.update(in_gdw=gd is not None, **labels(run, rec2, r.HL_area_km2, None if gd is None else float(gd.AREA_POLY),
                                                   None if gd is None else float(gd.YEAR_DAM), float(top.A_swo1)))
        rows.append(row)
        print(r.key, ' '.join(f"{row[f'PE_k{k}']:+.1f}" for k in KS), row['quality_class'], flush=True)
    return pd.DataFrame(rows)


def cmd_check():
    from scipy import stats
    t = reference_63()
    t['size'] = size_class(t.HL_area_km2)
    t.to_csv(OUT / 'check_reference_63.csv', index=False, float_format='%.5g')
    fx = t.filter(like='fixtest_ok')
    A = dict(n=int(len(t)), fixtest_reproduced=bool(fx.fillna(True).all().all()), fixtest_mismatch=int((~fx.fillna(True)).sum().sum()))
    for k in KS:
        x = t[f'PE_k{k}']
        A[f'k{k}'] = dict(abs_median=float(x.abs().median()), signed_median=float(x.median()), A_ratio_median=float(t[f'A_ratio_k{k}'].median()),
                          by_size={str(c): dict(n=int(len(g)), abs_median=float(g[f'PE_k{k}'].abs().median()))
                                   for c, g in t.groupby('size', observed=True)},
                          by_half={h: float(g[f'PE_k{k}'].abs().median()) for h, g in t.groupby('half')})
    for k in (2, 3):
        a, b = t[f'PE_k{k}'].abs(), t[f'PE_k{K_PRODUCT}'].abs()
        A[f'k{k}']['vs_k1'] = dict(better=int((a < b).sum()), worse=int((a > b).sum()), p_one_sided=float(stats.wilcoxon(a, b, alternative='less').pvalue))
    A['by_class_k1'] = {c: dict(n=int(len(g)), abs_median=float(g[f'PE_k{K_PRODUCT}'].abs().median()), signed_median=float(g[f'PE_k{K_PRODUCT}'].median()))
                        for c, g in t.groupby('quality_class')}
    A['flags'] = {c: int(t[c].sum()) for c in t.columns if c.startswith('flag_')}
    # evidence B: US NID normal capacity (GDW CAP_REP of US records)
    s = pd.read_csv(OUT / 'small_tier_summary.csv')
    n = s[s.GDW_COUNTRY.eq('United States') & (s.GDW_CAP_REP > 0)].copy()
    n['size'] = size_class(n.HL_Lake_area_km2)
    cols = {k: ('V_top_km3' if k == int(s.widen_cells.iloc[0]) else f'V_top_k{k}_km3') for k in KS}
    for k, c in cols.items():
        n[f'lnratio_k{k}'] = np.log(n[c].clip(lower=1e-12) / (n.GDW_CAP_REP * 1e-3))
    n.to_csv(OUT / 'check_nid.csv', index=False, float_format='%.6g')

    def st(x):
        return dict(n=int(len(x)), ln_median=float(x.median()), abs_ln_median=float(x.abs().median()), within_2x=float((x.abs() <= np.log(2)).mean()))
    B = {f'k{k}': dict(st(n[f'lnratio_k{k}']), by_size={str(c): st(g[f'lnratio_k{k}']) for c, g in n.groupby('size', observed=True)}) for k in KS}
    B['by_class_k1'] = {c: st(g[f'lnratio_k{K_PRODUCT}']) for c, g in n.groupby('quality_class')}
    B['zero_volume_k1'] = int((n[cols[K_PRODUCT]] <= 0).sum())
    # decision (spec section 4)
    dec = {}
    for k in (2, 3):
        c1 = A[f'k{k}']['abs_median'] <= A['k1']['abs_median'] - 2 and A[f'k{k}']['vs_k1']['p_one_sided'] < 0.05
        c2 = all(A[f'k{k}']['by_size'][c]['abs_median'] - A['k1']['by_size'][c]['abs_median'] <= 5 for c in A['k1']['by_size'])
        c3 = B[f'k{k}']['abs_ln_median'] < B['k1']['abs_ln_median']
        dec[f'k{k}'] = dict(rule1=bool(c1), rule2=bool(c2), rule3=bool(c3), all=bool(c1 and c2 and c3))
    ok = [k for k in (2, 3) if dec[f'k{k}']['all']]
    dec['k_product'] = min(ok, key=lambda k: A[f'k{k}']['abs_median']) if ok else K_PRODUCT
    out = dict(spec=str(SPEC.relative_to(ROOT)).replace(chr(92), '/'), reference_63=A, nid=B, decision=dec,
               finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (OUT / 'check_summary.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps(dec, indent=1)); print(json.dumps({k: {kk: v for kk, v in A[k].items() if kk != 'by_half'} for k in ['k0', 'k1', 'k2', 'k3']}, indent=1))
    print(json.dumps({k: B[k] for k in ['k0', 'k1', 'k2', 'k3']}, indent=1)); print(json.dumps(A['by_class_k1'], indent=1)); print(json.dumps(B['by_class_k1'], indent=1))


def main():
    warnings.filterwarnings('ignore')
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['list', 'run', 'retry-edge', 'lock', 'widen', 'widen-lock', 'summary', 'check'])
    p.add_argument('--part'); p.add_argument('--ids', nargs='*', type=int); p.add_argument('--k', type=int, default=K_PRODUCT)
    a = p.parse_args()
    dict(list=cmd_list, run=lambda: cmd_run(a), lock=cmd_lock, widen=lambda: cmd_widen(a), summary=lambda: cmd_summary(a), check=cmd_check,
         **{'retry-edge': lambda: cmd_retry_edge(a), 'widen-lock': cmd_widen_lock})[a.cmd]()


if __name__ == '__main__':
    main()
