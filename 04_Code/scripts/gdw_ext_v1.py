"""GDW extension of the global product (spec docs/plans/2026-10-27-gdw-extension-run-spec.md): the reservoirs that GDW v1.0
identifies as reservoirs but HydroLAKES v1.0 labels as lakes (gdw_coverage_v1 category candidate_hl10), run with the
frozen global code. Only the module paths of route1_global_run_v1 are redirected; no function is changed.

Subcommands:
  list        list with the product rules (Lake_area >= 1 km2, pour point in [56 S, 60 N)), excluded list, polygons;
              FABDEM tiles extracted from the local archive (route1_global_tiles_v1 rules, nothing is downloaded)
  run         per reservoir route1_global_run_v1.run_one (TC-cap, GLOBathy-est, switching), then r2_global_run_v1.run_one
              (TC-dam, dam height read at the HydroLAKES pour point)          [--size lo:hi] [--part i/n] [--ids ...]
  retry-edge  reservoirs that failed because F reached the grid border: inputs rebuilt with the 45 km margin cap
              (as the cap-extension pass of the global run), then run again
  lock        lock_manifest.json (SHA-256 of every output)
  summary     ext_summary.csv, ext_failed.csv, ext_aev.csv.gz; flags with the locked validated ranges of the global runs
Usage: .venv/Scripts/python.exe -I 04_Code/scripts/gdw_ext_v1.py list|run|retry-edge|lock|summary [...]
"""
from pathlib import Path
import sys, json, argparse, datetime, warnings, traceback, time, pickle, math, shutil, zipfile, subprocess

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import shapely
from pyogrio.raw import read
import route1_global_run_v1 as gr
import r2_global_run_v1 as r2
import route1_global_summary_v1 as gs
import route1_global_tiles_v1 as gt

ROOT = gr.ROOT
SPEC = ROOT / 'docs/plans/2026-10-27-gdw-extension-run-spec.md'
COVER = ROOT / '05_Results/by_date/2026-10-27/gdw_coverage_v1/gdw_reservoirs_category.csv'
EXT = ROOT / '05_Results/by_date/2026-10-27/gdw_ext_v1'
DATA = ROOT / '03_Data_Processed/gdw_ext_20261027'
VR = ROOT / '05_Results/by_date/2026-10-07/route1_global_v1/validated_range.json'
VR2 = ROOT / '05_Results/by_date/2026-10-10/r2_global_v1/validated_range_R2.json'
COLS = ['Hylak_id', 'Lake_name', 'Country', 'Continent', 'Poly_src', 'Lake_type', 'Grand_id', 'Lake_area', 'Vol_total',
        'Vol_res', 'Vol_src', 'Depth_avg', 'Elevation', 'Pour_long', 'Pour_lat']
GDW = {'GDW_ID': 'GDW_ID', 'RES_NAME': 'GDW_res_name', 'DAM_NAME': 'GDW_dam_name', 'ORIG_SRC': 'GDW_orig_src',
       'YEAR_DAM': 'GDW_year_dam', 'DAM_HGT_M': 'GDW_dam_hgt_m', 'CAP_MCM': 'GDW_cap_mcm', 'AREA_POLY': 'GDW_area_km2'}
GUARDED = [HERE / 'gdw_ext_v1.py', SPEC]
sha = gr.sha

# the frozen modules read these paths when their functions run
gr.OUT, gr.RUN, gr.DATA, gr.INPUTS = EXT, EXT / 'run', DATA, DATA / 'inputs'


def committed():
    head = r2.committed()              # the frozen global code and the R2 spec, tracked and unchanged
    for f in GUARDED:
        rel = str(f.relative_to(ROOT)).replace(chr(92), '/')
        if subprocess.run(['git', 'status', '--porcelain', '--', rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() or \
                subprocess.run(['git', 'ls-files', '--error-unmatch', rel], cwd=ROOT, capture_output=True).returncode != 0:
            raise SystemExit(f'{rel} is not committed')
    return head


def extract_tiles(d):
    """route1_global_tiles_v1.main for this list: 1-degree FABDEM tiles within 0.3 degrees of a polygon."""
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
    t.to_csv(EXT / 'fabdem_tiles.csv', index=False)
    print(t.status.value_counts().to_string(), flush=True)


def cmd_list():
    c = pd.read_csv(COVER, low_memory=False)
    c = c[c.category == 'candidate_hl10']
    ids = sorted(c.HYLAK_ID.astype(int).unique())
    meta, _, geoms, f = read(gr.GPKG, columns=COLS, where=f"Hylak_id IN ({','.join(map(str, ids))})")
    d = pd.DataFrame(dict(zip(meta['fields'], f)))
    wkb = dict(zip(d.Hylak_id.astype(int), geoms))
    d[['lon_min', 'lat_min', 'lon_max', 'lat_max']] = shapely.bounds(shapely.from_wkb(geoms))
    g = c.sort_values('GDW_ID').drop_duplicates('HYLAK_ID')[['HYLAK_ID'] + list(GDW)].rename(columns={'HYLAK_ID': 'Hylak_id', **GDW})
    d = d.merge(g, on='Hylak_id', how='left')
    d['excluded'] = np.where(d.Lake_area < 1, 'Lake_area < 1 km2',
                    np.where((d.Pour_lat < -56) | (d.Pour_lat >= 60), 'outside SWBD coverage (56 S - 60 N)', ''))
    EXT.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    keep = d[d.excluded == ''].drop(columns='excluded').sort_values('Hylak_id')
    keep['crosses_60N'] = keep.lat_max >= 60
    keep.to_csv(EXT / 'ext_list.csv', index=False)
    d[d.excluded != ''].sort_values('Hylak_id').to_csv(EXT / 'ext_excluded.csv', index=False)
    with open(DATA / 'polygons.pkl', 'wb') as fh:
        pickle.dump({int(i): wkb[int(i)] for i in keep.Hylak_id}, fh)
    print('candidates', len(d), '| listed', len(keep), '| excluded', d.excluded[d.excluded != ''].value_counts().to_dict(), flush=True)
    extract_tiles(keep)


def order(size='all'):
    d = pd.read_csv(EXT / 'ext_list.csv')
    lat = (d.lat_min + d.lat_max) / 2       # grid-size estimate of route1_global_run_v1.jobs
    w = (d.lon_max - d.lon_min) * 111.32 * np.cos(np.radians(lat)) + 30; h = (d.lat_max - d.lat_min) * 111.32 + 30
    d['est_Mcells'] = w * h / 0.0009 / 1e6
    if size != 'all':
        lo, hi = map(float, size.split(':'))
        d = d[(d.est_Mcells >= lo) & (d.est_Mcells < hi)]
    return d.sort_values('est_Mcells')


def resource(exc):
    return isinstance(exc, MemoryError) or 'allocat' in repr(exc)


def run_list(d, max_margin_km):
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
                print(hid, 'tc ok', rec['regime'][:5], f"W0 {rec['W0_km2']:.1f} F {rec['F_km2']:.1f}",
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
    d = order(a.size)
    if a.ids:
        d = d[d.Hylak_id.isin(a.ids)]
    if a.part:
        i, n = map(int, a.part.split('/')); d = d.iloc[i - 1::n]
    run_list(d, 15.0)


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
    d = pd.read_csv(EXT / 'ext_list.csv')
    runs, failed, missing = {}, {}, []
    for hid in d.Hylak_id:
        f = gr.RUN / str(hid)
        if (f / f'{hid}_run.json').exists():
            runs[int(hid)] = {p.name: sha(p) for p in sorted(f.glob('*')) if p.suffix in ('.tif', '.csv', '.json')}
        elif (f / f'{hid}_error.json').exists():
            failed[int(hid)] = json.loads((f / f'{hid}_error.json').read_text(encoding='utf-8'))['error'][:300]
        else:
            missing.append(int(hid))
    scripts = gr.SCRIPTS + ['r2_global_run_v1.py', 'r2_dev_v1.py', 'r2_candidates_round1.json', 'gdw_ext_v1.py']
    m = dict(spec=str(SPEC.relative_to(ROOT)).replace(chr(92), '/'), spec_sha256=sha(SPEC), git_head=head,
             list_sha256=sha(EXT / 'ext_list.csv'), scripts={n: sha(HERE / n) for n in scripts}, n_listed=int(len(d)),
             n_ok=len(runs), n_failed=len(failed), missing=missing, failed=failed, runs=runs,
             locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (EXT / 'lock_manifest.json').write_text(json.dumps(m, indent=0), encoding='utf-8')
    print('LOCKED ok', len(runs), 'failed', len(failed), 'missing', len(missing), flush=True)


def failure_class(hid, err):
    j = gr.INPUTS / f'{hid}_inputs.json'
    m = json.loads(j.read_text(encoding='utf-8')) if j.exists() else {}
    if 'no SWBD water' in err:
        return 'no SWBD (Feb 2000) water inside the polygon'
    if m.get('fabdem_finite_in_R') == 0:
        return 'no FABDEM'
    if 'grid border' in err:
        return f"F reaches the grid border at the {m.get('margin_m', 0) / 1e3:.0f} km margin cap"
    return 'other'


def cmd_summary():
    assert (EXT / 'lock_manifest.json').exists(), 'run lock first'
    lst = pd.read_csv(EXT / 'ext_list.csv')
    vr, vr2 = json.loads(VR.read_text(encoding='utf-8')), json.loads(VR2.read_text(encoding='utf-8'))
    rows, fails, curves = [], [], []
    for hid in lst.Hylak_id:
        f = gr.RUN / str(hid)
        if not (f / f'{hid}_run.json').exists():
            e = json.loads((f / f'{hid}_error.json').read_text(encoding='utf-8'))['error'] if (f / f'{hid}_error.json').exists() else 'no output'
            fails.append(dict(Hylak_id=hid, stage='TC-cap', failure_class=failure_class(hid, e), error=e[:300])); continue
        row = gs.row(json.loads((f / f'{hid}_run.json').read_text(encoding='utf-8')))
        a = pd.read_csv(f / f'{hid}_aev.csv')
        j2 = f / f'{hid}_R2_run.json'
        if j2.exists():
            r = json.loads(j2.read_text(encoding='utf-8')); m = r['R2']
            row.update(R2_status=r['status'], B1000_m=r['B_R_m'], B2000_m=r['B_2R_m'], D_dam_m=r['D_dam_m'], V_top_R2_km3=m['V_top_km3'],
                       A_top_R2_km2=m['A_top_km2'], max_depth_R2_m=m['max_depth_m'], r1c_recheck_max_abs_diff_m=r.get('r1c_recheck_max_abs_diff_m'))
        else:
            e = json.loads((f / f'{hid}_R2_error.json').read_text(encoding='utf-8'))['error'] if (f / f'{hid}_R2_error.json').exists() else 'no output'
            row.update(R2_status='failed'); fails.append(dict(Hylak_id=hid, stage='TC-dam', failure_class='TC-dam error', error=e[:300]))
        if (f / f'{hid}_aev_R2.csv').exists():
            a = a.merge(pd.read_csv(f / f'{hid}_aev_R2.csv'), on='H_m', how='outer').sort_values('H_m').fillna(0.0)
        else:
            a['A_R2_km2'], a['V_R2_km3'] = a.A_R1C_km2, a.V_R1C_km3
        a.insert(0, 'Hylak_id', hid); curves.append(a); rows.append(row)
    keep = ['Hylak_id', 'Lake_name', 'Country', 'Continent', 'Lake_type', 'Poly_src', 'Lake_area', 'Vol_total', 'Depth_avg', 'Pour_long',
            'Pour_lat', 'crosses_60N'] + list(GDW.values())
    s = lst[keep].rename(columns={'Lake_type': 'HL_Lake_type', 'Lake_area': 'HL_Lake_area_km2', 'Vol_total': 'HL_Vol_total_mcm',
                                  'Depth_avg': 'HL_Depth_avg_m'}).merge(pd.DataFrame(rows), on='Hylak_id')
    failed = s.R2_status.eq('failed')
    for col, unit in [('V_top', 'km3'), ('A_top', 'km2'), ('max_depth', 'm')]:      # failed TC-dam: the product falls back to TC-cap
        s.loc[failed, f'{col}_R2_{unit}'] = s.loc[failed, f'{col}_R1C_{unit}']
    s['flag_R2_failed'] = failed
    s['flag_R2_equals_R1C'] = s.R2_status.isin(['fallback_undefined', 'fallback_nonpositive', 'failed'])
    s['validated_region'] = s.Country.eq('United States of America')
    s['F_over_W0'] = s.F_km2 / s.W0_km2; s['R_over_W0'] = s.R_km2 / s.W0_km2
    s['flag_F_W0_beyond_validated'] = s.F_over_W0 > vr['F_over_W0_max']
    s['flag_R_W0_beyond_validated'] = s.R_over_W0 > vr['R_over_W0_max']
    s['flag_poly_not_swbd'] = s.Poly_src.ne('SWBD')
    s['C2_outside_validated_domain'] = s.regime.eq('F>W0') & (s.flag_poly_not_swbd | s.flag_F_W0_beyond_validated)
    s['flag_D_dam_beyond_validated'] = s.D_dam_m > vr2['D_dam_max_m']
    s['flag_area_beyond_validated'] = s.HL_Lake_area_km2 > vr2['HL_area_max_km2']
    s['flag_identity_gdw'] = True
    s['recommended'] = 'R2'
    s.to_csv(EXT / 'ext_summary.csv', index=False, float_format='%.6g')
    pd.DataFrame(fails, columns=['Hylak_id', 'stage', 'failure_class', 'error']).merge(
        lst[['Hylak_id', 'Lake_name', 'Country', 'Poly_src', 'Lake_area']], on='Hylak_id', how='left').to_csv(EXT / 'ext_failed.csv', index=False)
    pd.concat(curves).to_csv(EXT / 'ext_aev.csv.gz', index=False, float_format='%.6g', compression='gzip')
    print('listed', len(lst), '| with product', len(s), '| failed', len(fails))
    print(s.R2_status.value_counts().to_string()); print(s.regime.value_counts().to_string())
    print(s.filter(like='flag_').sum().to_string())


def main():
    warnings.filterwarnings('ignore')
    p = argparse.ArgumentParser(); p.add_argument('cmd', choices=['list', 'run', 'retry-edge', 'lock', 'summary'])
    p.add_argument('--size', default='all'); p.add_argument('--part'); p.add_argument('--ids', nargs='*', type=int)
    a = p.parse_args()
    dict(list=lambda: cmd_list(), run=lambda: cmd_run(a), lock=lambda: cmd_lock(), summary=lambda: cmd_summary(),
         **{'retry-edge': lambda: cmd_retry_edge(a)})[a.cmd]()


if __name__ == '__main__':
    main()
