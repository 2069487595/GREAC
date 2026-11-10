"""Reservoirs north of 60 N (spec docs/plans/2026-10-27-high-latitude-spec.md): the frozen global code with one input change,
W0 (the 2000 water body, from SRTM SWBD south of 60 N) taken from the JRC GSW yearly classification of 2000, permanent
water (class 3). NASADEM does not exist north of 60 N and is not needed by the methods. Only the module paths of
route1_global_run_v1 and its sample() are redirected; no method function is changed.

Targets: HydroLAKES Lake_type 2 with the pour point at or north of 60 N (117), and GDW reservoirs outside 56 S - 60 N that
have a HydroLAKES v1.0 polygon of >= 1 km2 (gdw_coverage_v1 category outside_lat). Pour point = HydroLAKES pour point.
Subcommands: list | run [--part i/n] [--ids ...] | lock | summary
Usage: .venv/Scripts/python.exe -I 04_Code/scripts/highlat_v1.py list|run|lock|summary
"""
from pathlib import Path
import sys, json, argparse, datetime, warnings, traceback, time, pickle, math, shutil, zipfile, subprocess

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import rasterio
import shapely
from pyogrio.raw import read
import route1_global_run_v1 as gr
import r2_global_run_v1 as r2
import route1_global_summary_v1 as gs
import route1_global_tiles_v1 as gt

ROOT = gr.ROOT
SPEC = ROOT / 'docs/plans/2026-10-27-high-latitude-spec.md'
G1 = ROOT / '05_Results/by_date/2026-10-07/route1_global_v1'
COVER = ROOT / '05_Results/by_date/2026-10-27/gdw_coverage_v1/gdw_reservoirs_category.csv'
GSW = ROOT / '02_Data_Raw/GSW_yearly2000_20261027'
HL = ROOT / '05_Results/by_date/2026-10-27/highlat_v1'
DATA = ROOT / '03_Data_Processed/highlat_20261027'
VR = G1 / 'validated_range.json'
VR2 = ROOT / '05_Results/by_date/2026-10-10/r2_global_v1/validated_range_R2.json'
COLS = ['Hylak_id', 'Lake_name', 'Country', 'Continent', 'Poly_src', 'Lake_type', 'Grand_id', 'Lake_area', 'Vol_total', 'Depth_avg', 'Pour_long', 'Pour_lat']
GUARDED = [HERE / 'highlat_v1.py', SPEC]
sha = gr.sha
SAMPLE = gr.sample

gr.OUT, gr.RUN, gr.DATA, gr.INPUTS = HL, HL / 'run', DATA, DATA / 'inputs'


def committed():
    head = r2.committed()
    for f in GUARDED:
        rel = str(f.relative_to(ROOT)).replace(chr(92), '/')
        if subprocess.run(['git', 'status', '--porcelain', '--', rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() or \
                subprocess.run(['git', 'ls-files', '--error-unmatch', rel], cwd=ROOT, capture_output=True).returncode != 0:
            raise SystemExit(f'{rel} is not committed')
    return head


def gsw_class(lon, lat):
    """GSW yearly classification 2000 at the cell centres (nearest); 0 where no tile."""
    out = np.zeros(lon.shape, 'uint8')
    tr_, tc_ = np.floor((80 - lat) / 10).astype(int), np.floor((lon + 180) / 10).astype(int)
    for a in np.unique(tr_):
        for b in np.unique(tc_[tr_ == a]):
            sel = (tr_ == a) & (tc_ == b)
            p = GSW / f'yearlyClassification2000-{a * 40000:010d}-{b * 40000:010d}.tif'
            if not p.exists():
                continue
            with rasterio.open(p) as s:
                rr = np.floor((lat[sel] - s.transform.f) / s.transform.e).astype(int); cc = np.floor((lon[sel] - s.transform.c) / s.transform.a).astype(int)
                rr = np.clip(rr, 0, s.height - 1); cc = np.clip(cc, 0, s.width - 1)
                win = rasterio.windows.Window(int(cc.min()), int(rr.min()), int(cc.max() - cc.min() + 1), int(rr.max() - rr.min() + 1))
                v = s.read(1, window=win); out[sel] = v[rr - rr.min(), cc - cc.min()]
    return out


def sample_gsw(lon, lat):
    """route1_global_run_v1.sample with the SWBD water layer replaced by GSW 2000 permanent water; NASADEM left as it is
    (no-data north of 60 N) but set to 0 where GSW has data, because the W0 selection requires a finite NASADEM value."""
    z, swb, occ, missing = SAMPLE(lon, lat)
    k = gsw_class(lon, lat)
    swb = np.where(k == 3, 255, 0).astype('uint8')
    z = np.where(k > 0, np.where(np.isfinite(z), z, 0.0), np.nan).astype('float32')
    missing = [m for m in missing if not m.startswith('NASADEM_')]
    return z, swb, occ, missing


def cmd_list():
    exc = pd.read_csv(G1 / 'global_excluded.csv')
    ids2 = exc[(exc.Lake_type == 2) & exc.excluded.str.startswith('outside')].Hylak_id.astype(int).tolist()
    g = pd.read_csv(COVER, low_memory=False)
    o = g[(g.category == 'outside_lat') & (g.HYLAK_ID > 0)]
    ids = sorted(set(ids2) | set(o.HYLAK_ID.astype(int)))
    meta, _, geoms, f = read(gr.GPKG, columns=COLS, where=f"Hylak_id IN ({','.join(map(str, ids))})")
    d = pd.DataFrame(dict(zip(meta['fields'], f)))
    wkb = dict(zip(d.Hylak_id.astype(int), geoms))
    d[['lon_min', 'lat_min', 'lon_max', 'lat_max']] = shapely.bounds(shapely.from_wkb(geoms))
    d['source'] = np.where(d.Hylak_id.isin(ids2), 'HydroLAKES reservoir', 'GDW')
    d['excluded'] = np.where(d.Lake_area < 1, 'Lake_area < 1 km2', np.where(d.Pour_lat < 60, 'pour point south of 60 N (outside rule by GDW point only)', ''))
    HL.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    keep = d[d.excluded == ''].drop(columns='excluded').sort_values('Hylak_id')
    keep.to_csv(HL / 'hl_list.csv', index=False); d[d.excluded != ''].to_csv(HL / 'hl_excluded.csv', index=False)
    with open(DATA / 'polygons.pkl', 'wb') as fh:
        pickle.dump({int(i): wkb[int(i)] for i in keep.Hylak_id}, fh)
    print('candidates', len(d), '| listed', len(keep), keep.source.value_counts().to_dict(), '| excluded', d.excluded[d.excluded != ''].value_counts().to_dict())
    need = {(la, lo) for r in keep.itertuples() for la in range(math.floor(r.lat_min - gt.PAD), math.floor(r.lat_max + gt.PAD) + 1)
            for lo in range(math.floor(r.lon_min - gt.PAD), math.floor(r.lon_max + gt.PAD) + 1)}
    have = {p.name[:7] for p in gt.CACHE.glob('*_FABDEM_V1-2.tif')}; rows = []
    for la, lo in sorted(need):
        name = f'{gt.tag(la, lo)}_FABDEM_V1-2.tif'
        if gt.tag(la, lo) in have:
            rows.append(dict(tile=gt.tag(la, lo), status='cached')); continue
        z = gt.archive(la, lo)
        if not z.exists():
            rows.append(dict(tile=gt.tag(la, lo), status='no archive')); continue
        with zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist() if Path(n).name == name]
            if not names:
                rows.append(dict(tile=gt.tag(la, lo), status='not in archive (no land)')); continue
            tmp = gt.CACHE / (name + '.part')
            with zf.open(names[0]) as s, tmp.open('wb') as o_:
                shutil.copyfileobj(s, o_, 8 * 1024 * 1024)
            tmp.replace(gt.CACHE / name)
        rows.append(dict(tile=gt.tag(la, lo), status='extracted'))
    t = pd.DataFrame(rows); t.to_csv(HL / 'fabdem_tiles.csv', index=False); print(t.status.value_counts().to_string())


def resource(exc):
    return isinstance(exc, MemoryError) or 'allocat' in repr(exc)


def cmd_run(a):
    warnings.filterwarnings('ignore')
    head = committed()
    gr.sample = sample_gsw
    d = pd.read_csv(HL / 'hl_list.csv')
    lat = (d.lat_min + d.lat_max) / 2
    d['est'] = ((d.lon_max - d.lon_min) * 111.32 * np.cos(np.radians(lat)) + 30) * ((d.lat_max - d.lat_min) * 111.32 + 30)
    d = d.sort_values('est')
    if a.ids:
        d = d[d.Hylak_id.isin(a.ids)]
    if a.part:
        i, n = map(int, a.part.split('/')); d = d.iloc[i - 1::n]
    with open(DATA / 'polygons.pkl', 'rb') as fh:
        polys = pickle.load(fh)
    dm = gr.erun.dmax_all(); gr.RUN.mkdir(parents=True, exist_ok=True)
    for r in d.itertuples():
        hid = int(r.Hylak_id); f = gr.RUN / str(hid); t0 = time.time()
        if not (f / f'{hid}_run.json').exists() and not (f / f'{hid}_error.json').exists():
            try:
                rec = gr.run_one(hid, shapely.from_wkb(polys[hid]), dm, 45000.0)
                print(hid, 'tc ok', rec['regime'][:5], f"W0 {rec['W0_km2']:.1f} F {rec['F_km2']:.1f}", f"V_R1C {rec['methods']['R1C']['V_top_km3']:.4g}", f'{time.time() - t0:.0f}s', flush=True)
            except Exception as exc:
                if resource(exc):
                    print(hid, 'RESOURCE tc', repr(exc)[:120], flush=True); continue
                f.mkdir(parents=True, exist_ok=True)
                (f / f'{hid}_error.json').write_text(json.dumps(dict(error=repr(exc), trace=traceback.format_exc())), encoding='utf-8')
                print(hid, 'FAILED tc', repr(exc)[:160], flush=True); continue
        if (f / f'{hid}_run.json').exists() and not (f / f'{hid}_R2_run.json').exists() and not (f / f'{hid}_R2_error.json').exists():
            try:
                rec = r2.run_one(r, gr.RUN, head, recheck=True)
                print(hid, 'dam', rec['status'], f"D {rec['D_dam_m']:.1f}", f"V_R2 {rec['R2']['V_top_km3']:.4g}", flush=True)
            except Exception as exc:
                if resource(exc):
                    print(hid, 'RESOURCE dam', repr(exc)[:120], flush=True); continue
                (f / f'{hid}_R2_error.json').write_text(json.dumps(dict(error=repr(exc), trace=traceback.format_exc())), encoding='utf-8')
                print(hid, 'FAILED dam', repr(exc)[:160], flush=True)


def cmd_lock():
    head = committed(); d = pd.read_csv(HL / 'hl_list.csv')
    runs, failed, missing = {}, {}, []
    for hid in d.Hylak_id:
        f = gr.RUN / str(hid)
        if (f / f'{hid}_run.json').exists():
            runs[int(hid)] = {p.name: sha(p) for p in sorted(f.glob('*')) if p.suffix in ('.tif', '.csv', '.json')}
        elif (f / f'{hid}_error.json').exists():
            failed[int(hid)] = json.loads((f / f'{hid}_error.json').read_text(encoding='utf-8'))['error'][:300]
        else:
            missing.append(int(hid))
    m = dict(spec=str(SPEC.relative_to(ROOT)).replace(chr(92), '/'), spec_sha256=sha(SPEC), git_head=head, list_sha256=sha(HL / 'hl_list.csv'),
             scripts={n: sha(HERE / n) for n in gr.SCRIPTS + ['r2_global_run_v1.py', 'r2_dev_v1.py', 'highlat_v1.py']}, n_listed=int(len(d)),
             n_ok=len(runs), n_failed=len(failed), missing=missing, failed=failed, runs=runs, locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (HL / 'lock_manifest.json').write_text(json.dumps(m, indent=0), encoding='utf-8')
    print('LOCKED ok', len(runs), 'failed', len(failed), 'missing', len(missing), flush=True)


def cmd_summary():
    d = pd.read_csv(HL / 'hl_list.csv')
    vr, vr2 = json.loads(VR.read_text(encoding='utf-8')), json.loads(VR2.read_text(encoding='utf-8'))
    rows, fails, curves = [], [], []
    for hid in d.Hylak_id:
        f = gr.RUN / str(hid)
        if not (f / f'{hid}_run.json').exists():
            e = json.loads((f / f'{hid}_error.json').read_text(encoding='utf-8'))['error'] if (f / f'{hid}_error.json').exists() else 'no output'
            fails.append(dict(Hylak_id=hid, error=e[:300])); continue
        row = gs.row(json.loads((f / f'{hid}_run.json').read_text(encoding='utf-8')))
        j2 = f / f'{hid}_R2_run.json'
        if j2.exists():
            rr = json.loads(j2.read_text(encoding='utf-8')); m = rr['R2']
            row.update(R2_status=rr['status'], D_dam_m=rr['D_dam_m'], V_top_R2_km3=m['V_top_km3'], A_top_R2_km2=m['A_top_km2'], max_depth_R2_m=m['max_depth_m'])
        else:
            row.update(R2_status='failed', V_top_R2_km3=row['V_top_R1C_km3'], A_top_R2_km2=row['A_top_R1C_km2'], max_depth_R2_m=row['max_depth_R1C_m'])
        a = pd.read_csv(f / f'{hid}_aev.csv')
        if (f / f'{hid}_aev_R2.csv').exists():
            a = a.merge(pd.read_csv(f / f'{hid}_aev_R2.csv'), on='H_m', how='outer').sort_values('H_m').fillna(0.0)
        else:
            a['A_R2_km2'], a['V_R2_km3'] = a.A_R1C_km2, a.V_R1C_km3
        a.insert(0, 'Hylak_id', hid); curves.append(a); rows.append(row)
    s = d[['Hylak_id', 'source', 'Lake_name', 'Country', 'Continent', 'Lake_type', 'Grand_id', 'Poly_src', 'Lake_area', 'Vol_total', 'Pour_long', 'Pour_lat']].rename(
        columns={'Lake_type': 'HL_Lake_type', 'Lake_area': 'HL_Lake_area_km2', 'Vol_total': 'HL_Vol_total_mcm'}).merge(pd.DataFrame(rows), on='Hylak_id')
    s['w0_source'] = 'GSW yearly 2000 permanent water'
    s['F_over_W0'] = s.F_km2 / s.W0_km2; s['R_over_W0'] = s.R_km2 / s.W0_km2
    s['flag_F_W0_beyond_validated'] = s.F_over_W0 > vr['F_over_W0_max']; s['flag_R_W0_beyond_validated'] = s.R_over_W0 > vr['R_over_W0_max']
    s['flag_R2_equals_R1C'] = s.R2_status.isin(['fallback_undefined', 'fallback_nonpositive', 'failed'])
    s['flag_D_dam_beyond_validated'] = s.D_dam_m > vr2['D_dam_max_m']; s['flag_area_beyond_validated'] = s.HL_Lake_area_km2 > vr2['HL_area_max_km2']
    s['recommended'] = 'R2'
    s.to_csv(HL / 'hl_summary.csv', index=False, float_format='%.6g')
    pd.DataFrame(fails, columns=['Hylak_id', 'error']).to_csv(HL / 'hl_failed.csv', index=False)
    if curves:
        pd.concat(curves).to_csv(HL / 'hl_aev.csv.gz', index=False, float_format='%.6g', compression='gzip')
    print('listed', len(d), '| with product', len(s), '| failed', len(fails)); print(s.R2_status.value_counts().to_string()); print(s.regime.value_counts().to_string())
    print('V_top TC-dam sum km3', round(s.V_top_R2_km3.sum(), 1), '| countries', s.Country.value_counts().to_dict())


def main():
    p = argparse.ArgumentParser(); p.add_argument('cmd', choices=['list', 'run', 'lock', 'summary']); p.add_argument('--part'); p.add_argument('--ids', nargs='*', type=int)
    a = p.parse_args()
    dict(list=cmd_list, run=lambda: cmd_run(a), lock=cmd_lock, summary=cmd_summary)[a.cmd]()


if __name__ == '__main__':
    main()
