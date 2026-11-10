"""Remnant fix pass (spec docs/plans/2026-10-27-remnant-fix-spec.md): the reservoirs of the global product (10-07) and of
the GDW extension (10-27) that had no product because F reached the grid border at the 45 km margin cap, or because there
is no FABDEM. The frozen global code is called unchanged; only the module paths and two inputs change:
  margin90  grid margin cap 90 km instead of 45 km
  nasadem   NASADEM (Feb 2000, 30 m, EGM96) in place of FABDEM, converted to EGM2008 with H08 = H96 + N96 - N08
            (local geoid grids us_nga_egm96_15 / us_nga_egm08_25, bilinear); flagged dem_substitute = NASADEM
Subcommands: list | run [--ids ...] | lock | summary
Usage: .venv/Scripts/python.exe -I 04_Code/scripts/remnant_fix_v1.py list|run|lock|summary
"""
from pathlib import Path
import sys, json, argparse, datetime, warnings, traceback, time, pickle, subprocess

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import rasterio
import shapely
from scipy.ndimage import map_coordinates
from scipy import ndimage as ndi
import route1_global_run_v1 as gr
import r2_global_run_v1 as r2
import route1_global_summary_v1 as gs

ROOT = gr.ROOT
SPEC = ROOT / 'docs/plans/2026-10-27-remnant-fix-spec.md'
G1 = ROOT / '05_Results/by_date/2026-10-07/route1_global_v1'
EXT = ROOT / '05_Results/by_date/2026-10-27/gdw_ext_v1'
POLY = {'global': ROOT / '03_Data_Processed/global_route1_20261007/polygons_type2.pkl',
        'gdw_ext': ROOT / '03_Data_Processed/gdw_ext_20261027/polygons.pkl'}
FIX = ROOT / '05_Results/by_date/2026-10-27/remnant_fix_v1'
DATA = ROOT / '03_Data_Processed/remnant_fix_20261027'
GEOID = ROOT / '03_Data_Processed/ground_truth/twdb_five_20260918/vertical_grids'
VR = G1 / 'validated_range.json'
VR2 = ROOT / '05_Results/by_date/2026-10-10/r2_global_v1/validated_range_R2.json'
KEEP = ['Hylak_id', 'Lake_name', 'Country', 'Continent', 'Grand_id', 'Poly_src', 'Lake_area', 'Vol_total', 'Depth_avg',
        'Pour_long', 'Pour_lat', 'lon_min', 'lat_min', 'lon_max', 'lat_max']
GUARDED = [HERE / 'remnant_fix_v1.py', SPEC]
sha = gr.sha

gr.OUT, gr.RUN, gr.DATA, gr.INPUTS = FIX, FIX / 'run', DATA, DATA / 'inputs'
FABDEM = gr.fabdem


def committed():
    head = r2.committed()
    for f in GUARDED:
        rel = str(f.relative_to(ROOT)).replace(chr(92), '/')
        if subprocess.run(['git', 'status', '--porcelain', '--', rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() or \
                subprocess.run(['git', 'ls-files', '--error-unmatch', rel], cwd=ROOT, capture_output=True).returncode != 0:
            raise SystemExit(f'{rel} is not committed')
    return head


def geoid(name, lon, lat):
    with rasterio.open(GEOID / name) as s:
        a, t = s.read(1).astype('float64'), s.transform
    col = (np.asarray(lon) - t.c) / t.a - 0.5; row = (np.asarray(lat) - t.f) / t.e - 0.5
    return map_coordinates(a, [row.ravel(), col.ravel()], order=1, mode='nearest').reshape(np.shape(lon))


def nasadem_dem(lon, lat, tr, crs):
    """Replaces route1_global_run_v1.fabdem: NASADEM at the cell centres (nearest, as route1_global_run_v1.sample),
    every 8-connected SWB water body flattened to its median elevation (spec revision: the NASADEM water surface is not flat,
    so the water plateau was not detected), converted from EGM96 to EGM2008."""
    z = np.full(lon.shape, np.nan, 'float32'); w = np.zeros(lon.shape, bool); missing = []
    fl, fo = np.floor(lat).astype(int), np.floor(lon).astype(int)
    for la in np.unique(fl):
        for lo in np.unique(fo[fl == la]):
            sel = (fl == la) & (fo == lo)
            stem = f'{"n" if la >= 0 else "s"}{abs(la):02d}{"e" if lo >= 0 else "w"}{abs(lo):03d}'
            hp, sp = gr.HGT / (stem + '.hgt'), gr.SWB / (stem + '.swb')
            if not hp.exists():
                missing.append('NASADEM_' + stem); continue
            h = np.memmap(hp, dtype='>i2', mode='r', shape=(3601, 3601))
            rr = np.rint((la + 1 - lat[sel]) * 3600).astype(int); cc = np.rint((lon[sel] - lo) * 3600).astype(int)
            z[sel] = h[rr, cc]
            if sp.exists():
                w[sel] = np.memmap(sp, dtype='u1', mode='r', shape=(3601, 3601))[rr, cc] == 255
    z[z == -32768] = np.nan
    z = (z + geoid('us_nga_egm96_15.tif', lon, lat) - geoid('us_nga_egm08_25.tif', lon, lat)).astype('float32')
    lab, n = ndi.label(w & np.isfinite(z), np.ones((3, 3), bool))     # after the datum change: the geoid difference tilts
    if n:                                                              # the (flat) SRTM water by centimetres to decimetres
        med = ndi.median(z, lab, index=np.arange(1, n + 1))
        z = np.where(lab > 0, np.round(np.asarray(med, 'float32'), 2)[np.maximum(lab - 1, 0)], z).astype('float32')
    return z, ['NASADEM (EGM96 -> EGM2008)'], missing


def cmd_list():
    rows, polys = [], {}
    g = pd.read_csv(G1 / 'global_failed.csv'); gl = pd.read_csv(G1 / 'global_list.csv')
    e = pd.read_csv(EXT / 'ext_failed.csv'); el = pd.read_csv(EXT / 'ext_list.csv')
    for src, f, lst in [('global', g, gl), ('gdw_ext', e, el)]:
        f = f[f.failure_class.str.contains('grid border|no FABDEM')]
        with open(POLY[src], 'rb') as fh:
            pk = pickle.load(fh)
        d = lst[lst.Hylak_id.isin(f.Hylak_id)][KEEP].copy()
        d['source'] = src
        d['fix'] = d.Hylak_id.map(dict(zip(f.Hylak_id, np.where(f.failure_class.str.contains('grid border'), 'margin90', 'nasadem'))))
        rows.append(d); polys.update({int(i): pk[int(i)] for i in d.Hylak_id})
    d = pd.concat(rows).sort_values(['fix', 'Hylak_id'])
    FIX.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    d.to_csv(FIX / 'fix_list.csv', index=False)
    with open(DATA / 'polygons.pkl', 'wb') as fh:
        pickle.dump(polys, fh)
    print(d.groupby(['fix', 'source']).size().to_string())


def resource(exc):
    return isinstance(exc, MemoryError) or 'allocat' in repr(exc)


def cmd_run(a):
    warnings.filterwarnings('ignore')
    head = committed()
    d = pd.read_csv(FIX / 'fix_list.csv')
    if a.ids:
        d = d[d.Hylak_id.isin(a.ids)]
    with open(DATA / 'polygons.pkl', 'rb') as fh:
        polys = pickle.load(fh)
    dm = gr.erun.dmax_all()
    gr.RUN.mkdir(parents=True, exist_ok=True)
    for r in d.itertuples():
        hid = int(r.Hylak_id); f = gr.RUN / str(hid); t0 = time.time()
        gr.fabdem = nasadem_dem if r.fix == 'nasadem' else FABDEM
        cap_km = 90.0 if r.fix == 'margin90' else 45.0
        if not (f / f'{hid}_run.json').exists() and not (f / f'{hid}_error.json').exists():
            try:
                rec = gr.run_one(hid, shapely.from_wkb(polys[hid]), dm, cap_km * 1000)
                print(hid, r.fix, 'tc ok', rec['regime'][:5], f"W0 {rec['W0_km2']:.1f} F {rec['F_km2']:.1f}",
                      f"margin {rec['inputs']['margin_m'] / 1e3:.0f} km", f"V_R1C {rec['methods']['R1C']['V_top_km3']:.4g}", f'{time.time() - t0:.0f}s', flush=True)
            except Exception as exc:
                if resource(exc):
                    print(hid, 'RESOURCE tc', repr(exc)[:120], flush=True); continue
                f.mkdir(parents=True, exist_ok=True)
                (f / f'{hid}_error.json').write_text(json.dumps(dict(error=repr(exc), trace=traceback.format_exc())), encoding='utf-8')
                print(hid, r.fix, 'FAILED tc', repr(exc)[:160], f'{time.time() - t0:.0f}s', flush=True); continue
        if (f / f'{hid}_run.json').exists() and not (f / f'{hid}_R2_run.json').exists() and not (f / f'{hid}_R2_error.json').exists():
            try:
                rec = r2.run_one(r, gr.RUN, head, recheck=True)
                print(hid, r.fix, 'dam', rec['status'], f"D {rec['D_dam_m']:.1f}", f"V_R2 {rec['R2']['V_top_km3']:.4g}", flush=True)
            except Exception as exc:
                if resource(exc):
                    print(hid, 'RESOURCE dam', repr(exc)[:120], flush=True); continue
                (f / f'{hid}_R2_error.json').write_text(json.dumps(dict(error=repr(exc), trace=traceback.format_exc())), encoding='utf-8')
                print(hid, r.fix, 'FAILED dam', repr(exc)[:160], flush=True)
    gr.fabdem = FABDEM


def cmd_lock():
    head = committed()
    d = pd.read_csv(FIX / 'fix_list.csv')
    runs, failed, missing = {}, {}, []
    for hid in d.Hylak_id:
        f = gr.RUN / str(hid)
        if (f / f'{hid}_run.json').exists():
            runs[int(hid)] = {p.name: sha(p) for p in sorted(f.glob('*')) if p.suffix in ('.tif', '.csv', '.json')}
        elif (f / f'{hid}_error.json').exists():
            failed[int(hid)] = json.loads((f / f'{hid}_error.json').read_text(encoding='utf-8'))['error'][:300]
        else:
            missing.append(int(hid))
    m = dict(spec=str(SPEC.relative_to(ROOT)).replace(chr(92), '/'), spec_sha256=sha(SPEC), git_head=head, list_sha256=sha(FIX / 'fix_list.csv'),
             scripts={n: sha(HERE / n) for n in gr.SCRIPTS + ['r2_global_run_v1.py', 'r2_dev_v1.py', 'remnant_fix_v1.py']},
             n_listed=int(len(d)), n_ok=len(runs), n_failed=len(failed), missing=missing, failed=failed, runs=runs,
             locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (FIX / 'lock_manifest.json').write_text(json.dumps(m, indent=0), encoding='utf-8')
    print('LOCKED ok', len(runs), 'failed', len(failed), 'missing', len(missing), flush=True)


def cmd_summary():
    assert (FIX / 'lock_manifest.json').exists(), 'run lock first'
    d = pd.read_csv(FIX / 'fix_list.csv')
    vr, vr2 = json.loads(VR.read_text(encoding='utf-8')), json.loads(VR2.read_text(encoding='utf-8'))
    rows, fails, curves = [], [], []
    for r in d.itertuples():
        hid = int(r.Hylak_id); f = gr.RUN / str(hid)
        if not (f / f'{hid}_run.json').exists():
            e = json.loads((f / f'{hid}_error.json').read_text(encoding='utf-8'))['error'] if (f / f'{hid}_error.json').exists() else 'no output'
            fails.append(dict(Hylak_id=hid, fix=r.fix, source=r.source, error=e[:300])); continue
        row = gs.row(json.loads((f / f'{hid}_run.json').read_text(encoding='utf-8')))
        rr = json.loads((f / f'{hid}_R2_run.json').read_text(encoding='utf-8')); m = rr['R2']
        row.update(R2_status=rr['status'], D_dam_m=rr['D_dam_m'], V_top_R2_km3=m['V_top_km3'], A_top_R2_km2=m['A_top_km2'], max_depth_R2_m=m['max_depth_m'])
        a = pd.read_csv(f / f'{hid}_aev.csv')
        if (f / f'{hid}_aev_R2.csv').exists():
            a = a.merge(pd.read_csv(f / f'{hid}_aev_R2.csv'), on='H_m', how='outer').sort_values('H_m').fillna(0.0)
        else:
            a['A_R2_km2'], a['V_R2_km3'] = a.A_R1C_km2, a.V_R1C_km3
        a.insert(0, 'Hylak_id', hid); curves.append(a); rows.append(row)
    s = d[['Hylak_id', 'source', 'fix', 'Lake_name', 'Country', 'Continent', 'Grand_id', 'Poly_src', 'Lake_area', 'Vol_total', 'Pour_long', 'Pour_lat']].rename(
        columns={'Lake_area': 'HL_Lake_area_km2', 'Vol_total': 'HL_Vol_total_mcm'}).merge(pd.DataFrame(rows), on='Hylak_id')
    s['dem_substitute'] = np.where(s.fix.eq('nasadem'), 'NASADEM (water flattened)', '')
    s['F_over_W0'] = s.F_km2 / s.W0_km2; s['R_over_W0'] = s.R_km2 / s.W0_km2
    s['flag_F_W0_beyond_validated'] = s.F_over_W0 > vr['F_over_W0_max']
    s['flag_R_W0_beyond_validated'] = s.R_over_W0 > vr['R_over_W0_max']
    s['flag_R2_equals_R1C'] = s.R2_status.isin(['fallback_undefined', 'fallback_nonpositive'])
    s['flag_D_dam_beyond_validated'] = s.D_dam_m > vr2['D_dam_max_m']
    s['flag_area_beyond_validated'] = s.HL_Lake_area_km2 > vr2['HL_area_max_km2']
    s['recommended'] = 'R2'
    s.to_csv(FIX / 'fix_summary.csv', index=False, float_format='%.6g')
    pd.DataFrame(fails, columns=['Hylak_id', 'fix', 'source', 'error']).to_csv(FIX / 'fix_failed.csv', index=False)
    if curves:
        pd.concat(curves).to_csv(FIX / 'fix_aev.csv.gz', index=False, float_format='%.6g', compression='gzip')
    print('listed', len(d), '| with product', len(s), '| failed', len(fails))
    print(s[['Hylak_id', 'fix', 'Country', 'HL_Lake_area_km2', 'regime', 'R2_status', 'D_dam_m', 'V_top_R1C_km3', 'V_top_R2_km3', 'R_over_W0', 'margin']].to_string()
          if 'margin' in s else s[['Hylak_id', 'fix', 'Country', 'HL_Lake_area_km2', 'regime', 'R2_status', 'D_dam_m', 'V_top_R1C_km3', 'V_top_R2_km3', 'R_over_W0']].to_string())


def main():
    p = argparse.ArgumentParser(); p.add_argument('cmd', choices=['list', 'run', 'lock', 'summary']); p.add_argument('--ids', nargs='*', type=int)
    a = p.parse_args()
    dict(list=cmd_list, run=lambda: cmd_run(a), lock=cmd_lock, summary=cmd_summary)[a.cmd]()


if __name__ == '__main__':
    main()
