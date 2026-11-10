"""Evaluation expansion, step 6 (protocol 446b182e1): run every method on TWDB new + USBR group A, then lock.

Reads only input grids (eval_expansion_registry_v1.load: FABDEM, SWBD W0, SWO, NASADEM) and, for GLOBathy, the
HydroLAKES polygon and GLOBathy's parameter table. No official curve, report number or sonar point is read here.
The frozen functions are called unchanged:
  R1-C    : route1_seed_level_v1.rule_C + run_tao (seeds min(DEM, H_seed), h0 = H_seed = max(rule C, H_flat))
  T-flat  : route1_tao_flat_v1.run_tao, seed_cap=False (released Tao behaviour on FABDEM)
  T-SWBD  : aev_observation_share_v1.compute_m0 (Tao on SWBD W0, frozen M0)
  mDEM    : route1_mdem_v1.run (frozen at 4afc3ae33; case 1 writes only its json)
  GLOBathy: globathy_compare_v1.globathy_native / to_grid (its released method); water surface = median DEM on the
            ring outside the polygon; Hylak_id = the polygon holding most of 400 sampled W0 cells (rule of
            baseline_products_ids_v1); Dmax_use_m from GLOBathy_basic_parameters; uses_observed_Dmax recorded.
Rasters per reservoir (EPSG of the input grid, NaN outside the method's own region):
  <key>_C_tao_F.tif, <key>_route1_tao_F.tif, <key>_M0.tif, <key>_mDEM_tao_F.tif (case 2), <key>_GLOBathy.tif
Resume: a reservoir with <key>_run.json is skipped. Failures are written to <key>_error.json and count as failures.
Usage: .venv/Scripts/python.exe 04_Code/scripts/eval_expansion_run_v1.py [--keys ...] [--part i/n]
       .venv/Scripts/python.exe 04_Code/scripts/eval_expansion_run_v1.py --lock   (after all parts finished)
"""
from pathlib import Path
import sys, json, argparse, hashlib, datetime, warnings, traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import rasterio
import shapely
from pyogrio.raw import read
from pyproj import Transformer
from scipy import ndimage as ndi
import aev_registry_v1 as reg
import aev_observation_share_v1 as aev
import route1_tao_flat_v1 as r1
import route1_seed_level_v1 as sl
import route1_mdem_v1 as md
import globathy_compare_v1 as gb
import eval_expansion_registry_v1 as er

OUT = reg.ROOT / '05_Results/by_date/2026-09-30/route1_expansion_run_v1'
PROTOCOL = reg.ROOT / 'docs/plans/2026-09-30-route1-eval-expansion-protocol.md'
GPKG = reg.ROOT / '02_Data_Raw/input_mead/HydroLAKES_polys_v10_wgs84.gpkg'
SCRIPTS = ['eval_expansion_run_v1.py', 'route1_seed_level_v1.py', 'route1_tao_flat_v1.py', 'aev_observation_share_v1.py',
           'route1_mdem_v1.py', 'globathy_compare_v1.py', 'eval_expansion_registry_v1.py', 'eval_expansion_inputs_v1.py']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sample_keys():
    t = pd.read_csv(er.RES / 'admission_twdb.csv')
    s = pd.read_csv(er.RES / 'usbr_split.csv')
    u = pd.read_csv(er.RES / 'admission_usbr.csv')
    tw = list(t[t.status == 'admitted'].key)
    ua = list(u[(u.status == 'admitted') & u.key.isin(s[s.group == 'A'].key)].key)
    return [('TWDB', k) for k in tw] + [('USBR-A', k) for k in ua]


def hylak_id(g, epsg):
    """HydroLAKES polygon containing most of 400 sampled W0 cells (baseline_products_ids_v1 rule)."""
    t = g['transform']; rr, cc = np.nonzero(g['W0'])
    sel = np.random.default_rng(0).choice(len(rr), min(400, len(rr)), replace=False)
    lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(t[2] + (cc[sel] + .5) * t[0], t[5] + (rr[sel] + .5) * t[4])
    meta, _, geoms, fields = read(GPKG, bbox=(lon.min() - .05, lat.min() - .05, lon.max() + .05, lat.max() + .05),
                                  columns=['Hylak_id', 'Lake_name', 'Lake_area'])
    f = dict(zip(meta['fields'], fields)); polys = shapely.from_wkb(geoms); pts = shapely.points(lon, lat)
    hits = [int(shapely.contains(p, pts).sum()) for p in polys]
    i = int(np.argmax(hits))
    return int(f['Hylak_id'][i]), shapely.make_valid(polys[i]), dict(Lake_name=str(f['Lake_name'][i]),
                                                                     Lake_area_km2=float(f['Lake_area'][i]), W0_points_inside=f'{hits[i]}/{len(pts)}')


def dmax_all():
    """GLOBathy_basic_parameters, all lakes in the local tables (as globathy_compare_v1.dmax_table, unfiltered)."""
    df = pd.concat([pd.read_csv(p) for p in sorted(gb.GLOB.glob('GLOBathy_basic_parameters(*K*LAKES).csv'))]).set_index('Hylak_id')
    df['uses_observed_Dmax'] = (df.Dmax_use_m - df.Dmax_est_PAVEW_m).abs() > 1e-6
    return df


def run_globathy(key, g, epsg, folder, dm):
    hid, poly, info = hylak_id(g, epsg)
    if hid not in dm.index:
        return dict(Hylak_id=hid, **info, status='Hylak_id not in the local GLOBathy parameter tables')
    dmax = float(dm.loc[hid, 'Dmax_use_m'])
    depth, origin = gb.globathy_native(poly, dmax)
    d = gb.to_grid(depth, origin, g, epsg)
    P = np.isfinite(d)
    H_surf = float(np.median(g['dem'][ndi.binary_dilation(P, np.ones((3, 3))) & ~P]))
    z = np.full(P.shape, np.nan); z[P] = H_surf - d[P]
    md.write(folder, f'{key}_GLOBathy.tif', z, g, epsg)
    return dict(Hylak_id=hid, **info, status='ok', Dmax_use_m=dmax, Dmax_est_PAVEW_m=float(dm.loc[hid, 'Dmax_est_PAVEW_m']),
                uses_observed_Dmax=bool(dm.loc[hid, 'uses_observed_Dmax']), H_surf_m=H_surf, polygon_km2_on_grid=float(P.sum() * 9e-4))


def run(batch, key, dm):
    folder = OUT / key
    folder.mkdir(parents=True, exist_ok=True)
    g = er.load(key)
    epsg = int(g['meta']['epsg'])
    reg.RESERVOIRS.setdefault(key, dict(name=key, crs=epsg))
    dem, W0, swo1 = g['dem'], g['W0'], g['swo1']
    F, H_flat, share = r1.flattened_region(dem, W0, swo1)
    if F[:3].any() or F[-3:].any() or F[:, :3].any() or F[:, -3:].any():
        raise RuntimeError(f'{key}: F touches the grid border')
    rec = dict(batch=batch, key=key, epsg=epsg, H_flat_m=H_flat, plateau_share_of_W0=share, F_km2=float(F.sum() * 9e-4),
               W0_km2=float(W0.sum() * 9e-4), swo1_km2=float(swo1.sum() * 9e-4))
    status = {}
    H_C, infoC = sl.rule_C(dem, F, g['occ'], swo1)
    H_seed = max(H_C, H_flat)
    rec.update(H_rule_C_m=H_C, H_seed_m=H_seed, floor_applied=bool(H_C < H_flat), rule_C_fallback=infoC.get('fallback'), rule_C=infoC)
    for name, fn in [('R1-C', lambda: sl.run_tao(key, g, F, H_seed, folder, 'C')[1]),
                     ('T-flat', lambda: r1.run_tao(key, g, F, H_flat, folder, seed_cap=False)[1]),
                     ('T-SWBD', lambda: str(aev.compute_m0(key, g, folder))),
                     ('mDEM', lambda: md.run(key, g, folder)),
                     ('GLOBathy', lambda: run_globathy(key, g, epsg, folder, dm))]:
        try:
            out = fn()
            status[name] = 'ok'
            if name == 'R1-C':
                rec['alpha_R1C'] = out['alpha']
            elif name == 'T-flat':
                rec['alpha_Tflat'] = out['alpha']
            elif name == 'mDEM':
                rec['mDEM_case'] = out['case']
            elif name == 'GLOBathy':
                rec['GLOBathy'] = out; status[name] = out['status']
        except Exception as exc:
            status[name] = 'failed: ' + repr(exc)[:200]
            (folder / f'{key}_{name}_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    rec['status'] = status
    rec['inputs_sha256'] = sha(er.INPUTS / f'{er.safe(key)}_inputs.npz')
    rec['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (folder / f'{key}_run.json').write_text(json.dumps(rec, indent=2, default=float), encoding='utf-8')
    return rec


def lock():
    runs, missing = {}, []
    for batch, key in sample_keys():
        f = OUT / key / f'{key}_run.json'
        if not f.exists():
            missing.append(key); continue
        rec = json.loads(f.read_text(encoding='utf-8'))
        runs[key] = dict(batch=batch, status=rec['status'], inputs_sha256=rec['inputs_sha256'],
                         outputs={p.name: sha(p) for p in sorted((OUT / key).glob('*')) if p.suffix in ('.tif', '.json', '.txt')})
    m = dict(protocol=str(PROTOCOL.relative_to(reg.ROOT)), protocol_sha256=sha(PROTOCOL),
             split_csv_sha256=sha(er.RES / 'usbr_split.csv'), scripts={n: sha(HERE / n) for n in SCRIPTS},
             n_reservoirs=len(runs), missing=missing, runs=runs,
             locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             note='written before any official curve, report number or sonar point of these reservoirs is compared with a model')
    (OUT / 'lock_manifest.json').write_text(json.dumps(m, indent=1), encoding='utf-8')
    failed = {k: {n: s for n, s in v['status'].items() if s != 'ok'} for k, v in runs.items()}
    print('LOCKED', len(runs), 'missing', missing, 'non-ok', {k: v for k, v in failed.items() if v}, flush=True)


def main():
    warnings.filterwarnings('ignore')
    p = argparse.ArgumentParser(); p.add_argument('--keys', nargs='*'); p.add_argument('--part'); p.add_argument('--lock', action='store_true')
    a = p.parse_args()
    if a.lock:
        return lock()
    jobs = sample_keys()
    if a.keys:
        jobs = [j for j in jobs if j[1] in a.keys]
    if a.part:
        i, n = map(int, a.part.split('/'))
        jobs = jobs[i - 1::n]
    dm = dmax_all()
    OUT.mkdir(parents=True, exist_ok=True)
    for batch, key in jobs:
        if (OUT / key / f'{key}_run.json').exists():
            continue
        t0 = datetime.datetime.now()
        try:
            rec = run(batch, key, dm)
            print(key, batch, rec['status'], f'{(datetime.datetime.now() - t0).seconds}s', flush=True)
        except Exception as exc:
            (OUT / key).mkdir(parents=True, exist_ok=True)
            (OUT / key / f'{key}_error.json').write_text(json.dumps(dict(error=repr(exc), trace=traceback.format_exc())), encoding='utf-8')
            print(key, batch, 'FAILED', repr(exc)[:200], flush=True)


if __name__ == '__main__':
    main()
