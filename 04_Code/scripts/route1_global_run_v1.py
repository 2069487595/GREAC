"""Step 3 (spec docs/plans/2026-10-07-route1-global-run-spec.md): R1-C and C2 for every listed HydroLAKES reservoir.

Inputs per reservoir (spec section 2), same sources and grid rules as the evaluation (eval_expansion_inputs_v1):
  - UTM 30 m grid; NASADEM HGT/SWB and SWO occurrence sampled at cell centres, FABDEM nearest;
  - identity: W0 = the SWBD water component (SWB = 255, finite NASADEM) inside the reservoir's own HydroLAKES polygon
    dilated by 2 cells, the component with the largest overlap with the polygon; cells of every other HydroLAKES
    polygon are removed first;
  - R (SWO>=1% extent) = occurrence >= 1 outside other HydroLAKES polygons, 8-connected to W0;
  - grid = bbox(W0 or polygon) + 3 km, widened by 3 km while R is within 25 cells of the edge, up to 15 km
    (evaluation: within 3 cells; changed 2026-10-07 because Tao's 20-cell bank-slope lines left the grid);
  - missing NASADEM / SWO / FABDEM tiles (no land) are no-data, counted in the record.
Methods (frozen code, called unchanged):
  R1-C          route1_tao_flat_v1.flattened_region + route1_seed_level_v1.rule_C / run_tao, H_seed = max(rule C, H_flat)
  GLOBathy-est  globathy_compare_v1.globathy_native / to_grid on the own polygon, Dmax = Dmax_est_PAVEW_m,
                H_surf = median DEM on the ring outside the polygon cells (NaN DEM ignored)
  C2            route1_hybrid_v1.surfaces: R1-C if area(F) <= area(W0), else GLOBathy-est
Curves: A and V inside R at 0.25 m from the lowest bed cell to H_top = SWO maximum level (top_level: median DEM on
the outer ring of R outside W0 and F, floored at H_flat); H_top below H_seed is flagged, not raised.
Outputs in RUN/<Hylak_id>/: <id>_inputs.json, <id>_C_tao_F.tif, <id>_GLOBathy_est.tif (intermediate, full grid),
  <id>_bed_R1C.tif (bed elevation EGM2008 m inside R, W0 and F, cropped), <id>_bed_C2.tif (only when F > W0),
  <id>_aev.csv, <id>_run.json; failures <id>_error.json. Inputs npz in DATA/inputs.
Usage: .venv/Scripts/python.exe 04_Code/scripts/route1_global_run_v1.py [--part i/n] [--size small|large|all|lo:hi] [--ids ...] [--defer csv] [--max-margin-km 45]
       .venv/Scripts/python.exe 04_Code/scripts/route1_global_run_v1.py --lock
"""
from pathlib import Path
import sys, json, argparse, hashlib, datetime, warnings, traceback, pickle, time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import rasterio
import shapely
import shapely.ops
from pyogrio.raw import read
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
from scipy import ndimage as ndi
import aev_registry_v1 as reg
import aev_observation_share_v1 as aev
import route1_tao_flat_v1 as r1
import route1_seed_level_v1 as sl
import globathy_compare_v1 as gb
import route1_hybrid_v1 as hy
import eval_expansion_run_v1 as erun
from twdb_nasadem_intersection_screen import HGT, SWB, SWO

ROOT = reg.ROOT
SPEC = ROOT / 'docs/plans/2026-10-07-route1-global-run-spec.md'
OUT = ROOT / '05_Results/by_date/2026-10-07/route1_global_v1'
RUN = OUT / 'run'
DATA = ROOT / '03_Data_Processed/global_route1_20261007'
INPUTS = DATA / 'inputs'
GPKG = ROOT / '02_Data_Raw/input_mead/HydroLAKES_polys_v10_wgs84.gpkg'
FAB = ROOT / '02_Data_Raw/FABDEM_tiles_extracted'
STEP = aev.STEP
LARGE_MCELLS = 30.0
# Tao's bank-slope lines (FirstSlopeCal / draw_slope_line) reach up to 600 m = 20 cells outward from the F shoreline;
# out-of-grid indices crash at the bottom/right edge and silently wrap at the top/left edge. So F must stay >= 21 cells
# from the border, and the grid margin grows while R (which holds F outside W0) is within 25 cells of the border.
TAO_BUFFER_CELLS = 21
EDGE_CELLS = 25
SCRIPTS = ['route1_global_run_v1.py', 'route1_global_list_v1.py', 'route1_tao_flat_v1.py', 'route1_seed_level_v1.py',
           'globathy_compare_v1.py', 'route1_hybrid_v1.py', 'aev_observation_share_v1.py', 'aev_registry_v1.py']
EIGHT = np.ones((3, 3), bool)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def near_edge(m, k):
    return bool(m[:k].any() or m[-k:].any() or m[:, :k].any() or m[:, -k:].any())


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def tag(lat, lon):
    return f'{"N" if lat >= 0 else "S"}{abs(lat):02d}{"E" if lon >= 0 else "W"}{abs(lon):03d}'


def grid(left, top, width, height):
    tr = from_origin(left, top, 30.0, 30.0)
    rr, cc = np.indices((height, width))
    return tr, left + (cc + 0.5) * 30.0, top - (rr + 0.5) * 30.0


def sample(lon, lat):
    """twdb_nasadem_intersection_screen.sample, with absent tiles left as no-data instead of raising."""
    z = np.full(lon.shape, np.nan, 'float32'); swb = np.full(lon.shape, 127, 'uint8'); occ = np.full(lon.shape, 255, 'uint8')
    missing = []
    fl, fo = np.floor(lat).astype(int), np.floor(lon).astype(int)
    for la in np.unique(fl):
        for lo in np.unique(fo[fl == la]):
            sel = (fl == la) & (fo == lo)
            stem = f'{"n" if la >= 0 else "s"}{abs(la):02d}{"e" if lo >= 0 else "w"}{abs(lo):03d}'
            hp, sp = HGT / (stem + '.hgt'), SWB / (stem + '.swb')
            if not hp.exists() or not sp.exists():
                missing.append('NASADEM_' + stem); continue
            h = np.memmap(hp, dtype='>i2', mode='r', shape=(3601, 3601)); s = np.memmap(sp, dtype='u1', mode='r', shape=(3601, 3601))
            rr = np.rint((la + 1 - lat[sel]) * 3600).astype(int); cc = np.rint((lon[sel] - lo) * 3600).astype(int)
            z[sel] = h[rr, cc]; swb[sel] = s[rr, cc]
    z[z == -32768] = np.nan
    cl, cf = (np.ceil(lat / 10) * 10).astype(int), (np.floor(lon / 10) * 10).astype(int)
    for la in np.unique(cl):
        for lo in np.unique(cf[cl == la]):
            sel = (cl == la) & (cf == lo)
            path = SWO / f'occurrence_{abs(lo)}{"E" if lo >= 0 else "W"}_{abs(la)}{"N" if la >= 0 else "S"}_v1_5_2024.tif'
            if not path.exists():
                missing.append('SWO_' + path.name); continue
            with rasterio.open(path) as src:
                rr = np.floor((lat[sel] - src.transform.f) / src.transform.e).astype(int)
                cc = np.floor((lon[sel] - src.transform.c) / src.transform.a).astype(int)
                win = Window(int(cc.min()), int(rr.min()), int(cc.max() - cc.min() + 1), int(rr.max() - rr.min() + 1))
                a = src.read(1, window=win); occ[sel] = a[rr - rr.min(), cc - cc.min()]
    return z, swb, occ, missing


def fabdem(lon, lat, tr, crs):
    """twdb_fabdem_intersection_screen.fabdem (nearest), reading the extracted tile cache; absent tiles = no-data."""
    z = np.full(lon.shape, np.nan, 'float32'); used, missing = [], []
    fl, fo = np.floor(lat).astype(int), np.floor(lon).astype(int)
    for la in np.unique(fl):
        for lo in np.unique(fo[fl == la]):
            path = FAB / f'{tag(la, lo)}_FABDEM_V1-2.tif'
            if not path.exists():
                missing.append('FABDEM_' + tag(la, lo)); continue
            with rasterio.open(path) as s:
                tmp = np.full(z.shape, np.nan, 'float32')
                reproject(rasterio.band(s, 1), tmp, src_transform=s.transform, src_crs=s.crs, src_nodata=s.nodata,
                          dst_transform=tr, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.nearest)
                ok = np.isfinite(tmp); z[ok] = tmp[ok]
            used.append(tag(la, lo))
    return z, used, missing


def masks_on_grid(hid, P, epsg, tr, shape, lon_bounds):
    """Own polygon cells and cells of every other HydroLAKES polygon on the grid."""
    Pm = rasterize([(P, 1)], out_shape=shape, transform=tr, dtype='uint8').astype(bool)
    meta, _, geoms, fields = read(GPKG, bbox=lon_bounds, columns=['Hylak_id'])
    ids = fields[0]
    to = Transformer.from_crs(4326, epsg, always_xy=True).transform
    others = [(shapely.ops.transform(to, shapely.from_wkb(g)), 1) for i, g in zip(ids, geoms) if int(i) != hid]
    other = rasterize(others, out_shape=shape, transform=tr, dtype='uint8').astype(bool) if others else np.zeros(shape, bool)
    return Pm, other & ~Pm


def select_W0(water, Pm):
    lab, _ = ndi.label(water & ndi.binary_dilation(Pm, EIGHT, iterations=2), EIGHT)
    counts = np.bincount(lab[Pm].ravel(), minlength=lab.max() + 1); counts[0] = 0
    if counts.max() == 0:
        raise RuntimeError('no SWBD water inside the polygon')
    return lab == np.argmax(counts)


def build_inputs(hid, poly_wgs, max_margin_m=15000.0):
    c = poly_wgs.representative_point()
    epsg = utm_epsg(c.x, c.y)
    P = shapely.make_valid(shapely.ops.transform(Transformer.from_crs(4326, epsg, always_xy=True).transform, poly_wgs))
    b = P.bounds
    for margin in np.arange(3000.0, max_margin_m + 1.0, 3000.0):
        left, top = np.floor((b[0] - margin) / 30) * 30, np.ceil((b[3] + margin) / 30) * 30
        width = int(np.ceil((b[2] + margin - left) / 30)); height = int(np.ceil((top - (b[1] - margin)) / 30))
        tr, x, y = grid(left, top, width, height)
        lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(x, y)
        del x, y
        nas, swb, occ, missing = sample(lon, lat)
        occ = np.where(occ > 100, 0, occ).astype('uint8')
        Pm, other = masks_on_grid(hid, P, epsg, tr, nas.shape, (float(lon.min()) - .01, float(lat.min()) - .01,
                                                                  float(lon.max()) + .01, float(lat.max()) + .01))
        W0 = select_W0((swb == 255) & np.isfinite(nas) & ~other, Pm)
        R = reg.connected_to((occ >= 1) & ~other, W0)
        touches = near_edge(R, EDGE_CELLS)
        if not touches:
            break
    z, used, fmissing = fabdem(lon, lat, tr, f'EPSG:{epsg}')
    del lon, lat
    INPUTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(INPUTS / f'{hid}_inputs.npz', fabdem=z.astype('float32'), W0=W0, occurrence=occ, R=R, P=Pm, other=other,
                        nasadem=nas.astype('float32'), transform=np.array(tr)[:6])
    meta = dict(Hylak_id=hid, epsg=epsg, shape=list(z.shape), margin_m=float(margin), R_touches_edge=touches,
                W0_km2=float(W0.sum() * 9e-4), polygon_km2_on_grid=float(Pm.sum() * 9e-4), R_km2=float(R.sum() * 9e-4),
                W0_cells_outside_polygon=int((W0 & ~Pm).sum()), other_lake_km2=float(other.sum() * 9e-4),
                fabdem_finite_in_R=float(np.isfinite(z[R]).mean()) if R.any() else None,
                fabdem_tiles=used, missing_tiles=sorted(set(missing + fmissing)),
                built_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (INPUTS / f'{hid}_inputs.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    return meta


def load(hid):
    with np.load(INPUTS / f'{hid}_inputs.npz') as d:
        g = dict(dem=d['fabdem'].astype(float), W0=d['W0'].astype(bool), occ=d['occurrence'].astype(int),
                 swo1=d['R'].astype(bool), P=d['P'].astype(bool), transform=np.asarray(d['transform'], float))
    g['meta'] = json.loads((INPUTS / f'{hid}_inputs.json').read_text(encoding='utf-8'))
    return g


def write_crop(path, arr, keep, g, epsg):
    """Write arr inside keep, cropped to keep's bounding box (+1 cell)."""
    rows, cols = np.flatnonzero(keep.any(1)), np.flatnonzero(keep.any(0))
    r0, r1_, c0, c1 = max(rows[0] - 1, 0), rows[-1] + 2, max(cols[0] - 1, 0), cols[-1] + 2
    a = np.where(keep, arr, np.nan)[r0:r1_, c0:c1].astype('float32')
    t = g['transform']
    with rasterio.open(path, 'w', driver='GTiff', height=a.shape[0], width=a.shape[1], count=1, dtype='float32', nodata=np.nan,
                       crs=f'EPSG:{epsg}', compress='deflate', predictor=3, tiled=True,
                       transform=rasterio.Affine(t[0], 0, t[2] + c0 * t[0], 0, t[4], t[5] + r0 * t[4])) as d:
        d.write(a, 1)


def globathy_est(hid, poly_wgs, g, epsg, folder, dm):
    """route1_hybrid_v1.globathy_est on the reservoir's own polygon (identity known), NaN DEM ignored on the ring."""
    dmax = float(dm.loc[hid, 'Dmax_est_PAVEW_m'])
    depth, origin = gb.globathy_native(shapely.make_valid(poly_wgs), dmax)
    d = gb.to_grid(depth, origin, g, epsg)
    P = np.isfinite(d)
    H_surf = float(np.nanmedian(g['dem'][ndi.binary_dilation(P, EIGHT) & ~P]))
    z = np.full(P.shape, np.nan); z[P] = H_surf - d[P]
    erun.md.write(folder, f'{hid}_GLOBathy_est.tif', z, g, epsg)
    return z, dict(Hylak_id=hid, Dmax_est_PAVEW_m=dmax, Dmax_use_m=float(dm.loc[hid, 'Dmax_use_m']),
                   uses_observed_Dmax=bool(dm.loc[hid, 'uses_observed_Dmax']), H_surf_m=H_surf, polygon_km2_on_grid=float(P.sum() * 9e-4))


def top_level(dem, R, W0, F, H_flat):
    """H_top = SWO maximum level: median DEM on the outer ring of R outside W0 and F (cells whose DEM is real terrain
    at the edge of the maximum observed extent), floored at the FABDEM water plateau H_flat (FABDEM water, 2011-15, lies
    inside the SWO record). Fewer than 10 ring cells -> H_flat."""
    ring = R & ~ndi.binary_erosion(R, EIGHT) & ~W0 & ~F & np.isfinite(dem)
    if ring.sum() < 10:
        return float(H_flat), dict(rule='plateau (ring < 10 cells)', n_ring=int(ring.sum()))
    v = dem[ring]
    H = float(np.median(v))
    return max(H, float(H_flat)), dict(rule='ring median' if H >= H_flat else 'plateau floor', n_ring=int(ring.sum()), H_ring_m=H,
                                       H_ring_p25_m=float(np.percentile(v, 25)), H_ring_p75_m=float(np.percentile(v, 75)))


def curves(z, region, g, F, top):
    swo1, W0 = g['swo1'], g['W0']
    m = swo1 | W0 | F | region
    rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    s = slice(max(rows[0] - 1, 0), rows[-1] + 2), slice(max(cols[0] - 1, 0), cols[-1] + 2)
    lo = np.floor(np.nanmin(z[region & swo1] if (region & swo1).any() else z[region]) / STEP) * STEP
    levels = np.append(np.arange(lo, top - 1e-9, STEP), top)
    return aev.curve_table(z[s], (W0 | F | region)[s], {'swo1': swo1[s]}, levels)


def run_one(hid, poly_wgs, dm, max_margin_m=15000.0):
    folder = RUN / str(hid)
    folder.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not (INPUTS / f'{hid}_inputs.json').exists():
        build_inputs(hid, poly_wgs, max_margin_m)
    t_in = time.time() - t0
    g = load(hid)
    epsg = int(g['meta']['epsg']); key = str(hid)
    reg.RESERVOIRS.setdefault(key, dict(name=key, crs=epsg))
    dem, W0, swo1 = g['dem'], g['W0'], g['swo1']
    F, H_flat, share = r1.flattened_region(dem, W0, swo1)
    if near_edge(F, TAO_BUFFER_CELLS):
        raise RuntimeError(f'F within {TAO_BUFFER_CELLS} cells of the grid border (Tao bank-slope lines would leave the grid)')
    H_C, infoC = sl.rule_C(dem, F, g['occ'], swo1)
    H_seed = max(H_C, H_flat)
    r1c, tao = sl.run_tao(key, g, F, H_seed, folder, 'C')
    gest, grec = globathy_est(hid, poly_wgs, g, epsg, folder, dm)
    surf, sinfo = hy.surfaces(g, F, H_seed, r1c, gest, grec)
    top, tinfo = top_level(dem, swo1, W0, F, H_flat)
    tabs, rec_m = {}, {}
    for name, short in [('R1-C', 'R1C'), ('GLOBathy-est', 'Gest'), ('C2', 'C2')]:
        z, region = surf[name]
        if name == 'C2' and sinfo['F_le_W0']:
            tabs[short] = tabs['R1C']
        else:
            tabs[short] = curves(z, region, g, F, top)
        t = tabs[short].iloc[-1]
        rec_m[short] = dict(V_top_km3=float(t.V_swo1), A_top_km2=float(t.A_swo1),
                            max_depth_m=float(top - np.nanmin(z[region & swo1])) if (region & swo1).any() else None)
    zR, _ = surf['R1-C']
    rec_m['R1C']['share_below_plateau'] = (float(np.sum(np.clip(H_flat - zR[F], 0, None))) * 9e-4 / 1e3 / rec_m['R1C']['V_top_km3']
                                           if rec_m['R1C']['V_top_km3'] > 0 else None)
    write_crop(folder / f'{hid}_bed_R1C.tif', zR, swo1 | W0 | F, g, epsg)
    if not sinfo['F_le_W0']:
        zC, regC = surf['C2']
        write_crop(folder / f'{hid}_bed_C2.tif', zC, swo1 | W0 | F | regC, g, epsg)
    a = pd.DataFrame(dict(H_m=tabs['R1C'].H))
    for short, tab in tabs.items():
        a = a.merge(pd.DataFrame({'H_m': tab.H, f'A_{short}_km2': tab.A_swo1, f'V_{short}_km3': tab.V_swo1}), on='H_m', how='outer')
    a.sort_values('H_m').to_csv(folder / f'{hid}_aev.csv', index=False, float_format='%.6g')
    rec = dict(Hylak_id=hid, epsg=epsg, inputs=g['meta'], H_flat_m=H_flat, plateau_share_of_W0=share, F_km2=float(F.sum() * 9e-4),
               W0_km2=float(W0.sum() * 9e-4), R_km2=float(swo1.sum() * 9e-4), F_le_W0=sinfo['F_le_W0'],
               H_rule_C_m=H_C, H_seed_m=H_seed, floor_applied=bool(H_C < H_flat), rule_C_fallback=infoC.get('fallback'),
               alpha_R1C=tao['alpha'], H_top_m=float(top), H_top=tinfo, H_top_below_seed=bool(top < H_seed), GLOBathy_est=grec, methods=rec_m,
               recommended='C2', regime='F<=W0: C2 = R1-C' if sinfo['F_le_W0'] else 'F>W0: C2 = GLOBathy-est',
               runtime_s=dict(inputs=round(t_in, 1), total=round(time.time() - t0, 1)),
               inputs_sha256=sha(INPUTS / f'{hid}_inputs.npz'), finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (folder / f'{hid}_run.json').write_text(json.dumps(rec, indent=1, default=float), encoding='utf-8')
    return rec


def jobs(size):
    d = pd.read_csv(OUT / 'global_list.csv')
    lat = (d.lat_min + d.lat_max) / 2
    w = (d.lon_max - d.lon_min) * 111.32 * np.cos(np.radians(lat)) + 30; h = (d.lat_max - d.lat_min) * 111.32 + 30
    d['est_Mcells'] = w * h / 0.0009 / 1e6
    if size == 'small':
        d = d[d.est_Mcells < LARGE_MCELLS]
    elif size == 'large':
        d = d[d.est_Mcells >= LARGE_MCELLS]
    elif ':' in size:  # 'lo:hi' in estimated million cells
        lo, hi = map(float, size.split(':'))
        d = d[(d.est_Mcells >= lo) & (d.est_Mcells < hi)]
    return [int(i) for i in d.sort_values('est_Mcells').Hylak_id]


def lock():
    ids = jobs('all')
    runs, failed, missing = {}, {}, []
    for hid in ids:
        f = RUN / str(hid)
        if (f / f'{hid}_run.json').exists():
            rec = json.loads((f / f'{hid}_run.json').read_text(encoding='utf-8'))
            runs[hid] = dict(inputs_sha256=rec['inputs_sha256'],
                             outputs={p.name: sha(p) for p in sorted(f.glob('*')) if p.suffix in ('.tif', '.csv', '.json')})
        elif (f / f'{hid}_error.json').exists():
            failed[hid] = json.loads((f / f'{hid}_error.json').read_text(encoding='utf-8'))['error'][:300]
        else:
            missing.append(hid)
    m = dict(spec=str(SPEC.relative_to(ROOT)), spec_sha256=sha(SPEC), list_sha256=sha(OUT / 'global_list.csv'),
             scripts={n: sha(HERE / n) for n in SCRIPTS}, n_listed=len(ids), n_ok=len(runs), n_failed=len(failed),
             missing=missing, failed=failed, runs=runs, locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (OUT / 'lock_manifest.json').write_text(json.dumps(m, indent=0), encoding='utf-8')
    print('LOCKED ok', len(runs), 'failed', len(failed), 'missing', len(missing), flush=True)


def main():
    warnings.filterwarnings('ignore')
    p = argparse.ArgumentParser(); p.add_argument('--part'); p.add_argument('--size', default='all'); p.add_argument('--ids', nargs='*', type=int)
    p.add_argument('--max-margin-km', type=float, default=15.0,
                   help='cap of the grid margin (cap-extension pass only; default = evaluation cap)')
    p.add_argument('--defer', help='CSV with a Hylak_id column: skipped in this pass (e.g. inputs waiting for a data fix)')
    p.add_argument('--lock', action='store_true')
    a = p.parse_args()
    if a.lock:
        return lock()
    ids = a.ids or jobs(a.size)
    if a.defer:
        skip = set(pd.read_csv(a.defer).Hylak_id.astype(int))
        ids = [i for i in ids if i not in skip]
    if a.part:
        i, n = map(int, a.part.split('/')); ids = ids[i - 1::n]
    with open(DATA / 'polygons_type2.pkl', 'rb') as fh:
        polys = pickle.load(fh)
    dm = erun.dmax_all()
    RUN.mkdir(parents=True, exist_ok=True)
    for hid in ids:
        f = RUN / str(hid)
        if (f / f'{hid}_run.json').exists() or (f / f'{hid}_error.json').exists():
            continue
        t0 = time.time()
        try:
            rec = run_one(hid, shapely.from_wkb(polys[hid]), dm, a.max_margin_km * 1000)
            print(hid, 'ok', rec['regime'][:5], f"W0 {rec['W0_km2']:.1f} F {rec['F_km2']:.1f}",
                  f"V_R1C {rec['methods']['R1C']['V_top_km3']:.4g} V_C2 {rec['methods']['C2']['V_top_km3']:.4g}", f'{time.time() - t0:.0f}s', flush=True)
        except Exception as exc:
            if isinstance(exc, MemoryError) or 'bad allocation' in repr(exc):  # resource, not method: retried in a later pass
                print(hid, 'RESOURCE', repr(exc)[:120], f'{time.time() - t0:.0f}s', flush=True)
                continue
            f.mkdir(parents=True, exist_ok=True)
            (f / f'{hid}_error.json').write_text(json.dumps(dict(error=repr(exc), trace=traceback.format_exc())), encoding='utf-8')
            print(hid, 'FAILED', repr(exc)[:200], f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
