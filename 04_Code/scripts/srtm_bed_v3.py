"""SRTM pre-impoundment bed, v3: srtm_bed_v2 with R cut at the GDW dam points and the FABDEM water plateau searched on all
of R when the occurrence >= 50 % rule finds none. Spec docs/plans/2026-10-27-srtm-bed-v3-spec.md.

  cut      600 m around every GDW barrier point in the window (river-snapped point, and the dam coordinates when given;
           removed / replaced / destroyed / subsumed / planned / under-construction barriers left out): water cells there do
           not connect, except those inside the reservoir's own polygon
  R        SWO occurrence >= 1 %, outside other polygons, outside the cut or inside the own polygon, 8-connected to the own
           polygon's water cells (R without the cut is recorded)
  plateau  the v1 rule (occurrence >= 50 % cells, mode >= 20 %); when it finds none, the mode on all R cells (>= 20 %)
  rest     as srtm_bed_v2 (curve top from NASADEM + offset where FABDEM is missing; flags and recommendation rules)
Subcommands: run [--size lo:hi] [--part i/n] | lock | summary | score
Outputs: 05_Results/by_date/2026-10-27/srtm_bed_v3/ (run/<key>/ as v1)
Usage: .venv/Scripts/python.exe -I 04_Code/scripts/srtm_bed_v3.py run|lock|summary|score
"""
from pathlib import Path
import sys, json, argparse, datetime, warnings, traceback, time, shutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import rasterio
import shapely, shapely.ops
from shapely.strtree import STRtree
from pyogrio.raw import read
from pyproj import Transformer
from rasterio.features import rasterize
from scipy import ndimage as ndi
import srtm_bed_v1 as sb
import route1_global_run_v1 as gr
import aev_registry_v1 as reg
import aev_observation_share_v1 as aev

ROOT = sb.ROOT
PLAN = ROOT / 'docs/plans/2026-10-27-srtm-bed-v3-spec.md'
V1 = sb.OUT
OUT = ROOT / '05_Results/by_date/2026-10-27/srtm_bed_v3'
BARRIERS = ROOT / '02_Data_Raw/GDW_v1_0_20261027/extracted/GDW_v1_0_shp/GDW_barriers_v1_0.shp'
CUT_M = 600.0
GONE = {'Removed', 'Replaced', 'Destroyed', 'Subsumed', 'Planned', 'Under construction'}
STEP, EDGE, E8 = sb.STEP, sb.EDGE, sb.E8
sb.GUARDED = [HERE / 'srtm_bed_v1.py', sb.PLAN, HERE / 'srtm_bed_v3.py', PLAN]


class Barriers:
    def __init__(self):
        meta, _, geoms, f = read(BARRIERS, columns=['GDW_ID', 'TIMELINE', 'LONG_DAM', 'LAT_DAM'])
        d = pd.DataFrame(dict(zip(meta['fields'], f)))
        keep = ~d.TIMELINE.isin(GONE).values
        pts = list(shapely.from_wkb(geoms)[keep])
        dm = d[keep & (d.LONG_DAM != 0).values & (d.LAT_DAM != 0).values]
        pts += list(shapely.points(dm.LONG_DAM.values, dm.LAT_DAM.values))
        self.pts = np.array(pts, dtype=object)
        self.tree = STRtree(self.pts)

    def within(self, bounds):
        return list(self.pts[self.tree.query(shapely.box(*bounds))])


def build(key, Pw, polys, bars, cap_m=45000.0):
    """srtm_bed_v1.build with the dam cut."""
    c = Pw.representative_point(); epsg = gr.utm_epsg(c.x, c.y)
    to = Transformer.from_crs(4326, epsg, always_xy=True).transform; back = Transformer.from_crs(epsg, 4326, always_xy=True).transform
    P = shapely.make_valid(shapely.ops.transform(to, Pw)); b = P.bounds
    for margin in np.arange(3000.0, cap_m + 1.0, 3000.0):
        left, top = np.floor((b[0] - margin) / 30) * 30, np.ceil((b[3] + margin) / 30) * 30
        width = int(np.ceil((b[2] + margin - left) / 30)); height = int(np.ceil((top - (b[1] - margin)) / 30))
        tr, x, y = gr.grid(left, top, width, height)
        lon, lat = back(x, y); del x, y
        nas, swb, occ, missing = gr.sample(lon, lat)
        occ = np.where(occ > 100, 0, occ).astype('uint8')
        bounds = (float(lon.min()) - .01, float(lat.min()) - .01, float(lon.max()) + .01, float(lat.max()) + .01)
        Pm = rasterize([(P, 1)], out_shape=nas.shape, transform=tr, dtype='uint8').astype(bool)
        oth = [shapely.ops.transform(to, p) for p in polys.others(Pw, bounds)]
        other = (rasterize([(p, 1) for p in oth], out_shape=nas.shape, transform=tr, dtype='uint8').astype(bool) if oth else np.zeros(nas.shape, bool)) & ~Pm
        seed = Pm & (occ >= 1)
        if not seed.any():
            raise RuntimeError('no SWO water inside the polygon')
        cps = [shapely.ops.transform(to, p) for p in bars.within(bounds)]
        cut = rasterize([(p.buffer(CUT_M), 1) for p in cps], out_shape=nas.shape, transform=tr, dtype='uint8').astype(bool) if cps else np.zeros(nas.shape, bool)
        water = (occ >= 1) & ~other
        R = reg.connected_to(water & (~cut | Pm), seed)
        touches = gr.near_edge(R, EDGE)
        if not touches:
            break
    R_uncut = reg.connected_to(water, seed)
    fab, used, fmiss = gr.fabdem(lon, lat, tr, f'EPSG:{epsg}')
    return dict(epsg=epsg, tr=tr, lon0=float(c.x), lat0=float(c.y), Pm=Pm, other=other, occ=occ, swb=swb, nas=nas, fab=fab, R=R,
                margin_m=float(margin), touches=bool(touches), missing=sorted(set(missing + fmiss)), n_cut_points=len(cps),
                R_uncut_km2=float(R_uncut.sum() * 9e-4), cut_cells_restored_km2=float((R & cut).sum() * 9e-4))


def plateau(R, occ, fab):
    """(F, H_flat, share, rule)"""
    for rule, w in (('occurrence >= 50 %', R & (occ >= 50) & np.isfinite(fab)), ('all R', R & np.isfinite(fab))):
        if w.sum() >= 20:
            vals, counts = np.unique(np.round(fab[w], 2), return_counts=True)
            i = counts.argmax(); share = float(counts[i] / counts.sum())
            if share >= 0.2:
                return R & (np.round(fab, 2) == vals[i]), float(vals[i]), share, rule
    return np.zeros(R.shape, bool), np.nan, 0.0, None


def run_one(key, polys, bars):
    t0 = time.time(); fo = OUT / 'run' / key; fo.mkdir(parents=True, exist_ok=True)
    g = build(key, polys.own(key), polys, bars)
    R, Pm, occ, swb, nas, fab = g['R'], g['Pm'], g['occ'], g['swb'], g['nas'], g['fab']
    land = ndi.binary_dilation(R, E8, iterations=30) & ~ndi.binary_dilation(R, E8, iterations=3) & (occ == 0) & np.isfinite(nas) & np.isfinite(fab)
    if land.sum() >= 100:
        off, off_src = float(np.median(fab[land] - nas[land])), 'stable land'
    else:
        off, off_src = sb.geoid_diff(g['lon0'], g['lat0']), 'geoid EGM96 - EGM2008'
    nas2 = nas + off
    F, H_flat, plateau_share, plateau_rule = plateau(R, occ, fab)
    bed = np.where(np.isfinite(fab), fab, nas2)
    bed[F] = nas2[F]
    bed = np.where(R, bed, np.nan)
    w2000 = R & (swb == 255)
    ring = R & ~ndi.binary_erosion(R, E8) & ~F
    zr = np.where(np.isfinite(fab), fab, nas2)[ring]; okr = np.isfinite(zr)
    ring_nas_share = float((~np.isfinite(fab[ring]))[okr].mean()) if okr.any() else None
    H_top = float(np.median(zr[okr])) if okr.sum() >= 10 else H_flat
    if np.isfinite(H_flat):
        H_top = max(H_top, H_flat) if np.isfinite(H_top) else H_flat
    if not np.isfinite(H_top):
        raise RuntimeError('no curve top (no ring cells and no plateau)')
    seed = Pm & R
    rows_, cols_ = np.flatnonzero(R.any(1)), np.flatnonzero(R.any(0))
    s = slice(max(rows_[0] - 1, 0), rows_[-1] + 2), slice(max(cols_[0] - 1, 0), cols_[-1] + 2)
    zb, Rs, Ss = bed[s], R[s], seed[s]
    lo = np.floor(np.nanmin(zb[Rs]) / STEP) * STEP
    levels = np.append(np.arange(lo, H_top - 1e-9, STEP), H_top)
    tab = aev.curve_table(np.where(Rs, zb, np.inf), Ss, {'R': Rs}, levels)
    tab[['H', 'A_R', 'V_R']].rename(columns={'H': 'H_m', 'A_R': 'A_km2', 'V_R': 'V_km3'}).to_csv(fo / f'{key}_aev.csv', index=False, float_format='%.6g')
    t = g['tr']; tr_s = rasterio.Affine(t.a, 0, t.c + s[1].start * t.a, 0, t.e, t.f + s[0].start * t.e)
    np.savez_compressed(fo / f'{key}_grid.npz', bed=zb.astype('float32'), R=Rs, seed=Ss, transform=np.array(tr_s)[:6], epsg=g['epsg'])
    with rasterio.open(fo / f'{key}_bed.tif', 'w', driver='GTiff', height=zb.shape[0], width=zb.shape[1], count=1, dtype='float32', nodata=np.nan,
                       crs=f"EPSG:{g['epsg']}", transform=tr_s, compress='deflate', predictor=3, tiled=True) as dst:
        dst.write(zb.astype('float32'), 1)
    last = tab.iloc[-1]
    Vf = float(np.sum(np.clip(H_flat - bed[F], 0, None)) * 9e-4 / 1e3) if F.any() else 0.0
    rec = dict(key=key, version='v3', epsg=g['epsg'], margin_m=g['margin_m'], R_touches_edge=g['touches'], missing_tiles=g['missing'],
               n_cut_points=g['n_cut_points'], R_uncut_km2=g['R_uncut_km2'], cut_cells_restored_km2=g['cut_cells_restored_km2'],
               polygon_km2=float(Pm.sum() * 9e-4), R_km2=float(R.sum() * 9e-4), F_km2=float(F.sum() * 9e-4), plateau_share=plateau_share,
               plateau_rule=plateau_rule, H_flat_m=H_flat, H_top_m=H_top, ring_nasadem_share=ring_nas_share, offset_m=off, offset_source=off_src,
               n_stable=int(land.sum()), offset_iqr_m=[float(np.percentile(fab[land] - nas[land], q)) for q in (25, 75)] if land.sum() >= 100 else None,
               w2000_share_R=float(w2000.sum() / R.sum()), w2000_share_F=float((F & w2000).sum() / F.sum()) if F.any() else None,
               bed_above_plateau_share_F=float((nas2[F] >= H_flat).mean()) if F.any() else None,
               fab_missing_share_R=float((~np.isfinite(fab[R])).mean()), nas_missing_share_R=float((~np.isfinite(nas[R])).mean()),
               V_top_km3=float(last.V_R), A_top_km2=float(last.A_R), V_below_flat_km3=Vf,
               max_depth_m=float(H_top - np.nanmin(zb[Rs])), runtime_s=round(time.time() - t0, 1),
               finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (fo / f'{key}_run.json').write_text(json.dumps(rec, indent=1, default=float), encoding='utf-8')
    return rec


def cmd_run(a):
    warnings.filterwarnings('ignore')
    sb.committed()
    t = sb.order(a.size)
    if a.keys:
        t = t[t.key.isin(a.keys)]
    if a.part:
        i, n = map(int, a.part.split('/')); t = t.iloc[i - 1::n]
    polys, bars = sb.Polys(), Barriers()
    for key in t.key:
        f = OUT / 'run' / key
        if (f / f'{key}_run.json').exists() or (f / f'{key}_error.json').exists():
            continue
        t0 = time.time()
        try:
            r = run_one(key, polys, bars)
            print(key, 'ok', f"R {r['R_km2']:.1f} (uncut {r['R_uncut_km2']:.1f}) F {r['F_km2']:.1f} {r['plateau_rule'] or '-'} V {r['V_top_km3']:.4g}", f'{time.time() - t0:.0f}s', flush=True)
        except Exception as exc:
            if isinstance(exc, MemoryError) or 'allocat' in repr(exc):
                print(key, 'RESOURCE', repr(exc)[:120], flush=True); continue
            f.mkdir(parents=True, exist_ok=True)
            (f / f'{key}_error.json').write_text(json.dumps(dict(error=repr(exc), trace=traceback.format_exc())), encoding='utf-8')
            print(key, 'FAILED', repr(exc)[:160], flush=True)


def cmd_lock():
    sb.OUT = OUT
    shutil.copy(V1 / 'target_list.csv', OUT / 'target_list.csv')
    sb.PLAN = PLAN
    sb.cmd_lock()
    m = json.loads((OUT / 'lock_manifest.json').read_text(encoding='utf-8'))
    m['script_sha256'] = sb.sha(HERE / 'srtm_bed_v3.py'); m['v1_script_sha256'] = sb.sha(HERE / 'srtm_bed_v1.py')
    (OUT / 'lock_manifest.json').write_text(json.dumps(m, indent=0), encoding='utf-8')


def cmd_summary():
    t = pd.read_csv(V1 / 'target_list.csv'); rows = []
    for key in t.key:
        p = OUT / 'run' / key / f'{key}_run.json'
        if p.exists():
            r = json.loads(p.read_text(encoding='utf-8')); r.pop('missing_tiles', None); r.pop('offset_iqr_m', None); rows.append(r)
    s = t.merge(pd.DataFrame(rows), on='key')
    s['R_over_polygon'] = s.R_km2 / s.polygon_km2
    s['flag_R_beyond_polygon'] = s.R_over_polygon > 3.31
    s['flag_w2000_ge_5pct'] = s.w2000_share_R >= 0.05
    s['flag_curve_top_from_nasadem'] = s.ring_nasadem_share.fillna(0) >= 0.5
    s['A_top_over_R'] = s.A_top_km2 / s.R_km2
    s['flag_top_area_below_half_R'] = s.A_top_over_R < 0.5
    s['recommended'] = ~s.flag_R_beyond_polygon & ~s.flag_top_area_below_half_R
    s.to_csv(OUT / 'srtm_bed_summary.csv', index=False, float_format='%.10g')
    print('flags', s.filter(like='flag_').sum().to_dict(), '| recommended', int(s.recommended.sum()), '| V_top km3 recommended', round(s.V_top_km3[s.recommended].sum(), 1))
    print('targets', len(t), '| with curves', len(s), '| plateau rule', s.plateau_rule.value_counts(dropna=False).to_dict())


def cmd_score():
    """srtm_bed_v1.cmd_score on the v3 lock and runs."""
    sb.OUT = OUT
    shutil.copy(V1 / 'validation_match.csv', OUT / 'validation_match.csv')
    sb.cmd_score()


def main():
    p = argparse.ArgumentParser(); p.add_argument('cmd', choices=['run', 'lock', 'summary', 'score'])
    p.add_argument('--size', default='all'); p.add_argument('--part'); p.add_argument('--keys', nargs='*')
    a = p.parse_args()
    dict(run=lambda: cmd_run(a), lock=cmd_lock, summary=cmd_summary, score=cmd_score)[a.cmd]()


if __name__ == '__main__':
    main()
