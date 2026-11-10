"""Reservoirs impounded after February 2000 (or empty then): curves read from the SRTM (NASADEM) pre-impoundment bed, no
reconstruction. Plan docs/plans/2026-10-27-post2000-srtm-bed-plan.md (option A, user 2026-10-27).

Targets
  GDW     gdw_coverage_v1 category candidate_hl11 (>= 1 km2, 56 S - 60 N, polygon only in HydroLAKES v1.1): the GDW polygon
  HL      the 63 reservoirs of no_swbd_diag_v1 whose bed SRTM recorded: their HydroLAKES polygon
  valid.  the Brazil ANA reservoirs impounded in 2000 or later that have an official curve (matched to GDW by the dam point
          within 5 km); each is one of the targets above
Per reservoir, on a 30 m UTM grid (polygon bbox + margin; margin grown by 3 km while R is within 25 cells of the border, cap 45 km):
  other   HydroLAKES and GDW polygons that are not the reservoir (overlap with the own polygon < 50 % of either area)
  R       SWO occurrence >= 1 % outside other polygons, 8-connected to the own polygon's water cells
  F       FABDEM water plateau inside R: the most frequent FABDEM value (0.01 m) on R cells with occurrence >= 50 %, when it
          covers >= 20 % of them; F = R cells at that value; empty when FABDEM shows no water (filled after 2011-15)
  offset  median FABDEM - NASADEM on stable land (3-30 cells outside R, occurrence 0, >= 100 cells; otherwise the EGM96 to
          EGM2008 geoid difference at the centre); NASADEM + offset is on EGM2008
  bed     inside F: NASADEM + offset; cells under water in February 2000 keep that water surface (option A, share flagged);
          elsewhere in R: FABDEM (bare earth); where FABDEM is missing: NASADEM + offset
  H_top   median FABDEM on the outer ring of R outside F (as route1_global_run_v1.top_level), floored at the plateau
  curves  A and V every 0.25 m from the lowest bed cell to H_top: cells <= H inside R connected to the own polygon
Subcommands: list | run [--part i/n] [--size lo:hi] [--keys ...] | lock | score | summary
Outputs: 05_Results/by_date/2026-10-27/srtm_bed_v1/ (run/<key>/: _bed.tif, _grid.npz, _aev.csv, _run.json)
Usage: .venv/Scripts/python.exe -I 04_Code/scripts/srtm_bed_v1.py list|run|lock|score|summary
"""
from pathlib import Path
import sys, json, argparse, datetime, warnings, traceback, time, pickle, subprocess

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
import route1_global_run_v1 as gr
import aev_registry_v1 as reg
import aev_observation_share_v1 as aev

ROOT = gr.ROOT
PLAN = ROOT / 'docs/plans/2026-10-27-post2000-srtm-bed-plan.md'
OUT = ROOT / '05_Results/by_date/2026-10-27/srtm_bed_v1'
COVER = ROOT / '05_Results/by_date/2026-10-27/gdw_coverage_v1/gdw_reservoirs_category.csv'
DIAG = ROOT / '05_Results/by_date/2026-10-27/no_swbd_diag_v1/per_reservoir.csv'
ADM = ROOT / '05_Results/by_date/2026-10-22/newdata_admission_v2/admission.csv'
GDW_SHP = ROOT / '02_Data_Raw/GDW_v1_0_20261027/extracted/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp'
HL_PKL = ROOT / '03_Data_Processed/global_route1_20261007/polygons_type2.pkl'
GEOID = ROOT / '03_Data_Processed/ground_truth/twdb_five_20260918/vertical_grids'
GUARDED = [HERE / 'srtm_bed_v1.py', PLAN]
STEP, EDGE, E8 = 0.25, 25, np.ones((3, 3), bool)
sha = gr.sha


def committed():
    for f in GUARDED:
        rel = str(f.relative_to(ROOT)).replace(chr(92), '/')
        if subprocess.run(['git', 'status', '--porcelain', '--', rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() or \
                subprocess.run(['git', 'ls-files', '--error-unmatch', rel], cwd=ROOT, capture_output=True).returncode != 0:
            raise SystemExit(f'{rel} is not committed')
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------- list
def validation_matches():
    """Brazil ANA reservoirs without product, impounded in 2000 or later, matched to the nearest GDW dam point (<= 5 km)."""
    a = pd.read_csv(ADM, encoding='utf-8-sig')
    a = a[(a.source == 'Brazil ANA') & a.reason.astype(str).str.startswith('not in the global product') & a.lat.notna()].copy()
    g = pd.read_csv(COVER, low_memory=False)
    t = STRtree(shapely.points(g.LONG_RIV.values * np.cos(np.radians(g.LAT_RIV.values)), g.LAT_RIV.values))
    q = shapely.points(a.lon.values * np.cos(np.radians(a.lat.values)), a.lat.values)
    idx = t.nearest(q)
    m = g.iloc[idx].reset_index(drop=True)
    a = a.reset_index(drop=True)
    a['gdw_km'] = shapely.distance(q, shapely.points(m.LONG_RIV.values * np.cos(np.radians(m.LAT_RIV.values)), m.LAT_RIV.values)) * 111.32
    for c in ['GDW_ID', 'category', 'YEAR_DAM', 'PRE_YEAR', 'GRAND_ID', 'HYLAK_ID']:
        a[c] = m[c].values
    return a[(a.gdw_km < 5) & ((a.YEAR_DAM >= 2000) | (a.PRE_YEAR >= 2000))]


def cmd_list():
    g = pd.read_csv(COVER, low_memory=False)
    gd = g[g.category == 'candidate_hl11'][['GDW_ID', 'COUNTRY', 'AREA_POLY', 'YEAR_DAM', 'GRAND_ID', 'HYLAK_ID']]
    rows = [dict(key=f'g{int(r.GDW_ID)}', source='GDW', GDW_ID=int(r.GDW_ID), Hylak_id=np.nan, Grand_id=int(r.GRAND_ID), country=r.COUNTRY,
                 poly_area_km2=r.AREA_POLY, year_dam=r.YEAR_DAM) for r in gd.itertuples()]
    d = pd.read_csv(DIAG)
    d = d[d.klass == 'bed_seen_by_srtm']
    rows += [dict(key=f'h{int(r.Hylak_id)}', source='HL', GDW_ID=np.nan, Hylak_id=int(r.Hylak_id), Grand_id=r.Grand_id, country=r.Country,
                  poly_area_km2=r.polygon_km2, year_dam=r.year_dam) for r in d.itertuples()]
    t = pd.DataFrame(rows)
    v = validation_matches()
    lst = pd.read_csv(gr.ROOT / '05_Results/by_date/2026-10-07/route1_global_v1/global_list.csv', usecols=['Hylak_id', 'Grand_id'])
    gr2hl = dict(zip(lst.Grand_id, lst.Hylak_id))
    keys = []
    for r in v.itertuples():
        k = f'g{int(r.GDW_ID)}' if r.category == 'candidate_hl11' else f'h{int(gr2hl.get(r.GRAND_ID, -1))}'
        keys.append(k if k in set(t.key) else None)
    v['key'] = keys
    t['validation'] = t.key.map(dict(zip(v.key.dropna(), v.candidate[v.key.notna()])))
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / 'target_list.csv', index=False)
    v[['candidate', 'name', 'GDW_ID', 'category', 'YEAR_DAM', 'gdw_km', 'key']].to_csv(OUT / 'validation_match.csv', index=False)
    print('targets', len(t), t.source.value_counts().to_dict(), '| validation matched', int(v.key.notna().sum()), 'of', len(v),
          '| not in targets:', v.loc[v.key.isna(), 'candidate'].tolist())


# ---------------------------------------------------------------- polygons
class Polys:
    def __init__(self):
        meta, _, geoms, f = read(GDW_SHP, columns=['GDW_ID'])
        self.gdw_id = np.asarray(f[0]).astype(int)
        self.gdw = shapely.make_valid(shapely.from_wkb(geoms))
        self.tree = STRtree(self.gdw)
        self.by_id = dict(zip(self.gdw_id, self.gdw))
        with open(HL_PKL, 'rb') as fh:
            self.hl = {k: shapely.make_valid(shapely.from_wkb(v)) for k, v in pickle.load(fh).items()}

    def own(self, key):
        return self.by_id[int(key[1:])] if key[0] == 'g' else self.hl[int(key[1:])]

    def others(self, own, bounds):
        """Polygons in the window that are not the reservoir (overlap < 50 % of either area)."""
        box = shapely.box(*bounds)
        cand = list(self.gdw[self.tree.query(box)])
        meta, _, geoms, f = read(gr.GPKG, bbox=bounds, columns=['Hylak_id'])
        cand += list(shapely.make_valid(shapely.from_wkb(geoms))) if len(geoms) else []
        oa = own.area; out = []
        for p in cand:
            inter = p.intersection(own).area if p.intersects(own) else 0.0
            if inter < 0.5 * min(p.area, oa):
                out.append(p)
        return out


# ---------------------------------------------------------------- one reservoir
def geoid_diff(lon, lat):
    v = []
    for n in ('us_nga_egm96_15.tif', 'us_nga_egm08_25.tif'):
        with rasterio.open(GEOID / n) as s:
            v.append(float(next(s.sample([(lon, lat)]))[0]))
    return v[0] - v[1]


def build(key, Pw, polys, cap_m=45000.0):
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
        R = reg.connected_to((occ >= 1) & ~other, seed)
        touches = gr.near_edge(R, EDGE)
        if not touches:
            break
    fab, used, fmiss = gr.fabdem(lon, lat, tr, f'EPSG:{epsg}')
    return dict(epsg=epsg, tr=tr, lon0=float(c.x), lat0=float(c.y), Pm=Pm, other=other, occ=occ, swb=swb, nas=nas, fab=fab, R=R,
                margin_m=float(margin), touches=bool(touches), missing=sorted(set(missing + fmiss)))


def run_one(key, polys):
    t0 = time.time(); folder = OUT / 'run' / key; folder.mkdir(parents=True, exist_ok=True)
    g = build(key, polys.own(key), polys)
    R, Pm, occ, swb, nas, fab = g['R'], g['Pm'], g['occ'], g['swb'], g['nas'], g['fab']
    land = ndi.binary_dilation(R, E8, iterations=30) & ~ndi.binary_dilation(R, E8, iterations=3) & (occ == 0) & np.isfinite(nas) & np.isfinite(fab)
    if land.sum() >= 100:
        off, off_src = float(np.median(fab[land] - nas[land])), 'stable land'
    else:
        off, off_src = geoid_diff(g['lon0'], g['lat0']), 'geoid EGM96 - EGM2008'
    nas2 = nas + off
    w = R & (occ >= 50) & np.isfinite(fab)
    F = np.zeros(R.shape, bool); H_flat = np.nan; plateau_share = 0.0
    if w.sum() >= 20:
        vals, counts = np.unique(np.round(fab[w], 2), return_counts=True)
        i = counts.argmax(); plateau_share = float(counts[i] / counts.sum())
        if plateau_share >= 0.2:
            H_flat = float(vals[i]); F = R & (np.round(fab, 2) == H_flat)
    bed = np.where(np.isfinite(fab), fab, nas2)
    bed[F] = nas2[F]
    bed = np.where(R, bed, np.nan)
    w2000 = R & (swb == 255)
    ring = R & ~ndi.binary_erosion(R, E8) & ~F & np.isfinite(fab)
    H_top = float(np.median(fab[ring])) if ring.sum() >= 10 else H_flat
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
    tab[['H', 'A_R', 'V_R']].rename(columns={'H': 'H_m', 'A_R': 'A_km2', 'V_R': 'V_km3'}).to_csv(folder / f'{key}_aev.csv', index=False, float_format='%.6g')
    t = g['tr']; tr_s = rasterio.Affine(t.a, 0, t.c + s[1].start * t.a, 0, t.e, t.f + s[0].start * t.e)
    np.savez_compressed(folder / f'{key}_grid.npz', bed=zb.astype('float32'), R=Rs, seed=Ss, transform=np.array(tr_s)[:6], epsg=g['epsg'])
    with rasterio.open(folder / f'{key}_bed.tif', 'w', driver='GTiff', height=zb.shape[0], width=zb.shape[1], count=1, dtype='float32', nodata=np.nan,
                       crs=f"EPSG:{g['epsg']}", transform=tr_s, compress='deflate', predictor=3, tiled=True) as dst:
        dst.write(zb.astype('float32'), 1)
    last = tab.iloc[-1]
    Vf = float(np.sum(np.clip(H_flat - bed[F], 0, None)) * 9e-4 / 1e3) if F.any() else 0.0
    rec = dict(key=key, epsg=g['epsg'], margin_m=g['margin_m'], R_touches_edge=g['touches'], missing_tiles=g['missing'],
               polygon_km2=float(Pm.sum() * 9e-4), R_km2=float(R.sum() * 9e-4), F_km2=float(F.sum() * 9e-4), plateau_share=plateau_share,
               H_flat_m=H_flat, H_top_m=H_top, offset_m=off, offset_source=off_src, n_stable=int(land.sum()),
               offset_iqr_m=[float(np.percentile(fab[land] - nas[land], q)) for q in (25, 75)] if land.sum() >= 100 else None,
               w2000_share_R=float(w2000.sum() / R.sum()), w2000_share_F=float((F & w2000).sum() / F.sum()) if F.any() else None,
               bed_above_plateau_share_F=float((nas2[F] >= H_flat).mean()) if F.any() else None,
               fab_missing_share_R=float((~np.isfinite(fab[R])).mean()), nas_missing_share_R=float((~np.isfinite(nas[R])).mean()),
               V_top_km3=float(last.V_R), A_top_km2=float(last.A_R), V_below_flat_km3=Vf,
               max_depth_m=float(H_top - np.nanmin(zb[Rs])), runtime_s=round(time.time() - t0, 1),
               finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (folder / f'{key}_run.json').write_text(json.dumps(rec, indent=1, default=float), encoding='utf-8')
    return rec


def order(size='all'):
    t = pd.read_csv(OUT / 'target_list.csv')
    t = t.assign(_s=t.poly_area_km2.fillna(1)).sort_values('_s')
    if size != 'all':
        lo, hi = map(float, size.split(':')); t = t[(t._s >= lo) & (t._s < hi)]
    return t


def cmd_run(a):
    warnings.filterwarnings('ignore')
    committed()
    t = order(a.size)
    if a.keys:
        t = t[t.key.isin(a.keys)]
    if a.part:
        i, n = map(int, a.part.split('/')); t = t.iloc[i - 1::n]
    polys = Polys()
    for key in t.key:
        f = OUT / 'run' / key
        if (f / f'{key}_run.json').exists() or (f / f'{key}_error.json').exists():
            continue
        t0 = time.time()
        try:
            r = run_one(key, polys)
            print(key, 'ok', f"R {r['R_km2']:.1f} F {r['F_km2']:.1f} off {r['offset_m']:.1f} w2000 {r['w2000_share_R']:.2f} V {r['V_top_km3']:.4g}", f'{time.time() - t0:.0f}s', flush=True)
        except Exception as exc:
            if isinstance(exc, MemoryError) or 'allocat' in repr(exc):
                print(key, 'RESOURCE', repr(exc)[:120], flush=True); continue
            f.mkdir(parents=True, exist_ok=True)
            (f / f'{key}_error.json').write_text(json.dumps(dict(error=repr(exc), trace=traceback.format_exc())), encoding='utf-8')
            print(key, 'FAILED', repr(exc)[:160], flush=True)


def cmd_lock():
    head = committed(); t = pd.read_csv(OUT / 'target_list.csv')
    runs, failed, missing = {}, {}, []
    for key in t.key:
        f = OUT / 'run' / key
        if (f / f'{key}_run.json').exists():
            runs[key] = {p.name: sha(p) for p in sorted(f.glob('*')) if p.suffix in ('.tif', '.csv', '.json', '.npz')}
        elif (f / f'{key}_error.json').exists():
            failed[key] = json.loads((f / f'{key}_error.json').read_text(encoding='utf-8'))['error'][:300]
        else:
            missing.append(key)
    m = dict(plan=str(PLAN.relative_to(ROOT)).replace(chr(92), '/'), plan_sha256=sha(PLAN), git_head=head, script_sha256=sha(HERE / 'srtm_bed_v1.py'),
             list_sha256=sha(OUT / 'target_list.csv'), n_listed=int(len(t)), n_ok=len(runs), n_failed=len(failed), missing=missing, failed=failed,
             runs=runs, locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (OUT / 'lock_manifest.json').write_text(json.dumps(m, indent=0), encoding='utf-8')
    print('LOCKED ok', len(runs), 'failed', len(failed), 'missing', len(missing), flush=True)


# ---------------------------------------------------------------- scoring (after the lock)
def model_curve(key, levels):
    with np.load(OUT / 'run' / key / f'{key}_grid.npz') as d:
        z, R, seed = d['bed'].astype(float), d['R'], d['seed']
    tab = aev.curve_table(np.where(R, z, np.inf), seed, {'R': R}, levels)
    return tab.H.values, tab.A_R.values, tab.V_R.values


def cmd_score():
    import newdata_reference_v1 as nr
    lock = json.loads((OUT / 'lock_manifest.json').read_text(encoding='utf-8'))
    v = pd.read_csv(OUT / 'validation_match.csv').dropna(subset=['key'])
    a = pd.read_csv(ADM, encoding='utf-8-sig').set_index('candidate')
    grids = dict(imbituba=nr.Grid(nr.IBGE / 'hgeoHNOR2020__IMBITUBA__fator-conversao.txt'), santana=nr.Grid(nr.IBGE / 'hgeoHNOR2020__SANTANA__fator-conversao.txt'),
                 west=nr.Grid(nr.IBGE / 'MAPGEO2015_SIRGAS2000_recorte-oeste.txt'), west_poly=nr.poly(nr.IBGE / 'polig_MAPGEO2015_OESTE.txt'),
                 santana_poly=nr.poly(nr.IBGE / 'polig_hgeoHNOR2020_SANTANA.txt'))
    rows = []
    for r in v.itertuples():
        key = r.key; rec = dict(candidate=r.candidate, key=key)
        if key not in lock['runs']:
            rec['status'] = 'no product (' + lock['failed'].get(key, 'missing')[:80] + ')'; rows.append(rec); continue
        for f, h in lock['runs'][key].items():
            if sha(OUT / 'run' / key / f) != h:
                raise SystemExit(f'{key}/{f} differs from the lock')
        run = json.loads((OUT / 'run' / key / f'{key}_run.json').read_text(encoding='utf-8'))
        ar = a.loc[r.candidate]
        try:
            c, npl, unit, info = nr.brazil_curve(ar)
            off = nr.brazil_offset(float(ar.lon), float(ar.lat), grids)
        except Exception as exc:
            rec['status'] = 'reference error ' + repr(exc)[:80]; rows.append(rec); continue
        c = c.sort_values('H_native'); H = c.H_native.values + off['metres']
        H_pool = float(npl + off['metres']); Vo = float(np.interp(npl, c.H_native, c.volume_km3)); Ao = float(np.interp(npl, c.H_native, c.area_km2))
        lo = np.floor(min(H.min(), H_pool - 1) / STEP) * STEP
        levels = np.unique(np.append(np.arange(lo, H_pool, STEP), H_pool))
        Hm, Am, Vm = model_curve(key, levels)
        sel = (H <= H_pool + 1e-6) & (H >= Hm.min())
        ae = aev.fit_stats(np.interp(H[sel], Hm, Am), c.area_km2.values[sel]); ve = aev.fit_stats(np.interp(H[sel], Hm, Vm), c.volume_km3.values[sel])
        rec.update(status='scored', unit=unit, H_pool_m=H_pool, H_top_m=run['H_top_m'], H_top_minus_pool_m=run['H_top_m'] - H_pool,
                   area_ratio_official_to_R=Ao / run['R_km2'], identity_ok=bool(0.5 <= Ao / run['R_km2'] <= 2),
                   V_PE_pct=(float(Vm[-1]) / Vo - 1) * 100, A_PE_pct=(float(Am[-1]) / Ao - 1) * 100,
                   AE_NRMSE_pct=ae.get('NRMSE_pct'), VE_NRMSE_pct=ve.get('NRMSE_pct'), offset_m=run['offset_m'], w2000_share_R=run['w2000_share_R'],
                   F_km2=run['F_km2'], R_km2=run['R_km2'], bed_above_plateau_share_F=run['bed_above_plateau_share_F'], lon=float(ar.lon), lat=float(ar.lat))
        rows.append(rec)
        print(r.candidate, 'scored', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'validation_score.csv', index=False, float_format='%.5g')
    s = d[d.status.eq('scored')]
    ok = s[s.identity_ok]
    q = lambda x: [round(float(v), 1) for v in np.nanpercentile(x, [25, 50, 75])]
    summ = dict(n_validation=int(len(d)), n_scored=int(len(s)), n_identity_ok=int(len(ok)),
                abs_PE_median=float(ok.V_PE_pct.abs().median()), abs_PE_q=q(ok.V_PE_pct.abs()), signed_PE_q=q(ok.V_PE_pct),
                within_20pct=int((ok.V_PE_pct.abs() <= 20).sum()), AE_NRMSE_median=float(ok.AE_NRMSE_pct.median()), VE_NRMSE_median=float(ok.VE_NRMSE_pct.median()),
                area_PE_q=q(ok.A_PE_pct), H_top_minus_pool_q=q(ok.H_top_minus_pool_m),
                by_w2000={k: dict(n=int(len(g)), abs_PE_median=float(g.V_PE_pct.abs().median())) for k, g in ok.groupby(ok.w2000_share_R >= 0.05)},
                spearman_PE_offset=float(ok[['V_PE_pct', 'offset_m']].corr('spearman').iloc[0, 1]),
                not_scored=d.loc[~d.status.eq('scored'), ['candidate', 'status']].values.tolist())
    (OUT / 'validation_summary.json').write_text(json.dumps(summ, indent=1, default=float, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summ, indent=1, default=float, ensure_ascii=False))


def cmd_summary():
    t = pd.read_csv(OUT / 'target_list.csv'); rows = []
    for key in t.key:
        p = OUT / 'run' / key / f'{key}_run.json'
        if p.exists():
            r = json.loads(p.read_text(encoding='utf-8')); r.pop('missing_tiles', None); r.pop('offset_iqr_m', None); rows.append(r)
    s = t.merge(pd.DataFrame(rows), on='key')
    # flags added after the scoring (descriptive; nothing in the run changes): R leaking into the sea, rivers or a cascade
    # (R / polygon above the evaluated R/W0 maximum of the global product, 3.31), and Feb-2000 water inside R (option A)
    s['R_over_polygon'] = s.R_km2 / s.polygon_km2
    s['flag_R_beyond_polygon'] = s.R_over_polygon > 3.31
    s['flag_w2000_ge_5pct'] = s.w2000_share_R >= 0.05
    s['recommended'] = ~s.flag_R_beyond_polygon
    s.to_csv(OUT / 'srtm_bed_summary.csv', index=False, float_format='%.10g')
    print('flags', s.filter(like='flag_').sum().to_dict(), '| recommended', int(s.recommended.sum()), '| V_top km3 recommended', round(s.V_top_km3[s.recommended].sum(), 1))
    print('targets', len(t), '| with curves', len(s), '| V_top sum km3', round(s.V_top_km3.sum(), 1), '| R sum km2', round(s.R_km2.sum()))


def main():
    p = argparse.ArgumentParser(); p.add_argument('cmd', choices=['list', 'run', 'lock', 'score', 'summary'])
    p.add_argument('--size', default='all'); p.add_argument('--part'); p.add_argument('--keys', nargs='*')
    a = p.parse_args()
    dict(list=cmd_list, run=lambda: cmd_run(a), lock=cmd_lock, score=cmd_score, summary=cmd_summary)[a.cmd]()


if __name__ == '__main__':
    main()
