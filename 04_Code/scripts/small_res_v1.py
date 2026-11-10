"""Small-reservoir study (plan docs/plans/2026-10-27-small-reservoir-study-plan.md), step 1: candidates, locations and the
HydroLAKES polygon of each, and whether a product already exists. Reads no official area, capacity or level.

Sources
  USBR / TWDB  the 9 reservoirs excluded from the evaluation only because the official pool area is < 1 km2
               (eval_expansion_v1 admission rule 4); location from the evaluation inputs (USBR) or the identity table (TWDB)
  Brazil ANA   the new-data reservoirs with an official pool area < 1 km2 (admission table; dam coordinates)
  OWRB         the 48 surveys without a product at the new-data admission; location and extent from the official survey
               boundary in the GIS package
  USGS MO      the 25 water-supply lakes of SIR 2023-5108 and SIR 2024-5114; extent from the bounding box of each lake's
               item in the ScienceBase data releases (metadata only)
Identity: the HydroLAKES v1.0 polygon (any type) with the largest overlap with the official boundary (OWRB) or bounding box
(USGS MO), or the polygon containing / nearest to the point (<= 2 km; USBR, TWDB, ANA).
Product: the Hylak_id in the global product, the GDW extension, the high-latitude run or the remnant fix pass.
Output: 05_Results/by_date/2026-10-27/small_res_v1/inventory.csv
Usage: .venv/Scripts/python.exe -I 04_Code/scripts/small_res_v1.py inventory
"""
from pathlib import Path
import sys, json, re, argparse, subprocess

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import shapely
from pyogrio.raw import read
from pyproj import Transformer

ROOT = HERE.parents[1]
OUT = ROOT / '05_Results/by_date/2026-10-27/small_res_v1'
GPKG = ROOT / '02_Data_Raw/input_mead/HydroLAKES_polys_v10_wgs84.gpkg'
EVAL = ROOT / '05_Results/by_date/2026-09-30/eval_expansion_v1'
EVAL_IN = ROOT / '03_Data_Processed/ground_truth/eval_expansion_20260930/inputs'
ADM = ROOT / '05_Results/by_date/2026-10-22/newdata_admission_v2/admission.csv'
OWRB = ROOT / '02_Data_Raw/new_surveys_20261013/OWRB_Bathymetry_20261012'
MO_RELEASES = {'2021': '62e01487d34e952be9098222', '2022': '64aed670d34e70357a294adc', '2023': '65b29221d34e36a390452eb8'}
PRODUCTS = {'global': ROOT / '05_Results/by_date/2026-10-07/route1_global_v1/global_summary.csv',
            'gdw_ext': ROOT / '05_Results/by_date/2026-10-27/gdw_ext_v1/ext_summary.csv',
            'highlat': ROOT / '05_Results/by_date/2026-10-27/highlat_v1/hl_list.csv',
            'remnant': ROOT / '05_Results/by_date/2026-10-27/remnant_fix_v1/fix_list.csv'}


def hl_near(geom, pad=0.03):
    """HydroLAKES polygons around a geometry (WGS84)."""
    b = geom.bounds
    meta, _, geoms, f = read(GPKG, bbox=(b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad), columns=['Hylak_id', 'Lake_type', 'Lake_area', 'Pour_long', 'Pour_lat'])
    d = pd.DataFrame(dict(zip(meta['fields'], f)))
    d['geom'] = list(shapely.make_valid(shapely.from_wkb(geoms))) if len(geoms) else []
    return d


GDW_SHP = ROOT / '02_Data_Raw/GDW_v1_0_20261027/extracted/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp'
_GDW = {}


def gdw_match(geom):
    """Fallback when HydroLAKES has no polygon: the GDW reservoir polygon with the largest overlap, or the nearest (<= 2 km)."""
    if not _GDW:
        meta, _, geoms, f = read(GDW_SHP, columns=['GDW_ID', 'LONG_RIV', 'LAT_RIV', 'AREA_POLY'])
        _GDW['d'] = pd.DataFrame(dict(zip(meta['fields'], f))); _GDW['g'] = shapely.make_valid(shapely.from_wkb(geoms))
        _GDW['t'] = shapely.STRtree(_GDW['g'])
    d, g, t = _GDW['d'], _GDW['g'], _GDW['t']
    idx = t.query(geom.buffer(0.02))
    if len(idx) == 0:
        return dict(match='none')
    if geom.geom_type in ('Polygon', 'MultiPolygon'):
        iou = np.array([g[i].intersection(geom).area / g[i].union(geom).area for i in idx])
        if iou.max() > 0:
            i = idx[int(np.argmax(iou))]
            return dict(match='GDW overlap', iou=float(iou.max()), GDW_ID=int(d.GDW_ID[i]), GDW_area_km2=float(d.AREA_POLY[i]), Pour_long=float(d.LONG_RIV[i]), Pour_lat=float(d.LAT_RIV[i]))
        geom = geom.centroid
    dist = np.array([g[i].distance(geom) for i in idx]) * 111.32 * np.cos(np.radians(geom.y))
    j = int(np.argmin(dist)); i = idx[j]
    if dist[j] > 2.0:
        return dict(match='none')
    return dict(match='GDW nearest', dist_km=float(dist[j]), GDW_ID=int(d.GDW_ID[i]), GDW_area_km2=float(d.AREA_POLY[i]), Pour_long=float(d.LONG_RIV[i]), Pour_lat=float(d.LAT_RIV[i]))


def match(geom):
    m = hl_match(geom)
    return m if m.get('Hylak_id') is not None else {**gdw_match(geom), **({'hl_nearest_km': m['nearest_km']} if 'nearest_km' in m else {})}


def hl_match(geom):
    d = hl_near(geom)
    if d.empty:
        return dict(match='none')
    if geom.geom_type in ('Polygon', 'MultiPolygon'):
        inter = np.array([g.intersection(geom).area for g in d.geom]); union = np.array([g.union(geom).area for g in d.geom])
        i = int(np.argmax(inter / union))
        if inter[i] > 0:
            r = d.iloc[i]
            return dict(match='overlap', iou=float(inter[i] / union[i]), Hylak_id=int(r.Hylak_id), HL_Lake_type=int(r.Lake_type), HL_area_km2=float(r.Lake_area),
                        Pour_long=float(r.Pour_long), Pour_lat=float(r.Pour_lat))
        geom = geom.centroid
    dist = np.array([g.distance(geom) for g in d.geom]) * 111.32 * np.cos(np.radians(geom.y))
    i = int(np.argmin(dist)); r = d.iloc[i]
    if dist[i] > 2.0:
        return dict(match='none', nearest_km=float(dist[i]))
    return dict(match='contains' if dist[i] == 0 else 'nearest', dist_km=float(dist[i]), Hylak_id=int(r.Hylak_id), HL_Lake_type=int(r.Lake_type),
                HL_area_km2=float(r.Lake_area), Pour_long=float(r.Pour_long), Pour_lat=float(r.Pour_lat))


def owrb_boundary(folder):
    z = next(folder.glob('*.zip'))
    import zipfile
    names = [n for n in zipfile.ZipFile(z).namelist() if n.lower().endswith('.shp')]
    import pyogrio
    poly_layers = [n for n in names if 'Polygon' in (pyogrio.read_info(f'/vsizip/{z.as_posix()}/{n}')['geometry_type'] or '')]
    shp = next((n for n in poly_layers if 'bound' in n.lower()), None) or next((n for n in poly_layers if re.search('shore|_lake', n.lower())), None)         or (poly_layers[0] if poly_layers else None)
    if shp is None:
        return None, None
    path = f'/vsizip/{z.as_posix()}/{shp}'
    info = pyogrio.read_info(path)
    meta, _, geoms, _ = read(path, read_geometry=True)
    g = shapely.union_all(shapely.make_valid(shapely.from_wkb(geoms)))
    if info['crs'] and info['crs'] != 'EPSG:4326':
        g = shapely.ops.transform(Transformer.from_crs(info['crs'], 4326, always_xy=True).transform, g)
    return g, shp


def mo_items():
    rows = []
    for year, pid in MO_RELEASES.items():
        js = subprocess.run(['curl', '-s', '--max-time', '60', f'https://www.sciencebase.gov/catalog/items?parentId={pid}&format=json&fields=title,spatial&max=50'],
                            capture_output=True, text=True).stdout
        for it in json.loads(js).get('items', []):
            bb = (it.get('spatial') or {}).get('boundingBox')
            if bb:
                rows.append(dict(release=year, title=it['title'], box=shapely.box(bb['minX'], bb['minY'], bb['maxX'], bb['maxY'])))
    return rows


def cmd_inventory():
    import shapely.ops
    globals()['shapely'].ops = shapely.ops
    rows = []
    adm = {s: pd.read_csv(EVAL / f'admission_{s}.csv', encoding='utf-8-sig') for s in ('twdb', 'usbr')}
    ida = pd.read_csv(EVAL / 'identity_auto.csv', encoding='utf-8-sig').set_index('key')
    for s, a in adm.items():
        for r in a[a.reason.astype(str).str.startswith('rule 4')].itertuples():
            if s == 'twdb':
                lon, lat = float(ida.loc[r.key, 'lon']), float(ida.loc[r.key, 'lat'])
            else:
                j = json.loads((EVAL_IN / f'{r.key}_inputs.json').read_text(encoding='utf-8')); lon, lat = j['lon'], j['lat']
            rows.append(dict(source=s.upper(), key=r.key, name=r.key, lon=lon, lat=lat, **match(shapely.Point(lon, lat))))
    a = pd.read_csv(ADM, encoding='utf-8-sig')
    for r in a[(a.source == 'Brazil ANA') & (a.official_area_npl_km2 < 1)].itertuples():
        if not (np.isfinite(r.lon) and np.isfinite(r.lat)):
            rows.append(dict(source='ANA', key=r.candidate, name=r.name, match='no coordinates')); continue
        rows.append(dict(source='ANA', key=r.candidate, name=r.name, lon=r.lon, lat=r.lat, **match(shapely.Point(r.lon, r.lat))))
    for r in a[(a.source == 'OWRB') & a.reason.astype(str).str.startswith('not in the global product')].itertuples():
        g, shp = owrb_boundary(OWRB / r.candidate)
        if g is None:
            rows.append(dict(source='OWRB', key=r.candidate, name=r.candidate, match='no boundary in the GIS package')); continue
        c = g.representative_point()
        rows.append(dict(source='OWRB', key=r.candidate, name=r.candidate, lon=c.x, lat=c.y, boundary_layer=shp, **match(g)))
    for it in mo_items():
        c = it['box'].centroid
        rows.append(dict(source='USGS_MO', key='MO_' + re.sub(r'[^A-Za-z0-9]+', '_', it['title'])[:60], name=it['title'], release=it['release'], lon=c.x, lat=c.y, **match(it['box'])))
    d = pd.DataFrame(rows)
    for name, p in PRODUCTS.items():
        ids = set(pd.read_csv(p, usecols=['Hylak_id']).Hylak_id.astype(int)) if p.exists() else set()
        d[f'in_{name}'] = d.Hylak_id.isin(ids)
    d['in_product'] = d.filter(like='in_').any(axis=1)
    OUT.mkdir(parents=True, exist_ok=True)
    d.drop(columns=[c for c in d.columns if c == 'geom'], errors='ignore').to_csv(OUT / 'inventory.csv', index=False, float_format='%.10g')     # %.6g truncated Hylak_id > 999,999 (2026-10-27)
    print(d.groupby('source').agg(n=('key', 'size'), matched=('Hylak_id', lambda x: int(x.notna().sum())), in_product=('in_product', 'sum')).to_string())
    print('HydroLAKES area of matched polygons (km2), quartiles by source:')
    print(d.groupby('source').HL_area_km2.quantile([.25, .5, .75]).unstack().round(2).to_string())


# ---------------------------------------------------------------- step 2: the product method on the HydroLAKES polygons
DATA = ROOT / '03_Data_Processed/small_res_20261027'
GUARDED = [HERE / 'small_res_v1.py', ROOT / 'docs/plans/2026-10-27-small-reservoir-study-plan.md']


def study_list():
    d = pd.read_csv(OUT / 'inventory.csv')
    return d[d.Hylak_id.notna() & ~d.in_product].drop_duplicates('Hylak_id')


def cmd_run(a):
    """route1_global_run_v1.run_one and r2_global_run_v1.run_one, unchanged, on the matched HydroLAKES polygons of the
    reservoirs without a product (frozen code; only the module paths are redirected). The GDW-polygon matches are left out:
    the method needs the GLOBathy parameters of a HydroLAKES polygon."""
    import warnings, traceback, time, pickle
    import route1_global_run_v1 as gr
    import r2_global_run_v1 as r2
    warnings.filterwarnings('ignore')
    for f in GUARDED:
        rel = str(f.relative_to(ROOT)).replace(chr(92), '/')
        if subprocess.run(['git', 'status', '--porcelain', '--', rel], cwd=ROOT, capture_output=True, text=True).stdout.strip():
            raise SystemExit(f'{rel} is not committed')
    head = r2.committed()
    gr.OUT, gr.RUN, gr.DATA, gr.INPUTS = OUT, OUT / 'run', DATA, DATA / 'inputs'
    d = study_list()
    ids = [int(i) for i in d.Hylak_id]
    meta, _, geoms, f = read(GPKG, columns=['Hylak_id'], where=f"Hylak_id IN ({','.join(map(str, ids))})")
    polys = dict(zip(np.asarray(f[0]).astype(int), geoms))
    dm = gr.erun.dmax_all(); gr.RUN.mkdir(parents=True, exist_ok=True)
    for r in d.itertuples():
        hid = int(r.Hylak_id); fo = gr.RUN / str(hid); t0 = time.time()
        if not (fo / f'{hid}_run.json').exists() and not (fo / f'{hid}_error.json').exists():
            try:
                rec = gr.run_one(hid, shapely.from_wkb(polys[hid]), dm, 45000.0)
                print(hid, r.source, 'tc ok', rec['regime'][:5], f"W0 {rec['W0_km2']:.2f} F {rec['F_km2']:.2f}", f'{time.time() - t0:.0f}s', flush=True)
            except Exception as exc:
                fo.mkdir(parents=True, exist_ok=True)
                (fo / f'{hid}_error.json').write_text(json.dumps(dict(error=repr(exc), trace=traceback.format_exc())), encoding='utf-8')
                print(hid, r.source, 'FAILED tc', repr(exc)[:140], flush=True); continue
        if (fo / f'{hid}_run.json').exists() and not (fo / f'{hid}_R2_run.json').exists() and not (fo / f'{hid}_R2_error.json').exists():
            row = pd.Series(dict(Hylak_id=hid, Pour_long=r.Pour_long, Pour_lat=r.Pour_lat))
            try:
                rec = r2.run_one(type('R', (), row.to_dict())(), gr.RUN, head, recheck=True)
                print(hid, 'dam', rec['status'], f"D {rec['D_dam_m']:.1f}", flush=True)
            except Exception as exc:
                (fo / f'{hid}_R2_error.json').write_text(json.dumps(dict(error=repr(exc), trace=traceback.format_exc())), encoding='utf-8')
                print(hid, 'FAILED dam', repr(exc)[:140], flush=True)


def cmd_lock():
    import datetime, hashlib
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    d = study_list(); runs, failed = {}, {}
    for hid in d.Hylak_id.astype(int):
        fo = OUT / 'run' / str(hid)
        if (fo / f'{hid}_run.json').exists():
            runs[int(hid)] = {p.name: sha(p) for p in sorted(fo.glob('*')) if p.suffix in ('.tif', '.csv', '.json')}
        elif (fo / f'{hid}_error.json').exists():
            failed[int(hid)] = json.loads((fo / f'{hid}_error.json').read_text(encoding='utf-8'))['error'][:300]
    m = dict(inventory_sha256=sha(OUT / 'inventory.csv'), script_sha256=sha(HERE / 'small_res_v1.py'), n_listed=int(len(d)), n_ok=len(runs),
             n_failed=len(failed), failed=failed, runs=runs, locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (OUT / 'lock_manifest.json').write_text(json.dumps(m, indent=0), encoding='utf-8')
    print('LOCKED ok', len(runs), 'failed', len(failed))


# ---------------------------------------------------------------- step 3: scoring (after the lock)
FT, ACRE_KM2, ACFT_KM3 = 0.3048, 0.00404685642, 1.23348183755e-6
G1, G2 = ROOT / '05_Results/by_date/2026-10-07/route1_global_v1', ROOT / '05_Results/by_date/2026-10-10/r2_global_v1'
EXT = ROOT / '05_Results/by_date/2026-10-27/gdw_ext_v1'
INPUTS = {'global': ROOT / '03_Data_Processed/global_route1_20261007/inputs', 'gdw_ext': ROOT / '03_Data_Processed/gdw_ext_20261027/inputs',
          'study': ROOT / '03_Data_Processed/small_res_20261027/inputs'}


def mo_ref(name, mo):
    n = re.sub(r'Number Three', 'Number 3', re.sub(r'Number Two', 'Number 2', re.sub(r'Number One', 'Number 1', name)))
    hits = [r for r in mo.itertuples() if r.lake.lower() in n.lower()]
    return max(hits, key=lambda r: len(r.lake)) if hits else None


def reference(r, adm, owrb, mo, nr, grids):
    """Official curve (H EGM2008 m, area km2, volume km3; one row when only the pool point is known), the pool level, a note
    and whether the full curve is available."""
    import eval_expansion_score_v1 as es
    import eval_expansion_registry_v1 as er
    if r.source in ('USBR', 'TWDB'):
        cur, H_pool, off, f = es.official(r.source, r.key)
        return cur[['H', 'area_km2', 'volume_km3']].reset_index(drop=True), float(H_pool), 'official curve', True
    if r.source == 'ANA':
        a = adm.loc[r.key]; c, npl, unit, info = nr.brazil_curve(a); off = nr.brazil_offset(float(a.lon), float(a.lat), grids)
        c = c.sort_values('H_native')
        return pd.DataFrame(dict(H=c.H_native.values + off['metres'], area_km2=c.area_km2.values, volume_km3=c.volume_km3.values)), float(npl + off['metres']), unit, True
    if r.source == 'OWRB':
        if r.key not in owrb.index:
            return None, None, 'no stated normal-pool values in the linked report', False
        o = owrb.loc[r.key]; off = er.egm_offset(float(r.lon), float(r.lat), o.datum)
        H = float(o.npl_ft) * FT + off['metres']
        return pd.DataFrame(dict(H=[H], area_km2=[o.area_acres * ACRE_KM2], volume_km3=[o.capacity_acft * ACFT_KM3])), H, o.datum, False
    m = mo_ref(r.name, mo)
    if m is None:
        return None, None, 'lake not found in the report tables', False
    off = er.egm_offset(float(r.lon), float(r.lat), 'NAVD88'); H = float(m.spillway_ft_NAVD88) * FT + off['metres']
    return pd.DataFrame(dict(H=[H], area_km2=[m.area_acres * ACRE_KM2], volume_km3=[m.capacity_acft * ACFT_KM3])), H, m.where, False


def cmd_score():
    import warnings
    warnings.filterwarnings('ignore')
    import route1_global_run_v1 as gr
    import newdata_score_v1 as ns
    import newdata_reference_v1 as nr
    lock = json.loads((OUT / 'lock_manifest.json').read_text(encoding='utf-8'))
    inv = pd.read_csv(OUT / 'inventory.csv')
    adm = pd.read_csv(ADM, encoding='utf-8-sig').set_index('candidate')
    owrb = pd.read_csv(HERE / 'small_res_owrb_reference_v1.csv').set_index('candidate')
    mo = pd.read_csv(HERE / 'small_res_mo_reference_v1.csv')
    grids = dict(imbituba=nr.Grid(nr.IBGE / 'hgeoHNOR2020__IMBITUBA__fator-conversao.txt'), santana=nr.Grid(nr.IBGE / 'hgeoHNOR2020__SANTANA__fator-conversao.txt'),
                 west=nr.Grid(nr.IBGE / 'MAPGEO2015_SIRGAS2000_recorte-oeste.txt'), west_poly=nr.poly(nr.IBGE / 'polig_MAPGEO2015_OESTE.txt'),
                 santana_poly=nr.poly(nr.IBGE / 'polig_hgeoHNOR2020_SANTANA.txt'))
    rows = []
    for r in inv[inv.Hylak_id.notna()].drop_duplicates('Hylak_id').itertuples():
        hid = int(r.Hylak_id)
        where = 'global' if r.in_global else 'gdw_ext' if r.in_gdw_ext else 'study' if str(hid) in lock['runs'] else None
        base = dict(source=r.source, key=r.key, name=r.name, Hylak_id=hid, HL_area_km2=r.HL_area_km2, product=where)
        if where is None:
            rows.append(dict(base, status='no product')); continue
        ns.R1, ns.R2 = {'global': (G1, G2), 'gdw_ext': (EXT, EXT), 'study': (OUT, OUT)}[where]
        gr.INPUTS = INPUTS[where]
        try:
            cur, H_pool, note, full = reference(r, adm, owrb, mo, nr, grids)
        except Exception as exc:
            rows.append(dict(base, status='reference error ' + repr(exc)[:80])); continue
        if cur is None:
            rows.append(dict(base, status=note)); continue
        run = json.loads((ns.R1 / 'run' / str(hid) / f'{hid}_run.json').read_text(encoding='utf-8'))
        r2p = ns.R2 / 'run' / str(hid) / f'{hid}_R2_run.json'
        r2s = json.loads(r2p.read_text(encoding='utf-8'))['status'] if r2p.exists() else 'no TC-dam record'
        for m in ns.score_one(hid, cur, H_pool, full):
            rows.append(dict(base, status='scored', reference=note, R_km2=run['R_km2'], W0_km2=run['W0_km2'], F_km2=run['F_km2'], regime=run['regime'][:5],
                             rule_C_fallback=run['rule_C_fallback'] is not None, R2_status=r2s, H_top_minus_pool_m=run['H_top_m'] - H_pool,
                             identity_ratio=m['A_pool_official_km2'] / run['R_km2'] if run['R_km2'] > 0 else np.nan, **m))
    d = pd.DataFrame(rows)
    d['identity_ok'] = d.identity_ratio.between(0.5, 2)
    d.to_csv(OUT / 'score_per_reservoir.csv', index=False, float_format='%.5g')
    s = d[d.status.eq('scored') & d.identity_ok]
    print('reservoirs scored', s.key.nunique(), '| identity excluded', d[d.status.eq('scored') & ~d.identity_ok].key.nunique(),
          '| not scored', d[~d.status.eq('scored')].drop_duplicates('key').status.value_counts().to_dict())
    print(s.groupby('method').Vmax_PE_pct.agg(n='size', abs_median=lambda x: x.abs().median(), signed_median='median').round(1).to_string())
    print(s.groupby(['product', 'method']).Vmax_PE_pct.agg(lambda x: x.abs().median()).round(1).unstack().to_string())


# ---------------------------------------------------------------- step 4: candidate fix (plan section 2b)
SEED, KS = 20261027, (1, 2)


def cmd_fixtest():
    """R widened by k cells for the volume integral (cells <= H, connected), TC-dam and TC-cap; k chosen on the development
    half (>= 3 points better than k = 0), tested once on the test half."""
    import warnings
    warnings.filterwarnings('ignore')
    from scipy import ndimage as ndi, stats
    import route1_global_run_v1 as gr
    import newdata_score_v1 as ns
    d = pd.read_csv(OUT / 'score_per_reservoir.csv')
    s = d[d.status.eq('scored') & d.identity_ok & d.method.eq('TC-dam')].drop_duplicates('key').reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    s['half'] = ''
    for src, g in s.groupby('source'):
        idx = g.index.values[rng.permutation(len(g))]
        s.loc[idx[: (len(idx) + 1) // 2], 'half'] = 'dev'; s.loc[idx[(len(idx) + 1) // 2:], 'half'] = 'test'
    rows = []
    for r in s.itertuples():
        hid = int(r.Hylak_id)
        ns.R1, ns.R2 = {'global': (G1, G2), 'gdw_ext': (EXT, EXT), 'study': (OUT, OUT)}[r.product]
        gr.INPUTS = INPUTS[r.product]
        g, F, H_flat, surf, info = ns.surfaces(hid)
        R0 = g['swo1']
        for k in (0,) + KS:
            gk = dict(g, swo1=ndi.binary_dilation(R0, np.ones((3, 3), bool), iterations=k) if k else R0)
            for name in ('TC-dam', 'TC-cap'):
                z, region = surf[name]
                tab = gr.curves(z, region, gk, F, r.H_pool_m)
                v = float(tab.iloc[-1].V_swo1); a = float(tab.iloc[-1].A_swo1)
                rows.append(dict(key=r.key, source=r.source, product=r.product, half=r.half, k=k, method=name,
                                 PE=(v / r.Vmax_official_km3 - 1) * 100, A_ratio=a / r.A_pool_official_km2))
    t = pd.DataFrame(rows); t.to_csv(OUT / 'fixtest_per_reservoir.csv', index=False, float_format='%.5g')
    dev = t[t.half.eq('dev') & t.method.eq('TC-dam')].groupby('k').PE.apply(lambda x: x.abs().median())
    base = dev.loc[0]; ok = {k: v for k, v in dev.items() if k and base - v >= 3}
    k_star = min(ok, key=ok.get) if ok else 0
    out = dict(seed=SEED, n_dev=int(s.half.eq('dev').sum()), n_test=int(s.half.eq('test').sum()), dev_abs_median={int(k): float(v) for k, v in dev.items()}, k_chosen=int(k_star))
    if k_star:
        for name in ('TC-dam', 'TC-cap'):
            x = t[t.half.eq('test') & t.method.eq(name)].pivot(index='key', columns='k', values='PE')
            a, b = x[k_star].abs(), x[0].abs()
            out[f'test_{name}'] = dict(n=int(len(x)), abs_median_k=float(a.median()), abs_median_0=float(b.median()), better=int((a < b).sum()), worse=int((a > b).sum()),
                                       p_one_sided=float(stats.wilcoxon(a, b, alternative='less').pvalue), signed_median_k=float(x[k_star].median()), signed_median_0=float(x[0].median()))
    ar = t[t.method.eq('TC-dam')].groupby('k').A_ratio.median()
    out['area_ratio_median_by_k'] = {int(k): float(v) for k, v in ar.items()}
    (OUT / 'fixtest_summary.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser(); p.add_argument('cmd', choices=['inventory', 'run', 'lock', 'score', 'fixtest']); a = p.parse_args()
    dict(inventory=cmd_inventory, run=lambda: cmd_run(a), lock=cmd_lock, score=cmd_score, fixtest=cmd_fixtest)[a.cmd]()


if __name__ == '__main__':
    main()
