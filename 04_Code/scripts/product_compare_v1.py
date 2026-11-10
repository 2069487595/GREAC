"""Same-reservoir comparison with GRDL, ReGeom and 3D-LAKES (plan docs/plans/2026-10-08-route1-product-comparison-plan.md).

Sample: the 101 step-1 reservoirs and the 35 of group B (official curves through eval_expansion_score_v1.official; model
curves R1-C / C2 / GLOBathy-est as written by the locked scoring runs, SWO>=1% extent).
  area-aligned (GRDL, ReGeom, ours): V at the level whose area equals the official normal-pool area; V(A) NRMSE.
  in-band (3D-LAKES L1, L2, ours): grbd_compare_v1.band_scores inside the product's own elevation band, after
      ellipsoidal -> EGM2008 (h - N, us_nga_egm08_25.tif).
Writes area_aligned.csv, in_band.csv, coverage.json, aggregate.json to 05_Results/by_date/2026-10-08/product_compare_v1.
Usage: .venv/Scripts/python.exe 04_Code/scripts/product_compare_v1.py
"""
from pathlib import Path
import sys, json, warnings

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import pyproj
from pyproj import Transformer
from pyogrio.raw import read
import aev_registry_v1 as reg
import eval_expansion_registry_v1 as er
import eval_expansion_score_v1 as sc
import grbd_compare_v1 as gc
import route1_global_check_v1 as chk

ROOT = reg.ROOT
OUT = ROOT / '05_Results/by_date/2026-10-08/product_compare_v1'
RAW = ROOT / '02_Data_Raw'
GRDL = RAW / 'baseline_products_20261008/GRDL'
REGEOM = RAW / 'baseline_products_20261008/ReGeom/ReGeom/Area_Strg_Dp'
L1, L2 = RAW / 'L1_LAKE_A-E', RAW / 'L2 A-E product'
STEP1 = ROOT / '05_Results/by_date/2026-09-30/route1_expansion_score_v1'
HYB = ROOT / '05_Results/by_date/2026-10-01/route1_hybrid_v1'
OURS = ['R1-C', 'C2', 'GLOBathy-est']


def our_curve(batch, key, model):
    if batch == 'USBR-B':
        f = HYB / 'test_score' / key / f'{key}_{model}_curves.csv'
    elif model == 'R1-C':
        f = STEP1 / key / f'{key}_{model}_curves.csv'
    else:
        f = HYB / 'dev' / key / f'{key}_{model}_curves.csv'
    t = pd.read_csv(f)
    return t.H.values, t.A_swo1.values, t.V_swo1.values


def grdl(gid):
    f = GRDL / 'grdl' / f'{gid:05d}.csv'
    if not f.exists():
        return None
    t = pd.read_csv(f, sep=';', skiprows=4)
    return t.iloc[:, 1].values.astype(float), t.iloc[:, 2].values.astype(float) / 1e3  # km2, km3


def regeom(gid):
    f = REGEOM / f'{gid}.csv'
    if not f.exists():
        return None
    lines = f.read_text(encoding='latin-1').splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip().startswith('Depth'))
    t = pd.read_csv(f, skiprows=i + 1, header=None, encoding='latin-1').apply(pd.to_numeric, errors='coerce').dropna(how='all')
    return t.iloc[:, 1].values.astype(float), t.iloc[:, 2].values.astype(float) / 1e3


def area_aligned(A, V, cur, H_pool):
    """V at the level whose area equals the official pool area, and V(A) NRMSE over official levels."""
    o = np.argsort(A, kind='stable'); A, V = np.maximum.accumulate(A[o]), np.maximum.accumulate(V[o])
    A_pool = float(np.interp(H_pool, cur.H, cur.area_km2)); V_pool = float(np.interp(H_pool, cur.H, cur.volume_km3))
    short = bool(A.max() < A_pool)
    V_at = float(V[-1] if short else np.interp(A_pool, A, V))
    sel = (cur.H <= H_pool + 1e-6) & (cur.area_km2 >= A[A > 0].min() if (A > 0).any() else False) & (cur.area_km2 <= min(A.max(), A_pool))
    out = dict(short=short, A_top_over_A_pool=float(A.max() / A_pool), Vmax_PE_pct=(V_at / V_pool - 1) * 100, n_levels=int(np.sum(sel)))
    if np.sum(sel) >= 3:
        ref = cur.volume_km3[sel].values
        e = np.interp(cur.area_km2[sel].values, A, V) - ref
        rng = ref.max() - ref.min()
        out['VA_NRMSE_pct'] = float(np.sqrt(np.mean(e * e)) / rng * 100) if rng > 0 else None
    return out


def geoid(lon, lat):
    pyproj.datadir.append_data_dir(str(reg.GRIDS))
    v = Transformer.from_pipeline('+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad '
                                  '+step +proj=vgridshift +grids=us_nga_egm08_25.tif +multiplier=1 '
                                  '+step +proj=unitconvert +xy_in=rad +xy_out=deg').transform(lon, lat, 0.0)[2]
    return -abs(float(v))  # EGM2008 undulation is negative over the conterminous US (sample is CONUS only)


def lakes3d(folder, suffix, hid, N):
    f = folder / f'{hid}_{suffix}.csv'
    if not f.exists():
        return None
    t = pd.read_csv(f).iloc[:, :2].dropna().drop_duplicates()
    t.columns = ['E', 'A']
    t = t.sort_values('E')
    return t.E.values - N, t.A.values / 1e6


def main():
    warnings.filterwarnings('ignore')
    OUT.mkdir(parents=True, exist_ok=True)
    ev = chk.evaluated().dropna(subset=['Hylak_id'])
    ev['Hylak_id'] = ev.Hylak_id.astype(int)
    meta, _, _, f = read(chk.gr.GPKG, read_geometry=False, columns=['Hylak_id', 'Grand_id', 'Lake_area'],
                         where=f"Hylak_id IN ({','.join(map(str, ev.Hylak_id))})")
    hl = pd.DataFrame(dict(zip(meta['fields'], f))).set_index('Hylak_id')
    ratio = pd.read_csv(GRDL / 'grand_area_ratio.csv').set_index('GRAND_ID').area_ratio
    aa, band, cover = [], [], dict(n_sample=int(len(ev)))
    for r in ev.itertuples():
        cur, H_pool, off, _ = sc.official(r.batch, r.key)
        gid = int(hl.loc[r.Hylak_id, 'Grand_id']) if r.Hylak_id in hl.index else 0
        base = dict(batch=r.batch, key=r.key, Hylak_id=r.Hylak_id, Grand_id=gid, A_pool_km2=float(np.interp(H_pool, cur.H, cur.area_km2)))
        curves = {m: our_curve(r.batch, r.key, m) for m in OURS}
        for m, (H, A, V) in curves.items():
            native = (float(np.interp(H_pool, H, V)) / float(np.interp(H_pool, cur.H, cur.volume_km3)) - 1) * 100
            aa.append(dict(base, product=m, Vmax_PE_native_pct=native, **area_aligned(A, V, cur, H_pool)))
        for name, fn in [('GRDL', grdl), ('ReGeom', regeom)]:
            c = fn(gid) if gid > 0 else None
            if c is not None:
                aa.append(dict(base, product=name, grdl_area_ratio=float(ratio.get(gid, np.nan)) if name == 'GRDL' else None,
                               **area_aligned(c[0], c[1], cur, H_pool)))
        m = json.loads((er.INPUTS / f'{er.safe(r.key)}_inputs.json').read_text(encoding='utf-8'))
        N = geoid(m['lon'], m['lat'])
        for name, folder, suf in [('3D-LAKES L1', L1, 'L1'), ('3D-LAKES L2', L2, 'L2')]:
            c = lakes3d(folder, suf, r.Hylak_id, N)
            if c is None or len(c[0]) < 2:
                continue
            E, A = c
            lo, top = float(E.min()), float(min(E.max(), H_pool))
            n_in = int(((cur.H >= lo) & (cur.H <= top)).sum())
            if top - lo <= 0 or n_in < 2:
                band.append(dict(base, band=name, product=name, note='band empty or < 2 official levels', band_m=top - lo)); continue
            Vc = np.concatenate([[0], np.cumsum(np.diff(E) * (A[1:] + A[:-1]) / 2)]) / 1e3  # km2 * m -> km3
            fa = lambda h: float(np.interp(h, E, A)); fv = lambda h: float(np.interp(h, E, Vc))
            band.append(dict(base, band=name, product=name, band_m=top - lo, **gc.band_scores(fa, fv, cur, lo, top)))
            for mm, (H, Am, Vm) in curves.items():
                band.append(dict(base, band=name, product=mm, band_m=top - lo,
                                 **gc.band_scores(lambda h: float(np.interp(h, H, Am)), lambda h: float(np.interp(h, H, Vm)), cur, lo, top)))
    a = pd.DataFrame(aa); b = pd.DataFrame(band)
    a.to_csv(OUT / 'area_aligned.csv', index=False, float_format='%.6g'); b.to_csv(OUT / 'in_band.csv', index=False, float_format='%.6g')
    a['abs'] = a.Vmax_PE_pct.abs()
    agg = dict(area_aligned={}, in_band={})
    for name in ['GRDL', 'ReGeom']:
        keys = set(a[a['product'] == name].key)
        sub = a[a.key.isin(keys)]
        p = sub.pivot(index='key', columns='product', values='abs')
        agg['area_aligned'][name] = dict(n=len(keys), short=int(a[(a['product'] == name)].short.sum()),
            median_abs_Vmax_PE={c: float(p[c].median()) for c in p.columns}, mean_abs_Vmax_PE={c: float(p[c].mean()) for c in p.columns},
            median_VA_NRMSE={c: float(v) for c, v in sub.groupby('product').VA_NRMSE_pct.median().items()},
            wins={f'{o} better than {name}': int((p[o] < p[name]).sum()) for o in OURS})
    both = set(a[a['product'] == 'GRDL'].key) & set(a[a['product'] == 'ReGeom'].key)
    p = a[a.key.isin(both)].pivot(index='key', columns='product', values='abs')
    agg['area_aligned']['common'] = dict(n=len(both), median_abs_Vmax_PE={c: float(p[c].median()) for c in p.columns})
    g = a[a['product'] == 'GRDL'].set_index('key')
    big = g.A_pool_km2 > 100
    agg['area_aligned']['GRDL_by_size'] = {'<=100 km2': dict(n=int((~big).sum()), median=float(g['abs'][~big].median())),
                                           '>100 km2': dict(n=int(big.sum()), median=float(g['abs'][big].median()))}
    ok = b.dropna(subset=['AE_NRMSE_pct']) if 'AE_NRMSE_pct' in b else b
    for name in ['3D-LAKES L1', '3D-LAKES L2']:
        s = ok[ok.band == name].assign(absV=lambda x: x.band_V_PE_pct.abs())
        pa = s.pivot(index='key', columns='product', values='AE_NRMSE_pct')
        agg['in_band'][name] = dict(n=int(s.key.nunique()), median_band_m=float(s.groupby('key').band_m.first().median()),
            median_AE_NRMSE={c: float(v) for c, v in s.groupby('product').AE_NRMSE_pct.median().items()},
            median_abs_band_V_PE={c: float(v) for c, v in s.groupby('product').absV.median().items()},
            wins_AE={f'{o} better than {name}': int((pa[o] < pa[name]).sum()) for o in OURS})
    cover.update(GRDL=int((a['product'] == 'GRDL').sum()), ReGeom=int((a['product'] == 'ReGeom').sum()),
                 L1=int(b[b.band == '3D-LAKES L1'].key.nunique()), L2=int(b[b.band == '3D-LAKES L2'].key.nunique()),
                 with_Grand_id=int((hl.Grand_id > 0).sum()))
    (OUT / 'coverage.json').write_text(json.dumps(cover, indent=1), encoding='utf-8')
    (OUT / 'aggregate.json').write_text(json.dumps(agg, indent=1), encoding='utf-8')
    print(json.dumps(cover)); print(json.dumps(agg, indent=1))


if __name__ == '__main__':
    main()
