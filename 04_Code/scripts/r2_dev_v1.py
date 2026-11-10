"""R2 development harness (protocol docs/plans/2026-10-08-route1-r2-development-protocol.md, frozen e9122f1).

DEVELOPMENT SET ONLY. Reservoirs come from resplit_v1/resplit.csv with set == 'development' (66, kind 'exp'), the 17 non-US
reservoirs of the SRTM-visible-bed validation (kind 'nonus') and the 12 reservoirs of the first small-scale work (kind
'old': D01-D06 of the original development, T01-T06 of the original USBR test; their own input grids and official curves,
HydroLAKES ids matched by pour point on W0 and area). Validation and test reservoirs are refused by the loader.
A candidate changes only the profile formula of the frozen kernel
    slope_now = s * (x / side) * ratio ** alpha_eff
through a parameter vector par = [anchor, alpha_base, D0_m, D1_m]:
    anchor      0 = endpoints at the second land cell (frozen), 1 = endpoints at the first land cell (shore), with
                ratio = min(1, (current + 2) / total)
    alpha_base  -1 = eq. 7 rule of the reservoir, otherwise the value (1 = released code, 0 = off)
    D0_m, D1_m  width dependence of the sediment correction: alpha_eff = alpha_base * clip((D - D0) / (D1 - D0), 0, 1),
                D = chord length in metres; D1 <= D0 means no width dependence
    --wpar W0,W1 (optional) makes alpha_base depend on the RESERVOIR's effective width W = 4 * mean distance to the shore
                inside F (metres): alpha_base *= clip((W - W0) / (W1 - W0), 0, 1)
    --dam R,Q (optional) rescales the reconstructed depth below H_seed inside F so that its maximum equals the dam height
                read from the DEM: D_dam = H_flat - (Q-th percentile of FABDEM within R metres of the HydroLAKES pour point,
                outside F and W0), target = D_dam + (H_seed - H_flat). No rescaling when the target is not positive.
    --guard     with --dam: when D_dam is not positive (or cannot be read) the frozen R1-C raster is returned instead
    --below     with --dam: the percentile is taken over the cells BELOW H_flat only (the valley below the dam; canyon
                walls and hillsides inside the disc no longer enter the statistic)
    --extrap    with --dam --below: the reading is extrapolated back to the dam for rivers that keep falling below it,
                D_dam = B(R) - max(0, B(2R) - B(R)), B(r) = the --below reading within r metres
    --shape edt the Tao recession is replaced by a cone: depth below H_seed proportional to the distance to the edge of F
                (ablation of the shape; needs --dam)
F, slope fitting and stretching, seed cap at the locked H_seed, recession order and neighbour weights are the frozen ones.
  --cache [--part i/n]                 fit the frozen initialization once per reservoir and store it
  --run NAME --par a,b,c,d [--part]    run one candidate, write r2_dev/<NAME>_<part>.csv
  --baseline                           metrics of the locked R1-C rasters (NAME = K0)
  --summary NAME [NAME ...]            tables by width class
Outputs: 05_Results/by_date/2026-10-08/r2_dev_v1/ ; cache: 03_Data_Processed/r2_dev_20261008/init/
"""
from pathlib import Path
import sys, json, argparse, warnings

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import srtm_bed_validation_v1 as v
import eval_expansion_registry_v1 as er
import eval_expansion_score_v1 as sc
import aev_observation_share_v1 as aev
import aev_registry_v1 as reg
import usbr_test_registry_v1 as tr
import usbr_official_curves_v1 as oc
import route1_seed_level_v1 as sl

sys.path.insert(0, str(v.reg.ROOT / '04_Code/versions/m0_m1_m2_runner_v4'))
import geometry_adapter as ga
from numba import njit

ROOT = v.reg.ROOT
kernel, run2, r1 = ga.kernel, ga.tao_run, v.r1
OUT = ROOT / '05_Results/by_date/2026-10-08/r2_dev_v1'
CACHE = ROOT / '03_Data_Processed/r2_dev_20261008/init'
SPLIT = ROOT / '05_Results/by_date/2026-10-08/resplit_v1/resplit.csv'
STEP1_RUN = ROOT / '05_Results/by_date/2026-09-30/route1_expansion_run_v1'
B_RUN = ROOT / '05_Results/by_date/2026-10-01/route1_hybrid_v1/test'
D_RUN = ROOT / '05_Results/by_date/2026-09-29/route1_seed_level_v1'
T_RUN = ROOT / '05_Results/by_date/2026-09-29/route1_test_v1'
GLIST = ROOT / '05_Results/by_date/2026-10-07/route1_global_v1/global_list.csv'
OLD = {'D01': 9490, 'D02': 9560, 'D03': 9514, 'D04': 9524, 'D05': 9343, 'D06': 839,
       'T01': 9070, 'T02': 8978, 'T03': 8297, 'T04': 110538, 'T05': 9350, 'T06': 8945}
WIDTH_EDGES = (24.7, 40.6)  # tertile edges of the 136 (resplit_inputs_draft.csv), cells
line_at = kernel.line_at


@njit(cache=True)
def directional_r2(slope, origin, water, rr, cc, nr, nc, dem_h, res, par):
    shore = par[0] > 0.5; alpha = par[1]; D0 = par[2]; D1 = par[3]
    step = np.hypot(rr - nr, cc - nc); vr = (rr - nr) / step; vc = (cc - nc) / step
    tr = float(rr); tc = float(cc)
    while water[int(tr), int(tc)] != 0:
        tr += vr; tc += vc
    tr -= vr; tc -= vc
    current = np.hypot(rr - int(tr), cc - int(tc))
    r1_ = float(rr); c1 = float(cc); r2_ = float(rr); c2 = float(cc)
    while origin[int(r1_), int(c1)] != 0:
        r1_ += vr; c1 += vc
    while origin[int(r2_), int(c2)] != 0:
        r2_ -= vr; c2 -= vc
    p1r = int(r1_ + vr); p1c = int(c1 + vc); p2r = int(r2_ - vr); p2c = int(c2 - vc)
    g1 = slope[p1r, p1c]; g2 = slope[p2r, p2c]
    length = max(abs(int(r1_) - int(r2_)), abs(int(c1) - int(c2))) + 1
    if g1 == 0 and g2 == 0:
        weight = .5
    else:
        a1 = g1; a2 = g2
        if a1 > 3 * a2:
            a2 = a1
        elif a2 > 3 * a1:
            a1 = a2
        weight = a1 / (a1 + a2)
    idx = max(1, int(weight * length)) - 1
    lc, lr = line_at(int(r2_), int(c2), int(r1_), int(c1), idx)
    if shore:
        e1r = int(r1_); e1c = int(c1); e2r = int(r2_); e2c = int(c2)
    else:
        e1r = p1r; e1c = p1c; e2r = p2r; e2c = p2c
    s1 = g1; s2 = g2
    total = np.hypot(e1r - e2r, e1c - e2c); x = np.hypot(rr - lr, cc - lc); x1 = np.hypot(lr - e1r, lc - e1c); x2 = np.hypot(lr - e2r, lc - e2c)
    if x1 == 0 or x2 == 0 or total == 0:
        return dem_h, float(length), 1
    if s1 > 3 * s2:
        s1 *= .5; s2 = s1
    elif s2 > 3 * s1:
        s2 *= .5; s1 = s2
    if min(e1r, lr) <= rr <= max(e1r, lr) and min(e1c, lc) <= cc <= max(e1c, lc):
        s = s1; side = x1
    else:
        s = s2; side = x2
    ratio = current / total
    if shore:
        ratio = min(1.0, (current + 2.0) / total)
    if D1 > D0:
        w = (total * res - D0) / (D1 - D0)
        if w < 0.:
            w = 0.
        elif w > 1.:
            w = 1.
        alpha = alpha * w
    slope_now = (s / side) * x * ratio ** alpha
    return dem_h - slope_now * res * step, float(length), 0


@njit(cache=True)
def advance_r2(points, water, origin, known, dem, slope, res, par):
    offsets = np.array([[0, -1], [0, 1], [-1, 0], [1, 0], [-1, -1], [-1, 1], [1, -1], [1, 1]])
    degenerate = 0; updated = 0
    for i in range(len(points)):
        rr = points[i, 0]; cc = points[i, 1]
        if water[rr, cc] == 0:
            continue
        numerator = 0.; denominator = 0.
        for j in range(8):
            nr = rr + offsets[j, 0]; nc = cc + offsets[j, 1]
            if not known[nr, nc]:
                continue
            value, length, deg = directional_r2(slope, origin, water, rr, cc, nr, nc, dem[nr, nc], res, par)
            if not np.isfinite(value) or length <= 0:
                raise ValueError('Nonfinite ray update')
            numerator += value / length; denominator += 1 / length; degenerate += deg
        if denominator > 0:
            dem[rr, cc] = numerator / denominator; known[rr, cc] = True; water[rr, cc] = 0; updated += 1
    return updated, degenerate


def old_grid(key):
    g = reg.load_reservoir(key) if key[0] == 'D' else tr.load_test(key)
    g['meta'] = dict(epsg=reg.RESERVOIRS[key]['crs'] if key[0] == 'D' else tr.TEST[key]['epsg'])
    return g


def old_table():
    f = OUT / 'dev_old12.csv'
    if not f.exists():
        from scipy import ndimage as ndi
        L = pd.read_csv(GLIST).set_index('Hylak_id'); rows = []
        for key, hid in OLD.items():
            g = old_grid(key); F, _, _ = r1.flattened_region(g['dem'], g['W0'], g['swo1'])
            w = 4.0 * float(ndi.distance_transform_edt(F)[F].mean())
            rows.append(dict(key=key, batch='D6' if key[0] == 'D' else 'T6', kind='old', width_cells=w,
                             width_class='narrow' if w <= WIDTH_EDGES[0] else 'mid' if w <= WIDTH_EDGES[1] else 'wide',
                             F_le_W0=bool(F.sum() <= g['W0'].sum()), Hylak_id=hid, Pour_long=L.loc[hid, 'Pour_long'], Pour_lat=L.loc[hid, 'Pour_lat']))
        pd.DataFrame(rows).to_csv(f, index=False)
    return pd.read_csv(f)


def dev_table():
    s = pd.read_csv(SPLIT)
    d = s[s.set == 'development'].assign(kind='exp')[['key', 'batch', 'kind', 'width_cells', 'width_class', 'F_le_W0']]
    a = v.admitted()
    n = pd.DataFrame(dict(key=a.Hylak_id.astype(str), batch='nonUS', kind='nonus'))
    o = old_table()[['key', 'batch', 'kind', 'width_cells', 'width_class', 'F_le_W0']]
    return pd.concat([d, n, o], ignore_index=True)


def load(row, allowed='development'):
    """Inputs, F, H_flat, locked H_seed and locked R1-C raster of one reservoir of the allowed set (development here;
    r2_validation_v1 passes 'validation' inside a registered validation round)."""
    if row.kind == 'exp':
        s = pd.read_csv(SPLIT).set_index('key')
        assert s.loc[row.key, 'set'] == allowed, f'{row.key} is not a {allowed} reservoir'
        g = er.load(row.key)
        folder = (B_RUN if row.batch == 'USBR-B' else STEP1_RUN) / row.key
        rec = json.loads((folder / f'{row.key}_run.json').read_text(encoding='utf-8'))
        locked = folder / f'{row.key}_C_tao_F.tif'
    elif row.kind == 'old':
        assert row.key in OLD
        g = old_grid(row.key)
        F, H_flat, _ = r1.flattened_region(g['dem'], g['W0'], g['swo1'])
        if row.key[0] == 'D':
            H_seed = max(sl.rule_C(g['dem'], F, g['occ'], g['swo1'])[0], H_flat); locked = D_RUN / f'{row.key}_C_tao_F.tif'
        else:
            H_seed = json.loads((T_RUN / row.key / f'{row.key}_run.json').read_text(encoding='utf-8'))['H_seed_m']
            locked = T_RUN / row.key / f'{row.key}_C_tao_F.tif'
        return g, F, H_flat, float(H_seed), locked
    else:
        hid = int(row.key); g = v.load(hid)
        rec = json.loads((v.gr.RUN / row.key / f'{hid}_run.json').read_text(encoding='utf-8'))
        locked = v.gr.RUN / row.key / f'{hid}_C_tao_F.tif'
    F, H_flat, _ = r1.flattened_region(g['dem'], g['W0'], g['swo1'])
    return g, F, H_flat, float(rec['H_seed_m']), locked


def init_cache(row, g, F):
    f = CACHE / f'{row.key}.npz'
    if not f.exists():
        init = ga.initialize(g['dem'], F)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, dem=init['dem'], known=init['known'], slope=init['slope'], alpha_rule=float(init['alpha_rule']),
                            buffer_slope_deg=float(init['info']['buffer_slope_deg']))
    with np.load(f) as d:
        return d['dem'].astype(float), d['known'].astype(bool), d['slope'], float(d['alpha_rule'])


def run_candidate(g, F, H_seed, cache, par, wpar=None):
    d0, k0, s0, rule = cache
    p = np.array(par, float)
    if p[1] < 0:
        p[1] = rule
    if wpar is not None:
        from scipy import ndimage as ndi
        W = 4.0 * float(ndi.distance_transform_edt(F)[F].mean()) * 30.0
        p[1] *= float(np.clip((W - wpar[0]) / (wpar[1] - wpar[0]), 0, 1))
    inp = g['dem'].copy(); inp[F] = np.nan
    orig_init, orig_adv = run2.initialize, kernel.advance

    def cached(z, unknown):
        return np.where(k0, np.minimum(d0, H_seed), d0).copy(), k0.copy(), s0.copy(), dict(seed_cap_m=H_seed)

    def adv(points, water, origin, known, dem, slope, alpha, res):
        return advance_r2(points, water, origin, known, dem, slope, res, p)
    run2.initialize, kernel.advance = cached, adv
    try:
        pred, stat = run2.recession(inp, F, float(p[1]), H_seed, 'r2')
    finally:
        run2.initialize, kernel.advance = orig_init, orig_adv
    if not stat['complete']:
        raise RuntimeError('incomplete')
    pred[~F] = np.nan
    return pred


POUR = {}


def pour_point(row):
    if not POUR:
        import product_compare_v1 as pc
        from pyogrio.raw import read
        f = OUT / 'dev_pour_points.csv'
        if not f.exists():
            ev = pc.chk.evaluated().dropna(subset=['Hylak_id']); ev['Hylak_id'] = ev.Hylak_id.astype(int)
            meta, _, _, fl = read(pc.chk.gr.GPKG, read_geometry=False, columns=['Hylak_id', 'Pour_long', 'Pour_lat'],
                                  where=f"Hylak_id IN ({','.join(map(str, ev.Hylak_id))})")
            hl = pd.DataFrame(dict(zip(meta['fields'], fl)))
            ev.merge(hl, on='Hylak_id')[['key', 'Hylak_id', 'Pour_long', 'Pour_lat']].to_csv(f, index=False)
        t = pd.read_csv(f)
        POUR.update({str(k): (float(a), float(b)) for k, a, b in zip(t.key, t.Pour_long, t.Pour_lat)})
        L = v.lst()
        POUR.update({str(k): (float(L.loc[k, 'Pour_long']), float(L.loc[k, 'Pour_lat'])) for k in v.admitted().Hylak_id})
        o = old_table()
        POUR.update({k: (float(a), float(b)) for k, a, b in zip(o.key, o.Pour_long, o.Pour_lat)})
    return POUR[str(row.key)]


def dam_depth(row, g, F, H_flat, radius, q, below=False):
    from pyproj import Transformer
    lon, lat = pour_point(row)
    t = g['transform']; shape = g['dem'].shape
    x, y = Transformer.from_crs(4326, int(g['meta']['epsg']), always_xy=True).transform(lon, lat)
    c0, r0 = (x - t[2]) / t[0], (y - t[5]) / t[4]; k = int(np.ceil(radius / 30.0)) + 2
    rs, re = max(int(r0) - k, 0), min(int(r0) + k + 1, shape[0]); cs, ce = max(int(c0) - k, 0), min(int(c0) + k + 1, shape[1])
    if re <= rs or ce <= cs:
        return np.nan
    rr, cc = np.mgrid[rs:re, cs:ce]
    m = (np.hypot(t[2] + (cc + .5) * t[0] - x, t[5] + (rr + .5) * t[4] - y) <= radius) & ~F[rs:re, cs:ce] & ~g['W0'][rs:re, cs:ce] & np.isfinite(g['dem'][rs:re, cs:ce])
    z = g['dem'][rs:re, cs:ce][m]
    if below:
        z = z[z < H_flat]
    if len(z) < 20:
        return np.nan
    return float(H_flat - np.percentile(z, q))


def rescale(pred, F, H_seed, target):
    d = H_seed - pred
    m = F & np.isfinite(pred) & (d > 0)
    if not (np.isfinite(target) and target > 0 and m.any()):
        return pred, False
    out = pred.copy(); out[m] = H_seed - d[m] * target / d[m].max()
    return out, True


def predict(row, g, F, H_flat, H_seed, cache, locked, spec):
    """One candidate on one reservoir. spec: par, wpar, dam=(R, Q) or None, below, guard, shape ('tao' / 'edt')."""
    dam = spec.get('dam'); extra = {}
    if dam is not None:
        D = dam_depth(row, g, F, H_flat, dam[0], dam[1], spec.get('below', False))
        if spec.get('extrap') and np.isfinite(D):
            D2 = dam_depth(row, g, F, H_flat, 2 * dam[0], dam[1], True)
            extra.update(B_R_m=D, B_2R_m=D2); D -= max(0.0, D2 - D) if np.isfinite(D2) else 0.0
        extra['D_dam_m'] = D
        if spec.get('guard') and not (np.isfinite(D) and D > 0):
            return v.reg.load_m0(locked, g), dict(extra, rescaled=False)
    if spec.get('shape', 'tao') == 'edt':
        from scipy import ndimage as ndi
        pred = np.where(F, H_seed - ndi.distance_transform_edt(F), np.nan)
    else:
        pred = run_candidate(g, F, H_seed, cache, spec['par'], spec.get('wpar'))
    if dam is not None:
        pred, extra['rescaled'] = rescale(pred, F, H_seed, D + (H_seed - H_flat))
    return pred, extra


def metrics(row, g, F, H_flat, pred, ctx):
    z = g['dem'].copy(); z[F] = pred[F]
    if row.kind == 'nonus':
        m = v.metrics(pred, ctx['zref'], ctx['X'], H_flat)
        return dict(below_PE_pct=m['V_PE_pct'], bed_RMSE_m=m['bed_RMSE_m'], bed_bias_m=m['bed_bias_m'])
    cur, H_pool = ctx['cur'], ctx['H_pool']
    s = sc.crop([g['swo1'], g['W0'], F])
    L0 = np.floor(H_pool / aev.STEP) * aev.STEP
    tab = aev.curve_table(z[s], (g['W0'] | F)[s], {'swo1': g['swo1'][s]}, [L0, L0 + aev.STEP])
    Vmax = float(np.interp(H_pool, cur.H, cur.volume_km3)); A_pool = float(np.interp(H_pool, cur.H, cur.area_km2))
    Vflat = float(np.interp(H_flat, cur.H, cur.volume_km3))
    zs = np.sort(z[F])
    n5 = int(np.ceil(0.05 * A_pool / 9e-4))
    D5_model = H_pool - float(zs[n5 - 1]) if 0 < n5 <= len(zs) else np.nan
    Am = np.maximum.accumulate(cur.area_km2.values)
    D5_off = H_pool - float(np.interp(0.05 * A_pool, Am, cur.H.values))
    return dict(Vmax_PE_pct=(float(np.interp(H_pool, tab.H, tab.V_swo1)) / Vmax - 1) * 100,
                below_PE_pct=(float(np.sum(np.clip(H_flat - z[F], 0, None))) * 9e-7 / Vflat - 1) * 100 if Vflat > 0 else np.nan,
                share_below_pct=Vflat / Vmax * 100, D5_ratio=D5_model / D5_off if D5_off > 0 else np.nan,
                Dmax_ratio=(H_pool - float(zs[0])) / (H_pool - float(cur.H.iloc[0])))


def context(row, g, F, H_flat):
    if row.kind == 'nonus':
        hid = int(row.key); L = v.lst(); adm = v.admitted().set_index('Hylak_id')
        _, _, X, _ = v.regions(g, float(L.loc[hid, 'Pour_long']), float(L.loc[hid, 'Pour_lat']))
        return dict(zref=g['nasadem'] + float(adm.loc[hid, 'offset_m']), X=X)
    if row.kind == 'old' and row.key[0] == 'D':
        off = reg.egm_offset_m(row.key, g); cur = reg.official_curve(row.key, off)
        return dict(cur=cur, H_pool=float(reg.to_egm(row.key, reg.RESERVOIRS[row.key]['pool_ft'], off)))
    if row.kind == 'old':
        off = tr.egm_offset_m(row.key); cur = tr.official_curve(row.key, off)
        return dict(cur=cur, H_pool=float(tr.to_egm(oc.POOL[row.key][0], off)))
    cur, H_pool, _, _ = sc.official(row.batch, row.key)
    return dict(cur=cur, H_pool=H_pool)


def loop(name, spec, part, baseline=False, cache_only=False, kind=None):
    t = dev_table(); tag = kind or 'all'
    if kind:
        t = t[t.kind == kind]
    if part:
        i, n = map(int, part.split('/')); t = t.iloc[i - 1::n]; tag += f'_p{i}of{n}'
    rows = []
    for row in t.itertuples():
        try:
            g, F, H_flat, H_seed, locked = load(row)
            cache = init_cache(row, g, F)
            if cache_only:
                print(row.key, 'cached', flush=True); continue
            pred, extra = (v.reg.load_m0(locked, g), {}) if baseline else predict(row, g, F, H_flat, H_seed, cache, locked, spec)
            rows.append(dict(key=row.key, batch=row.batch, kind=row.kind, candidate=name, alpha_rule=cache[3], **extra,
                             **metrics(row, g, F, H_flat, pred, context(row, g, F, H_flat))))
            print({k: (round(x, 2) if isinstance(x, float) else x) for k, x in rows[-1].items()}, flush=True)
        except Exception as exc:
            rows.append(dict(key=row.key, batch=row.batch, kind=row.kind, candidate=name, error=repr(exc)[:200]))
            print(row.key, 'FAILED', repr(exc)[:200], flush=True)
        if rows:
            pd.DataFrame(rows).to_csv(OUT / f'{name}_{tag}.csv', index=False, float_format='%.6g')


def summary(names, kinds=('exp',)):
    t = dev_table().set_index('key')
    d = pd.concat([pd.read_csv(f) for n in names for f in sorted(OUT.glob(f'{n}_*.csv'))])
    d['key'] = d.key.astype(str)
    d = d.join(t[['width_class', 'F_le_W0', 'width_cells']], on='key')
    d = d[~d.duplicated(['key', 'candidate'], keep='last')]
    e = d[d.kind.isin(kinds)].assign(absPE=lambda x: x.Vmax_PE_pct.abs())
    order = lambda g: g.reindex(names)
    print('development reservoirs with official curves:', e.key.nunique(), '| kinds', list(kinds))
    print(order(e.groupby('candidate').agg(n=('key', 'count'), med_abs_Vmax=('absPE', 'median'), mean_abs=('absPE', 'mean'), med_Vmax=('Vmax_PE_pct', 'median'),
                                           over50=('Vmax_PE_pct', lambda s: (s > 50).mean()), med_below=('below_PE_pct', 'median'),
                                           D5=('D5_ratio', 'median'), Dmax=('Dmax_ratio', 'median'))).round(2).to_string())
    pv = e.pivot(index='key', columns='candidate', values='absPE')
    if 'K0' in pv:
        from scipy import stats
        rows = []
        for c in names:
            if c == 'K0':
                continue
            dd = (pv[c] - pv['K0']).dropna(); nz = dd[dd != 0]
            rows.append(dict(candidate=c, wins=int((dd < 0).sum()), losses=int((dd > 0).sum()), median_diff=float(dd.median()), mean_diff=float(dd.mean()),
                             p_one_sided=float(stats.wilcoxon(nz, alternative='less').pvalue) if len(nz) else np.nan))
        print('paired against K0 on |Vmax PE| (development, descriptive)'); print(pd.DataFrame(rows).set_index('candidate').round(3).to_string())
    for col, lab in [('absPE', 'median |Vmax PE| by width class'), ('Vmax_PE_pct', 'median signed Vmax PE by width class'), ('below_PE_pct', 'median below-waterline PE by width class')]:
        print(lab); print(e.pivot_table(index='candidate', columns='width_class', values=col, aggfunc='median').reindex(names)[['narrow', 'mid', 'wide']].round(1).to_string())
    print('median |Vmax PE| by regime'); print(e.pivot_table(index='candidate', columns='F_le_W0', values='absPE', aggfunc='median').reindex(names).round(1).to_string())
    n = d[d.kind == 'nonus']
    if len(n):
        print('17 non-US (below FABDEM water level, in X)')
        print(order(n.groupby('candidate').agg(n=('key', 'count'), med_below=('below_PE_pct', 'median'), med_abs=('below_PE_pct', lambda s: s.abs().median()),
                                               RMSE=('bed_RMSE_m', 'median'), bias=('bed_bias_m', 'median'), over25=('below_PE_pct', lambda s: (s > 25).mean()))).round(2).to_string())


def main():
    warnings.filterwarnings('ignore')
    p = argparse.ArgumentParser()
    p.add_argument('--cache', action='store_true'); p.add_argument('--baseline', action='store_true')
    p.add_argument('--run'); p.add_argument('--par'); p.add_argument('--part'); p.add_argument('--summary', nargs='*')
    p.add_argument('--kind', choices=['exp', 'nonus', 'old']); p.add_argument('--wpar'); p.add_argument('--dam')
    p.add_argument('--guard', action='store_true'); p.add_argument('--kinds', default='exp')
    p.add_argument('--below', action='store_true'); p.add_argument('--shape', choices=['tao', 'edt'], default='tao')
    p.add_argument('--extrap', action='store_true')
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cache:
        loop('cache', None, a.part, cache_only=True, kind=a.kind)
    elif a.baseline:
        loop('K0', None, a.part, baseline=True, kind=a.kind)
    elif a.run:
        loop(a.run, dict(par=[float(x) for x in a.par.split(',')], wpar=[float(x) for x in a.wpar.split(',')] if a.wpar else None,
                         dam=[float(x) for x in a.dam.split(',')] if a.dam else None, guard=a.guard, below=a.below, shape=a.shape, extrap=a.extrap), a.part, kind=a.kind)
    elif a.summary:
        summary(a.summary, tuple(a.kinds.split(',')))


if __name__ == '__main__':
    main()
