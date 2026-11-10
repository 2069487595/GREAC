"""Route 1, seed water level for Tao on the DEM plateau: candidate rules A / B / C on the 6 dev reservoirs.

Rules and selection are fixed in docs/plans/2026-09-29-route1-seed-level-candidates.md (option 2, pinned
details at the end of that file) before anything here ran. Common to all rules:
  F, H_flat : as route1_tao_flat_v1 (FABDEM plateau and its elevation)
  S         : shoreline ring, cells outside F, 8-adjacent to F, DEM finite
  H_seed    : max(rule value, H_flat); known seeds used by Tao = min(DEM, H_seed); Tao start level h0 = H_seed
  A : GEV (scipy MLE) location of S heights at or below their 95th percentile
  B : median over S of z_i - s_i * d_i, d_i = 15 m (edge-adjacent to F) or 21.2 m (corner only),
      s_i = max(0, (median DEM of outward neighbours - z_i) / 30 m), outward = 8-neighbours not in F or S
  C : o_edge = median SWO occurrence of F's outermost cells; terrain outside F in the SWO>=1% region with
      occurrence in [o_edge-20, o_edge) binned by 1 point (bins with >= 10 cells); unweighted line through
      (median occurrence, median elevation) of the bins; H = line at o_edge; fewer than 5 bins -> rule A
Order: seed_levels.csv is written for all rules and reservoirs before Tao runs or any scoring.
Selection: smallest median |Vmax error| (SWO extent) over the 6; candidates within 2 points of the best
are tied and the one with the smallest max |error| wins. v1 / v2 appear as reference only.
Usage: .venv/Scripts/python.exe 04_Code/scripts/route1_seed_level_v1.py [--keys D01 ...] [--rules A B C]
"""
from pathlib import Path
import sys, json, argparse, datetime, hashlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi, stats
import aev_registry_v1 as reg
import aev_observation_share_v1 as aev
import route1_tao_flat_v1 as r1

OUT = reg.ROOT / '05_Results/by_date/2026-09-29/route1_seed_level_v1'
PLAN = reg.ROOT / 'docs/plans/2026-09-29-route1-seed-level-candidates.md'
RULES = ['A', 'B', 'C']
EIGHT = np.ones((3, 3), bool)
CROSS = ndi.generate_binary_structure(2, 1)


def shore_ring(dem, F):
    return ndi.binary_dilation(F, EIGHT) & ~F & np.isfinite(dem)


def rule_A(dem, F):
    x = dem[shore_ring(dem, F)]
    x = x[x <= np.quantile(x, 0.95)]
    c, loc, scale = stats.genextreme.fit(x)
    if not np.isfinite(loc):
        raise RuntimeError('GEV fit not finite')
    return float(loc), dict(n_ring=int(len(x)), gev_shape_c=float(c), gev_scale=float(scale))


def rule_B(dem, F):
    S = shore_ring(dem, F)
    outward = ndi.binary_dilation(S, EIGHT) & ~S & ~F & np.isfinite(dem)
    edge = ndi.binary_dilation(F, CROSS) & S
    H, slopes = [], []
    for r, c in zip(*np.nonzero(S)):
        win = (slice(r - 1, r + 2), slice(c - 1, c + 2))
        nb = dem[win][outward[win]]
        if not len(nb):
            continue
        s = max(0., (np.median(nb) - dem[r, c]) / 30.)
        H.append(dem[r, c] - s * (15. if edge[r, c] else 15. * np.sqrt(2)))
        slopes.append(s)
    return float(np.median(H)), dict(n_ring=int(S.sum()), n_used=len(H),
                                     median_slope_deg=float(np.degrees(np.arctan(np.median(slopes)))))


def rule_C(dem, F, occ, swo1):
    outer = F & ndi.binary_dilation(~F, EIGHT)
    o_edge = float(np.median(occ[outer]))
    terrain = ~F & swo1 & np.isfinite(dem)
    bins = []
    for k in range(20):
        lo = o_edge - 20 + k
        sel = terrain & (occ >= lo) & (occ < lo + 1)
        if sel.sum() >= 10:
            bins.append((float(np.median(occ[sel])), float(np.median(dem[sel])), int(sel.sum())))
    info = dict(o_edge=o_edge, n_bins=len(bins), bins=bins)
    if len(bins) < 5:
        H, infoA = rule_A(dem, F)
        return H, dict(info, fallback='A', **infoA)
    o, z, _ = np.array(bins).T
    b, a = np.polyfit(o, z, 1)
    return float(a + b * o_edge), dict(info, fallback=None, intercept=float(a), slope_m_per_pct=float(b))


def seed_levels(key):
    g = reg.load_reservoir(key)
    F, H_flat, _ = r1.flattened_region(g['dem'], g['W0'], g['swo1'])
    rows = []
    for rule in RULES:
        if rule == 'A':
            H, info = rule_A(g['dem'], F)
        elif rule == 'B':
            H, info = rule_B(g['dem'], F)
        else:
            H, info = rule_C(g['dem'], F, g['occ'], g['swo1'])
        rows.append(dict(id=key, rule=rule, H_flat_m=H_flat, H_rule_m=H, H_seed_m=max(H, H_flat),
                         floor_applied=bool(H < H_flat), info=json.dumps(info, default=float)))
    return rows


def run_tao(key, g, F, H_seed, folder, rule):
    sys.path.insert(0, str(reg.ROOT / '04_Code/versions/m0_m1_m2_runner_v4'))
    import geometry_adapter as ga
    dem = g['dem']
    init = ga.initialize(dem, F)
    inp = dem.copy(); inp[F] = np.nan
    original = ga.tao_run.initialize

    def capped(z, unknown):
        d, known, slope, info = original(z, unknown)
        return np.where(known, np.minimum(d, H_seed), d), known, slope, dict(info, seed_cap_m=H_seed)
    ga.tao_run.initialize = capped
    try:
        pred, stat = ga.tao_run.recession(inp, F, init['alpha_rule'], H_seed, f'{key}/{rule}')
    finally:
        ga.tao_run.initialize = original
    if not stat['complete']:
        raise RuntimeError(f'{key}/{rule}: incomplete, {stat["remaining_unknown"]} cells left')
    pred[~F] = np.nan
    t = g['transform']
    with rasterio.open(folder / f'{key}_{rule}_tao_F.tif', 'w', driver='GTiff', height=pred.shape[0],
                       width=pred.shape[1], count=1, dtype='float32', nodata=np.nan,
                       crs=f'EPSG:{reg.RESERVOIRS[key]["crs"]}', compress='deflate',
                       transform=rasterio.Affine(t[0], t[1], t[2], t[3], t[4], t[5])) as d:
        d.write(pred.astype('float32'), 1)
    return pred, dict(alpha=init['alpha_rule'], iterations=stat['iterations'])


def analyse(key, rule, H_seed, folder):
    rec = reg.RESERVOIRS[key]
    g = reg.load_reservoir(key)
    dem, W0 = g['dem'], g['W0']
    F, H_flat, _ = r1.flattened_region(dem, W0, g['swo1'])
    tao, meta = run_tao(key, g, F, H_seed, folder, rule)
    z = dem.copy(); z[F] = tao[F]
    # scoring only below this line: official curve and survey cells
    off = reg.egm_offset_m(key, g); cur = reg.official_curve(key, off)
    H_pool = float(reg.to_egm(key, rec['pool_ft'], off))
    masks = {'swo1': g['swo1'], 'dem_fill': None}
    if 'official' in g:
        masks['bound'] = g['official']
    lo = np.floor(min(np.nanmin(z[F]), cur.H.min()) / aev.STEP) * aev.STEP
    tab = aev.curve_table(z, W0 | F, masks, np.arange(lo, H_pool + 2 + aev.STEP, aev.STEP))
    tab.to_csv(folder / f'{key}_{rule}_curves.csv', index=False)
    off_at = lambda H, c: float(np.interp(H, cur.H, cur[c]))
    at = lambda H, c: float(np.interp(H, tab.H, tab[c]))
    Vmax = off_at(H_pool, 'volume_km3')
    F_km2 = F.sum() * reg.CELL_AREA / 1e6
    H_area = float(np.interp(F_km2, np.maximum.accumulate(cur.area_km2.values), cur.H.values))
    v_tao = float(np.sum(np.clip(H_flat - z[F], 0, None))) * reg.CELL_AREA / 1e9
    row = dict(id=key, name=rec['name'], rule=rule, H_flat_m=H_flat, H_seed_m=H_seed,
               H_area_matched_TWDB_m=H_area, seed_minus_area_matched_m=H_seed - H_area,
               alpha=meta['alpha'], iterations=meta['iterations'], Vmax_TWDB_km3=Vmax,
               route1_tao_part_PE_pct=(v_tao / off_at(H_flat, 'volume_km3') - 1) * 100)
    for name in masks:
        row[f'Vmax_{name}_PE_pct'] = (at(H_pool, f'V_{name}') / Vmax - 1) * 100
        sel = (cur.H <= H_pool) & (cur.H >= tab.H.min())
        mA = np.interp(cur.H[sel], tab.H, tab[f'A_{name}'])
        row[f'AE_NRMSE_{name}_pct'] = aev.fit_stats(mA, cur.area_km2[sel].values).get('NRMSE_pct')
    rr, cc, zr = r1.survey_cells(key, g)
    inW = W0[rr, cc]
    for pname, msk in dict(W0=inW, F=inW & F[rr, cc]).items():
        s = r1.err_stats(z[rr, cc][msk], zr[msk])
        row[f'cells_{pname}_RMSE'] = s.get('RMSE'); row[f'cells_{pname}_bias'] = s.get('bias')
        row[f'cells_{pname}_n'] = s.get('n', 0)
    return row


def select(summary):
    """Pre-registered choice: smallest median |Vmax error| (swo1); ties within 2 points -> smallest max."""
    e = summary.assign(abs_pe=summary.Vmax_swo1_PE_pct.abs()).groupby('rule').abs_pe
    score = pd.DataFrame(dict(median_abs_pct=e.median(), max_abs_pct=e.max(), n=e.count()))
    tied = score[score.median_abs_pct <= score.median_abs_pct.min() + 2]
    return tied.max_abs_pct.idxmin(), score, list(tied.index)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--keys', nargs='*', default=list(reg.RESERVOIRS))
    p.add_argument('--rules', nargs='*', default=RULES)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    proto = OUT / 'protocol.json'
    if not proto.exists():  # written once, before any seed level or prediction
        proto.write_text(json.dumps(dict(
            plan=str(PLAN.relative_to(reg.ROOT)), plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rules=RULES,
            selection='min median |Vmax_swo1 PE| over 6 dev reservoirs; within 2 points tied -> min max |PE|',
            reference_only=['route1_tao_flat_v1', 'route1_tao_flat_v2'],
            written_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=2), encoding='utf-8')
    seeds_csv = OUT / 'seed_levels.csv'
    if not seeds_csv.exists():  # locked before Tao runs
        rows = []
        for key in reg.RESERVOIRS:
            rows += seed_levels(key)
            print('SEEDS', key, [(r['rule'], round(r['H_rule_m'], 2), round(r['H_seed_m'], 2)) for r in rows[-3:]], flush=True)
        pd.DataFrame(rows).to_csv(seeds_csv, index=False, encoding='utf-8-sig')
    seeds = pd.read_csv(seeds_csv).set_index(['id', 'rule'])
    out_csv = OUT / 'summary_seed_rules.csv'
    rows = pd.read_csv(out_csv).to_dict('records') if out_csv.exists() else []
    done = {(r['id'], r['rule']) for r in rows if pd.isna(r.get('error'))}
    for rule in a.rules:
        for key in a.keys:
            if (key, rule) in done:
                continue
            print('RUN', rule, key, reg.RESERVOIRS[key]['name'], flush=True)
            rows = [r for r in rows if (r['id'], r['rule']) != (key, rule)]
            try:
                rows.append(analyse(key, rule, float(seeds.loc[(key, rule), 'H_seed_m']), OUT))
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
            except Exception as exc:
                import traceback; traceback.print_exc()
                rows.append(dict(id=key, name=reg.RESERVOIRS[key]['name'], rule=rule, error=repr(exc)))
            pd.DataFrame(rows).to_csv(out_csv, index=False, encoding='utf-8-sig')
    s = pd.DataFrame(rows)
    s = s[s['error'].isna()] if 'error' in s else s
    if len(s) == len(RULES) * len(reg.RESERVOIRS):
        chosen, score, tied = select(s)
        score.to_csv(OUT / 'selection_scores.csv', encoding='utf-8-sig')
        (OUT / 'selection.json').write_text(json.dumps(dict(chosen=chosen, tied=tied, scores=score.to_dict('index')),
                                                       indent=2, default=float), encoding='utf-8')
        print('SELECTED', chosen, 'tied', tied, flush=True); print(score, flush=True)


if __name__ == '__main__':
    main()
