"""Route-1 A-E-V diagnostic: observed share of full-pool volume and full-curve accuracy.

For each TWDB development reservoir two levels are separated, because FABDEM (2011-2015) and the
SWBD mask W0 (Feb 2000) come from different epochs:
  * H_W0   - DEM on the ring just outside W0; Tao-original (M0) reconstructs everything inside W0 from here.
  * H_flat - FABDEM's own flattened water plateau inside W0; below it the DEM holds no terrain at all.
Curves:
  * Tao W0 + DEM  = frozen v4 M0 inside W0 (Medina and Meredith run here with the same frozen code),
                    FABDEM elsewhere.
  * DEM only      = FABDEM everywhere, plateau as the floor: the observed band on its own, no Tao.
  Both are integrated level by level. The extent at level H is every
                   cell with z <= H that is 8-connected to W0, optionally restricted to a mask:
                     swo1     - SWO occurrence >= 1% region connected to W0 (globally available)
                     bound    - official boundary where one exists (Spence 1910-ft polygon;
                                Codex R closed by official boundaries for Buchanan/Travis/Sam Rayburn)
                     dem_fill - no mask at all (diagnostic for leaks past the dam / into other valleys)
  * Official curve (TWDB) converted to EGM2008 via aev_registry_v1.

Also a Tibet section: observed band between the 2000 NASADEM level and the SWO maximum level,
next to the Tao v3 volume below the 2000 level (no official curves; Tao 2026 Table 2 for context).

Nothing here selects parameters from bathymetry or official curves; they are used only for scoring.
Usage:  .venv/Scripts/python.exe 04_Code/scripts/aev_observation_share_v1.py [--keys D01 D03 ...] [--skip-tibet]
"""
from pathlib import Path
import sys, json, argparse, datetime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi
from scipy.stats import pearsonr
import aev_registry_v1 as reg

OUT = reg.ROOT / '05_Results/by_date/2026-09-29/aev_observation_share_v1'
STEP = 0.25  # m between evaluated levels


def compute_m0(key, g, folder):
    """Run Tao-original on W0 exactly as the frozen v4 runner does for M0."""
    sys.path.insert(0, str(reg.ROOT / '04_Code/versions/m0_m1_m2_runner_v4'))
    import geometry_adapter as ga
    z, W0 = g['dem'], g['W0']
    init = ga.initialize(z, W0)
    h0 = float(np.nanmedian(z[W0]))
    inp = z.copy(); inp[W0] = np.nan
    pred, stat = ga.tao_run.recession(inp, W0, init['alpha_rule'], h0, key + '/M0')
    pred[~W0] = np.nan
    if not stat['complete']:
        raise RuntimeError(f'{key}: Tao M0 incomplete, {stat["remaining_unknown"]} cells left')
    t = g['transform']
    path = folder / f'{key}_M0.tif'
    with rasterio.open(path, 'w', driver='GTiff', height=pred.shape[0], width=pred.shape[1], count=1,
                       dtype='float32', crs=f'EPSG:{reg.RESERVOIRS[key]["crs"]}', nodata=np.nan,
                       transform=rasterio.Affine(t[0], t[1], t[2], t[3], t[4], t[5]), compress='deflate') as d:
        d.write(pred.astype('float32'), 1)
    meta = dict(alpha=init['alpha_rule'], h0=h0, buffer_slope_deg=init['info']['buffer_slope_deg'],
                iterations=stat['iterations'], code='04_Code/versions/m0_m1_m2_runner_v4/geometry_adapter.py (frozen Tao v3 initialize + recession)')
    (folder / f'{key}_M0.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return path


def curve_table(z, W0, masks, levels):
    """Area (km2) and volume (km3) at each level for every extent variant."""
    rows = []
    for H in levels:
        wet = z <= H
        row = dict(H=H)
        for name, mask in masks.items():
            m = wet if mask is None else wet & mask
            keep = reg.connected_to(m, W0)
            row[f'A_{name}'] = keep.sum() * reg.CELL_AREA / 1e6
            row[f'V_{name}'] = float(np.sum(H - z[keep])) * reg.CELL_AREA / 1e9
            if mask is None:
                row['edge_dem_fill'] = bool(keep[0].any() or keep[-1].any() or keep[:, 0].any() or keep[:, -1].any())
        rows.append(row)
    return pd.DataFrame(rows)


def fit_stats(model, ref):
    ok = np.isfinite(model) & np.isfinite(ref)
    m, r = model[ok], ref[ok]
    if len(m) < 3:
        return dict(n=int(len(m)))
    e = m - r
    rng = r.max() - r.min()
    return dict(n=int(len(m)), RMSE=float(np.sqrt(np.mean(e * e))), MAE=float(np.mean(abs(e))),
                bias=float(e.mean()), NRMSE_pct=float(np.sqrt(np.mean(e * e)) / rng * 100) if rng > 0 else None,
                r=float(pearsonr(m, r)[0]))


def outer_ring_level(dem, region, W0):
    ring = region & ~ndi.binary_erosion(region, np.ones((3, 3))) & ~W0
    v = dem[ring & np.isfinite(dem)]
    return float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))


def water_plateau(dem, W0):
    """Most frequent DEM value inside W0 (the flattened water surface) and its share of W0."""
    vals, counts = np.unique(np.round(dem[W0 & np.isfinite(dem)], 2), return_counts=True)
    i = counts.argmax()
    return float(vals[i]), float(counts[i] / counts.sum())


def analyse_reservoir(key, folder):
    rec = reg.RESERVOIRS[key]
    g = reg.load_reservoir(key)
    off = reg.egm_offset_m(key, g)
    cur = reg.official_curve(key, off)
    if rec['m0'] is not None:
        m0_path = rec['m0']
    elif (folder / f'{key}_M0.tif').exists():
        m0_path = folder / f'{key}_M0.tif'
    else:
        m0_path = compute_m0(key, g, folder)
    m0 = reg.load_m0(m0_path, g)
    W0 = g['W0']
    missing = int((W0 & ~np.isfinite(m0)).sum())
    dem = g['dem']
    z = dem.copy()
    use = W0 & np.isfinite(m0)
    z[use] = m0[use]
    # Two different "water levels" when the DEM and the SWBD mask come from different epochs:
    #   H_W0   - DEM elevation on the ring just outside W0: where Tao-original (M0) starts, ~SWBD 2000 level
    #   H_flat - the DEM's own flattened water surface inside W0: below it the DEM holds no terrain
    H_W0 = float(np.median(dem[ndi.binary_dilation(W0, np.ones((3, 3))) & ~W0]))
    H_flat, flat_share = water_plateau(dem, W0)
    H_pool = float(reg.to_egm(key, rec['pool_ft'], off))
    H_swo, H_swo_p25, H_swo_p75 = outer_ring_level(dem, g['swo1'], W0)
    masks = {'swo1': g['swo1'], 'dem_fill': None}
    if 'official' in g:
        masks['bound'] = g['official']
    lo = np.floor(min(np.nanmin(z[W0]), cur.H.min()) / STEP) * STEP
    tab = curve_table(z, W0, masks, np.arange(lo, H_pool + 2 + STEP, STEP))
    tab.to_csv(folder / f'{key}_model_curves_TaoW0_plus_DEM.csv', index=False)
    # DEM only, no Tao anywhere: the flattened plateau is the floor. Tests the observed band on its own.
    obs_masks = {k: v for k, v in masks.items() if k != 'dem_fill'}
    tab_obs = curve_table(dem, W0, obs_masks, np.arange(np.floor(H_flat / STEP) * STEP, H_pool + 2 + STEP, STEP))
    tab_obs.to_csv(folder / f'{key}_model_curves_DEM_only.csv', index=False)

    def off_at(H, col):
        return float(np.interp(H, cur.H, cur[col]))

    def at(t, H, col):
        return float(np.interp(H, t.H, t[col]))

    Vmax = off_at(H_pool, 'volume_km3')
    V_m0_own = float(np.sum(np.clip(H_W0 - z[use], 0, None))) * reg.CELL_AREA / 1e9
    row = dict(id=key, name=rec['name'], datum=rec['datum'], egm_offset_m=off['metres'] + off['feet'] * reg.FT,
               H_flat_m=H_flat, flat_share_of_W0=flat_share, H_W0_m=H_W0, H_swo_max_m=H_swo,
               H_swo_max_iqr=f'{H_swo_p25:.2f}-{H_swo_p75:.2f}', H_pool_m=H_pool, pool_ft=rec['pool_ft'],
               swo_max_minus_pool_m=H_swo - H_pool,
               W0_km2=W0.sum() * reg.CELL_AREA / 1e6, A_TWDB_at_H_W0=off_at(H_W0, 'area_km2'),
               plateau_km2=flat_share * W0.sum() * reg.CELL_AREA / 1e6, A_TWDB_at_H_flat=off_at(H_flat, 'area_km2'),
               swo1_km2=g['swo1'].sum() * reg.CELL_AREA / 1e6,
               swo1_touches_edge=bool(g['swo1'][0].any() or g['swo1'][-1].any() or g['swo1'][:, 0].any() or g['swo1'][:, -1].any()),
               W0_cells_without_M0=missing, alpha_M0=None,
               Vmax_TWDB_km3=Vmax, A_pool_TWDB_km2=off_at(H_pool, 'area_km2'),
               share_below_H_flat_pct=off_at(H_flat, 'volume_km3') / Vmax * 100,
               share_below_H_W0_pct=off_at(H_W0, 'volume_km3') / Vmax * 100,
               V_M0_below_H_W0_km3=V_m0_own, V_TWDB_below_H_W0_km3=off_at(H_W0, 'volume_km3'),
               datum_1m_Vmax_pct=off_at(H_pool, 'area_km2') * 1e-3 / Vmax * 100)
    row['M0_own_domain_PE_pct'] = (V_m0_own / row['V_TWDB_below_H_W0_km3'] - 1) * 100
    band_twdb = Vmax - off_at(H_flat, 'volume_km3')
    row['observed_band_TWDB_km3'] = band_twdb
    sel = (cur.H <= H_pool) & (cur.H >= tab.H.min())
    ref_H = cur.H[sel].values
    stats = {}
    for name in masks:
        Vp = at(tab, H_pool, f'V_{name}')
        row[f'Vmax_TaoW0_{name}_km3'] = Vp
        row[f'Vmax_TaoW0_{name}_PE_pct'] = (Vp / Vmax - 1) * 100
        row[f'A_pool_{name}_km2'] = at(tab, H_pool, f'A_{name}')
        mA = np.interp(ref_H, tab.H, tab[f'A_{name}']); mV = np.interp(ref_H, tab.H, tab[f'V_{name}'])
        stats[f'TaoW0_{name}'] = dict(AE=fit_stats(mA, cur.area_km2[sel].values), VE=fit_stats(mV, cur.volume_km3[sel].values))
        row[f'AE_NRMSE_TaoW0_{name}_pct'] = stats[f'TaoW0_{name}']['AE'].get('NRMSE_pct')
        if name in obs_masks:
            vb = at(tab_obs, H_pool, f'V_{name}')
            row[f'observed_band_DEM_{name}_km3'] = vb
            row[f'observed_band_DEM_{name}_PE_pct'] = (vb / band_twdb - 1) * 100 if band_twdb > 0 else None
            above = (ref_H >= H_flat)
            oA = np.interp(ref_H[above], tab_obs.H, tab_obs[f'A_{name}'])
            stats[f'DEM_{name}_above_H_flat'] = dict(AE=fit_stats(oA, cur.area_km2[sel].values[above]))
            row[f'AE_NRMSE_DEM_{name}_above_flat_pct'] = stats[f'DEM_{name}_above_H_flat']['AE'].get('NRMSE_pct')
    lvl_pool = tab.iloc[(tab.H - H_pool).abs().argmin()]
    row['dem_fill_leak_at_pool'] = bool(lvl_pool['edge_dem_fill'] or lvl_pool['A_dem_fill'] > 1.5 * row['A_pool_TWDB_km2'])
    if rec['m0'] is None:
        row['alpha_M0'] = json.loads((folder / f'{key}_M0.json').read_text())['alpha']
    else:
        row['alpha_M0'] = json.loads((reg.V4 / f'formal/{key}/result.json').read_text())['alpha_M0']
    (folder / f'{key}_fit_stats.json').write_text(json.dumps(dict(offset=off, stats=stats), indent=2), encoding='utf-8')
    cur.to_csv(folder / f'{key}_official_curve_egm2008.csv', index=False)
    return row, tab, cur


def analyse_tibet(lake, folder):
    info = reg.TIBET[lake]
    with np.load(reg.TIBET_INPUTS / lake / 'frozen_terrain.npz') as d:
        dem, W0, occ, t = d['dem'].astype(float), d['W0'].astype(bool), d['occurrence'].astype(int), d['transform']
    with rasterio.open(reg.TIBET_TAO / lake / 'Tao_original_W0.tif') as s:
        tao = s.read(1).astype(float)
    assert tao.shape == dem.shape
    swo1 = reg.connected_to(occ >= 1, W0)
    z = dem.copy(); z[W0] = tao[W0]
    H0 = float(np.median(dem[W0]))
    Hs, Hs25, Hs75 = outer_ring_level(dem, swo1, W0)
    v_tao = float(np.sum(np.clip(H0 - z[W0], 0, None))) * reg.CELL_AREA / 1e9
    tab = curve_table(z, W0, {'swo1': swo1}, np.array([H0, Hs]))
    V_top = float(tab.V_swo1.iloc[1])
    return dict(id=lake, name=info['name'], H2000_m=H0, H_swo_max_m=Hs, H_swo_iqr=f'{Hs25:.2f}-{Hs75:.2f}',
                rise_m=Hs - H0, W0_km2=W0.sum() * reg.CELL_AREA / 1e6, A_at_swo_max_km2=float(tab.A_swo1.iloc[1]),
                V_tao_below_H2000_km3=v_tao, V_total_at_swo_max_km3=V_top,
                observed_band_km3=V_top - float(tab.V_swo1.iloc[0]),
                observed_share_pct=(V_top - float(tab.V_swo1.iloc[0])) / V_top * 100,
                paper_reference_km3=info['paper_volume_km3'], paper_tao_km3=info['paper_tao_volume_km3'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--keys', nargs='*', default=list(reg.RESERVOIRS))
    p.add_argument('--skip-tibet', action='store_true')
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in a.keys:
        print('RESERVOIR', key, reg.RESERVOIRS[key]['name'], flush=True)
        row, _, _ = analyse_reservoir(key, OUT)
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        pd.DataFrame(rows).to_csv(OUT / 'summary_reservoirs.csv', index=False, encoding='utf-8-sig')
    if not a.skip_tibet:
        trows = []
        for lake in reg.TIBET:
            print('TIBET', lake, flush=True)
            trows.append(analyse_tibet(lake, OUT))
            print(trows[-1], flush=True)
        pd.DataFrame(trows).to_csv(OUT / 'summary_tibet.csv', index=False, encoding='utf-8-sig')
    (OUT / 'run_info.json').write_text(json.dumps(dict(
        script='04_Code/scripts/aev_observation_share_v1.py', registry='04_Code/scripts/aev_registry_v1.py',
        run_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), level_step_m=STEP,
        keys=a.keys, tibet=not a.skip_tibet), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
