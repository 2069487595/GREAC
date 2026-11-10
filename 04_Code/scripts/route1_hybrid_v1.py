"""Step-2 hybrids of R1-C and GLOBathy (protocol docs/plans/2026-10-01-route1-hybrid-globathy-protocol.md, frozen 747e227).

GLOBathy-est: GLOBathy's released method (globathy_compare_v1.globathy_native / to_grid) with Dmax taken from the
  parameter table's model estimate Dmax_est_PAVEW_m for every lake; water surface H_surf = median DEM on the ring
  outside the polygon cells (as GLOBathy in step 1). Hylak_id by the W0-sampling rule (eval_expansion_run_v1.hylak_id).
Candidate surfaces (z_R = R1-C surface, z_G = GLOBathy-est surface; both FABDEM outside their own region):
  C1 mean          z = (z_R + z_G) / 2
  C2 select        z_R if area(F) <= area(W0) else z_G
  C3 select-mean   z_R if area(F) <= area(W0) else (z_R + z_G) / 2
  C4 rescale       inside F where z_R < H_seed: d = H_seed - z_R scaled to d * T / max(d), T = Dmax_est - (H_surf - H_seed);
                   other cells unchanged; T <= 0 -> R1-C
"""
from pathlib import Path
import sys, json

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
from scipy import ndimage as ndi
import aev_registry_v1 as reg
import globathy_compare_v1 as gb
import route1_mdem_v1 as md
import eval_expansion_run_v1 as run

CANDIDATES = ['C1', 'C2', 'C3', 'C4']


def globathy_est(key, g, epsg, folder, dm):
    """Write <key>_GLOBathy_est.tif (surface inside the polygon cells, NaN elsewhere) and return its record."""
    hid, poly, info = run.hylak_id(g, epsg)
    dmax = float(dm.loc[hid, 'Dmax_est_PAVEW_m'])
    depth, origin = gb.globathy_native(poly, dmax)
    d = gb.to_grid(depth, origin, g, epsg)
    P = np.isfinite(d)
    H_surf = float(np.median(g['dem'][ndi.binary_dilation(P, np.ones((3, 3))) & ~P]))
    z = np.full(P.shape, np.nan); z[P] = H_surf - d[P]
    md.write(folder, f'{key}_GLOBathy_est.tif', z, g, epsg)
    rec = dict(Hylak_id=hid, **info, Dmax_est_PAVEW_m=dmax, Dmax_use_m=float(dm.loc[hid, 'Dmax_use_m']), H_surf_m=H_surf,
               polygon_km2_on_grid=float(P.sum() * 9e-4))
    (folder / f'{key}_GLOBathy_est.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    return rec


def surfaces(g, F, H_seed, r1c_raster, gest_raster, gest_rec):
    """Full-grid surfaces and unknown regions for R1-C, GLOBathy-est and C1-C4."""
    dem, W0 = g['dem'], g['W0']
    zR = dem.copy(); zR[F] = r1c_raster[F]
    P = np.isfinite(gest_raster)
    zG = dem.copy(); zG[P] = gest_raster[P]
    lower = bool(F.sum() <= W0.sum())
    out = {'R1-C': (zR, F), 'GLOBathy-est': (zG, P)}
    out['C1'] = ((zR + zG) / 2, F | P)
    out['C2'] = (zR, F) if lower else (zG, P)
    out['C3'] = (zR, F) if lower else ((zR + zG) / 2, F | P)
    T = gest_rec['Dmax_est_PAVEW_m'] - (gest_rec['H_surf_m'] - H_seed)
    z4 = zR.copy()
    m = F & (zR < H_seed)
    if T > 0 and m.any():
        d = H_seed - zR[m]
        z4[m] = H_seed - d * T / d.max()
    out['C4'] = (z4, F)
    return out, dict(F_le_W0=lower, C4_T_m=float(T), C4_maxdepth_R1C_m=float((H_seed - zR[m]).max()) if m.any() else None,
                     C4_fallback=bool(not (T > 0 and m.any())))
