"""Waterbody registry and shared helpers for the Route-1 A-E-V diagnostics.

Route 1 = Tao reconstructs below the DEM water level; above it, DEM terrain inside the
SWO-observed extent is integrated directly. This module only gathers inputs that already
exist (Codex-prepared DEM / SWBD W0 / SWO occurrence grids, the Tao-original W0 runs, the
official TWDB curves) and converts official elevations to EGM2008. Nothing is re-surveyed,
re-screened, or tuned against bathymetry here.

Sources of the per-reservoir constants (pool elevation, vertical datum): the TWDB final
reports under 03_Data_Processed/ground_truth/twdb_five_20260918/*/ (Buchanan, Travis,
Medina, Meredith, Spence) and the Sam Rayburn reference manifest (NGVD29; the 2016
recalculated table tops out at the 164.4 ft conservation pool).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyproj
import rasterio
from pyproj import Transformer
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
CODEX = ROOT / '05_Results/codex/by_date'
V4 = CODEX / '2026-09-24/m0_m1_m2_runner_v4'
AUDIT = CODEX / '2026-09-18/twdb_five_reference_audit'
CURVES = ROOT / '03_Data_Processed/ground_truth/twdb_v4_clean/twdb_eac_tables_standardized_v4_clean.csv'
GRIDS = ROOT / '03_Data_Processed/ground_truth/twdb_five_20260918/vertical_grids'
FT = 0.3048
CELL_AREA = 900.0  # 30 m grids throughout

# datum: how the official curve elevations reach NAVD88
#   'NGVD29'       -> + VERTCON grid
#   'LCRA_legacy'  -> + navd_offset_ft (report: "legacy + x ft = NAVD88")
#   'Medina_local' -> - 7.8 ft to NGVD29 (report: Medina datum ~7.8 ft above NGVD29 numbers), then VERTCON
RESERVOIRS = {
    'D01': dict(name='E. V. Spence', entity='TWDB_E_V_SPENCE_RESERVOIR', table_id=76, pool_ft=1898.0,
                datum='NGVD29', crs=32614,
                inputs=AUDIT / 'TWDB_E_V_SPENCE_RESERVOIR_expanded_inputs.npz',
                m0=V4 / 'formal/D01/M0.tif',
                official_mask=(CODEX / '2026-09-20/spence_NE_sensitivity/SWO_audit_masks.npz', 'official1910'),
                official_note='TWDB 1910-ft topographic survey polygon (above the 1898-ft pool)'),
    'D02': dict(name='Medina', entity='TWDB_MEDINA', table_id=143, pool_ft=1072.0,
                datum='Medina_local', crs=32614,
                inputs=AUDIT / 'TWDB_MEDINA_expanded_inputs.npz', m0=None),
    'D03': dict(name='Buchanan', entity='TWDB_BUCHANAN', table_id=44, pool_ft=1020.0,
                datum='LCRA_legacy', navd_offset_ft=0.26, crs=32614,
                inputs=CODEX / '2026-09-23/dev_closed_domain_QC_v1/D03_inputs.npz',
                m0=V4 / 'formal/D03/M0.tif', official_mask=('input', 'R'),
                official_note='Codex R: SWO>=1% closed by the official high-water boundary'),
    'D04': dict(name='Travis', entity='TWDB_TRAVIS', table_id=202, pool_ft=681.0,
                datum='LCRA_legacy', navd_offset_ft=0.60, crs=32614,
                inputs=CODEX / '2026-09-23/dev_closed_domain_QC_v1/D04_inputs.npz',
                m0=V4 / 'formal/D04/M0.tif', official_mask=('input', 'R'),
                official_note='Codex R: SWO>=1% closed by the official high-water boundary'),
    'D05': dict(name='Meredith', entity='TWDB_MEREDITH', table_id=145, pool_ft=2936.5,
                datum='NGVD29', crs=32614,
                inputs=AUDIT / 'TWDB_MEREDITH_expanded_inputs.npz', m0=None),
    'D06': dict(name='Sam Rayburn', entity='TWDB_SAM_RAYBURN', table_id=180, pool_ft=164.4,
                datum='NGVD29', crs=32615,
                inputs=CODEX / '2026-09-23/dev_admission_closure_20260923_v1/D06_domain_inputs/D06_inputs.npz',
                m0=V4 / 'formal/D06/M0.tif', official_mask=('input', 'R'),
                official_note='Codex R: SWO>=1% closed by the official boundary'),
}

TIBET = {
    'laang_co': dict(name='Laang Co', paper_volume_km3=5.53, paper_tao_volume_km3=6.41),
    'mapang_yongco': dict(name='Mapang Yongco', paper_volume_km3=17.20, paper_tao_volume_km3=26.70),
    'ngangla_rinco': dict(name='Ngangla Rinco', paper_volume_km3=9.09, paper_tao_volume_km3=9.57),
}
TIBET_INPUTS = CODEX / '2026-09-15/tibet_three_lakes_v1'
TIBET_TAO = CODEX / '2026-09-16/tibet_three_group_v3_bank_alignment'


def place(src, src_transform, dst_shape, dst_transform, fill):
    """Paste a same-resolution, grid-aligned array into another grid by its affine origin."""
    res = dst_transform[0]
    r0 = int(round((dst_transform[5] - src_transform[5]) / res))
    c0 = int(round((src_transform[2] - dst_transform[2]) / res))
    out = np.full(dst_shape, fill, dtype=np.asarray(src).dtype)
    rs, cs = max(r0, 0), max(c0, 0)
    re, ce = min(r0 + src.shape[0], dst_shape[0]), min(c0 + src.shape[1], dst_shape[1])
    if re > rs and ce > cs:
        out[rs:re, cs:ce] = src[rs - r0:re - r0, cs - c0:ce - c0]
    return out


def connected_to(mask, seed):
    """Cells of mask in 8-connected components that touch seed."""
    lab, _ = ndi.label(mask, np.ones((3, 3)))
    ids = np.unique(lab[seed & mask])
    return np.isin(lab, ids[ids > 0])


def load_reservoir(key):
    """DEM, SWBD W0, SWO occurrence and transform on the analysis grid of one reservoir."""
    rec = RESERVOIRS[key]
    with np.load(rec['inputs']) as d:
        out = dict(dem=d['fabdem'].astype(float), W0=d['W0'].astype(bool),
                   occ=d['occurrence'].astype(int), transform=np.asarray(d['transform'], float),
                   R_input=d['R'].astype(bool))
    out['swo1'] = connected_to(out['occ'] >= 1, out['W0'])
    if 'official_mask' in rec:
        src, name = rec['official_mask']
        if src == 'input':
            out['official'] = out['R_input']
        else:
            with np.load(src) as m:
                out['official'] = place(m[name].astype(bool), m['transform'], out['dem'].shape,
                                        out['transform'], False)
    return out


def load_m0(path, grid):
    """Tao-original W0 reconstruction (NaN outside W0) pasted onto the analysis grid."""
    with rasterio.open(path) as s:
        a = s.read(1).astype(float)
        t = np.array([s.transform.a, s.transform.b, s.transform.c,
                      s.transform.d, s.transform.e, s.transform.f])
    return place(a, t, grid['dem'].shape, grid['transform'], np.nan)


def lonlat_center(grid, crs):
    rows, cols = np.nonzero(grid['W0'])
    t = grid['transform']
    x = t[2] + (cols.mean() + .5) * t[0]
    y = t[5] + (rows.mean() + .5) * t[4]
    return Transformer.from_crs(crs, 4326, always_xy=True).transform(x, y)


def egm_offset_m(key, grid):
    """Metres to add to (curve feet * 0.3048), after the datum-specific feet offset, to get EGM2008."""
    pyproj.datadir.append_data_dir(str(GRIDS))
    lon, lat = lonlat_center(grid, RESERVOIRS[key]['crs'])
    vertcon = Transformer.from_pipeline(
        '+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad '
        '+step +proj=vgridshift +grids=us_noaa_vertconc.tif +multiplier=1 '
        '+step +proj=unitconvert +xy_in=rad +xy_out=deg')
    navd_to_egm = Transformer.from_pipeline(
        '+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad '
        '+step +proj=vgridshift +grids=us_noaa_g2018u0.tif +multiplier=1 '
        '+step +inv +proj=vgridshift +grids=us_nga_egm08_25.tif +multiplier=1 '
        '+step +proj=unitconvert +xy_in=rad +xy_out=deg')
    geoid = navd_to_egm.transform(lon, lat, 0.0)[2]
    ngvd = vertcon.transform(lon, lat, 0.0)[2]
    datum = RESERVOIRS[key]['datum']
    if datum == 'NGVD29':
        return dict(feet=0.0, metres=ngvd + geoid, lon=lon, lat=lat)
    if datum == 'Medina_local':
        return dict(feet=-7.8, metres=ngvd + geoid, lon=lon, lat=lat)
    if datum == 'LCRA_legacy':
        return dict(feet=RESERVOIRS[key]['navd_offset_ft'], metres=geoid, lon=lon, lat=lat)
    raise ValueError(datum)


def official_curve(key, offset):
    """Official TWDB curve with elevations converted to EGM2008 metres."""
    rec = RESERVOIRS[key]
    cur = pd.read_csv(CURVES, low_memory=False)
    cur = cur[(cur.entity_id == rec['entity']) & (cur.table_id == rec['table_id'])]
    cur = cur.sort_values('elevation_ft').drop_duplicates('elevation_ft')
    if cur.empty:
        raise ValueError(f'no official curve for {key}')
    return pd.DataFrame(dict(
        elevation_ft=cur.elevation_ft.values,
        H=(cur.elevation_ft.values + offset['feet']) * FT + offset['metres'],
        area_km2=cur.area_km2.values, volume_km3=cur.capacity_km3.values))


def to_egm(key, feet, offset):
    return (feet + offset['feet']) * FT + offset['metres']
