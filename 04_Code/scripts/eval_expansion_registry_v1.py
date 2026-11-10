"""Shared helpers for the evaluation-expansion reservoirs: inputs, datum conversion, official curves, admission.

Inputs: eval_expansion_inputs_v1 grids. Datum (protocol section 4 rule 3): report feet + offset to NGVD29 or NAVD88,
NGVD29 -> NAVD88 by VERTCON (central or west grid, whichever covers the point), NAVD88 -> EGM2008 by GEOID18 - EGM2008.
Official curves: TWDB = the local v4_clean table chosen by eval_expansion_download_list_v1 (feet, acres/acre-feet
already converted to km2 / km3); USBR = curves/<key>_curve_ft.csv written from the reports.
Admission (rules 1, 4, 5, 6) uses only the official curve at the normal pool and the input grids.
"""
from pathlib import Path
import sys, json

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import pyproj
from pyproj import Transformer
import aev_registry_v1 as reg

ROOT = reg.ROOT
GT = ROOT / '03_Data_Processed/ground_truth'
DATA = GT / 'eval_expansion_20260930'
INPUTS = DATA / 'inputs'
CURVES = DATA / 'curves'
RES = ROOT / '05_Results/by_date/2026-09-30/eval_expansion_v1'
FT = reg.FT


def safe(key):
    return str(key).replace(' ', '_').replace('(', '').replace(')', '')


def load(key):
    with np.load(INPUTS / f'{safe(key)}_inputs.npz') as d:
        g = dict(dem=d['fabdem'].astype(float), W0=d['W0'].astype(bool), occ=d['occurrence'].astype(int),
                 nasadem=d['nasadem'].astype(float), transform=np.asarray(d['transform'], float), R_input=d['R'].astype(bool))
    g['swo1'] = reg.connected_to(g['occ'] >= 1, g['W0'])
    g['meta'] = json.loads((INPUTS / f'{safe(key)}_inputs.json').read_text(encoding='utf-8'))
    return g


def egm_offset(lon, lat, datum, feet=0.0):
    """dict(feet, metres): EGM2008 m = (report ft + feet) * 0.3048 + metres; datum is the target of `feet` (NGVD29 / NAVD88)."""
    pyproj.datadir.append_data_dir(str(reg.GRIDS))
    geoid = Transformer.from_pipeline(
        '+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad '
        '+step +proj=vgridshift +grids=us_noaa_g2018u0.tif +multiplier=1 '
        '+step +inv +proj=vgridshift +grids=us_nga_egm08_25.tif +multiplier=1 '
        '+step +proj=unitconvert +xy_in=rad +xy_out=deg').transform(lon, lat, 0.0)[2]
    vc = None
    for grid in ['us_noaa_vertconc.tif', 'us_noaa_vertconw.tif']:
        try:
            v = Transformer.from_pipeline('+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad '
                                          f'+step +proj=vgridshift +grids={grid} +multiplier=1 '
                                          '+step +proj=unitconvert +xy_in=rad +xy_out=deg').transform(lon, lat, 0.0)[2]
            if np.isfinite(v) and abs(v) < 5:
                vc = float(v); break
        except Exception:
            continue
    metres = float(geoid)
    if datum == 'NGVD29':
        if vc is None:
            raise RuntimeError('no VERTCON grid covers the point')
        metres += vc
    return dict(feet=float(feet), metres=metres, geoid18_minus_egm2008_m=float(geoid), vertcon_m=vc, datum=datum)


def twdb_curve(table_id):
    std = pd.read_csv(GT / 'twdb_v4_clean/twdb_eac_tables_standardized_v4_clean.csv',
                      usecols=['table_id', 'elevation_ft', 'area_km2', 'capacity_km3'], low_memory=False)
    c = std[std.table_id == table_id].sort_values('elevation_ft').drop_duplicates('elevation_ft')
    return pd.DataFrame(dict(elevation_ft=c.elevation_ft.values, area_km2=c.area_km2.values, volume_km3=c.capacity_km3.values))


def official(key, facts_row, table_id=None):
    """Official curve in EGM2008 plus the normal pool elevation, for a TWDB or USBR key."""
    c = twdb_curve(table_id) if table_id is not None else pd.read_csv(CURVES / f'{safe(key)}_curve_ft.csv')
    g_meta = json.loads((INPUTS / f'{safe(key)}_inputs.json').read_text(encoding='utf-8'))
    off = egm_offset(g_meta['lon'], g_meta['lat'], facts_row['datum'], facts_row.get('feet_offset', 0.0) or 0.0)
    cur = pd.DataFrame(dict(elevation_ft=c.elevation_ft, H=(c.elevation_ft + off['feet']) * FT + off['metres'],
                            area_km2=c.area_km2, volume_km3=c.volume_km3))
    H_pool = (facts_row['pool_ft'] + off['feet']) * FT + off['metres']
    return cur, float(H_pool), off
