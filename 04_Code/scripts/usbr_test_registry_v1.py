"""Registry, input grids and datum conversion for the 6 sealed USBR test reservoirs (Route-1 test).

Inputs are built from global products only (no survey data), with the rule of
docs/plans/2026-09-29-route1-test-protocol.md:
  W0   : NASADEM SWB water (255) with finite NASADEM, the largest 8-connected component within 5 km
         of the cohort coordinate (sampled on a +-25 km window, widened to +-50 km if it touches the edge)
  grid : UTM 30 m, bbox(W0) + 3 km; widened by 3 km while the SWO>=1% region connected to W0 touches the
         grid edge, up to 15 km (then clipped and flagged)
  dem  : FABDEM V1-2 (nearest), occurrence : JRC GSW v1.5 2024 (255 -> 0)
Datum (report is the primary source; archive metadata secondary):
  'NGVD29' -> + feet offset to NGVD29, then VERTCON (grid must cover the site), then GEOID18 - EGM2008
  'NAVD88' -> + feet offset to NAVD88, then GEOID18 - EGM2008
"""
from pathlib import Path
import sys, json, datetime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import pyproj
from pyproj import Transformer
from rasterio.transform import from_origin
from scipy import ndimage as ndi
import aev_registry_v1 as reg
from twdb_nasadem_intersection_screen import sample
from twdb_fabdem_intersection_screen import fabdem

ROOT = reg.ROOT
DATA = ROOT / '03_Data_Processed/ground_truth/usbr_test_20260929'
INPUTS = DATA / 'inputs'
SEALED = ROOT / '02_Data_Raw/sealed_test_bathymetry_20260922'
FT = reg.FT
TEST = {
    'T01': dict(name='Anderson Ranch', lon=-115.4501724, lat=43.3595096, epsg=32611, datum='NGVD29', feet=0.0,
                datum_note='project datum assumed NGVD29, shift to NAVD88 not determined (archive metadata)',
                archive='Anderson_Ranch.gdb.zip', sonar_layer='Bathymetricpoints'),
    'T02': dict(name='Cascade', lon=-116.050279, lat=44.524997, epsg=32611, datum='NGVD29', feet=-0.66,
                datum_note='report: Reclamation datum 0.66 ft higher than USC&GS datum (assumed NGVD29)',
                archive='Cascade.gdb.zip', sonar_layer=None),
    'T03': dict(name='Nelson', lon=-107.5186, lat=48.5297, epsg=32613, datum='NAVD88', feet=4.16,
                datum_note='report: project datum 1.70 ft below NGVD29; NGVD29 2.46 ft below NAVD88 at the 1999 control points',
                archive='Nelson.gdb.zip', sonar_layer='BathymetricPoints'),
    'T04': dict(name='Prineville', lon=-120.78738, lat=44.11125, epsg=32610, datum='NGVD29', feet=0.0,
                datum_note='project or construction datum, shift to NAVD88 not determined; treated as NGVD29',
                archive='Prineville.gdb.zip', sonar_layer='BathymetricPoints'),
    'T05': dict(name='Foss', lon=-99.1833, lat=35.55, epsg=32614, datum='NAVD88', feet=0.5,
                datum_note='report: project datum measured 0.3 ft above NGVD29 and 0.5 ft below NAVD88',
                archive='Foss.gdb.zip', sonar_layer='BathymetricPoints'),
    'T06': dict(name='Belle Fourche', lon=-103.6742, lat=44.7367, epsg=32613, datum='NAVD88', feet=1.6,
                datum_note='report: project datum near NGVD29 and around 1.6 ft lower than NAVD88',
                archive='Belle_Fourche.gdb.zip', sonar_layer='bathymetricpoints'),
}


def _grid(left, top, width, height):
    tr = from_origin(left, top, 30.0, 30.0)
    rr, cc = np.indices((height, width))
    return tr, left + (cc + 0.5) * 30.0, top - (rr + 0.5) * 30.0


def _lonlat(epsg, x, y):
    return Transformer.from_crs(epsg, 4326, always_xy=True).transform(x, y)


def select_W0(key):
    rec = TEST[key]
    x0, y0 = Transformer.from_crs(4326, rec['epsg'], always_xy=True).transform(rec['lon'], rec['lat'])
    for half in [25000.0, 50000.0]:
        left, top = np.floor((x0 - half) / 30) * 30, np.ceil((y0 + half) / 30) * 30
        n = int(2 * half / 30)
        tr, x, y = _grid(left, top, n, n)
        lon, lat = _lonlat(rec['epsg'], x, y)
        z, swb, _, _ = sample(lon, lat)
        water = (swb == 255) & np.isfinite(z)
        lab, _ = ndi.label(water, np.ones((3, 3)))
        near = np.hypot(x - x0, y - y0) <= 5000.0
        ids = np.unique(lab[near & water])
        if not len(ids):
            raise RuntimeError(f'{key}: no SWBD water within 5 km')
        sizes = ndi.sum(np.ones_like(lab), lab, index=ids)
        W0 = lab == ids[np.argmax(sizes)]
        edge = W0[0].any() or W0[-1].any() or W0[:, 0].any() or W0[:, -1].any()
        if not edge:
            rows, cols = np.nonzero(W0)
            bbox = (x[0, cols.min()] - 15, y[rows.max(), 0] - 15, x[0, cols.max()] + 15, y[rows.min(), 0] + 15)
            d = np.where(W0, np.hypot(x - x0, y - y0), np.inf)
            r, c = np.unravel_index(np.argmin(d), d.shape)
            seed_xy = (float(x[r, c]), float(y[r, c]))
            return bbox, seed_xy, dict(seed_xy=seed_xy, point_to_W0_m=float(d[r, c]), window_half_m=half, candidates_within_5km=len(ids), W0_km2_window=float(W0.sum() * 9e-4))
    raise RuntimeError(f'{key}: W0 touches the 50 km window')


def build_inputs(key):
    rec = TEST[key]
    bbox, seed_xy, info = select_W0(key)
    for margin in np.arange(3000.0, 15001.0, 3000.0):
        left, top = np.floor((bbox[0] - margin) / 30) * 30, np.ceil((bbox[3] + margin) / 30) * 30
        width = int(np.ceil((bbox[2] + margin - left) / 30)); height = int(np.ceil((top - (bbox[1] - margin)) / 30))
        tr, x, y = _grid(left, top, width, height)
        lon, lat = _lonlat(rec['epsg'], x, y)
        nas, swb, occ, _ = sample(lon, lat)
        occ = np.where(occ > 100, 0, occ).astype('uint8')
        water = (swb == 255) & np.isfinite(nas)
        lab, _ = ndi.label(water, np.ones((3, 3)))
        sr, sc = int((top - seed_xy[1]) // 30), int((seed_xy[0] - left) // 30)
        if not water[sr, sc]:
            raise RuntimeError(f'{key}: seed cell not water on the final grid')
        W0 = lab == lab[sr, sc]  # same component as selected on the window
        swo1 = reg.connected_to(occ >= 1, W0)
        touches = bool(swo1[:3].any() or swo1[-3:].any() or swo1[:, :3].any() or swo1[:, -3:].any())
        if not touches:
            break
    z, fab_sources = fabdem(lon, lat, tr, f'EPSG:{rec["epsg"]}')
    INPUTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(INPUTS / f'{key}_inputs.npz', fabdem=z.astype('float32'), W0=W0, occurrence=occ, R=swo1,
                        transform=np.array(tr)[:6])
    meta = dict(info, key=key, name=rec['name'], epsg=rec['epsg'], shape=list(z.shape), margin_m=float(margin),
                swo1_touches_edge=touches, W0_km2=float(W0.sum() * 9e-4), swo1_km2=float(swo1.sum() * 9e-4),
                fabdem_finite_in_swo1=float(np.isfinite(z[swo1]).mean()), occurrence_255_set_to_0=True,
                fabdem_tiles=sorted(set(fab_sources)), built_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (INPUTS / f'{key}_inputs.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta


def load_test(key):
    with np.load(INPUTS / f'{key}_inputs.npz') as d:
        g = dict(dem=d['fabdem'].astype(float), W0=d['W0'].astype(bool), occ=d['occurrence'].astype(int),
                 transform=np.asarray(d['transform'], float), R_input=d['R'].astype(bool))
    g['swo1'] = reg.connected_to(g['occ'] >= 1, g['W0'])
    return g


def egm_offset_m(key):
    """dict(feet, metres): EGM2008 metres = (report feet + feet) * 0.3048 + metres."""
    rec = TEST[key]
    pyproj.datadir.append_data_dir(str(reg.GRIDS))
    lon, lat = rec['lon'], rec['lat']
    geoid = Transformer.from_pipeline(
        '+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad '
        '+step +proj=vgridshift +grids=us_noaa_g2018u0.tif +multiplier=1 '
        '+step +inv +proj=vgridshift +grids=us_nga_egm08_25.tif +multiplier=1 '
        '+step +proj=unitconvert +xy_in=rad +xy_out=deg').transform(lon, lat, 0.0)[2]
    metres = geoid
    vertcon = None
    for grid in ['us_noaa_vertconc.tif', 'us_noaa_vertconw.tif']:
        try:
            v = Transformer.from_pipeline(
                '+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad '
                f'+step +proj=vgridshift +grids={grid} +multiplier=1 '
                '+step +proj=unitconvert +xy_in=rad +xy_out=deg').transform(lon, lat, 0.0)[2]
            if np.isfinite(v) and abs(v) < 5:
                vertcon = dict(grid=grid, navd88_minus_ngvd29_m=float(v)); break
        except Exception:
            continue
    if rec['datum'] == 'NGVD29':
        if vertcon is None:
            raise RuntimeError(f'{key}: no VERTCON grid covers the site')
        metres += vertcon['navd88_minus_ngvd29_m']
    return dict(feet=rec['feet'], metres=float(metres), geoid18_minus_egm2008_m=float(geoid), vertcon=vertcon)


def official_curve(key, offset):
    cur = pd.read_csv(DATA / f'{key}_official_curve_projectft.csv')
    return pd.DataFrame(dict(elevation_ft=cur.elevation_ft, H=(cur.elevation_ft + offset['feet']) * FT + offset['metres'],
                             area_km2=cur.area_km2, volume_km3=cur.volume_km3))


def to_egm(feet, offset):
    return (feet + offset['feet']) * FT + offset['metres']
