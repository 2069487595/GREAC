"""Step 3a of the frozen plan docs/plans/2026-10-22-new-data-independent-test-plan.md (section 13, item 4): the official
reference curves of the admitted reservoirs in one format, the normal pool in EGM2008, and the hashes of both. Reads
official tables only; reads no product value and computes no error.

Sources
  Brazil ANA   curve sheet of the technical workbook: SGB elevation (m), area (km2), volume (hm3). Multi-plant sheets:
               the column group whose area at the normal level is closest to the technical sheet's area. LIGHT sheets:
               the 'SGB Batimetria' column. Single-column sheets: that column (datum flagged at admission).
               Normal level on SGB from the admission (interpolated at the local level in the same curve).
  USGS         the release tables (feet NAVD88; acres; acre-feet).
  Morse, Geist SIR 2020-5067 tables 3 and 4 (feet NAVD88; acres; acre-feet), read with pdftotext -layout.
  Livingston   TWDB v4_clean table 135, the 2019 survey (feet NGVD29), as in the evaluation (eval_expansion_registry_v1.twdb_curve).
  OWRB         area and capacity at the normal pool, transcribed into newdata_owrb_reference_v1.csv (the report's stated values: summary table or results sentence; the 0.1-ft
               table row as a check). Only the normal-pool point: the appendix tables are not reliably readable as text,
               so curve-based descriptive metrics are not computed for OWRB.
EGM2008
  US      eval_expansion_registry_v1.egm_offset at the product reservoir's outlet (VERTCON for NGVD29, GEOID18 - EGM2008).
  Brazil  H_EGM2008 = H_SGB + hgeoHNOR2020 conversion factor - N_EGM2008 at the outlet, bilinear; inside the MAPGEO2015
          western cut-out or the Santana area the corresponding grid of the IBGE release.
Outputs: 05_Results/by_date/2026-10-23/newdata_reference_v1/{curves/<id>.csv, reference_index.csv, sources/}
Usage: .venv/Scripts/python.exe -I 04_Code/scripts/newdata_reference_v1.py
"""
from pathlib import Path
import sys, re, json, hashlib, subprocess
import numpy as np
import pandas as pd
import shapely
import pyproj
from pyproj import Transformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding='utf-8')
import xlsx_min as X
import aev_registry_v1 as reg
import eval_expansion_registry_v1 as er

ROOT = HERE.parents[1]
RAW = ROOT / '02_Data_Raw'
ADM = ROOT / '05_Results/by_date/2026-10-22/newdata_admission_v2/admission.csv'
MAN = HERE / 'newdata_admission_manual_v1.csv'
OWRB_MAN = HERE / 'newdata_owrb_reference_v1.csv'
GLOBAL = ROOT / '05_Results/by_date/2026-10-10/r2_global_v1/global_summary_R2.csv'
IBGE = RAW / 'new_surveys_20261022/IBGE_hgeoHNOR2020/grids'
OUT = ROOT / '05_Results/by_date/2026-10-23/newdata_reference_v1'
FT, ACRE_KM2, ACFT_KM3 = 0.3048, 0.00404685642, 1.23348183755e-6
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
safe = lambda c: re.sub(r'[^A-Za-z0-9_.-]+', '_', c)


# ---------------------------------------------------------------- vertical datums
class Grid:
    """IBGE text grid: longitude (0-360 E), latitude, value; regular 5' spacing; bilinear."""
    def __init__(self, path):
        a = np.loadtxt(path)
        self.lon = np.unique(a[:, 0]); self.lat = np.unique(a[:, 1])
        v = pd.DataFrame(a, columns=['x', 'y', 'v']).pivot(index='y', columns='x', values='v')
        self.v = v.reindex(index=self.lat, columns=self.lon).values

    def __call__(self, lon, lat):
        x = lon % 360
        i = np.searchsorted(self.lat, lat) - 1; j = np.searchsorted(self.lon, x) - 1
        if not (0 <= i < len(self.lat) - 1 and 0 <= j < len(self.lon) - 1):
            return np.nan
        ty = (lat - self.lat[i]) / (self.lat[i + 1] - self.lat[i]); tx = (x - self.lon[j]) / (self.lon[j + 1] - self.lon[j])
        q = self.v[i:i + 2, j:j + 2]
        return float((1 - ty) * ((1 - tx) * q[0, 0] + tx * q[0, 1]) + ty * ((1 - tx) * q[1, 0] + tx * q[1, 1]))


def poly(path):
    xy = np.loadtxt(path)
    return shapely.Polygon(xy)


def brazil_offset(lon, lat, grids):
    """metres to add to an SGB height to get an EGM2008 height."""
    pt = shapely.Point(lon, lat)
    if grids['santana_poly'].contains(pt):
        name, f = 'hgeoHNOR2020 Santana', grids['santana'](lon, lat)
    elif grids['west_poly'].contains(pt):
        name, f = 'MAPGEO2015 western cut-out', grids['west'](lon, lat)
    else:
        name, f = 'hgeoHNOR2020 Imbituba', grids['imbituba'](lon, lat)
    pyproj.datadir.append_data_dir(str(reg.GRIDS))
    n_egm = -Transformer.from_pipeline('+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad '
                                       '+step +inv +proj=vgridshift +grids=us_nga_egm08_25.tif +multiplier=1 '
                                       '+step +proj=unitconvert +xy_in=rad +xy_out=deg').transform(lon, lat, 0.0)[2]
    return dict(model=name, conversion_factor_m=float(f), N_EGM2008_m=float(n_egm), metres=float(f - n_egm))


# ---------------------------------------------------------------- readers (native units -> H_native, area_km2, volume_km3)
def brazil_curve(r):
    path = RAW / r.table_file
    sheet = next(n for n, _ in X.sheets(path) if 'curva' in n.lower() or 'cav' in n.lower())
    rows = X.read(path, sheet)
    short = lambda x, pat: isinstance(x, str) and len(x) < 60 and re.search(pat, x)
    hi = next(i for i, rr in enumerate(rows[:25]) if any(short(x, r'(?i)cota|altitude') for x in rr) and any(short(x, r'(?i)área|area') for x in rr))
    hdr = [re.sub(r'\s+', ' ', x).strip() if isinstance(x, str) else '' for x in rows[hi]]
    acols = [j for j, h in enumerate(hdr) if re.search(r'(?i)^área|^area', h)]
    groups = []
    for a in acols:
        v = next(j for j in range(a + 1, len(hdr)) if re.search(r'(?i)volume', hdr[j]))
        ecols = [j for j in range(max(0, a - 3), a) if re.search(r'(?i)cota|altitude', hdr[j])]
        loc = [j for j in ecols if re.search(r'(?i)opera|local|usina|r[eé]gua|\bSL\b', hdr[j])]
        sgb = [j for j in ecols if j not in loc and re.search(r'(?i)sgb|sbg|ibge|imbituba|geod', hdr[j])]
        if len(sgb) > 1:
            sgb = [j for j in sgb if re.search(r'(?i)batimetria', hdr[j])] or sgb[-1:]
        loc = loc or [j for j in ecols if j not in sgb]
        hcol = sgb[0] if sgb else ecols[-1]
        data = [(rr[hcol], rr[loc[0]] if loc else np.nan, rr[a], rr[v]) for rr in rows[hi + 1:]
                if max(hcol, a, v) < len(rr) and all(isinstance(rr[k], float) for k in (hcol, a, v))]
        c = pd.DataFrame(data, columns=['H', 'H_local', 'area_km2', 'vol_hm3']).sort_values('H').drop_duplicates('H')
        groups.append(dict(cols=(hdr[hcol], hdr[loc[0]] if loc else '', hdr[a], hdr[v]), curve=c, sgb=bool(sgb)))
    if len(groups) > 1:   # multi-plant sheet: the group matching the technical sheet's area at the normal level
        def gap(gr):
            c = gr['curve'].dropna(subset=['H_local']).sort_values('H_local')
            return abs(float(np.interp(r.npl, c.H_local, c.area_km2)) - float(r.official_area_npl_km2))
        g = min(groups, key=gap); note = f'multi-plant sheet: group {g["cols"]} chosen by area at the normal level'
    else:
        g = groups[0]; note = ''
    c = g['curve']
    npl_native = r.npl_sgb_m if pd.notna(r.npl_sgb_m) else None
    if npl_native is None:     # multi-plant sheet: convert the local level with the chosen group
        cl = c.dropna(subset=['H_local']).sort_values('H_local')
        npl_native = float(np.interp(r.npl, cl.H_local, cl.H)) if len(cl) else float(r.npl)
    out = pd.DataFrame(dict(H_native=c.H.values, area_km2=c.area_km2.values, volume_km3=c.vol_hm3.values / 1e3))
    return out, float(npl_native), 'm SGB' if g['sgb'] else 'm (single column, datum flagged)', dict(columns=list(g['cols']), note=note, sheet=sheet)


def usgs_curve(r):
    t = pd.read_csv(RAW / r.table_file)
    t.columns = [re.sub(r'\s+', ' ', c).strip() for c in t.columns]
    ec = next(c for c in t.columns if re.search(r'(?i)elev', c) and re.search(r'(?i)navd ?88', c))
    ac = next(c for c in t.columns if re.search(r'(?i)area', c))
    vc = next(c for c in t.columns if re.search(r'(?i)capacity|volume', c))
    # some tables write thousands separators inside quotes (" 53,181 ")
    t = t[[ec, ac, vc]].apply(lambda s: pd.to_numeric(s.astype(str).str.replace(',', '').str.strip(), errors='coerce')).dropna().sort_values(ec).drop_duplicates(ec)
    return (pd.DataFrame(dict(H_native=t[ec].values, area_km2=t[ac].values * ACRE_KM2, volume_km3=t[vc].values * ACFT_KM3)),
            float(r.npl), 'ft NAVD88', dict(columns=[ec, ac, vc]))


def morse_geist_curve(r):
    name = r.candidate.split(':')[1]
    pdf = RAW / 'new_surveys_20261012/USGS_P9A2ITC6_Morse_Geist_2016/sir20205067.pdf'
    txt = OUT / 'sources' / 'sir20205067_layout.txt'
    if not txt.exists():
        subprocess.run(['pdftotext', '-layout', str(pdf), str(txt)], check=True)
    t = txt.read_text(encoding='utf-8', errors='ignore')
    # the table titles also appear in the list of tables at the front, so the last occurrence is the table itself
    t3, t4 = t.rindex('Table 3. Reservoir surface area'), t.rindex('Table 4. Reservoir surface area')
    k = 'Table 3' if name == 'Morse' else 'Table 4'
    block = t[t3:t4] if name == 'Morse' else t[t4:t4 + 5000]
    trip = re.findall(r'(\d{3}\.\d{2})\s+([\d,]+(?:\.\d+)?)\s+([\d,]+(?:\.\d+)?)', block)
    num = lambda s: float(s.replace(',', ''))
    c = pd.DataFrame([(num(e), num(ar), num(v)) for e, ar, v in trip], columns=['e', 'a', 'v']).sort_values('e').drop_duplicates('e')
    return (pd.DataFrame(dict(H_native=c.e.values, area_km2=c.a.values * ACRE_KM2, volume_km3=c.v.values * ACFT_KM3)),
            float(r.npl), 'ft NAVD88', dict(source=f'SIR 2020-5067 {k}', rows=int(len(c))))


def livingston_curve(r):
    tid = int(pd.read_csv(er.RES / 'twdb_download_list.csv').set_index('entity_id').loc['TWDB_LIVINGSTON', 'table_id'])   # 135, the 2019 survey
    c = er.twdb_curve(tid)
    return pd.DataFrame(dict(H_native=c.elevation_ft.values, area_km2=c.area_km2.values, volume_km3=c.volume_km3.values)), float(r.npl), 'ft NGVD29', dict(table_id=tid)


def owrb_curve(r, om):
    """Normal-pool values transcribed from the report (om row); the full curve if newdata_owrb_curves_v1 parsed it."""
    f = OUT / 'owrb_parsed' / f'{safe(r.candidate)}.csv'
    pt = pd.DataFrame(dict(H_native=[float(r.npl)], area_km2=[om.area_acres_npl * ACRE_KM2], volume_km3=[om.capacity_acft_npl * ACFT_KM3]))
    info = dict(source=f'{om.report}, {om.where}', normal_pool_values='transcribed from the report', check=om.check if isinstance(om.check, str) else '', full_curve=f.exists())
    if f.exists():
        c = pd.read_csv(f)
        c = pd.DataFrame(dict(H_native=c.elevation_ft, area_km2=c.area_acres * ACRE_KM2, volume_km3=c.capacity_acft * ACFT_KM3))
        return c, float(r.npl), f'ft {r.npl_datum}', dict(info, pool_point=pt.iloc[0].to_dict())
    return pt, float(r.npl), f'ft {r.npl_datum}', info


# ---------------------------------------------------------------- main
def main():
    (OUT / 'curves').mkdir(parents=True, exist_ok=True); (OUT / 'sources').mkdir(exist_ok=True)
    a = pd.read_csv(ADM); a = a[a.status == 'admitted'].copy()
    g = pd.read_csv(GLOBAL, usecols=['Hylak_id', 'Pour_long', 'Pour_lat']).set_index('Hylak_id')
    man = pd.read_csv(MAN).set_index('candidate')
    om = pd.read_csv(OWRB_MAN).set_index('candidate') if OWRB_MAN.exists() else pd.DataFrame()
    grids = dict(imbituba=Grid(IBGE / 'hgeoHNOR2020__IMBITUBA__fator-conversao.txt'), santana=Grid(IBGE / 'hgeoHNOR2020__SANTANA__fator-conversao.txt'),
                 west=Grid(IBGE / 'MAPGEO2015_SIRGAS2000_recorte-oeste.txt'), west_poly=poly(IBGE / 'polig_MAPGEO2015_OESTE.txt'),
                 santana_poly=poly(IBGE / 'polig_hgeoHNOR2020_SANTANA.txt'))
    rows = []
    for r in a.itertuples():
        hid = int(r.Hylak_id); lon, lat = float(g.loc[hid, 'Pour_long']), float(g.loc[hid, 'Pour_lat'])
        rec = dict(candidate=r.candidate, source=r.source, Hylak_id=hid, lon=lon, lat=lat, flags=r.flags if isinstance(r.flags, str) else '')
        try:
            if r.source == 'Brazil ANA':
                c, npl, unit, info = brazil_curve(r)
                off = brazil_offset(lon, lat, grids); to_m = lambda h: h + off['metres']
            else:
                if r.source == 'OWRB':
                    if r.candidate not in om.index:
                        raise RuntimeError('normal-pool values not transcribed yet')
                    c, npl, unit, info = owrb_curve(r, om.loc[r.candidate])
                elif r.candidate.startswith('USGS_P9A2ITC6'):
                    c, npl, unit, info = morse_geist_curve(r)
                elif r.source == 'TWDB':
                    c, npl, unit, info = livingston_curve(r)
                else:
                    c, npl, unit, info = usgs_curve(r)
                datum = 'NGVD29' if 'NGVD29' in unit else 'NAVD88'
                off = er.egm_offset(lon, lat, datum); to_m = lambda h: h * FT + off['metres']
            c = c.sort_values('H_native').reset_index(drop=True)
            c['H_egm2008_m'] = to_m(c.H_native)
            path = OUT / 'curves' / f'{safe(r.candidate)}.csv'
            c[['H_native', 'H_egm2008_m', 'area_km2', 'volume_km3']].to_csv(path, index=False, float_format='%.8g')
            covers = bool(c.H_native.min() - 1e-6 <= npl <= c.H_native.max() + 1e-6)
            rec.update(npl_native=npl, native_unit=unit, egm_offset=json.dumps(off, default=float), H_pool_egm2008_m=float(to_m(npl)),
                       Vmax_official_km3=float(np.interp(npl, c.H_native, c.volume_km3)) if covers else np.nan,
                       A_pool_official_km2=float(np.interp(npl, c.H_native, c.area_km2)) if covers else np.nan,
                       curve_points=int(len(c)), curve_covers_npl=covers, curve_min_native=float(c.H_native.min()), curve_max_native=float(c.H_native.max()),
                       admission_official_area_km2=r.official_area_npl_km2, info=json.dumps(info, default=str, ensure_ascii=False), curve_sha256=sha(path))
        except Exception as e:
            rec['error'] = repr(e)[:200]
        rows.append(rec)
        print(r.candidate, rec.get('error') or f"V {rec['Vmax_official_km3']:.4g} km3 at {rec['H_pool_egm2008_m']:.2f} m EGM2008", flush=True)
    idx = pd.DataFrame(rows)
    idx.to_csv(OUT / 'reference_index.csv', index=False, encoding='utf-8-sig', float_format='%.8g')
    print('written', len(idx), 'errors', int(idx.get('error', pd.Series(dtype=str)).notna().sum()))


if __name__ == '__main__':
    main()
