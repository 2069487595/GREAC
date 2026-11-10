"""Official area-capacity curves of the 6 sealed USBR test reservoirs, from the USBR survey reports.

Sources (downloaded 2026-09-29 from usbr.gov/tsc/techreferences/reservoir.html into
02_Data_Raw/usbr_reports_20260929/, SHA-256 in the output manifest):
  T03 Nelson, T05 Foss : ACAP capacity equations (y = a1 + a2 x + a3 x^2, x = elevation - base), printed
                         in the separate "Area and Capacity Tables" PDFs; parsed from the text layer.
                         Area = dy/dx = a2 + 2 a3 x. Evaluated every 0.1 ft.
  T01, T02, T04, T06   : survey columns of "Table 2 - Summary of ... survey results" (contour intervals),
                         transcribed below. T01 and T02 are scanned pages: values checked against page images.
Elevations stay in each report's project datum (feet); datum conversion happens in usbr_test_registry_v1.
Normal pool = top of the highest conservation or joint-use zone (below flood-only and surcharge zones),
from the "Reservoir operations" section of each report. Nothing here reads survey points or runs a model.
Check: for tabulated curves, capacity increments vs trapezoidal integration of area.
"""
from pathlib import Path
import re, json, hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / '02_Data_Raw/usbr_reports_20260929'
TEXT = ROOT / '05_Results/by_date/2026-09-29/test_admission_v1/report_text'
OUT = ROOT / '03_Data_Processed/ground_truth/usbr_test_20260929'
ACRE_KM2 = 0.0040468564224
AF_KM3 = 1233.48183754752e-9

# elevation_ft, survey area (acres), survey capacity (acre-feet)
TABLE2 = {
    'T01': dict(report='Anderson_Ranch_Sediment_Survey_1998.pdf', page=18, columns='(4) 1998 area, (5) 1998 capacity', rows=[
        (4210, 5115, 543988), (4206, 5012, 523735), (4200, 4854, 494136), (4199, 4826, 489296), (4198.2, 4804, 485444),
        (4196, 4743, 474942), (4190, 4568, 447002), (4180, 4219, 403058), (4170, 3878, 362615), (4160, 3589, 325337),
        (4150, 3350, 290628), (4140, 3078, 258425), (4130, 2802, 229179), (4120, 2630, 202026), (4110, 2425, 176653),
        (4100, 2175, 153659), (4090, 1906, 133291), (4080, 1675, 115411), (4070, 1491, 99600), (4060, 1312, 85575),
        (4050, 1159, 73226), (4044, 1082, 66500), (4040, 1028, 62278), (4039.6, 1023, 61868), (4030, 906, 52619),
        (4020, 798, 44126), (4010, 716, 36555), (4000, 638, 29771), (3992, 578, 24912), (3990, 564, 23770),
        (3980, 496, 18458), (3970, 412, 13915), (3960, 337, 10173), (3950, 284, 7090), (3940, 233, 4496),
        (3930, 167, 2516), (3920, 101, 1187), (3910, 46, 458), (3900, 21, 132), (3890, 4, 8), (3886, 0, 0)]),
    'T02': dict(report='Cascade_Reservoir_1995_Sedimentation_Survey.pdf', page=18, columns='(4) 1995 area, (5) 1995 capacity', rows=[
        (4840.0, 32909, 1049146), (4835.0, 30260, 891228), (4833.6, 29484, 849404), (4830.0, 27466, 746895),
        (4828.0, 26307, 693123), (4825.0, 24512, 616819), (4820.0, 22062, 501099), (4815.0, 19245, 397828),
        (4810.0, 17009, 307252), (4805.0, 14746, 227835), (4800.0, 12437, 159714), (4795.0, 9731, 104297),
        (4790.0, 6998, 62698), (4787.5, 5837, 46662), (4785.0, 4744, 33459), (4780.0, 2836, 14405),
        (4775.0, 1364, 4381), (4770.0, 251, 542), (4766.0, 39, 62), (4765.0, 22, 32), (4759.2, 0, 0)]),
    'T04': dict(report='Prineville_Reservoir_1998_Sedimentation_Survey.pdf', page=20, columns='(4) 1998 area, (5) 1998 capacity', rows=[
        (3234.8, 3028, 150216), (3230, 2832, 136152), (3225, 2643, 122485), (3220, 2411, 109828), (3215, 2191, 98322),
        (3211.2, 2054, 90257), (3210, 2011, 87818), (3205, 1867, 78123), (3200, 1724, 69145), (3195, 1589, 60862),
        (3190, 1468, 53218), (3185, 1356, 46157), (3180, 1247, 39647), (3175, 1123, 33720), (3170, 968, 28493),
        (3165, 845, 23962), (3160, 755, 19963), (3155, 671, 16399), (3150, 574, 13286), (3145, 488, 10631),
        (3140, 405, 8399), (3135, 339, 6539), (3130, 290, 4968), (3125, 235, 3657), (3121.5, 203, 2890),
        (3120, 189, 2596), (3114, 149, 1583), (3112, 135, 1299), (3110, 122, 1041), (3105, 80, 536),
        (3100, 46, 221), (3095, 20, 57), (3090, 2, 3), (3087.2, 0, 0)]),
    'T06': dict(report='Belle_Fourche_Reservoir_2006_Sedimentation_Survey.pdf', page=23, columns='(8) 2006 area, (9) 2006 capacity', rows=[
        (2975, 8040.0, 172873), (2970, 7153.9, 134888), (2965, 5919.5, 102366), (2960, 5045.0, 75010),
        (2955, 3982.2, 52355), (2950, 2753.6, 35531), (2945, 1956.1, 23923), (2940, 1478.0, 15407),
        (2935, 1034.7, 9161), (2930, 684.1, 4866), (2925, 425.9, 2142), (2920, 212.6, 540), (2915, 18.0, 9),
        (2910, 0.0, 0)]),
}
EQUATIONS = {
    'T03': dict(report='Nelson_Reservoir_1999_Area_and_Capacity_Tables.pdf', text='Nelson_Reservoir_1999_Area_and_Capacity_Tables.txt',
                page=2, top_ft=2223.0),
    'T05': dict(report='Foss_Reservoir_2009_Capacity_Tables.pdf', text='Foss_Reservoir_2009_Capacity_Tables.txt',
                page='3-4', top_ft=1694.0),
}
POOL = {  # ft, project datum; zone text from "Reservoir operations"
    'T01': (4196.0, 'top of joint use (408,442 AF joint use 4,044.0-4,196.0; surcharge above)'),
    'T02': (4828.0, 'top of conservation (646,461 AF conservation 4,787.5-4,828.0; surcharge above)'),
    'T03': (2221.6, 'top of conservation (78,950 AF total storage at 2,221.6; never operated above)'),
    'T04': (3234.8, 'top of joint use / full pool and top of spillway (joint use 3,211.17-3,234.8; surcharge above)'),
    'T05': (1642.0, 'top of active conservation (1,597.2-1,642.0; flood control 1,642.0-1,662.2 above)'),
    'T06': (2975.0, 'top of conservation (2,927.0-2,975.0; surcharge above)'),
}


def parse_equations(txt):
    pat = re.compile(r'^\s*(\d+)\s+(\d{4}\.\d{2})\s+(\d+)\s+(-?\d*\.\d+)\s+(-?\d*\.\d+)\s+(-?\d*\.\d+)\s*$')
    rows = [tuple(float(v) for v in m.groups()[1:]) for line in txt.splitlines() if (m := pat.match(line))]
    return pd.DataFrame(rows, columns=['base', 'cap_base', 'a1', 'a2', 'a3']).drop_duplicates('base').sort_values('base')


def from_equations(eq, top):
    H = np.round(np.arange(eq.base.iloc[0], top + 1e-9, 0.1), 1)
    i = np.searchsorted(eq.base.values, H, side='right') - 1
    x = H - eq.base.values[i]
    V = eq.a1.values[i] + eq.a2.values[i] * x + eq.a3.values[i] * x * x
    A = eq.a2.values[i] + 2 * eq.a3.values[i] * x
    return pd.DataFrame(dict(elevation_ft=H, area_acres=A, capacity_af=V))


def trapezoid_check(df):
    d = df.sort_values('elevation_ft')
    dv = np.diff(d.capacity_af.values)
    trap = 0.5 * (d.area_acres.values[1:] + d.area_acres.values[:-1]) * np.diff(d.elevation_ft.values)
    ok = dv > 50
    rel = np.abs(trap[ok] / dv[ok] - 1)
    return dict(max_rel_mismatch=float(rel.max()), median_rel_mismatch=float(np.median(rel)),
                worst_interval_ft=[float(v) for v in d.elevation_ft.values[1:][ok][rel.argmax()] - np.array([np.diff(d.elevation_ft.values)[ok][rel.argmax()], 0])])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for key in ['T01', 'T02', 'T03', 'T04', 'T05', 'T06']:
        if key in TABLE2:
            src = TABLE2[key]
            df = pd.DataFrame(src['rows'], columns=['elevation_ft', 'area_acres', 'capacity_af']).sort_values('elevation_ft')
            info = dict(method='Table 2 survey columns, transcribed', page=src['page'], columns=src['columns'])
        else:
            src = EQUATIONS[key]
            eq = parse_equations((TEXT / src['text']).read_text(encoding='utf-8'))
            df = from_equations(eq, src['top_ft'])
            info = dict(method='ACAP capacity equations evaluated at 0.1 ft', page=src['page'], n_equations=len(eq),
                        equation_bases_ft=[float(eq.base.min()), float(eq.base.max())])
        check = trapezoid_check(df)
        pool, zone = POOL[key]
        V_pool = float(np.interp(pool, df.elevation_ft, df.capacity_af)); A_pool = float(np.interp(pool, df.elevation_ft, df.area_acres))
        df['area_km2'] = df.area_acres * ACRE_KM2; df['volume_km3'] = df.capacity_af * AF_KM3
        df.to_csv(OUT / f'{key}_official_curve_projectft.csv', index=False)
        pdf = REPORTS / src['report']
        manifest[key] = dict(info, report=src['report'], report_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                             pool_ft=pool, pool_zone=zone, area_at_pool_acres=A_pool, capacity_at_pool_af=V_pool,
                             elevation_range_ft=[float(df.elevation_ft.min()), float(df.elevation_ft.max())],
                             trapezoid_check=check, rows=len(df))
        print(key, {k: v for k, v in manifest[key].items() if k not in ('report_sha256',)}, flush=True)
    (OUT / 'official_curves_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
