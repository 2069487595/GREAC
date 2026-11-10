"""Option B of docs/plans/2026-11-08-hl11-small-plan.md (section 7, written before the run; user choice 2026-11-08):
the 4,469 HydroLAKES v1.1 small reservoirs without SWBD water are routed by the JRC GSW yearly classification of 2000
inside their polygon (classes 0 no data, 1 not water, 2 seasonal, 3 permanent):
  permanent >= 5 % of the valid cells -> small tier, W0 = permanent water of 2000 (highlat_v1.sample_gsw), otherwise as
                                        hl11_small_v1 (GDW polygon and river point, other GDW polygons, GLOBathy placeholder)
  no water at all in 2000            -> SRTM floor: the locked hl11 floor runs, v3 recommendation, flag_built_before_2000
  otherwise                          -> no product (water in February 2000 cannot be decided)
G1 is not applied (seed-gap plan, part B closed 2026-11-08).
Subcommands: route | run [--part i/n] | retry-edge | lock | widen [--part i/n] | widen-lock | tier-summary | floor | coverage
Outputs: 05_Results/by_date/2026-11-08/hl11_small_v1/b/ (tier/ with run/ and widen/; inputs in 03_Data_Processed/hl11_small_20261108/b_tier)
Usage: .venv/Scripts/python.exe -I 04_Code/scripts/hl11_small_b_v1.py <cmd> [...]
"""
from pathlib import Path
import sys, json, argparse, datetime, warnings, pickle, math

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import shapely
import hl11_small_v1 as h            # GLOBathy placeholder, GDW others, tier helpers
import highlat_v1 as hlat            # JRC 2000 sampling (its import redirects the frozen paths; redirected again below)

st, gr = h.st, h.gr
ROOT = h.ROOT
B = h.OUT / 'b'
BT = B / 'tier'
BDATA = h.DATA / 'b_tier'
TIER_BASE = h.TIER / 'gdw_part_with_tier.csv'
SRTM31 = ROOT / '05_Results/by_date/2026-11-08/srtm_bed_pre2000_v1/srtm_bed_summary_v3_1.csv'
PERM_MIN = 0.05
h.redirect(BT, BDATA)
st.GUARDED = st.GUARDED + [HERE / 'hl11_small_b_v1.py']


def classes(P):
    """Counts of the JRC 2000 classes at the native cell centres (0.00025 deg) inside polygon P (WGS84)."""
    s = 0.00025
    b = P.bounds
    lons = np.arange(math.floor(b[0] / s) * s + s / 2, b[2], s)
    lats = np.arange(math.floor(b[1] / s) * s + s / 2, b[3], s)
    LON, LAT = np.meshgrid(lons, lats)
    ins = shapely.contains_xy(P, LON, LAT)
    if not ins.any():
        c = P.representative_point(); LON, LAT, ins = np.array([[c.x]]), np.array([[c.y]]), np.array([[True]])
    k = hlat.gsw_class(LON[ins], LAT[ins])
    return {f'n{j}': int((k == j).sum()) for j in range(4)}


def cmd_route():
    t = pd.read_csv(h.OUT / 'floor' / 'target_list.csv')
    with open(h.DATA / 'polygons.pkl', 'rb') as fh:
        polys = pickle.load(fh)
    rows = []
    for i, r in enumerate(t.itertuples()):
        c = classes(shapely.make_valid(shapely.from_wkb(polys[int(r.Hylak_id)])))
        valid = c['n1'] + c['n2'] + c['n3']
        perm = c['n3'] / valid if valid else np.nan
        route = ('tier' if valid and perm >= PERM_MIN else 'floor' if valid and c['n2'] + c['n3'] == 0 else 'none')
        rows.append(dict(key=r.key, Hylak_id=int(r.Hylak_id), GDW_ID=int(r.GDW_ID), **c, perm_share=perm,
                         seasonal_share=c['n2'] / valid if valid else np.nan, route=route))
        if i % 500 == 0:
            print(i, flush=True)
    d = pd.DataFrame(rows)
    B.mkdir(parents=True, exist_ok=True)
    d.to_csv(B / 'route.csv', index=False, float_format='%.6g')
    lst = pd.read_csv(h.OUT / 'small_list.csv')
    tier = lst[lst.Hylak_id.isin(d.Hylak_id[d.route == 'tier'])]
    BT.mkdir(parents=True, exist_ok=True); BDATA.mkdir(parents=True, exist_ok=True)
    tier.to_csv(BT / 'small_list.csv', index=False, float_format='%.10g')
    with open(BDATA / 'polygons.pkl', 'wb') as fh:
        pickle.dump({int(i): polys[int(i)] for i in tier.Hylak_id}, fh)
    print('routes', d.route.value_counts().to_dict(), '| no valid JRC cells', int(((d.n1 + d.n2 + d.n3) == 0).sum()), flush=True)


def production():
    h.production()                      # reproduction check present and passed; other GDW polygons
    gr.sample = hlat.sample_gsw         # W0 = JRC 2000 permanent water


def cmd_tier_summary(a):
    st.coverage = lambda s: None
    st.cmd_summary(a)
    s = pd.read_csv(BT / 'small_tier_summary.csv')
    for c in h.GEST_COLS:
        if c in s.columns:
            s[c] = np.nan
    s['tier'] = 'small (< 1 km2), HydroLAKES v1.1 polygon from GDW, W0 from JRC 2000'
    s['w0_source'] = 'JRC GSW yearly classification 2000, permanent water'
    s['seed_gap_m'] = s.H_seed_m - s.H_flat_m
    s['flag_seed_gap'] = s.seed_gap_m >= 5
    s.to_csv(B / 'b_tier_summary.csv', index=False, float_format='%.6g')
    (BT / 'small_tier_summary.csv').unlink()
    print('B tier', len(s), '| classes', s.quality_class.value_counts().sort_index().to_dict(), '| seed gap >= 5 m', int(s.flag_seed_gap.sum()), flush=True)


def cmd_floor():
    r = pd.read_csv(B / 'route.csv')
    f = pd.read_csv(h.OUT / 'hl11_floor_summary.csv')
    f = f[f.Hylak_id.isin(r.Hylak_id[r.route == 'floor'])].copy()
    y = f.year_dam.where(f.year_dam > 0)
    f['flag_built_before_2000'] = (y < 2000).fillna(False)
    f['route'] = 'no JRC water in 2000'
    f.to_csv(B / 'b_floor_summary.csv', index=False, float_format='%.10g')
    print('B floor', len(f), '| recommended', int(f.recommended.sum()), '| built before 2000', int(f.flag_built_before_2000.sum()),
          '| recommended and before 2000', int((f.recommended & f.flag_built_before_2000).sum()), flush=True)


def cmd_coverage():
    p = pd.read_csv(TIER_BASE)
    lst = pd.read_csv(h.OUT / 'small_list.csv')
    ids = {int(k): [int(x) for x in str(v).split(';')] for k, v in zip(lst.Hylak_id, lst.GDW_IDs)}
    cap = p.CAP_MCM.clip(lower=0); A, V = p.AREA_POLY.sum(), cap.sum()

    def share(m):
        return dict(n=int(m.sum()), area_share=float(p.AREA_POLY[m].sum() / A), storage_share=float(cap[m].sum() / V))
    before = share(p.part.ne('none'))
    # SRTM floor v3.1: 51 reservoirs no longer recommended
    s = pd.read_csv(SRTM31)
    drop = s[s.recommended_v3 & ~s.recommended]
    gk = {int(k[1:]) for k in drop.key if k.startswith('g')}; hk = {int(k[1:]) for k in drop.key if k.startswith('h')}
    m = p.part.eq('srtm_bed') & (p.GDW_ID.isin(gk) | p.HYLAK_ID.isin(hk))
    p.loc[m, 'part'] = 'none'
    n_drop_records = int(m.sum())
    t1 = pd.read_csv(h.OUT / 'hl11_tier_summary.csv')
    t2 = pd.read_csv(B / 'b_tier_summary.csv')
    fb = pd.read_csv(B / 'b_floor_summary.csv'); fb = fb[fb.recommended]
    for name, hids in (('hl11_tier_swbd', t1.Hylak_id), ('hl11_tier_jrc2000', t2.Hylak_id), ('hl11_floor', fb.Hylak_id)):
        g = {x for hh in hids for x in ids[int(hh)]}
        assert not p.loc[p.GDW_ID.isin(g), 'part'].ne('none').any(), name
        p.loc[p.GDW_ID.isin(g), 'part'] = name
    after = share(p.part.ne('none'))
    n_res = dict(base_reservoirs=26072, srtm_floor_no_longer_recommended=int(len(drop)), hl11_tier_swbd=int(len(t1)),
                 hl11_tier_jrc2000=int(len(t2)), hl11_floor=int(len(fb)))
    n_res['total_reservoirs'] = 26072 - n_res['srtm_floor_no_longer_recommended'] + n_res['hl11_tier_swbd'] + n_res['hl11_tier_jrc2000'] + n_res['hl11_floor']
    out = dict(n_records=int(len(p)), before=before, srtm_floor_records_dropped=n_drop_records, after=after,
               no_product=share(p.part.eq('none')), reservoirs=n_res,
               by_part={k: share(p.part.eq(k)) for k in sorted(p.part.unique())},
               finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    p.to_csv(B / 'gdw_part_v1_0.csv', index=False)
    (B / 'coverage.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ['before', 'after', 'no_product', 'reservoirs', 'srtm_floor_records_dropped']}, indent=1), flush=True)


def main():
    warnings.filterwarnings('ignore')
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['route', 'run', 'retry-edge', 'lock', 'widen', 'widen-lock', 'tier-summary', 'floor', 'coverage'])
    p.add_argument('--part'); p.add_argument('--ids', nargs='*', type=int); p.add_argument('--k', type=int, default=st.K_PRODUCT)
    a = p.parse_args()
    if a.cmd in ('run', 'retry-edge'):
        production()
    {'route': cmd_route, 'run': lambda: st.cmd_run(a), 'retry-edge': lambda: st.cmd_retry_edge(a), 'lock': st.cmd_lock,
     'widen': lambda: st.cmd_widen(a), 'widen-lock': st.cmd_widen_lock, 'tier-summary': lambda: cmd_tier_summary(a),
     'floor': cmd_floor, 'coverage': cmd_coverage}[a.cmd]()


if __name__ == '__main__':
    main()
