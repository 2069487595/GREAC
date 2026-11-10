"""Global R2 run (spec docs/plans/2026-10-10-r2-global-run-spec.md). Production run of the frozen method, nothing is scored.

For every reservoir with a product in the locked global R1-C run (05_Results/by_date/2026-10-07/route1_global_v1):
  inputs   the locked inputs npz of that run (SHA-256 checked against its run.json); F and H_flat recomputed with the frozen
           function and checked against the locked record; H_seed and H_top read from the locked record
  R2       r2_dev_v1.predict with the frozen candidate R2e (r2_candidates_round1.json): dam height read from the DEM at the
           HydroLAKES pour point, recession without sediment correction, depth below H_seed rescaled to the reading;
           when the reading is undefined or not positive, R2 = the locked R1-C raster (nothing is recomputed)
  recheck  for recomputed reservoirs the R1-C raster is recomputed with the same initialization and code path and compared
           with the locked R1-C raster (maximum absolute difference recorded); a consistency check, not a selection
  curves   route1_global_run_v1.curves, 0.25 m levels up to the locked H_top
Outputs in OUT/run/<Hylak_id>/: <id>_bed_R2.tif and <id>_aev_R2.csv (not for fallbacks), <id>_R2_run.json; failures
<id>_R2_error.json. Memory errors are not failures (printed as RESOURCE, retried in a later single-process pass).
Usage: r2_global_run_v1.py [--size lo:hi] [--part i/n] [--ids ...] [--no-recheck]
       r2_global_run_v1.py --trial --ids ...      (script test: no git check, folder with the suffix _trial)
       r2_global_run_v1.py --lock
"""
from pathlib import Path
import sys, json, argparse, datetime, warnings, traceback, time, subprocess, collections

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import r2_dev_v1 as d
import route1_global_run_v1 as gr

ROOT = d.ROOT
SPEC_DOC = ROOT / 'docs/plans/2026-10-10-r2-global-run-spec.md'
OUT = ROOT / '05_Results/by_date/2026-10-10/r2_global_v1'
SUMMARY = gr.OUT / 'global_summary.csv'
SPEC = json.loads((HERE / 'r2_candidates_round1.json').read_text(encoding='utf-8'))['candidates']['R2e']
K0 = [0.0, -1.0, 0.0, 0.0]      # frozen R1-C settings of the development harness: frozen anchors, eq. 7 exponent
GUARDED = [HERE / 'r2_global_run_v1.py', HERE / 'r2_dev_v1.py', HERE / 'r2_candidates_round1.json', HERE / 'route1_global_run_v1.py', SPEC_DOC]
Row = collections.namedtuple('Row', 'key')
sha = gr.sha


def committed():
    for f in GUARDED:
        rel = str(f.relative_to(ROOT)).replace(chr(92), '/')
        if subprocess.run(['git', 'ls-files', '--error-unmatch', rel], cwd=ROOT, capture_output=True).returncode != 0:
            raise SystemExit(f'{rel} is not tracked by git')
        if subprocess.run(['git', 'status', '--porcelain', '--', rel], cwd=ROOT, capture_output=True, text=True).stdout.strip():
            raise SystemExit(f'{rel} differs from the committed version')
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def reading(row, g, F, H_flat):
    """The dam-height reading, with the same calls as r2_dev_v1.predict (used here only to decide whether the
    initialization is needed; the value used by the method is the one returned by predict and is checked against this)."""
    R, q = float(SPEC['dam'][0]), float(SPEC['dam'][1])
    b1 = d.dam_depth(row, g, F, H_flat, R, q, True)
    if not np.isfinite(b1):
        return np.nan, np.nan, np.nan
    b2 = d.dam_depth(row, g, F, H_flat, 2 * R, q, True)
    return b1, b2, b1 - (max(0.0, b2 - b1) if np.isfinite(b2) else 0.0)


def run_one(r, out, head, recheck=True):
    hid = int(r.Hylak_id); key = str(hid); t0 = time.time()
    folder = out / key; folder.mkdir(parents=True, exist_ok=True)
    src = gr.RUN / key
    rec0 = json.loads((src / f'{hid}_run.json').read_text(encoding='utf-8'))
    ish = sha(gr.INPUTS / f'{hid}_inputs.npz')
    if ish != rec0['inputs_sha256']:
        raise RuntimeError('inputs differ from the locked global run')
    g = gr.load(hid); g.pop('occ', None); g.pop('P', None)
    epsg = int(g['meta']['epsg']); swo1, W0 = g['swo1'], g['W0']
    F, H_flat, _ = d.r1.flattened_region(g['dem'], W0, swo1)
    if abs(H_flat - rec0['H_flat_m']) > 1e-6 or abs(float(F.sum() * 9e-4) - rec0['F_km2']) > 1e-6:
        raise RuntimeError('F or H_flat differ from the locked global run')
    H_seed, H_top = float(rec0['H_seed_m']), float(rec0['H_top_m'])
    d.POUR[key] = (float(r.Pour_long), float(r.Pour_lat)); row = Row(key)
    locked = src / f'{hid}_C_tao_F.tif'
    b1, b2, D = reading(row, g, F, H_flat)
    positive = bool(np.isfinite(D) and D > 0)
    rec = dict(Hylak_id=hid, git_head=head, spec=SPEC, inputs_sha256=ish, cells=int(F.size), H_flat_m=float(H_flat), H_seed_m=H_seed, H_top_m=H_top,
               F_km2=float(F.sum() * 9e-4), B_R_m=float(b1), B_2R_m=float(b2), D_dam_m=float(D))
    cache, t_init = None, 0.0
    if positive:
        t1 = time.time(); init = d.ga.initialize(g['dem'], F)
        cache = (init['dem'].astype(float), init['known'].astype(bool), init['slope'], float(init['alpha_rule'])); del init
        t_init = time.time() - t1; rec['alpha_rule_R1C'] = cache[3]
    t1 = time.time()
    pred, extra = d.predict(row, g, F, H_flat, H_seed, cache, locked, SPEC)
    t_pred = time.time() - t1
    Dp = float(extra['D_dam_m'])
    if not ((np.isnan(Dp) and np.isnan(D)) or Dp == D):
        raise RuntimeError(f'reading differs between the pre-check and predict: {D} vs {Dp}')
    rescaled = bool(extra.get('rescaled', False))
    status = 'rescaled' if rescaled else 'fallback_undefined' if not np.isfinite(D) else 'fallback_nonpositive' if D <= 0 else 'not_rescaled'
    rec.update(status=status, rescaled=rescaled, R2_equals_R1C=not positive)
    if positive:
        z = g['dem'].copy(); z[F] = pred[F]
        tab = gr.curves(z, F, g, F, H_top); last = tab.iloc[-1]
        rec['R2'] = dict(V_top_km3=float(last.V_swo1), A_top_km2=float(last.A_swo1),
                         max_depth_m=float(H_top - np.nanmin(z[F & swo1])) if (F & swo1).any() else None,
                         max_depth_below_seed_m=float(np.nanmax(H_seed - pred[F])), target_max_depth_m=float(D + (H_seed - H_flat)))
        gr.write_crop(folder / f'{hid}_bed_R2.tif', z, swo1 | W0 | F, g, epsg)
        pd.DataFrame({'H_m': tab.H, 'A_R2_km2': tab.A_swo1, 'V_R2_km3': tab.V_swo1}).to_csv(folder / f'{hid}_aev_R2.csv', index=False, float_format='%.6g')
        del z, tab, pred
        if recheck:
            t1 = time.time()
            k0 = d.run_candidate(g, F, H_seed, cache, K0); ref = d.v.reg.load_m0(locked, g)
            rec['r1c_recheck_max_abs_diff_m'] = float(np.nanmax(np.abs(k0[F] - ref[F]))); rec['r1c_recheck_s'] = round(time.time() - t1, 1)
    else:
        m = rec0['methods']['R1C']
        rec['R2'] = dict(V_top_km3=m['V_top_km3'], A_top_km2=m['A_top_km2'], max_depth_m=m['max_depth_m'], source='locked R1-C record')
    rec.update(runtime_s=dict(init=round(t_init, 1), predict=round(t_pred, 1), total=round(time.time() - t0, 1)),
               finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (folder / f'{hid}_R2_run.json').write_text(json.dumps(rec, indent=1, default=float), encoding='utf-8')
    return rec


def table(size='all'):
    s = pd.read_csv(SUMMARY)[['Hylak_id', 'Pour_long', 'Pour_lat']]
    order = {h: i for i, h in enumerate(gr.jobs(size))}       # ascending estimated grid size
    s = s[s.Hylak_id.isin(order)].copy()
    return s.assign(_o=s.Hylak_id.map(order)).sort_values('_o').drop(columns='_o').reset_index(drop=True)


def lock():
    head = committed(); out = OUT / 'run'; t = table()
    runs, failed, missing, status = {}, {}, [], collections.Counter()
    for hid in t.Hylak_id:
        f = out / str(hid)
        if (f / f'{hid}_R2_run.json').exists():
            rec = json.loads((f / f'{hid}_R2_run.json').read_text(encoding='utf-8')); status[rec['status']] += 1
            runs[int(hid)] = dict(status=rec['status'], inputs_sha256=rec['inputs_sha256'], outputs={p.name: sha(p) for p in sorted(f.glob('*')) if p.suffix in ('.tif', '.csv', '.json')})
        elif (f / f'{hid}_R2_error.json').exists():
            failed[int(hid)] = json.loads((f / f'{hid}_R2_error.json').read_text(encoding='utf-8'))['error'][:300]
        else:
            missing.append(int(hid))
    m = dict(spec=str(SPEC_DOC.relative_to(ROOT)).replace(chr(92), '/'), spec_sha256=sha(SPEC_DOC), git_head=head, method=SPEC,
             scripts={p.name: sha(p) for p in GUARDED if p.suffix in ('.py', '.json')}, r1c_lock_manifest_sha256=sha(gr.OUT / 'lock_manifest.json'),
             n_listed=int(len(t)), n_ok=len(runs), n_failed=len(failed), status=dict(status), missing=missing, failed=failed, runs=runs,
             locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (OUT / 'lock_manifest.json').write_text(json.dumps(m, indent=0), encoding='utf-8')
    print('LOCKED ok', len(runs), dict(status), 'failed', len(failed), 'missing', len(missing), flush=True)


def main():
    warnings.filterwarnings('ignore')
    p = argparse.ArgumentParser(); p.add_argument('--size', default='all'); p.add_argument('--part'); p.add_argument('--ids', nargs='*', type=int)
    p.add_argument('--trial', action='store_true'); p.add_argument('--no-recheck', action='store_true'); p.add_argument('--lock', action='store_true')
    a = p.parse_args()
    if a.lock:
        return lock()
    head = 'trial' if a.trial else committed()
    out = (OUT.parent / (OUT.name + '_trial') if a.trial else OUT) / 'run'
    t = table(a.size)
    if a.ids:
        t = t[t.Hylak_id.isin(a.ids)]
    if a.part:
        i, n = map(int, a.part.split('/')); t = t.iloc[i - 1::n]
    out.mkdir(parents=True, exist_ok=True)
    for r in t.itertuples():
        hid = int(r.Hylak_id); f = out / str(hid)
        if (f / f'{hid}_R2_run.json').exists() or (f / f'{hid}_R2_error.json').exists():
            continue
        t0 = time.time()
        try:
            rec = run_one(r, out, head, recheck=not a.no_recheck)
            print(hid, rec['status'], f"D {rec['D_dam_m']:.1f}", f"F {rec['F_km2']:.1f} km2", f"V_R2 {rec['R2']['V_top_km3']:.4g}",
                  f"recheck {rec.get('r1c_recheck_max_abs_diff_m', float('nan')):.2g}", f"init {rec['runtime_s']['init']:.0f}s total {rec['runtime_s']['total']:.0f}s", flush=True)
        except Exception as exc:
            if isinstance(exc, MemoryError) or 'allocat' in repr(exc):      # resource, not method: retried in a later pass
                print(hid, 'RESOURCE', repr(exc)[:120], f'{time.time() - t0:.0f}s', flush=True)
                continue
            f.mkdir(parents=True, exist_ok=True)
            (f / f'{hid}_R2_error.json').write_text(json.dumps(dict(error=repr(exc), trace=traceback.format_exc())), encoding='utf-8')
            print(hid, 'FAILED', repr(exc)[:200], f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
