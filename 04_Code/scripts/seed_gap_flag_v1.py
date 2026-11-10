"""Part A of docs/plans/2026-11-04-seed-gap-plan.md: the seed gap G = H_seed - H_flat and the flag G >= 5 m for every
reservoir of the reconstruction products. Reads the locked summaries only; no product value changes.

Product lines (locked summaries):
  global       05_Results/by_date/2026-10-10/r2_global_v1/global_summary_R2.csv     (5,168)
  gdw_ext      05_Results/by_date/2026-10-27/gdw_ext_v1/ext_summary.csv             (3,563)
  high_lat     05_Results/by_date/2026-10-27/highlat_v1/hl_summary.csv              (213)
  remnant      05_Results/by_date/2026-10-27/remnant_fix_v1/fix_summary.csv         (16)
  small_tier   05_Results/by_date/2026-10-28/small_tier_v1/small_tier_summary.csv   (15,457; with quality class)
Output: 05_Results/by_date/2026-11-04/seed_gap_flag_v1/ (seed_gap_per_reservoir.csv.gz, seed_gap_summary.json)
Usage: .venv/Scripts/python.exe -I 04_Code/scripts/seed_gap_flag_v1.py
"""
from pathlib import Path
import sys, json, datetime, hashlib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parents[2]
R = ROOT / '05_Results/by_date'
OUT = R / '2026-11-04/seed_gap_flag_v1'
THRESHOLD_M = 5.0
LINES = {'global': ('2026-10-10/r2_global_v1/global_summary_R2.csv', 5168),
         'gdw_ext': ('2026-10-27/gdw_ext_v1/ext_summary.csv', 3563),
         'high_lat': ('2026-10-27/highlat_v1/hl_summary.csv', 213),
         'remnant': ('2026-10-27/remnant_fix_v1/fix_summary.csv', 16),
         'small_tier': ('2026-10-28/small_tier_v1/small_tier_summary.csv', 15457)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stats(d):
    G = d.seed_gap_m
    return dict(n=int(len(d)), G_median=float(G.median()), G_p75=float(G.quantile(.75)), G_p90=float(G.quantile(.9)),
                n_flag=int(d.flag_seed_gap.sum()), share_flag=float(d.flag_seed_gap.mean()), n_G_ge_10=int((G >= 10).sum()),
                n_flag_with_positive_dam_reading=int((d.flag_seed_gap & (d.D_dam_m > 0)).sum()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts, inputs = [], {}
    for line, (rel, n) in LINES.items():
        cols = ['Hylak_id', 'H_flat_m', 'H_seed_m', 'D_dam_m', 'R2_status'] + (['quality_class'] if line == 'small_tier' else [])
        d = pd.read_csv(R / rel, usecols=cols, low_memory=False)
        assert len(d) == n and d.Hylak_id.is_unique and np.isfinite(d.H_seed_m).all() and np.isfinite(d.H_flat_m).all(), line
        assert (d.H_seed_m >= d.H_flat_m - 1e-6).all(), line                    # H_seed = max(rule C, H_flat)
        d.insert(0, 'product_line', line)
        parts.append(d); inputs[line] = dict(file=rel, sha256=sha(R / rel))
    d = pd.concat(parts, ignore_index=True)
    dup = d.Hylak_id.duplicated(keep=False)
    d['seed_gap_m'] = (d.H_seed_m - d.H_flat_m).clip(lower=0.0)
    d['flag_seed_gap'] = d.seed_gap_m >= THRESHOLD_M
    d.to_csv(OUT / 'seed_gap_per_reservoir.csv.gz', index=False, float_format='%.4f')
    s = dict(plan='docs/plans/2026-11-04-seed-gap-plan.md (part A)', threshold_m=THRESHOLD_M, inputs=inputs,
             n_total=int(len(d)), n_hylak_in_more_than_one_line=int(dup.sum()),
             all=stats(d), by_line={k: stats(v) for k, v in d.groupby('product_line', sort=False)},
             small_tier_by_class={k: stats(v) for k, v in d[d.product_line == 'small_tier'].groupby('quality_class')},
             by_status={k: stats(v) for k, v in d.groupby('R2_status')},
             finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (OUT / 'seed_gap_summary.json').write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding='utf-8', newline='\n')
    print(json.dumps({k: s[k] for k in ('n_total', 'n_hylak_in_more_than_one_line', 'all', 'by_line', 'small_tier_by_class', 'by_status')}, indent=1))


if __name__ == '__main__':
    main()
