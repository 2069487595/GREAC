"""Step 3: make sure every FABDEM 1-degree tile the global run can touch is extracted (local zips only, no download).

Tiles touched = every 1-degree cell within 0.3 degrees of a listed reservoir polygon (covers the 10 km W0 search
window plus the 15 km maximum margin). Missing tiles are extracted from 02_Data_Raw/FABDEM_V1-2_all/*.zip into
02_Data_Raw/FABDEM_tiles_extracted (the shared cache). Tiles absent from the archive (no land) are listed; the runner
treats them as FABDEM no-data. Writes 05_Results/by_date/2026-10-07/route1_global_v1/fabdem_tiles.csv.
Usage: .venv/Scripts/python.exe 04_Code/scripts/route1_global_tiles_v1.py
"""
from pathlib import Path
import math, zipfile, shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '05_Results/by_date/2026-10-07/route1_global_v1'
ARCH = ROOT / '02_Data_Raw/FABDEM_V1-2_all'
CACHE = ROOT / '02_Data_Raw/FABDEM_tiles_extracted'
PAD = 0.3


def tag(lat, lon):
    return f'{"N" if lat >= 0 else "S"}{abs(lat):02d}{"E" if lon >= 0 else "W"}{abs(lon):03d}'


def archive(lat, lon):
    la, lo = lat // 10 * 10, lon // 10 * 10
    return ARCH / f'{tag(la, lo)}-{tag(la + 10, lo + 10)}_FABDEM_V1-2.zip'


def main():
    d = pd.read_csv(OUT / 'global_list.csv')
    need = set()
    for r in d.itertuples():
        for la in range(math.floor(r.lat_min - PAD), math.floor(r.lat_max + PAD) + 1):
            for lo in range(math.floor(r.lon_min - PAD), math.floor(r.lon_max + PAD) + 1):
                need.add((la, lo))
    have = {p.name[:7] for p in CACHE.glob('*_FABDEM_V1-2.tif')}
    rows = []
    for la, lo in sorted(need):
        name = f'{tag(la, lo)}_FABDEM_V1-2.tif'
        if tag(la, lo) in have:
            rows.append(dict(tile=tag(la, lo), status='cached')); continue
        z = archive(la, lo)
        if not z.exists():
            rows.append(dict(tile=tag(la, lo), status='no archive (no land)')); continue
        with zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist() if Path(n).name == name]
            if not names:
                rows.append(dict(tile=tag(la, lo), status='not in archive (no land)')); continue
            tmp = CACHE / (name + '.part')
            with zf.open(names[0]) as s, tmp.open('wb') as o:
                shutil.copyfileobj(s, o, 8 * 1024 * 1024)
            tmp.replace(CACHE / name)
        rows.append(dict(tile=tag(la, lo), status='extracted'))
        print('extracted', name, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'fabdem_tiles.csv', index=False)
    print(t.status.value_counts().to_string())


if __name__ == '__main__':
    main()
