# Path changes

The scripts are copied unchanged except for the absolute paths of the machine they ran on:

| File | Line | Original | Replacement |
|---|---:|---|---|
| `04_Code/versions/Tao-original/FROZEN-20260921/project/04_Code/scripts/tao_compat.py` | 11 | `r'D:/00_Ningchuanwei/03_Research/01_Paper/10_References/Tao_source_code/scripts'` | `os.environ.get('GREAC_TAO_CODE', str(Path(__file__).resolve().parents[2] / '10_References/Tao_source_code/scripts'))` |
| `04_Code/versions/Tao-original/FROZEN-20260921/project/04_Code/scripts/tao_compat.py` | 8 | `(imports)` | `import os; from pathlib import Path` |
