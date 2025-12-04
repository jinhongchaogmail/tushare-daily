#!/usr/bin/env python3
"""
Fix amount column aliases in data parquet files.

If a file has `amount_yuan` but not `amount_cny`, create `amount_cny` as a copy
of `amount_yuan` and overwrite the parquet. Records modified files to
`data/amount_alias_migration.json`.
"""
import json
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
OUT_META = DATA_DIR / 'amount_alias_migration.json'

def process_file(p: Path):
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        return False, f'read_error: {e}'

    if 'amount_cny' in df.columns:
        return False, 'already_has_amount_cny'

    if 'amount_yuan' in df.columns:
        df['amount_cny'] = df['amount_yuan']
        try:
            df.to_parquet(p, index=False)
            return True, 'fixed'
        except Exception as e:
            return False, f'write_error: {e}'

    return False, 'no_amount_yuan'

def main():
    files = sorted(DATA_DIR.glob('*.parquet'))
    modified = {}
    for p in files:
        ok, msg = process_file(p)
        if ok:
            modified[p.name] = msg
            print(f'OK {p.name}: {msg}')
        else:
            if msg != 'already_has_amount_cny':
                # Only print non-trivial skips
                print(f'SKIP {p.name}: {msg}')

    with open(OUT_META, 'w') as f:
        json.dump({'modified': modified}, f, indent=2)

    print(f'Done. Modified {len(modified)} files. Log: {OUT_META}')

if __name__ == '__main__':
    main()
