#!/usr/bin/env python3
"""
Normalize units in data/*.parquet

This script infers per-file conversion factors so that:
  amount_cny ~ volume_shares * close
and creates new columns:
  - volume_shares: volume converted to shares
  - amount_cny: amount in CNY (recomputed as volume_shares * close)
  - net_mf_amount_cny: net_mf_amount scaled to amount_cny units (if present)

Run in trial mode first (default N=20) to inspect scales before applying to all files.
"""
import sys
import os
from pathlib import Path
import math
import json
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
OUT_META = Path(__file__).resolve().parents[1] / 'data_unit_scales.json'

def infer_and_apply(p: Path):
    df = pd.read_parquet(p)
    # ensure numeric
    df = df.copy()
    valid = (df['amount'].notna()) & (df['volume'].notna()) & (df['close'].notna()) & (df['close']>0)
    if valid.sum() < 3:
        return None

    implied_shares = df.loc[valid, 'amount'] / df.loc[valid, 'close']
    reported_volume = df.loc[valid, 'volume']
    # scale = median(implied_shares / reported_volume)
    ratios = (implied_shares / (reported_volume + 1e-12)).replace([float('inf'), -float('inf')], float('nan')).dropna()
    if ratios.empty:
        return None
    scale = float(ratios.median())

    # sanity bounds
    if not (1e-6 < scale < 1e6):
        # suspicious scale: ignore
        return None

    # create normalized columns
    df['volume_shares'] = df['volume'] * scale
    df['amount_cny'] = df['volume_shares'] * df['close']

    # adjust moneyflow amount if present
    if 'net_mf_amount' in df.columns and df['net_mf_amount'].notna().sum() > 0:
        # compute factor so that median(amount_cny) matches median(amount)
        orig_med = df.loc[valid, 'amount'].median()
        new_med = df.loc[valid, 'amount_cny'].median()
        if orig_med and not math.isclose(orig_med, 0.0):
            amt_factor = new_med / orig_med
            df['net_mf_amount_cny'] = df['net_mf_amount'] * amt_factor
        else:
            df['net_mf_amount_cny'] = df['net_mf_amount']

    return {'path': str(p), 'scale': scale, 'rows': len(df)} , df

def run(trial_n=20, apply_changes=False):
    files = sorted(DATA_DIR.glob('*.parquet'))
    meta = {}
    to_process = files[:trial_n]
    print(f'Processing {len(to_process)} files (trial={trial_n}, apply_changes={apply_changes})')
    for p in to_process:
        res = infer_and_apply(p)
        if res is None:
            print(f'[SKIP] {p.name}: cannot infer scale or insufficient data')
            continue
        info, df_new = res
        meta[p.name] = {'scale': info['scale'], 'rows': info['rows']}
        print(f'[OK] {p.name}: scale={info["scale"]:.6g} rows={info["rows"]}')
        if apply_changes:
            # overwrite parquet with added columns
            df_new.to_parquet(p, index=False)

    # save meta file
    with open(OUT_META, 'w') as f:
        json.dump(meta, f, indent=2)
    print('Wrote meta to', OUT_META)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--trial', type=int, default=20, help='number of files to trial')
    ap.add_argument('--apply', action='store_true', help='apply changes (overwrite parquet)')
    args = ap.parse_args()
    run(trial_n=args.trial, apply_changes=args.apply)
#!/usr/bin/env python3
"""
Normalize units in data/*.parquet

For each parquet file, infer a scale factor so that amount/(volume*close) ~= 1
and create a new column `volume_shares` = volume * scale. Also keep original
`volume` and `amount` columns. Writes back to the same parquet file (overwrites)
with added metadata columns: `volume_scale_inferred`.

Usage: python3 scripts/normalize_data_units.py
"""
import os
import glob
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, 'data')

def infer_scale(df):
    # compute amount/(volume*close) for valid rows
    mask = (~df['volume'].isna()) & (~df['amount'].isna()) & (~df['close'].isna()) & (df['volume']>0) & (df['close']>0) & (df['amount']>0)
    if mask.sum() < 10:
        return None
    ratios = (df.loc[mask, 'amount'] / (df.loc[mask, 'volume'] * df.loc[mask, 'close'] + 1e-12)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(ratios) < 5:
        return None
    # use median to be robust against outliers
    med = float(ratios.median())
    # sanity bounds
    if not (1e-6 < med < 1e6):
        return None
    return med

def process_file(path):
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"SKIP read error: {path}: {e}")
        return False

    if 'volume' not in df.columns or 'amount' not in df.columns or 'close' not in df.columns:
        print(f"SKIP missing cols: {os.path.basename(path)}")
        return False

    scale = infer_scale(df)
    if scale is None:
        print(f"SKIP cannot infer scale: {os.path.basename(path)}")
        return False

    # create normalized columns
    df['volume_shares'] = df['volume'] * scale
    df['amount_yuan'] = df['amount']
    df['volume_scale_inferred'] = scale

    # downcast floats to save space
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    try:
        df.to_parquet(path, engine='pyarrow', compression='zstd', index=False)
        print(f"OK  normalized: {os.path.basename(path)}  scale={scale:.6f}")
        return True
    except Exception as e:
        print(f"ERROR write failed: {path}: {e}")
        return False

def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
    print(f"Found {len(files)} files in {DATA_DIR}")
    succeeded = 0
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Processing {os.path.basename(f)}", end=' ')
        ok = process_file(f)
        if ok:
            succeeded += 1

    print(f"Done. Succeeded: {succeeded}/{len(files)}")

if __name__ == '__main__':
    main()
