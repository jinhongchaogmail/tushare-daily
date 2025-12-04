#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from daily_run import get_hist, TUSHARE_TOKEN, TS_SERVER, TS_ENV
from shared.features import apply_technical_indicators
import pandas as pd

STOCK = '000001.SZ'
OUT = 'tmp/test_single_stock.parquet'


def run():
    print(f"Testing end-to-end for {STOCK}")
    res = get_hist(STOCK)
    if not res:
        print('No data returned from get_hist')
        return 2
    code, df = res
    print('Raw columns:', df.columns.tolist())
    print('Rows:', len(df))

    df_feat = apply_technical_indicators(df.copy())
    print('Feature columns sample:', [c for c in df_feat.columns if c not in df.columns][:20])
    print(df_feat.tail(3))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df_feat.to_parquet(OUT, index=False)
    print('Saved to', OUT)
    return 0

if __name__ == '__main__':
    sys.exit(run())
