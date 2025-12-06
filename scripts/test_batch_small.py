#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from daily_run import list_main_board_cs, get_hist
from features_raw import apply_technical_indicators
import pandas as pd
import random

OUT_DIR = 'tmp/batch_test'
N = 20

os.makedirs(OUT_DIR, exist_ok=True)


def run():
    print(f'Running small batch test (N={N})')
    ts_list = list_main_board_cs()
    if ts_list is None or ts_list.empty:
        print('Failed to get stock list')
        return 2
    codes = ts_list['ts_code'].sample(min(N, len(ts_list))).tolist()
    print('Selected', len(codes), 'codes')

    for code in codes:
        try:
            res = get_hist(code)
            if not res:
                print(code, '-> no data')
                continue
            _, df = res
            df_feat = apply_technical_indicators(df.copy())
            out = os.path.join(OUT_DIR, f'{code}.parquet')
            df_feat.to_parquet(out, index=False)
            print(code, 'ok', df_feat.shape)
        except Exception as e:
            print(code, 'error', e)
    return 0

if __name__ == '__main__':
    sys.exit(run())
