#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from daily_run import list_main_board_cs, get_hist

if __name__ == '__main__':
    df = list_main_board_cs()
    print('Total stocks:', len(df))
    print(df.head(10).to_string(index=False))
    sample = df['ts_code'].tolist()[:10]
    print('\nTesting first 10 codes:')
    for code in sample:
        try:
            res = get_hist(code)
            print(code, '->', 'has data' if res else 'no data')
        except Exception as e:
            print(code, 'error', e)
