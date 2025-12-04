#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from daily_run import pro, START_DATE

CODES = ['000001.SZ','600000.SH','600004.SH','002496.SZ','002192.SZ']

for code in CODES:
    print('---', code)
    try:
        df = pro.daily(ts_code=code, start_date=START_DATE, end_date='')
        print('daily:', None if df is None else (len(df), df.columns.tolist()[:10]))
        if df is not None and len(df)>0:
            print(df.head(2))
    except Exception as e:
        print('daily error', e)
    try:
        df = pro.daily_basic(ts_code=code, start_date=START_DATE, end_date='')
        print('daily_basic:', None if df is None else (len(df), df.columns.tolist()[:10]))
        if df is not None and len(df)>0:
            print(df.head(2))
    except Exception as e:
        print('daily_basic error', e)
    try:
        df = pro.moneyflow(ts_code=code, start_date=START_DATE, end_date='')
        print('moneyflow:', None if df is None else (len(df), df.columns.tolist()[:10]))
        if df is not None and len(df)>0:
            print(df.head(2))
    except Exception as e:
        print('moneyflow error', e)
