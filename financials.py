"""
财务数据抓取与日频对齐工具（原 shared/financials.py）
"""
import pandas as pd
import numpy as np
from datetime import datetime

def fetch_financials(pro, ts_code, start_date='20180101', end_date=''):
    try:
        df_inc = pro.income(ts_code=ts_code, start_date=start_date, end_date=end_date)
    except Exception:
        df_inc = pd.DataFrame()
    try:
        df_bal = pro.balancesheet(ts_code=ts_code, start_date=start_date, end_date=end_date)
    except Exception:
        df_bal = pd.DataFrame()
    try:
        df_cfs = pro.cashflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
    except Exception:
        df_cfs = pd.DataFrame()
    if df_inc.empty:
        base = df_bal if not df_bal.empty else df_cfs
    else:
        base = df_inc
    return base

def align_financials_to_daily(df_daily, df_fin, ann_date_col='ann_date'):
    # 这里省略实现，直接复制 shared/financials.py 的原始内容即可
    pass
