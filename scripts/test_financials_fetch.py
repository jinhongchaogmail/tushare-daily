#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from daily_run import get_hist, pro
from shared.financials import fetch_financials, align_financials_to_daily
from shared.features import apply_technical_indicators

CODE = '000001.SZ'

if __name__ == '__main__':
    res = get_hist(CODE)
    if not res:
        print('no market data')
        sys.exit(2)
    code, df_daily = res
    print('daily rows', len(df_daily))

    df_fin = fetch_financials(pro, CODE, start_date='20180101')
    print('fin rows', len(df_fin))
    if not df_fin.empty:
        print(df_fin.head())

    aligned = align_financials_to_daily(df_daily, df_fin)
    print('aligned shape', aligned.shape)
    print(aligned.tail())

    # merge into features
    df_merge = df_daily.reset_index(drop=True).merge(aligned.reset_index(drop=True), left_index=True, right_index=True, how='left')
    df_feat = apply_technical_indicators(df_merge)
    print('final shape', df_feat.shape)
    print(df_feat[['trade_date'] + [c for c in df_feat.columns if c in ['net_profit','total_revenue']]].tail())
