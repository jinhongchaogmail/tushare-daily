import os
import xcsc_tushare as ts
import pandas as pd

# 配置
token = os.environ.get("TUSHARE_TOKEN")
server = "http://tushare.xcsc.com:7172"

print(f"Connecting to API server: {server}")
ts.set_token(token)
pro = ts.pro_api(env="prd", server=server)

print("\n=== 1. Analyzing daily_basic available fields ===")
try:
    # 不指定 fields，获取所有默认字段
    df = pro.daily_basic(ts_code='000001.SZ', trade_date='20231201')
    if not df.empty:
        print(f"✅ daily_basic returned {len(df.columns)} columns:")
        print(df.columns.tolist())
        print("Sample data:")
        print(df.iloc[0].to_dict())
    else:
        print("⚠️ daily_basic returned empty data.")
except Exception as e:
    print(f"❌ daily_basic failed: {e}")

print("\n=== 2. Analyzing other interfaces ===")
interfaces = {
    'daily (行情)': lambda: pro.daily(ts_code='000001.SZ', start_date='20231201', end_date='20231201'),
    'income (利润)': lambda: pro.income(ts_code='000001.SZ', start_date='20230101', end_date='20231231', fields='ts_code,basic_eps,net_profit'),
    'limit_list (涨跌停)': lambda: pro.limit_list(trade_date='20231201'),
    'moneyflow (资金流)': lambda: pro.moneyflow(ts_code='000001.SZ', trade_date='20231201'),
    'stk_factor (技术因子)': lambda: pro.stk_factor(ts_code='000001.SZ', start_date='20231201', end_date='20231201') # 尝试一些特色接口
}

for name, func in interfaces.items():
    try:
        df = func()
        if not df.empty:
            print(f"Interface '{name}': ✅ OK - Columns: {df.columns.tolist()}")
        else:
            print(f"Interface '{name}': ⚠️ Empty Data")
    except Exception as e:
        print(f"Interface '{name}': ❌ Failed ({e})")
