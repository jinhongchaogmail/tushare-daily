import os
import xcsc_tushare as ts
import pandas as pd

# 配置
token = os.environ.get("TUSHARE_TOKEN")
server = "http://116.128.206.39:7172"

if not token:
    print("Error: TUSHARE_TOKEN not found.")
    exit(1)

print(f"Testing connection to {server}...")
ts.set_token(token)
pro = ts.pro_api(env="prd", server=server)

# 测试 1: daily_basic (每日指标 - 市值、PE、PB)
try:
    print("\nTesting daily_basic (每日指标)...")
    # 使用一个较新的日期，但不要是周末
    df = pro.daily_basic(ts_code='000001.SZ', trade_date='20231201', fields='ts_code,trade_date,pe,pb,turnover_rate,total_mv')
    if not df.empty:
        print("✅ daily_basic success:")
        print(df.head(1).to_string(index=False))
    else:
        print("⚠️ daily_basic returned empty data.")
except Exception as e:
    print(f"❌ daily_basic failed: {e}")

# 测试 2: income (利润表 - 财务数据)
try:
    print("\nTesting income (财务数据)...")
    df = pro.income(ts_code='000001.SZ', start_date='20230101', end_date='20231231', fields='ts_code,end_date,basic_eps,total_revenue')
    if not df.empty:
        print("✅ income success:")
        print(df.head(1).to_string(index=False))
    else:
        print("⚠️ income returned empty data.")
except Exception as e:
    print(f"❌ income failed: {e}")

# 测试 3: limit_list (涨跌停)
try:
    print("\nTesting limit_list (涨跌停)...")
    df = pro.limit_list(trade_date='20231201')
    if not df.empty:
        print("✅ limit_list success:")
        print(df.head(1).to_string(index=False))
    else:
        print("⚠️ limit_list returned empty data.")
except Exception as e:
    print(f"❌ limit_list failed: {e}")
