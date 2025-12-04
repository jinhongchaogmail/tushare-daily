import os
import xcsc_tushare as ts
import pandas as pd
from datetime import datetime, timedelta

# 配置
token = os.environ.get("TUSHARE_TOKEN")
server = "http://tushare.xcsc.com:7172"

print(f"Connecting to API server: {server}")
ts.set_token(token)
pro = ts.pro_api(env="prd", server=server)

print("\n=== Re-verifying 'Failed' or 'Empty' Interfaces ===")

# 1. Re-verify stock_basic without 'area' and 'industry'
print("\n1. Testing stock_basic (minimal fields)...")
try:
    # 去掉 area, industry，保留基础字段
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,list_date')
    if not df.empty:
        print(f"  ✅ Success! Rows: {len(df)}")
        print(f"  Columns: {df.columns.tolist()}")
    else:
        print("  ⚠️ Still empty.")
except Exception as e:
    print(f"  ❌ Failed: {e}")

# 2. Re-verify limit_list with multiple dates
print("\n2. Testing limit_list (checking last 10 days)...")
end_date = datetime.now()
found_limit_data = False
for i in range(10):
    d = end_date - timedelta(days=i)
    d_str = d.strftime('%Y%m%d')
    try:
        df = pro.limit_list(trade_date=d_str)
        if not df.empty:
            print(f"  ✅ Found data on {d_str}! Rows: {len(df)}")
            print(f"  Columns: {df.columns.tolist()}")
            found_limit_data = True
            break
    except Exception:
        pass

if not found_limit_data:
    print("  ⚠️ No data found in the last 10 days for limit_list.")

# 3. Re-verify daily_basic fields mapping
print("\n3. Verifying daily_basic fields mapping...")
try:
    df = pro.daily_basic(ts_code='000001.SZ', trade_date='20231201')
    if not df.empty:
        cols = df.columns.tolist()
        print(f"  Available columns: {cols}")
        
        # Check for common aliases
        mapping_check = {
            'turnover_rate': 'turn' in cols,
            'total_mv': 'tot_mv' in cols,
            'circ_mv': 'mv' in cols,
            'pb': 'pb_new' in cols
        }
        print(f"  Field Mapping Check: {mapping_check}")
except Exception as e:
    print(f"  ❌ Failed: {e}")
