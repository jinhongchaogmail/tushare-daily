import os
import xcsc_tushare as ts

# 配置
token = os.environ.get("TUSHARE_TOKEN")
server = "http://116.128.206.39:7172"
ts.set_token(token)
pro = ts.pro_api(env="prd", server=server)

print("Testing daily_basic with single fields...")

fields_to_test = ['pe', 'pb', 'turnover_rate', 'total_mv', 'circ_mv']
for field in fields_to_test:
    try:
        df = pro.daily_basic(ts_code='000001.SZ', trade_date='20231201', fields=f'ts_code,trade_date,{field}')
        print(f"Field '{field}': ✅ OK")
    except Exception as e:
        print(f"Field '{field}': ❌ Failed ({e})")
