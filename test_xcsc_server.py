import os
import xcsc_tushare as ts

# 配置
token = os.environ.get("TUSHARE_TOKEN")
# 更新为用户提供的新 Server 地址
server = "http://tushare.xcsc.com:7172"

print(f"Connecting to new server: {server}")
ts.set_token(token)
pro = ts.pro_api(env="prd", server=server)

print("\n=== 1. Testing daily_basic fields (Market Value, PE, PB) ===")
fields_to_test = ['pe', 'pb', 'turnover_rate', 'total_mv', 'circ_mv']
for field in fields_to_test:
    try:
        # 使用最近的交易日
        df = pro.daily_basic(ts_code='000001.SZ', trade_date='20231201', fields=f'ts_code,trade_date,{field}')
        if not df.empty:
            print(f"Field '{field}': ✅ OK - Value: {df[field].iloc[0]}")
        else:
            print(f"Field '{field}': ⚠️ Empty Data")
    except Exception as e:
        print(f"Field '{field}': ❌ Failed ({e})")

print("\n=== 2. Testing other key interfaces ===")
interfaces = {
    'daily (行情)': lambda: pro.daily(ts_code='000001.SZ', start_date='20231201', end_date='20231201'),
    'income (利润)': lambda: pro.income(ts_code='000001.SZ', start_date='20230101', end_date='20231231', fields='ts_code,basic_eps'),
    'limit_list (涨跌停)': lambda: pro.limit_list(trade_date='20231201'),
    'moneyflow (资金流)': lambda: pro.moneyflow(ts_code='000001.SZ', trade_date='20231201')
}

for name, func in interfaces.items():
    try:
        df = func()
        if not df.empty:
            print(f"Interface '{name}': ✅ OK ({len(df)} rows)")
        else:
            print(f"Interface '{name}': ⚠️ Empty Data")
    except Exception as e:
        print(f"Interface '{name}': ❌ Failed ({e})")
