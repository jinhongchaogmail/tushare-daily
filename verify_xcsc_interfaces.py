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

# 辅助函数：获取最近的交易日范围
def get_recent_trading_dates():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10)
    return start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')

start_dt, end_dt = get_recent_trading_dates()
# 固定一个已知有数据的日期用于单日测试 (2023-12-01 是周五)
test_date = '20231201' 

results = {}

def check_interface(name, func, description):
    print(f"\nTesting {name} ({description})...")
    try:
        df = func()
        if not df.empty:
            print(f"  ✅ Success. Rows: {len(df)}")
            # 记录字段和类型
            dtypes = df.dtypes.astype(str).to_dict()
            results[name] = {
                "status": "Supported",
                "description": description,
                "columns": list(df.columns),
                "dtypes": dtypes,
                "sample": df.iloc[0].to_dict()
            }
        else:
            print(f"  ⚠️ Empty DataFrame returned.")
            results[name] = {"status": "Empty", "description": description}
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        results[name] = {"status": "Failed", "error": str(e), "description": description}

# --- 1. 基础数据 ---
check_interface("stock_basic", 
                lambda: pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date'), 
                "股票列表")

check_interface("trade_cal", 
                lambda: pro.trade_cal(exchange='', start_date=start_dt, end_date=end_dt), 
                "交易日历")

# --- 2. 行情数据 ---
check_interface("daily", 
                lambda: pro.daily(ts_code='000001.SZ', start_date=test_date, end_date=test_date), 
                "日线行情 (含复权)")

check_interface("daily_basic", 
                lambda: pro.daily_basic(ts_code='000001.SZ', trade_date=test_date), 
                "每日指标 (市值/估值/财务TTM)")

check_interface("adj_factor", 
                lambda: pro.adj_factor(ts_code='000001.SZ', trade_date=test_date), 
                "复权因子 (独立接口)")

check_interface("limit_list", 
                lambda: pro.limit_list(trade_date=test_date), 
                "涨跌停列表")

check_interface("moneyflow", 
                lambda: pro.moneyflow(ts_code='000001.SZ', trade_date=test_date), 
                "个股资金流向")

# --- 3. 财务数据 (尝试不指定字段，看默认返回) ---
check_interface("income", 
                lambda: pro.income(ts_code='000001.SZ', start_date='20230101', end_date='20231231'), 
                "利润表")

check_interface("balancesheet", 
                lambda: pro.balancesheet(ts_code='000001.SZ', start_date='20230101', end_date='20231231'), 
                "资产负债表")

check_interface("cashflow", 
                lambda: pro.cashflow(ts_code='000001.SZ', start_date='20230101', end_date='20231231'), 
                "现金流量表")

# --- 4. 生成报告 ---
import json
with open("xcsc_api_audit.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nAudit complete. Results saved to xcsc_api_audit.json")
