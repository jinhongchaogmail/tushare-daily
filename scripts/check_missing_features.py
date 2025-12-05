import os
import sys
import xcsc_tushare as ts
import pandas as pd

# 从环境变量获取 Token
token = os.environ.get('TUSHARE_TOKEN')
if not token:
    print("错误: 未在环境变量中找到 TUSHARE_TOKEN。请确保 source secrets.sh 已执行。")
    sys.exit(1)

# 强制关闭代理
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
print("已强制清除代理设置。")

print(f"使用 Token: {token[:5]}******")
ts.set_token(token)

# 指定内网服务器地址
server_url = "http://tushare.xcsc.com:7172"
print(f"连接服务器: {server_url}")
pro = ts.pro_api(server=server_url)

# 测试参数
stock_code = '000001.SZ' # 平安银行
start_date = '20241120'
end_date = '20241128'

print(f"\n正在测试股票 {stock_code} ({start_date} - {end_date}) 的数据可用性...\n")

# 1. 测试 daily_basic (目标: free_turnover)
print("-" * 30)
print("1. 测试 daily_basic 接口 (检查 free_turnover)")
try:
    # 尝试只查一天，不指定 fields，看返回什么
    check_date = '20241128'
    print(f"尝试查询单日 {check_date} 的所有字段...")
    df_basic = pro.daily_basic(ts_code=stock_code, trade_date=check_date)
    
    if df_basic is not None and not df_basic.empty:
        print(f"成功获取 {len(df_basic)} 行数据")
        cols = df_basic.columns.tolist()
        print(f"返回字段列表: {cols}")
        
        target_col = 'free_turnover'
        if target_col in cols:
            print(f"✅ [可用] {target_col} 存在。")
        else:
            print(f"❌ [缺失] {target_col} 不存在。")
            if 'turnover_rate_f' in cols:
                print("提示: 发现 'turnover_rate_f'")
    else:
        print("❌ daily_basic 接口返回为空 (即使是单日查询)。")
except Exception as e:
    print(f"❌ 调用 daily_basic 发生异常: {e}")


# 2. 测试 moneyflow (目标: net_mf_amount)
print("\n" + "-" * 30)
print("2. 测试 moneyflow 接口 (检查 net_mf_amount)")
try:
    df_flow = pro.moneyflow(ts_code=stock_code, start_date=start_date, end_date=end_date)
    
    if df_flow is not None and not df_flow.empty:
        print(f"成功获取 {len(df_flow)} 行数据")
        cols = df_flow.columns.tolist()
        
        target_col = 'net_mf_amount'
        if target_col in cols:
            print(f"✅ [可用] {target_col} 存在于返回结果中。")
            print("数据预览:")
            print(df_flow[['trade_date', target_col]].head(3))
        else:
            print(f"❌ [缺失] {target_col} 未在返回结果中找到。")
            print(f"现有字段: {cols}")
            
            # 检查是否可以通过小单/中单/大单/超大单计算
            components = ['buy_sm_amount', 'sell_sm_amount', 'buy_md_amount', 'sell_md_amount', 
                          'buy_lg_amount', 'sell_lg_amount', 'buy_elg_amount', 'sell_elg_amount']
            available_components = [c for c in components if c in cols]
            
            if len(available_components) == 8:
                print("✅ [可计算] 所有分单资金字段齐全，可以手动计算净流入。")
            else:
                print(f"⚠️ 分单字段不全，仅找到: {available_components}")

    else:
        print("❌ moneyflow 接口返回为空 (可能是权限不足或该股票无数据)。")
except Exception as e:
    print(f"❌ 调用 moneyflow 发生异常: {e}")

print("\n" + "-" * 30)
print("测试结束")
