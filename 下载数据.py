import os
import time
import pandas as pd
import requests
import xcsc_tushare as ts
from datetime import datetime
import concurrent.futures

TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN")
TS_SERVER = "http://116.128.206.39:7172"
TS_ENV = "prd"
START_DATE = "20220101"
OUT_DIR = "data"

if not TUSHARE_TOKEN:
    raise RuntimeError("Missing env TUSHARE_TOKEN")

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api(env=TS_ENV, server=TS_SERVER)
hist_fields = "trade_date,open,high,low,close,change,pct_chg,volume,amount"

def get_hist(ts_code: str):
    """获取历史数据，若数据不足则返回 None"""
    try:
        df = pro.daily(ts_code=ts_code, start_date=START_DATE, end_date="", fields=hist_fields)
    except Exception as e:
        # print(f"{ts_code} API请求失败: {e}", flush=True)
        raise e
        
    df = df.iloc[::-1].reset_index(drop=True)
    if len(df) > 21:  # 至少一个月的数据
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        return ts_code, df
    else:
        return None

def list_main_board_cs():
    """获取主板已上市股票列表"""
    today = datetime.today().strftime("%Y%m%d")
    temp0 = pro.stock_basic(market="CS", fields="ts_code,name,list_date,delist_date,list_board_name")
    temp0 = temp0[temp0["delist_date"].isna()]  # 未退市
    temp0 = temp0[temp0["list_board_name"] == "主板"]  # 主板
    temp0 = temp0[temp0["list_date"] <= today]  # 已经上市
    return temp0[["ts_code", "name"]].reset_index(drop=True)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加技术指标"""
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["volatility_10"] = df["close"].rolling(10).std()
    df["vol_ma5"] = df["volume"].rolling(5).mean()
    df["momentum_5"] = df["close"].pct_change(5)
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    return df

def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """降低浮点精度以节省空间"""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    return df

def process_one_stock(x):
    """处理单个股票的函数，用于多线程"""
    retry = 0
    max_retry = 3
    
    while retry < max_retry:
        try:
            res = get_hist(x)
            if res is None:
                return None # 数据不足
            
            ts_code, df = res
            df = add_features(df)
            df = downcast(df)
            out_file = os.path.join(OUT_DIR, f"{ts_code}.parquet")
            df.to_parquet(out_file, engine="pyarrow", compression="zstd", compression_level=3, index=False)
            return f"OK: {ts_code} ({len(df)} rows)"
            
        except requests.exceptions.ConnectionError:
            time.sleep(3)
            retry += 1
        except Exception as e:
            return f"ERR: {x} {e}"
            
    return f"FAIL: {x} Max retries"

def main():
    print("🚀 启动下载脚本...", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("📋 正在获取股票列表...", flush=True)
    try:
        ts_codes = list_main_board_cs()
    except Exception as e:
        print(f"❌ 获取股票列表失败: {e}", flush=True)
        return

    print(f"✅ 获取到 {len(ts_codes)} 只股票，开始并行下载...", flush=True)
    
    total = len(ts_codes)
    done_count = 0
    
    # 使用 ThreadPoolExecutor 并行下载
    # 注意：并发数不要太高，以免触发服务器限流
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_one_stock, row["ts_code"]): row["ts_code"] for _, row in ts_codes.iterrows()}
        
        for future in concurrent.futures.as_completed(futures):
            done_count += 1
            ts_code = futures[future]
            try:
                result = future.result()
                if result and result.startswith("OK"):
                    if done_count % 50 == 0: # 每50个打印一次进度，避免日志过多
                        print(f"[{done_count}/{total}] {result}", flush=True)
                elif result and (result.startswith("ERR") or result.startswith("FAIL")):
                    print(f"[{done_count}/{total}] {result}", flush=True)
            except Exception as exc:
                print(f"[{done_count}/{total}] 💥 {ts_code} generated an exception: {exc}", flush=True)

    print("🎉 RUN_DONE: 所有下载任务完成", flush=True)

if __name__ == "__main__":
    main()
