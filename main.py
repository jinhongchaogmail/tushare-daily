import os
import time
import pandas as pd
import requests
import xcsc_tushare as ts
from datetime import datetime

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
    df = pro.daily(ts_code=ts_code, start_date=START_DATE, end_date="", fields=hist_fields)
    df = df.iloc[::-1].reset_index(drop=True)
    if len(df) > 21:  # 至少一个月的数据
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        return ts_code, df
    else:
        print(f"{ts_code} 数据不足（仅 {len(df)} 行），跳过")
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

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ts_codes = list_main_board_cs().head(10)
    skipped = []

    for x in ts_codes["ts_code"]:
        i = None
        retry = 0
        max_retry = 3  # 网络错误时最多重试 3 次

        while i is None and retry < max_retry:
            try:
                i = get_hist(x)
                if i is None:  # 数据不足直接跳过，不再重试
                    skipped.append(x)
                    break
            except requests.exceptions.ConnectionError:
                print(f"{x} 网络错误，3秒后重试")
                time.sleep(3)
                retry += 1
                continue
            except Exception as e:
                print(f"{x} 出错: {e}，跳过")
                skipped.append(x)
                break

        if i is not None:
            ts_code, df = i
            df = add_features(df)
            df = downcast(df)
            out_file = os.path.join(OUT_DIR, f"{ts_code}.parquet")
            df.to_parquet(out_file, engine="pyarrow", compression="zstd", compression_level=3, index=False)
            print(f"写入: {out_file}, 行数={len(df)}")

    if skipped:
        pd.DataFrame(skipped, columns=["ts_code"]).to_csv("skipped.csv", index=False)
        print(f"跳过 {len(skipped)} 个股票，已写入 skipped.csv")

    print("RUN_DONE")

if __name__ == "__main__":
    main()
