import os
import time
import pandas as pd
import requests
import xcsc_tushare as ts 

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
    df = pro.daily(ts_code=ts_code, start_date=START_DATE, end_date="", fields=hist_fields)
    df = df.iloc[::-1].reset_index(drop=True)
    if len(df) > 21:
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        return ts_code, df
    return None

def list_main_board_cs():
    temp0 = pro.stock_basic(market="CS")
    temp0 = temp0[temp0["delist_date"].isna()]
    temp1 = temp0[temp0["list_board_name"] == "主板"]
    return temp1[["ts_code", "name"]].reset_index(drop=True)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
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
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    return df

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ts_codes = list_main_board_cs()

    for x in ts_codes["ts_code"]:
        i = None
        retry = 0
        max_retry = 5

        while i is None and retry < max_retry:
            try:
                i = get_hist(x)
                if i is None:
                    print(f"{x} 数据太短，等待5秒重试")
                    time.sleep(5)
                    retry += 1
            except requests.exceptions.ConnectionError:
                print(f"{x} 网络错误，3秒后重试")
                time.sleep(3)
                retry += 1
                continue
            except Exception as e:
                print(f"{x} 出错: {e}，跳过")
                break

        if i is not None:
            ts_code, df = i
            df = add_features(df)
            df = downcast(df)
            out_file = os.path.join(OUT_DIR, f"{ts_code}.parquet")
            df.to_parquet(out_file, engine="pyarrow", compression="zstd", compression_level=3, index=False)
            print(f"写入: {out_file}, 行数={len(df)}")

    print("RUN_DONE")

if __name__ == "__main__":
    main()
