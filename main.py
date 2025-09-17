import os
import time
import h5py
import pandas as pd
import requests
import xcsc_tushare as ts

# 环境变量读取 Token 与可选参数
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN")
TS_SERVER = os.environ.get("TS_SERVER", "http://116.128.206.39:7172")
TS_ENV = os.environ.get("TS_ENV", "prd")
START_DATE = os.environ.get("START_DATE", "20220101")
OUT_PATH = os.environ.get("OUT_PATH", "DB.h5")

if not TUSHARE_TOKEN:
    raise RuntimeError("Missing env TUSHARE_TOKEN")

# 初始化
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api(env=TS_ENV, server=TS_SERVER)
hist_fields = "trade_date,open,high,low,close,change,pct_chg,volume,amount"

def get_hist(ts_code: str):
    df = pro.daily(ts_code=ts_code, start_date=START_DATE, end_date="", fields=hist_fields)
    # 翻转时间顺序：从旧到新
    df = df.iloc[::-1].reset_index(drop=True)
    if len(df) > 21:
        return ts_code, df
    return None

def list_main_board_cs():
    temp0 = pro.stock_basic(market="CS")
    temp0 = temp0[temp0["delist_date"].isna()]
    temp1 = temp0[temp0["list_board_name"] == "主板"]
    return temp1[["ts_code", "name"]].reset_index(drop=True)

def main():
    ts_codes = list_main_board_cs()

    # 使用 gzip 压缩；注意：HDF5 压缩对数值类型友好
    with h5py.File(OUT_PATH, "w") as f:
        for x in ts_codes["ts_code"]:
            i = None
            retry = 0
            max_retry = 8

            while i is None and retry < max_retry:
                try:
                    i = get_hist(x)
                    if i is None:
                        print(f"{x} 请求返回空或数据太短，等待5秒重试")
                        time.sleep(5)
                        retry += 1
                except requests.exceptions.ConnectionError:
                    print(f"{x} 网络连接错误，3秒后重试")
                    time.sleep(3)
                    retry += 1
                    continue
                except Exception as e:
                    s = str(e)
                    if "抱歉,您每小时最多访问该接口4000次" in s or "最多访问该接口" in s:
                        print("达到请求限制，等待到下一个整点后重试")
                        now = time.time()
                        next_hour = now + (60 * 60 - now % (60 * 60))
                        time.sleep(max(1, next_hour - now + 3))
                    else:
                        print(f"{x} 出错: {e}，跳过该标的")
                        break

            if i is not None:
                ts_code, df = i
                # 保留 trade_date 作为第一列
                data = df.to_numpy()
                # 如果已存在同名数据集会报错，这里先删除再写入
                if ts_code in f.keys():
                    del f[ts_code]
                dset = f.create_dataset(
                    ts_code,
                    data=data,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True
                )
                # 写入列名作为属性，方便恢复为 DataFrame
                dset.attrs["columns"] = df.columns.to_list()
                print(f"写入: {ts_code}, 行数={len(df)}")

    print("RUN_DONE")

if __name__ == "__main__":
    main()
