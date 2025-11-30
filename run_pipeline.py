import os
import time
import pandas as pd
import requests
import xcsc_tushare as ts
import catboost as cb
import json
import sys
from datetime import datetime

# 添加 optuna/quant 到路径以便导入 feature_engineering
sys.path.append(os.path.join(os.getcwd(), 'optuna/quant'))
try:
    from feature_engineering import apply_technical_indicators
except ImportError:
    print("Warning: Could not import feature_engineering. Make sure you are running from root.")

# --- 配置 ---
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN")
TS_SERVER = "http://116.128.206.39:7172"
TS_ENV = "prd"
START_DATE = "20220101"
OUT_DIR = "data"

# 模型路径
MODEL_PATH = 'optuna/model/catboost_final_model.cbm'
PARAMS_PATH = 'optuna/model/final_model_params.json'
MIN_RETURN_THRESHOLD = 0.03

if not TUSHARE_TOKEN:
    raise RuntimeError("Missing env TUSHARE_TOKEN")

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api(env=TS_ENV, server=TS_SERVER)
hist_fields = "trade_date,open,high,low,close,change,pct_chg,volume,amount"

# --- 全局变量 ---
model = None
vol_multiplier = 0.89
report = []

def init_model():
    """初始化模型"""
    global model, vol_multiplier
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return False

    try:
        model = cb.CatBoostClassifier()
        model.load_model(MODEL_PATH)
        
        if os.path.exists(PARAMS_PATH):
            with open(PARAMS_PATH, 'r') as f:
                params = json.load(f)
                vol_multiplier = params.get('vol_multiplier_best', 0.89)
        
        print(f"模型加载成功. Vol Multiplier: {vol_multiplier:.4f}")
        return True
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False

def process_stock_data(ts_code, df):
    """处理单只股票数据：特征工程 + 预测"""
    global report
    
    if model is None:
        return

    try:
        # 1. 数据清洗
        # 确保按日期升序
        if 'trade_date' in df.columns:
            df = df.sort_values('trade_date').reset_index(drop=True)
        
        # 2. 特征工程
        # 只需要最后几十行来计算指标，减少计算量
        # 但为了 MA20, MACD 等，至少需要 60 行以上
        if len(df) < 60:
            return

        df_features = apply_technical_indicators(df)
        
        # 3. 预测
        # 取最后一行 (最新交易日)
        latest_row = df_features.iloc[[-1]].copy()
        current_date = latest_row['trade_date'].values[0]
        
        # 推理
        prob = model.predict_proba(latest_row)[0]
        prob_down, prob_flat, prob_up = prob[0], prob[1], prob[2]
        
        # 策略逻辑
        current_vol = latest_row['volatility_factor'].values[0]
        if pd.isna(current_vol): current_vol = 0.02
        
        implied_return = current_vol * vol_multiplier
        
        signal = "⚪ 观望"
        position = 0.0
        reason = ""
        is_candidate = False
        
        # 宽松筛选：只要上涨概率大于下跌概率，且大于35%，就列入观察
        if prob_up > prob_down and prob_up > 0.35:
            signal = "🔵 关注"
            reason = f"看涨({prob_up:.1%})"
            is_candidate = True

        # 强买入信号
        if prob_up > 0.4 and prob_up > prob_down and prob_up > prob_flat:
            if implied_return > MIN_RETURN_THRESHOLD:
                signal = "🔴 买入"
                position = min(1.0, 0.02 / (current_vol + 1e-5))
                reason = f"高胜率({prob_up:.0%}) 高赔率(>{implied_return:.1%})"
                is_candidate = True
        
        if is_candidate:
            item = {
                '代码': ts_code,
                '日期': pd.to_datetime(current_date).strftime('%Y-%m-%d'),
                '信号': signal,
                '上涨概率': f"{prob_up:.1%}",
                '波动率': f"{current_vol:.1%}",
                '预期收益': f"{implied_return:.1%}",
                '建议仓位': f"{position:.1%}",
                '理由': reason,
                'prob_up_raw': prob_up
            }
            report.append(item)
            # 实时打印好机会
            if "买入" in signal:
                print(f"!!! 发现机会 [{ts_code}]: {reason}")

    except Exception as e:
        # print(f"Error processing {ts_code}: {e}")
        pass

def get_hist(ts_code: str):
    """获取历史数据，若数据不足则返回 None"""
    df = pro.daily(ts_code=ts_code, start_date=START_DATE, end_date="", fields=hist_fields)
    df = df.iloc[::-1].reset_index(drop=True)
    if len(df) > 21:  # 至少一个月的数据
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        return ts_code, df
    else:
        # print(f"{ts_code} 数据不足（仅 {len(df)} 行），跳过")
        return None

def list_main_board_cs():
    """获取主板已上市股票列表"""
    today = datetime.today().strftime("%Y%m%d")
    temp0 = pro.stock_basic(market="CS", fields="ts_code,name,list_date,delist_date,list_board_name")
    temp0 = temp0[temp0["delist_date"].isna()]  # 未退市
    temp0 = temp0[temp0["list_board_name"] == "主板"]  # 主板
    temp0 = temp0[temp0["list_date"] <= today]  # 已经上市
    return temp0[["ts_code", "name"]].reset_index(drop=True)

def add_features_simple(df: pd.DataFrame) -> pd.DataFrame:
    """
    main.py 原有的简单特征添加 (仅用于数据存储，不用于预测)
    预测使用的是 feature_engineering.py 中的复杂逻辑
    """
    # 为了保持数据文件的一致性，这里保留原有的简单特征计算
    # 但实际上如果只为了预测，可以不存这些，直接用原始数据
    # 这里为了兼容性，还是加上
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

def generate_report():
    """生成最终报告"""
    today_str = datetime.now().strftime("%Y-%m-%d")
    report_path = "optuna/quant/strategy_report.md"
    
    if report:
        # 按上涨概率排序
        df_report = pd.DataFrame(report).sort_values('prob_up_raw', ascending=False).drop(columns=['prob_up_raw'])
        
        print(f"\n=== 每日策略报告 (Top 20 / {len(df_report)}) ===")
        print(df_report.head(20).to_markdown(index=False))
        
        # 保存为 Markdown
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write(f"# 每日量化策略报告 ({today_str})\n\n")
            f.write(f"入选机会: {len(df_report)}\n\n")
            f.write("### 🔴 重点关注 (Top 50)\n")
            f.write(df_report.head(50).to_markdown(index=False))
        print(f"报告已保存至: {report_path}")
        
        # 保存为 CSV (用于邮件附件或下载)
        csv_path = report_path.replace(".md", ".csv")
        df_report.to_csv(csv_path, index=False)
        print(f"CSV 报告已保存至: {csv_path}")
    else:
        print("今日无符合条件的交易机会。")
        # 也要生成一个空报告，防止 Action 报错
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write(f"# 每日量化策略报告 ({today_str})\n\n")
            f.write("今日无符合条件的交易机会。")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. 初始化模型
    if not init_model():
        print("模型初始化失败，将只进行数据下载。")
    
    # 2. 获取股票列表
    ts_codes = list_main_board_cs()
    print(f"获取到 {len(ts_codes)} 只主板股票，开始处理...")
    
    skipped = []
    count = 0

    for x in ts_codes["ts_code"]:
        i = None
        retry = 0
        max_retry = 3

        while i is None and retry < max_retry:
            try:
                i = get_hist(x)
                if i is None:
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
            
            # --- 流式处理核心 ---
            # 1. 先进行预测 (使用原始数据或清洗后的数据)
            # 注意：process_stock_data 内部会调用 feature_engineering
            # 我们传入原始 df 的副本，以免影响后续存储逻辑
            if model is not None:
                process_stock_data(ts_code, df.copy())
            
            # 2. 数据存储逻辑 (保持原有)
            df = add_features_simple(df)
            df = downcast(df)
            out_file = os.path.join(OUT_DIR, f"{ts_code}.parquet")
            df.to_parquet(out_file, engine="pyarrow", compression="zstd", compression_level=3, index=False)
            
            count += 1
            if count % 100 == 0:
                print(f"已处理 {count} 只股票...")

    if skipped:
        pd.DataFrame(skipped, columns=["ts_code"]).to_csv("skipped.csv", index=False)
        print(f"跳过 {len(skipped)} 个股票")

    # 3. 生成报告
    if model is not None:
        generate_report()

    print("RUN_DONE")

if __name__ == "__main__":
    main()
