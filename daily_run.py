import os
import sys
import time
import pandas as pd
import requests
import xcsc_tushare as ts
from datetime import datetime

# 添加 shared 到路径以便导入 feature_engineering
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shared'))
try:
    from features import apply_technical_indicators
    HAS_FEATURE_ENGINE = True
except ImportError as e:
    print(f"⚠️ 警告：无法导入特征工程 ({e})，将跳过预测功能", flush=True)
    HAS_FEATURE_ENGINE = False

# 尝试导入 CatBoost
try:
    import catboost as cb
    import json
    HAS_MODEL = True
except ImportError:
    print("⚠️ 警告：未安装 catboost，将跳过预测功能", flush=True)
    HAS_MODEL = False

TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN")
TS_SERVER = "http://116.128.206.39:7172"
TS_ENV = "prd"
START_DATE = "20220101"
OUT_DIR = "data"
MODEL_PATH = 'models/catboost_final_model.cbm'
PARAMS_PATH = 'models/final_model_params.json'
MIN_RETURN_THRESHOLD = 0.03

# 全局变量
model = None
vol_multiplier = 0.89
report = []
all_predictions = [] # 存储所有预测结果，用于强制输出
count_debug = 0

if not TUSHARE_TOKEN:
    raise RuntimeError("Missing env TUSHARE_TOKEN")

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api(env=TS_ENV, server=TS_SERVER)
hist_fields = "trade_date,open,high,low,close,change,pct_chg,volume,amount"

def init_model():
    """初始化预测模型"""
    global model, vol_multiplier
    
    if not HAS_MODEL or not HAS_FEATURE_ENGINE:
        return False
    
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ 未找到模型文件: {MODEL_PATH}，跳过预测功能", flush=True)
        return False

    try:
        model = cb.CatBoostClassifier()
        model.load_model(MODEL_PATH)
        
        if os.path.exists(PARAMS_PATH):
            with open(PARAMS_PATH, 'r') as f:
                params = json.load(f)
                vol_multiplier = params.get('vol_multiplier_best', 0.89)
        
        print(f"✅ 模型加载成功，波动率乘数: {vol_multiplier:.4f}", flush=True)
        return True
    except Exception as e:
        print(f"❌ 模型加载失败: {e}", flush=True)
        return False

def predict_stock(ts_code, df):
    """对单只股票进行预测 (多空双向)"""
    global report, count_debug, all_predictions
    
    if model is None or len(df) < 60:
        if len(df) < 60:
            print(f"  [{ts_code}] 数据不足 ({len(df)}行)，跳过预测", flush=True)
        return

    try:
        # 特征工程
        df_features = apply_technical_indicators(df)
        latest_row = df_features.iloc[[-1]].copy()
        current_date = latest_row['trade_date'].values[0]
        
        # 预测
        prob = model.predict_proba(latest_row)[0]
        prob_down, prob_flat, prob_up = prob[0], prob[1], prob[2]
        
        # 调试输出
        count_debug += 1
        # 如果总数少于 200 (调试模式)，则打印所有预测结果
        if len(report) < 200 or count_debug <= 5 or count_debug % 200 == 0 or prob_up > 0.25:
            print(f"  [{ts_code}] 预测: 跌{prob_down:.2f} 平{prob_flat:.2f} 涨{prob_up:.2f}", flush=True)

        # 策略逻辑
        current_vol = latest_row['volatility_factor'].values[0]
        if pd.isna(current_vol): 
            current_vol = 0.02
        
        implied_return = current_vol * vol_multiplier
        signal = "⚪ 观望"
        position = 0.0
        reason = ""
        is_candidate = False
        
        # 收集所有预测结果 (Debug用)
        all_predictions.append({
            '代码': ts_code,
            '日期': pd.to_datetime(current_date).strftime('%Y-%m-%d'),
            '信号': "Debug",
            '上涨概率': f"{prob_up:.1%}",
            '下跌概率': f"{prob_down:.1%}",
            '波动率': f"{current_vol:.1%}",
            '预期收益': f"{implied_return:.1%}",
            '建议仓位': "0.0%",
            '理由': "Debug记录",
            'prob_up_raw': prob_up,
            'prob_down_raw': prob_down,
            'max_prob': max(prob_up, prob_down)
        })
        
        # --- 阈值设置 (基于最新模型分析: Down 准, Up 保守) ---
        THRESHOLD_WATCH_UP = 0.28    # 降低做多门槛
        THRESHOLD_STRONG_UP = 0.38   # 强力做多门槛
        THRESHOLD_WATCH_DOWN = 0.35  # 做空门槛
        THRESHOLD_STRONG_DOWN = 0.45 # 强力做空门槛

        # --- 1. 做多信号 (Long) ---
        if prob_up > prob_down and prob_up > THRESHOLD_WATCH_UP:
            signal = "🔵 关注多"
            reason = f"看涨({prob_up:.1%})"
            is_candidate = True
            
            # 强力买入条件
            if prob_up > THRESHOLD_STRONG_UP and prob_up > prob_flat:
                if implied_return > MIN_RETURN_THRESHOLD:
                    signal = "🔴 强力做多"
                    position = min(1.0, 0.02 / (current_vol + 1e-5))
                    reason = f"高胜率({prob_up:.0%}) 高赔率(>{implied_return:.1%})"
                else:
                    signal = "🟠 潜伏做多" # 胜率高但波动率低
                    reason = f"高胜率({prob_up:.0%}) 低波动"

        # --- 2. 做空信号 (Short) ---
        elif prob_down > prob_up and prob_down > THRESHOLD_WATCH_DOWN:
            signal = "🟡 关注空"
            reason = f"看跌({prob_down:.1%})"
            is_candidate = True
            
            if prob_down > THRESHOLD_STRONG_DOWN and prob_down > prob_flat:
                signal = "🟢 强力做空"
                reason = f"高确信度({prob_down:.1%})"
                position = min(1.0, 0.02 / (current_vol + 1e-5))

        if is_candidate:
            item = {
                '代码': ts_code,
                '日期': pd.to_datetime(current_date).strftime('%Y-%m-%d'),
                '信号': signal,
                '上涨概率': f"{prob_up:.1%}",
                '下跌概率': f"{prob_down:.1%}",
                '波动率': f"{current_vol:.1%}",
                '预期收益': f"{implied_return:.1%}",
                '建议仓位': f"{position:.1%}",
                '理由': reason,
                'prob_up_raw': prob_up,
                'prob_down_raw': prob_down,
                'max_prob': max(prob_up, prob_down)
            }
            report.append(item)
            if "强力" in signal:
                print(f"  !!! 发现机会 [{ts_code}]: {signal} - {reason}", flush=True)

    except Exception as e:
        print(f"❌ [{ts_code}] 预测出错: {e}", flush=True)
        import traceback
        traceback.print_exc()

def generate_report():
    """生成预测报告 (分多空展示)"""
    today_str = datetime.now().strftime("%Y-%m-%d")
    report_path = "reports/strategy_report.md"
    
    if report:
        df_report = pd.DataFrame(report)
        
        # 分离多空
        df_long = df_report[df_report['信号'].str.contains('多')].sort_values('prob_up_raw', ascending=False)
        df_short = df_report[df_report['信号'].str.contains('空')].sort_values('prob_down_raw', ascending=False)
        
        # 移除原始排序列，保持表格整洁
        cols_to_drop = ['prob_up_raw', 'prob_down_raw', 'max_prob']
        df_long_display = df_long.drop(columns=cols_to_drop, errors='ignore')
        df_short_display = df_short.drop(columns=cols_to_drop, errors='ignore')
        
        print(f"\n=== 每日策略报告 (总计: {len(df_report)}) ===", flush=True)
        print(f"多头机会: {len(df_long)} | 空头机会: {len(df_short)}", flush=True)
        
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write(f"# 每日量化策略报告 ({today_str})\n\n")
            f.write(f"**总计入选**: {len(df_report)} (多头: {len(df_long)}, 空头: {len(df_short)})\n\n")
            
            f.write("## 🔴 多头机会 (Top 50)\n")
            if not df_long.empty:
                f.write(df_long_display.head(50).to_markdown(index=False))
            else:
                f.write("无符合条件的多头机会。\n")
            
            f.write("\n\n## 🟢 空头机会 (Top 50)\n")
            if not df_short.empty:
                f.write(df_short_display.head(50).to_markdown(index=False))
            else:
                f.write("无符合条件的空头机会。\n")
        
        csv_path = report_path.replace(".md", ".csv")
        # CSV 保留原始概率列，方便用户自己排序
        df_report.sort_values('max_prob', ascending=False).to_csv(csv_path, index=False)
        print(f"✅ 报告已保存: {report_path} 和 {csv_path}", flush=True)
    else:
        print("ℹ️ 今日无符合条件的交易机会", flush=True)
        
        # --- 强制输出 Top 10 (Debug) ---
        if all_predictions:
            print("⚠️ 强制输出 Top 10 预测结果 (即使未达阈值)", flush=True)
            df_all = pd.DataFrame(all_predictions)
            # 按最大概率排序
            df_top = df_all.sort_values('max_prob', ascending=False).head(10)
            
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w") as f:
                f.write(f"# 每日量化策略报告 ({today_str}) - DEBUG MODE\n\n")
                f.write("⚠️ **注意**: 今日无符合阈值的机会。以下为概率最高的 Top 10 股票 (仅供调试参考)。\n\n")
                f.write(df_top.drop(columns=['prob_up_raw', 'prob_down_raw', 'max_prob'], errors='ignore').to_markdown(index=False))
            
            print(f"✅ 强制报告已保存: {report_path}", flush=True)
        else:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w") as f:
                f.write(f"# 每日量化策略报告 ({today_str})\n\n")
                f.write("今日无符合条件的交易机会，且无任何有效预测数据。")

def get_hist(ts_code: str):
    """获取历史数据，若数据不足则返回 None"""
    try:
        df = pro.daily(ts_code=ts_code, start_date=START_DATE, end_date="", fields=hist_fields)
    except Exception as e:
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

def main():
    print("🚀 启动数据下载与预测脚本 (单线程模式)...", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 初始化模型（如果可用）
    model_enabled = init_model()
    if model_enabled:
        print("📊 预测功能已启用", flush=True)
    else:
        print("📊 预测功能未启用，仅下载数据", flush=True)
    
    print("📋 正在获取股票列表...", flush=True)
    try:
        ts_codes = list_main_board_cs()
    except Exception as e:
        print(f"❌ 获取股票列表失败: {e}", flush=True)
        return

    print(f"✅ 获取到 {len(ts_codes)} 只股票，开始处理...", flush=True)
    
    # --- 调试模式：限制处理数量 ---
    # 为了快速验证，仅处理前 100 只股票
    ts_codes = ts_codes.head(100)
    print(f"⚠️ 调试模式已开启：仅处理前 {len(ts_codes)} 只股票", flush=True)
    
    total = len(ts_codes)
    skipped = []
    
    for idx, row in ts_codes.iterrows():
        ts_code = row["ts_code"]
        
        # 进度打印
        if idx % 50 == 0:
            print(f"[{idx}/{total}] 正在处理: {ts_code}", flush=True)
            
        i = None
        retry = 0
        max_retry = 3

        while i is None and retry < max_retry:
            try:
                i = get_hist(ts_code)
                if i is None:
                    skipped.append(ts_code)
                    break
            except requests.exceptions.ConnectionError:
                print(f"⚠️ {ts_code} 网络错误，3秒后重试", flush=True)
                time.sleep(3)
                retry += 1
                continue
            except Exception as e:
                print(f"❌ {ts_code} 出错: {e}，跳过", flush=True)
                skipped.append(ts_code)
                break

        if i is not None:
            code, df = i
            try:
                # 预测（在保存前）
                if model_enabled and model is not None:
                    predict_stock(code, df.copy())
                
                # 添加特征并保存
                df = add_features(df)
                df = downcast(df)
                out_file = os.path.join(OUT_DIR, f"{code}.parquet")
                df.to_parquet(out_file, engine="pyarrow", compression="zstd", compression_level=3, index=False)
            except Exception as e:
                print(f"❌ {ts_code} 处理数据出错: {e}", flush=True)
                skipped.append(ts_code)

    if skipped:
        pd.DataFrame(skipped, columns=["ts_code"]).to_csv("skipped.csv", index=False)
        print(f"⚠️ 跳过 {len(skipped)} 个股票，已写入 skipped.csv", flush=True)

    # 生成预测报告
    if model_enabled and model is not None:
        generate_report()

    print("🎉 RUN_DONE: 所有任务完成", flush=True)

if __name__ == "__main__":
    main()
