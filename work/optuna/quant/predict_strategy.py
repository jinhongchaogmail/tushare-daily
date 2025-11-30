import pandas as pd
import catboost as cb
import json
import os
import glob
import sys
from datetime import datetime
from feature_engineering import apply_technical_indicators

# --- 配置 ---
# 假设脚本运行在 optuna/quant 目录下
# 数据在项目根目录的 data/ 下
DATA_DIR = '../../data' 
MODEL_PATH = '../model/catboost_final_model.cbm'
PARAMS_PATH = '../model/final_model_params.json'
MIN_RETURN_THRESHOLD = 0.03  # 最小预期收益门槛 (3%)

def load_parquet_data(file_path):
    """读取 main.py 生成的 parquet 文件并清洗"""
    try:
        df = pd.read_parquet(file_path)
        if df.empty:
            return None
            
        # 确保列名匹配 feature_engineering 的要求
        # main.py 的列: trade_date, open, high, low, close, volume, amount, ...
        # feature_engineering 需要: trade_date (或 date), open, high, low, close, volume
        
        # 重命名
        df = df.rename(columns={'trade_date': 'date'})
        
        # 确保按日期升序 (main.py 似乎已经是倒序或乱序，这里强制排序)
        df = df.sort_values('date').reset_index(drop=True)
        
        # 只保留原始列，重新计算特征以确保一致性
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return None
            
        return df[required_cols]
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    print("--- 启动每日预测策略 (基于本地 Parquet 数据) ---")
    
    # 1. 加载模型
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        # 尝试绝对路径 (调试用)
        print(f"Current working directory: {os.getcwd()}")
        return

    model = cb.CatBoostClassifier()
    model.load_model(MODEL_PATH)
    
    with open(PARAMS_PATH, 'r') as f:
        params = json.load(f)
        VOL_MULTIPLIER = params.get('vol_multiplier_best', 0.89)
        print(f"模型加载成功. Vol Multiplier: {VOL_MULTIPLIER:.4f}")

    # 2. 扫描数据文件
    parquet_files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
    if not parquet_files:
        print(f"Warning: No parquet files found in {DATA_DIR}")
        return
        
    print(f"找到 {len(parquet_files)} 个数据文件，开始分析...")
    
    report = []
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # 3. 遍历所有股票
    # 为了演示和速度，这里可以限制数量，或者全量跑
    # 在 GitHub Actions 中跑全量通常没问题，只要内存够
    for file_path in parquet_files:
        ts_code = os.path.basename(file_path).replace('.parquet', '')
        
        try:
            # 获取数据
            df = load_parquet_data(file_path)
            if df is None or len(df) < 60:
                continue
                
            # 特征工程
            df_features = apply_technical_indicators(df)
            
            # 取最后一行 (最新交易日)
            latest_row = df_features.iloc[[-1]].copy()
            current_date = latest_row['date'].values[0]
            
            # 简单的日期过滤：如果数据不是最近几天的，可能停牌或未更新，跳过
            # 这里暂不严格过滤，只在报告中显示日期
            
            # 推理
            prob = model.predict_proba(latest_row)[0]
            prob_down, prob_flat, prob_up = prob[0], prob[1], prob[2]
            
            # 策略逻辑
            current_vol = latest_row['volatility_factor'].values[0]
            if pd.isna(current_vol): current_vol = 0.02
            
            implied_return = current_vol * VOL_MULTIPLIER
            
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
                report.append({
                    '代码': ts_code,
                    '日期': pd.to_datetime(current_date).strftime('%Y-%m-%d'),
                    '信号': signal,
                    '上涨概率': f"{prob_up:.1%}",
                    '波动率': f"{current_vol:.1%}",
                    '预期收益': f"{implied_return:.1%}",
                    '建议仓位': f"{position:.1%}",
                    '理由': reason,
                    'prob_up_raw': prob_up
                })
            
        except Exception as e:
            # print(f"[{ts_code}] Error: {e}") # 减少日志噪音
            pass

    # 4. 生成报告
    if report:
        # 按上涨概率排序
        df_report = pd.DataFrame(report).sort_values('prob_up_raw', ascending=False).drop(columns=['prob_up_raw'])
        
        print(f"\n=== 每日策略报告 (Top 20 / {len(df_report)}) ===")
        print(df_report.head(20).to_markdown(index=False)) # 终端只打印前20
        
        # 保存为 Markdown
        with open("strategy_report.md", "w") as f:
            f.write(f"# 每日量化策略报告 ({today_str})\n\n")
            f.write(f"扫描股票数: {len(parquet_files)} | 入选机会: {len(df_report)}\n\n")
            f.write("### 🔴 重点关注 (Top 50)\n")
            f.write(df_report.head(50).to_markdown(index=False))
    else:
        print("今日无符合条件的交易机会。")

if __name__ == "__main__":
    main()
