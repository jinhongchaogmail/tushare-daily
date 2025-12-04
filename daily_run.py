import os
import sys
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
MODEL_PATH = 'models/catboost_final_model.cbm'
PARAMS_PATH = 'models/final_model_params.json'
MIN_RETURN_THRESHOLD = 0.03

# 动态导入特征工程逻辑 (优先使用模型绑定的 frozen_features.py)
# 这样可以保证预测时使用的特征计算逻辑与模型训练时完全一致，
# 即使 shared/features.py 已经更新或修改。
HAS_FEATURE_ENGINE = False
apply_technical_indicators = None

def load_feature_engineering():
    global apply_technical_indicators, HAS_FEATURE_ENGINE
    
    # 1. 尝试加载模型目录下的 frozen_features.py (模型伴生代码)
    frozen_features_path = os.path.join(os.path.dirname(MODEL_PATH), 'frozen_features.py')
    if os.path.exists(frozen_features_path):
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("frozen_features", frozen_features_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["frozen_features"] = module
            spec.loader.exec_module(module)
            apply_technical_indicators = module.apply_technical_indicators
            HAS_FEATURE_ENGINE = True
            print(f"✅ 已加载模型伴生特征代码: {frozen_features_path}", flush=True)
            return
        except Exception as e:
            print(f"⚠️ 加载 frozen_features.py 失败: {e}，将回退到 shared/features.py", flush=True)

    # 2. 回退到项目默认的 shared/features.py
    try:
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shared'))
        from features import apply_technical_indicators as shared_ati
        apply_technical_indicators = shared_ati
        HAS_FEATURE_ENGINE = True
        print("✅ 已加载默认特征代码: shared/features.py", flush=True)
    except ImportError as e:
        print(f"⚠️ 警告：无法导入特征工程 ({e})，将跳过预测功能", flush=True)
        HAS_FEATURE_ENGINE = False

load_feature_engineering()

# 尝试导入 CatBoost
try:
    import catboost as cb
    import json
    HAS_MODEL = True
except ImportError:
    print("⚠️ 警告：未安装 catboost，将跳过预测功能", flush=True)
    HAS_MODEL = False

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

# 1. 基础行情字段 (包含交易状态)
fields_daily = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,volume,amount,adj_factor,trade_status"
# 2. 每日指标字段 (注意 XCSC 特有字段名: tot_mv, turn)
fields_daily_basic = "ts_code,trade_date,tot_mv,mv,turn,pe,pe_ttm,pb_new,free_turnover,high_52w,low_52w"
# 3. 资金流向字段
fields_moneyflow = "ts_code,trade_date,buy_sm_vol,sell_sm_vol,buy_md_vol,sell_md_vol,buy_lg_vol,sell_lg_vol,buy_elg_vol,sell_elg_vol,net_mf_vol,net_mf_amount"

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
        # 1. 基础特征 (shared)
        # (v32: shared/features.py 现在包含所有特征逻辑，包括 close_lag1, volume_change 等)
        df_features = apply_technical_indicators(df)
        
        # 2. (v32: 移除手动补全，已统一到 shared/features.py)
        # if 'close_lag1' not in df_features.columns: ...
        
        latest_row = df_features.iloc[[-1]].copy()
        current_date = latest_row['trade_date'].values[0]
        
        # 预测
        # v28 修复: CatBoost 预测时特征顺序必须与训练时一致
        # 训练时的特征顺序 (参考 training/train.py get_feature_columns)
        # 这里我们动态获取模型需要的特征名
        model_feature_names = model.feature_names_
        
        # 检查缺失特征
        missing_features = [f for f in model_feature_names if f not in latest_row.columns]
        if missing_features:
            print(f"  [{ts_code}] 缺失特征: {missing_features}，跳过", flush=True)
            return

        # 按模型要求的顺序重排特征
        X_predict = latest_row[model_feature_names]
        
        prob = model.predict_proba(X_predict)[0]
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
    """
    获取全量数据：行情 + 每日指标 + 资金流向
    并合并为一个 DataFrame
    """
    try:
        # 1. 获取日线行情
        df_daily = pro.daily(ts_code=ts_code, start_date=START_DATE, end_date="", fields=fields_daily)
        if df_daily.empty:
            return None
            
        # 2. 获取每日指标 (市值, 换手, PE/PB)
        # 注意: 接口可能返回空，需处理
        try:
            df_basic = pro.daily_basic(ts_code=ts_code, start_date=START_DATE, end_date="", fields=fields_daily_basic)
        except Exception:
            df_basic = pd.DataFrame()
            
        # 3. 获取资金流向
        try:
            # moneyflow 有时在网络或服务端较慢，设置短超时保护
            # xcsc_tushare 的 pro.moneyflow 本身无 timeout 参数，因此使用线程包装以避免阻塞
            import concurrent.futures

            def call_moneyflow():
                return pro.moneyflow(ts_code=ts_code, start_date=START_DATE, end_date="", fields=fields_moneyflow)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(call_moneyflow)
                try:
                    df_flow = future.result(timeout=10)  # 10秒超时
                except Exception:
                    future.cancel()
                    df_flow = pd.DataFrame()
        except Exception:
            df_flow = pd.DataFrame()

        # --- 数据合并 ---
        # 以 daily 为主表，左连接其他表
        df_merge = df_daily
        
        if not df_basic.empty:
            df_merge = pd.merge(df_merge, df_basic, on=['ts_code', 'trade_date'], how='left')
            
        if not df_flow.empty:
            df_merge = pd.merge(df_merge, df_flow, on=['ts_code', 'trade_date'], how='left')

        # 统一按日期排序 (旧到新)
        df_merge = df_merge.sort_values('trade_date').reset_index(drop=True)
        # 确保日期格式
        df_merge["trade_date"] = pd.to_datetime(df_merge["trade_date"], format="%Y%m%d")

        # 在过滤停牌之前，尝试拉取并合并季报财务数据（安全对齐）
        try:
            from shared.financials import fetch_financials, align_financials_to_daily
            df_fin = fetch_financials(pro, ts_code, start_date=START_DATE)
            if df_fin is not None and not df_fin.empty:
                aligned_fin = align_financials_to_daily(df_merge, df_fin)
                # 合并到主表（按行对齐），保留财务字段
                try:
                    df_merge = pd.concat([df_merge.reset_index(drop=True), aligned_fin.reset_index(drop=True)], axis=1)
                except Exception:
                    pass
        except Exception:
            # 财务抓取非关键，失败时继续
            pass

        # 过滤停牌日并对刚复牌的股票做短期保护（防止复牌异常波动污染特征）
        RESUME_SAFE_DAYS = 5
        if 'trade_status' in df_merge.columns:
            # 标记是否交易：XCSC 使用中文字段值（例如 '交易'、'停牌'、'XD' 等）
            # 兼容常见值：'T','交易','TRADE','交易中','1'
            valid_trade_vals = set(['T', '交易', 'TRADE', '交易中', '1'])
            # 有些值可能包含空格或大小写差异，统一处理
            def is_trade_val(x):
                try:
                    s = str(x).strip()
                except Exception:
                    return False
                return s in valid_trade_vals or s == '交易'
            df_merge['is_trade'] = df_merge['trade_status'].apply(is_trade_val)
            # 将连续段分组
            grp = (df_merge['is_trade'] != df_merge['is_trade'].shift(fill_value=df_merge['is_trade'].iloc[0])).cumsum()
            df_merge['grp'] = grp
            df_merge['days_since_resume'] = 0
            grp_vals = df_merge.groupby('grp')['is_trade'].first().to_dict()
            groups = sorted(grp_vals.items(), key=lambda x: x[0])
            # iterate groups to find resume groups (group with is_trade==True and previous group is_trade==False)
            for idx in range(1, len(groups)):
                gid, val = groups[idx]
                prev_gid, prev_val = groups[idx-1]
                if val and not prev_val:
                    # this group is a resume after suspension
                    mask = df_merge['grp'] == gid
                    n = mask.sum()
                    # set 1..n
                    df_merge.loc[mask, 'days_since_resume'] = list(range(1, n+1))
            # 删除非交易日
            df_merge = df_merge[df_merge['is_trade']]
            # 删除复牌后短期数据
            df_merge = df_merge[~((df_merge['days_since_resume'] > 0) & (df_merge['days_since_resume'] <= RESUME_SAFE_DAYS))]
            # 清理辅助列
            df_merge.drop(columns=['is_trade', 'grp', 'days_since_resume'], inplace=True, errors='ignore')

        # --- 单位校正: 推断并统一量/额单位到“股”和人民币金额 ---
        try:
            if 'volume' in df_merge.columns and 'amount' in df_merge.columns and 'close' in df_merge.columns:
                mask = df_merge['volume'].notna() & df_merge['amount'].notna() & df_merge['close'].notna() & (df_merge['close']>0) & (df_merge['volume']>0)
                if mask.sum() >= 5:
                    ratios = (df_merge.loc[mask, 'amount'] / (df_merge.loc[mask, 'volume'] * df_merge.loc[mask, 'close'] + 1e-12)).replace([float('inf'), -float('inf')], pd.NA).dropna()
                    if len(ratios) >= 3:
                        scale = float(ratios.median())
                        if 1e-6 < scale < 1e6:
                            # 标准化列
                            df_merge['volume_shares'] = df_merge['volume'] * scale
                            df_merge['amount_cny'] = df_merge['volume_shares'] * df_merge['close']
                            df_merge['volume_scale_inferred'] = scale
                            # moneyflow 金额列调整
                            if 'net_mf_amount' in df_merge.columns and df_merge['net_mf_amount'].notna().sum() > 0:
                                orig_med = df_merge.loc[mask, 'amount'].median()
                                new_med = df_merge.loc[mask, 'amount_cny'].median()
                                if orig_med and abs(orig_med) > 0:
                                    amt_factor = new_med / orig_med
                                    df_merge['net_mf_amount_cny'] = df_merge['net_mf_amount'] * amt_factor
                                else:
                                    df_merge['net_mf_amount_cny'] = df_merge['net_mf_amount']
        except Exception:
            # 单个票的归一化失败不应中断整个流程
            pass

        # 仅进行降精度处理，不添加任何额外特征，保持数据纯洁
        df_merge = downcast(df_merge)

        if len(df_merge) > 21:
            return ts_code, df_merge
        else:
            return None

    except Exception as e:
        raise e

def list_main_board_cs():
    """获取主板已上市股票列表"""
    today = datetime.today().strftime("%Y%m%d")
    temp0 = pro.stock_basic(market="CS", fields="ts_code,name,list_date,delist_date,list_board_name")
    temp0 = temp0[temp0["delist_date"].isna()]  # 未退市
    temp0 = temp0[temp0["list_board_name"] == "主板"]  # 主板
    temp0 = temp0[temp0["list_date"] <= today]  # 已经上市
    return temp0[["ts_code", "name"]].reset_index(drop=True)

# (v32: 已移除 add_features 函数，确保保存的数据只包含原始行情)

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
    
    # 全量下载模式：默认处理所有符合条件的股票
    print("⚠️ 全量下载模式：处理所有符合条件的股票", flush=True)
    
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
                    # 使用副本进行预测，不污染原始数据
                    predict_stock(code, df.copy())

                # 保存（get_hist 已经完成归一化与 downcast）
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
