import pandas as pd
import catboost as cb
import matplotlib.pyplot as plt
import os
import sys
import glob
import random
import platform
import numpy as np

# 尝试导入 SHAP，如果不存在则提示
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️ 未检测到 shap 库，部分高级可视化功能将不可用。建议 pip install shap")

# === 配置路径 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
# =============

def setup_plotting_style():
    """配置绘图风格和中文字体"""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 尝试设置中文字体
    system = platform.system()
    fonts = []
    if system == 'Windows':
        fonts = ['SimHei', 'Microsoft YaHei']
    elif system == 'Darwin': # macOS
        fonts = ['Arial Unicode MS', 'PingFang SC']
    else: # Linux
        fonts = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'SimHei']
    
    for font in fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            break
        except:
            continue

def list_and_select(files, description):
    """通用的交互式选择函数"""
    if not files:
        print(f"❌ 在 {description} 中未找到任何文件。")
        return None
    
    print(f"\n--- 请选择 {description} ---")
    for i, f in enumerate(files):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        mtime = pd.to_datetime(os.path.getmtime(f), unit='s').strftime('%Y-%m-%d %H:%M')
        print(f"[{i+1}] {os.path.basename(f):<30} | {size_mb:.1f}MB | {mtime}")
    
    while True:
        choice = input(f"\n请输入序号 (1-{len(files)}, q退出): ").strip()
        if choice.lower() == 'q':
            return None
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return files[idx]
        print("❌ 输入无效，请重试。")

def get_data_file():
    """获取数据文件路径 (支持搜索或随机)"""
    all_files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
    if not all_files:
        print(f"❌ 目录 {DATA_DIR} 中没有找到 .parquet 数据文件。")
        return None

    while True:
        print(f"\n--- 数据选择 (共找到 {len(all_files)} 个文件) ---")
        print("[1] 🎲 随机选择一个")
        print("[2] 🔍 输入股票代码查找")
        print("[3] 📋 列出前 10 个文件")
        print("[q] 🚪 退出")
        
        choice = input("\n请选择模式: ").strip().lower()
        
        if choice == 'q':
            return None
        
        if choice == '1':
            selected = random.choice(all_files)
            print(f"🎲 随机选中: {os.path.basename(selected)}")
            return selected
        
        elif choice == '2':
            code = input("请输入股票代码 (例如 000001.SZ): ").strip()
            # 尝试几种常见格式
            candidates = [
                os.path.join(DATA_DIR, f"{code}.parquet"),
                os.path.join(DATA_DIR, f"{code}"),
            ]
            # 模糊搜索
            matches = [f for f in all_files if code in os.path.basename(f)]
            
            if matches:
                if len(matches) == 1:
                    return matches[0]
                else:
                    return list_and_select(matches, f"匹配 '{code}' 的文件")
            else:
                print(f"❌ 未找到包含 {code} 的数据文件。")
            
        elif choice == '3':
            return list_and_select(all_files[:10], "数据文件 (Top 10)")
        
    return None

def load_feature_engineering():
    """动态加载特征工程模块"""
    # 优先尝试加载 models/frozen_features.py
    if os.path.exists(os.path.join(MODELS_DIR, 'frozen_features.py')):
        try:
            sys.path.append(MODELS_DIR)
            import frozen_features
            print("✅ 已加载 frozen_features (模型专用)")
            return frozen_features
        except ImportError as e:
            print(f"⚠️ 加载 frozen_features 失败: {e}")
    
    # 回退到 shared/features.py
    if os.path.exists(os.path.join(BASE_DIR, 'shared', 'features.py')):
        try:
            sys.path.append(os.path.join(BASE_DIR, 'shared'))
            import features
            print("⚠️ 使用 shared/features.py (通用特征)")
            return features
        except ImportError as e:
            print(f"❌ 加载 shared/features 失败: {e}")
    
    return None

import time

def get_shap_explanation(model, X):
    """使用 CatBoost 原生加速计算 SHAP 值"""
    t0 = time.time()
    print("🚀 使用 CatBoost 原生接口加速计算 SHAP值...", end="", flush=True)
    pool = cb.Pool(X)
    # 返回 shape (N, F+1), 最后一列是 base_value
    shap_values_raw = model.get_feature_importance(pool, type=cb.EFstrType.ShapValues)
    
    values = shap_values_raw[:, :-1]
    base_values = shap_values_raw[:, -1]
    
    # 构造 SHAP Explanation 对象
    explanation = shap.Explanation(
        values=values,
        base_values=base_values,
        data=X,
        feature_names=X.columns.tolist()
    )
    print(f" 完成 ({time.time()-t0:.2f}s)")
    return explanation

def plot_shap_summary(explanation, filename_prefix):
    """生成 SHAP 摘要图 (Beeswarm)"""
    t0 = time.time()
    print("🎨 正在生成 SHAP 摘要图 (Beeswarm)...", end="", flush=True)
    
    plt.figure(figsize=(10, 8)) # 稍微减小尺寸
    plt.title(f"SHAP Summary: {filename_prefix}")
    # max_display=20 限制显示特征数，加快绘图
    shap.summary_plot(explanation, show=False, max_display=20, plot_size=None)
    
    out_file = os.path.join(REPORTS_DIR, f"shap_summary_{filename_prefix}.png")
    plt.savefig(out_file, bbox_inches='tight', dpi=150) # 降低 DPI 加速保存
    plt.close()
    print(f" 完成 ({time.time()-t0:.2f}s) -> {out_file}")

def plot_shap_bar(explanation, filename_prefix):
    """生成 SHAP 重要性条形图"""
    t0 = time.time()
    print("🎨 正在生成 SHAP 重要性条形图...", end="", flush=True)
    
    plt.figure(figsize=(10, 8))
    plt.title(f"Feature Importance: {filename_prefix}")
    shap.summary_plot(explanation, plot_type="bar", show=False, max_display=20, plot_size=None)
    
    out_file = os.path.join(REPORTS_DIR, f"shap_bar_{filename_prefix}.png")
    plt.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close()
    print(f" 完成 ({time.time()-t0:.2f}s) -> {out_file}")

def plot_latest_waterfall(explanation, filename_prefix):
    """生成最新一条数据的瀑布图 (解释单次预测)"""
    t0 = time.time()
    print("🎨 正在生成最新预测的瀑布图...", end="", flush=True)
    
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(explanation[-1], show=False, max_display=15)
    plt.title(f"Latest Prediction Explanation: {filename_prefix}")
    
    out_file = os.path.join(REPORTS_DIR, f"shap_waterfall_{filename_prefix}.png")
    plt.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close()
    print(f" 完成 ({time.time()-t0:.2f}s) -> {out_file}")

def main():
    setup_plotting_style()
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. 选择模型
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.cbm"))
    model_path = list_and_select(model_files, "模型文件")
    if not model_path: return

    # 2. 加载模型
    print(f"\n🔄 正在加载模型: {os.path.basename(model_path)} ...")
    try:
        model = cb.CatBoostClassifier()
        model.load_model(model_path)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 3. 加载特征工程
    feature_module = load_feature_engineering()
    if not feature_module:
        print("❌ 无法继续：缺少特征工程模块")
        return

    # 循环：允许用户不断更换数据进行分析
    while True:
        data_path = get_data_file()
        if not data_path: break

        print(f"\n🔄 正在读取数据: {os.path.basename(data_path)} ...")
        try:
            df = pd.read_parquet(data_path)
            if df.empty:
                print("❌ 数据为空")
                continue
                
            # 显示基本信息
            print(f"   📅 日期范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
            print(f"   📊 数据行数: {len(df)}")
            if 'close' in df.columns:
                print(f"   💰 最新收盘: {df['close'].iloc[-1]:.2f}")

            # --- (Fix) 补全可能缺失的原始数据列 (确保旧数据文件也能运行) ---
            # 1. 融资融券字段
            margin_cols = ['rzye', 'rqye', 'rzmre', 'rzche', 'rqmcl', 'rqchl', 'rzrqye']
            for col in margin_cols:
                if col not in df.columns:
                    df[col] = 0.0
                else:
                    df[col] = df[col].fillna(0.0)
            
            # 2. 龙虎榜字段
            top_cols = ['top_net_amount', 'top_buy_amount', 'top_sell_amount', 'top_count']
            for col in top_cols:
                if col not in df.columns:
                    df[col] = 0.0
                else:
                    df[col] = df[col].fillna(0.0)
            
            # 3. 大宗交易字段
            block_cols = ['block_vol', 'block_amount', 'block_count']
            for col in block_cols:
                if col not in df.columns:
                    df[col] = 0.0
                else:
                    df[col] = df[col].fillna(0.0)
            
            if 'block_avg_price' not in df.columns:
                df['block_avg_price'] = df['close'] if 'close' in df.columns else 0.0
            else:
                df['block_avg_price'] = df['block_avg_price'].fillna(df['close'] if 'close' in df.columns else 0.0)
            # -------------------------------------------------------

            # 应用特征工程
            df_features = feature_module.apply_technical_indicators(df)
            
            # 对齐特征
            model_feature_names = model.feature_names_
            missing_cols = [c for c in model_feature_names if c not in df_features.columns]
            if missing_cols:
                print(f"❌ 数据缺失模型所需的特征: {missing_cols[:5]}... (共缺失 {len(missing_cols)} 个)")
                continue

            X_full = df_features[model_feature_names].fillna(0.0)
            
            # 采样用于 SHAP 摘要 (最近 200 行，加速绘图)
            X_sample = X_full.tail(200)
            
            # 预测最新一天的概率
            latest_prob = model.predict_proba(X_full.iloc[[-1]])[0]
            print(f"\n🔮 最新预测 ({df['trade_date'].iloc[-1]}):")
            print(f"   📉 下跌概率: {latest_prob[0]:.2%}")
            print(f"   ➖ 震荡概率: {latest_prob[1]:.2%}")
            print(f"   📈 上涨概率: {latest_prob[2]:.2%}")

            if not HAS_SHAP:
                input("\n按 Enter 继续...")
                continue

            # 交互式绘图菜单
            explanation = None
            while True:
                print("\n--- 可视化分析菜单 ---")
                print("[1] 🐝 SHAP 摘要图 (Beeswarm) - 全局特征影响")
                print("[2] 📊 SHAP 重要性 (Bar) - 特征重要性排序")
                print("[3] 🌊 最新预测归因 (Waterfall) - 为什么预测这个结果？")
                print("[4] 🔙 更换数据文件")
                print("[q] 🚪 退出程序")
                
                viz_choice = input("\n请选择操作: ").strip().lower()
                
                if viz_choice == 'q':
                    return
                if viz_choice == '4':
                    break
                
                # 懒加载 explainer (使用原生加速)
                if explanation is None and viz_choice in ['1', '2', '3']:
                    try:
                        explanation = get_shap_explanation(model, X_sample)
                    except Exception as e:
                        print(f"❌ SHAP 计算失败: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                file_prefix = f"{os.path.basename(data_path).replace('.parquet','')}_{pd.Timestamp.now().strftime('%H%M%S')}"
                
                try:
                    if viz_choice == '1':
                        plot_shap_summary(explanation, file_prefix)
                    elif viz_choice == '2':
                        plot_shap_bar(explanation, file_prefix)
                    elif viz_choice == '3':
                        plot_latest_waterfall(explanation, file_prefix)
                except Exception as e:
                    print(f"❌ 绘图失败: {e}")
                    import traceback
                    traceback.print_exc()

        except Exception as e:
            print(f"❌ 处理数据出错: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()
