import pandas as pd
import catboost as cb
import shap
import matplotlib.pyplot as plt
import os
import sys
import glob
import random

# === 配置路径 ===
MODELS_DIR = 'models'
DATA_DIR = 'data'
# =============

def list_and_select(files, description):
    """通用的交互式选择函数"""
    if not files:
        print(f"❌ 在 {description} 中未找到任何文件。")
        return None
    
    print(f"\n--- 请选择 {description} ---")
    for i, f in enumerate(files):
        print(f"[{i+1}] {os.path.basename(f)}")
    
    while True:
        choice = input(f"\n请输入序号 (1-{len(files)}): ").strip()
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

    print(f"\n--- 数据选择 (共找到 {len(all_files)} 个文件) ---")
    print("[1] 随机选择一个")
    print("[2] 输入股票代码查找 (如 000001.SZ)")
    print("[3] 列出前 10 个文件")
    
    choice = input("\n请选择数据源模式 (1/2/3): ").strip()
    
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
        for c in candidates:
            if c in all_files:
                return c
        print(f"❌ 未找到代码为 {code} 的数据文件。")
        return None
        
    elif choice == '3':
        return list_and_select(all_files[:10], "数据文件 (Top 10)")
    
    return None

def main():
    # 1. 选择模型
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.cbm"))
    model_path = list_and_select(model_files, "模型文件")
    if not model_path: return

    # 2. 选择数据
    data_path = get_data_file()
    if not data_path: return

    # 3. 加载模型
    print(f"\n🔄 正在加载模型: {model_path} ...")
    try:
        model = cb.CatBoostClassifier()
        model.load_model(model_path)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 4. 加载特征工程逻辑
    # 优先尝试加载 models/frozen_features.py，否则回退到 shared/features.py
    feature_module = None
    try:
        sys.path.append('models')
        import frozen_features
        feature_module = frozen_features
        print("✅ 已加载 frozen_features (模型专用)")
    except ImportError:
        try:
            sys.path.append('shared')
            import features
            feature_module = features
            print("⚠️ 未找到 frozen_features，回退使用 shared/features.py")
        except ImportError:
            print("❌ 无法加载特征工程模块 (frozen_features 或 shared/features)。")
            return

    # 5. 处理数据
    print(f"🔄 正在读取数据: {data_path} ...")
    try:
        df = pd.read_parquet(data_path)
        
        # 应用特征工程
        df_features = feature_module.apply_technical_indicators(df)
        
        # 对齐特征
        model_feature_names = model.feature_names_
        missing_cols = [c for c in model_feature_names if c not in df_features.columns]
        if missing_cols:
            print(f"❌ 数据缺失模型所需的特征: {missing_cols[:5]}...")
            return

        X_sample = df_features[model_feature_names].fillna(0.0)
        
        # 采样 (如果数据太多，取最近 1000 行，既包含近期规律，计算也快)
        if len(X_sample) > 1000:
            X_sample = X_sample.tail(1000)
        
        print(f"📊 样本准备就绪: {X_sample.shape}")

    except Exception as e:
        print(f"❌ 数据处理出错: {e}")
        return

    # 6. SHAP 分析
    print("\n🧮 正在计算 SHAP 值 (请稍候)...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # 7. 绘图
        print("🎨 正在生成图表...")
        plt.figure(figsize=(12, 10))
        
        # 蜂群图
        plt.title(f"SHAP Summary: {os.path.basename(model_path)} @ {os.path.basename(data_path)}")
        shap.summary_plot(shap_values, X_sample, show=False)
        
        output_filename = f"shap_{os.path.basename(data_path).replace('.parquet','')}.png"
        plt.savefig(output_filename, bbox_inches='tight', dpi=300)
        print(f"\n🎉 成功！分析图已保存为: {output_filename}")
        
    except Exception as e:
        print(f"❌ SHAP 计算或绘图失败: {e}")

if __name__ == "__main__":
    main()
