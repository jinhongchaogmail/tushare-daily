#!/usr/bin/env python
# coding: utf-8

# @title 数据准备与Optuna寻优 (已合并为 Colab 终端脚本 - v15.1 修正版)
# 负责：加载原始数据 -> 特征工程 -> Optuna寻优
#
# 【v15.1 修正版 更新】:
# 1. (v15.1) 修复了主逻辑块中 'SHOW_VERBOSE_OUTPUT' is not defined 的 NameError。
# 2. (v15) 新增特征工程缓存 (feature_cache.parquet)。
# 3. (v15) 在后验评估后，保存最终的 .cbm 模型和 .json 阈值参数。
# 4. (v15) 在 post_process_data 中进行内存优化 (float64->float32)。
# 5. (v15) 在 process_single_file 中添加了 ATR, Stoch, CMF 等新特征。
# 6. (v15) 在 get_feature_columns 中移除了 O/H/L/C/V 原始数据。

import sys
import subprocess
import time
import glob
import os
import shutil
import tarfile
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
import importlib
import logging
import json
import hashlib
try:
    import yaml
except ImportError:
    yaml = None
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# 如果存在用户虚拟环境，尽早在进程启动时切换到该 Python 解释器，
# 以确保后续的 pip 安装与模块导入在同一解释器中生效。
_preferred_venv = os.environ.get('VENV_PATH', '/home/jin/.venv')
_preferred_python = os.path.join(_preferred_venv, 'bin', 'python')
if os.path.exists(_preferred_python):
    # 如果当前解释器不是虚拟环境的 python，则用 execv 切换到虚拟环境解释器
    try:
        if os.path.abspath(sys.executable) != os.path.abspath(_preferred_python):
            print(f"检测到虚拟环境 Python: {_preferred_python}，正在用该解释器重启脚本...")
            os.execv(_preferred_python, [_preferred_python] + sys.argv)
    except Exception:
        # 如果无法重启（权限等），继续当前解释器但后续会尝试用 venv 的 pip
        pass

# --- 步骤 0: 环境配置与依赖引导 ---

class Environment(Enum):
    COLAB = 'COLAB'
    PAI_DSW = 'PAI_DSW'
    LOCAL = 'LOCAL'
    UNKNOWN = 'UNKNOWN'

def get_environment():
    """检测当前运行环境 (已恢复 Colab 检测)"""
    if os.environ.get('COLAB_RELEASE_TAG') or os.path.exists('/content/drive') or os.getcwd() == '/content':
         return Environment.COLAB
    elif os.path.exists('/mnt/workspace'): # PAI-DSW
        return Environment.PAI_DSW
    elif os.path.exists(os.path.expanduser('~/gdrive')): # Local
        return Environment.LOCAL
    elif os.path.exists('/content'): # Colab 后备检查
        return Environment.COLAB
    return Environment.UNKNOWN

def bootstrap_dependencies():
    """
    (v12.1) (Colab 优化模式) 依赖安装程序。
    检查所有包，如果缺少则自动安装。
    """

    print("--- 引导程序: 检查并安装依赖 (v12.1 - Colab 优化模式) ---")
    current_env = get_environment()
    print(f"检测到环境: {current_env.value}")

    def run_pip_install(package_name, args=None, capture_output=True):
        """辅助函数：在当前 Python 内核中运行 pip install"""
        # 优先使用用户指定的虚拟环境来执行 pip（如果存在）
        command = [pip_python, "-m", "pip", "install", package_name]
        if args:
            command.extend(args)

        print(f"  -> 正在执行: {' '.join(command)}")
        try:
            # 静默安装，避免刷屏
            stdout = subprocess.DEVNULL if capture_output else None
            stderr = subprocess.DEVNULL if capture_output else None
            subprocess.check_call(command, stdout=stdout, stderr=stderr)
            print(f"  -> {package_name} 安装成功。")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  -> {package_name} 安装失败: {e}")
            if current_env == Environment.COLAB and (package_name == "pandas-ta" or package_name == "numba"):
                 print("【错误】: Colab 关键依赖安装失败，请检查环境。")
                 sys.exit(1)
            return False
        except Exception as e:
            print(f"  -> 安装时发生意外错误: {e}")
            return False

    # 检测并优先使用虚拟环境中的 python 来执行 pip（如果存在）
    venv_dir = os.environ.get('VENV_PATH', '/home/jin/.venv')
    venv_python = os.path.join(venv_dir, 'bin', 'python')
    if os.path.exists(venv_python):
        pip_python = venv_python
        print(f"使用虚拟环境 Python 作为 pip: {pip_python}")
    else:
        pip_python = sys.executable

    # 步骤 1: Colab 特殊依赖处理
    if current_env == Environment.COLAB:
        print("\n【Colab 模式】: 正在应用 'Colab-first' 依赖安装策略...")
        print("【Colab 步骤 1/2】: 安装 pandas-ta --no-dependencies")
        run_pip_install("pandas-ta", args=["--no-dependencies"])
        print("\n【Colab 步骤 2/2】: 安装 numba")
        run_pip_install("numba")
        print("【Colab 模式】: 特殊依赖处理完成。")


    # 步骤 2: 检查所有必需的包 (已合并两个单元)
    import_targets = {
        'psutil': 'psutil',
        'numba': 'numba',
        'pandas': 'pandas',
        'scikit-learn': 'sklearn',
        'pyarrow': 'pyarrow',
        'pandas_ta': 'pandas_ta',
        'optuna': 'optuna',
        'imblearn': 'imblearn',
        'catboost': 'catboost',
        'colorlog': 'colorlog',
        'plotly': 'plotly',
        'seaborn': 'seaborn',
        'matplotlib': 'matplotlib',
        'kaleido': 'kaleido' # 用于保存 plotly 图像
    }

    missing_packages = []
    print("\n【步骤 1/3】: 检查所有必需的包...")
    for pip_name, import_name in import_targets.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            # Colab 上刚安装的包可能需要重启才能导入
            if current_env == Environment.COLAB and (pip_name == 'pandas_ta' or pip_name == 'numba'):
                print(f" - [已知] {import_name} (pip: {pip_name}) 已在 Colab 上安装，等待内核刷新。")
            else:
                print(f" - [缺失] {import_name} (pip: {pip_name})")
                missing_packages.append(pip_name)

    # 步骤 3: 自动安装缺失的包
    if missing_packages:
        print(f"\n【步骤 2/3】: 发现缺失的包: {', '.join(missing_packages)}")
        print("【步骤 3/3】: 正在尝试自动安装...")
        for package in missing_packages:
            run_pip_install(package)
        print("\n【重要提示】: 依赖已更新。如果是首次安装，可能需要重新运行脚本。")
    else:
        print("\n【步骤 2/3】: 所有包均已存在。")
        print("【步骤 3/3】: 跳过安装。")
        print("\n【信息】:所有依赖已成功加载。")


# --- 运行引导程序 ---
bootstrap_dependencies()

# --- 步骤 1: 导入核心库 ---
try:
    import psutil
    import numpy as np
    import pandas as pd
    import pandas_ta as ta
    from sklearn.preprocessing import LabelEncoder
    from imblearn.under_sampling import RandomUnderSampler
    
    # 添加 shared 到路径以便导入 feature_engineering
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared'))
    try:
        from 特征工程 import apply_technical_indicators
    except ImportError:
        print("警告：无法从 shared 导入特征工程。如果可用，将使用本地定义。")

    # Optuna/CatBoost 部分的导入
    import optuna
    import catboost as cb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
    from sklearn.utils.class_weight import compute_class_weight
    import plotly.io as pio
    import optuna.visualization as vis
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # v25: 中文字体配置 (适配 Linux/Noto 字体)
    import matplotlib.font_manager as fm
    
    # 优先使用系统已安装的中文字体
    zh_fonts = ['Noto Sans CJK SC', 'Noto Serif CJK SC', 'WenQuanYi Micro Hei', 
                'AR PL UMing CN', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    
    # 检测可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    selected_font = None
    for font in zh_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font] + zh_fonts
        print(f"✓ 已选择中文字体: {selected_font}")
    else:
        plt.rcParams['font.sans-serif'] = zh_fonts
        print("⚠ 未找到首选中文字体，使用默认列表")
    
    plt.rcParams['axes.unicode_minus'] = False
    
except ImportError as e:
    print(f"\n【导入错误】: {e}")
    print("请检查依赖安装是否成功。")
    sys.exit(1)

# 忽略 future warnings, 避免干扰
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- 优化 1: 配置管理系统 ---
class ConfigManager:
    """集中管理所有配置参数"""
    
    DEFAULT_CONFIG = {
        'data': {'max_files': 3000, 'min_rows': 20, 'cache_enabled': True},
        'optuna': {'n_trials': 200, 'n_jobs': 2, 'tsc_splits': 3, 'pruner_startup': 2},
        'training': {'train_test_ratio': 0.8, 'random_seed': 42},
        'balance': {'penalty_threshold': 0.15, 'flat_multiplier': 10},
        'logging': {'level': 'INFO', 'verbose': False, 'log_interval': 100}
    }
    
    def __init__(self, config_path=None):
        self.config = self.DEFAULT_CONFIG.copy()
        if config_path and os.path.exists(config_path) and yaml:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f) or {}
                    self._deep_update(self.config, user_config)
                    print(f"✓ 配置文件加载成功: {config_path}")
            except Exception as e:
                print(f"⚠ 配置文件加载失败: {e}，使用默认配置")
    
    @staticmethod
    def _deep_update(d, u):
        """递归更新字典"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                ConfigManager._deep_update(d[k], v)
            else:
                d[k] = v
    
    def get(self, key, default=None):
        """获取配置值，支持点号路径: 'optuna.n_trials'"""
        keys = key.split('.')
        val = self.config
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
        return val if val is not None else default

# 初始化全局配置
# 使用脚本所在目录的相对 config 路径，便于移动整个项目目录
script_dir = os.path.dirname(os.path.abspath(__file__))
default_config_path = os.path.join(script_dir, 'config.yaml')
CONFIG = ConfigManager(default_config_path)

# --- 步骤 2: 核心函数定义 (数据准备) ---

def setup_paths(env: Environment):
    """(v13) 根据环境动态配置路径 (已恢复 Colab 路径)。"""
    paths = {}
    # 尝试使用脚本的上级目录作为项目根（便于从其他目录复制过来的代码直接运行）
    project_root = os.path.abspath(os.path.join(script_dir, '..')) if 'script_dir' in globals() else None

    if env == Environment.COLAB:
        paths['BASE_DIR'] = '/content/'
        # G-Drive 基础路径，用于保存 Optuna 数据库、报告和缓存
        paths['GDRIVE_BASE_PATH'] = '/content/drive/MyDrive/Colab Notebooks'
        # 数据源的完整路径
        paths['ARCHIVE_PATH'] = os.path.join(paths['GDRIVE_BASE_PATH'], 'parquet-data.tar.gz')
    elif env == Environment.PAI_DSW:
        paths['BASE_DIR'] = '/mnt/workspace/'
        paths['ARCHIVE_PATH'] = os.path.join(paths['BASE_DIR'], 'parquet-data.tar.gz')
        paths['GDRIVE_BASE_PATH'] = paths['BASE_DIR']
    elif env == Environment.LOCAL:
        # 优先使用项目根（脚本上级目录），否则回退到原有的 '~/gdrive'
        if project_root and os.path.exists(project_root):
            paths['BASE_DIR'] = project_root
        else:
            paths['BASE_DIR'] = os.path.expanduser('~/gdrive')
        paths['ARCHIVE_PATH'] = os.path.join(paths['BASE_DIR'], 'parquet-data.tar.gz')
        paths['GDRIVE_BASE_PATH'] = paths['BASE_DIR']
    else: # UNKNOWN
        # 使用项目根作为首选回退位置，若不存在则保持与之前一致的默认
        if project_root and os.path.exists(project_root):
            paths['BASE_DIR'] = project_root
        else:
            paths['BASE_DIR'] = '/mnt/workspace/'
        paths['ARCHIVE_PATH'] = os.path.join(paths['BASE_DIR'], 'parquet-data.tar.gz')
        paths['GDRIVE_BASE_PATH'] = paths['BASE_DIR']

    # 动态、相对地定义数据解压路径
    paths['DATA_DIR'] = os.path.join(paths['BASE_DIR'], 'data/')
    
    # 确保 GDrive 路径存在 (用于保存报告)
    if 'GDRIVE_BASE_PATH' in paths:
        try:
            os.makedirs(paths['GDRIVE_BASE_PATH'], exist_ok=True)
        except OSError as e:
            print(f"警告：无法创建 GDrive 保存路径 {paths['GDRIVE_BASE_PATH']}: {e}")
            print("将尝试在本地 /content/ 保存。")
            paths['GDRIVE_BASE_PATH'] = paths['BASE_DIR']
            
    return paths

def prepare_data_files(env: Environment, paths: dict):
    """处理数据文件的解压 (已移除 GDrive 挂载逻辑)"""
    print("\n--- 步骤 2: 数据准备 (条件解压) ---")
    archive_path = paths['ARCHIVE_PATH']
    data_dir = paths['DATA_DIR']

    if os.path.exists(data_dir) and os.listdir(data_dir):
        print(f"数据目录 {data_dir} 已存在且非空，跳过数据准备。")
        return True

    archive_exists = os.path.exists(archive_path)

    if env == Environment.COLAB and not archive_exists:
        print(f"错误: Colab 环境中未在 {archive_path} 找到压缩包。")
        print("请确保 Google Drive 已挂载，且文件存在于 'MyDrive/Colab Notebooks/parquet-data.tar.gz'")
        return False

    if not archive_exists:
        print(f"错误: 压缩包 {archive_path} 不存在，无法继续。")
        return False

    print(f"正在解压 {archive_path} 到 {paths['BASE_DIR']}...")
    try:
        os.makedirs(data_dir, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as tar:
            if sys.version_info >= (3, 12):
                tar.extractall(path=paths['BASE_DIR'], filter='data')
            else:
                tar.extractall(path=paths['BASE_DIR'])
        print("文件解压完成。")
        return True
    except Exception as e:
        print(f"解压文件时出错: {e}")
        return False

# apply_technical_indicators imported from shared.feature_engineering

def process_single_file(file_path):
    """
    对单个 Parquet 文件进行特征工程处理（优化2：流式处理）。
    此函数在子进程中运行，因此需要独立导入库。
    """
    import pandas as pd
    import os
    import numpy as np

    try:
        file_name = os.path.basename(file_path)
        stock_code = file_name.split('.')[0]

        df = pd.read_parquet(file_path)

        # 优化3：细化异常处理
        if 'trade_date' not in df.columns:
            raise ValueError(f"缺少 'trade_date' 列")
        if df.empty:
            raise ValueError("DataFrame 为空")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.set_index('trade_date')

        df = df.sort_index(ascending=True)

        if len(df) < CONFIG.get('data.min_rows', 20):
            return None

        # 优化2.1：去重的特征计算
        df = apply_technical_indicators(df)

        # 添加滞后特征 - 注：close_pct_change 已在 apply_technical_indicators 中计算
        df['close_lag1'] = df['close'].shift(1)
        # 注：return_5d 已在 apply_technical_indicators 中计算，无需重复
        
        # --- (v19 新增：时间特征) ---
        # 确保索引是日期类型
        if df.index.name == 'trade_date' or 'trade_date' in df.columns:
            date_col = df.index if df.index.name == 'trade_date' else pd.to_datetime(df['trade_date'])
            if not isinstance(date_col, pd.DatetimeIndex):
                date_col = pd.to_datetime(date_col)
            
            # v17 修正: 移除日历特征，防止过拟合
            # df['day_of_week'] = date_col.dayofweek
            # df['is_monday'] = (date_col.dayofweek == 0).astype(int)
            # df['is_friday'] = (date_col.dayofweek == 4).astype(int)
            
            # df['day_of_month'] = date_col.day
            # df['is_month_start'] = (date_col.day <= 5).astype(int)
            # df['is_month_end'] = (date_col.day >= 25).astype(int)
            
            # df['quarter'] = date_col.quarter
            
            # df['month'] = date_col.month
            # df['is_year_start'] = (date_col.month == 1).astype(int)
            # df['is_year_end'] = (date_col.month == 12).astype(int)
        # --- 时间特征结束 ---
        
        # --- (新增 v16：价量相关性特征) ---
        df['price_vol_corr'] = df['close'].rolling(20, min_periods=1).corr(df['volume'])
        df['volume_change'] = df['volume'].pct_change()

        # v19: 处理 Inf 值（除零等导致）
        df = df.replace([np.inf, -np.inf], np.nan)
        df.fillna(method='ffill', inplace=True)
        df.dropna(inplace=True)

        # --- (v21 优化: 内存优化下沉) ---
        # 在单文件级别进行类型转换，减少合并时的内存压力
        for col in df.select_dtypes(include='float64').columns:
            df[col] = df[col].astype('float32')
        # --- (优化结束) ---

        if df.empty:
            return None

        # 目标变量
        future_period = CONFIG.get('features.target_future_period', 5)
        df['future_close_5d'] = df['close'].shift(-future_period)
        df['future_return_5d'] = (df['future_close_5d'] - df['close']) / df['close']

        df['stock_code'] = stock_code
        df.drop(columns=['future_close_5d'], inplace=True)

        return df if not df.empty else None

    except (ValueError, KeyError):
        # 优化3：记录特定异常类型
        return None
    except Exception as e:
        # 只记录意外错误
        logger = logging.getLogger(__name__)
        logger.debug(f"处理文件 {os.path.basename(file_path)} 时发生错误: {type(e).__name__}")
        return None

def run_feature_engineering(data_dir, max_files):
    """
    动态适配并行/串行处理（优化6：进度条）。
    v21: 恢复 ProcessPoolExecutor 并行处理，保留分批逻辑以控制内存。
    """
    print(f"\n--- 步骤 3: 文件列表获取与特征工程 (动态适配) ---")

    parquet_files = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))  # 优化4：排序

    if not parquet_files:
        print(f"错误：数据目录 {data_dir} 为空，没有找到 .parquet 文件。")
        return []

    if len(parquet_files) > max_files:
        parquet_files = parquet_files[:max_files]
        print(f"已将处理文件数量限制为前 {max_files} 个。")

    print(f"找到 {len(parquet_files)} 个 .parquet 文件进行处理。")

    # v19: 分批处理策略（Colab 风格）- 避免内存峰值
    BATCH_SIZE = 500  # 每批处理文件数
    processed_dfs = []
    start_time = time.time()
    total_files = len(parquet_files)
    
    # v21: 使用 ProcessPoolExecutor
    max_workers = min(os.cpu_count(), 8) # 限制最大进程数
    print(f"【优化】: 使用分批并行处理模式（每批 {BATCH_SIZE} 个文件, 进程数 {max_workers}）")
    
    num_batches = (total_files + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, total_files)
        batch_files = parquet_files[batch_start:batch_end]
        
        print(f"\n--- 批次 {batch_idx + 1}/{num_batches}: 文件 {batch_start + 1}-{batch_end} ---")
        
        batch_dfs = []
        
        # v21: 并行执行
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            futures = {executor.submit(process_single_file, fp): fp for fp in batch_files}
            
            # 使用 tqdm 显示进度
            iterator = as_completed(futures)
            if tqdm:
                iterator = tqdm(iterator, total=len(batch_files), desc=f"批次 {batch_idx + 1}")
            
            for future in iterator:
                try:
                    result = future.result()
                    if result is not None:
                        batch_dfs.append(result)
                except Exception as e:
                    # 这里的异常通常是 process_single_file 内部未捕获的
                    pass
        
        if batch_dfs:
            # 每批处理完立即合并，释放单个 df 内存
            batch_combined = pd.concat(batch_dfs, axis=0, ignore_index=False)
            processed_dfs.append(batch_combined)
            print(f"批次 {batch_idx + 1} 完成: {len(batch_dfs)} 个文件 -> {len(batch_combined)} 行")
            
            # 清理批次内存
            del batch_dfs
            import gc
            gc.collect()

    end_time = time.time()
    print(f"\n特征工程完成，耗时: {end_time - start_time:.2f} 秒。共 {num_batches} 批次。")
    return processed_dfs


def post_process_data(processed_dfs):
    """合并所有 DataFrame，进行编码、排序、目标定义和内存优化"""
    print("\n--- 步骤 4: 合并、编码、排序与目标定义 ---")
    if not processed_dfs:
        print("没有可处理的数据，跳过此步骤。")
        return None

    combined_df = pd.concat(processed_dfs, axis=0)
    print(f"合并后数据形状 (初步): {combined_df.shape}")

    combined_df.dropna(subset=['future_return_5d'], inplace=True)
    if combined_df.empty:
        print("错误：在处理目标变量必需的 future_return_5d 后，DataFrame 为空。")
        return None

    print(f"移除 future_return_5d 为 NaN 的行后形状: {combined_df.shape}")

    print("编码股票代码...")
    le_stock = LabelEncoder()
    combined_df['stock_code_encoded'] = le_stock.fit_transform(combined_df['stock_code'])
    combined_df.drop(columns=['stock_code'], inplace=True)

    print("按交易日期排序...")
    combined_df = combined_df.sort_index()
    print("DataFrame 已排序。")

    print("定义三分类目标变量 (固定, 仅用于展示)...")
    lower_bound = -0.0205932534665812
    upper_bound = 0.0205932534665812
    print(f"持平范围 (固定): [{lower_bound:.3f}, {upper_bound:.3f}]")

    combined_df['target_class_3'] = np.nan
    combined_df.loc[combined_df['future_return_5d'] < lower_bound, 'target_class_3'] = -1
    combined_df.loc[(combined_df['future_return_5d'] >= lower_bound) & (combined_df['future_return_5d'] <= upper_bound), 'target_class_3'] = 0
    combined_df.loc[combined_df['future_return_5d'] > upper_bound, 'target_class_3'] = 1

    combined_df.dropna(subset=['target_class_3'], inplace=True)
    combined_df['target_class_3'] = combined_df['target_class_3'].astype(int)

    print("三分类目标 'target_class_3' (固定) 已创建。")
    print("类别分布 (固定):")
    print(combined_df['target_class_3'].value_counts(normalize=True))

    # --- (新增 优化 1.3) 内存优化 ---
    # v21: 内存优化已下沉至 process_single_file，此处仅需处理合并后产生的少量 int64
    print("\n[进度] 正在进行最终内存检查...")
    # 再次检查 float64 (防止遗漏)
    for col in combined_df.select_dtypes(include='float64').columns:
        combined_df[col] = combined_df[col].astype('float32')
        
    for col in combined_df.select_dtypes(include='int64').columns:
        # 股票编码和目标列可以进一步缩小
        if col == 'target_class_3':
             combined_df[col] = combined_df[col].astype('int16')
        elif col == 'stock_code_encoded':
            combined_df[col] = combined_df[col].astype('int32') # 假设股票数少于 20 亿
    print(f"[进度] ✓ 内存优化完成。优化后内存占用: {combined_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # --- (v17 重构：移除全局 Scaler，改为 CV 内部处理，避免数据泄露) ---
    # 注意：RobustScaler 已移至 objective() 函数的 CV 循环内
    # 每折仅用训练集 fit，再 transform 验证集
    print("[进度] 特征标准化将在 CV 循环内进行，避免数据泄露。")
    
    # --- (v17 修复：使用 close_pct_change 的 rolling(20).std() 计算历史波动率) ---
    print("[进度] 正在计算个股滚动波动率因子 (rolling 20 日, 基于 close_pct_change)...")
    # 按股票分组计算滚动波动率 - 使用历史数据 close_pct_change 而非未来数据
    if combined_df.index.name is None:
        combined_df.index.name = 'trade_date'
    combined_df = combined_df.sort_values(['stock_code_encoded', combined_df.index.name])
    combined_df['volatility_factor'] = combined_df.groupby('stock_code_encoded')['close_pct_change'].transform(
        lambda x: x.rolling(window=20, min_periods=5).std()
    )
    # 填充 NaN（前 20 行无法计算）
    # v17 改进: 使用 bfill 或 fillna(method='bfill') 向后填充，避免整体中位数引起的未来泄露
    combined_df['volatility_factor'] = combined_df.groupby('stock_code_encoded')['volatility_factor'].transform(
        lambda x: x.bfill().fillna(0)  # 向后填充，最后用 0 填充
    )
    # 归一化：除以当前已知数据的中位数
    combined_df['volatility_factor'] = combined_df['volatility_factor'].astype('float32')
    print(f"[进度] ✓ 历史波动率因子计算完成 (20日窗口, close_pct_change)。")
    
    # --- (v26 新增: 市场情绪 - 超额收益) ---
    print("[进度] 正在计算市场情绪指标 (超额收益)...")
    # 1. 计算每日全市场平均收益率 (作为大盘指数代理)
    market_return = combined_df.groupby(combined_df.index.name)['close_pct_change'].mean()
    market_return.name = 'market_return'
    
    # 2. 将大盘收益率映射回个股数据
    # 注意：combined_df 索引是 trade_date，可以直接 join 或 map
    # 但 combined_df 可能有重复索引 (不同股票同一天)，所以用 merge 或 map
    combined_df['market_return'] = combined_df.index.map(market_return)
    
    # 3. 计算超额收益 (Alpha)
    combined_df['excess_return'] = combined_df['close_pct_change'] - combined_df['market_return']
    
    # 4. 计算当日涨幅排名 (0~1) - (v27 新增: 截面特征)
    # 这能剔除大盘的影响，还原个股真实的强弱
    combined_df['rank_pct_chg'] = combined_df.groupby(combined_df.index.name)['close_pct_change'].rank(pct=True).astype('float32')
    
    # 5. 填充可能的 NaN (比如某天只有一只股票交易，或者数据缺失)
    combined_df['excess_return'] = combined_df['excess_return'].fillna(0.0).astype('float32')
    combined_df['rank_pct_chg'] = combined_df['rank_pct_chg'].fillna(0.5).astype('float32') # 默认排名居中
    
    # 移除中间变量 market_return 以节省内存 (如果不需要作为特征)
    combined_df.drop(columns=['market_return'], inplace=True)
    print(f"[进度] ✓ 超额收益 (excess_return) 和 涨幅排名 (rank_pct_chg) 计算完成。")
    # --- (新增结束) ---

    return combined_df


# --- 步骤 3: Optuna 寻优函数定义 (原第二单元) ---

def run_optuna_study(combined_df, base_save_path):
    """
    运行 Optuna + CatBoost 寻优。
    """
    print("\n\n--- 启动 Optuna 寻优单元 (CatBoost CPU/GPU 自动适配版) ---")

    print(f"Optuna 寻优：文件将保存到: {base_save_path}")
    GDRIVE_SAVE_PATH = base_save_path # 使用传入的 GDrive 路径
    os.makedirs(GDRIVE_SAVE_PATH, exist_ok=True)


    # --- 0. (核心配置区 - 优化1：从配置文件读取) ---
    # -----------------------------------------------------------------
    SHOW_VERBOSE_OUTPUT = CONFIG.get('logging.verbose', False)
    N_TRIALS = CONFIG.get('optuna.n_trials', 200)
    TRAIN_TEST_SPLIT_RATIO = CONFIG.get('training.train_test_ratio', 0.8)
    TSC_N_SPLITS = CONFIG.get('optuna.tsc_splits', 3)  # 优化：从5减为3
    FUTURE_GAP = CONFIG.get('optuna.future_gap', 5)  # v17: CV 间隔防止标签泄露
    BALANCE_PENALTY_THRESHOLD = CONFIG.get('balance.penalty_threshold', 0.15)
    BALANCE_FLAT_WEIGHT_MULTIPLIER = CONFIG.get('balance.flat_multiplier', 10)
    # -----------------------------------------------------------------

    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    warnings.filterwarnings("ignore", category=UserWarning)

    # --- 健壮的 GPU 自动检测 (CatBoost 版本) ---
    _GPU_AVAILABLE = False
    try:
        logger.info("正在运行 CatBoost GPU 检测...")
        temp_model = cb.CatBoostClassifier(
            task_type='GPU',
            devices='0',
            iterations=1,
            logging_level='Silent'
        )
        temp_model.fit(np.array([[1,2],[3,4]]), np.array([0,1]), verbose=False)
        _GPU_AVAILABLE = True
        logger.info("CatBoost GPU 检测成功。CatBoost 将使用 GPU 加速。")
        if SHOW_VERBOSE_OUTPUT:
            print("CatBoost GPU 检测成功。CatBoost 将使用 GPU 加速。")
    except Exception as e:
        logger.warning(f"CatBoost GPU 检测失败: {e}")
        logger.warning("CatBoost 将回退到 CPU (task_type='CPU')。")
    # --- GPU 检测结束 ---


    # 2. 必要的预检查 (combined_df 已作为参数传入)
    if combined_df is None or combined_df.empty:
        raise RuntimeError("combined_df 为空，Optuna 无法运行。")

    # 3. 确保 future_return_5d 存在
    if 'future_return_5d' not in combined_df.columns:
        raise RuntimeError("无法创建 future_return_5d：数据准备阶段失败。")

    # 4. 固定时间序列划分
    train_size = int(len(combined_df) * TRAIN_TEST_SPLIT_RATIO)
    train_data_raw = combined_df.iloc[:train_size].copy()
    test_data_raw = combined_df.iloc[train_size:].copy()

    if train_data_raw.empty or test_data_raw.empty:
        raise RuntimeError("训练集或测试集为空，请检查 combined_df 的长度。")

    logger.info(f"固定训练集范围: {train_data_raw.index.min()} 到 {train_data_raw.index.max()} (N={len(train_data_raw)})")
    logger.info(f"固定测试集范围: {test_data_raw.index.min()} 到 {test_data_raw.index.max()} (N={len(test_data_raw)})")

    # 5. 数值特征白名单
    def get_feature_columns(df):
        # --- (修改 优化 2.2) 特征选择 ---
        # 排除目标、未来收益，以及所有原始 O/H/L/C/V 数据
        exclude = {
            'target_class_3', 'future_return_5d',
            # 移除原始价格/成交量数据，迫使模型依赖指标
            'open', 'high', 'low', 'close', 'volume', 'amount',
            # 排除辅助列 (v17: volatility_factor 已移除，作为特征加入训练)
            'stock_code_encoded'
        }
        # 排除所有以 'future_' 或 'shift' 开头的 (保持原逻辑)
        exclude |= {c for c in df.columns if c.lower().startswith('future_') or 'shift' in c.lower() or c in exclude}
        
        num_cols = df.select_dtypes(include=[np.number]).columns.to_list()
        feature_cols = [c for c in num_cols if c not in exclude]
        # --- (修改结束) ---
        return feature_cols

    # --- (v26 新增: 特征名称汉化映射) ---
    FEATURE_NAME_MAP = {
        'volatility_factor': '波动率因子',
        'rsi_percentile_60': 'RSI分位数(60日)',
        'vol_ma5': '成交量均线(5日)',
        'price_percentile_60': '价格分位数(60日)',
        'OBV': '能量潮(OBV)',
        'ATRr_14': '真实波幅(ATR)',
        'price_vol_corr': '价量相关性',
        'excess_return': '超额收益',
        'close_ratio_20': '收盘价乖离(20日)',
        'AD': '累积派发线(AD)',
        'high_low_ratio': '高低价差比',
        'DMN_14': '趋向指标(DMN)',
        'CMF_20': '资金流量指标(CMF)',
        'macd_signal': 'MACD信号线',
        'return_10d': '10日收益率',
        'return_5d': '5日收益率',
        'return_3d': '3日收益率',
        'return_2d': '2日收益率',
        'trend_strength': '趋势强度(ADX)',
        'ma_alignment': '均线排列得分',
        'bb_position': '布林带位置',
        'bias_sma20': '20日均线乖离',
        'bb_width_ratio': '布林带宽比',
        'open_close_gap': '开盘收盘缺口',
        'close_position': '收盘位置(K线内)',
        'volume_ma_ratio': '量比(20日)',
        'price_volume_momentum': '价量动量',
        'macd_histogram_change': 'MACD柱变化',
        'rsi_volume_cross': 'RSI-成交量交叉',
        'macd_trend_cross': 'MACD-趋势交叉',
        'bb_volume_cross': '布林-成交量交叉',
        'atr_volume_cross': 'ATR-成交量交叉',
        'rsi_stoch_cross': 'RSI-KD交叉',
        'days_since_low_20': '距近期低点天数',
        'consecutive_up_days': '连涨天数',
        'rank_pct_chg': '当日涨幅排名',
        'volatility_5d': '波动率(5日)',
        'volatility_10d': '波动率(10日)',
        'volatility_60d': '波动率(60日)',
        'rsi_vol_20': 'RSI波动率',
        'bias_vol_20': '乖离率波动率',
        'macd_h_vol_20': 'MACD柱波动率'
    }

    feature_columns = get_feature_columns(train_data_raw)
    if not feature_columns:
        raise RuntimeError("未找到用于训练的数值特征，请检查数据和特征选择逻辑。")
    print(f"使用 {len(feature_columns)} 个数值特征进行训练。")
    if SHOW_VERBOSE_OUTPUT:
        print(f"特征示例: {feature_columns[:10]}")

    # 6. create_targets - v18 支持动态阈值
    def create_targets(df, base_threshold, offset, use_dynamic=True):
        """创建目标变量，支持动态阈值"""
        dfc = df.copy()
        if 'future_return_5d' not in dfc.columns:
            raise ValueError("缺少 'future_return_5d' 列。")
        
        if use_dynamic and 'volatility_factor' in dfc.columns:
            # v23: 动态阈值 = 波动率系数(base_threshold) * 波动率因子(volatility_factor)
            # 假设 volatility_factor 为原始波动率 (如 0.02)
            vol_raw = dfc['volatility_factor'].fillna(0.02) # 填充默认波动率
            dynamic_threshold = base_threshold * vol_raw
            # 限制阈值范围，防止过大或过小 (0.5% ~ 5%)
            dynamic_threshold = dynamic_threshold.clip(0.005, 0.05)
            
            lower_bound = offset - dynamic_threshold
            upper_bound = offset + dynamic_threshold
        else:
            # 固定阈值
            lower_bound = offset - base_threshold
            upper_bound = offset + base_threshold
        
        dfc['target_class_3'] = np.nan
        valid = dfc['future_return_5d'].notna()
        dfc.loc[valid & (dfc['future_return_5d'] > upper_bound), 'target_class_3'] = 1
        dfc.loc[valid & (dfc['future_return_5d'] < lower_bound), 'target_class_3'] = -1
        # 处理中间类别
        dfc.loc[valid & dfc['target_class_3'].isna(), 'target_class_3'] = 0
        
        dfc = dfc.dropna(subset=['target_class_3']).copy()
        dfc['target_class_3'] = dfc['target_class_3'].astype('int16')
        return dfc

    # 7. CatBoost 参数函数 (CPU/GPU 自动适配) - v18 优化搜索空间
    def build_cb_params(trial):
        # 从 Config 读取搜索空间
        depth_range = CONFIG.get('optuna.search_space.depth', [6, 10])
        iter_range = CONFIG.get('optuna.search_space.iterations', [600, 1000])
        lr_range = CONFIG.get('optuna.search_space.learning_rate', [0.03, 0.2])
        
        params = {
            'loss_function': 'MultiClass',
            'eval_metric': 'MultiClass',
            'classes_count': 3,
            'logging_level': 'Silent',
            'random_seed': 42,
            # v18: 优化搜索空间 (v17建议: 缩小范围)
            'iterations': trial.suggest_int('iterations', iter_range[0], iter_range[1]),
            'learning_rate': trial.suggest_float('learning_rate', lr_range[0], lr_range[1], log=True),
            'depth': trial.suggest_int('depth', depth_range[0], depth_range[1]),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'bootstrap_type': 'Bernoulli',
            # v18 新增：正则化参数
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        }

        if _GPU_AVAILABLE:
            params['task_type'] = 'GPU'
            params['devices'] = '0'
            # v27.1 修复: 恢复独占显存模式 (0.95)，因为已回退到串行训练
            params['gpu_ram_part'] = 0.95
        else:
            params['task_type'] = 'CPU'
            params['colsample_bylevel'] = trial.suggest_float('colsample_bylevel', 0.5, 1.0)
            # v18 新增：特征采样 (CPU only)
            params['rsm'] = trial.suggest_float('rsm', 0.5, 1.0)

        return params

    # 8. Optuna 目标函数
    def objective(trial):
        logger.info(f"【试验 {trial.number + 1}】 开始...")
        print(f"\n{'='*60}")
        print(f"【试验 {trial.number + 1}】 开始优化试验")
        print(f"{'='*60}")
        trial_start_time = time.time()

        # 8.1. 动态标签 - v23 适配 A 股波动率
        try:
            # v23: 搜索波动率乘数 (vol_multiplier)，从 Config 读取范围
            vol_range = CONFIG.get('optuna.search_space.vol_multiplier', [0.5, 1.5])
            vol_multiplier = trial.suggest_float('vol_multiplier', vol_range[0], vol_range[1])
            
            offset_range = CONFIG.get('optuna.search_space.offset', [0.0, 0.0])
            offset = trial.suggest_float('offset', offset_range[0], offset_range[1]) if offset_range[0] != offset_range[1] else 0.0
            use_dynamic = True
            
            # 为了兼容 create_targets 签名，将 multiplier 传给 base_threshold
            threshold = vol_multiplier 
            
            logger.info(f"试验 {trial.number}: 波动率乘数={vol_multiplier:.4f}, 偏移量={offset:.4f}")
        except Exception as e:
            logger.error(f"试验 {trial.number}: 设置参数错误: {e}")
            return 0.0

        # 8.2. 创建标签 - v18 传递动态阈值参数
        try:
            train_df = create_targets(train_data_raw.copy(), threshold, offset, use_dynamic=use_dynamic)
        except Exception as e:
            logger.error(f"试验 {trial.number}: 创建目标时发生意外错误: {e}")
            return 0.0

        if train_df.empty:
            logger.warning(f"试验 {trial.number}: 创建目标后 train_df 为空。")
            return 0.0

        class_counts_raw = train_df['target_class_3'].value_counts()

        if SHOW_VERBOSE_OUTPUT:
            print(f"\n试验 {trial.number}: 训练集目标类别分布:")
            print(class_counts_raw.to_frame(name='count'))

        # 8.3. 特征与目标
        try:
            X = train_df[feature_columns].copy()
            # v19: 处理 Inf 和 NaN，避免 sklearn 报错
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            y = train_df['target_class_3'].values
            if X.empty or len(y) == 0:
                logger.warning(f"试验 {trial.number}: 分离后 X 或 y 为空。")
                return 0.0
        except Exception as e:
            logger.error(f"试验 {trial.number}: 分离特征/目标时出错: {e}")
            return 0.0

        label_mapping = {-1: 0, 0: 1, 1: 2}
        y_mapped = np.array([label_mapping[val] for val in y])

        # 8.4. 类别权重 - v18 添加 Focal Loss 风格权重
        try:
            classes = np.unique(y_mapped)
            if len(classes) < 2:
                logger.warning(f"试验 {trial.number}: 训练数据中仅有 {len(classes)} 个唯一类别。")
                return 0.0
            
            # 基础平衡权重
            class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_mapped)
            weight_map = dict(zip(classes, class_weights))
            
            # v18: Focal Loss 风格权重增强（对少数类给予更高权重）
            class_counts = pd.Series(y_mapped).value_counts()
            total = len(y_mapped)
            gamma = 2.0  # Focal Loss gamma 参数
            
            for cls in classes:
                cls_ratio = class_counts.get(cls, 0) / total
                focal_factor = (1 - cls_ratio) ** gamma
                weight_map[cls] = weight_map[cls] * (1 + focal_factor * 0.5)  # 温和增强
            
            # v26: 手动增强上涨类权重 (Class 1)
            if 1 in weight_map:
                weight_map[1] = weight_map[1] * 1.2
            
            # 归一化权重
            max_weight = max(weight_map.values())
            weight_map = {k: v / max_weight * 3 for k, v in weight_map.items()}
            
        except Exception as e:
            logger.error(f"试验 {trial.number}: 计算类别权重时出错: {e}")
            return 0.0

        # 8.5. CatBoost 参数
        try:
            cb_params = build_cb_params(trial)
        except Exception as e:
            logger.error(f"试验 {trial.number}: 构建 CatBoost 参数时出错: {e}")
            return 0.0

        # 8.6. TimeSeriesSplit 验证 (v17: 添加 future_gap 和 CV 内 Scaler)
        try:
            from sklearn.preprocessing import RobustScaler
            
            n_splits = TSC_N_SPLITS
            if len(X) < n_splits * 2:
                logger.warning(f"试验 {trial.number}: 数据不足以进行 {n_splits} 折划分。")
                return 0.0

            tscv = TimeSeriesSplit(n_splits=n_splits)
            f1_scores = []
            verbose_eval_setting = False

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                # --- v17: 添加 future_gap 间隔，防止标签偷看未来 ---
                # 训练集结束位置需要与验证集开始位置保持 FUTURE_GAP 的间隔
                gap = FUTURE_GAP
                if train_idx[-1] + gap >= val_idx[0]:
                    # 需要调整：从训练集末尾移除 gap 个样本
                    safe_train_end = val_idx[0] - gap
                    train_idx = train_idx[train_idx < safe_train_end]
                    if len(train_idx) < 50:  # 确保训练集足够大
                        logger.warning(f"试验 {trial.number}, 第 {fold+1} 折: 调整间隔后训练集太小。")
                        continue
                
                X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
                y_tr_mapped, y_val_mapped = y_mapped[train_idx], y_mapped[val_idx]

                if X_tr.empty or X_val.empty or len(y_tr_mapped) == 0 or len(y_val_mapped) == 0:
                    logger.warning(f"试验 {trial.number}, 第 {fold+1} 折: 训练或验证数据为空。")
                    continue

                # --- v17: CV 内部进行特征标准化，避免数据泄露 ---
                scaler = RobustScaler()
                X_tr_scaled = pd.DataFrame(
                    scaler.fit_transform(X_tr), 
                    index=X_tr.index, 
                    columns=X_tr.columns
                )
                X_val_scaled = pd.DataFrame(
                    scaler.transform(X_val), 
                    index=X_val.index, 
                    columns=X_val.columns
                )
                # --- 标准化结束 ---

                sw_tr = np.array([weight_map[val] for val in y_tr_mapped])

                model = cb.CatBoostClassifier(**cb_params)

                try:
                    # v19: 添加早停策略
                    model.fit(
                        X_tr_scaled, y_tr_mapped,
                        sample_weight=sw_tr,
                        eval_set=(X_val_scaled, y_val_mapped),
                        verbose=verbose_eval_setting,
                        early_stopping_rounds=50  # 连续 50 轮无改善则早停
                    )
                except Exception as e:
                    logger.error(f"试验 {trial.number}, 第 {fold+1} 折: 模型拟合期间出错: {e}")
                    return 0.0

                try:
                    y_val_pred_mapped = model.predict(X_val_scaled).flatten()
                    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
                    y_val_pred = pd.Series(y_val_pred_mapped).map(reverse_label_mapping).values
                    f1 = f1_score(y[val_idx], y_val_pred, average='macro', labels=[-1, 0, 1], zero_division=0)
                    f1_scores.append(f1)
                except Exception as e:
                    logger.error(f"试验 {trial.number}, 第 {fold+1} 折: 预测或评估期间出错: {e}")
                    f1_scores.append(0.0)

                # Pruning
                if f1_scores:
                    intermediate_value = np.mean(f1_scores)
                    trial.report(intermediate_value, fold)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                else:
                    trial.report(0.0, fold)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            if not f1_scores:
                logger.warning(f"试验 {trial.number}: 所有折叠均未记录 F1 分数。")
                return 0.0
            mean_f1 = float(np.mean(f1_scores))

        except Exception as e:
            logger.error(f"试验 {trial.number}: TimeSeriesSplit 或 CV 期间发生意外错误。", exc_info=True)
            return 0.0

        # 8.7. 平衡因子
        try:
            class_counts_final = pd.Series(y).value_counts()
            count_down = class_counts_final.get(-1, 0)
            count_flat = class_counts_final.get(0, 0)
            count_up = class_counts_final.get(1, 0)
            total_samples = count_down + count_flat + count_up
            if total_samples == 0:
                logger.warning(f"试验 {trial.number}: 平衡指标的总样本数为 0。")
                balance_metric = 0.1

            weight_down = 1.0
            weight_flat = 2.0
            weight_up = 1.0
            baseline = 1e-6

            max_count = max(count_down, count_flat, count_up)
            if max_count == 0:
                balance_metric = 0.1
            else:
                ratio_down = count_down / max_count
                ratio_flat = count_flat / max_count
                ratio_up = count_up / max_count

                if count_flat < total_samples * BALANCE_PENALTY_THRESHOLD:
                    balance_metric = baseline
                else:
                    total_weighted_ratios = (ratio_down * weight_down) + (ratio_flat * weight_flat * BALANCE_FLAT_WEIGHT_MULTIPLIER) + (ratio_up * weight_up)
                    total_weights = weight_down + (weight_flat * BALANCE_FLAT_WEIGHT_MULTIPLIER) + weight_up
                    balance_metric = total_weighted_ratios / total_weights

                balance_metric += baseline

            final_score = mean_f1 * balance_metric

            if SHOW_VERBOSE_OUTPUT:
                print(f"\n--- 试验 {trial.number} 摘要 ---")
                print(f"目标值 (Mean F1 * Balance): {final_score:.4f}")
                print(f"平均 Macro F1 分数 (across folds): {mean_f1:.4f}")
                print(f"平衡指标 (weighted): {balance_metric:.4f}")
                print(f"目标计数: 下跌={count_down}, 持平={count_flat}, 上涨={count_up}")
                print("---------------------------\n")
            
            # 总是输出简要进度
            elapsed = time.time() - trial_start_time
            print(f"[✓ 试验 {trial.number + 1} 完成] F1={mean_f1:.4f} | 平衡度={balance_metric:.4f} | 得分={final_score:.4f} | 耗时 {elapsed:.1f}s")

        except Exception as e:
            logger.error(f"试验 {trial.number}: 计算最终得分时出错: {e}")
            return 0.0

        return final_score

    # 9. 创建 study (启用 Pruner 和持久化)
    OPTUNA_DB_PATH = os.path.join(GDRIVE_SAVE_PATH, "optuna_catboost_study.db")
    OPTUNA_STUDY_NAME = CONFIG.get('optuna.study_name', 'catboost_stock_3class_v16')
    STORAGE_URL = f"sqlite:///{OPTUNA_DB_PATH}"

    if SHOW_VERBOSE_OUTPUT:
        print(f"--- Optuna 持久化已启用 (CatBoost) ---")
        print(f"数据库文件: {OPTUNA_DB_PATH}")
        print(f"Study 名称: {OPTUNA_STUDY_NAME}")

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=CONFIG.get('optuna.pruner_startup', 2),
        n_warmup_steps=1
    )

    study = optuna.create_study(
        direction='maximize',
        pruner=pruner,
        study_name=OPTUNA_STUDY_NAME,
        storage=STORAGE_URL,
        load_if_exists=True
    )

    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if SHOW_VERBOSE_OUTPUT:
        print(f"加载 Study 成功。已完成 {completed_trials} / {N_TRIALS} 次试验。")

    cpu_count = os.cpu_count() or 1
    # v27.1 修复: GPU 模式下强制串行 (n_jobs=1)，避免 "State == nullptr" 和 OOM 错误
    # CatBoost 对多进程 GPU 上下文支持不稳定，串行是最安全的方案。
    n_jobs_optuna = 1 if _GPU_AVAILABLE else min(CONFIG.get('optuna.n_jobs', 2), cpu_count)


    # --- (v17 新增：Optuna 回调函数 - 实时监控) ---
    def optuna_callback(study, trial):
        """每完成 10 次试验，自动保存优化历史图表"""
        if trial.number % 10 == 0:
            try:
                # 绘制优化历史
                fig_history = optuna.visualization.plot_optimization_history(study)
                fig_history.write_image(os.path.join(GDRIVE_SAVE_PATH, "optuna_history_realtime.png"))
                
                # 绘制参数重要性 (如果试验足够多)
                if len(study.trials) > 20:
                    fig_importance = optuna.visualization.plot_param_importances(study)
                    fig_importance.write_image(os.path.join(GDRIVE_SAVE_PATH, "optuna_param_importance_realtime.png"))
                
                if SHOW_VERBOSE_OUTPUT:
                    print(f"[回调] 试验 {trial.number}: 实时图表已更新。")
            except Exception as e:
                # 绘图失败不应中断训练
                logger.warning(f"[回调] 绘图失败: {e}")
    # --- (新增结束) ---

    if completed_trials < N_TRIALS:
        if SHOW_VERBOSE_OUTPUT:
            if _GPU_AVAILABLE:
                print(f"开始 Optuna 寻优 (目标总数={N_TRIALS}) - 检测到 GPU，n_jobs={n_jobs_optuna}")
            else:
                print(f"开始 Optuna 寻优 (目标总数={N_TRIALS}, n_jobs={n_jobs_optuna}) - 使用 CPU")
        else:
            print(f"开始 Optuna 寻优 (目标 {N_TRIALS} 次)...")

        try:
            study.optimize(
                objective,
                n_trials=N_TRIALS,
                show_progress_bar=False, 
                n_jobs=n_jobs_optuna,
                callbacks=[optuna_callback]  # 添加回调
            )
        except optuna.exceptions.TrialPruned:
            logger.info("发现被 Pruner 提前中止的试验。")
        except KeyboardInterrupt:
            print("\n[寻优被用户手动中断]")
        except Exception as e:
            logger.error(f"寻优过程中发生异常: {e}")
    else:
        print("所有试验已完成，无需再次运行 optimize()。跳过寻优步骤。")


    print("\n--- 寻优结束 ---")
    print(f"成功试验数: {len([t for t in study.trials if t.state.is_finished()])}")
    if study.trials and study.best_trial:
        print(f"最佳值: {study.best_value:.6f}")
        print(f"最佳参数:")
        print(json.dumps(study.best_params, indent=2))
    else:
        logger.warning("没有成功的试验或未找到最佳试验。")

    # --- 将所有试验结果保存到 CSV 表格 ---
    try:
        REPORT_FILENAME = os.path.join(GDRIVE_SAVE_PATH, "optuna_trials_report_catboost.csv")
        if SHOW_VERBOSE_OUTPUT:
            print(f"\n正在将所有试验结果保存到表格: {REPORT_FILENAME}")

        trials_df = study.trials_dataframe()
        trials_df = trials_df[trials_df['state'] != 'RUNNING'].copy()
        trials_df = trials_df.sort_values(by='value', ascending=False)
        trials_df.to_csv(REPORT_FILENAME, index=False)

        if SHOW_VERBOSE_OUTPUT:
            print(f"成功保存 {len(trials_df)} 条试验记录到 {REPORT_FILENAME}")
            print("\n表格预览 (前 5 名最佳试验):")
            print(trials_df.head(5))

    except Exception as e:
        logger.error(f"保存试验结果到 CSV 失败: {e}")


    # 10. 可视化 (始终显示，保存为文件)
    if study.trials and study.best_trial and study.best_value > 0.0:
        print("\n--- 寻优结果可视化 (正在保存到文件...) ---")
        
        img_path_hist = os.path.join(GDRIVE_SAVE_PATH, "opt_history_cb.png")
        img_path_imp = os.path.join(GDRIVE_SAVE_PATH, "opt_importance_cb.png")
        img_path_slice = os.path.join(GDRIVE_SAVE_PATH, "opt_slice_cb.png")
        
        try:
            # v24: Optuna 图表汉化
            fig_hist = vis.plot_optimization_history(study)
            fig_hist.update_layout(
                title="寻优过程历史",
                xaxis_title="试验次数",
                yaxis_title="目标值 (Macro F1)"
            )
            fig_hist.write_image(img_path_hist)
            print(f"已保存: {img_path_hist}")

            fig_imp = vis.plot_param_importances(study)
            fig_imp.update_layout(
                title="超参数重要性排行",
                xaxis_title="重要性权重",
                yaxis_title="超参数"
            )
            fig_imp.write_image(img_path_imp)
            print(f"已保存: {img_path_imp}")

            # v23: 更新切片参数以匹配新的搜索空间
            slice_params = [p for p in ['vol_multiplier', 'learning_rate', 'depth', 'iterations'] if p in study.best_params]
            if slice_params:
                fig_slice = vis.plot_slice(study, params=slice_params)
                fig_slice.update_layout(title="参数切片分析")
                fig_slice.write_image(img_path_slice)
                print(f"已保存: {img_path_slice}")
            else:
                logger.warning("跳过切片图：在 best_params 中未找到默认参数。")
        
        except ImportError as ie:
             logger.error(f"可视化失败: {ie}。请确保 'kaleido' 已安装 (pip install kaleido)。")
        except Exception as e:
            logger.error(f"可视化失败: {e}，尝试保存为静态图像。")
           
    else:
        logger.warning("没有成功的试验 (best_value=0.0)，跳过可视化。")


    # 11. 可选：后验评估 (CatBoost 版本) (始终显示)
    if study.best_params:
        print("\n--- 在固定测试集上进行后验评估 ---")
        best_params = study.best_params.copy()
        
        # v23: 适配波动率乘数参数
        threshold_best = 1.0 # 默认乘数
        offset_best = 0.0
        
        if 'vol_multiplier' in best_params: 
            threshold_best = best_params['vol_multiplier']
            print(f"使用最佳波动率乘数: {threshold_best:.4f}")
        elif 'threshold' in best_params:
            threshold_best = best_params['threshold']
            
        if 'offset' in best_params: offset_best = best_params['offset']

        # 注意: create_targets 现在期望 base_threshold 是 multiplier (如果 use_dynamic=True)
        try:
            # 强制开启 use_dynamic=True 以匹配 v23 逻辑
            train_final = create_targets(train_data_raw.copy(), threshold_best, offset_best, use_dynamic=True)
            test_final = create_targets(test_data_raw.copy(), threshold_best, offset_best, use_dynamic=True)
            print(f"后验评估: 已应用动态阈值 (Multiplier={threshold_best:.4f})")
        except Exception as e:
            logger.error(f"创建最终训练/测试目标时出错: {e}")
            train_final = pd.DataFrame()
            test_final = pd.DataFrame()

        if not train_final.empty and not test_final.empty:
            try:
                X_train_final = train_final[feature_columns].fillna(0.0)
                y_train_final = train_final['target_class_3'].values
                X_test_final = test_final[feature_columns].fillna(0.0)
                y_test_final = test_final['target_class_3'].values

                if X_train_final.empty or X_test_final.empty or len(y_train_final) == 0 or len(y_test_final) == 0:
                    logger.warning("准备后验评估数据后，最终训练/测试数据为空。")
                else:
                    label_mapping = {-1: 0, 0: 1, 1: 2}
                    y_train_final_mapped = np.array([label_mapping[val] for val in y_train_final])
                    y_test_final_mapped = np.array([label_mapping[val] for val in y_test_final])

                    try:
                        fixed_trial_params_keys = [
                            'iterations', 'learning_rate', 'depth',
                            'subsample', 'colsample_bylevel',
                            'l2_leaf_reg', 'bootstrap_type'
                        ]
                        fixed_trial_params = {k: v for k, v in best_params.items() if k in fixed_trial_params_keys}

                        cb_params_post = fixed_trial_params.copy()
                        cb_params_post.update({
                            'loss_function': 'MultiClass',
                            'eval_metric': 'MultiClass',
                            'classes_count': 3,
                            'random_seed': 42,
                            'logging_level': 'Silent',
                            'bootstrap_type': 'Bernoulli'
                        })

                        if _GPU_AVAILABLE:
                            cb_params_post['task_type'] = 'GPU'
                            cb_params_post['devices'] = '0'
                            # v27.1: 提高后验评估显存配额
                            cb_params_post['gpu_ram_part'] = 0.95
                            cb_params_post.pop('colsample_bylevel', None)
                            if SHOW_VERBOSE_OUTPUT: print("后验评估：使用 GPU (gpu_ram_part=0.95)。")
                        else:
                            cb_params_post['task_type'] = 'CPU'
                            if SHOW_VERBOSE_OUTPUT: print("后验评估：使用 CPU。")

                    except Exception as e:
                        logger.error(f"构建后验评估 CatBoost 参数时出错: {e}，跳过后验评估。")
                        cb_params_post = None

                    if cb_params_post:
                        model_final = cb.CatBoostClassifier(**cb_params_post)
                        try:
                            classes_final_train = np.unique(y_train_final_mapped)
                            if len(classes_final_train) < 2:
                                logger.warning("最终训练数据中类别不足，无法计算类别权重。")
                                sw_train_final = None
                            else:
                                cw = compute_class_weight(class_weight='balanced', classes=classes_final_train, y=y_train_final_mapped)
                                wm = dict(zip(classes_final_train, cw))
                                sw_train_final = np.array([wm[val] for val in y_train_final_mapped])
                        except Exception as e:
                            logger.error(f"计算最终训练类别权重时出错: {e}")
                            sw_train_final = None

                        try:
                            if SHOW_VERBOSE_OUTPUT: print("正在拟合最终模型以进行后验评估...")
                            # v17 修正: 添加 eval_set 以绘制验证集曲线
                            model_final.fit(
                                X_train_final, y_train_final_mapped, 
                                sample_weight=sw_train_final,
                                eval_set=(X_test_final, y_test_final_mapped),
                                verbose=False
                            )
                            if SHOW_VERBOSE_OUTPUT: print("最终模型拟合完成。")

                            y_test_pred_final_mapped = model_final.predict(X_test_final).flatten()

                            reverse_label_mapping = {v: k for k, v in label_mapping.items()}
                            y_test_pred_final = pd.Series(y_test_pred_final_mapped).map(reverse_label_mapping).values

                            test_f1 = f1_score(y_test_final, y_test_pred_final, average='macro', labels=[-1,0,1], zero_division=0)
                            print(f"后验测试集 macro F1: {test_f1:.4f}")

                            print("\n后验测试集分类报告:")
                            print(classification_report(y_test_final, y_test_pred_final, labels=[-1, 0, 1], target_names=['下跌 (-1)', '持平 (0)', '上涨 (1)'], zero_division=0))
                            print("\n后验测试集混淆矩阵:")
                            cm_final = confusion_matrix(y_test_final, y_test_pred_final, labels=[-1, 0, 1])
                            
                            cm_img_path = os.path.join(GDRIVE_SAVE_PATH, "confusion_matrix_post_eval.png")
                            plt.figure(figsize=(8, 6))
                            # v24: 混淆矩阵汉化
                            sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues',
                                        xticklabels=['预测下跌', '预测持平', '预测上涨'],
                                        yticklabels=['实际下跌', '实际持平', '实际上涨'])
                            plt.xlabel('预测标签')
                            plt.ylabel('真实标签')
                            plt.title('后验评估混淆矩阵 (三分类) - CatBoost')
                            plt.savefig(cm_img_path)
                            print(f"混淆矩阵已保存到: {cm_img_path}")
                            plt.close()

                            # --- (新增 优化 1.2) 保存最终模型和参数 ---
                            print("正在保存最终模型和参数...")
                            model_path = os.path.join(GDRIVE_SAVE_PATH, 'catboost_final_model.cbm')
                            model_final.save_model(model_path)
                            print(f"最终模型已保存到: {model_path}")
                            
                            # v23: 动态阈值模式下，bounds 是 per-sample 的，不再保存全局值
                            final_params = {
                                'vol_multiplier_best': threshold_best,  # v23: 这里存的是波动率乘数
                                'offset_best': offset_best,
                                'mode': 'dynamic_volatility',
                                'note': 'Dynamic thresholding enabled. Bounds are per-sample based on volatility.'
                            }
                            params_path = os.path.join(GDRIVE_SAVE_PATH, 'final_model_params.json')
                            with open(params_path, 'w') as f:
                                json.dump(final_params, f, indent=2)
                            print(f"模型阈值参数已保存到: {params_path}")
                            
                            # --- (v17 新增：高级模型评估与可视化) ---
                            print("正在生成高级评估报告...")
                            
                            # 1. 导出特征重要性
                            try:
                                feature_importance = model_final.get_feature_importance()
                                fi_df = pd.DataFrame({'feature': feature_columns, 'importance': feature_importance})
                                fi_df = fi_df.sort_values(by='importance', ascending=False)
                                fi_path = os.path.join(GDRIVE_SAVE_PATH, 'catboost_feature_importance.csv')
                                fi_df.to_csv(fi_path, index=False)
                                print(f"特征重要性已保存到: {fi_path}")
                                
                                # v24: 特征重要性图汉化 (应用映射)
                                plt.figure(figsize=(10, 8))
                                # 创建用于绘图的汉化 DataFrame
                                fi_df_plot = fi_df.head(20).copy()
                                fi_df_plot['feature_cn'] = fi_df_plot['feature'].map(FEATURE_NAME_MAP).fillna(fi_df_plot['feature'])
                                
                                sns.barplot(x='importance', y='feature_cn', data=fi_df_plot)
                                plt.title('CatBoost 特征重要性前 20 名')
                                plt.xlabel('重要性')
                                plt.ylabel('特征名称')
                                plt.tight_layout()
                                plt.savefig(os.path.join(GDRIVE_SAVE_PATH, 'catboost_feature_importance.png'))
                                plt.close()
                            except Exception as e:
                                logger.error(f"特征重要性导出失败: {e}")

                            # 2. 绘制学习曲线 (Loss) - v24 汉化
                            try:
                                evals_result = model_final.get_evals_result()
                                if evals_result:
                                    plt.figure(figsize=(10, 6))
                                    for metric in evals_result['learn'].keys():
                                        plt.plot(evals_result['learn'][metric], label=f'训练集 {metric}')
                                        if 'validation' in evals_result:
                                            plt.plot(evals_result['validation'][metric], label=f'验证集 {metric}')
                                    plt.title('训练与验证学习曲线')
                                    plt.xlabel('迭代次数')
                                    plt.ylabel('损失值')
                                    plt.legend()
                                    plt.grid(True)
                                    plt.savefig(os.path.join(GDRIVE_SAVE_PATH, 'catboost_learning_curve.png'))
                                    plt.close()
                            except Exception as e:
                                logger.error(f"学习曲线绘制失败: {e}")

                            # 3. 详细分类报告 CSV (修正变量名: y_pred_final -> y_test_pred_final)
                            try:
                                report_dict = classification_report(
                                    y_test_final, 
                                    y_test_pred_final,  # <--- 修正点 1
                                    labels=[-1, 0, 1],
                                    target_names=['Down', 'Flat', 'Up'], 
                                    output_dict=True, 
                                    zero_division=0
                                )
                                pd.DataFrame(report_dict).transpose().to_csv(
                                    os.path.join(GDRIVE_SAVE_PATH, 'classification_report_detailed.csv')
                                )
                                print("详细分类报告已保存。")
                            except Exception as e:
                                logger.error(f"分类报告保存失败: {e}")

                            # 4. 概率分布图 - v24 汉化
                            try:
                                # CatBoost 可以直接处理未缩放的 X_test_final
                                y_proba = model_final.predict_proba(X_test_final)
                                
                                plt.figure(figsize=(10, 6))
                                for i, label in enumerate(['下跌概率', '持平概率', '上涨概率']):
                                    sns.kdeplot(y_proba[:, i], label=label, fill=True, alpha=0.3)
                                
                                plt.title('预测概率分布 (测试集)')
                                plt.xlabel('概率值')
                                plt.ylabel('密度')
                                plt.legend()
                                plt.savefig(os.path.join(GDRIVE_SAVE_PATH, 'prediction_proba_distribution.png'))
                                plt.close()
                                print("概率分布图已保存。")
                            except Exception as e:
                                logger.error(f"概率分布图绘制失败: {e}")
                            
                            print("高级评估报告生成完成。")
                            # --- (新增结束) ---

                        except Exception as e:
                            logger.error(f"后验评估失败：{e}")
            except Exception as e:
                logger.error(f"Error preparing data for post-evaluation: {e}")
        else:
            logger.warning("后验评估跳过：训练或测试目标集为空（可能阈值导致）。")
    else:
        logger.warning("最佳 trial 中未包含 threshold/offset，跳过后验评估。")

    print("\n--- 优化单元完成 (CatBoost V6 安静模式) ---")



# --- 步骤 4: 主执行逻辑 ---
if __name__ == '__main__':
    # --- 全局参数 ---
    # MAX_FILES_TO_PROCESS = 3000 # 减少数量以便快速测试
    MAX_FILES_TO_PROCESS = CONFIG.get('data.max_files_to_process', 10000) # 从配置读取，默认 10000


    # --- (!! 修正 v15.1 !!) ---
    # 在主逻辑中定义 SHOW_VERBOSE_OUTPUT
    # True: 显示详细的 DataFrame 预览和 Optuna 内部日志
    # False: 保持安静
    SHOW_VERBOSE_OUTPUT = False 
    # --- (修正结束) ---

    total_start_time = time.time()

    # 步骤 1: 环境设置
    print("\n--- 步骤 1: 自动环境检测与路径配置 ---")
    current_env = get_environment()
    paths_config = setup_paths(current_env)

    print(f"检测到环境: {current_env.value}")
    if paths_config:
        for key, value in paths_config.items():
            if value:
                print(f"{key}: {value}")
    else:
        print("路径配置失败。")
        sys.exit(1)


    # --- (v20 新增) 交互式训练模式选择 ---
    import datetime
    
    # 定义结果根目录 (在 GDrive/Workspace 下的 optuna_results 文件夹)
    base_results_root = os.path.join(paths_config['GDRIVE_BASE_PATH'], 'optuna_results')
    os.makedirs(base_results_root, exist_ok=True)
    
    print(f"\n--- 训练模式配置 (v20) ---")
    print(f"结果存储根目录: {base_results_root}")
    print("请选择模式:")
    print("  [n] 新训练 (New)      - 创建带时间戳的新目录 (例如 run_2023...)")
    print("  [c] 继续训练 (Continue) - 选择已有目录继续优化")
    
    try:
        mode = input("请输入模式 (n/c) [默认 n]: ").strip().lower()
    except (EOFError, OSError):
        mode = 'n'
    if not mode or mode not in ['n', 'c']: 
        mode = 'n'
        print(">> 输入无效，默认使用新训练模式。")
    
    current_run_path = ""
    
    if mode == 'c':
        try:
            dirs = [d for d in os.listdir(base_results_root) if os.path.isdir(os.path.join(base_results_root, d))]
            dirs.sort(key=lambda x: os.path.getmtime(os.path.join(base_results_root, x)), reverse=True)
        except Exception:
            dirs = []
            
        if not dirs:
            print(">> 未找到历史记录，自动切换为新训练模式。")
            mode = 'n'
        else:
            print("\n历史训练记录 (按时间倒序):")
            for i, d in enumerate(dirs[:10]):
                print(f" [{i}] {d}")
            
            try:
                idx_str = input(f"请选择序号 (0-{min(len(dirs)-1, 9)}) [默认 0]: ").strip()
            except EOFError:
                idx_str = '0'
            if not idx_str: idx_str = '0'
            
            try:
                idx = int(idx_str)
                if 0 <= idx < len(dirs):
                    current_run_path = os.path.join(base_results_root, dirs[idx])
                    print(f">> 已选择继续训练目录: {dirs[idx]}")
                else:
                    print(">> 序号无效，默认使用最新记录。")
                    current_run_path = os.path.join(base_results_root, dirs[0])
            except:
                print(">> 输入无效，默认使用最新记录。")
                current_run_path = os.path.join(base_results_root, dirs[0])
    
    if mode == 'n':
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        current_run_path = os.path.join(base_results_root, f"run_{timestamp}")
        os.makedirs(current_run_path, exist_ok=True)
        print(f">> 已创建新训练目录: run_{timestamp}")
        
    print(f"本次运行结果将保存至: {current_run_path}")
    # --- (选择逻辑结束) ---


    # --- 优化8: 参数哈希缓存系统 ---
    combined_df = None
    cache_enabled = CONFIG.get('data.cache_enabled', True)
    
    # 计算特征工程参数哈希
    feature_config = str(CONFIG.config.get('features', {}))
    config_hash = hashlib.md5(feature_config.encode()).hexdigest()[:8]
    cache_path = os.path.join(
        paths_config['GDRIVE_BASE_PATH'],
        f'feature_cache_{MAX_FILES_TO_PROCESS}_{config_hash}.parquet'
    )
    print(f"检查缓存文件: {cache_path}")

    if cache_enabled and os.path.exists(cache_path):
        # v22: 增加交互式缓存确认，防止代码修改后误用旧缓存
        use_cache = 'y'
        print(f"\n[!] 发现已存在的特征缓存: {os.path.basename(cache_path)}")
        print("    如果你修改了特征工程代码(apply_technical_indicators)，请选择 'n' 以重新计算。")
        try:
            use_cache = input("    是否加载此缓存? [y/n] (默认 y): ").strip().lower()
        except (EOFError, OSError):
            use_cache = 'y'

        if use_cache != 'n':
            try:
                print("✓ 正在加载缓存...")
                combined_df = pd.read_parquet(cache_path)
                print(f"✓ 从缓存加载数据完成。形状: {combined_df.shape}")
            except Exception as e:
                print(f"⚠ 加载缓存失败: {e}。将重新生成。")
                combined_df = None
        else:
            print(">> 用户选择跳过缓存，强制重新计算特征。")
            combined_df = None
            
    elif not cache_enabled:
        print("✓ 缓存已禁用，执行完整数据准备流程。")
    else:
        print("✓ 未发现缓存，执行完整数据准备流程。")

    if combined_df is None:
        # 步骤 2: 数据准备
        data_available = prepare_data_files(current_env, paths_config)

        if data_available:
            # 步骤 3: 特征工程 (DATA_DIR 指向 /content/data/)
            list_of_dfs = run_feature_engineering(paths_config['DATA_DIR'], MAX_FILES_TO_PROCESS)

            # 步骤 4: 后处理 (包含内存优化)
            if list_of_dfs:
                combined_df = post_process_data(list_of_dfs)
                
                # 优化8: 写入缓存
                if cache_enabled and combined_df is not None and not combined_df.empty:
                    try:
                        print(f"✓ 正在写入缓存到: {cache_path}")
                        combined_df.to_parquet(cache_path)
                        print("✓ 缓存写入成功。")
                    except Exception as e:
                        print(f"⚠ 缓存写入失败: {e}")
    # --- (缓存逻辑结束) ---


    # 步骤 5: 最终结果展示 (并启动 Optuna)
    if combined_df is not None and not combined_df.empty:
        print("\n--- 步骤 5: 处理后的总数据表已准备就绪 ---")
        print(f"最终数据形状: {combined_df.shape}")
        
        # --- (!! 修正 v15.1 !!) ---
        # 此处现在可以正确访问 SHOW_VERBOSE_OUTPUT
        if SHOW_VERBOSE_OUTPUT:
            print("\n--- 最终 combined_df 特征列表 ---")
            print(combined_df.columns.tolist())
            print("\n--- combined_df 数据前 5 行 ---")
            print(combined_df.head())

        total_end_time_part1 = time.time()
        print(f"\n*** 数据准备完成。耗时: {total_end_time_part1 - total_start_time:.2f} 秒。 ***")

        # --- (新增) 调用 Optuna 寻优
        # 将 GDrive 路径传递给 Optuna 用于保存
        run_optuna_study(combined_df, current_run_path)
        # --- (新增结束) ---

    else:
        print("\n数据处理流程结束，但未生成最终的 DataFrame。无法启动 Optuna。")

    
    total_end_time = time.time()
    print(f"\n*** 脚本总执行完毕。总耗时: {total_end_time - total_start_time:.2f} 秒。 ***")