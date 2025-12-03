import pandas as pd
import pandas_ta as ta
import numpy as np

def apply_technical_indicators(df):
    """
    提取技术指标计算逻辑 (v25 优化版 - 预测专用)
    保持与训练阶段完全一致的特征生成逻辑。
    """
    # 确保数据按日期升序排列
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    # 计算基础技术指标
    df.ta.sma(length=5, append=True)
    df.ta.sma(length=10, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.bbands(append=True)
    df.ta.adx(append=True)
    df.ta.atr(append=True)
    df.ta.stoch(append=True)
    df.ta.cmf(append=True)
    
    # --- (v25 新增) 高效比率特征 ---
    # 乖离率
    df['bias_sma20'] = (df['close'] - df['SMA_20']) / (df['SMA_20'] + 1e-8)
    
    # 布林带宽比率
    bb_width = df['BBU_5_2.0_2.0'] - df['BBL_5_2.0_2.0']
    df['bb_width_ratio'] = bb_width / (df['SMA_20'] + 1e-8)
    
    # --- (v16：原始价格/量特征) ---
    df['close_ratio_20'] = df['close'] / df['close'].shift(20)
    df['high_low_ratio'] = (df['high'] - df['low']) / (df['close'] + 1e-8)
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['open_close_gap'] = (df['close'] - df['open']) / (df['open'] + 1e-8)
    
    # 量能特征
    df['volume_ma_ratio'] = df['volume'] / (df['volume'].rolling(20, min_periods=1).mean() + 1e-8)
    df['price_volume_momentum'] = df['close'].pct_change() * df['volume']
    
    # 多周期收益率
    for lag in [2, 3, 5, 10]:
        df[f'return_{lag}d'] = df['close'].pct_change(lag)
    
    # --- 量能指标 ---
    df.ta.obv(append=True)
    df.ta.ad(append=True)
    
    # --- (v18：分位数特征) ---
    df['rsi_percentile_60'] = df['RSI_14'].rolling(60, min_periods=20).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8), raw=False
    )
    
    rolling_low_60 = df['low'].rolling(60, min_periods=20).min()
    rolling_high_60 = df['high'].rolling(60, min_periods=20).max()
    df['price_percentile_60'] = (df['close'] - rolling_low_60) / (rolling_high_60 - rolling_low_60 + 1e-8)
    
    df['volume_percentile_60'] = df['volume'].rolling(60, min_periods=20).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8), raw=False
    )
    
    # MACD 柱状图变化率
    df['macd_histogram_change'] = df['MACDh_12_26_9'].diff()
    
    # --- 趋势强度特征 ---
    df['trend_strength'] = df['ADX_14'] / 100.0
    
    # 均线排列
    df['ma_alignment'] = (
        (df['SMA_5'] > df['SMA_10']).astype(int) +
        (df['SMA_10'] > df['SMA_20']).astype(int)
    ) / 2.0
    
    # 布林带位置
    bb_middle = (df['BBU_5_2.0_2.0'] + df['BBL_5_2.0_2.0']) / 2
    df['bb_position'] = (df['close'] - bb_middle) / (bb_width / 2 + 1e-8)
    df['bb_position'] = df['bb_position'].clip(-2, 2)
    
    # --- (v19：交叉特征) ---
    df['close_pct_change'] = df['close'].pct_change()
    df['rsi_volume_cross'] = (df['RSI_14'] / 50 - 1) * df['volume_ma_ratio']
    
    df['macd_trend_cross'] = df['MACDh_12_26_9'] * df['trend_strength']
    df['bb_volume_cross'] = df['bb_position'] * df['volume_percentile_60']
    df['atr_volume_cross'] = df['ATRr_14'] * df['volume_ma_ratio']
    df['rsi_stoch_cross'] = (df['RSI_14'] / 100) * (df['STOCHk_14_3_3'] / 100)
    
    # [新增] 计算波动率因子 (Top 2 特征，用于动态阈值)
    # 使用 20日滚动标准差，确保与主脚本逻辑一致
    df['volatility_factor'] = df['close'].pct_change().rolling(window=20).std()

    # --- (v27 新增: 多周期波动率与周期性特征波动率) ---
    # 1. 不同周期的价格波动率 (捕捉短期和长期市场情绪)
    df['volatility_5d'] = df['close'].pct_change().rolling(window=5).std()
    df['volatility_10d'] = df['close'].pct_change().rolling(window=10).std()
    df['volatility_60d'] = df['close'].pct_change().rolling(window=60).std()
    
    # 2. 周期性指标的波动率 (衡量指标本身的稳定性)
    # RSI 的波动率 (衡量动量的稳定性: 波动大=情绪不稳, 波动小=趋势稳定)
    df['rsi_vol_20'] = df['RSI_14'].rolling(window=20).std()
    
    # 乖离率的波动率 (衡量价格回归均值的剧烈程度)
    df['bias_vol_20'] = df['bias_sma20'].rolling(window=20).std()
    
    # MACD 柱状图的波动率 (衡量趋势变化的剧烈程度)
    df['macd_h_vol_20'] = df['MACDh_12_26_9'].rolling(window=20).std()

    # --- (v26 新增: 趋势特征) ---
    # 1. 距离近期最低点的天数 (20日)
    # 使用 argmin 计算窗口内最小值的索引位置 (0 表示今天就是最低点)
    df['days_since_low_20'] = df['low'].rolling(20, min_periods=5).apply(
        lambda x: len(x) - 1 - np.argmin(x), raw=True
    )
    
    # 2. 连涨天数
    # 定义上涨: 收盘价 > 前一日收盘价
    s = (df['close'] > df['close'].shift(1)).astype(int)
    # 计算连续上涨天数 (下跌则重置为0)
    df['consecutive_up_days'] = s * (s.groupby((s != s.shift()).cumsum()).cumcount() + 1)

    # --- (v32 新增: 统一补全缺失特征) ---
    # 将分散在 train.py 和 daily_run.py 中的特征逻辑统一至此
    
    # 1. 滞后特征
    if 'close_lag1' not in df.columns:
        df['close_lag1'] = df['close'].shift(1)
        
    # 2. 量能变化
    if 'volume_change' not in df.columns:
        df['volume_change'] = df['volume'].pct_change()
        
    # 3. 价量相关性 (过去20天)
    if 'price_vol_corr' not in df.columns:
        df['price_vol_corr'] = df['close'].rolling(20, min_periods=1).corr(df['volume'])
        
    # 4. 超额收益 (简单近似)
    # 注意: train.py 中有更复杂的 excess_return 计算 (减去大盘)，这里仅作为 fallback
    if 'excess_return' not in df.columns:
        df['excess_return'] = df['close'].pct_change()

    # --- (v25) 删除冗余/低重要性列 ---
    # 修复: pandas_ta 可能生成小写列名 (ma5, ma10) 导致无法删除
    # 显式添加常见的小写变体
    # cols_to_drop = [
    #     'SMA_5', 'SMA_10', 'SMA_20',
    #     'ma5', 'ma10', 'ma20', 'SMA_5.0', 'SMA_10.0', 'SMA_20.0', # 常见变体
    #     'BBU_5_2.0_2.0', 'BBL_5_2.0_2.0', 'BBM_5_2.0_2.0', 'BBP_5_2.0_2.0',
    #     'MACD_12_26_9', 'MACDs_12_26_9'
    # ]
    
    # 动态查找并删除所有以 SMA_ 或 ma (后跟数字) 开头的列，如果它们不在保留列表中
    # 这里简单起见，直接使用扩展的列表
    # existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    # df.drop(columns=existing_cols_to_drop, inplace=True, errors='ignore')
    
    # 兼容性修复: 重命名 pandas_ta 生成的列以匹配旧模型 (ma5, ma10, ma20)
    if 'SMA_5' in df.columns: df['ma5'] = df['SMA_5']
    if 'SMA_10' in df.columns: df['ma10'] = df['SMA_10']
    if 'SMA_20' in df.columns: df['ma20'] = df['SMA_20']
    
    return df
