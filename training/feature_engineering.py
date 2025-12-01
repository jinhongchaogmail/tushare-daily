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

    # --- (v25) 删除冗余/低重要性列 ---
    cols_to_drop = [
        'SMA_5', 'SMA_10', 'SMA_20',
        'BBU_5_2.0_2.0', 'BBL_5_2.0_2.0', 'BBM_5_2.0_2.0', 'BBP_5_2.0_2.0',
        'MACD_12_26_9', 'MACDs_12_26_9'
    ]
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=existing_cols_to_drop, inplace=True, errors='ignore')
    
    return df
