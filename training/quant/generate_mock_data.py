import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# 模拟 main.py 的输出目录
DATA_DIR = '../../data'
os.makedirs(DATA_DIR, exist_ok=True)

def generate_mock_stock(ts_code, days=200):
    end_date = datetime.now()
    dates = [(end_date - timedelta(days=i)).strftime('%Y%m%d') for i in range(days)]
    dates.reverse() # main.py output might be sorted or not, but let's generate sorted first
    
    # 随机生成价格数据
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(days))
    high = close + np.random.rand(days) * 2
    low = close - np.random.rand(days) * 2
    open_ = close + np.random.randn(days) * 0.5
    volume = np.random.randint(1000, 10000, days) * 100
    
    df = pd.DataFrame({
        'trade_date': dates,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        # main.py 还会生成一些指标，虽然 predict_strategy 会重新计算，但为了模拟真实情况，我们加上
        'ma5': pd.Series(close).rolling(5).mean(),
        'ma10': pd.Series(close).rolling(10).mean(),
        'ma20': pd.Series(close).rolling(20).mean(),
        'volatility_10': pd.Series(close).rolling(10).std(),
        'vol_ma5': pd.Series(volume).rolling(5).mean(),
        'momentum_5': pd.Series(close).pct_change(5),
        'rsi14': np.random.rand(days) * 100, # 模拟值
        'macd': np.random.randn(days),
        'macd_signal': np.random.randn(days)
    })
    
    # main.py 保存时可能是乱序的 (iloc[::-1])，我们模拟一下
    # 但 predict_strategy.py 会负责排序
    
    file_path = os.path.join(DATA_DIR, f'{ts_code}.parquet')
    df.to_parquet(file_path)
    print(f"Generated {file_path}")

if __name__ == '__main__':
    generate_mock_stock('000001.SZ')
    generate_mock_stock('600519.SH')
