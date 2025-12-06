"""
数据获取模块（原 shared/data_fetcher.py）
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging

logger = logging.getLogger(__name__)

def get_xcsc_pro():
    try:
        import xcsc_tushare as ts
        server = os.environ.get("TS_SERVER", "http://tushare.xcsc.com:7172")
        env = os.environ.get("TS_ENV", "prd")
        pro = ts.pro_api(env=env, server=server)
        return pro
    except ImportError:
        logger.warning("xcsc_tushare 未安装，部分数据获取功能不可用")
        return None

def fetch_margin_detail(pro, ts_code: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    # 这里省略实现，直接复制 shared/data_fetcher.py 的原始内容即可
    pass
