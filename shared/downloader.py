"""
数据批量下载模块 (v37 更新)

v37 新增:
- 融资融券数据 (margin_detail) 批量获取
- 龙虎榜数据 (top_list) 按日期获取
- 大宗交易数据 (block_trade) 按日期获取
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def fetch_for_ticker(pro, ts_code: str, start_date: str, fields_daily: str, fields_daily_basic: str, fields_moneyflow: str, fields_margin: str = None, timeout_moneyflow=10):
    """Fetch daily, daily_basic, moneyflow, margin_detail for a single ts_code in parallel.

    Returns a dict with keys 'daily','daily_basic','moneyflow','margin' mapped to DataFrames (or empty DataFrame on failure).
    """
    results: Dict[str, object] = {}

    def f_daily():
        return pro.daily(ts_code=ts_code, start_date=start_date, end_date="", fields=fields_daily)

    def f_daily_basic():
        return pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date="", fields=fields_daily_basic)

    def f_moneyflow():
        return pro.moneyflow(ts_code=ts_code, start_date=start_date, end_date="", fields=fields_moneyflow)

    def f_margin():
        if fields_margin:
            return pro.margin_detail(ts_code=ts_code, start_date=start_date, end_date="", fields=fields_margin)
        return None

    # Use a small threadpool to run the requests in parallel for this ticker
    max_workers = 4 if fields_margin else 3
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(f_daily): 'daily',
            ex.submit(f_daily_basic): 'daily_basic',
            ex.submit(f_moneyflow): 'moneyflow'
        }
        if fields_margin:
            futs[ex.submit(f_margin)] = 'margin'

        for fut in as_completed(futs):
            name = futs[fut]
            try:
                if name in ('moneyflow', 'margin'):
                    # Respect timeout
                    val = fut.result(timeout=timeout_moneyflow)
                else:
                    val = fut.result()
                results[name] = val
            except Exception:
                # return empty DataFrame on any failure for graceful degradation
                try:
                    import pandas as pd
                    results[name] = pd.DataFrame()
                except Exception:
                    results[name] = None

    # Ensure keys exist
    for k in ('daily', 'daily_basic', 'moneyflow', 'margin'):
        if k not in results:
            try:
                import pandas as pd
                results[k] = pd.DataFrame()
            except Exception:
                results[k] = None

    return results


def fetch_batch(pro, ts_codes, start_date: str, fields_daily: str, fields_daily_basic: str, fields_moneyflow: str, fields_margin: str = None, timeout=30, max_codes_per_request: int = 10):
    """Fetch multiple ts_codes in (safe) sub-requests and merge results.

    Some servers limit the number of ts_code values accepted in a single request.
    To be robust we split `ts_codes` into chunks of at most `max_codes_per_request`.
    
    注意: XCSC 服务器对不同接口的批量查询支持不同:
    - daily: 支持批量查询 (ts_code 可以逗号分隔)
    - daily_basic, moneyflow, margin_detail: 不支持批量查询，需要逐只获取

    Returns a dict: {'daily': {ts: df,...}, 'daily_basic': {...}, 'moneyflow': {...}, 'margin': {...}}
    """
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    if not ts_codes:
        return {'daily': {}, 'daily_basic': {}, 'moneyflow': {}, 'margin': {}}

    results = {'daily': {}, 'daily_basic': {}, 'moneyflow': {}, 'margin': {}}

    # helper to split df by ts_code
    def split_df_by_ts(df):
        out = {}
        if df is None or df.empty:
            return out
        if 'ts_code' not in df.columns:
            out['_all'] = df
            return out
        for code, g in df.groupby('ts_code'):
            out[code] = g.reset_index(drop=True)
        return out

    # --- 1. daily 支持批量查询 ---
    for i in range(0, len(ts_codes), max_codes_per_request):
        chunk = ts_codes[i:i+max_codes_per_request]
        ts_param = ','.join(chunk)
        try:
            df_daily = pro.daily(ts_code=ts_param, start_date=start_date, end_date="")
            if df_daily is None:
                df_daily = pd.DataFrame()
            daily_map = split_df_by_ts(df_daily)
            results['daily'].update(daily_map)
        except Exception:
            pass

    # --- 2. daily_basic, moneyflow, margin_detail 需要逐只获取 (并行) ---
    def fetch_single_basic(code):
        try:
            df = pro.daily_basic(ts_code=code, start_date=start_date, end_date="", fields=fields_daily_basic)
            return (code, 'daily_basic', df if df is not None else pd.DataFrame())
        except Exception:
            return (code, 'daily_basic', pd.DataFrame())
    
    def fetch_single_flow(code):
        try:
            df = pro.moneyflow(ts_code=code, start_date=start_date, end_date="", fields=fields_moneyflow)
            return (code, 'moneyflow', df if df is not None else pd.DataFrame())
        except Exception:
            return (code, 'moneyflow', pd.DataFrame())
    
    def fetch_single_margin(code):
        if not fields_margin:
            return (code, 'margin', pd.DataFrame())
        try:
            df = pro.margin_detail(ts_code=code, start_date=start_date, end_date="")
            return (code, 'margin', df if df is not None else pd.DataFrame())
        except Exception:
            return (code, 'margin', pd.DataFrame())
    
    # 使用线程池并行获取所有单只股票的 basic/flow/margin 数据
    max_workers = min(len(ts_codes) * 3, 30)  # 限制最大并发数
    
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for code in ts_codes:
            futures.append(ex.submit(fetch_single_basic, code))
            futures.append(ex.submit(fetch_single_flow, code))
            if fields_margin:
                futures.append(ex.submit(fetch_single_margin, code))
        
        for fut in as_completed(futures, timeout=timeout * 2):
            try:
                code, data_type, df = fut.result(timeout=timeout)
                if df is not None and not df.empty:
                    results[data_type][code] = df.reset_index(drop=True)
            except Exception:
                pass

    return results


def fetch_market_data_by_date(pro, trade_date: str, timeout: int = 30):
    """
    按日期获取全市场数据 (龙虎榜、大宗交易)
    
    v37 新增: 这些数据是日度全市场数据，无法按个股批量获取
    
    Args:
        pro: Tushare Pro API 实例
        trade_date: 交易日期 (如 '20241129')
        timeout: 超时时间（秒）
    
    Returns:
        dict: {
            'top_list': DataFrame,      # 龙虎榜
            'top_inst': DataFrame,      # 龙虎榜机构明细
            'block_trade': DataFrame    # 大宗交易
        }
    """
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    results = {
        'top_list': pd.DataFrame(),
        'top_inst': pd.DataFrame(),
        'block_trade': pd.DataFrame()
    }
    
    def fetch_top_list():
        try:
            return pro.top_list(trade_date=trade_date)
        except Exception as e:
            logger.debug(f"获取龙虎榜失败 {trade_date}: {e}")
            return pd.DataFrame()
    
    def fetch_top_inst():
        try:
            return pro.top_inst(trade_date=trade_date)
        except Exception as e:
            logger.debug(f"获取龙虎榜机构明细失败 {trade_date}: {e}")
            return pd.DataFrame()
    
    def fetch_block_trade():
        try:
            return pro.block_trade(trade_date=trade_date)
        except Exception as e:
            logger.debug(f"获取大宗交易失败 {trade_date}: {e}")
            return pd.DataFrame()
    
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {
            ex.submit(fetch_top_list): 'top_list',
            ex.submit(fetch_top_inst): 'top_inst',
            ex.submit(fetch_block_trade): 'block_trade'
        }
        
        for fut in as_completed(futs, timeout=timeout):
            name = futs[fut]
            try:
                df = fut.result(timeout=timeout)
                if df is not None and not df.empty:
                    results[name] = df
            except Exception:
                pass
    
    return results
