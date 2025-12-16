"""
数据获取模块 - 从 XCSC Tushare 获取各类数据

v37 新增：
- 融资融券数据 (margin_detail)
- 龙虎榜数据 (top_list, top_inst)  
- 大宗交易数据 (block_trade)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging

logger = logging.getLogger(__name__)


def get_xcsc_pro():
    """获取 XCSC Tushare Pro API 实例"""
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
    """
    获取个股融资融券明细数据
    
    Args:
        pro: Tushare Pro API 实例
        ts_code: 股票代码 (如 '000001.SZ')
        start_date: 开始日期 (如 '20220101')
        end_date: 结束日期 (默认今天)
    
    Returns:
        DataFrame with columns:
        - trade_date: 交易日期
        - ts_code: 股票代码
        - rzye: 融资余额 (元)
        - rqye: 融券余额 (元)
        - rzmre: 融资买入额 (元)
        - rzche: 融资偿还额 (元)
        - rqmcl: 融券卖出量 (股)
        - rqchl: 融券偿还量 (股)
        - rzrqye: 融资融券余额 (元)
    """
    if pro is None:
        return pd.DataFrame()
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    
    try:
        df = pro.margin_detail(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is not None and not df.empty:
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            return df
    except Exception as e:
        logger.debug(f"获取融资融券数据失败 {ts_code}: {e}")
    
    return pd.DataFrame()


def fetch_margin_by_date(pro, trade_date: str) -> pd.DataFrame:
    """
    按日期获取全市场融资融券数据
    
    Args:
        pro: Tushare Pro API 实例
        trade_date: 交易日期 (如 '20241129')
    
    Returns:
        DataFrame with margin data for all stocks on that date
    """
    if pro is None:
        return pd.DataFrame()
    
    try:
        df = pro.margin_detail(trade_date=trade_date)
        if df is not None and not df.empty:
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            return df
    except Exception as e:
        logger.debug(f"获取全市场融资融券数据失败 {trade_date}: {e}")
    
    return pd.DataFrame()


def fetch_top_list(pro, trade_date: str) -> pd.DataFrame:
    """
    获取龙虎榜数据
    
    Args:
        pro: Tushare Pro API 实例
        trade_date: 交易日期 (如 '20241129')
    
    Returns:
        DataFrame with columns:
        - trade_date: 交易日期
        - ts_code: 股票代码
        - name: 股票名称
        - close: 收盘价
        - pct_chg: 涨跌幅
        - turnover_rate: 换手率
        - amount: 成交额
        - l_sell: 龙虎榜卖出额
        - l_buy: 龙虎榜买入额
        - l_amount: 龙虎榜成交额
        - net_amount: 龙虎榜净买额
        - net_rate: 净买额占比
        - reason: 上榜原因
    """
    if pro is None:
        return pd.DataFrame()
    
    try:
        df = pro.top_list(trade_date=trade_date)
        if df is not None and not df.empty:
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            return df
    except Exception as e:
        logger.debug(f"获取龙虎榜数据失败 {trade_date}: {e}")
    
    return pd.DataFrame()


def fetch_top_inst(pro, trade_date: str) -> pd.DataFrame:
    """
    获取龙虎榜机构交易明细
    
    Args:
        pro: Tushare Pro API 实例
        trade_date: 交易日期 (如 '20241129')
    
    Returns:
        DataFrame with institutional trading details
    """
    if pro is None:
        return pd.DataFrame()
    
    try:
        df = pro.top_inst(trade_date=trade_date)
        if df is not None and not df.empty:
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            return df
    except Exception as e:
        logger.debug(f"获取龙虎榜机构数据失败 {trade_date}: {e}")
    
    return pd.DataFrame()


def fetch_block_trade(pro, trade_date: str) -> pd.DataFrame:
    """
    获取大宗交易数据
    
    Args:
        pro: Tushare Pro API 实例
        trade_date: 交易日期 (如 '20241129')
    
    Returns:
        DataFrame with columns:
        - ts_code: 股票代码
        - trade_date: 交易日期
        - price: 成交价
        - vol: 成交量 (万股)
        - amount: 成交金额 (万元)
        - buyer: 买方营业部
        - seller: 卖方营业部
    """
    if pro is None:
        return pd.DataFrame()
    
    try:
        df = pro.block_trade(trade_date=trade_date)
        if df is not None and not df.empty:
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            return df
    except Exception as e:
        logger.debug(f"获取大宗交易数据失败 {trade_date}: {e}")
    
    return pd.DataFrame()


def aggregate_top_inst_by_stock(df_top_inst: pd.DataFrame) -> pd.DataFrame:
    """
    将龙虎榜机构明细按股票聚合
    
    Args:
        df_top_inst: 龙虎榜机构明细 DataFrame
    
    Returns:
        DataFrame with aggregated columns:
        - ts_code: 股票代码
        - trade_date: 交易日期
        - inst_buy_count: 机构买入次数
        - inst_sell_count: 机构卖出次数
        - inst_net_buy: 机构净买入额
        - inst_buy_total: 机构买入总额
        - inst_sell_total: 机构卖出总额
    """
    if df_top_inst.empty:
        return pd.DataFrame()
    
    # 识别机构席位 (包含"机构"关键字的营业部)
    df = df_top_inst.copy()
    
    # 分离买入和卖出
    df_buy = df[df['side'] == 0].copy() if 'side' in df.columns else pd.DataFrame()
    df_sell = df[df['side'] == 1].copy() if 'side' in df.columns else pd.DataFrame()
    
    result = []
    
    for (ts_code, trade_date), group in df.groupby(['ts_code', 'trade_date']):
        buy_group = group[group.get('side', 0) == 0] if 'side' in group.columns else group
        sell_group = group[group.get('side', 0) == 1] if 'side' in group.columns else pd.DataFrame()
        
        # 统计机构席位
        inst_buy = buy_group[buy_group['exalter'].str.contains('机构', na=False)] if 'exalter' in buy_group.columns else pd.DataFrame()
        inst_sell = sell_group[sell_group['exalter'].str.contains('机构', na=False)] if 'exalter' in sell_group.columns else pd.DataFrame()
        
        result.append({
            'ts_code': ts_code,
            'trade_date': trade_date,
            'inst_buy_count': len(inst_buy),
            'inst_sell_count': len(inst_sell),
            'inst_buy_total': inst_buy['buy'].sum() if 'buy' in inst_buy.columns and len(inst_buy) > 0 else 0,
            'inst_sell_total': inst_sell['sell'].sum() if 'sell' in inst_sell.columns and len(inst_sell) > 0 else 0,
        })
    
    df_result = pd.DataFrame(result)
    if not df_result.empty:
        df_result['inst_net_buy'] = df_result['inst_buy_total'] - df_result['inst_sell_total']
    
    return df_result


def aggregate_block_trade_by_stock(df_block: pd.DataFrame, df_daily: pd.DataFrame = None) -> pd.DataFrame:
    """
    将大宗交易按股票聚合，并计算折溢价率
    
    Args:
        df_block: 大宗交易 DataFrame
        df_daily: 日线数据 DataFrame (用于计算折溢价)
    
    Returns:
        DataFrame with aggregated columns:
        - ts_code: 股票代码
        - trade_date: 交易日期
        - block_count: 大宗交易笔数
        - block_vol: 大宗成交量 (万股)
        - block_amount: 大宗成交额 (万元)
        - block_avg_price: 大宗平均成交价
        - block_premium: 大宗折溢价率 (相对收盘价)
    """
    if df_block.empty:
        return pd.DataFrame()
    
    result = []
    
    for (ts_code, trade_date), group in df_block.groupby(['ts_code', 'trade_date']):
        block_vol = group['vol'].sum() if 'vol' in group.columns else 0
        block_amount = group['amount'].sum() if 'amount' in group.columns else 0
        block_avg_price = block_amount / block_vol if block_vol > 0 else 0
        
        # 获取当日收盘价计算折溢价
        block_premium = 0.0
        if df_daily is not None and not df_daily.empty:
            close_row = df_daily[(df_daily['ts_code'] == ts_code) & (df_daily['trade_date'] == trade_date)]
            if not close_row.empty and 'close' in close_row.columns:
                close_price = close_row['close'].values[0]
                if close_price > 0 and block_avg_price > 0:
                    block_premium = (block_avg_price - close_price) / close_price
        
        result.append({
            'ts_code': ts_code,
            'trade_date': trade_date,
            'block_count': len(group),
            'block_vol': block_vol,
            'block_amount': block_amount,
            'block_avg_price': block_avg_price,
            'block_premium': block_premium,
        })
    
    return pd.DataFrame(result)


def merge_alternative_data(df_main: pd.DataFrame, 
                           df_margin: pd.DataFrame = None,
                           df_top: pd.DataFrame = None,
                           df_block: pd.DataFrame = None) -> pd.DataFrame:
    """
    将另类数据合并到主表
    
    Args:
        df_main: 主数据表 (必须包含 ts_code, trade_date)
        df_margin: 融资融券数据
        df_top: 龙虎榜聚合数据
        df_block: 大宗交易聚合数据
    
    Returns:
        合并后的 DataFrame
    """
    df = df_main.copy()
    merge_keys = ['ts_code', 'trade_date']
    
    # 确保日期格式一致
    if 'trade_date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
            df['trade_date'] = pd.to_datetime(df['trade_date'])
    
    # 合并融资融券
    if df_margin is not None and not df_margin.empty:
        margin_cols = ['ts_code', 'trade_date', 'rzye', 'rqye', 'rzmre', 'rzche', 'rqmcl', 'rqchl', 'rzrqye']
        margin_cols = [c for c in margin_cols if c in df_margin.columns]
        if len(margin_cols) > 2:
            df_margin_clean = df_margin[margin_cols].copy()
            if not pd.api.types.is_datetime64_any_dtype(df_margin_clean['trade_date']):
                df_margin_clean['trade_date'] = pd.to_datetime(df_margin_clean['trade_date'])
            df = pd.merge(df, df_margin_clean, on=merge_keys, how='left')
    
    # 合并龙虎榜
    if df_top is not None and not df_top.empty:
        if not pd.api.types.is_datetime64_any_dtype(df_top['trade_date']):
            df_top['trade_date'] = pd.to_datetime(df_top['trade_date'])
        df = pd.merge(df, df_top, on=merge_keys, how='left')
    
    # 合并大宗交易
    if df_block is not None and not df_block.empty:
        if not pd.api.types.is_datetime64_any_dtype(df_block['trade_date']):
            df_block['trade_date'] = pd.to_datetime(df_block['trade_date'])
        df = pd.merge(df, df_block, on=merge_keys, how='left')
    
    return df


def fetch_income(pro, ts_code: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """
    获取利润表数据
    
    Args:
        pro: Tushare Pro API 实例
        ts_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
    """
    if pro is None:
        return pd.DataFrame()
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
        
    try:
        # 关键字段: 归母净利润, 营业总收入, 基本每股收益
        fields = 'ts_code,ann_date,f_ann_date,end_date,report_type,n_income_attr_p,total_revenue,basic_eps'
        df = pro.income(ts_code=ts_code, start_date=start_date, end_date=end_date, fields=fields)
        if df is not None and not df.empty:
            # 转换日期
            for col in ['ann_date', 'f_ann_date', 'end_date']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], format='%Y%m%d')
            return df
    except Exception as e:
        logger.debug(f"获取利润表失败 {ts_code}: {e}")
        
    return pd.DataFrame()


def fetch_balancesheet(pro, ts_code: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """
    获取资产负债表数据
    
    Args:
        pro: Tushare Pro API 实例
        ts_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
    """
    if pro is None:
        return pd.DataFrame()
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
        
    try:
        # 关键字段: 资产总计, 负债合计, 股东权益合计
        fields = 'ts_code,ann_date,f_ann_date,end_date,report_type,total_assets,total_liab,total_hldr_eqy_exc_min_int'
        df = pro.balancesheet(ts_code=ts_code, start_date=start_date, end_date=end_date, fields=fields)
        if df is not None and not df.empty:
            # 转换日期
            for col in ['ann_date', 'f_ann_date', 'end_date']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], format='%Y%m%d')
            return df
    except Exception as e:
        logger.debug(f"获取资产负债表失败 {ts_code}: {e}")
        
    return pd.DataFrame()
