"""
财务数据抓取与日频对齐工具
- 提供从 XCSC （xcsc_tushare pro）获取季报并向日频对齐的函数
- 对齐策略: 使用财报公布日 (`ann_date`) 进行安全向前填充，避免未来函数泄露

API:
- fetch_financials(ts_code, start_date, end_date)
- align_financials_to_daily(df_daily, df_financials, ann_date_col='ann_date')
"""

import pandas as pd
import numpy as np
from datetime import datetime


def fetch_financials(pro, ts_code, start_date='20180101', end_date=''):
    """
    从 pro 获取利润表/资产负债表/现金流量表的常用字段（以利润表为主）
    返回合并后的季度财报 DataFrame，包含 ann_date、end_date、report_date 等列
    """
    # 利润表
    try:
        df_inc = pro.income(ts_code=ts_code, start_date=start_date, end_date=end_date)
    except Exception:
        df_inc = pd.DataFrame()
    # 资产负债表
    try:
        df_bal = pro.balancesheet(ts_code=ts_code, start_date=start_date, end_date=end_date)
    except Exception:
        df_bal = pd.DataFrame()
    # 现金流
    try:
        df_cfs = pro.cashflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
    except Exception:
        df_cfs = pd.DataFrame()

    # 以 end_date/ann_date 为对齐键，优先以 income 为基础
    if df_inc.empty:
        base = df_bal if not df_bal.empty else df_cfs
    else:
        base = df_inc.copy()

    # 选择常用字段
    fields = {}
    if 'n_income_attr_p' in base.columns:
        fields['net_profit'] = 'n_income_attr_p'
    elif 'n_income' in base.columns:
        fields['net_profit'] = 'n_income'

    if 'total_revenue' in base.columns:
        fields['total_revenue'] = 'total_revenue'
    if 'basic_eps' in base.columns:
        fields['basic_eps'] = 'basic_eps'
        
    # (v39 新增) 资产负债表字段
    if 'total_assets' in base.columns:
        fields['total_assets'] = 'total_assets'
    if 'total_liab' in base.columns:
        fields['total_liab'] = 'total_liab'
    if 'total_hldr_eqy_exc_min_int' in base.columns:
        fields['total_hldr_eqy_exc_min_int'] = 'total_hldr_eqy_exc_min_int'

    # 合并各表（健壮合并：根据两个表中都存在的键进行 merge）
    dfs = [df for df in [df_inc, df_bal, df_cfs] if not df.empty]
    if not dfs:
        return pd.DataFrame()
    merged = dfs[0].copy()
    for other in dfs[1:]:
        # 找到两个表共有的键列
        common_keys = [k for k in ['ts_code', 'end_date', 'ann_date', 'report_date'] if k in merged.columns and k in other.columns]
        if common_keys:
            merged = pd.merge(merged, other, on=common_keys, how='outer')
        else:
            # 如果没有共同的主键，按 ts_code + 类似日期列进行合并（降级策略）
            if 'ts_code' in merged.columns and 'ts_code' in other.columns:
                # 尝试按 ts_code + end_date
                if 'end_date' in merged.columns and 'end_date' in other.columns:
                    merged = pd.merge(merged, other, on=['ts_code', 'end_date'], how='outer')
                else:
                    # 退回到简单的 concat（可能造成重复）
                    merged = pd.concat([merged, other], axis=0, ignore_index=True, sort=False)
            else:
                merged = pd.concat([merged, other], axis=0, ignore_index=True, sort=False)

    # 保留关键列并去重
    keep_cols = ['ts_code','end_date','ann_date','report_date'] + list(set(fields.values()))
    keep_cols = [c for c in keep_cols if c in merged.columns]
    merged = merged[keep_cols].drop_duplicates()

    # 规范日期
    if 'ann_date' in merged.columns:
        merged['ann_date'] = pd.to_datetime(merged['ann_date'], format='%Y%m%d', errors='coerce')
    if 'end_date' in merged.columns:
        merged['end_date'] = pd.to_datetime(merged['end_date'], format='%Y%m%d', errors='coerce')

    merged = merged.sort_values(['ts_code','ann_date'])
    return merged


def align_financials_to_daily(df_daily, df_fin, ann_date_col='ann_date'):
    """
    将季度财报 `df_fin` 对齐到日频 `df_daily`，基于公告日 (`ann_date`) 进行安全向前填充。
    
    规则:
    - 使用 `pd.merge_asof` 基于 `ann_date` 对齐，确保只能看到过去已发布的财报。
    - 避免数据泄露：`direction='backward'` 确保对于日 t，只能使用 ann_date <= t 的财报。
    - 数据清洗：剔除无效的 `ann_date`，确保公告日不早于财报期末日。
    
    参数:
    - df_daily: 日频数据，需包含 'trade_date' 和 'ts_code' 列。
    - df_fin: 财报数据，需包含 'ann_date', 'end_date', 'ts_code' 以及财务字段。
    - ann_date_col: 公告日期列名，默认 'ann_date'。
    
    返回:
    - 与 df_daily 索引对齐的 DataFrame，包含财务字段。
    """
    if df_fin.empty:
        # 返回空列框架
        return pd.DataFrame(index=df_daily.index)
    
    # 数据清洗：确保 df_fin 准备好
    df_fin = df_fin.copy()
    
    # 1. 确保 ann_date 是 datetime 格式
    if ann_date_col in df_fin.columns:
        df_fin[ann_date_col] = pd.to_datetime(df_fin[ann_date_col], errors='coerce')
    else:
        raise ValueError(f"财报数据中缺少 {ann_date_col} 列")
    
    # 2. 剔除 ann_date 为空的行
    df_fin = df_fin.dropna(subset=[ann_date_col])
    
    # 3. 剔除 ann_date 早于 end_date 的异常行（如果 end_date 存在）
    if 'end_date' in df_fin.columns:
        df_fin['end_date'] = pd.to_datetime(df_fin['end_date'], errors='coerce')
        df_fin = df_fin[df_fin[ann_date_col] >= df_fin['end_date']]
    
    # 4. 按 ts_code 和 ann_date 排序，确保 merge_asof 正确工作
    df_fin = df_fin.sort_values(['ts_code', ann_date_col])
    
    # 准备 df_daily
    df_daily = df_daily.copy()
    if 'trade_date' not in df_daily.columns:
        raise ValueError("日频数据中缺少 'trade_date' 列")
    df_daily['trade_date'] = pd.to_datetime(df_daily['trade_date'], errors='coerce')
    df_daily = df_daily.dropna(subset=['trade_date'])
    df_daily = df_daily.sort_values(['ts_code', 'trade_date'])
    
    # 确定要保留的财务字段（排除标识列）
    exclude_cols = ['ts_code', 'end_date', ann_date_col, 'report_date']
    value_cols = [c for c in df_fin.columns if c not in exclude_cols]
    
    # 使用 pd.merge_asof 进行对齐
    merged = pd.merge_asof(
        left=df_daily[['ts_code', 'trade_date']],
        right=df_fin[['ts_code', ann_date_col] + value_cols],
        left_on='trade_date',
        right_on=ann_date_col,
        by='ts_code',
        direction='backward'  # 只能使用 ann_date <= trade_date 的财报
    )
    
    # 清理结果：删除重复的标识列，保留财务字段
    result = merged[value_cols].copy()
    result.index = df_daily.index  # 保持与 df_daily 的索引对齐
    
    return result
