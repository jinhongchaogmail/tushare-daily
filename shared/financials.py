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
    将季度财报 `df_fin` 向前填充到日频 `df_daily`。
    规则:
    - 对于日 t，使用最近一个已公布且 ann_date <= t 的财报数据
    - 如果没有已公布的财报，使用 NaN
    - 返回一个与 df_daily 索引对齐的新 DataFrame（含财务字段）
    """
    if df_fin.empty:
        # 返回空列框架
        return pd.DataFrame(index=df_daily.index)

    df_fin = df_fin.copy()
    # 确保日期列
    df_fin[ann_date_col] = pd.to_datetime(df_fin[ann_date_col])

    # 仅保留财务值列（除去标识列）
    value_cols = [c for c in df_fin.columns if c not in ['ts_code','end_date','ann_date','report_date']]

    # 创建按 ann_date 排序的时间序列并向前填充
    # 对于每个日频日期，找到最大 ann_date <= date
    df_daily = df_daily.copy()
    df_daily['__date'] = pd.to_datetime(df_daily['trade_date'])

    out = pd.DataFrame(index=df_daily.index)
    out.index.name = df_daily.index.name

    # 若 df_fin 中 ann_date 为 NaT，则使用 end_date
    df_fin['ref_date'] = df_fin[ann_date_col].fillna(df_fin.get('end_date'))
    df_fin = df_fin.dropna(subset=['ref_date'])
    df_fin = df_fin.sort_values('ref_date')

    # 为加速，将 df_fin 转为按 ref_date 分组的字典
    fin_list = []
    for _, row in df_fin.iterrows():
        fin_list.append((row['ref_date'], row[value_cols].to_dict()))

    # 对每个日频日期选择最近的公布值
    fin_dates = [d for d,_ in fin_list]
    fin_vals = [v for _,v in fin_list]

    import bisect
    rows = []
    for i, dt in enumerate(df_daily['__date']):
        pos = bisect.bisect_right(fin_dates, dt) - 1
        if pos >= 0:
            out_row = fin_vals[pos]
        else:
            out_row = {c: np.nan for c in value_cols}
        rows.append(out_row)

    out = pd.DataFrame(rows, columns=value_cols)
    out.index = df_daily.index
    return out
