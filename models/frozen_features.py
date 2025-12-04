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
    
    # 多周期收益率
    for lag in [2, 3, 5, 10]:
        df[f'return_{lag}d'] = df['close'].pct_change(lag, fill_method=None)
    
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
    df['close_pct_change'] = df['close'].pct_change(fill_method=None)
    df['rsi_volume_cross'] = (df['RSI_14'] / 50 - 1) * df['volume_ma_ratio']
    
    df['macd_trend_cross'] = df['MACDh_12_26_9'] * df['trend_strength']
    df['bb_volume_cross'] = df['bb_position'] * df['volume_percentile_60']
    df['atr_volume_cross'] = df['ATRr_14'] * df['volume_ma_ratio']
    df['rsi_stoch_cross'] = (df['RSI_14'] / 100) * (df['STOCHk_14_3_3'] / 100)
    
    # [新增] 计算波动率因子 (Top 2 特征，用于动态阈值)
    # 使用 20日滚动标准差，确保与主脚本逻辑一致
    df['volatility_factor'] = df['close'].pct_change(fill_method=None).rolling(window=20).std()

    # --- (v27 新增: 多周期波动率与周期性特征波动率) ---
    # 1. 不同周期的价格波动率 (捕捉短期和长期市场情绪)
    df['volatility_5d'] = df['close'].pct_change(fill_method=None).rolling(window=5).std()
    df['volatility_10d'] = df['close'].pct_change(fill_method=None).rolling(window=10).std()
    df['volatility_60d'] = df['close'].pct_change(fill_method=None).rolling(window=60).std()
    
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

    # --- (v33 新增: 基本面与资金流特征) ---
    # 1. 估值因子 (需 daily_basic 数据)
    if 'pe_ttm' in df.columns:
        # PE 历史分位数 (60日)
        df['pe_rank'] = df['pe_ttm'].rolling(60, min_periods=20).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8), raw=False
        )
    
    if 'pb_new' in df.columns:
        df['pb_rank'] = df['pb_new'].rolling(60, min_periods=20).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8), raw=False
        )

    # 2. 市值因子
    if 'tot_mv' in df.columns:
        df['log_mv'] = np.log(df['tot_mv'] + 1e-8)
        
    # 3. 换手率因子
    if 'turn' in df.columns:
        df['turn_ma5'] = df['turn'].rolling(5).mean()
        df['turn_volatility'] = df['turn'].rolling(20).std()
    
    # (v34 补充) 自由换手率 - 更真实的流动性
    if 'free_turnover' in df.columns:
        df['free_turn_ma5'] = df['free_turnover'].rolling(5).mean()
        
    # 4. 资金流向因子 (需 moneyflow 数据)
    if 'buy_elg_vol' in df.columns and 'sell_elg_vol' in df.columns:
        # 主力净流入 (大单+超大单)
        main_in = df['buy_lg_vol'] + df['buy_elg_vol']
        main_out = df['sell_lg_vol'] + df['sell_elg_vol']
        df['main_force_net_inflow'] = main_in - main_out
        
        # 主力资金占比 (假设单位一致)
        df['main_force_ratio'] = (main_in + main_out) / (df['volume'] + 1e-8)
        
        # 散户情绪 (小单净买入占比)
        retail_net = df['buy_sm_vol'] - df['sell_sm_vol']
        df['retail_sentiment'] = retail_net / (df['volume'] + 1e-8)

        # --- 中单 (mid orders) 特征 (v34 新增) ---
        # 中单常被视为跟风盘或中户行为，能补充主力/散户之间的中间层资金动向
        if 'buy_md_vol' in df.columns and 'sell_md_vol' in df.columns:
            mid_in = df['buy_md_vol']
            mid_out = df['sell_md_vol']
            df['mid_force_net_inflow'] = mid_in - mid_out
            df['mid_force_ratio'] = (mid_in + mid_out) / (df['volume'] + 1e-8)

    # (v34 补充) 资金流向金额 - 区分“量”与“钱”
    if 'net_mf_amount' in df.columns:
        # 归一化金额 (除以成交额)，避免高价股数值过大
        df['net_mf_amt_ratio'] = df['net_mf_amount'] / (df['amount'] + 1e-8)

    # 5. 价格位置 (52周)
    if 'high_52w' in df.columns and 'low_52w' in df.columns:
        df['price_position_52w'] = (df['close'] - df['low_52w']) / (df['high_52w'] - df['low_52w'] + 1e-8)

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
           df['excess_return'] = df['close'].pct_change(fill_method=None)

    # --- (v35 新增: 基于季报的财务特征) ---
    # 如果日表中包含对齐后的季报字段（如 total_revenue, n_income_attr_p, basic_eps）
    if 'total_revenue' in df.columns and 'n_income_attr_p' in df.columns:
        # 同比营收增长率: (当前 - 同期上一年) / 同期上一年
        df['total_revenue_yoy'] = df['total_revenue'] / df['total_revenue'].shift(252) - 1
        # 同比净利润增长率
        df['net_profit_yoy'] = df['n_income_attr_p'] / df['n_income_attr_p'].shift(252) - 1
        # 净利率
        df['net_profit_margin'] = df['n_income_attr_p'] / (df['total_revenue'] + 1e-8)
        # 每股收益变化
        if 'basic_eps' in df.columns:
            df['basic_eps_yoy'] = df['basic_eps'] / df['basic_eps'].shift(252) - 1

    # --- (v36 优化) 精简特征，删除冗余别名 ---
    # 
    # 设计原则:
    # 1. 不创建与原始特征完全相同的别名（如 rsi14=RSI_14），避免模型学习冗余
    # 2. 只保留有独立信息增益的衍生特征
    # 3. 使用 pandas_ta 的原始命名（SMA_5, RSI_14, MACD_12_26_9 等）
    #
    # 已删除的冗余别名（v36）:
    # - ma5/ma10/ma20: 与 SMA_5/SMA_10/SMA_20 完全相同
    # - volatility_10: 与 volatility_10d 完全相同
    # - rsi14: 与 RSI_14 完全相同
    # - macd: 与 MACD_12_26_9 完全相同
    # - macd_signal: 与 MACDs_12_26_9 完全相同

    # --- 保留有价值的衍生特征 ---
    
    # vol_ma5: 波动率的平滑版本（有独立信息，重要性 2.53）
    if 'volatility_10d' in df.columns and 'vol_ma5' not in df.columns:
        df['vol_ma5'] = df['volatility_10d'].rolling(5, min_periods=1).mean()

    # momentum_5: 5日动量（与 return_5d 计算方式相同，但命名更直观）
    # 注意：如果模型已有 return_5d，可考虑删除此特征
    if 'momentum_5' not in df.columns:
        df['momentum_5'] = df['close'].pct_change(5, fill_method=None)

    # rank_pct_chg: 当日涨幅在近60日的分位数（有独立信息，重要性 1.58）
    if 'pct_chg' in df.columns and 'rank_pct_chg' not in df.columns:
        df['rank_pct_chg'] = df['pct_chg'].rolling(60, min_periods=5).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8), raw=False
        )
    
    # --- (v37 新增: 融资融券特征) ---
    # 融资融券是重要的杠杆资金指标，能反映市场情绪
    if 'rzye' in df.columns:
        # 1. 融资余额变化率 (日度)
        df['margin_balance_change'] = df['rzye'].pct_change(fill_method=None)
        
        # 2. 融资余额 5日均线
        df['margin_balance_ma5'] = df['rzye'].rolling(5, min_periods=1).mean()
        
        # 3. 融资买入额占成交额比例 (杠杆情绪)
        if 'rzmre' in df.columns and 'amount' in df.columns:
            df['margin_buy_ratio'] = df['rzmre'] / (df['amount'] + 1e-8)
        
        # 4. 融资净买入 (买入 - 偿还)
        if 'rzmre' in df.columns and 'rzche' in df.columns:
            df['margin_net_buy'] = df['rzmre'] - df['rzche']
            # 归一化
            df['margin_net_buy_ratio'] = df['margin_net_buy'] / (df['amount'] + 1e-8)
    
    # 融券数据 (做空指标)
    if 'rqye' in df.columns:
        # 5. 融券余额变化率
        df['short_balance_change'] = df['rqye'].pct_change(fill_method=None)
        
        # 6. 融券/融资比 (做空情绪 vs 做多情绪)
        if 'rzye' in df.columns:
            df['short_long_ratio'] = df['rqye'] / (df['rzye'] + 1e-8)
    
    # 融资融券总余额占流通市值比例 (如果有市值数据)
    if 'rzrqye' in df.columns and 'tot_mv' in df.columns:
        df['margin_mv_ratio'] = df['rzrqye'] / (df['tot_mv'] * 1e4 + 1e-8)  # tot_mv 单位是万元
    
    # --- (v37 新增: 龙虎榜特征) ---
    # 龙虎榜数据反映异常交易和机构动向
    # 字段来源: merge_and_postprocess 中的聚合
    # - top_buy_amount: 买入总额
    # - top_sell_amount: 卖出总额  
    # - top_net_amount: 净买入额
    # - top_count: 上榜次数
    
    if 'top_net_amount' in df.columns:
        # 1. 龙虎榜净买入占成交额比例
        if 'amount' in df.columns:
            df['top_net_buy_ratio'] = df['top_net_amount'] / (df['amount'] + 1e-8)
        
        # 2. 龙虎榜交易活跃度 (买+卖总额占成交额比)
        if 'top_buy_amount' in df.columns and 'top_sell_amount' in df.columns:
            df['top_activity_ratio'] = (df['top_buy_amount'] + df['top_sell_amount']) / (df['amount'] + 1e-8)
        
        # 3. 龙虎榜买卖力量对比
        if 'top_buy_amount' in df.columns and 'top_sell_amount' in df.columns:
            df['top_buy_sell_ratio'] = df['top_buy_amount'] / (df['top_sell_amount'] + 1e-8)
    
    # 龙虎榜上榜频率 (滚动30日上榜次数)
    if 'top_count' in df.columns:
        # 二值化：当日是否上榜
        df['is_on_top_list'] = (df['top_count'] > 0).astype(int)
        # 近30日上榜次数
        df['top_count_30d'] = df['is_on_top_list'].rolling(30, min_periods=1).sum()
    
    # --- (v37 新增: 大宗交易特征) ---
    # 大宗交易反映大资金动向和折溢价情况
    # 字段来源: merge_and_postprocess 中的聚合
    # - block_vol: 成交量 (股)
    # - block_amount: 成交额 (元)
    # - block_avg_price: 平均成交价
    # - block_count: 交易笔数
    
    if 'block_amount' in df.columns and 'amount' in df.columns:
        # 1. 大宗成交额占比
        df['block_amount_ratio'] = df['block_amount'] / (df['amount'] + 1e-8)
    
    if 'block_vol' in df.columns and 'volume' in df.columns:
        # 2. 大宗成交量占比
        df['block_vol_ratio'] = df['block_vol'] / (df['volume'] + 1e-8)
    
    if 'block_avg_price' in df.columns and 'close' in df.columns:
        # 3. 大宗折溢价率 (大宗成交价 vs 收盘价)
        # < 0 折价，可能暗示看空或大宗减持
        # > 0 溢价，可能暗示看多或大宗锁定
        df['block_premium_rate'] = (df['block_avg_price'] - df['close']) / (df['close'] + 1e-8)
    
    if 'block_count' in df.columns:
        # 4. 大宗交易频率 (近30日大宗次数)
        df['has_block_trade'] = (df['block_count'] > 0).astype(int)
        df['block_count_30d'] = df['has_block_trade'].rolling(30, min_periods=1).sum()
    
    # --- (v37 新增: 组合特征 - 多信号融合) ---
    # 1. 杠杆资金综合信号 (融资买入强度 + 融券做空程度)
    if 'margin_buy_ratio' in df.columns and 'short_long_ratio' in df.columns:
        # 做多信号强 + 做空信号弱 = 看涨
        df['leverage_sentiment'] = df['margin_buy_ratio'] - df['short_long_ratio']
    
    # 2. 机构-散户分歧度 (使用龙虎榜净买入替代原有的 inst_net_buy)
    if 'top_net_buy_ratio' in df.columns and 'retail_sentiment' in df.columns:
        # 龙虎榜资金买入，散户卖出 = 机构抄底信号
        df['smart_money_divergence'] = df['top_net_buy_ratio'] - df['retail_sentiment']
    
    return df
