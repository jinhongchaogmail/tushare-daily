---
title: "对绝对值特征（MACD/OBV/资金绝对额）进行归一化或替换为比率"
labels: [enhancement, medium-priority]
assignees: []
---

问题概述
--
当前 `shared/features.py` 中存在若干依赖原始量级的特征（例如 `MACD_12_26_9`, `OBV`, `margin_balance` 等），这些特征因股票价格与成交规模差异而具有强烈的规模依赖性，可能导致模型对价格尺度而非模式建模。

影响
--
- 模型泛化能力下降，跨不同市值/价格区间的股票表现不稳定。

建议修复
--
对以下类别的绝对特征实施归一化策略：

1. MACD 系列：使用比率或相对值，例如 `MACD / close` 或 `MACD / MA(close, 26)`。
2. OBV/AD：使用 `ROC(OBV)` 或 `(OBV - MA(OBV))/MA(OBV)`。
3. 资金流绝对额（`margin_balance`, `top_net_buy` 等）：转换为占比，例如 `margin_balance / circ_mv`、`top_net_buy / amount`。

代码示例（`shared/features.py`）
--
```python
if 'MACD_12_26_9' in df.columns:
    df['MACD_ratio'] = df['MACD_12_26_9'] / (df['close'] + 1e-8)

if 'OBV' in df.columns:
    df['OBV_zscore'] = (df['OBV'] - df['OBV'].rolling(60).mean()) / (df['OBV'].rolling(60).std() + 1e-8)

if 'margin_balance' in df.columns and 'mv' in df.columns:
    df['margin_balance_ratio'] = df['margin_balance'] / (df['mv'] + 1e-8)
```

验收标准
--
- 新增归一化/比率特征被加入并在 `training/train.py` 的排除名单中移除对应的原始绝对值特征。
