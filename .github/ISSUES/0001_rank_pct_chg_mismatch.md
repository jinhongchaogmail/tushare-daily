---
title: "rank_pct_chg 在训练与推断中定义不一致（截面 vs 滚动）"
labels: [bug, high-priority, data-quality]
assignees: []
---

问题概述
--
训练阶段 (`training/train.py`) 使用的是截面排名（cross-sectional rank）来构造 `rank_pct_chg`，而推断阶段（`models/frozen_features.py` / `shared/features.py`）使用的是滚动时间序列排名（rolling rank）。这会导致训练学到的模式在推断时不匹配，实盘信号严重失真。

影响
--
- 实盘预测与训练目标不一致，导致性能下降或策略失效。

复现步骤
--
1. 在训练数据上计算 `rank_pct_chg`（截面）并训练模型。
2. 在推断时仅使用该股票历史窗口计算滚动 `rank_pct_chg`，观察模型输出与验证不一致。

建议修复
--
有两种可行方向（任选其一）：

方案 A（推荐，当可用全市场数据时）
- 在推断时也计算截面排名：每日加载全市场当日涨幅，计算横向百分位并作为 `rank_pct_chg` 输入。
- 优点：保持训练/推断一致，模型无需重训。
- 代价：推断需要访问全市场数据（内存/IO 增加）。

方案 B（当推断无法获取全市场数据时）
- 修改训练逻辑：将训练时的特征从截面排名改为滚动排名（window=60 等），并重新训练模型。
- 优点：推断开销低。
- 代价：需要重新训练并验证性能。

修改点建议（代码片段）
--
1) 在 `training/train.py`（若选择方案 B）

```python
# 替换当前截面 rank 逻辑为滚动 rank
df['rank_pct_chg_rolling'] = df['pct_chg'].rolling(60, min_periods=5).apply(
    lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8), raw=False
)
# 将训练使用的特征名替换为 rank_pct_chg_rolling 或更新 feature list
```

2) 在 `daily_run.py` 或推断路径（若选择方案 A）

```python
# 在推断时加载当日所有股票的 pct_chg 列并计算横向分位
grouped = all_stocks_df.groupby('trade_date')
all_stocks_df['rank_pct_chg'] = grouped['pct_chg'].rank(pct=True)
```

验收标准
--
- 训练和推断使用同一逻辑生成 `rank_pct_chg`，或者训练已基于滚动排名并且重训练后验证指标恢复或提升。
