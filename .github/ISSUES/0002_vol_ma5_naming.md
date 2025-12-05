---
title: "vol_ma5 命名含义混淆（volatility_ma5 vs volume_ma5）"
labels: [bug, medium-priority, docs]
assignees: []
---

问题概述
--
`vol_ma5` 在不同模块中含义不一致：在某处被实现为基于波动率的 5 日均值（`volatility_10d.rolling(5).mean()`），在另一些地方语义可能为成交量的 5 日均值。这种命名混淆会导致特征误引用和模型输入不一致。

影响
--
- 特征命名不一致会导致模型训练与推断或后续分析误用特征，影响结果可解释性。

建议修复
--
- 将 `vol_ma5` 明确重命名为 `volatility_ma5`（如果表示波动率均线）或 `volume_ma5`（如果表示成交量均线），并全仓库替换引用。
- 在 `shared/features.py` 添加兼容别名（短期内）并输出 deprecation 警告，给团队时间迁移。

代码修复示例
--
在 `shared/features.py` 中：

```python
if 'volatility_10d' in df.columns and 'volatility_ma5' not in df.columns:
    df['volatility_ma5'] = df['volatility_10d'].rolling(5, min_periods=1).mean()
# 兼容别名（短期）
if 'vol_ma5' not in df.columns:
    df['vol_ma5'] = df['volatility_ma5']
    # logger.warning("`vol_ma5` 将被重命名为 `volatility_ma5`，请更新引用。")
```

验收标准
--
- 代码库中只保留一个明确的字段名（`volatility_ma5` 或 `volume_ma5`），并且所有引用被更新或保持向后兼容一段时间后移除兼容别名。
