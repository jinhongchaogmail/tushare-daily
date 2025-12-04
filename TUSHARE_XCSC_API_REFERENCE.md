# XCSC Tushare API Reference

本文档基于对私有服务器 `http://tushare.xcsc.com:7172` 的实测结果整理，列出了支持的接口、字段及数据类型。

**服务器地址**: `http://tushare.xcsc.com:7172`
**测试日期**: 2025-12-03

## 1. 基础数据

### 1.1 交易日历 (`trade_cal`)
获取交易所交易日期安排。

*   **调用方式**: `pro.trade_cal(exchange='', start_date='...', end_date='...')`
*   **支持字段**:

| 字段名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `exchange` | object | 交易所代码 |
| `cal_date` | object | 日历日期 (YYYYMMDD) |
| `is_open` | int64 | 是否交易 (0/1) |

### 1.2 股票列表 (`stock_basic`)
**状态**: ✅ **可用** (需注意字段)
*   **调用方式**: `pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,list_date')`
*   **注意**: 该接口**不包含**地域 (`area`) 和行业 (`industry`) 信息，请求这些字段会导致报错。

---

## 2. 行情数据

### 2.1 日线行情 (`daily`)
包含开高低收、成交量及**复权数据**。

*   **调用方式**: `pro.daily(ts_code='...', start_date='...', end_date='...')`
*   **支持字段**:

| 字段名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `ts_code` | object | 股票代码 |
| `trade_date` | object | 交易日期 |
| `open` / `high` / `low` / `close` | float64 | 原始价格 |
| `pre_close` | float64 | 昨收价 |
| `change` | float64 | 涨跌额 |
| `pct_chg` | float64 | 涨跌幅 (%) |
| `volume` | float64 | 成交量 (手) |
| `amount` | float64 | 成交额 (千元) |
| `adj_open` / `adj_high` / `adj_low` / `adj_close` | float64 | **复权后价格** |
| `adj_factor` | float64 | 复权因子 |
| `trade_status` | object | 交易状态 |

### 2.2 每日指标 (`daily_basic`)
包含市值、估值、换手率及 TTM 财务数据。**这是最核心的增强接口。**

*   **调用方式**: `pro.daily_basic(ts_code='...', trade_date='...')`
*   **字段映射表 (标准版 -> XCSC版)**:

| 标准版字段 | XCSC版字段 | 含义 |
| :--- | :--- | :--- |
| `total_mv` | `tot_mv` | 总市值 (万元) |
| `circ_mv` | `mv` | 流通市值 (万元) |
| `turnover_rate` | `turn` | 换手率 (%) |
| `pb` | `pb_new` | 市净率 |
| `pe` | `pe` | 市盈率 |
| `pe_ttm` | `pe_ttm` | 滚动市盈率 |

*   **其他重要字段**:
    *   `ps` / `ps_ttm`: 市销率 / 滚动市销率
    *   `free_turnover`: 自由换手率 (%)
    *   `high_52w` / `low_52w`: 52周最高/最低价
    *   `net_profit_parent_comp_ttm`: 归母净利润 (TTM)
    *   `oper_rev_ttm`: 营业收入 (TTM)
    *   `net_cash_flows_oper_act_ttm`: 经营现金流 (TTM)

### 2.3 个股资金流向 (`moneyflow`)
包含大小单买卖数据。

*   **调用方式**: `pro.moneyflow(ts_code='...', trade_date='...')`
*   **支持字段**:

| 字段名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `buy_sm_vol` / `sell_sm_vol` | int64 | 小单买入/卖出量 |
| `buy_md_vol` / `sell_md_vol` | int64 | 中单买入/卖出量 |
| `buy_lg_vol` / `sell_lg_vol` | int64 | 大单买入/卖出量 |
| `buy_elg_vol` / `sell_elg_vol` | int64 | 超大单买入/卖出量 |
| `net_mf_vol` | int64 | 净流入量 |
| `net_mf_amount` | float64 | 净流入额 (万元) |

### 2.4 复权因子 (`adj_factor`)
独立接口，也可直接从 `daily` 获取。
*   **调用方式**: `pro.adj_factor(ts_code='...', trade_date='...')`

---

## 3. 财务数据

### 3.1 利润表 (`income`)
*   **调用方式**: `pro.income(ts_code='...', start_date='...', end_date='...')`
*   **注意**: 之前报错 `net_profit` 缺失，实测发现字段名为 `n_income` (净利润) 或 `n_income_attr_p` (归母净利润)。
*   **关键字段**:
    *   `total_revenue`: 营业总收入
    *   `n_income`: 净利润
    *   `n_income_attr_p`: 归母净利润
    *   `basic_eps`: 基本每股收益

### 3.2 资产负债表 (`balancesheet`)
*   **调用方式**: `pro.balancesheet(ts_code='...', start_date='...', end_date='...')`
*   **关键字段**:
    *   `total_assets`: 总资产
    *   `total_liab`: 总负债
    *   `total_hldr_eqy_exc_min_int`: 股东权益 (不含少数股东权益)

### 3.3 现金流量表 (`cashflow`)
*   **调用方式**: `pro.cashflow(ts_code='...', start_date='...', end_date='...')`
*   **关键字段**:
    *   `n_cashflow_act`: 经营活动产生的现金流量净额

---

## 4. 不支持/异常接口

| 接口名 | 状态 | 备注 |
| :--- | :--- | :--- |
| `limit_list` | ❌ 无数据 | 经测试最近10个交易日均无数据返回 |
| `stk_factor` | ❌ 不存在 | 接口不存在 |
| `suspend_d` | ❓ 未测试 | 建议使用 `daily` 中的 `trade_status` 判断停牌 |

## 5. 使用示例

```python
import xcsc_tushare as ts

# 初始化
ts.set_token("YOUR_TOKEN")
pro = ts.pro_api(env="prd", server="http://tushare.xcsc.com:7172")

# 1. 获取行情 + 市值 + 换手率
df_daily = pro.daily(ts_code='000001.SZ', start_date='20231201', end_date='20231201')
df_basic = pro.daily_basic(ts_code='000001.SZ', trade_date='20231201')

# 合并数据 (示例)
if not df_daily.empty and not df_basic.empty:
    # 注意字段名映射: tot_mv(总市值), turn(换手率)
    df_merge = pd.merge(df_daily, df_basic[['ts_code', 'trade_date', 'tot_mv', 'pe', 'turn']], on=['ts_code', 'trade_date'])
    print(df_merge.head())
```
