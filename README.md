# Tushare Daily Stock Prediction

这是一个基于 GitHub Actions 的每日 A 股量化预测系统。

## 项目结构

- **.github/workflows/**: GitHub Actions 工作流配置
  - `run.yml`: 每日定时运行的任务 (下载数据 -> 预测 -> 发送邮件)
- **training/**: 模型训练与优化
  - `train.py`: 核心训练脚本 (Optuna + CatBoost)
  - `config.yaml`: 训练配置文件
  - `v16_OPTIMIZATION_ARCHIVE.md`: 历史优化文档归档
- **shared/**: 公共模块
  - `features.py`: 统一的特征计算逻辑 (SSOT)
- **daily_run.py**: 生产环境主程序 (下载 -> 预测 -> 报告)
- **data/**: 存放下载的股票数据 (.parquet 格式)
- **optuna_results/**: 训练结果与日志

## 运行流程

1. **每日预测**:
   - GitHub Actions 触发 `daily_run.py`。
   - 脚本下载最新数据，调用 `shared/features.py` 计算特征。
   - 加载 `catboost_final_model.cbm` 进行预测 (支持做多与做空)。
   - 生成 HTML/Markdown 报告并通过邮件发送。

2. **模型训练**:
   - 运行 `training/train.py`。
   - 使用 Optuna 进行超参数寻优。
   - 生成新的模型文件 `catboost_final_model.cbm`。

## 本地运行

1. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```
2. 设置 Tushare Token:
   ```bash
   export TUSHARE_TOKEN="your_token_here"
   ```

更多项目说明与精简文档请参阅：`docs/DOCUMENTATION_SUMMARY.md`
3. 运行预测:
   ```bash
   python daily_run.py
   ```

## 本地代理与加速 GitHub 访问

如果本地有代理（例如在本机启动了 SOCKS/HTTP 代理），可以加速对 GitHub 或外部资源的访问。

1. 将 `secrets.sh` 加载到当前 shell（此文件包含 `TUSHARE_TOKEN` 以及代理环境变量）:
```bash
source ./secrets.sh
```

2. 本仓库提供一个辅助脚本来设置/取消 git 全局代理:
```bash
# 启用 git http/https 代理（指向本机 127.0.0.1:1089）
./scripts/setup_proxy_git.sh enable

# 取消代理
./scripts/setup_proxy_git.sh disable
```

3. 临时在当前 shell 中使用代理（不修改全局配置）:
```bash
export HTTP_PROXY="http://127.0.0.1:1089"
export HTTPS_PROXY="http://127.0.0.1:1089"
export ALL_PROXY="socks5://127.0.0.1:1088"
# 然后运行 pip/git/其他需要加速的命令
```

注意: Git 对 SOCKS 代理的原生支持有限，如需通过 SOCKS 使用 git，可使用 `proxychains` 或 `torsocks`。

