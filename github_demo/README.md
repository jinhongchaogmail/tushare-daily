# GitHub 操作示例

这个目录包含了一些演示如何与GitHub API交互的示例脚本。

## 文件说明

- [github_actions_example.py](github_actions_example.py) - 展示了如何使用Python与GitHub API进行交互的基本示例

## GitHub API 功能演示

### 1. 搜索仓库
脚本会自动搜索GitHub上与"机器学习"相关的Python仓库，并显示前5个最受欢迎的结果。

### 2. 列出Issues
可以通过函数列出任何公开仓库的issues。

### 3. 创建Issue
提供了创建新issue的函数模板（需要有效的访问令牌）。

## 如何使用

1. 安装依赖:
```bash
pip install requests
```

2. 运行示例:
```bash
python github_actions_example.py
```

## 关于访问令牌

要执行写入操作（如创建issue），您需要一个GitHub个人访问令牌：

1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token"
3. 选择适当的权限（对于issues，至少需要 `public_repo` 权限）
4. 生成令牌并在代码中使用它

注意：永远不要将真实的访问令牌提交到版本控制系统中！

## 实际应用案例

这些示例可以扩展用于：

1. 自动化项目管理任务
2. 监控仓库活动
3. 自动生成报告
4. 与其他CI/CD系统集成