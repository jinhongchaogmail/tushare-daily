#!/usr/bin/env python3
"""
本地Git操作示例脚本
展示如何使用Python与本地Git仓库交互
"""

import subprocess
import os
import json
from datetime import datetime

def run_git_command(command):
    """
    运行Git命令并返回结果
    
    Args:
        command (str): Git命令
    
    Returns:
        str: 命令输出
    """
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd="/home/jin/work"
        )
        result.check_returncode()
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"错误: {e.stderr}"

def get_current_branch():
    """获取当前Git分支"""
    return run_git_command("git rev-parse --abbrev-ref HEAD")

def get_last_commit():
    """获取最后一次提交信息"""
    commit_hash = run_git_command("git rev-parse HEAD")
    commit_msg = run_git_command("git log -1 --pretty=%B")
    commit_author = run_git_command("git log -1 --pretty=%an")
    commit_date = run_git_command("git log -1 --pretty=%ad")
    
    return {
        "hash": commit_hash,
        "message": commit_msg,
        "author": commit_author,
        "date": commit_date
    }

def get_status():
    """获取Git状态"""
    return run_git_command("git status --porcelain")

def get_recent_commits(limit=5):
    """获取最近的几次提交"""
    command = f"git log --oneline -n {limit}"
    result = run_git_command(command)
    commits = []
    for line in result.split('\n'):
        if line:
            parts = line.split(' ', 1)
            if len(parts) == 2:
                commits.append({
                    "hash": parts[0],
                    "message": parts[1]
                })
    return commits

def create_summary_report():
    """创建Git仓库摘要报告"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "current_branch": get_current_branch(),
        "last_commit": get_last_commit(),
        "uncommitted_changes": get_status(),
        "recent_commits": get_recent_commits()
    }
    return report

if __name__ == "__main__":
    print("本地Git操作示例")
    print("=" * 30)
    
    # 显示当前分支
    print(f"当前分支: {get_current_branch()}")
    
    # 显示最后一次提交
    last_commit = get_last_commit()
    print(f"\n最后一次提交:")
    print(f"  哈希: {last_commit['hash']}")
    print(f"  作者: {last_commit['author']}")
    print(f"  日期: {last_commit['date']}")
    print(f"  消息: {last_commit['message']}")
    
    # 显示未提交的更改
    status = get_status()
    if status:
        print(f"\n未提交的更改:")
        for line in status.split('\n'):
            if line:
                print(f"  {line}")
    else:
        print("\n工作区干净，无未提交的更改")
    
    # 显示最近的提交
    recent_commits = get_recent_commits()
    print(f"\n最近 {len(recent_commits)} 次提交:")
    for commit in recent_commits:
        print(f"  {commit['hash'][:7]} {commit['message']}")
    
    # 生成并保存报告
    report = create_summary_report()
    with open('/home/jin/work/github_demo/git_summary.json', 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n摘要报告已保存到 github_demo/git_summary.json")