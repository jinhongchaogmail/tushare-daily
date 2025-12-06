#!/usr/bin/env python3
"""
GitHub Actions 示例脚本
展示如何使用Python与GitHub API交互
"""

import os
import json
import requests

def list_repo_issues(owner, repo, token=None):
    """
    列出指定仓库的所有issues
    
    Args:
        owner (str): 仓库所有者
        repo (str): 仓库名称
        token (str, optional): GitHub个人访问令牌
    
    Returns:
        list: issues列表
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    
    if token:
        headers["Authorization"] = f"token {token}"
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    return response.json()

def create_issue(owner, repo, title, body, token):
    """
    在指定仓库创建一个新issue
    
    Args:
        owner (str): 仓库所有者
        repo (str): 仓库名称
        title (str): issue标题
        body (str): issue正文
        token (str): GitHub个人访问令牌
    
    Returns:
        dict: 创建的issue信息
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {token}"
    }
    
    data = {
        "title": title,
        "body": body
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    return response.json()

def search_repos(query, token=None):
    """
    搜索GitHub仓库
    
    Args:
        query (str): 搜索查询
        token (str, optional): GitHub个人访问令牌
    
    Returns:
        dict: 搜索结果
    """
    url = f"https://api.github.com/search/repositories?q={query}"
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    
    if token:
        headers["Authorization"] = f"token {token}"
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    return response.json()

if __name__ == "__main__":
    print("GitHub API交互示例")
    print("=" * 30)
    
    # 示例：搜索与机器学习相关的仓库
    try:
        results = search_repos("machine learning language:python")
        print(f"\n找到 {results['total_count']} 个与'机器学习'相关的Python仓库")
        print("前5个结果:")
        for i, item in enumerate(results['items'][:5]):
            print(f"{i+1}. {item['full_name']} - ⭐ {item['stargazers_count']}")
    except Exception as e:
        print(f"搜索仓库时出错: {e}")
