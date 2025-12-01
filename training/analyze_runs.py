#!/usr/bin/env python
# coding: utf-8
"""
模型训练结果分析与比较工具

功能：
1. 加载多个 Optuna 训练运行的结果
2. 比较不同运行的性能指标
3. 生成可视化图表
4. 输出中文摘要报告

用法：
    python analyze_runs.py --results-dir /path/to/optuna_results
    python analyze_runs.py --runs run_20251201_160807 run_20251201_215750
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# 尝试导入必要的库
try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"错误: 缺少必要的库 {e}。请运行: pip install pandas numpy")
    sys.exit(1)

try:
    import optuna
except ImportError:
    optuna = None
    print("警告: 未安装 optuna，部分功能受限。")

try:
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    # 中文字体配置
    zh_fonts = ['Noto Sans CJK SC', 'Noto Serif CJK SC', 'WenQuanYi Micro Hei', 
                'AR PL UMing CN', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in zh_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font] + zh_fonts
            break
    plt.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: 未安装 matplotlib，跳过可视化。")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class RunAnalyzer:
    """单个训练运行分析器"""
    
    # 默认 study name，可以通过类变量修改
    DEFAULT_STUDY_NAME = "catboost_stock_3class_v16"
    
    def __init__(self, run_path: str, study_name: str = None):
        self.run_path = run_path
        self.run_name = os.path.basename(run_path)
        self.study_name = study_name or self.DEFAULT_STUDY_NAME
        self.study = None
        self.trials_df = None
        self.best_params = None
        self.metrics = {}
        
    def load(self) -> bool:
        """加载运行数据"""
        try:
            # 1. 尝试加载 Optuna 数据库
            db_path = os.path.join(self.run_path, "optuna_catboost_study.db")
            if os.path.exists(db_path) and optuna:
                storage_url = f"sqlite:///{db_path}"
                try:
                    self.study = optuna.load_study(
                        study_name=self.study_name,
                        storage=storage_url
                    )
                except Exception:
                    # 尝试其他 study name
                    studies = optuna.get_all_study_names(storage_url)
                    if studies:
                        self.study = optuna.load_study(
                            study_name=studies[0],
                            storage=storage_url
                        )
            
            # 2. 加载试验 CSV 报告
            csv_path = os.path.join(self.run_path, "optuna_trials_report_catboost.csv")
            if os.path.exists(csv_path):
                self.trials_df = pd.read_csv(csv_path)
            
            # 3. 加载最佳参数
            params_path = os.path.join(self.run_path, "final_model_params.json")
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    self.best_params = json.load(f)
            
            # 4. 提取分类报告
            report_path = os.path.join(self.run_path, "classification_report_detailed.csv")
            if os.path.exists(report_path):
                self.metrics['classification_report'] = pd.read_csv(report_path, index_col=0)
            
            # 5. 提取关键指标
            self._extract_metrics()
            
            return True
            
        except Exception as e:
            print(f"警告: 加载 {self.run_name} 时出错: {e}")
            return False
    
    def _extract_metrics(self):
        """从已加载数据提取关键指标"""
        # 从 study 提取
        if self.study and self.study.best_trial:
            self.metrics['best_value'] = self.study.best_value
            self.metrics['n_trials'] = len(self.study.trials)
            self.metrics['n_completed'] = len([t for t in self.study.trials 
                                               if t.state == optuna.trial.TrialState.COMPLETE])
            self.metrics['n_pruned'] = len([t for t in self.study.trials 
                                            if t.state == optuna.trial.TrialState.PRUNED])
            self.metrics['best_params'] = self.study.best_params
        
        # 从 CSV 提取（作为备用）
        elif self.trials_df is not None and not self.trials_df.empty:
            self.metrics['best_value'] = self.trials_df['value'].max()
            self.metrics['n_trials'] = len(self.trials_df)
            if 'state' in self.trials_df.columns:
                self.metrics['n_completed'] = (self.trials_df['state'] == 'COMPLETE').sum()
                self.metrics['n_pruned'] = (self.trials_df['state'] == 'PRUNED').sum()
        
        # 从 best_params JSON
        if self.best_params:
            self.metrics['vol_multiplier'] = self.best_params.get('vol_multiplier_best')
            self.metrics['offset'] = self.best_params.get('offset_best')
            self.metrics['mode'] = self.best_params.get('mode')
        
        # 解析运行时间戳
        try:
            if self.run_name.startswith('run_'):
                ts = self.run_name.replace('run_', '')
                self.metrics['timestamp'] = datetime.strptime(ts, '%Y%m%d_%H%M%S')
        except Exception:
            pass
    
    def get_summary(self) -> Dict:
        """返回运行摘要"""
        return {
            'run_name': self.run_name,
            'run_path': self.run_path,
            **self.metrics
        }


class MultiRunComparator:
    """多运行比较器"""
    
    def __init__(self, results_dir: str = None, run_names: List[str] = None, study_name: str = None):
        self.results_dir = results_dir
        self.run_names = run_names or []
        self.study_name = study_name
        self.analyzers: List[RunAnalyzer] = []
        self.comparison_df = None
        
    def discover_runs(self) -> List[str]:
        """发现所有可用的运行目录"""
        if not self.results_dir or not os.path.exists(self.results_dir):
            print(f"警告: 结果目录不存在: {self.results_dir}")
            return []
        
        runs = []
        for name in os.listdir(self.results_dir):
            run_path = os.path.join(self.results_dir, name)
            if os.path.isdir(run_path) and name.startswith('run_'):
                runs.append(name)
        
        # 按时间排序
        runs.sort(reverse=True)
        return runs
    
    def load_runs(self, run_names: List[str] = None):
        """加载指定的运行"""
        if run_names:
            self.run_names = run_names
        
        if not self.run_names:
            self.run_names = self.discover_runs()
        
        if not self.run_names:
            print("错误: 未找到任何训练运行记录。")
            return
        
        print(f"\n正在加载 {len(self.run_names)} 个训练运行...")
        
        for name in self.run_names:
            if self.results_dir:
                run_path = os.path.join(self.results_dir, name)
            else:
                run_path = name  # 假设传入的是完整路径
            
            if not os.path.exists(run_path):
                print(f"  跳过: {name} (目录不存在)")
                continue
            
            analyzer = RunAnalyzer(run_path, study_name=self.study_name)
            if analyzer.load():
                self.analyzers.append(analyzer)
                print(f"  ✓ 已加载: {name}")
            else:
                print(f"  ✗ 加载失败: {name}")
        
        print(f"\n成功加载 {len(self.analyzers)} 个运行。")
    
    def compare(self) -> pd.DataFrame:
        """比较所有运行"""
        if not self.analyzers:
            print("错误: 没有已加载的运行数据。")
            return pd.DataFrame()
        
        summaries = [a.get_summary() for a in self.analyzers]
        self.comparison_df = pd.DataFrame(summaries)
        
        # 按 best_value 排序
        if 'best_value' in self.comparison_df.columns:
            self.comparison_df = self.comparison_df.sort_values(
                'best_value', ascending=False
            ).reset_index(drop=True)
        
        return self.comparison_df
    
    def print_summary(self):
        """打印比较摘要"""
        if self.comparison_df is None:
            self.compare()
        
        if self.comparison_df.empty:
            print("没有可用的比较数据。")
            return
        
        print("\n" + "=" * 70)
        print("                    模型训练结果分析报告")
        print("=" * 70)
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"分析运行数: {len(self.comparison_df)}")
        print("=" * 70)
        
        # 最佳运行
        if 'best_value' in self.comparison_df.columns:
            best_idx = self.comparison_df['best_value'].idxmax()
            best_run = self.comparison_df.iloc[best_idx]
            
            print("\n📊 最佳运行")
            print("-" * 40)
            print(f"  运行名称: {best_run['run_name']}")
            print(f"  最佳得分: {best_run['best_value']:.6f}")
            if 'n_trials' in best_run:
                print(f"  试验次数: {best_run.get('n_trials', 'N/A')}")
            if 'vol_multiplier' in best_run and pd.notna(best_run['vol_multiplier']):
                print(f"  波动率乘数: {best_run['vol_multiplier']:.4f}")
        
        # 运行列表
        print("\n📋 所有运行排名 (按得分降序)")
        print("-" * 70)
        print(f"{'排名':<5} {'运行名称':<25} {'最佳得分':<12} {'试验数':<10} {'完成率':<10}")
        print("-" * 70)
        
        for idx, row in self.comparison_df.iterrows():
            rank = idx + 1
            name = row['run_name'][:24] if len(row['run_name']) > 24 else row['run_name']
            score = f"{row.get('best_value', 0):.6f}" if pd.notna(row.get('best_value')) else "N/A"
            n_trials = row.get('n_trials', 'N/A')
            n_completed = row.get('n_completed', 0)
            if n_trials and n_trials != 'N/A' and n_trials > 0:
                completion_rate = f"{n_completed / n_trials * 100:.1f}%"
            else:
                completion_rate = "N/A"
            
            print(f"{rank:<5} {name:<25} {score:<12} {str(n_trials):<10} {completion_rate:<10}")
        
        print("-" * 70)
        
        # 统计摘要
        if 'best_value' in self.comparison_df.columns:
            values = self.comparison_df['best_value'].dropna()
            if len(values) > 0:
                print("\n📈 得分统计")
                print("-" * 40)
                print(f"  最高分: {values.max():.6f}")
                print(f"  最低分: {values.min():.6f}")
                print(f"  平均分: {values.mean():.6f}")
                print(f"  标准差: {values.std():.6f}")
        
        print("\n" + "=" * 70)
    
    def plot_comparison(self, save_path: str = None):
        """生成比较可视化"""
        if not HAS_MATPLOTLIB:
            print("跳过可视化: matplotlib 未安装。")
            return
        
        if self.comparison_df is None or self.comparison_df.empty:
            print("没有可用的数据进行可视化。")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('模型训练结果比较分析', fontsize=14, fontweight='bold')
        
        # 1. 最佳得分对比柱状图
        ax1 = axes[0, 0]
        if 'best_value' in self.comparison_df.columns:
            df_plot = self.comparison_df.dropna(subset=['best_value']).head(10)
            if not df_plot.empty:
                colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(df_plot))]
                bars = ax1.barh(df_plot['run_name'], df_plot['best_value'], color=colors)
                ax1.set_xlabel('最佳得分 (Macro F1 * Balance)')
                ax1.set_title('各运行最佳得分对比')
                ax1.invert_yaxis()
                
                # 添加数值标签 - 使用相对偏移量
                max_val = df_plot['best_value'].max()
                min_val = df_plot['best_value'].min()
                offset = (max_val - min_val) * 0.02 if max_val > min_val else 0.005
                for bar, val in zip(bars, df_plot['best_value']):
                    ax1.text(bar.get_width() + offset, bar.get_y() + bar.get_height()/2,
                            f'{val:.4f}', va='center', fontsize=9)
        
        # 2. 试验完成率饼图
        ax2 = axes[0, 1]
        if 'n_completed' in self.comparison_df.columns and 'n_pruned' in self.comparison_df.columns:
            total_completed = int(self.comparison_df['n_completed'].fillna(0).sum())
            total_pruned = int(self.comparison_df['n_pruned'].fillna(0).sum())
            if 'n_trials' in self.comparison_df.columns:
                total_trials = int(self.comparison_df['n_trials'].fillna(0).sum())
            else:
                total_trials = total_completed + total_pruned
            
            if total_trials > 0:
                other = max(0, total_trials - total_completed - total_pruned)
                # 过滤掉大小为 0 的部分
                sizes = []
                labels = []
                colors_list = []
                if total_completed > 0:
                    sizes.append(total_completed)
                    labels.append(f'完成 ({total_completed})')
                    colors_list.append('#2ecc71')
                if total_pruned > 0:
                    sizes.append(total_pruned)
                    labels.append(f'剪枝 ({total_pruned})')
                    colors_list.append('#e74c3c')
                if other > 0:
                    sizes.append(other)
                    labels.append(f'其他 ({other})')
                    colors_list.append('#95a5a6')
                
                if sizes:
                    ax2.pie(sizes, labels=labels, colors=colors_list, autopct='%1.1f%%', startangle=90)
                ax2.set_title('全部试验状态分布')
        
        # 3. 得分趋势（如果有时间戳）
        ax3 = axes[1, 0]
        if 'timestamp' in self.comparison_df.columns and 'best_value' in self.comparison_df.columns:
            df_time = self.comparison_df.dropna(subset=['timestamp', 'best_value']).copy()
            if not df_time.empty:
                df_time = df_time.sort_values('timestamp')
                ax3.plot(df_time['timestamp'], df_time['best_value'], 
                        marker='o', linewidth=2, markersize=8, color='#3498db')
                ax3.fill_between(df_time['timestamp'], df_time['best_value'], alpha=0.3)
                ax3.set_xlabel('运行时间')
                ax3.set_ylabel('最佳得分')
                ax3.set_title('得分随时间变化趋势')
                ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, '无时间序列数据', ha='center', va='center', fontsize=12)
            ax3.set_title('得分趋势')
        
        # 4. 参数分布（如果有 vol_multiplier）
        ax4 = axes[1, 1]
        if 'vol_multiplier' in self.comparison_df.columns:
            vol_data = self.comparison_df['vol_multiplier'].dropna()
            if len(vol_data) > 0:
                if HAS_SEABORN:
                    sns.histplot(vol_data, kde=True, ax=ax4, color='#9b59b6')
                else:
                    ax4.hist(vol_data, bins=10, color='#9b59b6', edgecolor='white')
                ax4.set_xlabel('波动率乘数')
                ax4.set_ylabel('频次')
                ax4.set_title('最佳波动率乘数分布')
        else:
            ax4.text(0.5, 0.5, '无参数数据', ha='center', va='center', fontsize=12)
            ax4.set_title('参数分布')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 比较图表已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def export_report(self, output_path: str):
        """导出详细报告为 CSV"""
        if self.comparison_df is None:
            self.compare()
        
        if self.comparison_df.empty:
            print("没有可导出的数据。")
            return
        
        self.comparison_df.to_csv(output_path, index=False)
        print(f"✓ 比较报告已保存到: {output_path}")
    
    def get_best_run(self) -> Optional[RunAnalyzer]:
        """返回最佳运行的分析器"""
        if not self.analyzers:
            return None
        
        if self.comparison_df is None:
            self.compare()
        
        if 'best_value' not in self.comparison_df.columns:
            return self.analyzers[0] if self.analyzers else None
        
        best_name = self.comparison_df.iloc[0]['run_name']
        for analyzer in self.analyzers:
            if analyzer.run_name == best_name:
                return analyzer
        
        return None


def main():
    parser = argparse.ArgumentParser(
        description='模型训练结果分析与比较工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析指定目录下的所有运行
  python analyze_runs.py --results-dir ~/gdrive/optuna_results

  # 分析特定的运行
  python analyze_runs.py --results-dir ~/gdrive/optuna_results \\
      --runs run_20251201_160807 run_20251201_215750

  # 导出报告和图表
  python analyze_runs.py --results-dir ~/gdrive/optuna_results \\
      --output-csv comparison.csv --output-plot comparison.png
        """
    )
    
    parser.add_argument(
        '--results-dir', '-d',
        type=str,
        help='Optuna 结果根目录 (包含 run_* 子目录)'
    )
    
    parser.add_argument(
        '--runs', '-r',
        nargs='+',
        type=str,
        help='要分析的特定运行名称 (如 run_20251201_160807)'
    )
    
    parser.add_argument(
        '--output-csv', '-o',
        type=str,
        help='输出比较 CSV 报告的路径'
    )
    
    parser.add_argument(
        '--output-plot', '-p',
        type=str,
        help='输出比较图表的路径'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='静默模式，只输出错误'
    )
    
    parser.add_argument(
        '--study-name', '-s',
        type=str,
        default=None,
        help='Optuna study 名称 (默认: catboost_stock_3class_v16)'
    )
    
    args = parser.parse_args()
    
    # 默认结果目录
    if not args.results_dir:
        # 尝试常见位置
        possible_dirs = [
            os.path.expanduser('~/gdrive/optuna_results'),
            os.path.expanduser('~/optuna_results'),
            '/mnt/workspace/optuna_results',
            '/content/drive/MyDrive/Colab Notebooks/optuna_results'
        ]
        for d in possible_dirs:
            if os.path.exists(d):
                args.results_dir = d
                break
    
    if not args.results_dir or not os.path.exists(args.results_dir):
        print("错误: 请使用 --results-dir 指定有效的结果目录。")
        print("\n使用 --help 查看帮助信息。")
        sys.exit(1)
    
    # 创建比较器
    comparator = MultiRunComparator(results_dir=args.results_dir, study_name=args.study_name)
    comparator.load_runs(args.runs)
    
    if not comparator.analyzers:
        print("错误: 未能加载任何训练运行数据。")
        sys.exit(1)
    
    # 比较
    comparator.compare()
    
    # 输出摘要
    if not args.quiet:
        comparator.print_summary()
    
    # 导出 CSV
    if args.output_csv:
        comparator.export_report(args.output_csv)
    
    # 生成图表
    if args.output_plot:
        comparator.plot_comparison(args.output_plot)
    elif HAS_MATPLOTLIB and not args.quiet:
        # 默认保存到结果目录
        default_plot_path = os.path.join(args.results_dir, 'runs_comparison.png')
        comparator.plot_comparison(default_plot_path)
    
    # 返回最佳运行信息
    best_run = comparator.get_best_run()
    if best_run and not args.quiet:
        print(f"\n推荐使用: {best_run.run_name}")
        if best_run.best_params:
            print("最佳参数配置:")
            print(json.dumps(best_run.best_params, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
