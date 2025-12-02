#!/usr/bin/env python3
"""
Recursive Optimizer for CatBoost Stock Model
------------------------------------------
This script implements an infinite loop of:
1. Run Training (using existing 模型训练.py)
2. Analyze Results (parse Optuna report)
3. Adjust Configuration (zoom in on best parameters)
4. Repeat

Usage:
    python3 recursive_optimizer.py
"""

import os
import sys
import time
import yaml
import pandas as pd
import shutil
import subprocess
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [RecursiveOptimizer] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recursive_optimizer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CONFIG_PATH = "training/config.yaml"
TRAINING_SCRIPT = "training/模型训练.py"
RESULTS_BASE_DIR = "optuna_results"

def load_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(config):
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)

def get_latest_run_dir():
    if not os.path.exists(RESULTS_BASE_DIR):
        return None
    dirs = [d for d in os.listdir(RESULTS_BASE_DIR) if d.startswith("run_")]
    if not dirs:
        return None
    # Sort by timestamp in name
    dirs.sort(reverse=True)
    return os.path.join(RESULTS_BASE_DIR, dirs[0])

def generate_ai_report(cycle, run_dir, success, error_msg=None, best_params=None, best_value=None):
    """
    Generates a markdown report specifically for the AI assistant to read.
    User can simply paste this file content to the chat.
    """
    report_path = "AI_CONSULT_REPORT.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# AI Consultation Report\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Cycle:** {cycle}\n")
        f.write(f"**Status:** {'SUCCESS' if success else 'FAILURE'}\n")
        f.write(f"**Run Directory:** `{run_dir}`\n\n")
        
        if error_msg:
            f.write(f"## 🚨 Critical Error\n")
            f.write(f"The script encountered an error it could not handle:\n")
            f.write(f"```\n{error_msg}\n```\n\n")
        
        if best_params:
            f.write(f"## 🏆 Best Results So Far\n")
            f.write(f"- **Best F1 Score:** {best_value}\n")
            f.write(f"- **Best Parameters:**\n")
            f.write(f"```json\n{best_params}\n```\n\n")
            
        f.write(f"## ⚙️ Current Configuration State\n")
        try:
            config = load_config()
            search_space = config.get('optuna', {}).get('search_space', {})
            f.write(f"```yaml\n{yaml.dump(search_space, default_flow_style=False)}\n```\n\n")
        except Exception:
            f.write("Could not load current config.\n\n")

        f.write(f"## 📝 Request to AI\n")
        f.write(f"Please analyze the above. \n")
        if not success:
            f.write("Diagnose the error and provide a fix for `recursive_optimizer.py` or `模型训练.py`.\n")
        else:
            f.write("Evaluate if the search space is converging correctly. Should we widen it, shift it, or change strategy?\n")

    logger.info(f"AI Report generated at {report_path}")

def analyze_and_update_config(run_dir, cycle):
    """
    Reads the optuna report from the run_dir and updates config.yaml
    to narrow down the search space around the best parameters.
    """
    report_path = os.path.join(run_dir, "optuna_trials_report_catboost.csv")
    if not os.path.exists(report_path):
        msg = f"Report not found: {report_path}"
        logger.error(msg)
        generate_ai_report(cycle, run_dir, False, error_msg=msg)
        return False

    try:
        df = pd.read_csv(report_path)
        # Filter for completed trials
        df = df[df['state'] == 'COMPLETE']
        if df.empty:
            msg = "No completed trials found in the report."
            logger.warning(msg)
            generate_ai_report(cycle, run_dir, False, error_msg=msg)
            return False
        
        # Get top 5 trials
        top_trials = df.sort_values(by='value', ascending=False).head(5)
        best_trial = top_trials.iloc[0]
        
        best_value = best_trial['value']
        # Extract params for report
        best_params_dict = {
            'depth': int(best_trial['params_depth']),
            'iterations': int(best_trial['params_iterations']),
            'learning_rate': float(best_trial['params_learning_rate']),
            'vol_multiplier': float(best_trial['params_vol_multiplier']) if 'params_vol_multiplier' in best_trial else None
        }

        logger.info(f"Best Trial Value: {best_value}")
        logger.info("Top 5 Trials Parameters:")
        print(top_trials[['value', 'params_depth', 'params_iterations', 'params_learning_rate', 'params_vol_multiplier']])

        # Load current config
        config = load_config()
        search_space = config['optuna']['search_space']

        # --- Strategy: Zoom In ---
        # We will update the ranges to be centered around the best value, 
        # but keep them wide enough to avoid local optima.
        
        # 1. Depth (Integer)
        best_depth = int(best_trial['params_depth'])
        new_depth_min = max(6, best_depth - 2)
        new_depth_max = min(16, best_depth + 2)
        search_space['depth'] = [new_depth_min, new_depth_max]
        logger.info(f"Updated Depth: {search_space['depth']}")

        # 2. Iterations (Integer)
        best_iter = int(best_trial['params_iterations'])
        # Widen the range slightly around the best, but respect bounds
        new_iter_min = max(100, int(best_iter * 0.8))
        new_iter_max = int(best_iter * 1.2)
        search_space['iterations'] = [new_iter_min, new_iter_max]
        logger.info(f"Updated Iterations: {search_space['iterations']}")

        # 3. Learning Rate (Float - Log scale usually, but here we do simple zoom)
        best_lr = float(best_trial['params_learning_rate'])
        new_lr_min = best_lr * 0.5
        new_lr_max = best_lr * 1.5
        search_space['learning_rate'] = [float(f"{new_lr_min:.5f}"), float(f"{new_lr_max:.5f}")]
        logger.info(f"Updated Learning Rate: {search_space['learning_rate']}")

        # 4. Vol Multiplier
        if 'params_vol_multiplier' in best_trial:
            best_vol = float(best_trial['params_vol_multiplier'])
            new_vol_min = max(0.1, best_vol - 0.3)
            new_vol_max = best_vol + 0.3
            search_space['vol_multiplier'] = [float(f"{new_vol_min:.3f}"), float(f"{new_vol_max:.3f}")]
            logger.info(f"Updated Vol Multiplier: {search_space['vol_multiplier']}")

        # Save updated config
        config['optuna']['search_space'] = search_space
        save_config(config)
        logger.info("Config updated successfully.")
        
        # Generate Success Report
        generate_ai_report(cycle, run_dir, True, best_params=best_params_dict, best_value=best_value)
        
        return True

    except Exception as e:
        msg = f"Error analyzing results: {str(e)}"
        logger.error(msg)
        generate_ai_report(cycle, run_dir, False, error_msg=msg)
        return False

def run_training_cycle(cycle_num, resume=False):
    logger.info(f"=== Starting Cycle {cycle_num} (Resume={resume}) ===")
    
    # v28: 使用命令行参数避免交互式输入卡死
    # --mode n: 新训练
    # --mode c: 继续训练
    # --use-cache y: 使用缓存 (假设特征工程未变)
    
    mode_arg = 'c' if resume else 'n'
    # 使用 runpy 方式运行，以规避在部分环境下对非 ASCII 文件名的编码问题
    # 通过 ASCII 命名的启动器避免终端/locale 对中文文件名的编码问题
    launcher = "training/run_training.py"
    cmd = f"export PYTHONPATH=$PYTHONPATH:$(pwd)/shared && {sys.executable} {launcher} --mode {mode_arg} --use-cache y"
    
    logger.info(f"Executing: {cmd}")
    
    # 不再需要 stdin pipe，因为使用了命令行参数
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=sys.stdout, 
        stderr=sys.stderr,
        text=True
    )
    
    process.wait()
    
    if process.returncode != 0:
        logger.error("Training script exited with error.")
        return False
    
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume the latest training run for the first cycle')
    args = parser.parse_args()

    cycle = 1
    first_run = True
    
    while True:
        logger.info(f"Initiating Recursive Optimization Loop: Cycle {cycle}")
        
        # Determine if we should resume
        # Only resume on the very first cycle if the flag is set
        should_resume = args.resume and first_run
        
        # 1. Run Training
        success = run_training_cycle(cycle, resume=should_resume)
        if not success:
            logger.error("Training failed. Aborting loop.")
            break
        
        # After first run, disable resume flag so subsequent loops start new runs
        first_run = False
            
        # 2. Get Results
        latest_run = get_latest_run_dir()
        if not latest_run:
            logger.error("Could not find run directory.")
            break
            
        logger.info(f"Cycle {cycle} completed. Results in: {latest_run}")
        
        # 3. Analyze & Update
        logger.info("Analyzing results and updating configuration...")
        updated = analyze_and_update_config(latest_run, cycle)
        
        if not updated:
            logger.warning("Could not update config. Keeping previous config for next run.")
        
        # 4. Wait / Cool down
        logger.info("Waiting 60 seconds before next cycle...")
        time.sleep(60)
        
        cycle += 1

if __name__ == "__main__":
    main()
