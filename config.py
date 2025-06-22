#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目配置文件
CDS503 Group Project - 首尔自行车需求预测
"""

import os
from pathlib import Path

class ProjectConfig:
    """项目配置类"""
    
    # 基础路径配置
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "outputs"
    FIGURES_DIR = OUTPUT_DIR / "figures"
    MODELS_DIR = OUTPUT_DIR / "models"
    RESULTS_DIR = OUTPUT_DIR / "results"
    
    # 数据文件配置
    RAW_DATA_FILE = "SeoulBikeData.csv"
    
    # 确保输出目录存在
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        for dir_path in [cls.OUTPUT_DIR, cls.FIGURES_DIR, cls.MODELS_DIR, cls.RESULTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # 数据处理配置
    DATA_CONFIG = {
        'target_column': 'Rented Bike Count',
        'date_column': 'Date',
        'date_format': 'mixed',
        'encoding': ['utf-8', 'latin-1', 'cp1252', 'unicode_escape'],
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'random_state': 42
    }
    
    # 特征工程配置
    FEATURE_CONFIG = {
        'categorical_features': ['Seasons', 'Holiday', 'Functioning Day'],
        'time_features': {
            'use_hour_encoding': True,
            'use_day_encoding': True,
            'use_month_encoding': True,
            'use_weekend_flag': True
        },
        'weather_thresholds': {
            'temp_ranges': [-50, 0, 10, 20, 30, 50],
            'humidity_ranges': [0, 30, 50, 70, 100],
            'wind_ranges': [0, 2, 4, 6, 20]
        },
        'interaction_features': True,
        'lag_features': False,  # 默认关闭，避免数据泄露
        'outlier_method': 'iqr',
        'outlier_factor': 1.5,
        'scaling_method': 'standard'
    }
    
    # 模型配置
    MODEL_CONFIG = {
        'cross_validation': {
            'method': 'TimeSeriesSplit',
            'n_splits': 5
        },
        'metrics': ['RMSE', 'MAE', 'R2', 'MAPE'],
        'performance_thresholds': {
            'rmse_threshold': 200,
            'r2_threshold': 0.75,
            'mape_threshold': 25
        }
    }
    
    # 可视化配置
    VIZ_CONFIG = {
        'figure_size': (20, 24),
        'dpi': 300,
        'font_family': ['SimHei', 'Arial Unicode MS', 'PingFang SC', 'Microsoft YaHei', 'DejaVu Sans'],
        'color_palette': 'Set2',
        'style': 'whitegrid'
    }
    
    # 实验配置
    EXPERIMENT_CONFIG = {
        'sample_size_ratios': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'hyperparameter_tuning': {
            'search_method': 'grid',  # 'grid', 'random', 'bayesian'
            'n_iter': 50,  # for random/bayesian search
            'scoring': 'neg_mean_squared_error',
            'n_jobs': -1
        }
    }
    
    # 日志配置
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'log_file': OUTPUT_DIR / 'project.log'
    }

# 创建全局配置实例
config = ProjectConfig()
config.create_directories() 