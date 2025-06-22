#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用工具函数
CDS503 Group Project - 首尔自行车需求预测
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import json

from config import config

# 配置警告和字体
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = config.VIZ_CONFIG['font_family']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style(config.VIZ_CONFIG['style'])
sns.set_palette(config.VIZ_CONFIG['color_palette'])

class Logger:
    """日志管理类"""
    
    def __init__(self, name, level=None):
        self.logger = logging.getLogger(name)
        
        if not self.logger.handlers:
            # 设置日志级别
            level = level or config.LOGGING_CONFIG['level']
            self.logger.setLevel(getattr(logging, level))
            
            # 创建格式器
            formatter = logging.Formatter(config.LOGGING_CONFIG['format'])
            
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # 文件处理器
            file_handler = logging.FileHandler(config.LOGGING_CONFIG['log_file'], encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def debug(self, message):
        self.logger.debug(message)

class DataValidator:
    """数据验证类"""
    
    @staticmethod
    def validate_dataframe(df, required_columns=None):
        """验证DataFrame的基本属性"""
        if df is None or df.empty:
            raise ValueError("数据框为空")
        
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"缺少必需的列: {missing_cols}")
        
        return True
    
    @staticmethod
    def check_missing_values(df):
        """检查缺失值"""
        missing_summary = df.isnull().sum()
        missing_percent = (missing_summary / len(df)) * 100
        
        if missing_summary.sum() > 0:
            missing_info = pd.DataFrame({
                '缺失数量': missing_summary,
                '缺失比例(%)': missing_percent
            }).round(2)
            return missing_info[missing_info['缺失数量'] > 0]
        
        return None
    
    @staticmethod
    def check_data_types(df, expected_types=None):
        """检查数据类型"""
        type_info = pd.DataFrame({
            '列名': df.columns,
            '数据类型': df.dtypes,
            '非空数量': df.count(),
            '唯一值数量': [df[col].nunique() for col in df.columns]
        })
        
        if expected_types:
            type_mismatches = []
            for col, expected_type in expected_types.items():
                if col in df.columns and not df[col].dtype == expected_type:
                    type_mismatches.append(f"{col}: 期望 {expected_type}, 实际 {df[col].dtype}")
            
            if type_mismatches:
                return type_info, type_mismatches
        
        return type_info, None

class MetricsCalculator:
    """评估指标计算类"""
    
    @staticmethod
    def calculate_regression_metrics(y_true, y_pred):
        """计算回归评估指标"""
        try:
            # 确保输入是numpy数组
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # 移除可能的nan值
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) == 0:
                raise ValueError("所有预测值或真实值都是NaN")
            
            metrics = {
                'RMSE': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
                'MAE': mean_absolute_error(y_true_clean, y_pred_clean),
                'R2': r2_score(y_true_clean, y_pred_clean),
                'MAPE': np.mean(np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-8))) * 100,
                'MSE': mean_squared_error(y_true_clean, y_pred_clean),
                'Max_Error': np.max(np.abs(y_true_clean - y_pred_clean)),
                'Mean_Error': np.mean(y_true_clean - y_pred_clean),
                'Std_Error': np.std(y_true_clean - y_pred_clean)
            }
            
            return metrics
            
        except Exception as e:
            raise ValueError(f"计算指标时出错: {str(e)}")
    
    @staticmethod
    def format_metrics(metrics, decimal_places=4):
        """格式化指标输出"""
        formatted = {}
        for metric, value in metrics.items():
            if metric in ['MAPE']:
                formatted[metric] = f"{value:.{decimal_places-2}f}%"
            elif metric in ['R2']:
                formatted[metric] = f"{value:.{decimal_places}f}"
            else:
                formatted[metric] = f"{value:.{decimal_places-2}f}"
        
        return formatted
    
    @staticmethod
    def compare_models(results_dict):
        """比较多个模型的性能"""
        comparison_df = pd.DataFrame(results_dict).T
        
        # 按RMSE排序（越小越好）
        comparison_df = comparison_df.sort_values('RMSE')
        
        # 添加排名
        comparison_df['Rank'] = range(1, len(comparison_df) + 1)
        
        return comparison_df

class DataLoader:
    """数据加载类"""
    
    def __init__(self, logger=None):
        self.logger = logger or Logger(__name__)
    
    def load_data(self, file_path=None):
        """智能加载数据，自动尝试不同编码"""
        if file_path is None:
            file_path = config.RAW_DATA_FILE
        
        # 尝试不同编码
        for encoding in config.DATA_CONFIG['encoding']:
            try:
                self.logger.info(f"尝试使用 {encoding} 编码加载数据...")
                df = pd.read_csv(file_path, encoding=encoding)
                self.logger.info(f"✅ 数据加载成功！使用编码: {encoding}")
                self.logger.info(f"数据形状: {df.shape}")
                return df
            except UnicodeDecodeError:
                self.logger.warning(f"❌ {encoding} 编码失败，尝试下一个...")
                continue
            except Exception as e:
                self.logger.error(f"加载数据时出错 ({encoding}): {str(e)}")
                continue
        
        raise ValueError("所有编码方式都失败，无法加载数据")

class ResultSaver:
    """结果保存类"""
    
    def __init__(self, logger=None):
        self.logger = logger or Logger(__name__)
    
    def save_dataframe(self, df, filename, subdir=None, index=False):
        """保存DataFrame到CSV"""
        try:
            if subdir:
                save_path = config.OUTPUT_DIR / subdir / f"{filename}.csv"
                save_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                save_path = config.OUTPUT_DIR / f"{filename}.csv"
            
            df.to_csv(save_path, index=index, encoding='utf-8-sig')
            self.logger.info(f"✅ 数据已保存: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"保存数据框失败: {str(e)}")
            raise
    
    def save_figure(self, fig, filename, subdir="figures", dpi=None):
        """保存图表"""
        try:
            dpi = dpi or config.VIZ_CONFIG['dpi']
            save_path = config.OUTPUT_DIR / subdir / f"{filename}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            self.logger.info(f"✅ 图表已保存: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"保存图表失败: {str(e)}")
            raise
    
    def save_json(self, data, filename, subdir=None):
        """保存JSON数据"""
        try:
            if subdir:
                save_path = config.OUTPUT_DIR / subdir / f"{filename}.json"
                save_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                save_path = config.OUTPUT_DIR / f"{filename}.json"
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"✅ JSON数据已保存: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"保存JSON失败: {str(e)}")
            raise
    
    def save_model_results(self, model_name, metrics, predictions=None, 
                          feature_importance=None, hyperparams=None):
        """保存模型结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_data = {
                'model_name': model_name,
                'timestamp': timestamp,
                'metrics': metrics,
                'hyperparameters': hyperparams
            }
            
            # 保存主要结果
            save_path = self.save_json(result_data, f"{model_name}_results_{timestamp}", "results")
            
            # 保存预测结果
            if predictions is not None:
                pred_df = pd.DataFrame({
                    'predictions': predictions
                })
                self.save_dataframe(pred_df, f"{model_name}_predictions_{timestamp}", "results")
            
            # 保存特征重要性
            if feature_importance is not None:
                imp_df = pd.DataFrame({
                    'feature': feature_importance.keys() if isinstance(feature_importance, dict) else range(len(feature_importance)),
                    'importance': feature_importance.values() if isinstance(feature_importance, dict) else feature_importance
                }).sort_values('importance', ascending=False)
                self.save_dataframe(imp_df, f"{model_name}_feature_importance_{timestamp}", "results")
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"保存模型结果失败: {str(e)}")
            raise

class VisualizationHelper:
    """可视化辅助类"""
    
    def __init__(self, logger=None):
        self.logger = logger or Logger(__name__)
        self.result_saver = ResultSaver(logger)
    
    def create_subplots(self, nrows, ncols, figsize=None):
        """创建子图"""
        figsize = figsize or config.VIZ_CONFIG['figure_size']
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        return fig, axes
    
    def plot_correlation_matrix(self, df, target_col=None, save_name=None):
        """绘制相关性矩阵"""
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 创建遮罩（只显示下三角）
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # 绘制热力图
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title('特征相关性矩阵', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_name:
            self.result_saver.save_figure(fig, save_name)
        
        return fig, ax
    
    def plot_target_distribution(self, y, save_name=None):
        """绘制目标变量分布"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 直方图
        axes[0].hist(y, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_title('目标变量分布', fontweight='bold')
        axes[0].set_xlabel('租借数量')
        axes[0].set_ylabel('频次')
        
        # 箱线图
        axes[1].boxplot(y)
        axes[1].set_title('目标变量箱线图', fontweight='bold')
        axes[1].set_ylabel('租借数量')
        
        # Q-Q图
        from scipy import stats
        stats.probplot(y, dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q 正态性检验', fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            self.result_saver.save_figure(fig, save_name)
        
        return fig, axes
    
    def plot_time_series(self, df, date_col, target_col, save_name=None):
        """绘制时间序列图"""
        # 确保日期列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 每日趋势
        daily_avg = df.groupby(df[date_col].dt.date)[target_col].mean()
        axes[0, 0].plot(daily_avg.index, daily_avg.values)
        axes[0, 0].set_title('每日平均租借量趋势', fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 每小时模式
        hourly_avg = df.groupby('Hour')[target_col].mean()
        axes[0, 1].plot(hourly_avg.index, hourly_avg.values, marker='o')
        axes[0, 1].set_title('每小时平均租借量', fontweight='bold')
        axes[0, 1].set_xlabel('小时')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 每月模式
        monthly_avg = df.groupby(df[date_col].dt.month)[target_col].mean()
        axes[1, 0].bar(monthly_avg.index, monthly_avg.values)
        axes[1, 0].set_title('每月平均租借量', fontweight='bold')
        axes[1, 0].set_xlabel('月份')
        
        # 工作日vs周末
        df['is_weekend'] = df[date_col].dt.weekday >= 5
        weekend_avg = df.groupby('is_weekend')[target_col].mean()
        axes[1, 1].bar(['工作日', '周末'], weekend_avg.values)
        axes[1, 1].set_title('工作日 vs 周末', fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            self.result_saver.save_figure(fig, save_name)
        
        return fig, axes

def print_section_header(title, level=1):
    """打印格式化的章节标题"""
    if level == 1:
        print("\n" + "="*60)
        print(f"🔍 {title}")
        print("="*60)
    elif level == 2:
        print(f"\n📊 {title}")
        print("-"*50)
    elif level == 3:
        print(f"\n💡 {title}")
        print("-"*30)

def get_timestamp():
    """获取当前时间戳"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def safe_divide(numerator, denominator, default=0):
    """安全除法，避免除零错误"""
    try:
        return numerator / denominator if denominator != 0 else default
    except:
        return default 