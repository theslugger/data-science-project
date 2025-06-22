#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强数据探索分析
CDS503 Group Project - 首尔自行车需求预测

全面的数据探索和可视化分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import json
import os
from scipy import stats
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# 科学论文级别的图形样式配置
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    # 字体设置 - 科学论文标准
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'SimSun'],
    'font.sans-serif': ['Arial', 'Helvetica', 'SimHei', 'Microsoft YaHei'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    
    # 图形样式 - 学术标准
    'figure.figsize': (12, 8),
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    
    # 轴和网格样式
    'axes.grid': True,
    'axes.grid.axis': 'both',
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'axes.edgecolor': 'black',
    
    # 色彩和线条
    'axes.prop_cycle': plt.cycler('color', [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]),
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'patch.linewidth': 0.5,
    'patch.facecolor': '#1f77b4',
    'patch.edgecolor': 'black',
    
    # 中文支持
    'axes.unicode_minus': False,
})

# 定义科学论文配色方案
SCIENTIFIC_COLORS = {
    'primary': '#2E86AB',      # 主色调：科学蓝
    'secondary': '#A23B72',    # 次色调：深紫红
    'accent': '#F18F01',       # 强调色：暖橙
    'success': '#C73E1D',      # 成功色：深红
    'info': '#845EC2',         # 信息色：紫色
    'warning': '#FF8500',      # 警告色：橙色
    'light': '#F8F9FA',        # 浅色
    'dark': '#212529',         # 深色
    'muted': '#6C757D',        # 中性色
}

# 专业调色板
SCIENTIFIC_PALETTES = {
    'categorical': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#845EC2', '#FF8500'],
    'sequential_blue': ['#EDF4F8', '#D1E5F0', '#92C5DE', '#4393C3', '#2166AC', '#053061'],
    'sequential_red': ['#FFF5F0', '#FEE0D2', '#FCBBA1', '#FC9272', '#FB6A4A', '#DE2D26'],
    'diverging': ['#8E0152', '#C51B7D', '#DE77AE', '#F1B6DA', '#FDE0EF', '#E6F5D0', '#B8E186', '#7FBC41', '#4D9221', '#276419'],
    'temperature': ['#313695', '#4575B4', '#74ADD1', '#ABD9E9', '#E0F3F8', '#FEE090', '#FDAE61', '#F46D43', '#D73027', '#A50026']
}

class BikeDataExplorer:
    """首尔自行车数据探索分析类"""
    
    def __init__(self, data_file='SeoulBikeData.csv'):
        self.data_file = data_file
        self.df = None
        self.insights = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建输出目录
        self.output_dir = 'outputs'
        self.eda_dir = os.path.join(self.output_dir, 'eda')
        self.figures_dir = os.path.join(self.output_dir, 'figures')
        
        for directory in [self.output_dir, self.eda_dir, self.figures_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def load_and_basic_info(self):
        """加载数据并获取基本信息"""
        print("🔍 数据加载与基本信息分析")
        print("="*60)
        
        # 加载数据
        try:
            self.df = pd.read_csv(self.data_file, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                self.df = pd.read_csv(self.data_file, encoding='latin-1')
            except UnicodeDecodeError:
                self.df = pd.read_csv(self.data_file, encoding='cp1252')
        
        print(f"✅ 数据加载成功: {self.data_file}")
        print(f"📊 数据维度: {self.df.shape[0]} 行 × {self.df.shape[1]} 列")
        
        # 基本信息
        print(f"\n📋 数据基本信息:")
        print(f"  数据集大小: {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        print(f"  数据类型分布:")
        print(self.df.dtypes.value_counts().to_string())
        
        # 列名信息
        print(f"\n📝 列名列表:")
        for i, col in enumerate(self.df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # 前几行数据
        print(f"\n🔍 前5行数据:")
        print(self.df.head().to_string())
        
        self.insights.append(f"数据集包含{self.df.shape[0]}条记录和{self.df.shape[1]}个特征")
        
        return self.df.describe()
    
    def data_quality_check(self):
        """数据质量检查"""
        print("\n🔎 数据质量检查")
        print("="*60)
        
        # 缺失值检查
        missing_info = self.df.isnull().sum()
        missing_percent = (missing_info / len(self.df)) * 100
        
        print("📊 缺失值统计:")
        if missing_info.sum() == 0:
            print("  ✅ 没有发现缺失值")
            self.insights.append("数据集完整，无缺失值")
        else:
            missing_df = pd.DataFrame({
                '缺失数量': missing_info,
                '缺失比例(%)': missing_percent
            })
            missing_df = missing_df[missing_df['缺失数量'] > 0].sort_values('缺失数量', ascending=False)
            print(missing_df.to_string())
            self.insights.append(f"发现{missing_df.shape[0]}个字段存在缺失值")
        
        # 重复值检查
        duplicates = self.df.duplicated().sum()
        print(f"\n🔄 重复值统计:")
        if duplicates == 0:
            print("  ✅ 没有发现重复行")
            self.insights.append("数据集无重复记录")
        else:
            print(f"  ⚠️  发现 {duplicates} 条重复记录 ({duplicates/len(self.df)*100:.2f}%)")
            self.insights.append(f"发现{duplicates}条重复记录")
        
        # 数据类型检查
        print(f"\n📊 数据类型详情:")
        print(self.df.dtypes.to_string())
        
        # 唯一值统计
        print(f"\n🔢 唯一值统计:")
        for col in self.df.columns:
            unique_count = self.df[col].nunique()
            unique_ratio = unique_count / len(self.df)
            print(f"  {col}: {unique_count} 个唯一值 ({unique_ratio:.3f})")
        
        return missing_info, duplicates
    
    def target_analysis(self):
        """目标变量分析"""
        print("\n🎯 目标变量分析 (租借自行车数量)")
        print("="*60)
        
        target_col = 'Rented Bike Count'
        
        # 基础统计
        stats_info = self.df[target_col].describe()
        print("📊 基础统计信息:")
        print(stats_info.to_string())
        
        # 分布特征
        skewness = self.df[target_col].skew()
        kurtosis = self.df[target_col].kurtosis()
        
        print(f"\n📈 分布特征:")
        print(f"  偏度 (Skewness): {skewness:.4f}")
        print(f"  峰度 (Kurtosis): {kurtosis:.4f}")
        
        if skewness > 1:
            skew_desc = "严重右偏"
        elif skewness > 0.5:
            skew_desc = "轻微右偏"
        elif skewness < -1:
            skew_desc = "严重左偏"
        elif skewness < -0.5:
            skew_desc = "轻微左偏"
        else:
            skew_desc = "近似正态"
        
        print(f"  分布特征: {skew_desc}")
        
        # 创建目标变量分布图 - 科学论文样式
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Rental Bike Count Distribution Analysis', fontsize=16, fontweight='bold', y=0.95)
        
        # 直方图 - 优雅的蓝色系
        n, bins, patches = axes[0, 0].hist(self.df[target_col], bins=50, alpha=0.8, 
                                          color=SCIENTIFIC_COLORS['primary'], 
                                          edgecolor='white', linewidth=0.5)
        axes[0, 0].set_title('(a) Distribution Histogram', fontweight='bold', pad=15)
        axes[0, 0].set_xlabel('Rental Count', fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 箱线图 - 专业配色
        bp = axes[0, 1].boxplot(self.df[target_col], patch_artist=True, 
                               boxprops=dict(facecolor=SCIENTIFIC_COLORS['secondary'], alpha=0.8),
                               medianprops=dict(color='white', linewidth=2),
                               whiskerprops=dict(color=SCIENTIFIC_COLORS['dark'], linewidth=1.5),
                               capprops=dict(color=SCIENTIFIC_COLORS['dark'], linewidth=1.5),
                               flierprops=dict(marker='o', markerfacecolor=SCIENTIFIC_COLORS['accent'], 
                                             markersize=4, alpha=0.6))
        axes[0, 1].set_title('(b) Box Plot', fontweight='bold', pad=15)
        axes[0, 1].set_ylabel('Rental Count', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q图 - 统计学标准
        stats.probplot(self.df[target_col], dist="norm", plot=axes[1, 0])
        axes[1, 0].get_lines()[0].set_markerfacecolor(SCIENTIFIC_COLORS['info'])
        axes[1, 0].get_lines()[0].set_markeredgecolor('white')
        axes[1, 0].get_lines()[0].set_markersize(4)
        axes[1, 0].get_lines()[1].set_color(SCIENTIFIC_COLORS['accent'])
        axes[1, 0].get_lines()[1].set_linewidth(2)
        axes[1, 0].set_title('(c) Q-Q Plot (Normality Test)', fontweight='bold', pad=15)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 累积分布图 - 科学配色
        sorted_data = np.sort(self.df[target_col])
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        axes[1, 1].plot(sorted_data, cumulative, color=SCIENTIFIC_COLORS['success'], 
                       linewidth=2.5, alpha=0.9)
        axes[1, 1].fill_between(sorted_data, cumulative, alpha=0.2, 
                               color=SCIENTIFIC_COLORS['success'])
        axes[1, 1].set_title('(d) Cumulative Distribution Function', fontweight='bold', pad=15)
        axes[1, 1].set_xlabel('Rental Count', fontweight='bold')
        axes[1, 1].set_ylabel('Cumulative Probability', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, f'target_distribution_{self.timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 异常值检测
        Q1 = self.df[target_col].quantile(0.25)
        Q3 = self.df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.df[(self.df[target_col] < lower_bound) | (self.df[target_col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percent = outlier_count / len(self.df) * 100
        
        print(f"\n🚨 异常值检测 (IQR方法):")
        print(f"  下边界: {lower_bound:.2f}")
        print(f"  上边界: {upper_bound:.2f}")
        print(f"  异常值数量: {outlier_count} ({outlier_percent:.2f}%)")
        
        self.insights.append(f"目标变量呈{skew_desc}分布，异常值占比{outlier_percent:.1f}%")
        
        return stats_info, outlier_count
    
    def temporal_analysis(self):
        """时间序列分析"""
        print("\n⏰ 时间序列分析")
        print("="*60)
        
        # 转换日期格式
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d/%m/%Y')
        self.df['DateTime'] = pd.to_datetime(
            self.df['Date'].dt.strftime('%Y-%m-%d') + ' ' + 
            self.df['Hour'].astype(str) + ':00:00'
        )
        
        # 时间范围
        date_range = f"{self.df['Date'].min().strftime('%Y-%m-%d')} 到 {self.df['Date'].max().strftime('%Y-%m-%d')}"
        total_days = (self.df['Date'].max() - self.df['Date'].min()).days + 1
        
        print(f"📅 时间范围: {date_range} ({total_days} 天)")
        
        # 按时间维度聚合分析
        hourly_avg = self.df.groupby('Hour')['Rented Bike Count'].agg(['mean', 'std']).round(2)
        daily_avg = self.df.groupby('Date')['Rented Bike Count'].agg(['mean', 'sum']).round(2)
        monthly_avg = self.df.groupby(self.df['Date'].dt.month)['Rented Bike Count'].mean().round(2)
        seasonal_avg = self.df.groupby('Seasons')['Rented Bike Count'].mean().round(2)
        
        print(f"\n📊 时间模式分析:")
        print(f"  高峰小时: {hourly_avg['mean'].idxmax()}时 (平均{hourly_avg['mean'].max():.0f}辆)")
        print(f"  低谷小时: {hourly_avg['mean'].idxmin()}时 (平均{hourly_avg['mean'].min():.0f}辆)")
        print(f"  最高单日总量: {daily_avg['sum'].max():.0f}辆")
        print(f"  最低单日总量: {daily_avg['sum'].min():.0f}辆")
        
        # 创建时间序列可视化 - 科学论文样式
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Temporal Patterns in Bike Rental Demand', fontsize=16, fontweight='bold', y=0.95)
        
        # 小时模式 - 双峰曲线
        line = axes[0, 0].plot(hourly_avg.index, hourly_avg['mean'], 
                              marker='o', linewidth=2.5, markersize=6,
                              color=SCIENTIFIC_COLORS['primary'], markerfacecolor='white',
                              markeredgecolor=SCIENTIFIC_COLORS['primary'], markeredgewidth=2)
        axes[0, 0].fill_between(hourly_avg.index, 
                                hourly_avg['mean'] - hourly_avg['std'],
                                hourly_avg['mean'] + hourly_avg['std'], 
                                alpha=0.25, color=SCIENTIFIC_COLORS['primary'])
        axes[0, 0].set_title('(a) Hourly Rental Pattern', fontweight='bold', pad=15)
        axes[0, 0].set_xlabel('Hour of Day', fontweight='bold')
        axes[0, 0].set_ylabel('Average Rental Count', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xticks(range(0, 24, 3))
        
        # 日趋势 - 时间序列
        axes[0, 1].plot(daily_avg.index, daily_avg['mean'], 
                       alpha=0.8, linewidth=1.5, color=SCIENTIFIC_COLORS['secondary'])
        axes[0, 1].set_title('(b) Daily Average Trend', fontweight='bold', pad=15)
        axes[0, 1].set_xlabel('Date', fontweight='bold')
        axes[0, 1].set_ylabel('Daily Average Count', fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 月份模式 - 渐变柱状图
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, 12))
        bars = axes[1, 0].bar(range(1, 13), monthly_avg.values, 
                             color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        axes[1, 0].set_title('(c) Monthly Rental Pattern', fontweight='bold', pad=15)
        axes[1, 0].set_xlabel('Month', fontweight='bold')
        axes[1, 0].set_ylabel('Average Rental Count', fontweight='bold')
        axes[1, 0].set_xticks(range(1, 13))
        axes[1, 0].set_xticklabels(month_names)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 在柱状图上添加数值
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 10,
                           f'{height:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 季节模式 - 专业配色
        season_colors = {
            'Spring': SCIENTIFIC_COLORS['info'],     # 紫色
            'Summer': SCIENTIFIC_COLORS['accent'],   # 橙色
            'Autumn': SCIENTIFIC_COLORS['warning'],  # 深橙
            'Winter': SCIENTIFIC_COLORS['primary']   # 蓝色
        }
        bars = axes[1, 1].bar(seasonal_avg.index, seasonal_avg.values,
                             color=[season_colors.get(season, SCIENTIFIC_COLORS['muted']) 
                                   for season in seasonal_avg.index],
                             alpha=0.85, edgecolor='white', linewidth=1)
        axes[1, 1].set_title('(d) Seasonal Rental Pattern', fontweight='bold', pad=15)
        axes[1, 1].set_xlabel('Season', fontweight='bold')
        axes[1, 1].set_ylabel('Average Rental Count', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # 在柱状图上添加数值
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 15,
                           f'{height:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, f'time_series_{self.timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 周内模式
        self.df['Weekday'] = self.df['Date'].dt.dayofweek
        weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        weekday_avg = self.df.groupby('Weekday')['Rented Bike Count'].mean()
        
        print(f"\n📅 周内模式:")
        for i, avg in weekday_avg.items():
            print(f"  {weekday_names[i]}: {avg:.0f}辆")
        
        peak_hour = hourly_avg['mean'].idxmax()
        peak_season = seasonal_avg.idxmax()
        
        self.insights.append(f"一日双峰模式：高峰在{peak_hour}时")
        self.insights.append(f"季节性强：{peak_season}季需求最高")
        
        return hourly_avg, seasonal_avg
    
    def weather_analysis(self):
        """天气因素分析"""
        print("\n🌤️ 天气因素分析")
        print("="*60)
        
        weather_cols = ['Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 
                       'Visibility (10m)', 'Rainfall(mm)', 'Snowfall (cm)']
        
        # 天气统计
        weather_stats = self.df[weather_cols].describe().round(2)
        print("📊 天气变量统计:")
        print(weather_stats.to_string())
        
        # 创建天气相关性分析图 - 科学论文样式
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Weather Factors Impact on Bike Rental Demand', fontsize=16, fontweight='bold', y=0.95)
        axes = axes.flatten()
        
        # 科学配色方案 - 每个天气因子使用不同颜色
        weather_colors = SCIENTIFIC_PALETTES['categorical']
        
        for i, col in enumerate(weather_cols):
            # 散点图显示与租借量的关系 - 使用密度着色
            scatter = axes[i].scatter(self.df[col], self.df['Rented Bike Count'], 
                                    alpha=0.6, s=25, c=weather_colors[i % len(weather_colors)],
                                    edgecolors='white', linewidth=0.2)
            
            # 计算相关系数
            correlation = self.df[col].corr(self.df['Rented Bike Count'])
            
            # 添加趋势线 - 使用对比色
            z = np.polyfit(self.df[col], self.df['Rented Bike Count'], 1)
            p = np.poly1d(z)
            axes[i].plot(self.df[col], p(self.df[col]), color=SCIENTIFIC_COLORS['dark'], 
                        linestyle='--', alpha=0.9, linewidth=2.5)
            
            # 标题包含子图标号
            subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
            col_short = col.replace('(°C)', '').replace('(%)', '').replace(' (m/s)', '').replace(' (10m)', '').replace('(mm)', '').replace(' (cm)', '')
            axes[i].set_title(f'{subplot_labels[i]} {col_short}\n(r = {correlation:.3f})', 
                            fontweight='bold', pad=15)
            axes[i].set_xlabel(col, fontweight='bold')
            axes[i].set_ylabel('Rental Count', fontweight='bold')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, f'weather_correlation_{self.timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 天气条件分类分析
        print(f"\n🌡️ 温度影响分析:")
        temp_bins = [-20, 0, 10, 20, 30, 40]
        temp_labels = ['严寒(<0°C)', '寒冷(0-10°C)', '凉爽(10-20°C)', '温暖(20-30°C)', '炎热(>30°C)']
        self.df['Temp_Category'] = pd.cut(self.df['Temperature(°C)'], bins=temp_bins, labels=temp_labels)
        temp_analysis = self.df.groupby('Temp_Category')['Rented Bike Count'].agg(['count', 'mean', 'std']).round(2)
        print(temp_analysis.to_string())
        
        print(f"\n💧 降水影响分析:")
        rain_analysis = self.df.groupby(self.df['Rainfall(mm)'] > 0)['Rented Bike Count'].agg(['count', 'mean']).round(2)
        rain_analysis.index = ['无降雨', '有降雨']
        print(rain_analysis.to_string())
        
        snow_analysis = self.df.groupby(self.df['Snowfall (cm)'] > 0)['Rented Bike Count'].agg(['count', 'mean']).round(2)
        snow_analysis.index = ['无降雪', '有降雪']
        print(snow_analysis.to_string())
        
        # 找出强相关的天气因子
        correlations = {}
        for col in weather_cols:
            correlations[col] = abs(self.df[col].corr(self.df['Rented Bike Count']))
        
        strongest_factor = max(correlations, key=correlations.get)
        self.insights.append(f"最强天气相关因子：{strongest_factor} (r={correlations[strongest_factor]:.3f})")
        
        return weather_stats, correlations
    
    def categorical_analysis(self):
        """分类变量分析"""
        print("\n🏷️ 分类变量分析")
        print("="*60)
        
        categorical_cols = ['Seasons', 'Holiday', 'Functioning Day']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
        fig.suptitle('Categorical Variables Impact on Bike Rental', fontsize=16, fontweight='bold', y=0.95)
        
        subplot_labels = ['(a)', '(b)', '(c)']
        colors = [SCIENTIFIC_COLORS['primary'], SCIENTIFIC_COLORS['secondary'], SCIENTIFIC_COLORS['accent']]
        
        for i, col in enumerate(categorical_cols):
            category_stats = self.df.groupby(col)['Rented Bike Count'].agg(['count', 'mean', 'std']).round(2)
            print(f"\n📊 {col} 分析:")
            print(category_stats.to_string())
            
            # 准备数据用于箱线图
            groups = [self.df[self.df[col] == category]['Rented Bike Count'].values 
                     for category in self.df[col].unique()]
            labels = self.df[col].unique()
            
            # 创建专业箱线图 - 使用matplotlib而不是pandas
            bp = axes[i].boxplot(groups, labels=labels, patch_artist=True)
            
            # 自定义箱线图样式
            for patch in bp['boxes']:
                patch.set_facecolor(colors[i])
                patch.set_alpha(0.7)
                patch.set_edgecolor(SCIENTIFIC_COLORS['dark'])
                patch.set_linewidth(1.2)
            
            for median in bp['medians']:
                median.set_color('white')
                median.set_linewidth(2)
            
            for whisker in bp['whiskers']:
                whisker.set_color(SCIENTIFIC_COLORS['dark'])
                whisker.set_linewidth(1.2)
            
            for cap in bp['caps']:
                cap.set_color(SCIENTIFIC_COLORS['dark'])
                cap.set_linewidth(1.2)
            
            for flier in bp['fliers']:
                flier.set_markerfacecolor(colors[i])
                flier.set_markeredgecolor('white')
                flier.set_markersize(4)
                flier.set_alpha(0.6)
            
            axes[i].set_title(f'{subplot_labels[i]} {col}', fontweight='bold', pad=15)
            axes[i].set_xlabel(col, fontweight='bold')
            axes[i].set_ylabel('Rental Count', fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            
            # 旋转x轴标签以避免重叠
            axes[i].tick_params(axis='x', rotation=45)
            
        plt.suptitle('Categorical Variables Impact on Bike Rental', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, f'categorical_analysis_{self.timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 非运营日影响
        functioning_impact = self.df.groupby('Functioning Day')['Rented Bike Count'].mean()
        non_functioning_days = (self.df['Functioning Day'] == 'No').sum()
        non_functioning_ratio = non_functioning_days / len(self.df) * 100
        
        print(f"\n🚫 非运营日影响:")
        print(f"  非运营日数量: {non_functioning_days} ({non_functioning_ratio:.2f}%)")
        
        self.insights.append(f"非运营日占比{non_functioning_ratio:.1f}%")
        
        return category_stats
    
    def correlation_analysis(self):
        """相关性分析"""
        print("\n🔗 特征相关性分析")
        print("="*60)
        
        # 选择数值特征
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 计算相关性矩阵
        correlation_matrix = self.df[numeric_cols].corr()
        
        # 与目标变量的相关性
        target_corr = correlation_matrix['Rented Bike Count'].abs().sort_values(ascending=False)
        
        print("📊 与租借量相关性最强的前10个特征:")
        for i, (feature, corr) in enumerate(target_corr.head(11).items(), 1):  # 11个因为包含目标变量自己
            if feature != 'Rented Bike Count':
                print(f"  {i-1:2d}. {feature}: {corr:.4f}")
        
        # 创建相关性热图 - 科学论文样式
        plt.figure(figsize=(16, 12))
        
        # 创建上三角遮罩
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # 使用科学配色方案
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r',  # 经典的红蓝发散配色
                   center=0,
                   square=True, 
                   linewidths=0.5,
                   linecolor='white',
                   cbar_kws={
                       "shrink": .8, 
                       "label": "Correlation Coefficient",
                       "orientation": "vertical"
                   },
                   fmt='.2f',
                   annot_kws={
                       'size': 8, 
                       'fontweight': 'bold',
                       'color': 'black'
                   })
        
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Features', fontsize=12, fontweight='bold')
        plt.ylabel('Features', fontsize=12, fontweight='bold')
        
        # 设置坐标轴标签样式
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, f'correlation_matrix_{self.timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 多重共线性检查
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = abs(correlation_matrix.iloc[i, j])
                if corr_val > 0.8:  # 高相关性阈值
                    high_corr_pairs.append((
                        correlation_matrix.columns[i], 
                        correlation_matrix.columns[j], 
                        corr_val
                    ))
        
        print(f"\n⚠️ 高相关性特征对 (|r|>0.8):")
        if high_corr_pairs:
            for feat1, feat2, corr in high_corr_pairs:
                print(f"  {feat1} - {feat2}: {corr:.4f}")
        else:
            print("  未发现高相关性特征对")
        
        strongest_predictor = target_corr.drop('Rented Bike Count').index[0]
        self.insights.append(f"最强预测因子：{strongest_predictor} (r={target_corr[strongest_predictor]:.3f})")
        
        return correlation_matrix, target_corr
    
    def advanced_analysis(self):
        """高级分析"""
        print("\n🧮 高级统计分析")
        print("="*60)
        
        # 双峰检测
        hourly_avg = self.df.groupby('Hour')['Rented Bike Count'].mean()
        
        # 找到峰值
        peaks = []
        for i in range(1, len(hourly_avg)-1):
            if hourly_avg.iloc[i] > hourly_avg.iloc[i-1] and hourly_avg.iloc[i] > hourly_avg.iloc[i+1]:
                if hourly_avg.iloc[i] > hourly_avg.mean():  # 只考虑高于平均值的峰
                    peaks.append((hourly_avg.index[i], hourly_avg.iloc[i]))
        
        print(f"🏔️ 需求峰值检测:")
        for hour, demand in peaks:
            print(f"  {hour}时: {demand:.0f}辆")
        
        # 天气阈值分析
        print(f"\n🌡️ 最优天气条件分析:")
        optimal_temp = self.df.loc[self.df['Rented Bike Count'].idxmax(), 'Temperature(°C)']
        temp_range = self.df[(self.df['Temperature(°C)'] >= optimal_temp-5) & 
                           (self.df['Temperature(°C)'] <= optimal_temp+5)]
        optimal_temp_demand = temp_range['Rented Bike Count'].mean()
        
        print(f"  最优温度范围: {optimal_temp-5:.1f}°C - {optimal_temp+5:.1f}°C")
        print(f"  该温度范围平均需求: {optimal_temp_demand:.0f}辆")
        
        # 需求分级
        percentiles = [25, 50, 75, 90, 95]
        demand_thresholds = self.df['Rented Bike Count'].quantile([p/100 for p in percentiles])
        
        print(f"\n📊 需求分级阈值:")
        demand_levels = ['低需求', '中需求', '高需求', '极高需求', '峰值需求']
        for i, (p, threshold) in enumerate(zip(percentiles, demand_thresholds)):
            if i < len(demand_levels):
                print(f"  {demand_levels[i]} (>{p}分位): >{threshold:.0f}辆")
        
        self.insights.append(f"识别出{len(peaks)}个需求峰值")
        
        return peaks, demand_thresholds
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("\n📋 生成综合数据探索报告")
        print("="*60)
        
        # 汇总所有洞察 - 确保所有数据类型可JSON序列化
        report = {
            'dataset_info': {
                'file_name': self.data_file,
                'shape': [int(x) for x in self.df.shape],  # 转换为int列表
                'memory_usage_mb': float(self.df.memory_usage(deep=True).sum() / 1024 / 1024),
                'analysis_timestamp': self.timestamp
            },
            'data_quality': {
                'missing_values': int(self.df.isnull().sum().sum()),
                'duplicate_rows': int(self.df.duplicated().sum()),
                'data_types': {str(k): int(v) for k, v in self.df.dtypes.value_counts().items()}  # 转换键和值
            },
            'target_variable_stats': {
                'mean': float(self.df['Rented Bike Count'].mean()),
                'std': float(self.df['Rented Bike Count'].std()),
                'min': float(self.df['Rented Bike Count'].min()),
                'max': float(self.df['Rented Bike Count'].max()),
                'skewness': float(self.df['Rented Bike Count'].skew()),
                'kurtosis': float(self.df['Rented Bike Count'].kurtosis())
            },
            'temporal_patterns': {
                'date_range': {
                    'start': self.df['Date'].min().strftime('%Y-%m-%d'),
                    'end': self.df['Date'].max().strftime('%Y-%m-%d'),
                    'total_days': int((self.df['Date'].max() - self.df['Date'].min()).days + 1)
                },
                'peak_hour': int(self.df.groupby('Hour')['Rented Bike Count'].mean().idxmax()),
                'peak_season': str(self.df.groupby('Seasons')['Rented Bike Count'].mean().idxmax())
            },
            'key_insights': self.insights,
            'recommendations': [
                "考虑时间特征工程（小时、季节、工作日）",
                "温度是最重要的预测因子，需要详细建模",
                "处理非运营日数据（排除或特殊处理）",
                "考虑双峰模式的特殊建模方法",
                "异常值需要适当处理（winsorization）"
            ]
        }
        
        # 保存报告
        report_file = os.path.join(self.eda_dir, f'comprehensive_eda_results_{self.timestamp}.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存处理后的数据
        self.df.to_csv(os.path.join(self.eda_dir, f'explored_data_{self.timestamp}.csv'), 
                      index=False, encoding='utf-8')
        
        print(f"✅ 报告已保存:")
        print(f"  📊 分析结果: {report_file}")
        print(f"  💾 数据文件: {os.path.join(self.eda_dir, f'explored_data_{self.timestamp}.csv')}")
        
        # 打印核心洞察摘要
        print(f"\n💡 核心洞察摘要:")
        for i, insight in enumerate(self.insights, 1):
            print(f"  {i}. {insight}")
        
        return report
    
    def run_complete_exploration(self):
        """运行完整的数据探索分析"""
        print("🚀 开始完整数据探索分析")
        print("="*80)
        
        try:
            # 1. 基本信息
            basic_stats = self.load_and_basic_info()
            
            # 2. 数据质量检查
            missing_info, duplicates = self.data_quality_check()
            
            # 3. 目标变量分析
            target_stats, outliers = self.target_analysis()
            
            # 4. 时间序列分析
            temporal_patterns = self.temporal_analysis()
            
            # 5. 天气分析
            weather_analysis = self.weather_analysis()
            
            # 6. 分类变量分析
            categorical_analysis = self.categorical_analysis()
            
            # 7. 相关性分析
            correlation_results = self.correlation_analysis()
            
            # 8. 高级分析
            advanced_results = self.advanced_analysis()
            
            # 9. 生成综合报告
            final_report = self.generate_comprehensive_report()
            
            print("\n🎉 数据探索分析完成！")
            print(f"📁 所有结果已保存到 '{self.output_dir}' 目录")
            
            return final_report
            
        except Exception as e:
            print(f"❌ 分析过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函数"""
    print("🚴‍♂️ 首尔自行车数据探索分析")
    print("="*80)
    
    # 创建探索器
    explorer = BikeDataExplorer('SeoulBikeData.csv')
    
    # 运行完整分析
    results = explorer.run_complete_exploration()
    
    if results:
        print("\n✨ 分析总结:")
        print("  - 数据质量检查完成")
        print("  - 时间模式识别完成")
        print("  - 天气影响分析完成")
        print("  - 特征相关性分析完成")
        print("  - 高级统计分析完成")
        print("  - 可视化图表已生成")
        print("  - 综合报告已保存")
        
        print(f"\n📂 输出文件位置:")
        print(f"  📊 EDA结果: outputs/eda/")
        print(f"  📈 图表: outputs/figures/")
    else:
        print("\n❌ 分析失败，请检查数据文件和代码")

if __name__ == "__main__":
    main() 