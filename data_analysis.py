#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seoul Bike Data 深度分析
重新观察数据，寻找新的洞察
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_explore_data():
    """加载并探索数据"""
    print('📊 重新加载和观察Seoul Bike数据...')
    print('='*60)
    
    # 加载数据
    df = pd.read_csv('SeoulBikeData.csv', encoding='unicode_escape')
    
    print(f'\n🔍 数据基本信息:')
    print(f'数据形状: {df.shape}')
    print(f'时间范围: {df["Date"].min()} 到 {df["Date"].max()}')
    print(f'内存使用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB')
    
    print(f'\n📋 列信息:')
    for i, col in enumerate(df.columns):
        null_count = df[col].isnull().sum()
        print(f'{i+1:2d}. {col:30s} - {str(df[col].dtype):10s} - 缺失值: {null_count}')
    
    return df

def analyze_target_variable(df):
    """分析目标变量"""
    target_col = 'Rented Bike Count'
    print(f'\n📈 目标变量 "{target_col}" 深度分析:')
    print('='*50)
    
    # 基础统计
    print(f'平均值: {df[target_col].mean():.2f}')
    print(f'中位数: {df[target_col].median():.2f}')
    print(f'标准差: {df[target_col].std():.2f}')
    print(f'最小值: {df[target_col].min()}')
    print(f'最大值: {df[target_col].max()}')
    print(f'零值数量: {(df[target_col] == 0).sum()}')
    print(f'零值比例: {(df[target_col] == 0).mean()*100:.2f}%')
    
    # 分位数分析
    print(f'\n分位数分析:')
    for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        print(f'Q{q*100:4.0f}: {df[target_col].quantile(q):8.2f}')
    
    # 异常值分析
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[target_col] < lower_bound) | (df[target_col] > upper_bound)]
    print(f'\nIQR异常值分析:')
    print(f'下界: {lower_bound:.2f}, 上界: {upper_bound:.2f}')
    print(f'异常值数量: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)')
    
    return df[target_col]

def analyze_time_patterns(df):
    """分析时间模式"""
    print(f'\n⏰ 时间模式分析:')
    print('='*40)
    
    # 转换时间（处理混合日期格式）
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.dayofweek
    df['Weekend'] = (df['Weekday'] >= 5).astype(int)
    
    target_col = 'Rented Bike Count'
    
    # 按年分析
    print('\n📅 按年份分析:')
    yearly_stats = df.groupby('Year')[target_col].agg(['count', 'mean', 'std', 'min', 'max'])
    print(yearly_stats)
    
    # 按月分析
    print('\n📅 按月份分析:')
    monthly_stats = df.groupby('Month')[target_col].agg(['count', 'mean', 'std'])
    print(monthly_stats)
    
    # 按小时分析
    print('\n🕐 按小时分析:')
    hourly_stats = df.groupby('Hour')[target_col].agg(['mean', 'std'])
    print(hourly_stats)
    
    # 工作日vs周末
    print('\n📊 工作日 vs 周末:')
    weekend_stats = df.groupby('Weekend')[target_col].agg(['count', 'mean', 'std'])
    weekend_stats.index = ['工作日', '周末']
    print(weekend_stats)
    
    return df

def analyze_weather_impact(df):
    """分析天气影响"""
    print(f'\n🌤️ 天气因素影响分析:')
    print('='*45)
    
    target_col = 'Rented Bike Count'
    
    # 温度影响
    print('\n🌡️ 温度分析:')
    temp_col = 'Temperature(°C)'
    print(f'温度范围: {df[temp_col].min():.1f}°C 到 {df[temp_col].max():.1f}°C')
    
    # 温度分段分析
    df['Temp_Range'] = pd.cut(df[temp_col], bins=[-50, 0, 10, 20, 30, 50], 
                             labels=['严寒(<0°C)', '寒冷(0-10°C)', '凉爽(10-20°C)', '温暖(20-30°C)', '炎热(>30°C)'])
    temp_impact = df.groupby('Temp_Range')[target_col].agg(['count', 'mean', 'std'])
    print(temp_impact)
    
    # 湿度影响
    print('\n💧 湿度分析:')
    humidity_col = 'Humidity(%)'
    df['Humidity_Range'] = pd.cut(df[humidity_col], bins=[0, 30, 50, 70, 100], 
                                 labels=['低湿度(<30%)', '中等湿度(30-50%)', '高湿度(50-70%)', '极高湿度(>70%)'])
    humidity_impact = df.groupby('Humidity_Range')[target_col].agg(['count', 'mean', 'std'])
    print(humidity_impact)
    
    # 风速影响
    print('\n💨 风速分析:')
    wind_col = 'Wind speed (m/s)'
    print(f'风速范围: {df[wind_col].min():.1f} 到 {df[wind_col].max():.1f} m/s')
    df['Wind_Range'] = pd.cut(df[wind_col], bins=[0, 2, 4, 6, 20], 
                             labels=['微风(<2m/s)', '轻风(2-4m/s)', '和风(4-6m/s)', '强风(>6m/s)'])
    wind_impact = df.groupby('Wind_Range')[target_col].agg(['count', 'mean', 'std'])
    print(wind_impact)
    
    # 降雨影响
    print('\n🌧️ 降雨分析:')
    rain_col = 'Rainfall(mm)'
    rain_stats = df.groupby(df[rain_col] > 0)[target_col].agg(['count', 'mean', 'std'])
    rain_stats.index = ['无雨', '有雨']
    print(rain_stats)
    print(f'降雨天数: {(df[rain_col] > 0).sum()} ({(df[rain_col] > 0).mean()*100:.1f}%)')
    
    # 降雪影响
    print('\n❄️ 降雪分析:')
    snow_col = 'Snowfall (cm)'
    snow_stats = df.groupby(df[snow_col] > 0)[target_col].agg(['count', 'mean', 'std'])
    snow_stats.index = ['无雪', '有雪']
    print(snow_stats)
    print(f'降雪天数: {(df[snow_col] > 0).sum()} ({(df[snow_col] > 0).mean()*100:.1f}%)')
    
    return df

def analyze_special_conditions(df):
    """分析特殊条件"""
    print(f'\n🏷️ 特殊条件分析:')
    print('='*35)
    
    target_col = 'Rented Bike Count'
    
    # 季节分析
    print('\n🍃 季节分析:')
    seasonal_stats = df.groupby('Seasons')[target_col].agg(['count', 'mean', 'std'])
    print(seasonal_stats)
    
    # 假期分析
    print('\n🏖️ 假期分析:')
    holiday_stats = df.groupby('Holiday')[target_col].agg(['count', 'mean', 'std'])
    print(holiday_stats)
    
    # 运营状态分析
    print('\n🚴 运营状态分析:')
    functioning_stats = df.groupby('Functioning Day')[target_col].agg(['count', 'mean', 'std'])
    print(functioning_stats)
    
    # 零值分析
    print('\n🔍 零值深度分析:')
    zero_mask = df[target_col] == 0
    zero_data = df[zero_mask]
    print(f'零值总数: {zero_mask.sum()}')
    
    if len(zero_data) > 0:
        print('\n零值条件分析:')
        print(f'运营状态: {zero_data["Functioning Day"].value_counts()}')
        print(f'季节分布: {zero_data["Seasons"].value_counts()}')
        print(f'假期分布: {zero_data["Holiday"].value_counts()}')
        
        print(f'\n零值时间分布:')
        print(f'小时分布: {zero_data["Hour"].value_counts().sort_index()}')
        
        print(f'\n零值天气条件:')
        print(f'平均温度: {zero_data["Temperature(°C)"].mean():.2f}°C')
        print(f'平均湿度: {zero_data["Humidity(%)"].mean():.2f}%')
        print(f'平均风速: {zero_data["Wind speed (m/s)"].mean():.2f} m/s')
        print(f'降雨天数: {(zero_data["Rainfall(mm)"] > 0).sum()}')
        print(f'降雪天数: {(zero_data["Snowfall (cm)"] > 0).sum()}')
    
    return df

def correlation_analysis(df):
    """相关性分析"""
    print(f'\n📊 特征相关性分析:')
    print('='*40)
    
    # 选择数值型特征
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_col = 'Rented Bike Count'
    
    if target_col in numeric_cols:
        # 计算与目标变量的相关性
        correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
        
        print(f'\n与目标变量的相关性排序:')
        for col, corr in correlations.items():
            if col != target_col:
                print(f'{col:30s}: {corr:.4f}')
        
        # 强相关特征（|r| > 0.3）
        strong_corr = correlations[correlations > 0.3]
        strong_corr = strong_corr[strong_corr.index != target_col]
        print(f'\n强相关特征 (|r| > 0.3):')
        for col, corr in strong_corr.items():
            print(f'{col:30s}: {corr:.4f}')
    
    return correlations

def identify_peak_patterns(df):
    """识别峰值模式"""
    print(f'\n📈 峰值模式识别:')
    print('='*35)
    
    target_col = 'Rented Bike Count'
    
    # 高峰时段识别
    top_90_percentile = df[target_col].quantile(0.9)
    peak_data = df[df[target_col] >= top_90_percentile]
    
    print(f'高峰阈值（90分位数）: {top_90_percentile:.0f}')
    print(f'高峰记录数: {len(peak_data)} ({len(peak_data)/len(df)*100:.2f}%)')
    
    print(f'\n高峰时段特征:')
    print(f'小时分布:')
    peak_hours = peak_data['Hour'].value_counts().sort_index()
    for hour, count in peak_hours.items():
        print(f'  {hour:2d}时: {count:3d}次 ({count/len(peak_data)*100:.1f}%)')
    
    print(f'\n季节分布:')
    peak_seasons = peak_data['Seasons'].value_counts()
    for season, count in peak_seasons.items():
        print(f'  {season}: {count:3d}次 ({count/len(peak_data)*100:.1f}%)')
    
    print(f'\n天气条件:')
    print(f'平均温度: {peak_data["Temperature(°C)"].mean():.2f}°C')
    print(f'平均湿度: {peak_data["Humidity(%)"].mean():.2f}%')
    print(f'平均风速: {peak_data["Wind speed (m/s)"].mean():.2f} m/s')
    
    # 低谷时段识别（排除零值）
    non_zero_data = df[df[target_col] > 0]
    bottom_10_percentile = non_zero_data[target_col].quantile(0.1)
    low_data = non_zero_data[non_zero_data[target_col] <= bottom_10_percentile]
    
    print(f'\n低谷时段特征:')
    print(f'低谷阈值（非零10分位数）: {bottom_10_percentile:.0f}')
    print(f'低谷记录数: {len(low_data)} ({len(low_data)/len(non_zero_data)*100:.2f}%)')
    
    print(f'小时分布:')
    low_hours = low_data['Hour'].value_counts().sort_index()
    for hour, count in low_hours.items():
        print(f'  {hour:2d}时: {count:3d}次 ({count/len(low_data)*100:.1f}%)')
    
    return peak_data, low_data

def main():
    """主函数"""
    # 加载数据
    df = load_and_explore_data()
    
    # 分析目标变量
    target_values = analyze_target_variable(df)
    
    # 时间模式分析
    df = analyze_time_patterns(df)
    
    # 天气影响分析
    df = analyze_weather_impact(df)
    
    # 特殊条件分析
    df = analyze_special_conditions(df)
    
    # 相关性分析
    correlations = correlation_analysis(df)
    
    # 峰值模式识别
    peak_data, low_data = identify_peak_patterns(df)
    
    print(f'\n🎯 数据洞察总结:')
    print('='*40)
    print('✅ 数据重新分析完成！')
    print('📝 发现的关键模式将有助于特征工程和模型优化')

if __name__ == "__main__":
    main() 