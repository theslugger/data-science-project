#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
首尔自行车共享数据探索性分析
CDS503 Group Project - EDA Script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
print("=" * 60)
print("首尔自行车共享数据探索性分析")
print("=" * 60)

df = pd.read_csv('SeoulBikeData.csv', encoding='utf-8')

# 1. 基本数据信息
print("\n1. 数据基本信息")
print("-" * 40)
print(f"数据集形状: {df.shape}")
print(f"特征数量: {df.shape[1]}")
print(f"样本数量: {df.shape[0]}")

print("\n列名:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

# 2. 数据类型和缺失值
print("\n2. 数据类型和缺失值分析")
print("-" * 40)
info_df = pd.DataFrame({
    '列名': df.columns,
    '数据类型': df.dtypes,
    '非空值数量': df.count(),
    '缺失值数量': df.isnull().sum(),
    '缺失值比例(%)': (df.isnull().sum() / len(df) * 100).round(2)
})
print(info_df.to_string(index=False))

# 3. 目标变量分析
print("\n3. 目标变量分析 (Rented Bike Count)")
print("-" * 40)
target = df['Rented Bike Count']
print(f"最小值: {target.min()}")
print(f"最大值: {target.max()}")
print(f"平均值: {target.mean():.2f}")
print(f"中位数: {target.median():.2f}")
print(f"标准差: {target.std():.2f}")
print(f"偏度: {target.skew():.2f}")
print(f"峰度: {target.kurtosis():.2f}")

# 4. 数值特征统计描述
print("\n4. 数值特征统计描述")
print("-" * 40)
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(df[numeric_cols].describe().round(2))

# 5. 分类特征分析
print("\n5. 分类特征分析")
print("-" * 40)
categorical_cols = ['Seasons', 'Holiday', 'Functioning Day']
for col in categorical_cols:
    print(f"\n{col} 分布:")
    value_counts = df[col].value_counts()
    percentages = (df[col].value_counts(normalize=True) * 100).round(2)
    for val, count, pct in zip(value_counts.index, value_counts.values, percentages.values):
        print(f"  {val}: {count} ({pct}%)")

# 6. 时间特征分析
print("\n6. 时间特征分析")
print("-" * 40)
print("小时分布:")
hour_dist = df['Hour'].value_counts().sort_index()
for hour, count in hour_dist.items():
    print(f"  {hour:2d}点: {count:4d} 条记录")

# 7. 数据质量检查
print("\n7. 数据质量检查")
print("-" * 40)
# 检查异常值
print("异常值检查 (使用IQR方法):")
for col in numeric_cols:
    if col != 'Hour':  # 小时是正常的0-23范围
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if len(outliers) > 0:
            print(f"  {col}: {len(outliers)} 个异常值 ({len(outliers)/len(df)*100:.2f}%)")

# 8. 相关性分析
print("\n8. 数值特征相关性分析")
print("-" * 40)
# 计算与目标变量的相关性
correlations = df[numeric_cols].corr()['Rented Bike Count'].abs().sort_values(ascending=False)
print("与目标变量的相关性 (绝对值):")
for feature, corr in correlations.items():
    if feature != 'Rented Bike Count':
        print(f"  {feature}: {corr:.3f}")

# 9. 可视化分析
print("\n9. 生成可视化图表...")
print("-" * 40)

# 创建图表
fig = plt.figure(figsize=(20, 24))

# 9.1 目标变量分布
plt.subplot(4, 3, 1)
plt.hist(df['Rented Bike Count'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('租借自行车数量分布', fontsize=14, fontweight='bold')
plt.xlabel('租借数量')
plt.ylabel('频次')

# 9.2 按小时的租借量
plt.subplot(4, 3, 2)
hourly_avg = df.groupby('Hour')['Rented Bike Count'].mean()
plt.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, markersize=6)
plt.title('每小时平均租借量', fontsize=14, fontweight='bold')
plt.xlabel('小时')
plt.ylabel('平均租借数量')
plt.grid(True, alpha=0.3)

# 9.3 按季节的租借量
plt.subplot(4, 3, 3)
seasonal_avg = df.groupby('Seasons')['Rented Bike Count'].mean()
plt.bar(seasonal_avg.index, seasonal_avg.values, color=['lightblue', 'lightgreen', 'orange', 'lightcoral'])
plt.title('各季节平均租借量', fontsize=14, fontweight='bold')
plt.xlabel('季节')
plt.ylabel('平均租借数量')
plt.xticks(rotation=45)

# 9.4 温度 vs 租借量
plt.subplot(4, 3, 4)
plt.scatter(df['Temperature(°C)'], df['Rented Bike Count'], alpha=0.6, s=20)
plt.title('温度 vs 租借量', fontsize=14, fontweight='bold')
plt.xlabel('温度 (°C)')
plt.ylabel('租借数量')

# 9.5 湿度 vs 租借量
plt.subplot(4, 3, 5)
plt.scatter(df['Humidity(%)'], df['Rented Bike Count'], alpha=0.6, s=20, color='orange')
plt.title('湿度 vs 租借量', fontsize=14, fontweight='bold')
plt.xlabel('湿度 (%)')
plt.ylabel('租借数量')

# 9.6 风速 vs 租借量
plt.subplot(4, 3, 6)
plt.scatter(df['Wind speed (m/s)'], df['Rented Bike Count'], alpha=0.6, s=20, color='green')
plt.title('风速 vs 租借量', fontsize=14, fontweight='bold')
plt.xlabel('风速 (m/s)')
plt.ylabel('租借数量')

# 9.7 假期 vs 非假期
plt.subplot(4, 3, 7)
holiday_avg = df.groupby('Holiday')['Rented Bike Count'].mean()
plt.bar(holiday_avg.index, holiday_avg.values, color=['lightcoral', 'lightblue'])
plt.title('假期 vs 非假期平均租借量', fontsize=14, fontweight='bold')
plt.xlabel('假期状态')
plt.ylabel('平均租借数量')

# 9.8 运营日 vs 非运营日
plt.subplot(4, 3, 8)
functioning_avg = df.groupby('Functioning Day')['Rented Bike Count'].mean()
plt.bar(functioning_avg.index, functioning_avg.values, color=['red', 'green'])
plt.title('运营日 vs 非运营日平均租借量', fontsize=14, fontweight='bold')
plt.xlabel('运营状态')
plt.ylabel('平均租借数量')

# 9.9 降雨量 vs 租借量
plt.subplot(4, 3, 9)
plt.scatter(df['Rainfall(mm)'], df['Rented Bike Count'], alpha=0.6, s=20, color='purple')
plt.title('降雨量 vs 租借量', fontsize=14, fontweight='bold')
plt.xlabel('降雨量 (mm)')
plt.ylabel('租借数量')

# 9.10 相关性热力图
plt.subplot(4, 3, 10)
# 选择主要数值特征
main_features = ['Rented Bike Count', 'Temperature(°C)', 'Humidity(%)', 
                'Wind speed (m/s)', 'Visibility (10m)', 'Solar Radiation (MJ/m2)', 
                'Rainfall(mm)', 'Hour']
corr_matrix = df[main_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', cbar_kws={"shrink": .8})
plt.title('特征相关性热力图', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# 9.11 一周中各天的模式 (如果有日期信息)
plt.subplot(4, 3, 11)
# 转换日期格式并提取星期
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['Weekday'] = df['Date'].dt.day_name()
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_avg = df.groupby('Weekday')['Rented Bike Count'].mean().reindex(weekday_order)
plt.bar(range(len(weekday_avg)), weekday_avg.values, color='lightsteelblue')
plt.title('一周各天平均租借量', fontsize=14, fontweight='bold')
plt.xlabel('星期')
plt.ylabel('平均租借数量')
plt.xticks(range(len(weekday_avg)), ['周一', '周二', '周三', '周四', '周五', '周六', '周日'])

# 9.12 月份模式
plt.subplot(4, 3, 12)
df['Month'] = df['Date'].dt.month
monthly_avg = df.groupby('Month')['Rented Bike Count'].mean()
plt.plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2, markersize=8, color='darkgreen')
plt.title('各月份平均租借量', fontsize=14, fontweight='bold')
plt.xlabel('月份')
plt.ylabel('平均租借数量')
plt.grid(True, alpha=0.3)
plt.xticks(range(1, 13))

plt.tight_layout()
plt.savefig('seoul_bike_eda.png', dpi=300, bbox_inches='tight')
print("可视化图表已保存为 'seoul_bike_eda.png'")

# 10. 总结和见解
print("\n10. 数据探索总结")
print("=" * 60)
print("关键发现:")
print("1. 数据质量: 无缺失值，数据相对完整")
print("2. 目标变量: 租借量分布右偏，存在高峰和低谷")
print("3. 时间模式: 明显的通勤高峰 (7-9点, 17-19点)")
print("4. 季节影响: 不同季节的租借模式存在差异")
print("5. 天气影响: 温度、湿度、降雨等天气因素与租借量相关")
print("6. 假期效应: 假期和非假期的租借模式不同")
print("7. 运营状态: 非运营日租借量显著降低")

print("\n建议的预处理步骤:")
print("1. 特征工程: 提取更多时间特征 (周末/工作日, 月份等)")
print("2. 异常值处理: 处理极端天气条件的异常值")
print("3. 特征缩放: 对数值特征进行标准化或归一化")
print("4. 编码处理: 对分类变量进行独热编码")
print("5. 交互特征: 创建天气和时间的交互特征")

print("\nEDA分析完成!") 