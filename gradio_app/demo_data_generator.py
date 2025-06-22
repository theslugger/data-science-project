#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示数据生成器
生成类似首尔自行车数据的示例数据用于测试
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_demo_data(num_records=1000):
    """生成演示数据"""
    
    print(f"🔄 正在生成 {num_records} 条演示数据...")
    
    # 设置随机种子确保可重现性
    np.random.seed(42)
    random.seed(42)
    
    # 生成日期时间序列
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(num_records)]
    
    # 初始化数据字典
    data = {
        'Date': [d.strftime('%d/%m/%Y') for d in dates],
        'Hour': [d.hour for d in dates],
        'Temperature(°C)': np.random.normal(15, 10, num_records),
        'Humidity(%)': np.random.normal(60, 20, num_records),
        'Wind speed (m/s)': np.random.exponential(2, num_records),
        'Visibility (10m)': np.random.normal(1500, 500, num_records),
        'Solar Radiation (MJ/m2)': np.random.exponential(1.5, num_records),
        'Rainfall(mm)': np.random.exponential(0.5, num_records),
        'Snowfall (cm)': np.random.exponential(0.1, num_records)
    }
    
    # 生成运营状态
    functioning_status = ['Yes'] * int(num_records * 0.95) + ['No'] * int(num_records * 0.05)
    random.shuffle(functioning_status)
    data['Functioning Day'] = functioning_status
    
    # 添加季节信息
    seasons = []
    for d in dates:
        month = d.month
        if month in [12, 1, 2]:
            seasons.append('Winter')
        elif month in [3, 4, 5]:
            seasons.append('Spring')
        elif month in [6, 7, 8]:
            seasons.append('Summer')
        else:
            seasons.append('Autumn')
    
    data['Seasons'] = seasons
    
    # 生成目标变量（自行车租赁数量）
    print("📊 生成目标变量...")
    bike_counts = []
    
    for i in range(num_records):
        hour = data['Hour'][i]
        temp = data['Temperature(°C)'][i]
        season = data['Seasons'][i]
        functioning = data['Functioning Day'][i]
        humidity = data['Humidity(%)'][i]
        rainfall = data['Rainfall(mm)'][i]
        
        # 基础需求量
        base_demand = 500
        
        # 时间因子（模拟通勤高峰）
        if hour in [8, 9, 17, 18, 19]:  # 高峰时段
            time_factor = 1.6
        elif hour in [10, 11, 12, 13, 14, 15, 16, 20]:  # 白天
            time_factor = 1.2
        elif hour in [6, 7, 21, 22]:  # 早晚
            time_factor = 0.8
        else:  # 深夜凌晨
            time_factor = 0.15
        
        # 温度因子（舒适温度需求更高）
        if 18 <= temp <= 25:
            temp_factor = 1.3
        elif 10 <= temp <= 30:
            temp_factor = 1.0
        elif temp < 0 or temp > 35:
            temp_factor = 0.3
        else:
            temp_factor = 0.7
        
        # 季节因子
        season_factors = {
            'Spring': 1.15, 
            'Summer': 1.25, 
            'Autumn': 1.05, 
            'Winter': 0.75
        }
        season_factor = season_factors[season]
        
        # 天气因子
        weather_factor = 1.0
        if rainfall > 5:  # 大雨
            weather_factor = 0.4
        elif rainfall > 0:  # 小雨
            weather_factor = 0.7
        
        if humidity > 85:  # 高湿度
            weather_factor *= 0.8
        elif humidity < 30:  # 低湿度
            weather_factor *= 0.9
        
        # 运营因子
        functioning_factor = 1.0 if functioning == 'Yes' else 0.0
        
        # 计算最终需求
        demand = (base_demand * time_factor * temp_factor * 
                 season_factor * weather_factor * functioning_factor)
        
        # 添加随机噪声
        demand = max(0, int(demand + np.random.normal(0, 50)))
        
        bike_counts.append(demand)
    
    data['Rented Bike Count'] = bike_counts
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 数值范围限制
    df['Temperature(°C)'] = np.clip(df['Temperature(°C)'], -20, 40)
    df['Humidity(%)'] = np.clip(df['Humidity(%)'], 0, 100)
    df['Wind speed (m/s)'] = np.clip(df['Wind speed (m/s)'], 0, 15)
    df['Visibility (10m)'] = np.clip(df['Visibility (10m)'], 100, 2000)
    df['Solar Radiation (MJ/m2)'] = np.clip(df['Solar Radiation (MJ/m2)'], 0, 10)
    df['Rainfall(mm)'] = np.clip(df['Rainfall(mm)'], 0, 50)
    df['Snowfall (cm)'] = np.clip(df['Snowfall (cm)'], 0, 20)
    
    # 数值精度处理
    for col in ['Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 
                'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)']:
        df[col] = df[col].round(2)
    
    return df

def save_demo_data():
    """保存演示数据到文件"""
    print("🚴‍♂️ 首尔自行车演示数据生成器")
    print("=" * 50)
    
    # 生成数据
    df = generate_demo_data(1000)
    
    # 保存文件
    filename = "seoul_bike_demo_data.csv"
    df.to_csv(filename, index=False, encoding='utf-8')
    
    print(f"\n✅ 演示数据已成功保存: {filename}")
    print(f"📊 数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"📋 特征列表:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\n📄 数据预览:")
    print(df.head().to_string(index=False))
    
    print(f"\n📈 目标变量统计:")
    target_col = 'Rented Bike Count'
    print(f"  平均值: {df[target_col].mean():.1f}")
    print(f"  中位数: {df[target_col].median():.1f}")
    print(f"  最小值: {df[target_col].min()}")
    print(f"  最大值: {df[target_col].max()}")
    print(f"  零值数: {(df[target_col] == 0).sum()}")
    
    print(f"\n🎯 使用方法:")
    print(f"  1. 启动Gradio应用: python app.py")
    print(f"  2. 上传生成的文件: {filename}")
    print(f"  3. 开始数据分析和特征工程")
    
    return filename

if __name__ == "__main__":
    save_demo_data()