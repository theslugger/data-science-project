#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理模块
CDS503 Group Project - 首尔自行车需求预测
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class BikeDataPreprocessor:
    """自行车数据预处理类"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def load_data(self, file_path='SeoulBikeData.csv'):
        """加载数据"""
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='latin-1')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='cp1252')
        
        print(f"数据加载完成: {df.shape[0]} 行, {df.shape[1]} 列")
        return df
    
    def create_time_features(self, df):
        """创建时间特征"""
        df = df.copy()
        
        # 转换日期格式
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        
        # 基础时间特征
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['Weekday'] = df['Date'].dt.weekday  # 0=Monday
        df['Quarter'] = df['Date'].dt.quarter
        
        # 是否周末
        df['Is_Weekend'] = (df['Weekday'] >= 5).astype(int)
        
        # 小时的周期性编码
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        
        # 一年中的天数
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['DayOfYear_Sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
        df['DayOfYear_Cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
        
        print("时间特征创建完成")
        return df
    
    def create_interaction_features(self, df):
        """创建交互特征"""
        df = df.copy()
        
        # 温度相关交互特征
        df['Temp_Humidity'] = df['Temperature(°C)'] * df['Humidity(%)']
        df['Temp_Hour'] = df['Temperature(°C)'] * df['Hour']
        df['Temp_Weekend'] = df['Temperature(°C)'] * df['Is_Weekend']
        
        # 天气组合特征
        df['Weather_Index'] = (df['Temperature(°C)'] + 20) / 60 * \
                             (100 - df['Humidity(%)']) / 100 * \
                             (df['Visibility (10m)'] / 2000)
        
        # 通勤时间标识
        df['Rush_Hour'] = ((df['Hour'].between(7, 9)) | 
                          (df['Hour'].between(17, 19))).astype(int)
        
        # 降水相关
        df['Has_Rain'] = (df['Rainfall(mm)'] > 0).astype(int)
        df['Has_Snow'] = (df['Snowfall (cm)'] > 0).astype(int)
        df['Precipitation'] = df['Rainfall(mm)'] + df['Snowfall (cm)']
        
        print("交互特征创建完成")
        return df
    
    def create_lag_features(self, df, target_col='Rented Bike Count'):
        """创建滞后特征（注意数据泄露）"""
        df = df.copy()
        df = df.sort_values(['Date', 'Hour']).reset_index(drop=True)
        
        # 1小时前的租借量
        df['Rent_Lag_1h'] = df[target_col].shift(1)
        
        # 24小时前的租借量（同一时间昨天）
        df['Rent_Lag_24h'] = df[target_col].shift(24)
        
        # 7天前同一时间的租借量
        df['Rent_Lag_168h'] = df[target_col].shift(168)
        
        # 填充缺失值
        df['Rent_Lag_1h'].fillna(df[target_col].mean(), inplace=True)
        df['Rent_Lag_24h'].fillna(df[target_col].mean(), inplace=True)
        df['Rent_Lag_168h'].fillna(df[target_col].mean(), inplace=True)
        
        print("滞后特征创建完成")
        return df
    
    def encode_categorical_features(self, df):
        """编码分类特征"""
        df = df.copy()
        
        # 对分类特征进行独热编码
        categorical_cols = ['Seasons', 'Holiday', 'Functioning Day']
        
        for col in categorical_cols:
            # 创建独热编码
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
        
        # 删除原始分类列
        df.drop(categorical_cols, axis=1, inplace=True)
        
        print("分类特征编码完成")
        return df
    
    def remove_outliers(self, df, target_col='Rented Bike Count', method='iqr', factor=1.5):
        """移除异常值"""
        df = df.copy()
        
        if method == 'iqr':
            Q1 = df[target_col].quantile(0.25)
            Q3 = df[target_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            outliers_mask = (df[target_col] < lower_bound) | (df[target_col] > upper_bound)
            outliers_count = outliers_mask.sum()
            
            print(f"检测到 {outliers_count} 个异常值 ({outliers_count/len(df)*100:.2f}%)")
            
            # 不删除异常值，而是进行winsorization
            df.loc[df[target_col] < lower_bound, target_col] = lower_bound
            df.loc[df[target_col] > upper_bound, target_col] = upper_bound
            
            print("异常值处理完成（使用winsorization）")
        
        return df
    
    def handle_non_operating_days(self, df):
        """处理非运营日数据"""
        print("🚫 处理非运营日数据...")
        
        # 标记非运营日
        non_operating_mask = df['Functioning Day'] == 'No'
        non_operating_count = non_operating_mask.sum()
        
        print(f"发现 {non_operating_count} 条非运营日记录 ({non_operating_count/len(df)*100:.2f}%)")
        
        # 排除非运营日数据
        df_filtered = df[~non_operating_mask].copy()
        
        print(f"排除后剩余 {len(df_filtered)} 条记录")
        return df_filtered
    
    def create_enhanced_weather_features(self, df):
        """基于数据洞察创建增强天气特征"""
        print("🌤️ 创建增强天气特征...")
        df = df.copy()
        
        # 1. 温度分段特征（基于数据分析发现的阈值）
        df['Temp_Severe_Cold'] = (df['Temperature(°C)'] < 0).astype(int)
        df['Temp_Cold'] = ((df['Temperature(°C)'] >= 0) & (df['Temperature(°C)'] < 10)).astype(int)
        df['Temp_Cool'] = ((df['Temperature(°C)'] >= 10) & (df['Temperature(°C)'] < 20)).astype(int)
        df['Temp_Warm'] = ((df['Temperature(°C)'] >= 20) & (df['Temperature(°C)'] < 30)).astype(int)
        df['Temp_Hot'] = (df['Temperature(°C)'] >= 30).astype(int)
        
        # 2. 湿度阈值特征（基于数据分析发现）
        df['Humidity_Low'] = (df['Humidity(%)'] < 30).astype(int)
        df['Humidity_Medium'] = ((df['Humidity(%)'] >= 30) & (df['Humidity(%)'] < 50)).astype(int)
        df['Humidity_High'] = ((df['Humidity(%)'] >= 50) & (df['Humidity(%)'] < 70)).astype(int)
        df['Humidity_Very_High'] = (df['Humidity(%)'] >= 70).astype(int)
        
        # 3. 降水阈值特征
        df['Has_Rain'] = (df['Rainfall(mm)'] > 0).astype(int)
        df['Has_Snow'] = (df['Snowfall (cm)'] > 0).astype(int)
        df['Has_Precipitation'] = ((df['Rainfall(mm)'] > 0) | (df['Snowfall (cm)'] > 0)).astype(int)
        
        # 4. 风速分级
        df['Wind_Calm'] = (df['Wind speed (m/s)'] < 2).astype(int)
        df['Wind_Light'] = ((df['Wind speed (m/s)'] >= 2) & (df['Wind speed (m/s)'] < 4)).astype(int)
        df['Wind_Moderate'] = ((df['Wind speed (m/s)'] >= 4) & (df['Wind speed (m/s)'] < 6)).astype(int)
        df['Wind_Strong'] = (df['Wind speed (m/s)'] >= 6).astype(int)
        
        print("增强天气特征创建完成")
        return df
    
    def create_enhanced_time_features(self, df):
        """基于双峰模式创建增强时间特征"""
        print("⏰ 创建增强时间特征（双峰模式）...")
        df = df.copy()
        
        # 1. 基于数据分析的时间段划分
        df['Hour_Deep_Night'] = (df['Hour'].isin([0, 1, 2, 3, 4, 5])).astype(int)  # 深夜低谷
        df['Hour_Early_Morning'] = (df['Hour'].isin([6, 7])).astype(int)  # 早晨上升
        df['Hour_Morning_Peak'] = (df['Hour'].isin([8, 9])).astype(int)  # 早高峰
        df['Hour_Morning_Decline'] = (df['Hour'].isin([10, 11, 12])).astype(int)  # 上午下降
        df['Hour_Afternoon'] = (df['Hour'].isin([13, 14, 15, 16])).astype(int)  # 下午平稳
        df['Hour_Evening_Peak'] = (df['Hour'].isin([17, 18, 19])).astype(int)  # 晚高峰
        df['Hour_Evening_Decline'] = (df['Hour'].isin([20, 21, 22, 23])).astype(int)  # 晚间下降
        
        # 2. 双峰特征
        df['Is_Peak_Hour'] = ((df['Hour'].isin([8, 17, 18, 19]))).astype(int)
        df['Is_Low_Hour'] = ((df['Hour'].isin([3, 4, 5]))).astype(int)
        
        # 3. 季节性周期特征（基于数据分析的季节差异）
        df['Season_Summer'] = (df['Seasons'] == 'Summer').astype(int)
        df['Season_Winter'] = (df['Seasons'] == 'Winter').astype(int)
        df['Season_Spring'] = (df['Seasons'] == 'Spring').astype(int)
        df['Season_Autumn'] = (df['Seasons'] == 'Autumn').astype(int)
        
        # 4. 月份的周期性编码（捕捉季节变化）
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # 5. 工作日vs周末的细分
        df['Is_Weekday'] = (df['Weekday'] < 5).astype(int)
        df['Is_Weekend'] = (df['Weekday'] >= 5).astype(int)
        df['Is_Friday'] = (df['Weekday'] == 4).astype(int)  # 周五特殊
        df['Is_Monday'] = (df['Weekday'] == 0).astype(int)  # 周一特殊
        
        print("增强时间特征创建完成")
        return df
    
    def create_comfort_and_extreme_features(self, df):
        """创建舒适度和极端天气特征"""
        print("😌 创建舒适度和极端天气特征...")
        df = df.copy()
        
        # 1. 舒适度指数（基于温度和湿度）
        # 基于数据分析：最佳温度20-30°C，湿度30-70%
        df['Comfort_Index'] = np.where(
            (df['Temperature(°C)'].between(20, 30)) & (df['Humidity(%)'].between(30, 70)),
            1.0,  # 最舒适
            np.where(
                (df['Temperature(°C)'].between(10, 35)) & (df['Humidity(%)'].between(20, 80)),
                0.7,  # 较舒适
                np.where(
                    (df['Temperature(°C)'].between(0, 40)) & (df['Humidity(%)'].between(10, 90)),
                    0.4,  # 一般
                    0.1   # 不舒适
                )
            )
        )
        
        # 2. 体感温度（Heat Index简化版）
        df['Heat_Index'] = df['Temperature(°C)'] + 0.5 * (df['Humidity(%)'] - 50) / 100 * df['Temperature(°C)']
        
        # 3. 极端天气标识
        df['Extreme_Cold'] = (df['Temperature(°C)'] < -10).astype(int)
        df['Extreme_Hot'] = (df['Temperature(°C)'] > 35).astype(int)
        df['Extreme_Humid'] = (df['Humidity(%)'] > 90).astype(int)
        df['Extreme_Dry'] = (df['Humidity(%)'] < 20).astype(int)
        df['Heavy_Rain'] = (df['Rainfall(mm)'] > 10).astype(int)
        df['Heavy_Snow'] = (df['Snowfall (cm)'] > 5).astype(int)
        
        # 4. 恶劣天气组合
        df['Bad_Weather'] = (
            (df['Extreme_Cold'] == 1) |
            (df['Extreme_Hot'] == 1) |
            (df['Heavy_Rain'] == 1) |
            (df['Heavy_Snow'] == 1) |
            (df['Extreme_Humid'] == 1)
        ).astype(int)
        
        # 5. 完美天气（高需求预期）
        df['Perfect_Weather'] = (
            (df['Temperature(°C)'].between(20, 28)) &
            (df['Humidity(%)'].between(40, 60)) &
            (df['Rainfall(mm)'] == 0) &
            (df['Snowfall (cm)'] == 0) &
            (df['Wind speed (m/s)'] < 3)
        ).astype(int)
        
        print("舒适度和极端天气特征创建完成")
        return df
    
    def create_enhanced_interaction_features(self, df):
        """创建增强交互特征"""
        print("🔗 创建增强交互特征...")
        df = df.copy()
        
        # 1. 温度×季节交互特征（重点优化）
        df['Temp_Summer'] = df['Temperature(°C)'] * df['Season_Summer']
        df['Temp_Winter'] = df['Temperature(°C)'] * df['Season_Winter']
        df['Temp_Spring'] = df['Temperature(°C)'] * df['Season_Spring']
        df['Temp_Autumn'] = df['Temperature(°C)'] * df['Season_Autumn']
        
        # 2. 时间×天气交互
        df['Peak_Hour_Good_Weather'] = df['Is_Peak_Hour'] * (1 - df['Bad_Weather'])
        df['Peak_Hour_Bad_Weather'] = df['Is_Peak_Hour'] * df['Bad_Weather']
        df['Weekend_Good_Weather'] = df['Is_Weekend'] * (1 - df['Bad_Weather'])
        
        # 3. 舒适度×时间交互
        df['Comfort_Peak'] = df['Comfort_Index'] * df['Is_Peak_Hour']
        df['Comfort_Weekend'] = df['Comfort_Index'] * df['Is_Weekend']
        
        # 4. 季节×时间交互（捕捉季节性使用模式差异）
        df['Summer_Peak'] = df['Season_Summer'] * df['Is_Peak_Hour']
        df['Winter_Peak'] = df['Season_Winter'] * df['Is_Peak_Hour']
        df['Summer_Weekend'] = df['Season_Summer'] * df['Is_Weekend']
        df['Winter_Weekend'] = df['Season_Winter'] * df['Is_Weekend']
        
        print("增强交互特征创建完成")
        return df
    
    def prepare_features(self, df, use_lag_features=False, exclude_non_operating=True):
        """基于数据洞察的增强特征工程"""
        print("🚀 开始基于数据洞察的增强特征工程...")
        df = df.copy()
        
        # 1. 处理非运营日数据
        if exclude_non_operating:
            df = self.handle_non_operating_days(df)
        
        # 2. 基础时间特征
        df = self.create_time_features(df)
        
        # 3. 增强天气特征
        df = self.create_enhanced_weather_features(df)
        
        # 4. 增强时间特征（双峰模式）
        df = self.create_enhanced_time_features(df)
        
        # 5. 舒适度和极端天气特征
        df = self.create_comfort_and_extreme_features(df)
        
        # 6. 原有交互特征
        df = self.create_interaction_features(df)
        
        # 7. 增强交互特征
        df = self.create_enhanced_interaction_features(df)
        
        # 8. 滞后特征
        if use_lag_features:
            df = self.create_lag_features(df)
        
        # 9. 编码分类特征
        df = self.encode_categorical_features(df)
        
        # 10. 处理异常值
        df = self.remove_outliers(df)
        
        # 选择特征列（排除原始日期和目标变量）
        exclude_cols = ['Date', 'Rented Bike Count']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        self.feature_names = feature_cols
        
        print(f"✅ 最终特征数量: {len(feature_cols)}")
        print(f"📋 特征列表: {feature_cols}")
        
        return df
    
    def split_data_temporal(self, df, target_col='Rented Bike Count', 
                           train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """时间序列友好的数据分割"""
        
        # 按时间排序
        df = df.sort_values(['Date', 'Hour']).reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # 分割数据
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        print(f"数据分割完成:")
        print(f"  训练集: {len(train_df)} 条 ({len(train_df)/n*100:.1f}%)")
        print(f"  验证集: {len(val_df)} 条 ({len(val_df)/n*100:.1f}%)")
        print(f"  测试集: {len(test_df)} 条 ({len(test_df)/n*100:.1f}%)")
        
        # 提取特征和目标
        feature_cols = self.feature_names
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]
        
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test):
        """特征标准化"""
        # 只在数值特征上进行标准化
        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()
        
        # 拟合标准化器
        self.scaler.fit(X_train[numeric_features])
        
        # 应用标准化
        X_train_scaled[numeric_features] = self.scaler.transform(X_train[numeric_features])
        X_val_scaled[numeric_features] = self.scaler.transform(X_val[numeric_features])
        X_test_scaled[numeric_features] = self.scaler.transform(X_test[numeric_features])
        
        print("特征标准化完成")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def get_time_series_cv(self, n_splits=5):
        """获取时间序列交叉验证器"""
        return TimeSeriesSplit(n_splits=n_splits)
    
    def prepare_stratified_data(self, df):
        """为分层建模准备数据"""
        print("🎯 准备分层建模数据...")
        
        # 创建分层标识
        stratified_data = {}
        
        # 1. 按运营状态分层（如果有非运营日）
        if 'Functioning Day' in df.columns:
            operating_data = df[df['Functioning Day'] == 'Yes'].copy()
        else:
            operating_data = df.copy()
        
        stratified_data['operating'] = operating_data
        
        # 2. 按季节分层（使用独热编码后的季节特征）
        season_mapping = {
            'spring': 'Seasons_Spring',
            'summer': 'Seasons_Summer', 
            'autumn': 'Seasons_Winter',  # 注意：由于drop_first=True，Autumn被表示为其他季节都为0
            'winter': 'Seasons_Winter'
        }
        
        for season_name, season_col in season_mapping.items():
            if season_name == 'autumn':
                # Autumn是参考类别，所有其他季节列都为0
                season_data = operating_data[
                    (operating_data.get('Seasons_Spring', 0) == 0) & 
                    (operating_data.get('Seasons_Summer', 0) == 0) & 
                    (operating_data.get('Seasons_Winter', 0) == 0)
                ].copy()
            elif season_col in operating_data.columns:
                season_data = operating_data[operating_data[season_col] == 1].copy()
            else:
                print(f"  ⚠️  {season_name.title()} 列 {season_col} 不存在，跳过")
                continue
                
            if len(season_data) > 0:
                stratified_data[f'season_{season_name}'] = season_data
                print(f"  {season_name.title()}: {len(season_data)} 条记录")
            else:
                print(f"  ⚠️  {season_name.title()}: 0 条记录")
        
        # 3. 按天气条件分层
        good_weather = operating_data[
            (operating_data['Temperature(°C)'].between(10, 30)) &
            (operating_data['Humidity(%)'].between(30, 70)) &
            (operating_data['Rainfall(mm)'] == 0) &
            (operating_data['Snowfall (cm)'] == 0)
        ].copy()
        
        bad_weather = operating_data[
            (operating_data['Temperature(°C)'] < 0) |
            (operating_data['Temperature(°C)'] > 35) |
            (operating_data['Rainfall(mm)'] > 5) |
            (operating_data['Snowfall (cm)'] > 2)
        ].copy()
        
        stratified_data['good_weather'] = good_weather
        stratified_data['bad_weather'] = bad_weather
        
        print(f"  好天气: {len(good_weather)} 条记录")
        print(f"  恶劣天气: {len(bad_weather)} 条记录")
        
        # 4. 按时间段分层
        peak_hours = operating_data[operating_data['Hour'].isin([8, 17, 18, 19])].copy()
        off_peak_hours = operating_data[~operating_data['Hour'].isin([8, 17, 18, 19])].copy()
        
        stratified_data['peak_hours'] = peak_hours
        stratified_data['off_peak_hours'] = off_peak_hours
        
        print(f"  高峰时段: {len(peak_hours)} 条记录")
        print(f"  非高峰时段: {len(off_peak_hours)} 条记录")
        
        return stratified_data

def main():
    """主函数 - 演示数据预处理流程"""
    
    # 初始化预处理器
    preprocessor = BikeDataPreprocessor()
    
    # 加载数据
    df = preprocessor.load_data()
    
    # 准备特征
    df_processed = preprocessor.prepare_features(df, use_lag_features=True)
    
    # 分割数据
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data_temporal(df_processed)
    
    # 标准化特征
    X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_val, X_test)
    
    print("\n数据预处理完成!")
    print(f"训练特征形状: {X_train_scaled.shape}")
    print(f"训练目标形状: {y_train.shape}")
    
    # 保存处理后的数据
    print("\n保存预处理数据...")
    np.save('X_train.npy', X_train_scaled.values)
    np.save('X_val.npy', X_val_scaled.values)
    np.save('X_test.npy', X_test_scaled.values)
    np.save('y_train.npy', y_train.values)
    np.save('y_val.npy', y_val.values)
    np.save('y_test.npy', y_test.values)
    
    # 保存特征名称
    with open('feature_names.txt', 'w') as f:
        for name in preprocessor.feature_names:
            f.write(f"{name}\n")
    
    print("预处理数据已保存到文件")

if __name__ == "__main__":
    main() 