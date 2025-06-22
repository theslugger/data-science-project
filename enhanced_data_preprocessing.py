#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版数据预处理
CDS503 Group Project - 首尔自行车需求预测
基于数据洞察的智能特征工程
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
import warnings

from config import config
from utils import (Logger, DataValidator, DataLoader, ResultSaver, 
                  print_section_header, get_timestamp)

warnings.filterwarnings('ignore')

class EnhancedDataPreprocessor:
    """增强版数据预处理类"""
    
    def __init__(self):
        self.logger = Logger(self.__class__.__name__)
        self.data_loader = DataLoader(self.logger)
        self.validator = DataValidator()
        self.result_saver = ResultSaver(self.logger)
        
        # 预处理组件
        self.scaler = None
        self.feature_selector = None
        self.label_encoders = {}
        
        # 存储特征信息
        self.feature_names = []
        self.feature_metadata = {}
        self.preprocessing_results = {}
        
    def load_and_validate_data(self, file_path=None):
        """加载并验证数据"""
        print_section_header("数据加载与初步验证", level=1)
        
        # 加载数据
        self.df = self.data_loader.load_data(file_path)
        
        # 验证必需列
        required_cols = [
            config.DATA_CONFIG['target_column'],
            config.DATA_CONFIG['date_column']
        ] + config.FEATURE_CONFIG['categorical_features']
        
        self.validator.validate_dataframe(self.df, required_cols)
        
        # 数据质量检查
        missing_info = self.validator.check_missing_values(self.df)
        if missing_info is not None:
            self.logger.warning(f"发现缺失值:\n{missing_info}")
        else:
            self.logger.info("✅ 数据完整，无缺失值")
        
        print(f"原始数据形状: {self.df.shape}")
        return self.df
    
    def handle_datetime_features(self):
        """处理日期时间特征"""
        print_section_header("日期时间特征处理", level=2)
        
        date_col = config.DATA_CONFIG['date_column']
        
        # 转换日期格式
        try:
            self.df[date_col] = pd.to_datetime(self.df[date_col], format='mixed', dayfirst=True)
        except:
            self.df[date_col] = pd.to_datetime(self.df[date_col])
        
        # 提取基础时间特征
        self.df['Year'] = self.df[date_col].dt.year
        self.df['Month'] = self.df[date_col].dt.month
        self.df['Day'] = self.df[date_col].dt.day
        self.df['Weekday'] = self.df[date_col].dt.weekday
        self.df['DayOfYear'] = self.df[date_col].dt.dayofyear
        self.df['Quarter'] = self.df[date_col].dt.quarter
        
        # 周末标识
        self.df['IsWeekend'] = (self.df['Weekday'] >= 5).astype(int)
        self.df['IsWeekday'] = (self.df['Weekday'] < 5).astype(int)
        
        # 特殊日期标识
        self.df['IsMonday'] = (self.df['Weekday'] == 0).astype(int)
        self.df['IsFriday'] = (self.df['Weekday'] == 4).astype(int)
        
        # 周期性编码（三角函数）
        if config.FEATURE_CONFIG['time_features']['use_hour_encoding']:
            self.df['Hour_Sin'] = np.sin(2 * np.pi * self.df['Hour'] / 24)
            self.df['Hour_Cos'] = np.cos(2 * np.pi * self.df['Hour'] / 24)
        
        if config.FEATURE_CONFIG['time_features']['use_day_encoding']:
            self.df['DayOfYear_Sin'] = np.sin(2 * np.pi * self.df['DayOfYear'] / 365)
            self.df['DayOfYear_Cos'] = np.cos(2 * np.pi * self.df['DayOfYear'] / 365)
        
        if config.FEATURE_CONFIG['time_features']['use_month_encoding']:
            self.df['Month_Sin'] = np.sin(2 * np.pi * self.df['Month'] / 12)
            self.df['Month_Cos'] = np.cos(2 * np.pi * self.df['Month'] / 12)
        
        time_features = [
            'Year', 'Month', 'Day', 'Weekday', 'DayOfYear', 'Quarter',
            'IsWeekend', 'IsWeekday', 'IsMonday', 'IsFriday'
        ]
        
        if config.FEATURE_CONFIG['time_features']['use_hour_encoding']:
            time_features.extend(['Hour_Sin', 'Hour_Cos'])
        
        if config.FEATURE_CONFIG['time_features']['use_day_encoding']:
            time_features.extend(['DayOfYear_Sin', 'DayOfYear_Cos'])
        
        if config.FEATURE_CONFIG['time_features']['use_month_encoding']:
            time_features.extend(['Month_Sin', 'Month_Cos'])
        
        self.logger.info(f"创建了 {len(time_features)} 个时间特征")
        return time_features
    
    def create_advanced_time_features(self):
        """创建高级时间特征（基于数据洞察）"""
        print_section_header("高级时间特征创建", level=2)
        
        # 基于双峰模式的时间段特征
        self.df['Hour_Deep_Night'] = (self.df['Hour'].isin([0, 1, 2, 3, 4, 5])).astype(int)
        self.df['Hour_Early_Morning'] = (self.df['Hour'].isin([6, 7])).astype(int)
        self.df['Hour_Morning_Peak'] = (self.df['Hour'].isin([8, 9])).astype(int)
        self.df['Hour_Morning_Decline'] = (self.df['Hour'].isin([10, 11, 12])).astype(int)
        self.df['Hour_Afternoon'] = (self.df['Hour'].isin([13, 14, 15, 16])).astype(int)
        self.df['Hour_Evening_Peak'] = (self.df['Hour'].isin([17, 18, 19])).astype(int)
        self.df['Hour_Evening_Decline'] = (self.df['Hour'].isin([20, 21, 22, 23])).astype(int)
        
        # 峰值时间标识
        self.df['Is_Peak_Hour'] = (self.df['Hour'].isin([8, 17, 18, 19])).astype(int)
        self.df['Is_Low_Hour'] = (self.df['Hour'].isin([3, 4, 5])).astype(int)
        
        # 通勤时间标识
        self.df['Is_Rush_Hour'] = ((self.df['Hour'].between(7, 9)) | 
                                 (self.df['Hour'].between(17, 19))).astype(int)
        
        advanced_time_features = [
            'Hour_Deep_Night', 'Hour_Early_Morning', 'Hour_Morning_Peak',
            'Hour_Morning_Decline', 'Hour_Afternoon', 'Hour_Evening_Peak',
            'Hour_Evening_Decline', 'Is_Peak_Hour', 'Is_Low_Hour', 'Is_Rush_Hour'
        ]
        
        self.logger.info(f"创建了 {len(advanced_time_features)} 个高级时间特征")
        return advanced_time_features
    
    def create_weather_features(self):
        """创建天气特征"""
        print_section_header("天气特征工程", level=2)
        
        weather_features = []
        
        # 温度分段特征
        if 'Temperature(°C)' in self.df.columns:
            temp_col = 'Temperature(°C)'
            temp_ranges = config.FEATURE_CONFIG['weather_thresholds']['temp_ranges']
            
            self.df['Temp_Severe_Cold'] = (self.df[temp_col] < temp_ranges[1]).astype(int)
            self.df['Temp_Cold'] = ((self.df[temp_col] >= temp_ranges[1]) & 
                                  (self.df[temp_col] < temp_ranges[2])).astype(int)
            self.df['Temp_Cool'] = ((self.df[temp_col] >= temp_ranges[2]) & 
                                  (self.df[temp_col] < temp_ranges[3])).astype(int)
            self.df['Temp_Warm'] = ((self.df[temp_col] >= temp_ranges[3]) & 
                                  (self.df[temp_col] < temp_ranges[4])).astype(int)
            self.df['Temp_Hot'] = (self.df[temp_col] >= temp_ranges[4]).astype(int)
            
            weather_features.extend(['Temp_Severe_Cold', 'Temp_Cold', 'Temp_Cool', 'Temp_Warm', 'Temp_Hot'])
        
        # 湿度分段特征
        if 'Humidity(%)' in self.df.columns:
            humidity_col = 'Humidity(%)'
            humidity_ranges = config.FEATURE_CONFIG['weather_thresholds']['humidity_ranges']
            
            self.df['Humidity_Low'] = (self.df[humidity_col] < humidity_ranges[1]).astype(int)
            self.df['Humidity_Medium'] = ((self.df[humidity_col] >= humidity_ranges[1]) & 
                                        (self.df[humidity_col] < humidity_ranges[2])).astype(int)
            self.df['Humidity_High'] = ((self.df[humidity_col] >= humidity_ranges[2]) & 
                                      (self.df[humidity_col] < humidity_ranges[3])).astype(int)
            self.df['Humidity_Very_High'] = (self.df[humidity_col] >= humidity_ranges[3]).astype(int)
            
            weather_features.extend(['Humidity_Low', 'Humidity_Medium', 'Humidity_High', 'Humidity_Very_High'])
        
        # 风速分段特征
        if 'Wind speed (m/s)' in self.df.columns:
            wind_col = 'Wind speed (m/s)'
            wind_ranges = config.FEATURE_CONFIG['weather_thresholds']['wind_ranges']
            
            self.df['Wind_Calm'] = (self.df[wind_col] < wind_ranges[1]).astype(int)
            self.df['Wind_Light'] = ((self.df[wind_col] >= wind_ranges[1]) & 
                                   (self.df[wind_col] < wind_ranges[2])).astype(int)
            self.df['Wind_Moderate'] = ((self.df[wind_col] >= wind_ranges[2]) & 
                                      (self.df[wind_col] < wind_ranges[3])).astype(int)
            self.df['Wind_Strong'] = (self.df[wind_col] >= wind_ranges[3]).astype(int)
            
            weather_features.extend(['Wind_Calm', 'Wind_Light', 'Wind_Moderate', 'Wind_Strong'])
        
        # 降水特征
        if 'Rainfall(mm)' in self.df.columns:
            self.df['Has_Rain'] = (self.df['Rainfall(mm)'] > 0).astype(int)
            self.df['Light_Rain'] = ((self.df['Rainfall(mm)'] > 0) & 
                                   (self.df['Rainfall(mm)'] <= 2.5)).astype(int)
            self.df['Moderate_Rain'] = ((self.df['Rainfall(mm)'] > 2.5) & 
                                      (self.df['Rainfall(mm)'] <= 10)).astype(int)
            self.df['Heavy_Rain'] = (self.df['Rainfall(mm)'] > 10).astype(int)
            
            weather_features.extend(['Has_Rain', 'Light_Rain', 'Moderate_Rain', 'Heavy_Rain'])
        
        if 'Snowfall (cm)' in self.df.columns:
            self.df['Has_Snow'] = (self.df['Snowfall (cm)'] > 0).astype(int)
            self.df['Light_Snow'] = ((self.df['Snowfall (cm)'] > 0) & 
                                   (self.df['Snowfall (cm)'] <= 2)).astype(int)
            self.df['Heavy_Snow'] = (self.df['Snowfall (cm)'] > 2).astype(int)
            
            weather_features.extend(['Has_Snow', 'Light_Snow', 'Heavy_Snow'])
        
        # 降水总量
        if 'Rainfall(mm)' in self.df.columns and 'Snowfall (cm)' in self.df.columns:
            self.df['Total_Precipitation'] = self.df['Rainfall(mm)'] + self.df['Snowfall (cm)']
            self.df['Has_Precipitation'] = ((self.df['Rainfall(mm)'] > 0) | 
                                          (self.df['Snowfall (cm)'] > 0)).astype(int)
            weather_features.extend(['Total_Precipitation', 'Has_Precipitation'])
        
        self.logger.info(f"创建了 {len(weather_features)} 个天气特征")
        return weather_features
    
    def create_comfort_index_features(self):
        """创建舒适度指数特征"""
        print_section_header("舒适度指数特征", level=2)
        
        comfort_features = []
        
        if 'Temperature(°C)' in self.df.columns and 'Humidity(%)' in self.df.columns:
            temp_col = 'Temperature(°C)'
            humidity_col = 'Humidity(%)'
            
            # 舒适度指数（基于温度和湿度）
            self.df['Comfort_Index'] = np.where(
                (self.df[temp_col].between(20, 30)) & (self.df[humidity_col].between(30, 70)),
                1.0,  # 最舒适
                np.where(
                    (self.df[temp_col].between(10, 35)) & (self.df[humidity_col].between(20, 80)),
                    0.7,  # 较舒适
                    np.where(
                        (self.df[temp_col].between(0, 40)) & (self.df[humidity_col].between(10, 90)),
                        0.4,  # 一般
                        0.1   # 不舒适
                    )
                )
            )
            
            # 体感温度（Heat Index简化版）
            self.df['Heat_Index'] = (self.df[temp_col] + 
                                   0.5 * (self.df[humidity_col] - 50) / 100 * self.df[temp_col])
            
            # 完美天气标识
            perfect_weather_conditions = [
                self.df[temp_col].between(20, 28),
                self.df[humidity_col].between(40, 60),
                self.df.get('Rainfall(mm)', pd.Series([0]*len(self.df))) == 0,
                self.df.get('Snowfall (cm)', pd.Series([0]*len(self.df))) == 0
            ]
            
            if 'Wind speed (m/s)' in self.df.columns:
                perfect_weather_conditions.append(self.df['Wind speed (m/s)'] < 3)
            
            self.df['Perfect_Weather'] = pd.concat(perfect_weather_conditions, axis=1).all(axis=1).astype(int)
            
            # 极端天气标识
            extreme_conditions = [
                self.df[temp_col] < -10,
                self.df[temp_col] > 35,
                self.df[humidity_col] > 90,
                self.df[humidity_col] < 20
            ]
            
            if 'Rainfall(mm)' in self.df.columns:
                extreme_conditions.append(self.df['Rainfall(mm)'] > 10)
            if 'Snowfall (cm)' in self.df.columns:
                extreme_conditions.append(self.df['Snowfall (cm)'] > 5)
            
            self.df['Extreme_Weather'] = pd.concat(extreme_conditions, axis=1).any(axis=1).astype(int)
            
            comfort_features = ['Comfort_Index', 'Heat_Index', 'Perfect_Weather', 'Extreme_Weather']
        
        self.logger.info(f"创建了 {len(comfort_features)} 个舒适度特征")
        return comfort_features
    
    def create_interaction_features(self):
        """创建交互特征"""
        print_section_header("交互特征创建", level=2)
        
        if not config.FEATURE_CONFIG['interaction_features']:
            self.logger.info("跳过交互特征创建")
            return []
        
        interaction_features = []
        
        # 温度×时间交互
        if 'Temperature(°C)' in self.df.columns:
            self.df['Temp_Hour'] = self.df['Temperature(°C)'] * self.df['Hour']
            self.df['Temp_Weekend'] = self.df['Temperature(°C)'] * self.df['IsWeekend']
            self.df['Temp_Peak'] = self.df['Temperature(°C)'] * self.df['Is_Peak_Hour']
            interaction_features.extend(['Temp_Hour', 'Temp_Weekend', 'Temp_Peak'])
        
        # 季节×时间交互
        if 'Seasons' in self.df.columns:
            for season in self.df['Seasons'].unique():
                season_col = f'Season_{season}'
                self.df[season_col] = (self.df['Seasons'] == season).astype(int)
                
                peak_interaction_col = f'{season}_Peak'
                weekend_interaction_col = f'{season}_Weekend'
                
                self.df[peak_interaction_col] = self.df[season_col] * self.df['Is_Peak_Hour']
                self.df[weekend_interaction_col] = self.df[season_col] * self.df['IsWeekend']
                
                interaction_features.extend([season_col, peak_interaction_col, weekend_interaction_col])
        
        # 舒适度×时间交互
        if 'Comfort_Index' in self.df.columns:
            self.df['Comfort_Peak'] = self.df['Comfort_Index'] * self.df['Is_Peak_Hour']
            self.df['Comfort_Weekend'] = self.df['Comfort_Index'] * self.df['IsWeekend']
            interaction_features.extend(['Comfort_Peak', 'Comfort_Weekend'])
        
        # 天气组合特征
        if 'Temperature(°C)' in self.df.columns and 'Humidity(%)' in self.df.columns:
            self.df['Temp_Humidity'] = self.df['Temperature(°C)'] * self.df['Humidity(%)'] / 100
            interaction_features.append('Temp_Humidity')
        
        self.logger.info(f"创建了 {len(interaction_features)} 个交互特征")
        return interaction_features
    
    def create_lag_features(self, target_col=None):
        """创建滞后特征（谨慎使用，避免数据泄露）"""
        print_section_header("滞后特征创建", level=2)
        
        if not config.FEATURE_CONFIG['lag_features']:
            self.logger.info("跳过滞后特征创建（避免数据泄露）")
            return []
        
        target_col = target_col or config.DATA_CONFIG['target_column']
        
        # 按时间排序
        self.df = self.df.sort_values([config.DATA_CONFIG['date_column'], 'Hour']).reset_index(drop=True)
        
        lag_features = []
        
        # 1小时前的需求
        self.df['Demand_Lag_1h'] = self.df[target_col].shift(1)
        
        # 24小时前的需求（昨天同时间）
        self.df['Demand_Lag_24h'] = self.df[target_col].shift(24)
        
        # 7天前的需求（上周同时间）
        self.df['Demand_Lag_168h'] = self.df[target_col].shift(168)
        
        # 移动平均特征
        self.df['Demand_MA_3h'] = self.df[target_col].rolling(window=3, min_periods=1).mean().shift(1)
        self.df['Demand_MA_24h'] = self.df[target_col].rolling(window=24, min_periods=1).mean().shift(1)
        
        lag_features = ['Demand_Lag_1h', 'Demand_Lag_24h', 'Demand_Lag_168h', 'Demand_MA_3h', 'Demand_MA_24h']
        
        # 填充缺失值（用均值）
        for feature in lag_features:
            self.df[feature].fillna(self.df[target_col].mean(), inplace=True)
        
        self.logger.warning(f"⚠️  创建了 {len(lag_features)} 个滞后特征，请注意数据泄露风险")
        return lag_features
    
    def encode_categorical_features(self):
        """编码分类特征"""
        print_section_header("分类特征编码", level=2)
        
        categorical_cols = config.FEATURE_CONFIG['categorical_features']
        encoded_features = []
        
        for col in categorical_cols:
            if col in self.df.columns:
                # 使用独热编码
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
                
                # 记录编码后的特征名
                encoded_features.extend(dummies.columns.tolist())
                
                # 删除原始列
                self.df.drop(col, axis=1, inplace=True)
        
        self.logger.info(f"编码了 {len(categorical_cols)} 个分类特征，生成 {len(encoded_features)} 个新特征")
        return encoded_features
    
    def handle_outliers(self, target_col=None):
        """处理异常值"""
        print_section_header("异常值处理", level=2)
        
        target_col = target_col or config.DATA_CONFIG['target_column']
        method = config.FEATURE_CONFIG['outlier_method']
        factor = config.FEATURE_CONFIG['outlier_factor']
        
        if method == 'iqr':
            Q1 = self.df[target_col].quantile(0.25)
            Q3 = self.df[target_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            outliers_mask = (self.df[target_col] < lower_bound) | (self.df[target_col] > upper_bound)
            outliers_count = outliers_mask.sum()
            
            self.logger.info(f"检测到 {outliers_count} 个异常值 ({outliers_count/len(self.df)*100:.2f}%)")
            
            # 使用Winsorization而不是删除
            self.df.loc[self.df[target_col] < lower_bound, target_col] = lower_bound
            self.df.loc[self.df[target_col] > upper_bound, target_col] = upper_bound
            
            self.logger.info("使用Winsorization处理异常值")
            
        return outliers_count
    
    def handle_non_functioning_days(self):
        """处理非运营日数据"""
        print_section_header("非运营日数据处理", level=2)
        
        if 'Functioning Day' not in self.df.columns:
            self.logger.info("未发现运营状态列，跳过处理")
            return len(self.df)
        
        # 统计非运营日
        non_functioning_mask = self.df['Functioning Day'] == 'No'
        non_functioning_count = non_functioning_mask.sum()
        
        if non_functioning_count > 0:
            self.logger.info(f"发现 {non_functioning_count} 条非运营日记录 ({non_functioning_count/len(self.df)*100:.2f}%)")
            
            # 移除非运营日数据
            self.df = self.df[~non_functioning_mask].reset_index(drop=True)
            self.logger.info(f"移除后剩余 {len(self.df)} 条记录")
        else:
            self.logger.info("未发现非运营日记录")
        
        return len(self.df)
    
    def select_features(self, target_col=None, method='correlation', k=None):
        """特征选择"""
        print_section_header("特征选择", level=2)
        
        target_col = target_col or config.DATA_CONFIG['target_column']
        
        # 获取所有数值特征（排除目标变量和日期）
        exclude_cols = [target_col, config.DATA_CONFIG['date_column']]
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        feature_candidates = [col for col in numeric_features if col not in exclude_cols]
        
        self.logger.info(f"候选特征数量: {len(feature_candidates)}")
        
        if method == 'correlation':
            # 基于相关性选择
            correlations = self.df[feature_candidates + [target_col]].corr()[target_col].abs()
            correlations = correlations.drop(target_col).sort_values(ascending=False)
            
            # 选择前k个或相关性>阈值的特征
            if k:
                selected_features = correlations.head(k).index.tolist()
            else:
                threshold = 0.1  # 相关性阈值
                selected_features = correlations[correlations > threshold].index.tolist()
            
            self.logger.info(f"基于相关性选择了 {len(selected_features)} 个特征")
            
        elif method == 'univariate':
            # 单变量统计检验
            X = self.df[feature_candidates].fillna(0)
            y = self.df[target_col]
            
            k = k or min(20, len(feature_candidates))  # 默认选择20个特征
            selector = SelectKBest(score_func=f_regression, k=k)
            selector.fit(X, y)
            
            selected_features = [feature_candidates[i] for i in selector.get_support(indices=True)]
            self.logger.info(f"基于单变量检验选择了 {len(selected_features)} 个特征")
            
        elif method == 'rfe':
            # 递归特征消除
            X = self.df[feature_candidates].fillna(0)
            y = self.df[target_col]
            
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            k = k or min(15, len(feature_candidates))  # 默认选择15个特征
            
            rfe = RFE(estimator=estimator, n_features_to_select=k, step=1)
            rfe.fit(X, y)
            
            selected_features = [feature_candidates[i] for i in rfe.get_support(indices=True)]
            self.logger.info(f"基于RFE选择了 {len(selected_features)} 个特征")
            
        else:
            # 保留所有特征
            selected_features = feature_candidates
            self.logger.info(f"保留所有 {len(selected_features)} 个特征")
        
        self.feature_names = selected_features
        return selected_features
    
    def scale_features(self, X_train, X_val, X_test):
        """特征标准化"""
        print_section_header("特征标准化", level=2)
        
        scaling_method = config.FEATURE_CONFIG['scaling_method']
        
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.logger.info("跳过特征标准化")
            return X_train, X_val, X_test
        
        # 只对数值特征进行标准化
        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()
        
        # 拟合和转换
        X_train_scaled[numeric_features] = self.scaler.fit_transform(X_train[numeric_features])
        X_val_scaled[numeric_features] = self.scaler.transform(X_val[numeric_features])
        X_test_scaled[numeric_features] = self.scaler.transform(X_test[numeric_features])
        
        self.logger.info(f"使用 {scaling_method} 方法标准化了 {len(numeric_features)} 个数值特征")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def split_data_temporal(self, target_col=None):
        """时间序列友好的数据分割"""
        print_section_header("时间序列数据分割", level=2)
        
        target_col = target_col or config.DATA_CONFIG['target_column']
        
        # 按时间排序
        date_col = config.DATA_CONFIG['date_column']
        self.df = self.df.sort_values([date_col, 'Hour']).reset_index(drop=True)
        
        # 分割比例
        train_ratio = config.DATA_CONFIG['train_ratio']
        val_ratio = config.DATA_CONFIG['val_ratio']
        test_ratio = config.DATA_CONFIG['test_ratio']
        
        n = len(self.df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # 分割数据
        train_df = self.df.iloc[:train_end].copy()
        val_df = self.df.iloc[train_end:val_end].copy()
        test_df = self.df.iloc[val_end:].copy()
        
        self.logger.info(f"数据分割完成:")
        self.logger.info(f"  训练集: {len(train_df)} 条 ({len(train_df)/n*100:.1f}%)")
        self.logger.info(f"  验证集: {len(val_df)} 条 ({len(val_df)/n*100:.1f}%)")
        self.logger.info(f"  测试集: {len(test_df)} 条 ({len(test_df)/n*100:.1f}%)")
        
        # 提取特征和目标
        X_train = train_df[self.feature_names]
        y_train = train_df[target_col]
        
        X_val = val_df[self.feature_names]
        y_val = val_df[target_col]
        
        X_test = test_df[self.feature_names]
        y_test = test_df[target_col]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_cross_validation_splitter(self):
        """获取时间序列交叉验证分割器"""
        cv_config = config.MODEL_CONFIG['cross_validation']
        
        if cv_config['method'] == 'TimeSeriesSplit':
            return TimeSeriesSplit(n_splits=cv_config['n_splits'])
        else:
            return TimeSeriesSplit(n_splits=5)  # 默认
    
    def preprocess_pipeline(self, use_lag_features=False, feature_selection_method='correlation', 
                          feature_selection_k=None):
        """完整的预处理流水线"""
        print_section_header("完整预处理流水线", level=1)
        
        timestamp = get_timestamp()
        
        try:
            # 1. 数据加载与验证
            self.load_and_validate_data()
            
            # 2. 处理非运营日
            remaining_count = self.handle_non_functioning_days()
            
            # 3. 日期时间特征
            time_features = self.handle_datetime_features()
            
            # 4. 高级时间特征
            advanced_time_features = self.create_advanced_time_features()
            
            # 5. 天气特征
            weather_features = self.create_weather_features()
            
            # 6. 舒适度特征
            comfort_features = self.create_comfort_index_features()
            
            # 7. 交互特征
            interaction_features = self.create_interaction_features()
            
            # 8. 滞后特征（可选）
            lag_features = []
            if use_lag_features:
                lag_features = self.create_lag_features()
            
            # 9. 分类特征编码
            encoded_features = self.encode_categorical_features()
            
            # 10. 异常值处理
            outliers_count = self.handle_outliers()
            
            # 11. 特征选择
            selected_features = self.select_features(method=feature_selection_method, k=feature_selection_k)
            
            # 12. 数据分割
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data_temporal()
            
            # 13. 特征标准化
            X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(X_train, X_val, X_test)
            
            # 保存预处理结果
            self.preprocessing_results = {
                'timestamp': timestamp,
                'original_shape': self.df.shape,
                'final_shape': (len(X_train_scaled) + len(X_val_scaled) + len(X_test_scaled), 
                               len(selected_features)),
                'feature_counts': {
                    'time_features': len(time_features),
                    'advanced_time_features': len(advanced_time_features),
                    'weather_features': len(weather_features),
                    'comfort_features': len(comfort_features),
                    'interaction_features': len(interaction_features),
                    'lag_features': len(lag_features),
                    'encoded_features': len(encoded_features),
                    'selected_features': len(selected_features)
                },
                'outliers_handled': outliers_count,
                'non_functioning_removed': self.df.shape[0] != remaining_count,
                'feature_names': selected_features,
                'preprocessing_config': {
                    'use_lag_features': use_lag_features,
                    'feature_selection_method': feature_selection_method,
                    'feature_selection_k': feature_selection_k
                }
            }
            
            # 保存预处理后的数据
            self.save_preprocessed_data(X_train_scaled, X_val_scaled, X_test_scaled,
                                      y_train, y_val, y_test, timestamp)
            
            # 生成预处理报告
            self.generate_preprocessing_report()
            
            return (X_train_scaled, X_val_scaled, X_test_scaled, 
                   y_train, y_val, y_test, self.preprocessing_results)
            
        except Exception as e:
            self.logger.error(f"预处理过程中出错: {str(e)}")
            raise
    
    def save_preprocessed_data(self, X_train, X_val, X_test, y_train, y_val, y_test, timestamp):
        """保存预处理后的数据"""
        print_section_header("保存预处理数据", level=2)
        
        # 保存特征数据
        datasets = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        for name, data in datasets.items():
            if hasattr(data, 'values'):
                np.save(config.OUTPUT_DIR / f"{name}_{timestamp}.npy", data.values)
            else:
                np.save(config.OUTPUT_DIR / f"{name}_{timestamp}.npy", data)
        
        # 保存特征名称
        self.result_saver.save_json(self.feature_names, f"feature_names_{timestamp}")
        
        # 保存预处理器状态
        import pickle
        preprocessor_state = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'preprocessing_results': self.preprocessing_results
        }
        
        with open(config.OUTPUT_DIR / f"preprocessor_{timestamp}.pkl", 'wb') as f:
            pickle.dump(preprocessor_state, f)
        
        self.logger.info(f"✅ 预处理数据已保存，时间戳: {timestamp}")
    
    def generate_preprocessing_report(self):
        """生成预处理报告"""
        print_section_header("预处理总结报告", level=1)
        
        results = self.preprocessing_results
        
        print("🎯 预处理完成摘要:")
        print(f"1. 原始数据: {results['original_shape'][0]} 行 × {results['original_shape'][1]} 列")
        print(f"2. 最终数据: {results['final_shape'][0]} 行 × {results['final_shape'][1]} 列")
        
        print(f"\n📊 特征工程统计:")
        feature_counts = results['feature_counts']
        for feature_type, count in feature_counts.items():
            print(f"  {feature_type}: {count} 个")
        
        print(f"\n🔧 数据处理统计:")
        print(f"  异常值处理: {results['outliers_handled']} 个")
        print(f"  非运营日移除: {'是' if results['non_functioning_removed'] else '否'}")
        
        print(f"\n⚙️  预处理配置:")
        config_info = results['preprocessing_config']
        for key, value in config_info.items():
            print(f"  {key}: {value}")
        
        print(f"\n🏆 Top 10 选择特征:")
        for i, feature in enumerate(results['feature_names'][:10], 1):
            print(f"  {i:2d}. {feature}")
        
        # 保存完整结果
        self.result_saver.save_json(results, f"preprocessing_results_{results['timestamp']}", "preprocessing")
        
        print(f"\n✅ 预处理完成！")
        print(f"📁 结果保存位置: {config.OUTPUT_DIR}")

def main():
    """主函数"""
    print_section_header("首尔自行车共享数据 - 增强版预处理", level=1)
    
    # 创建预处理器
    preprocessor = EnhancedDataPreprocessor()
    
    try:
        # 运行完整预处理流水线
        (X_train, X_val, X_test, y_train, y_val, y_test, results) = preprocessor.preprocess_pipeline(
            use_lag_features=False,  # 默认不使用滞后特征
            feature_selection_method='correlation',  # 基于相关性选择特征
            feature_selection_k=None  # 自动确定特征数量
        )
        
        print(f"\n✅ 预处理管道执行成功！")
        print(f"📊 最终数据形状:")
        print(f"  训练集: {X_train.shape}")
        print(f"  验证集: {X_val.shape}")
        print(f"  测试集: {X_test.shape}")
        
        return results
        
    except Exception as e:
        print(f"❌ 预处理失败: {str(e)}")
        raise

if __name__ == "__main__":
    results = main() 