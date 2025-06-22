#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版数据探索性分析
CDS503 Group Project - 首尔自行车需求预测
整合统一流程和工具
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

from config import config
from utils import (Logger, DataValidator, DataLoader, ResultSaver, 
                  VisualizationHelper, print_section_header, get_timestamp)

warnings.filterwarnings('ignore')

class EnhancedDataExplorer:
    """增强版数据探索类"""
    
    def __init__(self):
        self.logger = Logger(self.__class__.__name__)
        self.data_loader = DataLoader(self.logger)
        self.validator = DataValidator()
        self.result_saver = ResultSaver(self.logger)
        self.viz_helper = VisualizationHelper(self.logger)
        
        self.df = None
        self.exploration_results = {}
        
    def load_and_validate_data(self, file_path=None):
        """加载并验证数据"""
        print_section_header("数据加载与验证", level=1)
        
        # 加载数据
        self.df = self.data_loader.load_data(file_path)
        
        # 基础验证
        required_cols = [config.DATA_CONFIG['target_column'], 
                        config.DATA_CONFIG['date_column']]
        self.validator.validate_dataframe(self.df, required_cols)
        
        # 数据基本信息
        print_section_header("数据基本信息", level=2)
        print(f"数据集形状: {self.df.shape}")
        print(f"特征数量: {self.df.shape[1]}")
        print(f"样本数量: {self.df.shape[0]}")
        print(f"内存使用量: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # 列信息
        print(f"\n列信息:")
        for i, col in enumerate(self.df.columns, 1):
            print(f"{i:2d}. {col}")
        
        # 数据类型检查
        type_info, type_mismatches = self.validator.check_data_types(self.df)
        print(f"\n数据类型信息:")
        print(type_info.to_string(index=False))
        
        if type_mismatches:
            self.logger.warning(f"数据类型不匹配: {type_mismatches}")
        
        # 缺失值检查
        missing_info = self.validator.check_missing_values(self.df)
        if missing_info is not None:
            print(f"\n⚠️  发现缺失值:")
            print(missing_info)
        else:
            print(f"\n✅ 无缺失值，数据完整度100%")
        
        # 保存基础信息
        self.exploration_results['basic_info'] = {
            'shape': self.df.shape,
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'has_missing_values': missing_info is not None,
            'data_types': self.df.dtypes.to_dict()
        }
        
        return self.df
    
    def analyze_target_variable(self):
        """分析目标变量"""
        print_section_header("目标变量深度分析", level=1)
        
        target_col = config.DATA_CONFIG['target_column']
        target_values = self.df[target_col]
        
        # 基础统计
        print_section_header("基础统计描述", level=2)
        stats = {
            '数据点数量': len(target_values),
            '平均值': target_values.mean(),
            '中位数': target_values.median(),
            '标准差': target_values.std(),
            '方差': target_values.var(),
            '最小值': target_values.min(),
            '最大值': target_values.max(),
            '偏度': target_values.skew(),
            '峰度': target_values.kurtosis(),
            '零值数量': (target_values == 0).sum(),
            '零值比例(%)': (target_values == 0).mean() * 100
        }
        
        for key, value in stats.items():
            print(f"{key}: {value:.2f}")
        
        # 分位数分析
        print_section_header("分位数分析", level=3)
        quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        for q in quantiles:
            print(f"Q{q*100:5.1f}: {target_values.quantile(q):8.2f}")
        
        # 异常值分析
        print_section_header("异常值分析 (IQR方法)", level=3)
        Q1 = target_values.quantile(0.25)
        Q3 = target_values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_mask = (target_values < lower_bound) | (target_values > upper_bound)
        outliers_count = outliers_mask.sum()
        
        print(f"下界: {lower_bound:.2f}")
        print(f"上界: {upper_bound:.2f}")
        print(f"异常值数量: {outliers_count} ({outliers_count/len(target_values)*100:.2f}%)")
        
        # 保存目标变量分析结果
        self.exploration_results['target_analysis'] = {
            'basic_stats': stats,
            'quantiles': {f'Q{q*100:.1f}': target_values.quantile(q) for q in quantiles},
            'outliers': {
                'count': outliers_count,
                'percentage': outliers_count/len(target_values)*100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        }
        
        # 可视化目标变量分布
        self.viz_helper.plot_target_distribution(target_values, 
                                                save_name=f"target_distribution_{get_timestamp()}")
        
        return target_values
    
    def analyze_numerical_features(self):
        """分析数值特征"""
        print_section_header("数值特征分析", level=1)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        target_col = config.DATA_CONFIG['target_column']
        
        print(f"数值特征数量: {len(numeric_cols)}")
        print(f"数值特征列表: {numeric_cols}")
        
        # 基础统计描述
        print_section_header("数值特征统计描述", level=2)
        desc_stats = self.df[numeric_cols].describe()
        print(desc_stats.round(2).to_string())
        
        # 相关性分析
        print_section_header("特征相关性分析", level=2)
        correlations = self.df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
        
        print("与目标变量的相关性 (绝对值排序):")
        for feature, corr in correlations.items():
            if feature != target_col:
                correlation_level = "强" if corr > 0.5 else "中" if corr > 0.3 else "弱"
                print(f"  {feature:30s}: {corr:.4f} ({correlation_level})")
        
        # 强相关特征识别
        strong_corr_features = correlations[correlations > 0.3]
        strong_corr_features = strong_corr_features[strong_corr_features.index != target_col]
        
        if len(strong_corr_features) > 0:
            print(f"\n🎯 强相关特征 (|r| > 0.3):")
            for feature, corr in strong_corr_features.items():
                print(f"  {feature}: {corr:.4f}")
        
        # 异常值检查
        print_section_header("数值特征异常值检查", level=2)
        outlier_summary = {}
        
        for col in numeric_cols:
            if col != 'Hour':  # 跳过小时列（0-23是正常范围）
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                outlier_count = len(outliers)
                outlier_percentage = outlier_count / len(self.df) * 100
                
                if outlier_count > 0:
                    print(f"  {col:30s}: {outlier_count:4d} 个异常值 ({outlier_percentage:.2f}%)")
                    outlier_summary[col] = {
                        'count': outlier_count,
                        'percentage': outlier_percentage,
                        'bounds': [lower_bound, upper_bound]
                    }
        
        if not outlier_summary:
            print("  ✅ 未发现显著异常值")
        
        # 保存数值特征分析结果
        self.exploration_results['numerical_analysis'] = {
            'feature_count': len(numeric_cols),
            'descriptive_stats': desc_stats.to_dict(),
            'correlations': correlations.to_dict(),
            'strong_correlations': strong_corr_features.to_dict(),
            'outliers': outlier_summary
        }
        
        # 绘制相关性矩阵
        self.viz_helper.plot_correlation_matrix(self.df, target_col, 
                                              save_name=f"correlation_matrix_{get_timestamp()}")
        
        return numeric_cols, correlations
    
    def analyze_categorical_features(self):
        """分析分类特征"""
        print_section_header("分类特征分析", level=1)
        
        categorical_cols = config.FEATURE_CONFIG['categorical_features']
        target_col = config.DATA_CONFIG['target_column']
        
        categorical_analysis = {}
        
        for col in categorical_cols:
            if col in self.df.columns:
                print_section_header(f"{col} 分布分析", level=2)
                
                # 值计数和比例
                value_counts = self.df[col].value_counts()
                percentages = (self.df[col].value_counts(normalize=True) * 100).round(2)
                
                print("类别分布:")
                for val, count, pct in zip(value_counts.index, value_counts.values, percentages.values):
                    print(f"  {val:15s}: {count:5d} ({pct:5.2f}%)")
                
                # 各类别的目标变量统计
                print(f"\n各类别的{target_col}统计:")
                category_stats = self.df.groupby(col)[target_col].agg(['count', 'mean', 'std', 'min', 'max'])
                print(category_stats.round(2).to_string())
                
                categorical_analysis[col] = {
                    'value_counts': value_counts.to_dict(),
                    'percentages': percentages.to_dict(),
                    'target_stats': category_stats.to_dict()
                }
        
        # 保存分类特征分析结果
        self.exploration_results['categorical_analysis'] = categorical_analysis
        
        return categorical_analysis
    
    def analyze_time_patterns(self):
        """分析时间模式"""
        print_section_header("时间模式分析", level=1)
        
        date_col = config.DATA_CONFIG['date_column']
        target_col = config.DATA_CONFIG['target_column']
        
        # 转换日期格式
        try:
            self.df[date_col] = pd.to_datetime(self.df[date_col], format='mixed', dayfirst=True)
        except:
            self.df[date_col] = pd.to_datetime(self.df[date_col])
        
        # 提取时间特征
        self.df['Year'] = self.df[date_col].dt.year
        self.df['Month'] = self.df[date_col].dt.month
        self.df['Day'] = self.df[date_col].dt.day
        self.df['Weekday'] = self.df[date_col].dt.weekday
        self.df['WeekdayName'] = self.df[date_col].dt.day_name()
        self.df['IsWeekend'] = (self.df['Weekday'] >= 5).astype(int)
        
        time_analysis = {}
        
        # 年度分析
        if self.df['Year'].nunique() > 1:
            print_section_header("年度分析", level=2)
            yearly_stats = self.df.groupby('Year')[target_col].agg(['count', 'mean', 'std', 'min', 'max'])
            print(yearly_stats.round(2).to_string())
            time_analysis['yearly'] = yearly_stats.to_dict()
        
        # 月度分析
        print_section_header("月度分析", level=2)
        monthly_stats = self.df.groupby('Month')[target_col].agg(['count', 'mean', 'std'])
        print(monthly_stats.round(2).to_string())
        time_analysis['monthly'] = monthly_stats.to_dict()
        
        # 小时分析
        print_section_header("小时分析", level=2)
        hourly_stats = self.df.groupby('Hour')[target_col].agg(['count', 'mean', 'std'])
        print(hourly_stats.round(2).to_string())
        time_analysis['hourly'] = hourly_stats.to_dict()
        
        # 工作日vs周末
        print_section_header("工作日 vs 周末分析", level=2)
        weekend_stats = self.df.groupby('IsWeekend')[target_col].agg(['count', 'mean', 'std'])
        weekend_stats.index = ['工作日', '周末']
        print(weekend_stats.round(2).to_string())
        time_analysis['weekend'] = weekend_stats.to_dict()
        
        # 一周中各天
        print_section_header("一周各天分析", level=2)
        weekday_stats = self.df.groupby('WeekdayName')[target_col].agg(['count', 'mean', 'std'])
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_stats = weekday_stats.reindex(weekday_order)
        print(weekday_stats.round(2).to_string())
        time_analysis['weekday'] = weekday_stats.to_dict()
        
        # 高峰时段识别
        print_section_header("高峰时段识别", level=2)
        hourly_avg = self.df.groupby('Hour')[target_col].mean()
        peak_threshold = hourly_avg.quantile(0.8)
        peak_hours = hourly_avg[hourly_avg >= peak_threshold].index.tolist()
        
        print(f"高峰阈值（80分位数）: {peak_threshold:.2f}")
        print(f"高峰时段: {peak_hours}")
        
        time_analysis['peaks'] = {
            'threshold': peak_threshold,
            'peak_hours': peak_hours
        }
        
        # 保存时间分析结果
        self.exploration_results['time_analysis'] = time_analysis
        
        # 绘制时间序列图
        self.viz_helper.plot_time_series(self.df, date_col, target_col,
                                        save_name=f"time_series_{get_timestamp()}")
        
        return time_analysis
    
    def analyze_weather_impact(self):
        """分析天气影响"""
        print_section_header("天气因素影响分析", level=1)
        
        target_col = config.DATA_CONFIG['target_column']
        weather_analysis = {}
        
        # 温度影响
        print_section_header("温度影响分析", level=2)
        temp_col = 'Temperature(°C)'
        if temp_col in self.df.columns:
            print(f"温度范围: {self.df[temp_col].min():.1f}°C 到 {self.df[temp_col].max():.1f}°C")
            
            # 温度分段分析
            temp_ranges = config.FEATURE_CONFIG['weather_thresholds']['temp_ranges']
            temp_labels = ['严寒(<0°C)', '寒冷(0-10°C)', '凉爽(10-20°C)', '温暖(20-30°C)', '炎热(>30°C)']
            
            self.df['TempRange'] = pd.cut(self.df[temp_col], bins=temp_ranges, labels=temp_labels, include_lowest=True)
            temp_impact = self.df.groupby('TempRange')[target_col].agg(['count', 'mean', 'std'])
            print(temp_impact.round(2).to_string())
            weather_analysis['temperature'] = temp_impact.to_dict()
        
        # 湿度影响
        print_section_header("湿度影响分析", level=2)
        humidity_col = 'Humidity(%)'
        if humidity_col in self.df.columns:
            humidity_ranges = config.FEATURE_CONFIG['weather_thresholds']['humidity_ranges']
            humidity_labels = ['低湿度(<30%)', '中等湿度(30-50%)', '高湿度(50-70%)', '极高湿度(>70%)']
            
            self.df['HumidityRange'] = pd.cut(self.df[humidity_col], bins=humidity_ranges, 
                                            labels=humidity_labels, include_lowest=True)
            humidity_impact = self.df.groupby('HumidityRange')[target_col].agg(['count', 'mean', 'std'])
            print(humidity_impact.round(2).to_string())
            weather_analysis['humidity'] = humidity_impact.to_dict()
        
        # 风速影响
        print_section_header("风速影响分析", level=2)
        wind_col = 'Wind speed (m/s)'
        if wind_col in self.df.columns:
            print(f"风速范围: {self.df[wind_col].min():.1f} 到 {self.df[wind_col].max():.1f} m/s")
            
            wind_ranges = config.FEATURE_CONFIG['weather_thresholds']['wind_ranges']
            wind_labels = ['微风(<2m/s)', '轻风(2-4m/s)', '和风(4-6m/s)', '强风(>6m/s)']
            
            self.df['WindRange'] = pd.cut(self.df[wind_col], bins=wind_ranges, 
                                        labels=wind_labels, include_lowest=True)
            wind_impact = self.df.groupby('WindRange')[target_col].agg(['count', 'mean', 'std'])
            print(wind_impact.round(2).to_string())
            weather_analysis['wind'] = wind_impact.to_dict()
        
        # 降雨影响
        print_section_header("降雨影响分析", level=2)
        rain_col = 'Rainfall(mm)'
        if rain_col in self.df.columns:
            rain_impact = self.df.groupby(self.df[rain_col] > 0)[target_col].agg(['count', 'mean', 'std'])
            rain_impact.index = ['无雨', '有雨']
            print(rain_impact.round(2).to_string())
            
            rain_days = (self.df[rain_col] > 0).sum()
            rain_percentage = rain_days / len(self.df) * 100
            print(f"降雨天数: {rain_days} ({rain_percentage:.1f}%)")
            weather_analysis['rainfall'] = {
                'impact': rain_impact.to_dict(),
                'rainy_days': rain_days,
                'rain_percentage': rain_percentage
            }
        
        # 降雪影响
        print_section_header("降雪影响分析", level=2)
        snow_col = 'Snowfall (cm)'
        if snow_col in self.df.columns:
            snow_impact = self.df.groupby(self.df[snow_col] > 0)[target_col].agg(['count', 'mean', 'std'])
            snow_impact.index = ['无雪', '有雪']
            print(snow_impact.round(2).to_string())
            
            snow_days = (self.df[snow_col] > 0).sum()
            snow_percentage = snow_days / len(self.df) * 100
            print(f"降雪天数: {snow_days} ({snow_percentage:.1f}%)")
            weather_analysis['snowfall'] = {
                'impact': snow_impact.to_dict(),
                'snowy_days': snow_days,
                'snow_percentage': snow_percentage
            }
        
        # 保存天气分析结果
        self.exploration_results['weather_analysis'] = weather_analysis
        
        return weather_analysis
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print_section_header("综合分析报告生成", level=1)
        
        # 保存所有分析结果
        self.result_saver.save_json(self.exploration_results, 
                                   f"comprehensive_eda_results_{get_timestamp()}", 
                                   "eda")
        
        # 生成总结报告
        print_section_header("数据探索总结", level=1)
        
        target_col = config.DATA_CONFIG['target_column']
        
        print("🎯 关键发现:")
        print(f"1. 数据质量: {self.df.shape[0]} 条记录，{self.df.shape[1]} 个特征")
        
        if 'target_analysis' in self.exploration_results:
            zero_pct = self.exploration_results['target_analysis']['basic_stats']['零值比例(%)']
            outlier_pct = self.exploration_results['target_analysis']['outliers']['percentage']
            print(f"2. 目标变量: 零值占比 {zero_pct:.1f}%，异常值占比 {outlier_pct:.1f}%")
        
        if 'numerical_analysis' in self.exploration_results:
            strong_corr = self.exploration_results['numerical_analysis']['strong_correlations']
            print(f"3. 强相关特征: {len(strong_corr)} 个特征与目标变量强相关")
        
        if 'time_analysis' in self.exploration_results:
            peak_hours = self.exploration_results['time_analysis']['peaks']['peak_hours']
            print(f"4. 时间模式: 高峰时段为 {peak_hours}")
        
        if 'weather_analysis' in self.exploration_results:
            print("5. 天气影响: 温度、湿度、降雨降雪对需求有显著影响")
        
        print(f"\n📝 建议的后续步骤:")
        print("1. 特征工程: 基于时间模式创建通勤时段特征")
        print("2. 天气特征: 创建温度舒适度和极端天气标识")
        print("3. 交互特征: 考虑天气×时间的交互效应")
        print("4. 异常值处理: 使用稳健的方法处理极端值")
        print("5. 数据分割: 采用时间序列友好的分割方法")
        
        # 保存基础数据框（添加了时间特征）
        processed_df_path = self.result_saver.save_dataframe(
            self.df, f"explored_data_{get_timestamp()}", "eda"
        )
        
        print(f"\n✅ 探索性分析完成！")
        print(f"📊 结果已保存到: {config.OUTPUT_DIR}")
        
        return self.exploration_results

def main():
    """主函数"""
    print_section_header("首尔自行车共享数据 - 增强版探索性分析", level=1)
    
    # 创建探索器实例
    explorer = EnhancedDataExplorer()
    
    try:
        # 1. 数据加载与验证
        df = explorer.load_and_validate_data()
        
        # 2. 目标变量分析
        target_values = explorer.analyze_target_variable()
        
        # 3. 数值特征分析
        numeric_cols, correlations = explorer.analyze_numerical_features()
        
        # 4. 分类特征分析
        categorical_analysis = explorer.analyze_categorical_features()
        
        # 5. 时间模式分析
        time_analysis = explorer.analyze_time_patterns()
        
        # 6. 天气影响分析
        weather_analysis = explorer.analyze_weather_impact()
        
        # 7. 生成综合报告
        results = explorer.generate_comprehensive_report()
        
        return results
        
    except Exception as e:
        explorer.logger.error(f"探索性分析过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    results = main() 