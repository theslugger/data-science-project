#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版数据深度分析
CDS503 Group Project - 首尔自行车需求预测
基于数据洞察的深度模式挖掘
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
import warnings

from config import config
from utils import (Logger, DataValidator, DataLoader, ResultSaver, 
                  VisualizationHelper, MetricsCalculator, print_section_header, get_timestamp)

warnings.filterwarnings('ignore')

class EnhancedDataAnalyzer:
    """增强版数据深度分析类"""
    
    def __init__(self):
        self.logger = Logger(self.__class__.__name__)
        self.data_loader = DataLoader(self.logger)
        self.validator = DataValidator()
        self.result_saver = ResultSaver(self.logger)
        self.viz_helper = VisualizationHelper(self.logger)
        self.metrics_calc = MetricsCalculator()
        
        self.df = None
        self.analysis_results = {}
        
    def load_and_prepare_data(self, file_path=None):
        """加载并准备数据"""
        print_section_header("数据加载与预处理", level=1)
        
        # 加载数据
        self.df = self.data_loader.load_data(file_path)
        
        # 转换日期格式（处理混合格式）
        date_col = config.DATA_CONFIG['date_column']
        try:
            self.df[date_col] = pd.to_datetime(self.df[date_col], format='mixed', dayfirst=True)
        except:
            self.df[date_col] = pd.to_datetime(self.df[date_col])
        
        # 添加基础时间特征
        self.df['Year'] = self.df[date_col].dt.year
        self.df['Month'] = self.df[date_col].dt.month
        self.df['Day'] = self.df[date_col].dt.day
        self.df['Weekday'] = self.df[date_col].dt.weekday
        self.df['IsWeekend'] = (self.df['Weekday'] >= 5).astype(int)
        
        print(f"✅ 数据准备完成: {self.df.shape}")
        return self.df
    
    def deep_target_analysis(self):
        """深度目标变量分析"""
        print_section_header("目标变量深度洞察分析", level=1)
        
        target_col = config.DATA_CONFIG['target_column']
        target_values = self.df[target_col]
        
        analysis_results = {}
        
        # 1. 高级统计分析
        print_section_header("高级统计特征", level=2)
        
        # 偏度和峰度分析
        skewness = target_values.skew()
        kurtosis = target_values.kurtosis()
        
        print(f"偏度 (Skewness): {skewness:.4f}")
        if skewness > 1:
            print("  - 强右偏分布")
        elif skewness > 0.5:
            print("  - 中等右偏分布")
        elif skewness < -1:
            print("  - 强左偏分布")
        elif skewness < -0.5:
            print("  - 中等左偏分布")
        else:
            print("  - 近似对称分布")
        
        print(f"峰度 (Kurtosis): {kurtosis:.4f}")
        if kurtosis > 3:
            print("  - 尖峰分布（厚尾）")
        elif kurtosis < 3:
            print("  - 平峰分布（薄尾）")
        else:
            print("  - 正态峰度")
        
        # 正态性检验
        print_section_header("正态性检验", level=3)
        try:
            # Shapiro-Wilk检验（适用于小样本）
            if len(target_values) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(target_values.sample(5000) if len(target_values) > 5000 else target_values)
                print(f"Shapiro-Wilk 检验: 统计量={shapiro_stat:.4f}, p值={shapiro_p:.6f}")
            
            # Kolmogorov-Smirnov检验
            ks_stat, ks_p = stats.kstest(target_values, 'norm')
            print(f"Kolmogorov-Smirnov 检验: 统计量={ks_stat:.4f}, p值={ks_p:.6f}")
            
            if ks_p < 0.05:
                print("  - 拒绝正态分布假设")
            else:
                print("  - 无法拒绝正态分布假设")
                
        except Exception as e:
            self.logger.warning(f"正态性检验失败: {str(e)}")
        
        # 2. 零值模式深度分析
        print_section_header("零值模式深度分析", level=2)
        
        zero_mask = target_values == 0
        zero_count = zero_mask.sum()
        zero_percentage = zero_count / len(target_values) * 100
        
        print(f"零值总数: {zero_count} ({zero_percentage:.2f}%)")
        
        if zero_count > 0:
            zero_data = self.df[zero_mask].copy()
            
            # 零值的时间分布
            print("零值时间分布模式:")
            zero_hour_dist = zero_data['Hour'].value_counts().sort_index()
            for hour, count in zero_hour_dist.items():
                pct = count / zero_count * 100
                print(f"  {hour:2d}时: {count:3d}次 ({pct:5.1f}%)")
            
            # 零值的季节分布
            if 'Seasons' in zero_data.columns:
                print(f"\n零值季节分布:")
                zero_season_dist = zero_data['Seasons'].value_counts()
                for season, count in zero_season_dist.items():
                    pct = count / zero_count * 100
                    print(f"  {season}: {count}次 ({pct:.1f}%)")
            
            # 零值的运营状态
            if 'Functioning Day' in zero_data.columns:
                print(f"\n零值运营状态:")
                zero_func_dist = zero_data['Functioning Day'].value_counts()
                for status, count in zero_func_dist.items():
                    pct = count / zero_count * 100
                    print(f"  {status}: {count}次 ({pct:.1f}%)")
            
            # 零值天气条件分析
            print(f"\n零值天气条件:")
            weather_cols = ['Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 'Rainfall(mm)', 'Snowfall (cm)']
            for col in weather_cols:
                if col in zero_data.columns:
                    mean_val = zero_data[col].mean()
                    overall_mean = self.df[col].mean()
                    print(f"  {col}: {mean_val:.2f} (全体平均: {overall_mean:.2f})")
        
        # 3. 分布形状分析
        print_section_header("分布形状特征分析", level=2)
        
        # 多峰检测
        hist, bin_edges = np.histogram(target_values, bins=50)
        peaks = []
        for i in range(1, len(hist)-1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peak_value = (bin_edges[i] + bin_edges[i+1]) / 2
                peaks.append(peak_value)
        
        print(f"检测到 {len(peaks)} 个峰值:")
        for i, peak in enumerate(peaks):
            print(f"  峰{i+1}: {peak:.1f}")
        
        if len(peaks) > 1:
            print("  - 多峰分布，可能存在不同的使用模式")
        else:
            print("  - 单峰分布")
        
        # 4. 极值分析
        print_section_header("极值分析", level=2)
        
        # 最高需求分析
        max_demand = target_values.max()
        max_indices = target_values[target_values == max_demand].index
        
        print(f"最高需求: {max_demand} 辆")
        print(f"最高需求出现次数: {len(max_indices)}")
        
        if len(max_indices) > 0:
            max_conditions = self.df.loc[max_indices[0]]
            print(f"最高需求条件:")
            print(f"  日期: {max_conditions[config.DATA_CONFIG['date_column']]}")
            print(f"  小时: {max_conditions['Hour']}")
            if 'Seasons' in max_conditions:
                print(f"  季节: {max_conditions['Seasons']}")
            if 'Temperature(°C)' in max_conditions:
                print(f"  温度: {max_conditions['Temperature(°C)']}°C")
        
        # 保存深度分析结果
        analysis_results['advanced_stats'] = {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'zero_analysis': {
                'count': zero_count,
                'percentage': zero_percentage
            },
            'peaks': peaks,
            'max_demand': {
                'value': max_demand,
                'occurrences': len(max_indices)
            }
        }
        
        self.analysis_results['deep_target_analysis'] = analysis_results
        return analysis_results
    
    def advanced_time_pattern_analysis(self):
        """高级时间模式分析"""
        print_section_header("高级时间模式洞察", level=1)
        
        target_col = config.DATA_CONFIG['target_column']
        analysis_results = {}
        
        # 1. 双峰模式详细分析
        print_section_header("双峰模式详细分析", level=2)
        
        hourly_avg = self.df.groupby('Hour')[target_col].mean()
        hourly_std = self.df.groupby('Hour')[target_col].std()
        
        # 识别峰值
        peaks = []
        valleys = []
        
        for hour in range(1, 23):
            prev_val = hourly_avg.iloc[hour-1]
            curr_val = hourly_avg.iloc[hour]
            next_val = hourly_avg.iloc[hour+1]
            
            # 峰值检测
            if curr_val > prev_val and curr_val > next_val:
                peaks.append((hour, curr_val))
            
            # 谷值检测
            if curr_val < prev_val and curr_val < next_val:
                valleys.append((hour, curr_val))
        
        print(f"发现 {len(peaks)} 个峰值:")
        for hour, value in peaks:
            print(f"  {hour:2d}时: {value:.1f}辆")
        
        print(f"发现 {len(valleys)} 个谷值:")
        for hour, value in valleys:
            print(f"  {hour:2d}时: {value:.1f}辆")
        
        # 2. 工作日vs周末模式对比
        print_section_header("工作日vs周末模式对比", level=2)
        
        weekday_pattern = self.df[self.df['IsWeekend'] == 0].groupby('Hour')[target_col].mean()
        weekend_pattern = self.df[self.df['IsWeekend'] == 1].groupby('Hour')[target_col].mean()
        
        pattern_diff = weekday_pattern - weekend_pattern
        max_diff_hour = pattern_diff.abs().idxmax()
        max_diff_value = pattern_diff[max_diff_hour]
        
        print(f"最大差异时段: {max_diff_hour}时")
        print(f"差异值: {max_diff_value:.1f}辆")
        print(f"工作日平均: {weekday_pattern[max_diff_hour]:.1f}辆")
        print(f"周末平均: {weekend_pattern[max_diff_hour]:.1f}辆")
        
        # 3. 季节性变化模式
        print_section_header("季节性变化模式", level=2)
        
        if 'Seasons' in self.df.columns:
            seasonal_stats = {}
            for season in self.df['Seasons'].unique():
                season_data = self.df[self.df['Seasons'] == season]
                seasonal_hourly = season_data.groupby('Hour')[target_col].mean()
                
                # 计算季节性特征
                daily_total = season_data.groupby(season_data[config.DATA_CONFIG['date_column']].dt.date)[target_col].sum().mean()
                peak_hour = seasonal_hourly.idxmax()
                peak_value = seasonal_hourly.max()
                
                seasonal_stats[season] = {
                    'daily_average': daily_total,
                    'peak_hour': peak_hour,
                    'peak_value': peak_value,
                    'hourly_pattern': seasonal_hourly.to_dict()
                }
                
                print(f"{season}:")
                print(f"  日均总量: {daily_total:.1f}辆")
                print(f"  峰值时间: {peak_hour}时")
                print(f"  峰值需求: {peak_value:.1f}辆")
        
        # 4. 月度趋势分析
        print_section_header("月度趋势分析", level=2)
        
        monthly_avg = self.df.groupby('Month')[target_col].mean()
        monthly_growth = monthly_avg.pct_change() * 100
        
        print("月度平均需求变化率:")
        for month, growth in monthly_growth.dropna().items():
            growth_trend = "增长" if growth > 0 else "下降"
            print(f"  {month:2d}月: {growth:+.1f}% ({growth_trend})")
        
        # 保存高级时间分析结果
        analysis_results.update({
            'peaks': peaks,
            'valleys': valleys,
            'weekday_weekend_diff': {
                'max_diff_hour': max_diff_hour,
                'max_diff_value': max_diff_value
            },
            'seasonal_stats': seasonal_stats if 'Seasons' in self.df.columns else None,
            'monthly_growth': monthly_growth.to_dict()
        })
        
        self.analysis_results['advanced_time_analysis'] = analysis_results
        return analysis_results
    
    def weather_impact_deep_dive(self):
        """天气影响深度挖掘"""
        print_section_header("天气影响深度挖掘", level=1)
        
        target_col = config.DATA_CONFIG['target_column']
        analysis_results = {}
        
        # 1. 温度舒适区间分析
        print_section_header("温度舒适区间分析", level=2)
        
        temp_col = 'Temperature(°C)'
        if temp_col in self.df.columns:
            # 按温度区间分析需求
            temp_bins = np.arange(-20, 41, 5)  # 5度一个区间
            self.df['TempBin'] = pd.cut(self.df[temp_col], bins=temp_bins)
            
            temp_demand = self.df.groupby('TempBin')[target_col].agg(['mean', 'count'])
            temp_demand = temp_demand[temp_demand['count'] >= 10]  # 至少10个样本
            
            # 找到最优温度区间
            optimal_temp_bin = temp_demand['mean'].idxmax()
            optimal_demand = temp_demand['mean'].max()
            
            print(f"最优温度区间: {optimal_temp_bin}")
            print(f"该区间平均需求: {optimal_demand:.1f}辆")
            
            # 计算温度敏感性
            temp_corr = self.df[temp_col].corr(self.df[target_col])
            print(f"温度相关性: {temp_corr:.4f}")
            
            analysis_results['temperature'] = {
                'optimal_range': str(optimal_temp_bin),
                'optimal_demand': optimal_demand,
                'correlation': temp_corr
            }
        
        # 2. 复合天气条件分析
        print_section_header("复合天气条件分析", level=2)
        
        # 定义理想天气条件
        ideal_weather_mask = (
            (self.df['Temperature(°C)'].between(15, 25)) &
            (self.df['Humidity(%)'].between(40, 70)) &
            (self.df['Wind speed (m/s)'] < 4) &
            (self.df['Rainfall(mm)'] == 0) &
            (self.df['Snowfall (cm)'] == 0)
        )
        
        ideal_demand = self.df[ideal_weather_mask][target_col].mean()
        ideal_count = ideal_weather_mask.sum()
        ideal_percentage = ideal_count / len(self.df) * 100
        
        print(f"理想天气条件:")
        print(f"  出现次数: {ideal_count} ({ideal_percentage:.1f}%)")
        print(f"  平均需求: {ideal_demand:.1f}辆")
        print(f"  vs 总体平均: {self.df[target_col].mean():.1f}辆")
        
        # 恶劣天气条件分析
        extreme_weather_mask = (
            (self.df['Temperature(°C)'] < 0) |
            (self.df['Temperature(°C)'] > 35) |
            (self.df['Humidity(%)'] > 90) |
            (self.df['Wind speed (m/s)'] > 8) |
            (self.df['Rainfall(mm)'] > 10) |
            (self.df['Snowfall (cm)'] > 2)
        )
        
        extreme_demand = self.df[extreme_weather_mask][target_col].mean()
        extreme_count = extreme_weather_mask.sum()
        extreme_percentage = extreme_count / len(self.df) * 100
        
        print(f"\n恶劣天气条件:")
        print(f"  出现次数: {extreme_count} ({extreme_percentage:.1f}%)")
        print(f"  平均需求: {extreme_demand:.1f}辆")
        print(f"  vs 总体平均: {self.df[target_col].mean():.1f}辆")
        
        # 3. 天气×时间交互效应
        print_section_header("天气×时间交互效应", level=2)
        
        # 分析不同天气条件下的时间模式
        weather_conditions = {
            '理想天气': ideal_weather_mask,
            '恶劣天气': extreme_weather_mask,
            '普通天气': ~(ideal_weather_mask | extreme_weather_mask)
        }
        
        for condition_name, condition_mask in weather_conditions.items():
            if condition_mask.sum() > 0:
                condition_hourly = self.df[condition_mask].groupby('Hour')[target_col].mean()
                peak_hour = condition_hourly.idxmax()
                peak_value = condition_hourly.max()
                
                print(f"{condition_name}:")
                print(f"  高峰时间: {peak_hour}时")
                print(f"  高峰需求: {peak_value:.1f}辆")
        
        # 保存天气深度分析结果
        analysis_results.update({
            'ideal_weather': {
                'count': ideal_count,
                'percentage': ideal_percentage,
                'average_demand': ideal_demand
            },
            'extreme_weather': {
                'count': extreme_count,
                'percentage': extreme_percentage,
                'average_demand': extreme_demand
            },
            'weather_time_interaction': {
                condition: {
                    'peak_hour': self.df[mask].groupby('Hour')[target_col].mean().idxmax() if mask.sum() > 0 else None,
                    'peak_demand': self.df[mask].groupby('Hour')[target_col].mean().max() if mask.sum() > 0 else None
                }
                for condition, mask in weather_conditions.items()
            }
        })
        
        self.analysis_results['weather_deep_analysis'] = analysis_results
        return analysis_results
    
    def demand_pattern_segmentation(self):
        """需求模式分割分析"""
        print_section_header("需求模式分割分析", level=1)
        
        target_col = config.DATA_CONFIG['target_column']
        analysis_results = {}
        
        # 1. 需求水平分层
        print_section_header("需求水平分层", level=2)
        
        # 定义需求等级
        demand_quantiles = self.df[target_col].quantile([0.1, 0.3, 0.7, 0.9])
        
        def categorize_demand(demand):
            if demand <= demand_quantiles[0.1]:
                return '极低需求'
            elif demand <= demand_quantiles[0.3]:
                return '低需求'
            elif demand <= demand_quantiles[0.7]:
                return '中等需求'
            elif demand <= demand_quantiles[0.9]:
                return '高需求'
            else:
                return '极高需求'
        
        self.df['DemandLevel'] = self.df[target_col].apply(categorize_demand)
        
        demand_distribution = self.df['DemandLevel'].value_counts()
        print("需求分层分布:")
        for level, count in demand_distribution.items():
            percentage = count / len(self.df) * 100
            print(f"  {level}: {count} ({percentage:.1f}%)")
        
        # 2. 各需求层级的特征分析
        print_section_header("各需求层级特征分析", level=2)
        
        level_characteristics = {}
        
        for level in demand_distribution.index:
            level_data = self.df[self.df['DemandLevel'] == level]
            
            # 时间特征
            peak_hour = level_data['Hour'].mode().iloc[0] if len(level_data) > 0 else None
            peak_season = level_data['Seasons'].mode().iloc[0] if 'Seasons' in level_data.columns and len(level_data) > 0 else None
            weekend_ratio = level_data['IsWeekend'].mean() if len(level_data) > 0 else 0
            
            # 天气特征
            avg_temp = level_data['Temperature(°C)'].mean() if 'Temperature(°C)' in level_data.columns else None
            avg_humidity = level_data['Humidity(%)'].mean() if 'Humidity(%)' in level_data.columns else None
            
            level_characteristics[level] = {
                'count': len(level_data),
                'peak_hour': peak_hour,
                'peak_season': peak_season,
                'weekend_ratio': weekend_ratio,
                'avg_temp': avg_temp,
                'avg_humidity': avg_humidity
            }
            
            print(f"{level}:")
            print(f"  主要时间: {peak_hour}时")
            if peak_season:
                print(f"  主要季节: {peak_season}")
            print(f"  周末比例: {weekend_ratio:.2f}")
            if avg_temp:
                print(f"  平均温度: {avg_temp:.1f}°C")
        
        # 3. 异常高需求事件分析
        print_section_header("异常高需求事件分析", level=2)
        
        # 定义异常高需求（99分位数以上）
        extreme_threshold = self.df[target_col].quantile(0.99)
        extreme_events = self.df[self.df[target_col] >= extreme_threshold]
        
        print(f"异常高需求阈值: {extreme_threshold:.0f}辆")
        print(f"异常事件数量: {len(extreme_events)}")
        
        if len(extreme_events) > 0:
            # 分析异常事件特征
            extreme_hour_dist = extreme_events['Hour'].value_counts().head(3)
            extreme_season_dist = extreme_events['Seasons'].value_counts().head(3) if 'Seasons' in extreme_events.columns else None
            
            print(f"异常事件高发时段:")
            for hour, count in extreme_hour_dist.items():
                print(f"  {hour}时: {count}次")
            
            if extreme_season_dist is not None:
                print(f"异常事件高发季节:")
                for season, count in extreme_season_dist.items():
                    print(f"  {season}: {count}次")
        
        # 保存需求分割分析结果
        analysis_results.update({
            'demand_levels': demand_distribution.to_dict(),
            'level_characteristics': level_characteristics,
            'extreme_events': {
                'threshold': extreme_threshold,
                'count': len(extreme_events),
                'characteristics': {
                    'hour_distribution': extreme_hour_dist.to_dict() if len(extreme_events) > 0 else {},
                    'season_distribution': extreme_season_dist.to_dict() if len(extreme_events) > 0 and extreme_season_dist is not None else {}
                }
            }
        })
        
        self.analysis_results['demand_segmentation'] = analysis_results
        return analysis_results
    
    def predictive_insights_analysis(self):
        """预测性洞察分析"""
        print_section_header("预测性洞察分析", level=1)
        
        target_col = config.DATA_CONFIG['target_column']
        analysis_results = {}
        
        # 1. 特征重要性初步评估
        print_section_header("特征重要性初步评估", level=2)
        
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [col for col in numeric_features if col != target_col]
        
        feature_importance = {}
        
        for feature in numeric_features:
            if feature in self.df.columns:
                # 计算相关性
                correlation = abs(self.df[feature].corr(self.df[target_col]))
                
                # 计算互信息（简化版）
                try:
                    from sklearn.feature_selection import mutual_info_regression
                    mi_score = mutual_info_regression(
                        self.df[[feature]].fillna(0), 
                        self.df[target_col], 
                        random_state=42
                    )[0]
                except:
                    mi_score = 0
                
                feature_importance[feature] = {
                    'correlation': correlation,
                    'mutual_info': mi_score,
                    'combined_score': correlation * 0.7 + mi_score * 0.3
                }
        
        # 排序特征重要性
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: x[1]['combined_score'], 
                               reverse=True)
        
        print("特征重要性排序（综合评分）:")
        for i, (feature, scores) in enumerate(sorted_features[:10], 1):
            print(f"{i:2d}. {feature:25s}: {scores['combined_score']:.4f} "
                  f"(相关性={scores['correlation']:.3f}, 互信息={scores['mutual_info']:.3f})")
        
        # 2. 可预测性分析
        print_section_header("可预测性分析", level=2)
        
        # 基于自相关分析可预测性
        try:
            from statsmodels.tsa.stattools import acf
            
            # 计算自相关函数
            autocorr = acf(self.df[target_col].values, nlags=24, fft=True)
            
            # 1小时滞后相关性
            lag_1h_corr = autocorr[1]
            # 24小时滞后相关性（同时间昨天）
            lag_24h_corr = autocorr[24] if len(autocorr) > 24 else 0
            
            print(f"1小时自相关性: {lag_1h_corr:.4f}")
            print(f"24小时自相关性: {lag_24h_corr:.4f}")
            
            if lag_1h_corr > 0.3:
                print("  - 短期预测性较好")
            if lag_24h_corr > 0.3:
                print("  - 日模式预测性较好")
            
            analysis_results['predictability'] = {
                'lag_1h_correlation': lag_1h_corr,
                'lag_24h_correlation': lag_24h_corr
            }
            
        except ImportError:
            self.logger.warning("statsmodels不可用，跳过自相关分析")
        
        # 3. 建模策略建议
        print_section_header("建模策略建议", level=2)
        
        strategies = []
        
        # 基于数据特征给出建议
        if self.analysis_results.get('deep_target_analysis', {}).get('advanced_stats', {}).get('skewness', 0) > 1:
            strategies.append("考虑对目标变量进行对数变换以处理右偏分布")
        
        if len(sorted_features) > 0 and sorted_features[0][1]['combined_score'] > 0.5:
            strategies.append(f"重点关注特征 '{sorted_features[0][0]}'，它与目标变量关系最强")
        
        if 'predictability' in analysis_results:
            if analysis_results['predictability']['lag_24h_correlation'] > 0.3:
                strategies.append("可以考虑使用滞后特征，但要注意数据泄露")
        
        zero_percentage = self.analysis_results.get('deep_target_analysis', {}).get('advanced_stats', {}).get('zero_analysis', {}).get('percentage', 0)
        if zero_percentage > 10:
            strategies.append("考虑使用零膨胀模型处理大量零值")
        
        strategies.append("使用时间序列交叉验证确保模型泛化能力")
        strategies.append("考虑集成方法结合不同算法的优势")
        
        print("建模策略建议:")
        for i, strategy in enumerate(strategies, 1):
            print(f"{i}. {strategy}")
        
        # 保存预测性洞察结果
        analysis_results.update({
            'feature_importance': {k: v for k, v in sorted_features[:10]},
            'modeling_strategies': strategies
        })
        
        self.analysis_results['predictive_insights'] = analysis_results
        return analysis_results
    
    def generate_comprehensive_insights_report(self):
        """生成综合洞察报告"""
        print_section_header("综合洞察报告", level=1)
        
        # 保存所有分析结果
        timestamp = get_timestamp()
        self.result_saver.save_json(self.analysis_results, 
                                   f"deep_analysis_results_{timestamp}", 
                                   "analysis")
        
        # 生成执行摘要
        print_section_header("执行摘要", level=1)
        
        print("🎯 关键洞察:")
        
        # 1. 数据质量洞察
        zero_pct = self.analysis_results.get('deep_target_analysis', {}).get('advanced_stats', {}).get('zero_analysis', {}).get('percentage', 0)
        print(f"1. 数据质量: 零值占比 {zero_pct:.1f}%，需要特别处理")
        
        # 2. 时间模式洞察
        peaks = self.analysis_results.get('advanced_time_analysis', {}).get('peaks', [])
        if peaks:
            peak_hours = [str(p[0]) for p in peaks]
            print(f"2. 时间模式: 存在明显双峰模式，高峰时段为 {', '.join(peak_hours)}时")
        
        # 3. 天气影响洞察
        ideal_pct = self.analysis_results.get('weather_deep_analysis', {}).get('ideal_weather', {}).get('percentage', 0)
        extreme_pct = self.analysis_results.get('weather_deep_analysis', {}).get('extreme_weather', {}).get('percentage', 0)
        print(f"3. 天气影响: 理想天气占比 {ideal_pct:.1f}%，恶劣天气占比 {extreme_pct:.1f}%")
        
        # 4. 需求分层洞察
        demand_levels = self.analysis_results.get('demand_segmentation', {}).get('demand_levels', {})
        if demand_levels:
            high_demand_pct = demand_levels.get('高需求', 0) + demand_levels.get('极高需求', 0)
            print(f"4. 需求分层: 高需求时段占比 {high_demand_pct:.1f}%")
        
        # 5. 可预测性洞察
        feature_importance = self.analysis_results.get('predictive_insights', {}).get('feature_importance', {})
        if feature_importance:
            top_feature = list(feature_importance.keys())[0]
            top_score = list(feature_importance.values())[0]['combined_score']
            print(f"5. 可预测性: 最重要特征为 '{top_feature}'，综合评分 {top_score:.3f}")
        
        print(f"\n📋 建模建议:")
        strategies = self.analysis_results.get('predictive_insights', {}).get('modeling_strategies', [])
        for i, strategy in enumerate(strategies[:5], 1):
            print(f"{i}. {strategy}")
        
        print(f"\n✅ 深度分析完成！")
        print(f"📊 结果已保存到: {config.OUTPUT_DIR}")
        
        return self.analysis_results

def main():
    """主函数"""
    print_section_header("首尔自行车共享数据 - 深度洞察分析", level=1)
    
    # 创建分析器实例
    analyzer = EnhancedDataAnalyzer()
    
    try:
        # 1. 数据加载与准备
        df = analyzer.load_and_prepare_data()
        
        # 2. 深度目标变量分析
        target_analysis = analyzer.deep_target_analysis()
        
        # 3. 高级时间模式分析
        time_analysis = analyzer.advanced_time_pattern_analysis()
        
        # 4. 天气影响深度挖掘
        weather_analysis = analyzer.weather_impact_deep_dive()
        
        # 5. 需求模式分割分析
        segmentation_analysis = analyzer.demand_pattern_segmentation()
        
        # 6. 预测性洞察分析
        predictive_analysis = analyzer.predictive_insights_analysis()
        
        # 7. 生成综合洞察报告
        results = analyzer.generate_comprehensive_insights_report()
        
        return results
        
    except Exception as e:
        analyzer.logger.error(f"深度分析过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    results = main() 