#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验4：训练样本大小影响分析 (成员D)
CDS503 Group Project - 首尔自行车需求预测

目标：分析训练样本大小对模型性能的影响
方法：以10%递增的方式测试不同训练集大小（10%-100%）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import time
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import BikeDataPreprocessor

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SampleSizeAnalysis:
    """训练样本大小分析实验类"""
    
    def __init__(self):
        self.results = {}
        self.learning_curves = {}
        self.preprocessor = BikeDataPreprocessor()
        
    def calculate_metrics(self, y_true, y_pred):
        """计算评估指标"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape
        }
    
    def create_models(self):
        """创建要测试的模型"""
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        return models
    
    def train_with_sample_size(self, model, X_train, y_train, X_val, X_test, 
                              y_val, y_test, sample_ratio, model_name):
        """使用指定比例的训练数据训练模型"""
        
        # 计算样本数量
        n_samples = int(len(X_train) * sample_ratio)
        
        # 随机采样（保持时间顺序）
        indices = np.arange(len(X_train))
        np.random.seed(42)  # 确保可重复性
        selected_indices = np.sort(np.random.choice(indices, size=n_samples, replace=False))
        
        X_train_sample = X_train.iloc[selected_indices]
        y_train_sample = y_train.iloc[selected_indices]
        
        # 训练模型
        start_time = time.time()
        try:
            model.fit(X_train_sample, y_train_sample)
            training_time = time.time() - start_time
            
            # 预测
            y_train_pred = model.predict(X_train_sample)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            
            # 计算指标
            train_metrics = self.calculate_metrics(y_train_sample, y_train_pred)
            val_metrics = self.calculate_metrics(y_val, y_val_pred)
            test_metrics = self.calculate_metrics(y_test, y_test_pred)
            
            return {
                'sample_size': n_samples,
                'sample_ratio': sample_ratio,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'training_time': training_time,
                'success': True
            }
            
        except Exception as e:
            print(f"  ❌ {model_name} 在样本比例 {sample_ratio:.1%} 时训练失败: {str(e)}")
            return {
                'sample_size': n_samples,
                'sample_ratio': sample_ratio,
                'success': False,
                'error': str(e)
            }
    
    def analyze_learning_curve(self, model_name, model, X_train, y_train, 
                              X_val, X_test, y_val, y_test):
        """分析特定模型的学习曲线"""
        print(f"\n📈 分析 {model_name} 的学习曲线...")
        
        # 定义样本比例（10%到100%，以10%递增）
        sample_ratios = np.arange(0.1, 1.1, 0.1)
        
        results = []
        
        for ratio in sample_ratios:
            print(f"  训练样本比例: {ratio:.1%} ({int(len(X_train) * ratio)} 样本)")
            
            # 创建模型副本以避免重复使用已训练的模型
            if model_name == 'Linear Regression':
                model_copy = LinearRegression()
            elif model_name == 'Random Forest':
                model_copy = RandomForestRegressor(
                    n_estimators=100, max_depth=20, random_state=42
                )
            elif model_name == 'XGBoost':
                model_copy = xgb.XGBRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
                )
            elif model_name == 'SVR':
                model_copy = SVR(kernel='rbf', C=1.0, gamma='scale')
            else:
                model_copy = model
            
            result = self.train_with_sample_size(
                model_copy, X_train, y_train, X_val, X_test, 
                y_val, y_test, ratio, model_name
            )
            
            if result['success']:
                print(f"    验证集 RMSE: {result['val_metrics']['RMSE']:.2f}")
                print(f"    测试集 RMSE: {result['test_metrics']['RMSE']:.2f}")
                print(f"    训练时间: {result['training_time']:.2f}秒")
            
            results.append(result)
        
        return results
    
    def run_experiment(self):
        """运行训练样本大小分析实验"""
        print("🚀 开始实验4：训练样本大小影响分析")
        print("="*50)
        
        # 数据预处理
        print("1. 数据预处理...")
        df = self.preprocessor.load_data()
        df_processed = self.preprocessor.prepare_features(df, use_lag_features=False)
        
        # 数据分割
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.preprocessor.split_data_temporal(df_processed)
        
        # 特征标准化
        X_train_scaled, X_val_scaled, X_test_scaled = \
            self.preprocessor.scale_features(X_train, X_val, X_test)
        
        print(f"\n数据概览:")
        print(f"  特征数量: {X_train_scaled.shape[1]}")
        print(f"  训练样本总数: {X_train_scaled.shape[0]}")
        print(f"  验证样本: {X_val_scaled.shape[0]}")
        print(f"  测试样本: {X_test_scaled.shape[0]}")
        
        # 创建模型
        models = self.create_models()
        
        print(f"\n2. 开始分析 {len(models)} 个模型的学习曲线...")
        
        # 分析每个模型的学习曲线
        for i, (model_name, model) in enumerate(models.items(), 1):
            print(f"\n进度: {i}/{len(models)}")
            
            learning_curve_results = self.analyze_learning_curve(
                model_name, model,
                X_train_scaled, y_train, X_val_scaled, X_test_scaled,
                y_val, y_test
            )
            
            self.learning_curves[model_name] = learning_curve_results
        
        print("\n✅ 训练样本大小分析完成!")
        
        # 生成报告
        self.generate_analysis_report()
        
        # 保存结果
        self.save_results()
        
        return self.learning_curves
    
    def generate_analysis_report(self):
        """生成分析报告"""
        print("\n📊 生成训练样本大小分析报告...")
        
        # 创建详细结果DataFrame
        detailed_results = []
        
        for model_name, results in self.learning_curves.items():
            for result in results:
                if result['success']:
                    row = {
                        'Model': model_name,
                        'Sample_Ratio': result['sample_ratio'],
                        'Sample_Size': result['sample_size'],
                        'Train_RMSE': result['train_metrics']['RMSE'],
                        'Val_RMSE': result['val_metrics']['RMSE'],
                        'Test_RMSE': result['test_metrics']['RMSE'],
                        'Test_R²': result['test_metrics']['R²'],
                        'Training_Time': result['training_time']
                    }
                    detailed_results.append(row)
        
        results_df = pd.DataFrame(detailed_results)
        
        # 保存详细结果
        results_df.to_csv('experiment_4_detailed_results.csv', index=False)
        
        # 创建汇总表（每个模型在100%数据上的最终性能）
        summary_results = []
        for model_name in self.learning_curves.keys():
            model_results = results_df[results_df['Model'] == model_name]
            if not model_results.empty:
                # 获取100%数据的结果
                full_data_result = model_results[model_results['Sample_Ratio'] == 1.0]
                if not full_data_result.empty:
                    best_result = full_data_result.iloc[0]
                    summary_results.append({
                        'Model': model_name,
                        'Final_Test_RMSE': best_result['Test_RMSE'],
                        'Final_Test_R²': best_result['Test_R²'],
                        'Final_Training_Time': best_result['Training_Time'],
                        'Data_Efficiency': self.calculate_data_efficiency(model_name, results_df)
                    })
        
        summary_df = pd.DataFrame(summary_results)
        summary_df = summary_df.sort_values('Final_Test_RMSE')
        
        print("\n📈 模型最终性能汇总 (100%训练数据):")
        print("="*70)
        print(summary_df.round(4).to_string(index=False))
        
        # 保存汇总结果
        summary_df.to_csv('experiment_4_summary.csv', index=False)
        
        # 创建可视化
        self.create_visualizations(results_df, summary_df)
        
        # 数据效率分析
        print(f"\n📊 数据效率分析:")
        self.analyze_data_efficiency(results_df)
        
        # 性能分析
        print(f"\n🏆 最佳模型 (100%数据):")
        best_model = summary_df.iloc[0]
        print(f"  模型: {best_model['Model']}")
        print(f"  测试集RMSE: {best_model['Final_Test_RMSE']:.2f}")
        print(f"  测试集R²: {best_model['Final_Test_R²']:.4f}")
        print(f"  数据效率: {best_model['Data_Efficiency']:.2f}")
        
        # 推荐最小训练集大小
        print(f"\n💡 推荐训练集大小:")
        self.recommend_training_size(results_df)
    
    def calculate_data_efficiency(self, model_name, results_df):
        """计算数据效率指标"""
        model_results = results_df[results_df['Model'] == model_name].sort_values('Sample_Ratio')
        
        if len(model_results) < 2:
            return 0.0
        
        # 计算从50%到100%数据时的性能改进
        result_50 = model_results[model_results['Sample_Ratio'] >= 0.5]
        result_100 = model_results[model_results['Sample_Ratio'] == 1.0]
        
        if len(result_50) > 0 and len(result_100) > 0:
            rmse_50 = result_50.iloc[0]['Test_RMSE']
            rmse_100 = result_100.iloc[0]['Test_RMSE']
            
            # 数据效率 = (RMSE_50 - RMSE_100) / RMSE_100 * 100
            # 值越高表示增加数据的边际效益越大
            efficiency = (rmse_50 - rmse_100) / rmse_100 * 100
            return max(0, efficiency)  # 确保非负
        
        return 0.0
    
    def analyze_data_efficiency(self, results_df):
        """分析数据效率"""
        
        for model_name in results_df['Model'].unique():
            model_results = results_df[results_df['Model'] == model_name].sort_values('Sample_Ratio')
            
            print(f"\n  {model_name}:")
            
            # 找到性能稳定点（RMSE变化<1%的点）
            stable_point = None
            for i in range(1, len(model_results)):
                prev_rmse = model_results.iloc[i-1]['Test_RMSE']
                curr_rmse = model_results.iloc[i]['Test_RMSE']
                
                improvement = (prev_rmse - curr_rmse) / prev_rmse
                if improvement < 0.01:  # 改进小于1%
                    stable_point = model_results.iloc[i]['Sample_Ratio']
                    break
            
            if stable_point:
                print(f"    性能稳定点: {stable_point:.1%} 训练数据")
            else:
                print(f"    性能稳定点: 未找到 (建议使用更多数据)")
            
            # 计算边际效益
            if len(model_results) >= 5:
                rmse_values = model_results['Test_RMSE'].values
                marginal_benefits = []
                for i in range(1, len(rmse_values)):
                    benefit = (rmse_values[i-1] - rmse_values[i]) / rmse_values[i-1] * 100
                    marginal_benefits.append(benefit)
                
                avg_marginal_benefit = np.mean(marginal_benefits)
                print(f"    平均边际效益: {avg_marginal_benefit:.2f}%")
    
    def recommend_training_size(self, results_df):
        """推荐最佳训练集大小"""
        
        target_rmse = 200  # 目标RMSE
        
        for model_name in results_df['Model'].unique():
            model_results = results_df[results_df['Model'] == model_name].sort_values('Sample_Ratio')
            
            # 找到达到目标性能的最小数据量
            target_achieved = model_results[model_results['Test_RMSE'] <= target_rmse]
            
            if not target_achieved.empty:
                min_ratio = target_achieved['Sample_Ratio'].min()
                min_samples = target_achieved[target_achieved['Sample_Ratio'] == min_ratio]['Sample_Size'].iloc[0]
                print(f"  {model_name}: 最少需要 {min_ratio:.1%} 数据 ({min_samples} 样本) 达到RMSE≤{target_rmse}")
            else:
                print(f"  {model_name}: 无法达到RMSE≤{target_rmse}目标")
    
    def create_visualizations(self, results_df, summary_df):
        """创建可视化图表"""
        
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 学习曲线 - 测试集RMSE
        plt.subplot(2, 4, 1)
        for model_name in results_df['Model'].unique():
            model_data = results_df[results_df['Model'] == model_name].sort_values('Sample_Ratio')
            plt.plot(model_data['Sample_Ratio'], model_data['Test_RMSE'], 
                    marker='o', label=model_name, linewidth=2, markersize=6)
        
        plt.xlabel('训练样本比例')
        plt.ylabel('测试集 RMSE')
        plt.title('学习曲线 - 测试集RMSE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0.1, 1.0)
        
        # 2. 学习曲线 - 测试集R²
        plt.subplot(2, 4, 2)
        for model_name in results_df['Model'].unique():
            model_data = results_df[results_df['Model'] == model_name].sort_values('Sample_Ratio')
            plt.plot(model_data['Sample_Ratio'], model_data['Test_R²'], 
                    marker='s', label=model_name, linewidth=2, markersize=6)
        
        plt.xlabel('训练样本比例')
        plt.ylabel('测试集 R²')
        plt.title('学习曲线 - 测试集R²')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0.1, 1.0)
        
        # 3. 训练时间与样本量关系
        plt.subplot(2, 4, 3)
        for model_name in results_df['Model'].unique():
            model_data = results_df[results_df['Model'] == model_name].sort_values('Sample_Ratio')
            plt.plot(model_data['Sample_Size'], model_data['Training_Time'], 
                    marker='^', label=model_name, linewidth=2, markersize=6)
        
        plt.xlabel('训练样本数量')
        plt.ylabel('训练时间 (秒)')
        plt.title('训练时间 vs 样本量')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. 过拟合分析 (训练集vs测试集RMSE)
        plt.subplot(2, 4, 4)
        sample_ratios = [0.3, 0.6, 1.0]  # 选择几个代表性的比例
        colors = ['red', 'blue', 'green']
        
        for i, ratio in enumerate(sample_ratios):
            ratio_data = results_df[results_df['Sample_Ratio'] == ratio]
            plt.scatter(ratio_data['Train_RMSE'], ratio_data['Test_RMSE'], 
                       s=100, alpha=0.7, c=colors[i], label=f'{ratio:.0%} 数据')
        
        # 添加对角线
        min_rmse = min(results_df['Train_RMSE'].min(), results_df['Test_RMSE'].min())
        max_rmse = max(results_df['Train_RMSE'].max(), results_df['Test_RMSE'].max())
        plt.plot([min_rmse, max_rmse], [min_rmse, max_rmse], 'k--', alpha=0.8, label='理想线')
        
        plt.xlabel('训练集 RMSE')
        plt.ylabel('测试集 RMSE')
        plt.title('过拟合分析')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. 边际效益分析
        plt.subplot(2, 4, 5)
        for model_name in results_df['Model'].unique():
            model_data = results_df[results_df['Model'] == model_name].sort_values('Sample_Ratio')
            
            # 计算边际效益
            rmse_values = model_data['Test_RMSE'].values
            ratios = model_data['Sample_Ratio'].values
            
            marginal_benefits = []
            for i in range(1, len(rmse_values)):
                benefit = (rmse_values[i-1] - rmse_values[i]) / rmse_values[i-1] * 100
                marginal_benefits.append(benefit)
            
            if marginal_benefits:
                plt.plot(ratios[1:], marginal_benefits, marker='d', 
                        label=model_name, linewidth=2, markersize=6)
        
        plt.xlabel('训练样本比例')
        plt.ylabel('边际效益 (%)')
        plt.title('数据增加的边际效益')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0.2, 1.0)
        
        # 6. 最终性能比较
        plt.subplot(2, 4, 6)
        models = summary_df['Model']
        final_rmse = summary_df['Final_Test_RMSE']
        
        bars = plt.bar(range(len(models)), final_rmse, color='skyblue', alpha=0.7)
        plt.xlabel('模型')
        plt.ylabel('最终测试集 RMSE')
        plt.title('最终模型性能比较 (100%数据)')
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        
        # 添加数值标签
        for i, v in enumerate(final_rmse):
            plt.text(i, v + 5, f'{v:.1f}', ha='center', va='bottom')
        
        plt.grid(axis='y', alpha=0.3)
        
        # 7. 数据效率比较
        plt.subplot(2, 4, 7)
        efficiency = summary_df['Data_Efficiency']
        
        bars = plt.bar(range(len(models)), efficiency, color='lightgreen', alpha=0.7)
        plt.xlabel('模型')
        plt.ylabel('数据效率指标')
        plt.title('模型数据效率比较')
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        
        # 添加数值标签
        for i, v in enumerate(efficiency):
            plt.text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom')
        
        plt.grid(axis='y', alpha=0.3)
        
        # 8. 性能收敛分析
        plt.subplot(2, 4, 8)
        for model_name in results_df['Model'].unique():
            model_data = results_df[results_df['Model'] == model_name].sort_values('Sample_Ratio')
            
            # 计算相对于最终性能的差距
            final_rmse = model_data['Test_RMSE'].iloc[-1]
            relative_gap = (model_data['Test_RMSE'] - final_rmse) / final_rmse * 100
            
            plt.plot(model_data['Sample_Ratio'], relative_gap, 
                    marker='o', label=model_name, linewidth=2, markersize=6)
        
        plt.xlabel('训练样本比例')
        plt.ylabel('相对性能差距 (%)')
        plt.title('性能收敛分析')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0.1, 1.0)
        plt.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='5%阈值')
        
        plt.tight_layout()
        plt.savefig('experiment_4_sample_size_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📸 可视化图表已保存: experiment_4_sample_size_analysis.png")
    
    def save_results(self):
        """保存实验结果"""
        # 保存学习曲线数据
        learning_curve_summary = {}
        for model_name, results in self.learning_curves.items():
            learning_curve_summary[model_name] = [
                {
                    'sample_ratio': r['sample_ratio'],
                    'sample_size': r['sample_size'],
                    'test_rmse': r['test_metrics']['RMSE'] if r['success'] else None,
                    'test_r2': r['test_metrics']['R²'] if r['success'] else None,
                    'training_time': r['training_time'] if r['success'] else None,
                    'success': r['success']
                }
                for r in results
            ]
        
        # 保存为JSON
        import json
        with open('experiment_4_learning_curves.json', 'w', encoding='utf-8') as f:
            json.dump(learning_curve_summary, f, indent=2, ensure_ascii=False)
        
        print("💾 实验结果已保存:")
        print("  - experiment_4_detailed_results.csv")
        print("  - experiment_4_summary.csv")
        print("  - experiment_4_learning_curves.json")
        print("  - experiment_4_sample_size_analysis.png")

def main():
    """主函数"""
    
    # 创建实验实例
    experiment = SampleSizeAnalysis()
    
    # 运行实验
    results = experiment.run_experiment()
    
    print("\n🎉 实验4完成！")
    print(f"分析了 {len(results)} 个模型的学习曲线")
    
    # 输出最终总结
    print("\n📋 实验总结:")
    print("- 已完成训练样本大小影响分析")
    print("- 生成了完整的学习曲线")
    print("- 分析了数据效率和边际效益")
    print("- 提供了最佳训练集大小建议")

if __name__ == "__main__":
    main() 