#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验1：机器学习算法比较 (成员A)
CDS503 Group Project - 首尔自行车需求预测

目标：比较不同机器学习算法在自行车需求预测任务上的性能
算法：线性回归、随机森林、XGBoost、支持向量回归、神经网络
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb
import time
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import BikeDataPreprocessor

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AlgorithmComparison:
    """算法比较实验类"""
    
    def __init__(self):
        self.results = {}
        self.models = {}
        self.preprocessor = BikeDataPreprocessor()
        
    def calculate_metrics(self, y_true, y_pred):
        """计算评估指标"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape
        }
    
    def define_algorithms(self):
        """定义要比较的算法"""
        algorithms = {
            'Linear Regression': {
                'model': LinearRegression(),
                'params': {}
            },
            'Ridge Regression': {
                'model': Ridge(),
                'params': {
                    'alpha': [0.1, 1, 10, 100]
                }
            },
            'Lasso Regression': {
                'model': Lasso(),
                'params': {
                    'alpha': [0.1, 1, 10, 100]
                }
            },
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'XGBoost': {
                'model': xgb.XGBRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0]
                }
            },
            'SVR': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto'],
                    'kernel': ['rbf', 'linear']
                }
            },
            'Neural Network': {
                'model': MLPRegressor(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
        
        return algorithms
    
    def train_and_evaluate_model(self, name, model_config, X_train, X_val, X_test, 
                                y_train, y_val, y_test, use_cv=True):
        """训练和评估单个模型"""
        print(f"\n=== 训练 {name} ===")
        
        start_time = time.time()
        
        if use_cv and model_config['params']:
            # 使用时间序列交叉验证进行超参数调优
            tscv = TimeSeriesSplit(n_splits=3)
            
            # 简化参数网格以加快速度
            if name in ['XGBoost', 'Random Forest', 'Neural Network']:
                # 对于计算密集的模型，减少参数组合
                simplified_params = {}
                for key, values in model_config['params'].items():
                    simplified_params[key] = values[:2] if len(values) > 2 else values
                param_grid = simplified_params
            else:
                param_grid = model_config['params']
            
            print(f"参数网格搜索: {param_grid}")
            
            grid_search = GridSearchCV(
                model_config['model'],
                param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            print(f"最佳参数: {grid_search.best_params_}")
            print(f"最佳CV分数: {-grid_search.best_score_:.2f}")
            
        else:
            # 直接使用默认参数
            best_model = model_config['model']
            best_model.fit(X_train, y_train)
        
        # 预测
        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)
        y_test_pred = best_model.predict(X_test)
        
        training_time = time.time() - start_time
        
        # 计算评估指标
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        val_metrics = self.calculate_metrics(y_val, y_val_pred)
        test_metrics = self.calculate_metrics(y_test, y_test_pred)
        
        print(f"训练时间: {training_time:.2f}秒")
        print(f"验证集 RMSE: {val_metrics['RMSE']:.2f}")
        print(f"测试集 RMSE: {test_metrics['RMSE']:.2f}")
        print(f"测试集 R²: {test_metrics['R²']:.4f}")
        
        # 存储结果
        self.results[name] = {
            'model': best_model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'training_time': training_time,
            'predictions': {
                'y_train_pred': y_train_pred,
                'y_val_pred': y_val_pred,
                'y_test_pred': y_test_pred
            }
        }
        
        return best_model
    
    def run_experiment(self, use_lag_features=False):
        """运行算法比较实验"""
        print("🚀 开始实验1：机器学习算法比较")
        print("="*50)
        
        # 数据预处理
        print("1. 数据预处理...")
        df = self.preprocessor.load_data()
        df_processed = self.preprocessor.prepare_features(df, use_lag_features=use_lag_features)
        
        # 数据分割
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.preprocessor.split_data_temporal(df_processed)
        
        # 特征标准化
        X_train_scaled, X_val_scaled, X_test_scaled = \
            self.preprocessor.scale_features(X_train, X_val, X_test)
        
        print(f"\n数据概览:")
        print(f"  特征数量: {X_train_scaled.shape[1]}")
        print(f"  训练样本: {X_train_scaled.shape[0]}")
        print(f"  验证样本: {X_val_scaled.shape[0]}")
        print(f"  测试样本: {X_test_scaled.shape[0]}")
        
        # 定义算法
        algorithms = self.define_algorithms()
        
        print(f"\n2. 开始训练 {len(algorithms)} 个算法...")
        
        # 训练所有算法
        for i, (name, config) in enumerate(algorithms.items(), 1):
            print(f"\n进度: {i}/{len(algorithms)}")
            try:
                self.train_and_evaluate_model(
                    name, config,
                    X_train_scaled, X_val_scaled, X_test_scaled,
                    y_train, y_val, y_test,
                    use_cv=True
                )
            except Exception as e:
                print(f"❌ {name} 训练失败: {str(e)}")
                continue
        
        print("\n✅ 所有算法训练完成!")
        
        # 生成结果报告
        self.generate_comparison_report(y_test)
        
        # 保存结果
        self.save_results()
        
        return self.results
    
    def generate_comparison_report(self, y_test):
        """生成比较报告"""
        print("\n📊 生成算法比较报告...")
        
        # 创建结果DataFrame
        comparison_data = []
        for name, result in self.results.items():
            row = {
                'Algorithm': name,
                'Train_RMSE': result['train_metrics']['RMSE'],
                'Val_RMSE': result['val_metrics']['RMSE'],
                'Test_RMSE': result['test_metrics']['RMSE'],
                'Test_MAE': result['test_metrics']['MAE'],
                'Test_R²': result['test_metrics']['R²'],
                'Test_MAPE': result['test_metrics']['MAPE'],
                'Training_Time': result['training_time']
            }
            comparison_data.append(row)
        
        results_df = pd.DataFrame(comparison_data)
        results_df = results_df.sort_values('Test_RMSE')
        
        print("\n📈 算法性能比较 (按测试集RMSE排序):")
        print("="*80)
        print(results_df.round(4).to_string(index=False))
        
        # 保存结果表格
        results_df.to_csv('experiment_1_results.csv', index=False)
        
        # 创建可视化
        self.create_visualizations(results_df, y_test)
        
        # 性能分析
        print(f"\n🏆 最佳算法:")
        best_algorithm = results_df.iloc[0]
        print(f"  算法: {best_algorithm['Algorithm']}")
        print(f"  测试集RMSE: {best_algorithm['Test_RMSE']:.2f}")
        print(f"  测试集R²: {best_algorithm['Test_R²']:.4f}")
        print(f"  训练时间: {best_algorithm['Training_Time']:.2f}秒")
        
        # 达到目标检查
        target_rmse = 200
        target_r2 = 0.75
        
        print(f"\n🎯 目标达成情况:")
        for _, row in results_df.iterrows():
            rmse_ok = "✅" if row['Test_RMSE'] < target_rmse else "❌"
            r2_ok = "✅" if row['Test_R²'] > target_r2 else "❌"
            print(f"  {row['Algorithm']}: RMSE<200 {rmse_ok}, R²>0.75 {r2_ok}")
    
    def create_visualizations(self, results_df, y_test):
        """创建可视化图表"""
        
        # 设置图形样式
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 算法性能比较条形图
        plt.subplot(2, 3, 1)
        algorithms = results_df['Algorithm']
        rmse_values = results_df['Test_RMSE']
        
        bars = plt.bar(range(len(algorithms)), rmse_values, color='skyblue', alpha=0.7)
        plt.xlabel('算法')
        plt.ylabel('RMSE')
        plt.title('测试集RMSE比较')
        plt.xticks(range(len(algorithms)), algorithms, rotation=45, ha='right')
        
        # 添加数值标签
        for i, v in enumerate(rmse_values):
            plt.text(i, v + 5, f'{v:.1f}', ha='center', va='bottom')
        
        plt.grid(axis='y', alpha=0.3)
        
        # 2. R²分数比较
        plt.subplot(2, 3, 2)
        r2_values = results_df['Test_R²']
        bars = plt.bar(range(len(algorithms)), r2_values, color='lightgreen', alpha=0.7)
        plt.xlabel('算法')
        plt.ylabel('R² Score')
        plt.title('测试集R²比较')
        plt.xticks(range(len(algorithms)), algorithms, rotation=45, ha='right')
        
        # 添加数值标签
        for i, v in enumerate(r2_values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.grid(axis='y', alpha=0.3)
        
        # 3. 训练时间比较
        plt.subplot(2, 3, 3)
        time_values = results_df['Training_Time']
        bars = plt.bar(range(len(algorithms)), time_values, color='orange', alpha=0.7)
        plt.xlabel('算法')
        plt.ylabel('训练时间 (秒)')
        plt.title('训练时间比较')
        plt.xticks(range(len(algorithms)), algorithms, rotation=45, ha='right')
        
        # 添加数值标签
        for i, v in enumerate(time_values):
            plt.text(i, v + max(time_values)*0.02, f'{v:.1f}', ha='center', va='bottom')
        
        plt.grid(axis='y', alpha=0.3)
        
        # 4. RMSE vs R²散点图
        plt.subplot(2, 3, 4)
        plt.scatter(rmse_values, r2_values, s=100, alpha=0.7, c='purple')
        
        for i, alg in enumerate(algorithms):
            plt.annotate(alg, (rmse_values.iloc[i], r2_values.iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('RMSE')
        plt.ylabel('R² Score')
        plt.title('RMSE vs R² 关系图')
        plt.grid(True, alpha=0.3)
        
        # 5. 预测效果对比（最佳算法）
        plt.subplot(2, 3, 5)
        best_alg_name = results_df.iloc[0]['Algorithm']
        best_predictions = self.results[best_alg_name]['predictions']['y_test_pred']
        
        # 随机选择一些点进行可视化
        sample_indices = np.random.choice(len(y_test), size=min(200, len(y_test)), replace=False)
        sample_indices = sorted(sample_indices)
        
        plt.scatter(y_test.iloc[sample_indices], best_predictions[sample_indices], 
                   alpha=0.6, s=20)
        
        # 添加完美预测线
        min_val = min(y_test.min(), best_predictions.min())
        max_val = max(y_test.max(), best_predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
        
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title(f'{best_alg_name} - 预测效果')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. 误差分布
        plt.subplot(2, 3, 6)
        errors = y_test.values - best_predictions
        plt.hist(errors, bins=30, alpha=0.7, color='red', edgecolor='black')
        plt.xlabel('预测误差')
        plt.ylabel('频次')
        plt.title(f'{best_alg_name} - 误差分布')
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.8)
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        plt.text(0.02, 0.98, f'均值: {errors.mean():.2f}\n标准差: {errors.std():.2f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('experiment_1_algorithm_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📸 可视化图表已保存: experiment_1_algorithm_comparison.png")
    
    def save_results(self):
        """保存实验结果"""
        # 保存模型性能总结
        summary = {}
        for name, result in self.results.items():
            summary[name] = {
                'test_rmse': result['test_metrics']['RMSE'],
                'test_r2': result['test_metrics']['R²'],
                'test_mae': result['test_metrics']['MAE'],
                'test_mape': result['test_metrics']['MAPE'],
                'training_time': result['training_time']
            }
        
        # 保存为JSON
        import json
        with open('experiment_1_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print("💾 实验结果已保存:")
        print("  - experiment_1_results.csv")
        print("  - experiment_1_summary.json")
        print("  - experiment_1_algorithm_comparison.png")

def main():
    """主函数"""
    
    # 创建实验实例
    experiment = AlgorithmComparison()
    
    # 运行实验
    results = experiment.run_experiment(use_lag_features=False)
    
    print("\n🎉 实验1完成！")
    print(f"总共测试了 {len(results)} 个算法")
    
    # 输出最终总结
    print("\n📋 实验总结:")
    print("- 已完成机器学习算法性能比较")
    print("- 识别最佳算法用于自行车需求预测")
    print("- 生成详细性能报告和可视化")
    print("- 为后续实验提供基线模型")

if __name__ == "__main__":
    main() 