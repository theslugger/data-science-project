#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验3：基于90特征的优化集成学习
CDS503 Group Project - 首尔自行车需求预测

针对数据预处理结果优化：
- 90个增强特征
- 8465条有效记录（排除295条非运营日）
- 152个异常值已处理
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import (
    BaggingRegressor, VotingRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import BikeDataPreprocessor

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class OptimizedEnsembleExperiment:
    """基于90特征的优化集成学习实验类"""
    
    def __init__(self):
        self.results = {}
        self.models = {}
        self.preprocessor = BikeDataPreprocessor()
        
    def calculate_metrics(self, y_true, y_pred):
        """计算全面的回归指标"""
        from sklearn.metrics import explained_variance_score, mean_squared_log_error
        
        # 基础指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 避免除零错误和负值
        y_true_safe = np.maximum(y_true, 1e-8)
        y_pred_safe = np.maximum(y_pred, 1e-8)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
        
        # MSLE (Mean Squared Log Error)
        try:
            if np.all(y_true >= 0) and np.all(y_pred >= 0):
                msle = mean_squared_log_error(y_true_safe, y_pred_safe)
            else:
                msle = np.nan
        except:
            msle = np.nan
        
        # Explained Variance Score
        evs = explained_variance_score(y_true, y_pred)
        
        # SMAPE (Symmetric Mean Absolute Percentage Error)
        smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape,
            'MSLE': msle,
            'EVS': evs,
            'SMAPE': smape
        }
    
    def create_optimized_bagging_models(self):
        """针对90特征优化的Bagging方法"""
        print("🎯 创建针对90特征优化的Bagging模型...")
        
        bagging_models = {
            'Bagging Linear (90F)': BaggingRegressor(
                estimator=LinearRegression(),
                n_estimators=100,  # 线性模型集成数量
                max_samples=0.8,  # 样本采样
                max_features=0.7,  # 70%特征采样，约63个特征
                bootstrap=True,
                bootstrap_features=True,
                random_state=42,
                n_jobs=-1
            ),
            
            'Bagging Ridge (90F)': BaggingRegressor(
                estimator=Ridge(alpha=0.01, solver='auto', max_iter=5000),
                n_estimators=100,
                max_samples=0.8,
                max_features=0.7,  # 63个特征
                bootstrap=True,
                bootstrap_features=True,
                random_state=42,
                n_jobs=-1
            ),
            
            'Bagging KNN (90F)': BaggingRegressor(
                estimator=KNeighborsRegressor(
                    n_neighbors=15,  # 针对5925个训练样本
                    weights='distance',  # 距离加权
                    algorithm='auto',
                    leaf_size=30,
                    n_jobs=-1
                ),
                n_estimators=50,  # KNN集成数量
                max_samples=0.8,
                max_features=0.6,  # 54个特征，避免维度灾难
                bootstrap=True,
                bootstrap_features=True,
                random_state=42,
                n_jobs=1  # KNN已经并行
            ),
            
            'Bagging SVR (90F)': BaggingRegressor(
                estimator=SVR(
                    kernel='rbf',
                    C=10.0,
                    gamma='scale',
                    epsilon=0.01,
                    cache_size=300
                ),
                n_estimators=30,  # SVR计算量大
                max_samples=0.7,
                max_features=0.5,  # 45个特征，避免过拟合
                bootstrap=True,
                bootstrap_features=True,
                random_state=42,
                n_jobs=-1
            ),
            
            'Bagging Neural Net (90F)': BaggingRegressor(
                estimator=MLPRegressor(
                    hidden_layer_sizes=(128, 64, 32),  # 针对90特征的网络
                    activation='relu',
                    solver='adam',
                    learning_rate='adaptive',
                    learning_rate_init=0.001,
                    alpha=0.001,
                    max_iter=3000,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=50,
                    random_state=42
                ),
                n_estimators=20,  # 神经网络集成数量
                max_samples=0.85,
                max_features=0.8,  # 72个特征
                random_state=42,
                n_jobs=1  # 避免嵌套并行
            )
        }
        
        return bagging_models
    
    def create_base_models(self):
        """创建四种基础模型"""
        print("🚀 创建四种基础模型（线性回归、KNN、神经网络、SVR）...")
        
        base_models = {
            'Linear Regression (90F)': LinearRegression(),
            
            'Ridge Regression (90F)': Ridge(
                alpha=0.01,  # 针对90特征的正则化
                solver='auto',
                max_iter=5000
            ),
            
            'Lasso Regression (90F)': Lasso(
                alpha=0.001,  # 较小的正则化保留更多特征
                max_iter=4000,
                selection='random'
            ),
            
            'KNN Regression (90F)': KNeighborsRegressor(
                n_neighbors=20,  # 针对5925个训练样本
                weights='distance',  # 距离加权
                algorithm='auto',
                leaf_size=30,
                metric='minkowski',
                p=2,  # 欧几里得距离
                n_jobs=-1
            ),
            
            'Neural Network Basic (90F)': MLPRegressor(
                hidden_layer_sizes=(128, 64),  # 基础网络
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                alpha=0.001,
                max_iter=3000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=50,
                random_state=42
            ),
            
            'Neural Network Deep (90F)': MLPRegressor(
                hidden_layer_sizes=(180, 90, 45, 22),  # 深度网络
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                learning_rate_init=0.0005,
                alpha=0.0001,
                max_iter=4000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=60,
                random_state=42
            ),
            
            'SVR Linear (90F)': SVR(
                kernel='linear',
                C=1.0,
                epsilon=0.01,
                max_iter=5000
            ),
            
            'SVR RBF (90F)': SVR(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                epsilon=0.01,
                cache_size=500
            ),
            
            'SVR Poly (90F)': SVR(
                kernel='poly',
                degree=3,
                C=1.0,
                gamma='scale',
                epsilon=0.01,
                cache_size=300
            )
        }
        
        return base_models
    
    def create_optimized_voting_models(self):
        """针对90特征优化的Voting方法"""
        print("🗳️ 创建针对90特征优化的Voting模型...")
        
        # 基础模型集合（四种类型）
        base_linear = LinearRegression()
        
        base_ridge = Ridge(alpha=0.01, solver='auto', max_iter=5000)
        
        base_knn = KNeighborsRegressor(
            n_neighbors=20, weights='distance', 
            algorithm='auto', n_jobs=-1
        )
        
        base_nn = MLPRegressor(
            hidden_layer_sizes=(180, 90, 45),  # 2x, 1x, 0.5x特征数
            activation='relu', solver='adam',
            learning_rate='adaptive', learning_rate_init=0.001,
            alpha=0.001, max_iter=4000,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=60, random_state=42
        )
        
        base_svr = SVR(
            kernel='rbf', C=10.0, gamma='scale',
            epsilon=0.01, cache_size=500
        )
        
        voting_models = {
            'Voting Linear Models (90F)': VotingRegressor(
                estimators=[
                    ('linear', base_linear),
                    ('ridge', base_ridge)
                ],
                weights=[1.0, 1.2]  # Ridge权重稍高
            ),
            
            'Voting All Four (90F)': VotingRegressor(
                estimators=[
                    ('linear', base_linear),
                    ('knn', base_knn),
                    ('nn', base_nn),
                    ('svr', base_svr)
                ],
                weights=[1.0, 0.8, 1.3, 1.1]  # NN权重最高
            ),
            
            'Voting Linear+NN (90F)': VotingRegressor(
                estimators=[
                    ('ridge', base_ridge),
                    ('nn', base_nn)
                ],
                weights=[0.8, 1.2]  # NN权重更高
            ),
            
            'Voting KNN+SVR (90F)': VotingRegressor(
                estimators=[
                    ('knn', base_knn),
                    ('svr', base_svr)
                ],
                weights=[1.0, 1.0]  # 等权重
            )
        }
        
        return voting_models
    
    def create_optimized_stacking_models(self, X_train, y_train, X_val, y_val):
        """针对90特征优化的Stacking方法"""
        print("🏗️ 创建针对90特征优化的Stacking模型...")
        
        # 第一层：四种基础模型的多个变体
        base_models = {
            'linear': LinearRegression(),
            
            'ridge': Ridge(alpha=0.01, solver='auto', max_iter=5000),
            
            'lasso': Lasso(alpha=0.001, max_iter=4000, selection='random'),
            
            'knn1': KNeighborsRegressor(
                n_neighbors=15, weights='distance', 
                algorithm='auto', n_jobs=-1
            ),
            
            'knn2': KNeighborsRegressor(
                n_neighbors=25, weights='uniform',
                algorithm='auto', n_jobs=-1
            ),
            
            'svr_linear': SVR(
                kernel='linear', C=1.0, epsilon=0.01, max_iter=5000
            ),
            
            'svr_rbf': SVR(
                kernel='rbf', C=10.0, gamma='scale',
                epsilon=0.01, cache_size=500
            ),
            
            'nn1': MLPRegressor(
                hidden_layer_sizes=(128, 64),  # 基础网络
                activation='relu', solver='adam',
                learning_rate='adaptive', learning_rate_init=0.001,
                alpha=0.001, max_iter=3000,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=40, random_state=42
            ),
            
            'nn2': MLPRegressor(
                hidden_layer_sizes=(180, 90, 45),  # 深度网络
                activation='relu', solver='adam',
                learning_rate='adaptive', learning_rate_init=0.0005,
                alpha=0.0001, max_iter=4000,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=50, random_state=43
            )
        }
        
        # 训练基础模型并获取预测
        print("  训练9个基础模型...")
        base_predictions_train = np.zeros((X_train.shape[0], len(base_models)))
        base_predictions_val = np.zeros((X_val.shape[0], len(base_models)))
        
        for i, (name, model) in enumerate(base_models.items()):
            print(f"    训练基础模型 {i+1}/9: {name}")
            model.fit(X_train, y_train)
            base_predictions_train[:, i] = model.predict(X_train)
            base_predictions_val[:, i] = model.predict(X_val)
        
        # 第二层：元学习器（针对9个基学习器输出优化）
        print("  创建元学习器...")
        meta_models = {
            'Stacking Linear (90F)': LinearRegression(),
            
            'Stacking Ridge (90F)': Ridge(alpha=0.1, solver='auto'),
            
            'Stacking Lasso (90F)': Lasso(alpha=0.02, max_iter=2000),
            
            'Stacking KNN (90F)': KNeighborsRegressor(
                n_neighbors=5, weights='distance', algorithm='auto'
            ),
            
            'Stacking SVR (90F)': SVR(
                kernel='rbf', C=1.0, gamma='scale', epsilon=0.01
            ),
            
            'Stacking NN (90F)': MLPRegressor(
                hidden_layer_sizes=(18, 9),  # 针对9个输入的网络
                activation='relu', solver='adam',
                learning_rate='constant', learning_rate_init=0.01,
                alpha=0.01, max_iter=2000,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=30, random_state=42
            )
        }
        
        stacking_results = {}
        
        for name, meta_model in meta_models.items():
            print(f"    训练元模型: {name}")
            meta_model.fit(base_predictions_train, y_train)
            
            # 创建组合模型类
            class StackingModel:
                def __init__(self, base_models, meta_model):
                    self.base_models = base_models
                    self.meta_model = meta_model
                
                def predict(self, X):
                    base_preds = np.zeros((X.shape[0], len(self.base_models)))
                    for i, model in enumerate(self.base_models.values()):
                        base_preds[:, i] = model.predict(X)
                    return self.meta_model.predict(base_preds)
                
                def fit(self, X, y):
                    pass  # 已经训练好了
            
            stacking_model = StackingModel(base_models, meta_model)
            stacking_results[name] = stacking_model
        
        return stacking_results
    
    def evaluate_model(self, model, X_train, X_val, X_test, y_train, y_val, y_test, name):
        """评估单个模型"""
        print(f"\n=== 评估 {name} ===")
        
        start_time = time.time()
        
        # 如果模型还没有训练，则训练它
        if hasattr(model, 'fit') and not hasattr(model, 'base_models'):
            model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # 预测
        try:
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
        except Exception as e:
            print(f"❌ {name} 预测失败: {str(e)}")
            return None
        
        # 计算指标
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        val_metrics = self.calculate_metrics(y_val, y_val_pred)
        test_metrics = self.calculate_metrics(y_test, y_test_pred)
        
        print(f"训练时间: {training_time:.2f}秒")
        print(f"验证集 RMSE: {val_metrics['RMSE']:.2f}")
        print(f"测试集 RMSE: {test_metrics['RMSE']:.2f}")
        print(f"测试集 R²: {test_metrics['R²']:.4f}")
        print(f"测试集 MAPE: {test_metrics['MAPE']:.2f}%")
        
        return {
            'model': model,
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
    
    def run_optimized_experiment(self):
        """运行针对90特征优化的集成学习实验"""
        print("🚀 开始针对90特征的优化集成学习实验")
        print("="*60)
        
        # 1. 数据预处理（使用滞后特征）
        print("1. 基于90特征的数据预处理...")
        df = self.preprocessor.load_data()
        df_processed = self.preprocessor.prepare_features(
            df, use_lag_features=True, exclude_non_operating=True
        )
        
        print(f"处理后数据: {len(df_processed)} 条记录")
        print(f"特征数量: {len(self.preprocessor.feature_names)}")
        
        # 2. 数据分割
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.preprocessor.split_data_temporal(df_processed)
        
        # 3. 特征标准化
        X_train_scaled, X_val_scaled, X_test_scaled = \
            self.preprocessor.scale_features(X_train, X_val, X_test)
        
        print(f"\n数据概览:")
        print(f"  特征数量: {X_train_scaled.shape[1]}")
        print(f"  训练样本: {X_train_scaled.shape[0]}")
        print(f"  验证样本: {X_val_scaled.shape[0]}")
        print(f"  测试样本: {X_test_scaled.shape[0]}")
        
        # 4. 创建所有优化模型
        print("\n2. 创建针对90特征优化的集成模型...")
        all_models = {}
        
        # 基础模型
        base_models = self.create_base_models()
        all_models.update(base_models)
        
        # Bagging方法
        bagging_models = self.create_optimized_bagging_models()
        all_models.update(bagging_models)
        
        # Voting方法
        voting_models = self.create_optimized_voting_models()
        all_models.update(voting_models)
        
        # Stacking方法
        stacking_models = self.create_optimized_stacking_models(
            X_train_scaled, y_train, X_val_scaled, y_val
        )
        all_models.update(stacking_models)
        
        # 5. 评估所有模型
        print(f"\n3. 评估 {len(all_models)} 个优化集成模型...")
        
        for i, (name, model) in enumerate(all_models.items(), 1):
            print(f"\n模型进度: {i}/{len(all_models)}")
            try:
                result = self.evaluate_model(
                    model, X_train_scaled, X_val_scaled, X_test_scaled,
                    y_train, y_val, y_test, name
                )
                if result:
                    self.results[name] = result
            except Exception as e:
                print(f"❌ 模型 {name} 评估失败: {str(e)}")
                continue
        
        print(f"\n✅ 成功评估了 {len(self.results)} 个模型")
        
        # 6. 生成报告
        if self.results:
            self.generate_optimized_report()
            self.save_optimized_results()
        
        return self.results
    
    def generate_optimized_report(self):
        """生成优化实验报告"""
        print("\n📊 生成90特征优化集成学习报告...")
        
        # 创建结果DataFrame
        comparison_data = []
        for name, result in self.results.items():
            # 确定模型类型
            if 'Bagging' in name:
                method_type = 'Bagging'
            elif 'Voting' in name:
                method_type = 'Voting'
            elif 'Stacking' in name:
                method_type = 'Stacking'
            elif any(base_type in name for base_type in ['Linear', 'Ridge', 'Lasso', 'KNN', 'Neural', 'SVR']):
                method_type = 'Base Model'
            else:
                method_type = 'Other'
            
            row = {
                'Model': name,
                'Type': method_type,
                'Train_RMSE': result['train_metrics']['RMSE'],
                'Val_RMSE': result['val_metrics']['RMSE'],
                'Test_RMSE': result['test_metrics']['RMSE'],
                'Test_MAE': result['test_metrics']['MAE'],
                'Test_R²': result['test_metrics']['R²'],
                'Test_MAPE': result['test_metrics']['MAPE'],
                'Test_SMAPE': result['test_metrics']['SMAPE'],
                'Training_Time': result['training_time']
            }
            comparison_data.append(row)
        
        results_df = pd.DataFrame(comparison_data)
        results_df = results_df.sort_values('Test_RMSE')
        
        print("\n📈 90特征优化集成学习结果 (按测试集RMSE排序):")
        print("="*120)
        print(results_df.round(4).to_string(index=False))
        
        # 保存结果表格
        results_df.to_csv('optimized_90f_ensemble_results.csv', index=False)
        
        # 最佳模型分析
        best_model = results_df.iloc[0]
        print(f"\n🏆 最佳90特征优化模型:")
        print(f"  模型: {best_model['Model']}")
        print(f"  类型: {best_model['Type']}")
        print(f"  测试集RMSE: {best_model['Test_RMSE']:.2f}")
        print(f"  测试集R²: {best_model['Test_R²']:.4f}")
        print(f"  测试集MAPE: {best_model['Test_MAPE']:.2f}%")
        print(f"  训练时间: {best_model['Training_Time']:.2f}秒")
        
        # 按类型分析
        print(f"\n📊 按集成类型分析 (90特征优化):")
        for method_type in ['Base Model', 'Bagging', 'Voting', 'Stacking']:
            type_results = results_df[results_df['Type'] == method_type]
            if not type_results.empty:
                best_in_type = type_results.iloc[0]
                avg_rmse = type_results['Test_RMSE'].mean()
                print(f"  {method_type}:")
                print(f"    最佳: {best_in_type['Model']} (RMSE: {best_in_type['Test_RMSE']:.2f})")
                print(f"    平均RMSE: {avg_rmse:.2f}")
        
        # 目标达成情况
        target_rmse = 200
        target_r2 = 0.75
        
        print(f"\n🎯 90特征模型目标达成情况:")
        successful_models = 0
        for _, row in results_df.iterrows():
            rmse_ok = row['Test_RMSE'] < target_rmse
            r2_ok = row['Test_R²'] > target_r2
            if rmse_ok and r2_ok:
                successful_models += 1
                print(f"  ✅ {row['Model']}: RMSE={row['Test_RMSE']:.2f}, R²={row['Test_R²']:.3f}")
            elif rmse_ok:
                print(f"  🟡 {row['Model']}: RMSE达标({row['Test_RMSE']:.2f}), R²={row['Test_R²']:.3f}")
        
        print(f"\n📈 完全达标模型: {successful_models}/{len(results_df)} ({successful_models/len(results_df)*100:.1f}%)")
        
        return results_df
    
    def save_optimized_results(self):
        """保存优化实验结果"""
        # 保存90特征优化总结
        optimized_summary = {}
        for name, result in self.results.items():
            optimized_summary[name] = {
                'test_rmse': result['test_metrics']['RMSE'],
                'test_r2': result['test_metrics']['R²'],
                'test_mae': result['test_metrics']['MAE'],
                'test_mape': result['test_metrics']['MAPE'],
                'test_smape': result['test_metrics']['SMAPE'],
                'training_time': result['training_time']
            }
        
        # 保存为JSON
        import json
        with open('optimized_90f_ensemble_summary.json', 'w', encoding='utf-8') as f:
            json.dump(optimized_summary, f, indent=2, ensure_ascii=False)
        
        print("💾 90特征优化实验结果已保存:")
        print("  - optimized_90f_ensemble_results.csv")
        print("  - optimized_90f_ensemble_summary.json")

def main():
    """主函数"""
    print("=" * 80)
    print("🚀 基于90特征的优化集成学习实验")
    print("=" * 80)
    print("针对数据预处理结果的优化:")
    print("✅ 90个增强特征（包含滞后特征）")
    print("✅ 8465条有效记录（排除295条非运营日）")
    print("✅ 152个异常值已处理（winsorization）")
    print("✅ 针对特征数量优化的模型参数")
    print("=" * 80)
    
    # 创建实验实例
    experiment = OptimizedEnsembleExperiment()
    
    # 运行优化实验
    results = experiment.run_optimized_experiment()
    
    if results:
        print(f"\n🎉 实验完成！成功测试了 {len(results)} 个优化集成模型")
        
        # 输出关键洞察
        print("\n💡 90特征优化关键洞察:")
        print("- 基于四种核心算法：线性回归、KNN、神经网络、SVR")
        print("- 神经网络层级设计考虑特征维度 (180-90-45)")
        print("- KNN使用距离加权避免维度灾难")
        print("- SVR采用多种核函数（线性、RBF、多项式）")
        print("- Stacking元学习器针对9个基学习器输出优化")
        print("- 考虑了5925个训练样本的模型复杂度平衡")
        
    else:
        print("\n❌ 实验失败，请检查数据和代码")
    
    print("\n=" * 80)

if __name__ == "__main__":
    main() 