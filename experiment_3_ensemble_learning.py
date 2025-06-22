#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验3：集成学习方法比较 (成员C)
CDS503 Group Project - 首尔自行车需求预测

目标：比较不同集成学习方法的性能
方法：Bagging、Boosting、Voting、Stacking
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import (
    RandomForestRegressor, BaggingRegressor, AdaBoostRegressor,
    GradientBoostingRegressor, VotingRegressor, ExtraTreesRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import time
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import BikeDataPreprocessor

# 设置中文字体 (跨平台通用)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class OptimalEnsembleExperiment:
    """基于R²最优模型的集成学习实验类"""
    
    def __init__(self):
        self.results = {}
        self.preprocessor = BikeDataPreprocessor()
        # 目标变量归一化器
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        self.target_scaler = MinMaxScaler()  # 使用MinMax归一化，保持值在[0,1]范围内
        
    def get_optimal_models(self):
        """获取R²最优的三个模型"""
        
        # 基于之前实验结果，R²最优的三个模型：
        # 1. Neural Network (R²: 0.6786)
        # 2. XGBoost (R²: 0.6507)  
        # 3. Stacking Neural Network (R²: 0.6364)
        
        optimal_models = {
            'Optimal_Neural_Network': MLPRegressor(
                # 网络架构：针对87个增强特征优化
                hidden_layer_sizes=(256, 128, 64, 32),  # 加深网络以处理更多特征
                
                # 激活函数和求解器
                activation='relu',
                solver='adam',
                
                # 学习率参数
                learning_rate='adaptive',
                learning_rate_init=0.0005,  # 更小的学习率
                
                # 正则化
                alpha=0.0001,  # 更小的正则化，让模型学到更多
                
                # 训练参数
                max_iter=8000,  # 更多迭代
                tol=1e-7,
                
                # 早停策略
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=100,  # 更多耐心
                
                # 其他参数
                batch_size='auto',
                shuffle=True,
                random_state=42
            ),
            
            'Optimal_XGBoost': xgb.XGBRegressor(
                # 针对87个特征和归一化目标优化
                n_estimators=1200,  # 更多树
                learning_rate=0.02,  # 更小学习率
                max_depth=15,  # 更深的树
                min_child_weight=2,
                subsample=0.9,
                colsample_bytree=0.85,
                colsample_bylevel=0.85,
                colsample_bynode=0.85,
                reg_alpha=0.01,
                reg_lambda=0.1,
                gamma=0.05,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            ),
            
            'Optimal_Gradient_Boosting': GradientBoostingRegressor(
                # 基于Gradient Boosting优化（作为Stacking的替代）
                n_estimators=1000,
                learning_rate=0.03,
                max_depth=12,
                subsample=0.9,
                min_samples_split=3,
                min_samples_leaf=2,
                max_features='sqrt',  # 使用sqrt(87) ≈ 9个特征
                random_state=42
            )
        }
        
        return optimal_models
    
    def create_ensemble_from_optimal(self, X_train, y_train, X_val, y_val):
        """基于最优三个模型创建集成"""
        
        print("🏆 训练R²最优的三个模型...")
        
        # 获取最优模型
        optimal_models = self.get_optimal_models()
        trained_models = {}
        
        # 训练每个最优模型
        for name, model in optimal_models.items():
            print(f"  训练 {name}...")
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # 验证性能
            val_pred = model.predict(X_val)
            val_r2 = r2_score(y_val, val_pred)
            print(f"    验证集R²: {val_r2:.4f}, 训练时间: {training_time:.2f}秒")
            
            trained_models[name] = model
        
        # 创建集成方法
        ensemble_models = {}
        
        # 1. 简单平均集成
        class SimpleAverageEnsemble:
            def __init__(self, models):
                self.models = models
            
            def predict(self, X):
                predictions = np.array([model.predict(X) for model in self.models.values()])
                return np.mean(predictions, axis=0)
            
            def fit(self, X, y):
                pass  # 已经训练好了
        
        ensemble_models['Average_Ensemble'] = SimpleAverageEnsemble(trained_models)
        
        # 2. 加权平均集成（基于验证集R²）
        class WeightedAverageEnsemble:
            def __init__(self, models, weights):
                self.models = models
                self.weights = weights
            
            def predict(self, X):
                predictions = np.array([model.predict(X) for model in self.models.values()])
                weighted_pred = np.average(predictions, axis=0, weights=self.weights)
                return weighted_pred
            
            def fit(self, X, y):
                pass
        
        # 计算权重（基于验证集R²）
        val_r2_scores = []
        for model in trained_models.values():
            val_pred = model.predict(X_val)
            val_r2 = r2_score(y_val, val_pred)
            val_r2_scores.append(max(val_r2, 0))  # 确保非负
        
        # 归一化权重
        weights = np.array(val_r2_scores)
        weights = weights / np.sum(weights)
        print(f"  集成权重: {dict(zip(trained_models.keys(), weights))}")
        
        ensemble_models['Weighted_Ensemble'] = WeightedAverageEnsemble(trained_models, weights)
        
        # 3. Voting Regressor
        from sklearn.ensemble import VotingRegressor
        voting_estimators = [(name, model) for name, model in trained_models.items()]
        ensemble_models['Voting_Ensemble'] = VotingRegressor(
            estimators=voting_estimators,
            weights=weights
        )
        
        # 训练Voting集成
        ensemble_models['Voting_Ensemble'].fit(X_train, y_train)
        
        # 返回所有模型（个体+集成）
        all_models = {**trained_models, **ensemble_models}
        return all_models
    
    def calculate_metrics(self, y_true, y_pred):
        """计算全面的回归指标（与主类相同）"""
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
        
        # MSLE (Mean Squared Log Error) - 需要确保值为正
        try:
            if np.all(y_true >= 0) and np.all(y_pred >= 0):
                msle = mean_squared_log_error(y_true_safe, y_pred_safe)
            else:
                msle = np.nan
        except:
            msle = np.nan
        
        # Explained Variance Score
        evs = explained_variance_score(y_true, y_pred)
        
        # Mean Absolute Scaled Error (MASE) - 相对于naive forecast
        naive_error = np.mean(np.abs(np.diff(y_true)))
        if naive_error > 0:
            mase = mae / naive_error
        else:
            mase = np.nan
        
        # Symmetric Mean Absolute Percentage Error (SMAPE)
        smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
        
        # Median Absolute Error
        medae = np.median(np.abs(y_true - y_pred))
        
        # Max Error
        max_error = np.max(np.abs(y_true - y_pred))
        
        # Mean Absolute Error in %
        mae_percent = (mae / np.mean(y_true)) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape,
            'MSLE': msle,
            'EVS': evs,  # Explained Variance Score
            'MASE': mase,  # Mean Absolute Scaled Error
            'SMAPE': smape,  # Symmetric MAPE
            'MedAE': medae,  # Median Absolute Error
            'MaxError': max_error,  # Maximum Error
            'MAE%': mae_percent  # MAE as percentage of mean
        }
    
    def evaluate_model(self, model, X_train, X_val, X_test, y_train, y_val, y_test, name):
        """评估单个模型（包含归一化处理）"""
        print(f"\n=== 评估 {name} ===")
        
        start_time = time.time()
        
        # 如果模型还没有训练，则训练它
        if hasattr(model, 'fit') and not hasattr(model, 'models'):
            model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # 预测（归一化的目标值）
        try:
            y_train_pred_norm = model.predict(X_train)
            y_val_pred_norm = model.predict(X_val)
            y_test_pred_norm = model.predict(X_test)
            
            # 反归一化预测值
            y_train_pred = self.target_scaler.inverse_transform(y_train_pred_norm.reshape(-1, 1)).flatten()
            y_val_pred = self.target_scaler.inverse_transform(y_val_pred_norm.reshape(-1, 1)).flatten()
            y_test_pred = self.target_scaler.inverse_transform(y_test_pred_norm.reshape(-1, 1)).flatten()
            
            # 反归一化真实值
            y_train_true = self.target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
            y_val_true = self.target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
            y_test_true = self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            
        except Exception as e:
            print(f"❌ {name} 预测失败: {str(e)}")
            return None
        
        # 计算指标（基于原始尺度）
        train_metrics = self.calculate_metrics(y_train_true, y_train_pred)
        val_metrics = self.calculate_metrics(y_val_true, y_val_pred)
        test_metrics = self.calculate_metrics(y_test_true, y_test_pred)
        
        print(f"训练时间: {training_time:.2f}秒")
        print(f"验证集 RMSE: {val_metrics['RMSE']:.2f}")
        print(f"测试集 RMSE: {test_metrics['RMSE']:.2f}")
        print(f"测试集 R²: {test_metrics['R²']:.4f}")
        print(f"测试集 MAPE: {test_metrics['MAPE']:.2f}%")
        print(f"测试集 SMAPE: {test_metrics['SMAPE']:.2f}%")
        
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
    
    def run_optimal_experiment(self):
        """运行基于R²最优模型的集成学习实验"""
        print("🎯 开始基于R²最优模型的集成学习实验")
        print("="*60)
        
        # 1. 数据预处理
        print("1. 增强数据预处理...")
        df = self.preprocessor.load_data()
        df_processed = self.preprocessor.prepare_features(df, use_lag_features=False, exclude_non_operating=True)
        
        # 2. 数据分割
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_data_temporal(df_processed)
        
        # 3. 目标变量归一化
        print("2. 目标变量归一化...")
        y_train_norm = self.target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_val_norm = self.target_scaler.transform(y_val.values.reshape(-1, 1)).flatten()
        y_test_norm = self.target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
        
        print(f"目标变量范围: {y_train.min():.0f} - {y_train.max():.0f}")
        print(f"归一化后范围: {y_train_norm.min():.3f} - {y_train_norm.max():.3f}")
        
        # 4. 特征标准化
        X_train_scaled, X_val_scaled, X_test_scaled = self.preprocessor.scale_features(X_train, X_val, X_test)
        
        print(f"\n数据概览:")
        print(f"  特征数量: {X_train_scaled.shape[1]}")
        print(f"  训练样本: {X_train_scaled.shape[0]}")
        print(f"  验证样本: {X_val_scaled.shape[0]}")
        print(f"  测试样本: {X_test_scaled.shape[0]}")
        
        # 5. 创建和训练最优模型集成
        print("\n3. 创建R²最优模型集成...")
        all_models = self.create_ensemble_from_optimal(
            X_train_scaled, y_train_norm, X_val_scaled, y_val_norm
        )
        
        # 6. 评估所有模型
        print(f"\n4. 评估 {len(all_models)} 个优化模型...")
        for i, (name, model) in enumerate(all_models.items(), 1):
            print(f"\n模型进度: {i}/{len(all_models)}")
            try:
                result = self.evaluate_model(
                    model, X_train_scaled, X_val_scaled, X_test_scaled,
                    y_train_norm, y_val_norm, y_test_norm, name
                )
                if result:
                    self.results[name] = result
            except Exception as e:
                print(f"❌ 模型 {name} 评估失败: {str(e)}")
                continue
        
        # 7. 生成报告
        if self.results:
            print(f"\n✅ 成功评估了 {len(self.results)} 个模型")
            self.generate_optimal_report()
            self.save_optimal_results()
        else:
            print("❌ 没有成功评估的模型")
        
        return self.results
    
    def generate_optimal_report(self):
        """生成优化模型比较报告"""
        print("\n📊 生成优化模型比较报告...")
        
        # 创建结果DataFrame
        comparison_data = []
        for name, result in self.results.items():
            row = {
                'Model': name,
                'Train_RMSE': result['train_metrics']['RMSE'],
                'Val_RMSE': result['val_metrics']['RMSE'],
                'Test_RMSE': result['test_metrics']['RMSE'],
                'Test_MAE': result['test_metrics']['MAE'],
                'Test_R²': result['test_metrics']['R²'],
                'Test_MAPE': result['test_metrics']['MAPE'],
                'Test_SMAPE': result['test_metrics']['SMAPE'],
                'Test_MSLE': result['test_metrics']['MSLE'],
                'Test_EVS': result['test_metrics']['EVS'],
                'Test_MASE': result['test_metrics']['MASE'],
                'Test_MedAE': result['test_metrics']['MedAE'],
                'Test_MaxError': result['test_metrics']['MaxError'],
                'Test_MAE%': result['test_metrics']['MAE%'],
                'Training_Time': result['training_time']
            }
            comparison_data.append(row)
        
        results_df = pd.DataFrame(comparison_data)
        results_df = results_df.sort_values('Test_R²', ascending=False)  # 按R²排序
        
        print("\n📈 优化模型性能比较 (按R²排序):")
        print("="*120)
        
        # 显示主要指标
        main_cols = ['Model', 'Test_RMSE', 'Test_MAE', 'Test_R²', 'Test_MAPE', 'Test_SMAPE', 'Training_Time']
        print(results_df[main_cols].round(4).to_string(index=False))
        
        print("\n📊 详细回归指标:")
        print("="*120)
        detail_cols = ['Model', 'Test_MSLE', 'Test_EVS', 'Test_MASE', 'Test_MedAE', 'Test_MaxError', 'Test_MAE%']
        detail_df = results_df[detail_cols].round(4)
        print(detail_df.to_string(index=False))
        
        # 保存结果表格
        results_df.to_csv('optimal_ensemble_results.csv', index=False)
        
        # 最佳模型分析
        best_model = results_df.iloc[0]
        print(f"\n🏆 最佳优化模型:")
        print(f"  模型: {best_model['Model']}")
        print(f"  测试集R²: {best_model['Test_R²']:.4f}")
        print(f"  测试集RMSE: {best_model['Test_RMSE']:.2f}")
        print(f"  测试集MAPE: {best_model['Test_MAPE']:.2f}%")
        print(f"  测试集SMAPE: {best_model['Test_SMAPE']:.2f}%")
        print(f"  训练时间: {best_model['Training_Time']:.2f}秒")
        
        # 目标达成情况
        target_rmse = 200
        target_r2 = 0.75
        
        print(f"\n🎯 目标达成情况:")
        successful_models = 0
        for _, row in results_df.iterrows():
            rmse_ok = row['Test_RMSE'] < target_rmse
            r2_ok = row['Test_R²'] > target_r2
            if rmse_ok and r2_ok:
                successful_models += 1
                print(f"  ✅ {row['Model']}: RMSE={row['Test_RMSE']:.2f}, R²={row['Test_R²']:.3f}")
            elif rmse_ok:
                print(f"  🟡 {row['Model']}: RMSE达标({row['Test_RMSE']:.2f}), R²={row['Test_R²']:.3f}")
            elif r2_ok:
                print(f"  🟡 {row['Model']}: R²达标({row['Test_R²']:.3f}), RMSE={row['Test_RMSE']:.2f}")
        
        print(f"\n📈 完全达标模型: {successful_models}/{len(results_df)} ({successful_models/len(results_df)*100:.1f}%)")
        
        return results_df
    
    def save_optimal_results(self):
        """保存优化实验结果"""
        # 保存优化模型总结
        optimal_summary = {}
        for name, result in self.results.items():
            optimal_summary[name] = {
                'test_rmse': result['test_metrics']['RMSE'],
                'test_r2': result['test_metrics']['R²'],
                'test_mae': result['test_metrics']['MAE'],
                'test_mape': result['test_metrics']['MAPE'],
                'test_smape': result['test_metrics']['SMAPE'],
                'test_msle': result['test_metrics']['MSLE'],
                'test_evs': result['test_metrics']['EVS'],
                'test_mase': result['test_metrics']['MASE'],
                'training_time': result['training_time']
            }
        
        # 保存为JSON
        import json
        with open('optimal_ensemble_summary.json', 'w', encoding='utf-8') as f:
            json.dump(optimal_summary, f, indent=2, ensure_ascii=False)
        
        print("💾 优化实验结果已保存:")
        print("  - optimal_ensemble_results.csv")
        print("  - optimal_ensemble_summary.json")

class EnsembleLearningExperiment:
    """原始集成学习实验类（保持兼容性）"""
    
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
        
        # MSLE (Mean Squared Log Error) - 需要确保值为正
        try:
            if np.all(y_true >= 0) and np.all(y_pred >= 0):
                msle = mean_squared_log_error(y_true_safe, y_pred_safe)
            else:
                msle = np.nan
        except:
            msle = np.nan
        
        # Explained Variance Score
        evs = explained_variance_score(y_true, y_pred)
        
        # Mean Absolute Scaled Error (MASE) - 相对于naive forecast
        naive_error = np.mean(np.abs(np.diff(y_true)))
        if naive_error > 0:
            mase = mae / naive_error
        else:
            mase = np.nan
        
        # Symmetric Mean Absolute Percentage Error (SMAPE)
        smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
        
        # Median Absolute Error
        medae = np.median(np.abs(y_true - y_pred))
        
        # Max Error
        max_error = np.max(np.abs(y_true - y_pred))
        
        # Mean Absolute Error in %
        mae_percent = (mae / np.mean(y_true)) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape,
            'MSLE': msle,
            'EVS': evs,  # Explained Variance Score
            'MASE': mase,  # Mean Absolute Scaled Error
            'SMAPE': smape,  # Symmetric MAPE
            'MedAE': medae,  # Median Absolute Error
            'MaxError': max_error,  # Maximum Error
            'MAE%': mae_percent  # MAE as percentage of mean
        }
    
    def create_base_models(self):
        """创建基础模型（针对特征精细优化版本）"""
        # 特征数量为33，据此优化模型参数
        base_models = {
            'Linear Regression': LinearRegression(),
            
            'Ridge Regression': Ridge(
                alpha=0.05,  # 针对33个特征的适中正则化
                solver='saga',  # 适合中等规模数据
                max_iter=3000
            ),
            
            'Lasso Regression': Lasso(
                alpha=0.005,  # 更小的正则化以保留更多特征
                max_iter=3000,
                selection='random',  # 随机特征选择
                tol=1e-5
            ),
            
            'Random Forest': RandomForestRegressor(
                n_estimators=800,  # 增加树的数量
                max_depth=None,  # 不限制深度，让树充分生长
                min_samples_split=3,  # 针对6132个训练样本
                min_samples_leaf=2,
                max_features=int(33**0.5),  # sqrt(33) ≈ 6个特征
                bootstrap=True,
                oob_score=True,  # 使用out-of-bag评分
                random_state=42, 
                n_jobs=-1
            ),
            
            'XGBoost': xgb.XGBRegressor(
                n_estimators=800,
                learning_rate=0.03,  # 更小的学习率配合更多估计器
                max_depth=12,  # 适中深度
                min_child_weight=3,  # 防止过拟合
                subsample=0.85,  # 行采样
                colsample_bytree=0.8,  # 特征采样
                colsample_bylevel=0.9,
                colsample_bynode=0.9,
                reg_alpha=0.05,  # L1正则化
                reg_lambda=1.2,  # L2正则化
                gamma=0.1,  # 最小分裂损失
                random_state=42, 
                n_jobs=-1, 
                verbosity=0
            ),
            
            'SVR': SVR(
                kernel='rbf',
                C=50.0,  # 增加复杂度容忍度
                gamma='scale',  # 自动缩放
                epsilon=0.005,  # 更小的epsilon管道
                tol=1e-4,  # 收敛容忍度
                cache_size=500  # 增加缓存
            ),
            
            'Neural Network': MLPRegressor(
                # 网络架构：针对33个特征设计
                hidden_layer_sizes=(128, 64, 32, 16),  # 4层递减架构
                
                # 激活函数和求解器
                activation='relu',  # ReLU激活函数
                solver='adam',  # Adam优化器
                
                # 学习率参数
                learning_rate='adaptive',  # 自适应学习率
                learning_rate_init=0.001,  # 初始学习率
                
                # 正则化
                alpha=0.001,  # L2正则化
                
                # 训练参数
                max_iter=5000,  # 更多迭代次数
                tol=1e-6,  # 更严格的收敛条件
                
                # 早停策略
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=50,  # 50次迭代无改进则停止
                
                # 批处理
                batch_size='auto',  # 自动批大小
                
                # 其他参数
                shuffle=True,  # 每轮训练打乱数据
                random_state=42,
                warm_start=False
            )
        }
        return base_models
    
    def bagging_methods(self):
        """Bagging方法"""
        print("🎯 测试Bagging方法...")
        
        bagging_models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=1000,  # 更多树
                max_depth=None,  # 不限制深度
                min_samples_split=3,
                min_samples_leaf=2,
                max_features=int(33**0.5),  # 针对33个特征优化
                bootstrap=True,
                oob_score=True,
                random_state=42,
                n_jobs=-1
            ),
            
            'Extra Trees': ExtraTreesRegressor(
                n_estimators=1000,  # 更多树
                max_depth=None,  # 不限制深度
                min_samples_split=3,
                min_samples_leaf=2,
                max_features=int(33**0.5),  # 针对33个特征优化
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            ),
            
            'Bagging (Linear)': BaggingRegressor(
                estimator=LinearRegression(),
                n_estimators=100,  # 增加基学习器数量
                max_samples=0.75,  # 采样比例
                max_features=0.85,  # 特征采样比例（针对33个特征）
                bootstrap=True,
                bootstrap_features=True,
                random_state=42,
                n_jobs=-1
            ),
            
            'Bagging (Ridge)': BaggingRegressor(
                estimator=Ridge(alpha=0.05, solver='auto', max_iter=5000),
                n_estimators=100,
                max_samples=0.75,
                max_features=0.85,
                bootstrap=True,
                bootstrap_features=True,
                random_state=42,
                n_jobs=-1
            ),
            
            'Bagging (SVR)': BaggingRegressor(
                estimator=SVR(
                    kernel='rbf', 
                    C=20.0, 
                    gamma='scale',
                    epsilon=0.01,
                    cache_size=300
                ),
                n_estimators=50,  # SVR计算量大，适度数量
                max_samples=0.7,
                max_features=0.9,  # SVR对特征敏感，保留更多特征
                random_state=42,
                n_jobs=-1
            ),
            
            'Bagging (Neural Network)': BaggingRegressor(
                estimator=MLPRegressor(
                    hidden_layer_sizes=(64, 32),  # 相对简单的网络
                    activation='relu',
                    solver='adam',
                    learning_rate='adaptive',
                    learning_rate_init=0.002,
                    alpha=0.01,
                    max_iter=2000,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=30,
                    random_state=42
                ),
                n_estimators=30,  # 神经网络组合数量
                max_samples=0.8,
                max_features=0.9,
                random_state=42,
                n_jobs=1  # 神经网络避免嵌套并行
            )
        }
        
        return bagging_models
    
    def boosting_methods(self):
        """Boosting方法"""
        print("🚀 测试Boosting方法...")
        
        boosting_models = {
            'AdaBoost (Linear)': AdaBoostRegressor(
                estimator=LinearRegression(),
                n_estimators=200,  # 大幅增加
                learning_rate=0.8,
                random_state=42
            ),
            'AdaBoost (Ridge)': AdaBoostRegressor(
                estimator=Ridge(alpha=0.1),
                n_estimators=200,  # 大幅增加
                learning_rate=0.8,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=500,  # 大幅增加
                learning_rate=0.05,
                max_depth=8,
                subsample=0.9,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=500,  # 大幅增加
                learning_rate=0.05,
                max_depth=10,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            ),
            'XGBoost Tuned': xgb.XGBRegressor(
                n_estimators=1000,  # 大幅增加
                learning_rate=0.03,
                max_depth=12,
                subsample=0.95,
                colsample_bytree=0.95,
                colsample_bylevel=0.9,
                reg_alpha=0.05,
                reg_lambda=1.5,
                gamma=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            ),
            'LightGBM': xgb.XGBRegressor(  # 添加另一个高性能模型
                n_estimators=800,
                learning_rate=0.04,
                max_depth=11,
                subsample=0.92,
                colsample_bytree=0.92,
                reg_alpha=0.08,
                reg_lambda=1.2,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        }
        
        return boosting_models
    
    def voting_methods(self):
        """Voting方法"""
        print("🗳️ 测试Voting方法...")
        
        # 创建基础模型集合1（线性模型）
        linear_models = [
            ('linear', LinearRegression()),
            ('ridge', Ridge(alpha=1.0, solver='auto', max_iter=5000)),
            ('lasso', Lasso(alpha=0.1, max_iter=5000))
        ]
        
        # 创建基础模型集合2（高性能树模型）
        tree_models = [
            ('rf', RandomForestRegressor(n_estimators=300, max_depth=25, random_state=42, n_jobs=-1)),
            ('xgb', xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=10, random_state=42, n_jobs=-1, verbosity=0)),
            ('et', ExtraTreesRegressor(n_estimators=300, max_depth=25, random_state=42, n_jobs=-1))
        ]
        
        # 创建基础模型集合3（高性能混合模型）
        mixed_models = [
            ('linear', LinearRegression()),
            ('ridge', Ridge(alpha=0.1, solver='auto', max_iter=5000)),
            ('rf', RandomForestRegressor(n_estimators=300, max_depth=25, random_state=42, n_jobs=-1)),
            ('xgb', xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=10, random_state=42, n_jobs=-1, verbosity=0)),
            ('svr', SVR(kernel='rbf', C=10.0, gamma='scale'))
        ]
        
        # 创建基础模型集合4（所有高性能模型）
        all_models = [
            ('linear', LinearRegression()),
            ('ridge', Ridge(alpha=0.1, solver='auto', max_iter=5000)),
            ('lasso', Lasso(alpha=0.01, max_iter=5000)),
            ('rf', RandomForestRegressor(n_estimators=300, max_depth=25, random_state=42, n_jobs=-1)),
            ('xgb', xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=10, random_state=42, n_jobs=-1, verbosity=0)),
            ('et', ExtraTreesRegressor(n_estimators=300, max_depth=25, random_state=42, n_jobs=-1)),
            ('svr', SVR(kernel='rbf', C=10.0, gamma='scale'))
        ]
        
        voting_models = {
            'Voting (Linear Models)': VotingRegressor(
                estimators=linear_models,
                weights=None  # 等权重
            ),
            'Voting (Tree Models)': VotingRegressor(
                estimators=tree_models,
                weights=[1, 1.5, 1.2]  # 给XGBoost更高权重
            ),
            'Voting (Tree Models - Equal)': VotingRegressor(
                estimators=tree_models,
                weights=None  # 等权重
            ),
            'Voting (Mixed - Equal)': VotingRegressor(
                estimators=mixed_models,
                weights=None  # 等权重
            ),
            'Voting (Mixed - Weighted)': VotingRegressor(
                estimators=mixed_models,
                weights=[0.5, 0.8, 2.5, 3.0, 1.2]  # 给树模型更高权重
            ),
            'Voting (All Models - Equal)': VotingRegressor(
                estimators=all_models,
                weights=None  # 等权重
            ),
            'Voting (All Models - Weighted)': VotingRegressor(
                estimators=all_models,
                weights=[0.5, 0.8, 0.6, 2.5, 3.0, 2.2, 1.0]  # 偏向树模型
            ),
            'Voting (Best Models)': VotingRegressor(
                estimators=[
                    ('rf', RandomForestRegressor(n_estimators=400, max_depth=25, random_state=42, n_jobs=-1)),
                    ('xgb', xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=10, random_state=42, n_jobs=-1, verbosity=0)),
                    ('et', ExtraTreesRegressor(n_estimators=400, max_depth=25, random_state=42, n_jobs=-1))
                ],
                weights=[1.2, 1.5, 1.0]  # XGBoost权重最高
            )
        }
        
        return voting_models
    
    def stacking_methods(self, X_train, y_train, X_val, y_val):
        """Stacking方法"""
        print("🏗️ 测试Stacking方法...")
        
        # 第一层模型（针对特征优化的基础学习器）
        base_models = {
            'linear': LinearRegression(),
            
            'ridge': Ridge(
                alpha=0.05, 
                solver='auto', 
                max_iter=5000
            ),
            
            'lasso': Lasso(
                alpha=0.005, 
                max_iter=3000, 
                selection='random'
            ),
            
            'rf': RandomForestRegressor(
                n_estimators=400, 
                max_depth=None, 
                min_samples_split=3,
                min_samples_leaf=2,
                max_features=int(33**0.5),
                oob_score=True,
                random_state=42, 
                n_jobs=-1
            ),
            
            'xgb': xgb.XGBRegressor(
                n_estimators=400, 
                learning_rate=0.03, 
                max_depth=12,
                min_child_weight=3,
                subsample=0.85,
                colsample_bytree=0.8,
                reg_alpha=0.05,
                reg_lambda=1.2,
                random_state=42, 
                n_jobs=-1, 
                verbosity=0
            ),
            
            'et': ExtraTreesRegressor(
                n_estimators=400, 
                max_depth=None,
                min_samples_split=3,
                min_samples_leaf=2,
                max_features=int(33**0.5),
                random_state=42, 
                n_jobs=-1
            ),
            
            'svr': SVR(
                kernel='rbf', 
                C=30.0, 
                gamma='scale',
                epsilon=0.01,
                cache_size=400
            ),
            
            'nn': MLPRegressor(
                hidden_layer_sizes=(96, 48, 24),  # 针对Stacking优化的网络
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                alpha=0.001,
                max_iter=3000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=40,
                random_state=42
            )
        }
        
        # 训练基础模型并获取预测
        base_predictions_train = np.zeros((X_train.shape[0], len(base_models)))
        base_predictions_val = np.zeros((X_val.shape[0], len(base_models)))
        
        for i, (name, model) in enumerate(base_models.items()):
            print(f"  训练基础模型: {name}")
            model.fit(X_train, y_train)
            base_predictions_train[:, i] = model.predict(X_train)
            base_predictions_val[:, i] = model.predict(X_val)
        
        # 第二层模型（针对基学习器输出优化的元学习器）
        # 基学习器有8个输出，针对这个特征数量设计元学习器
        meta_models = {
            'Stacking (Linear)': LinearRegression(),
            
            'Stacking (Ridge)': Ridge(
                alpha=0.1,  # 针对8个基学习器输出的正则化
                solver='auto'
            ),
            
            'Stacking (Lasso)': Lasso(
                alpha=0.02,  # 适中的稀疏化
                max_iter=2000
            ),
            
            'Stacking (RF)': RandomForestRegressor(
                n_estimators=300,
                max_depth=10,  # 较浅的树，避免在少量特征上过拟合
                min_samples_split=5,
                min_samples_leaf=3,
                max_features=3,  # 在8个特征中选择3个
                random_state=42, 
                n_jobs=-1
            ),
            
            'Stacking (XGBoost)': xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.08,  # 相对较高的学习率
                max_depth=6,  # 较浅的树
                min_child_weight=5,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.5,
                random_state=42, 
                n_jobs=-1, 
                verbosity=0
            ),
            
            'Stacking (ExtraTrees)': ExtraTreesRegressor(
                n_estimators=300,
                max_depth=8,  # 控制深度
                min_samples_split=5,
                min_samples_leaf=3,
                max_features=3,  # 针对8个输入特征
                random_state=42, 
                n_jobs=-1
            ),
            
            'Stacking (Neural Network)': MLPRegressor(
                hidden_layer_sizes=(24, 12),  # 针对8个输入的简单网络
                activation='relu',
                solver='adam',
                learning_rate='constant',
                learning_rate_init=0.01,  # 较高的学习率
                alpha=0.01,
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=42
            )
        }
        
        stacking_results = {}
        
        for name, meta_model in meta_models.items():
            print(f"  训练元模型: {name}")
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
    
    def run_experiment(self):
        """运行集成学习实验"""
        print("🚀 开始实验3：集成学习方法比较")
        print("="*50)
        
        # 数据预处理 - 使用增强特征工程
        print("1. 基于数据洞察的增强数据预处理...")
        df = self.preprocessor.load_data()
        print(f"原始数据: {len(df)} 条记录")
        
        # 使用增强特征工程，排除非运营日
        df_processed = self.preprocessor.prepare_features(df, use_lag_features=False, exclude_non_operating=True)
        print(f"处理后数据: {len(df_processed)} 条记录")
        print("增强特征包括: 温度分段、双峰时间模式、舒适度指数、极端天气标识等")
        
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
        
        print("\n2. 评估基础模型（与实验一一致）...")
        
        # 首先评估基础模型作为参考
        base_models = self.create_base_models()
        print(f"评估 {len(base_models)} 个基础模型...")
        
        for i, (name, model) in enumerate(base_models.items(), 1):
            print(f"\n基础模型进度: {i}/{len(base_models)}")
            try:
                result = self.evaluate_model(
                    model, X_train_scaled, X_val_scaled, X_test_scaled,
                    y_train, y_val, y_test, f"Base: {name}"
                )
                if result:
                    self.results[f"Base: {name}"] = result
            except Exception as e:
                print(f"❌ 基础模型 {name} 评估失败: {str(e)}")
                continue
        
        print("\n3. 开始集成学习实验...")
        
        # 测试各种集成方法
        all_models = {}
        
        # Bagging方法
        bagging_models = self.bagging_methods()
        all_models.update(bagging_models)
        
        # Boosting方法
        boosting_models = self.boosting_methods()
        all_models.update(boosting_models)
        
        # Voting方法
        voting_models = self.voting_methods()
        all_models.update(voting_models)
        
        # Stacking方法
        stacking_models = self.stacking_methods(
            X_train_scaled, y_train, X_val_scaled, y_val
        )
        all_models.update(stacking_models)
        
        # 评估所有集成模型
        print(f"\n4. 评估 {len(all_models)} 个集成模型...")
        
        for i, (name, model) in enumerate(all_models.items(), 1):
            print(f"\n集成模型进度: {i}/{len(all_models)}")
            try:
                result = self.evaluate_model(
                    model, X_train_scaled, X_val_scaled, X_test_scaled,
                    y_train, y_val, y_test, f"Ensemble: {name}"
                )
                if result:
                    self.results[f"Ensemble: {name}"] = result
            except Exception as e:
                print(f"❌ 集成模型 {name} 评估失败: {str(e)}")
                continue
        
        print("\n✅ 集成学习实验完成!")
        
        # 生成报告
        self.generate_comparison_report(y_test)
        
        # 保存结果
        self.save_results()
        
        return self.results
    
    def generate_comparison_report(self, y_test):
        """生成集成方法比较报告"""
        print("\n📊 生成集成学习比较报告...")
        
        # 创建结果DataFrame
        comparison_data = []
        for name, result in self.results.items():
            # 确定模型类型
            if name.startswith('Base:'):
                method_type = 'Base Model'
            elif 'Random Forest' in name or 'Extra Trees' in name or 'Bagging' in name:
                method_type = 'Bagging'
            elif 'Boosting' in name or 'XGBoost' in name or 'AdaBoost' in name:
                method_type = 'Boosting'
            elif 'Voting' in name:
                method_type = 'Voting'
            elif 'Stacking' in name:
                method_type = 'Stacking'
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
                'Training_Time': result['training_time']
            }
            comparison_data.append(row)
        
        results_df = pd.DataFrame(comparison_data)
        results_df = results_df.sort_values('Test_RMSE')
        
        print("\n📈 集成学习方法比较 (按测试集RMSE排序):")
        print("="*100)
        print(results_df.round(4).to_string(index=False))
        
        # 保存结果表格
        results_df.to_csv('experiment_3_results.csv', index=False)
        
        # 按类型分析
        print(f"\n📊 按模型类型分析:")
        for method_type in ['Base Model', 'Bagging', 'Boosting', 'Voting', 'Stacking']:
            type_results = results_df[results_df['Type'] == method_type]
            if not type_results.empty:
                best_model = type_results.iloc[0]
                print(f"  {method_type}: 最佳模型 {best_model['Model']}, RMSE={best_model['Test_RMSE']:.2f}")
        
        # 基础模型 vs 集成模型比较
        base_results = results_df[results_df['Type'] == 'Base Model']
        ensemble_results = results_df[results_df['Type'] != 'Base Model']
        
        if not base_results.empty and not ensemble_results.empty:
            best_base = base_results.iloc[0]
            best_ensemble = ensemble_results.iloc[0]
            
            print(f"\n🥇 基础模型 vs 集成模型:")
            print(f"  最佳基础模型: {best_base['Model']} (RMSE: {best_base['Test_RMSE']:.2f})")
            print(f"  最佳集成模型: {best_ensemble['Model']} (RMSE: {best_ensemble['Test_RMSE']:.2f})")
            
            improvement = (best_base['Test_RMSE'] - best_ensemble['Test_RMSE']) / best_base['Test_RMSE'] * 100
            print(f"  集成学习改进: {improvement:.2f}%")
        
        # 创建可视化
        self.create_visualizations(results_df, y_test)
        
        # 性能分析
        print(f"\n🏆 最佳集成模型:")
        best_model = results_df.iloc[0]
        print(f"  模型: {best_model['Model']}")
        print(f"  类型: {best_model['Type']}")
        print(f"  测试集RMSE: {best_model['Test_RMSE']:.2f}")
        print(f"  测试集R²: {best_model['Test_R²']:.4f}")
        print(f"  训练时间: {best_model['Training_Time']:.2f}秒")
        
        # 达到目标检查
        target_rmse = 200
        target_r2 = 0.75
        
        print(f"\n🎯 目标达成情况:")
        successful_models = 0
        for _, row in results_df.iterrows():
            rmse_ok = row['Test_RMSE'] < target_rmse
            r2_ok = row['Test_R²'] > target_r2
            if rmse_ok and r2_ok:
                successful_models += 1
                print(f"  ✅ {row['Model']}: RMSE={row['Test_RMSE']:.2f}, R²={row['Test_R²']:.3f}")
        
        print(f"\n📈 成功达标模型: {successful_models}/{len(results_df)} ({successful_models/len(results_df)*100:.1f}%)")
    
    def create_visualizations(self, results_df, y_test):
        """创建可视化图表"""
        
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 20))
        
        # 1. 各类集成方法性能比较
        plt.subplot(3, 3, 1)
        method_types = results_df['Type'].unique()
        type_rmse = []
        type_labels = []
        
        for method_type in method_types:
            type_data = results_df[results_df['Type'] == method_type]
            avg_rmse = type_data['Test_RMSE'].mean()
            type_rmse.append(avg_rmse)
            type_labels.append(f"{method_type}\n({len(type_data)}个)")
        
        bars = plt.bar(range(len(type_labels)), type_rmse, color='skyblue', alpha=0.7)
        plt.xlabel('集成方法类型')
        plt.ylabel('平均测试集 RMSE')
        plt.title('集成方法类型性能比较')
        plt.xticks(range(len(type_labels)), type_labels)
        
        # 添加数值标签
        for i, v in enumerate(type_rmse):
            plt.text(i, v + 5, f'{v:.1f}', ha='center', va='bottom')
        
        plt.grid(axis='y', alpha=0.3)
        
        # 2. 所有模型RMSE比较
        plt.subplot(3, 3, 2)
        models = results_df['Model'][:10]  # 只显示前10个
        rmse_values = results_df['Test_RMSE'][:10]
        
        bars = plt.bar(range(len(models)), rmse_values, color='lightcoral', alpha=0.7)
        plt.xlabel('模型')
        plt.ylabel('测试集 RMSE')
        plt.title('Top 10 模型性能比较')
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        
        # 添加数值标签
        for i, v in enumerate(rmse_values):
            plt.text(i, v + 5, f'{v:.1f}', ha='center', va='bottom')
        
        plt.grid(axis='y', alpha=0.3)
        
        # 3. R²分数比较
        plt.subplot(3, 3, 3)
        r2_values = results_df['Test_R²'][:10]
        bars = plt.bar(range(len(models)), r2_values, color='lightgreen', alpha=0.7)
        plt.xlabel('模型')
        plt.ylabel('测试集 R²')
        plt.title('Top 10 模型 R²比较')
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        
        # 添加数值标签
        for i, v in enumerate(r2_values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.grid(axis='y', alpha=0.3)
        
        # 4. 训练时间比较
        plt.subplot(3, 3, 4)
        time_values = results_df['Training_Time'][:10]
        bars = plt.bar(range(len(models)), time_values, color='orange', alpha=0.7)
        plt.xlabel('模型')
        plt.ylabel('训练时间 (秒)')
        plt.title('Top 10 模型训练时间比较')
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        
        # 添加数值标签
        for i, v in enumerate(time_values):
            plt.text(i, v + max(time_values)*0.02, f'{v:.1f}', ha='center', va='bottom')
        
        plt.grid(axis='y', alpha=0.3)
        
        # 5. 性能vs复杂度散点图
        plt.subplot(3, 3, 5)
        plt.scatter(results_df['Training_Time'], results_df['Test_RMSE'], 
                   s=100, alpha=0.7, c='purple')
        
        for i, model in enumerate(results_df['Model'][:10]):
            plt.annotate(model, (results_df['Training_Time'].iloc[i], results_df['Test_RMSE'].iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('训练时间 (秒)')
        plt.ylabel('测试集 RMSE')
        plt.title('性能 vs 训练复杂度')
        plt.grid(True, alpha=0.3)
        
        # 6. 最佳模型预测效果
        plt.subplot(3, 3, 6)
        best_model_name = results_df.iloc[0]['Model']
        best_predictions = self.results[best_model_name]['predictions']['y_test_pred']
        
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
        plt.title(f'{best_model_name} - 预测效果')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. 各类型方法的R²分布
        plt.subplot(3, 3, 7)
        type_r2_data = []
        type_labels_clean = []
        
        for method_type in method_types:
            type_data = results_df[results_df['Type'] == method_type]
            type_r2_data.append(type_data['Test_R²'].values)
            type_labels_clean.append(method_type)
        
        plt.boxplot(type_r2_data, labels=type_labels_clean)
        plt.ylabel('测试集 R²')
        plt.title('各集成类型 R²分布')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 8. 误差分布（最佳模型）
        plt.subplot(3, 3, 8)
        errors = y_test.values - best_predictions
        plt.hist(errors, bins=30, alpha=0.7, color='red', edgecolor='black')
        plt.xlabel('预测误差')
        plt.ylabel('频次')
        plt.title(f'{best_model_name} - 误差分布')
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.8)
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        plt.text(0.02, 0.98, f'均值: {errors.mean():.2f}\n标准差: {errors.std():.2f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 9. 方法效率分析
        plt.subplot(3, 3, 9)
        # 计算效率指标（性能/时间）
        efficiency = 1 / (results_df['Test_RMSE'] * results_df['Training_Time'])
        
        bars = plt.bar(range(len(models)), efficiency[:10], color='gold', alpha=0.7)
        plt.xlabel('模型')
        plt.ylabel('效率指标 (1/(RMSE×时间))')
        plt.title('模型效率比较')
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('experiment_3_ensemble_learning.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📸 可视化图表已保存: experiment_3_ensemble_learning.png")
    
    def save_results(self):
        """保存实验结果"""
        # 保存集成学习总结
        ensemble_summary = {}
        for name, result in self.results.items():
            ensemble_summary[name] = {
                'test_rmse': result['test_metrics']['RMSE'],
                'test_r2': result['test_metrics']['R²'],
                'test_mae': result['test_metrics']['MAE'],
                'test_mape': result['test_metrics']['MAPE'],
                'training_time': result['training_time']
            }
        
        # 保存为JSON
        import json
        with open('experiment_3_summary.json', 'w', encoding='utf-8') as f:
            json.dump(ensemble_summary, f, indent=2, ensure_ascii=False)
        
        print("💾 实验结果已保存:")
        print("  - experiment_3_results.csv")
        print("  - experiment_3_summary.json")
        print("  - experiment_3_ensemble_learning.png")

def run_stratified_experiment(X_train, X_val, X_test, y_train, y_val, y_test, 
                             stratum_name="分层"):
    """运行分层实验"""
    print(f"\n🔬 {stratum_name} 分层实验")
    print("-" * 60)
    
    # 检查数据大小
    if len(X_train) < 100:
        print(f"⚠️  数据量过小 ({len(X_train)} 条)，跳过该分层")
        return None
    
    # 创建实验对象
    experiment = EnsembleLearningExperiment()
    
    # 重新初始化结果字典
    experiment.results = {}
    
    # 运行主要集成方法（简化版）
    print("运行主要集成方法...")
    
    try:
        # 选择最有效的几种集成方法
        selected_models = {}
        
        # Bagging: Random Forest
        selected_models['RF Enhanced'] = RandomForestRegressor(
            n_estimators=300, max_depth=None, 
            min_samples_split=3, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )
        
        # Boosting: XGBoost
        selected_models['XGBoost Enhanced'] = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=10,
            subsample=0.85, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=0
        )
        
        # Voting: Best combination
        selected_models['Voting Enhanced'] = VotingRegressor(
            estimators=[
                ('rf', RandomForestRegressor(n_estimators=200, max_depth=25, random_state=42, n_jobs=-1)),
                ('xgb', xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=10, random_state=42, n_jobs=-1, verbosity=0)),
                ('et', ExtraTreesRegressor(n_estimators=200, max_depth=25, random_state=42, n_jobs=-1))
            ],
            weights=[1.2, 1.5, 1.0]
        )
        
        # 评估选定的模型
        for name, model in selected_models.items():
            try:
                result = experiment.evaluate_model(
                    model, X_train, X_val, X_test, y_train, y_val, y_test, name
                )
                if result:
                    experiment.results[name] = result
                    rmse = result['test_metrics']['RMSE']
                    r2 = result['test_metrics']['R²']
                    print(f"  ✅ {name}: RMSE={rmse:.2f}, R²={r2:.3f}")
            except Exception as e:
                print(f"  ❌ {name} 失败: {e}")
        
        # 返回最佳结果
        if experiment.results:
            best_result = min(experiment.results.items(), key=lambda x: x[1]['test_metrics']['RMSE'])
            return {
                'best_model': best_result[0],
                'best_rmse': best_result[1]['test_metrics']['RMSE'],
                'best_r2': best_result[1]['test_metrics']['R²'],
                'results_count': len(experiment.results)
            }
        else:
            return None
            
    except Exception as e:
        print(f"❌ 分层实验失败: {e}")
        return None

def main():
    """基于数据洞察的增强实验三"""
    print("=" * 80)
    print("🚀 实验三：基于数据洞察的增强集成学习")
    print("=" * 80)
    
    # 创建实验实例
    experiment = EnsembleLearningExperiment()
    
    # 运行主实验
    print("📊 运行全数据集集成学习实验...")
    results = experiment.run_experiment()
    
    print(f"\n✅ 主实验完成！测试了 {len(results)} 个集成模型")
    
    # 分层建模实验（可选）
    print("\n" + "=" * 80)
    print("🎯 分层建模实验（基于数据洞察）")
    print("=" * 80)
    
    try:
        # 准备预处理器
        preprocessor = BikeDataPreprocessor()
        df = preprocessor.load_data()
        df_processed = preprocessor.prepare_features(df, use_lag_features=False, exclude_non_operating=True)
        
        # 准备分层数据
        stratified_data = preprocessor.prepare_stratified_data(df_processed)
        
        stratified_results = {}
        
        # 按季节分层实验
        print("\n📅 按季节分层实验")
        for season in ['summer', 'winter', 'spring', 'autumn']:
            if f'season_{season}' in stratified_data:
                season_df = stratified_data[f'season_{season}']
                if len(season_df) > 500:  # 确保有足够数据
                    try:
                        # 特征工程
                        season_processed = preprocessor.prepare_features(
                            season_df, use_lag_features=False, exclude_non_operating=False
                        )
                        
                        # 分割数据
                        X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s = \
                            preprocessor.split_data_temporal(season_processed)
                        
                        # 标准化
                        X_train_s_scaled, X_val_s_scaled, X_test_s_scaled = \
                            preprocessor.scale_features(X_train_s, X_val_s, X_test_s)
                        
                        # 运行实验
                        season_results = run_stratified_experiment(
                            X_train_s_scaled, X_val_s_scaled, X_test_s_scaled,
                            y_train_s, y_val_s, y_test_s, 
                            f"{season.title()}季节"
                        )
                        
                        if season_results:
                            stratified_results[f'season_{season}'] = season_results
                    except Exception as e:
                        print(f"❌ {season.title()}季节实验失败: {e}")
                else:
                    print(f"⚠️  {season.title()}季节数据量不足 ({len(season_df)} 条)")
        
        # 按天气条件分层实验
        print("\n🌤️ 按天气条件分层实验")
        for weather_type in ['good_weather', 'bad_weather']:
            if weather_type in stratified_data:
                weather_df = stratified_data[weather_type]
                if len(weather_df) > 300:
                    try:
                        # 特征工程
                        weather_processed = preprocessor.prepare_features(
                            weather_df, use_lag_features=False, exclude_non_operating=False
                        )
                        
                        # 分割数据
                        X_train_w, X_val_w, X_test_w, y_train_w, y_val_w, y_test_w = \
                            preprocessor.split_data_temporal(weather_processed)
                        
                        # 标准化
                        X_train_w_scaled, X_val_w_scaled, X_test_w_scaled = \
                            preprocessor.scale_features(X_train_w, X_val_w, X_test_w)
                        
                        # 运行实验
                        weather_name = "好天气" if weather_type == "good_weather" else "恶劣天气"
                        weather_results = run_stratified_experiment(
                            X_train_w_scaled, X_val_w_scaled, X_test_w_scaled,
                            y_train_w, y_val_w, y_test_w, 
                            weather_name
                        )
                        
                        if weather_results:
                            stratified_results[weather_type] = weather_results
                    except Exception as e:
                        print(f"❌ {weather_name}实验失败: {e}")
                else:
                    print(f"⚠️  {weather_type}数据量不足 ({len(weather_df)} 条)")
        
        # 分层结果总结
        if stratified_results:
            print("\n📊 分层建模结果总结:")
            for stratum, result in stratified_results.items():
                print(f"  {stratum}: {result['best_model']} - RMSE={result['best_rmse']:.2f}, R²={result['best_r2']:.3f}")
        else:
            print("\n⚠️  分层建模实验未产生有效结果")
    
    except Exception as e:
        print(f"⚠️  分层建模实验失败: {e}")
        print("继续使用全数据集结果...")
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 实验总结")
    print("=" * 80)
    print("基于数据洞察的特征工程改进:")
    print("✅ 排除非运营日数据 (295条)")
    print("✅ 温度分段特征 (<0, 0-10, 10-20, 20-30, >30°C)")
    print("✅ 双峰时间模式识别 (8am, 6pm峰值)")
    print("✅ 温度×季节交互特征")
    print("✅ 舒适度指数 (温度+湿度组合)")
    print("✅ 降水阈值特征 (有/无降水)")
    print("✅ 极端天气条件处理")
    print("✅ 分层建模策略 (按季节/天气条件)")
    
    # 输出最终总结
    print("\n📋 集成学习方法总结:")
    print("- 测试了Bagging、Boosting、Voting、Stacking四类集成方法")
    print("- 基于数据分析优化了模型参数")
    print("- 重点关注温度、小时、露点温度三个强相关特征")
    print("- 实施了分层建模策略")
    print("- 处理了极端天气条件的特殊情况")
    
    print("\n🎉 实验三完成！")
    print("=" * 80)

def run_optimal_experiment_main():
    """运行基于R²最优模型的实验"""
    print("🏆 启动R²最优模型集成学习实验")
    print("="*60)
    
    # 运行优化实验
    optimal_experiment = OptimalEnsembleExperiment()
    results = optimal_experiment.run_optimal_experiment()
    
    if results:
        print(f"\n✅ 优化实验完成！成功测试了 {len(results)} 个优化模型")
        print("📊 结果已保存到 optimal_ensemble_results.csv 和 optimal_ensemble_summary.json")
    else:
        print("\n❌ 优化实验失败，请检查数据和代码")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'optimal':
        # 运行优化实验
        run_optimal_experiment_main()
    else:
        # 运行常规实验
        main()
        print("\n💡 提示: 使用 'python experiment_3_ensemble_learning.py optimal' 运行R²最优模型实验") 