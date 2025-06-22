#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验2：特征选择方法比较 (成员B)
CDS503 Group Project - 首尔自行车需求预测

目标：比较不同特征选择方法对模型性能的影响
方法：相关性分析、RFE、LASSO、树模型特征重要性、方差选择
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import BikeDataPreprocessor

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class FeatureSelectionExperiment:
    """特征选择实验类"""
    
    def __init__(self):
        self.results = {}
        self.feature_importance_scores = {}
        self.selected_features = {}
        self.preprocessor = BikeDataPreprocessor()
        
    def calculate_metrics(self, y_true, y_pred):
        """计算评估指标"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return {'RMSE': rmse, 'R²': r2}
    
    def correlation_based_selection(self, X_train, y_train, threshold=0.1):
        """基于相关性的特征选择"""
        print("🔍 执行相关性分析特征选择...")
        
        # 计算特征与目标变量的相关性
        correlations = {}
        for col in X_train.columns:
            corr = np.abs(np.corrcoef(X_train[col], y_train)[0, 1])
            correlations[col] = corr if not np.isnan(corr) else 0
        
        # 按相关性排序
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # 选择高相关性特征
        selected_features = [feat for feat, corr in sorted_features if corr >= threshold]
        
        print(f"  相关性阈值: {threshold}")
        print(f"  选择特征数量: {len(selected_features)}/{len(X_train.columns)}")
        
        self.feature_importance_scores['Correlation'] = correlations
        return selected_features
    
    def mutual_info_selection(self, X_train, y_train, k=20):
        """基于互信息的特征选择"""
        print("🔍 执行互信息特征选择...")
        
        # 计算互信息分数
        mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
        
        # 创建特征重要性字典
        mi_dict = dict(zip(X_train.columns, mi_scores))
        
        # 选择top-k特征
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        selector.fit(X_train, y_train)
        
        selected_features = X_train.columns[selector.get_support()].tolist()
        
        print(f"  选择特征数量: {len(selected_features)}")
        
        self.feature_importance_scores['Mutual_Info'] = mi_dict
        return selected_features
    
    def rfe_selection(self, X_train, y_train, n_features=20):
        """递归特征消除(RFE)"""
        print("🔍 执行RFE特征选择...")
        
        # 使用线性回归作为基础估计器
        estimator = LinearRegression()
        
        # RFE选择器
        rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
        rfe.fit(X_train, y_train)
        
        selected_features = X_train.columns[rfe.support_].tolist()
        
        # 获取特征排名（1是最重要的）
        feature_ranking = dict(zip(X_train.columns, rfe.ranking_))
        
        print(f"  选择特征数量: {len(selected_features)}")
        
        self.feature_importance_scores['RFE'] = feature_ranking
        return selected_features
    
    def lasso_selection(self, X_train, y_train, alpha=0.1):
        """LASSO特征选择"""
        print("🔍 执行LASSO特征选择...")
        
        # LASSO回归
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(X_train, y_train)
        
        # 获取非零系数的特征
        selected_features = X_train.columns[lasso.coef_ != 0].tolist()
        
        # 特征重要性（系数绝对值）
        feature_importance = dict(zip(X_train.columns, np.abs(lasso.coef_)))
        
        print(f"  LASSO alpha: {alpha}")
        print(f"  选择特征数量: {len(selected_features)}")
        
        self.feature_importance_scores['LASSO'] = feature_importance
        return selected_features
    
    def tree_based_selection(self, X_train, y_train, threshold='mean'):
        """基于树模型的特征选择"""
        print("🔍 执行树模型特征选择...")
        
        # 随机森林
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # 使用SelectFromModel进行特征选择
        selector = SelectFromModel(rf, threshold=threshold)
        selector.fit(X_train, y_train)
        
        selected_features = X_train.columns[selector.get_support()].tolist()
        
        # 特征重要性
        feature_importance = dict(zip(X_train.columns, rf.feature_importances_))
        
        print(f"  选择阈值: {threshold}")
        print(f"  选择特征数量: {len(selected_features)}")
        
        self.feature_importance_scores['Tree_Based'] = feature_importance
        return selected_features
    
    def variance_threshold_selection(self, X_train, threshold=0.01):
        """方差阈值特征选择"""
        print("🔍 执行方差阈值特征选择...")
        
        # 计算特征方差
        variances = X_train.var()
        
        # 选择方差大于阈值的特征
        selected_features = X_train.columns[variances > threshold].tolist()
        
        print(f"  方差阈值: {threshold}")
        print(f"  选择特征数量: {len(selected_features)}")
        
        self.feature_importance_scores['Variance'] = variances.to_dict()
        return selected_features
    
    def evaluate_feature_set(self, X_train, X_val, X_test, y_train, y_val, y_test, 
                           selected_features, method_name):
        """评估特征集的性能"""
        
        if not selected_features:
            print(f"  ⚠️ {method_name}: 没有选择任何特征")
            return None
        
        # 选择特征子集
        X_train_selected = X_train[selected_features]
        X_val_selected = X_val[selected_features]
        X_test_selected = X_test[selected_features]
        
        # 训练模型（使用随机森林）
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_selected, y_train)
        
        # 预测
        y_train_pred = model.predict(X_train_selected)
        y_val_pred = model.predict(X_val_selected)
        y_test_pred = model.predict(X_test_selected)
        
        # 计算指标
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        val_metrics = self.calculate_metrics(y_val, y_val_pred)
        test_metrics = self.calculate_metrics(y_test, y_test_pred)
        
        print(f"  📊 {method_name} 性能:")
        print(f"    特征数量: {len(selected_features)}")
        print(f"    验证集 RMSE: {val_metrics['RMSE']:.2f}")
        print(f"    测试集 RMSE: {test_metrics['RMSE']:.2f}")
        print(f"    测试集 R²: {test_metrics['R²']:.4f}")
        
        return {
            'method': method_name,
            'n_features': len(selected_features),
            'selected_features': selected_features,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'model': model
        }
    
    def run_experiment(self):
        """运行特征选择实验"""
        print("🚀 开始实验2：特征选择方法比较")
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
        print(f"  原始特征数量: {X_train_scaled.shape[1]}")
        print(f"  训练样本: {X_train_scaled.shape[0]}")
        
        # 基准模型（使用所有特征）
        print("\n2. 训练基准模型（所有特征）...")
        baseline_result = self.evaluate_feature_set(
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test,
            X_train_scaled.columns.tolist(), 'Baseline (All Features)'
        )
        
        if baseline_result:
            self.results['Baseline'] = baseline_result
        
        print("\n3. 执行特征选择方法...")
        
        # 方法1：相关性分析
        corr_features = self.correlation_based_selection(X_train_scaled, y_train, threshold=0.1)
        corr_result = self.evaluate_feature_set(
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test, corr_features, 'Correlation Analysis'
        )
        if corr_result:
            self.results['Correlation'] = corr_result
        
        # 方法2：互信息
        mi_features = self.mutual_info_selection(X_train_scaled, y_train, k=20)
        mi_result = self.evaluate_feature_set(
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test, mi_features, 'Mutual Information'
        )
        if mi_result:
            self.results['Mutual_Info'] = mi_result
        
        # 方法3：RFE
        rfe_features = self.rfe_selection(X_train_scaled, y_train, n_features=20)
        rfe_result = self.evaluate_feature_set(
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test, rfe_features, 'RFE'
        )
        if rfe_result:
            self.results['RFE'] = rfe_result
        
        # 方法4：LASSO
        lasso_features = self.lasso_selection(X_train_scaled, y_train, alpha=0.1)
        lasso_result = self.evaluate_feature_set(
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test, lasso_features, 'LASSO'
        )
        if lasso_result:
            self.results['LASSO'] = lasso_result
        
        # 方法5：树模型特征重要性
        tree_features = self.tree_based_selection(X_train_scaled, y_train, threshold='mean')
        tree_result = self.evaluate_feature_set(
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test, tree_features, 'Tree Based'
        )
        if tree_result:
            self.results['Tree_Based'] = tree_result
        
        print("\n✅ 特征选择实验完成!")
        
        # 生成报告
        self.generate_comparison_report()
        
        # 保存结果
        self.save_results()
        
        return self.results
    
    def generate_comparison_report(self):
        """生成特征选择比较报告"""
        print("\n📊 生成特征选择比较报告...")
        
        # 创建结果DataFrame
        comparison_data = []
        for method, result in self.results.items():
            row = {
                'Method': method,
                'N_Features': result['n_features'],
                'Train_RMSE': result['train_metrics']['RMSE'],
                'Val_RMSE': result['val_metrics']['RMSE'],
                'Test_RMSE': result['test_metrics']['RMSE'],
                'Test_R²': result['test_metrics']['R²'],
            }
            comparison_data.append(row)
        
        results_df = pd.DataFrame(comparison_data)
        results_df = results_df.sort_values('Test_RMSE')
        
        print("\n📈 特征选择方法比较 (按测试集RMSE排序):")
        print("="*70)
        print(results_df.round(4).to_string(index=False))
        
        # 保存结果表格
        results_df.to_csv('experiment_2_results.csv', index=False)
        
        # 创建可视化
        self.create_visualizations(results_df)
        
        # 性能分析
        print(f"\n🏆 最佳特征选择方法:")
        best_method = results_df.iloc[0]
        print(f"  方法: {best_method['Method']}")
        print(f"  特征数量: {best_method['N_Features']}")
        print(f"  测试集RMSE: {best_method['Test_RMSE']:.2f}")
        print(f"  测试集R²: {best_method['Test_R²']:.4f}")
        
        # 效率分析
        baseline_rmse = results_df[results_df['Method'] == 'Baseline']['Test_RMSE'].iloc[0]
        print(f"\n📊 相对基准模型的改进:")
        for _, row in results_df.iterrows():
            if row['Method'] != 'Baseline':
                improvement = (baseline_rmse - row['Test_RMSE']) / baseline_rmse * 100
                feature_reduction = (1 - row['N_Features'] / results_df[results_df['Method'] == 'Baseline']['N_Features'].iloc[0]) * 100
                print(f"  {row['Method']}: RMSE改进 {improvement:.1f}%, 特征减少 {feature_reduction:.1f}%")
    
    def create_visualizations(self, results_df):
        """创建可视化图表"""
        
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 特征数量 vs 性能
        plt.subplot(2, 3, 1)
        methods = results_df['Method']
        n_features = results_df['N_Features']
        test_rmse = results_df['Test_RMSE']
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
        
        for i, (method, n_feat, rmse) in enumerate(zip(methods, n_features, test_rmse)):
            plt.scatter(n_feat, rmse, s=100, c=[colors[i]], alpha=0.7, label=method)
        
        plt.xlabel('特征数量')
        plt.ylabel('测试集 RMSE')
        plt.title('特征数量 vs 模型性能')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 2. 方法性能比较
        plt.subplot(2, 3, 2)
        bars = plt.bar(range(len(methods)), test_rmse, color='lightcoral', alpha=0.7)
        plt.xlabel('特征选择方法')
        plt.ylabel('测试集 RMSE')
        plt.title('特征选择方法性能比较')
        plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
        
        # 添加数值标签
        for i, v in enumerate(test_rmse):
            plt.text(i, v + 5, f'{v:.1f}', ha='center', va='bottom')
        
        plt.grid(axis='y', alpha=0.3)
        
        # 3. R²分数比较
        plt.subplot(2, 3, 3)
        r2_values = results_df['Test_R²']
        bars = plt.bar(range(len(methods)), r2_values, color='lightgreen', alpha=0.7)
        plt.xlabel('特征选择方法')
        plt.ylabel('测试集 R²')
        plt.title('特征选择方法 R²比较')
        plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
        
        # 添加数值标签
        for i, v in enumerate(r2_values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.grid(axis='y', alpha=0.3)
        
        # 4. 特征重要性热图（如果有数据）
        plt.subplot(2, 3, 4)
        if self.feature_importance_scores:
            # 选择几个重要的特征选择方法
            methods_to_show = ['Correlation', 'Tree_Based', 'LASSO']
            
            # 获取所有特征
            all_features = list(next(iter(self.feature_importance_scores.values())).keys())
            
            # 创建重要性矩阵
            importance_matrix = []
            method_labels = []
            
            for method in methods_to_show:
                if method in self.feature_importance_scores:
                    scores = self.feature_importance_scores[method]
                    importance_row = [scores.get(feat, 0) for feat in all_features[:20]]  # 只显示前20个特征
                    importance_matrix.append(importance_row)
                    method_labels.append(method)
            
            if importance_matrix:
                importance_matrix = np.array(importance_matrix)
                
                # 标准化到0-1范围
                for i in range(len(importance_matrix)):
                    row = importance_matrix[i]
                    if row.max() > 0:
                        importance_matrix[i] = row / row.max()
                
                im = plt.imshow(importance_matrix, cmap='YlOrRd', aspect='auto')
                plt.colorbar(im, shrink=0.8)
                plt.ylabel('特征选择方法')
                plt.xlabel('特征')
                plt.title('特征重要性热图 (Top 20)')
                plt.yticks(range(len(method_labels)), method_labels)
                plt.xticks(range(min(20, len(all_features))), all_features[:20], rotation=90)
        
        # 5. 模型复杂度分析
        plt.subplot(2, 3, 5)
        plt.scatter(n_features, r2_values, s=100, alpha=0.7, c='purple')
        
        for i, method in enumerate(methods):
            plt.annotate(method, (n_features.iloc[i], r2_values.iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('特征数量')
        plt.ylabel('测试集 R²')
        plt.title('模型复杂度 vs 性能')
        plt.grid(True, alpha=0.3)
        
        # 6. 训练集 vs 测试集性能
        plt.subplot(2, 3, 6)
        train_rmse = results_df['Train_RMSE']
        
        plt.scatter(train_rmse, test_rmse, s=100, alpha=0.7, c='orange')
        
        # 添加对角线（理想情况）
        min_rmse = min(train_rmse.min(), test_rmse.min())
        max_rmse = max(train_rmse.max(), test_rmse.max())
        plt.plot([min_rmse, max_rmse], [min_rmse, max_rmse], 'r--', alpha=0.8, label='理想线')
        
        for i, method in enumerate(methods):
            plt.annotate(method, (train_rmse.iloc[i], test_rmse.iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('训练集 RMSE')
        plt.ylabel('测试集 RMSE')
        plt.title('训练集 vs 测试集性能')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('experiment_2_feature_selection.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📸 可视化图表已保存: experiment_2_feature_selection.png")
    
    def save_results(self):
        """保存实验结果"""
        # 保存特征选择结果
        feature_selection_summary = {}
        for method, result in self.results.items():
            feature_selection_summary[method] = {
                'n_features': result['n_features'],
                'selected_features': result['selected_features'],
                'test_rmse': result['test_metrics']['RMSE'],
                'test_r2': result['test_metrics']['R²']
            }
        
        # 保存为JSON
        import json
        with open('experiment_2_summary.json', 'w', encoding='utf-8') as f:
            json.dump(feature_selection_summary, f, indent=2, ensure_ascii=False)
        
        # 保存特征重要性分数
        with open('feature_importance_scores.json', 'w', encoding='utf-8') as f:
            json.dump(self.feature_importance_scores, f, indent=2, ensure_ascii=False)
        
        print("💾 实验结果已保存:")
        print("  - experiment_2_results.csv")
        print("  - experiment_2_summary.json")
        print("  - feature_importance_scores.json")
        print("  - experiment_2_feature_selection.png")

def main():
    """主函数"""
    
    # 创建实验实例
    experiment = FeatureSelectionExperiment()
    
    # 运行实验
    results = experiment.run_experiment()
    
    print("\n🎉 实验2完成！")
    print(f"总共测试了 {len(results)} 种特征选择方法")
    
    # 输出最终总结
    print("\n📋 实验总结:")
    print("- 已完成特征选择方法比较")
    print("- 识别最重要的特征子集")
    print("- 分析特征数量对模型性能的影响")
    print("- 为模型优化提供特征选择建议")

if __name__ == "__main__":
    main() 