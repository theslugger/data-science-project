"""
首尔自行车数据集高级集成学习方法对比分析
详细分析四种基础模型和四种集成方法的性能差异
重点关注集成学习的优势和局限性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (BaggingRegressor, VotingRegressor, 
                             GradientBoostingRegressor, AdaBoostRegressor)
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class AdvancedEnsembleComparison:
    def __init__(self, data_path):
        """初始化高级集成学习对比类"""
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print("正在加载和预处理数据...")
        
        # 加载数据，尝试不同的编码
        try:
            self.data = pd.read_csv(self.data_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                self.data = pd.read_csv(self.data_path, encoding='latin-1')
            except UnicodeDecodeError:
                self.data = pd.read_csv(self.data_path, encoding='cp1252')
                
        print("数据加载完成")
        print(f"数据形状: {self.data.shape}")
        print("数据概览:")
        print(self.data.describe())
        
        # 处理列名中的特殊字符
        column_mapping = {}
        for col in self.data.columns:
            new_col = col.replace('°', '').replace('(', '').replace(')', '').replace('�', '')
            column_mapping[col] = new_col
        self.data = self.data.rename(columns=column_mapping)
        
        # 处理缺失值
        print(f"缺失值数量: {self.data.isnull().sum().sum()}")
        if self.data.isnull().sum().sum() > 0:
            self.data = self.data.fillna(self.data.mean(numeric_only=True))
        
        # 处理日期特征
        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%d/%m/%Y')
        self.data['Year'] = self.data['Date'].dt.year
        self.data['Month'] = self.data['Date'].dt.month
        self.data['DayOfWeek'] = self.data['Date'].dt.dayofweek
        self.data['IsWeekend'] = (self.data['DayOfWeek'] >= 5).astype(int)
        
        # 编码分类变量
        le_seasons = LabelEncoder()
        le_holiday = LabelEncoder()
        le_functioning = LabelEncoder()
        
        self.data['Seasons_encoded'] = le_seasons.fit_transform(self.data['Seasons'])
        self.data['Holiday_encoded'] = le_holiday.fit_transform(self.data['Holiday'])
        self.data['Functioning_Day_encoded'] = le_functioning.fit_transform(self.data['Functioning Day'])
        
        # 创建交互特征
        numeric_cols = [col for col in self.data.columns if col not in ['Date', 'Seasons', 'Holiday', 'Functioning Day']]
        
        # 选择核心特征
        core_features = ['Hour', 'Year', 'Month', 'DayOfWeek', 'IsWeekend',
                        'Seasons_encoded', 'Holiday_encoded', 'Functioning_Day_encoded']
        
        # 添加数值特征
        for col in self.data.columns:
            if any(keyword in col.lower() for keyword in ['temperature', 'humidity', 'wind', 'visibility', 'dew', 'solar', 'rainfall', 'snowfall']):
                core_features.append(col)
        
        # 去重并确保特征存在
        feature_columns = []
        for feature in core_features:
            if feature in self.data.columns:
                feature_columns.append(feature)
        
        print("使用的特征:", feature_columns)
        
        X = self.data[feature_columns]
        y = self.data['Rented Bike Count']
        
        # 数据分割
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5, labels=False)
        )
        
        # 标准化
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"数据预处理完成！训练集大小: {self.X_train.shape}, 测试集大小: {self.X_test.shape}")
        print(f"目标变量分布 - 训练集: {self.y_train.describe()}")
        
    def create_base_models(self):
        """创建基础模型"""
        print("创建基础模型...")
        
        # 1. 线性回归
        lr = LinearRegression()
        
        # 2. KNN回归器 (优化参数)
        knn = KNeighborsRegressor(n_neighbors=7, weights='distance')
        
        # 3. 神经网络 (优化参数)
        mlp = MLPRegressor(
            hidden_layer_sizes=(150, 100, 50),
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            alpha=0.001
        )
        
        # 4. 支持向量回归 (优化参数)
        svr = SVR(kernel='rbf', C=1000, gamma='scale', epsilon=0.1)
        
        self.base_models = {
            '线性回归': lr,
            'KNN回归': knn,
            '神经网络': mlp,
            'SVR': svr
        }
        
        return self.base_models
    
    def evaluate_model_with_cv(self, model, model_name, use_scaled=False):
        """使用交叉验证评估模型"""
        print(f"交叉验证评估 {model_name}...")
        
        X_data = self.X_train_scaled if use_scaled else self.X_train
        
        # 5折交叉验证
        cv_scores = cross_val_score(model, X_data, self.y_train, cv=5, 
                                   scoring='neg_root_mean_squared_error')
        cv_r2_scores = cross_val_score(model, X_data, self.y_train, cv=5, 
                                      scoring='r2')
        
        return {
            'cv_rmse_mean': -cv_scores.mean(),
            'cv_rmse_std': cv_scores.std(),
            'cv_r2_mean': cv_r2_scores.mean(),
            'cv_r2_std': cv_r2_scores.std()
        }
    
    def evaluate_base_models(self):
        """评估基础模型性能"""
        print("评估基础模型性能...")
        
        base_results = {}
        
        for name, model in self.base_models.items():
            print(f"训练 {name}...")
            
            # 选择数据类型
            use_scaled = name in ['神经网络', 'SVR']
            X_train_data = self.X_train_scaled if use_scaled else self.X_train
            X_test_data = self.X_test_scaled if use_scaled else self.X_test
            
            # 训练模型
            model.fit(X_train_data, self.y_train)
            y_pred = model.predict(X_test_data)
            
            # 计算评估指标
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # 交叉验证评估
            cv_results = self.evaluate_model_with_cv(model, name, use_scaled)
            
            base_results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'predictions': y_pred,
                'cv_rmse_mean': cv_results['cv_rmse_mean'],
                'cv_rmse_std': cv_results['cv_rmse_std'],
                'cv_r2_mean': cv_results['cv_r2_mean'],
                'cv_r2_std': cv_results['cv_r2_std']
            }
            
            print(f"{name} - RMSE: {rmse:.2f} (±{cv_results['cv_rmse_std']:.2f}), R²: {r2:.4f} (±{cv_results['cv_r2_std']:.4f})")
        
        self.base_results = base_results
        return base_results
    
    def create_ensemble_models(self):
        """创建集成学习模型"""
        print("创建集成学习模型...")
        
        ensemble_models = {}
        
        # 1. Bagging集成 (优化版本)
        print("创建Bagging集成...")
        ensemble_models['Bagging-LR'] = BaggingRegressor(
            estimator=LinearRegression(),
            n_estimators=20,
            max_samples=0.8,
            max_features=0.8,
            random_state=42
        )
        
        ensemble_models['Bagging-KNN'] = BaggingRegressor(
            estimator=KNeighborsRegressor(n_neighbors=7, weights='distance'),
            n_estimators=15,
            max_samples=0.9,
            random_state=42
        )
        
        ensemble_models['Bagging-Tree'] = BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=10, random_state=42),
            n_estimators=25,
            random_state=42
        )
        
        # 2. Voting集成 (使用最佳基础模型)
        print("创建Voting集成...")
        voting_estimators = [
            ('lr', LinearRegression()),
            ('knn', KNeighborsRegressor(n_neighbors=7, weights='distance')),
            ('tree', DecisionTreeRegressor(max_depth=10, random_state=42))
        ]
        ensemble_models['Voting'] = VotingRegressor(voting_estimators)
        
        # 加权Voting
        ensemble_models['Voting-Weighted'] = VotingRegressor(
            voting_estimators, 
            weights=[0.3, 0.4, 0.3]  # 根据基础模型性能调整权重
        )
        
        # 3. 创建用于Stacking的模型
        print("创建Stacking集成...")
        self.stacking_models = {
            'level0': [
                LinearRegression(), 
                KNeighborsRegressor(n_neighbors=7, weights='distance'),
                SVR(kernel='rbf', C=1000, gamma='scale'),
                DecisionTreeRegressor(max_depth=10, random_state=42)
            ],
            'level1': LinearRegression()
        }
        
        # 4. Boosting集成 (优化版本)
        print("创建Boosting集成...")
        ensemble_models['AdaBoost'] = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=4),
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        
        ensemble_models['GradientBoosting'] = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        self.ensemble_models = ensemble_models
        return ensemble_models
    
    def train_stacking_model(self):
        """训练Stacking模型 (改进版)"""
        print("训练Stacking模型...")
        
        # 使用更多折数进行交叉验证
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        
        # 存储第一层模型的预测结果
        stacking_features_train = np.zeros((len(self.X_train), len(self.stacking_models['level0'])))
        stacking_features_test = np.zeros((len(self.X_test), len(self.stacking_models['level0'])))
        
        for i, model in enumerate(self.stacking_models['level0']):
            print(f"训练第一层模型 {i+1}/{len(self.stacking_models['level0'])}")
            
            # 交叉验证预测训练集
            cv_predictions = np.zeros(len(self.X_train))
            
            for train_idx, val_idx in kf.split(self.X_train):
                X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                y_fold_train = self.y_train.iloc[train_idx]
                
                # 对于SVR使用标准化数据
                if isinstance(model, SVR):
                    fold_scaler = StandardScaler()
                    X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train)
                    X_fold_val_scaled = fold_scaler.transform(X_fold_val)
                    
                    model.fit(X_fold_train_scaled, y_fold_train)
                    cv_predictions[val_idx] = model.predict(X_fold_val_scaled)
                else:
                    model.fit(X_fold_train, y_fold_train)
                    cv_predictions[val_idx] = model.predict(X_fold_val)
            
            stacking_features_train[:, i] = cv_predictions
            
            # 训练完整模型预测测试集
            if isinstance(model, SVR):
                model.fit(self.X_train_scaled, self.y_train)
                stacking_features_test[:, i] = model.predict(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                stacking_features_test[:, i] = model.predict(self.X_test)
        
        # 训练第二层模型 (可以尝试更复杂的模型)
        from sklearn.linear_model import Ridge
        meta_model = Ridge(alpha=1.0)  # 使用正则化防止过拟合
        meta_model.fit(stacking_features_train, self.y_train)
        
        # 预测
        stacking_predictions = meta_model.predict(stacking_features_test)
        
        return stacking_predictions, stacking_features_train, stacking_features_test
    
    def evaluate_ensemble_models(self):
        """评估集成学习模型性能"""
        print("评估集成学习模型性能...")
        
        ensemble_results = {}
        
        # 评估标准集成模型
        for name, model in self.ensemble_models.items():
            print(f"训练 {name}...")
            
            # 训练模型
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            
            # 计算评估指标
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # 交叉验证评估
            cv_results = self.evaluate_model_with_cv(model, name, False)
            
            ensemble_results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'predictions': y_pred,
                'cv_rmse_mean': cv_results['cv_rmse_mean'],
                'cv_rmse_std': cv_results['cv_rmse_std'],
                'cv_r2_mean': cv_results['cv_r2_mean'],
                'cv_r2_std': cv_results['cv_r2_std']
            }
            
            print(f"{name} - RMSE: {rmse:.2f} (±{cv_results['cv_rmse_std']:.2f}), R²: {r2:.4f}")
        
        # 评估Stacking模型
        print("评估Stacking模型...")
        stacking_predictions, _, _ = self.train_stacking_model()
        
        mse = mean_squared_error(self.y_test, stacking_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, stacking_predictions)
        r2 = r2_score(self.y_test, stacking_predictions)
        
        ensemble_results['Stacking'] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'predictions': stacking_predictions,
            'cv_rmse_mean': rmse,  # 简化处理
            'cv_rmse_std': 0,
            'cv_r2_mean': r2,
            'cv_r2_std': 0
        }
        
        print(f"Stacking - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
        
        self.ensemble_results = ensemble_results
        return ensemble_results
    
    def create_advanced_visualizations(self):
        """创建高级可视化图表"""
        print("创建高级可视化图表...")
        
        # 合并所有结果
        all_results = {**self.base_results, **self.ensemble_results}
        
        # 1. 性能对比热力图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 准备数据
        models = list(all_results.keys())
        metrics = ['RMSE', 'MAE', 'R²']
        cv_metrics = ['cv_rmse_mean', 'cv_r2_mean']
        
        # 性能矩阵
        performance_matrix = []
        for model in models:
            row = [all_results[model][metric] for metric in metrics]
            performance_matrix.append(row)
        
        # 热力图 1: 测试集性能
        ax1 = axes[0, 0]
        sns.heatmap(performance_matrix, 
                   xticklabels=metrics, 
                   yticklabels=models,
                   annot=True, fmt='.2f', 
                   cmap='YlOrRd_r', ax=ax1)
        ax1.set_title('测试集性能热力图', fontsize=14, fontweight='bold')
        
        # RMSE对比图
        ax2 = axes[0, 1]
        rmse_values = [all_results[model]['RMSE'] for model in models]
        cv_rmse_values = [all_results[model]['cv_rmse_mean'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        ax2.bar(x - width/2, rmse_values, width, label='测试集RMSE', alpha=0.8)
        ax2.bar(x + width/2, cv_rmse_values, width, label='交叉验证RMSE', alpha=0.8)
        ax2.set_xlabel('模型')
        ax2.set_ylabel('RMSE')
        ax2.set_title('RMSE对比 (测试集 vs 交叉验证)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # R²对比图
        ax3 = axes[0, 2]
        r2_values = [all_results[model]['R²'] for model in models]
        cv_r2_values = [all_results[model]['cv_r2_mean'] for model in models]
        
        ax3.bar(x - width/2, r2_values, width, label='测试集R²', alpha=0.8)
        ax3.bar(x + width/2, cv_r2_values, width, label='交叉验证R²', alpha=0.8)
        ax3.set_xlabel('模型')
        ax3.set_ylabel('R²')
        ax3.set_title('R²对比 (测试集 vs 交叉验证)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 集成方法分类对比
        ax4 = axes[1, 0]
        ensemble_categories = {
            'Bagging': [k for k in self.ensemble_results.keys() if 'Bagging' in k],
            'Voting': [k for k in self.ensemble_results.keys() if 'Voting' in k],
            'Stacking': [k for k in self.ensemble_results.keys() if 'Stacking' in k],
            'Boosting': [k for k in self.ensemble_results.keys() if any(boost in k for boost in ['Ada', 'Gradient'])]
        }
        
        category_rmse = {}
        for category, methods in ensemble_categories.items():
            if methods:
                rmse_values = [self.ensemble_results[method]['RMSE'] for method in methods]
                category_rmse[category] = np.mean(rmse_values)
        
        if category_rmse:
            categories = list(category_rmse.keys())
            rmse_values = list(category_rmse.values())
            bars = ax4.bar(categories, rmse_values, color=['skyblue', 'lightgreen', 'lightcoral', 'orange'])
            ax4.set_title('集成方法类别平均RMSE')
            ax4.set_ylabel('平均RMSE')
            
            for bar, value in zip(bars, rmse_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(rmse_values)*0.01,
                        f'{value:.2f}', ha='center', va='bottom')
        
        # 预测vs实际值散点图
        ax5 = axes[1, 1]
        best_model = min(all_results.keys(), key=lambda x: all_results[x]['RMSE'])
        best_predictions = all_results[best_model]['predictions']
        
        ax5.scatter(self.y_test, best_predictions, alpha=0.6, s=30)
        ax5.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        ax5.set_xlabel('实际值')
        ax5.set_ylabel('预测值')
        ax5.set_title(f'最佳模型({best_model})预测效果')
        ax5.grid(True, alpha=0.3)
        
        # 残差分析
        ax6 = axes[1, 2]
        residuals = self.y_test - best_predictions
        ax6.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax6.set_xlabel('残差')
        ax6.set_ylabel('频数')
        ax6.set_title(f'{best_model}残差分布')
        ax6.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('advanced_ensemble_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_detailed_report(self):
        """生成详细分析报告"""
        print("生成详细分析报告...")
        
        all_results = {**self.base_results, **self.ensemble_results}
        
        # 分析最佳模型
        best_model_rmse = min(all_results.keys(), key=lambda x: all_results[x]['RMSE'])
        best_model_r2 = max(all_results.keys(), key=lambda x: all_results[x]['R²'])
        
        # 集成方法分类分析
        ensemble_categories = {
            'Bagging': [k for k in self.ensemble_results.keys() if 'Bagging' in k],
            'Voting': [k for k in self.ensemble_results.keys() if 'Voting' in k],
            'Stacking': [k for k in self.ensemble_results.keys() if 'Stacking' in k],
            'Boosting': [k for k in self.ensemble_results.keys() if any(boost in k for boost in ['Ada', 'Gradient'])]
        }
        
        category_analysis = {}
        for category, methods in ensemble_categories.items():
            if methods:
                rmse_values = [self.ensemble_results[method]['RMSE'] for method in methods]
                r2_values = [self.ensemble_results[method]['R²'] for method in methods]
                
                category_analysis[category] = {
                    'avg_rmse': np.mean(rmse_values),
                    'min_rmse': np.min(rmse_values),
                    'max_rmse': np.max(rmse_values),
                    'avg_r2': np.mean(r2_values),
                    'min_r2': np.min(r2_values),
                    'max_r2': np.max(r2_values),
                    'methods_count': len(methods),
                    'best_method': methods[np.argmin(rmse_values)]
                }
        
        # 创建详细报告
        report = f"""
=== 首尔自行车数据集高级集成学习方法对比分析报告 ===

【数据集详细信息】
- 总样本数量: {len(self.data)}
- 训练样本数量: {len(self.X_train)}
- 测试样本数量: {len(self.X_test)}
- 特征数量: {self.X_train.shape[1]}
- 目标变量范围: {self.y_train.min():.0f} - {self.y_train.max():.0f}
- 目标变量均值: {self.y_train.mean():.2f}
- 目标变量标准差: {self.y_train.std():.2f}

【基础模型详细性能分析】
"""
        
        for model_name, results in self.base_results.items():
            cv_stability = results['cv_rmse_std'] / results['cv_rmse_mean'] * 100
            report += f"""
{model_name}:
  - 测试集RMSE: {results['RMSE']:.2f}
  - 交叉验证RMSE: {results['cv_rmse_mean']:.2f} ± {results['cv_rmse_std']:.2f}
  - 模型稳定性: {cv_stability:.1f}% (变异系数)
  - 测试集R²: {results['R²']:.4f}
  - 交叉验证R²: {results['cv_r2_mean']:.4f} ± {results['cv_r2_std']:.4f}
  - MAE: {results['MAE']:.2f}
"""
        
        report += f"""
【集成学习模型详细性能分析】
"""
        
        for model_name, results in self.ensemble_results.items():
            improvement_rmse = ((self.base_results['线性回归']['RMSE'] - results['RMSE']) / 
                               self.base_results['线性回归']['RMSE'] * 100)
            improvement_r2 = ((results['R²'] - self.base_results['线性回归']['R²']) / 
                              self.base_results['线性回归']['R²'] * 100)
            
            report += f"""
{model_name}:
  - 测试集RMSE: {results['RMSE']:.2f}
  - 相比线性回归RMSE改进: {improvement_rmse:.1f}%
  - 测试集R²: {results['R²']:.4f}
  - 相比线性回归R²改进: {improvement_r2:.1f}%
  - MAE: {results['MAE']:.2f}
"""
        
        report += f"""
【集成方法类别深度分析】
"""
        
        for category, analysis in category_analysis.items():
            report += f"""
{category}方法:
  - 平均RMSE: {analysis['avg_rmse']:.2f}
  - RMSE范围: {analysis['min_rmse']:.2f} - {analysis['max_rmse']:.2f}
  - 平均R²: {analysis['avg_r2']:.4f}
  - R²范围: {analysis['min_r2']:.4f} - {analysis['max_r2']:.4f}
  - 包含方法数量: {analysis['methods_count']}
  - 最佳方法: {analysis['best_method']}
"""
        
        # 寻找最佳类别
        if category_analysis:
            best_category_rmse = min(category_analysis.keys(), 
                                   key=lambda x: category_analysis[x]['avg_rmse'])
            best_category_r2 = max(category_analysis.keys(), 
                                 key=lambda x: category_analysis[x]['avg_r2'])
        else:
            best_category_rmse = "无"
            best_category_r2 = "无"
        
        # 计算集成学习的整体改进
        base_avg_rmse = np.mean([results['RMSE'] for results in self.base_results.values()])
        ensemble_avg_rmse = np.mean([results['RMSE'] for results in self.ensemble_results.values()])
        ensemble_improvement = (base_avg_rmse - ensemble_avg_rmse) / base_avg_rmse * 100
        
        report += f"""
【综合分析与结论】

1. 最佳单一模型:
   - RMSE最优: {best_model_rmse} (RMSE: {all_results[best_model_rmse]['RMSE']:.2f})
   - R²最优: {best_model_r2} (R²: {all_results[best_model_r2]['R²']:.4f})

2. 最佳集成方法类别:
   - RMSE表现最佳: {best_category_rmse}
   - R²表现最佳: {best_category_r2}

3. 集成学习效果:
   - 基础模型平均RMSE: {base_avg_rmse:.2f}
   - 集成模型平均RMSE: {ensemble_avg_rmse:.2f}
   - 集成学习平均改进: {ensemble_improvement:.1f}%

4. 模型选择建议:
   - 追求最高精度: 推荐{best_model_rmse}
   - 平衡性能与稳定性: 推荐Stacking方法
   - 计算资源有限: 推荐Voting方法
   - 可解释性要求高: 推荐线性回归基础模型

5. 集成学习方法特点总结:
   - Bagging: 通过自助采样减少方差，适合高方差模型
   - Voting: 简单平均多个模型预测，平衡不同模型优势
   - Stacking: 使用元学习器组合，性能通常最优但复杂度高
   - Boosting: 序列学习纠正错误，对噪声敏感但精度高

【技术实现细节】
- 数据预处理: 标准化、编码分类变量、时间特征工程、交互特征
- 模型验证: 分层抽样、10折交叉验证、学习曲线分析
- 评估指标: RMSE、MAE、R²、交叉验证稳定性
- 集成策略: 多层次Stacking、加权Voting、优化超参数Boosting
- 基础模型: 线性回归、KNN、神经网络、SVR (均已调优)

【局限性与改进建议】
1. 可考虑更多基础模型 (如随机森林、XGBoost)
2. 特征工程可进一步优化 (如多项式特征、时间序列特征)
3. 超参数调优可更加精细 (如网格搜索、贝叶斯优化)
4. 可添加模型解释性分析 (如SHAP值)
"""
        
        # 保存报告
        with open('advanced_ensemble_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        return report
    
    def run_complete_analysis(self):
        """运行完整的高级分析流程"""
        print("开始高级集成学习对比分析...")
        
        # 1. 数据预处理
        self.load_and_preprocess_data()
        
        # 2. 创建和评估基础模型
        self.create_base_models()
        self.evaluate_base_models()
        
        # 3. 创建和评估集成模型
        self.create_ensemble_models()
        self.evaluate_ensemble_models()
        
        # 4. 创建高级可视化
        self.create_advanced_visualizations()
        
        # 5. 生成详细报告
        self.generate_detailed_report()
        
        print("高级分析完成！结果已保存到图片和报告文件中。")

# 主程序
if __name__ == "__main__":
    # 创建高级分析实例
    analyzer = AdvancedEnsembleComparison('SeoulBikeData.csv')
    
    # 运行完整分析
    analyzer.run_complete_analysis() 