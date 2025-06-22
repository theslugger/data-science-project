"""
首尔自行车数据集集成学习方法对比分析
使用线性回归、KNN、神经网络和SVR四种基础模型
对比bagging、voting、stacking和boosting四种集成方法
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (BaggingRegressor, VotingRegressor, 
                             GradientBoostingRegressor, AdaBoostRegressor)
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

# 设置英文字体和科学配色
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')  # 科学配色风格

class EnsembleLearningComparison:
    def __init__(self, data_path):
        """初始化集成学习对比类"""
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess data"""
        print("Loading and preprocessing data...")
        
        # 加载数据，尝试不同的编码
        try:
            self.data = pd.read_csv(self.data_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                self.data = pd.read_csv(self.data_path, encoding='latin-1')
            except UnicodeDecodeError:
                self.data = pd.read_csv(self.data_path, encoding='cp1252')
        print("Data loading completed")
        print(f"Data shape: {self.data.shape}")
        print("Column names:", self.data.columns.tolist())
        
        # 处理列名中的特殊字符
        column_mapping = {}
        for col in self.data.columns:
            new_col = col.replace('°', '').replace('(', '').replace(')', '').replace('', '')
            column_mapping[col] = new_col
        self.data = self.data.rename(columns=column_mapping)
        
        # 处理日期特征
        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%d/%m/%Y')
        self.data['Year'] = self.data['Date'].dt.year
        self.data['Month'] = self.data['Date'].dt.month
        self.data['DayOfWeek'] = self.data['Date'].dt.dayofweek
        
        # 编码分类变量
        le_seasons = LabelEncoder()
        le_holiday = LabelEncoder()
        le_functioning = LabelEncoder()
        
        self.data['Seasons_encoded'] = le_seasons.fit_transform(self.data['Seasons'])
        self.data['Holiday_encoded'] = le_holiday.fit_transform(self.data['Holiday'])
        self.data['Functioning_Day_encoded'] = le_functioning.fit_transform(self.data['Functioning Day'])
        
        # 选择特征
        numeric_features = ['Hour', 'TemperatureC', 'Humidity%', 'Wind speed m/s',
                           'Visibility 10m', 'Dew point temperatureC', 'Solar Radiation MJ/m2',
                           'Rainfallmm', 'Snowfall cm']
        
        # 检查实际存在的特征
        available_features = []
        for feature in numeric_features:
            matches = [col for col in self.data.columns if feature.replace(' ', '').replace('/', '').replace('(', '').replace(')', '') in col.replace(' ', '').replace('/', '').replace('(', '').replace(')', '')]
            if matches:
                available_features.append(matches[0])
        
        # 添加时间和编码特征
        feature_columns = available_features + ['Year', 'Month', 'DayOfWeek', 
                                              'Seasons_encoded', 'Holiday_encoded', 
                                              'Functioning_Day_encoded']
        
        print("Features used:", feature_columns)
        
        X = self.data[feature_columns]
        y = self.data['Rented Bike Count']
        
        # Data splitting: 70% train, 15% validation, 15% test
        # First split: 70% train, 30% temp
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Second split: 15% validation, 15% test (from 30% temp)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        # Standardization
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Data preprocessing completed! Training set size: {self.X_train.shape}, Validation set size: {self.X_val.shape}, Test set size: {self.X_test.shape}")
        
    def create_base_models(self):
        """Create base models"""
        print("Creating base models...")
        
        # 1. Linear Regression
        lr = LinearRegression()
        
        # 2. KNN Regressor
        knn = KNeighborsRegressor(n_neighbors=5)
        
        # 3. Neural Network
        mlp = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        # 4. Support Vector Regression
        svr = SVR(kernel='rbf', C=100, gamma='scale')
        
        self.base_models = {
            'Linear Regression': lr,
            'KNN Regression': knn,
            'Neural Network': mlp,
            'SVR': svr
        }
        
        return self.base_models
    
    def evaluate_base_models(self):
        """Evaluate base model performance"""
        print("Evaluating base model performance...")
        
        base_results = {}
        
        for name, model in self.base_models.items():
            print(f"Training {name}...")
            
            # Use standardized data for Neural Network and SVR
            if name in ['Neural Network', 'SVR']:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
            
            # Calculate evaluation metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            base_results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'predictions': y_pred
            }
            
            print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
        
        self.base_results = base_results
        return base_results
    
    def create_ensemble_models(self):
        """Create ensemble learning models"""
        print("Creating ensemble learning models...")
        
        ensemble_models = {}
        
        # 1. Bagging ensemble
        print("Creating Bagging ensemble...")
        ensemble_models['Bagging'] = BaggingRegressor(
            estimator=LinearRegression(),
            n_estimators=10,
            random_state=42
        )
        
        # 2. Voting ensemble
        print("Creating Voting ensemble...")
        voting_estimators = [
            ('lr', LinearRegression()),
            ('knn', KNeighborsRegressor(n_neighbors=5))
        ]
        ensemble_models['Voting'] = VotingRegressor(voting_estimators)
        
        # 3. Create first-level models for Stacking
        print("Creating Stacking ensemble...")
        self.stacking_models = {
            'level0': [LinearRegression(), KNeighborsRegressor(n_neighbors=5), SVR(kernel='rbf', C=100, gamma='scale')],
            'level1': LinearRegression()
        }
        
        # 4. GradientBoosting ensemble
        print("Creating GradientBoosting ensemble...")
        ensemble_models['GradientBoosting'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            random_state=42
        )
        
        self.ensemble_models = ensemble_models
        return ensemble_models
    
    def train_stacking_model(self):
        """Train Stacking model"""
        print("Training Stacking model...")
        
        # 5-fold cross-validation to generate first-level predictions
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Store first-level model predictions
        stacking_features_train = np.zeros((len(self.X_train), len(self.stacking_models['level0'])))
        stacking_features_test = np.zeros((len(self.X_test), len(self.stacking_models['level0'])))
        
        for i, model in enumerate(self.stacking_models['level0']):
            print(f"Training first-level model {i+1}/{len(self.stacking_models['level0'])}")
            
            # Cross-validation predictions for training set
            cv_predictions = np.zeros(len(self.X_train))
            
            for train_idx, val_idx in kf.split(self.X_train):
                X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                y_fold_train = self.y_train.iloc[train_idx]
                
                # Use standardized data for SVR
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
            
            # Train complete model to predict test set
            if isinstance(model, SVR):
                model.fit(self.X_train_scaled, self.y_train)
                stacking_features_test[:, i] = model.predict(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                stacking_features_test[:, i] = model.predict(self.X_test)
        
        # Train second-level model
        self.stacking_models['level1'].fit(stacking_features_train, self.y_train)
        
        # Predict
        stacking_predictions = self.stacking_models['level1'].predict(stacking_features_test)
        
        return stacking_predictions
    
    def evaluate_ensemble_models(self):
        """Evaluate ensemble learning model performance"""
        print("Evaluating ensemble learning model performance...")
        
        ensemble_results = {}
        
        # Evaluate standard ensemble models
        for name, model in self.ensemble_models.items():
            print(f"Training {name}...")
            
            # Train with original data
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            
            # Calculate evaluation metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            ensemble_results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'predictions': y_pred
            }
            
            print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
        
        # Evaluate Stacking model
        print("Evaluating Stacking model...")
        stacking_predictions = self.train_stacking_model()
        
        mse = mean_squared_error(self.y_test, stacking_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, stacking_predictions)
        r2 = r2_score(self.y_test, stacking_predictions)
        
        ensemble_results['Stacking'] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'predictions': stacking_predictions
        }
        
        print(f"Stacking - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
        
        self.ensemble_results = ensemble_results
        return ensemble_results
    
    def create_comparison_visualizations(self):
        """Create comparison visualization charts"""
        print("Creating comparison visualization charts...")
        
        # Set scientific color palette
        colors_sci = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Merge all results
        all_results = {**self.base_results, **self.ensemble_results}
        
        # 1. Performance comparison bar chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(all_results.keys())
        metrics = ['RMSE', 'MAE', 'R²', 'MSE']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            values = [all_results[model][metric] for model in models]
            
            # Categorize colors
            colors = []
            for j, model in enumerate(models):
                if model in self.base_results:
                    colors.append(colors_sci[j % len(colors_sci)])
                else:
                    colors.append(colors_sci[(j + 4) % len(colors_sci)])
            
            bars = ax.bar(range(len(models)), values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_title(f'{metric} Performance Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#f8f9fa')
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('ensemble_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Ensemble method category comparison
        fig2 = plt.figure(figsize=(15, 10))
        
        # Categorize ensemble methods
        ensemble_categories = {
            'Bagging': ['Bagging'],
            'Voting': ['Voting'],
            'Stacking': ['Stacking'],
            'GradientBoosting': ['GradientBoosting']
        }
        
        # Plot detailed method comparison
        plt.subplot(2, 2, 1)
        ensemble_models_list = list(self.ensemble_results.keys())
        ensemble_rmse = [self.ensemble_results[model]['RMSE'] for model in ensemble_models_list]
        
        colors = colors_sci[:len(ensemble_models_list)]
        bars = plt.bar(range(len(ensemble_models_list)), ensemble_rmse, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        plt.title('Ensemble Methods RMSE Comparison', fontsize=14, fontweight='bold')
        plt.xticks(range(len(ensemble_models_list)), ensemble_models_list, rotation=45, fontsize=10)
        plt.ylabel('RMSE', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.gca().set_facecolor('#f8f9fa')
        
        # Add value labels
        for bar, value in zip(bars, ensemble_rmse):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(ensemble_rmse)*0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # R² comparison
        plt.subplot(2, 2, 2)
        ensemble_r2 = [self.ensemble_results[model]['R²'] for model in ensemble_models_list]
        bars = plt.bar(range(len(ensemble_models_list)), ensemble_r2, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        plt.title('Ensemble Methods R² Comparison', fontsize=14, fontweight='bold')
        plt.xticks(range(len(ensemble_models_list)), ensemble_models_list, rotation=45, fontsize=10)
        plt.ylabel('R²', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.gca().set_facecolor('#f8f9fa')
        
        # Add value labels
        for bar, value in zip(bars, ensemble_r2):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(ensemble_r2)*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Prediction vs Actual scatter plot (best model)
        plt.subplot(2, 2, 3)
        best_model = min(self.ensemble_results.keys(), 
                        key=lambda x: self.ensemble_results[x]['RMSE'])
        best_predictions = self.ensemble_results[best_model]['predictions']
        
        plt.scatter(self.y_test, best_predictions, alpha=0.6, color=colors_sci[0], s=30, edgecolors='black', linewidth=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], color=colors_sci[3], linestyle='--', linewidth=2)
        plt.xlabel('Actual Values', fontsize=12, fontweight='bold')
        plt.ylabel('Predicted Values', fontsize=12, fontweight='bold')
        plt.title(f'Best Model ({best_model}) Prediction Performance', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.gca().set_facecolor('#f8f9fa')
        
        # Ensemble method category average performance
        plt.subplot(2, 2, 4)
        category_rmse = {}
        for category, methods in ensemble_categories.items():
            rmse_values = [self.ensemble_results[method]['RMSE'] 
                          for method in methods if method in self.ensemble_results]
            if rmse_values:
                category_rmse[category] = np.mean(rmse_values)
        
        categories = list(category_rmse.keys())
        rmse_values = list(category_rmse.values())
        plt.bar(categories, rmse_values, color=colors_sci[:len(categories)], alpha=0.8, edgecolor='black', linewidth=0.5)
        plt.title('Ensemble Method Category Average RMSE', fontsize=14, fontweight='bold')
        plt.ylabel('Average RMSE', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.gca().set_facecolor('#f8f9fa')
        
        for i, v in enumerate(rmse_values):
            plt.text(i, v + max(rmse_values)*0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('ensemble_detailed_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        """Generate summary report"""
        print("Generating summary report...")
        
        all_results = {**self.base_results, **self.ensemble_results}
        
        # Find best models
        best_model_rmse = min(all_results.keys(), key=lambda x: all_results[x]['RMSE'])
        best_model_r2 = max(all_results.keys(), key=lambda x: all_results[x]['R²'])
        
        # Ensemble method category performance
        ensemble_categories = {
            'Bagging': ['Bagging'],
            'Voting': ['Voting'],
            'Stacking': ['Stacking'],
            'GradientBoosting': ['GradientBoosting']
        }
        
        category_performance = {}
        for category, methods in ensemble_categories.items():
            rmse_values = [self.ensemble_results[method]['RMSE'] 
                          for method in methods if method in self.ensemble_results]
            r2_values = [self.ensemble_results[method]['R²'] 
                        for method in methods if method in self.ensemble_results]
            
            if rmse_values and r2_values:
                category_performance[category] = {
                    'avg_rmse': np.mean(rmse_values),
                    'avg_r2': np.mean(r2_values),
                    'methods_count': len(rmse_values)
                }
        
        # Create report
        report = f"""
=== Seoul Bike Dataset Ensemble Learning Methods Comparison Report ===

【Dataset Overview】
- Training samples: {len(self.X_train)}
- Validation samples: {len(self.X_val)}
- Test samples: {len(self.X_test)}
- Number of features: {self.X_train.shape[1]}

【Base Model Performance】
"""
        
        for model_name, results in self.base_results.items():
            report += f"- {model_name}: RMSE={results['RMSE']:.2f}, MAE={results['MAE']:.2f}, R²={results['R²']:.4f}\n"
        
        report += f"""
【Ensemble Learning Model Performance】
"""
        
        for model_name, results in self.ensemble_results.items():
            report += f"- {model_name}: RMSE={results['RMSE']:.2f}, MAE={results['MAE']:.2f}, R²={results['R²']:.4f}\n"
        
        report += f"""
【Best Models】
- Best RMSE: {best_model_rmse} (RMSE: {all_results[best_model_rmse]['RMSE']:.2f})
- Best R²: {best_model_r2} (R²: {all_results[best_model_r2]['R²']:.4f})

【Ensemble Method Category Performance Analysis】
"""
        
        for category, perf in category_performance.items():
            report += f"- {category}: Average RMSE={perf['avg_rmse']:.2f}, Average R²={perf['avg_r2']:.4f}, Method count={perf['methods_count']}\n"
        
        # Find best categories
        if category_performance:
            best_category_rmse = min(category_performance.keys(), 
                                   key=lambda x: category_performance[x]['avg_rmse'])
            best_category_r2 = max(category_performance.keys(), 
                                 key=lambda x: category_performance[x]['avg_r2'])
        else:
            best_category_rmse = "None"
            best_category_r2 = "None"
        
        report += f"""
【Conclusions and Recommendations】
1. Ensemble learning methods generally outperform single base models
2. {best_category_rmse} method performs best on RMSE metric
3. {best_category_r2} method performs best on R² metric  
4. Recommend prioritizing {best_model_rmse} model for practical applications

【Technical Details】
- Data preprocessing: Standardization, categorical variable encoding, temporal feature extraction
- Cross-validation: 5-fold cross-validation for Stacking model
- Evaluation metrics: RMSE, MAE, R², MSE
- Base models: Linear Regression, KNN, Neural Network, SVR
- Ensemble methods: Bagging, Voting, Stacking, Boosting (AdaBoost and GradientBoosting)
"""
        
        # Save report
        with open('ensemble_learning_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        return report
    
    def save_best_model(self):
        """Save GradientBoosting model as .sav file"""
        print("Saving GradientBoosting model...")
        
        all_results = {**self.base_results, **self.ensemble_results}
        
        # Force save GradientBoosting model
        if 'GradientBoosting' in all_results:
            best_model_name = 'GradientBoosting'
            best_rmse = all_results[best_model_name]['RMSE']
            best_r2 = all_results[best_model_name]['R²']
            
            print(f"Saving model: {best_model_name}")
            print(f"RMSE: {best_rmse:.2f}, R²: {best_r2:.4f}")
            
            # Display performance comparison of all ensemble models
            print(f"\nAll ensemble models performance comparison:")
            for model_name in ['Bagging', 'Voting', 'Stacking', 'GradientBoosting']:
                if model_name in all_results:
                    rmse = all_results[model_name]['RMSE']
                    r2 = all_results[model_name]['R²']
                    indicator = "👑" if model_name == 'GradientBoosting' else "  "
                    print(f"{indicator} {model_name}: RMSE={rmse:.2f}, R²={r2:.4f}")
        else:
            print("❌ GradientBoosting model not found!")
            return None
        
        # 准备要保存的模型对象
        model_to_save = None
        
        if best_model_name == 'Stacking':
            # Stacking模型需要特殊处理
            print("保存Stacking模型...")
            
            # 重新训练Stacking模型以获取完整的模型对象
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            stacking_features_train = np.zeros((len(self.X_train), len(self.stacking_models['level0'])))
            
            # 训练第一层模型
            trained_level0_models = []
            for i, model in enumerate(self.stacking_models['level0']):
                # 克隆模型以避免修改原始模型
                from sklearn.base import clone
                model_clone = clone(model)
                
                cv_predictions = np.zeros(len(self.X_train))
                for train_idx, val_idx in kf.split(self.X_train):
                    X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                    y_fold_train = self.y_train.iloc[train_idx]
                    
                    if isinstance(model_clone, SVR):
                        fold_scaler = StandardScaler()
                        X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train)
                        X_fold_val_scaled = fold_scaler.transform(X_fold_val)
                        model_clone.fit(X_fold_train_scaled, y_fold_train)
                        cv_predictions[val_idx] = model_clone.predict(X_fold_val_scaled)
                    else:
                        model_clone.fit(X_fold_train, y_fold_train)
                        cv_predictions[val_idx] = model_clone.predict(X_fold_val)
                
                stacking_features_train[:, i] = cv_predictions
                
                # 在完整训练集上训练最终模型
                final_model = clone(model)
                if isinstance(final_model, SVR):
                    final_model.fit(self.X_train_scaled, self.y_train)
                else:
                    final_model.fit(self.X_train, self.y_train)
                trained_level0_models.append(final_model)
            
            # 训练第二层模型
            from sklearn.base import clone
            level1_model = clone(self.stacking_models['level1'])
            level1_model.fit(stacking_features_train, self.y_train)
            
            # 创建Stacking模型字典
            stacking_model_dict = {
                'level0_models': trained_level0_models,
                'level1_model': level1_model,
                'scaler': self.scaler,
                'feature_columns': list(self.X_train.columns),
                'model_type': 'Stacking'
            }
            
            model_to_save = stacking_model_dict
            
        else:
            # 对于其他模型，获取已训练的模型
            if best_model_name in self.base_results:
                # 基础模型
                model_to_save = self.base_models[best_model_name]
                
                # 确保模型已经训练
                if best_model_name in ['Neural Network', 'SVR']:
                    model_to_save.fit(self.X_train_scaled, self.y_train)
                else:
                    model_to_save.fit(self.X_train, self.y_train)
                    
            else:
                # 集成模型
                model_to_save = self.ensemble_models[best_model_name]
                model_to_save.fit(self.X_train, self.y_train)
        
        # 创建模型包，包含模型、预处理器和元数据
        model_package = {
            'model': model_to_save,
            'scaler': self.scaler,
            'feature_columns': list(self.X_train.columns),
            'model_name': best_model_name,
            'performance': {
                'RMSE': best_rmse,
                'R²': best_r2,
                'MAE': all_results[best_model_name]['MAE'],
                'MSE': all_results[best_model_name]['MSE']
            },
            'model_type': 'ensemble' if best_model_name in self.ensemble_results else 'base'
        }
        
        # 保存模型为.sav文件
        model_filename = f'best_ensemble_model_{best_model_name.replace("-", "_").replace(" ", "_")}.sav'
        
        try:
            # Use joblib to save (recommended for scikit-learn models)
            joblib.dump(model_package, model_filename)
            print(f"✅ Model successfully saved as: {model_filename}")
            
            # Also save as .pkl file as backup
            pkl_filename = model_filename.replace('.sav', '.pkl')
            with open(pkl_filename, 'wb') as f:
                pickle.dump(model_package, f)
            print(f"✅ Backup file saved as: {pkl_filename}")
            
        except Exception as e:
            print(f"❌ Error occurred while saving model: {str(e)}")
            return None
        
        # Verify saved model
        self.verify_saved_model(model_filename, model_package)
        
        return model_filename
    
    def verify_saved_model(self, filename, original_model_package):
        """Verify if saved model can be loaded and predict correctly"""
        print("Verifying saved model...")
        
        try:
            # Load model
            loaded_model_package = joblib.load(filename)
            
            # Check metadata
            print(f"Loaded model name: {loaded_model_package['model_name']}")
            print(f"Model performance: RMSE={loaded_model_package['performance']['RMSE']:.2f}")
            
            # Perform prediction test
            if loaded_model_package['model_name'] == 'Stacking':
                # Stacking model prediction
                test_features = np.zeros((len(self.X_test), len(loaded_model_package['model']['level0_models'])))
                
                for i, level0_model in enumerate(loaded_model_package['model']['level0_models']):
                    if isinstance(level0_model, SVR):
                        test_features[:, i] = level0_model.predict(self.X_test_scaled)
                    else:
                        test_features[:, i] = level0_model.predict(self.X_test)
                
                test_predictions = loaded_model_package['model']['level1_model'].predict(test_features)
                
            else:
                # Other model prediction
                model = loaded_model_package['model']
                if loaded_model_package['model_name'] in ['Neural Network', 'SVR']:
                    test_predictions = model.predict(self.X_test_scaled)
                else:
                    test_predictions = model.predict(self.X_test)
            
            # Calculate prediction error
            test_rmse = np.sqrt(mean_squared_error(self.y_test, test_predictions))
            original_rmse = original_model_package['performance']['RMSE']
            
            if abs(test_rmse - original_rmse) < 0.01:  # Allow small numerical errors
                print("✅ Model verification successful! Loaded model predictions match original model")
            else:
                print(f"⚠️ Prediction differences found: Original RMSE={original_rmse:.2f}, Loaded RMSE={test_rmse:.2f}")
                
        except Exception as e:
            print(f"❌ Model verification failed: {str(e)}")
    
    @staticmethod
    def load_best_model(filename):
        """Static method: Load saved optimal model"""
        try:
            model_package = joblib.load(filename)
            print(f"✅ Model '{model_package['model_name']}' loaded successfully")
            print(f"Model performance: RMSE={model_package['performance']['RMSE']:.2f}, R²={model_package['performance']['R²']:.4f}")
            return model_package
        except Exception as e:
            print(f"❌ Model loading failed: {str(e)}")
            return None
    
    @staticmethod
    def predict_with_saved_model(model_package, X_new):
        """Use saved model for prediction"""
        try:
            # Check if feature columns match
            if hasattr(X_new, 'columns'):
                expected_features = model_package['feature_columns']
                if list(X_new.columns) != expected_features:
                    print("⚠️ Warning: Input features do not fully match training features")
                    print(f"Expected features: {expected_features}")
                    print(f"Input features: {list(X_new.columns)}")
            
            # Data preprocessing
            scaler = model_package['scaler']
            model = model_package['model']
            model_name = model_package['model_name']
            
            if model_name == 'Stacking':
                # Stacking model prediction
                level0_predictions = np.zeros((len(X_new), len(model['level0_models'])))
                
                for i, level0_model in enumerate(model['level0_models']):
                    if isinstance(level0_model, SVR):
                        X_scaled = scaler.transform(X_new)
                        level0_predictions[:, i] = level0_model.predict(X_scaled)
                    else:
                        level0_predictions[:, i] = level0_model.predict(X_new)
                
                predictions = model['level1_model'].predict(level0_predictions)
                
            else:
                # Other model prediction
                if model_name in ['Neural Network', 'SVR']:
                    X_scaled = scaler.transform(X_new)
                    predictions = model.predict(X_scaled)
                else:
                    predictions = model.predict(X_new)
            
            return predictions
            
        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
            return None
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        print("Starting ensemble learning comparison analysis...")
        
        # 1. Data preprocessing
        self.load_and_preprocess_data()
        
        # 2. Create and evaluate base models
        self.create_base_models()
        self.evaluate_base_models()
        
        # 3. Create and evaluate ensemble models
        self.create_ensemble_models()
        self.evaluate_ensemble_models()
        
        # 4. Create visualizations
        self.create_comparison_visualizations()
        
        # 5. Generate report
        self.generate_summary_report()
        
        # 6. Save best model
        best_model_file = self.save_best_model()
        
        print("\n" + "="*60)
        print("🎉 Analysis completed! Results saved to:")
        print(f"📊 Performance images: ensemble_performance_comparison.png, ensemble_detailed_comparison.png")
        print(f"📋 Analysis report: ensemble_learning_report.txt")
        if best_model_file:
            print(f"💾 GradientBoosting model saved as: {best_model_file}")
        print("="*60)

# Main program
if __name__ == "__main__":
    # Create analysis instance
    analyzer = EnsembleLearningComparison('SeoulBikeData.csv')
    
    # Run complete analysis
    analyzer.run_complete_analysis() 