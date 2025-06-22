"""
使用保存的最优集成模型进行预测的示例代码
"""

import pandas as pd
import numpy as np
from ensemble_learning_comparison import EnsembleLearningComparison

def main():
    """演示如何加载和使用保存的模型"""
    
    print("=== 保存的集成模型使用示例 ===\n")
    
    # 示例1: 加载保存的模型
    print("1. 加载保存的最优模型...")
    
    # 这里需要根据实际保存的文件名来修改
    # 文件名格式通常是: best_ensemble_model_[ModelName].sav
    model_filename = "best_ensemble_model_GradientBoosting.sav"  # 示例文件名
    
    try:
        # 加载模型
        model_package = EnsembleLearningComparison.load_best_model(model_filename)
        
        if model_package is None:
            print("❌ 未找到保存的模型文件，请先运行完整分析生成模型")
            print("运行命令: python ensemble_learning_comparison.py")
            return
            
    except FileNotFoundError:
        print(f"❌ 文件不存在: {model_filename}")
        print("请先运行 ensemble_learning_comparison.py 生成模型文件")
        return
    
    # 示例2: 准备新数据进行预测
    print("\n2. 准备新数据进行预测...")
    
    # 这里使用模拟数据作为示例
    # 实际使用时，您需要提供真实的新数据
    np.random.seed(42)
    
    # 根据保存的特征列创建示例数据
    feature_columns = model_package['feature_columns']
    n_samples = 5  # 预测5个样本
    
    # 创建模拟数据（实际应用中用您的真实数据替换）
    sample_data = {}
    for feature in feature_columns:
        if 'Hour' in feature:
            sample_data[feature] = np.random.randint(0, 24, n_samples)
        elif 'Temperature' in feature:
            sample_data[feature] = np.random.uniform(-10, 35, n_samples)
        elif 'Humidity' in feature:
            sample_data[feature] = np.random.uniform(0, 100, n_samples)
        elif 'Wind' in feature:
            sample_data[feature] = np.random.uniform(0, 10, n_samples)
        elif 'Year' in feature:
            sample_data[feature] = np.random.choice([2017, 2018], n_samples)
        elif 'Month' in feature:
            sample_data[feature] = np.random.randint(1, 13, n_samples)
        elif 'DayOfWeek' in feature:
            sample_data[feature] = np.random.randint(0, 7, n_samples)
        elif 'encoded' in feature:
            sample_data[feature] = np.random.randint(0, 4, n_samples)
        else:
            sample_data[feature] = np.random.uniform(0, 100, n_samples)
    
    X_new = pd.DataFrame(sample_data)
    
    print("新数据样本:")
    print(X_new)
    
    # 示例3: 使用模型进行预测
    print("\n3. 使用模型进行预测...")
    
    predictions = EnsembleLearningComparison.predict_with_saved_model(model_package, X_new)
    
    if predictions is not None:
        print("\n预测结果:")
        for i, pred in enumerate(predictions):
            print(f"样本 {i+1}: 预测租赁数量 = {pred:.0f}")
        
        print(f"\n平均预测值: {np.mean(predictions):.0f}")
        print(f"最小预测值: {np.min(predictions):.0f}")
        print(f"最大预测值: {np.max(predictions):.0f}")
    
    # 示例4: 显示模型信息
    print(f"\n4. 模型信息:")
    print(f"模型名称: {model_package['model_name']}")
    print(f"模型类型: {model_package['model_type']}")
    print(f"模型性能:")
    for metric, value in model_package['performance'].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n=== 使用示例完成 ===")

def create_real_prediction_example():
    """创建真实数据预测示例"""
    print("\n=== 真实数据预测示例 ===")
    
    # 示例：创建一个具体的预测场景
    # 假设我们想预测明天下午2点的自行车租赁量
    
    # 加载模型（实际文件名需要根据生成的文件调整）
    model_files = [
        "best_ensemble_model_GradientBoosting.sav",
        "best_ensemble_model_Stacking.sav",
        "best_ensemble_model_Voting.sav"
    ]
    
    for model_file in model_files:
        try:
            model_package = EnsembleLearningComparison.load_best_model(model_file)
            break
        except FileNotFoundError:
            continue
    else:
        print("未找到任何保存的模型文件")
        return
    
    # 创建具体的预测场景
    scenarios = [
        {
            'name': '晴朗的春日下午',
            'data': {
                'Hour': 14,
                'TemperatureC': 22.0,
                'Humidity%': 45.0,
                'Wind speed m/s': 2.1,
                'Visibility 10m': 2000,
                'Dew point temperatureC': 8.0,
                'Solar Radiation MJ/m2': 1.5,
                'Rainfallmm': 0.0,
                'Snowfall cm': 0.0,
                'Year': 2018,
                'Month': 4,
                'DayOfWeek': 1,  # 周二
                'Seasons_encoded': 1,  # 春季
                'Holiday_encoded': 0,  # 非假日
                'Functioning_Day_encoded': 1  # 正常运营日
            }
        },
        {
            'name': '雨天的冬日傍晚',
            'data': {
                'Hour': 18,
                'TemperatureC': 2.0,
                'Humidity%': 85.0,
                'Wind speed m/s': 4.5,
                'Visibility 10m': 500,
                'Dew point temperatureC': -1.0,
                'Solar Radiation MJ/m2': 0.2,
                'Rainfallmm': 3.0,
                'Snowfall cm': 0.5,
                'Year': 2018,
                'Month': 12,
                'DayOfWeek': 5,  # 周六
                'Seasons_encoded': 3,  # 冬季
                'Holiday_encoded': 0,  # 非假日
                'Functioning_Day_encoded': 1  # 正常运营日
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        
        # 创建DataFrame
        scenario_df = pd.DataFrame([scenario['data']])
        
        # 确保列顺序与训练时一致
        feature_columns = model_package['feature_columns']
        scenario_df = scenario_df.reindex(columns=feature_columns, fill_value=0)
        
        # 进行预测
        prediction = EnsembleLearningComparison.predict_with_saved_model(model_package, scenario_df)
        
        if prediction is not None:
            print(f"预测租赁数量: {prediction[0]:.0f} 辆")
        
        # 显示关键条件
        print(f"  天气条件: {scenario['data']['TemperatureC']}°C, 湿度{scenario['data']['Humidity%']}%, 降雨{scenario['data']['Rainfallmm']}mm")
        print(f"  时间: {scenario['data']['Hour']}时, 星期{scenario['data']['DayOfWeek']+1}")

if __name__ == "__main__":
    main()
    create_real_prediction_example() 