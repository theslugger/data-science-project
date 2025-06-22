#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CDS503 Group Project - 首尔自行车需求预测
主实验启动脚本 - 运行所有4个实验

实验1: 机器学习算法比较 (成员A)
实验2: 特征选择方法比较 (成员B)  
实验3: 集成学习方法比较 (成员C)
实验4: 训练样本大小影响分析 (成员D)
"""

import sys
import time
import traceback
from datetime import datetime

def run_experiment_1():
    """运行实验1：机器学习算法比较"""
    print("\n" + "="*80)
    print("🚀 启动实验1：机器学习算法比较")
    print("="*80)
    
    try:
        from experiment_1_algorithm_comparison import AlgorithmComparison
        experiment = AlgorithmComparison()
        results = experiment.run_experiment(use_lag_features=False)
        
        print("✅ 实验1完成成功!")
        return True, results
        
    except Exception as e:
        print(f"❌ 实验1失败: {str(e)}")
        traceback.print_exc()
        return False, None

def run_experiment_2():
    """运行实验2：特征选择方法比较"""
    print("\n" + "="*80)
    print("🔍 启动实验2：特征选择方法比较")
    print("="*80)
    
    try:
        from experiment_2_feature_selection import FeatureSelectionExperiment
        experiment = FeatureSelectionExperiment()
        results = experiment.run_experiment()
        
        print("✅ 实验2完成成功!")
        return True, results
        
    except Exception as e:
        print(f"❌ 实验2失败: {str(e)}")
        traceback.print_exc()
        return False, None

def run_experiment_3():
    """运行实验3：集成学习方法比较"""
    print("\n" + "="*80)
    print("🎯 启动实验3：集成学习方法比较")
    print("="*80)
    
    try:
        from experiment_3_ensemble_learning import EnsembleLearningExperiment
        experiment = EnsembleLearningExperiment()
        results = experiment.run_experiment()
        
        print("✅ 实验3完成成功!")
        return True, results
        
    except Exception as e:
        print(f"❌ 实验3失败: {str(e)}")
        traceback.print_exc()
        return False, None

def run_experiment_4():
    """运行实验4：训练样本大小影响分析"""
    print("\n" + "="*80)
    print("📈 启动实验4：训练样本大小影响分析")
    print("="*80)
    
    try:
        from experiment_4_sample_size_analysis import SampleSizeAnalysis
        experiment = SampleSizeAnalysis()
        results = experiment.run_experiment()
        
        print("✅ 实验4完成成功!")
        return True, results
        
    except Exception as e:
        print(f"❌ 实验4失败: {str(e)}")
        traceback.print_exc()
        return False, None

def generate_overall_summary(results_dict):
    """生成总体实验总结"""
    print("\n" + "="*80)
    print("📊 生成总体实验总结")
    print("="*80)
    
    summary = {
        "experiment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiments_completed": len([k for k, v in results_dict.items() if v[0]]),
        "experiments_failed": len([k for k, v in results_dict.items() if not v[0]]),
        "details": {}
    }
    
    print(f"\n🎯 实验完成情况:")
    print(f"  总实验数: 4")
    print(f"  成功完成: {summary['experiments_completed']}")
    print(f"  失败数量: {summary['experiments_failed']}")
    
    print(f"\n📋 各实验状态:")
    for exp_name, (success, results) in results_dict.items():
        status = "✅ 成功" if success else "❌ 失败"
        result_count = len(results) if success and results else 0
        print(f"  {exp_name}: {status} ({result_count} 个结果)")
        
        summary["details"][exp_name] = {
            "success": success,
            "result_count": result_count
        }
    
    # 如果所有实验都成功，进行综合分析
    if summary['experiments_completed'] == 4:
        print(f"\n🏆 综合分析:")
        
        # 从各实验中提取最佳结果
        best_results = {}
        
        # 实验1的最佳算法
        if results_dict["实验1"][0] and results_dict["实验1"][1]:
            exp1_results = results_dict["实验1"][1]
            best_algorithm = min(exp1_results.items(), 
                               key=lambda x: x[1]['test_metrics']['RMSE'])
            best_results["最佳单一算法"] = {
                "name": best_algorithm[0],
                "rmse": best_algorithm[1]['test_metrics']['RMSE'],
                "r2": best_algorithm[1]['test_metrics']['R²']
            }
            print(f"  最佳单一算法: {best_algorithm[0]} (RMSE: {best_algorithm[1]['test_metrics']['RMSE']:.2f})")
        
        # 实验2的最佳特征选择
        if results_dict["实验2"][0] and results_dict["实验2"][1]:
            exp2_results = results_dict["实验2"][1]
            best_features = min(exp2_results.items(), 
                              key=lambda x: x[1]['test_metrics']['RMSE'])
            best_results["最佳特征选择"] = {
                "name": best_features[0],
                "rmse": best_features[1]['test_metrics']['RMSE'],
                "n_features": best_features[1]['n_features']
            }
            print(f"  最佳特征选择: {best_features[0]} (RMSE: {best_features[1]['test_metrics']['RMSE']:.2f}, {best_features[1]['n_features']}个特征)")
        
        # 实验3的最佳集成方法
        if results_dict["实验3"][0] and results_dict["实验3"][1]:
            exp3_results = results_dict["实验3"][1]
            best_ensemble = min(exp3_results.items(), 
                              key=lambda x: x[1]['test_metrics']['RMSE'])
            best_results["最佳集成方法"] = {
                "name": best_ensemble[0],
                "rmse": best_ensemble[1]['test_metrics']['RMSE'],
                "r2": best_ensemble[1]['test_metrics']['R²']
            }
            print(f"  最佳集成方法: {best_ensemble[0]} (RMSE: {best_ensemble[1]['test_metrics']['RMSE']:.2f})")
        
        # 检查目标达成情况
        target_rmse = 200
        target_r2 = 0.75
        
        print(f"\n🎯 目标达成分析 (RMSE < {target_rmse}, R² > {target_r2}):")
        
        successful_methods = []
        for method_type, result in best_results.items():
            if 'rmse' in result and 'r2' in result:
                rmse_ok = result['rmse'] < target_rmse
                r2_ok = result['r2'] > target_r2
                
                if rmse_ok and r2_ok:
                    successful_methods.append(method_type)
                    print(f"  ✅ {method_type}: 达标")
                else:
                    print(f"  ❌ {method_type}: RMSE={result['rmse']:.2f}, R²={result['r2']:.3f}")
        
        if successful_methods:
            print(f"\n🌟 {len(successful_methods)} 种方法成功达标!")
        else:
            print(f"\n⚠️ 没有方法完全达标，需要进一步优化")
        
        summary["best_results"] = best_results
        summary["target_achievement"] = {
            "target_rmse": target_rmse,
            "target_r2": target_r2,
            "successful_methods": successful_methods
        }
    
    # 保存总结
    import json
    with open('overall_experiment_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 总体实验总结已保存: overall_experiment_summary.json")
    
    return summary

def main():
    """主函数 - 运行所有实验"""
    print("🎉 CDS503 首尔自行车需求预测项目 - 启动所有实验")
    print("="*80)
    print("📅 开始时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("👥 团队成员:")
    print("  - 成员A: 机器学习算法比较")
    print("  - 成员B: 特征选择方法比较")
    print("  - 成员C: 集成学习方法比较")
    print("  - 成员D: 训练样本大小影响分析")
    print("="*80)
    
    start_time = time.time()
    
    # 存储所有实验结果
    results_dict = {}
    
    # 依次运行所有实验
    experiments = [
        ("实验1", run_experiment_1),
        ("实验2", run_experiment_2),
        ("实验3", run_experiment_3),
        ("实验4", run_experiment_4)
    ]
    
    for exp_name, exp_func in experiments:
        exp_start_time = time.time()
        
        print(f"\n⏰ 开始时间: {datetime.now().strftime('%H:%M:%S')}")
        success, results = exp_func()
        
        exp_duration = time.time() - exp_start_time
        print(f"⏱️ {exp_name}用时: {exp_duration:.1f}秒")
        
        results_dict[exp_name] = (success, results)
        
        if not success:
            print(f"⚠️ {exp_name}失败，继续下一个实验...")
        
        # 短暂休息，避免系统过载
        time.sleep(2)
    
    # 生成总体总结
    overall_summary = generate_overall_summary(results_dict)
    
    total_duration = time.time() - start_time
    
    print("\n" + "="*80)
    print("🎊 所有实验完成!")
    print("="*80)
    print(f"📅 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️ 总用时: {total_duration:.1f}秒 ({total_duration/60:.1f}分钟)")
    print(f"✅ 成功实验: {overall_summary['experiments_completed']}/4")
    
    if overall_summary['experiments_completed'] == 4:
        print("\n🏆 恭喜！所有实验都成功完成！")
        print("📄 请查看以下输出文件：")
        print("  📊 可视化图表: experiment_*_*.png")
        print("  📈 结果数据: experiment_*_results.csv")
        print("  📋 详细报告: experiment_*_summary.json")
        print("  📑 总体总结: overall_experiment_summary.json")
    else:
        print(f"\n⚠️ 有 {4 - overall_summary['experiments_completed']} 个实验失败")
        print("请检查错误信息并重新运行失败的实验")
    
    print("\n🎯 项目目标检查:")
    print("  - 算法比较: ✅ 完成" if results_dict["实验1"][0] else "  - 算法比较: ❌ 未完成")
    print("  - 特征选择: ✅ 完成" if results_dict["实验2"][0] else "  - 特征选择: ❌ 未完成")
    print("  - 集成学习: ✅ 完成" if results_dict["实验3"][0] else "  - 集成学习: ❌ 未完成")
    print("  - 样本分析: ✅ 完成" if results_dict["实验4"][0] else "  - 样本分析: ❌ 未完成")
    
    print("\n📚 感谢使用！祝项目顺利完成！")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⛔ 用户中断实验")
        print("已完成的实验结果已保存")
    except Exception as e:
        print(f"\n\n💥 程序异常: {str(e)}")
        traceback.print_exc()
        print("请检查错误信息并重新运行") 