#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
首尔自行车需求预测 - 主流程控制脚本
CDS503 Group Project - 完整数据分析流水线
"""

import sys
import argparse
from pathlib import Path
import warnings

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config import config
from utils import Logger, print_section_header, get_timestamp

warnings.filterwarnings('ignore')

class BikeDataPipeline:
    """自行车数据分析主流水线"""
    
    def __init__(self, verbose=True):
        self.logger = Logger(self.__class__.__name__)
        self.verbose = verbose
        self.pipeline_results = {}
        
        # 确保输出目录存在
        config.create_directories()
        
    def run_data_exploration(self, save_results=True):
        """运行数据探索分析"""
        print_section_header("第一阶段：数据探索分析 (EDA)", level=1)
        
        try:
            from enhanced_data_exploration import EnhancedDataExplorer
            
            explorer = EnhancedDataExplorer()
            
            # 执行完整的EDA流程
            self.logger.info("开始数据探索分析...")
            
            # 数据加载与验证
            df = explorer.load_and_validate_data()
            
            # 目标变量分析
            target_analysis = explorer.analyze_target_variable()
            
            # 数值特征分析
            numeric_analysis, correlations = explorer.analyze_numerical_features()
            
            # 分类特征分析
            categorical_analysis = explorer.analyze_categorical_features()
            
            # 时间模式分析
            time_analysis = explorer.analyze_time_patterns()
            
            # 天气影响分析
            weather_analysis = explorer.analyze_weather_impact()
            
            # 生成综合报告
            eda_results = explorer.generate_comprehensive_report()
            
            self.pipeline_results['eda'] = eda_results
            self.logger.info("✅ 数据探索分析完成")
            
            return eda_results
            
        except Exception as e:
            self.logger.error(f"❌ 数据探索分析失败: {str(e)}")
            raise
    
    def run_deep_analysis(self, save_results=True):
        """运行深度数据分析"""
        print_section_header("第二阶段：深度数据洞察分析", level=1)
        
        try:
            from enhanced_data_analysis import EnhancedDataAnalyzer
            
            analyzer = EnhancedDataAnalyzer()
            
            # 执行深度分析流程
            self.logger.info("开始深度数据分析...")
            
            # 数据加载与准备
            df = analyzer.load_and_prepare_data()
            
            # 深度目标变量分析
            target_analysis = analyzer.deep_target_analysis()
            
            # 高级时间模式分析
            time_analysis = analyzer.advanced_time_pattern_analysis()
            
            # 天气影响深度挖掘
            weather_analysis = analyzer.weather_impact_deep_dive()
            
            # 需求模式分割分析
            segmentation_analysis = analyzer.demand_pattern_segmentation()
            
            # 预测性洞察分析
            predictive_analysis = analyzer.predictive_insights_analysis()
            
            # 生成综合洞察报告
            deep_analysis_results = analyzer.generate_comprehensive_insights_report()
            
            self.pipeline_results['deep_analysis'] = deep_analysis_results
            self.logger.info("✅ 深度数据分析完成")
            
            return deep_analysis_results
            
        except Exception as e:
            self.logger.error(f"❌ 深度数据分析失败: {str(e)}")
            raise
    
    def run_data_preprocessing(self, use_lag_features=False, feature_selection_method='correlation'):
        """运行数据预处理"""
        print_section_header("第三阶段：智能数据预处理", level=1)
        
        try:
            from enhanced_data_preprocessing import EnhancedDataPreprocessor
            
            preprocessor = EnhancedDataPreprocessor()
            
            # 执行预处理流程
            self.logger.info("开始数据预处理...")
            
            # 运行完整预处理流水线
            (X_train, X_val, X_test, y_train, y_val, y_test, preprocessing_results) = \
                preprocessor.preprocess_pipeline(
                    use_lag_features=use_lag_features,
                    feature_selection_method=feature_selection_method,
                    feature_selection_k=None
                )
            
            self.pipeline_results['preprocessing'] = {
                'results': preprocessing_results,
                'data_shapes': {
                    'X_train': X_train.shape,
                    'X_val': X_val.shape,
                    'X_test': X_test.shape,
                    'y_train': y_train.shape,
                    'y_val': y_val.shape,
                    'y_test': y_test.shape
                },
                'feature_names': preprocessing_results['feature_names']
            }
            
            self.logger.info("✅ 数据预处理完成")
            
            return (X_train, X_val, X_test, y_train, y_val, y_test, preprocessing_results)
            
        except Exception as e:
            self.logger.error(f"❌ 数据预处理失败: {str(e)}")
            raise
    
    def run_complete_pipeline(self, include_deep_analysis=True, 
                            use_lag_features=False, 
                            feature_selection_method='correlation'):
        """运行完整的数据分析流水线"""
        print_section_header("首尔自行车需求预测 - 完整数据分析流水线", level=1)
        
        timestamp = get_timestamp()
        pipeline_start_time = timestamp
        
        try:
            # 阶段1：数据探索分析
            self.logger.info("🚀 启动完整数据分析流水线...")
            eda_results = self.run_data_exploration()
            
            # 阶段2：深度数据分析（可选）
            if include_deep_analysis:
                deep_analysis_results = self.run_deep_analysis()
            else:
                self.logger.info("⏭️  跳过深度数据分析")
                deep_analysis_results = None
            
            # 阶段3：数据预处理
            preprocessing_data = self.run_data_preprocessing(
                use_lag_features=use_lag_features,
                feature_selection_method=feature_selection_method
            )
            
            # 整合所有结果
            complete_results = {
                'pipeline_info': {
                    'timestamp': pipeline_start_time,
                    'stages_completed': ['eda', 'deep_analysis', 'preprocessing'] if include_deep_analysis else ['eda', 'preprocessing'],
                    'configuration': {
                        'include_deep_analysis': include_deep_analysis,
                        'use_lag_features': use_lag_features,
                        'feature_selection_method': feature_selection_method
                    }
                },
                'eda_results': eda_results,
                'deep_analysis_results': deep_analysis_results,
                'preprocessing_results': self.pipeline_results['preprocessing']
            }
            
            # 生成流水线总结报告
            self.generate_pipeline_summary(complete_results)
            
            # 保存完整结果
            from utils import ResultSaver
            result_saver = ResultSaver(self.logger)
            result_saver.save_json(complete_results, f"complete_pipeline_results_{pipeline_start_time}", "pipeline")
            
            self.logger.info("🎉 完整数据分析流水线执行成功！")
            
            return complete_results, preprocessing_data
            
        except Exception as e:
            self.logger.error(f"❌ 流水线执行失败: {str(e)}")
            raise
    
    def generate_pipeline_summary(self, results):
        """生成流水线总结报告"""
        print_section_header("数据分析流水线总结报告", level=1)
        
        pipeline_info = results['pipeline_info']
        eda_results = results['eda_results']
        preprocessing_results = results['preprocessing_results']['results']
        
        print("🎯 流水线执行摘要:")
        print(f"  执行时间戳: {pipeline_info['timestamp']}")
        print(f"  完成阶段: {', '.join(pipeline_info['stages_completed'])}")
        
        print(f"\n📊 数据探索关键发现:")
        if 'basic_info' in eda_results:
            basic_info = eda_results['basic_info']
            print(f"  原始数据规模: {basic_info['shape']}")
            print(f"  数据完整度: {'100%' if not basic_info['has_missing_values'] else '存在缺失值'}")
        
        if 'target_analysis' in eda_results:
            target_info = eda_results['target_analysis']
            zero_pct = target_info['basic_stats']['零值比例(%)']
            outlier_pct = target_info['outliers']['percentage']
            print(f"  目标变量零值: {zero_pct:.1f}%")
            print(f"  目标变量异常值: {outlier_pct:.1f}%")
        
        if 'numerical_analysis' in eda_results:
            num_info = eda_results['numerical_analysis']
            strong_corr_count = len(num_info['strong_correlations'])
            print(f"  强相关特征数量: {strong_corr_count}")
        
        print(f"\n🔧 预处理执行摘要:")
        feature_counts = preprocessing_results['feature_counts']
        print(f"  最终特征数量: {feature_counts['selected_features']}")
        print(f"  时间特征: {feature_counts['time_features'] + feature_counts['advanced_time_features']}")
        print(f"  天气特征: {feature_counts['weather_features'] + feature_counts['comfort_features']}")
        print(f"  交互特征: {feature_counts['interaction_features']}")
        print(f"  滞后特征: {feature_counts['lag_features']}")
        
        data_shapes = results['preprocessing_results']['data_shapes']
        print(f"\n📈 最终数据集:")
        print(f"  训练集: {data_shapes['X_train']}")
        print(f"  验证集: {data_shapes['X_val']}")
        print(f"  测试集: {data_shapes['X_test']}")
        
        print(f"\n💡 建模准备建议:")
        if results.get('deep_analysis_results') and 'predictive_insights' in results['deep_analysis_results']:
            strategies = results['deep_analysis_results']['predictive_insights']['modeling_strategies']
            for i, strategy in enumerate(strategies[:5], 1):
                print(f"  {i}. {strategy}")
        else:
            print("  1. 使用时间序列交叉验证")
            print("  2. 考虑集成学习方法")
            print("  3. 注意特征工程的有效性")
            print("  4. 监控模型过拟合")
            print("  5. 重点关注高峰时段预测")
        
        print(f"\n🎯 下一步行动:")
        print("  1. 运行实验1：机器学习算法比较")
        print("  2. 运行实验2：特征选择优化")
        print("  3. 运行实验3：集成学习方法")
        print("  4. 运行实验4：训练样本大小分析")
        
        print(f"\n✅ 数据分析流水线完成，可以开始模型训练实验！")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='首尔自行车需求预测 - 数据分析流水线')
    
    parser.add_argument('--stage', type=str, choices=['eda', 'deep', 'preprocess', 'all'], 
                       default='all', help='执行的阶段')
    parser.add_argument('--no-deep-analysis', action='store_true', 
                       help='跳过深度数据分析')
    parser.add_argument('--use-lag-features', action='store_true',
                       help='使用滞后特征（注意数据泄露风险）')
    parser.add_argument('--feature-selection', type=str, 
                       choices=['correlation', 'univariate', 'rfe', 'none'],
                       default='correlation', help='特征选择方法')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='详细输出')
    
    args = parser.parse_args()
    
    # 创建流水线实例
    pipeline = BikeDataPipeline(verbose=args.verbose)
    
    try:
        if args.stage == 'eda':
            # 只运行数据探索
            results = pipeline.run_data_exploration()
            print("✅ 数据探索分析完成")
            
        elif args.stage == 'deep':
            # 只运行深度分析
            results = pipeline.run_deep_analysis()
            print("✅ 深度数据分析完成")
            
        elif args.stage == 'preprocess':
            # 只运行预处理
            results = pipeline.run_data_preprocessing(
                use_lag_features=args.use_lag_features,
                feature_selection_method=args.feature_selection
            )
            print("✅ 数据预处理完成")
            
        else:  # args.stage == 'all'
            # 运行完整流水线
            complete_results, preprocessing_data = pipeline.run_complete_pipeline(
                include_deep_analysis=not args.no_deep_analysis,
                use_lag_features=args.use_lag_features,
                feature_selection_method=args.feature_selection
            )
            print("✅ 完整流水线执行成功")
        
        return 0
        
    except Exception as e:
        print(f"❌ 流水线执行失败: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 