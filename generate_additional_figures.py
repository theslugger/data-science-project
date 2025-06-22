#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
额外图表生成器 - 高质量可视化
生成额外的高质量图表和分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# 设置高质量图表参数
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

class AdditionalFigureGenerator:
    """额外图表生成器"""
    
    def __init__(self, results_file='optimized_90f_ensemble_results.csv'):
        self.results_file = results_file
        self.results_df = None
        self.figures_dir = 'paper_figures'
        
        import os
        os.makedirs(self.figures_dir, exist_ok=True)
    
    def load_results(self):
        """加载实验结果"""
        try:
            self.results_df = pd.read_csv(self.results_file)
            print(f"✅ 成功加载 {len(self.results_df)} 个模型的结果")
            return True
        except FileNotFoundError:
            print(f"❌ 找不到结果文件: {self.results_file}")
            return False
    
    def generate_detailed_performance_table_figure(self):
        """生成详细性能表格图"""
        if self.results_df is None:
            return
        
        # 创建表格样式的图
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.axis('tight')
        ax.axis('off')
        
        # 准备表格数据
        table_data = self.results_df.copy()
        table_data = table_data.sort_values('Test_RMSE')
        
        # 选择要显示的列
        display_cols = ['Model', 'Type', 'Test_RMSE', 'Test_R²', 'Test_MAE', 'Test_MAPE', 'Training_Time']
        table_subset = table_data[display_cols].head(15)  # 显示前15个模型
        
        # 格式化数值
        table_subset = table_subset.copy()
        table_subset['Test_RMSE'] = table_subset['Test_RMSE'].round(2)
        table_subset['Test_R²'] = table_subset['Test_R²'].round(4)
        table_subset['Test_MAE'] = table_subset['Test_MAE'].round(2)
        table_subset['Test_MAPE'] = table_subset['Test_MAPE'].round(2)
        table_subset['Training_Time'] = table_subset['Training_Time'].round(2)
        
        # 创建表格
        table = ax.table(cellText=table_subset.values,
                        colLabels=display_cols,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # 设置表头样式
        for i in range(len(display_cols)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置行颜色交替
        for i in range(1, len(table_subset) + 1):
            for j in range(len(display_cols)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('white')
                
                # 高亮最佳值
                if i == 1:  # 第一行（最佳模型）
                    table[(i, j)].set_facecolor('#FFE082')
                    table[(i, j)].set_text_props(weight='bold')
        
        plt.title('Top 15 Models - Detailed Performance Comparison\n(Sorted by Test RMSE)', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/detailed_performance_table.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f'{self.figures_dir}/detailed_performance_table.pdf', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print("✅ 详细性能表格图已生成")
    
    def generate_method_comparison_boxplot(self):
        """生成方法类型对比箱线图"""
        if self.results_df is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Statistical Distribution Analysis by Method Type', 
                     fontsize=16, fontweight='bold')
        
        metrics = ['Test_RMSE', 'Test_R²', 'Test_MAE', 'Test_MAPE']
        titles = ['Test RMSE Distribution', 'Test R² Distribution', 
                 'Test MAE Distribution', 'Test MAPE Distribution']
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
            ax = axes[idx // 2, idx % 2]
            
            # 准备数据
            data_by_type = []
            labels = []
            for method_type in self.results_df['Type'].unique():
                type_data = self.results_df[self.results_df['Type'] == method_type][metric]
                data_by_type.append(type_data.values)
                labels.append(f'{method_type}\n(n={len(type_data)})')
            
            # 创建箱线图
            bp = ax.boxplot(data_by_type, labels=labels, patch_artist=True)
            
            # 设置颜色
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel(metric.replace('Test_', '').replace('_', ' '))
            ax.grid(True, alpha=0.3)
            
            # 添加均值点
            means = [np.mean(data) for data in data_by_type]
            ax.scatter(range(1, len(means) + 1), means, 
                      color='red', marker='D', s=50, zorder=10, label='Mean')
            
            # 添加数值标签
            for i, mean_val in enumerate(means):
                ax.text(i + 1, mean_val, f'{mean_val:.2f}', 
                       ha='center', va='bottom', fontweight='bold', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/method_comparison_boxplot.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f'{self.figures_dir}/method_comparison_boxplot.pdf', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print("✅ 方法类型对比箱线图已生成")
    
    def generate_correlation_heatmap(self):
        """生成性能指标相关性热图"""
        if self.results_df is None:
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 选择性能指标
        metrics = ['Test_RMSE', 'Test_MAE', 'Test_R²', 'Test_MAPE', 'Test_SMAPE', 'Training_Time']
        corr_matrix = self.results_df[metrics].corr()
        
        # 创建热图
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 只显示下三角
        
        heatmap = sns.heatmap(corr_matrix, 
                             mask=mask,
                             annot=True, 
                             cmap='RdBu_r', 
                             center=0,
                             square=True,
                             fmt='.3f',
                             cbar_kws={"shrink": .8},
                             ax=ax)
        
        # 设置标签
        ax.set_title('Performance Metrics Correlation Matrix', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # 旋转标签
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/correlation_heatmap.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f'{self.figures_dir}/correlation_heatmap.pdf', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print("✅ 相关性热图已生成")
    
    def generate_ensemble_effectiveness_analysis(self):
        """生成集成学习有效性分析图"""
        if self.results_df is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Ensemble Learning Effectiveness Analysis', 
                     fontsize=16, fontweight='bold')
        
        # 1. 基础模型 vs 集成模型性能对比
        ax1 = axes[0, 0]
        base_models = self.results_df[self.results_df['Type'] == 'Base Model']
        ensemble_models = self.results_df[self.results_df['Type'] != 'Base Model']
        
        base_rmse = base_models['Test_RMSE'].values
        ensemble_rmse = ensemble_models['Test_RMSE'].values
        
        ax1.hist(base_rmse, alpha=0.7, label=f'Base Models (n={len(base_rmse)})', 
                bins=10, color='#FF6B6B', edgecolor='black')
        ax1.hist(ensemble_rmse, alpha=0.7, label=f'Ensemble Models (n={len(ensemble_rmse)})', 
                bins=10, color='#4ECDC4', edgecolor='black')
        
        ax1.axvline(np.mean(base_rmse), color='red', linestyle='--', 
                   label=f'Base Mean: {np.mean(base_rmse):.2f}')
        ax1.axvline(np.mean(ensemble_rmse), color='teal', linestyle='--', 
                   label=f'Ensemble Mean: {np.mean(ensemble_rmse):.2f}')
        
        ax1.set_xlabel('Test RMSE')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Base vs Ensemble Models RMSE Distribution', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 集成方法类型对比
        ax2 = axes[0, 1]
        ensemble_types = ['Bagging', 'Voting', 'Stacking']
        ensemble_means = []
        ensemble_stds = []
        
        for ens_type in ensemble_types:
            type_data = self.results_df[self.results_df['Type'] == ens_type]['Test_RMSE']
            if len(type_data) > 0:
                ensemble_means.append(type_data.mean())
                ensemble_stds.append(type_data.std())
            else:
                ensemble_means.append(0)
                ensemble_stds.append(0)
        
        bars = ax2.bar(ensemble_types, ensemble_means, yerr=ensemble_stds, 
                      capsize=5, color=['#FF9999', '#66B2FF', '#99FF99'], 
                      edgecolor='black', alpha=0.8)
        
        # 添加基础模型平均线
        ax2.axhline(np.mean(base_rmse), color='red', linestyle='--', 
                   label=f'Base Model Average: {np.mean(base_rmse):.2f}')
        
        ax2.set_ylabel('Average Test RMSE')
        ax2.set_title('Ensemble Method Comparison', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, mean_val in zip(bars, ensemble_means):
            if mean_val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{mean_val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. 性能改进分析
        ax3 = axes[1, 0]
        
        # 计算每种集成方法相对于基础模型的改进
        base_best = base_models['Test_RMSE'].min()
        improvements = []
        method_names = []
        
        for ens_type in ensemble_types:
            type_data = self.results_df[self.results_df['Type'] == ens_type]
            if len(type_data) > 0:
                ens_best = type_data['Test_RMSE'].min()
                improvement = ((base_best - ens_best) / base_best) * 100
                improvements.append(improvement)
                method_names.append(f'{ens_type}\n({improvement:.1f}%)')
            else:
                improvements.append(0)
                method_names.append(f'{ens_type}\n(0.0%)')
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax3.bar(range(len(method_names)), improvements, 
                      color=colors, alpha=0.7, edgecolor='black')
        
        ax3.set_xticks(range(len(method_names)))
        ax3.set_xticklabels(method_names)
        ax3.set_ylabel('RMSE Improvement (%)')
        ax3.set_title('Performance Improvement over Best Base Model', fontweight='bold')
        ax3.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax3.grid(True, alpha=0.3)
        
        # 4. 训练时间 vs 性能改进
        ax4 = axes[1, 1]
        
        # 计算每个模型相对于最差基础模型的改进
        base_worst = base_models['Test_RMSE'].max()
        
        for method_type in self.results_df['Type'].unique():
            if method_type != 'Base Model':
                type_data = self.results_df[self.results_df['Type'] == method_type]
                improvements = ((base_worst - type_data['Test_RMSE']) / base_worst) * 100
                training_times = type_data['Training_Time']
                
                ax4.scatter(training_times, improvements, 
                           label=method_type, s=60, alpha=0.7)
        
        ax4.set_xlabel('Training Time (seconds)')
        ax4.set_ylabel('RMSE Improvement over Worst Base Model (%)')
        ax4.set_title('Training Time vs Performance Improvement', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/ensemble_effectiveness_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f'{self.figures_dir}/ensemble_effectiveness_analysis.pdf', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print("✅ 集成学习有效性分析图已生成")
    
    def generate_model_ranking_figure(self):
        """生成模型排名图"""
        if self.results_df is None:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('Model Performance Ranking Analysis', 
                     fontsize=16, fontweight='bold')
        
        # 1. 前10名模型排名
        ax1 = axes[0]
        top10 = self.results_df.nsmallest(10, 'Test_RMSE')
        
        # 创建颜色映射
        type_colors = {'Base Model': '#FF6B6B', 'Bagging': '#4ECDC4', 
                      'Voting': '#45B7D1', 'Stacking': '#96CEB4'}
        colors = [type_colors.get(t, 'gray') for t in top10['Type']]
        
        bars = ax1.barh(range(len(top10)), top10['Test_RMSE'], color=colors, alpha=0.8)
        
        # 设置标签
        ax1.set_yticks(range(len(top10)))
        ax1.set_yticklabels([f"{i+1}. {name[:30]}{'...' if len(name) > 30 else ''}" 
                            for i, name in enumerate(top10['Model'])])
        ax1.set_xlabel('Test RMSE')
        ax1.set_title('Top 10 Models by RMSE Performance', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, (bar, rmse, r2) in enumerate(zip(bars, top10['Test_RMSE'], top10['Test_R²'])):
            ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'RMSE: {rmse:.2f}\nR²: {r2:.3f}', 
                    va='center', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # 添加图例
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, label=method) 
                          for method, color in type_colors.items()]
        ax1.legend(handles=legend_elements, loc='lower right')
        
        # 2. 性能分布雷达图（前5名）
        ax2 = axes[1]
        top5 = self.results_df.nsmallest(5, 'Test_RMSE')
        
        # 标准化指标（0-1范围）
        metrics = ['Test_RMSE', 'Test_MAE', 'Test_MAPE']
        normalized_data = top5[metrics].copy()
        
        # RMSE和MAE需要反转（越小越好）
        for metric in ['Test_RMSE', 'Test_MAE', 'Test_MAPE']:
            max_val = self.results_df[metric].max()
            min_val = self.results_df[metric].min()
            normalized_data[metric] = 1 - (top5[metric] - min_val) / (max_val - min_val)
        
        # R²直接标准化（越大越好）
        r2_max = self.results_df['Test_R²'].max()
        r2_min = self.results_df['Test_R²'].min()
        normalized_data['Test_R²'] = (top5['Test_R²'] - r2_min) / (r2_max - r2_min)
        
        # 创建散点图显示多维性能
        x_vals = normalized_data['Test_RMSE']
        y_vals = normalized_data['Test_R²']
        sizes = (1 - normalized_data['Test_MAE']) * 300 + 50  # 基于MAE的点大小
        colors_scatter = [type_colors.get(t, 'gray') for t in top5['Type']]
        
        scatter = ax2.scatter(x_vals, y_vals, s=sizes, c=colors_scatter, alpha=0.7, edgecolors='black')
        
        ax2.set_xlabel('Normalized RMSE Performance (higher is better)')
        ax2.set_ylabel('Normalized R² Performance (higher is better)')
        ax2.set_title('Top 5 Models - Multi-dimensional Performance\n(Bubble size: MAE performance)', 
                     fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 添加模型标签
        for i, (x, y, name) in enumerate(zip(x_vals, y_vals, top5['Model'])):
            ax2.annotate(f'{i+1}. {name[:20]}...', 
                        xy=(x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, ha='left',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/model_ranking_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f'{self.figures_dir}/model_ranking_analysis.pdf', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print("✅ 模型排名分析图已生成")
    
    def generate_all_additional_figures(self):
        """生成所有额外图表"""
        if not self.load_results():
            return
        
        print("\n🎨 开始生成额外的图表...")
        print("="*60)
        
        print("1. 生成详细性能表格图...")
        self.generate_detailed_performance_table_figure()
        
        print("\n2. 生成方法类型对比箱线图...")
        self.generate_method_comparison_boxplot()
        
        print("\n3. 生成相关性热图...")
        self.generate_correlation_heatmap()
        
        print("\n4. 生成集成学习有效性分析图...")
        self.generate_ensemble_effectiveness_analysis()
        
        print("\n5. 生成模型排名分析图...")
        self.generate_model_ranking_figure()
        
        print("\n" + "="*60)
        print("🎉 所有额外图表生成完成！")
        print("="*60)
        print(f"📁 所有图表保存在: {self.figures_dir}/")
        print("\n新生成的图表:")
        print("  📋 detailed_performance_table.png/pdf")
        print("  📊 method_comparison_boxplot.png/pdf")
        print("  🔥 correlation_heatmap.png/pdf")
        print("  📈 ensemble_effectiveness_analysis.png/pdf")
        print("  🏆 model_ranking_analysis.png/pdf")

def main():
    """主函数"""
    print("🎨 额外图表生成器")
    print("="*50)
    
    generator = AdditionalFigureGenerator()
    generator.generate_all_additional_figures()

if __name__ == "__main__":
    main() 