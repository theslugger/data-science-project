#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图表生成器
CDS503 Group Project - 首尔自行车需求预测集成学习研究

生成高质量图表（≥300 DPI）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon
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
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体

class FigureGenerator:
    """图表生成器"""
    
    def __init__(self, results_file='optimized_90f_ensemble_results.csv'):
        self.results_file = results_file
        self.results_df = None
        self.figures_dir = 'paper_figures'
        
        # 创建目录
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
            print("请先运行 experiment_3_optimized_ensemble.py 生成结果")
            return False
    

    
    def generate_performance_comparison_figure(self):
        """生成性能对比图（高质量）"""
        if self.results_df is None:
            return
        
        # 创建综合性能对比图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Ensemble Learning Performance Comparison\n(Seoul Bike Sharing Demand Prediction)', 
                     fontsize=16, fontweight='bold')
        
        # 1. RMSE对比（按类型）
        ax1 = axes[0, 0]
        type_rmse = self.results_df.groupby('Type')['Test_RMSE'].agg(['mean', 'std', 'min'])
        bars = ax1.bar(type_rmse.index, type_rmse['mean'], yerr=type_rmse['std'], 
                      capsize=4, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        ax1.set_title('Average Test RMSE by Method Type', fontweight='bold')
        ax1.set_ylabel('RMSE')
        ax1.set_xlabel('Method Type')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, mean_val in zip(bars, type_rmse['mean']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{mean_val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. R²对比（按类型）
        ax2 = axes[0, 1]
        type_r2 = self.results_df.groupby('Type')['Test_R²'].agg(['mean', 'std', 'max'])
        bars2 = ax2.bar(type_r2.index, type_r2['mean'], yerr=type_r2['std'], 
                       capsize=4, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        ax2.set_title('Average Test R² by Method Type', fontweight='bold')
        ax2.set_ylabel('R² Score')
        ax2.set_xlabel('Method Type')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, mean_val in zip(bars2, type_r2['mean']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. 前10个模型性能
        ax3 = axes[1, 0]
        top10 = self.results_df.nsmallest(10, 'Test_RMSE')
        bars3 = ax3.barh(range(len(top10)), top10['Test_RMSE'], 
                        color=plt.cm.viridis(np.linspace(0, 1, len(top10))))
        ax3.set_yticks(range(len(top10)))
        ax3.set_yticklabels([name[:25] + '...' if len(name) > 25 else name 
                            for name in top10['Model']], fontsize=9)
        ax3.set_xlabel('Test RMSE')
        ax3.set_title('Top 10 Models by Test RMSE', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, (idx, row) in enumerate(top10.iterrows()):
            ax3.text(row['Test_RMSE'] + 1, i, f'{row["Test_RMSE"]:.1f}', 
                    va='center', fontsize=8)
        
        # 4. 训练时间 vs 性能散点图
        ax4 = axes[1, 1]
        type_colors = {'Base Model': '#2E86AB', 'Bagging': '#A23B72', 
                      'Voting': '#F18F01', 'Stacking': '#C73E1D'}
        
        for method_type in self.results_df['Type'].unique():
            type_data = self.results_df[self.results_df['Type'] == method_type]
            ax4.scatter(type_data['Training_Time'], type_data['Test_RMSE'], 
                       c=type_colors.get(method_type, 'gray'), 
                       label=method_type, s=60, alpha=0.7)
        
        ax4.set_xlabel('Training Time (seconds)')
        ax4.set_ylabel('Test RMSE')
        ax4.set_title('Performance vs Training Time Trade-off', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/performance_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f'{self.figures_dir}/performance_comparison.pdf', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print("✅ 性能对比图已生成 (300+ DPI)")
    
    def generate_all_figures(self):
        """生成所有图表"""
        if not self.load_results():
            return
        
        print("\n🚀 开始生成高质量图表...")
        print("="*60)
        
        # 生成性能对比图
        print("生成性能对比图...")
        self.generate_performance_comparison_figure()
        
        print("\n" + "="*60)
        print("🎉 图表生成完成！")
        print("="*60)
        print(f"📁 图表文件夹: {self.figures_dir}/")
    


def main():
    """主函数"""
    print("📝 图表生成器")
    print("="*50)
    
    # 创建图表生成器
    generator = FigureGenerator()
    
    # 生成所有图表
    generator.generate_all_figures()

if __name__ == "__main__":
    main() 