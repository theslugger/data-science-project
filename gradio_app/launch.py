#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio应用启动脚本
自动检查依赖并启动应用
"""

import subprocess
import sys
import os

def check_dependencies():
    """检查所需依赖包"""
    required_packages = [
        'gradio', 'pandas', 'numpy', 'matplotlib', 
        'seaborn', 'plotly', 'scipy', 'scikit-learn'
    ]
    
    missing_packages = []
    print("🔍 检查依赖包...")
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package:15s} - 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package:15s} - 未安装")
    
    return missing_packages

def install_dependencies():
    """安装缺失的依赖包"""
    try:
        print("\n📦 开始安装依赖包...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ 依赖包安装完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败: {str(e)}")
        return False

def launch_app():
    """启动Gradio应用"""
    print("\n🚀 正在启动应用...")
    print("📱 应用将在浏览器中自动打开")
    print("🌐 本地访问地址: http://localhost:7860")
    print("🛑 按 Ctrl+C 停止应用\n")
    
    try:
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n👋 应用已停止运行")
    except FileNotFoundError:
        print("❌ 找不到app.py文件，请确保在正确目录中运行")
    except Exception as e:
        print(f"❌ 启动失败: {str(e)}")

def main():
    """主函数"""
    print("🚴‍♂️ 首尔自行车需求预测数据处理平台")
    print("🌟 Gradio可视化应用启动器")
    print("=" * 60)
    
    # 检查关键文件
    if not os.path.exists("app.py"):
        print("❌ 错误: 找不到app.py文件")
        print("💡 请确保在gradio_app目录中运行此脚本")
        return
    
    if not os.path.exists("requirements.txt"):
        print("❌ 错误: 找不到requirements.txt文件")
        return
    
    # 检查依赖
    missing = check_dependencies()
    
    if missing:
        print(f"\n⚠️  发现 {len(missing)} 个缺失的依赖包:")
        for pkg in missing:
            print(f"   • {pkg}")
        
        user_input = input("\n是否自动安装缺失的依赖包? (y/n): ").lower().strip()
        
        if user_input in ['y', 'yes', '是', 'Y']:
            if install_dependencies():
                print("\n🎉 依赖安装成功，准备启动应用...")
            else:
                print("\n❌ 依赖安装失败，请手动安装后重试")
                print("💡 手动安装命令: pip install -r requirements.txt")
                return
        else:
            print("\n❌ 请手动安装依赖后重新运行启动脚本")
            print("💡 安装命令: pip install -r requirements.txt")
            return
    else:
        print("\n🎉 所有依赖包已安装完毕!")
    
    # 启动应用
    launch_app()

if __name__ == "__main__":
    main()