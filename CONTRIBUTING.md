# 贡献指南 / Contributing Guide

感谢您对项目的兴趣！我们欢迎各种形式的贡献。

## 🤝 如何贡献

### 报告问题
- 使用GitHub Issues报告bug
- 提供详细的重现步骤
- 包含环境信息

### 提出功能请求
- 在Issues中描述新功能
- 说明使用场景和需求
- 讨论实现方案

### 代码贡献
1. Fork项目
2. 创建功能分支: `git checkout -b feature/amazing-feature`
3. 提交更改: `git commit -m 'Add amazing feature'`
4. 推送分支: `git push origin feature/amazing-feature`
5. 创建Pull Request

## 📝 提交规范

使用以下前缀：
- `✨ feat:` 新功能
- `🐛 fix:` 修复bug
- `📝 docs:` 文档更新
- `🎨 style:` 代码格式
- `♻️ refactor:` 重构
- `⚡ perf:` 性能优化
- `🧪 test:` 测试相关

## 🧪 测试

提交前请确保：
```bash
# 运行测试
python -m pytest

# 检查代码风格
flake8 .
```

## 📋 开发环境

```bash
# 克隆仓库
git clone https://github.com/theslugger/data-science-project.git

# 安装依赖
pip install -r requirements.txt

# 运行项目
python main_pipeline.py
```

谢谢您的贡献！🙏 