# 规划模板文件说明

本目录包含用于 `run_flow` 模式的规划模板文件。

## 文件列表

| 文件名 | 用途 | 适用场景 |
|--------|------|----------|
| `customer_service_template.txt` | 客户服务流程 | 处理客户投诉、咨询、满意度调查等 |
| `data_analysis_template.txt` | 数据分析流程 | 销售分析、用户行为分析、业务指标分析等 |
| `software_development_template.txt` | 软件开发流程 | 功能开发、Bug修复、代码重构等 |
| `simple_example.txt` | 简单示例 | 学习模板格式和基本用法 |

## 使用方法

### 方法1：使用测试脚本（推荐）

```bash
python examples/test_run_flow_with_template.py
```

### 方法2：直接运行主程序

```bash
python run_flow.py
# 输入模板文件路径：examples/planning_templates/customer_service_template.txt
```

## 模板格式

所有模板文件都遵循以下格式：

```markdown
# 业务名称规划模板

## 业务概述
说明模板用途

## 标准流程步骤
### 1. 阶段名称
- 步骤描述

## 注意事项
- 重要规则

## 质量标准
- 质量指标
```

## 自定义模板

您可以基于现有模板创建符合自己业务需求的模板：
1. 复制现有模板文件
2. 修改业务概述和流程步骤
3. 添加您的业务规则和质量标准
4. 保存为新文件

## 更多信息

- 快速开始：`../QUICK_START.md`
- 详细文档：`../README_planning_templates.md`
- 修改记录：`../../MODIFICATION_RECORD.md`
