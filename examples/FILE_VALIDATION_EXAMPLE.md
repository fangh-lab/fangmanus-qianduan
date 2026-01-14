# 文件要素完整性检查示例

## 概述

这是一个文件要素完整性检查的业务示例，展示如何使用规划模板和业务文件进行文件验证任务。

## 文件说明

### 规划模板
- `examples/planning_templates/file_validation_template.txt` - 文件验证流程模板

### 业务文件（workspace/fileexample目录）
- `sample_config.json` - 完整的配置文件示例（参考标准）
- `incomplete_config.json` - 待检查的配置文件（有缺失字段）
- `validation_requirements.txt` - 检查要求和规则

**注意**：所有文件都放在 `workspace/fileexample/` 子目录中，生成的文件也会保存在此目录。

## 使用示例

### 运行命令

```bash
python run_flow.py
```

### 交互流程

```
Enter planning template file path: examples/planning_templates/file_validation_template.txt

Workspace directory: D:\fanghao\gihub\fanghaomanus\workspace
Enter business file paths (relative to workspace, one per line, empty line to finish):
  > fileexample/incomplete_config.json
  > fileexample/validation_requirements.txt
  >

Enter your prompt: 检查 incomplete_config.json 文件的要素完整性，参考 validation_requirements.txt 中的要求
```

## 预期规划步骤

基于模板和业务文件，智能体应该生成类似以下的规划：

1. 读取并分析 incomplete_config.json 文件
2. 读取 validation_requirements.txt 了解检查要求
3. 识别配置文件中的关键要素和字段
4. 检查必需字段是否存在（application.version, database.name, features.logging, settings.max_connections）
5. 验证字段值的完整性和格式
6. 生成详细的检查报告，列出所有缺失和错误的字段
7. 保存检查报告到文件

## 检查要点

### 应该发现的问题

1. **缺失字段**：
   - `application.version` 值为空字符串（应为非空）
   - `database.name` 字段缺失
   - `features.logging` 字段缺失
   - `settings.max_connections` 字段缺失

2. **格式问题**：
   - `application.version` 为空字符串，不符合版本号格式要求

## 功能特点

### 1. 规划修改功能
- 如果生成的规划不符合预期，输入 `n` 拒绝
- 提供修改反馈，例如："添加对比 sample_config.json 的步骤"
- 系统会基于反馈重新生成规划

### 2. 步骤结果反馈
- 每个步骤执行完成后会显示结果
- 如果结果有问题，可以选择：
  - 提供修正意见（会存入记忆）
  - 标记步骤为阻塞状态
  - 继续执行

### 3. 记忆存储
- 修正意见会存入智能体的记忆系统
- 后续执行类似任务时会参考这些修正

## 输出示例

检查报告应该包含：

```
配置文件要素完整性检查报告
========================================

文件：incomplete_config.json
检查时间：2024-01-14

【文件基本信息】
- 文件格式：JSON（有效）
- 文件大小：XXX bytes
- 编码：UTF-8

【缺失字段】
1. database.name - 必需字段缺失
2. features.logging - 必需字段缺失
3. settings.max_connections - 必需字段缺失

【格式错误】
1. application.version - 值为空字符串，应为非空版本号

【修复建议】
1. 添加 database.name 字段，建议值："mydb"
2. 添加 features.logging 字段，建议值：true
3. 添加 settings.max_connections 字段，建议值：100
4. 修复 application.version 字段，建议值："1.0.0"

【完整性评分】
必需字段完整性：60% (6/10)
整体完整性：75%
```

## 扩展使用

### 检查其他文件类型

可以修改 `validation_requirements.txt` 来检查不同类型的文件：

- **CSV文件**：检查列完整性、数据类型、数据范围
- **文档文件**：检查章节完整性、格式规范性
- **配置文件**：检查配置项完整性、值有效性

### 自定义检查规则

在 `validation_requirements.txt` 中定义：
- 必需字段清单
- 字段格式要求
- 数据范围要求
- 业务规则约束

## 注意事项

1. 确保待检查文件在 `workspace/fileexample/` 目录中
2. 输入文件路径时使用相对路径：`fileexample/文件名`
3. 检查要求文件要清晰明确
4. 参考标准文件（如 sample_config.json）有助于理解正确格式
5. 检查报告会保存在 `workspace/fileexample/` 目录中
6. 所有生成的文件都会保存在 `fileexample` 子目录，保持工作区整洁

## 相关文件

- 规划模板：`examples/planning_templates/file_validation_template.txt`
- 完整示例：`workspace/fileexample/sample_config.json`
- 待检查文件：`workspace/fileexample/incomplete_config.json`
- 检查要求：`workspace/fileexample/validation_requirements.txt`

## 目录结构说明

```
workspace/
└── fileexample/              # 文件示例专用目录
    ├── README.md             # 目录说明
    ├── sample_config.json    # 完整示例文件
    ├── incomplete_config.json # 待检查文件
    ├── validation_requirements.txt # 检查要求
    └── [生成的检查报告等文件会保存在这里]
```

**优势**：
- 文件组织更清晰，不同类型的业务文件可以放在不同子目录
- 生成的文件不会污染 workspace 根目录
- 便于管理和清理
