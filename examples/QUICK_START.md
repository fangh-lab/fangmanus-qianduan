# 快速开始指南 - Run Flow 模式改造功能

## 5分钟快速体验

### 步骤1：准备规划模板文件

已提供的示例模板文件位于 `examples/planning_templates/` 目录：
- `customer_service_template.txt` - 客户服务流程
- `data_analysis_template.txt` - 数据分析流程
- `software_development_template.txt` - 软件开发流程
- `simple_example.txt` - 简单示例

### 步骤2：准备业务文件（可选）

将需要处理的业务文件放到 `workspace/` 目录中。已提供的示例文件：
- `workspace/customer_complaint_12345.txt` - 客户投诉工单
- `workspace/sales_data_january.csv` - 销售数据
- `workspace/todo_app_requirements.txt` - 应用需求文档

### 步骤3：运行测试脚本（推荐）

```bash
# 进入项目根目录
cd d:\fanghao\gihub\fanghaomanus

# 运行测试脚本
python examples/test_run_flow_with_template.py
```

测试脚本会引导您：
1. 选择规划模板（或输入自定义路径）
2. 选择示例提示词（或自定义输入）
3. 体验完整的人类反馈流程

### 步骤4：直接运行主程序

```bash
python run_flow.py
```

然后按照提示：
1. 输入规划模板文件路径（可选，直接按 Enter 跳过）
   ```
   Enter planning template file path: examples/planning_templates/customer_service_template.txt
   ```

2. 输入业务文件路径（可选，相对于 workspace，每行一个，空行结束）
   ```
   Workspace directory: D:\fanghao\gihub\fanghaomanus\workspace
   Enter business file paths (relative to workspace, one per line, empty line to finish):
     > customer_complaint_12345.txt
     >
   ```

3. 输入业务请求
   ```
   Enter your prompt: 处理客户投诉工单 #12345，客户反映产品质量问题
   ```

3. 确认生成的规划
   ```
   Do you want to proceed with this plan? (y/n, default=y): y
   ```

4. 每步执行后提供反馈
   ```
   Please review the step result. Enter feedback (optional):
   Do you want to continue to the next step? (y/n, default=y): y
   ```

## 完整示例演示

### 示例1：客户服务场景

**规划模板**：`examples/planning_templates/customer_service_template.txt`

**业务请求**：
```
处理客户投诉工单 #12345，客户反映产品质量问题，需要尽快解决
```

**预期交互流程**：

1. **规划生成后**：
```
================================================================================
GENERATED PLAN - Please Review
================================================================================
Plan: 处理客户投诉工单 (ID: plan_1234567890)
================================================================================
Progress: 0/5 steps completed (0.0%)
Steps:
0. [ ] 接收并分析投诉工单 #12345
1. [ ] 联系客户了解详细情况
2. [ ] 制定解决方案
3. [ ] 执行解决方案
4. [ ] 跟进并确认客户满意度
================================================================================

Do you want to proceed with this plan? (y/n, default=y): y
```

2. **步骤0执行后**：
```
--------------------------------------------------------------------------------
STEP 0 EXECUTION RESULT
--------------------------------------------------------------------------------
Step: 接收并分析投诉工单 #12345

Result:
已成功接收投诉工单 #12345
工单类型：产品质量投诉
紧急程度：高
客户信息：已提取
问题描述：客户反映产品在使用过程中出现质量问题...
--------------------------------------------------------------------------------

Please review the step result. Enter feedback (optional, press Enter to skip):
结果符合预期，继续执行

Do you want to continue to the next step? (y/n, default=y): y
```

### 示例2：数据分析场景

**规划模板**：`examples/planning_templates/data_analysis_template.txt`

**业务请求**：
```
分析最近一个月的销售数据，找出销售趋势和潜在问题
```

**预期规划步骤**：
1. 理解分析目标和业务问题
2. 收集和准备销售数据
3. 进行数据分析和趋势识别
4. 创建可视化图表
5. 撰写分析报告和建议
6. 向业务方展示结果

### 示例3：软件开发场景

**规划模板**：`examples/planning_templates/software_development_template.txt`

**业务请求**：
```
开发一个简单的待办事项管理应用，支持添加、删除和标记完成功能
```

**预期规划步骤**：
1. 分析功能需求和用户故事
2. 设计应用架构和数据结构
3. 搭建开发环境
4. 实现核心功能
5. 编写测试用例
6. 部署和发布应用
7. 编写使用文档

## 创建自定义模板

### 模板格式要求

1. **使用 Markdown 格式**
2. **包含以下部分**：
   - 业务概述
   - 标准流程步骤（分阶段）
   - 注意事项
   - 质量标准

### 模板示例结构

```markdown
# 您的业务名称规划模板

## 业务概述
说明模板的用途和适用场景

## 标准流程步骤

### 1. 第一阶段名称
- 步骤1描述
- 步骤2描述

### 2. 第二阶段名称
- 步骤1描述
- 步骤2描述

## 注意事项
- 重要规则1
- 重要规则2

## 质量标准
- 指标1：要求
- 指标2：要求
```

### 保存模板

将模板文件保存到 `examples/planning_templates/` 目录，或任何您方便的位置。

## 常见问题

### Q1: 规划文件路径怎么写？
**A**: 可以使用：
- 相对路径：`examples/planning_templates/customer_service_template.txt`
- 绝对路径：`D:\fanghao\gihub\fanghaomanus\examples\planning_templates\customer_service_template.txt`

### Q2: 可以不使用规划模板吗？
**A**: 可以！直接按 Enter 跳过文件路径输入即可。

### Q3: 人类反馈环节可以跳过吗？
**A**: 可以！直接按 Enter 使用默认值（y）即可快速通过。

### Q4: 如何修改规划？
**A**: 目前版本中，如果规划不符合预期：
1. 选择 "n" 拒绝规划，流程会终止
2. 提供反馈（虽然当前版本不会自动修改规划，但反馈会被记录）
3. 修改规划模板文件后重新运行

### Q5: 步骤反馈有什么用？
**A**: 步骤反馈会：
1. 被存储到计划的步骤备注中
2. 帮助您记录执行过程中的问题和建议
3. 为未来的规划优化提供参考

## 下一步

- 查看详细文档：`examples/README_planning_templates.md`
- 查看修改记录：`MODIFICATION_RECORD.md`
- 自定义您的业务模板
- 集成到您的业务流程中

## 技术支持

如有问题，请查看：
- 代码实现：`app/flow/planning.py`
- 人类交互：`app/flow/human_interaction.py`
- 主程序：`run_flow.py`
