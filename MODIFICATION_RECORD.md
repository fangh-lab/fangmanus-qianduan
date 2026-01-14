# 修改记录 - Run Flow 模式改造

## 修改概述

本次改造主要针对 `run_flow` 模式，添加了以下三个核心功能：
1. **文件上传功能**：支持上传规划文件作为规划提示的基础
2. **规划后人类反馈确认**：在规划生成后，需要人类确认才能继续执行
3. **步骤执行后人类反馈**：每个步骤执行完成后，需要人类反馈确认才能继续下一步

## 修改文件清单

### 1. `run_flow.py`
**修改内容**：
- 添加规划模板文件路径输入功能
- 添加业务文件路径输入功能（支持多个文件，相对于 workspace）
- 集成 `LocalFileOperator` 读取文件
- 将规划模板内容作为 `planning_context` 传递给 flow
- 将业务文件内容作为 `business_files` 传递给 flow
- 添加 `resolve_workspace_path()` 函数处理工作区路径解析

**修改原因**：
- 支持用户上传业务规划模板或指导文件
- 支持从 workspace 目录读取业务文件
- 让智能体能够基于部门的标准流程进行规划
- 让智能体能够访问和处理实际的业务数据文件
- 提高规划的专业性和一致性

**关键代码**：
```python
# 添加文件读取功能
planning_file_path = input("Enter planning file path (optional, press Enter to skip): ").strip()
planning_context = ""
if planning_file_path:
    if Path(planning_file_path).exists():
        planning_context = await read_planning_file(planning_file_path)

# 传递规划上下文给 flow
result = await flow.execute(prompt, planning_context=planning_context)
```

---

### 2. `app/flow/human_interaction.py` (新建文件)
**修改内容**：
- 创建人类交互辅助函数模块
- 实现 `ask_human_confirmation()`：确认对话框
- 实现 `ask_human_feedback()`：获取用户反馈
- 实现 `display_text_with_pagination()`：分页显示长文本

**修改原因**：
- 统一人类交互接口，便于维护和扩展
- 提供友好的用户交互体验
- 支持分页显示，避免终端输出过长

**关键功能**：
```python
def ask_human_confirmation(prompt: str, default: str = "y") -> bool
def ask_human_feedback(prompt: str, allow_empty: bool = True) -> str
def display_text_with_pagination(text: str, page_size: int = 20)
```

---

### 3. `app/flow/planning.py`
**修改内容**：

#### 3.1 导入人类交互模块
- 添加 `human_interaction` 模块的导入

#### 3.2 修改 `execute()` 方法
- 添加 `planning_context` 参数（可选）
- 在规划创建后调用 `_confirm_plan_with_human()` 进行确认
- 在每步执行后调用 `_confirm_step_result()` 进行反馈

**修改原因**：
- 支持规划上下文传递
- 确保人类对规划有控制权
- 每步执行后及时获取反馈，避免错误累积

#### 3.3 修改 `_create_initial_plan()` 方法
- 添加 `planning_context` 参数（规划模板）
- 添加 `business_files` 参数（业务文件字典）
- 将规划上下文整合到系统提示中
- 将业务文件内容整合到系统提示中
- 指导 LLM 基于规划模板和业务文件生成计划

**修改原因**：
- 让规划生成过程能够参考用户提供的业务标准
- 让规划生成过程能够考虑实际的业务数据
- 提高规划的专业性和符合度
- 确保规划步骤能够处理具体的业务文件

#### 3.4 新增 `_confirm_plan_with_human()` 方法
- 显示生成的规划内容
- 询问用户是否确认执行
- 支持用户提供反馈（为未来规划修改功能预留）

**修改原因**：
- 确保规划符合用户期望
- 在执行前给用户审查和修改的机会
- 提高任务执行的成功率

#### 3.5 新增 `_confirm_step_result()` 方法
- 显示步骤执行结果
- 收集用户反馈
- 将反馈存储到计划的步骤备注中
- 询问是否继续下一步

**修改原因**：
- 及时发现和纠正执行错误
- 收集用户反馈用于改进
- 给用户控制执行流程的机会

**关键代码**：
```python
async def execute(self, input_text: str, planning_context: str = "") -> str:
    # ... 创建规划 ...
    if not await self._confirm_plan_with_human():
        return "Plan execution cancelled by user."

    # ... 执行步骤 ...
    if not await self._confirm_step_result(step_result):
        result += "\n[Execution stopped by user feedback]"
        break
```

---

## 功能流程说明

### 改造前流程
```
用户输入 → 创建规划 → 自动执行所有步骤 → 完成
```

### 改造后流程
```
用户输入 → [可选]上传规划文件 → 创建规划（基于规划文件）
    → 人类确认规划 → 执行步骤1 → 人类反馈步骤1结果
    → 执行步骤2 → 人类反馈步骤2结果 → ... → 完成
```

---

## 使用示例

### 1. 使用规划模板和业务文件
```bash
python run_flow.py
# Enter planning template file path: examples/planning_templates/customer_service_template.txt
# Enter business file paths (relative to workspace):
#   > customer_complaint_12345.txt
#   >
# Enter your prompt: 处理客户投诉工单 #12345
```

### 2. 仅使用业务文件（不使用规划模板）
```bash
python run_flow.py
# Enter planning template file path: [直接按 Enter]
# Enter business file paths:
#   > sales_data_january.csv
#   >
# Enter your prompt: 分析一月份的销售数据
```

### 3. 不使用任何文件（传统方式）
```bash
python run_flow.py
# Enter planning template file path: [直接按 Enter]
# Enter business file paths: [直接按 Enter]
# Enter your prompt: 处理客户投诉工单
```

### 3. 规划确认交互
```
GENERATED PLAN - Please Review
================================================================================
Plan: 处理客户投诉工单 (ID: plan_1234567890)
================================================================================
Progress: 0/5 steps completed (0.0%)
Steps:
0. [ ] 接收并分析投诉工单
1. [ ] 联系客户了解详细情况
2. [ ] 制定解决方案
3. [ ] 执行解决方案
4. [ ] 跟进并确认客户满意度
================================================================================

Do you want to proceed with this plan? (y/n, default=y): y
```

### 4. 步骤反馈交互
```
--------------------------------------------------------------------------------
STEP 0 EXECUTION RESULT
--------------------------------------------------------------------------------
Step: 接收并分析投诉工单

Result:
已成功接收投诉工单 #12345，客户反映产品质量问题...
--------------------------------------------------------------------------------

Please review the step result. Enter feedback (optional, press Enter to skip):
结果符合预期，继续执行

Do you want to continue to the next step? (y/n, default=y): y
```

---

## 技术实现细节

### 1. 文件读取
- 使用 `LocalFileOperator` 读取本地文件
- 支持多种编码格式（UTF-8, GBK, GB2312等）
- 支持相对路径（相对于 workspace）和绝对路径
- 错误处理：文件不存在时给出警告但不中断流程
- 业务文件内容会被整合到规划提示中（大文件会截断预览）

### 2. 规划上下文整合
- 规划模板文件内容作为系统提示的一部分
- 业务文件内容作为系统提示的一部分
- LLM 会参考规划模板和业务文件生成符合业务标准的计划
- 保持向后兼容：规划模板和业务文件都为可选参数
- 业务文件路径相对于 workspace 目录解析

### 3. 人类反馈机制
- 使用同步输入（`input()`）进行交互
- 支持默认值，提高用户体验
- 反馈信息存储到计划的 `step_notes` 中

### 4. 错误处理
- 所有人类交互都有异常处理
- 出错时提供默认选项，避免流程中断
- 记录详细的日志信息

---

## 兼容性说明

### 向后兼容
- `planning_context` 参数有默认值 `""`，不影响现有调用
- 人类反馈确认可以跳过（通过默认值）
- 不提供规划文件时，行为与原来一致

### 接口变更
- `PlanningFlow.execute()` 新增可选参数 `planning_context`
- 不影响其他 Flow 类型的实现

---

## 未来扩展方向

1. **规划修改功能**：根据用户反馈自动修改规划
2. **反馈学习**：基于历史反馈优化规划生成
3. **批量模式**：支持非交互模式，跳过人类反馈
4. **反馈模板**：提供标准化的反馈模板
5. **可视化界面**：将终端交互改为图形界面

---

## 测试建议

1. **测试规划文件读取**：
   - 测试文件存在/不存在的情况
   - 测试不同编码格式的文件
   - 测试大文件读取

2. **测试人类反馈**：
   - 测试确认/拒绝规划
   - 测试步骤反馈输入
   - 测试空输入处理

3. **测试错误处理**：
   - 测试文件读取错误
   - 测试交互异常情况
   - 测试流程中断恢复

---

## 最新更新（2024-01-14）

### 新增功能

#### 1. 规划修改功能
**问题**：用户拒绝规划时无法修改，只能重新开始

**解决方案**：
- 实现 `_modify_plan_with_feedback()` 方法
- 当用户拒绝规划时，可以输入修改反馈
- 系统使用 LLM 和 PlanningTool 基于反馈重新生成规划
- 支持最多 3 次修改尝试
- 修改后的规划会保留未变更步骤的状态

**使用方式**：
```
Do you want to proceed with this plan? (y/n, default=y): n
Plan Modification Options:
1. Provide feedback to modify the plan
2. Cancel and exit
Enter your choice (1/2, default=1): 1
Please provide your feedback: 添加对比参考文件的步骤
```

#### 2. 步骤结果反馈和修正功能
**问题**：步骤执行结果有问题时无法修正，无法存入记忆

**解决方案**：
- 增强 `_confirm_step_result()` 方法
- 当结果不可接受时，提供三种选项：
  1. 提供修正意见（存入记忆和步骤备注）
  2. 标记步骤为阻塞状态
  3. 继续执行
- 实现 `_store_step_correction()` 方法，将修正存入智能体记忆
- 实现 `_store_step_feedback()` 方法，存储反馈到步骤备注

**使用方式**：
```
Is this step result acceptable? (y/n, default=y): n
Result Modification Options:
1. Provide correction/feedback (will be stored in memory)
2. Mark step as needing re-execution
3. Continue anyway
Enter your choice (1/2/3, default=1): 1
Please provide the correction: 应该检查所有必需字段，不仅仅是前三个
✓ Correction has been stored in memory and step notes.
```

#### 3. 文件要素完整性检查业务模板
**新增文件**：
- `examples/planning_templates/file_validation_template.txt` - 文件验证流程模板
- `workspace/sample_config.json` - 完整配置文件示例
- `workspace/incomplete_config.json` - 待检查配置文件
- `workspace/validation_requirements.txt` - 检查要求文档
- `examples/FILE_VALIDATION_EXAMPLE.md` - 使用示例文档

**特点**：
- 专注于智能体可以完成的任务（文件读取、格式检查、数据验证）
- 不涉及需要人工交互的任务（如打电话、联系客户）
- 提供完整的检查流程和标准

### 技术实现细节

#### 规划修改实现
```python
async def _modify_plan_with_feedback(self, feedback: str) -> bool:
    # 使用 LLM + PlanningTool 的 update 命令修改规划
    # 保留未变更步骤的状态和备注
```

#### 修正存储实现
```python
async def _store_step_correction(self, correction: str, original_result: str):
    # 1. 存储到步骤备注
    # 2. 添加到智能体记忆系统
    # 3. 后续执行会参考这些修正
```

### 使用示例

#### 文件要素完整性检查示例

```bash
python run_flow.py
```

输入：
```
Enter planning template file path: examples/planning_templates/file_validation_template.txt
Enter business file paths:
  > incomplete_config.json
  > validation_requirements.txt
Enter your prompt: 检查 incomplete_config.json 文件的要素完整性
```

预期流程：
1. 生成规划（可修改）
2. 读取文件
3. 识别要素
4. 检查完整性
5. 生成报告
6. 每步执行后确认结果（可修正）

---

## 修改日期
2024年1月14日

## 修改人员
根据实际情况填写
