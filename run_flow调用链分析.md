# run_flow.py 调用链详细分析

## 概述

本文档详细分析了运行 `run_flow.py` 后，整个系统的执行流程和调用关系，并与 `main.py` 的执行流程进行对比。

---

## 一、入口点：run_flow.py

### 1.1 程序启动

```1:52:run_flow.py
import asyncio
import time

from app.agent.data_analysis import DataAnalysis
from app.agent.manus import Manus
from app.config import config
from app.flow.flow_factory import FlowFactory, FlowType
from app.logger import logger


async def run_flow():
    agents = {
        "manus": Manus(),
    }
    if config.run_flow_config.use_data_analysis_agent:
        agents["data_analysis"] = DataAnalysis()
    try:
        prompt = input("Enter your prompt: ")

        if prompt.strip().isspace() or not prompt:
            logger.warning("Empty prompt provided.")
            return

        flow = FlowFactory.create_flow(
            flow_type=FlowType.PLANNING,
            agents=agents,
        )
        logger.warning("Processing your request...")

        try:
            start_time = time.time()
            result = await asyncio.wait_for(
                flow.execute(prompt),
                timeout=3600,  # 60 minute timeout for the entire execution
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Request processed in {elapsed_time:.2f} seconds")
            logger.info(result)
        except asyncio.TimeoutError:
            logger.error("Request processing timed out after 1 hour")
            logger.info(
                "Operation terminated due to timeout. Please try a simpler request."
            )

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user.")
    except Exception as e:
        logger.error(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(run_flow())
```

### 1.2 执行步骤

1. **创建 Agent 字典**：创建 `agents` 字典，包含 `Manus` agent
2. **可选添加 DataAnalysis Agent**：根据配置决定是否添加 `DataAnalysis` agent
3. **获取用户输入**：通过 `input("Enter your prompt: ")` 提示用户输入
4. **创建 Flow**：通过 `FlowFactory.create_flow()` 创建 `PlanningFlow`
5. **执行 Flow**：调用 `flow.execute(prompt)` 开始执行流程
6. **超时控制**：使用 `asyncio.wait_for()` 设置 1 小时超时
7. **记录结果**：记录执行时间和结果

---

## 二、Flow 创建：FlowFactory.create_flow()

### 2.1 FlowFactory 类

```13:30:app/flow/flow_factory.py
class FlowFactory:
    """Factory for creating different types of flows with support for multiple agents"""

    @staticmethod
    def create_flow(
        flow_type: FlowType,
        agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]],
        **kwargs,
    ) -> BaseFlow:
        flows = {
            FlowType.PLANNING: PlanningFlow,
        }

        flow_class = flows.get(flow_type)
        if not flow_class:
            raise ValueError(f"Unknown flow type: {flow_type}")

        return flow_class(agents, **kwargs)
```

### 2.2 创建流程

1. **选择 Flow 类型**：根据 `flow_type` 选择对应的 Flow 类（这里是 `PlanningFlow`）
2. **创建实例**：使用传入的 `agents` 字典创建 `PlanningFlow` 实例

---

## 三、PlanningFlow 初始化

### 3.1 PlanningFlow 类结构

```45:93:app/flow/planning.py
class PlanningFlow(BaseFlow):
    """A flow that manages planning and execution of tasks using agents."""

    llm: LLM = Field(default_factory=lambda: LLM())
    planning_tool: PlanningTool = Field(default_factory=PlanningTool)
    executor_keys: List[str] = Field(default_factory=list)
    active_plan_id: str = Field(default_factory=lambda: f"plan_{int(time.time())}")
    current_step_index: Optional[int] = None

    def __init__(
        self, agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]], **data
    ):
        # Set executor keys before super().__init__
        if "executors" in data:
            data["executor_keys"] = data.pop("executors")

        # Set plan ID if provided
        if "plan_id" in data:
            data["active_plan_id"] = data.pop("plan_id")

        # Initialize the planning tool if not provided
        if "planning_tool" not in data:
            planning_tool = PlanningTool()
            data["planning_tool"] = planning_tool

        # Call parent's init with the processed data
        super().__init__(agents, **data)

        # Set executor_keys to all agent keys if not specified
        if not self.executor_keys:
            self.executor_keys = list(self.agents.keys())
```

### 3.2 初始化流程

1. **初始化 PlanningTool**：创建 `PlanningTool` 实例用于管理计划
2. **设置 Agent 字典**：通过父类 `BaseFlow.__init__()` 设置 agents
3. **设置执行器列表**：如果没有指定 `executor_keys`，则使用所有 agent 的 key
4. **生成计划 ID**：自动生成一个唯一的计划 ID（基于时间戳）

---

## 四、Flow 执行：PlanningFlow.execute()

### 4.1 完整执行流程

```94:134:app/flow/planning.py
    async def execute(self, input_text: str) -> str:
        """Execute the planning flow with agents."""
        try:
            if not self.primary_agent:
                raise ValueError("No primary agent available")

            # Create initial plan if input provided
            if input_text:
                await self._create_initial_plan(input_text)

                # Verify plan was created successfully
                if self.active_plan_id not in self.planning_tool.plans:
                    logger.error(
                        f"Plan creation failed. Plan ID {self.active_plan_id} not found in planning tool."
                    )
                    return f"Failed to create plan for: {input_text}"

            result = ""
            while True:
                # Get current step to execute
                self.current_step_index, step_info = await self._get_current_step_info()

                # Exit if no more steps or plan completed
                if self.current_step_index is None:
                    result += await self._finalize_plan()
                    break

                # Execute current step with appropriate agent
                step_type = step_info.get("type") if step_info else None
                executor = self.get_executor(step_type)
                step_result = await self._execute_step(executor, step_info)
                result += step_result + "\n"

                # Check if agent wants to terminate
                if hasattr(executor, "state") and executor.state == AgentState.FINISHED:
                    break

            return result
        except Exception as e:
            logger.error(f"Error in PlanningFlow: {str(e)}")
            return f"Execution failed: {str(e)}"
```

### 4.2 执行步骤

1. **创建初始计划**：`await self._create_initial_plan(input_text)` 创建任务计划
2. **验证计划创建**：检查计划是否成功创建
3. **执行循环**：
   - 获取当前步骤：`await self._get_current_step_info()`
   - 如果没有更多步骤，执行 `_finalize_plan()` 并退出
   - 选择执行器：根据步骤类型选择合适的 Agent
   - 执行步骤：`await self._execute_step(executor, step_info)`
   - 检查 Agent 状态：如果 Agent 状态为 FINISHED，退出循环
4. **返回结果**：返回所有步骤的执行结果

---

## 五、创建初始计划：_create_initial_plan()

### 5.1 计划创建流程

```136:211:app/flow/planning.py
    async def _create_initial_plan(self, request: str) -> None:
        """Create an initial plan based on the request using the flow's LLM and PlanningTool."""
        logger.info(f"Creating initial plan with ID: {self.active_plan_id}")

        system_message_content = (
            "You are a planning assistant. Create a concise, actionable plan with clear steps. "
            "Focus on key milestones rather than detailed sub-steps. "
            "Optimize for clarity and efficiency."
        )
        agents_description = []
        for key in self.executor_keys:
            if key in self.agents:
                agents_description.append(
                    {
                        "name": key.upper(),
                        "description": self.agents[key].description,
                    }
                )
        if len(agents_description) > 1:
            # Add description of agents to select
            system_message_content += (
                f"\nNow we have {agents_description} agents. "
                f"The infomation of them are below: {json.dumps(agents_description)}\n"
                "When creating steps in the planning tool, please specify the agent names using the format '[agent_name]'."
            )

        # Create a system message for plan creation
        system_message = Message.system_message(system_message_content)

        # Create a user message with the request
        user_message = Message.user_message(
            f"Create a reasonable plan with clear steps to accomplish the task: {request}"
        )

        # Call LLM with PlanningTool
        response = await self.llm.ask_tool(
            messages=[user_message],
            system_msgs=[system_message],
            tools=[self.planning_tool.to_param()],
            tool_choice=ToolChoice.AUTO,
        )

        # Process tool calls if present
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.function.name == "planning":
                    # Parse the arguments
                    args = tool_call.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse tool arguments: {args}")
                            continue

                    # Ensure plan_id is set correctly and execute the tool
                    args["plan_id"] = self.active_plan_id

                    # Execute the tool via ToolCollection instead of directly
                    result = await self.planning_tool.execute(**args)

                    logger.info(f"Plan creation result: {str(result)}")
                    return

        # If execution reached here, create a default plan
        logger.warning("Creating default plan")

        # Create default plan using the ToolCollection
        await self.planning_tool.execute(
            **{
                "command": "create",
                "plan_id": self.active_plan_id,
                "title": f"Plan for: {request[:50]}{'...' if len(request) > 50 else ''}",
                "steps": ["Analyze request", "Execute task", "Verify results"],
            }
        )
```

**执行步骤**：

1. **构建系统提示**：创建计划助手提示，包含可用 Agent 的描述（如果有多个）
2. **创建用户消息**：将用户请求包装为用户消息
3. **调用 LLM**：使用 `PlanningTool` 调用 LLM 创建计划
4. **处理工具调用**：解析 LLM 返回的工具调用，执行 `planning` 工具的 `create` 命令
5. **创建默认计划**：如果 LLM 没有返回有效的工具调用，创建默认计划

### 5.2 PlanningTool 创建计划

PlanningTool 的 `create` 命令会在内部存储计划，包括：
- `plan_id`：计划 ID
- `title`：计划标题
- `steps`：步骤列表
- `step_statuses`：每个步骤的状态（初始为 "not_started"）
- `step_notes`：每个步骤的注释（初始为空）

---

## 六、获取当前步骤：_get_current_step_info()

### 6.1 步骤信息获取

```213:275:app/flow/planning.py
    async def _get_current_step_info(self) -> tuple[Optional[int], Optional[dict]]:
        """
        Parse the current plan to identify the first non-completed step's index and info.
        Returns (None, None) if no active step is found.
        """
        if (
            not self.active_plan_id
            or self.active_plan_id not in self.planning_tool.plans
        ):
            logger.error(f"Plan with ID {self.active_plan_id} not found")
            return None, None

        try:
            # Direct access to plan data from planning tool storage
            plan_data = self.planning_tool.plans[self.active_plan_id]
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])

            # Find first non-completed step
            for i, step in enumerate(steps):
                if i >= len(step_statuses):
                    status = PlanStepStatus.NOT_STARTED.value
                else:
                    status = step_statuses[i]

                if status in PlanStepStatus.get_active_statuses():
                    # Extract step type/category if available
                    step_info = {"text": step}

                    # Try to extract step type from the text (e.g., [SEARCH] or [CODE])
                    import re

                    type_match = re.search(r"\[([A-Z_]+)\]", step)
                    if type_match:
                        step_info["type"] = type_match.group(1).lower()

                    # Mark current step as in_progress
                    try:
                        await self.planning_tool.execute(
                            command="mark_step",
                            plan_id=self.active_plan_id,
                            step_index=i,
                            step_status=PlanStepStatus.IN_PROGRESS.value,
                        )
                    except Exception as e:
                        logger.warning(f"Error marking step as in_progress: {e}")
                        # Update step status directly if needed
                        if i < len(step_statuses):
                            step_statuses[i] = PlanStepStatus.IN_PROGRESS.value
                        else:
                            while len(step_statuses) < i:
                                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
                            step_statuses.append(PlanStepStatus.IN_PROGRESS.value)

                        plan_data["step_statuses"] = step_statuses

                    return i, step_info

            return None, None  # No active step found

        except Exception as e:
            logger.warning(f"Error finding current step index: {e}")
            return None, None
```

**执行步骤**：

1. **获取计划数据**：从 `planning_tool.plans` 中获取当前计划
2. **查找第一个未完成的步骤**：遍历步骤，找到状态为 "not_started" 或 "in_progress" 的步骤
3. **提取步骤类型**：如果步骤文本中包含 `[AGENT_NAME]` 格式，提取 Agent 名称
4. **标记为进行中**：将步骤状态更新为 "in_progress"
5. **返回步骤信息**：返回步骤索引和步骤信息字典

---

## 七、执行步骤：_execute_step()

### 7.1 步骤执行

```277:304:app/flow/planning.py
    async def _execute_step(self, executor: BaseAgent, step_info: dict) -> str:
        """Execute the current step with the specified agent using agent.run()."""
        # Prepare context for the agent with current plan status
        plan_status = await self._get_plan_text()
        step_text = step_info.get("text", f"Step {self.current_step_index}")

        # Create a prompt for the agent to execute the current step
        step_prompt = f"""
        CURRENT PLAN STATUS:
        {plan_status}

        YOUR CURRENT TASK:
        You are now working on step {self.current_step_index}: "{step_text}"

        Please only execute this current step using the appropriate tools. When you're done, provide a summary of what you accomplished.
        """

        # Use agent.run() to execute the step
        try:
            step_result = await executor.run(step_prompt)

            # Mark the step as completed after successful execution
            await self._mark_step_completed()

            return step_result
        except Exception as e:
            logger.error(f"Error executing step {self.current_step_index}: {e}")
            return f"Error executing step {self.current_step_index}: {str(e)}"
```

**执行步骤**：

1. **获取计划状态**：调用 `_get_plan_text()` 获取当前计划的文本表示
2. **构建步骤提示**：创建包含计划状态和当前步骤的提示
3. **执行 Agent**：调用 `executor.run(step_prompt)` 执行步骤（这里会触发 Agent 的完整执行循环）
4. **标记步骤完成**：步骤执行成功后，调用 `_mark_step_completed()` 标记为完成
5. **返回结果**：返回步骤执行结果

### 7.2 Agent 执行（递归调用）

当 `executor.run(step_prompt)` 被调用时，会触发 Agent 的完整执行流程：
- `BaseAgent.run()` → `ReActAgent.step()` → `ToolCallAgent.think()` → `ToolCallAgent.act()`

这与 `main.py` 中的 Agent 执行流程完全相同。

---

## 八、标记步骤完成：_mark_step_completed()

### 8.1 步骤完成标记

```306:335:app/flow/planning.py
    async def _mark_step_completed(self) -> None:
        """Mark the current step as completed."""
        if self.current_step_index is None:
            return

        try:
            # Mark the step as completed
            await self.planning_tool.execute(
                command="mark_step",
                plan_id=self.active_plan_id,
                step_index=self.current_step_index,
                step_status=PlanStepStatus.COMPLETED.value,
            )
            logger.info(
                f"Marked step {self.current_step_index} as completed in plan {self.active_plan_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to update plan status: {e}")
            # Update step status directly in planning tool storage
            if self.active_plan_id in self.planning_tool.plans:
                plan_data = self.planning_tool.plans[self.active_plan_id]
                step_statuses = plan_data.get("step_statuses", [])

                # Ensure the step_statuses list is long enough
                while len(step_statuses) <= self.current_step_index:
                    step_statuses.append(PlanStepStatus.NOT_STARTED.value)

                # Update the status
                step_statuses[self.current_step_index] = PlanStepStatus.COMPLETED.value
                plan_data["step_statuses"] = step_statuses
```

**执行步骤**：

1. **调用 PlanningTool**：使用 `mark_step` 命令更新步骤状态为 "completed"
2. **错误处理**：如果调用失败，直接更新内部存储的步骤状态

---

## 九、完成计划：_finalize_plan()

### 9.1 计划完成处理

```406:442:app/flow/planning.py
    async def _finalize_plan(self) -> str:
        """Finalize the plan and provide a summary using the flow's LLM directly."""
        plan_text = await self._get_plan_text()

        # Create a summary using the flow's LLM directly
        try:
            system_message = Message.system_message(
                "You are a planning assistant. Your task is to summarize the completed plan."
            )

            user_message = Message.user_message(
                f"The plan has been completed. Here is the final plan status:\n\n{plan_text}\n\nPlease provide a summary of what was accomplished and any final thoughts."
            )

            response = await self.llm.ask(
                messages=[user_message], system_msgs=[system_message]
            )

            return f"Plan completed:\n\n{response}"
        except Exception as e:
            logger.error(f"Error finalizing plan with LLM: {e}")

            # Fallback to using an agent for the summary
            try:
                agent = self.primary_agent
                summary_prompt = f"""
                The plan has been completed. Here is the final plan status:

                {plan_text}

                Please provide a summary of what was accomplished and any final thoughts.
                """
                summary = await agent.run(summary_prompt)
                return f"Plan completed:\n\n{summary}"
            except Exception as e2:
                logger.error(f"Error finalizing plan with agent: {e2}")
                return "Plan completed. Error generating summary."
```

**执行步骤**：

1. **获取计划文本**：获取完整的计划状态文本
2. **生成总结**：使用 LLM 或 Agent 生成计划完成的总结
3. **返回结果**：返回总结文本

---

## 十、完整调用流程图

```
┌─────────────────────────────────────────────────────────────┐
│                      run_flow.py                            │
│  asyncio.run(run_flow())                                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   run_flow() 函数                            │
│  1. 创建 agents 字典（Manus，可选 DataAnalysis）            │
│  2. prompt = input("Enter your prompt: ")                   │
│  3. flow = FlowFactory.create_flow(FlowType.PLANNING, agents) │
│  4. result = await flow.execute(prompt)  ←────────────────┐ │
│  5. 记录执行时间和结果                                      │
└───────────────────────┬─────────────────────────────────────┘ │
                        │                                     │ │
                        ▼                                     │ │
┌─────────────────────────────────────────────────────────────┐ │
│        FlowFactory.create_flow()                            │ │
│  1. 选择 PlanningFlow 类                                    │ │
│  2. 创建 PlanningFlow(agents) 实例                          │ │
└───────────────────────┬─────────────────────────────────────┘ │
                        │                                     │ │
                        ▼                                     │ │
┌─────────────────────────────────────────────────────────────┐ │
│        PlanningFlow.__init__()                              │ │
│  1. 初始化 PlanningTool                                     │ │
│  2. 调用 BaseFlow.__init__(agents)                         │ │
│  3. 设置 executor_keys（默认所有 agent keys）                │ │
│  4. 生成 active_plan_id                                     │ │
└───────────────────────┬─────────────────────────────────────┘ │
                        │                                     │ │
                        ▼                                     │ │
┌─────────────────────────────────────────────────────────────┐ │
│        PlanningFlow.execute(input_text)                     │ │
│  1. await self._create_initial_plan(input_text)  ←───┐    │ │
│  2. while True:                                       │    │ │
│       step_index, step_info = await _get_current_step_info() │ │
│       if step_index is None:                          │    │ │
│           result += await _finalize_plan()            │    │ │
│           break                                       │    │ │
│       executor = get_executor(step_type)              │    │ │
│       step_result = await _execute_step(executor, step_info) │ │
│       if executor.state == FINISHED:                  │    │ │
│           break                                       │    │ │
│  3. return result                                     │    │ │
└───────────────────────┬─────────────────────────────────────┘ │
                        │                                     │ │
                        │                                     │ │
                        ▼                                     │ │
┌─────────────────────────────────────────────────────────────┐ │
│        _create_initial_plan(request)                        │ │
│  1. 构建系统提示（包含可用 Agent 描述）                      │ │
│  2. 创建用户消息（请求创建计划）                             │ │
│  3. response = await self.llm.ask_tool(                    │ │
│       messages=[user_message],                             │ │
│       system_msgs=[system_message],                        │ │
│       tools=[planning_tool.to_param()]                     │ │
│     )                                                       │ │
│  4. 处理工具调用，执行 planning.create 命令                 │ │
│  5. PlanningTool 创建计划并存储                            │ │
└─────────────────────────────────────────────────────────────┘ │
                        │                                     │ │
                        │                                     │ │
                        ▼                                     │ │
┌─────────────────────────────────────────────────────────────┐ │
│        _get_current_step_info()                             │ │
│  1. 从 planning_tool.plans 获取计划数据                     │ │
│  2. 查找第一个状态为 "not_started" 或 "in_progress" 的步骤  │ │
│  3. 提取步骤类型（如果有 [AGENT_NAME] 格式）                │ │
│  4. 标记步骤为 "in_progress"                                │ │
│  5. 返回 (step_index, step_info)                           │ │
└───────────────────────┬─────────────────────────────────────┘ │
                        │                                     │ │
                        │                                     │ │
                        ▼                                     │ │
┌─────────────────────────────────────────────────────────────┐ │
│        _execute_step(executor, step_info)                   │ │
│  1. plan_status = await _get_plan_text()                    │ │
│  2. 构建 step_prompt（包含计划状态和当前步骤）               │ │
│  3. step_result = await executor.run(step_prompt)  ←───┐  │ │
│  4. await _mark_step_completed()                        │  │ │
│  5. return step_result                                  │  │ │
└───────────────────────┬─────────────────────────────────────┘ │
                        │                                     │ │
                        │                                     │ │
                        ▼                                     │ │
┌─────────────────────────────────────────────────────────────┐ │
│        executor.run(step_prompt)                            │ │
│  【递归调用 Agent 的执行流程，与 main.py 相同】              │ │
│  1. BaseAgent.run(step_prompt)                             │ │
│  2. 循环执行 ReActAgent.step()                             │ │
│     - ToolCallAgent.think()                                │ │
│     - ToolCallAgent.act()                                  │ │
│  3. 返回执行结果                                            │ │
└───────────────────────┬─────────────────────────────────────┘ │
                        │                                     │ │
                        │                                     │ │
                        ▼                                     │ │
┌─────────────────────────────────────────────────────────────┐ │
│        _mark_step_completed()                               │ │
│  1. await planning_tool.execute(                           │ │
│       command="mark_step",                                 │ │
│       step_status="completed"                              │ │
│     )                                                       │ │
│  2. 更新计划中步骤状态为 "completed"                        │ │
└───────────────────────┬─────────────────────────────────────┘ │
                        │                                     │ │
                        │                                     │ │
                        └─────────────────────────────────────┘ │
                                                                │
                        ┌───────────────────────────────────────┘
                        │
                        ▼
                    返回 step_result
                        │
                        └──→ 继续循环执行下一个步骤
```

---

## 十一、与 main.py 的主要区别

### 11.1 架构差异

| 方面 | main.py | run_flow.py |
|------|---------|-------------|
| **执行模式** | 直接使用 Agent | 使用 Flow 管理 Agent |
| **任务规划** | Agent 内部自主规划 | Flow 先创建计划，再逐步执行 |
| **执行结构** | 单个 Agent 的 Think-Act 循环 | Flow 管理多个步骤，每个步骤由 Agent 执行 |
| **Agent 数量** | 单个 Manus Agent | 支持多个 Agent（Manus + 可选 DataAnalysis） |
| **计划管理** | 无显式计划 | 使用 PlanningTool 管理计划 |

### 11.2 执行流程对比

#### main.py 的执行流程

```
用户输入
  ↓
创建 Manus Agent
  ↓
agent.run(prompt)
  ↓
循环执行 step()（Think-Act 模式）
  ↓
直接执行任务，Agent 内部自主规划
```

#### run_flow.py 的执行流程

```
用户输入
  ↓
创建多个 Agent（Manus + 可选 DataAnalysis）
  ↓
创建 PlanningFlow
  ↓
flow.execute(prompt)
  ↓
1. 创建初始计划（使用 LLM + PlanningTool）
  ↓
2. 循环执行计划中的每个步骤：
   - 获取当前步骤
   - 选择执行器 Agent
   - 调用 executor.run(step_prompt)（递归执行 Agent）
   - 标记步骤完成
  ↓
3. 完成计划并生成总结
```

### 11.3 关键区别总结

1. **计划先行**：
   - `main.py`：Agent 在执行过程中自主规划
   - `run_flow.py`：Flow 先创建完整计划，然后逐步执行

2. **步骤管理**：
   - `main.py`：无显式步骤管理，Agent 自主决定下一步
   - `run_flow.py`：显式管理步骤，每个步骤有明确的状态（not_started/in_progress/completed）

3. **多 Agent 支持**：
   - `main.py`：单个 Agent
   - `run_flow.py`：支持多个 Agent，可以根据步骤类型选择不同的 Agent

4. **执行粒度**：
   - `main.py`：Agent 级别的 Think-Act 循环
   - `run_flow.py`：Flow 级别的步骤循环，每个步骤内部是 Agent 级别的 Think-Act 循环

5. **Agent 初始化**：
   - `main.py`：使用 `Manus.create()` 异步初始化（包括 MCP 服务器连接）
   - `run_flow.py`：直接实例化 `Manus()` 和 `DataAnalysis()`（同步创建）

6. **超时控制**：
   - `main.py`：无超时控制
   - `run_flow.py`：使用 `asyncio.wait_for()` 设置 1 小时超时

7. **资源清理**：
   - `main.py`：在 `finally` 块中调用 `agent.cleanup()`
   - `run_flow.py`：无显式清理（依赖 Agent 内部的清理机制）

### 11.4 使用场景

- **main.py 适用于**：
  - 简单的单步任务
  - 需要 Agent 自主规划的场景
  - 快速原型开发

- **run_flow.py 适用于**：
  - 复杂的多步骤任务
  - 需要显式计划和进度跟踪的场景
  - 需要多个 Agent 协作的场景
  - 需要任务可追溯和可重试的场景

---

## 十二、执行流程详细对比表

| 阶段 | main.py | run_flow.py |
|------|---------|-------------|
| **初始化** | `Manus.create()`（异步，连接 MCP） | 直接实例化 `Manus()` 和 `DataAnalysis()`（同步） |
| **计划创建** | 无（Agent 内部自主规划） | `_create_initial_plan()`（使用 LLM + PlanningTool） |
| **执行循环** | `BaseAgent.run()` → `step()` 循环 | `PlanningFlow.execute()` → 步骤循环 |
| **每次迭代** | Think → Act | 获取步骤 → 选择 Agent → 执行 Agent → 标记完成 |
| **Agent 执行** | 直接执行 | 递归调用 `executor.run(step_prompt)` |
| **状态管理** | AgentState（IDLE/RUNNING/FINISHED） | 计划步骤状态（not_started/in_progress/completed） |
| **工具使用** | Agent 的工具集合 | Agent 的工具集合 + PlanningTool |
| **多 Agent** | 不支持 | 支持（通过 executor_keys 和 step_type） |
| **进度跟踪** | 无显式跟踪 | 通过 PlanningTool 跟踪计划进度 |
| **完成处理** | Agent 状态变为 FINISHED | `_finalize_plan()` 生成总结 |

---

## 十三、总结

### 13.1 run_flow.py 的核心特点

1. **计划驱动执行**：先创建计划，再逐步执行
2. **步骤状态管理**：每个步骤有明确的状态跟踪
3. **多 Agent 协作**：支持根据步骤类型选择不同的 Agent
4. **可追溯性**：所有步骤和状态都被记录在 PlanningTool 中

### 13.2 关键设计模式

- **Flow 模式**：使用 Flow 来管理复杂的多步骤任务
- **计划模式**：先规划后执行，提高任务的可控性
- **策略模式**：根据步骤类型选择不同的执行器 Agent

### 13.3 执行效率对比

- **main.py**：适合简单任务，执行更直接，开销更小
- **run_flow.py**：适合复杂任务，虽然增加了计划创建的开销，但提供了更好的可控性和可追溯性

---

**文档生成时间**：2024年
**版本**：基于当前代码库分析
