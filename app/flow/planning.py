import json
import time
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import Field

from app.agent.base import BaseAgent
from app.flow.base import BaseFlow
from app.flow.human_interaction import (
    ask_human_confirmation,
    ask_human_feedback,
    display_text_with_pagination,
)
from app.human_io import get_human_io
from app.llm import LLM
from app.logger import logger
from app.schema import AgentState, Message, ToolChoice
from app.tool import PlanningTool


class PlanStepStatus(str, Enum):
    """Enum class defining possible statuses of a plan step"""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"

    @classmethod
    def get_all_statuses(cls) -> list[str]:
        """Return a list of all possible step status values"""
        return [status.value for status in cls]

    @classmethod
    def get_active_statuses(cls) -> list[str]:
        """Return a list of values representing active statuses (not started or in progress)"""
        return [cls.NOT_STARTED.value, cls.IN_PROGRESS.value]

    @classmethod
    def get_status_marks(cls) -> Dict[str, str]:
        """Return a mapping of statuses to their marker symbols"""
        return {
            cls.COMPLETED.value: "[✓]",
            cls.IN_PROGRESS.value: "[→]",
            cls.BLOCKED.value: "[!]",
            cls.NOT_STARTED.value: "[ ]",
        }


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

    def get_executor(self, step_type: Optional[str] = None) -> BaseAgent:
        """
        Get an appropriate executor agent for the current step.
        Can be extended to select agents based on step type/requirements.
        """
        # If step type is provided and matches an agent key, use that agent
        if step_type and step_type in self.agents:
            return self.agents[step_type]

        # Otherwise use the first available executor or fall back to primary agent
        for key in self.executor_keys:
            if key in self.agents:
                return self.agents[key]

        # Fallback to primary agent
        return self.primary_agent

    async def execute(
        self,
        input_text: str,
        planning_context: str = "",
        business_files: dict[str, str] = None
    ) -> str:
        """
        Execute the planning flow with agents.

        Args:
            input_text: The user's request/prompt
            planning_context: Optional context from uploaded planning template file
            business_files: Optional dictionary mapping file paths to their contents
        """
        if business_files is None:
            business_files = {}
        try:
            if not self.primary_agent:
                raise ValueError("No primary agent available")

            # Create initial plan if input provided
            if input_text:
                await self._create_initial_plan(input_text, planning_context, business_files)

                # Verify plan was created successfully
                if self.active_plan_id not in self.planning_tool.plans:
                    logger.error(
                        f"Plan creation failed. Plan ID {self.active_plan_id} not found in planning tool."
                    )
                    return f"Failed to create plan for: {input_text}"

                # Human feedback: confirm plan before execution
                if not await self._confirm_plan_with_human():
                    logger.info("Plan execution cancelled by user.")
                    return "Plan execution cancelled by user."

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

                logger.info(f"[PLANNING FLOW] Starting execution of step {self.current_step_index}")
                await get_human_io().emit(
                    {
                        "type": "step_start",
                        "plan_id": self.active_plan_id,
                        "step_index": self.current_step_index,
                        "step": step_info,
                    }
                )
                step_result = await self._execute_step(executor, step_info)
                logger.info(f"[PLANNING FLOW] Step {self.current_step_index} execution returned. Result type: {type(step_result)}, Length: {len(str(step_result)) if step_result else 0}")

                result += step_result + "\n"

                # 发送 step_end 事件
                await get_human_io().emit(
                    {
                        "type": "step_end",
                        "plan_id": self.active_plan_id,
                        "step_index": self.current_step_index,
                        "result": str(step_result) if step_result is not None else "",
                    }
                )
                logger.info(f"[PLANNING FLOW] Step {self.current_step_index} step_end event emitted")

                # Human feedback: confirm step result before continuing
                # Force flush before human interaction
                import sys
                sys.stdout.flush()
                sys.stderr.flush()

                logger.info(f"[PLANNING FLOW] Step {self.current_step_index} execution completed. Result length: {len(str(step_result)) if step_result else 0}")
                logger.info(f"[PLANNING FLOW] About to call _confirm_step_result for step {self.current_step_index}")
                print(f"\n[DEBUG] Step {self.current_step_index} completed. Preparing for human feedback...")
                sys.stdout.flush()

                # 确保步骤确认不会永久卡住，添加超时保护
                try:
                    import asyncio
                    logger.info(f"[PLANNING FLOW] Calling _confirm_step_result for step {self.current_step_index} with timeout 600s")
                    should_continue = await asyncio.wait_for(
                        self._confirm_step_result(str(step_result) if step_result else "[No result]"),
                        timeout=600.0  # 10分钟超时，避免永久卡住
                    )
                    logger.info(f"[PLANNING FLOW] Step {self.current_step_index} confirmation completed. Result: {should_continue}")
                    if not should_continue:
                        logger.info("Step execution cancelled by user. Stopping execution.")
                        result += "\n[Execution stopped by user feedback]"
                        break
                except asyncio.TimeoutError:
                    logger.warning(f"[PLANNING FLOW] Step {self.current_step_index} confirmation timed out after 10 minutes, defaulting to continue")
                    should_continue = True  # 超时默认继续，避免卡住
                    await get_human_io().emit({
                        "type": "error",
                        "message": f"步骤 {self.current_step_index} 确认超时，将自动继续执行下一步"
                    })
                except Exception as e:
                    logger.error(f"[PLANNING FLOW] Error in step result confirmation for step {self.current_step_index}: {e}", exc_info=True)
                    import traceback
                    traceback.print_exc()
                    # 错误时默认继续，避免卡住整个流程
                    should_continue = True
                    await get_human_io().emit({
                        "type": "error",
                        "message": f"步骤 {self.current_step_index} 确认过程中出现错误: {str(e)}，将继续执行下一步"
                    })

                logger.info(f"[PLANNING FLOW] Step {self.current_step_index} human feedback completed, should_continue={should_continue}")

                # 如果用户选择继续，立即发送事件通知前端，然后继续下一个步骤
                if should_continue:
                    logger.info(f"[PLANNING FLOW] User confirmed step {self.current_step_index}, proceeding to next step")
                    await get_human_io().emit({
                        "type": "step_confirmed",
                        "plan_id": self.active_plan_id,
                        "step_index": self.current_step_index,
                        "action": "continue"
                    })
                    # 继续循环，执行下一个步骤
                    continue
                else:
                    logger.info(f"[PLANNING FLOW] User rejected step {self.current_step_index}, stopping execution")
                    await get_human_io().emit({
                        "type": "step_confirmed",
                        "plan_id": self.active_plan_id,
                        "step_index": self.current_step_index,
                        "action": "stop"
                    })
                    break

            return result
        except Exception as e:
            logger.error(f"Error in PlanningFlow: {str(e)}")
            return f"Execution failed: {str(e)}"

    async def _create_initial_plan(
        self,
        request: str,
        planning_context: str = "",
        business_files: dict[str, str] = None
    ) -> None:
        """
        Create an initial plan based on the request using the flow's LLM and PlanningTool.

        Args:
            request: The user's request/prompt
            planning_context: Optional context from uploaded planning template file to guide plan creation
            business_files: Optional dictionary mapping file paths to their contents
        """
        if business_files is None:
            business_files = {}

        logger.info(f"Creating initial plan with ID: {self.active_plan_id}")

        system_message_content = (
            "You are a planning assistant. Create a concise, actionable plan with clear steps. "
            "Focus on key milestones rather than detailed sub-steps. "
            "Optimize for clarity and efficiency."
        )

        # Add planning context if provided
        if planning_context:
            system_message_content += (
                f"\n\nIMPORTANT: The following planning guidelines/context should be used as a reference "
                f"when creating the plan. Base your plan steps on these guidelines:\n\n{planning_context}\n\n"
                "Use these guidelines to structure your plan, but adapt them to the specific request."
            )

        # Add business files context if provided
        if business_files:
            system_message_content += (
                f"\n\nBUSINESS FILES CONTEXT:\n"
                f"The following business files are available in the workspace and should be considered when creating the plan:\n\n"
            )
            for file_path, content in business_files.items():
                # Truncate very long files to avoid token limits
                content_preview = content[:2000] + "..." if len(content) > 2000 else content
                system_message_content += (
                    f"File: {file_path}\n"
                    f"Content:\n{content_preview}\n\n"
                )
            system_message_content += (
                "These files contain relevant business information. Reference them when creating plan steps. "
                "The agent can access these files directly from the workspace during execution."
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
        user_message_content = f"Create a reasonable plan with clear steps to accomplish the task: {request}"
        if planning_context:
            user_message_content += f"\n\nPlease refer to the planning guidelines provided in the system message."
        if business_files:
            user_message_content += (
                f"\n\nNote: Relevant business files are available in the workspace. "
                f"The agent can read and process these files during plan execution."
            )

        user_message = Message.user_message(user_message_content)

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

    async def _execute_step(self, executor: BaseAgent, step_info: dict) -> str:
        """Execute the current step with the specified agent using agent.run()."""
        # 重置 Agent 的状态，确保每个 planning step 都从 step 0 开始计数
        # 强制重置，确保每次执行都从干净状态开始
        if hasattr(executor, 'current_step'):
            executor.current_step = 0
            logger.info(f"[PLANNING FLOW] Reset agent current_step to 0 for step {self.current_step_index}")
        if hasattr(executor, 'state'):
            executor.state = AgentState.IDLE
            logger.info(f"[PLANNING FLOW] Reset agent state to IDLE for step {self.current_step_index}")

        # Prepare context for the agent with current plan status
        plan_status = await self._get_plan_text()
        step_text = step_info.get("text", f"Step {self.current_step_index}")

        # Create a prompt for the agent to execute the current step
        step_prompt = f"""
        CURRENT PLAN STATUS:
        {plan_status}

        YOUR CURRENT TASK:
        You are now working on step {self.current_step_index}: "{step_text}"

        IMPORTANT INSTRUCTIONS:
        1. You must complete this step 100% before marking it as finished.
        2. Use the appropriate tools to accomplish the task fully.
        3. When you have COMPLETELY finished this step (100% done), you MUST set your state to FINISHED to indicate completion.
        4. Only mark as FINISHED when the step is truly complete - do not finish prematurely.
        5. If you reach the maximum number of steps but the task is not complete, return a summary indicating what was accomplished and what remains.

        When you're done, provide a CLEAR and USER-FRIENDLY summary of what you accomplished, focusing on the actual information extracted or actions completed, not technical implementation details.

        Your final summary should be readable and meaningful to a human user, not just technical logs.
        """

        # Use agent.run() to execute the step
        try:
            # 记录执行前的状态
            initial_state = executor.state if hasattr(executor, 'state') else None
            logger.info(f"[PLANNING FLOW] Starting execution of planning step {self.current_step_index}. Agent initial state: {initial_state}")

            step_result = await executor.run(step_prompt)

            # 从结果中提取完成状态信息（Agent在结果中标记了完成状态）
            is_completed = False
            steps_taken = 0
            max_steps = executor.max_steps if hasattr(executor, 'max_steps') else 15

            # 检查结果中是否包含完成标记
            if step_result and "[AGENT_COMPLETED:" in step_result:
                is_completed = True
                # 提取步数信息
                import re
                match = re.search(r'\[AGENT_COMPLETED:100%:steps=(\d+)\]', step_result)
                if match:
                    steps_taken = int(match.group(1))
                logger.info(f"[PLANNING FLOW] Planning step {self.current_step_index} completed successfully (100%) after {steps_taken} steps")
            elif step_result and "[AGENT_INCOMPLETE:" in step_result:
                is_completed = False
                # 提取步数信息
                import re
                match = re.search(r'\[AGENT_INCOMPLETE:steps=(\d+)/(\d+)\]', step_result)
                if match:
                    steps_taken = int(match.group(1))
                    max_steps = int(match.group(2))
                logger.warning(f"[PLANNING FLOW] Planning step {self.current_step_index} may not be fully completed. Steps: {steps_taken}/{max_steps}")
            else:
                # 没有明确标记，假设完成（向后兼容）
                is_completed = True
                logger.info(f"[PLANNING FLOW] Planning step {self.current_step_index} execution completed (no explicit completion marker)")

            # 优先从 Agent 的 memory 中提取最后的 assistant message（包含总结）
            meaningful_result = None
            if hasattr(executor, 'memory') and executor.memory and hasattr(executor.memory, 'messages'):
                # 从后往前查找 assistant message
                for msg in reversed(executor.memory.messages):
                    if hasattr(msg, 'role') and msg.role == 'assistant' and hasattr(msg, 'content'):
                        content = msg.content or ""
                        # 如果内容包含总结性关键词且长度足够，使用它
                        if len(content) > 100 and any(keyword in content for keyword in ["总结", "完成", "已经", "收集", "获取", "##", "###", "GitHub", "功能", "信息"]):
                            meaningful_result = content
                            logger.info(f"[PLANNING FLOW] Found meaningful result in agent memory (length: {len(content)})")
                            break

            # Clean up the result - extract meaningful content from technical logs
            # 先移除完成状态标记，然后提取有意义的内容
            cleaned_result = step_result
            # 移除Agent添加的完成状态标记
            import re
            cleaned_result = re.sub(r'\[AGENT_(COMPLETED|INCOMPLETE|UNCERTAIN):[^\]]+\]\n?', '', cleaned_result)

            # 如果从 memory 中找到了有意义的结果，优先使用它
            if meaningful_result:
                cleaned_result = meaningful_result
            else:
                # 否则从 step_result 中提取
                cleaned_result = self._extract_meaningful_result(cleaned_result)

            # 根据完成状态添加用户友好的标记
            if is_completed:
                # 任务完成，在结果中添加完成标记
                cleaned_result = f"[✅ 步骤 {self.current_step_index} 已完成 100%]\n\n{cleaned_result}"
            else:
                # 任务可能未完成，在结果中添加警告
                cleaned_result = f"[⚠️ 步骤 {self.current_step_index} 可能未完全完成（执行了 {steps_taken}/{max_steps} 步）]\n\n{cleaned_result}\n\n[提示]：如果任务未完成，请提供反馈或选择继续执行。"

            # Mark the step as completed after execution (无论是否100%完成，都标记为完成，让人类决定是否继续)
            await self._mark_step_completed()

            return cleaned_result
        except Exception as e:
            logger.error(f"Error executing step {self.current_step_index}: {e}")
            return f"Error executing step {self.current_step_index}: {str(e)}"

    def _extract_meaningful_result(self, raw_result: str) -> str:
        """Extract meaningful content from technical tool output."""
        if not raw_result:
            return "[步骤执行完成，但未产生明确结果]"

        import re
        import json

        # 优先查找包含总结性内容的部分（通常以 ## 或 ### 开头，包含"总结"、"完成"等关键词）
        # 匹配格式：## 标题\n...内容... 或包含总结性关键词的长文本
        summary_patterns = [
            r'(##[^\n]+\n.*?)(?=\n\n|$)',
            r'(###[^\n]+\n.*?)(?=\n\n|$)',
            r'(.{200,}?(?:总结|完成|已经|收集|获取|功能|信息).{200,}?)(?=\n\n|$)',
        ]

        for pattern in summary_patterns:
            matches = re.findall(pattern, raw_result, re.DOTALL | re.IGNORECASE)
            if matches:
                for match in matches:
                    match = match.strip()
                    # 如果匹配的内容包含总结性关键词且长度足够
                    if len(match) > 100 and any(keyword in match for keyword in ["总结", "完成", "已经", "收集", "获取", "##", "###", "GitHub", "功能", "信息"]):
                        # 清理技术性前缀
                        match = re.sub(r'^Observed output of cmd.*?executed:\s*', '', match, flags=re.IGNORECASE)
                        match = re.sub(r'^The interaction has been completed.*?$', '', match, flags=re.IGNORECASE | re.MULTILINE)
                        match = match.strip()
                        if len(match) > 100:
                            return match[:3000] + ("\n...(内容已截断)" if len(match) > 3000 else "")

        # 其次查找提取的内容（浏览器工具提取的页面内容）
        # 匹配格式：Extracted from page: {'text': '...', ...}
        extracted_patterns = [
            r"Extracted from page[:\s]*(\{.*?\})",
            r"extracted_content[:\s]*(\{.*?\})",
            r"'text':\s*'([^']{50,})'",  # 至少50字符的文本
            r'"text":\s*"([^"]{50,})"',
        ]

        for pattern in extracted_patterns:
            matches = re.findall(pattern, raw_result, re.DOTALL | re.IGNORECASE)
            if matches:
                for match in matches:
                    try:
                        # 尝试解析 JSON
                        if match.startswith('{'):
                            data = json.loads(match)
                            if 'text' in data:
                                text = data['text']
                                # 如果文本太长，截取前1000字符
                                if len(text) > 1000:
                                    return text[:1000] + "\n...(内容已截断)"
                                return text
                            # 如果有其他有意义的内容
                            if 'extracted_content' in data:
                                content = data['extracted_content']
                                if isinstance(content, dict) and 'text' in content:
                                    return content['text'][:1000] if len(content['text']) > 1000 else content['text']
                        else:
                            # 直接返回匹配的文本
                            if len(match) > 50:
                                return match[:1000] + ("\n...(内容已截断)" if len(match) > 1000 else "")
                    except json.JSONDecodeError:
                        # JSON 解析失败，尝试直接提取文本
                        if len(match) > 50:
                            return match[:1000] + ("\n...(内容已截断)" if len(match) > 1000 else "")
                    except Exception:
                        pass

        # 如果没有找到提取的内容，清理技术性日志
        # 移除 "Observed output of cmd" 前缀
        cleaned = re.sub(r"Observed output of cmd `[^`]+` executed:\s*", "", raw_result, flags=re.IGNORECASE)

        # 移除重复的技术性日志
        cleaned = re.sub(r"Scrolled (down|up) by \d+ pixels\s*\n?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"The interaction has been completed with status: success\s*\n?", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r"Error: Browser action.*?failed.*?\n", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"ERROR\s+\[browser\].*?\n", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r"WARNING\s+\[browser\].*?\n", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)

        # 查找有意义的内容行
        lines = cleaned.split('\n')
        meaningful_lines = []
        skip_patterns = [
            r'^Observed output',
            r'^Scrolled',
            r'^步骤 \d+:',
            r'^\[执行了',
        ]

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 跳过技术性日志
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
                continue
            # 保留有意义的内容
            if len(line) > 20:
                meaningful_lines.append(line)
                if len('\n'.join(meaningful_lines)) > 1000:
                    break

        if meaningful_lines:
            result = '\n'.join(meaningful_lines)
            return result[:1000] + ("\n...(内容已截断)" if len(result) > 1000 else result)

        # 如果还是没有找到有意义的内容，返回清理后的原始结果（截断）
        if cleaned:
            return cleaned[:500] + ("\n...(内容已截断，完整内容见事件日志)" if len(cleaned) > 500 else "")

        return raw_result[:500] + ("\n...(内容已截断)" if len(raw_result) > 500 else raw_result)

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

    async def _get_plan_text(self) -> str:
        """Get the current plan as formatted text."""
        try:
            result = await self.planning_tool.execute(
                command="get", plan_id=self.active_plan_id
            )
            return result.output if hasattr(result, "output") else str(result)
        except Exception as e:
            logger.error(f"Error getting plan: {e}")
            return self._generate_plan_text_from_storage()

    def _generate_plan_text_from_storage(self) -> str:
        """Generate plan text directly from storage if the planning tool fails."""
        try:
            if self.active_plan_id not in self.planning_tool.plans:
                return f"Error: Plan with ID {self.active_plan_id} not found"

            plan_data = self.planning_tool.plans[self.active_plan_id]
            title = plan_data.get("title", "Untitled Plan")
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])
            step_notes = plan_data.get("step_notes", [])

            # Ensure step_statuses and step_notes match the number of steps
            while len(step_statuses) < len(steps):
                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
            while len(step_notes) < len(steps):
                step_notes.append("")

            # Count steps by status
            status_counts = {status: 0 for status in PlanStepStatus.get_all_statuses()}

            for status in step_statuses:
                if status in status_counts:
                    status_counts[status] += 1

            completed = status_counts[PlanStepStatus.COMPLETED.value]
            total = len(steps)
            progress = (completed / total) * 100 if total > 0 else 0

            plan_text = f"Plan: {title} (ID: {self.active_plan_id})\n"
            plan_text += "=" * len(plan_text) + "\n\n"

            plan_text += (
                f"Progress: {completed}/{total} steps completed ({progress:.1f}%)\n"
            )
            plan_text += f"Status: {status_counts[PlanStepStatus.COMPLETED.value]} completed, {status_counts[PlanStepStatus.IN_PROGRESS.value]} in progress, "
            plan_text += f"{status_counts[PlanStepStatus.BLOCKED.value]} blocked, {status_counts[PlanStepStatus.NOT_STARTED.value]} not started\n\n"
            plan_text += "Steps:\n"

            status_marks = PlanStepStatus.get_status_marks()

            for i, (step, status, notes) in enumerate(
                zip(steps, step_statuses, step_notes)
            ):
                # Use status marks to indicate step status
                status_mark = status_marks.get(
                    status, status_marks[PlanStepStatus.NOT_STARTED.value]
                )

                plan_text += f"{i}. {status_mark} {step}\n"
                if notes:
                    plan_text += f"   Notes: {notes}\n"

            return plan_text
        except Exception as e:
            logger.error(f"Error generating plan text from storage: {e}")
            return f"Error: Unable to retrieve plan with ID {self.active_plan_id}"

    async def _confirm_plan_with_human(self) -> bool:
        """
        Display the created plan and ask human for confirmation before execution.
        Supports plan modification based on user feedback.

        Returns:
            True if user confirms, False if user rejects
        """
        max_modification_attempts = 3
        modification_count = 0

        while modification_count < max_modification_attempts:
            try:
                plan_text = await self._get_plan_text()

                # Force flush any pending output before displaying plan
                import sys
                sys.stdout.flush()

                print("\n" + "=" * 80)
                print("GENERATED PLAN - Please Review")
                print("=" * 80)
                await get_human_io().emit(
                    {
                        "type": "plan",
                        "plan_id": self.active_plan_id,
                        "text": plan_text,
                    }
                )
                await display_text_with_pagination(plan_text, page_size=30)
                print("=" * 80)
                sys.stdout.flush()  # Ensure plan is displayed before asking for confirmation

                # Ask for confirmation
                # Add a small delay to ensure all output is flushed
                import asyncio
                await asyncio.sleep(0.1)  # Small delay to ensure output is flushed
                confirmed = await ask_human_confirmation(
                    "\nDo you want to proceed with this plan?",
                    default="y"
                )

                if confirmed:
                    return True

                # User rejected the plan - offer modification
                print("\n" + "-" * 80)
                print("Plan Modification Options:")
                print("-" * 80)
                print("1. Provide feedback to modify the plan")
                print("2. Cancel and exit")
                print("-" * 80)

                choice = await get_human_io().choose(
                    "\n你选择了【不接受】当前计划。\n请选择下一步操作：\n1 = 提供反馈，让系统修改计划（推荐）\n2 = 取消并退出\n请输入 1 或 2",
                    choices=["1", "2"],
                    default="1",
                )

                if choice == "2":
                    logger.info("Plan execution cancelled by user.")
                    return False

                # Get modification feedback
                feedback = await ask_human_feedback(
                    "\nPlease provide your feedback to modify the plan (e.g., 'Add step X', 'Remove step Y', 'Modify step Z to...'):",
                    allow_empty=False
                )

                if not feedback:
                    return await ask_human_confirmation(
                        "\nNo feedback provided. Do you want to proceed with the original plan?",
                        default="n"
                    )

                logger.info(f"User feedback for plan modification (attempt {modification_count + 1}): {feedback}")

                # Modify the plan based on feedback
                success = await self._modify_plan_with_feedback(feedback)
                if success:
                    modification_count += 1
                    # Force flush output before showing modified plan
                    import sys
                    sys.stdout.flush()
                    print(f"\n✓ Plan has been modified. Please review the updated plan.")
                    sys.stdout.flush()
                    # Continue loop to show modified plan (will show plan and ask for confirmation again)
                else:
                    print("\n✗ Failed to modify plan. Please try again or proceed with original plan.")
                    sys.stdout.flush()
                    return await ask_human_confirmation(
                        "\nDo you want to proceed with the original plan?",
                        default="n"
                    )

            except Exception as e:
                logger.error(f"Error in plan confirmation: {e}")
                return await ask_human_confirmation(
                    "\nError displaying plan. Do you want to proceed anyway?",
                    default="y"
                )

        # Max attempts reached
        print(f"\n⚠ Maximum modification attempts ({max_modification_attempts}) reached.")
        return await ask_human_confirmation(
            "\nDo you want to proceed with the current plan?",
            default="n"
        )

    async def _modify_plan_with_feedback(self, feedback: str) -> bool:
        """
        Modify the plan based on user feedback using LLM.

        Args:
            feedback: User's feedback on how to modify the plan

        Returns:
            True if modification was successful, False otherwise
        """
        try:
            current_plan_text = await self._get_plan_text()

            system_message_content = (
                "You are a planning assistant. Your task is to modify an existing plan based on user feedback. "
                "You should use the planning tool to update the plan with the requested changes. "
                "When modifying steps, preserve the status of steps that haven't changed."
            )

            user_message_content = (
                f"Current plan:\n\n{current_plan_text}\n\n"
                f"User feedback: {feedback}\n\n"
                f"Please modify the plan according to the feedback. Use the planning tool's 'update' command "
                f"to update the plan (plan_id: {self.active_plan_id}). "
                f"Make sure to preserve step statuses for unchanged steps."
            )

            system_message = Message.system_message(system_message_content)
            user_message = Message.user_message(user_message_content)

            # Call LLM with PlanningTool to modify the plan
            response = await self.llm.ask_tool(
                messages=[user_message],
                system_msgs=[system_message],
                tools=[self.planning_tool.to_param()],
                tool_choice=ToolChoice.AUTO,
            )

            # Process tool calls
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call.function.name == "planning":
                        args = tool_call.function.arguments
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse tool arguments: {args}")
                                continue

                        # Ensure plan_id is set correctly
                        args["plan_id"] = self.active_plan_id

                        # Execute the tool
                        result = await self.planning_tool.execute(**args)
                        logger.info(f"Plan modification result: {str(result)}")
                        return True

            logger.warning("No planning tool call found in LLM response")
            return False

        except Exception as e:
            logger.error(f"Error modifying plan with feedback: {e}")
            return False

    async def _confirm_step_result(self, step_result: str) -> bool:
        """
        Display step execution result and ask human for feedback/confirmation.
        Supports result modification and storing corrections in memory.

        Args:
            step_result: The result from executing the current step

        Returns:
            True if user wants to continue, False if user wants to stop
        """
        try:
            # Validate input
            if step_result is None:
                step_result = "[No result returned]"
            elif not isinstance(step_result, str):
                step_result = str(step_result)

            logger.info(f"[_confirm_step_result] Displaying step {self.current_step_index} result confirmation (result length: {len(step_result)})")

            step_info = None
            if self.current_step_index is not None:
                plan_data = self.planning_tool.plans.get(self.active_plan_id, {})
                steps = plan_data.get("steps", [])
                if self.current_step_index < len(steps):
                    step_info = steps[self.current_step_index]

            # Force flush to ensure output is displayed
            import sys
            sys.stdout.flush()
            sys.stderr.flush()

            # Add a very visible separator
            print("\n" + "=" * 80)
            print("=" * 80)
            print(f"  STEP {self.current_step_index} EXECUTION RESULT - HUMAN FEEDBACK REQUIRED")
            print("=" * 80)
            print("=" * 80)
            if step_info:
                print(f"\nStep: {step_info}")
            else:
                print(f"\nStep {self.current_step_index}: [No step info]")

            # Display result with pagination if too long
            if len(step_result) > 2000:
                print(f"\nResult (truncated, full length: {len(step_result)} characters):\n{step_result[:2000]}...")
                print("\n[Result is too long, showing first 2000 characters. Full result will be used for feedback.]")
            else:
                print(f"\nResult:\n{step_result}")

            print("=" * 80)
            print("=" * 80)
            sys.stdout.flush()  # Force flush again
            sys.stderr.flush()

            logger.info(f"[_confirm_step_result] About to ask user: Is this step result acceptable?")

            # Emit step confirmation request to frontend (for UI display only)
            step_info_text = step_info.get("text", str(step_info)) if isinstance(step_info, dict) else str(step_info) if step_info else "未知步骤"
            await get_human_io().emit(
                {
                    "type": "step_confirmation_request",
                    "plan_id": self.active_plan_id,
                    "step_index": self.current_step_index,
                    "step_info": step_info_text,
                    "result_preview": step_result[:500] + "..." if len(step_result) > 500 else step_result,
                }
            )

            # Ask if result is acceptable and whether to continue to next step (with timeout protection)
            # This will create a proper human_request with request_id
            try:
                import asyncio
                # 改进提示词，明确询问是否继续下一步
                confirmation_prompt = f"""
步骤 {self.current_step_index} 执行完成。

步骤内容: {step_info_text}

结果预览:
{step_result[:500] + "..." if len(step_result) > 500 else step_result}

请选择：
- 输入 'y' 或 'yes'：接受当前结果并继续执行下一步
- 输入 'n' 或 'no'：不接受当前结果，需要修改或反馈

是否接受当前结果并继续下一步？"""

                result_acceptable = await asyncio.wait_for(
                    ask_human_confirmation(
                        confirmation_prompt,
                        default="y"
                    ),
                    timeout=600.0  # 10分钟超时，避免永久卡住
                )
            except asyncio.TimeoutError:
                logger.warning(f"Step {self.current_step_index} confirmation timed out, defaulting to continue")
                result_acceptable = True  # 超时默认继续
                await get_human_io().emit({
                    "type": "error",
                    "message": f"步骤 {self.current_step_index} 确认超时，将自动继续"
                })

            if not result_acceptable:
                # Result is not acceptable - offer modification options
                print("\n" + "-" * 80)
                print("Result Modification Options:")
                print("-" * 80)
                print("1. Provide correction/feedback (will be stored in memory)")
                print("2. Mark step as needing re-execution")
                print("3. Continue anyway")
                print("-" * 80)

                choice = await get_human_io().choose(
                    "\n你选择了【不接受】当前步骤结果。\n\n请选择下一步操作：\n\n1 = 提供纠正/反馈（会记录到 notes/记忆，便于后续步骤参考）\n2 = 标记该步骤为阻塞/需要重新执行\n3 = 接受当前结果并继续（可选填写备注）\n\n请输入选项编号：",
                    choices=["1", "2", "3"],
                    default="1",
                )

                if choice == "2":
                    # Mark step as blocked and ask if user wants to continue
                    try:
                        await self.planning_tool.execute(
                            command="mark_step",
                            plan_id=self.active_plan_id,
                            step_index=self.current_step_index,
                            step_status=PlanStepStatus.BLOCKED.value,
                        )
                        logger.info(f"Marked step {self.current_step_index} as blocked")
                    except Exception as e:
                        logger.warning(f"Failed to mark step as blocked: {e}")

                    return await ask_human_confirmation(
                        "\nStep marked as blocked. Do you want to continue to next step?",
                        default="y"
                    )

                elif choice == "3":
                    # Continue anyway
                    feedback = await ask_human_feedback(
                        "\nEnter any notes about this step (optional):",
                        allow_empty=True
                    )
                    if feedback:
                        await self._store_step_feedback(feedback)
                    return True

                else:
                    # Get correction feedback
                    correction = await ask_human_feedback(
                        "\nPlease provide the correction or what should be changed:",
                        allow_empty=False
                    )

                    if correction:
                        # Store correction in memory and step notes
                        await self._store_step_correction(correction, step_result)
                        print("\n✓ Correction has been stored in memory and step notes.")
                        # 提供反馈后，询问是否继续
                        continue_after_feedback = await ask_human_confirmation(
                            "\n反馈已记录。是否继续执行下一步？",
                            default="y"
                        )
                        logger.info(f"[_confirm_step_result] User provided feedback, continue_after_feedback={continue_after_feedback}")
                        return continue_after_feedback

            else:
                # Result is acceptable - 用户已经选择了"接受并继续"，直接返回 True
                # 不需要再询问是否继续，因为用户已经在第一步确认中选择了继续
                logger.info(f"[_confirm_step_result] Step {self.current_step_index} result accepted, will continue to next step")
                return True
        except Exception as e:
            logger.error(f"Error in step result confirmation: {e}")
            return await ask_human_confirmation(
                "\nError displaying step result. Do you want to continue?",
                default="y"
            )

    async def _store_step_feedback(self, feedback: str) -> None:
        """Store feedback in step notes."""
        try:
            plan_data = self.planning_tool.plans.get(self.active_plan_id, {})
            step_notes = plan_data.get("step_notes", [])
            while len(step_notes) <= self.current_step_index:
                step_notes.append("")

            current_note = step_notes[self.current_step_index]
            if current_note:
                step_notes[self.current_step_index] = f"{current_note}\n[Feedback] {feedback}"
            else:
                step_notes[self.current_step_index] = f"[Feedback] {feedback}"

            plan_data["step_notes"] = step_notes
            logger.info(f"Stored feedback for step {self.current_step_index}")
        except Exception as e:
            logger.warning(f"Failed to store step feedback: {e}")

    async def _store_step_correction(self, correction: str, original_result: str) -> None:
        """
        Store correction in step notes and add to agent memory for learning.

        Args:
            correction: The correction provided by user
            original_result: The original step result that was incorrect
        """
        try:
            # Store in step notes
            plan_data = self.planning_tool.plans.get(self.active_plan_id, {})
            step_notes = plan_data.get("step_notes", [])
            while len(step_notes) <= self.current_step_index:
                step_notes.append("")

            step_info = ""
            steps = plan_data.get("steps", [])
            if self.current_step_index < len(steps):
                step_info = steps[self.current_step_index]

            correction_note = (
                f"[Correction Required]\n"
                f"Original Result: {original_result[:500]}...\n"
                f"Correction: {correction}\n"
                f"Step: {step_info}"
            )

            current_note = step_notes[self.current_step_index]
            if current_note:
                step_notes[self.current_step_index] = f"{current_note}\n\n{correction_note}"
            else:
                step_notes[self.current_step_index] = correction_note

            plan_data["step_notes"] = step_notes

            # Add to agent memory for future reference
            if self.primary_agent and hasattr(self.primary_agent, 'memory'):
                correction_message = (
                    f"Step {self.current_step_index} correction:\n"
                    f"Step: {step_info}\n"
                    f"Original result was incorrect: {original_result[:300]}...\n"
                    f"Correct approach: {correction}"
                )
                self.primary_agent.memory.add_message(
                    Message.user_message(correction_message)
                )
                logger.info(f"Added correction to agent memory for step {self.current_step_index}")

            logger.info(f"Stored correction for step {self.current_step_index}")
        except Exception as e:
            logger.warning(f"Failed to store step correction: {e}")

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
