from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from app.llm import LLM
from app.logger import logger
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import ROLE_TYPE, AgentState, Memory, Message


class BaseAgent(BaseModel, ABC):
    """Abstract base class for managing agent state and execution.

    Provides foundational functionality for state transitions, memory management,
    and a step-based execution loop. Subclasses must implement the `step` method.
    """

    # Core attributes
    name: str = Field(..., description="Unique name of the agent")
    description: Optional[str] = Field(None, description="Optional agent description")

    # Prompts
    system_prompt: Optional[str] = Field(
        None, description="System-level instruction prompt"
    )
    next_step_prompt: Optional[str] = Field(
        None, description="Prompt for determining next action"
    )

    # Dependencies
    llm: LLM = Field(default_factory=LLM, description="Language model instance")
    memory: Memory = Field(default_factory=Memory, description="Agent's memory store")
    state: AgentState = Field(
        default=AgentState.IDLE, description="Current agent state"
    )

    # Execution control
    max_steps: int = Field(default=10, description="Maximum steps before termination")
    current_step: int = Field(default=0, description="Current step in execution")

    duplicate_threshold: int = 2

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields for flexibility in subclasses

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """Initialize agent with default settings if not provided."""
        if self.llm is None or not isinstance(self.llm, LLM):
            self.llm = LLM(config_name=self.name.lower())
        if not isinstance(self.memory, Memory):
            self.memory = Memory()
        return self

    @asynccontextmanager
    async def state_context(self, new_state: AgentState):
        """Context manager for safe agent state transitions.

        Args:
            new_state: The state to transition to during the context.

        Yields:
            None: Allows execution within the new state.

        Raises:
            ValueError: If the new_state is invalid.
        """
        if not isinstance(new_state, AgentState):
            raise ValueError(f"Invalid state: {new_state}")

        previous_state = self.state
        self.state = new_state
        try:
            yield
        except Exception as e:
            self.state = AgentState.ERROR  # Transition to ERROR on failure
            raise e
        finally:
            self.state = previous_state  # Revert to previous state

    def update_memory(
        self,
        role: ROLE_TYPE,  # type: ignore
        content: str,
        base64_image: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Add a message to the agent's memory.

        Args:
            role: The role of the message sender (user, system, assistant, tool).
            content: The message content.
            base64_image: Optional base64 encoded image.
            **kwargs: Additional arguments (e.g., tool_call_id for tool messages).

        Raises:
            ValueError: If the role is unsupported.
        """
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }

        if role not in message_map:
            raise ValueError(f"Unsupported message role: {role}")

        # Create message with appropriate parameters based on role
        kwargs = {"base64_image": base64_image, **(kwargs if role == "tool" else {})}
        self.memory.add_message(message_map[role](content, **kwargs))

    async def run(self, request: Optional[str] = None) -> str:
        """Execute the agent's main loop asynchronously.

        Args:
            request: Optional initial user request to process.

        Returns:
            A string summarizing the execution results.

        Raises:
            RuntimeError: If the agent is not in IDLE state at start.
        """
        # 重置状态，确保每次 run() 都从干净的状态开始
        # 无论当前状态如何，都重置为初始状态
        logger.info(f"[AGENT] Resetting agent state before run: current_step={self.current_step}, state={self.state}")
        self.current_step = 0  # 总是重置步骤计数
        if self.state != AgentState.IDLE:
            logger.warning(f"Agent state is {self.state}, resetting to IDLE")
            self.state = AgentState.IDLE
        logger.info(f"[AGENT] Agent state after reset: current_step={self.current_step}, state={self.state}")

        if request:
            self.update_memory("user", request)

        results: List[str] = []
        last_step_result = ""  # 保存最后一步的结果，用于返回
        all_step_results = []  # 保存所有步骤的结果摘要

        async with self.state_context(AgentState.RUNNING):
            while (
                self.current_step < self.max_steps and self.state != AgentState.FINISHED
            ):
                self.current_step += 1
                logger.info(f"Executing step {self.current_step}/{self.max_steps}")
                step_result = await self.step()
                last_step_result = step_result  # 保存每一步的结果

                # 提取步骤结果的关键信息（前200字符）
                if step_result:
                    summary = step_result[:200] + "..." if len(step_result) > 200 else step_result
                    all_step_results.append(f"步骤 {self.current_step}: {summary}")

                # Check for stuck state
                if self.is_stuck():
                    self.handle_stuck_state()

            # 处理完成或达到最大步数的情况
            # 保存完成状态，因为后面会重置状态
            was_finished = (self.state == AgentState.FINISHED)
            steps_taken = self.current_step

            if was_finished:
                # Agent 明确标记任务已完成
                logger.info(f"[AGENT] Task completed successfully after {steps_taken} steps")

                # 优先从 memory 中提取最后的 assistant message（通常包含总结）
                # 如果最后一条 assistant message 包含总结性内容，使用它作为结果
                final_result = last_step_result if last_step_result else "任务执行完成。"

                # 尝试从 memory 中获取最后的思考内容
                if hasattr(self, 'memory') and self.memory and hasattr(self.memory, 'messages'):
                    # 从后往前查找 assistant message
                    for msg in reversed(self.memory.messages):
                        if hasattr(msg, 'role') and msg.role == 'assistant' and hasattr(msg, 'content'):
                            content = msg.content or ""
                            # 如果内容包含总结性关键词且长度足够，使用它
                            if len(content) > 100 and any(keyword in content for keyword in ["总结", "完成", "已经", "收集", "获取", "##", "###", "GitHub", "功能"]):
                                final_result = content
                                logger.info(f"[AGENT] Using final assistant message as result (length: {len(content)})")
                                break

                # 在结果前添加完成标记，方便后续识别
                result = f"[AGENT_COMPLETED:100%:steps={steps_taken}]\n{final_result}"
                self.state = AgentState.IDLE
            elif self.current_step >= self.max_steps:
                # 达到最大步数但未完成 - 需要人类决定是否继续
                logger.warning(f"[AGENT] Reached max_steps ({self.max_steps}) but task not finished. State: {self.state}")
                self.state = AgentState.IDLE
                if last_step_result:
                    summary_text = "\n".join(all_step_results[-3:]) if all_step_results else ""  # 只显示最后3步
                    result = f"[AGENT_INCOMPLETE:steps={self.max_steps}/{self.max_steps}]\n[⚠️ 达到最大步数 {self.max_steps}，任务可能未完全完成]:\n{last_step_result}\n\n[执行摘要]:\n{summary_text}\n\n[提示]：如果任务未完成，请与人类交互后决定是否继续执行。"
                else:
                    result = f"[AGENT_INCOMPLETE:steps={self.max_steps}/{self.max_steps}]\n[⚠️ 达到最大步数 {self.max_steps}，但未产生明确结果。请检查任务是否过于复杂，或增加最大步数。]"
            else:
                # 正常完成（理论上不应该到这里，因为如果完成应该是 FINISHED 状态）
                logger.warning(f"[AGENT] Unexpected state: current_step={self.current_step}, max_steps={self.max_steps}, state={self.state}")
                result = last_step_result if last_step_result else "任务执行完成，但未产生明确结果。"
                # 标记为可能完成，但不确定
                result = f"[AGENT_UNCERTAIN:steps={self.current_step}/{self.max_steps}]\n{result}"
                self.state = AgentState.IDLE

        await SANDBOX_CLIENT.cleanup()
        # 重置 current_step，为下次执行做准备
        self.current_step = 0
        return result

    @abstractmethod
    async def step(self) -> str:
        """Execute a single step in the agent's workflow.

        Must be implemented by subclasses to define specific behavior.
        """

    def handle_stuck_state(self):
        """Handle stuck state by adding a prompt to change strategy"""
        stuck_prompt = "\
        Observed duplicate responses. Consider new strategies and avoid repeating ineffective paths already attempted."
        self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"
        logger.warning(f"Agent detected stuck state. Added prompt: {stuck_prompt}")

    def is_stuck(self) -> bool:
        """Check if the agent is stuck in a loop by detecting duplicate content"""
        if len(self.memory.messages) < 2:
            return False

        last_message = self.memory.messages[-1]
        if not last_message.content:
            return False

        # Count identical content occurrences
        duplicate_count = sum(
            1
            for msg in reversed(self.memory.messages[:-1])
            if msg.role == "assistant" and msg.content == last_message.content
        )

        return duplicate_count >= self.duplicate_threshold

    @property
    def messages(self) -> List[Message]:
        """Retrieve a list of messages from the agent's memory."""
        return self.memory.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """Set the list of messages in the agent's memory."""
        self.memory.messages = value
