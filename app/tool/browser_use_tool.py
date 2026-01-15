import asyncio
import base64
import json
from typing import Generic, Optional, TypeVar

from browser_use import Browser as BrowserUseBrowser
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.dom.service import DomService
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from app.config import config
from app.llm import LLM
from app.logger import logger
from app.tool.base import BaseTool, ToolResult
from app.tool.web_search import WebSearch


_BROWSER_DESCRIPTION = """\
A powerful browser automation tool that allows interaction with web pages through various actions.
* This tool provides commands for controlling a browser session, navigating web pages, and extracting information
* It maintains state across calls, keeping the browser session alive until explicitly closed
* Use this when you need to browse websites, fill forms, click buttons, extract content, or perform web searches
* Each action requires specific parameters as defined in the tool's dependencies

Key capabilities include:
* Navigation: Go to specific URLs, go back, search the web, or refresh pages
* Interaction: Click elements, input text, select from dropdowns, send keyboard commands
* Scrolling: Scroll up/down by pixel amount or scroll to specific text
* Content extraction: Extract and analyze content from web pages based on specific goals
* Tab management: Switch between tabs, open new tabs, or close tabs

Note: When using element indices, refer to the numbered elements shown in the current browser state.
"""

Context = TypeVar("Context")


class BrowserUseTool(BaseTool, Generic[Context]):
    name: str = "browser_use"
    description: str = _BROWSER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "go_to_url",
                    "click_element",
                    "input_text",
                    "scroll_down",
                    "scroll_up",
                    "scroll_to_text",
                    "send_keys",
                    "get_dropdown_options",
                    "select_dropdown_option",
                    "go_back",
                    "web_search",
                    "wait",
                    "extract_content",
                    "switch_tab",
                    "open_tab",
                    "close_tab",
                ],
                "description": "The browser action to perform",
            },
            "url": {
                "type": "string",
                "description": "URL for 'go_to_url' or 'open_tab' actions",
            },
            "index": {
                "type": "integer",
                "description": "Element index for 'click_element', 'input_text', 'get_dropdown_options', or 'select_dropdown_option' actions",
            },
            "text": {
                "type": "string",
                "description": "Text for 'input_text', 'scroll_to_text', or 'select_dropdown_option' actions",
            },
            "scroll_amount": {
                "type": "integer",
                "description": "Pixels to scroll (positive for down, negative for up) for 'scroll_down' or 'scroll_up' actions",
            },
            "tab_id": {
                "type": "integer",
                "description": "Tab ID for 'switch_tab' action",
            },
            "query": {
                "type": "string",
                "description": "Search query for 'web_search' action",
            },
            "goal": {
                "type": "string",
                "description": "Extraction goal for 'extract_content' action",
            },
            "keys": {
                "type": "string",
                "description": "Keys to send for 'send_keys' action",
            },
            "seconds": {
                "type": "integer",
                "description": "Seconds to wait for 'wait' action",
            },
        },
        "required": ["action"],
        "dependencies": {
            "go_to_url": ["url"],
            "click_element": ["index"],
            "input_text": ["index", "text"],
            "switch_tab": ["tab_id"],
            "open_tab": ["url"],
            "scroll_down": ["scroll_amount"],
            "scroll_up": ["scroll_amount"],
            "scroll_to_text": ["text"],
            "send_keys": ["keys"],
            "get_dropdown_options": ["index"],
            "select_dropdown_option": ["index", "text"],
            "go_back": [],
            "web_search": ["query"],
            "wait": ["seconds"],
            "extract_content": ["goal"],
        },
    }

    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    browser: Optional[BrowserUseBrowser] = Field(default=None, exclude=True)
    context: Optional[BrowserContext] = Field(default=None, exclude=True)
    dom_service: Optional[DomService] = Field(default=None, exclude=True)
    web_search_tool: WebSearch = Field(default_factory=WebSearch, exclude=True)

    # Context for generic functionality
    tool_context: Optional[Context] = Field(default=None, exclude=True)

    llm: Optional[LLM] = Field(default_factory=LLM)

    @field_validator("parameters", mode="before")
    def validate_parameters(cls, v: dict, info: ValidationInfo) -> dict:
        if not v:
            raise ValueError("Parameters cannot be empty")
        return v

    async def _ensure_browser_initialized(self) -> BrowserContext:
        """Ensure browser and context are initialized."""
        if self.browser is None:
            browser_config_kwargs = {"headless": False, "disable_security": True}

            if config.browser_config:
                from browser_use.browser.browser import ProxySettings

                # handle proxy settings.
                if config.browser_config.proxy and config.browser_config.proxy.server:
                    browser_config_kwargs["proxy"] = ProxySettings(
                        server=config.browser_config.proxy.server,
                        username=config.browser_config.proxy.username,
                        password=config.browser_config.proxy.password,
                    )

                browser_attrs = [
                    "headless",
                    "disable_security",
                    "extra_chromium_args",
                    "chrome_instance_path",
                    "wss_url",
                    "cdp_url",
                ]

                for attr in browser_attrs:
                    value = getattr(config.browser_config, attr, None)
                    if value is not None:
                        if not isinstance(value, list) or value:
                            browser_config_kwargs[attr] = value

            # 如果只配置了 chrome_instance_path 但没有配置 cdp_url 或 wss_url
            # 说明用户想使用指定的浏览器路径启动新实例，而不是连接现有实例
            # browser_use 库可能会尝试连接现有实例，所以我们需要确保它启动新实例
            # 方法：添加额外的参数来强制启动新实例，或者使用 executable_path 而不是 chrome_instance_path
            chrome_path = browser_config_kwargs.get("chrome_instance_path")
            has_cdp = "cdp_url" in browser_config_kwargs or "wss_url" in browser_config_kwargs

            if chrome_path and not has_cdp:
                # 只配置了 chrome_instance_path，没有配置连接 URL
                # 这意味着要启动新实例，而不是连接现有实例
                # 使用 executable_path 参数（如果 browser_use 支持）或者移除 chrome_instance_path 让 Playwright 自动启动
                logger.info(f"Browser instance path configured: {chrome_path}")
                logger.info("Will start a new browser instance (not connecting to existing one)")
                # 确保不设置 cdp_url/wss_url，让工具启动新实例
                # browser_use 库应该会自动使用 chrome_instance_path 启动新实例
                # 如果仍然尝试连接，可能需要使用不同的配置方式
            elif chrome_path and has_cdp:
                # 同时配置了 chrome_instance_path 和 cdp_url/wss_url
                # 这意味着要连接到现有实例
                logger.info("Connecting to existing browser instance via CDP/WSS")
            elif "cdp_url" not in browser_config_kwargs and "wss_url" not in browser_config_kwargs:
                # 没有配置任何浏览器相关参数，让 browser_use 自动启动浏览器
                logger.info("No browser config specified, will auto-start browser")

            try:
                self.browser = BrowserUseBrowser(BrowserConfig(**browser_config_kwargs))
            except Exception as e:
                error_msg = str(e)
                # 如果是连接错误，且只配置了 chrome_instance_path（没有 cdp_url/wss_url）
                # 尝试移除 chrome_instance_path，让 Playwright 自动启动浏览器
                if ("ECONNREFUSED" in error_msg or "connect" in error_msg.lower() or "Debug mode" in error_msg) and \
                   "chrome_instance_path" in browser_config_kwargs and \
                   "cdp_url" not in browser_config_kwargs and "wss_url" not in browser_config_kwargs:
                    logger.warning(f"Browser connection failed with chrome_instance_path, trying without it: {error_msg}")
                    # 移除 chrome_instance_path，让 Playwright 自动启动浏览器
                    fallback_config = browser_config_kwargs.copy()
                    fallback_config.pop("chrome_instance_path", None)
                    try:
                        logger.info("Retrying browser initialization without chrome_instance_path (auto-start)")
                        self.browser = BrowserUseBrowser(BrowserConfig(**fallback_config))
                        logger.info("Successfully started browser without chrome_instance_path")
                    except Exception as e2:
                        # 如果还是失败，提供友好的错误信息
                        raise RuntimeError(
                            f"浏览器启动失败: {str(e2)}\n"
                            f"原始错误: {error_msg}\n"
                            f"解决方案:\n"
                            f"1. 关闭所有正在运行的 Chrome/Edge 浏览器实例，然后重试\n"
                            f"2. 或者使用 web_search 工具作为替代方案（不需要浏览器）\n"
                            f"3. 如果必须使用浏览器，请确保 Playwright 已正确安装"
                        ) from e2
                else:
                    # 其他错误，提供友好的错误信息
                    if "ECONNREFUSED" in error_msg or "connect" in error_msg.lower() or "Debug mode" in error_msg:
                        raise RuntimeError(
                            f"浏览器连接失败: {error_msg}\n"
                            f"解决方案:\n"
                            f"1. 关闭所有正在运行的 Chrome/Edge 浏览器实例\n"
                            f"2. 或者让工具自动启动浏览器（不配置 cdp_url/wss_url）\n"
                            f"3. 如果必须使用现有浏览器，请以调试模式启动: msedge.exe --remote-debugging-port=9222"
                        ) from e
                    raise

        if self.context is None:
            context_config = BrowserContextConfig()

            # if there is context config in the config, use it.
            if (
                config.browser_config
                and hasattr(config.browser_config, "new_context_config")
                and config.browser_config.new_context_config
            ):
                context_config = config.browser_config.new_context_config

            self.context = await self.browser.new_context(context_config)
            self.dom_service = DomService(await self.context.get_current_page())

        return self.context

    async def execute(
        self,
        action: str,
        url: Optional[str] = None,
        index: Optional[int] = None,
        text: Optional[str] = None,
        scroll_amount: Optional[int] = None,
        tab_id: Optional[int] = None,
        query: Optional[str] = None,
        goal: Optional[str] = None,
        keys: Optional[str] = None,
        seconds: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """
        Execute a specified browser action.

        Args:
            action: The browser action to perform
            url: URL for navigation or new tab
            index: Element index for click or input actions
            text: Text for input action or search query
            scroll_amount: Pixels to scroll for scroll action
            tab_id: Tab ID for switch_tab action
            query: Search query for Google search
            goal: Extraction goal for content extraction
            keys: Keys to send for keyboard actions
            seconds: Seconds to wait
            **kwargs: Additional arguments

        Returns:
            ToolResult with the action's output or error
        """
        async with self.lock:
            try:
                context = await self._ensure_browser_initialized()

                # Get max content length from config
                max_content_length = getattr(
                    config.browser_config, "max_content_length", 2000
                )

                # Navigation actions
                if action == "go_to_url":
                    if not url:
                        return ToolResult(
                            error="URL is required for 'go_to_url' action"
                        )
                    try:
                        page = await context.get_current_page()
                        await asyncio.wait_for(page.goto(url), timeout=30.0)
                        await asyncio.wait_for(page.wait_for_load_state(), timeout=10.0)
                        return ToolResult(output=f"Navigated to {url}")
                    except asyncio.TimeoutError:
                        return ToolResult(error=f"Navigation to {url} timed out after 30 seconds")
                    except Exception as e:
                        return ToolResult(error=f"Navigation error: {str(e)}")

                elif action == "go_back":
                    await context.go_back()
                    return ToolResult(output="Navigated back")

                elif action == "refresh":
                    await context.refresh_page()
                    return ToolResult(output="Refreshed current page")

                elif action == "web_search":
                    if not query:
                        return ToolResult(
                            error="Query is required for 'web_search' action"
                        )
                    # Execute the web search and return results directly without browser navigation
                    search_response = await self.web_search_tool.execute(
                        query=query, fetch_content=True, num_results=1
                    )
                    if not search_response.results:
                        return ToolResult(error="No search results found")

                    # Navigate to the first search result
                    first_search_result = search_response.results[0]
                    url_to_navigate = first_search_result.url

                    try:
                        page = await context.get_current_page()
                        await asyncio.wait_for(page.goto(url_to_navigate), timeout=30.0)
                        await asyncio.wait_for(page.wait_for_load_state(), timeout=10.0)
                    except asyncio.TimeoutError:
                        return ToolResult(error=f"Navigation to search result timed out, but search results: {search_response.output}")
                    except Exception as e:
                        return ToolResult(error=f"Navigation error: {str(e)}, but search results: {search_response.output}")

                    return search_response

                # Element interaction actions
                elif action == "click_element":
                    if index is None:
                        return ToolResult(
                            error="Index is required for 'click_element' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    download_path = await context._click_element_node(element)
                    output = f"Clicked element at index {index}"
                    if download_path:
                        output += f" - Downloaded file to {download_path}"
                    return ToolResult(output=output)

                elif action == "input_text":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'input_text' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    await context._input_text_element_node(element, text)
                    return ToolResult(
                        output=f"Input '{text}' into element at index {index}"
                    )

                elif action == "scroll_down" or action == "scroll_up":
                    direction = 1 if action == "scroll_down" else -1
                    amount = (
                        scroll_amount
                        if scroll_amount is not None
                        else context.config.browser_window_size["height"]
                    )
                    await context.execute_javascript(
                        f"window.scrollBy(0, {direction * amount});"
                    )
                    return ToolResult(
                        output=f"Scrolled {'down' if direction > 0 else 'up'} by {amount} pixels"
                    )

                elif action == "scroll_to_text":
                    if not text:
                        return ToolResult(
                            error="Text is required for 'scroll_to_text' action"
                        )
                    page = await context.get_current_page()
                    try:
                        locator = page.get_by_text(text, exact=False)
                        await locator.scroll_into_view_if_needed()
                        return ToolResult(output=f"Scrolled to text: '{text}'")
                    except Exception as e:
                        return ToolResult(error=f"Failed to scroll to text: {str(e)}")

                elif action == "send_keys":
                    if not keys:
                        return ToolResult(
                            error="Keys are required for 'send_keys' action"
                        )
                    page = await context.get_current_page()
                    await page.keyboard.press(keys)
                    return ToolResult(output=f"Sent keys: {keys}")

                elif action == "get_dropdown_options":
                    if index is None:
                        return ToolResult(
                            error="Index is required for 'get_dropdown_options' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    page = await context.get_current_page()
                    options = await page.evaluate(
                        """
                        (xpath) => {
                            const select = document.evaluate(xpath, document, null,
                                XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                            if (!select) return null;
                            return Array.from(select.options).map(opt => ({
                                text: opt.text,
                                value: opt.value,
                                index: opt.index
                            }));
                        }
                    """,
                        element.xpath,
                    )
                    return ToolResult(output=f"Dropdown options: {options}")

                elif action == "select_dropdown_option":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'select_dropdown_option' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    page = await context.get_current_page()
                    await page.select_option(element.xpath, label=text)
                    return ToolResult(
                        output=f"Selected option '{text}' from dropdown at index {index}"
                    )

                # Content extraction actions
                elif action == "extract_content":
                    if not goal:
                        return ToolResult(
                            error="Goal is required for 'extract_content' action"
                        )

                    page = await context.get_current_page()
                    import markdownify

                    content = markdownify.markdownify(await page.content())

                    prompt = f"""\
Your task is to extract the content of the page. You will be given a page and a goal, and you should extract all relevant information around this goal from the page. If the goal is vague, summarize the page. Respond in json format.
Extraction goal: {goal}

Page content:
{content[:max_content_length]}
"""
                    messages = [{"role": "system", "content": prompt}]

                    # Define extraction function schema
                    extraction_function = {
                        "type": "function",
                        "function": {
                            "name": "extract_content",
                            "description": "Extract specific information from a webpage based on a goal",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "extracted_content": {
                                        "type": "object",
                                        "description": "The content extracted from the page according to the goal",
                                        "properties": {
                                            "text": {
                                                "type": "string",
                                                "description": "Text content extracted from the page",
                                            },
                                            "metadata": {
                                                "type": "object",
                                                "description": "Additional metadata about the extracted content",
                                                "properties": {
                                                    "source": {
                                                        "type": "string",
                                                        "description": "Source of the extracted content",
                                                    }
                                                },
                                            },
                                        },
                                    }
                                },
                                "required": ["extracted_content"],
                            },
                        },
                    }

                    # Use LLM to extract content with required function calling
                    response = await self.llm.ask_tool(
                        messages,
                        tools=[extraction_function],
                        tool_choice="required",
                    )

                    if response and response.tool_calls:
                        args = json.loads(response.tool_calls[0].function.arguments)
                        extracted_content = args.get("extracted_content", {})
                        return ToolResult(
                            output=f"Extracted from page:\n{extracted_content}\n"
                        )

                    return ToolResult(output="No content was extracted from the page.")

                # Tab management actions
                elif action == "switch_tab":
                    if tab_id is None:
                        return ToolResult(
                            error="Tab ID is required for 'switch_tab' action"
                        )
                    await context.switch_to_tab(tab_id)
                    page = await context.get_current_page()
                    await page.wait_for_load_state()
                    return ToolResult(output=f"Switched to tab {tab_id}")

                elif action == "open_tab":
                    if not url:
                        return ToolResult(error="URL is required for 'open_tab' action")
                    await context.create_new_tab(url)
                    return ToolResult(output=f"Opened new tab with {url}")

                elif action == "close_tab":
                    await context.close_current_tab()
                    return ToolResult(output="Closed current tab")

                # Utility actions
                elif action == "wait":
                    seconds_to_wait = seconds if seconds is not None else 3
                    await asyncio.sleep(seconds_to_wait)
                    return ToolResult(output=f"Waited for {seconds_to_wait} seconds")

                else:
                    return ToolResult(error=f"Unknown action: {action}")

            except Exception as e:
                error_msg = str(e)
                # 如果是浏览器连接错误，提供更友好的错误信息和建议
                if "ECONNREFUSED" in error_msg or "connect" in error_msg.lower() or "Debug mode" in error_msg or "9222" in error_msg:
                    friendly_error = (
                        f"浏览器连接失败: {error_msg}\n\n"
                        f"建议:\n"
                        f"1. 关闭所有正在运行的 Chrome/Edge 浏览器实例，然后重试\n"
                        f"2. 或者使用 web_search 工具作为替代方案（不需要浏览器）\n"
                        f"3. 如果必须使用浏览器，请确保浏览器以调试模式启动"
                    )
                    return ToolResult(error=friendly_error)
                return ToolResult(error=f"Browser action '{action}' failed: {str(e)}")

    async def get_current_state(
        self, context: Optional[BrowserContext] = None
    ) -> ToolResult:
        """
        Get the current browser state as a ToolResult.
        If context is not provided, uses self.context.
        """
        try:
            # Use provided context or fall back to self.context
            ctx = context or self.context
            if not ctx:
                return ToolResult(error="Browser context not initialized")

            state = await ctx.get_state()

            # Create a viewport_info dictionary if it doesn't exist
            viewport_height = 0
            if hasattr(state, "viewport_info") and state.viewport_info:
                viewport_height = state.viewport_info.height
            elif hasattr(ctx, "config") and hasattr(ctx.config, "browser_window_size"):
                viewport_height = ctx.config.browser_window_size.get("height", 0)

            # Take a screenshot for the state
            page = await ctx.get_current_page()

            await page.bring_to_front()
            await page.wait_for_load_state()

            screenshot = await page.screenshot(
                full_page=True, animations="disabled", type="jpeg", quality=100
            )

            screenshot = base64.b64encode(screenshot).decode("utf-8")

            # Build the state info with all required fields
            state_info = {
                "url": state.url,
                "title": state.title,
                "tabs": [tab.model_dump() for tab in state.tabs],
                "help": "[0], [1], [2], etc., represent clickable indices corresponding to the elements listed. Clicking on these indices will navigate to or interact with the respective content behind them.",
                "interactive_elements": (
                    state.element_tree.clickable_elements_to_string()
                    if state.element_tree
                    else ""
                ),
                "scroll_info": {
                    "pixels_above": getattr(state, "pixels_above", 0),
                    "pixels_below": getattr(state, "pixels_below", 0),
                    "total_height": getattr(state, "pixels_above", 0)
                    + getattr(state, "pixels_below", 0)
                    + viewport_height,
                },
                "viewport_height": viewport_height,
            }

            return ToolResult(
                output=json.dumps(state_info, indent=4, ensure_ascii=False),
                base64_image=screenshot,
            )
        except Exception as e:
            return ToolResult(error=f"Failed to get browser state: {str(e)}")

    async def cleanup(self):
        """Clean up browser resources."""
        async with self.lock:
            if self.context is not None:
                await self.context.close()
                self.context = None
                self.dom_service = None
            if self.browser is not None:
                await self.browser.close()
                self.browser = None

    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        if self.browser is not None or self.context is not None:
            try:
                asyncio.run(self.cleanup())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.cleanup())
                loop.close()

    @classmethod
    def create_with_context(cls, context: Context) -> "BrowserUseTool[Context]":
        """Factory method to create a BrowserUseTool with a specific context."""
        tool = cls()
        tool.tool_context = context
        return tool
