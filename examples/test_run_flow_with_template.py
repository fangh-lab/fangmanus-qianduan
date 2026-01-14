"""
测试 run_flow 模式的文件上传和人类反馈功能

使用方法：
1. 准备一个规划模板文件（如 examples/planning_templates/customer_service_template.txt）
2. 运行此脚本进行测试
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.agent.data_analysis import DataAnalysis
from app.agent.manus import Manus
from app.config import config
from app.flow.flow_factory import FlowFactory, FlowType
from app.logger import logger
from app.tool.file_operators import LocalFileOperator


async def read_file(file_path: str) -> str:
    """Read content from a file."""
    try:
        operator = LocalFileOperator()
        content = await operator.read_file(file_path)
        logger.info(f"Successfully read file: {file_path}")
        return content
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        raise


def resolve_project_path(file_path: str) -> Path:
    """Resolve file path relative to project root directory."""
    path = Path(file_path)
    if path.is_absolute():
        return path
    project_path = config.root_path / file_path
    return project_path


def resolve_workspace_path(file_path: str) -> Path:
    """Resolve file path relative to workspace directory."""
    path = Path(file_path)
    if path.is_absolute():
        return path
    workspace_path = config.workspace_root / file_path
    return workspace_path


async def read_business_files(file_paths: list[str]) -> dict[str, str]:
    """Read multiple business files from workspace."""
    business_files = {}
    operator = LocalFileOperator()

    for file_path in file_paths:
        if not file_path.strip():
            continue
        try:
            resolved_path = resolve_workspace_path(file_path)
            if not resolved_path.exists():
                logger.warning(f"Business file not found: {file_path}")
                continue
            content = await operator.read_file(str(resolved_path))
            business_files[file_path] = content
            logger.info(f"Loaded business file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to read business file {file_path}: {e}")

    return business_files


async def test_run_flow_with_template():
    """测试使用规划模板的 run_flow"""
    agents = {
        "manus": Manus(),
    }
    if config.run_flow_config.use_data_analysis_agent:
        agents["data_analysis"] = DataAnalysis()

    try:
        # 使用示例模板文件
        template_dir = project_root / "examples" / "planning_templates"
        templates = {
            "1": ("customer_service_template.txt", "客户服务流程"),
            "2": ("data_analysis_template.txt", "数据分析流程"),
            "3": ("software_development_template.txt", "软件开发流程"),
        }

        print("\n" + "=" * 80)
        print("可用的规划模板：")
        print("=" * 80)
        for key, (filename, desc) in templates.items():
            print(f"{key}. {desc} - {filename}")
        print("0. 不使用模板（直接输入）")
        print("=" * 80)

        choice = input("\n请选择模板编号（0-3）: ").strip()

        planning_context = ""
        if choice in templates:
            template_file = template_dir / templates[choice][0]
            if template_file.exists():
                planning_context = await read_file(str(template_file))
                print(f"\n✓ 已加载模板: {templates[choice][1]}")
            else:
                print(f"\n✗ 模板文件不存在: {template_file}")
        elif choice == "0":
            print("\n✓ 将不使用模板")
        else:
            # 允许用户直接输入文件路径（相对于项目根目录）
            custom_path = input("\n请输入规划文件路径（相对于项目根目录，或按 Enter 跳过）: ").strip()
            if custom_path:
                resolved_path = resolve_project_path(custom_path)
                if resolved_path.exists():
                    planning_context = await read_file(str(resolved_path))
                    print(f"\n✓ 已加载自定义规划文件: {resolved_path}")
                else:
                    print(f"\n✗ 文件不存在: {custom_path} (解析路径: {resolved_path})")
                    print(f"项目根目录: {config.root_path}")

        # 业务文件选择
        print("\n" + "=" * 80)
        print("可用的业务文件（在 workspace 目录中）：")
        print("=" * 80)
        workspace_files = {
            "1": ("customer_complaint_12345.txt", "客户投诉工单 #12345"),
            "2": ("sales_data_january.csv", "一月份销售数据"),
            "3": ("todo_app_requirements.txt", "待办应用需求文档"),
        }
        for key, (filename, desc) in workspace_files.items():
            file_path = config.workspace_root / filename
            exists_mark = "✓" if file_path.exists() else "✗"
            print(f"{key}. {exists_mark} {desc} - {filename}")
        print("0. 不使用业务文件")
        print("4. 自定义输入文件路径")
        print("=" * 80)

        file_choice = input("\n请选择业务文件（可多选，用逗号分隔，如 1,2）: ").strip()

        business_file_paths = []
        if file_choice:
            choices = [c.strip() for c in file_choice.split(",")]
            for c in choices:
                if c in workspace_files:
                    business_file_paths.append(workspace_files[c][0])
                elif c == "4":
                    custom_file = input("请输入业务文件路径（相对于 workspace）: ").strip()
                    if custom_file:
                        business_file_paths.append(custom_file)

        business_files = {}
        if business_file_paths:
            business_files = await read_business_files(business_file_paths)
            if business_files:
                print(f"\n✓ 已加载 {len(business_files)} 个业务文件")
            else:
                print("\n✗ 未能加载任何业务文件")

        # 示例提示词
        example_prompts = {
            "1": "处理客户投诉工单 #12345，客户反映产品质量问题，需要尽快解决",
            "2": "分析最近一个月的销售数据，找出销售趋势和潜在问题",
            "3": "开发一个简单的待办事项管理应用，支持添加、删除和标记完成功能",
        }

        print("\n" + "=" * 80)
        print("示例提示词：")
        print("=" * 80)
        for key, prompt in example_prompts.items():
            print(f"{key}. {prompt}")
        print("0. 自定义输入")
        print("=" * 80)

        prompt_choice = input("\n请选择提示词（0-3）: ").strip()

        if prompt_choice in example_prompts:
            prompt = example_prompts[prompt_choice]
            print(f"\n使用提示词: {prompt}")
        else:
            prompt = input("\n请输入您的业务请求: ").strip()

        if not prompt:
            logger.warning("Empty prompt provided.")
            return

        flow = FlowFactory.create_flow(
            flow_type=FlowType.PLANNING,
            agents=agents,
        )
        logger.info("开始处理请求...")

        try:
            import time
            start_time = time.time()
            result = await asyncio.wait_for(
                flow.execute(
                    prompt,
                    planning_context=planning_context,
                    business_files=business_files
                ),
                timeout=3600,
            )
            elapsed_time = time.time() - start_time
            logger.info(f"请求处理完成，耗时 {elapsed_time:.2f} 秒")
            print("\n" + "=" * 80)
            print("最终结果：")
            print("=" * 80)
            print(result)
        except asyncio.TimeoutError:
            logger.error("请求处理超时（1小时）")
        except KeyboardInterrupt:
            logger.info("用户中断操作")
        except Exception as e:
            logger.error(f"处理出错: {str(e)}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        logger.error(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Run Flow 模式测试 - 文件上传和人类反馈功能")
    print("=" * 80)
    asyncio.run(test_run_flow_with_template())
