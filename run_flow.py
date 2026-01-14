import asyncio
import time
from pathlib import Path

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
    """
    Resolve file path relative to project root directory.
    If path is absolute, return as is. If relative, resolve against project root.
    """
    path = Path(file_path)
    if path.is_absolute():
        return path
    # Relative path - resolve against project root
    project_path = config.root_path / file_path
    return project_path


def resolve_workspace_path(file_path: str) -> Path:
    """
    Resolve file path relative to workspace directory.
    If path is absolute, return as is. If relative, resolve against workspace root.
    """
    path = Path(file_path)
    if path.is_absolute():
        return path
    # Relative path - resolve against workspace root
    workspace_path = config.workspace_root / file_path
    return workspace_path


async def read_business_files(file_paths: list[str]) -> dict[str, str]:
    """
    Read multiple business files from workspace.

    Args:
        file_paths: List of file paths (relative to workspace or absolute)

    Returns:
        Dictionary mapping file paths to their contents
    """
    business_files = {}
    operator = LocalFileOperator()

    for file_path in file_paths:
        if not file_path.strip():
            continue

        try:
            # Resolve path relative to workspace
            resolved_path = resolve_workspace_path(file_path)

            if not resolved_path.exists():
                logger.warning(f"Business file not found: {file_path} (resolved: {resolved_path})")
                continue

            content = await operator.read_file(str(resolved_path))
            business_files[file_path] = content
            logger.info(f"Loaded business file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to read business file {file_path}: {e}")

    return business_files


async def run_flow():
    agents = {
        "manus": Manus(),
    }
    if config.run_flow_config.use_data_analysis_agent:
        agents["data_analysis"] = DataAnalysis()
    try:
        # Ask for planning template file (optional)
        planning_file_path = input(
            "Enter planning template file path (optional, press Enter to skip): "
        ).strip()

        planning_context = ""
        if planning_file_path:
            # Planning template paths are relative to project root, not workspace
            resolved_path = resolve_project_path(planning_file_path)
            if resolved_path.exists():
                planning_context = await read_file(str(resolved_path))
                logger.info(f"Planning template loaded from: {planning_file_path} (resolved: {resolved_path})")
            else:
                logger.warning(f"Planning template not found: {planning_file_path} (resolved: {resolved_path})")
                logger.info(f"Project root: {config.root_path}")
                logger.info(f"Please check if the file exists at the resolved path above.")

        # Ask for business files (optional, multiple files supported)
        print(f"\nWorkspace directory: {config.workspace_root}")
        print("Enter business file paths (relative to workspace, one per line, empty line to finish):")
        business_file_paths = []
        while True:
            file_path = input("  > ").strip()
            if not file_path:
                break
            business_file_paths.append(file_path)

        business_files = {}
        if business_file_paths:
            business_files = await read_business_files(business_file_paths)
            if business_files:
                logger.info(f"Loaded {len(business_files)} business file(s)")

        prompt = input("\nEnter your prompt: ")

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
            # Pass prompt, planning context, and business files to flow
            result = await asyncio.wait_for(
                flow.execute(
                    prompt,
                    planning_context=planning_context,
                    business_files=business_files
                ),
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
