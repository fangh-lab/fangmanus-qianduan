# fangmanus-qianduan

## Web 可视化界面（FastAPI + HTML/CSS/JS）

本仓库已新增一个轻量 Web UI：支持**上传 planning 模板 / 业务文件**、输入 prompt，并在页面里实时展示**生成计划**与**每一步执行结果**，遇到需要人类确认/反馈时会在页面弹窗等待输入。

### 启动

1) 安装依赖（确保已创建 venv）

```bash
pip install -r requirements.txt
```

2) 启动 Web 服务

```bash
python run_web.py
```

3) 打开浏览器访问

`http://127.0.0.1:8000`

### 说明

- **后端执行引擎**：复用现有 `PlanningFlow`（即 `run_flow.py` 的 flow 模式逻辑），只是把命令行 `input()` 改成了可插拔的人类交互层，Web 模式通过事件+前端回传来完成交互。
- **文件处理**：UI 上传的文件会以“文件名 -> 内容字符串”的形式传入 flow（用于 planning context 与 business files 上下文）。
