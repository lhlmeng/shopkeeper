# Shopkeeper Brain

基于 FastAPI、LangGraph、Milvus、MongoDB、MinIO 和 BGE 的企业级 RAG 知识库项目。

## 当前能力

- PDF/Markdown 文档导入。
- PDF 转 Markdown、图片提取和对象存储。
- 文档切分、商品名识别、混合向量化和 Milvus 入库。
- 向量检索、HyDE、网络搜索、RRF 融合和 Rerank 重排序。
- 基于 SSE 的流式问答。
- MongoDB 会话历史。
- 导入流程和查询流程双 LangGraph。
- MCP 网络搜索能力。

## 目录结构

```text
shopkeeper/
├── knowledge/
│   ├── api/                 # FastAPI 路由
│   ├── core/                # 配置、日志、异常和应用入口
│   ├── front/               # 静态前端页面
│   ├── processor/           # 导入和查询 LangGraph
│   ├── prompts/             # Prompt 模板
│   ├── schema/              # Pydantic 请求和响应模型
│   ├── service/             # 应用服务层
│   ├── test/                # 原有手工测试脚本
│   └── utils/               # 客户端、存储和通用工具
├── tests/                   # pytest 自动化测试
├── docker-compose.yml
├── pyproject.toml
└── main.py
```

## 环境要求

- Python 3.12
- Docker Desktop
- uv
- Git

安装 uv：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 本地启动

1. 创建环境变量文件。

```powershell
Copy-Item .env.example .env
```

如果项目中已经存在 `knowledge/.env`，应用会优先读取它；确认无关后，再把配置迁移到根目录 `.env` 并删除旧文件。

2. 按需修改 `.env` 中的模型、MinerU 和服务地址。

3. 安装依赖。

```powershell
uv sync --extra dev
```

如果继续使用现有 `knowledge/.venv`，可以先执行：

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$PWD\knowledge\.venv"
uv sync --extra dev
```

`knowledge/requirements.txt` 只是由 `uv.lock` 导出的兼容文件，不要手工修改。

4. 启动基础设施。

```powershell
docker compose up -d
```

5. 检查容器状态。

```powershell
docker compose ps
```

6. 启动应用。

```powershell
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8011
```

7. 访问服务。

- 健康检查：`http://localhost:8011/health`
- API 文档：`http://localhost:8011/docs`
- 导入页面：`http://localhost:8011/front/import.html`
- 问答页面：`http://localhost:8011/front/chat.html`
- MinIO 控制台：`http://localhost:9001`

## 开发命令

```powershell
uv run ruff check .
uv run ruff format .
uv run pytest
uv run pre-commit install
uv run pre-commit run --all-files
```

## 配置原则

- 新配置统一写在项目根目录 `.env`。
- 当前兼容读取 `knowledge/.env`，迁移期间优先使用旧文件中的已有值。
- 密钥、Token 和密码不能提交到 Git。
- 运行时数据、模型缓存和上传文件不提交到 Git。

## 服务端口

| 服务 | 端口 |
|---|---:|
| FastAPI | 8011 |
| Milvus | 19530 |
| Milvus metrics | 9091 |
| MongoDB | 27017 |
| Redis | 6379 |
| MinIO API | 9000 |
| MinIO Console | 9001 |
