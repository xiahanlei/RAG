# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个基于 Python 构建的 **RAG（检索增强生成）+ MCP Agent** 系统，包含两个主要子系统：

1. **向量数据库与 RAG 服务** — 通过 Milvus + DashScope（Qwen）实现文档入库、Embedding 向量化和语义检索
2. **MCP Agent 服务** — 基于 LangGraph 的 AI 智能体，连接 MCP（Model Context Protocol）服务器以实现工具调用能力

## 技术栈

| 组件 | 技术 |
|---|---|
| 大语言模型 | DashScope（Qwen / 通义千问），通过 `ChatTongyi` 和 OpenAI 兼容 API 调用 |
| 向量数据库 | Milvus v2.6.13 |
| Embedding | DashScope `text-embedding-v1`（备用：HuggingFace `all-MiniLM-L6-v2`） |
| 编排框架 | LangChain + LangGraph |
| MCP | `langchain-mcp-adapters`, `mcp`（FastMCP） |
| 后端（RAG） | Flask + Flask-CORS |
| 后端（Agent） | FastAPI + Uvicorn |
| 前端 | Vue 3 + Vite（`example/vector_databases/rag_front/`） |
| 基础设施 | Docker Compose（Milvus + etcd + vector-db-service + mcp-agent） |

## 目录结构

```
mult_agent/
├── Dockerfile                      # 多阶段构建：下载 pip 包用于离线安装
├── docker-compose.yml              # 4 个服务：milvus, etcd, vector-db-service, mcp-agent
├── example/
│   ├── vector_databases/           # RAG / 向量数据库子系统
│   │   ├── server.py               # Flask 应用入口（端口 5000）
│   │   ├── vector_db_manager.py    # Milvus 连接、文档加载/切分/入库
│   │   ├── vector_retriever.py     # 语义检索 + LLM 问答
│   │   ├── api_integration.py      # Flask Blueprint：/api/vector/* 路由
│   │   ├── document_loader.py      # 按文件类型加载文档（TXT/PDF/CSV/DOCX/XLSX）
│   │   ├── upload_document.py      # 命令行文档上传工具
│   │   ├── query_system.py         # 命令行查询接口
│   │   ├── rag_front/              # Vue 3 前端
│   │   └── test/                   # Milvus/Dashscope 测试脚本
│   ├── mcp_agent/                  # MCP Agent 子系统
│   │   ├── api_server.py           # FastAPI 应用入口（端口 8000），/chat 接口
│   │   ├── client.py               # 命令行聊天循环，支持 MCP 工具调用
│   │   ├── write_server.py         # 简易 MCP 服务器（文件写入工具）
│   │   ├── weather_server.py       # MCP 天气服务示例
│   │   └── agent_prompts.txt       # Agent 系统提示词
│   └── langgraph_example/          # LangGraph 学习示例与实验代码
├── packages/                       # 预下载的 Python wheel 包（用于 Docker 离线构建）
└── volumes/                        # Docker 持久化数据
```

## 核心架构

### RAG 流程（vector_databases）

```
文档 → Loader → Splitter → Embedding（DashScope） → Milvus
用户提问 → Embedding → Milvus 检索 → Top-K 文档 → LLM（Qwen） → 回答
```

- `VectorDatabaseManager`（`vector_db_manager.py`）：管理 Milvus 连接、文档加载（支持 TXT/PDF/CSV/DOCX/XLSX）、文本切分（`RecursiveCharacterTextSplitter`，chunk_size=500, overlap=50）和向量入库
- `VectorRetriever`（`vector_retriever.py`）：封装 `VectorDatabaseManager` 提供检索和问答功能。使用 OpenAI 兼容 API 调用 Qwen 生成回答
- `api_integration.py`：Flask Blueprint，在 `/api/vector/` 路径下暴露 REST 接口

### MCP Agent 流程（mcp_agent）

```
用户聊天 → FastAPI /chat → LangGraph ReAct Agent → MCP 工具 → LLM（Qwen） → 返回结果
```

- `api_server.py`：FastAPI 应用，提供 `/chat` POST 接口。通过 `MultiServerMCPClient` 连接 `servers_config.json` 中配置的 MCP 服务器，将工具注入 LangGraph `create_react_agent`
- `client.py`：同一 Agent 的命令行版本
- MCP 服务器配置在 `servers_config.json` 中（未提交到版本库，需本地创建）

### 模块路径设置

两个子系统都在运行时修改 `sys.path`，将项目根目录加入路径，以便 `example.*` 的导入能正常工作。每个子模块通过 `Path(__file__).parent / '.env'` 从自身目录加载 `.env` 环境变量。

## 常用命令

### 环境准备

```bash
# 本地安装依赖
pip install -r example/vector_databases/requirements.txt
pip install -r example/mcp_agent/requirements.txt

# 在 example/vector_databases/ 和 example/mcp_agent/ 下分别创建 .env 文件
# 必要环境变量：DASHSCOPE_API_KEY, MILVUS_HOST, MILVUS_PORT, MODEL
```

### Docker（全栈启动）

```bash
docker-compose up -d          # 启动所有服务（milvus, etcd, vector-db, mcp-agent）
docker-compose down           # 停止服务
docker-compose logs -f        # 查看日志
```

服务端口映射：
- Milvus: `localhost:19530`
- Vector DB（Flask）: `localhost:5001`（容器内 5000）
- MCP Agent（FastAPI）: `localhost:8000`

### 本地运行各服务

```bash
# RAG 后端（Flask）
python example/vector_databases/server.py
# → http://localhost:5000

# MCP Agent 后端（FastAPI）
uvicorn example.mcp_agent.api_server:app --host 0.0.0.0 --port 8000
# → http://localhost:8000

# MCP Agent 命令行模式
python example/mcp_agent/client.py

# 前端
cd example/vector_databases/rag_front
npm install && npm run dev
# → http://localhost:5173
```

### API 接口

**Vector DB（Flask，端口 5000）：**
- `POST /api/vector/upload_document` — 通过文件路径处理文档（JSON: `{file_path, collection_name}`）
- `POST /api/vector/upload_file` — 通过 multipart form 上传文件
- `POST /api/vector/query` — 问答（JSON: `{question, collection_name, k?}`）
- `POST /api/vector/search` — 相似度搜索（JSON: `{query, collection_name, k?}`）
- `GET /api/vector/collection_info?collection_name=xxx` — 集合信息
- `POST /api/vector/clear_collection` — 清空集合
- `GET /api/vector/health` — 健康检查

**MCP Agent（FastAPI，端口 8000）：**
- `POST /chat` — 与 Agent 对话（JSON: `{message, thread_id?}`）

## 配置说明

- `.env` 文件需放在各子模块目录下（`example/vector_databases/.env`、`example/mcp_agent/.env`）
- MCP 服务器配置：`example/mcp_agent/servers_config.json`（格式：`{"mcpServers": {...}}`）
- Agent 提示词：`example/mcp_agent/agent_prompts.txt`
- Milvus 数据持久化路径：`volumes/milvus/`

## 注意事项

- Docker 使用 Python 3.10 基础镜像，但本地 wheel 包是为 cp313 编译的 — 在 Docker 外开发时注意 Python 版本匹配
- Milvus 连接带重试逻辑（5 次，间隔 5 秒）— 即使 Milvus 不可用服务也能启动（降级模式）
- Schema 不兼容错误会触发 `VectorDatabaseManager.add_documents_to_db()` 中的自动集合重建
- `packages/` 目录包含预下载的 wheel 包，用于 Docker 离线构建，不要手动修改
