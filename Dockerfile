# 第一阶段：在 Docker 环境中下载包
FROM python:3.10-slim as downloader

WORKDIR /download

# 配置国内镜像源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 复制依赖文件
COPY example/mcp_agent/requirements.txt /tmp/requirements_mcp.txt
COPY example/vector_databases/requirements.txt /tmp/requirements_vector.txt

# 下载所有包（在 Docker 环境中，平台自动正确）
RUN pip download \
    -r /tmp/requirements_mcp.txt \
    -r /tmp/requirements_vector.txt \
    sentence-transformers \
    langchain-huggingface \
    -d /download/packages

# 第二阶段：最终镜像
FROM python:3.10-slim

WORKDIR /app

# 配置国内镜像源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 从第一阶段复制下载的包
COPY --from=downloader /download/packages /tmp/packages

# 复制依赖文件
COPY example/mcp_agent/requirements.txt /tmp/requirements_mcp.txt
COPY example/vector_databases/requirements.txt /tmp/requirements_vector.txt

# 从本地安装
RUN pip install --no-cache-dir \
    --no-index \
    --find-links=/tmp/packages \
    -r /tmp/requirements_mcp.txt \
    -r /tmp/requirements_vector.txt \
    sentence-transformers \
    langchain-huggingface

# 清理
RUN rm -rf /tmp/packages

COPY . .

# CMD ["uvicorn", "example.mcp_agent.api_server:app", "--host", "0.0.0.0", "--port", "8000"]