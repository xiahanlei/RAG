"""
向量数据库管理模块
基于 pymilvus.MilvusClient 和 Milvus 实现文档切分、向量化存储和检索功能
"""

import time
import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Load env vars from explicit path
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# LangChain and Milvus imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings, DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader
)
from pymilvus import MilvusClient, DataType
from pymilvus.milvus_client import IndexParams

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default dimension for embedding vectors (will be detected on first use)
DEFAULT_DIM = 768


class VectorDatabaseManager:
    """向量数据库管理器 (Milvus后端)"""

    def __init__(self,
                 milvus_host: str = None,
                 milvus_port: int = None,
                 collection_name: str = None,
                 embedding_model: str = None,
                 dashscope_api_key: str = None,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        self.milvus_host = milvus_host or os.getenv("MILVUS_HOST", "127.0.0.1")
        self.milvus_port = str(milvus_port or os.getenv("MILVUS_PORT", "19530"))
        self.milvus_uri = f"http://{self.milvus_host}:{self.milvus_port}"
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "agent_rag")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-v1")
        self.dashscope_api_key = dashscope_api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 初始化嵌入模型
        self._init_embeddings()

        # 初始化文档切分器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )

        # 标记 Milvus 是否可用
        self.milvus_available = False
        self._client: Optional[MilvusClient] = None

        # 连接到Milvus
        self._connect_to_milvus_with_retry()

    def _init_embeddings(self):
        """初始化嵌入模型"""
        try:
            if not self.dashscope_api_key:
                logger.warning("未提供 DashScope API Key，将尝试从环境变量获取")
                self.dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY", "")

            self.embeddings = DashScopeEmbeddings(
                model=self.embedding_model,
                dashscope_api_key=self.dashscope_api_key
            )
            try:
                self.embeddings.embed_query("test")
                logger.info(f"成功加载并验证 DashScope嵌入模型: {self.embedding_model}")
            except Exception as e:
                logger.error(f"DashScope 模型验证失败: {e}")
                raise e

        except Exception as e:
            logger.error(f"加载DashScope模型失败: {e}")
            logger.warning("使用备用HuggingFace模型")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )

    def _connect_to_milvus_with_retry(self, max_retries=5, delay=5):
        """带重试的Milvus连接"""
        for attempt in range(max_retries):
            try:
                logger.info(f"尝试连接Milvus (第 {attempt + 1}/{max_retries} 次): {self.milvus_uri}")
                self._client = MilvusClient(uri=self.milvus_uri)
                # 验证连接是否有效
                self._client.list_collections()
                logger.info(f"✅ 成功连接到Milvus: {self.milvus_uri}")
                self.milvus_available = True
                return
            except Exception as e:
                logger.warning(f"连接Milvus失败 (第 {attempt + 1}/{max_retries} 次): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"等待 {delay} 秒后重试...")
                    time.sleep(delay)
                else:
                    logger.error(f"❌ 无法连接到Milvus，已达到最大重试次数")
                    logger.warning("服务将以降级模式运行（向量搜索功能不可用）")
                    self.milvus_available = False

    def _ensure_collection(self, collection_name: str, dim: int) -> str:
        """确保集合存在，返回集合名"""
        if collection_name in self._client.list_collections():
            return collection_name

        schema = self._client.create_schema(
            auto_id=True,
            primary_field_name="id",
            primary_field_type=DataType.INT64,
            description=f"Collection: {collection_name}",
            enable_dynamic_field=True,
        )
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)

        self._client.create_collection(
            collection_name=collection_name,
            dimension=dim,
            metric_type="IP",
            schema=schema,
        )
        logger.info(f"成功创建集合: {collection_name}")
        return collection_name

    def _create_index(self, collection_name: str):
        """创建向量索引"""
        try:
            index_params = IndexParams()
            index_params.add_index(
                field_name="vector",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128},
            )
            self._client.create_index(
                collection_name=collection_name,
                index_params=index_params,
            )
            self._client.load_collection(collection_name=collection_name)
        except Exception as e:
            if "already exist" not in str(e).lower() and "duplicated index" not in str(e).lower():
                raise

    def load_document(self, file_path: str) -> List[Document]:
        """根据文件类型加载文档"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_extension = Path(file_path).suffix.lower()

        try:
            if file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension == '.csv':
                loader = CSVLoader(file_path, encoding='utf-8')
            elif file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                loader = UnstructuredExcelLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding='utf-8')
                logger.warning(f"未识别的文件类型 {file_extension}, 使用文本加载器")

            documents = loader.load()
            logger.info(f"成功加载文档: {file_path}, 共 {len(documents)} 个文档块")
            return documents

        except Exception as e:
            logger.error(f"加载文档失败 {file_path}: {e}")
            return []

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """切分文档"""
        try:
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"文档切分完成: {len(documents)} -> {len(split_docs)} 个块")
            return split_docs
        except Exception as e:
            logger.error(f"文档切分失败: {e}")
            return documents

    def add_documents_to_db(self, documents: List[Document], collection_name: str = None):
        """将文档添加到Milvus数据库"""
        if not self.milvus_available or not self._client:
            logger.error("Milvus 不可用，无法添加文档")
            raise Exception("Milvus service is not available")

        if not documents:
            logger.warning("没有文档需要添加")
            return

        target_collection = collection_name or self.collection_name
        logger.info(f"目标集合: {target_collection}")

        try:
            # 生成向量和元数据
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            vectors = self.embeddings.embed_documents(texts)
            dim = len(vectors[0])

            # 确保集合存在
            self._ensure_collection(target_collection, dim)

            # 分批插入
            batch_size = 100
            total_inserted = 0
            for i in range(0, len(texts), batch_size):
                batch_end = min(i + batch_size, len(texts))
                data_batch = []
                for j in range(i, batch_end):
                    data_batch.append({
                        "text": texts[j],
                        "vector": vectors[j],
                        **metadatas[j],
                    })
                self._client.insert(collection_name=target_collection, data=data_batch)
                total_inserted += (batch_end - i)

            # 创建索引
            self._create_index(target_collection)

            logger.info(f"成功向集合 '{target_collection}' 插入 {total_inserted} 条文档")

        except Exception as e:
            logger.error(f"添加文档到Milvus失败: {e}")
            raise e

    def process_file(self, file_path: str, collection_name: str = None) -> bool:
        """处理单个文件：加载、切分、存储"""
        try:
            logger.info(f"开始处理文件: {file_path}")
            documents = self.load_document(file_path)
            if not documents:
                return False

            split_docs = self.split_documents(documents)
            self.add_documents_to_db(split_docs, collection_name)

            logger.info(f"文件处理完成: {file_path}")
            return True

        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {e}")
            return False

    def process_csv_data(self, csv_path: str,
                         text_columns: List[str] = None,
                         metadata_columns: List[str] = None) -> bool:
        """处理CSV数据文件"""
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, encoding='utf-8')
            logger.info(f"读取CSV文件: {csv_path}, 共 {len(df)} 行数据")
            if text_columns is None or not text_columns:
                object_cols = [c for c in df.columns if df[c].dtype == 'object']
                text_columns = [c for c in object_cols if not str(c).startswith('Unnamed')]

            if metadata_columns is None:
                metadata_columns = [c for c in df.columns if c not in text_columns]

            documents = []
            for idx, row in df.iterrows():
                content_parts = []
                metadata = {"source": csv_path, "row_index": idx}

                for col in text_columns:
                    if pd.notna(row[col]):
                        text = str(row[col]).strip()
                        if text:
                            content_parts.append(f"{col}: {text}")

                for col in metadata_columns:
                    val = row.get(col)
                    if pd.notna(val):
                        metadata[str(col)] = str(val)

                if content_parts:
                    content = "\n".join(content_parts)
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)

            logger.info(f"构建了 {len(documents)} 个文档")
            split_docs = self.split_documents(documents)
            self.add_documents_to_db(split_docs)

            return True

        except Exception as e:
            logger.error(f"处理CSV数据失败: {e}")
            return False

    def get_embedding(self, texts: List[str]) -> List[List[float]]:
        """使用嵌入模型为一组文本生成嵌入向量"""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {e}")
            return []

    def search(self, query: str, k: int = 5, filter_dict: Dict = None, collection_name: Optional[str] = None) -> List[Tuple[Document, float]]:
        """相似性搜索"""
        if not self.milvus_available or not self._client:
            logger.warning("Milvus 不可用，返回空结果")
            return []

        target_collection = collection_name or self.collection_name

        if target_collection not in self._client.list_collections():
            logger.warning(f"集合 {target_collection} 不存在")
            return []

        try:
            # 确保集合已加载
            self._client.load_collection(collection_name=target_collection)

            query_vector = self.embeddings.embed_query(query)
            results = self._client.search(
                collection_name=target_collection,
                data=[query_vector],
                limit=k,
                output_fields=["text", "source", "row_index", "page", "filename", "sheet_name"],
            )

            docs_with_scores = []
            for hit in results[0]:
                page_content = hit.get("text", "")
                metadata = {key: hit[key] for key in hit if key not in ("id", "text", "distance", "entity")}
                score = hit.get("distance", 0)
                doc = Document(page_content=page_content, metadata=metadata)
                docs_with_scores.append((doc, score))

            logger.info(f"搜索查询: '{query}', 返回 {len(docs_with_scores)} 个结果")
            return docs_with_scores

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

    def get_database_info(self, collection_name: str = None) -> Dict[str, Any]:
        """获取数据库信息"""
        target_collection = collection_name or self.collection_name

        info = {
            "milvus_host": self.milvus_host,
            "milvus_port": self.milvus_port,
            "collection_name": target_collection,
            "is_initialized": self._client is not None,
            "milvus_available": self.milvus_available
        }

        if not self.milvus_available or not self._client:
            info["error"] = "Milvus service is not available"
            return info

        try:
            if target_collection in self._client.list_collections():
                stats = self._client.get_collection_stats(target_collection)
                info["document_count"] = stats.get("row_count", 0)
            else:
                info["document_count"] = 0
        except Exception as e:
            logger.error(f"获取Milvus集合信息失败: {e}")
            info["error"] = str(e)

        return info

    def clear_database(self, collection_name: str = None):
        """清空Milvus集合"""
        if not self._client:
            return
        target = collection_name or self.collection_name
        try:
            if target in self._client.list_collections():
                self._client.drop_collection(target)
                logger.info(f"Milvus集合 '{target}' 已被删除")
        except Exception as e:
            logger.error(f"清空Milvus集合失败: {e}")


def main():
    """测试函数"""
    try:
        client = MilvusClient(uri="http://127.0.0.1:19530")
        print("Collections:", client.list_collections())
        client.close()
    except Exception as e:
        logger.error("无法连接到Milvus服务，请确保您已通过 docker-compose up -d 启动了Milvus。")
        logger.error(f"错误: {e}")
        return

    db_manager = VectorDatabaseManager()

    print("清空现有数据库...")
    db_manager.clear_database()

    csv_path = "../data.csv"
    if os.path.exists(csv_path):
        print("处理CSV数据...")
        success = db_manager.process_csv_data(csv_path)
        if success:
            print("CSV数据处理成功！")
            print("\n测试搜索功能:")
            results = db_manager.search("示例查询", k=3)
            for i, (doc, score) in enumerate(results):
                print(f"\n结果 {i+1} (相似度: {score:.4f}):")
                print(f"内容: {doc.page_content[:200]}...")
                print(f"元数据: {doc.metadata}")
        else:
            print("CSV数据处理失败！")
    else:
        print(f"未找到数据文件: {csv_path}")

    print("\n数据库信息:")
    info = db_manager.get_database_info()
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
