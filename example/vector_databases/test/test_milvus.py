#!/usr/bin/env python
"""
测试 Milvus 连接和基本功能
"""
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_milvus_connection():
    """测试 Milvus 连接"""
    try:
        from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
        
        # 连接 Milvus
        logger.info("=" * 50)
        logger.info("测试 Milvus 连接")
        logger.info("=" * 50)
        
        # 连接参数（根据你的配置调整）
        host = "milvus"  # 如果使用 docker-compose 中的服务名
        # host = "localhost"  # 如果直接连接本地
        port = "19530"
        
        logger.info(f"尝试连接 Milvus: {host}:{port}")
        connections.connect("default", host=host, port=port)
        logger.info("✓ 成功连接到 Milvus")
        
        # 获取 Milvus 版本
        version = utility.get_server_version()
        logger.info(f"✓ Milvus 版本: {version}")
        
        # 列出所有集合
        collections = utility.list_collections()
        logger.info(f"✓ 现有集合列表: {collections}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Milvus 连接失败: {e}")
        return False

def test_collection_operations():
    """测试集合操作"""
    try:
        from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
        import numpy as np
        
        logger.info("\n" + "=" * 50)
        logger.info("测试集合操作")
        logger.info("=" * 50)
        
        # 连接
        connections.connect("default", host="milvus", port="19530")
        
        # 创建测试集合
        test_collection_name = "test_collection_temp"
        
        # 如果测试集合已存在，先删除
        if utility.has_collection(test_collection_name):
            logger.info(f"删除已存在的测试集合: {test_collection_name}")
            utility.drop_collection(test_collection_name)
        
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)
        ]
        
        # 创建集合 schema
        schema = CollectionSchema(fields, description="Test collection")
        
        # 创建集合
        logger.info(f"创建测试集合: {test_collection_name}")
        collection = Collection(test_collection_name, schema)
        logger.info("✓ 集合创建成功")
        
        # 插入测试数据
        logger.info("插入测试数据...")
        num_entities = 10
        texts = [f"测试文本 {i}" for i in range(num_entities)]
        vectors = [[np.random.random(128).tolist()] for _ in range(num_entities)]
        
        entities = [texts, vectors]
        insert_result = collection.insert(entities)
        logger.info(f"✓ 成功插入 {len(insert_result.primary_keys)} 条数据")
        
        # 创建索引
        logger.info("创建索引...")
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index("vector", index_params)
        logger.info("✓ 索引创建成功")
        
        # 加载集合
        logger.info("加载集合到内存...")
        collection.load()
        logger.info("✓ 集合加载成功")
        
        # 搜索测试
        logger.info("执行搜索测试...")
        search_vectors = [np.random.random(128).tolist()]
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        results = collection.search(
            data=search_vectors,
            anns_field="vector",
            param=search_params,
            limit=5,
            output_fields=["text"]
        )
        
        logger.info(f"✓ 搜索成功，返回 {len(results[0])} 条结果")
        for i, hit in enumerate(results[0]):
            logger.info(f"  - 结果 {i+1}: {hit.entity.get('text')}, 距离: {hit.distance:.4f}")
        
        # 清理测试数据
        logger.info("清理测试数据...")
        collection.drop()
        logger.info("✓ 测试集合已删除")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 集合操作测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_performance():
    """测试 Milvus 性能"""
    try:
        from pymilvus import connections, Collection
        import time
        import numpy as np
        
        logger.info("\n" + "=" * 50)
        logger.info("测试性能")
        logger.info("=" * 50)
        
        connections.connect("default", host="milvus", port="19530")
        
        # 使用现有集合或创建临时集合
        test_collection = "performance_test"
        
        # 创建测试集合
        from pymilvus import FieldSchema, CollectionSchema, DataType
        
        if utility.has_collection(test_collection):
            utility.drop_collection(test_collection)
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)
        ]
        schema = CollectionSchema(fields)
        collection = Collection(test_collection, schema)
        
        # 测试插入性能
        logger.info("测试插入性能...")
        num_vectors = 1000
        vectors = [[np.random.random(128).tolist()] for _ in range(num_vectors)]
        
        start_time = time.time()
        collection.insert(vectors)
        insert_time = time.time() - start_time
        logger.info(f"✓ 插入 {num_vectors} 条向量耗时: {insert_time:.2f}秒")
        logger.info(f"  - 平均每条: {insert_time/num_vectors*1000:.2f}毫秒")
        
        # 创建索引
        logger.info("创建索引...")
        start_time = time.time()
        index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        collection.create_index("vector", index_params)
        index_time = time.time() - start_time
        logger.info(f"✓ 索引创建耗时: {index_time:.2f}秒")
        
        # 加载集合
        logger.info("加载集合...")
        start_time = time.time()
        collection.load()
        load_time = time.time() - start_time
        logger.info(f"✓ 集合加载耗时: {load_time:.2f}秒")
        
        # 测试搜索性能
        logger.info("测试搜索性能...")
        search_vectors = [np.random.random(128).tolist()]
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        num_searches = 100
        start_time = time.time()
        for _ in range(num_searches):
            results = collection.search(
                data=search_vectors,
                anns_field="vector",
                param=search_params,
                limit=10
            )
        search_time = time.time() - start_time
        logger.info(f"✓ {num_searches} 次搜索耗时: {search_time:.2f}秒")
        logger.info(f"  - 平均每次: {search_time/num_searches*1000:.2f}毫秒")
        
        # 清理
        collection.drop()
        logger.info("✓ 测试完成，已清理")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 性能测试失败: {e}")
        return False

def test_vector_db_manager():
    """测试 VectorDatabaseManager"""
    try:
        import sys
        sys.path.append('/app/example/vector_databases')
        from vector_db_manager import VectorDatabaseManager
        
        logger.info("\n" + "=" * 50)
        logger.info("测试 VectorDatabaseManager")
        logger.info("=" * 50)
        
        # 创建管理器
        logger.info("创建 VectorDatabaseManager...")
        db_manager = VectorDatabaseManager(
            milvus_host="milvus",
            milvus_port=19530,
            collection_name="test_manager",
            chunk_size=500,
            chunk_overlap=50
        )
        logger.info("✓ VectorDatabaseManager 创建成功")
        
        # 获取数据库信息
        logger.info("获取数据库信息...")
        info = db_manager.get_database_info()
        logger.info(f"✓ 数据库信息: {info}")
        
        # 测试添加文档
        logger.info("测试添加文档...")
        from langchain_core.documents import Document
        
        test_docs = [
            Document(page_content="这是测试文档1", metadata={"source": "test1"}),
            Document(page_content="这是测试文档2", metadata={"source": "test2"}),
            Document(page_content="这是测试文档3", metadata={"source": "test3"})
        ]
        
        db_manager.add_documents_to_db(test_docs, "test_manager")
        logger.info("✓ 文档添加成功")
        
        # 测试搜索
        logger.info("测试搜索...")
        results = db_manager.search("测试", k=2, collection_name="test_manager")
        logger.info(f"✓ 搜索成功，返回 {len(results)} 条结果")
        
        # 清理
        db_manager.clear_database()
        logger.info("✓ 测试完成，已清理")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ VectorDatabaseManager 测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("\n")
    print("=" * 60)
    print("Milvus 可用性测试工具")
    print("=" * 60)
    
    # 测试连接
    conn_ok = test_milvus_connection()
    
    if not conn_ok:
        print("\n✗ Milvus 连接失败，请检查：")
        print("  1. Milvus 服务是否运行: docker ps | grep milvus")
        print("  2. 端口是否正确: netstat -an | grep 19530")
        print("  3. 主机地址是否正确")
        sys.exit(1)
    
    # 测试集合操作
    collection_ok = test_collection_operations()
    
    # 测试性能
    performance_ok = test_performance()
    
    # 测试 VectorDatabaseManager
    manager_ok = test_vector_db_manager()
    
    # 输出总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    print(f"Milvus 连接: {'✓ 正常' if conn_ok else '✗ 异常'}")
    print(f"集合操作: {'✓ 正常' if collection_ok else '✗ 异常'}")
    print(f"性能测试: {'✓ 正常' if performance_ok else '✗ 异常'}")
    print(f"VectorDatabaseManager: {'✓ 正常' if manager_ok else '✗ 异常'}")
    
    if all([conn_ok, collection_ok, performance_ok, manager_ok]):
        print("\n✓ 所有测试通过！Milvus 工作正常。")
        sys.exit(0)
    else:
        print("\n✗ 部分测试失败，请检查 Milvus 配置。")
        sys.exit(1)