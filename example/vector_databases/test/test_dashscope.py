#!/usr/bin/env python
"""
测试 DashScope 嵌入模型
"""
import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dashscope_embedding():
    """测试 DashScope 嵌入模型"""
    
    # 配置
    api_key = "sk-8c497e3d49a849d3a9c161c551793fe2"
    model = "text-embedding-v1"
    
    logger.info("=" * 50)
    logger.info("开始测试 DashScope 嵌入模型")
    logger.info(f"API Key: {api_key[:10]}...{api_key[-10:]}")
    logger.info(f"模型: {model}")
    logger.info("=" * 50)
    
    try:
        # 导入必要的库
        from langchain_community.embeddings import DashScopeEmbeddings
        
        # 创建嵌入模型实例
        logger.info("1. 创建 DashScopeEmbeddings 实例...")
        embeddings = DashScopeEmbeddings(
            model=model,
            dashscope_api_key=api_key
        )
        logger.info("✓ 实例创建成功")
        
        # 测试单个文本嵌入
        logger.info("\n2. 测试单个文本嵌入 (embed_query)...")
        test_text = "这是一个测试文本，用于验证 DashScope 嵌入模型是否正常工作"
        
        start_time = time.time()
        embedding = embeddings.embed_query(test_text)
        elapsed_time = time.time() - start_time
        
        if embedding:
            logger.info(f"✓ 嵌入生成成功")
            logger.info(f"  - 向量维度: {len(embedding)}")
            logger.info(f"  - 前5个值: {embedding[:5]}")
            logger.info(f"  - 耗时: {elapsed_time:.2f} 秒")
        else:
            logger.error("✗ 嵌入生成失败，返回空向量")
            return False
        
        # 测试批量文本嵌入
        logger.info("\n3. 测试批量文本嵌入 (embed_documents)...")
        test_texts = [
            "第一个测试文本",
            "第二个测试文本，内容不同",
            "第三个测试文本，用于测试批量处理能力"
        ]
        
        start_time = time.time()
        embeddings_list = embeddings.embed_documents(test_texts)
        elapsed_time = time.time() - start_time
        
        if embeddings_list:
            logger.info(f"✓ 批量嵌入生成成功")
            logger.info(f"  - 批次大小: {len(embeddings_list)}")
            logger.info(f"  - 向量维度: {len(embeddings_list[0]) if embeddings_list else 0}")
            logger.info(f"  - 耗时: {elapsed_time:.2f} 秒")
            
            # 检查向量是否不同
            if len(embeddings_list) >= 2:
                import numpy as np
                vec1 = np.array(embeddings_list[0])
                vec2 = np.array(embeddings_list[1])
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                logger.info(f"  - 前两个向量的相似度: {similarity:.4f}")
        else:
            logger.error("✗ 批量嵌入生成失败")
            return False
        
        # 测试连接稳定性
        logger.info("\n4. 测试连接稳定性（连续调用）...")
        success_count = 0
        for i in range(3):
            try:
                test_embed = embeddings.embed_query(f"测试调用 {i+1}")
                if test_embed:
                    success_count += 1
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"第{i+1}次调用失败: {e}")
        
        logger.info(f"✓ 连续调用成功率: {success_count}/3")
        
        logger.info("\n" + "=" * 50)
        logger.info("✓ DashScope 嵌入模型测试全部通过！")
        logger.info("=" * 50)
        return True
        
    except ImportError as e:
        logger.error(f"✗ 导入错误: {e}")
        logger.error("请确保已安装 langchain-community: pip install langchain-community")
        return False
        
    except Exception as e:
        logger.error(f"✗ 测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_with_vector_db_manager():
    """通过 VectorDatabaseManager 测试"""
    logger.info("\n" + "=" * 50)
    logger.info("通过 VectorDatabaseManager 测试")
    logger.info("=" * 50)
    
    try:
        # 动态导入，避免循环依赖
        sys.path.append(os.path.dirname(__file__))
        from vector_db_manager import VectorDatabaseManager
        
        # 创建管理器实例
        logger.info("创建 VectorDatabaseManager 实例...")
        db_manager = VectorDatabaseManager(
            milvus_host="milvus",  # 使用 docker-compose 中的服务名
            milvus_port=19530,
            collection_name="test_collection",
            embedding_model="text-embedding-v1",
            dashscope_api_key="sk-8c497e3d49a849d3a9c161c551793fe2",
            chunk_size=500,
            chunk_overlap=50
        )
        
        logger.info("✓ VectorDatabaseManager 创建成功")
        
        # 测试嵌入功能
        logger.info("\n测试嵌入功能...")
        test_texts = ["测试文本1", "测试文本2"]
        embeddings = db_manager.get_embedding(test_texts)
        
        if embeddings and len(embeddings) == 2:
            logger.info(f"✓ 嵌入功能正常")
            logger.info(f"  - 向量维度: {len(embeddings[0])}")
        else:
            logger.error("✗ 嵌入功能异常")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"✗ VectorDatabaseManager 测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def check_network_connectivity():
    """检查网络连接"""
    logger.info("\n" + "=" * 50)
    logger.info("检查网络连接")
    logger.info("=" * 50)
    
    try:
        import requests
        
        # 测试 DashScope API 连通性
        url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding-v1"
        headers = {
            "Authorization": f"Bearer sk-8c497e3d49a849d3a9c161c551793fe2",
            "Content-Type": "application/json"
        }
        
        logger.info("测试 DashScope API 连通性...")
        start_time = time.time()
        response = requests.get("https://dashscope.aliyuncs.com", timeout=5)
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            logger.info(f"✓ DashScope API 可访问 (耗时: {elapsed_time:.2f}秒)")
        else:
            logger.warning(f"⚠ DashScope API 响应状态码: {response.status_code}")
            
        return True
        
    except requests.exceptions.Timeout:
        logger.error("✗ 网络超时，无法连接到 DashScope API")
        return False
    except requests.exceptions.ConnectionError:
        logger.error("✗ 网络连接错误，请检查网络设置")
        return False
    except Exception as e:
        logger.error(f"✗ 网络检查失败: {e}")
        return False

if __name__ == "__main__":
    print("\n")
    print("=" * 60)
    print("DashScope 嵌入模型诊断工具")
    print("=" * 60)
    
    # 检查网络
    network_ok = check_network_connectivity()
    
    # 测试嵌入模型
    embedding_ok = test_dashscope_embedding()
    
    # 测试 VectorDatabaseManager
    if embedding_ok:
        manager_ok = test_with_vector_db_manager()
    else:
        manager_ok = False
    
    # 输出总结
    print("\n" + "=" * 60)
    print("诊断结果总结")
    print("=" * 60)
    print(f"网络连接: {'✓ 正常' if network_ok else '✗ 异常'}")
    print(f"DashScope 嵌入模型: {'✓ 正常' if embedding_ok else '✗ 异常'}")
    print(f"VectorDatabaseManager: {'✓ 正常' if manager_ok else '✗ 异常'}")
    
    if embedding_ok and manager_ok:
        print("\n✓ 所有测试通过！DashScope 嵌入模型工作正常。")
        sys.exit(0)
    else:
        print("\n✗ 部分测试失败，请检查配置。")
        sys.exit(1)