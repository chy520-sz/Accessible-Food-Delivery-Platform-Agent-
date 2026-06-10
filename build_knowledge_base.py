"""
知识库构建脚本 —— 离线构建/重建向量数据库。

读取 data/ 目录下的 JSON 知识文件，使用 RecursiveCharacterTextSplitter 进行
中文友好的文本分块，通过 BGE 嵌入模型向量化后持久化到 Chroma 向量数据库。

运行方式:
  python build_knowledge_base.py            # 首次构建
  python build_knowledge_base.py --rebuild  # 强制重建（删除现有数据）

知识集合:
  - dish_knowledge: 菜品语义知识（口味、做法、菜系等）
  - dietary_knowledge: 饮食健康知识（疾病饮食限制、营养建议等）
  - faq: 常见问题（点餐流程、配送、支付等）
"""
import argparse
import json
import os
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

from config import (
    RAG_PERSIST_DIR,
    RAG_COLLECTION_FOOD,
    RAG_COLLECTION_DIETARY,
    RAG_COLLECTION_FAQ,
)

# 项目根目录（即本脚本所在目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# 中文文本分块配置
CHUNK_SIZE = 300       # 每块最大字符数
CHUNK_OVERLAP = 50     # 块间重叠字符数

# 中文友好的分隔符：优先在段落、句子边界断开
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


def load_json(filename: str) -> list[dict]:
    """从 data/ 目录加载 JSON 知识文件。

    参数:
        filename: JSON 文件名，如 "dish_knowledge.json"

    返回:
        解析后的字典列表。文件不存在时退出并报错。
    """
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"[ERROR] 文件不存在: {filepath}")
        print("   请确保 data/ 目录下有对应的知识 JSON 文件。")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  [OK] 加载 {filename}: {len(data)} 条记录")
    return data


def build_food_knowledge() -> list[Document]:
    """将菜品知识 JSON 转换为 LangChain Document 列表。

    每条菜品生成为独立文档，元数据包含名称、分类和标签，
    便于检索时做过滤和展示。
    """
    entries = load_json("dish_knowledge.json")
    docs = []
    for item in entries:
        name = item.get("name", "")
        category = item.get("category", "")
        taste = item.get("taste", "")
        cooking = item.get("cooking_method", "")
        ingredients = "、".join(item.get("main_ingredients", []))
        description = item.get("description", "")
        suitable = "、".join(item.get("suitable_for", []))
        tags = "、".join(item.get("tags", []))

        # 构造语义丰富的文本
        text = (
            f"菜品：{name}\n"
            f"分类：{category}\n"
            f"口味：{taste}\n"
            f"烹饪方式：{cooking}\n"
            f"主要食材：{ingredients}\n"
            f"描述：{description}\n"
            f"适合人群：{suitable}"
        )

        docs.append(Document(
            page_content=text,
            metadata={
                "source": "dish_knowledge",
                "name": name,
                "category": category,
                "tags": tags,
            },
        ))
    return docs


def build_dietary_knowledge() -> list[Document]:
    """将饮食健康知识 JSON 转换为 LangChain Document 列表。

    每条健康建议生成为独立文档，元数据包含健康状况和关键词。
    """
    entries = load_json("dietary_knowledge.json")
    docs = []
    for item in entries:
        condition = item.get("condition", "")
        keywords = "、".join(item.get("keywords", []))
        restricted = "、".join(item.get("restricted_foods", []))
        recommended = "、".join(item.get("recommended_foods", []))
        advice = item.get("advice", "")
        tags = "、".join(item.get("tags", []))

        text = (
            f"健康状况：{condition}\n"
            f"关键词：{keywords}\n"
            f"应避免的食物：{restricted}\n"
            f"推荐食物：{recommended}\n"
            f"饮食建议：{advice}"
        )

        docs.append(Document(
            page_content=text,
            metadata={
                "source": "dietary_knowledge",
                "condition": condition,
                "tags": tags,
            },
        ))
    return docs


def build_faq_knowledge() -> list[Document]:
    """将 FAQ JSON 转换为 LangChain Document 列表。

    每个问答对生成为独立文档，元数据包含分类和关键词。
    """
    entries = load_json("faq_knowledge.json")
    docs = []
    for item in entries:
        question = item.get("question", "")
        answer = item.get("answer", "")
        category = item.get("category", "")
        keywords = "、".join(item.get("keywords", []))

        text = f"问题：{question}\n回答：{answer}"

        docs.append(Document(
            page_content=text,
            metadata={
                "source": "faq",
                "category": category,
                "keywords": keywords,
            },
        ))
    return docs


def main():
    parser = argparse.ArgumentParser(
        description="构建/重建 take-out Agent 的 RAG 知识库"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="强制重建所有集合（删除现有数据）",
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  take-out Agent -- RAG 知识库构建工具")
    print("=" * 55)

    # 导入嵌入模型（放在函数内避免脚本启动时的额外延迟）
    from knowledge_base import _get_embedding_model

    # 1. 加载嵌入模型
    print("\n[1/4] 加载嵌入模型...")
    try:
        embeddings = _get_embedding_model()
    except Exception as e:
        print(f"[ERROR] 嵌入模型加载失败: {e}")
        sys.exit(1)

    # 2. 准备文本分块器
    print(f"\n[2/4] 初始化文本分块器 (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHINESE_SEPARATORS,
        keep_separator=True,  # 保留句号等标点，保持语义完整
    )

    # 3. 构建各知识集合
    print(f"\n[3/4] 构建知识集合并持久化到 {RAG_PERSIST_DIR} ...")
    os.makedirs(RAG_PERSIST_DIR, exist_ok=True)

    collections = [
        ("dish_knowledge", build_food_knowledge, RAG_COLLECTION_FOOD),
        ("dietary_knowledge", build_dietary_knowledge, RAG_COLLECTION_DIETARY),
        ("faq", build_faq_knowledge, RAG_COLLECTION_FAQ),
    ]

    total_chunks = 0

    for name, builder, collection_name in collections:
        print(f"\n  --- {name} ---")
        # 构建原始文档
        raw_docs = builder()
        print(f"    原始文档: {len(raw_docs)} 条")

        # 分块
        chunks = splitter.split_documents(raw_docs)
        print(f"    分块后: {len(chunks)} 条")

        # 持久化到 Chroma
        if args.rebuild:
            # 删除已有集合
            try:
                old_store = Chroma(
                    persist_directory=RAG_PERSIST_DIR,
                    collection_name=collection_name,
                    embedding_function=embeddings,
                )
                old_store.delete_collection()
                print(f"    已删除旧集合")
            except Exception:
                pass

        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=RAG_PERSIST_DIR,
            collection_name=collection_name,
        )
        print(f"    [OK] 已持久化到 Chroma")
        total_chunks += len(chunks)

    # 4. 完成
    print(f"\n[4/4] [OK] 知识库构建完成！")
    print(f"    持久化目录: {os.path.abspath(RAG_PERSIST_DIR)}")
    print(f"    集合数量: {len(collections)}")
    print(f"    总向量数: {total_chunks}")
    print(f"\n现在可以启动 Agent 服务，LLM 将能通过工具使用这些知识。")
    print(f"  python main.py")


if __name__ == "__main__":
    main()
