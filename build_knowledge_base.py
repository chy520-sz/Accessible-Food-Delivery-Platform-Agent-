"""
知识库构建脚本 —— 将 data/ 下的 JSON 知识构建到 Milvus（百炼 Qwen 嵌入，1024 维）。

构建流水线:
  校验 JSON → 补齐来源与业务标识 → 分块 → 生成稳定主键与内容哈希
  → 批量云端嵌入（带本地缓存）→ 写入新版本集合 → 校验数量/维度/检索 → 输出报告

设计要点:
  - 幂等：同一集合全量重建时先删后建，重复构建不会追加重复向量；
          主键由 逻辑类型|业务标识|块序号|内容哈希 稳定生成。
  - 更新文档：重建会整体替换集合，旧分块随之清除，不会残留陈旧块。
  - FAQ 优先保留完整问答；菜品优先保留完整条目，超过 CHUNK_SIZE 再分块。
  - 第一轮保留 300 字符 / 50 重叠作为比较基线。
  - 旧 Chroma 的 BGE 向量维度/模型均不同，不能搬迁，必须重新嵌入。

运行:
  python build_knowledge_base.py --dry-run   # 仅校验 JSON 与分块，不连 Milvus、不耗额度
  python build_knowledge_base.py             # 全量构建三个集合
  python build_knowledge_base.py --only dish # 只构建某个逻辑类型
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    RAG_BM25_DROP_RATIO,
    RAG_CONTENT_VERSION,
    RAG_COLLECTION_DIETARY,
    RAG_COLLECTION_FAQ,
    RAG_COLLECTION_FOOD,
    RAG_EMBEDDING_DIMENSIONS,
    RAG_EMBEDDING_MODEL,
    RAG_INDEX_TYPE,
    RAG_METRIC_TYPE,
    RAG_SPARSE_FIELD,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# 第一轮基线：300 字符 / 50 重叠
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

# 构建版本（写入元数据，便于排查集合是用哪版逻辑生成的）
BUILD_VERSION = time.strftime("%Y%m%d")


@dataclass
class BuildSpec:
    logical_type: str
    filename: str
    collection: str


SPECS = [
    BuildSpec("dish", "dish_knowledge.json", RAG_COLLECTION_FOOD),
    BuildSpec("dietary", "dietary_knowledge.json", RAG_COLLECTION_DIETARY),
    BuildSpec("faq", "faq_knowledge.json", RAG_COLLECTION_FAQ),
]


# ==================== JSON 加载与校验 ====================

def load_json(filename: str) -> list[dict]:
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"知识文件不存在: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{filename} 顶层必须是 JSON 数组，当前为 {type(data).__name__}")
    return data


def _require(item: dict, keys: tuple[str, ...]) -> None:
    missing = [k for k in keys if not item.get(k)]
    if missing:
        raise ValueError(f"记录缺少必填字段 {missing}：{json.dumps(item, ensure_ascii=False)[:120]}")


# ==================== 条目 -> 文本 + 业务标识 ====================

def _dish_entry(item: dict) -> tuple[str, str, dict]:
    _require(item, ("name",))
    name = item.get("name", "")
    category = item.get("category", "")
    ingredients = "、".join(item.get("main_ingredients", []) or [])
    suitable = "、".join(item.get("suitable_for", []) or [])
    tags = "、".join(item.get("tags", []) or [])
    allergens = "、".join(item.get("allergens", []) or [])
    nut = item.get("nutrition", {}) or {}
    nutrition = ""
    if nut:
        nutrition = (
            f"营养（每100g）：热量{nut.get('calories', '未知')}kcal、"
            f"蛋白质{nut.get('protein', '未知')}g、碳水{nut.get('carbs', '未知')}g、"
            f"脂肪{nut.get('fat', '未知')}g\n"
        )
    text = (
        f"菜品：{name}\n分类：{category}\n口味：{item.get('taste', '')}\n"
        f"烹饪方式：{item.get('cooking_method', '')}\n主要食材：{ingredients}\n"
        f"{nutrition}"
        f"描述：{item.get('description', '')}\n适合人群：{suitable}\n"
        f"标签：{tags}\n过敏原：{allergens}"
    )
    business_id = str(item.get("dish_id") or f"{name}|{item.get('shop', '')}")
    # 写入 Milvus 的元数据：用于标量过滤（category/shop/spicy_level/taste 等）
    # 注意：spicy_level 保持 int 类型，便于数值范围过滤；tags 保持字符串便于 like 模糊匹配
    meta = {
        "name": name,
        "category": category,
        "tags": tags,
        "shop": item.get("shop", ""),
        "taste": item.get("taste", ""),
        "cooking_method": item.get("cooking_method", ""),
        "spicy_level": item.get("spicy_level") if item.get("spicy_level") is not None else -1,
        "price": float(item.get("price", 0)) if item.get("price") else 0.0,
    }
    return text, business_id, meta


def _dietary_entry(item: dict) -> tuple[str, str, dict]:
    _require(item, ("condition",))
    condition = item.get("condition", "")
    keywords = "、".join(item.get("keywords", []) or [])
    restricted = "、".join(item.get("restricted_foods", []) or [])
    recommended = "、".join(item.get("recommended_foods", []) or [])
    text = (
        f"健康状况：{condition}\n关键词：{keywords}\n应避免的食物：{restricted}\n"
        f"推荐食物：{recommended}\n饮食建议：{item.get('advice', '')}"
    )
    meta = {"condition": condition, "tags": "、".join(item.get("tags", []) or [])}
    return text, condition, meta


def _faq_entry(item: dict) -> tuple[str, str, dict]:
    _require(item, ("question", "answer"))
    question, answer = item["question"], item["answer"]
    category = item.get("category", "")
    text = f"问题：{question}\n回答：{answer}"
    meta = {"category": category, "keywords": "、".join(item.get("keywords", []) or [])}
    return text, question, meta


_ENTRY_BUILDERS = {
    "dish": _dish_entry,
    "dietary": _dietary_entry,
    "faq": _faq_entry,
}


# ==================== 稳定主键 / 内容哈希 ====================

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_pk(logical_type: str, business_id: str, chunk_idx: int, content: str) -> str:
    """稳定分块主键：逻辑类型|业务标识|块序号|内容哈希。内容不变则主键不变。"""
    digest = _sha(f"{logical_type}|{business_id}|{chunk_idx}|{RAG_CONTENT_VERSION}|{content}")
    return f"{logical_type}_{digest[:24]}"


# ==================== 分块 ====================

def _make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHINESE_SEPARATORS,
        keep_separator=True,
    )


def build_documents(spec: BuildSpec) -> list[Document]:
    """把一个知识 JSON 转成带稳定主键与完整元数据的 Document 列表。"""
    entries = load_json(spec.filename)
    splitter = _make_splitter()
    docs: list[Document] = []

    for item in entries:
        text, business_id, extra_meta = _ENTRY_BUILDERS[spec.logical_type](item)
        doc_id = _sha(f"{spec.logical_type}|{business_id}")
        # 菜品/FAQ 优先保留完整条目；超长才分块
        if len(text) <= CHUNK_SIZE:
            pieces = [text]
        else:
            pieces = splitter.split_text(text)
        for idx, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            content_hash = _sha(piece)
            metadata = {
                "knowledge_type": spec.logical_type,
                "source": spec.filename,
                "doc_id": doc_id,
                "business_id": business_id,
                "chunk_idx": idx,
                "chunk_count": len(pieces),
                "content_hash": content_hash,
                "embedding_model": RAG_EMBEDDING_MODEL,
                "embedding_dim": RAG_EMBEDDING_DIMENSIONS,
                "chunk_version": RAG_CONTENT_VERSION,
                "build_version": BUILD_VERSION,
                **extra_meta,
            }
            docs.append(
                Document(page_content=piece, metadata=metadata, id=stable_pk(
                    spec.logical_type, business_id, idx, piece
                ))
            )
    return docs


# ==================== 写入 Milvus ====================

def _create_store(collection: str):
    """使用官方 MilvusClient 创建一个全新的集合（稠密向量 + BM25 稀疏向量双路）。"""
    from pymilvus import DataType, Function, FunctionType
    from knowledge_base import get_milvus_client, reset_store_cache

    client = get_milvus_client()
    if client.has_collection(collection):
        client.drop_collection(collection)
        print(f"    已删除旧集合 {collection}")
    reset_store_cache()

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("pk", DataType.VARCHAR, is_primary=True, max_length=128)
    # text 字段启用中文分析器，作为 BM25 Function 的输入
    schema.add_field(
        "text",
        DataType.VARCHAR,
        max_length=65535,
        enable_analyzer=True,
        analyzer_params={"type": "chinese"},
    )
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=RAG_EMBEDDING_DIMENSIONS)
    # BM25 稀疏向量字段：由 BM25 Function 从 text 自动生成，写入时无需手动提供
    schema.add_field(RAG_SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)
    # 注册 BM25 Function：text -> sparse_vector
    schema.add_function(Function(
        name="bm25_fn",
        input_field_names=["text"],
        output_field_names=[RAG_SPARSE_FIELD],
        function_type=FunctionType.BM25,
    ))

    index_params = client.prepare_index_params()
    # 稠密向量索引（语义检索）
    index_params.add_index(
        field_name="vector",
        index_type=RAG_INDEX_TYPE,
        metric_type=RAG_METRIC_TYPE,
        params={},
    )
    # 稀疏向量倒排索引（BM25 关键词检索）
    index_params.add_index(
        field_name=RAG_SPARSE_FIELD,
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"drop_ratio_build": RAG_BM25_DROP_RATIO},
    )
    client.create_collection(
        collection_name=collection,
        schema=schema,
        index_params=index_params,
    )
    client.load_collection(collection)
    return client


def _verify(collection: str, expected: int, docs: list[Document]) -> None:
    from knowledge_base import collection_count

    actual = collection_count(collection)
    if actual != expected:
        raise RuntimeError(f"集合 {collection} 数量校验失败：期望 {expected}，实际 {actual}")
    # 用首条文本做一次检索冒烟，确认维度匹配、可检索
    from knowledge_base import search_knowledge
    probe = docs[0].page_content[:40]
    result = search_knowledge(probe, collection, top_k=3, min_score=0.0)
    if "未在知识库中找到" in result:
        raise RuntimeError(f"集合 {collection} 检索冒烟失败：构建后无法检索到内容")


def build_one(spec: BuildSpec, embeddings, dry_run: bool) -> int:
    print(f"\n  --- {spec.logical_type} -> {spec.collection} ({spec.filename}) ---")
    docs = build_documents(spec)
    # 主键唯一性校验
    pks = [d.id for d in docs]
    if len(set(pks)) != len(pks):
        dup = len(pks) - len(set(pks))
        raise RuntimeError(f"{spec.filename} 生成了 {dup} 个重复主键，构建中止")
    print(f"    分块数: {len(docs)}（chunk={CHUNK_SIZE}/overlap={CHUNK_OVERLAP}）")

    if dry_run:
        print("    [dry-run] 跳过嵌入与写入。")
        return len(docs)

    store = _create_store(spec.collection)
    t0 = time.time()
    vectors = embeddings.embed_documents([doc.page_content for doc in docs])
    rows = [
        {"pk": doc.id, "text": doc.page_content, "vector": vector, **doc.metadata}
        for doc, vector in zip(docs, vectors)
    ]
    result = store.insert(collection_name=spec.collection, data=rows)
    store.flush(spec.collection)
    inserted = int(result.get("insert_count", len(rows)))
    print(f"    写入 {inserted} 条向量，耗时 {time.time() - t0:.1f}s")
    if inserted != len(rows):
        raise RuntimeError(f"集合 {spec.collection} 写入数量不符：期望 {len(rows)}，实际 {inserted}")
    _verify(spec.collection, len(docs), docs)
    print(f"    [OK] 集合 {spec.collection} 数量/维度/检索校验通过")
    return len(docs)


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 take-out Agent 的 Milvus 知识库")
    parser.add_argument("--dry-run", action="store_true", help="只校验 JSON 与分块，不连 Milvus、不消耗嵌入额度")
    parser.add_argument("--only", choices=[s.logical_type for s in SPECS], help="只构建指定逻辑类型")
    args = parser.parse_args()

    print("=" * 60)
    print("  take-out Agent —— Milvus 知识库构建")
    print(f"  嵌入模型: {RAG_EMBEDDING_MODEL} / {RAG_EMBEDDING_DIMENSIONS} 维 / {RAG_METRIC_TYPE}")
    print("=" * 60)

    specs = [s for s in SPECS if (args.only is None or s.logical_type == args.only)]

    # dry-run 不需要嵌入客户端
    embeddings = None
    if not args.dry_run:
        from embedding_client import get_embeddings
        embeddings = get_embeddings()
        # 提前确认 Milvus 可达，避免嵌入计费后才发现连不上
        from knowledge_base import ping_milvus
        if not ping_milvus():
            print("[ERROR] Milvus 不可达，请先用 docker compose -f docker-compose.milvus.yml up -d 启动。")
            return 1

    total = 0
    try:
        for spec in specs:
            total += build_one(spec, embeddings, args.dry_run)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] 知识数据校验失败: {e}")
        return 1
    except Exception as e:
        print(f"[ERROR] 构建失败: {type(e).__name__}: {e}")
        return 1

    print("\n" + "=" * 60)
    if args.dry_run:
        print(f"  [dry-run] 校验通过，总分块 {total}，未写入。")
    else:
        print(f"  [OK] 构建完成，总向量数 {total}。可运行 python check_rag.py 自检。")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
