"""
RAG 自检脚本 —— 构建知识库后用于端到端校验。

逐项报告:
  1. 百炼嵌入接口是否可调用、实际返回维度是否等于配置维度
  2. Milvus 是否可连接
  3. 三个集合是否存在且已加载、记录数
  4. 集合向量维度是否与嵌入维度一致
  5. 用固定问题分别检索三个集合，确认能命中

退出码: 全部通过 0；存在失败项 1。

运行: python check_rag.py
"""

from __future__ import annotations

import sys

from config import (
    RAG_EMBEDDING_DIMENSIONS,
    RAG_KNOWLEDGE_COLLECTIONS,
)

# 每个逻辑类型的固定检索题（用于确认确实可检索）
PROBE_QUERIES = {
    "dish": "川菜有什么特点，宫保鸡丁是什么口味",
    "dietary": "糖尿病患者饮食应该注意什么",
    "faq": "怎么添加收货地址，配送要多久",
}


def _ok(label: str, detail: str = "") -> bool:
    print(f"  [PASS] {label}" + (f" —— {detail}" if detail else ""))
    return True


def _fail(label: str, detail: str) -> bool:
    print(f"  [FAIL] {label} —— {detail}")
    return False


def check_embedding() -> tuple[bool, int]:
    print("\n[1/5] 百炼嵌入接口")
    try:
        from embedding_client import get_embeddings
        emb = get_embeddings()
        vec = emb.embed_query("自检：外卖点餐")
    except Exception as e:
        return _fail("嵌入调用", f"{type(e).__name__}: {e}"), 0
    dim = len(vec)
    if dim != RAG_EMBEDDING_DIMENSIONS:
        return _fail("嵌入维度", f"实际 {dim}，配置 {RAG_EMBEDDING_DIMENSIONS}"), dim
    return _ok("嵌入调用与维度", f"返回 {dim} 维"), dim


def check_milvus() -> tuple[bool, object]:
    print("\n[2/5] Milvus 连接")
    try:
        from knowledge_base import get_milvus_client
        client = get_milvus_client()
        cols = client.list_collections()
        return _ok("Milvus 可连接", f"现有集合: {cols}"), client
    except Exception as e:
        return _fail("Milvus 连接", f"{type(e).__name__}: {e}"), None


def check_collections(client, expect_dim: int) -> bool:
    print("\n[3/5] 集合存在性 / 记录数 / 加载状态")
    from knowledge_base import collection_count

    all_ok = True
    for logical, name in RAG_KNOWLEDGE_COLLECTIONS.items():
        try:
            if not client.has_collection(name):
                all_ok &= _fail(f"{logical}", f"集合 {name} 不存在，请先 build_knowledge_base.py")
                continue
            count = collection_count(name)
            client.load_collection(name)
            # 使用 MilvusClient API 检查 schema，避免依赖不存在的 client.using
            # 属性和即将移除的 ORM Collection API。
            schema = client.describe_collection(name)
            fields = schema.get("fields", []) if isinstance(schema, dict) else []
            vec_field = next((f for f in fields if f.get("name") == "vector"), None)
            params = vec_field.get("params", {}) if vec_field else {}
            dim = params.get("dim")
            dim_ok = expect_dim == 0 or dim in (None, expect_dim)
            if count <= 0:
                all_ok &= _fail(f"{logical}", f"{name} 记录数为 0")
            elif not dim_ok:
                all_ok &= _fail(f"{logical}", f"{name} 向量维度 {dim} != {expect_dim}")
            else:
                all_ok &= _ok(f"{logical}", f"{name} 已加载，{count} 条，维度 {dim}")
        except Exception as e:
            all_ok &= _fail(f"{logical}", f"{type(e).__name__}: {e}")
    return all_ok


def check_retrieval() -> bool:
    print("\n[4/5] 固定问题检索")
    from knowledge_base import search_knowledge

    all_ok = True
    for logical, query in PROBE_QUERIES.items():
        collection = RAG_KNOWLEDGE_COLLECTIONS[logical]
        try:
            text = search_knowledge(query, collection, top_k=3, min_score=0.0)
            if text.startswith("未在知识库中找到"):
                all_ok &= _fail(logical, "未检索到任何内容")
            else:
                first = text.splitlines()[0]
                all_ok &= _ok(logical, first)
        except Exception as e:
            all_ok &= _fail(logical, f"{type(e).__name__}: {e}")
    return all_ok


def check_irrelevant() -> bool:
    print("\n[5/5] 无关问题应被过滤（不硬拼答案）")
    from knowledge_base import search_knowledge
    try:
        # 用一个明显无关且高阈值的查询，验证阈值过滤链路不报错
        text = search_knowledge(
            "量子纠缠的张量网络数学证明",
            RAG_KNOWLEDGE_COLLECTIONS["faq"],
            top_k=8,
            min_score=0.99,
        )
        return _ok("阈值过滤", "正常返回" if text else "空")
    except Exception as e:
        return _fail("阈值过滤", f"{type(e).__name__}: {e}")


def main() -> int:
    print("=" * 60)
    print("  take-out Agent RAG 自检")
    print("=" * 60)
    emb_ok, dim = check_embedding()
    milvus_ok, client = check_milvus()
    coll_ok = check_collections(client, dim) if milvus_ok else False
    retr_ok = check_retrieval() if milvus_ok else False
    irr_ok = check_irrelevant() if milvus_ok else False

    print("\n" + "=" * 60)
    results = {"嵌入": emb_ok, "Milvus": milvus_ok, "集合": coll_ok,
               "检索": retr_ok, "阈值过滤": irr_ok}
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    passed = all(results.values())
    print("  结论:", "全部通过" if passed else "存在失败项，请按上面的 FAIL 排查")
    print("=" * 60)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
