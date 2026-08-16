"""
知识库检索模块 —— 基于 Chroma 向量数据库 + BGE 中文嵌入模型。

功能:
  - 使用 BAAI/bge-small-zh-v1.5 本地嵌入模型（无需 API Key，~100MB）
  - Chroma 向量存储，持久化到磁盘
  - 语义相似度检索，支持多个知识集合
  - 带 tenacity 自动重试的鲁棒检索

模型下载策略:
  1. 优先使用 ModelScope 国内镜像下载（速度快，网络兼容性好）
  2. 回退到 HuggingFace 官方源
  3. 支持通过环境变量指定本地已下载的模型路径

使用方式:
  from knowledge_base import search_knowledge
  result = search_knowledge("川菜有什么特点？", "dish_knowledge", top_k=5)
"""
import os
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from config import (
    RAG_PERSIST_DIR,
    RAG_EMBEDDING_MODEL,
    RAG_TOP_K,
    MAX_RETRIES,
    RETRY_DELAY,
)

# ==================== 嵌入模型管理 ====================

_embedding_model = None  # 懒加载单例


class _STEmbeddings(Embeddings):
    """sentence-transformers 适配 LangChain Embeddings 接口。

    将 BGE 中文嵌入模型包装为 LangChain 兼容的 Embeddings 类，
    实现 embed_documents 和 embed_query 两个必需方法。
    """

    def __init__(self, model):
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文档（构建知识库时使用）"""
        return self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=True
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        """向量化查询文本（检索时使用）"""
        return self._model.encode(
            text, normalize_embeddings=True
        ).tolist()


def _download_model_from_modelscope(model_name: str) -> str:
    """通过 ModelScope（阿里云国内镜像）下载模型，返回本地路径。

    ModelScope 无需特殊网络配置，适合国内开发环境。
    首次下载约 100MB，后续使用缓存。
    """
    from modelscope import snapshot_download

    # ModelScope 上的模型路径映射
    modelscope_id = model_name  # BAAI/bge-small-zh-v1.5 在 ModelScope 上路径相同
    print(f"[RAG] 正在从 ModelScope 下载模型: {modelscope_id} ...")
    local_path = snapshot_download(modelscope_id)
    print(f"[RAG] ModelScope 下载完成: {local_path}")
    return local_path


def _download_model_from_huggingface(model_name: str) -> str:
    """通过 HuggingFace 下载模型，返回本地路径。

    如果环境有 HuggingFace 镜像（如 hf-mirror.com），
    可通过环境变量 HF_ENDPOINT 指定。
    """
    from huggingface_hub import snapshot_download

    endpoint = os.environ.get("HF_ENDPOINT", "")
    if endpoint:
        print(f"[RAG] 使用 HuggingFace 镜像: {endpoint}")
    print(f"[RAG] 正在从 HuggingFace 下载模型: {model_name} ...")
    local_path = snapshot_download(model_name)
    print(f"[RAG] HuggingFace 下载完成: {local_path}")
    return local_path


def _get_embedding_model():
    """懒加载 BGE 中文嵌入模型。

    下载策略：
      1. 如果环境变量 RAG_MODEL_PATH 已设置，直接使用本地模型
      2. 优先尝试 ModelScope（国内镜像，速度快）
      3. 回退到 HuggingFace 官方源

    模型首次下载约 100MB，后续调用复用缓存。
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    from sentence_transformers import SentenceTransformer

    # 0. 检查是否指定了本地模型路径
    local_path = os.environ.get("RAG_MODEL_PATH", "")
    if local_path and os.path.exists(local_path):
        print(f"[RAG] 使用本地模型: {local_path}")
        model = SentenceTransformer(local_path)
        _embedding_model = _STEmbeddings(model)
        print("[RAG] 嵌入模型加载完成。")
        return _embedding_model

    # 1. 尝试 ModelScope
    model_path = None
    try:
        model_path = _download_model_from_modelscope(RAG_EMBEDDING_MODEL)
    except ImportError:
        print("[RAG] ModelScope 未安装。如需使用国内镜像加速下载，请运行:")
        print("       pip install modelscope")
    except Exception as e:
        print(f"[RAG] ModelScope 下载失败: {e}")

    # 2. 回退到 HuggingFace
    if model_path is None:
        try:
            model_path = _download_model_from_huggingface(RAG_EMBEDDING_MODEL)
        except ImportError:
            raise RuntimeError(
                "未安装 huggingface_hub 库。请运行:\n"
                "  pip install huggingface_hub\n"
                "或者使用国内镜像:\n"
                "  pip install modelscope"
            )
        except Exception as e:
            raise RuntimeError(
                f"模型下载失败（ModelScope 和 HuggingFace 均失败）。\n"
                f"HuggingFace 错误: {e}\n"
                f"\n"
                f"请尝试以下方法之一：\n"
                f"  1. 使用 ModelScope: pip install modelscope 后重试\n"
                f"  2. 设置 HuggingFace 镜像: 设置环境变量 HF_ENDPOINT=https://hf-mirror.com\n"
                f"  3. 手动下载模型并设置: RAG_MODEL_PATH=/path/to/model\n"
                f"     下载地址: https://modelscope.cn/models/BAAI/bge-small-zh-v1.5"
            )

    # 3. 加载模型
    print(f"[RAG] 正在加载嵌入模型: {RAG_EMBEDDING_MODEL} ...")
    try:
        model = SentenceTransformer(model_path)
        _embedding_model = _STEmbeddings(model)
        print("[RAG] 嵌入模型加载完成。")
        return _embedding_model
    except ImportError:
        raise RuntimeError(
            "未安装 sentence-transformers 库。请运行:\n"
            "  pip install sentence-transformers"
        )
    except Exception as e:
        raise RuntimeError(
            f"加载嵌入模型失败: {e}\n"
            f"模型路径: {model_path}"
        )


# ==================== 向量存储管理 ====================

_vector_stores: dict[str, Chroma] = {}  # 按集合名称缓存


def _get_vector_store(collection_name: str) -> Optional[Chroma]:
    """获取指定集合的 Chroma 向量存储实例。

    如果 Chroma DB 目录不存在或集合为空，返回 None。
    此函数不会自动创建集合——集合由 build_knowledge_base.py 构建。
    """
    if collection_name in _vector_stores:
        return _vector_stores[collection_name]

    # 检查持久化目录是否存在
    if not os.path.exists(RAG_PERSIST_DIR):
        return None

    try:
        embeddings = _get_embedding_model()
        store = Chroma(
            persist_directory=RAG_PERSIST_DIR,
            collection_name=collection_name,
            embedding_function=embeddings,
        )
        # 检查集合是否有数据
        try:
            count = store._collection.count()
            if count == 0:
                return None
        except Exception:
            pass
        _vector_stores[collection_name] = store
        return store
    except Exception as e:
        print(f"[RAG] 加载向量存储失败 ({collection_name}): {e}")
        return None


# ==================== 核心检索函数 ====================

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_fixed(RETRY_DELAY),
    retry=retry_if_exception_type((OSError, ConnectionError, RuntimeError)),
    reraise=True,
)
def search_knowledge(
    query: str,
    collection: str = "dish_knowledge",
    top_k: int = RAG_TOP_K,
) -> str:
    """语义检索知识库，返回格式化文本结果。

    使用 BGE 模型将查询向量化，在 Chroma 集合中做相似度检索，
    将 top_k 条结果格式化为 LLM 可直接阅读的字符串。

    参数:
        query: 自然语言查询，如"川菜有什么特点"、"糖尿病患者能吃什么"
        collection: 知识集合名称，可选 dish_knowledge / dietary_knowledge / faq
        top_k: 返回条数，默认5

    返回:
        格式化的检索结果字符串。若无结果或知识库不可用，返回友好的中文提示。
    """
    # 1. 获取向量存储
    store = _get_vector_store(collection)
    if store is None:
        collection_names = {
            "dish_knowledge": "菜品知识库",
            "dietary_knowledge": "饮食健康知识库",
            "faq": "常见问题知识库",
        }
        cn_name = collection_names.get(collection, collection)
        return (
            f"知识库（{cn_name}）暂不可用，可能是因为还没有构建向量数据库。\n"
            f"请运行以下命令构建知识库:\n"
            f"  python build_knowledge_base.py\n"
            f"如果尚未下载嵌入模型，请先安装 modelscope:\n"
            f"  pip install modelscope"
        )

    # 2. 执行相似度检索
    try:
        docs = store.similarity_search_with_score(query, k=top_k)
    except Exception as e:
        print(f"[RAG] 检索异常: {e}")
        return "知识库检索时出现异常，请稍后再试。"

    # 3. 格式化结果
    if not docs:
        return "抱歉，没有在知识库中找到相关信息。您可以换个说法试试，或者告诉我更具体的需求。"

    source_labels = {
        "dish_knowledge": "菜品知识",
        "dietary_knowledge": "饮食知识",
        "faq": "常见问题",
    }
    source_label = source_labels.get(collection, collection)

    lines = [f"为您找到以下{source_label}（共{len(docs)}条）："]
    for i, (doc, score) in enumerate(docs, 1):
        # 相似度分数越小表示越相关（余弦距离），转换为易读标记
        if score < 0.3:
            relevance = "★★★ 高度相关"
        elif score < 0.6:
            relevance = "★★☆ 相关"
        else:
            relevance = "★☆☆ 低相关"

        metadata = doc.metadata or {}
        meta_hint = ""
        if collection == "dish_knowledge":
            name = metadata.get("name", "")
            category = metadata.get("category", "")
            if name:
                meta_hint = f"【{name}】"
            if category:
                meta_hint += f"[{category}] "
        elif collection == "dietary_knowledge":
            condition = metadata.get("condition", "")
            if condition:
                meta_hint = f"【{condition}】"
        elif collection == "faq":
            category = metadata.get("category", "")
            if category:
                meta_hint = f"【{category}】"

        lines.append(
            f"  {i}. {meta_hint}（{relevance}）\n"
            f"     {doc.page_content.strip()[:300]}"
        )

    return "\n".join(lines)
