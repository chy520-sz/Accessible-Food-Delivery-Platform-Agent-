"""
智能 Agent 核心模块 —— LangChain v1 create_agent + LangGraph checkpointer。

核心架构:
  - 使用 DeepSeek (deepseek-chat) 作为推理引擎（共享 LLM 单例）
  - LangChain v1 标准入口 create_agent；短期记忆保存在进程内 InMemorySaver，
    以 thread_id=session_id 绑定服务端会话（取代旧 AgentExecutor+ConversationBufferMemory）
  - 29 个工具函数供 LLM 调用（闭包绑定当前会话，底层调用 Java 后端 API）
  - 服务端会话归属校验：客户端 session_id 不是访问历史/已保存 JWT 的凭据，
    换账号会重新绑定并隔离旧历史；同一会话串行执行，避免状态交叉
  - JWT 只存服务端认证模块，绝不进入模型消息与 checkpoint
"""

import asyncio
import hashlib
import json
import logging
import time
import traceback
import uuid
from typing import Optional

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

import backend_client as bc
from config import (
    AGENT_MAX_TOOL_CALLS,
    AGENT_MAX_MODEL_CALLS,
    AGENT_MESSAGE_WINDOW,
    AGENT_RECURSION_LIMIT,
    AGENT_TOTAL_TIMEOUT_SECONDS,
    JAVA_BASE_URL,
    LLM_HEALTH_TTL,
    LLM_MODEL,
    SESSION_EXPIRE_SECONDS,
)
from llm_client import (
    aclose_llm,
    get_llm,
    llm_healthy,
    mark_llm_failure,
    mark_llm_success,
    probe_llm,
)
from prompts import (
    DELIVERY_AGENT_PROMPT,
    MERCHANT_AGENT_PROMPT,
    XIAOLU_SYSTEM_PROMPT,
)
from text_utils import clean_text_for_tts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent")

# 导入底层工具函数
from tools import (
    login as _login,
    search_dishes as _search_dishes,
    get_dish_detail as _get_dish_detail,
    recommend_dishes as _recommend_dishes,
    rate_item as _rate_item,
    get_my_rating_history as _get_my_rating_history,
    get_cart as _get_cart,
    add_to_cart as _add_to_cart,
    clear_cart as _clear_cart,
    get_user_orders as _get_user_orders,
    get_user_addresses as _get_user_addresses,
    place_order as _place_order,
    get_shop_status as _get_shop_status,
    estimate_dish_nutrition as _estimate_dish_nutrition,
    agent_accept_order as _agent_accept_order,
    agent_start_delivery as _agent_start_delivery,
    agent_complete_order as _agent_complete_order,
    agent_get_order_status as _agent_get_order_status,
    set_session,
    get_session,
)
from health_tools import (
    save_health_profile as _save_health_profile,
    get_health_profile as _get_health_profile,
    record_weight as _record_weight,
    get_weight_history as _get_weight_history,
    save_diet_plan as _save_diet_plan,
    get_active_diet_plan as _get_active_diet_plan,
    get_diet_plan_history as _get_diet_plan_history,
    fetch_dishes_for_planning as _fetch_dishes_for_planning,
    fetch_dietary_rules_for_goal as _fetch_dietary_rules_for_goal,
)
from health_validator import validate_health_profile
from knowledge_base import KnowledgeStoreError, search_knowledge
from config import RAG_COLLECTION_DIETARY, RAG_COLLECTION_FAQ, RAG_COLLECTION_FOOD, RAG_TOP_K


# ==================== 进程内 checkpointer ====================
# 单进程部署：所有会话的短期记忆保存在这里，按 thread_id 隔离。
_checkpointer = InMemorySaver()


# ==================== 下单确认闸门（服务端待确认状态） ====================

class OrderConfirmationGate:
    """两阶段下单：第一次调用只登记“待确认”，用户明确确认后的第二次相同调用才提交。

    幂等键由 用户|地址|备注 稳定派生，同一确认动作重复提交复用同一键，
    并直接返回首次结果，避免重复下单。待确认记录带 TTL，超时自动失效。
    """

    TTL_SECONDS = 300

    def __init__(self) -> None:
        self._pending: dict[str, dict] = {}

    @staticmethod
    def _idempotency_key(user_id, address_id, remark) -> str:
        raw = f"{user_id}|{address_id}|{(remark or '').strip()}"
        return "agent-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

    def _purge(self, now: float) -> None:
        expired = [k for k, v in self._pending.items() if now - v["ts"] > self.TTL_SECONDS]
        for k in expired:
            self._pending.pop(k, None)

    def evaluate(self, session_id: str, address_id: int, remark: str) -> tuple[str, str]:
        """返回 (幂等键, 阶段)，阶段为 NEED_CONFIRM / CONFIRMED / ALREADY_DONE。"""
        now = time.time()
        self._purge(now)
        sess = get_session(session_id)
        uid = sess.get("user_id", 0) if sess else 0
        key = self._idempotency_key(uid, address_id, remark)
        rec = self._pending.get(key)
        if rec is None:
            self._pending[key] = {"ts": now, "state": "awaiting"}
            return key, "NEED_CONFIRM"
        if rec.get("result") is not None:
            return key, "ALREADY_DONE"
        return key, "CONFIRMED"

    def store_result(self, key: str, result: str) -> None:
        self._pending[key] = {"ts": time.time(), "state": "done", "result": result}

    def get_result(self, key: str) -> str:
        return self._pending.get(key, {}).get("result", "")

    def clear(self) -> None:
        self._pending.clear()


# ==================== 会话管理 ====================

class AgentSession:
    """用户会话：绑定 checkpointer 线程、登录归属、串行锁、下单待确认状态。"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.last_active = time.time()
        self.agent = None  # 懒构建的 create_agent 编译图
        self.logged_in = False
        self.username = ""
        self.owner_user_id: Optional[object] = None
        self.order_gate = OrderConfirmationGate()
        self._lock: Optional[asyncio.Lock] = None

    def lock(self) -> asyncio.Lock:
        # 同一会话串行执行，避免连续两条消息导致状态交叉/重复购物车操作
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def reset_run_counters(self) -> None:
        """保留会话重置入口，调用预算由 LangChain 中间件按轮管理。"""

    def config(self) -> dict:
        return {
            "configurable": {"thread_id": self.session_id},
            # 注意：recursion_limit 是图步数上限，不等价旧 max_iterations
            "recursion_limit": AGENT_RECURSION_LIMIT,
        }

    def ensure_agent(self):
        if self.agent is None:
            self.agent = _build_agent(self)
        return self.agent


_sessions: dict[str, AgentSession] = {}


def get_or_create_session(session_id: Optional[str] = None) -> AgentSession:
    if session_id and session_id in _sessions:
        sess = _sessions[session_id]
        sess.last_active = time.time()
        # JWT 过期则标记未登录（真正的重新校验在 reconcile_auth 完成）
        if not get_session(session_id):
            sess.logged_in = False
        return sess
    new_id = session_id or uuid.uuid4().hex[:12]
    sess = AgentSession(new_id)
    _sessions[new_id] = sess
    return sess


def _delete_session(session_id: str) -> None:
    """统一清理：会话对象、JWT、checkpoint 线程、待确认订单、锁。"""
    sess = _sessions.pop(session_id, None)
    bc.clear_session(session_id)
    try:
        _checkpointer.delete_thread(session_id)
    except Exception:
        logger.debug("[session] 删除 checkpoint 线程 %s 异常", session_id, exc_info=True)
    if sess is not None:
        sess.order_gate.clear()
        sess.agent = None
        sess._lock = None


def delete_session(session_id: str) -> bool:
    """对外统一会话清理入口，返回是否原本存在。"""
    existed = session_id in _sessions
    _delete_session(session_id)
    return existed


def cleanup_sessions(now: Optional[float] = None) -> int:
    """清理过期会话及其 token / checkpoint / 待确认状态，返回清理数量。"""
    now = time.time() if now is None else now
    expired = [
        sid for sid, sess in _sessions.items()
        if now - sess.last_active > SESSION_EXPIRE_SECONDS
    ]
    for sid in expired:
        _delete_session(sid)
    bc.purge_expired_sessions(now)
    if expired:
        logger.info("[session_cleanup] 清理过期会话 %d 个", len(expired))
    return len(expired)


async def _clear_conversation(sess: AgentSession) -> None:
    """清空某会话的对话历史（换账号隔离），并重置待确认状态与计数。"""
    sess.order_gate.clear()
    sess.reset_run_counters()
    if sess.agent is None:
        return
    try:
        state = await sess.agent.aget_state(sess.config())
        msgs = state.values.get("messages", [])
        removes = [RemoveMessage(id=m.id) for m in msgs if getattr(m, "id", None)]
        if removes:
            await sess.agent.aupdate_state(sess.config(), {"messages": removes})
    except Exception:
        logger.debug("[session] 清空对话历史异常", exc_info=True)


async def _trim_history(sess: AgentSession) -> None:
    """长对话消息窗口：只在 HumanMessage 边界裁剪，保证工具调用/结果成对完整。"""
    agent = sess.ensure_agent()
    state = await agent.aget_state(sess.config())
    msgs = state.values.get("messages", [])
    if len(msgs) <= AGENT_MESSAGE_WINDOW:
        return
    human_idx = [i for i, m in enumerate(msgs) if getattr(m, "type", "") == "human"]
    cut = 0
    for hi in human_idx:
        if len(msgs) - hi <= AGENT_MESSAGE_WINDOW:
            cut = hi
            break
    if cut == 0:
        return
    removes = [RemoveMessage(id=m.id) for m in msgs[:cut] if getattr(m, "id", None)]
    if removes:
        await agent.aupdate_state(sess.config(), {"messages": removes})
        logger.info("[memory] session=%s 裁剪旧消息 %d 条", sess.session_id, len(removes))


# ==================== LangChain 工具定义 ====================

def _create_tools(sess: AgentSession):
    """创建绑定到指定会话的工具闭包（LLM 无需感知 session_id）。"""
    session_id = sess.session_id

    @tool
    def login(phone: str, password: str) -> str:
        """用户登录。仅在用户尚未登录或登录已过期时使用。如果用户已经在网页端登录过了，不需要再次调用此工具。
        参数 phone: 手机号（11位数字）
        参数 password: 登录密码
        """
        if sess.logged_in:
            return f"用户已登录，当前用户是 {sess.username}。无需重复登录，直接为ta提供服务即可。"
        result = _login(session_id, phone, password)
        if "登录成功" in result:
            sess.logged_in = True
            data = get_session(session_id)
            if data:
                sess.username = data.get("username", "")
                sess.owner_user_id = data.get("user_id")
        return result

    @tool
    def search_dishes(keyword: str = "", category_id: Optional[int] = None) -> str:
        """使用知识库语义检索和实时菜单联合搜索菜品。
        可以按关键词（如"宫保鸡丁""辣的""素食"）或分类ID查找菜品。
        工具内部始终先查询 Milvus 菜品知识库，再查询平台实时在售数据；
        最终推荐必须以实时在售结果为准。
        参数 keyword: 搜索关键词，可为空
        参数 category_id: 菜品分类ID数字，可为空
        """
        if keyword.strip():
            rag_query = keyword.strip()
        elif category_id is not None:
            rag_query = f"分类ID为{category_id}的菜品"
        else:
            rag_query = "平台有哪些菜品"
        knowledge = _rag(rag_query, RAG_COLLECTION_FOOD, "dish")
        realtime = _search_dishes(session_id, keyword, category_id)
        return (
            "【Milvus 菜品知识库语义检索】\n"
            f"{knowledge}\n\n"
            "【平台实时在售菜品】\n"
            f"{realtime}\n\n"
            "回答和推荐时以平台实时在售菜品为准，知识库内容用于理解食材、口味和语义关联。"
        )

    @tool
    def get_dish_detail(dish_id: int) -> str:
        """查看某个菜品的详细信息，包括价格、描述、月销量等。
        参数 dish_id: 菜品ID数字
        """
        return _get_dish_detail(session_id, dish_id)

    @tool
    def recommend_dishes(limit: int = 5) -> str:
        """根据用户历史评分偏好、菜品平均评分、评分数量、时间衰减和销量推荐菜品。
        参数 limit: 推荐数量，默认5
        """
        return _recommend_dishes(session_id, limit)

    @tool
    def rate_item(target_type: str, target_id: int, score: int, comment: str = "") -> str:
        """给菜品、套餐或店铺评分。target_type 必须是 DISH、COMBO 或 SHOP；score 为1到5星；comment 可为空。
        参数 target_type: DISH/COMBO/SHOP
        参数 target_id: 评分对象ID
        参数 score: 1到5星
        参数 comment: 可选评语
        """
        return _rate_item(session_id, target_type, target_id, score, comment)

    @tool
    def get_my_rating_history(limit: int = 10) -> str:
        """查看当前用户最近的评分历史，用于理解用户偏好。"""
        return _get_my_rating_history(session_id, limit)

    @tool
    def get_cart() -> str:
        """查看当前购物车里的所有菜品和总价。"""
        return _get_cart(session_id)

    @tool
    def add_to_cart(dish_id: int, quantity: int = 1) -> str:
        """把菜品加入购物车。
        参数 dish_id: 菜品ID数字
        参数 quantity: 数量，默认1
        """
        result = _add_to_cart(session_id, dish_id, quantity)
        sess.order_gate.clear()  # 购物车变化后旧的待确认订单失效
        return result

    @tool
    def clear_cart() -> str:
        """清空购物车里的所有菜品。"""
        sess.order_gate.clear()  # 购物车变化后旧的待确认订单失效
        return _clear_cart(session_id)

    @tool
    def get_user_orders() -> str:
        """查看用户的所有历史订单，包括订单编号、金额、状态。"""
        return _get_user_orders(session_id)

    @tool
    def get_user_addresses() -> str:
        """查看用户保存的所有收货地址。"""
        return _get_user_addresses(session_id)

    @tool
    def place_order(address_id: int, remark: str = "") -> str:
        """提交下单！系统会自动从购物车读取菜品计算总价，下单后购物车自动清空。
        下单前必须先调用 get_cart/get_user_addresses，完整复述订单内容并请用户明确确认。
        本工具采用两阶段确认：第一次调用只登记待确认（不会扣款下单），
        用户明确确认后再次以相同参数调用才会真正提交；重复提交会自动幂等去重。
        参数 address_id: 收货地址ID数字（先通过 get_user_addresses 获取）
        参数 remark: 订单备注，如"少辣""多加醋"等，可为空
        """
        key, phase = sess.order_gate.evaluate(session_id, address_id, remark)
        if phase == "NEED_CONFIRM":
            return (
                "订单尚未提交，当前处于待用户确认状态。请先向用户完整复述收货地址、备注以及"
                "通过 get_cart 得到的菜品与总价，明确询问是否确认下单；只有用户明确表示确认后，"
                "再用完全相同的 address_id 与 remark 调用一次 place_order 才会真正提交。"
            )
        if phase == "ALREADY_DONE":
            return sess.order_gate.get_result(key)
        result = _place_order(session_id, address_id, remark, idempotency_key=key)
        # 仅在确实下单成功时缓存结果用于幂等复用；失败允许用户修正后重试
        if "下单成功" in result:
            sess.order_gate.store_result(key, result)
        return result

    @tool
    def get_shop_status() -> str:
        """查询店铺当前的营业状态（是否在营业中）。无需登录即可查询。"""
        return _get_shop_status()

    @tool
    def estimate_dish_nutrition(dish_name: str, dish_desc: str = "") -> str:
        """估算菜品的营养热量。根据菜品名称估算千卡数，仅供参考。
        参数 dish_name: 菜品名称
        参数 dish_desc: 菜品描述（可选）
        """
        return _estimate_dish_nutrition(dish_name, dish_desc)

    @tool
    def simulate_multi_agent(order_summary: str) -> str:
        """【仅演示】模拟多智能体协同工作流程（导购Agent → 商家接单Agent → 配送Agent）。
        生成的是虚构演示数据（演示订单号/骑手/时间均为随机编造）。
        仅当用户明确要求观看"演示""展示"多智能体流程时使用；
        真实订单的进度必须使用 query_order_status 查询。
        参数 order_summary: 订单摘要文本
        """
        return _simulate_multi_agent_flow(order_summary)

    @tool
    def merchant_accept_order(order_id: int, user_id: int) -> str:
        """【真实桥接】商家接单Agent：确认订单并开始备餐，真实调用 Java 后端。
        参数 order_id: 订单ID
        参数 user_id: 用户ID
        """
        return _agent_accept_order(session_id, order_id, user_id)

    @tool
    def delivery_pickup_order(order_id: int, user_id: int) -> str:
        """【真实桥接】配送Agent：标记订单进入配送状态，真实调用 Java 后端。
        参数 order_id: 订单ID
        参数 user_id: 用户ID
        """
        return _agent_start_delivery(session_id, order_id, user_id)

    @tool
    def delivery_complete_order(order_id: int, user_id: int) -> str:
        """【真实桥接】配送Agent：订单已送达，真实调用 Java 后端。
        参数 order_id: 订单ID
        参数 user_id: 用户ID
        """
        return _agent_complete_order(session_id, order_id, user_id)

    @tool
    def query_order_status(order_id: int) -> str:
        """查询订单的真实配送状态（是否接单、骑手分配、预计送达、关键时间点）。
        用户询问"订单到哪了""什么时候送到""帮我查一下订单进度"时使用。
        参数 order_id: 订单ID
        """
        return _agent_get_order_status(session_id, order_id)

    # ==================== RAG 知识库检索工具 ====================

    def _rag(query: str, collection: str, logical: str) -> str:
        rag_id = uuid.uuid4().hex[:8]
        logger.info(
            "[RAG tool][%s] session=%s type=%s query=%r",
            rag_id, session_id, logical, " ".join(query.split())[:120],
        )
        try:
            result = search_knowledge(
                query,
                collection,
                top_k=RAG_TOP_K,
                logical_type=logical,
                trace_id=rag_id,
            )
            logger.info("[RAG tool][%s] RETURN chars=%d", rag_id, len(result))
            return result
        except KnowledgeStoreError as e:
            # 基础设施问题要显式暴露，绝不伪装成“没有知识”
            logger.warning("[RAG tool][%s] %s 检索失败: %s", rag_id, logical, e)
            return f"知识库当前不可用（{type(e).__name__}）。不要据此编造知识，可改用实时菜品查询或提示稍后再试。"

    @tool
    def search_food_knowledge(query: str) -> str:
        """搜索美食知识库。可以查询菜品的口味特点、烹饪方法、食材搭配、菜系文化等知识。
        当用户询问菜品本身的特点（如"宫保鸡丁是什么口味""川菜有什么特点""什么菜适合冬天吃"）
        或者需要了解菜系、食材、做法等美食文化知识时使用此工具。
        用户按食材组成筛选菜品（如“含有牛肉的菜品有哪些”“不含花生的菜”）时必须先调用本工具，
        再调用 search_dishes 核对平台当前在售状态。
        参数 query: 要查询的问题或关键词，如"不辣的低脂菜""粤菜清淡菜品""适合减肥的鸡肉做法"
        """
        return _rag(query, RAG_COLLECTION_FOOD, "dish")

    @tool
    def search_dietary_knowledge(query: str) -> str:
        """搜索饮食健康知识库。可以查询关于疾病饮食限制、营养建议、食物禁忌、过敏原等信息。
        当用户询问健康相关问题时使用，例如"糖尿病能吃什么""什么食物低脂""高血压饮食注意什么"
        "减肥应该怎么吃""痛风不能吃什么"等。这是用户健康饮食的权威参考来源。
        参数 query: 要查询的健康问题或关键词
        """
        return _rag(query, RAG_COLLECTION_DIETARY, "dietary")

    @tool
    def search_faq(query: str) -> str:
        """搜索常见问题知识库。可以查询关于外卖点餐流程、订单修改、配送时间、支付方式、
        账户管理等平台操作类常见问题。
        当用户询问操作性/流程性问题时使用，例如"如何修改订单""配送要多久""怎么退款"
        "如何添加地址""怎么使用语音点餐"等。
        参数 query: 要查询的问题或关键词
        """
        return _rag(query, RAG_COLLECTION_FAQ, "faq")

    # ==================== 健康管理工具 ====================

    @tool
    def record_health_profile(
        age: int, gender: str, height: float, weight: float,
        activity_level: str, diet_preference: str, health_goal: str,
    ) -> str:
        """录入或更新用户的健康档案。保存前先校验数值范围。
        参数 age: 年龄（10-100）
        参数 gender: 性别（男或女）
        参数 height: 身高（厘米，100-250）
        参数 weight: 体重（公斤，30-200）
        参数 activity_level: 活动水平（低/中/高）
        参数 diet_preference: 饮食偏好（均衡/低脂/素食/高蛋白/低碳水）
        参数 health_goal: 健康目标（减肥/增肌/维持）
        """
        result = validate_health_profile(
            age=age, gender=gender, height=height, weight=weight,
            activity_level=activity_level, diet_preference=diet_preference,
            health_goal=health_goal,
        )
        if not result.valid:
            return '{"action": "error", "message": "' + result.message + '"}'
        n = result.normalized
        return _save_health_profile(
            session_id, n["age"], n["gender"], n["height"], n["weight"],
            n["activity_level"], n["diet_preference"], n["health_goal"],
        )

    @tool
    def check_health_profile() -> str:
        """查看当前用户的健康档案。包含年龄、性别、身高、体重、活动水平、饮食偏好、健康目标。"""
        return _get_health_profile(session_id)

    @tool
    def record_weight(weight: float, record_date: str = "") -> str:
        """记录当前体重。
        参数 weight: 体重（公斤，30-200）
        参数 record_date: 日期（yyyy-MM-dd），不填默认今天
        """
        if weight < 30 or weight > 200:
            return '{"action": "error", "message": "体重需在30到200公斤之间"}'
        return _record_weight(session_id, weight, record_date)

    @tool
    def view_weight_history(weeks: int = 12) -> str:
        """查看体重历史记录。
        参数 weeks: 最近几周，默认12周
        """
        return _get_weight_history(session_id, weeks)

    @tool
    def generate_diet_plan() -> str:
        """根据健康档案和体重趋势，从本平台已有菜品库中选取真实菜品，生成一周个性化饮食计划，
        生成后自动保存到后端（如后端已实现健康接口）。
        生成前自动检查档案是否存在，并从数据库和知识库获取可用菜品及饮食规则。
        所有菜品均来自本平台，绝不编造不存在的食物。"""
        profile_str = _get_health_profile(session_id)
        try:
            p = json.loads(profile_str)
            if p.get("action") == "profile_not_found":
                return json.dumps(
                    {"action": "error", "message": "请先录入健康档案，对我说'录入健康档案'"},
                    ensure_ascii=False
                )
            if p.get("action") == "error":
                return profile_str
        except json.JSONDecodeError:
            return json.dumps({"action": "error", "message": "获取档案出错，请重试"}, ensure_ascii=False)

        weight_str = _get_weight_history(session_id, weeks=4)
        dishes_str = _fetch_dishes_for_planning(session_id)
        profile_data = p.get("profile", {})
        health_goal = profile_data.get("health_goal", "维持")
        dietary_rules_str = _fetch_dietary_rules_for_goal(health_goal)

        from diet_planner import generate_weekly_diet_plan
        plan_result = generate_weekly_diet_plan(
            profile_str=profile_str,
            weight_history_str=weight_str,
            dishes_str=dishes_str,
            dietary_rules_str=dietary_rules_str,
        )

        try:
            plan_obj = json.loads(plan_result)
            if isinstance(plan_obj, dict) and plan_obj.get("action") == "diet_plan" and plan_obj.get("plan"):
                saved = _save_diet_plan(
                    session_id,
                    json.dumps(plan_obj["plan"], ensure_ascii=False),
                )
                try:
                    saved_obj = json.loads(saved)
                    if saved_obj.get("action") == "diet_plan_saved":
                        plan_obj["message"] = f"{plan_obj.get('message', '')}（计划已保存）"
                    else:
                        plan_obj["message"] = (
                            f"{plan_obj.get('message', '')}（保存提示：{saved_obj.get('message', '')}）"
                        )
                except (json.JSONDecodeError, AttributeError):
                    logger.debug("[diet_plan] 保存回执解析失败", exc_info=True)
                plan_result = json.dumps(plan_obj, ensure_ascii=False)
        except json.JSONDecodeError:
            logger.debug("[diet_plan] 计划结果解析失败", exc_info=True)
        return plan_result

    @tool
    def view_diet_plan() -> str:
        """查看当前有效的周饮食计划。"""
        return _get_active_diet_plan(session_id)

    @tool
    def view_diet_plan_history(limit: int = 5) -> str:
        """查看饮食计划历史。
        参数 limit: 最近几条，默认5
        """
        return _get_diet_plan_history(session_id, limit)

    return [
        login, search_dishes, get_dish_detail, recommend_dishes, rate_item,
        get_my_rating_history, get_cart, add_to_cart, clear_cart,
        get_user_orders, get_user_addresses, place_order, get_shop_status,
        estimate_dish_nutrition, simulate_multi_agent, merchant_accept_order,
        delivery_pickup_order, delivery_complete_order, query_order_status,
        search_food_knowledge, search_dietary_knowledge, search_faq,
        record_health_profile, check_health_profile, record_weight,
        view_weight_history, generate_diet_plan, view_diet_plan,
        view_diet_plan_history,
    ]


# ==================== create_agent 构建 ====================

def _build_agent(sess: AgentSession):
    """为会话构建 LangChain v1 create_agent（共享 LLM 与进程内 checkpointer）。"""
    llm = get_llm()
    tools = _create_tools(sess)
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=XIAOLU_SYSTEM_PROMPT,
        checkpointer=_checkpointer,
        middleware=[
            ModelCallLimitMiddleware(
                run_limit=AGENT_MAX_MODEL_CALLS,
                exit_behavior="end",
            ),
            ToolCallLimitMiddleware(
                run_limit=AGENT_MAX_TOOL_CALLS,
                exit_behavior="continue",
            ),
        ],
    )


# ==================== 多智能体协同模拟（仅演示） ====================

def _simulate_multi_agent_flow(order_summary: str) -> str:
    """模拟三智能体协同流程：导购 → 商家接单 → 配送（虚构演示数据）。"""
    import random

    rider_names = ["赵明", "钱伟", "孙强", "李磊", "周洋"]
    rider_name = random.choice(rider_names)
    rider_phone = f"138{random.randint(10000000, 99999999)}"
    prep_time = random.randint(15, 30)
    delivery_time = random.randint(20, 40)
    order_no = f"ORD{uuid.uuid4().hex[:8].upper()}"

    lines = [
        "══════════════════════════",
        "🐾  多智能体协同流程演示（虚构数据）  🐾",
        "══════════════════════════",
        "",
        "【第一步】🛍️  导购Agent（小鹿）",
        f"  已完成订单确认：{order_summary}",
        "  ✓ 订单已提交至商家后台",
        "",
        "【第二步】👨‍🍳 商家接单Agent",
        f"  {MERCHANT_AGENT_PROMPT.strip().format(order_no=order_no, prep_time=prep_time)}",
        "",
        "【第三步】🛵 配送Agent",
        f"  {DELIVERY_AGENT_PROMPT.strip().format(delivery_time=delivery_time, rider_name=rider_name, rider_phone=rider_phone)}",
        "",
        "══════════════════════════",
        f"  演示订单 {order_no} 全流程展示完毕",
        "  如需查询真实订单进度，请对我说'查一下我的订单'",
        "══════════════════════════",
    ]
    return "\n".join(lines)


# ==================== 登录态同步 ====================

def _decode_jwt_payload(auth_token: str) -> Optional[dict]:
    """base64 解码 JWT payload（仅读取展示字段，签名由后端验证）。"""
    import base64
    try:
        payload_b64 = auth_token.split(".")[1]
        missing_padding = (4 - len(payload_b64) % 4) % 4
        payload_b64 += "=" * missing_padding
        return json.loads(base64.urlsafe_b64decode(payload_b64).decode("utf-8"))
    except Exception as e:
        logger.warning("[sync_login] JWT payload 解码失败: %s", type(e).__name__)
        return None


def _validate_and_store_token(session_id: str, auth_token: str) -> tuple[bool, str, object]:
    """后端真实校验 token 并写入会话存储，返回 (ok, username, user_id)。"""
    payload = _decode_jwt_payload(auth_token)
    if payload is None:
        return False, "", None
    if not bc.validate_user_token(auth_token):
        logger.warning("[sync_login] 会话 %s token 后端校验未通过", session_id)
        return False, "", None
    now = time.time()
    expires_at = now + SESSION_EXPIRE_SECONDS
    exp = payload.get("exp")
    if isinstance(exp, (int, float)) and exp > now:
        expires_at = min(expires_at, exp)
    user_id = payload.get("userId") or payload.get("sub", 0)
    username = payload.get("username", "用户")
    set_session(session_id, {
        "token": auth_token,
        "user_id": user_id,
        "username": username,
        "expires_at": expires_at,
    })
    return True, username, user_id


async def sync_login_token(session_id: str, auth_token: Optional[str]) -> tuple[bool, str]:
    """将前端登录态同步到 Agent 会话，处理首次登录、换号、退出三种场景。

    换号（token 归属用户变化）时清空旧对话历史，实现账号间历史隔离。
    """
    sess = _sessions.get(session_id)
    if not sess:
        logger.warning("[sync_login] 会话 %s 不存在", session_id)
        return False, ""

    if auth_token:
        stored = get_session(session_id)
        if stored and stored.get("token") == auth_token and sess.logged_in:
            logger.info("[sync_login] 会话 %s token 未变化，跳过", session_id)
            return sess.logged_in, sess.username

        ok, username, user_id = _validate_and_store_token(session_id, auth_token)
        if not ok:
            return False, ""
        switched = sess.logged_in and sess.owner_user_id not in (None, user_id)
        sess.logged_in = True
        sess.username = username
        sess.owner_user_id = user_id
        if switched:
            # 换账号：隔离旧历史与旧待确认订单
            await _clear_conversation(sess)
            logger.info("[sync_login] 会话 %s 换号，已隔离旧历史 user=%s", session_id, username)
        logger.info("[sync_login] 会话 %s 登录成功 username=%s", session_id, username)
        return True, username

    # 退出登录：清除登录态、JWT、对话历史、待确认状态
    logger.info("[sync_login] 会话 %s 清除登录态", session_id)
    bc.clear_session(session_id)
    sess.logged_in = False
    sess.username = ""
    sess.owner_user_id = None
    await _clear_conversation(sess)
    return False, ""


async def _reconcile_auth(sess: AgentSession, auth_token: Optional[str]) -> None:
    """每轮对话都重新核对传入 token（修复旧实现只在未登录时同步的问题）。"""
    stored = get_session(sess.session_id)
    if auth_token:
        if stored and stored.get("token") == auth_token and sess.logged_in:
            sess.logged_in = True
            return
        ok, username, user_id = _validate_and_store_token(sess.session_id, auth_token)
        if ok:
            switched = sess.logged_in and sess.owner_user_id not in (None, user_id)
            sess.logged_in = True
            sess.username = username
            sess.owner_user_id = user_id
            if switched:
                await _clear_conversation(sess)
                logger.info("[chat] 会话 %s 检测到换号，已隔离旧历史", sess.session_id)
        else:
            # 传入了无效 token：不得沿用旧登录态
            sess.logged_in = False
    elif stored is None:
        # 本轮没带 token 且服务端也没有有效 JWT
        sess.logged_in = False


# ==================== 对话接口 ====================

def _extract_final_text(result: dict) -> str:
    """从 create_agent 返回状态中提取最后一条 AI 消息文本（兼容字符串/内容块）。"""
    messages = result.get("messages", [])
    if not messages:
        return ""
    last = messages[-1]
    content = getattr(last, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # 多模态/内容块：拼接其中的文本块
        parts = [
            blk.get("text", "") if isinstance(blk, dict) else str(blk)
            for blk in content
        ]
        return "".join(p for p in parts if p)
    return str(content)


def _friendly_error(trace_id: str, error_msg: str) -> str:
    error_lower = error_msg.lower()
    if "timeout" in error_lower or "timed out" in error_lower:
        return f"网络有点慢呢，大模型响应超时了，请稍等片刻再试试～\n（追踪ID: {trace_id}）"
    if "rate" in error_lower and ("limit" in error_lower or "exceeded" in error_lower):
        return f"当前访问人数较多，大模型服务繁忙，请稍后再试～\n（追踪ID: {trace_id}）"
    if any(k in error_lower for k in ("api key", "auth", "unauthorized", "401")):
        return f"大模型服务认证失败，请联系管理员检查 API Key 配置。\n（追踪ID: {trace_id}）"
    if "content" in error_lower and any(k in error_lower for k in ("filter", "safety", "moderation")):
        return f"您输入的内容包含了不被允许的敏感信息，请修改后重试。\n（追踪ID: {trace_id}）"
    if any(k in error_lower for k in ("service unavailable", "503", "overloaded")):
        return f"大模型服务暂时不可用，请稍等 1-2 分钟后再试。\n（追踪ID: {trace_id}）"
    if any(k in error_lower for k in ("connect", "network", "dns")):
        return f"连接大模型服务失败，请检查网络连接后重试。\n（追踪ID: {trace_id}）"
    return f"小鹿遇到了一点小问题，请您稍后再试试好吗？\n（追踪ID: {trace_id}）"


async def chat(session_id: Optional[str], user_input: str, auth_token: Optional[str] = None) -> dict:
    """处理用户文本对话，返回 session_id/reply/tts_text/is_new_session/trace_id。"""
    trace_id = uuid.uuid4().hex[:12]
    is_new = session_id is None
    sess = None
    reply = ""

    try:
        sess = get_or_create_session(session_id)
        async with sess.lock():  # 同一会话串行
            sess.last_active = time.time()
            await _reconcile_auth(sess, auth_token)

            logger.info(
                "[chat] trace=%s session=%s logged_in=%s username=%s input=%s",
                trace_id, sess.session_id, sess.logged_in, sess.username, user_input[:80]
            )

            if not sess.logged_in:
                return {
                    "session_id": sess.session_id,
                    "reply": "您还没有登录哦～请先登录后再使用点餐功能。您可以在登录页面输入手机号和密码完成登录，小鹿会一直在这里等您～",
                    "tts_text": "您还没有登录哦，请先登录后再使用点餐功能。您可以在登录页面输入手机号和密码完成登录，小鹿会一直在这里等您。",
                    "is_new_session": is_new,
                    "trace_id": trace_id,
                }

            sess.reset_run_counters()
            agent = sess.ensure_agent()
            cfg = sess.config()
            await _trim_history(sess)

            result = await asyncio.wait_for(
                agent.ainvoke({"messages": [HumanMessage(content=user_input)]}, cfg),
                timeout=AGENT_TOTAL_TIMEOUT_SECONDS,
            )
            reply = _extract_final_text(result)
            if not reply or not reply.strip():
                reply = "抱歉呀，我刚才没有理解您的意思。能换个说法再告诉我一遍吗？小鹿在认真听呢～"
            mark_llm_success()

    except asyncio.TimeoutError:
        mark_llm_failure()
        logger.error("[chat] trace=%s 总耗时超过 %.0fs", trace_id, AGENT_TOTAL_TIMEOUT_SECONDS)
        reply = f"这一轮处理时间过长已被中断，请简化需求或稍后再试～\n（追踪ID: {trace_id}）"
    except Exception as e:
        mark_llm_failure()
        logger.error(
            "[chat] trace=%s session=%s ERROR type=%s msg=%s\n%s",
            trace_id, session_id or "(new)", type(e).__name__, str(e), traceback.format_exc(),
        )
        reply = _friendly_error(trace_id, str(e))

    tts_text = clean_text_for_tts(reply)
    result_session_id = sess.session_id if sess else (session_id or "")
    return {
        "session_id": result_session_id,
        "reply": reply,
        "tts_text": tts_text,
        "is_new_session": is_new,
        "trace_id": trace_id,
    }


# ==================== 健康状态 ====================

_java_probe = {"ts": 0.0, "ok": False}
_JAVA_PROBE_TTL = 30.0


async def _probe_java_backend() -> bool:
    now = time.time()
    if now - _java_probe["ts"] < _JAVA_PROBE_TTL:
        return _java_probe["ok"]
    try:
        bc.get_public("/api/shop/status", timeout=3.0)
        _java_probe.update(ts=now, ok=True)
    except Exception as e:
        logger.debug("[health] Java 探测失败: %s", type(e).__name__)
        _java_probe.update(ts=now, ok=False)
    return _java_probe["ok"]


async def get_agent_status() -> dict:
    llm_alive = llm_healthy(LLM_HEALTH_TTL)
    if not llm_alive:
        llm_alive = await probe_llm(ttl=LLM_HEALTH_TTL)
    java_ok = await _probe_java_backend()

    # RAG（Milvus + 集合）状态；探测失败不影响整体服务存活
    rag = {"milvus_up": False, "ready": False, "collections": {}}
    try:
        from knowledge_base import rag_status
        rag = rag_status()
    except Exception as e:
        logger.debug("[health] RAG 状态获取失败: %s", type(e).__name__)

    # 嵌入探测缓存状态
    emb = {"checked": False, "ok": False}
    try:
        from embedding_client import get_embeddings
        embeddings = get_embeddings()
        emb = embeddings.health(LLM_HEALTH_TTL)
        if not emb.get("ok"):
            # 固定探测文本会命中磁盘缓存；首次启动仍能验证配置与返回维度，
            # 后续健康检查不会反复消耗云端额度。
            await embeddings.aembed_query("外卖知识检索连通性测试")
            emb = embeddings.health(LLM_HEALTH_TTL)
    except Exception:
        logger.debug("[health] 嵌入探测失败", exc_info=True)

    overall = "ok" if llm_alive else "degraded"
    return {
        "status": overall,
        "model": LLM_MODEL,
        "active_sessions": len(_sessions),
        "llm_connected": llm_alive,
        "java_backend": "up" if java_ok else "down",
        "java_base_url": JAVA_BASE_URL,
        "milvus_up": rag.get("milvus_up", False),
        "rag_ready": rag.get("ready", False),
        "rag_collections": rag.get("collections", {}),
        "embedding_ok": emb.get("ok", False),
    }


async def shutdown() -> None:
    """释放全局资源（服务关闭时调用）。"""
    await aclose_llm()
    bc.close_client()
    try:
        from embedding_client import aclose_embeddings
        await aclose_embeddings()
    except Exception:
        logger.debug("[shutdown] 嵌入客户端关闭异常", exc_info=True)
    try:
        from knowledge_base import close_milvus_client
        close_milvus_client()
    except Exception:
        logger.debug("[shutdown] Milvus 客户端关闭异常", exc_info=True)
