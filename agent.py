"""
智能 Agent 核心模块 —— LangChain AgentExecutor + ConversationBufferMemory。

核心架构:
  - 使用 DeepSeek (deepseek-chat) 作为推理引擎（共享 LLM 单例）
  - 29 个工具函数供 LLM 调用（底层调用 Java 后端 API）
  - 会话级记忆管理，每个用户独立对话历史，定期清理过期会话
  - 多智能体协同：导购 → 商家接单 → 配送，桥接工具真实调用 Java 后端
"""
import json
import logging
import time
import traceback
import uuid
from typing import Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

import backend_client as bc
from config import (
    JAVA_BASE_URL,
    LLM_HEALTH_TTL,
    LLM_MODEL,
    SESSION_EXPIRE_SECONDS,
)
from llm_client import (
    close_llm,
    get_llm,
    llm_healthy,
    mark_llm_failure,
    mark_llm_success,
    probe_llm,
)
from text_utils import clean_text_for_tts

# 修复[错误分级]：配置结构化日志，记录完整上下文用于排障
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent")
from prompts import (
    XIAOLU_SYSTEM_PROMPT,
    MERCHANT_AGENT_PROMPT,
    DELIVERY_AGENT_PROMPT,
)

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
# 健康管理工具（依赖 Java 后端 /api/user/health/* 接口，调用失败时有容错处理）
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


# ==================== 会话管理 ====================

class AgentSession:
    """用户会话，包含 LangChain Memory + JWT Token。

    每个用户（按 session_id 区分）拥有独立的:
      - ConversationBufferMemory: 对话历史
      - AgentExecutor: 绑定了该会话的 Agent 实例
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.last_active = time.time()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
        )
        self.agent_executor = _create_agent_executor(session_id, self.memory)
        # 标记用户是否已登录
        self.logged_in = False
        self.username = ""


# 全局会话池: session_id -> AgentSession
_sessions: dict[str, AgentSession] = {}


def get_or_create_session(session_id: Optional[str] = None) -> AgentSession:
    """获取或创建用户会话。

    参数:
        session_id: 会话ID，为 None 时自动创建新会话

    返回:
        AgentSession 实例
    """
    if session_id and session_id in _sessions:
        sess = _sessions[session_id]
        sess.last_active = time.time()
        # 检查 JWT 是否过期
        if get_session(session_id) or not sess.logged_in:
            return sess
        # Token 已过期，标记为未登录
        sess.logged_in = False
        return sess
    # 创建新会话
    new_id = session_id or uuid.uuid4().hex[:12]
    sess = AgentSession(new_id)
    _sessions[new_id] = sess
    return sess


def cleanup_sessions(now: Optional[float] = None) -> int:
    """清理超过 SESSION_EXPIRE_SECONDS 未活动的会话及其 token，返回清理数量。"""
    now = time.time() if now is None else now
    expired = [
        sid for sid, sess in _sessions.items()
        if now - sess.last_active > SESSION_EXPIRE_SECONDS
    ]
    for sid in expired:
        sess = _sessions.pop(sid, None)
        if sess is not None:
            sess.agent_executor = None  # 释放 executor/memory 引用
    bc.purge_expired_sessions(now)
    if expired:
        logger.info("[session_cleanup] 清理过期会话 %d 个", len(expired))
    return len(expired)


# ==================== LangChain 工具定义 ====================

def _create_tools(session_id: str):
    """创建绑定到指定会话的工具列表。

    每个工具函数内部闭包捕获 session_id，
    使得 LLM 不需要感知 session_id 参数。

    参数:
        session_id: 当前用户会话ID

    返回:
        LangChain Tool 列表
    """

    @tool
    def login(phone: str, password: str) -> str:
        """用户登录。仅在用户尚未登录或登录已过期时使用。如果用户已经在网页端登录过了，不需要再次调用此工具。
        参数 phone: 手机号（11位数字）
        参数 password: 登录密码
        """
        # 检查会话是否已经通过 JWT 同步登录
        sess = _sessions.get(session_id)
        if sess and sess.logged_in:
            return f"用户已登录，当前用户是 {sess.username}。无需重复登录，直接为ta提供服务即可。"
        result = _login(session_id, phone, password)
        # 登录成功后更新会话状态
        if "登录成功" in result:
            if sess:
                sess.logged_in = True
                sess_data = get_session(session_id)
                if sess_data:
                    sess.username = sess_data.get("username", "")
        return result

    @tool
    def search_dishes(keyword: str = "", category_id: Optional[int] = None) -> str:
        """搜索菜品。可以按关键词（如"宫保鸡丁""辣的""素食"）或分类ID查找菜品。
        参数 keyword: 搜索关键词，可为空
        参数 category_id: 菜品分类ID数字，可为空
        """
        return _search_dishes(session_id, keyword, category_id)

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
        return _add_to_cart(session_id, dish_id, quantity)

    @tool
    def clear_cart() -> str:
        """清空购物车里的所有菜品。"""
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
        下单前必须完整复述订单内容并请用户确认！
        参数 address_id: 收货地址ID数字（先通过 get_user_addresses 获取）
        参数 remark: 订单备注，如"少辣""多加醋"等，可为空
        """
        return _place_order(session_id, address_id, remark)

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

    @tool
    def search_food_knowledge(query: str) -> str:
        """搜索美食知识库。可以查询菜品的口味特点、烹饪方法、食材搭配、菜系文化等知识。
        当用户询问菜品本身的特点（如"宫保鸡丁是什么口味""川菜有什么特点""什么菜适合冬天吃"）
        或者需要了解菜系、食材、做法等美食文化知识时使用此工具。
        参数 query: 要查询的问题或关键词，如"不辣的低脂菜""粤菜清淡菜品""适合减肥的鸡肉做法"
        """
        from knowledge_base import search_knowledge
        from config import RAG_COLLECTION_FOOD, RAG_TOP_K
        return search_knowledge(query, RAG_COLLECTION_FOOD, top_k=RAG_TOP_K)

    @tool
    def search_dietary_knowledge(query: str) -> str:
        """搜索饮食健康知识库。可以查询关于疾病饮食限制、营养建议、食物禁忌、过敏原等信息。
        当用户询问健康相关问题时使用，例如"糖尿病能吃什么""什么食物低脂""高血压饮食注意什么"
        "减肥应该怎么吃""痛风不能吃什么"等。这是用户健康饮食的权威参考来源。
        参数 query: 要查询的健康问题或关键词
        """
        from knowledge_base import search_knowledge
        from config import RAG_COLLECTION_DIETARY, RAG_TOP_K
        return search_knowledge(query, RAG_COLLECTION_DIETARY, top_k=RAG_TOP_K)

    @tool
    def search_faq(query: str) -> str:
        """搜索常见问题知识库。可以查询关于外卖点餐流程、订单修改、配送时间、支付方式、
        账户管理等平台操作类常见问题。
        当用户询问操作性/流程性问题时使用，例如"如何修改订单""配送要多久""怎么退款"
        "如何添加地址""怎么使用语音点餐"等。
        参数 query: 要查询的问题或关键词
        """
        from knowledge_base import search_knowledge
        from config import RAG_COLLECTION_FAQ, RAG_TOP_K
        return search_knowledge(query, RAG_COLLECTION_FAQ, top_k=RAG_TOP_K)

    # ==================== 健康管理工具 ====================
    # ⚠️ 注意：以下工具依赖 Java 后端 /api/user/health/* 接口。
    # 如果后端健康管理 API 尚未实现，工具调用会返回友好的错误提示，
    # 不会影响点餐核心流程。

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
        """记录用户当前体重。
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

        # 获取体重历史（最近4周）
        weight_str = _get_weight_history(session_id, weeks=4)

        # 获取本平台已有菜品（优先 Java 后端，回退知识库 JSON）
        dishes_str = _fetch_dishes_for_planning(session_id)

        # 获取饮食规则（从知识库中匹配健康目标）
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

        # 生成成功后自动保存到后端（后端未实现时不影响返回计划内容）
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
                    pass
                plan_result = json.dumps(plan_obj, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
        return plan_result

    @tool
    def view_diet_plan() -> str:
        """查看当前有效的周饮食计划。"""
        return _get_active_diet_plan(session_id)

    @tool
    def view_diet_plan_history(limit: int = 5) -> str:
        """查看饮食计划历史。
        参数 limit: 最近几条，默认5条
        """
        return _get_diet_plan_history(session_id, limit)

    return [
        login,
        search_dishes,
        get_dish_detail,
        recommend_dishes,
        rate_item,
        get_my_rating_history,
        get_cart,
        add_to_cart,
        clear_cart,
        get_user_orders,
        get_user_addresses,
        place_order,
        get_shop_status,
        estimate_dish_nutrition,
        simulate_multi_agent,
        merchant_accept_order,
        delivery_pickup_order,
        delivery_complete_order,
        query_order_status,
        search_food_knowledge,       # RAG: 美食知识库检索
        search_dietary_knowledge,    # RAG: 饮食健康知识库检索
        search_faq,                  # RAG: 常见问题知识库检索
        record_health_profile,       # 健康: 录入档案
        check_health_profile,        # 健康: 查看档案
        record_weight,               # 健康: 记录体重
        view_weight_history,         # 健康: 体重历史
        generate_diet_plan,          # 健康: AI生成饮食计划
        view_diet_plan,              # 健康: 查看饮食计划
        view_diet_plan_history,      # 健康: 饮食计划历史
    ]


# ==================== AgentExecutor 创建 ====================

def _create_agent_executor(
    session_id: str, memory: ConversationBufferMemory
) -> AgentExecutor:
    """创建绑定到指定会话的 AgentExecutor。

    参数:
        session_id: 会话ID
        memory: 对话记忆实例

    返回:
        配置好的 AgentExecutor
    """
    llm = get_llm()
    tools = _create_tools(session_id)

    # 构造 ChatPromptTemplate —— create_tool_calling_agent 要求的标准格式
    prompt = ChatPromptTemplate.from_messages([
        ("system", XIAOLU_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 使用 create_tool_calling_agent 创建支持 function calling 的 agent
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    # 创建 AgentExecutor
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=10,          # 最多调用 10 次工具
        early_stopping_method="generate",
        return_intermediate_steps=False,  # 生产环境不返回中间步骤
    )

    return executor


# ==================== 多智能体协同模拟（仅演示） ====================

def _simulate_multi_agent_flow(order_summary: str) -> str:
    """模拟三智能体协同流程：导购 → 商家接单 → 配送（虚构演示数据）。

    此函数仅用于演示多智能体协作概念，返回的订单号/骑手/时间均为随机编造。
    真实订单的进度必须通过 query_order_status 工具查询 Java 后端。

    参数:
        order_summary: 导购Agent确认后的订单摘要

    返回:
        完整的协同流程文本
    """
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
    """base64 解码 JWT payload（仅用于读取展示字段，签名由后端验证）。"""
    import base64
    try:
        payload_b64 = auth_token.split(".")[1]
        missing_padding = (4 - len(payload_b64) % 4) % 4
        payload_b64 += "=" * missing_padding
        return json.loads(base64.urlsafe_b64decode(payload_b64).decode("utf-8"))
    except Exception as e:
        logger.warning("[sync_login] JWT payload 解码失败: %s", type(e).__name__)
        return None


def sync_login_token(session_id: str, auth_token: Optional[str]) -> tuple[bool, str]:
    """将前端登录态同步到 Agent 会话（免密码复用登录态）。

    处理三种场景：首次登录、换号、退出登录。
    安全说明：token 的签名/过期/黑名单状态由 Java 后端真实校验
    （validate_user_token），Agent 不信任客户端直接传来的 payload。

    参数:
        session_id: 会话ID
        auth_token: 前端传递的 JWT token（None 或空字符串表示退出登录）

    返回:
        (是否已登录, 用户名)
    """
    sess = _sessions.get(session_id)
    if not sess:
        logger.warning("[sync_login] 会话 %s 不存在", session_id)
        return False, ""

    stored = get_session(session_id)

    if auth_token:
        if not stored or stored.get("token") != auth_token:
            payload = _decode_jwt_payload(auth_token)
            if payload is None:
                return False, ""
            # 后端真实校验（验签 + 过期 + 黑名单）
            if not bc.validate_user_token(auth_token):
                logger.warning("[sync_login] 会话 %s token 后端校验未通过", session_id)
                return False, ""
            now = time.time()
            expires_at = now + SESSION_EXPIRE_SECONDS
            exp = payload.get("exp")
            if isinstance(exp, (int, float)) and exp > now:
                expires_at = min(expires_at, exp)
            set_session(session_id, {
                "token": auth_token,
                "user_id": payload.get("userId") or payload.get("sub", 0),
                "username": payload.get("username", "用户"),
                "expires_at": expires_at,
            })
            sess.logged_in = True
            sess.username = payload.get("username", "用户")
            logger.info(
                "[sync_login] 会话 %s 登录成功: username=%s",
                session_id, sess.username,
            )
        else:
            logger.info("[sync_login] 会话 %s token未变化，跳过", session_id)
        return sess.logged_in, sess.username
    else:
        # 前端无 token（用户退出登录）：清除 Agent 会话中的登录态
        logger.info("[sync_login] 会话 %s 清除登录态", session_id)
        if stored:
            set_session(session_id, {
                "token": "", "user_id": 0, "username": "",
                "expires_at": 0,
            })
            sess.logged_in = False
            sess.username = ""
        return False, ""


# ==================== 对话接口 ====================

async def chat(session_id: Optional[str], user_input: str, auth_token: Optional[str] = None) -> dict:
    """处理用户的文本对话请求。

    参数:
        session_id: 会话ID，None 时创建新会话
        user_input: 用户输入的文本
        auth_token: 前端传递的 JWT token（可选，用于免密复用登录态）

    返回:
        { "session_id": str, "reply": str, "is_new_session": bool }
    """
    # 每次请求生成唯一 trace_id，贯穿日志和错误响应
    trace_id = uuid.uuid4().hex[:12]
    is_new = session_id is None
    sess = None

    try:
        sess = get_or_create_session(session_id)
        sess.last_active = time.time()

        # 同步前端登录态到 Agent 会话
        if auth_token and not sess.logged_in:
            sync_login_token(sess.session_id, auth_token)

        logger.info(
            "[chat] trace=%s session=%s logged_in=%s username=%s input=%s",
            trace_id, sess.session_id, sess.logged_in, sess.username, user_input[:80]
        )

        # 同步后仍未登录 → 提示用户先登录，不让 LLM 乱调工具
        if not sess.logged_in:
            return {
                "session_id": sess.session_id,
                "reply": "您还没有登录哦～请先登录后再使用点餐功能。您可以在登录页面输入手机号和密码完成登录，小鹿会一直在这里等您～",
                "tts_text": "您还没有登录哦，请先登录后再使用点餐功能。您可以在登录页面输入手机号和密码完成登录，小鹿会一直在这里等您。",
                "is_new_session": is_new,
                "trace_id": trace_id,
            }

        result = await sess.agent_executor.ainvoke({"input": user_input})
        reply = result.get("output", "")
        if not reply or not reply.strip():
            reply = "抱歉呀，我刚才没有理解您的意思。能换个说法再告诉我一遍吗？小鹿在认真听呢～"
        mark_llm_success()

    except Exception as e:
        # 记录完整 traceback，按错误类型分级返回用户友好的提示
        error_msg = str(e)
        error_type = type(e).__name__
        mark_llm_failure()
        logger.error(
            "[chat] trace=%s session=%s ERROR type=%s msg=%s\n%s",
            trace_id,
            session_id or "(new)",
            error_type,
            error_msg,
            traceback.format_exc(),
        )

        # 分类识别大模型 API 的错误类型，给出精确定向提示
        error_lower = error_msg.lower()
        if "timeout" in error_lower or "timed out" in error_lower:
            reply = (
                f"网络有点慢呢，大模型响应超时了，请稍等片刻再试试～"
                f"\n（追踪ID: {trace_id}）"
            )
        elif "rate" in error_lower and ("limit" in error_lower or "exceeded" in error_lower):
            reply = (
                f"当前访问人数较多，大模型服务繁忙，请稍后再试～"
                f"\n（追踪ID: {trace_id}）"
            )
        elif "api key" in error_lower or "auth" in error_lower or "unauthorized" in error_lower or "401" in error_lower:
            reply = (
                f"大模型服务认证失败，请联系管理员检查 API Key 配置。"
                f"\n（追踪ID: {trace_id}）"
            )
        elif "content" in error_lower and ("filter" in error_lower or "safety" in error_lower or "moderation" in error_lower):
            reply = (
                f"您输入的内容包含了不被允许的敏感信息，请修改后重试。"
                f"\n（追踪ID: {trace_id}）"
            )
        elif "service unavailable" in error_lower or "503" in error_lower or "overloaded" in error_lower:
            reply = (
                f"大模型服务暂时不可用，请稍等 1-2 分钟后再试。"
                f"\n（追踪ID: {trace_id}）"
            )
        elif "connect" in error_lower or "network" in error_lower or "dns" in error_lower:
            reply = (
                f"连接大模型服务失败，请检查网络连接后重试。"
                f"\n（追踪ID: {trace_id}）"
            )
        else:
            reply = (
                f"小鹿遇到了一点小问题，请您稍后再试试好吗？"
                f"\n（追踪ID: {trace_id}）"
            )

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
    """轻量探测 Java 后端（店铺状态公开接口，带 30 秒缓存）。"""
    now = time.time()
    if now - _java_probe["ts"] < _JAVA_PROBE_TTL:
        return _java_probe["ok"]
    try:
        bc.get_public("/api/shop/status", timeout=3.0)
        _java_probe.update(ts=now, ok=True)
    except Exception:
        _java_probe.update(ts=now, ok=False)
    return _java_probe["ok"]


async def get_agent_status() -> dict:
    """获取 Agent 服务健康状态（基于缓存，不每次真调大模型）。

    返回:
        { "status": str, "model": str, "active_sessions": int, ... }
    """
    llm_alive = llm_healthy(LLM_HEALTH_TTL)
    if not llm_alive:
        # 从未探测或缓存失效时补一次探测（probe_llm 内部有节流）
        llm_alive = await probe_llm(ttl=LLM_HEALTH_TTL)
    java_ok = await _probe_java_backend()

    return {
        "status": "ok" if llm_alive else "degraded",
        "model": LLM_MODEL,
        "active_sessions": len(_sessions),
        "llm_connected": llm_alive,
        "java_backend": "up" if java_ok else "down",
        "java_base_url": JAVA_BASE_URL,
    }


def shutdown() -> None:
    """释放全局资源（服务关闭时调用）。"""
    close_llm()
    bc.close_client()
