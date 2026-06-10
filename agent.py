"""
智能 Agent 核心模块 —— LangChain AgentExecutor + ConversationBufferMemory。

核心架构:
  - 使用 DeepSeek (deepseek-chat) 作为推理引擎
  - 11 个核心工具函数供 LLM 调用（底层调用 Java 后端 API）
  - 会话级记忆管理，每个用户独立对话历史
  - 多智能体协同：导购 → 商家接单 → 配送，以函数串联展示

工具列表（供 LLM 调用）:
  1. search_dishes     - 搜索菜品
  2. get_cart          - 查看购物车
  3. add_to_cart       - 加入购物车
  4. get_user_orders   - 查看历史订单
  5. get_user_addresses- 查看收货地址
  6. place_order       - 提交下单
  7. get_shop_status   - 查询店铺状态
  8. get_dish_detail   - 查看菜品详情
  9. clear_cart        - 清空购物车
  10. estimate_dish_nutrition - 估算菜品营养
  11. simulate_multi_agent - 多智能体协同演示
"""
import json
import logging
import re
import traceback
import uuid
from typing import Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    LLM_MODEL,
    LLM_REQUEST_TIMEOUT,
    LLM_TEMPERATURE,
    MAX_LLM_TOKENS,
    SESSION_EXPIRE_SECONDS,
    SSL_VERIFY,
)

# 修复[错误分级]：配置结构化日志，记录完整上下文用于排障
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent")
from prompts import (
    XIAOLU_SYSTEM_PROMPT,
    MULTI_AGENT_FLOW,
    WELCOME_MESSAGE,
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
    get_dish_nutrition as _get_dish_nutrition,
    agent_accept_order as _agent_accept_order,
    agent_start_delivery as _agent_start_delivery,
    agent_complete_order as _agent_complete_order,
    set_session,
    get_session,
)

# ==================== 会话管理 ====================

class AgentSession:
    """用户会话，包含 LangChain Memory + JWT Token。

    每个用户（按 session_id 区分）拥有独立的:
      - ConversationBufferMemory: 对话历史
      - AgentExecutor: 绑定了该会话的 Agent 实例
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
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


# ==================== LLM 初始化 ====================

def _create_llm() -> ChatOpenAI:
    """创建 DeepSeek LLM 实例，通过 DeepSeek API 调用。

    DeepSeek 提供与 OpenAI 兼容的 API 端点，
    因此可以直接使用 langchain-openai 的 ChatOpenAI 类。
    """
    import httpx

    # 修复[LLM超时]：为 httpx 客户端设置与 LLM_REQUEST_TIMEOUT 一致的超时，防止 AgentExecutor
    # 长链工具调用被过早截断。同步/异步客户端均需配置，否则默认超时可能仅 5-10 秒。
    timeout = httpx.Timeout(LLM_REQUEST_TIMEOUT, connect=10.0)
    http_client = httpx.Client(verify=SSL_VERIFY, timeout=timeout)
    async_http_client = httpx.AsyncClient(verify=SSL_VERIFY, timeout=timeout)
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=MAX_LLM_TOKENS,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        streaming=False,
        request_timeout=LLM_REQUEST_TIMEOUT,
        http_client=http_client,
        http_async_client=async_http_client,
    )


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
    def get_dish_nutrition(dish_id: int) -> str:
        """获取某个菜品的营养信息（热量、蛋白质、脂肪等）。需要在登录后使用。
        参数 dish_id: 菜品ID数字
        """
        return _get_dish_nutrition(session_id, dish_id)

    @tool
    def simulate_multi_agent(order_summary: str) -> str:
        """【演示用】模拟多智能体协同工作流程：导购Agent → 商家接单Agent → 配送Agent。
        参数 order_summary: 订单摘要文本
        """
        return _simulate_multi_agent_flow(order_summary)

    @tool
    def merchant_accept_order(order_id: int, user_id: int) -> str:
        """【多智能体演示】商家接单Agent：确认订单并开始备餐。
        参数 order_id: 订单ID
        参数 user_id: 用户ID
        """
        return _agent_accept_order(order_id, user_id)

    @tool
    def delivery_pickup_order(order_id: int, user_id: int) -> str:
        """【多智能体演示】配送Agent：配送员取餐开始配送。
        参数 order_id: 订单ID
        参数 user_id: 用户ID
        """
        return _agent_start_delivery(order_id, user_id)

    @tool
    def delivery_complete_order(order_id: int, user_id: int) -> str:
        """【多智能体演示】配送Agent：订单已送达。
        参数 order_id: 订单ID
        参数 user_id: 用户ID
        """
        return _agent_complete_order(order_id, user_id)

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
        get_dish_nutrition,
        simulate_multi_agent,
        merchant_accept_order,
        delivery_pickup_order,
        delivery_complete_order,
        search_food_knowledge,       # RAG: 美食知识库检索
        search_dietary_knowledge,    # RAG: 饮食健康知识库检索
        search_faq,                  # RAG: 常见问题知识库检索
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
    llm = _create_llm()
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


# ==================== 多智能体协同模拟 ====================

def _simulate_multi_agent_flow(order_summary: str) -> str:
    """模拟三智能体协同流程：导购 → 商家接单 → 配送。

    此函数用于在作品中生动展示多智能体协作概念。
    实际生产环境中，商家接单和配送状态由 Java 后端管理，
    Agent 通过查询订单状态接口获取最新进展。

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
        "🐾  多智能体协同流程演示  🐾",
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
        f"  订单 {order_no} 全流程追踪完毕",
        "  小鹿将持续为您跟进订单状态～",
        "══════════════════════════",
    ]
    return "\n".join(lines)


# ==================== 登录态同步 ====================

def sync_login_token(session_id: str, auth_token: Optional[str]) -> tuple[bool, str]:
    """将前端登录态同步到 Agent 会话（免密码复用登录态）。

    处理三种场景：首次登录、换号、退出登录。

    参数:
        session_id: 会话ID
        auth_token: 前端传递的 JWT token（None 或空字符串表示退出登录）

    返回:
        (是否已登录, 用户名)
    """
    import base64, json, time

    sess = _sessions.get(session_id)
    if not sess:
        print(f"[sync_login] 会话 {session_id} 不存在")
        return False, ""

    stored = get_session(session_id)

    if auth_token:
        if not stored or stored.get("token") != auth_token:
            try:
                payload_b64 = auth_token.split(".")[1]
                missing_padding = (4 - len(payload_b64) % 4) % 4
                payload_b64 += "=" * missing_padding
                payload = json.loads(base64.urlsafe_b64decode(payload_b64).decode("utf-8"))
                print(f"[sync_login] JWT载荷: {json.dumps(payload, ensure_ascii=False)}")
                set_session(session_id, {
                    "token": auth_token,
                    "user_id": payload.get("userId") or payload.get("sub", 0),
                    "username": payload.get("username", "用户"),
                    "expires_at": time.time() + 1800,
                })
                sess.logged_in = True
                sess.username = payload.get("username", "用户")
                print(f"[sync_login] 会话 {session_id} 登录成功: username={sess.username}, userId={payload.get('userId') or payload.get('sub')}")
            except Exception as e:
                print(f"[sync_login] 会话 {session_id} JWT解码失败: {e}, token前20字符={auth_token[:20]}")
        else:
            print(f"[sync_login] 会话 {session_id} token未变化，跳过")
        print(f"[sync_login] 会话 {session_id} 最终状态: logged_in={sess.logged_in}, username={sess.username}")
        return sess.logged_in, sess.username
    else:
        # 前端无 token（用户退出登录）：清除 Agent 会话中的登录态
        print(f"[sync_login] 会话 {session_id} 清除登录态")
        if stored:
            set_session(session_id, {
                "token": "", "user_id": 0, "username": "",
                "expires_at": 0,
            })
            sess.logged_in = False
            sess.username = ""
        return False, ""


# ==================== TTS 文本清理 ====================

def clean_text_for_tts(text: str) -> str:
    """清理文本，移除表情和 Markdown 符号，用于语音合成。
    保留中文语义和自然停顿，去掉 TTS 会读出来的符号和表情。
    """
    # 移除 emoji（覆盖常见 Unicode 表情范围）
    emoji_pattern = re.compile(
        '['
        '\U0001F600-\U0001F64F'   # 表情符号
        '\U0001F300-\U0001F5FF'   # 杂项符号和象形文字
        '\U0001F680-\U0001F6FF'   # 交通工具和地图
        '\U0001F1E0-\U0001F1FF'   # 旗帜
        '\U0001F900-\U0001F9FF'   # 补充符号和象形文字
        '\U0001FA00-\U0001FA6F'   # 象棋符号
        '\U0001FA70-\U0001FAFF'   # 符号扩展-A
        '☀-➿'           # 杂项符号（包含 ☀⭐ 等）
        '⭐'                   # ⭐
        '️'                   # 变体选择器
        '‍'                   # 零宽连接符
        ']+', flags=re.UNICODE)
    text = emoji_pattern.sub('', text)

    # 移除 markdown 加粗 **text** → text
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)

    # 移除 markdown 斜体标记
    text = re.sub(r'\*(.+?)\*', r'\1', text)

    # 移除 markdown 代码标记
    text = re.sub(r'`(.+?)`', r'\1', text)

    # 将 — （em dash）替换为逗号停顿
    text = text.replace('—', '，')

    # 将 # 标题标记去掉内容保留
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)

    # 移除多余空格，保留换行作为停顿
    text = re.sub(r'[ \t]+', ' ', text)

    # 多个连续换行压缩为单个
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


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
    # 修复[错误分级]：每次请求生成唯一 trace_id，贯穿日志和错误响应，便于用户反馈时定位
    trace_id = uuid.uuid4().hex[:12]
    is_new = session_id is None

    # 修复[根因]：将 get_or_create_session 移入 try 块内。此前该调用在 try 外部（原第547行），
    # 若 AgentSession 初始化（LLM/httpx 创建）失败，异常直接逃逸到 FastAPI，导致无 trace_id 的 500。
    try:
        sess = get_or_create_session(session_id)

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

    except Exception as e:
        # 修复[错误分级]：不再简单截断错误消息，而是记录完整 traceback 到日志，
        # 同时按错误类型分级返回用户友好的提示，并在回复末尾附加 trace_id 供反馈
        error_msg = str(e)
        error_type = type(e).__name__
        logger.error(
            "[chat] trace=%s session=%s ERROR type=%s msg=%s\n%s",
            trace_id,
            session_id or "(new)",
            error_type,
            error_msg,
            traceback.format_exc(),
        )

        # 修复[错误分级]：分类识别大模型 API 的错误类型，给出精确定向提示
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

    return {
        "session_id": sess.session_id,
        "reply": reply,
        "tts_text": tts_text,
        "is_new_session": is_new,
        "trace_id": trace_id,
    }


async def get_agent_status() -> dict:
    """获取 Agent 服务健康状态。

    返回:
        { "status": "ok", "model": str, "active_sessions": int, ... }
    """
    try:
        llm = _create_llm()
        # 发送一个极短的测试请求验证 LLM 连通性
        test_msg = await llm.ainvoke("回复'OK'")
        llm_alive = "OK" in test_msg.content
    except Exception:
        llm_alive = False

    return {
        "status": "ok" if llm_alive else "degraded",
        "model": LLM_MODEL,
        "active_sessions": len(_sessions),
        "llm_connected": llm_alive,
        "java_backend": "http://localhost:3000",
    }
