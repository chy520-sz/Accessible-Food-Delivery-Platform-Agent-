"""
工具元数据注册表 —— 全量 Agent 工具的"说明书"单一事实来源。

每个工具在此声明：
  - kind          : query（只读查询，无副作用） / mutation（有副作用的执行类）
  - risk          : low / high（high = 花钱、不可逆、或推进交易状态机）
  - auth_required : 是否需要用户 JWT 登录
  - when_to_use   : 适用场景
  - when_not_to_use: 不适用场景（避免 LLM 误用）
  - params        : 参数 JSON Schema（OpenAI function calling 兼容）
  - returns       : 返回结构说明
  - errors        : 可能返回的错误码（见 tool_errors.py）
  - side_effect   : mutation 才有，描述对系统的改变
  - draft_required: high-risk 工具为 True，必须先生成草稿经用户确认后再提交

分类视图：
  - queries()     : 所有只读查询工具
  - mutations()   : 所有有副作用工具
  - high_risk()   : 其中高风险（草稿闸门）工具

注意：这里的元数据是文档/治理用，不替换 LangChain @tool 从类型注解自动推导的入参 schema；
它补充了"何时该用/不该用"和"返回什么/错在哪"这两块 LLM 看不见的信息。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from tool_errors import (
    AUTH_REQUIRED, BAD_PARAM, CONFLICT, DRAFT_REQUIRED, DRAFT_EXPIRED,
    INTERNAL_ERROR, NETWORK_ERROR, NOT_FOUND, OK, UPSTREAM_ERROR,
)


# ==================== 数据结构 ====================

@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str                       # "string" / "integer" / "number" / "boolean" / "array"
    description: str
    required: bool = True
    enum: Optional[list] = None      # 若有限定取值
    default: Any = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    kind: str                       # "query" | "mutation"
    risk: str                       # "low" | "high"
    auth_required: bool
    description: str
    when_to_use: tuple[str, ...]
    when_not_to_use: tuple[str, ...]
    params: tuple[ParamSpec, ...]
    returns: str
    errors: tuple[str, ...]
    side_effect: str = ""
    draft_required: bool = False


# ==================== 工具元数据 ====================

SPECS: dict[str, ToolSpec] = {}


def _reg(spec: ToolSpec) -> None:
    SPECS[spec.name] = spec


# ---------- 点餐核心：查询类 ----------

_reg(ToolSpec(
    name="search_dishes",
    kind="query", risk="low", auth_required=True,
    description="语义+实时联合搜索菜品（先 Milvus 知识库，再平台实时在售）",
    when_to_use=(
        "用户想按菜名/口味/菜系找菜（如'宫保鸡丁''辣的''素菜'）",
        "用户给了一个想吃的方向，需要候选菜品列表",
    ),
    when_not_to_use=(
        "用户只问菜品知识/做法（用 search_food_knowledge）",
        "用户要看某家店全部菜品（用 search_dishes_by_shop）",
        "已经知道具体 dish_id 要看详情（用 get_dish_detail）",
    ),
    params=(
        ParamSpec("keyword", "string", "搜索关键词，可为空", required=False, default=""),
        ParamSpec("category_id", "integer", "分类ID，由 list_categories 获取", required=False, default=None),
    ),
    returns="纯文本：命中条数 + 每行【ID | 名称 | 价格 | 店铺 | 月销 | 分类 | 描述】；未命中返回引导语",
    errors=(OK, AUTH_REQUIRED, NOT_FOUND, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
))

_reg(ToolSpec(
    name="get_dish_detail",
    kind="query", risk="low", auth_required=True,
    description="按 dish_id 查看单个菜品完整信息",
    when_to_use=(
        "用户已经从搜索结果里点了某道菜，要看价格/库存/描述",
    ),
    when_not_to_use=(
        "还不知道 dish_id（先 search_dishes）",
        "用户想看套餐（用 get_combo_detail）",
    ),
    params=(ParamSpec("dish_id", "integer", "菜品ID，来自搜索结果", required=True),),
    returns="纯文本：名称/店铺/价格/分类/月销/库存/描述；ID 无效返回提示",
    errors=(OK, AUTH_REQUIRED, NOT_FOUND, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
))

_reg(ToolSpec(
    name="search_combos",
    kind="query", risk="low", auth_required=True,
    description="搜索套餐",
    when_to_use=(
        "用户说'双人餐''家庭餐''套餐'等组合需求",
    ),
    when_not_to_use=(
        "用户只要单点菜品（用 search_dishes）",
        "已知 combo_id 要看内容（用 get_combo_detail）",
    ),
    params=(
        ParamSpec("keyword", "string", "套餐名关键词", required=False, default=""),
        ParamSpec("category_id", "integer", "分类ID", required=False, default=None),
    ),
    returns="纯文本：套餐列表（ID/名称/价格/店铺/月销）",
    errors=(OK, AUTH_REQUIRED, NOT_FOUND, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
))

_reg(ToolSpec(
    name="get_combo_detail",
    kind="query", risk="low", auth_required=True,
    description="按 combo_id 查看套餐详情及所含菜品明细",
    when_to_use=("用户选定某个套餐，要看里面具体含哪几道菜",),
    when_not_to_use=("还不知道 combo_id（先 search_combos）",),
    params=(ParamSpec("combo_id", "integer", "套餐ID", required=True),),
    returns="纯文本：套餐基本信息 + 逐行列出所含菜品名与数量",
    errors=(OK, AUTH_REQUIRED, NOT_FOUND, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
))

_reg(ToolSpec(
    name="recommend_dishes",
    kind="query", risk="low", auth_required=True,
    description="基于用户历史评分/评分均值/销量/时间衰减的个性化推荐",
    when_to_use=(
        "用户说'随便推荐点什么''有什么好吃的'",
        "首次进店/没有明确目标时",
    ),
    when_not_to_use=(
        "用户已有明确菜品名（直接 search_dishes）",
        "要求按健康/营养筛选（结合 diet 工具）",
    ),
    params=(ParamSpec("limit", "integer", "推荐条数", required=False, default=5),),
    returns="纯文本：推荐列表 + 每条推荐理由",
    errors=(OK, AUTH_REQUIRED, NOT_FOUND, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
))

_reg(ToolSpec(
    name="get_my_rating_history",
    kind="query", risk="low", auth_required=True,
    description="查询当前用户历史评分",
    when_to_use=("分析用户偏好、回顾过往评价",),
    when_not_to_use=("用户是新用户/还没评过分时直接告知即可",),
    params=(ParamSpec("limit", "integer", "返回条数", required=False, default=10),),
    returns="纯文本：评分历史列表",
    errors=(OK, AUTH_REQUIRED, NOT_FOUND, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
))

_reg(ToolSpec(
    name="get_cart",
    kind="query", risk="low", auth_required=True,
    description="查看购物车内容与合计金额",
    when_to_use=(
        "用户问'我购物车里有什么''多少钱'",
        "下单前复述订单明细",
    ),
    when_not_to_use=("需要修改购物车（用 add_to_cart/clear_cart）",),
    params=(),
    returns="纯文本：逐行菜品/套餐 + 数量单价 + 合计；空车引导去选菜",
    errors=(OK, AUTH_REQUIRED, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
))

_reg(ToolSpec(
    name="get_user_orders",
    kind="query", risk="low", auth_required=True,
    description="查询用户全部历史订单",
    when_to_use=("用户问'我的订单''上次点了什么'",),
    when_not_to_use=("跟踪某订单实时配送（用 query_order_status）",),
    params=(),
    returns="纯文本：订单号/金额/状态中文/时间/明细",
    errors=(OK, AUTH_REQUIRED, NOT_FOUND, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
))

_reg(ToolSpec(
    name="get_user_addresses",
    kind="query", risk="low", auth_required=True,
    description="查询用户已保存收货地址",
    when_to_use=("下单前让用户选地址",),
    when_not_to_use=("用户报了一个新地址要新增（当前系统不支持新增地址接口）",),
    params=(),
    returns="纯文本：地址ID/姓名/电话/地址，默认地址有标记",
    errors=(OK, AUTH_REQUIRED, NOT_FOUND, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
))

_reg(ToolSpec(
    name="get_shop_status",
    kind="query", risk="low", auth_required=False,
    description="查询店铺营业状态（公开）",
    when_to_use=("用户准备下单前确认是否打烊",),
    when_not_to_use=("已在下单流程中无需重复查询",),
    params=(),
    returns="纯文本：营业中/已打烊",
    errors=(OK, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
))

_reg(ToolSpec(
    name="list_categories",
    kind="query", risk="low", auth_required=False,
    description="列出所有菜品分类（公开，用于导航）",
    when_to_use=("用户想按分类逛菜",),
    when_not_to_use=("已知分类ID 直接 search_dishes(category_id=...)",),
    params=(),
    returns="纯文本：序号 + 分类名 + 分类ID",
    errors=(OK, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
))

_reg(ToolSpec(
    name="list_shops",
    kind="query", risk="low", auth_required=False,
    description="列出所有商家（公开）",
    when_to_use=("用户想按商家/店铺点菜",),
    when_not_to_use=("已知 shop_id 直接 search_dishes_by_shop",),
    params=(),
    returns="纯文本：序号 + 商家名 + 商家ID + 评分 + 评论数",
    errors=(OK, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
))

_reg(ToolSpec(
    name="search_dishes_by_shop",
    kind="query", risk="low", auth_required=False,
    description="按商家ID分页查询该店菜品（公开）",
    when_to_use=("用户说'XX店有什么菜'",),
    when_not_to_use=("跨店搜某道菜（用 search_dishes）",),
    params=(
        ParamSpec("shop_id", "integer", "商家ID，来自 list_shops", required=True),
        ParamSpec("keyword", "string", "店内关键词筛选", required=False, default=""),
        ParamSpec("page", "integer", "页码，从1开始", required=False, default=1),
        ParamSpec("page_size", "integer", "每页条数", required=False, default=20),
    ),
    returns="纯文本：当前页菜品 + 总数 + 翻页提示",
    errors=(OK, NOT_FOUND, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
))

_reg(ToolSpec(
    name="estimate_dish_nutrition",
    kind="query", risk="low", auth_required=False,
    description="本地关键词匹配估算菜品热量（纯本地计算，不调后端）",
    when_to_use=("用户问'这道菜多少大卡''热量高不高'",),
    when_not_to_use=(
        "需要精确营养（当前无数据源，应告知仅供参考）",
        "健康饮食规则类问题（用 search_dietary_knowledge）",
    ),
    params=(
        ParamSpec("dish_name", "string", "菜品名", required=True),
        ParamSpec("dish_desc", "string", "菜品描述，辅助匹配", required=False, default=""),
    ),
    returns="纯文本：估算热量区间 + 匹配关键词 + 免责提示",
    errors=(OK, INTERNAL_ERROR),
))

_reg(ToolSpec(
    name="query_order_status",
    kind="query", risk="low", auth_required=True,
    description="查询订单真实配送状态（商家接单/骑手/时间点）",
    when_to_use=("用户问'订单到哪了''什么时候送到'",),
    when_not_to_use=("演示多智能体流程（用 simulate_multi_agent）",),
    params=(ParamSpec("order_id", "integer", "订单ID", required=True),),
    returns="纯文本：状态中文 + 骑手信息 + 关键时间点",
    errors=(OK, AUTH_REQUIRED, NOT_FOUND, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
))

_reg(ToolSpec(
    name="search_food_knowledge",
    kind="query", risk="low", auth_required=False,
    description="RAG 检索美食知识库（口味/做法/食材/菜系）",
    when_to_use=(
        "用户问菜品文化/做法/食材搭配",
        "按食材/忌口筛选菜品前先用本工具",
    ),
    when_not_to_use=("查平台实时在售（用 search_dishes）",),
    params=(ParamSpec("query", "string", "自然语言问题", required=True),),
    returns="纯文本：知识库检索片段",
    errors=(OK, NETWORK_ERROR, INTERNAL_ERROR),
))

_reg(ToolSpec(
    name="search_dietary_knowledge",
    kind="query", risk="low", auth_required=False,
    description="RAG 检索饮食健康知识库（疾病饮食/忌口/营养）",
    when_to_use=("糖尿病/高血压/痛风/减脂等饮食建议类问题",),
    when_not_to_use=("查询平台菜品营养估算（用 estimate_dish_nutrition）",),
    params=(ParamSpec("query", "string", "健康问题", required=True),),
    returns="纯文本：健康知识片段",
    errors=(OK, NETWORK_ERROR, INTERNAL_ERROR),
))

_reg(ToolSpec(
    name="search_faq",
    kind="query", risk="low", auth_required=False,
    description="RAG 检索平台常见问题（下单/支付/配送/账户流程）",
    when_to_use=("操作流程类问题，如'怎么退款''怎么加地址'",),
    when_not_to_use=("具体订单问题（用 get_user_orders / query_order_status）",),
    params=(ParamSpec("query", "string", "流程问题", required=True),),
    returns="纯文本：FAQ 片段",
    errors=(OK, NETWORK_ERROR, INTERNAL_ERROR),
))

_reg(ToolSpec(
    name="check_health_profile",
    kind="query", risk="low", auth_required=True,
    description="查看当前用户健康档案",
    when_to_use=("生成饮食计划前/用户问'我的档案'",),
    when_not_to_use=("还没档案时引导先 record_health_profile",),
    params=(),
    returns='JSON 字符串：{"action","profile","message"}；无档案返回 profile_not_found',
    errors=(OK, AUTH_REQUIRED, NOT_FOUND, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
))

_reg(ToolSpec(
    name="view_weight_history",
    kind="query", risk="low", auth_required=True,
    description="查询体重历史",
    when_to_use=("看体重趋势、生成饮食计划前取近4周数据",),
    when_not_to_use=("单次记体重（用 record_weight）",),
    params=(ParamSpec("weeks", "integer", "回溯周数", required=False, default=12),),
    returns='JSON 字符串：{"action":"weight_history","history":[...],"message"}',
    errors=(OK, AUTH_REQUIRED, NOT_FOUND, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
))

_reg(ToolSpec(
    name="view_diet_plan",
    kind="query", risk="low", auth_required=True,
    description="查看当前生效的周饮食计划",
    when_to_use=("用户问'我这周的计划'",),
    when_not_to_use=("还没计划时引导 generate_diet_plan",),
    params=(),
    returns='JSON 字符串：{"action":"diet_plan","plan":{...},"week_start_date"}',
    errors=(OK, AUTH_REQUIRED, NOT_FOUND, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
))

_reg(ToolSpec(
    name="view_diet_plan_history",
    kind="query", risk="low", auth_required=True,
    description="查询历史饮食计划列表",
    when_to_use=("回顾过往计划",),
    when_not_to_use=("看当前生效计划（用 view_diet_plan）",),
    params=(ParamSpec("limit", "integer", "条数", required=False, default=5),),
    returns='JSON 字符串：{"action":"diet_plan_history","history":[...]}',
    errors=(OK, AUTH_REQUIRED, NOT_FOUND, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
))

_reg(ToolSpec(
    name="simulate_multi_agent",
    kind="query", risk="low", auth_required=False,
    description="【演示用】模拟多智能体接单/配送流程（虚构数据）",
    when_to_use=("用户明确要求'演示''看看多智能体效果'",),
    when_not_to_use=(
        "真实订单进度（用 query_order_status）",
        "任何真实下单/履约场景",
    ),
    params=(ParamSpec("order_summary", "string", "订单摘要文本", required=True),),
    returns="纯文本：虚构的接单→配送→完成演示流程",
    errors=(OK, INTERNAL_ERROR),
))

# ---------- 执行类：低风险 mutation ----------

_reg(ToolSpec(
    name="login",
    kind="mutation", risk="low", auth_required=False,
    description="手机号+密码登录，写入 JWT 到会话",
    when_to_use=("用户尚未登录且需要访问需鉴权接口",),
    when_not_to_use=("已登录（重复登录会被工具拒绝）",),
    params=(
        ParamSpec("phone", "string", "11位手机号", required=True),
        ParamSpec("password", "string", "登录密码", required=True),
    ),
    returns="纯文本：登录成功欢迎语 / 失败原因",
    errors=(OK, BAD_PARAM, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
    side_effect="在会话 store 写入 JWT（有效期约30分钟）",
))

_reg(ToolSpec(
    name="rate_item",
    kind="mutation", risk="low", auth_required=True,
    description="给菜品/套餐/店铺打分",
    when_to_use=("用户用餐后主动评价",),
    when_not_to_use=("查看历史评分（用 get_my_rating_history）",),
    params=(
        ParamSpec("target_type", "string", "对象类型", required=True,
                  enum=["DISH", "COMBO", "SHOP"]),
        ParamSpec("target_id", "integer", "对象ID", required=True),
        ParamSpec("score", "integer", "1-5 星", required=True, enum=[1, 2, 3, 4, 5]),
        ParamSpec("comment", "string", "评语", required=False, default=""),
    ),
    returns="纯文本：评分已提交回执",
    errors=(OK, AUTH_REQUIRED, BAD_PARAM, NOT_FOUND, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
    side_effect="在后端写入一条评分记录",
))

_reg(ToolSpec(
    name="add_to_cart",
    kind="mutation", risk="low", auth_required=True,
    description="把菜品加入购物车（已存在则累加数量）",
    when_to_use=("用户明确说'加这个菜'",),
    when_not_to_use=("只是看看价格（用 get_dish_detail）",),
    params=(
        ParamSpec("dish_id", "integer", "菜品ID", required=True),
        ParamSpec("quantity", "integer", "数量", required=False, default=1),
    ),
    returns="纯文本：加入成功提示",
    errors=(OK, AUTH_REQUIRED, BAD_PARAM, NOT_FOUND, CONFLICT, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
    side_effect="购物车新增/累加菜品；旧的待确认订单草稿自动失效",
))

_reg(ToolSpec(
    name="add_combo_to_cart",
    kind="mutation", risk="low", auth_required=True,
    description="把套餐加入购物车",
    when_to_use=("用户明确说'加这个套餐'",),
    when_not_to_use=("单点菜品（用 add_to_cart）",),
    params=(
        ParamSpec("combo_id", "integer", "套餐ID", required=True),
        ParamSpec("quantity", "integer", "数量", required=False, default=1),
    ),
    returns="纯文本：加入成功提示",
    errors=(OK, AUTH_REQUIRED, BAD_PARAM, NOT_FOUND, CONFLICT, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
    side_effect="购物车新增/累加套餐；旧的待确认订单草稿自动失效",
))

_reg(ToolSpec(
    name="record_health_profile",
    kind="mutation", risk="low", auth_required=True,
    description="保存/更新健康档案（带参数校验）",
    when_to_use=("用户首次录入或修改健康信息",),
    when_not_to_use=("只是查看（用 check_health_profile）",),
    params=(
        ParamSpec("age", "integer", "10-100", required=True),
        ParamSpec("gender", "string", "男/女", required=True, enum=["男", "女"]),
        ParamSpec("height", "number", "身高 cm，100-250", required=True),
        ParamSpec("weight", "number", "体重 kg，30-200", required=True),
        ParamSpec("activity_level", "string", "活动水平", required=True, enum=["低", "中", "高"]),
        ParamSpec("diet_preference", "string", "饮食偏好", required=True),
        ParamSpec("health_goal", "string", "健康目标", required=True, enum=["减肥", "增肌", "维持"]),
    ),
    returns='JSON 字符串：{"action":"profile_save"|"error",...}',
    errors=(OK, AUTH_REQUIRED, BAD_PARAM, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
    side_effect="覆盖式写入用户健康档案",
))

_reg(ToolSpec(
    name="record_weight",
    kind="mutation", risk="low", auth_required=True,
    description="记录一次体重",
    when_to_use=("用户报今日体重",),
    when_not_to_use=("看趋势（用 view_weight_history）",),
    params=(
        ParamSpec("weight", "number", "kg，30-200", required=True),
        ParamSpec("record_date", "string", "yyyy-MM-dd，空=今天", required=False, default=""),
    ),
    returns='JSON 字符串：{"action":"weight_recorded",...}',
    errors=(OK, AUTH_REQUIRED, BAD_PARAM, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
    side_effect="追加一条体重记录",
))

_reg(ToolSpec(
    name="generate_diet_plan",
    kind="mutation", risk="low", auth_required=True,
    description="基于档案与体重趋势生成本周饮食计划并落库",
    when_to_use=("用户要求'帮我制定这周饮食计划'",),
    when_not_to_use=("还没健康档案（先 record_health_profile）",),
    params=(),
    returns='JSON 字符串：{"action":"diet_plan","plan":{7天安排},"message"}',
    errors=(OK, AUTH_REQUIRED, BAD_PARAM, NOT_FOUND, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
    side_effect="生成计划并保存到后端（diet-plan 表）",
))

# ---------- 执行类：高风险 mutation（草稿确认闸门） ----------

_reg(ToolSpec(
    name="place_order",
    kind="mutation", risk="high", auth_required=True,
    description="提交订单（两阶段：首次生成草稿，用户确认后真正下单）",
    when_to_use=(
        "购物车已就绪、地址已选定、用户已确认",
    ),
    when_not_to_use=(
        "还没选好菜/没选地址（先引导选车和地址）",
        "用户只是看看多少钱（用 get_cart）",
    ),
    params=(
        ParamSpec("address_id", "integer", "地址ID，先 get_user_addresses", required=True),
        ParamSpec("remark", "string", "订单备注，如少辣", required=False, default=""),
    ),
    returns=(
        "首次调用：DRAFT_REQUIRED 草稿提示（不下单）；"
        "确认后：下单成功（订单号/金额/状态）；重复提交：幂等返回首次结果"
    ),
    errors=(OK, AUTH_REQUIRED, BAD_PARAM, NOT_FOUND, CONFLICT,
             DRAFT_REQUIRED, DRAFT_EXPIRED, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
    side_effect="扣减库存、生成订单、清空购物车（花钱，不可逆）",
    draft_required=True,
))

_reg(ToolSpec(
    name="clear_cart",
    kind="mutation", risk="high", auth_required=True,
    description="清空购物车全部菜品/套餐",
    when_to_use=("用户明确说'清空购物车''不要了重新点'",),
    when_not_to_use=(
        "只想删某一道菜（当前无单删接口，应提示用户这是清空全部）",
        "只是查看（用 get_cart）",
    ),
    params=(),
    returns="首次：DRAFT_REQUIRED 草稿；确认后：购物车已清空",
    errors=(OK, AUTH_REQUIRED, CONFLICT, DRAFT_REQUIRED, DRAFT_EXPIRED,
             UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
    side_effect="删除购物车所有条目（误操作需重新选菜）",
    draft_required=True,
))

_reg(ToolSpec(
    name="merchant_accept_order",
    kind="mutation", risk="high", auth_required=True,
    description="【商家Agent】接单并开始备餐（推进订单状态机）",
    when_to_use=("待接单订单，需要商家侧确认备餐",),
    when_not_to_use=("订单已接单/已完成（状态机不允许）",),
    params=(
        ParamSpec("order_id", "integer", "订单ID", required=True),
        ParamSpec("user_id", "integer", "下单用户ID", required=True),
    ),
    returns="首次：DRAFT_REQUIRED；确认后：接单成功 + 预计出餐时间",
    errors=(OK, AUTH_REQUIRED, BAD_PARAM, NOT_FOUND, CONFLICT,
             DRAFT_REQUIRED, DRAFT_EXPIRED, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
    side_effect="订单状态 pending → accepted（不可逆）",
    draft_required=True,
))

_reg(ToolSpec(
    name="delivery_pickup_order",
    kind="mutation", risk="high", auth_required=True,
    description="【配送Agent】标记订单进入配送中",
    when_to_use=("已接单订单，骑手取餐后启动配送",),
    when_not_to_use=("订单未接单/已完成",),
    params=(
        ParamSpec("order_id", "integer", "订单ID", required=True),
        ParamSpec("user_id", "integer", "下单用户ID", required=True),
    ),
    returns="首次：DRAFT_REQUIRED；确认后：配送中 + 骑手电话",
    errors=(OK, AUTH_REQUIRED, BAD_PARAM, NOT_FOUND, CONFLICT,
             DRAFT_REQUIRED, DRAFT_EXPIRED, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
    side_effect="订单状态 accepted → delivering（不可逆）",
    draft_required=True,
))

_reg(ToolSpec(
    name="delivery_complete_order",
    kind="mutation", risk="high", auth_required=True,
    description="【配送Agent】标记订单已送达完成",
    when_to_use=("配送中订单，确认已送达",),
    when_not_to_use=("订单未在配送中",),
    params=(
        ParamSpec("order_id", "integer", "订单ID", required=True),
        ParamSpec("user_id", "integer", "下单用户ID", required=True),
    ),
    returns="首次：DRAFT_REQUIRED；确认后：送达完成",
    errors=(OK, AUTH_REQUIRED, BAD_PARAM, NOT_FOUND, CONFLICT,
             DRAFT_REQUIRED, DRAFT_EXPIRED, UPSTREAM_ERROR, NETWORK_ERROR, INTERNAL_ERROR),
    side_effect="订单状态 delivering → completed（不可逆，触发交易完结）",
    draft_required=True,
))


# ---------- Agent 元工具：分层长期记忆 ----------

_reg(ToolSpec(
    name="remember_context",
    kind="mutation", risk="low", auth_required=True,
    description="按策略保存或更新用户明确授权的长期记忆；任务状态和外部知识会被分别路由",
    when_to_use=(
        "用户明确说『记住』『以后都这样』或确认了 Agent 的复述",
        "当前任务有临时进度需要跨轮保存时使用 task_state",
        "用户提供外部知识资料时使用 external_knowledge_reference 路由到 RAG 审核",
    ),
    when_not_to_use=(
        "仅从对话中猜测出的偏好或目标",
        "一次性选择和无长期价值的闲聊",
        "密码、令牌、支付凭证等任何秘密信息",
    ),
    params=(
        ParamSpec("memory_type", "string", "记忆类型", required=True, enum=[
            "user_preference", "user_goal", "project_background", "task_state",
            "historical_conclusion", "external_knowledge_reference",
        ]),
        ParamSpec("key", "string", "稳定、简短的记忆键；同类型同键会更新旧值", required=True),
        ParamSpec("content", "string", "用户授权保存的事实内容", required=True),
        ParamSpec("user_confirmed", "boolean", "敏感信息是否已经用户明确确认", required=False, default=False),
        ParamSpec("importance", "number", "重要性，0 到 1", required=False, default=0.5),
    ),
    returns="保存/更新结果与记忆 ID；任务状态返回临时保存；外部知识返回 RAG 引用 ID",
    errors=(OK, AUTH_REQUIRED, BAD_PARAM, INTERNAL_ERROR),
    side_effect="写入或更新当前登录用户的记忆、会话任务状态或 RAG 审核收件箱",
))

_reg(ToolSpec(
    name="forget_context",
    kind="mutation", risk="low", auth_required=True,
    description="按记忆 ID 删除当前登录用户明确要求忘记的长期记忆",
    when_to_use=("用户明确要求忘记一条已检索到且 ID 确定的记忆",),
    when_not_to_use=("用户没有明确提出删除", "不知道准确记忆 ID 时猜测删除"),
    params=(ParamSpec("memory_id", "string", "相关记忆上下文中给出的记忆 ID", required=True),),
    returns="删除成功或未找到；始终按当前登录用户隔离",
    errors=(OK, AUTH_REQUIRED, NOT_FOUND, INTERNAL_ERROR),
    side_effect="将指定长期记忆标记为已删除",
))

_reg(ToolSpec(
    name="supersede_conclusion",
    kind="mutation", risk="low", auth_required=True,
    description="新证据推翻旧历史结论时，将旧结论标为过时并降低置信度和重要性",
    when_to_use=("相关记忆中的历史结论已被用户陈述或工具证据明确推翻",),
    when_not_to_use=("只是怀疑旧结论不准确", "目标或偏好普通变更（应用相同 key 更新）"),
    params=(
        ParamSpec("memory_id", "string", "被推翻的 historical_conclusion 记忆 ID", required=True),
        ParamSpec("reason", "string", "推翻旧结论的新证据摘要", required=True),
    ),
    returns="旧结论已标记过时并降权，或未找到",
    errors=(OK, AUTH_REQUIRED, NOT_FOUND, INTERNAL_ERROR),
    side_effect="更新旧历史结论的状态、置信度和重要性，保留审计记录",
))


# ---------- Agent 元工具：问题排查调查板 ----------

_reg(ToolSpec(
    name="update_investigation",
    kind="query", risk="low", auth_required=False,
    description="更新问题排查调查板（Agent 自身的结构化排查工作记忆，写入 LangGraph state，不调用后端）",
    when_to_use=(
        "用户反馈异常类问题：下单失败、订单卡住、一直没送到、金额/扣款不对、登录失败等",
        "需要跨多步提出假设、调查询工具核实、记录已确认事实与已排除猜测时",
    ),
    when_not_to_use=(
        "正常点餐流程（搜菜、加购、下单、查进度）",
        "单步即可回答、无需排查的问题",
    ),
    params=(
        ParamSpec("goal", "string", "一句话排查目标，如『定位用户下单失败原因』", required=True),
        ParamSpec("current_hypothesis", "string", "当前最可能的原因", required=False, default=""),
        ParamSpec("confirmed_facts", "array", "已被工具结果证实的事实（字符串列表）", required=False, default=[]),
        ParamSpec("rejected_hypotheses", "array", "已排除的猜测（字符串列表）", required=False, default=[]),
        ParamSpec("next_actions", "array", "下一步要验证或查询的事项（字符串列表）", required=False, default=[]),
        ParamSpec("status", "string", "investigating=排查中，resolved=已定位",
                  required=False, enum=["investigating", "resolved"], default="investigating"),
        ParamSpec("conclusion", "string", "定位后的结论，resolved 时填写", required=False, default=""),
    ),
    returns="Command：更新 state.investigation 并回一条 ToolMessage（调查板已更新）",
    errors=(OK, INTERNAL_ERROR),
    draft_required=False,
))


# ==================== 分类视图 ====================

def all_specs() -> dict[str, ToolSpec]:
    return dict(SPECS)


def queries() -> dict[str, ToolSpec]:
    return {n: s for n, s in SPECS.items() if s.kind == "query"}


def mutations() -> dict[str, ToolSpec]:
    return {n: s for n, s in SPECS.items() if s.kind == "mutation"}


def high_risk() -> dict[str, ToolSpec]:
    return {n: s for n, s in SPECS.items() if s.risk == "high" and s.draft_required}


def get(name: str) -> Optional[ToolSpec]:
    return SPECS.get(name)


def to_openai_schema(spec: ToolSpec) -> dict:
    """把单个 ToolSpec 转成 OpenAI function calling 的 JSON Schema 片段。
    仅作治理/文档用途；运行时 schema 仍由 LangChain @tool 从类型注解生成。
    """
    properties: dict[str, dict] = {}
    required: list[str] = []
    for p in spec.params:
        prop: dict[str, Any] = {"type": p.type, "description": p.description}
        if p.type == "array":
            prop["items"] = {"type": "string"}
        if p.enum is not None:
            prop["enum"] = p.enum
        if p.default is not None:
            prop["default"] = p.default
        properties[p.name] = prop
        if p.required:
            required.append(p.name)
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": f"{spec.description}\n何时用：{'；'.join(spec.when_to_use)}\n"
                           f"何时不用：{'；'.join(spec.when_not_to_use)}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
