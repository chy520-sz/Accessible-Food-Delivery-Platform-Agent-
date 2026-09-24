# 工具清单与治理规范（TOOLS.md）

> 本文档由 `tool_registry.py` 自动生成，是 Agent 工具的单一事实来源。
> 修改工具元数据请改 `tool_registry.py`，再重新运行 `python _gen_tools_md.py`。

## 一、设计原则

1. **查询类（query）与执行类（mutation）严格分离**：query 只读、无副作用、可安全重试；
   mutation 会写后端状态，写操作**绝不自动重试**（防重复提交）。
2. **高风险操作两步确认**：凡花钱、不可逆、或推进交易状态机的工具，第一次调用只生成
   草稿（返回 `[code=DRAFT_REQUIRED]`），必须向用户复述影响并获明确确认后，再以相同
   参数调用一次才真正执行；同参数重复提交幂等去重。草稿默认 5 分钟过期。
3. **统一错误码**：所有工具错误以 `[code=XXX]` 形式追加在返回文本末尾，机器可读、稳定不变。

## 二、错误码总表

| 错误码 | 含义 | 可重试 | 期望动作 |
|---|---|---|---|
| `AUTH_REQUIRED` | 未登录或登录已过期 | 否 | 引导用户先调用 login 登录，再重试原操作 |
| `BAD_PARAM` | 入参非法或超出取值范围 | 否 | 修正参数后重试（如 score 必须在 1-5、ID 必须是数字） |
| `NOT_FOUND` | 目标资源不存在或已下架 | 否 | 提示用户换一个关键词/ID，或先搜索获取最新 ID |
| `CONFLICT` | 当前状态不允许该操作 | 否 | 向用户解释冲突原因（如售罄、订单已完成），不要原样重试 |
| `DRAFT_REQUIRED` | 高风险操作已生成草稿，等待用户确认 | 否 | 把草稿内容完整复述给用户，等待用户明确确认后再以相同参数调用一次 |
| `DRAFT_EXPIRED` | 草稿已过期 | 否 | 提示用户重新发起该操作，不要复用旧草稿 |
| `UPSTREAM_ERROR` | Java 后端返回业务错误 | 否 | 把后端 message 转达给用户；写操作失败不要自动重试 |
| `NETWORK_ERROR` | 网络/超时/服务端 5xx | 是 | 查询类可自动重试；写操作类提示用户稍后再试 |
| `INTERNAL_ERROR` | 未预期的内部错误 | 否 | 记录日志并提示用户稍后再试，不要编造结果 |

## 三、工具分类总览

- 查询类（query）：**24** 个
- 执行类（mutation）：**12** 个
  - 其中高风险（草稿闸门）：**5** 个 —— `place_order`, `clear_cart`, `merchant_accept_order`, `delivery_pickup_order`, `delivery_complete_order`

## 查询类工具（只读，无副作用）

### `check_health_profile`

- **用途**：查看当前用户健康档案
- **鉴权**：需登录　|　**类型**：query　|　**风险**：low
- **适用场景**：
  - 生成饮食计划前/用户问'我的档案'
- **不适用场景**：
  - 还没档案时引导先 record_health_profile
- **参数 Schema**：
  （无参数）
- **返回结构**：JSON 字符串：{"action","profile","message"}；无档案返回 profile_not_found
- **可能错误码**：`OK`, `AUTH_REQUIRED`, `NOT_FOUND`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `estimate_dish_nutrition`

- **用途**：本地关键词匹配估算菜品热量（纯本地计算，不调后端）
- **鉴权**：公开/无需登录　|　**类型**：query　|　**风险**：low
- **适用场景**：
  - 用户问'这道菜多少大卡''热量高不高'
- **不适用场景**：
  - 需要精确营养（当前无数据源，应告知仅供参考）
  - 健康饮食规则类问题（用 search_dietary_knowledge）
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `dish_name` | string | 是 | 菜品名 |
  | `dish_desc` | string | 否 | 菜品描述，辅助匹配，默认 '' |
- **返回结构**：纯文本：估算热量区间 + 匹配关键词 + 免责提示
- **可能错误码**：`OK`, `INTERNAL_ERROR`

### `get_cart`

- **用途**：查看购物车内容与合计金额
- **鉴权**：需登录　|　**类型**：query　|　**风险**：low
- **适用场景**：
  - 用户问'我购物车里有什么''多少钱'
  - 下单前复述订单明细
- **不适用场景**：
  - 需要修改购物车（用 add_to_cart/clear_cart）
- **参数 Schema**：
  （无参数）
- **返回结构**：纯文本：逐行菜品/套餐 + 数量单价 + 合计；空车引导去选菜
- **可能错误码**：`OK`, `AUTH_REQUIRED`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `get_combo_detail`

- **用途**：按 combo_id 查看套餐详情及所含菜品明细
- **鉴权**：需登录　|　**类型**：query　|　**风险**：low
- **适用场景**：
  - 用户选定某个套餐，要看里面具体含哪几道菜
- **不适用场景**：
  - 还不知道 combo_id（先 search_combos）
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `combo_id` | integer | 是 | 套餐ID |
- **返回结构**：纯文本：套餐基本信息 + 逐行列出所含菜品名与数量
- **可能错误码**：`OK`, `AUTH_REQUIRED`, `NOT_FOUND`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `get_dish_detail`

- **用途**：按 dish_id 查看单个菜品完整信息
- **鉴权**：需登录　|　**类型**：query　|　**风险**：low
- **适用场景**：
  - 用户已经从搜索结果里点了某道菜，要看价格/库存/描述
- **不适用场景**：
  - 还不知道 dish_id（先 search_dishes）
  - 用户想看套餐（用 get_combo_detail）
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `dish_id` | integer | 是 | 菜品ID，来自搜索结果 |
- **返回结构**：纯文本：名称/店铺/价格/分类/月销/库存/描述；ID 无效返回提示
- **可能错误码**：`OK`, `AUTH_REQUIRED`, `NOT_FOUND`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `get_my_rating_history`

- **用途**：查询当前用户历史评分
- **鉴权**：需登录　|　**类型**：query　|　**风险**：low
- **适用场景**：
  - 分析用户偏好、回顾过往评价
- **不适用场景**：
  - 用户是新用户/还没评过分时直接告知即可
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `limit` | integer | 否 | 返回条数，默认 10 |
- **返回结构**：纯文本：评分历史列表
- **可能错误码**：`OK`, `AUTH_REQUIRED`, `NOT_FOUND`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `get_shop_status`

- **用途**：查询店铺营业状态（公开）
- **鉴权**：公开/无需登录　|　**类型**：query　|　**风险**：low
- **适用场景**：
  - 用户准备下单前确认是否打烊
- **不适用场景**：
  - 已在下单流程中无需重复查询
- **参数 Schema**：
  （无参数）
- **返回结构**：纯文本：营业中/已打烊
- **可能错误码**：`OK`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `get_user_addresses`

- **用途**：查询用户已保存收货地址
- **鉴权**：需登录　|　**类型**：query　|　**风险**：low
- **适用场景**：
  - 下单前让用户选地址
- **不适用场景**：
  - 用户报了一个新地址要新增（当前系统不支持新增地址接口）
- **参数 Schema**：
  （无参数）
- **返回结构**：纯文本：地址ID/姓名/电话/地址，默认地址有标记
- **可能错误码**：`OK`, `AUTH_REQUIRED`, `NOT_FOUND`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `get_user_orders`

- **用途**：查询用户全部历史订单
- **鉴权**：需登录　|　**类型**：query　|　**风险**：low
- **适用场景**：
  - 用户问'我的订单''上次点了什么'
- **不适用场景**：
  - 跟踪某订单实时配送（用 query_order_status）
- **参数 Schema**：
  （无参数）
- **返回结构**：纯文本：订单号/金额/状态中文/时间/明细
- **可能错误码**：`OK`, `AUTH_REQUIRED`, `NOT_FOUND`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `list_categories`

- **用途**：列出所有菜品分类（公开，用于导航）
- **鉴权**：公开/无需登录　|　**类型**：query　|　**风险**：low
- **适用场景**：
  - 用户想按分类逛菜
- **不适用场景**：
  - 已知分类ID 直接 search_dishes(category_id=...)
- **参数 Schema**：
  （无参数）
- **返回结构**：纯文本：序号 + 分类名 + 分类ID
- **可能错误码**：`OK`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `list_shops`

- **用途**：列出所有商家（公开）
- **鉴权**：公开/无需登录　|　**类型**：query　|　**风险**：low
- **适用场景**：
  - 用户想按商家/店铺点菜
- **不适用场景**：
  - 已知 shop_id 直接 search_dishes_by_shop
- **参数 Schema**：
  （无参数）
- **返回结构**：纯文本：序号 + 商家名 + 商家ID + 评分 + 评论数
- **可能错误码**：`OK`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `query_order_status`

- **用途**：查询订单真实配送状态（商家接单/骑手/时间点）
- **鉴权**：需登录　|　**类型**：query　|　**风险**：low
- **适用场景**：
  - 用户问'订单到哪了''什么时候送到'
- **不适用场景**：
  - 演示多智能体流程（用 simulate_multi_agent）
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `order_id` | integer | 是 | 订单ID |
- **返回结构**：纯文本：状态中文 + 骑手信息 + 关键时间点
- **可能错误码**：`OK`, `AUTH_REQUIRED`, `NOT_FOUND`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `recommend_dishes`

- **用途**：基于用户历史评分/评分均值/销量/时间衰减的个性化推荐
- **鉴权**：需登录　|　**类型**：query　|　**风险**：low
- **适用场景**：
  - 用户说'随便推荐点什么''有什么好吃的'
  - 首次进店/没有明确目标时
- **不适用场景**：
  - 用户已有明确菜品名（直接 search_dishes）
  - 要求按健康/营养筛选（结合 diet 工具）
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `limit` | integer | 否 | 推荐条数，默认 5 |
- **返回结构**：纯文本：推荐列表 + 每条推荐理由
- **可能错误码**：`OK`, `AUTH_REQUIRED`, `NOT_FOUND`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `search_combos`

- **用途**：搜索套餐
- **鉴权**：需登录　|　**类型**：query　|　**风险**：low
- **适用场景**：
  - 用户说'双人餐''家庭餐''套餐'等组合需求
- **不适用场景**：
  - 用户只要单点菜品（用 search_dishes）
  - 已知 combo_id 要看内容（用 get_combo_detail）
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `keyword` | string | 否 | 套餐名关键词，默认 '' |
  | `category_id` | integer | 否 | 分类ID |
- **返回结构**：纯文本：套餐列表（ID/名称/价格/店铺/月销）
- **可能错误码**：`OK`, `AUTH_REQUIRED`, `NOT_FOUND`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `search_dietary_knowledge`

- **用途**：RAG 检索饮食健康知识库（疾病饮食/忌口/营养）
- **鉴权**：公开/无需登录　|　**类型**：query　|　**风险**：low
- **适用场景**：
  - 糖尿病/高血压/痛风/减脂等饮食建议类问题
- **不适用场景**：
  - 查询平台菜品营养估算（用 estimate_dish_nutrition）
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `query` | string | 是 | 健康问题 |
- **返回结构**：纯文本：健康知识片段
- **可能错误码**：`OK`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `search_dishes`

- **用途**：语义+实时联合搜索菜品（先 Milvus 知识库，再平台实时在售）
- **鉴权**：需登录　|　**类型**：query　|　**风险**：low
- **适用场景**：
  - 用户想按菜名/口味/菜系找菜（如'宫保鸡丁''辣的''素菜'）
  - 用户给了一个想吃的方向，需要候选菜品列表
- **不适用场景**：
  - 用户只问菜品知识/做法（用 search_food_knowledge）
  - 用户要看某家店全部菜品（用 search_dishes_by_shop）
  - 已经知道具体 dish_id 要看详情（用 get_dish_detail）
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `keyword` | string | 否 | 搜索关键词，可为空，默认 '' |
  | `category_id` | integer | 否 | 分类ID，由 list_categories 获取 |
- **返回结构**：纯文本：命中条数 + 每行【ID | 名称 | 价格 | 店铺 | 月销 | 分类 | 描述】；未命中返回引导语
- **可能错误码**：`OK`, `AUTH_REQUIRED`, `NOT_FOUND`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `search_dishes_by_shop`

- **用途**：按商家ID分页查询该店菜品（公开）
- **鉴权**：公开/无需登录　|　**类型**：query　|　**风险**：low
- **适用场景**：
  - 用户说'XX店有什么菜'
- **不适用场景**：
  - 跨店搜某道菜（用 search_dishes）
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `shop_id` | integer | 是 | 商家ID，来自 list_shops |
  | `keyword` | string | 否 | 店内关键词筛选，默认 '' |
  | `page` | integer | 否 | 页码，从1开始，默认 1 |
  | `page_size` | integer | 否 | 每页条数，默认 20 |
- **返回结构**：纯文本：当前页菜品 + 总数 + 翻页提示
- **可能错误码**：`OK`, `NOT_FOUND`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `search_faq`

- **用途**：RAG 检索平台常见问题（下单/支付/配送/账户流程）
- **鉴权**：公开/无需登录　|　**类型**：query　|　**风险**：low
- **适用场景**：
  - 操作流程类问题，如'怎么退款''怎么加地址'
- **不适用场景**：
  - 具体订单问题（用 get_user_orders / query_order_status）
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `query` | string | 是 | 流程问题 |
- **返回结构**：纯文本：FAQ 片段
- **可能错误码**：`OK`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `search_food_knowledge`

- **用途**：RAG 检索美食知识库（口味/做法/食材/菜系）
- **鉴权**：公开/无需登录　|　**类型**：query　|　**风险**：low
- **适用场景**：
  - 用户问菜品文化/做法/食材搭配
  - 按食材/忌口筛选菜品前先用本工具
- **不适用场景**：
  - 查平台实时在售（用 search_dishes）
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `query` | string | 是 | 自然语言问题 |
- **返回结构**：纯文本：知识库检索片段
- **可能错误码**：`OK`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `simulate_multi_agent`

- **用途**：【演示用】模拟多智能体接单/配送流程（虚构数据）
- **鉴权**：公开/无需登录　|　**类型**：query　|　**风险**：low
- **适用场景**：
  - 用户明确要求'演示''看看多智能体效果'
- **不适用场景**：
  - 真实订单进度（用 query_order_status）
  - 任何真实下单/履约场景
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `order_summary` | string | 是 | 订单摘要文本 |
- **返回结构**：纯文本：虚构的接单→配送→完成演示流程
- **可能错误码**：`OK`, `INTERNAL_ERROR`

### `update_investigation`

- **用途**：更新问题排查调查板（Agent 自身的结构化排查工作记忆，写入 LangGraph state，不调用后端）
- **鉴权**：公开/无需登录　|　**类型**：query　|　**风险**：low
- **适用场景**：
  - 用户反馈异常类问题：下单失败、订单卡住、一直没送到、金额/扣款不对、登录失败等
  - 需要跨多步提出假设、调查询工具核实、记录已确认事实与已排除猜测时
- **不适用场景**：
  - 正常点餐流程（搜菜、加购、下单、查进度）
  - 单步即可回答、无需排查的问题
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `goal` | string | 是 | 一句话排查目标，如『定位用户下单失败原因』 |
  | `current_hypothesis` | string | 否 | 当前最可能的原因，默认 '' |
  | `confirmed_facts` | string | 否 | 已被工具结果证实的事实（列表），默认 '[]' |
  | `rejected_hypotheses` | string | 否 | 已排除的猜测（列表），避免重复怀疑，默认 '[]' |
  | `next_actions` | string | 否 | 下一步要验证或查询的事项（列表），默认 '[]' |
  | `status` | string | 否 | investigating=排查中，resolved=已定位，取值 ['investigating', 'resolved']，默认 'investigating' |
  | `conclusion` | string | 否 | 定位后的结论，resolved 时填写，默认 '' |
- **返回结构**：Command：更新 state.investigation 并回一条 ToolMessage（调查板已更新）
- **可能错误码**：`OK`, `INTERNAL_ERROR`

### `view_diet_plan`

- **用途**：查看当前生效的周饮食计划
- **鉴权**：需登录　|　**类型**：query　|　**风险**：low
- **适用场景**：
  - 用户问'我这周的计划'
- **不适用场景**：
  - 还没计划时引导 generate_diet_plan
- **参数 Schema**：
  （无参数）
- **返回结构**：JSON 字符串：{"action":"diet_plan","plan":{...},"week_start_date"}
- **可能错误码**：`OK`, `AUTH_REQUIRED`, `NOT_FOUND`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `view_diet_plan_history`

- **用途**：查询历史饮食计划列表
- **鉴权**：需登录　|　**类型**：query　|　**风险**：low
- **适用场景**：
  - 回顾过往计划
- **不适用场景**：
  - 看当前生效计划（用 view_diet_plan）
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `limit` | integer | 否 | 条数，默认 5 |
- **返回结构**：JSON 字符串：{"action":"diet_plan_history","history":[...]}
- **可能错误码**：`OK`, `AUTH_REQUIRED`, `NOT_FOUND`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `view_weight_history`

- **用途**：查询体重历史
- **鉴权**：需登录　|　**类型**：query　|　**风险**：low
- **适用场景**：
  - 看体重趋势、生成饮食计划前取近4周数据
- **不适用场景**：
  - 单次记体重（用 record_weight）
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `weeks` | integer | 否 | 回溯周数，默认 12 |
- **返回结构**：JSON 字符串：{"action":"weight_history","history":[...],"message"}
- **可能错误码**：`OK`, `AUTH_REQUIRED`, `NOT_FOUND`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

## 执行类工具 · 低风险（直接生效）

### `add_combo_to_cart`

- **用途**：把套餐加入购物车
- **鉴权**：需登录　|　**类型**：mutation　|　**风险**：low
- **适用场景**：
  - 用户明确说'加这个套餐'
- **不适用场景**：
  - 单点菜品（用 add_to_cart）
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `combo_id` | integer | 是 | 套餐ID |
  | `quantity` | integer | 否 | 数量，默认 1 |
- **返回结构**：纯文本：加入成功提示
- **副作用**：购物车新增/累加套餐；旧的待确认订单草稿自动失效
- **可能错误码**：`OK`, `AUTH_REQUIRED`, `BAD_PARAM`, `NOT_FOUND`, `CONFLICT`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `add_to_cart`

- **用途**：把菜品加入购物车（已存在则累加数量）
- **鉴权**：需登录　|　**类型**：mutation　|　**风险**：low
- **适用场景**：
  - 用户明确说'加这个菜'
- **不适用场景**：
  - 只是看看价格（用 get_dish_detail）
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `dish_id` | integer | 是 | 菜品ID |
  | `quantity` | integer | 否 | 数量，默认 1 |
- **返回结构**：纯文本：加入成功提示
- **副作用**：购物车新增/累加菜品；旧的待确认订单草稿自动失效
- **可能错误码**：`OK`, `AUTH_REQUIRED`, `BAD_PARAM`, `NOT_FOUND`, `CONFLICT`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `generate_diet_plan`

- **用途**：基于档案与体重趋势生成本周饮食计划并落库
- **鉴权**：需登录　|　**类型**：mutation　|　**风险**：low
- **适用场景**：
  - 用户要求'帮我制定这周饮食计划'
- **不适用场景**：
  - 还没健康档案（先 record_health_profile）
- **参数 Schema**：
  （无参数）
- **返回结构**：JSON 字符串：{"action":"diet_plan","plan":{7天安排},"message"}
- **副作用**：生成计划并保存到后端（diet-plan 表）
- **可能错误码**：`OK`, `AUTH_REQUIRED`, `BAD_PARAM`, `NOT_FOUND`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `login`

- **用途**：手机号+密码登录，写入 JWT 到会话
- **鉴权**：公开/无需登录　|　**类型**：mutation　|　**风险**：low
- **适用场景**：
  - 用户尚未登录且需要访问需鉴权接口
- **不适用场景**：
  - 已登录（重复登录会被工具拒绝）
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `phone` | string | 是 | 11位手机号 |
  | `password` | string | 是 | 登录密码 |
- **返回结构**：纯文本：登录成功欢迎语 / 失败原因
- **副作用**：在会话 store 写入 JWT（有效期约30分钟）
- **可能错误码**：`OK`, `BAD_PARAM`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `rate_item`

- **用途**：给菜品/套餐/店铺打分
- **鉴权**：需登录　|　**类型**：mutation　|　**风险**：low
- **适用场景**：
  - 用户用餐后主动评价
- **不适用场景**：
  - 查看历史评分（用 get_my_rating_history）
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `target_type` | string | 是 | 对象类型，取值 ['DISH', 'COMBO', 'SHOP'] |
  | `target_id` | integer | 是 | 对象ID |
  | `score` | integer | 是 | 1-5 星，取值 [1, 2, 3, 4, 5] |
  | `comment` | string | 否 | 评语，默认 '' |
- **返回结构**：纯文本：评分已提交回执
- **副作用**：在后端写入一条评分记录
- **可能错误码**：`OK`, `AUTH_REQUIRED`, `BAD_PARAM`, `NOT_FOUND`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `record_health_profile`

- **用途**：保存/更新健康档案（带参数校验）
- **鉴权**：需登录　|　**类型**：mutation　|　**风险**：low
- **适用场景**：
  - 用户首次录入或修改健康信息
- **不适用场景**：
  - 只是查看（用 check_health_profile）
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `age` | integer | 是 | 10-100 |
  | `gender` | string | 是 | 男/女，取值 ['男', '女'] |
  | `height` | number | 是 | 身高 cm，100-250 |
  | `weight` | number | 是 | 体重 kg，30-200 |
  | `activity_level` | string | 是 | 活动水平，取值 ['低', '中', '高'] |
  | `diet_preference` | string | 是 | 饮食偏好 |
  | `health_goal` | string | 是 | 健康目标，取值 ['减肥', '增肌', '维持'] |
- **返回结构**：JSON 字符串：{"action":"profile_save"|"error",...}
- **副作用**：覆盖式写入用户健康档案
- **可能错误码**：`OK`, `AUTH_REQUIRED`, `BAD_PARAM`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `record_weight`

- **用途**：记录一次体重
- **鉴权**：需登录　|　**类型**：mutation　|　**风险**：low
- **适用场景**：
  - 用户报今日体重
- **不适用场景**：
  - 看趋势（用 view_weight_history）
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `weight` | number | 是 | kg，30-200 |
  | `record_date` | string | 否 | yyyy-MM-dd，空=今天，默认 '' |
- **返回结构**：JSON 字符串：{"action":"weight_recorded",...}
- **副作用**：追加一条体重记录
- **可能错误码**：`OK`, `AUTH_REQUIRED`, `BAD_PARAM`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

## 执行类工具 · 高风险（先生成草稿，确认后提交）

### `clear_cart`

- **用途**：清空购物车全部菜品/套餐
- **鉴权**：需登录　|　**类型**：mutation　|　**风险**：high　|　**草稿确认**：是
- **适用场景**：
  - 用户明确说'清空购物车''不要了重新点'
- **不适用场景**：
  - 只想删某一道菜（当前无单删接口，应提示用户这是清空全部）
  - 只是查看（用 get_cart）
- **参数 Schema**：
  （无参数）
- **返回结构**：首次：DRAFT_REQUIRED 草稿；确认后：购物车已清空
- **副作用**：删除购物车所有条目（误操作需重新选菜）
- **可能错误码**：`OK`, `AUTH_REQUIRED`, `CONFLICT`, `DRAFT_REQUIRED`, `DRAFT_EXPIRED`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `delivery_complete_order`

- **用途**：【配送Agent】标记订单已送达完成
- **鉴权**：需登录　|　**类型**：mutation　|　**风险**：high　|　**草稿确认**：是
- **适用场景**：
  - 配送中订单，确认已送达
- **不适用场景**：
  - 订单未在配送中
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `order_id` | integer | 是 | 订单ID |
  | `user_id` | integer | 是 | 下单用户ID |
- **返回结构**：首次：DRAFT_REQUIRED；确认后：送达完成
- **副作用**：订单状态 delivering → completed（不可逆，触发交易完结）
- **可能错误码**：`OK`, `AUTH_REQUIRED`, `BAD_PARAM`, `NOT_FOUND`, `CONFLICT`, `DRAFT_REQUIRED`, `DRAFT_EXPIRED`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `delivery_pickup_order`

- **用途**：【配送Agent】标记订单进入配送中
- **鉴权**：需登录　|　**类型**：mutation　|　**风险**：high　|　**草稿确认**：是
- **适用场景**：
  - 已接单订单，骑手取餐后启动配送
- **不适用场景**：
  - 订单未接单/已完成
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `order_id` | integer | 是 | 订单ID |
  | `user_id` | integer | 是 | 下单用户ID |
- **返回结构**：首次：DRAFT_REQUIRED；确认后：配送中 + 骑手电话
- **副作用**：订单状态 accepted → delivering（不可逆）
- **可能错误码**：`OK`, `AUTH_REQUIRED`, `BAD_PARAM`, `NOT_FOUND`, `CONFLICT`, `DRAFT_REQUIRED`, `DRAFT_EXPIRED`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `merchant_accept_order`

- **用途**：【商家Agent】接单并开始备餐（推进订单状态机）
- **鉴权**：需登录　|　**类型**：mutation　|　**风险**：high　|　**草稿确认**：是
- **适用场景**：
  - 待接单订单，需要商家侧确认备餐
- **不适用场景**：
  - 订单已接单/已完成（状态机不允许）
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `order_id` | integer | 是 | 订单ID |
  | `user_id` | integer | 是 | 下单用户ID |
- **返回结构**：首次：DRAFT_REQUIRED；确认后：接单成功 + 预计出餐时间
- **副作用**：订单状态 pending → accepted（不可逆）
- **可能错误码**：`OK`, `AUTH_REQUIRED`, `BAD_PARAM`, `NOT_FOUND`, `CONFLICT`, `DRAFT_REQUIRED`, `DRAFT_EXPIRED`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

### `place_order`

- **用途**：提交订单（两阶段：首次生成草稿，用户确认后真正下单）
- **鉴权**：需登录　|　**类型**：mutation　|　**风险**：high　|　**草稿确认**：是
- **适用场景**：
  - 购物车已就绪、地址已选定、用户已确认
- **不适用场景**：
  - 还没选好菜/没选地址（先引导选车和地址）
  - 用户只是看看多少钱（用 get_cart）
- **参数 Schema**：

  | 参数 | 类型 | 必填 | 说明 |
  |---|---|---|---|
  | `address_id` | integer | 是 | 地址ID，先 get_user_addresses |
  | `remark` | string | 否 | 订单备注，如少辣，默认 '' |
- **返回结构**：首次调用：DRAFT_REQUIRED 草稿提示（不下单）；确认后：下单成功（订单号/金额/状态）；重复提交：幂等返回首次结果
- **副作用**：扣减库存、生成订单、清空购物车（花钱，不可逆）
- **可能错误码**：`OK`, `AUTH_REQUIRED`, `BAD_PARAM`, `NOT_FOUND`, `CONFLICT`, `DRAFT_REQUIRED`, `DRAFT_EXPIRED`, `UPSTREAM_ERROR`, `NETWORK_ERROR`, `INTERNAL_ERROR`

