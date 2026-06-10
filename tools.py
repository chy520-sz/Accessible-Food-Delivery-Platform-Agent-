"""
工具函数模块 —— 封装对 Java 后端 API 的 HTTP 调用。

所有需要认证的请求自动携带当前会话的 JWT token。
使用 tenacity 库实现失败自动重试，提升视障用户交互稳定性。
"""
import json
import time
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from config import JAVA_BASE_URL, MAX_RETRIES, RETRY_DELAY, CALORIE_ESTIMATE_MAP, SSL_VERIFY, AGENT_API_KEY


# ==================== 会话级 JWT Token 存储 ====================
# key: session_id (str) → value: dict { "token", "user_id", "username", "expires_at" }
_session_store: dict[str, dict] = {}


def set_session(session_id: str, data: dict) -> None:
    """将 JWT 登录信息存入会话。"""
    _session_store[session_id] = data


def get_session(session_id: str) -> dict | None:
    """获取会话信息，若过期则自动清除并返回 None。"""
    sess = _session_store.get(session_id)
    if not sess:
        return None
    # 检查是否过期
    if time.time() > sess.get("expires_at", 0):
        _session_store.pop(session_id, None)
        return None
    return sess


def _get_auth_headers(session_id: str) -> dict[str, str]:
    """构造带 Bearer token 的 Authorization 请求头。"""
    sess = get_session(session_id)
    if not sess:
        raise PermissionError("用户未登录或登录已过期，请先登录。")
    return {"Authorization": f"Bearer {sess['token']}", "Content-Type": "application/json"}


# ==================== 底层 HTTP 请求封装 ====================

def _is_retryable_error(exception: BaseException) -> bool:
    """判断异常是否可重试（网络超时、5xx 服务端错误可重试，4xx 不可重试）。"""
    if isinstance(exception, httpx.TimeoutException):
        return True
    if isinstance(exception, httpx.HTTPStatusError):
        return 500 <= exception.response.status_code < 600
    if isinstance(exception, httpx.ConnectError):
        return True
    return False


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_fixed(RETRY_DELAY),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError)),
    reraise=True,
)
def _get(path: str, session_id: str, params: dict | None = None) -> dict:
    """发送 GET 请求到 Java 后端，自动重试。"""
    url = f"{JAVA_BASE_URL}{path}"
    headers = _get_auth_headers(session_id)
    # Content-Type 头只对 POST/PUT 有意义，GET 去掉
    headers.pop("Content-Type", None)
    with httpx.Client(verify=SSL_VERIFY, timeout=15.0) as client:
        resp = client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        return resp.json()


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_fixed(RETRY_DELAY),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError)),
    reraise=True,
)
def _post(path: str, session_id: str, body: dict | None = None) -> dict:
    """发送 POST 请求到 Java 后端，自动重试。"""
    url = f"{JAVA_BASE_URL}{path}"
    headers = _get_auth_headers(session_id)
    with httpx.Client(verify=SSL_VERIFY, timeout=15.0) as client:
        resp = client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        return resp.json()


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_fixed(RETRY_DELAY),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError)),
    reraise=True,
)
def _put(path: str, session_id: str, body: dict | None = None) -> dict:
    """发送 PUT 请求到 Java 后端，自动重试。"""
    url = f"{JAVA_BASE_URL}{path}"
    headers = _get_auth_headers(session_id)
    with httpx.Client(verify=SSL_VERIFY, timeout=15.0) as client:
        resp = client.put(url, headers=headers, json=body)
        resp.raise_for_status()
        return resp.json()


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_fixed(RETRY_DELAY),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError)),
    reraise=True,
)
def _delete(path: str, session_id: str) -> dict:
    """发送 DELETE 请求到 Java 后端，自动重试。"""
    url = f"{JAVA_BASE_URL}{path}"
    headers = _get_auth_headers(session_id)
    headers.pop("Content-Type", None)
    with httpx.Client(verify=SSL_VERIFY, timeout=15.0) as client:
        resp = client.delete(url, headers=headers)
        resp.raise_for_status()
        return resp.json()


def _extract_data(result: dict) -> object:
    """从 Java 统一响应 {code, message, data} 中提取 data 字段。
    如果 code != 200 则抛出异常，便于上层感知业务错误。
    """
    code = result.get("code", -1)
    if code != 200:
        msg = result.get("message", "未知错误")
        raise RuntimeError(f"Java 后端返回错误 (code={code}): {msg}")
    return result.get("data")


def _extract_records(data: object) -> list:
    """兼容 Java 后端直接返回列表或分页对象两种结构。"""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        records = data.get("records")
        return records if isinstance(records, list) else []
    return []


# ==================== LangChain 工具函数 ====================

def login(session_id: str, phone: str, password: str) -> str:
    """用户登录认证。
    使用手机号和密码登录 Java 后端，获取 JWT token 并存入当前会话。
    这是所有需要认证操作的前置步骤。

    参数:
        phone: 用户手机号（11位）
        password: 用户密码

    返回:
        登录结果描述，包含用户名和用户ID。
    """
    # 登录接口不需要认证，直接请求
    url = f"{JAVA_BASE_URL}/api/auth/user/login"
    try:
        with httpx.Client(verify=SSL_VERIFY, timeout=10.0) as client:
            resp = client.post(url, json={"phone": phone, "password": password})
            resp.raise_for_status()
            result = resp.json()
            if result.get("code") != 200:
                return f"登录失败：{result.get('message', '未知错误')}"
            data = result["data"]
            # 存入会话：token 有效期约 7 天，这里按 30 分钟做会话过期
            set_session(session_id, {
                "token": data["token"],
                "user_id": data["id"],
                "username": data["username"],
                "expires_at": time.time() + 1800,
            })
            return f"登录成功！欢迎 {data['username']}，您的用户ID是 {data['id']}。"
    except httpx.HTTPStatusError as e:
        return f"登录请求失败（HTTP {e.response.status_code}），请检查账号密码。"
    except httpx.ConnectError:
        return "无法连接到外卖后端服务，请确保服务已启动。"
    except Exception as e:
        return f"登录时发生异常：{str(e)}"


def search_dishes(session_id: str, keyword: str = "", category_id: Optional[int] = None) -> str:
    """搜索菜品。
    支持按关键词模糊搜索，也可以按分类ID筛选。
    返回菜品名称、所属店铺、价格、月销量、描述等信息。

    参数:
        keyword: 搜索关键词，如"宫保鸡丁"、"素菜"等。可为空表示不限制关键词。
        category_id: 分类ID数字，用于按分类筛选。可为空。

    返回:
        匹配的菜品列表（JSON 格式文本）。
    """
    params = {}
    if keyword:
        params["keyword"] = keyword
    if category_id is not None:
        params["categoryId"] = category_id
    try:
        result = _get("/api/user/dishes", session_id, params=params)
        dishes = _extract_records(_extract_data(result))
        if not dishes:
            keyword_hint = f"关于{keyword}的" if keyword else " "
            return f"没有找到{keyword_hint}菜品。您可以换个关键词试试，或者告诉我想吃什么口味的，我帮您推荐。"
        # 格式化为易读文本
        lines = []
        for d in dishes:
            name = d.get("name", "未知")
            price = d.get("price", 0)
            sales = d.get("monthlySales", 0)
            desc = d.get("description", "")
            stock = d.get("stock", 0)
            cat = d.get("categoryName", "")
            shop = d.get("shopName") or "未知店铺"
            dish_id = d.get("id", "")
            status_text = "【售罄】" if stock == 0 else ""
            lines.append(
                f"  · ID:{dish_id} | {status_text}{name} | ¥{price} | "
                f"店铺:{shop} | 月销{sales}单 | 分类:{cat} | {desc[:40]}{'...' if len(desc) > 40 else ''}"
            )
        return f"为您找到 {len(dishes)} 个菜品：\n" + "\n".join(lines)
    except PermissionError as e:
        return str(e)
    except RuntimeError as e:
        return f"查询菜品失败：{str(e)}"
    except Exception as e:
        return f"查询菜品时网络异常：{str(e)}"


def get_dish_detail(session_id: str, dish_id: int) -> str:
    """获取单个菜品详情。
    返回菜品的完整信息，包括名称、所属店铺、价格、描述、库存、月销量、分类等。

    参数:
        dish_id: 菜品ID数字

    返回:
        菜品详情文本。
    """
    try:
        result = _get(f"/api/user/dishes/{dish_id}", session_id)
        d = _extract_data(result)
        if not d:
            return f"未找到ID为 {dish_id} 的菜品。"
        lines = [
            f"菜品详情 —— {d.get('name', '未知')}",
            f"  ID: {d.get('id')}",
            f"  店铺: {d.get('shopName', '未知店铺')}",
            f"  价格: ¥{d.get('price', 0)}",
            f"  分类: {d.get('categoryName', '未知')}",
            f"  月销量: {d.get('monthlySales', 0)} 单",
            f"  库存: {d.get('stock', 0)} 份",
            f"  描述: {d.get('description', '暂无描述')}",
        ]
        return "\n".join(lines)
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"查询菜品详情失败：{str(e)}"


def recommend_dishes(session_id: str, limit: int = 5) -> str:
    """根据用户历史评分、平均分、评分数量、销量和时间衰减推荐菜品。"""
    try:
        result = _get("/api/user/recommendations/dishes", session_id, params={"limit": limit})
        dishes = _extract_data(result)
        if not dishes:
            return "暂时没有可推荐的菜品。您可以告诉我想吃什么口味，我再帮您查找。"
        lines = ["根据您的历史评分偏好，为您推荐："]
        for d in dishes[:limit]:
            name = d.get("name", "未知菜品")
            shop = d.get("shopName") or "未知店铺"
            price = d.get("price", 0)
            rating = d.get("rating") or 0
            review_count = d.get("reviewCount") or 0
            sales = d.get("monthlySales") or 0
            reason = "评分和销量综合较好"
            if review_count == 0:
                reason = "销量较好，适合作为新尝试"
            lines.append(
                f"  · ID:{d.get('id')} | {name} | 店铺:{shop} | ¥{price} | "
                f"评分{rating}分/{review_count}条评价 | 月售{sales} | 推荐理由:{reason}"
            )
        return "\n".join(lines)
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"获取推荐失败：{str(e)}"


def rate_item(session_id: str, target_type: str, target_id: int, score: int, comment: str = "") -> str:
    """给菜品、套餐或店铺评分。target_type 支持 DISH、COMBO、SHOP。"""
    try:
        body = {
            "targetType": target_type.upper(),
            "targetId": target_id,
            "score": score,
            "comment": comment,
        }
        result = _post("/api/user/ratings", session_id, body)
        rating = _extract_data(result)
        return (
            f"评分已提交：{rating.get('targetType')} ID {rating.get('targetId')}，"
            f"{rating.get('score')}星。"
        )
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"提交评分失败：{str(e)}"


def get_my_rating_history(session_id: str, limit: int = 10) -> str:
    """查询当前用户最近的评分历史。"""
    try:
        result = _get("/api/user/ratings/mine", session_id, params={"page": 1, "pageSize": limit})
        data = _extract_data(result)
        records = _extract_records(data)
        if not records:
            return "您还没有评分记录。用餐后可以给菜品、套餐或店铺打一到五星。"
        lines = ["您最近的评分记录："]
        for r in records:
            lines.append(
                f"  · {r.get('targetType')} ID:{r.get('targetId')} | "
                f"{r.get('score')}星 | {r.get('comment') or '无评语'}"
            )
        return "\n".join(lines)
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"查询评分历史失败：{str(e)}"


def get_cart(session_id: str) -> str:
    """查看当前用户的购物车。
    返回购物车中所有菜品/套餐的名称、数量、单价。

    返回:
        购物车内容文本。
    """
    try:
        result = _get("/api/user/cart", session_id)
        items = _extract_data(result)
        if not items or (isinstance(items, list) and len(items) == 0):
            return "您的购物车是空的。对我说'帮我推荐一些菜品'来开始点餐吧。"
        lines = ["您的购物车："]
        total = 0
        for i, item in enumerate(items, 1):
            name = item.get("name", "未知")
            shop = item.get("shopName") or "未知店铺"
            price = item.get("price", 0)
            qty = item.get("quantity", 1)
            subtotal = float(price) * qty
            total += subtotal
            lines.append(f"  {i}. {name}（店铺:{shop}）× {qty}  ¥{price}/份  小计 ¥{subtotal:.2f}")
        lines.append(f"  ——————————————")
        lines.append(f"  合计: ¥{total:.2f}")
        return "\n".join(lines)
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"获取购物车失败：{str(e)}"


def add_to_cart(session_id: str, dish_id: int, quantity: int = 1) -> str:
    """将菜品加入购物车。
    如果菜品已在购物车中，数量会累加。

    参数:
        dish_id: 菜品ID数字
        quantity: 添加数量，默认为1

    返回:
        操作结果文本。
    """
    try:
        body = {"dishId": dish_id, "quantity": quantity}
        _post("/api/user/cart", session_id, body)
        return f"已将 {quantity} 份菜品(ID:{dish_id})加入购物车。您可以继续选菜，或者对我说'去结算'。"
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"加入购物车失败：{str(e)}"


def clear_cart(session_id: str) -> str:
    """清空购物车中的所有菜品。

    返回:
        操作结果文本。
    """
    try:
        _delete("/api/user/cart", session_id)
        return "购物车已清空。"
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"清空购物车失败：{str(e)}"


def get_user_orders(session_id: str) -> str:
    """查询当前用户的历史订单。
    返回订单编号、金额、状态、下单时间等信息。

    返回:
        订单列表文本。
    """
    try:
        result = _get("/api/user/orders", session_id)
        orders = _extract_data(result)
        if not orders or (isinstance(orders, list) and len(orders) == 0):
            return "您还没有任何订单。对我说'我想点餐'来开始吧。"
        lines = [f"您共有 {len(orders)} 笔订单："]
        # 状态中文映射
        status_map = {
            "pending": "待接单", "accepted": "商家已接单",
            "delivering": "配送中", "completed": "已完成", "cancelled": "已取消",
        }
        for o in orders:
            order_no = o.get("orderNo", "未知")
            total = o.get("totalPrice", 0)
            status_en = o.get("status", "unknown")
            status_cn = status_map.get(status_en, status_en)
            created = o.get("createdAt", "")
            items = o.get("items", [])
            item_names = ", ".join([
                f"{it.get('name', '?')}（店铺:{it.get('shopName') or '未知店铺'}）"
                for it in items
            ]) if items else "无明细"
            lines.append(
                f"  · {order_no} | ¥{total} | {status_cn} | {created[:19]} | {item_names}"
            )
        return "\n".join(lines)
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"查询订单失败：{str(e)}"


def get_user_addresses(session_id: str) -> str:
    """查询当前用户的收货地址列表。

    返回:
        地址列表文本。
    """
    try:
        result = _get("/api/user/addresses", session_id)
        addrs = _extract_data(result)
        if not addrs or (isinstance(addrs, list) and len(addrs) == 0):
            return "您还没有保存收货地址。请告诉我您的收货地址（如'XX路XX号'），我来帮您下单。"
        lines = ["您的收货地址："]
        for a in addrs:
            is_default = "【默认】" if a.get("isDefault") == 1 else ""
            lines.append(
                f"  ID:{a.get('id')} {is_default} {a.get('name')} {a.get('phone')} {a.get('address')}"
            )
        return "\n".join(lines)
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"查询地址失败：{str(e)}"


def place_order(session_id: str, address_id: int, remark: str = "") -> str:
    """提交下单。
    系统会自动从购物车读取菜品并计算总价，下单后购物车自动清空。

    参数:
        address_id: 收货地址ID数字（需先通过 get_user_addresses 获取）
        remark: 订单备注，如"少辣"、"不要香菜"等

    返回:
        下单结果文本，包含订单编号和金额。
    """
    try:
        body = {"addressId": address_id}
        if remark:
            body["remark"] = remark
        result = _post("/api/user/orders", session_id, body)
        order = _extract_data(result)
        order_no = order.get("orderNo", "未知")
        total = order.get("totalPrice", 0)
        status = order.get("status", "pending")
        return (
            f"下单成功！\n"
            f"  订单编号: {order_no}\n"
            f"  应付金额: ¥{total}\n"
            f"  订单状态: {status}（等待商家接单）\n"
            f"商家接单后我会第一时间通知您。请保持手机畅通，注意接听配送电话。"
        )
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"下单失败：{str(e)}"


def get_dish_nutrition(session_id: str, dish_id: int) -> str:
    """获取菜品的营养估算信息。
    调用 Java 后端的营养估算接口，返回热量、蛋白质、脂肪等数据。
    数据基于菜品名称估算，仅供参考。

    参数:
        dish_id: 菜品ID数字

    返回:
        营养信息文本。
    """
    try:
        result = _get(f"/api/user/dishes/{dish_id}/nutrition", session_id)
        nutrition = _extract_data(result)
        if not nutrition:
            return f"未找到菜品(ID:{dish_id})的营养信息。"
        lines = [f"「{nutrition.get('dishName', '未知')}」营养估算："]
        for key, label in [("calories", "热量"), ("protein", "蛋白质"), ("fat", "脂肪")]:
            val = nutrition.get(key, "")
            if val:
                lines.append(f"  {label}: {val}")
        note = nutrition.get("note", "")
        if note:
            lines.append(f"  ⚠ {note}")
        return "\n".join(lines)
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"获取营养信息失败：{str(e)}"


def agent_accept_order(order_id: int, user_id: int) -> str:
    """【多智能体-商家接单】模拟商家确认订单并开始备餐。
    使用 Agent API Key 认证，无需用户 JWT。

    参数:
        order_id: 订单ID
        user_id: 用户ID

    返回:
        接单结果描述。
    """
    url = f"{JAVA_BASE_URL}/api/agent/orders/{order_id}/accept"
    try:
        with httpx.Client(verify=SSL_VERIFY, timeout=10.0) as client:
            resp = client.post(
                url,
                headers={"X-Agent-Key": AGENT_API_KEY, "Content-Type": "application/json"},
                json={"userId": user_id},
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            return (
                f"商家已接单！订单编号 {data.get('orderNo', 'N/A')}\n"
                f"预计 {data.get('prepTimeMin', 20)} 分钟出餐，请耐心等待～"
            )
    except Exception as e:
        return f"商家接单失败：{str(e)}"


def agent_start_delivery(order_id: int, user_id: int) -> str:
    """【多智能体-配送开始】模拟配送员取餐并开始配送。

    参数:
        order_id: 订单ID
        user_id: 用户ID

    返回:
        配送信息描述。
    """
    url = f"{JAVA_BASE_URL}/api/agent/orders/{order_id}/deliver"
    try:
        with httpx.Client(verify=SSL_VERIFY, timeout=10.0) as client:
            resp = client.post(
                url,
                headers={"X-Agent-Key": AGENT_API_KEY, "Content-Type": "application/json"},
                json={"userId": user_id},
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            return (
                f"配送员 {data.get('riderName', 'N/A')} 已取餐！\n"
                f"预计 {data.get('deliveryTimeMin', 25)} 分钟送达。\n"
                f"配送员电话: {data.get('riderPhone', 'N/A')}（如有需要可联系）"
            )
    except Exception as e:
        return f"配送启动失败：{str(e)}"


def agent_complete_order(order_id: int, user_id: int) -> str:
    """【多智能体-配送完成】标记订单已送达。

    参数:
        order_id: 订单ID
        user_id: 用户ID

    返回:
        完成信息描述。
    """
    url = f"{JAVA_BASE_URL}/api/agent/orders/{order_id}/complete"
    try:
        with httpx.Client(verify=SSL_VERIFY, timeout=10.0) as client:
            resp = client.post(
                url,
                headers={"X-Agent-Key": AGENT_API_KEY, "Content-Type": "application/json"},
                json={"userId": user_id},
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            return data.get("message", "订单已送达，祝您用餐愉快！")
    except Exception as e:
        return f"订单完成失败：{str(e)}"


def get_shop_status() -> str:
    """查询店铺当前营业状态（无需登录）。

    返回:
        店铺是否在营业的描述。
    """
    url = f"{JAVA_BASE_URL}/api/shop/status"
    try:
        with httpx.Client(verify=SSL_VERIFY, timeout=10.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            result = resp.json()
            if result.get("code") != 200:
                return f"无法获取店铺状态：{result.get('message', '未知错误')}"
            data = result["data"]
            is_open = data.get("isOpen", 0)
            if is_open == 1:
                return "店铺正在营业中，可以正常下单。"
            else:
                return "店铺当前已打烊，暂时无法下单。请稍后再来，或提前选好菜品等营业后再下单。"
    except Exception as e:
        return f"查询店铺状态失败：{str(e)}"


def estimate_dish_nutrition(dish_name: str, dish_desc: str = "") -> str:
    """估算菜品的营养热量（千卡）。
    根据菜品名称和描述中的关键词进行近似热量估算。
    精确营养数据请以实际餐品为准。

    参数:
        dish_name: 菜品名称
        dish_desc: 菜品描述（可选，用于更精确的匹配）

    返回:
        热量估算文本。
    """
    # 综合菜品名称与描述进行关键词匹配
    combined = (dish_name + dish_desc).lower()
    total_cal = 0
    matched_keywords = []

    # 多匹配：在合并文本中搜索每个关键词
    for kw, cal in CALORIE_ESTIMATE_MAP.items():
        if kw in combined:
            total_cal += cal
            matched_keywords.append(f"{kw}(~{cal}千卡/100g)")

    if not matched_keywords:
        # 无法匹配时给出通用估计
        return f"「{dish_name}」暂无精确营养数据。按常见外卖菜品估算，每份约 400~600 千卡。具体热量以实际餐品为准。"

    # 取平均值更合理
    avg_cal = total_cal // max(len(matched_keywords), 1)
    lines = [
        f"「{dish_name}」营养估算（仅供参考）：",
        f"  每份预估热量: 约 {avg_cal}~{total_cal} 千卡",
        f"  匹配依据: " + "、".join(matched_keywords),
        f"温馨提示：如果您有特殊饮食需求（如低盐、低糖、过敏等），请告知我，我帮您筛选合适的菜品。",
    ]
    return "\n".join(lines)
