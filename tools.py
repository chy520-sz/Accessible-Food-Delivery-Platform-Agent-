"""
工具函数模块 —— 封装对 Java 后端 API 的 HTTP 调用。

所有需要认证的请求自动携带当前会话的 JWT token；统一走 backend_client。
重试策略：GET 查询可安全重试；下单/加购/评分等变更操作不重试（防重复提交）。
"""
import time
from typing import Optional

import backend_client as bc

from config import AGENT_API_KEY, CALORIE_ESTIMATE_MAP


# 兼容旧引用：agent.py 仍从 tools 导入 set_session / get_session
set_session = bc.set_session
get_session = bc.get_session


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
    try:
        result = bc.post_public("/api/auth/user/login", {
            "phone": phone, "password": password,
        })
        if result.get("code") != 200:
            return f"登录失败：{result.get('message', '未知错误')}"
        data = result["data"]
        # 存入会话：token 有效期约 30 分钟（与 Java 端 access-expiration 一致）
        set_session(session_id, {
            "token": data["token"],
            "user_id": data["id"],
            "username": data["username"],
            "expires_at": time.time() + 1800,
        })
        return f"登录成功！欢迎 {data['username']}，您的用户ID是 {data['id']}。"
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
        result = bc.get("/api/user/dishes", session_id, params=params)
        dishes = bc.extract_records(bc.extract_data(result))
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
        result = bc.get(f"/api/user/dishes/{dish_id}", session_id)
        d = bc.extract_data(result)
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


def search_combos(session_id: str, keyword: str = "", category_id: Optional[int] = None) -> str:
    """搜索套餐。
    支持按套餐名称关键词模糊搜索，也可以按分类ID筛选。
    返回套餐名称、所属店铺、价格、描述等信息。

    参数:
        keyword: 搜索关键词，如"双人餐""家庭餐"等。可为空表示不限制。
        category_id: 分类ID数字，用于按分类筛选。可为空。

    返回:
        匹配的套餐列表（易读文本）。
    """
    params = {"page": 1, "pageSize": 50}
    if keyword:
        params["keyword"] = keyword
    if category_id is not None:
        params["categoryId"] = category_id
    try:
        result = bc.get("/api/user/combos", session_id, params=params)
        data = bc.extract_data(result)
        combos = bc.extract_records(data)
        if not combos:
            keyword_hint = f"关于{keyword}的" if keyword else " "
            return f"没有找到{keyword_hint}套餐。您可以换个关键词试试，或者告诉我想吃什么，我帮您推荐菜品。"
        lines = []
        for c in combos:
            name = c.get("name", "未知")
            price = c.get("price", 0)
            shop = c.get("shopName") or "未知店铺"
            combo_id = c.get("id", "")
            desc = c.get("description", "")
            sales = c.get("monthlySales", 0)
            stock = c.get("stock", 0)
            status_text = "【售罄】" if stock == 0 else ""
            lines.append(
                f"  · ID:{combo_id} | {status_text}{name} | ¥{price} | "
                f"店铺:{shop} | 月销{sales}单 | {desc[:50]}{'...' if len(desc) > 50 else ''}"
            )
        return f"为您找到 {len(combos)} 个套餐：\n" + "\n".join(lines)
    except PermissionError as e:
        return str(e)
    except RuntimeError as e:
        return f"查询套餐失败：{str(e)}"
    except Exception as e:
        return f"查询套餐时网络异常：{str(e)}"


def get_combo_detail(session_id: str, combo_id: int) -> str:
    """获取单个套餐详情。
    返回套餐的完整信息，包括名称、所属店铺、价格、描述、库存，以及套餐包含的所有菜品明细。

    参数:
        combo_id: 套餐ID数字

    返回:
        套餐详情文本（含菜品明细）。
    """
    try:
        result = bc.get(f"/api/user/combos/{combo_id}", session_id)
        c = bc.extract_data(result)
        if not c:
            return f"未找到ID为 {combo_id} 的套餐。"
        lines = [
            f"套餐详情 —— {c.get('name', '未知')}",
            f"  ID: {c.get('id')}",
            f"  店铺: {c.get('shopName', '未知店铺')}",
            f"  套餐价: ¥{c.get('price', 0)}",
            f"  库存: {c.get('stock', 0)} 份",
            f"  月销量: {c.get('monthlySales', 0)} 单",
            f"  描述: {c.get('description', '暂无描述')}",
        ]
        # 套餐包含的菜品明细
        items = c.get("items") or c.get("comboDishes") or []
        if items:
            lines.append("  包含菜品：")
            for it in items:
                dish_name = it.get("dishName") or it.get("name", "未知菜品")
                qty = it.get("quantity", 1)
                lines.append(f"    - {dish_name} × {qty}")
        return "\n".join(lines)
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"查询套餐详情失败：{str(e)}"


def recommend_dishes(session_id: str, limit: int = 5) -> str:
    """根据用户历史评分、平均分、评分数量、销量和时间衰减推荐菜品。"""
    try:
        result = bc.get("/api/user/recommendations/dishes", session_id, params={"limit": limit})
        dishes = bc.extract_data(result)
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
        result = bc.post("/api/user/ratings", session_id, body)
        rating = bc.extract_data(result)
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
        result = bc.get("/api/user/ratings/mine", session_id, params={"page": 1, "pageSize": limit})
        data = bc.extract_data(result)
        records = bc.extract_records(data)
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
    返回购物车中所有菜品/套餐的名称、数量、单价，并区分是单品还是套餐。

    返回:
        购物车内容文本。
    """
    try:
        result = bc.get("/api/user/cart", session_id)
        items = bc.extract_data(result)
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
            # 区分单品和套餐
            if item.get("comboId"):
                item_type = "套餐"
            else:
                item_type = "菜品"
            lines.append(f"  {i}. [{item_type}] {name}（店铺:{shop}）× {qty}  ¥{price}/份  小计 ¥{subtotal:.2f}")
        lines.append("  ——————————————")
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
        bc.post("/api/user/cart", session_id, body)
        return f"已将 {quantity} 份菜品(ID:{dish_id})加入购物车。您可以继续选菜，或者对我说'去结算'。"
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"加入购物车失败：{str(e)}"


def add_combo_to_cart(session_id: str, combo_id: int, quantity: int = 1) -> str:
    """将套餐加入购物车。
    如果套餐已在购物车中，数量会累加。

    参数:
        combo_id: 套餐ID数字
        quantity: 添加数量，默认为1

    返回:
        操作结果文本。
    """
    try:
        body = {"comboId": combo_id, "quantity": quantity}
        bc.post("/api/user/cart", session_id, body)
        return f"已将 {quantity} 份套餐(ID:{combo_id})加入购物车。您可以继续选菜，或者对我说'去结算'。"
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
        bc.delete("/api/user/cart", session_id)
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
        result = bc.get("/api/user/orders", session_id)
        orders = bc.extract_data(result)
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
        result = bc.get("/api/user/addresses", session_id)
        addrs = bc.extract_data(result)
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


def place_order(session_id: str, address_id: int, remark: str = "",
                idempotency_key: str = "") -> str:
    """提交下单。
    系统会自动从购物车读取菜品并计算总价，下单后购物车自动清空。
    此操作**不自动重试**，并通过 idempotency_key 做服务端幂等，
    同一次已确认订单重复请求复用同一幂等键，避免重复下单。

    参数:
        address_id: 收货地址ID数字（需先通过 get_user_addresses 获取）
        remark: 订单备注，如"少辣"、"不要香菜"等
        idempotency_key: 服务端待确认订单派生的稳定幂等键

    返回:
        下单结果文本，包含订单编号和金额。
    """
    try:
        body = {"addressId": address_id}
        if remark:
            body["remark"] = remark
        if idempotency_key:
            body["idempotencyKey"] = idempotency_key
        result = bc.post("/api/user/orders", session_id, body)
        order = bc.extract_data(result)
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


def get_shop_status() -> str:
    """查询店铺当前营业状态（无需登录）。

    返回:
        店铺是否在营业的描述。
    """
    try:
        result = bc.get_public("/api/shop/status")
        if result.get("code") != 200:
            return f"无法获取店铺状态：{result.get('message', '未知错误')}"
        data = result["data"]
        is_open = data.get("isOpen", 0)
        if is_open == 1:
            return "店铺正在营业中，可以正常下单。"
        return "店铺当前已打烊，暂时无法下单。请稍后再来，或提前选好菜品等营业后再下单。"
    except Exception as e:
        return f"查询店铺状态失败：{str(e)}"


def list_categories() -> str:
    """获取所有菜品分类列表（用于分类浏览导航，无需登录）。

    返回:
        分类列表文本，包含分类名称和分类ID。
    """
    try:
        result = bc.get_public("/api/user/categories")
        if result.get("code") != 200:
            return f"无法获取分类列表：{result.get('message', '未知错误')}"
        data = result.get("data", [])
        if not data:
            return "暂无可用分类。"
        lines = ["可用菜品分类："]
        for idx, cat in enumerate(data, 1):
            name = cat.get("name", "未知")
            cat_id = cat.get("id", "")
            lines.append(f"  {idx}. {name}（分类ID: {cat_id}）")
        lines.append(f"\n共 {len(data)} 个分类。可使用 search_dishes 按 category_id 筛选菜品。")
        return "\n".join(lines)
    except Exception as e:
        return f"获取分类列表失败：{str(e)}"


def list_shops() -> str:
    """获取所有已启用的商家列表（用于商家浏览和按商家筛选菜品，无需登录）。

    返回:
        商家列表文本，包含商家名称、ID、评分、评论数。
    """
    try:
        result = bc.get_public("/api/shop/list")
        if result.get("code") != 200:
            return f"无法获取商家列表：{result.get('message', '未知错误')}"
        data = result.get("data", [])
        if not data:
            return "暂无可用商家。"
        lines = ["可用商家列表："]
        for idx, shop in enumerate(data, 1):
            name = shop.get("name", "未知")
            shop_id = shop.get("id", "")
            rating = shop.get("rating", 0)
            review_count = shop.get("reviewCount", 0)
            lines.append(f"  {idx}. {name}（商家ID: {shop_id}，评分: {rating}，评论数: {review_count}）")
        lines.append(f"\n共 {len(data)} 家商家。可使用 search_dishes_by_shop 按商家查看菜品。")
        return "\n".join(lines)
    except Exception as e:
        return f"获取商家列表失败：{str(e)}"


def search_dishes_by_shop(shop_id: int, keyword: str = "", page: int = 1, page_size: int = 20) -> str:
    """按商家ID搜索该店铺的菜品，可叠加关键词筛选（无需登录）。

    参数:
        shop_id: 商家ID
        keyword: 搜索关键词（可选）
        page: 页码，从1开始（默认1）
        page_size: 每页条数（默认20）

    返回:
        菜品列表文本，包含名称、价格、库存、菜品ID。
    """
    try:
        params = {"shopId": shop_id, "page": page, "pageSize": page_size}
        if keyword:
            params["keyword"] = keyword
        result = bc.get_public("/api/user/dishes", params=params)
        if result.get("code") != 200:
            return f"搜索失败：{result.get('message', '未知错误')}"
        data = result.get("data", {})
        records = data.get("records", [])
        total = data.get("total", 0)
        if not records:
            return f"未找到商家ID {shop_id} 的菜品" + (f"（关键词: {keyword}）" if keyword else "") + "。"
        lines = [f"商家ID {shop_id} 的菜品" + (f"（关键词: {keyword}）" if keyword else "") + f"（共 {total} 道，第 {page} 页）："]
        for d in records:
            name = d.get("name", "未知")
            price = d.get("price", 0)
            stock = d.get("stock", 0)
            dish_id = d.get("id", "")
            shop_name = d.get("shopName", "")
            lines.append(f"  - {name} | ¥{price} | 库存:{stock} | 菜品ID:{dish_id}" + (f" | 商家:{shop_name}" if shop_name else ""))
        if total > page * page_size:
            lines.append(f"\n还有更多菜品，可指定 page={page + 1} 查看下一页。")
        return "\n".join(lines)
    except Exception as e:
        return f"按商家搜索菜品失败：{str(e)}"


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
        "  匹配依据: " + "、".join(matched_keywords),
        "温馨提示：如果您有特殊饮食需求（如低盐、低糖、过敏等），请告知我，我帮您筛选合适的菜品。",
    ]
    return "\n".join(lines)


# ==================== Agent 桥接（商家接单 / 配送 / 订单状态） ====================
# 这些接口要求同时携带用户 JWT（证明操作者是本人）与 X-Agent-Key（证明调用方是 Agent）。

def agent_accept_order(session_id: str, order_id: int, user_id: int) -> str:
    """【多智能体-商家接单】确认订单并开始备餐（真实调用 Java 后端）。
    使用用户 JWT + Agent API Key 双重认证。

    参数:
        session_id: 当前会话ID
        order_id: 订单ID
        user_id: 用户ID

    返回:
        接单结果描述。
    """
    headers = {"X-Agent-Key": AGENT_API_KEY, "Content-Type": "application/json"}
    try:
        result = bc.post(
            f"/api/agent/orders/{order_id}/accept",
            session_id,
            body={"userId": user_id},
            extra_headers=headers,
        )
        data = bc.extract_data(result)
        return (
            f"商家已接单！订单编号 {data.get('orderNo', 'N/A')}\n"
            f"预计 {data.get('prepTimeMin', 20)} 分钟出餐，请耐心等待～"
        )
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"商家接单失败：{str(e)}"


def agent_start_delivery(session_id: str, order_id: int, user_id: int) -> str:
    """【多智能体-配送开始】标记订单进入配送状态（真实调用 Java 后端）。

    参数:
        session_id: 当前会话ID
        order_id: 订单ID
        user_id: 用户ID

    返回:
        配送信息描述（基于后端真实返回，不编造骑手信息）。
    """
    headers = {"X-Agent-Key": AGENT_API_KEY, "Content-Type": "application/json"}
    try:
        result = bc.post(
            f"/api/agent/orders/{order_id}/deliver",
            session_id,
            body={"userId": user_id},
            extra_headers=headers,
        )
        data = bc.extract_data(result)
        order_no = data.get("orderNo", "N/A")
        rider_phone = data.get("riderPhone")
        if rider_phone:
            return (
                f"配送员已取餐，正在为您配送！订单编号 {order_no}\n"
                f"配送员电话：{rider_phone}（脱敏显示）"
            )
        if data.get("message"):
            return f"订单 {order_no}：{data['message']}"
        return f"订单 {order_no} 已进入配送状态，正在等待骑手接单。"
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"配送启动失败：{str(e)}"


def agent_complete_order(session_id: str, order_id: int, user_id: int) -> str:
    """【多智能体-配送完成】标记订单已送达（真实调用 Java 后端）。

    参数:
        session_id: 当前会话ID
        order_id: 订单ID
        user_id: 用户ID

    返回:
        完成信息描述。
    """
    headers = {"X-Agent-Key": AGENT_API_KEY, "Content-Type": "application/json"}
    try:
        result = bc.post(
            f"/api/agent/orders/{order_id}/complete",
            session_id,
            body={"userId": user_id},
            extra_headers=headers,
        )
        data = bc.extract_data(result)
        return data.get("message", "订单已送达，祝您用餐愉快！")
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"订单完成失败：{str(e)}"


def agent_get_order_status(session_id: str, order_id: int) -> str:
    """查询订单的真实配送状态（真实调用 Java 后端）。

    参数:
        session_id: 当前会话ID
        order_id: 订单ID

    返回:
        订单编号、状态、骑手分配情况、关键时间点等真实信息。
    """
    headers = {"X-Agent-Key": AGENT_API_KEY}
    try:
        result = bc.get(
            f"/api/agent/orders/{order_id}/status",
            session_id,
            extra_headers=headers,
        )
        d = bc.extract_data(result)
        status_map = {
            "pending": "待接单", "accepted": "商家已接单",
            "delivering": "配送中", "completed": "已完成", "cancelled": "已取消",
        }
        order_no = d.get("orderNo", "N/A")
        status = status_map.get(d.get("status"), d.get("status", "未知"))
        delivery_status = d.get("deliveryStatus", "")
        lines = [
            f"订单 {order_no} 当前状态：{status}",
            f"  配送状态: {delivery_status or '未分配'}",
        ]
        if d.get("riderAssigned"):
            phone = d.get("riderPhone")
            lines.append(f"  骑手已接单，联系电话：{phone or '未知'}")
        else:
            lines.append("  暂未分配骑手，正在等待接单。")
        for label, key in [
            ("预计送达", "scheduledDeliveryTime"),
            ("分配时间", "assignedAt"),
            ("取餐时间", "pickedUpAt"),
            ("送达时间", "deliveredAt"),
        ]:
            val = d.get(key)
            if val:
                lines.append(f"  {label}: {val[:19]}")
        return "\n".join(lines)
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"查询订单状态失败：{str(e)}"
