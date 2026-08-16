"""
健康管理工具函数 —— 封装对 Java 后端健康模块 API 的 HTTP 调用。

所有需要认证的请求自动携带当前会话的 JWT token；统一走 backend_client。

⚠️ 注意：Java 后端健康管理 API（/api/user/health/*）当前可能尚未实现。
工具调用失败时会返回友好的错误提示，不影响点餐核心流程。

工具列表:
  - save_health_profile    — 保存/更新健康档案
  - get_health_profile     — 获取健康档案
  - record_weight          — 记录体重
  - get_weight_history     — 查询体重历史
  - save_diet_plan         — 保存饮食计划
  - get_active_diet_plan   — 获取当前有效饮食计划
  - get_diet_plan_history  — 查询饮食计划历史
"""

import json
import os

import backend_client as bc


# ==================== 健康档案工具 ====================

def save_health_profile(
    session_id: str,
    age: int,
    gender: str,
    height: float,
    weight: float,
    activity_level: str,
    diet_preference: str,
    health_goal: str,
) -> str:
    """保存或更新用户的健康档案。

    参数:
        age: 年龄（10-100）
        gender: 性别（男/女）
        height: 身高（cm）
        weight: 体重（kg）
        activity_level: 活动水平（低/中/高）
        diet_preference: 饮食偏好（均衡/低脂/素食等）
        health_goal: 健康目标（减肥/增肌/维持）

    返回:
        操作结果描述文本
    """
    try:
        body = {
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight,
            "activityLevel": activity_level,
            "dietPreference": diet_preference,
            "healthGoal": health_goal,
        }
        bc.post("/api/user/health/profile", session_id, body)
        output = {
            "action": "profile_save",
            "profile": {
                "age": age,
                "gender": gender,
                "height": height,
                "weight": weight,
                "activity_level": activity_level,
                "diet_preference": diet_preference,
                "health_goal": health_goal,
            },
            "message": (
                f"健康档案已保存：年龄{age}，性别{gender}，"
                f"身高{int(height)}厘米，体重{int(weight)}公斤，"
                f"活动水平{activity_level}，饮食偏好{diet_preference}，"
                f"健康目标{health_goal}"
            ),
        }
        return json.dumps(output, ensure_ascii=False)
    except PermissionError as e:
        return json.dumps({"action": "error", "message": str(e)}, ensure_ascii=False)
    except RuntimeError as e:
        return json.dumps({"action": "error", "message": str(e)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"action": "error", "message": f"保存健康档案失败：{str(e)}"}, ensure_ascii=False)


def get_health_profile(session_id: str) -> str:
    """获取用户的健康档案。

    返回:
        健康档案文本（JSON 格式）或未找到提示
    """
    try:
        result = bc.get("/api/user/health/profile", session_id)
        data = bc.extract_data(result)
        if not data:
            return json.dumps(
                {"action": "profile_not_found", "message": "您还没有创建健康档案。对我说'录入健康档案'来开始吧。"},
                ensure_ascii=False
            )
        profile = {
            "action": "profile_found",
            "profile": {
                "age": data.get("age"),
                "gender": data.get("gender"),
                "height": data.get("height"),
                "weight": data.get("weight"),
                "activity_level": data.get("activityLevel"),
                "diet_preference": data.get("dietPreference"),
                "health_goal": data.get("healthGoal"),
            },
            "message": (
                f"您的健康档案：年龄{data.get('age')}，性别{data.get('gender')}，"
                f"身高{int(data.get('height', 0))}厘米，体重{int(data.get('weight', 0))}公斤，"
                f"活动水平{data.get('activityLevel')}，饮食偏好{data.get('dietPreference')}，"
                f"健康目标{data.get('healthGoal')}"
            ),
        }
        return json.dumps(profile, ensure_ascii=False)
    except PermissionError as e:
        return json.dumps({"action": "error", "message": str(e)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"action": "error", "message": f"获取健康档案失败：{str(e)}"}, ensure_ascii=False)


# ==================== 体重记录工具 ====================

def record_weight(session_id: str, weight: float, record_date: str = "") -> str:
    """记录当前用户的体重。

    参数:
        weight: 体重（kg）
        record_date: 记录日期（yyyy-MM-dd），为空则默认当天

    返回:
        操作结果描述文本
    """
    try:
        body = {"weight": weight}
        if record_date:
            body["recordDate"] = record_date
        result = bc.post("/api/user/health/weight", session_id, body)
        data = bc.extract_data(result)
        record_date_str = data.get("recordDate", "今天")
        output = {
            "action": "weight_recorded",
            "weight": weight,
            "record_date": record_date_str,
            "message": f"体重已记录：{weight}公斤，日期{record_date_str}",
        }
        return json.dumps(output, ensure_ascii=False)
    except PermissionError as e:
        return json.dumps({"action": "error", "message": str(e)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"action": "error", "message": f"记录体重失败：{str(e)}"}, ensure_ascii=False)


def get_weight_history(session_id: str, weeks: int = 12) -> str:
    """查询用户的体重历史记录。

    参数:
        weeks: 查询最近几周的记录，默认 12 周（约 3 个月）

    返回:
        体重历史列表（JSON 格式）
    """
    try:
        result = bc.get("/api/user/health/weight", session_id, params={"weeks": weeks})
        records = bc.extract_data(result)
        if not records or (isinstance(records, list) and len(records) == 0):
            return json.dumps(
                {"action": "weight_history", "history": [], "message": "您还没有体重记录。对我说'记录体重'来开始吧。"},
                ensure_ascii=False
            )
        history_list = []
        for i, r in enumerate(records):
            history_list.append({
                "week": i + 1,
                "weight": r.get("weight"),
                "record_date": r.get("recordDate"),
            })
        # 生成语音友好的摘要
        summary_parts = []
        for h in history_list[:8]:  # 最多说 8 条
            summary_parts.append(f"第{h['week']}周{h['weight']}公斤")
        summary = "，".join(summary_parts)
        if len(history_list) > 8:
            summary += f"...共{len(history_list)}条记录"
        output = {
            "action": "weight_history",
            "history": history_list,
            "message": f"您最近{len(history_list)}周的体重记录：{summary}",
        }
        return json.dumps(output, ensure_ascii=False)
    except PermissionError as e:
        return json.dumps({"action": "error", "message": str(e)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"action": "error", "message": f"查询体重历史失败：{str(e)}"}, ensure_ascii=False)


# ==================== 饮食计划工具 ====================

def save_diet_plan(session_id: str, plan_data: str, week_start_date: str = "") -> str:
    """保存 AI 生成的饮食计划。

    参数:
        plan_data: 7 天饮食计划 JSON 字符串
        week_start_date: 计划周起始日期（yyyy-MM-dd），为空则默认当天

    返回:
        操作结果描述文本
    """
    try:
        body = {"planData": plan_data}
        if week_start_date:
            body["weekStartDate"] = week_start_date
        result = bc.post("/api/user/health/diet-plan", session_id, body)
        data = bc.extract_data(result)
        output = {
            "action": "diet_plan_saved",
            "plan_id": data.get("id"),
            "message": f"饮食计划已保存，计划ID：{data.get('id')}",
        }
        return json.dumps(output, ensure_ascii=False)
    except PermissionError as e:
        return json.dumps({"action": "error", "message": str(e)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"action": "error", "message": f"保存饮食计划失败：{str(e)}"}, ensure_ascii=False)


def get_active_diet_plan(session_id: str) -> str:
    """获取当前有效的饮食计划。

    返回:
        当前饮食计划 JSON 或未找到提示
    """
    try:
        result = bc.get("/api/user/health/diet-plan/active", session_id)
        data = bc.extract_data(result)
        if not data:
            return json.dumps(
                {"action": "diet_plan", "plan": None, "message": "您还没有饮食计划。对我说'生成饮食计划'来开始吧。"},
                ensure_ascii=False
            )
        plan_json = data.get("planData", "{}")
        # planData 可能是字符串也可能是已解析的 JSON
        if isinstance(plan_json, str):
            plan_obj = json.loads(plan_json)
        else:
            plan_obj = plan_json
        output = {
            "action": "diet_plan",
            "plan": plan_obj,
            "week_start_date": data.get("weekStartDate"),
            "message": "已找到您的本周饮食计划",
        }
        return json.dumps(output, ensure_ascii=False)
    except PermissionError as e:
        return json.dumps({"action": "error", "message": str(e)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"action": "error", "message": f"获取饮食计划失败：{str(e)}"}, ensure_ascii=False)


def get_diet_plan_history(session_id: str, limit: int = 10) -> str:
    """查询饮食计划历史列表。

    参数:
        limit: 返回条数，默认 10

    返回:
        饮食计划历史列表 JSON
    """
    try:
        result = bc.get("/api/user/health/diet-plan/history", session_id, params={"limit": limit})
        records = bc.extract_data(result)
        if not records or (isinstance(records, list) and len(records) == 0):
            return json.dumps(
                {"action": "diet_plan_history", "history": [], "message": "您还没有饮食计划历史记录。"},
                ensure_ascii=False
            )
        history_list = []
        for r in records:
            plan_data = r.get("planData", "{}")
            if isinstance(plan_data, str):
                try:
                    plan_data = json.loads(plan_data)
                except json.JSONDecodeError:
                    pass
            history_list.append({
                "id": r.get("id"),
                "plan": plan_data,
                "week_start_date": r.get("weekStartDate"),
                "status": r.get("status"),
                "created_at": r.get("createdAt"),
            })
        output = {
            "action": "diet_plan_history",
            "history": history_list,
            "message": f"您共有{len(history_list)}条饮食计划历史",
        }
        return json.dumps(output, ensure_ascii=False)
    except PermissionError as e:
        return json.dumps({"action": "error", "message": str(e)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"action": "error", "message": f"查询饮食计划历史失败：{str(e)}"}, ensure_ascii=False)


# ==================== 饮食计划数据源 ====================

def fetch_dishes_for_planning(session_id: str) -> str:
    """从 Java 后端获取全部可用菜品（结构化数据），供 AI 饮食计划生成使用。

    优先调用 Java 后端 API 获取实时菜品数据，
    如果后端不可用则回退到本地 dish_knowledge.json 知识库。

    返回:
        JSON 字符串: {"action":"dishes_for_planning","dishes":[...]}
        每个菜品包含: id, name, shop, price, category, calories, protein, carbs, fat, tags
    """
    dishes = []
    source = ""

    # 方案1: 从 Java 后端获取（实时数据）
    try:
        result = bc.get("/api/user/dishes", session_id)
        raw = bc.extract_data(result)
        records = []
        if isinstance(raw, list):
            records = raw
        elif isinstance(raw, dict) and "records" in raw:
            records = raw["records"]
        for d in records:
            dishes.append({
                "id": d.get("id"),
                "name": d.get("name", "未知"),
                "shop": d.get("shopName", "未知店铺"),
                "price": d.get("price", 0),
                "category": d.get("categoryName", ""),
                "description": (d.get("description", "") or "")[:80],
                "calories": None,  # Java API 不直接返回热量
                "protein": None,
                "carbs": None,
                "fat": None,
                "tags": [],
            })
        if dishes:
            source = "java_backend"
    except Exception:
        pass

    # 方案2: 回退到本地知识库 JSON（含完整营养数据）
    if not dishes:
        try:
            kb_path = os.path.join(os.path.dirname(__file__), "data", "dish_knowledge.json")
            if os.path.exists(kb_path):
                with open(kb_path, "r", encoding="utf-8") as f:
                    kb_dishes = json.load(f)
                for d in kb_dishes:
                    nut = d.get("nutrition", {})
                    dishes.append({
                        "id": d.get("dish_id"),
                        "name": d.get("name", "未知"),
                        "shop": d.get("shop", "未知店铺"),
                        "price": d.get("price", 0),
                        "category": d.get("category", ""),
                        "description": d.get("description", "")[:80],
                        "calories": nut.get("calories"),
                        "protein": nut.get("protein"),
                        "carbs": nut.get("carbs"),
                        "fat": nut.get("fat"),
                        "tags": d.get("tags", []),
                        "suitable_for": d.get("suitable_for", []),
                        "allergens": d.get("allergens", []),
                        "spicy_level": d.get("spicy_level", 0),
                    })
                source = "knowledge_base"
        except Exception:
            pass

    if not dishes:
        return json.dumps({"action": "error", "message": "暂时无法获取菜品数据，请稍后重试"}, ensure_ascii=False)

    return json.dumps({
        "action": "dishes_for_planning",
        "source": source,
        "count": len(dishes),
        "dishes": dishes,
    }, ensure_ascii=False)


def fetch_dietary_rules_for_goal(health_goal: str) -> str:
    """从本地饮食知识库中查找与用户健康目标匹配的饮食规则。

    参数:
        health_goal: 用户健康目标（减肥/增肌/维持）

    返回:
        JSON 字符串，包含匹配到的饮食规则
    """
    try:
        kb_path = os.path.join(os.path.dirname(__file__), "data", "dietary_knowledge.json")
        if not os.path.exists(kb_path):
            return json.dumps({"action": "no_rules", "rules": []}, ensure_ascii=False)

        with open(kb_path, "r", encoding="utf-8") as f:
            rules = json.load(f)

        # 关键词映射
        goal_keywords = {
            "减肥": ["减肥", "减脂", "低卡", "低热量", "轻食"],
            "增肌": ["增肌", "健身", "高蛋白"],
            "维持": ["均衡"],
        }
        keywords = goal_keywords.get(health_goal, [])

        matched = []
        for rule in rules:
            rule_kws = rule.get("keywords", [])
            if any(kw in " ".join(rule_kws) for kw in keywords):
                matched.append({
                    "condition": rule.get("condition", ""),
                    "restricted_foods": rule.get("restricted_foods", []),
                    "recommended_foods": rule.get("recommended_foods", []),
                    "advice": rule.get("advice", ""),
                })

        if not matched:
            # 没精确匹配就用通用饮食建议
            matched.append({
                "condition": "通用",
                "restricted_foods": ["油炸食品", "高糖食品"],
                "recommended_foods": ["蔬菜", "瘦肉", "鱼", "全谷物"],
                "advice": "保持饮食均衡，多摄入蔬菜和优质蛋白",
            })

        return json.dumps({
            "action": "dietary_rules",
            "health_goal": health_goal,
            "rules": matched,
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"action": "error", "message": f"获取饮食规则失败：{str(e)}"}, ensure_ascii=False)
