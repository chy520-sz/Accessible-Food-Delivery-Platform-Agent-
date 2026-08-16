"""
AI 饮食计划生成器 —— 从已有数据库和知识库中选取真实菜品，组合成周饮食计划。

核心流程:
  1. 解析用户健康档案 → 计算每日推荐热量 (Mifflin-St Jeor)
  2. 分析体重趋势 → 决定热量调整方向
  3. 读取已有菜品数据（Java 后端 / 知识库 JSON）→ 真实菜品池
  4. 读取饮食知识库 → 匹配健康目标的饮食规则
  5. 调用 DeepSeek LLM → 从菜品池中智能选取并排列成 7 天计划
  6. 格式化输出 → JSON（引用真实菜品 ID 和名称）

LLM 的角色: 从已有菜品中选择 + 排列，绝不编造不存在的菜品。

⚠️ 注意：本模块依赖 Java 后端健康管理 API 返回菜品数据。
如果后端健康 API 未实现，会回退到本地知识库 JSON。
"""

import json
import logging
import re

logger = logging.getLogger("diet_planner")


# ==================== 热量计算 ====================

def calculate_daily_calories(
    age: int, gender: str, height_cm: float, weight_kg: float,
    activity_level: str, health_goal: str,
) -> dict:
    """根据用户档案计算每日推荐热量 (Mifflin-St Jeor 公式)。"""
    if gender == "男":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

    activity_multipliers = {"低": 1.2, "中": 1.55, "高": 1.725}
    multiplier = activity_multipliers.get(activity_level, 1.55)
    tdee = bmr * multiplier

    goal_adjustments = {"减肥": -500, "增肌": 300, "维持": 0}
    adjustment = goal_adjustments.get(health_goal, 0)
    target = max(1200, int(tdee + adjustment))

    return {"bmr": int(bmr), "tdee": int(tdee), "target": target, "goal": health_goal}


# ==================== 体重趋势分析 ====================

def analyze_weight_trend(weight_history: list[dict]) -> dict:
    """分析体重变化趋势，返回趋势方向和调整建议。"""
    if not weight_history or len(weight_history) < 2:
        return {
            "trend": "数据不足", "change_per_week": 0, "total_change": 0,
            "is_healthy": True, "advice": "体重数据不足，建议先记录至少2周体重",
            "adjustment_needed": "maintain",
        }

    weights = [h["weight"] for h in weight_history]
    total_change = weights[0] - weights[-1]
    weeks = max(len(weight_history) - 1, 1)
    change_per_week = total_change / weeks

    if change_per_week > 0.5:
        trend = "下降"
    elif change_per_week < -0.3:
        trend = "上升"
    else:
        trend = "稳定"

    is_healthy = abs(change_per_week) <= 1.5

    if abs(change_per_week) > 1.5:
        if change_per_week > 0:
            adjustment_needed = "increase"
            advice = f"每周下降{abs(change_per_week):.1f}公斤，降速偏快，建议适当增加热量"
        else:
            adjustment_needed = "decrease"
            advice = f"每周上升{abs(change_per_week):.1f}公斤，增速偏快，建议控制饮食"
    else:
        adjustment_needed = "maintain"
        if trend == "下降":
            advice = f"每周下降{abs(change_per_week):.1f}公斤，属于健康范围"
        elif trend == "上升":
            advice = f"每周上升{abs(change_per_week):.1f}公斤，请关注体重变化"
        else:
            advice = "体重保持稳定"

    return {
        "trend": trend, "change_per_week": round(change_per_week, 1),
        "total_change": round(total_change, 1), "is_healthy": is_healthy,
        "advice": advice, "adjustment_needed": adjustment_needed,
    }


# ==================== 菜品筛选 ====================

def filter_dishes_for_profile(dishes: list[dict], profile: dict, dietary_rules: list[dict]) -> list[dict]:
    """根据用户档案和饮食规则，预筛选出适合的菜品池。

    筛选逻辑:
      - 根据饮食偏好过滤（素食 → 排除含肉类标签的菜品）
      - 根据健康目标过滤（减肥 → 排除油炸类、高热量类）
      - 根据饮食规则过滤（如糖尿病 → 排除高糖菜品）
      - 对通过的菜品打适用分（越高越推荐）
    """
    diet_preference = profile.get("diet_preference", "均衡")
    health_goal = profile.get("health_goal", "维持")

    # 收集所有需排除的食物类型
    restricted_keywords = set()
    recommended_keywords = set()
    for rule in dietary_rules:
        for food in rule.get("restricted_foods", []):
            restricted_keywords.add(food)
        for food in rule.get("recommended_foods", []):
            recommended_keywords.add(food)

    # 素食特殊处理
    if diet_preference == "素食":
        restricted_keywords.update(["肉", "鸡", "鸭", "鱼", "虾", "蟹", "猪", "牛", "羊", "排骨", "五花", "里脊"])
    elif diet_preference == "低脂":
        restricted_keywords.update(["油炸", "炸", "红烧肉", "五花", "肥", "奶油"])

    # 减肥目标排除高热量
    if health_goal == "减肥":
        restricted_keywords.update(["油炸", "炸", "红烧肉", "奶茶", "蛋糕", "含糖饮料"])

    filtered = []
    for dish in dishes:
        name = dish.get("name", "")
        tags = dish.get("tags", [])
        desc = dish.get("description", "")
        combined = name + " ".join(tags) + desc

        # 检查是否包含受限关键词
        is_restricted = False
        for kw in restricted_keywords:
            if kw in combined:
                is_restricted = True
                break
        if is_restricted:
            continue

        # 计算推荐分
        score = 0
        for kw in recommended_keywords:
            if kw in combined:
                score += 2
        # 偏好匹配加分
        if diet_preference == "高蛋白":
            protein = dish.get("protein") or 0
            if protein and protein >= 20:
                score += 3
        elif diet_preference == "低脂":
            fat = dish.get("fat") or 999
            if fat and fat <= 15:
                score += 3
        # 减肥目标：低热量加分
        if health_goal == "减肥":
            cal = dish.get("calories") or 999
            if cal and cal <= 500:
                score += 3
        # 增肌目标：高蛋白加分
        if health_goal == "增肌":
            protein = dish.get("protein") or 0
            if protein and protein >= 25:
                score += 3

        dish["_score"] = score
        filtered.append(dish)

    # 按分数降序排列
    filtered.sort(key=lambda d: d.get("_score", 0), reverse=True)
    return filtered


# ==================== LLM 选菜提示词 ====================

DIET_PLAN_PROMPT = """你是一位专业的营养师，需要为用户从已有菜品库中挑选并安排一周的健康饮食计划。

## 用户档案
- 年龄: {age}，性别: {gender}，身高: {height}cm，体重: {weight}kg
- 活动水平: {activity_level}，饮食偏好: {diet_preference}，健康目标: {health_goal}
- 每日目标热量: {target_kcal} 千卡

## 体重趋势
{trend_text}

## 饮食规则
{dietary_rules_text}

## 可用菜品池（共 {dish_count} 道，按推荐度排列）
{dishes_text}

## 你的任务
从上面的菜品池中，为用户挑选并安排周一至周日共7天的饮食计划。
每天包含: 早餐、午餐、晚餐、加餐（共4餐）

## 规则
1. **只能从上面列出的菜品中选择**，绝不编造不存在的菜品。每餐写 dish_id 和完整菜品名称。
2. 每日总热量尽量接近 {target_kcal}±200 千卡范围
3. 同一天内尽量避免同一店铺的菜品超过2个
4. 7天内菜品尽量不重复（同一道菜最多出现2次）
5. 考虑饮食偏好和健康目标选择合适的菜品
6. 计划末尾给出简短摘要（用于语音播报，每句不超过20字）

## 输出格式（严格 JSON，不要任何其他内容）
```json
{{
  "周一": {{
    "早餐": {{"dish_id": 6, "name": "番茄鸡蛋面", "calories": 450, "reason": "清淡开胃，适合早餐"}},
    "午餐": {{"dish_id": 4, "name": "鱼香肉丝饭", "calories": 620, "reason": "荤素搭配均衡"}},
    "晚餐": {{"dish_id": 14, "name": "清蒸鲈鱼套餐", "calories": 380, "reason": "低脂高蛋白，适合晚餐"}},
    "加餐": {{"dish_id": null, "name": "苹果1个", "calories": 80, "reason": "水果加餐补充维生素"}}
  }},
  ...
  "周日": {{...}},
  "plan_summary": "这是一个{health_goal}饮食计划，每日约{target_kcal}千卡。周一早餐番茄鸡蛋面...（7天概括）",
  "nutrition_notes": "此计划全部来自本平台真实菜品。实际热量以商家制作为准。如有特殊疾病请咨询医生。"
}}
```

**注意**: 加餐如果没有合适的数据库菜品，可以用常见水果/坚果代替（如苹果、香蕉、酸奶），dish_id 填 null。

请直接输出 JSON："""


# ==================== 主入口 ====================

def generate_weekly_diet_plan(profile_str: str, weight_history_str: str, dishes_str: str = "", dietary_rules_str: str = "") -> str:
    """生成一周饮食计划（基于真实菜品）。

    参数:
        profile_str: get_health_profile 返回的 JSON
        weight_history_str: get_weight_history 返回的 JSON
        dishes_str: fetch_dishes_for_planning 返回的 JSON（菜品池）
        dietary_rules_str: fetch_dietary_rules_for_goal 返回的 JSON（饮食规则）

    返回:
        JSON 字符串: {"action":"diet_plan","plan":{...},"message":"..."}
    """
    # 1. 解析健康档案
    try:
        profile_data = json.loads(profile_str)
        if profile_data.get("action") == "error":
            return profile_str
        if profile_data.get("action") == "profile_not_found":
            return json.dumps(
                {"action": "error", "message": "请先录入健康档案，对我说'录入健康档案'"},
                ensure_ascii=False
            )
        profile = profile_data.get("profile", {})
    except json.JSONDecodeError:
        return json.dumps({"action": "error", "message": "健康档案数据解析失败"}, ensure_ascii=False)

    # 2. 解析体重历史
    weight_history = []
    try:
        wdata = json.loads(weight_history_str)
        weight_history = wdata.get("history", []) if isinstance(wdata, dict) else []
    except (json.JSONDecodeError, AttributeError):
        pass

    # 3. 解析菜品池
    dishes = []
    try:
        ddata = json.loads(dishes_str) if dishes_str else {"dishes": []}
        dishes = ddata.get("dishes", []) if isinstance(ddata, dict) else []
    except (json.JSONDecodeError, AttributeError):
        pass

    if not dishes:
        return json.dumps(
            {"action": "error", "message": "暂无可用菜品数据。请确保菜品库已初始化。"},
            ensure_ascii=False
        )

    # 4. 解析饮食规则
    dietary_rules = []
    try:
        rdata = json.loads(dietary_rules_str) if dietary_rules_str else {"rules": []}
        dietary_rules = rdata.get("rules", []) if isinstance(rdata, dict) else []
    except (json.JSONDecodeError, AttributeError):
        pass

    # 5. 提取个人信息
    age = profile.get("age", 30)
    gender = profile.get("gender", "男")
    height = profile.get("height", 170)
    weight = profile.get("weight", 70)
    activity_level = profile.get("activity_level", "中")
    diet_preference = profile.get("diet_preference", "均衡")
    health_goal = profile.get("health_goal", "维持")

    # 6. 热量计算 + 趋势调整
    calories = calculate_daily_calories(age, gender, height, weight, activity_level, health_goal)
    trend = analyze_weight_trend(weight_history)
    if trend.get("adjustment_needed") == "increase":
        calories["target"] += 200
    elif trend.get("adjustment_needed") == "decrease":
        calories["target"] -= 200
    calories["target"] = max(1200, calories["target"])

    # 7. 预筛选菜品（根据档案和饮食规则）
    filtered_dishes = filter_dishes_for_profile(dishes, profile, dietary_rules)

    # 至少保留 30 道菜品供 LLM 选择
    candidate_dishes = filtered_dishes[:60]

    # 8. 构建 LLM 提示词
    # 趋势文本
    trend_text = f"趋势: {trend.get('trend','未知')}，{trend.get('advice','')}"

    # 饮食规则文本
    rules_lines = []
    for r in dietary_rules[:3]:
        rules_lines.append(f"- {r.get('condition','')}: {r.get('advice','')}")
        if r.get("recommended_foods"):
            rules_lines.append(f"  推荐: {', '.join(r['recommended_foods'][:6])}")
        if r.get("restricted_foods"):
            rules_lines.append(f"  避免: {', '.join(r['restricted_foods'][:6])}")
    dietary_rules_text = "\n".join(rules_lines) if rules_lines else "无特殊饮食规则"

    # 菜品文本（每道菜一行，含关键信息）
    dish_lines = []
    for d in candidate_dishes[:60]:
        cal_str = f"{d.get('calories')}kcal" if d.get('calories') else "?kcal"
        prot_str = f"蛋白{d.get('protein')}g" if d.get('protein') else ""
        dish_lines.append(
            f"  ID:{d.get('id')} | {d.get('name')} | {d.get('shop','')} | "
            f"{cal_str} | {prot_str} | {d.get('category','')} | "
            f"标签:{','.join(d.get('tags',[])[:3])}"
        )
    dishes_text = "\n".join(dish_lines)

    # 9. 调用 LLM
    prompt = DIET_PLAN_PROMPT.format(
        age=age, gender=gender, height=int(height), weight=int(weight),
        activity_level=activity_level, diet_preference=diet_preference,
        health_goal=health_goal, target_kcal=calories["target"],
        trend_text=trend_text, dietary_rules_text=dietary_rules_text,
        dish_count=len(candidate_dishes), dishes_text=dishes_text,
    )

    try:
        plan_json_str = _call_llm_for_plan(prompt)
        plan_obj = json.loads(plan_json_str)
    except Exception as e:
        logger.error(f"[diet_planner] LLM 失败: {e}")
        # 降级方案：用规则直接生成简单计划
        return _fallback_plan(candidate_dishes, calories, profile)

    # 10. 构建返回
    summary = plan_obj.pop("plan_summary", "")
    nutrition_notes = plan_obj.pop("nutrition_notes", "此计划全部来自本平台真实菜品，仅供参考。")
    tts_summary = format_plan_for_tts(json.dumps({"plan": plan_obj}, ensure_ascii=False))

    return json.dumps({
        "action": "diet_plan",
        "plan": plan_obj,
        "calories": calories,
        "trend_analysis": trend,
        "dish_source": "本平台菜品库",
        "message": summary or f"已为您生成{health_goal}饮食计划，每日约{calories['target']}千卡",
        "nutrition_notes": nutrition_notes,
        "tts_text": tts_summary,
    }, ensure_ascii=False)


# ==================== 降级方案：规则选菜 ====================

def _fallback_plan(candidate_dishes: list[dict], calories: dict, profile: dict) -> str:
    """当 LLM 调用失败时，用简单规则生成饮食计划（不需要 LLM）。

    规则:
      - 每天从菜品池中按推荐度取 3-4 道菜作为三餐+加餐
      - 早餐优先面食/粥类，午餐优先盖饭/套餐，晚餐优先蒸/煮类低热量
      - 7 天尽量不重复
    """
    target = calories["target"]
    health_goal = profile.get("health_goal", "维持")

    # 按类别分组
    breakfast_candidates = []
    lunch_candidates = []
    dinner_candidates = []
    snack_candidates = []

    for d in candidate_dishes:
        name = d.get("name", "")
        tags = " ".join(d.get("tags", []))
        cat = d.get("category", "")
        cal = d.get("calories") or 500

        if any(kw in name + tags for kw in ["面", "粥", "蛋", "包", "饼", "豆浆", "牛奶"]):
            if cal <= 500:
                breakfast_candidates.append(d)
        if any(kw in cat + name + tags for kw in ["主食", "盖饭", "套餐", "饭"]):
            lunch_candidates.append(d)
        if any(kw in name + tags for kw in ["蒸", "煮", "汤", "凉", "拌", "沙拉"]):
            if cal <= 500:
                dinner_candidates.append(d)
        else:
            dinner_candidates.append(d)  # 没有专门的晚餐菜就全部放进去
        if any(kw in name + tags + cat for kw in ["饮", "甜", "水果", "小食"]):
            snack_candidates.append(d)

    # 兜底
    if not breakfast_candidates:
        breakfast_candidates = [d for d in candidate_dishes if (d.get("calories") or 999) <= 500]
    if not lunch_candidates:
        lunch_candidates = candidate_dishes[:]
    if not dinner_candidates:
        dinner_candidates = [d for d in candidate_dishes if (d.get("calories") or 999) <= 600]
    if not snack_candidates:
        snack_candidates = candidate_dishes[:]

    days = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    used_ids = set()
    plan = {}

    for i, day in enumerate(days):
        def pick(pool, max_cal, prefer_low_fat=False):
            for d in pool:
                did = d.get("id")
                if did is not None and did in used_ids:
                    continue
                cal = d.get("calories") or 500
                if cal <= max_cal:
                    used_ids.add(did)
                    return {"dish_id": did, "name": d.get("name", ""),
                            "calories": cal, "reason": "根据您的档案自动推荐"}
            # 放宽条件
            for d in pool:
                did = d.get("id")
                if did is not None and did in used_ids:
                    continue
                used_ids.add(did)
                return {"dish_id": did, "name": d.get("name", ""),
                        "calories": d.get("calories") or 500, "reason": "根据您的档案自动推荐"}
            return {"dish_id": None, "name": "建议自行搭配", "calories": 0, "reason": "菜品不足"}

        bf = pick(breakfast_candidates, int(target * 0.3))
        lu = pick(lunch_candidates, int(target * 0.4))
        di = pick(dinner_candidates, int(target * 0.3))
        sn = {"dish_id": None, "name": "苹果1个" if health_goal == "减肥" else "酸奶1杯",
              "calories": 80, "reason": "常见加餐"}

        # 换用不同的字段名以兼容前端
        plan[day] = {
            "早餐": bf, "午餐": lu, "晚餐": di, "加餐": sn
        }

    return json.dumps({
        "action": "diet_plan",
        "plan": plan,
        "calories": calories,
        "message": f"已为您自动生成{health_goal}饮食计划（系统自动编排，未使用AI优化）",
        "nutrition_notes": "此计划通过规则自动生成。如需更智能的搭配，请稍后重试。",
    }, ensure_ascii=False)


# ==================== LLM 调用 ====================

def _call_llm_for_plan(prompt: str) -> str:
    """调用 DeepSeek LLM 从菜品池中选菜并生成计划 JSON。"""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    from config import (
        DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL,
        LLM_REQUEST_TIMEOUT, MAX_LLM_TOKENS,
    )
    from llm_client import get_http_clients

    # 复用共享 httpx 客户端，避免每次生成计划都新建连接池
    http_client, async_http_client = get_http_clients()

    llm = ChatOpenAI(
        model=LLM_MODEL, temperature=0.3, max_tokens=MAX_LLM_TOKENS * 2,
        api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL,
        streaming=False, request_timeout=LLM_REQUEST_TIMEOUT,
        http_client=http_client, http_async_client=async_http_client,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content

    # 清理 markdown 代码块
    content = re.sub(r'^```json\s*', '', content.strip())
    content = re.sub(r'\s*```$', '', content.strip())
    content = content.strip()

    # 验证 JSON
    try:
        json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            content = match.group(0)
        else:
            raise ValueError(f"LLM 返回无法解析: {content[:200]}")

    return content


# ==================== 计划格式化 ====================

def format_plan_for_tts(plan_json_str: str) -> str:
    """将饮食计划 JSON 格式化为适合 TTS 播报的简短文本。"""
    try:
        data = json.loads(plan_json_str)
        plan = data.get("plan", data)
    except json.JSONDecodeError:
        return "计划数据解析失败"

    if not plan:
        return "暂无饮食计划"

    days = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    lines = ["您的一周饮食计划："]
    for day in days:
        if day in plan:
            meals = plan[day]
            bf = meals.get("早餐", {})
            lu = meals.get("午餐", {})
            lines.append(f"{day}：早餐{bf.get('name','')}，午餐{lu.get('name','')}")
            if len(lines) >= 8:
                break
    return "。".join(lines)
