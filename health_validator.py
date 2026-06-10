"""
健康数据校验器 —— 对语音录入的健康档案进行数值合理性检查。

校验规则:
  - 年龄: 10 ~ 100
  - 性别: 男/女
  - 身高: 100 ~ 250 cm
  - 体重: 30 ~ 200 kg
  - 活动水平: 低/中/高
  - 饮食偏好: 均衡/低脂/素食/高蛋白等
  - 健康目标: 减肥/增肌/维持等

模糊信息处理:
  当用户表达不够精确时（如"活动水平中等"→"中"），
  做归一化映射；无法映射时返回 None 触发 Agent 追问。
"""

# ==================== 合法值映射 ====================

VALID_GENDERS = {"男", "女"}

VALID_ACTIVITY_LEVELS = {
    "低", "中", "高",
    "轻度", "中等", "高度",  # 兼容口语表达
    "不怎么运动", "偶尔运动", "经常运动",
    "久坐", "一般", "活跃",
}

VALID_DIET_PREFERENCES = {
    "均衡", "低脂", "素食", "高蛋白",
    "低碳水", "清淡", "无偏好", "随便",
}

VALID_HEALTH_GOALS = {
    "减肥", "增肌", "维持", "保持健康",
    "减重", "减脂", "塑形",
}

# ==================== 归一化映射 ====================

ACTIVITY_LEVEL_MAP = {
    "低": "低", "轻度": "低", "不怎么运动": "低", "久坐": "低",
    "中": "中", "中等": "中", "偶尔运动": "中", "一般": "中",
    "高": "高", "高度": "高", "经常运动": "高", "活跃": "高",
}

DIET_PREFERENCE_MAP = {
    "均衡": "均衡", "无偏好": "均衡", "随便": "均衡",
    "低脂": "低脂", "清淡": "低脂",
    "素食": "素食",
    "高蛋白": "高蛋白",
    "低碳水": "低碳水",
}

HEALTH_GOAL_MAP = {
    "减肥": "减肥", "减重": "减肥", "减脂": "减肥",
    "增肌": "增肌", "塑形": "增肌",
    "维持": "维持", "保持健康": "维持",
}


def normalize_activity_level(raw: str) -> str | None:
    """将用户口语化的活动水平归一化为 低/中/高。"""
    if not raw:
        return None
    cleaned = raw.strip()
    return ACTIVITY_LEVEL_MAP.get(cleaned)


def normalize_diet_preference(raw: str) -> str | None:
    """将用户口语化的饮食偏好归一化。"""
    if not raw:
        return None
    cleaned = raw.strip()
    return DIET_PREFERENCE_MAP.get(cleaned)


def normalize_health_goal(raw: str) -> str | None:
    """将用户口语化的健康目标归一化。"""
    if not raw:
        return None
    cleaned = raw.strip()
    return HEALTH_GOAL_MAP.get(cleaned)


# ==================== 数值校验 ====================

class ValidationResult:
    """校验结果"""
    def __init__(self, valid: bool, message: str = "", normalized: dict | None = None):
        self.valid = valid
        self.message = message
        self.normalized = normalized or {}


def validate_health_profile(
    age: int | None = None,
    gender: str | None = None,
    height: float | None = None,
    weight: float | None = None,
    activity_level: str | None = None,
    diet_preference: str | None = None,
    health_goal: str | None = None,
) -> ValidationResult:
    """
    校验健康档案各字段。

    参数:
        age: 年龄（10-100）
        gender: 性别（男/女）
        height: 身高 cm（100-250）
        weight: 体重 kg（30-200）
        activity_level: 活动水平（低/中/高，自动归一化）
        diet_preference: 饮食偏好（均衡/低脂/素食等，自动归一化）
        health_goal: 健康目标（减肥/增肌/维持，自动归一化）

    返回:
        ValidationResult: 含 valid, message, normalized 字段
    """
    normalized = {}

    # 年龄校验
    if age is not None:
        if not isinstance(age, (int, float)) or age < 10 or age > 100:
            return ValidationResult(False, "年龄需在10到100之间，请重试")
        normalized["age"] = int(age)

    # 性别校验
    if gender is not None:
        gender_clean = gender.strip()
        if gender_clean not in VALID_GENDERS:
            return ValidationResult(False, "性别只能是男或女，请重试")
        normalized["gender"] = gender_clean

    # 身高校验 (cm)
    if height is not None:
        if not isinstance(height, (int, float)) or height < 100 or height > 250:
            return ValidationResult(False, "身高需在100到250厘米之间，请重试")
        normalized["height"] = float(height)

    # 体重校验 (kg)
    if weight is not None:
        if not isinstance(weight, (int, float)) or weight < 30 or weight > 200:
            return ValidationResult(False, "体重需在30到200公斤之间，请重试")
        normalized["weight"] = float(weight)

    # 活动水平归一化
    if activity_level is not None:
        level = normalize_activity_level(activity_level)
        if level is None:
            return ValidationResult(
                False,
                f"活动水平「{activity_level}」我不太确定，请问是低、中还是高？"
                f"比如：'不怎么运动'是低，'偶尔运动'是中，'经常运动'是高"
            )
        normalized["activity_level"] = level

    # 饮食偏好归一化
    if diet_preference is not None:
        pref = normalize_diet_preference(diet_preference)
        if pref is None:
            return ValidationResult(
                False,
                f"饮食偏好「{diet_preference}」我不太确定，"
                f"请问是均衡、低脂、素食、高蛋白还是低碳水？"
            )
        normalized["diet_preference"] = pref

    # 健康目标归一化
    if health_goal is not None:
        goal = normalize_health_goal(health_goal)
        if goal is None:
            return ValidationResult(
                False,
                f"健康目标「{health_goal}」我不太确定，请问是减肥、增肌还是维持健康？"
            )
        normalized["health_goal"] = goal

    return ValidationResult(True, "校验通过", normalized)


def calculate_bmi(height_cm: float, weight_kg: float) -> dict:
    """计算 BMI 并返回分类和健康建议。

    返回:
        {"bmi": 23.1, "category": "正常", "advice": "体重正常，继续保持"}
    """
    if not height_cm or not weight_kg or height_cm <= 0:
        return {"bmi": 0, "category": "数据不足", "advice": "请先录入身高和体重"}

    height_m = height_cm / 100
    bmi = round(weight_kg / (height_m * height_m), 1)

    if bmi < 18.5:
        category = "偏瘦"
        advice = "体重偏瘦，建议适当增加营养摄入"
    elif bmi < 24:
        category = "正常"
        advice = "体重正常，继续保持"
    elif bmi < 28:
        category = "偏胖"
        advice = "体重偏胖，适当控制饮食，增加运动"
    else:
        category = "肥胖"
        advice = "属于肥胖范围，建议控制饮食并咨询医生"

    return {"bmi": bmi, "category": category, "advice": advice}
