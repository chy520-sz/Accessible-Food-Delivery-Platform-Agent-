"""health_validator 单元测试（纯函数，无外部依赖）。"""

from health_validator import (
    calculate_bmi,
    normalize_activity_level,
    normalize_diet_preference,
    normalize_health_goal,
    validate_health_profile,
)


def test_valid_profile():
    result = validate_health_profile(
        age=30, gender="男", height=175, weight=70,
        activity_level="偶尔运动", diet_preference="清淡", health_goal="减脂",
    )
    assert result.valid
    assert result.normalized["age"] == 30
    assert result.normalized["activity_level"] == "中"
    assert result.normalized["diet_preference"] == "低脂"
    assert result.normalized["health_goal"] == "减肥"


def test_invalid_age():
    result = validate_health_profile(age=5)
    assert not result.valid


def test_invalid_gender():
    result = validate_health_profile(gender="其他")
    assert not result.valid


def test_unknown_activity_level():
    result = validate_health_profile(activity_level="超级活跃")
    assert not result.valid


def test_bmi_normal():
    r = calculate_bmi(175, 70)
    assert r["bmi"] == 22.9
    assert r["category"] == "正常"


def test_normalize_mappings():
    assert normalize_activity_level("久坐") == "低"
    assert normalize_diet_preference("随便") == "均衡"
    assert normalize_health_goal("保持健康") == "维持"
    assert normalize_activity_level("") is None
