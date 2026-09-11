"""diet_planner 纯函数单元测试（模块级无外部依赖）。"""

from diet_planner import (
    analyze_weight_trend,
    calculate_daily_calories,
    filter_dishes_for_profile,
    format_plan_for_tts,
)


def test_calculate_daily_calories_male():
    r = calculate_daily_calories(30, "男", 175, 70, "中", "维持")
    assert r["target"] > 2000
    assert r["goal"] == "维持"


def test_calculate_daily_calories_floor():
    r = calculate_daily_calories(80, "女", 150, 40, "低", "减肥")
    assert r["target"] >= 1200


def test_weight_trend_insufficient_data():
    r = analyze_weight_trend([])
    assert r["trend"] == "数据不足"
    assert r["adjustment_needed"] == "maintain"


def test_weight_trend_direction():
    r = analyze_weight_trend([{"weight": 70}, {"weight": 69}, {"weight": 68}])
    assert r["trend"] == "下降"


def test_filter_vegetarian():
    dishes = [
        {"id": 1, "name": "宫保鸡丁", "tags": ["肉"], "description": ""},
        {"id": 2, "name": "清炒时蔬", "tags": ["素"], "description": ""},
    ]
    profile = {"diet_preference": "素食", "health_goal": "维持"}
    out = filter_dishes_for_profile(dishes, profile, [])
    assert [d["id"] for d in out] == [2]


def test_format_plan_for_tts():
    plan = {
        "周一": {
            "早餐": {"name": "燕麦粥"},
            "午餐": {"name": "鸡胸肉饭"},
            "晚餐": {"name": "清蒸鱼"},
        }
    }
    text = format_plan_for_tts('{"plan": ' + json_dumps(plan) + "}")
    assert "周一" in text
    assert "燕麦粥" in text


def json_dumps(obj):
    import json
    return json.dumps(obj, ensure_ascii=False)
