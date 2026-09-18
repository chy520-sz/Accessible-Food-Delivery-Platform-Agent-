"""
菜品知识库字段补全脚本 —— 为 backend_sync 同步的菜品智能补全所有空缺字段。

补全字段: taste, cooking_method, main_ingredients, description,
          suitable_for, tags, allergens, spicy_level, nutrition

规则: 基于菜品名称关键词 + 分类 + 店铺，用规则引擎推断。
用法: python fill_dish_fields.py
"""

import json
import os
import re
import random
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DISH_KB_PATH = os.path.join(DATA_DIR, "dish_knowledge.json")

# ==================== 食材关键词库 ====================

INGREDIENT_KEYWORDS = {
    # 肉类
    "鸡肉": ["鸡", "鸡腿", "鸡翅", "鸡胸", "鸡排", "鸡块", "鸡丝", "鸡丁", "土鸡", "母鸡", "公鸡", "凤爪", "鸡爪"],
    "牛肉": ["牛", "牛肉", "牛排", "牛腩", "牛腱", "牛舌", "肥牛", "嫩牛", "黑椒牛", "牛仔骨"],
    "猪肉": ["猪", "猪肉", "排骨", "五花肉", "里脊", "肉丝", "肉片", "肉末", "猪蹄", "猪耳", "猪头", "腊肠", "培根", "火腿", "香肠", "叉烧"],
    "羊肉": ["羊", "羊肉", "羊排", "羊蝎", "羊腿", "肥羊"],
    "鸭肉": ["鸭", "鸭肉", "烤鸭", "烧鸭", "白切鸭", "鸭脚", "鸭翅", "鸭脖"],
    "鹅肉": ["鹅", "鹅肉", "烧鹅", "卤鹅"],
    # 海鲜
    "鱼": ["鱼", "鱼片", "鱼头", "鱼丸", "鱼香", "鲈鱼", "草鱼", "鲤鱼", "鲫鱼", "鳕鱼", "三文鱼", "金枪鱼", "鳗鱼", "黑鱼", "鲶鱼", "武昌鱼", "带鱼"],
    "虾": ["虾", "虾仁", "虾滑", "鲜虾", "大虾", "基围虾", "小龙虾", "皮皮虾"],
    "蟹": ["蟹", "螃蟹", "蟹黄", "蟹肉", "大闸蟹"],
    "贝类": ["贝", "扇贝", "生蚝", "牡蛎", "花甲", "花蛤", "蛏子", "海螺"],
    "海鲜": ["海鲜", "鱿鱼", "章鱼", "墨鱼", "海参", "海胆", "鲍鱼", "龙虾"],
    # 素食
    "豆腐": ["豆腐", "豆干", "腐竹", "豆皮", "豆花", "豆浆", "纳豆"],
    "鸡蛋": ["蛋", "鸡蛋", "鸭蛋", "皮蛋", "咸蛋", "蛋花", "炒蛋", "煎蛋", "煮蛋", "玉子"],
    "蔬菜": ["蔬菜", "青菜", "白菜", "生菜", "菠菜", "油菜", "芹菜", "韭菜", "香菜", "葱", "姜", "蒜", "洋葱", "番茄", "西红柿", "土豆", "马铃薯", "红薯", "地瓜", "山药", "芋头", "藕", "莲藕", "萝卜", "胡萝卜", "黄瓜", "丝瓜", "冬瓜", "南瓜", "苦瓜", "茄子", "辣椒", "青椒", "红椒", "玉米", "青豆", "豌豆", "蚕豆", "豆角", "四季豆", "扁豆", "空心菜", "油麦菜", "茼蒿", "苋菜", "蕨菜", "竹笋", "笋", "茭白", "荸荠", "菱角", "菌菇", "蘑菇", "香菇", "平菇", "金针菇", "杏鲍菇", "茶树菇", "草菇", "口蘑", "牛肝菌", "羊肚菌", "木耳", "银耳", "海带", "紫菜", "裙带菜"],
    # 主食
    "米饭": ["米饭", "炒饭", "盖饭", "盖浇饭", "煲仔饭", "拌饭", "寿司", "饭团", "粥", "稀饭"],
    "面条": ["面", "面条", "拉面", "拌面", "汤面", "炒面", "烩面", "刀削面", "牛肉面", "肥肠面", "阳春面", "燃面", "螺蛳粉", "米粉", "米线", "河粉", "粿条", "意大利面", "意面", "通心粉", "千层面"],
    "面包": ["面包", "吐司", "汉堡", "三明治", "披萨", "饼", "馅饼", "葱油饼", "手抓饼", "煎饼", "烙饼", "烧饼", "锅盔", "肉夹馍", "包子", "馒头", "花卷", "饺子", "馄饨", "烧麦", "春卷", "肠粉", "蛋挞", "蛋糕", "饼干", "曲奇", "泡芙", "慕斯", "布朗尼", "华夫饼", "班戟", "小贝"],
    # 奶制品
    "牛奶": ["牛奶", "鲜奶", "酸奶", "奶酪", "芝士", "奶油", "黄油", "炼乳", "奶昔", "奶茶", "双皮奶", "姜撞奶", "布丁", "冰淇淋", "圣代", "绵绵冰"],
    # 坚果
    "坚果": ["花生", "瓜子", "核桃", "杏仁", "腰果", "开心果", "榛子", "松子", "板栗", "莲子", "芝麻"],
    # 水果
    "水果": ["水果", "苹果", "香蕉", "橙子", "橘子", "柠檬", "草莓", "蓝莓", "芒果", "西瓜", "哈密瓜", "葡萄", "桃子", "梨", "菠萝", "火龙果", "猕猴桃", "樱桃", "荔枝", "龙眼", "椰子", "榴莲", "牛油果", "柚子", "柿子", "枣", "山楂", "杨梅", "石榴"],
}

# ==================== 口味规则 ====================

TASTE_PATTERNS = [
    (r"麻辣|麻辣烫|麻辣香锅|麻辣拌", "麻辣浓郁"),
    (r"变态辣|爆辣|特辣|魔鬼辣", "劲爆特辣"),
    (r"香辣|辣子|剁椒|小米辣|野山椒|泡椒", "香辣开胃"),
    (r"酸辣|酸汤|酸菜鱼|酸辣粉", "酸辣爽口"),
    (r"甜辣|韩式|炸鸡", "甜辣酥脆"),
    (r"糖醋|蜜汁|叉烧|照烧|咕咾", "酸甜可口"),
    (r"香甜|甜|糖水|红豆|绿豆|芋圆|烧仙草", "香甜软糯"),
    (r"咸鲜|酱香|卤|酱|红烧|黄焖", "咸鲜酱香"),
    (r"清淡|清蒸|白灼|白切|盐水|水煮(?!鱼)|原味", "清淡鲜美"),
    (r"蒜香|蒜蓉|蒜泥", "蒜香浓郁"),
    (r"葱香|葱油|葱烧", "葱香扑鼻"),
    (r"孜然|烧烤|烤肉|烤串", "孜然香辣"),
    (r"黑椒|黑胡椒", "黑椒辛香"),
    (r"咖喱", "咖喱浓郁"),
    (r"酸|酸爽|柠檬|醋", "酸爽开胃"),
    (r"苦|苦瓜|咖啡|美式", "微苦回甘"),
    (r"鲜|海鲜|鱼|虾|蟹|贝", "鲜美多汁"),
    (r"奶香|芝士|奶酪|奶油", "奶香浓郁"),
    (r"抹茶", "抹茶清香"),
    (r"巧克力|可可", "巧克力浓郁"),
    (r"焦糖", "焦糖香甜"),
    (r"薄荷|清凉|冰", "清凉爽口"),
]

# 分类默认口味
CATEGORY_DEFAULT_TASTE = {
    "中式炒菜": "咸鲜入味",
    "粉面粥点": "汤鲜面滑",
    "米饭盖浇": "酱香浓郁",
    "西餐汉堡": "层次丰富",
    "炸鸡炸串": "外酥里嫩",
    "火锅冒菜": "麻辣鲜香",
    "烧烤烤肉": "孜然香辣",
    "轻食沙拉": "清爽低卡",
    "寿司日料": "清淡鲜美",
    "披萨意面": "芝士浓郁",
    "包子面点": "暄软鲜香",
    "饺子馄饨": "皮薄馅大",
    "卤味熟食": "酱香浓郁",
    "凉菜冷盘": "清爽开胃",
    "汤品炖品": "鲜美滋补",
    "咖啡茶饮": "香醇回甘",
    "果汁鲜奶": "清甜爽口",
    "冰淇淋": "绵密香甜",
    "甜品点心": "香甜软糯",
    "早餐精选": "清淡营养",
    "夜宵专区": "香辣过瘾",
    "热销推荐": "招牌美味",
    "小食饮品": "香脆可口",
    "主食套餐": "丰盛满足",
}

# ==================== 烹饪方式规则 ====================

COOKING_PATTERNS = [
    (r"爆炒|小炒|炒|翻炒|滑炒", "大火爆炒"),
    (r"红烧|烧|烩|焖", "红烧焖煮"),
    (r"清蒸|蒸|炖|盅", "清蒸慢炖"),
    (r"水煮|涮|冒菜|火锅", "水煮涮烫"),
    (r"油炸|炸|酥|脆皮|香酥", "高温油炸"),
    (r"烤|烧烤|烤肉|烤串|焗", "明火烤制"),
    (r"煎|铁板|石锅", "煎制"),
    (r"煲|砂锅|瓦罐", "煲制"),
    (r"凉拌|拌|捞", "凉拌"),
    (r"卤|酱|卤味", "卤制"),
    (r"披萨|焗饭|千层面", "烤箱烘焙"),
    (r"汉堡|三明治|肉夹馍|包子|馒头|花卷", "手工制作"),
    (r"寿司|刺身|饭团", "手工制作"),
    (r"奶茶|茶|咖啡|饮品|果汁|奶昔", "现调"),
    (r"冰淇淋|冰沙|绵绵冰|圣代", "冷冻制作"),
    (r"蛋糕|面包|饼干|蛋挞|泡芙|慕斯|布朗尼|华夫|班戟|小贝|曲奇", "烘焙"),
    (r"粥|稀饭", "慢火熬煮"),
    (r"面|粉|米线|河粉", "煮制"),
    (r"饭|炒饭|盖饭|煲仔饭|拌饭", "烹制"),
    (r"饺子|馄饨|烧麦|春卷|肠粉", "蒸制"),
    (r"汤|炖品|羹", "慢炖"),
]

CATEGORY_DEFAULT_COOKING = {
    "中式炒菜": "大火爆炒",
    "粉面粥点": "煮制",
    "米饭盖浇": "烹制",
    "西餐汉堡": "煎制+组装",
    "炸鸡炸串": "高温油炸",
    "火锅冒菜": "火锅涮煮",
    "烧烤烤肉": "明火烤制",
    "轻食沙拉": "凉拌",
    "寿司日料": "手工制作",
    "披萨意面": "烤箱烘焙",
    "包子面点": "蒸制",
    "饺子馄饨": "煮制",
    "卤味熟食": "卤制",
    "凉菜冷盘": "凉拌",
    "汤品炖品": "慢炖",
    "咖啡茶饮": "现调",
    "果汁鲜奶": "鲜榨/调制",
    "冰淇淋": "冷冻制作",
    "甜品点心": "烘焙",
    "早餐精选": "蒸制/煮制",
    "夜宵专区": "烤制/卤制",
    "热销推荐": "招牌烹制",
    "小食饮品": "炸制/调制",
    "主食套餐": "组合烹制",
}

# ==================== 辣度规则 ====================

SPICY_PATTERNS = [
    (r"变态辣|魔鬼辣|爆辣", 5),
    (r"特辣|重辣|超辣", 4),
    (r"麻辣|香辣|辣子|剁椒|小米辣|野山椒|泡椒|酸辣|辣|川味|湘味|重庆|四川|湖南|火锅|冒菜", 3),
    (r"微辣|少辣|轻辣", 1),
]

CATEGORY_DEFAULT_SPICY = {
    "中式炒菜": 2,
    "火锅冒菜": 3,
    "烧烤烤肉": 2,
    "夜宵专区": 2,
    "粉面粥点": 1,
}

# ==================== 过敏原规则 ====================

ALLERGEN_RULES = [
    ("花生", r"花生|花生酱|宫保|怪味|挂霜"),
    ("海鲜", r"鱼|虾|蟹|贝|海鲜|鱿鱼|章鱼|墨鱼|海参|鲍鱼|龙虾|生蚝|牡蛎|花甲|花蛤|蛏子|海螺|扇贝"),
    ("牛奶", r"牛奶|鲜奶|酸奶|奶酪|芝士|奶油|黄油|炼乳|奶昔|奶茶|双皮奶|姜撞奶|布丁|冰淇淋|圣代|绵绵冰|慕斯|提拉米苏"),
    ("鸡蛋", r"蛋|鸡蛋|鸭蛋|皮蛋|咸蛋|蛋花|炒蛋|煎蛋|煮蛋|玉子|蛋挞|蛋糕|布朗尼|泡芙|华夫|班戟|小贝|曲奇|饼干"),
    ("小麦", r"面|面条|面包|吐司|汉堡|三明治|披萨|饼|馅饼|葱油饼|手抓饼|煎饼|烙饼|烧饼|锅盔|肉夹馍|包子|馒头|花卷|饺子|馄饨|烧麦|春卷|肠粉|蛋挞|蛋糕|饼干|曲奇|泡芙|慕斯|布朗尼|华夫|班戟|小贝|意面|通心粉|千层面"),
    ("大豆", r"大豆|豆腐|豆干|腐竹|豆皮|豆花|豆浆|纳豆|酱油|味噌|毛豆|黄豆|红豆|绿豆"),
    ("坚果", r"花生|瓜子|核桃|杏仁|腰果|开心果|榛子|松子|板栗|莲子|芝麻|坚果"),
]

# ==================== 标签规则 ====================

CATEGORY_TAGS = {
    "中式炒菜": ["热菜", "下饭菜"],
    "粉面粥点": ["主食", "汤粉"],
    "米饭盖浇": ["主食", "盖饭"],
    "西餐汉堡": ["西餐", "快餐"],
    "炸鸡炸串": ["油炸", "小食"],
    "火锅冒菜": ["火锅", "聚餐"],
    "烧烤烤肉": ["烧烤", "夜宵"],
    "轻食沙拉": ["轻食", "低卡"],
    "寿司日料": ["日料", "生鲜"],
    "披萨意面": ["意餐", "芝士"],
    "包子面点": ["面点", "早餐"],
    "饺子馄饨": ["主食", "汤饺"],
    "卤味熟食": ["卤味", "下酒菜"],
    "凉菜冷盘": ["凉菜", "开胃"],
    "汤品炖品": ["汤品", "滋补"],
    "咖啡茶饮": ["饮品", "下午茶"],
    "果汁鲜奶": ["饮品", "鲜榨"],
    "冰淇淋": ["甜品", "冰品"],
    "甜品点心": ["甜品", "下午茶"],
    "早餐精选": ["早餐", "营养"],
    "夜宵专区": ["夜宵", "解馋"],
    "热销推荐": ["招牌", "热销"],
    "小食饮品": ["小食", "搭配"],
    "主食套餐": ["套餐", "丰盛"],
}

# ==================== 适合人群规则 ====================

def infer_suitable_for(name: str, category: str, spicy: int, tags: list) -> list:
    suitable = []
    if spicy >= 3:
        suitable.append("喜欢辣味的人")
    elif spicy == 0 and category in ["轻食沙拉", "寿司日料", "汤品炖品", "早餐精选"]:
        suitable.append("老人小孩")
        suitable.append("肠胃敏感者")
    if "低卡" in tags or "轻食" in tags:
        suitable.append("减肥健身人群")
    if category in ["咖啡茶饮"]:
        suitable.append("需要提神的人")
        suitable.append("下午茶爱好者")
    if category in ["甜品点心", "冰淇淋"]:
        suitable.append("喜欢甜食的人")
    if category in ["火锅冒菜", "烧烤烤肉", "夜宵专区"]:
        suitable.append("朋友聚餐")
    if category in ["早餐精选", "包子面点"]:
        suitable.append("上班族早餐")
    if not suitable:
        suitable.append("大众口味")
    # 去重
    return list(dict.fromkeys(suitable))

# ==================== 营养估算规则 ====================

def estimate_nutrition(name: str, category: str, price: float) -> dict:
    """根据菜品类型和价格估算营养（每100g或每份）。"""
    # 基础热量按价格区间估算（价格越高通常分量越大/食材越好）
    base_cal = 150 + price * 3  # 基础热量

    # 按分类调整
    category_factors = {
        "中式炒菜": (1.2, "protein", 1.0),
        "粉面粥点": (1.0, "carbs", 1.5),
        "米饭盖浇": (1.3, "carbs", 1.4),
        "西餐汉堡": (1.4, "protein", 1.2),
        "炸鸡炸串": (1.8, "fat", 1.5),
        "火锅冒菜": (1.1, "protein", 0.8),
        "烧烤烤肉": (1.3, "protein", 0.9),
        "轻食沙拉": (0.4, "protein", 0.5),
        "寿司日料": (0.9, "protein", 1.0),
        "披萨意面": (1.3, "carbs", 1.3),
        "包子面点": (1.1, "carbs", 1.1),
        "饺子馄饨": (1.0, "protein", 1.0),
        "卤味熟食": (1.2, "protein", 0.7),
        "凉菜冷盘": (0.6, "protein", 0.4),
        "汤品炖品": (0.5, "protein", 0.3),
        "咖啡茶饮": (0.3, "carbs", 0.2),
        "果汁鲜奶": (0.4, "carbs", 0.3),
        "冰淇淋": (0.8, "fat", 0.8),
        "甜品点心": (1.0, "carbs", 1.0),
        "早餐精选": (0.9, "carbs", 0.8),
        "夜宵专区": (1.2, "protein", 0.9),
        "热销推荐": (1.1, "protein", 1.0),
        "小食饮品": (0.8, "fat", 0.6),
        "主食套餐": (1.4, "carbs", 1.2),
    }

    factor, main_macro, macro_factor = category_factors.get(category, (1.0, "protein", 1.0))
    calories = int(base_cal * factor)

    # 三大营养素估算
    if main_macro == "protein":
        protein = int(calories * 0.25 / 4)
        carbs = int(calories * 0.35 / 4)
        fat = int(calories * 0.40 / 9)
    elif main_macro == "carbs":
        protein = int(calories * 0.15 / 4)
        carbs = int(calories * 0.55 / 4)
        fat = int(calories * 0.30 / 9)
    else:  # fat
        protein = int(calories * 0.10 / 4)
        carbs = int(calories * 0.30 / 4)
        fat = int(calories * 0.60 / 9)

    return {
        "calories": calories,
        "protein": protein,
        "carbs": carbs,
        "fat": fat,
    }

# ==================== 描述生成 ====================

DESC_TEMPLATES = {
    "中式炒菜": "精选{ingredients}，大火爆炒，{taste}，锅气十足，配米饭超下饭。",
    "粉面粥点": "{taste}，面条劲道，汤底浓郁，配料丰富，一碗下肚满足感满满。",
    "米饭盖浇": "米饭上铺满满{ingredients}，{taste}，酱汁浓郁，拌匀后每一粒米都裹满滋味。",
    "西餐汉堡": "{taste}，食材新鲜，层次丰富，搭配薯条和饮料，经典西式美味。",
    "炸鸡炸串": "高温炸至金黄酥脆，外酥里嫩，{taste}，搭配蘸酱更美味，追剧小食首选。",
    "火锅冒菜": "{taste}，食材丰富，涮煮后蘸上秘制蘸料，热气腾腾，聚餐必备。",
    "烧烤烤肉": "明火烤制，{taste}，肉香四溢，搭配啤酒更过瘾，夜宵解馋首选。",
    "轻食沙拉": "新鲜{ingredients}，{taste}，低卡健康，营养均衡，健身减脂好选择。",
    "寿司日料": "精选{ingredients}，{taste}，食材新鲜，摆盘精致，一口一个满足感。",
    "披萨意面": "{taste}，芝士拉丝，面条劲道，配料丰富，意式风情满满。",
    "包子面点": "手工制作，{taste}，皮薄馅大，暄软可口，早餐配豆浆绝配。",
    "饺子馄饨": "手工包制，{taste}，皮薄馅大，汤鲜味美，一口一个超满足。",
    "卤味熟食": "秘制卤汁慢卤，{taste}，入味十足，越嚼越香，下酒追剧好搭档。",
    "凉菜冷盘": "{taste}，清爽开胃，食材新鲜，解腻爽口，餐前小食首选。",
    "汤品炖品": "慢火细炖，{taste}，汤鲜味美，营养滋补，暖胃又暖心。",
    "咖啡茶饮": "现点现调，{taste}，香气浓郁，口感醇厚，提神醒脑，下午茶必备。",
    "果汁鲜奶": "新鲜水果现榨，{taste}，清甜爽口，营养丰富，健康饮品好选择。",
    "冰淇淋": "{taste}，绵密丝滑，入口即化，香甜可口，夏日解暑神器。",
    "甜品点心": "精致烘焙，{taste}，香甜软糯，层次丰富，下午茶配咖啡绝配。",
    "早餐精选": "{taste}，营养丰富，搭配合理，元气满满开启新的一天。",
    "夜宵专区": "{taste}，香辣过瘾，分量十足，深夜解馋，满足感爆棚。",
    "热销推荐": "招牌必点，{taste}，人气爆款，好评如潮，闭眼点不踩雷。",
    "小食饮品": "{taste}，香脆可口，搭配主食更美味，追剧解馋小能手。",
    "主食套餐": "丰盛搭配，{taste}，主食+小食+饮品一应俱全，一份吃饱吃好。",
}

def generate_description(name: str, category: str, taste: str, ingredients: list) -> str:
    template = DESC_TEMPLATES.get(category, DESC_TEMPLATES["热销推荐"])
    ing_str = "、".join(ingredients[:3]) if ingredients else "精选食材"
    return template.format(ingredients=ing_str, taste=taste)

# ==================== 主补全逻辑 ====================

def extract_ingredients(name: str, category: str) -> list:
    """从菜品名称中提取主要食材。"""
    found = []
    for ingredient, keywords in INGREDIENT_KEYWORDS.items():
        for kw in keywords:
            if kw in name:
                found.append(ingredient)
                break
    # 如果没提取到，按分类给默认
    if not found:
        category_defaults = {
            "中式炒菜": ["蔬菜"],
            "粉面粥点": ["面条"],
            "米饭盖浇": ["米饭"],
            "西餐汉堡": ["牛肉", "面包"],
            "炸鸡炸串": ["鸡肉"],
            "火锅冒菜": ["蔬菜", "牛肉"],
            "烧烤烤肉": ["牛肉"],
            "轻食沙拉": ["蔬菜"],
            "寿司日料": ["米饭", "海鲜"],
            "披萨意面": ["芝士", "面条"],
            "包子面点": ["面粉"],
            "饺子馄饨": ["猪肉", "面粉"],
            "卤味熟食": ["牛肉"],
            "凉菜冷盘": ["蔬菜"],
            "汤品炖品": ["蔬菜"],
            "咖啡茶饮": ["茶叶"],
            "果汁鲜奶": ["水果", "牛奶"],
            "冰淇淋": ["牛奶"],
            "甜品点心": ["面粉", "鸡蛋"],
            "早餐精选": ["鸡蛋", "面粉"],
            "夜宵专区": ["牛肉"],
            "热销推荐": ["精选食材"],
            "小食饮品": ["鸡肉"],
            "主食套餐": ["米饭", "鸡肉"],
        }
        found = category_defaults.get(category, ["精选食材"])
    return list(dict.fromkeys(found))  # 去重保序

def infer_taste(name: str, category: str) -> str:
    for pattern, taste in TASTE_PATTERNS:
        if re.search(pattern, name):
            return taste
    return CATEGORY_DEFAULT_TASTE.get(category, "美味可口")

def infer_cooking(name: str, category: str) -> str:
    for pattern, cooking in COOKING_PATTERNS:
        if re.search(pattern, name):
            return cooking
    return CATEGORY_DEFAULT_COOKING.get(category, "烹制")

def infer_spicy(name: str, category: str) -> int:
    for pattern, level in SPICY_PATTERNS:
        if re.search(pattern, name):
            return level
    return CATEGORY_DEFAULT_SPICY.get(category, 0)

def infer_allergens(name: str, ingredients: list) -> list:
    allergens = []
    for allergen, pattern in ALLERGEN_RULES:
        if re.search(pattern, name):
            allergens.append(allergen)
    # 额外根据食材推断
    if "海鲜" in ingredients or "鱼" in ingredients or "虾" in ingredients or "蟹" in ingredients or "贝类" in ingredients:
        if "海鲜" not in allergens:
            allergens.append("海鲜")
    if "牛奶" in ingredients:
        if "牛奶" not in allergens:
            allergens.append("牛奶")
    if "鸡蛋" in ingredients:
        if "鸡蛋" not in allergens:
            allergens.append("鸡蛋")
    if "豆腐" in ingredients:
        if "大豆" not in allergens:
            allergens.append("大豆")
    if "坚果" in ingredients:
        if "坚果" not in allergens:
            allergens.append("坚果")
    return allergens

def infer_tags(name: str, category: str, spicy: int, ingredients: list) -> list:
    tags = list(CATEGORY_TAGS.get(category, []))
    if spicy >= 3:
        tags.append("辣味")
    elif spicy == 0:
        tags.append("不辣")
    if "海鲜" in ingredients or "鱼" in ingredients or "虾" in ingredients:
        tags.append("海鲜")
    if "牛肉" in ingredients:
        tags.append("牛肉")
    if "鸡肉" in ingredients:
        tags.append("鸡肉")
    if "猪肉" in ingredients:
        tags.append("猪肉")
    if "蔬菜" in ingredients:
        tags.append("素菜")
    if "豆腐" in ingredients:
        tags.append("豆制品")
    if "轻食" in tags or "低卡" in tags:
        tags.append("健康")
    return list(dict.fromkeys(tags))  # 去重保序

def fill_dish(dish: dict) -> dict:
    """补全单个菜品的所有空缺字段。"""
    name = dish.get("name", "")
    category = dish.get("category", "")
    price = dish.get("price", 20.0)

    # 提取食材
    ingredients = extract_ingredients(name, category)

    # 推断口味
    taste = infer_taste(name, category)

    # 推断烹饪方式
    cooking = infer_cooking(name, category)

    # 推断辣度
    spicy = infer_spicy(name, category)

    # 推断过敏原
    allergens = infer_allergens(name, ingredients)

    # 推断标签
    tags = infer_tags(name, category, spicy, ingredients)

    # 推断适合人群
    suitable = infer_suitable_for(name, category, spicy, tags)

    # 估算营养
    nutrition = estimate_nutrition(name, category, price)

    # 生成描述
    description = generate_description(name, category, taste, ingredients)

    # 写回字段（只补全空缺的，保留已有值）
    if not dish.get("taste"):
        dish["taste"] = taste
    if not dish.get("cooking_method"):
        dish["cooking_method"] = cooking
    if not dish.get("main_ingredients"):
        dish["main_ingredients"] = ingredients
    # 描述：只要是简单模板（长度<40 或包含模板关键词）就替换
    desc = dish.get("description", "")
    template_keywords = ["美味可口", "精选", "人气爆款", "秘制出品", "醇厚之选", "厨师推荐",
                          "秘制调配", "醇厚享受", "每日限量", "地道选材", "丰富体验", "新鲜食材",
                          "传统烹饪", "清爽", "经典口味", "现做工艺", "酥脆回味", "正宗工艺",
                          "醇厚回味", "手工选材", "嫩滑体验", "精心烹饪", "本店特色", "值得一试",
                          "传统调配", "醇厚享受", "地道调配", "清爽享受"]
    if not desc or len(desc) < 40 or any(kw in desc for kw in template_keywords):
        dish["description"] = description
    if not dish.get("suitable_for"):
        dish["suitable_for"] = suitable
    if not dish.get("tags"):
        dish["tags"] = tags
    if not dish.get("allergens"):
        dish["allergens"] = allergens
    if dish.get("spicy_level", 0) == 0 and not any(k in name for k in ["不辣", "原味", "清淡"]):
        # 只有当原辣度为0且名称中没有明确不辣时，才用推断值
        # 但如果原数据就是0（默认值），我们用推断值覆盖
        dish["spicy_level"] = spicy
    if not dish.get("nutrition"):
        dish["nutrition"] = nutrition

    return dish

def main():
    print("=" * 60)
    print("  菜品知识库字段补全")
    print("=" * 60)

    # 读取
    with open(DISH_KB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[1/4] 已加载 {len(data)} 道菜")

    # 筛选需要补全的菜品（source=backend_sync 或字段空缺的）
    to_fill = [d for d in data if d.get("source") == "backend_sync" or not d.get("taste")]
    print(f"[2/4] 需要补全的菜品: {len(to_fill)} 道")

    # 补全
    for dish in to_fill:
        fill_dish(dish)

    # 统计补全结果
    fields = ["taste", "cooking_method", "main_ingredients", "description",
              "suitable_for", "tags", "allergens", "spicy_level", "nutrition"]
    print("[3/4] 补全后字段覆盖率:")
    for f in fields:
        filled = sum(1 for d in data if d.get(f) not in [None, "", [], {}, 0])
        print(f"  {f}: {filled}/{len(data)} ({filled*100//len(data)}%)")

    # 备份原文件
    backup_path = DISH_KB_PATH + ".bak_before_fill"
    with open(DISH_KB_PATH, "r", encoding="utf-8") as f:
        with open(backup_path, "w", encoding="utf-8") as bf:
            bf.write(f.read())
    print(f"[4/4] 原文件已备份至 {backup_path}")

    # 写回
    with open(DISH_KB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] 已写入 {DISH_KB_PATH}")

    # 抽样展示
    print("\n=== 抽样展示（前3道补全后的菜品）===")
    for d in to_fill[:3]:
        print(f"\n  【{d['name']}】({d['category']})")
        print(f"    口味: {d.get('taste')}")
        print(f"    烹饪: {d.get('cooking_method')}")
        print(f"    食材: {d.get('main_ingredients')}")
        print(f"    辣度: {d.get('spicy_level')}")
        print(f"    过敏原: {d.get('allergens')}")
        print(f"    标签: {d.get('tags')}")
        print(f"    适合: {d.get('suitable_for')}")
        print(f"    营养: {d.get('nutrition')}")
        print(f"    描述: {d.get('description')[:80]}...")

    print("\n" + "=" * 60)
    print("  补全完成！接下来可以运行 build_knowledge_base.py --only dish 重建向量库")
    print("=" * 60)

if __name__ == "__main__":
    main()
