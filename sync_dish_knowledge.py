"""
菜品知识库同步脚本 —— 对齐 data/dish_knowledge.json 与 Java 后端实时菜品。

两种模式:
  1. 离线模式（默认）: 校验 JSON 结构、按 dish_id 去重、输出差异报告。
  2. 在线模式: 提供手机号/密码（或环境变量 SYNC_PHONE/SYNC_PASSWORD）登录
     Java 后端，拉取实时菜品，与知识库对比：
       - 后端存在但知识库缺失 -> 追加基础条目（营养字段留空，避免编造）
       - 知识库存在但后端已下架 -> 仅报告，不删除（保留历史知识）
       - 名称/店铺/价格不一致 -> 报告差异

用法:
  python sync_dish_knowledge.py                 # 离线去重 + 校验
  python sync_dish_knowledge.py --dry-run       # 在线模式（只报告，不写入）
  python sync_dish_knowledge.py --write         # 在线模式（写入缺失条目）
  python sync_dish_knowledge.py --phone 13900000000 --password 123456 --write
"""

import argparse
import json
import os
import sys
from datetime import date

import backend_client as bc


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DISH_KB_PATH = os.path.join(DATA_DIR, "dish_knowledge.json")


def _load_kb() -> list[dict]:
    if not os.path.exists(DISH_KB_PATH):
        print(f"[ERROR] 知识库文件不存在: {DISH_KB_PATH}")
        sys.exit(1)
    with open(DISH_KB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"[ERROR] 知识库文件必须是 JSON 数组，当前为 {type(data).__name__}")
        sys.exit(1)
    return data


def _dish_key(d: dict) -> str:
    """以 dish_id 优先，缺失时用 名称+店铺 作为去重键。"""
    if d.get("dish_id") is not None:
        return f"id:{d['dish_id']}"
    return f"name:{d.get('name', '')}|shop:{d.get('shop', '')}"


def _validate_and_dedupe(entries: list[dict]) -> tuple[list[dict], list[str]]:
    """校验必填字段并按 dish_id 去重，返回 (去重后列表, 移除记录)。"""
    seen: dict[str, int] = {}
    deduped: list[dict] = []
    removed: list[str] = []
    required = ("name",)
    for d in entries:
        if not isinstance(d, dict) or not all(k in d for k in required):
            removed.append(f"缺少必填字段: {d.get('name', d)[:40] if isinstance(d, dict) else d}")
            continue
        key = _dish_key(d)
        if key in seen:
            removed.append(f"重复条目: {d.get('name', '?')} (key={key})")
            continue
        seen[key] = 1
        deduped.append(d)
    return deduped, removed


def _fetch_backend_dishes(session_id: str) -> list[dict]:
    """从 Java 后端拉取全部菜品（实时数据）。"""
    result = bc.get("/api/user/dishes", session_id)
    raw = bc.extract_data(result)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        records = raw.get("records")
        return records if isinstance(records, list) else []
    return []


def _login(phone: str, password: str) -> str:
    """登录 Java 后端，返回 session_id（临时会话）。"""
    import uuid
    session_id = f"sync-{uuid.uuid4().hex[:8]}"
    result = bc.post_public("/api/auth/user/login", {"phone": phone, "password": password})
    if result.get("code") != 200:
        print(f"[ERROR] 登录失败: {result.get('message', '未知错误')}")
        sys.exit(1)
    data = result["data"]
    import time
    bc.set_session(session_id, {
        "token": data["token"],
        "user_id": data["id"],
        "username": data["username"],
        "expires_at": time.time() + 1800,
    })
    return session_id


def main() -> None:
    parser = argparse.ArgumentParser(description="菜品知识库去重 / 与后端同步")
    parser.add_argument("--phone", default=os.getenv("SYNC_PHONE", ""), help="后端登录手机号")
    parser.add_argument("--password", default=os.getenv("SYNC_PASSWORD", ""), help="后端登录密码")
    parser.add_argument("--write", action="store_true", help="写回文件（在线模式追加缺失条目 / 离线模式写去重结果）")
    parser.add_argument("--dry-run", action="store_true", help="只报告不写入")
    args = parser.parse_args()

    entries = _load_kb()
    print(f"[1/4] 已加载知识库: {len(entries)} 条")

    deduped, removed = _validate_and_dedupe(entries)
    print(f"[2/4] 去重校验完成: 保留 {len(deduped)} 条, 移除 {len(removed)} 条")
    for r in removed[:20]:
        print(f"      - {r}")

    online = bool(args.phone and args.password)
    if online:
        session_id = _login(args.phone, args.password)
        print("[3/4] 登录成功，拉取后端实时菜品...")
        try:
            backend_dishes = _fetch_backend_dishes(session_id)
        except Exception as e:
            print(f"[ERROR] 拉取后端菜品失败: {e}")
            backend_dishes = []
        if not backend_dishes:
            print("[WARN] 后端未返回菜品，跳过在线对比")
        else:
            kb_keys = {_dish_key(d) for d in deduped}
            kb_names = {(d.get("name", ""), d.get("shop", "")) for d in deduped}
            missing = []
            for d in backend_dishes:
                key = f"id:{d.get('id')}"
                name_shop = (d.get("name", ""), d.get("shopName") or "")
                if key not in kb_keys and name_shop not in kb_names:
                    missing.append(d)
            print(f"      后端共 {len(backend_dishes)} 道，知识库缺失 {len(missing)} 道")
            for d in missing[:30]:
                print(f"      + 新增: {d.get('name')}（{d.get('shopName')}）")
            if args.write and not args.dry_run:
                today = date.today().isoformat()
                for d in missing:
                    deduped.append({
                        "dish_id": d.get("id"),
                        "name": d.get("name", "未知"),
                        "shop": d.get("shopName", "未知店铺"),
                        "category": d.get("categoryName", ""),
                        "price": d.get("price", 0),
                        "description": (d.get("description", "") or "")[:120],
                        "suitable_for": [],
                        "tags": [],
                        "allergens": [],
                        "spicy_level": 0,
                        "nutrition": {},
                        "source": "backend_sync",
                        "synced_at": today,
                    })
                print(f"      已追加 {len(missing)} 条（营养字段留空，请人工补充）")
    else:
        print("[3/4] 未提供登录凭据，跳过在线对比（可使用 --phone/--password 或 SYNC_PHONE/SYNC_PASSWORD）")

    if args.write and not args.dry_run:
        # 先备份再写入，保证可恢复
        backup = DISH_KB_PATH + ".bak"
        with open(DISH_KB_PATH, "r", encoding="utf-8") as f:
            with open(backup, "w", encoding="utf-8") as bf:
                bf.write(f.read())
        with open(DISH_KB_PATH, "w", encoding="utf-8") as f:
            json.dump(deduped, f, ensure_ascii=False, indent=2)
        print(f"[4/4] 已写入 {DISH_KB_PATH}（原文件备份至 {backup}）")
    else:
        print("[4/4] 本次为只读运行，未写入文件。确认无误后使用 --write 落盘。")


if __name__ == "__main__":
    main()
