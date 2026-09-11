"""TTS 文本清理单元测试。"""

from text_utils import clean_text_for_tts


def test_removes_emoji():
    assert "🦌" not in clean_text_for_tts("您好呀🦌，我是小鹿")


def test_removes_markdown():
    out = clean_text_for_tts("**加粗** 和 `代码` 以及 *斜体*")
    assert "**" not in out
    assert "`" not in out
    assert "加粗" in out and "代码" in out and "斜体" in out


def test_em_dash_to_comma():
    assert "，" in clean_text_for_tts("今天—明天")


def test_strips_heading_markers():
    out = clean_text_for_tts("## 标题\n内容")
    assert out.startswith("标题")


def test_compresses_newlines():
    out = clean_text_for_tts("第一行\n\n\n\n第二行")
    assert "\n\n\n" not in out
