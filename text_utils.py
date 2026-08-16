"""
文本处理工具 —— 供 TTS 播报前清理文本。
独立成模块，避免测试时引入 langchain 等重量级依赖。
"""

import re


def clean_text_for_tts(text: str) -> str:
    """清理文本，移除表情和 Markdown 符号，用于语音合成。
    保留中文语义和自然停顿，去掉 TTS 会读出来的符号和表情。
    """
    # 移除 emoji（覆盖常见 Unicode 表情范围）
    emoji_pattern = re.compile(
        '['
        '\U0001F600-\U0001F64F'   # 表情符号
        '\U0001F300-\U0001F5FF'   # 杂项符号和象形文字
        '\U0001F680-\U0001F6FF'   # 交通工具和地图
        '\U0001F1E0-\U0001F1FF'   # 旗帜
        '\U0001F900-\U0001F9FF'   # 补充符号和象形文字
        '\U0001FA00-\U0001FA6F'   # 象棋符号
        '\U0001FA70-\U0001FAFF'   # 符号扩展-A
        '☀-➿'           # 杂项符号（包含 ☀⭐ 等）
        '⭐'                   # ⭐
        '️'                   # 变体选择器
        '‍'                   # 零宽连接符
        ']+', flags=re.UNICODE)
    text = emoji_pattern.sub('', text)

    # 移除 markdown 加粗 **text** → text
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)

    # 移除 markdown 斜体标记
    text = re.sub(r'\*(.+?)\*', r'\1', text)

    # 移除 markdown 代码标记
    text = re.sub(r'`(.+?)`', r'\1', text)

    # 将 — （em dash）替换为逗号停顿
    text = text.replace('—', '，')

    # 将 # 标题标记去掉内容保留
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)

    # 移除多余空格，保留换行作为停顿
    text = re.sub(r'[ \t]+', ' ', text)

    # 多个连续换行压缩为单个
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()
