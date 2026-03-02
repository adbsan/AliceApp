"""
prompt_shaper_module.py
ユーザー入力を推論用データ構造へ変換するモジュール。

責務:
  - ユーザー入力文字列 → payload dict への変換
  - チャット履歴の整形（Gemini API フォーマット）
  - ペルソナ（システム指示）の適用

制約:
  - 推論を実行しない（推論は heart.py の責務）
  - 外部 API 呼び出しを行わない
  - 設定は env_binder_module 経由でのみ取得
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List

from module import env_binder_module as env

# ============================================================
# デフォルトペルソナ
# ============================================================
# 修正 v2: Gemini / TinyLlama 両対応。
# - Gemini は contents[0] が "model" だと続き補完になるバグとは別に、
#   ペルソナ指示が弱いと脱線しやすい。明確なルール形式にすることで
#   どちらのモデルでも安定した応答を得る。
# - "Do NOT continue a sentence" で途中から始まる生成を防止。
# - ロールプレイ禁止を強調。
_DEFAULT_PERSONA = (
    "You are Alice, a helpful AI assistant. "
    "Follow these rules strictly:\n"
    "1. Always respond in Japanese.\n"
    "2. Start your response from the very beginning of a new sentence. "
    "Never continue or complete a sentence that was already started.\n"
    "3. Only respond to the user's actual message. "
    "Never generate roleplay, fictional dialogue, or narrative scenes.\n"
    "4. Be concise, warm, and direct. "
    "Do not add unrelated information or unsolicited questions."
)


# ============================================================
# メッセージデータ型
# ============================================================
@dataclass
class Message:
    """チャット履歴の1エントリ。"""
    role: str          # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))

    def to_genai_format(self) -> Dict[str, Any]:
        """Gemini API が要求する形式に変換する。"""
        return {
            "role": "user" if self.role == "user" else "model",
            "parts": [{"text": self.content}],
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(**{k: v for k, v in data.items() if k in ("role", "content", "timestamp")})


# ============================================================
# ペイロード構築
# ============================================================
def build_payload(
    user_input: str,
    history: List[Message],
    max_history: int = 50,
    persona: str = "",
    temperature: float = 0.9,
) -> Dict[str, Any]:
    """
    ユーザー入力とチャット履歴から推論用ペイロードを構築する。

    Args:
        user_input:   ユーザーの入力テキスト
        history:      これまでのチャット履歴（Messageのリスト）
        max_history:  履歴に含める最大メッセージ数
        persona:      システム指示文（空文字列の場合はデフォルトを使用）
        temperature:  生成温度

    Returns:
        heart.py の execute() に渡す payload dict
    """
    _persona = persona if persona.strip() else _DEFAULT_PERSONA

    # 直近 max_history 件の履歴を Gemini API 形式に変換
    trimmed = history[-max_history:] if len(history) > max_history else history
    contents = [msg.to_genai_format() for msg in trimmed]

    # -------------------------------------------------------
    # 修正: Gemini API は contents[0].role が "model" だと
    # そのモデル発話の「続き」として補完してしまう。
    # get_greeting() がアシスタント発話を履歴の先頭に追加するため
    # この状態が発生する。先頭が "model" の場合は除去する。
    # -------------------------------------------------------
    while contents and contents[0].get("role") == "model":
        contents.pop(0)

    # 現在のユーザー入力を末尾に追加
    contents.append({
        "role": "user",
        "parts": [{"text": user_input}],
    })

    return {
        "user_input": user_input,
        "contents": contents,
        "system_instruction": _persona,
        "temperature": temperature,
        "max_output_tokens": 2048,
    }


def build_greeting_payload(persona: str = "") -> Dict[str, Any]:
    """
    起動時の挨拶生成用ペイロードを構築する（履歴なし）。

    Returns:
        heart.py の execute() に渡す payload dict
    """
    _persona = persona if persona.strip() else _DEFAULT_PERSONA
    name = env.get("ALICE_NAME")

    prompt = (
        f"あなたは {name} です。ユーザーに対して短く温かく挨拶してください。"
        "2〜3文程度で、明るく親しみやすいトーンでお願いします。"
    )

    return {
        "user_input": prompt,
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "system_instruction": _persona,
        "temperature": 0.8,
        "max_output_tokens": 200,
    }


def new_message(role: str, content: str) -> Message:
    """新しい Message インスタンスを生成する。"""
    return Message(role=role, content=content)
