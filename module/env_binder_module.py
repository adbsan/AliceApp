"""
env_binder_module.py
環境変数の統一管理モジュール。
すべての設定値はこのモジュール経由でのみ取得する。

責務:
  - .env の読み込み
  - 型定義された設定値の提供
  - 環境変数へのアクセスを単一窓口に集約

制約:
  - 推論ロジックへの依存を持たない
  - 外部への副作用を持たない（読み取り専用）
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger

try:
    from dotenv import load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False
    logger.warning("python-dotenv が未インストールです。.envの読み込みをスキップします。")

# ============================================================
# 環境変数キー定義（すべての設定キーをここで管理）
# ============================================================
_ENV_SCHEMA: Dict[str, Dict[str, Any]] = {
    # Google AI
    "GOOGLE_API_KEY":         {"type": str,   "default": ""},
    "ALICE_MODEL":            {"type": str,   "default": "gemini-2.0-flash"},
    "ALICE_NAME":             {"type": str,   "default": "Alice"},
    "AI_BACKEND":             {"type": str,   "default": "auto"},  # auto | gemini | ollama | local

    # Local LLM (llama.cpp + GGUF)
    "LOCAL_MODEL_REPO":       {"type": str,   "default": "auto"},
    "LOCAL_MODEL_FILE":       {"type": str,   "default": "auto"},
    "LOCAL_MODEL_DIR":        {"type": str,   "default": "assets/models"},
    "LOCAL_MODEL_N_CTX":      {"type": int,   "default": 2048},
    "LOCAL_MODEL_MAX_TOKENS": {"type": int,   "default": 256},
    "LOCAL_MODEL_TEMPERATURE":{"type": float, "default": 0.7},
    "LOCAL_MODEL_TOP_P":      {"type": float, "default": 0.95},
    "LOCAL_MODEL_THREADS":    {"type": int,   "default": 0},
    "LOCAL_MODEL_N_GPU_LAYERS":{"type": int,  "default": 0},
    "LOCAL_MODEL_CHAT_FORMAT":{"type": str,   "default": ""},
    "HF_TOKEN":               {"type": str,   "default": ""},
    "OLLAMA_URL":             {"type": str,   "default": "http://localhost:11434"},
    "OLLAMA_MODEL_NAME":      {"type": str,   "default": "alice-auto"},
    "ACTIVE_OLLAMA_MODEL":    {"type": str,   "default": ""},

    # VOICEVOX
    "VOICEVOX_URL":           {"type": str,   "default": "http://localhost:50021"},
    "VOICEVOX_SPEAKER_ID":    {"type": int,   "default": 1},
    "VOICEVOX_SPEED":         {"type": float, "default": 1.0},
    "VOICEVOX_PITCH":         {"type": float, "default": 0.0},
    "VOICEVOX_INTONATION":    {"type": float, "default": 1.0},
    "VOICEVOX_VOLUME":        {"type": float, "default": 1.0},

    # Git
    "GIT_URL":                {"type": str,   "default": "https://github.com/adbsan/AppAI.git"},
    "GIT_BRANCH":             {"type": str,   "default": "testbranch"},

    # App
    "APP_THEME":              {"type": str,   "default": "dark"},
    "APP_LOG_LEVEL":          {"type": str,   "default": "INFO"},
}

# ============================================================
# 内部状態
# ============================================================
_loaded: bool = False
_env_path: Optional[Path] = None

# キー名マッチ用パターンをモジュールロード時にコンパイルしてキャッシュ
# 修正: write_key() でコメント行を誤って上書きしないよう、
#       行頭コメント（#）を除外する正規表現を事前コンパイルする
_KEY_PATTERNS: Dict[str, re.Pattern] = {}


def _get_key_pattern(key: str) -> re.Pattern:
    """キー名に対応する行マッチパターンを返す（キャッシュ付き）。"""
    if key not in _KEY_PATTERNS:
        # ^ の直後に # が来る行（コメント行）を除外するため
        # (?!#) の否定先読みを使用して非コメント行のみにマッチさせる
        _KEY_PATTERNS[key] = re.compile(
            rf"^(?![ \t]*#){re.escape(key)}\s*="
        )
    return _KEY_PATTERNS[key]


def load(env_path: Optional[str] = None) -> bool:
    """
    .env ファイルを読み込む。AliceApp.py の起動時に一度だけ呼び出す。

    Args:
        env_path: .env ファイルのパス。None の場合は実行ディレクトリから探す。

    Returns:
        読み込み成功なら True。
    """
    global _loaded, _env_path

    path = Path(env_path) if env_path else Path(".env")
    _env_path = path.resolve()

    if not _DOTENV_AVAILABLE:
        logger.warning("python-dotenv が利用できないため .env を読み込めません。")
        _loaded = False
        return False

    if not _env_path.exists():
        logger.warning(f".env が見つかりません: {_env_path}")
        _loaded = False
        return False

    load_dotenv(_env_path, override=False)
    _loaded = True
    logger.info(f".env を読み込みました: {_env_path}")
    return True


def get(key: str, default: Any = None) -> Any:
    """
    環境変数を型変換して返す。

    Args:
        key:     環境変数キー（_ENV_SCHEMA で定義されたもの）
        default: スキーマに未定義の場合のデフォルト値

    Returns:
        型変換済みの値。
    """
    schema = _ENV_SCHEMA.get(key)
    raw = os.environ.get(key)

    if raw is None:
        if schema is not None:
            return schema["default"]
        return default

    if schema is not None:
        typ = schema["type"]
        try:
            if typ is bool:
                return raw.lower() in ("1", "true", "yes", "on")
            return typ(raw)
        except (ValueError, TypeError):
            logger.warning(f"環境変数 {key} の型変換に失敗しました: '{raw}' → {typ.__name__}")
            return schema["default"]

    return raw


def get_all() -> Dict[str, Any]:
    """
    スキーマに定義されたすべての設定値を辞書で返す。

    Returns:
        キー → 型変換済み値 の辞書。
    """
    return {key: get(key) for key in _ENV_SCHEMA}


def is_loaded() -> bool:
    """
    .env が正常に読み込まれているか確認する。

    Returns:
        読み込み済みなら True。
    """
    return _loaded


def write_key(key: str, value: Any) -> bool:
    """
    .env ファイルの特定キーを更新する。
    GUI設定保存時に使用する。

    修正: コメント行（# KEY=... 形式）を誤って上書きしないよう
          (?![ \\t]*#) 否定先読みを使ってコメント行をスキップする。

    Args:
        key:   環境変数キー
        value: 設定する値

    Returns:
        書き込み成功なら True。
    """
    if _env_path is None or not _env_path.exists():
        logger.error(".env が読み込まれていないため書き込めません。")
        return False

    try:
        content = _env_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        new_line = f"{key}={value}"
        pattern = _get_key_pattern(key)
        found = False

        for i, line in enumerate(lines):
            if pattern.match(line):
                lines[i] = new_line
                found = True
                break

        if not found:
            lines.append(new_line)

        _env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.environ[key] = str(value)
        logger.debug(f".env に書き込みました: {key}")
        return True
    except Exception as e:
        logger.error(f".env 書き込みエラー: {e}")
        return False
