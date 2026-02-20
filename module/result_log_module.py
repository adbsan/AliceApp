"""
result_log_module.py
推論結果の永続化・出力管理モジュール。

責務:
  - 推論結果のコンソール出力
  - チャット履歴の JSON ファイルへの永続化
  - ログローテーション管理

制約:
  - 推論を実行しない
  - 設定は env_binder_module 経由でのみ取得
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger

# ============================================================
# ファイルパス定数
# ============================================================
_LOGS_DIR = Path("logs")
_CHAT_HISTORY_FILE = _LOGS_DIR / "chat_history.json"
_MAX_HISTORY_PERSIST = 200   # ファイルに保存する最大メッセージ数


def save_result(payload: Dict[str, Any], result: Dict[str, Any]) -> bool:
    """
    推論結果をチャット履歴ファイルに追記保存する。

    Args:
        payload: prompt_shaper_module.build_payload() が返した入力ペイロード
        result:  heart.py の execute() が返した結果 dict

    Returns:
        保存成功なら True。
    """
    _ensure_logs_dir()

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": payload.get("user_input", ""),
        "alice": result.get("response", ""),
        "error": result.get("error"),
    }
    _append_to_history(entry)
    _log_to_console(entry)
    return True


def save_history(messages: list) -> bool:
    """
    Messageリストを丸ごとチャット履歴ファイルに保存する。

    Args:
        messages: prompt_shaper_module.Message のリスト

    Returns:
        保存成功なら True。
    """
    _ensure_logs_dir()
    try:
        data = [m.to_dict() if hasattr(m, "to_dict") else m for m in messages]
        # 最大件数を超えた分を切り詰める
        if len(data) > _MAX_HISTORY_PERSIST:
            data = data[-_MAX_HISTORY_PERSIST:]

        with open(_CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"履歴保存エラー: {e}")
        return False


def load_history() -> List[Dict[str, Any]]:
    """
    チャット履歴ファイルを読み込む。

    Returns:
        メッセージ dict のリスト。ファイルが存在しない場合は空リスト。
    """
    if not _CHAT_HISTORY_FILE.exists():
        return []
    try:
        with open(_CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"チャット履歴を読み込みました: {len(data)}件")
        return data
    except Exception as e:
        logger.error(f"履歴読み込みエラー: {e}")
        return []


def clear_history() -> bool:
    """チャット履歴ファイルを削除する。"""
    try:
        if _CHAT_HISTORY_FILE.exists():
            _CHAT_HISTORY_FILE.unlink()
        logger.info("チャット履歴を削除しました。")
        return True
    except Exception as e:
        logger.error(f"履歴削除エラー: {e}")
        return False


# ============================================================
# 内部処理
# ============================================================

def _ensure_logs_dir() -> None:
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _append_to_history(entry: Dict[str, Any]) -> None:
    """履歴ファイルにエントリを追記する。"""
    try:
        existing: List[Dict] = []
        if _CHAT_HISTORY_FILE.exists():
            with open(_CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        existing.append(entry)
        # 最大件数を超えた分を切り詰める
        if len(existing) > _MAX_HISTORY_PERSIST:
            existing = existing[-_MAX_HISTORY_PERSIST:]
        with open(_CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"履歴追記エラー: {e}")


def _log_to_console(entry: Dict[str, Any]) -> None:
    """推論結果をログに出力する。"""
    user = entry.get("user", "")
    alice = entry.get("alice", "")
    error = entry.get("error")

    if error:
        logger.error(f"推論エラー | user: '{user[:50]}' | error: {error}")
    else:
        logger.debug(f"推論完了 | user: '{user[:50]}' | response: {len(alice)}文字")
