"""
neural_loader_module.py
AIモデルの存在確認・ロード管理モジュール。

責務:
  - モデル（Google Gemini クライアント）の存在確認
  - クライアントの初期化と返却
  - モデル名の検証（API経由）

制約:
  - 推論を実行しない（推論は heart.py の責務）
  - 設定は env_binder_module 経由でのみ取得
"""

from __future__ import annotations

from typing import Optional, Tuple
from loguru import logger

from module import env_binder_module as env

try:
    from google import genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    logger.error("google-genai が未インストールです。pip install google-genai を実行してください。")

# ============================================================
# 内部状態（シングルトン管理）
# ============================================================
_client: Optional[object] = None
_model_name: str = ""


def load() -> Tuple[bool, Optional[object], str]:
    """
    Gemini クライアントをロードして返す。
    2回目以降はキャッシュを返す（再接続不要）。

    Returns:
        (success: bool, client: genai.Client | None, model_name: str)
    """
    global _client, _model_name

    if _client is not None:
        return True, _client, _model_name

    if not _GENAI_AVAILABLE:
        return False, None, ""

    api_key = env.get("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY が未設定です。チャット機能は利用できません。")
        return False, None, ""

    model = env.get("ALICE_MODEL")
    try:
        client = genai.Client(api_key=api_key)
        _client = client
        _model_name = model
        logger.info(f"Gemini クライアントをロードしました: model={model}")
        return True, client, model
    except Exception as e:
        logger.error(f"Gemini クライアントのロードに失敗しました: {e}")
        return False, None, ""


def verify_model() -> Tuple[bool, str]:
    """
    設定されたモデルが実際に利用可能か API で確認する。

    Returns:
        (available: bool, message: str)
    """
    if not _GENAI_AVAILABLE:
        return False, "google-genai が未インストールです。"

    api_key = env.get("GOOGLE_API_KEY")
    if not api_key:
        return False, "GOOGLE_API_KEY が未設定です。"

    model_name = env.get("ALICE_MODEL")
    try:
        client = genai.Client(api_key=api_key)
        models = list(client.models.list())
        model_ids = [m.name for m in models]
        full_name = f"models/{model_name}"
        found = (
            model_name in model_ids
            or full_name in model_ids
            or any(model_name in m for m in model_ids)
        )
        if found:
            return True, f"モデル '{model_name}' が確認できました。"
        else:
            return False, f"モデル '{model_name}' が見つかりません。利用可能モデル数: {len(model_ids)}"
    except Exception as e:
        return False, f"モデル確認エラー: {e}"


def reset() -> None:
    """クライアントキャッシュをリセットする（設定変更後の再初期化用）。"""
    global _client, _model_name
    _client = None
    _model_name = ""
    logger.debug("neural_loader_module: キャッシュをリセットしました。")


def is_ready() -> bool:
    """クライアントがロード済みか確認する。"""
    return _client is not None
