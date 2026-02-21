"""
neural_loader_module.py
AIモデルの存在確認・ロード管理モジュール。

責務:
  - モデル（Google Gemini クライアント）の存在確認
  - クライアントの初期化と返却
  - APIキーおよびモデル名の検証（API経由）

制約:
  - 推論を実行しない（推論は heart.py の責務）
  - 設定は env_binder_module 経由でのみ取得

無料枠で利用可能なモデル（2026年2月現在 / Gemini Developer API）:
  - gemini-2.0-flash       : 5 RPM  / 1,500 RPD  ← デフォルト推奨
  - gemini-2.0-flash-lite  : 30 RPM / 1,500 RPD  ← 高頻度用途
  - gemini-2.5-flash       : 10 RPM / 250 RPD
  - gemini-2.5-flash-lite  : 15 RPM / 1,000 RPD
  - gemini-2.5-pro         : 無料枠廃止（有料のみ）
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
_verified: bool = False  # APIキー検証済みフラグ


def load() -> Tuple[bool, Optional[object], str]:
    """
    Gemini クライアントをロードし、APIキーとモデルの実動作を確認して返す。
    2回目以降はキャッシュを返す（再接続不要）。

    ★ 修正: クライアント生成だけでなく、APIキーが実際に有効かを
            小さなリクエストで確認してからキャッシュする。
            APIキー無効時は False を返し AliceHeart を生成させない。

    Returns:
        (success: bool, client: genai.Client | None, model_name: str)
    """
    global _client, _model_name, _verified

    if _client is not None and _verified:
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

        # ── APIキーの実際の有効性チェック ──────────────────────────
        # models.list() を呼ぶことで 400/401 をここで捕捉する。
        # クライアント生成自体はAPIキーが無効でも成功してしまうため必須。
        try:
            model_list = list(client.models.list())
            model_ids = [m.name for m in model_list]
        except Exception as api_err:
            err_str = str(api_err)
            if "API_KEY_INVALID" in err_str or "400" in err_str or "401" in err_str:
                logger.error(f"APIキーが無効です: {api_err}")
                return False, None, ""
            raise  # 予期しないエラーは再 raise

        # ── モデル存在確認 ─────────────────────────────────────────
        full_name = f"models/{model}"
        found = (
            model in model_ids
            or full_name in model_ids
            or any(model in m for m in model_ids)
        )
        if not found:
            logger.warning(
                f"モデル '{model}' が見つかりません。"
                f"利用可能モデル数: {len(model_ids)}。"
                f"デフォルト 'gemini-2.0-flash' で続行します。"
            )
            model = "gemini-2.0-flash"

        _client = client
        _model_name = model
        _verified = True
        logger.info(f"Gemini クライアントをロードしました: model={model}")
        return True, client, model

    except Exception as e:
        logger.error(f"Gemini クライアントのロードに失敗しました: {e}")
        return False, None, ""


def verify_model() -> Tuple[bool, str]:
    """
    ロード済みクライアントのモデルが利用可能か確認する。
    load() が成功していれば既にキャッシュ済みのため即座に返す。

    Returns:
        (available: bool, message: str)
    """
    if not _GENAI_AVAILABLE:
        return False, "google-genai が未インストールです。"

    if _verified and _client is not None:
        return True, f"モデル '{_model_name}' が確認済みです（キャッシュ）。"

    # load() 未実行 or 失敗後の再確認
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
