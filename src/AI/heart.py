"""
heart.py
AI推論の心臓部（中層構造）。

設計原則:
  - 外部依存を排除（module/ への import 禁止）
  - 推論ロジックの純粋性を維持
  - 入力 → 推論 → 整形 → 返却 の直列処理のみ
  - 戻り値は常に dict 形式

責務:
  - モデルオブジェクトを受け取り推論を実行する
  - レスポンスを整形して dict で返す
  - 永続化・設定参照・物理操作は行わない

制約 (責務分離マトリクス):
  - 物理処理: ×
  - 設定参照: ×
  - 推論:     ○
  - 永続化:   ×

修正:
  - _stream() の generate_content_stream 呼び出しを
    コンテキストマネージャ形式と直接イテラブル形式の両方に対応させた。
    google-genai のバージョンによって返却型が異なるため、
    __enter__ の有無を確認してから適切な方法でストリームを読み取る。
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from loguru import logger

try:
    from google.genai import types
    _TYPES_AVAILABLE = True
except ImportError:
    _TYPES_AVAILABLE = False

# ============================================================
# リトライ設定（推論レイヤーで保持）
# ============================================================
_MAX_RETRIES = 3
_RETRY_BASE_WAIT = 5.0


class AliceHeart:
    """
    AI推論の心臓部クラス。

    Usage:
        heart = AliceHeart(client=gemini_client, model_name=model_name)
        result = heart.execute(payload)
    """

    def __init__(self, client: Any, model_name: str) -> None:
        """
        Args:
            client:     neural_loader_module.load() が返す Gemini クライアント
            model_name: 使用するモデル名
        """
        self._client = client
        self._model_name = model_name
        logger.info(f"AliceHeart 初期化完了 (APIキー・モデル検証済み): model={model_name}")

    # ============================================================
    # 主要メソッド（直列処理パイプライン）
    # ============================================================

    def execute(
        self,
        payload: Dict[str, Any],
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        推論を実行し、結果を dict で返す。

        処理順序（直列・逆流禁止）:
          1. 内部コンテキスト構築
          2. AI推論実行
          3. レスポンス整形
          4. 結果返却

        Args:
            payload:  prompt_shaper_module.build_payload() が返した dict
            on_chunk: ストリーミング時の各チャンクコールバック（任意）

        Returns:
            {
                "response": str,       # 生成テキスト
                "success": bool,       # 成功フラグ
                "error": str | None,   # エラーメッセージ（失敗時）
                "elapsed_ms": int,     # 処理時間（ミリ秒）
                "model": str,          # 使用モデル名
            }
        """
        t_start = time.monotonic()

        # Step 1: 内部コンテキスト構築
        ctx = self._build_context(payload)

        # Step 2: AI推論実行
        raw_response, error = self._run_inference(ctx, on_chunk)

        # Step 3: レスポンス整形
        shaped = self._shape_response(raw_response, error)

        # Step 4: 結果返却
        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        return {
            "response": shaped,
            "success": error is None,
            "error": error,
            "elapsed_ms": elapsed_ms,
            "model": self._model_name,
        }

    # ============================================================
    # Step 1: 内部コンテキスト構築
    # ============================================================

    def _build_context(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """ペイロードから推論コンテキストを構築する。"""
        return {
            "contents": payload.get("contents", []),
            "system_instruction": payload.get("system_instruction", ""),
            "temperature": float(payload.get("temperature", 0.9)),
            "max_output_tokens": int(payload.get("max_output_tokens", 2048)),
        }

    # ============================================================
    # Step 2: AI推論実行（リトライ付き）
    # ============================================================

    def _run_inference(
        self,
        ctx: Dict[str, Any],
        on_chunk: Optional[Callable[[str], None]],
    ) -> tuple:
        """
        Gemini API を呼び出す。リトライは 429 のみ対象。

        Returns:
            (response_text, error_message | None)
        """
        if not _TYPES_AVAILABLE:
            return "", "google-genai の types モジュールが利用できません。"

        cfg = types.GenerateContentConfig(
            system_instruction=ctx["system_instruction"],
            temperature=ctx["temperature"],
            max_output_tokens=ctx["max_output_tokens"],
        )

        for attempt in range(_MAX_RETRIES):
            try:
                if on_chunk is not None:
                    return self._stream(ctx["contents"], cfg, on_chunk), None
                else:
                    return self._generate(ctx["contents"], cfg), None

            except Exception as exc:
                exc_str = str(exc)

                # 429: レート制限 → リトライ
                if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
                    wait = self._parse_retry_delay(exc)
                    if attempt < _MAX_RETRIES - 1:
                        logger.warning(
                            f"レート制限 (attempt {attempt + 1}/{_MAX_RETRIES}). "
                            f"{wait:.0f}秒後にリトライします。"
                        )
                        if on_chunk:
                            on_chunk(
                                f"\n[レート制限: {wait:.0f}秒後に再試行します...]\n"
                            )
                        time.sleep(wait)
                        continue
                    return "", self._format_error(exc)

                # それ以外はリトライしない
                logger.error(f"推論エラー: {exc}")
                return "", self._format_error(exc)

        return "", "最大リトライ回数に達しました。"

    def _stream(
        self,
        contents: list,
        cfg: Any,
        on_chunk: Callable[[str], None],
    ) -> str:
        """
        ストリーミング推論。

        修正: google-genai のバージョンによって generate_content_stream() が
              コンテキストマネージャ（with 文対応）を返す場合と、
              直接イテラブルを返す場合がある。
              __enter__ 属性の有無で判定し、両方のパターンに対応する。
        """
        full = ""
        response = self._client.models.generate_content_stream(
            model=self._model_name,
            contents=contents,
            config=cfg,
        )

        # コンテキストマネージャ形式か直接イテラブルかを判定
        if hasattr(response, "__enter__"):
            # コンテキストマネージャ形式（旧 API）
            stream = response.__enter__()
            try:
                for chunk in stream:
                    text = chunk.text or ""
                    full += text
                    if text:
                        on_chunk(text)
            finally:
                response.__exit__(None, None, None)
        else:
            # 直接イテラブル形式（新 API）
            for chunk in response:
                text = chunk.text or ""
                full += text
                if text:
                    on_chunk(text)

        return full

    def _generate(self, contents: list, cfg: Any) -> str:
        """非ストリーミング推論。"""
        resp = self._client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=cfg,
        )
        return resp.text or ""

    # ============================================================
    # Step 3: レスポンス整形
    # ============================================================

    def _shape_response(self, raw: str, error: Optional[str]) -> str:
        """生の推論結果を整形する。"""
        if error:
            return ""
        return raw.strip()

    # ============================================================
    # ユーティリティ（推論レイヤー内部のみ使用）
    # ============================================================

    @staticmethod
    def _parse_retry_delay(exc: Exception) -> float:
        """429レスポンスから retryDelay を取得する。"""
        import re
        m = re.search(r"retryDelay.*?(\d+)s", str(exc))
        if m:
            return float(m.group(1)) + 2.0
        return 30.0

    @staticmethod
    def _format_error(exc: Exception) -> str:
        """エラーをユーザー向けメッセージに変換する。"""
        msg = str(exc)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            return (
                "クォータ制限に達しました。\n"
                "しばらく時間をおいてから再試行してください。"
            )
        if "401" in msg or "API_KEY" in msg.upper():
            return "APIキーが無効です。設定を確認してください。"
        if "404" in msg:
            return "モデルが見つかりません。設定でモデル名を確認してください。"
        return f"推論エラー: {type(exc).__name__}: {exc}"
