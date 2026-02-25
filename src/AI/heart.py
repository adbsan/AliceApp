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
  - chunk.text アクセスを getattr で安全化（安全フィルタ等でテキストなしの
    チャンクが返ることがあるため AttributeError を防止）。
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
        """
        t_start = time.monotonic()
        ctx = self._build_context(payload)
        raw_response, error = self._run_inference(ctx, on_chunk)
        shaped = self._shape_response(raw_response, error)
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

        修正: chunk.text を getattr で安全にアクセス。
        安全フィルタ等によりテキストを持たないチャンクが返る場合があり、
        直接 chunk.text を参照すると AttributeError が発生する。
        """
        full = ""
        response = self._client.models.generate_content_stream(
            model=self._model_name,
            contents=contents,
            config=cfg,
        )

        def _process_chunk(chunk) -> None:
            nonlocal full
            # 【修正】chunk.text → getattr で AttributeError を防止
            text = getattr(chunk, "text", None) or ""
            full += text
            if text:
                on_chunk(text)

        if hasattr(response, "__enter__"):
            # コンテキストマネージャ形式（旧 API）
            stream = response.__enter__()
            try:
                for chunk in stream:
                    _process_chunk(chunk)
            finally:
                response.__exit__(None, None, None)
        else:
            # 直接イテラブル形式（新 API）
            for chunk in response:
                _process_chunk(chunk)

        return full

    def _generate(self, contents: list, cfg: Any) -> str:
        """非ストリーミング推論。"""
        resp = self._client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=cfg,
        )
        # 【修正】resp.text も getattr で安全にアクセス
        return getattr(resp, "text", None) or ""

    # ============================================================
    # Step 3: レスポンス整形
    # ============================================================

    def _shape_response(self, raw: str, error: Optional[str]) -> str:
        if error:
            return ""
        return raw.strip()

    # ============================================================
    # ユーティリティ
    # ============================================================

    @staticmethod
    def _parse_retry_delay(exc: Exception) -> float:
        import re
        m = re.search(r"retryDelay.*?(\d+)s", str(exc))
        if m:
            return float(m.group(1)) + 2.0
        return 30.0

    @staticmethod
    def _format_error(exc: Exception) -> str:
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
