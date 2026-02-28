"""
local_heart.py
ローカルLLM（llama.cpp）用の推論ハート実装。

責務:
  - prompt_shaper_module が作った payload をローカルモデル入力へ変換
  - ストリーミング / 非ストリーミング応答を execute() で返却
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


class LocalAliceHeart:
    def __init__(
        self,
        llm: Any,
        model_name: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> None:
        self._llm = llm
        self._model_name = model_name
        self._max_tokens = max(64, int(max_tokens))
        self._temperature = float(temperature)
        self._top_p = float(top_p)
        logger.info(f"LocalAliceHeart 初期化完了: model={model_name}")

    def execute(
        self,
        payload: Dict[str, Any],
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        t_start = time.monotonic()
        raw, err = self._run_inference(payload, on_chunk)
        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        return {
            "response": raw.strip() if err is None else "",
            "success": err is None,
            "error": err,
            "elapsed_ms": elapsed_ms,
            "model": self._model_name,
        }

    def _run_inference(
        self,
        payload: Dict[str, Any],
        on_chunk: Optional[Callable[[str], None]],
    ) -> tuple[str, Optional[str]]:
        try:
            messages = self._to_chat_messages(payload)
            req_temp = float(payload.get("temperature", self._temperature))
            req_tokens = int(payload.get("max_output_tokens", self._max_tokens))
            max_tokens = min(max(32, req_tokens), self._max_tokens)

            if on_chunk is not None:
                stream = self._llm.create_chat_completion(
                    messages=messages,
                    temperature=req_temp,
                    top_p=self._top_p,
                    max_tokens=max_tokens,
                    stream=True,
                )
                full = ""
                for chunk in stream:
                    text = self._extract_stream_text(chunk)
                    if text:
                        full += text
                        on_chunk(text)
                return full, None

            resp = self._llm.create_chat_completion(
                messages=messages,
                temperature=req_temp,
                top_p=self._top_p,
                max_tokens=max_tokens,
                stream=False,
            )
            text = self._extract_response_text(resp)
            return text, None
        except Exception as e:
            logger.error(f"ローカル推論エラー: {e}")
            return "", f"ローカル推論エラー: {type(e).__name__}: {e}"

    @staticmethod
    def _to_chat_messages(payload: Dict[str, Any]) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        system_instruction = str(payload.get("system_instruction") or "").strip()
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        for item in payload.get("contents", []):
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "user"))
            mapped = "assistant" if role in ("model", "assistant") else "user"
            text = LocalAliceHeart._extract_parts_text(item.get("parts"))
            if text:
                messages.append({"role": mapped, "content": text})

        if not messages:
            fallback = str(payload.get("user_input") or "").strip()
            if fallback:
                messages.append({"role": "user", "content": fallback})
        return messages

    @staticmethod
    def _extract_parts_text(parts: Any) -> str:
        if not isinstance(parts, list):
            return ""
        texts: List[str] = []
        for p in parts:
            if not isinstance(p, dict):
                continue
            t = p.get("text")
            if t is not None:
                texts.append(str(t))
        return "\n".join(t for t in texts if t).strip()

    @staticmethod
    def _extract_stream_text(chunk: Any) -> str:
        if not isinstance(chunk, dict):
            return ""
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        c0 = choices[0] if isinstance(choices[0], dict) else {}
        delta = c0.get("delta") if isinstance(c0.get("delta"), dict) else {}
        text = delta.get("content")
        if text:
            return str(text)
        msg = c0.get("message") if isinstance(c0.get("message"), dict) else {}
        text2 = msg.get("content")
        return str(text2) if text2 else ""

    @staticmethod
    def _extract_response_text(resp: Any) -> str:
        if not isinstance(resp, dict):
            return ""
        choices = resp.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        c0 = choices[0] if isinstance(choices[0], dict) else {}
        msg = c0.get("message") if isinstance(c0.get("message"), dict) else {}
        text = msg.get("content")
        return str(text) if text else ""

