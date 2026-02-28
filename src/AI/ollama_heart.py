"""
ollama_heart.py
Ollama API を使ってローカル推論を行う Heart 実装。
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional

import requests
from loguru import logger


class OllamaAliceHeart:
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._max_tokens = max(64, int(max_tokens))
        self._temperature = float(temperature)
        self._top_p = float(top_p)
        logger.info(f"OllamaAliceHeart 初期化完了: model={model_name}, base_url={self._base_url}")

    def execute(
        self,
        payload: Dict[str, Any],
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        t_start = time.monotonic()
        response_text, err = self._run_inference(payload, on_chunk=on_chunk)
        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        return {
            "response": response_text.strip() if err is None else "",
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
        messages = self._to_chat_messages(payload)
        req_temp = float(payload.get("temperature", self._temperature))
        req_tokens = int(payload.get("max_output_tokens", self._max_tokens))
        max_tokens = min(max(32, req_tokens), self._max_tokens)

        body = {
            "model": self._model_name,
            "messages": messages,
            "stream": on_chunk is not None,
            "options": {
                "temperature": req_temp,
                "top_p": self._top_p,
                "num_predict": max_tokens,
            },
        }

        try:
            if on_chunk is None:
                resp = requests.post(
                    f"{self._base_url}/api/chat",
                    json=body,
                    timeout=120,
                )
                if resp.status_code != 200:
                    return "", f"Ollama API エラー: {resp.status_code} {resp.text[:200]}"
                data = resp.json()
                text = str((data.get("message") or {}).get("content") or "")
                return text, None

            stream_resp = requests.post(
                f"{self._base_url}/api/chat",
                json=body,
                stream=True,
                timeout=(10, 300),
            )
            if stream_resp.status_code != 200:
                return "", f"Ollama API エラー: {stream_resp.status_code} {stream_resp.text[:200]}"

            full = ""
            for raw_line in stream_resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                try:
                    item = json.loads(raw_line)
                except Exception:
                    continue
                piece = str((item.get("message") or {}).get("content") or "")
                if piece:
                    full += piece
                    on_chunk(piece)
            return full, None
        except Exception as e:
            logger.error(f"Ollama 推論エラー: {e}")
            return "", f"Ollama 推論エラー: {type(e).__name__}: {e}"

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
            text = OllamaAliceHeart._extract_parts_text(item.get("parts"))
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

