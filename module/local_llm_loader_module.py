"""
local_llm_loader_module.py
ローカルLLM（GGUF + llama.cpp）のダウンロード/ロード管理モジュール。

責務:
  - Hugging Face Hub から GGUF モデルをローカルへ取得
  - llama-cpp-python でモデルを初期化
  - ローダー状態をキャッシュし再利用
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger

from module import env_binder_module as env

_llm: Optional[object] = None
_model_label: str = ""
_model_path: str = ""


def load() -> Tuple[bool, Optional[object], str]:
    """
    ローカルLLMをロードする。未取得なら自動でダウンロードする。

    Returns:
        (success, llm, model_label)
    """
    global _llm, _model_label, _model_path

    if _llm is not None:
        return True, _llm, _model_label

    try:
        from llama_cpp import Llama
    except ImportError:
        logger.warning(
            "llama-cpp-python が未インストールです。"
            "ローカルモデル機能を使うには `pip install llama-cpp-python` を実行してください。"
        )
        return False, None, ""

    repo_id = str(env.get("LOCAL_MODEL_REPO") or "").strip()
    filename = str(env.get("LOCAL_MODEL_FILE") or "").strip()
    model_dir = Path(str(env.get("LOCAL_MODEL_DIR") or "assets/models")).resolve()
    n_ctx = max(512, int(env.get("LOCAL_MODEL_N_CTX")))
    n_threads = int(env.get("LOCAL_MODEL_THREADS"))
    n_gpu_layers = int(env.get("LOCAL_MODEL_N_GPU_LAYERS"))
    chat_format = str(env.get("LOCAL_MODEL_CHAT_FORMAT") or "").strip()
    hf_token = str(env.get("HF_TOKEN") or "").strip()

    if not repo_id or not filename:
        logger.warning("LOCAL_MODEL_REPO / LOCAL_MODEL_FILE が未設定です。")
        return False, None, ""

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "ローカルモデルを準備します: repo='{}', file='{}', dir='{}'",
        repo_id, filename, model_dir
    )

    try:
        kwargs = {
            "repo_id": repo_id,
            "filename": filename,
            "local_dir": str(model_dir),
            "local_dir_use_symlinks": False,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "verbose": False,
        }
        if n_threads > 0:
            kwargs["n_threads"] = n_threads
        if chat_format:
            kwargs["chat_format"] = chat_format

        llm = Llama.from_pretrained(**kwargs)
        _llm = llm
        _model_path = str(getattr(llm, "model_path", ""))
        _model_label = f"{repo_id}:{filename}"
        logger.info(f"ローカルモデルをロードしました: {_model_label} ({_model_path})")
        return True, llm, _model_label
    except Exception as e:
        logger.error(f"ローカルモデルのロードに失敗しました: {e}")
        return False, None, ""


def reset() -> None:
    """ローカルLLMキャッシュをリセットする。"""
    global _llm, _model_label, _model_path
    _llm = None
    _model_label = ""
    _model_path = ""
    logger.debug("local_llm_loader_module: キャッシュをリセットしました。")


def is_ready() -> bool:
    return _llm is not None


def model_path() -> str:
    return _model_path

