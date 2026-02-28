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
import platform
import ctypes
from pathlib import Path
from typing import Dict, Optional, Tuple

from loguru import logger

from module import env_binder_module as env

_llm: Optional[object] = None
_model_label: str = ""
_model_path: str = ""
_last_selection: Dict[str, object] = {}

_AUTO_MODEL_PROFILES = [
    {
        "max_ram_gb": 4.0,
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
        "label": "TinyLlama 1.1B Q2_K (low memory)",
    },
    {
        "max_ram_gb": 6.0,
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf",
        "label": "TinyLlama 1.1B Q3_K_M",
    },
    {
        "max_ram_gb": 8.0,
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "label": "TinyLlama 1.1B Q4_K_M (balanced)",
    },
    {
        "max_ram_gb": 12.0,
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
        "label": "TinyLlama 1.1B Q5_K_M",
    },
    {
        "max_ram_gb": 10_000.0,
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q6_K.gguf",
        "label": "TinyLlama 1.1B Q6_K (quality)",
    },
]

_LEGACY_DEFAULT_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
_LEGACY_DEFAULT_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"


def _total_ram_gb() -> float:
    # Try psutil first if installed.
    try:
        import psutil
        return float(psutil.virtual_memory().total) / (1024 ** 3)
    except Exception:
        pass

    # Windows fallback.
    if platform.system().lower().startswith("win"):
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return float(stat.ullTotalPhys) / (1024 ** 3)
        except Exception:
            pass

    return 8.0


def _pick_profile_by_spec() -> Dict[str, object]:
    ram_gb = _total_ram_gb()
    cpu_count = os.cpu_count() or 4
    chosen = _AUTO_MODEL_PROFILES[-1]
    for p in _AUTO_MODEL_PROFILES:
        if ram_gb <= float(p["max_ram_gb"]):
            chosen = p
            break
    result = dict(chosen)
    result["ram_gb"] = round(ram_gb, 1)
    result["cpu_count"] = int(cpu_count)
    return result


def _resolve_model_choice(repo_id: str, filename: str) -> Tuple[str, str, Dict[str, object]]:
    legacy_default = (repo_id == _LEGACY_DEFAULT_REPO and filename == _LEGACY_DEFAULT_FILE)
    auto_repo = (not repo_id) or (repo_id.lower() == "auto")
    auto_file = (not filename) or (filename.lower() == "auto")
    if legacy_default:
        auto_repo = True
        auto_file = True
    if auto_repo or auto_file:
        profile = _pick_profile_by_spec()
        return (
            str(profile["repo_id"]),
            str(profile["filename"]),
            {
                "auto_selected": True,
                "profile_label": str(profile.get("label", "")),
                "ram_gb": profile.get("ram_gb"),
                "cpu_count": profile.get("cpu_count"),
            },
        )
    return repo_id, filename, {"auto_selected": False}


def load() -> Tuple[bool, Optional[object], str, Dict[str, object]]:
    """
    ローカルLLMをロードする。未取得なら自動でダウンロードする。

    Returns:
        (success, llm, model_label, selection_info)
    """
    global _llm, _model_label, _model_path, _last_selection

    if _llm is not None:
        return True, _llm, _model_label, dict(_last_selection)

    try:
        from llama_cpp import Llama
    except ImportError:
        logger.warning(
            "llama-cpp-python が未インストールです。"
            "ローカルモデル機能を使うには `pip install llama-cpp-python` を実行してください。"
        )
        return False, None, "", {}

    repo_id = str(env.get("LOCAL_MODEL_REPO") or "").strip()
    filename = str(env.get("LOCAL_MODEL_FILE") or "").strip()
    repo_id, filename, auto_info = _resolve_model_choice(repo_id, filename)
    model_dir = Path(str(env.get("LOCAL_MODEL_DIR") or "assets/models")).resolve()
    n_ctx = max(512, int(env.get("LOCAL_MODEL_N_CTX")))
    n_threads = int(env.get("LOCAL_MODEL_THREADS"))
    n_gpu_layers = int(env.get("LOCAL_MODEL_N_GPU_LAYERS"))
    chat_format = str(env.get("LOCAL_MODEL_CHAT_FORMAT") or "").strip()
    hf_token = str(env.get("HF_TOKEN") or "").strip()

    if not repo_id or not filename:
        logger.warning("LOCAL_MODEL_REPO / LOCAL_MODEL_FILE が未設定です。")
        return False, None, "", {}

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "ローカルモデルを準備します: repo='{}', file='{}', dir='{}'",
        repo_id, filename, model_dir
    )
    if auto_info.get("auto_selected"):
        logger.info(
            "PCスペックに基づきローカルモデルを自動選択: {} (RAM={}GB, CPU={} cores)",
            auto_info.get("profile_label", f"{repo_id}:{filename}"),
            auto_info.get("ram_gb"),
            auto_info.get("cpu_count"),
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
        _last_selection = {
            "backend": "local",
            "model_label": _model_label,
            "model_path": _model_path,
            **auto_info,
        }
        logger.info(f"ローカルモデルをロードしました: {_model_label} ({_model_path})")
        return True, llm, _model_label, dict(_last_selection)
    except Exception as e:
        logger.error(f"ローカルモデルのロードに失敗しました: {e}")
        return False, None, "", {}


def reset() -> None:
    """ローカルLLMキャッシュをリセットする。"""
    global _llm, _model_label, _model_path, _last_selection
    _llm = None
    _model_label = ""
    _model_path = ""
    _last_selection = {}
    logger.debug("local_llm_loader_module: キャッシュをリセットしました。")


def is_ready() -> bool:
    return _llm is not None


def model_path() -> str:
    return _model_path


def last_selection_info() -> Dict[str, object]:
    return dict(_last_selection)
