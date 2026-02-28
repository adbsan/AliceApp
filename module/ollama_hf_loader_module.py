"""
ollama_hf_loader_module.py
Hugging Face の GGUF モデルを PC スペックに応じて選択・ダウンロードし、
Ollama モデルとして登録して利用するためのローダー。
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
from loguru import logger

from module import env_binder_module as env

_client_info: Optional[Dict[str, str]] = None
_model_name: str = ""
_last_selection: Dict[str, object] = {}

_AUTO_MODEL_PROFILES = [
    {
        "max_ram_gb": 4.0,
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
        "label": "TinyLlama 1.1B Q2_K (low memory)",
        "ollama_suffix": "tinyllama-q2k",
    },
    {
        "max_ram_gb": 6.0,
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf",
        "label": "TinyLlama 1.1B Q3_K_M",
        "ollama_suffix": "tinyllama-q3km",
    },
    {
        "max_ram_gb": 8.0,
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "label": "TinyLlama 1.1B Q4_K_M (balanced)",
        "ollama_suffix": "tinyllama-q4km",
    },
    {
        "max_ram_gb": 12.0,
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
        "label": "TinyLlama 1.1B Q5_K_M",
        "ollama_suffix": "tinyllama-q5km",
    },
    {
        "max_ram_gb": 10_000.0,
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q6_K.gguf",
        "label": "TinyLlama 1.1B Q6_K (quality)",
        "ollama_suffix": "tinyllama-q6k",
    },
]


def _total_ram_gb() -> float:
    try:
        import psutil

        return float(psutil.virtual_memory().total) / (1024 ** 3)
    except Exception:
        pass

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
    selected = _AUTO_MODEL_PROFILES[-1]
    for profile in _AUTO_MODEL_PROFILES:
        if ram_gb <= float(profile["max_ram_gb"]):
            selected = profile
            break
    data = dict(selected)
    data["ram_gb"] = round(ram_gb, 1)
    data["cpu_count"] = int(cpu_count)
    return data


def _resolve_model_choice(repo_id: str, filename: str) -> Tuple[str, str, Dict[str, object]]:
    auto_repo = (not repo_id) or (repo_id.lower() == "auto")
    auto_file = (not filename) or (filename.lower() == "auto")
    if auto_repo or auto_file:
        profile = _pick_profile_by_spec()
        return (
            str(profile["repo_id"]),
            str(profile["filename"]),
            {
                "auto_selected": True,
                "profile_label": str(profile.get("label", "")),
                "profile_suffix": str(profile.get("ollama_suffix", "auto")),
                "ram_gb": profile.get("ram_gb"),
                "cpu_count": profile.get("cpu_count"),
            },
        )
    return repo_id, filename, {"auto_selected": False, "profile_suffix": "manual"}


def _ping_ollama(base_url: str) -> bool:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def _ensure_ollama_server(base_url: str) -> tuple[bool, str]:
    if _ping_ollama(base_url):
        return True, ""

    ollama_exe = shutil.which("ollama")
    if not ollama_exe:
        return False, "Ollama が見つかりません。https://ollama.com からインストールしてください。"

    logger.info("Ollama サーバーを起動します。")
    try:
        kwargs = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "stdin": subprocess.DEVNULL,
            "cwd": str(Path.cwd()),
        }
        if platform.system().lower().startswith("win"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        subprocess.Popen([ollama_exe, "serve"], **kwargs)
    except Exception as e:
        return False, f"Ollama サーバー起動に失敗: {e}"

    for _ in range(30):
        if _ping_ollama(base_url):
            return True, ""
        time.sleep(0.4)
    return False, "Ollama サーバー起動待機がタイムアウトしました。"


def _download_gguf(repo_id: str, filename: str, model_dir: Path, hf_token: str) -> Path:
    from huggingface_hub import hf_hub_download

    model_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        token=hf_token or None,
    )
    return Path(path).resolve()


def _model_exists(base_url: str, model_name: str) -> bool:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=4)
        if resp.status_code != 200:
            return False
        data = resp.json()
        for item in data.get("models", []):
            name = str(item.get("name", ""))
            if name == model_name or name.startswith(f"{model_name}:"):
                return True
        return False
    except Exception:
        return False


def _create_ollama_model(model_name: str, gguf_path: Path, n_ctx: int) -> tuple[bool, str]:
    ollama_exe = shutil.which("ollama")
    if not ollama_exe:
        return False, "ollama コマンドが見つかりません。"

    modelfile = gguf_path.parent / f"Modelfile.{model_name.replace(':', '_')}"
    content = (
        f"FROM {gguf_path.as_posix()}\n"
        f"PARAMETER num_ctx {max(1024, int(n_ctx))}\n"
        "TEMPLATE \"\"\"{{ .Prompt }}\"\"\"\n"
    )
    modelfile.write_text(content, encoding="utf-8")

    try:
        proc = subprocess.run(
            [ollama_exe, "create", model_name, "-f", str(modelfile)],
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout or "unknown error").strip()
        return True, ""
    except Exception as e:
        return False, str(e)


def load() -> Tuple[bool, Optional[Dict[str, str]], str, Dict[str, object]]:
    global _client_info, _model_name, _last_selection

    if _client_info is not None and _model_name:
        return True, dict(_client_info), _model_name, dict(_last_selection)

    base_url = str(env.get("OLLAMA_URL") or "http://localhost:11434").strip()
    base_name = str(env.get("OLLAMA_MODEL_NAME") or "alice-auto").strip() or "alice-auto"
    repo_id = str(env.get("LOCAL_MODEL_REPO") or "").strip()
    filename = str(env.get("LOCAL_MODEL_FILE") or "").strip()
    model_dir = Path(str(env.get("LOCAL_MODEL_DIR") or "assets/models")).resolve()
    n_ctx = int(env.get("LOCAL_MODEL_N_CTX"))
    hf_token = str(env.get("HF_TOKEN") or "").strip()

    repo_id, filename, auto_info = _resolve_model_choice(repo_id, filename)
    suffix = str(auto_info.get("profile_suffix") or "manual")
    model_name = f"{base_name}-{suffix}" if suffix and suffix != "manual" else base_name

    logger.info(
        "Ollama+HF モデルを準備します: repo='{}', file='{}', model='{}'",
        repo_id,
        filename,
        model_name,
    )
    ok_server, server_err = _ensure_ollama_server(base_url)
    if not ok_server:
        logger.warning(server_err)
        return False, None, "", {}

    try:
        gguf_path = _download_gguf(repo_id, filename, model_dir, hf_token)
    except Exception as e:
        logger.error(f"Hugging Face モデル取得失敗: {e}")
        return False, None, "", {}

    if not _model_exists(base_url, model_name):
        ok_create, create_err = _create_ollama_model(model_name, gguf_path, n_ctx=n_ctx)
        if not ok_create:
            logger.error(f"Ollama モデル作成失敗: {create_err}")
            return False, None, "", {}

    if not _model_exists(base_url, model_name):
        logger.error("Ollama モデル登録確認に失敗しました。")
        return False, None, "", {}

    prev_model = str(env.get("ACTIVE_OLLAMA_MODEL") or "").strip()
    switched = prev_model != model_name
    if switched:
        env.write_key("ACTIVE_OLLAMA_MODEL", model_name)

    _client_info = {"base_url": base_url}
    _model_name = model_name
    _last_selection = {
        "backend": "ollama",
        "model_name": model_name,
        "hf_repo": repo_id,
        "hf_file": filename,
        "model_path": str(gguf_path),
        "model_switched": switched,
        "auto_selected": bool(auto_info.get("auto_selected")) or switched,
        "profile_label": auto_info.get("profile_label", ""),
        "ram_gb": auto_info.get("ram_gb"),
        "cpu_count": auto_info.get("cpu_count"),
    }
    if switched:
        _last_selection["message"] = f"Ollamaモデルを切り替えました: {model_name}"
    elif auto_info.get("auto_selected"):
        _last_selection["message"] = f"PCスペックに基づきOllamaモデルを自動選択しました: {model_name}"

    logger.info("Ollama モデル準備完了: {}", model_name)
    return True, dict(_client_info), model_name, dict(_last_selection)


def reset() -> None:
    global _client_info, _model_name, _last_selection
    _client_info = None
    _model_name = ""
    _last_selection = {}
    logger.debug("ollama_hf_loader_module: キャッシュをリセットしました。")


def is_ready() -> bool:
    return _client_info is not None and bool(_model_name)


def last_selection_info() -> Dict[str, object]:
    return dict(_last_selection)

