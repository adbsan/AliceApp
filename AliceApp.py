"""
AliceApp.py
エントリポイント。外装モジュールと中層（heart.py）の結合を担う。

実行シーケンス（設計書 §5 準拠・逆流禁止）:
  1. Load Config      - env_binder_module で .env を読み込む
  2. Ensure Model     - neural_loader_module でモデルをロード
  3. Initialize Heart - AliceHeart を生成
  4. Process Input    - prompt_shaper_module でペイロードを構築
  5. Core Execution   - heart.execute() で推論
  6. Handle Output    - result_log_module で永続化
"""

from __future__ import annotations

import re
import queue
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from loguru import logger

# ============================================================
# ログ設定（起動最初期）
# ============================================================
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.remove()
logger.add(
    sys.stderr, level="INFO", colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | "
           "<cyan>{name}</cyan> - <white>{message}</white>"
)
logger.add(
    LOG_DIR / "alice.log", level="DEBUG",
    rotation="10 MB", retention="30 days", encoding="utf-8"
)

# ============================================================
# モジュールインポート
# ============================================================
from module import env_binder_module as env
from module import local_llm_loader_module as local_loader
from module import neural_loader_module as neural
from module import prompt_shaper_module as shaper
from module import result_log_module as result_log
from src.AI.heart import AliceHeart
from src.AI.local_heart import LocalAliceHeart


# ============================================================
# AliceEngine（パイプライン制御）
# ============================================================

class AliceEngine:
    """heart.py を呼び出すパイプラインコントローラ。"""

    def __init__(self, heart: Any) -> None:
        self._heart = heart
        self._history: List[shaper.Message] = []

    @property
    def history(self) -> List[shaper.Message]:
        return list(self._history)

    def send_message(
        self,
        user_input: str,
        on_chunk: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        if not user_input.strip():
            return

        self._history.append(shaper.new_message("user", user_input))
        payload = shaper.build_payload(
            user_input=user_input,
            history=self._history[:-1],
            max_history=50,
            persona="",
            temperature=0.9,
        )

        result = self._heart.execute(payload, on_chunk=on_chunk)

        if result["success"]:
            response = result["response"]
            self._history.append(shaper.new_message("assistant", response))
            result_log.save_history(self._history)
            if on_complete:
                on_complete(response)
        else:
            if self._history and self._history[-1].role == "user":
                self._history.pop()
            if on_error:
                on_error(result["error"] or "不明なエラーが発生しました。")

    def get_greeting(self) -> str:
        name = env.get("ALICE_NAME")
        msg = (
            f"こんにちは！私は {name} です。\n"
            "何でもお気軽に話しかけてください。お手伝いします！"
        )
        self._history.append(shaper.new_message("assistant", msg))
        return msg

    def clear_history(self) -> None:
        self._history.clear()
        result_log.clear_history()

    def load_history(self) -> None:
        raw = result_log.load_history()
        self._history = [
            shaper.Message.from_dict(m) for m in raw
            if isinstance(m, dict) and "role" in m and "content" in m
        ]


# ============================================================
# GitManager（外装モジュール相当）
# ============================================================

class GitManager:
    TARGET_BRANCH = "testbranch"

    def __init__(self, repo_path: str = ".") -> None:
        self._repo_path = Path(repo_path).resolve()
        self._repo = None
        try:
            import git
            from git import InvalidGitRepositoryError, Repo
            try:
                self._repo = Repo(self._repo_path)
                logger.info(f"Git リポジトリを検出: {self._repo_path}")
                self._ensure_target_branch()
            except InvalidGitRepositoryError:
                logger.info("Git リポジトリが未初期化。新規初期化します。")
                self._repo = Repo.init(self._repo_path)
                self._create_initial_commit()
        except ImportError:
            logger.warning("GitPython が未インストールです。Git 機能は無効化されます。")
        except Exception as e:
            logger.error(f"Git 初期化エラー: {e}")
            self._repo = None

    @property
    def is_available(self) -> bool:
        return self._repo is not None

    def get_status(self) -> dict:
        if not self.is_available:
            return {"error": "Git 利用不可"}
        try:
            repo = self._repo
            branch = repo.active_branch.name
            changed = [item.a_path for item in repo.index.diff(None)]
            untracked = list(repo.untracked_files)
            staged = []
            if repo.head.is_valid():
                try:
                    staged = [item.a_path for item in repo.index.diff("HEAD")]
                except Exception:
                    pass
            commits_ahead = 0
            try:
                commits_ahead = len(list(repo.iter_commits(f"origin/{branch}..{branch}")))
            except Exception:
                pass
            return {
                "branch": branch,
                "is_target_branch": branch == self.TARGET_BRANCH,
                "changed_files": changed,
                "untracked_files": untracked,
                "staged_files": staged,
                "commits_ahead": commits_ahead,
                "last_commit": self._get_last_commit(),
            }
        except Exception as e:
            return {"error": str(e)}

    def auto_commit(self, message: Optional[str] = None) -> tuple:
        if not self.is_available:
            return False, "Git 利用不可"
        try:
            self._ensure_target_branch()
            repo = self._repo
            source_patterns = ["*.py", "*.txt", "*.md", "*.example"]
            staged_count = 0
            for pattern in source_patterns:
                for f in self._repo_path.rglob(pattern):
                    rel = f.relative_to(self._repo_path)
                    parts = rel.parts
                    if any(p in ("assets", "logs", "venvAlice", "__pycache__") for p in parts):
                        continue
                    try:
                        repo.index.add([str(rel)])
                        staged_count += 1
                    except Exception:
                        pass
            if staged_count == 0:
                return False, "コミット対象のファイルがありません。"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_msg = message or f"[Alice] 自動コミット - {timestamp}"
            repo.index.commit(commit_msg)
            logger.info(f"Git commit 完了: {commit_msg}")
            return True, f"コミット完了: {commit_msg}"
        except Exception as e:
            logger.error(f"Git commit エラー: {e}")
            return False, str(e)

    def switch_branch(self, branch_name: str) -> tuple:
        if not self.is_available:
            return False, "Git 利用不可"
        try:
            repo = self._repo
            if branch_name in repo.heads:
                repo.heads[branch_name].checkout()
            else:
                repo.create_head(branch_name)
                repo.heads[branch_name].checkout()
            logger.info(f"ブランチ切り替え: {branch_name}")
            return True, f"ブランチ '{branch_name}' に切り替えました。"
        except Exception as e:
            logger.error(f"ブランチ切り替えエラー: {e}")
            return False, str(e)

    def get_branches(self) -> list:
        if not self.is_available:
            return []
        try:
            return [h.name for h in self._repo.heads]
        except Exception:
            return []

    def get_log(self, max_count: int = 20) -> list:
        if not self.is_available:
            return []
        try:
            commits = []
            for commit in self._repo.iter_commits(max_count=max_count):
                commits.append({
                    "hash": commit.hexsha[:8],
                    "message": commit.message.strip(),
                    "author": str(commit.author),
                    "date": datetime.fromtimestamp(commit.committed_date).strftime("%Y-%m-%d %H:%M"),
                })
            return commits
        except Exception as e:
            logger.error(f"Git log エラー: {e}")
            return []

    def _ensure_target_branch(self) -> None:
        try:
            repo = self._repo
            if not repo.head.is_valid():
                return
            current = repo.active_branch.name
            if current != self.TARGET_BRANCH:
                if self.TARGET_BRANCH not in [h.name for h in repo.heads]:
                    repo.create_head(self.TARGET_BRANCH)
                repo.heads[self.TARGET_BRANCH].checkout()
                logger.info(f"ブランチを '{self.TARGET_BRANCH}' に切り替えました。")
        except Exception as e:
            logger.warning(f"ブランチ切り替えをスキップ: {e}")

    def _create_initial_commit(self) -> None:
        try:
            repo = self._repo
            readme = self._repo_path / "README.md"
            if readme.exists():
                repo.index.add(["README.md"])
            repo.index.commit("[Alice] 初期コミット")
            repo.create_head(self.TARGET_BRANCH)
            repo.heads[self.TARGET_BRANCH].checkout()
            logger.info("初期コミットを作成し、testbranch に切り替えました。")
        except Exception as e:
            logger.error(f"初回コミット作成エラー: {e}")

    def _get_last_commit(self) -> Optional[dict]:
        try:
            if not self._repo.head.is_valid():
                return None
            commit = self._repo.head.commit
            return {
                "hash": commit.hexsha[:8],
                "message": commit.message.strip(),
                "date": datetime.fromtimestamp(commit.committed_date).strftime("%Y-%m-%d %H:%M"),
            }
        except Exception:
            return None


# ============================================================
# VoiceEngine（外装モジュール相当）
# ============================================================

class VoiceEngine:
    """VOICEVOX を使った音声出力エンジン。"""

    # 事前コンパイル済み正規表現パターン
    _CODE_PATTERNS = [
        (re.compile(r"```[\s\S]*?```"),          ""),
        (re.compile(r"`[^`]+`"),                 ""),
        (re.compile(r"^#{1,6}\s*", re.M),        ""),
        (re.compile(r"^[\*\-\+]\s+", re.M),     ""),
        (re.compile(r"^\d+\.\s+", re.M),        ""),
        (re.compile(r"\*{1,3}([^\*]+)\*{1,3}"), r"\1"),
        (re.compile(r"https?://\S+"),            ""),
        (re.compile(r"[=\-_#\*]{3,}"),           ""),
    ]

    def __init__(self) -> None:
        self._is_speaking = False
        self._stop_flag = False
        self._lock = threading.Lock()
        self._speech_queue: "queue.Queue[str]" = queue.Queue(maxsize=30)
        self._voicevox_url = env.get("VOICEVOX_URL")
        self._speaker_id   = int(env.get("VOICEVOX_SPEAKER_ID"))
        self._speed        = float(env.get("VOICEVOX_SPEED"))
        self._pitch        = float(env.get("VOICEVOX_PITCH"))
        self._intonation   = float(env.get("VOICEVOX_INTONATION"))
        self._volume       = float(env.get("VOICEVOX_VOLUME"))

        try:
            import requests
            self._requests_available = True
        except ImportError:
            self._requests_available = False
            logger.warning("requests が未インストールです。VOICEVOX通信ができません。")

        try:
            import pygame
        except ImportError:
            self._pygame_available = False
            logger.info("pygame 未インストール。winsound にフォールバックします。")
        else:
            try:
                pygame.mixer.pre_init(44100, -16, 2, 512)
                pygame.mixer.init()
                self._pygame_available = True
                logger.info("pygame 音声エンジンを初期化しました。")
            except Exception as e:
                self._pygame_available = False
                logger.warning(f"pygame mixer 初期化失敗: {e} → winsound にフォールバックします。")

        self._speech_worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._speech_worker_thread.start()

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    @property
    def has_pending_speech(self) -> bool:
        return self._is_speaking or (not self._speech_queue.empty())

    def speak(self, text: str) -> None:
        if not self._requests_available:
            return
        cleaned = self._clean_text(text)
        if not cleaned.strip():
            return
        self._enqueue_speech(cleaned)

    def _clean_text(self, text: str) -> str:
        result = text
        for pattern, replacement in self._CODE_PATTERNS:
            result = pattern.sub(replacement, result)
        result = re.sub(r"\n{3,}", "\n\n", result)
        lines = [line.strip() for line in result.splitlines()]
        return " ".join(line for line in lines if line).strip()

    def _speak_thread(self, text: str) -> None:
        with self._lock:
            self._is_speaking = True
            self._stop_flag = False
            try:
                wav = self._text_to_wav(text)
                if wav and not self._stop_flag:
                    self._play_wav(wav)
            except Exception as e:
                logger.error(f"音声再生エラー: {e}")
            finally:
                self._is_speaking = False

    def _enqueue_speech(self, text: str) -> None:
        try:
            self._speech_queue.put_nowait(text)
            return
        except queue.Full:
            pass

        try:
            self._speech_queue.get_nowait()
            logger.warning("音声キューが満杯のため古い発話を破棄しました。")
        except queue.Empty:
            pass

        try:
            self._speech_queue.put_nowait(text)
        except queue.Full:
            logger.warning("音声キューが満杯のため発話を破棄しました。")

    def _clear_speech_queue(self) -> None:
        while True:
            try:
                self._speech_queue.get_nowait()
            except queue.Empty:
                break

    def _speech_worker(self) -> None:
        while True:
            text = self._speech_queue.get()
            try:
                if text:
                    self._speak_thread(text)
            except Exception as e:
                logger.error(f"音声キューワーカーエラー: {e}")

    def stop(self) -> None:
        self._stop_flag = True
        self._clear_speech_queue()
        if self._pygame_available:
            try:
                import pygame
                pygame.mixer.stop()
            except Exception:
                pass
        self._is_speaking = False

    def check_connection(self) -> bool:
        try:
            import requests
            resp = requests.get(f"{self._voicevox_url}/version", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def _text_to_wav(self, text: str) -> Optional[bytes]:
        try:
            import json
            import requests
            query_resp = requests.post(
                f"{self._voicevox_url}/audio_query",
                params={"text": text, "speaker": self._speaker_id},
                timeout=10,
            )
            if query_resp.status_code != 200:
                logger.error(f"audio_query 失敗: {query_resp.status_code}")
                return None
            query = query_resp.json()
            query["speedScale"]      = self._speed
            query["pitchScale"]      = self._pitch
            query["intonationScale"] = self._intonation
            query["volumeScale"]     = self._volume
            timeout = max(30, len(text) // 10)
            synth_resp = requests.post(
                f"{self._voicevox_url}/synthesis",
                params={"speaker": self._speaker_id},
                headers={"Content-Type": "application/json"},
                data=json.dumps(query),
                timeout=timeout,
            )
            if synth_resp.status_code != 200:
                logger.error(f"synthesis 失敗: {synth_resp.status_code}")
                return None
            return synth_resp.content
        except Exception as e:
            logger.error(f"WAV 生成エラー: {e}")
            return None

    def _play_wav(self, wav_data: bytes) -> None:
        if self._pygame_available:
            try:
                import io
                import pygame
                sound = pygame.mixer.Sound(io.BytesIO(wav_data))
                sound.set_volume(self._volume)
                channel = sound.play()
                while channel.get_busy():
                    if self._stop_flag:
                        channel.stop()
                        break
                    pygame.time.wait(50)
            except Exception as e:
                logger.error(f"pygame 再生エラー: {e}")
                self._play_wav_winsound(wav_data)
        else:
            self._play_wav_winsound(wav_data)

    def _play_wav_winsound(self, wav_data: bytes) -> None:
        import os
        import tempfile
        tmp = None
        try:
            import winsound
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_data)
                tmp = f.name
            winsound.PlaySound(tmp, winsound.SND_FILENAME | winsound.SND_SYNC)
        except Exception as e:
            logger.error(f"winsound 再生エラー: {e}")
        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except Exception:
                    pass


# ============================================================
# CharacterLoader（外装モジュール相当）
# ============================================================

class CharacterLoader:
    """
    キャラクター画像ファイルを読み込む外装モジュール。

    修正点:
      - initialize() をバックグラウンドスレッドで実行（非ブロッキング）
      - 起動時に検索ディレクトリと各ファイルの有無を INFO ログに出力
      - 指定ポーズの PNG が存在しない場合は alice_default.png で代替
      - get_image() のスレッドセーフ性を強化
    """

    _POSE_MAP = {
        "default":  "alice_default",
        "idle":     "alice_idle",
        "speaking": "alice_speaking",
        "thinking": "alice_thinking",
        "greeting": "alice_greeting",
    }

    def __init__(self) -> None:
        self._cache: dict = {}
        self._lock = threading.Lock()
        self._images_dir = ROOT_DIR / "assets" / "images"
        self._images_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CharacterLoader: 画像ディレクトリ = {self._images_dir}")

    def initialize(self) -> None:
        """バックグラウンドで全ポーズ画像をプリロードする（非ブロッキング）。"""
        threading.Thread(target=self._preload, daemon=True).start()

    def get_image(self, state: str = "default"):
        """
        指定ステートの画像を返す。
        指定ポーズが存在しない場合は alice_default で代替する。
        """
        if not self._pil_available():
            logger.error("CharacterLoader: Pillow が利用できません。pip install Pillow")
            return None

        with self._lock:
            if state in self._cache:
                return self._cache[state]

        # キャッシュになければファイルから読み込む
        img = self._load_from_file(state)
        if img is None and state != "default":
            img = self._load_from_file("default")

        if img is not None:
            with self._lock:
                self._cache[state] = img

        return img

    def reload(self) -> None:
        with self._lock:
            self._cache.clear()
        threading.Thread(target=self._preload, daemon=True).start()
        logger.info("CharacterLoader: キャッシュクリア、再読み込み開始")

    # ---- 内部処理 ----

    def _load_from_file(self, state: str):
        """指定ステートの PNG を読み込んで返す。失敗時は None。"""
        try:
            from PIL import Image
        except ImportError:
            return None

        fname = self._POSE_MAP.get(state, "alice_default")
        path = self._images_dir / f"{fname}.png"

        if not path.exists():
            logger.debug(f"CharacterLoader: ファイルなし → {path}")
            return None

        try:
            img = Image.open(path).convert("RGBA")
            logger.debug(
                f"CharacterLoader: 読み込み成功 [{state}] "
                f"{path.name} ({img.width}x{img.height})"
            )
            return img
        except Exception as e:
            logger.error(f"CharacterLoader: 読み込みエラー [{state}] {path}: {e}")
            return None

    def _preload(self) -> None:
        """全ポーズ画像をバックグラウンドでキャッシュに読み込む。"""
        logger.info(f"CharacterLoader: プリロード開始 ({self._images_dir})")

        found, missing = [], []
        for state, fname in self._POSE_MAP.items():
            path = self._images_dir / f"{fname}.png"
            (found if path.exists() else missing).append(fname)

        logger.info(f"CharacterLoader: 検出ファイル  = {found}")
        if missing:
            logger.warning(
                f"CharacterLoader: 未検出ファイル = {missing} "
                "→ alice_default.png で代替します"
            )

        for state in self._POSE_MAP:
            self.get_image(state)

        with self._lock:
            count = len(self._cache)
        logger.info(f"CharacterLoader: プリロード完了 ({count} ステートをキャッシュ済み)")

    def _pil_available(self) -> bool:
        try:
            from PIL import Image
            return True
        except ImportError:
            return False


# ============================================================
# AliceApp（起動制御・実行シーケンス）
# ============================================================

class AliceApp:
    """起動シーケンスを管理するエントリポイントクラス。"""
    _MODEL_CANDIDATES = (
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
    )

    def __init__(self) -> None:
        self._heart:  Optional[Any]             = None
        self._engine: Optional[AliceEngine]     = None
        self._voice:  Optional[VoiceEngine]     = None
        self._git:    Optional[GitManager]      = None
        self._loader: Optional[CharacterLoader] = None
        self._startup_notice: Dict[str, Any]    = {}

    def start(self) -> None:
        logger.info("=" * 60)
        logger.info("Alice AI 起動開始")
        logger.info("=" * 60)

        self._load_config()
        ok, backend, client, model_name, startup_notice = self._ensure_model()
        self._startup_notice = startup_notice or {}

        if ok and client:
            if backend == "local":
                self._heart = LocalAliceHeart(
                    llm=client,
                    model_name=model_name,
                    max_tokens=int(env.get("LOCAL_MODEL_MAX_TOKENS")),
                    temperature=float(env.get("LOCAL_MODEL_TEMPERATURE")),
                    top_p=float(env.get("LOCAL_MODEL_TOP_P")),
                )
            else:
                self._heart = AliceHeart(client=client, model_name=model_name)
        else:
            logger.warning("モデルが利用できません。チャット機能は無効化されます。")

        self._engine = AliceEngine(self._heart) if self._heart else None
        if self._engine:
            self._engine.load_history()

        self._voice  = self._init_voice()
        self._git    = GitManager(repo_path=str(ROOT_DIR))
        self._loader = CharacterLoader()
        self._loader.initialize()

        self._launch_gui()

    def _load_config(self) -> None:
        env_path = ROOT_DIR / ".env"
        loaded = env.load(str(env_path))
        if not loaded:
            logger.warning(".env の読み込みに失敗しました。デフォルト設定で動作します。")

    def _ensure_model(self) -> tuple:
        """
        AIバックエンドを初期化する。

        Returns:
            (ok, backend, client, model_name, startup_notice)
            backend: "gemini" | "local" | ""
        """
        backend_pref = str(env.get("AI_BACKEND") or "auto").strip().lower()
        if backend_pref not in ("auto", "gemini", "local"):
            backend_pref = "auto"

        if backend_pref in ("auto", "gemini"):
            ok, client, model_name, startup_notice = self._ensure_gemini_model()
            if ok and client:
                return True, "gemini", client, model_name, startup_notice

        if backend_pref in ("auto", "local"):
            ok, client, model_name, startup_notice = self._ensure_local_model()
            if ok and client:
                return True, "local", client, model_name, startup_notice

        logger.error("利用可能なAIバックエンドが見つかりませんでした。")
        return False, "", None, "", {}

    def _ensure_gemini_model(self) -> tuple:
        ok, client, model_name = neural.load()
        if ok:
            logger.info(f"Gemini モデルロードOK: {model_name}")
            return True, client, model_name, {}

        api_key = str(env.get("GOOGLE_API_KEY") or "").strip()
        if not api_key:
            logger.warning("GOOGLE_API_KEY が未設定のため、Gemini接続をスキップします。")
            return False, None, "", {}

        selected = self._auto_select_model(api_key)
        if selected:
            logger.warning(f"Geminiモデルを自動選択しました: {selected}")
            env.write_key("ALICE_MODEL", selected)
            neural.reset()
            ok2, client2, model_name2 = neural.load()
            if ok2:
                logger.info(f"Gemini モデルロードOK（自動選択）: {model_name2}")
                return True, client2, model_name2, {
                    "auto_selected": True,
                    "backend": "gemini",
                    "model_name": model_name2,
                    "message": f"Geminiモデルを自動選択しました: {model_name2}",
                }

        logger.warning("Gemini クライアントのロードに失敗しました。")
        return False, None, "", {}

    def _ensure_local_model(self) -> tuple:
        ok, llm, model_label, selection_info = local_loader.load()
        if ok and llm:
            logger.info(f"Local モデルロードOK: {model_label}")
            notice = dict(selection_info or {})
            if notice.get("auto_selected"):
                notice.setdefault("backend", "local")
                notice.setdefault("model_name", model_label)
                notice.setdefault("message", f"ローカルモデルを自動選択しました: {model_label}")
            return True, llm, model_label, notice
        logger.warning("Local LLM のロードに失敗しました。")
        return False, None, "", {}

    @classmethod
    def _auto_select_model(cls, api_key: str) -> Optional[str]:
        try:
            from google import genai
        except Exception as e:
            logger.warning(f"モデル自動選択をスキップしました（google-genai未利用）: {e}")
            return None

        try:
            client = genai.Client(api_key=api_key)
            models = list(client.models.list())
            model_ids = [str(getattr(m, "name", "") or "") for m in models]
            if not model_ids:
                return None

            for cand in cls._MODEL_CANDIDATES:
                if cls._is_model_available(cand, model_ids):
                    return cand

            for mid in model_ids:
                norm = cls._normalize_model_name(mid)
                if "gemini" in norm and "flash" in norm:
                    return norm
            return cls._normalize_model_name(model_ids[0])
        except Exception as e:
            logger.warning(f"モデル自動選択に失敗しました: {e}")
            return None

    @staticmethod
    def _normalize_model_name(model_id: str) -> str:
        return model_id[7:] if model_id.startswith("models/") else model_id

    @classmethod
    def _is_model_available(cls, candidate: str, model_ids: List[str]) -> bool:
        full = f"models/{candidate}"
        return (
            candidate in model_ids
            or full in model_ids
            or any(candidate == cls._normalize_model_name(mid) for mid in model_ids)
            or any(candidate in mid for mid in model_ids)
        )

    def _init_voice(self) -> Optional[VoiceEngine]:
        try:
            v = VoiceEngine()
            logger.info("VoiceEngine 初期化完了")
            return v
        except Exception as e:
            logger.error(f"VoiceEngine 初期化エラー: {e}")
            return None

    def _launch_gui(self) -> None:
        try:
            from module.window_module import AliceMainWindow
            window = AliceMainWindow(
                env_binder   = env,
                alice_engine = self._engine,
                voice_engine = self._voice,
                git_manager  = self._git,
                char_loader  = self._loader,
                startup_notice = self._startup_notice,
            )
            logger.info("GUI 起動完了。メインループを開始します。")
            window.run()
        except ImportError as e:
            logger.error(f"GUI モジュールのインポートエラー: {e}")
            sys.exit(1)
        except Exception as e:
            logger.exception(f"GUI 起動エラー: {e}")
            sys.exit(1)


# ============================================================
# エントリポイント
# ============================================================

def main() -> None:
    try:
        AliceApp().start()
    except KeyboardInterrupt:
        logger.info("中断されました。")
    except Exception as e:
        logger.exception(f"致命的なエラー: {e}")
        sys.exit(1)
    finally:
        logger.info("Alice AI 終了")


if __name__ == "__main__":
    main()
