"""
window_module.py
Alice AI メインGUIウィンドウ。
キャラクターアニメーション、チャット表示、設定、Git管理を提供する。

責務:
  - メインウィンドウの構築・表示
  - ユーザー入力の受け付けと AliceEngine への委譲
  - キャラクターアニメーションの制御
  - 設定ダイアログ・Git ダイアログの提供

制約:
  - 推論を実行しない（AliceEngine に委譲）
  - 設定参照は env_binder_module 経由のみ
"""

from __future__ import annotations

import math
import queue
import subprocess
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk
from typing import Callable, Dict, Optional

from loguru import logger

from module.display_mode_module import (
    AppMode, CharacterState, LayoutConfig, Theme,
    get_layout, DEFAULT_ANIMATION,
)

try:
    from PIL import Image, ImageTk
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ================================================================== #
# カスタムウィジェット
# ================================================================== #

class AutoScrollText(tk.Text):
    """末尾に追記すると自動スクロールするテキストウィジェット。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._auto_scroll = True
        self.bind("<MouseWheel>", lambda e: setattr(self, "_auto_scroll", False))

    def append(self, text: str, tag: Optional[str] = None) -> None:
        self.configure(state="normal")
        if tag:
            self.insert("end", text, tag)
        else:
            self.insert("end", text)
        self.configure(state="disabled")
        if self._auto_scroll:
            self.see("end")

    def clear(self) -> None:
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.configure(state="disabled")


class PlaceholderEntry(tk.Text):
    """プレースホルダー付き・自動リサイズ入力欄。文字が隠れない設計。"""

    _PLACEHOLDER_TAG = "placeholder"

    def __init__(self, parent, placeholder: str = "", min_height: int = 3,
                 max_height: int = 8, **kwargs):
        super().__init__(parent, **kwargs)
        self._placeholder = placeholder
        self._min_height = min_height
        self._max_height = max_height
        self._has_placeholder = False
        self.tag_configure(self._PLACEHOLDER_TAG, foreground="#606080")
        self._show_placeholder()
        self.bind("<FocusIn>",  self._on_focus_in)
        self.bind("<FocusOut>", self._on_focus_out)
        self.bind("<KeyRelease>", self._auto_resize)

    def _show_placeholder(self):
        self.delete("1.0", "end")
        self.insert("1.0", self._placeholder, self._PLACEHOLDER_TAG)
        self._has_placeholder = True
        self.configure(height=self._min_height)

    def _on_focus_in(self, _=None):
        if self._has_placeholder:
            self.delete("1.0", "end")
            self._has_placeholder = False

    def _on_focus_out(self, _=None):
        if not self.get("1.0", "end").strip():
            self._show_placeholder()

    def _auto_resize(self, _=None):
        if self._has_placeholder:
            return
        lines = int(self.index("end-1c").split(".")[0])
        new_h = max(self._min_height, min(lines, self._max_height))
        if int(self.cget("height")) != new_h:
            self.configure(height=new_h)

    def get_text(self) -> str:
        return "" if self._has_placeholder else self.get("1.0", "end").strip()

    def clear(self):
        self.delete("1.0", "end")
        self._show_placeholder()


# ================================================================== #
# キャラクターアニメーター
# ================================================================== #

class CharacterAnimator:
    """既存画像ファイルを使った浮遊アニメーション。"""

    def __init__(self, canvas: tk.Canvas) -> None:
        self.canvas = canvas
        self._images: Dict[str, Optional[Image.Image]] = {}
        self._tk_image: Optional[ImageTk.PhotoImage] = None
        self._image_id: Optional[int] = None
        self._state = CharacterState.IDLE
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time = time.time()
        self._breath_amp  = DEFAULT_ANIMATION.breath_amplitude
        self._breath_ms   = DEFAULT_ANIMATION.breath_period_ms
        self._speak_amp   = DEFAULT_ANIMATION.speak_bounce_amp
        self._speak_ms    = DEFAULT_ANIMATION.speak_bounce_period_ms
        self._fps         = DEFAULT_ANIMATION.fps

    def load_images(self, images: Dict[str, Optional[Image.Image]]) -> None:
        self._images = {k: v for k, v in images.items() if v is not None}
        logger.info(f"CharacterAnimator: {len(self._images)} 枚の画像をロードしました。")

    def set_state(self, state: CharacterState) -> None:
        self._state = state

    def start(self) -> None:
        if self._running or not _PIL_AVAILABLE:
            return
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        interval = 1.0 / max(1, self._fps)
        while self._running:
            t = time.time() - self._start_time
            try:
                self.canvas.after_idle(self._render, t)
            except tk.TclError:
                break
            time.sleep(interval)

    def _render(self, t: float) -> None:
        if not _PIL_AVAILABLE:
            return
        state_key = self._state.value
        img = (
            self._images.get(state_key)
            or self._images.get("default")
            or self._images.get("idle")
        )
        if img is None:
            return
        try:
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw <= 1 or ch <= 1:
                return
            ratio = min(cw / img.width, ch / img.height) * 0.90
            nw = int(img.width * ratio)
            nh = int(img.height * ratio)
            resized = img.resize((nw, nh), Image.LANCZOS)
            if self._state == CharacterState.SPEAKING:
                amp, period = self._speak_amp, self._speak_ms / 1000.0
            else:
                amp, period = self._breath_amp, self._breath_ms / 1000.0
            offset_y = int(amp * math.sin(2 * math.pi * t / period))
            x = (cw - nw) // 2
            y = (ch - nh) // 2 + offset_y
            self._tk_image = ImageTk.PhotoImage(resized)
            if self._image_id:
                self.canvas.coords(self._image_id, x, y)
                self.canvas.itemconfig(self._image_id, image=self._tk_image)
            else:
                self._image_id = self.canvas.create_image(
                    x, y, anchor="nw", image=self._tk_image
                )
        except Exception as e:
            logger.error(f"アニメーションレンダリングエラー: {e}")


# ================================================================== #
# 設定ダイアログ
# ================================================================== #

class SettingsDialog(tk.Toplevel):
    def __init__(self, parent, env_binder, on_save: Optional[Callable] = None):
        super().__init__(parent)
        self._env = env_binder
        self._on_save = on_save
        self._vars: Dict[str, tk.Variable] = {}
        theme_name = env_binder.get("APP_THEME") if env_binder else "dark"
        c = Theme.get(theme_name)
        self.title("Alice AI - 設定")
        self.geometry("700x640")
        self.configure(bg=c.bg_primary)
        self.transient(parent)
        self.grab_set()
        self._build(c)
        self._load_values()

    def _build(self, c):
        style = ttk.Style()
        style.configure("S.TNotebook", background=c.bg_primary, borderwidth=0)
        style.configure("S.TNotebook.Tab",
                        background=c.bg_secondary, foreground=c.text_secondary,
                        padding=[12, 6])
        style.map("S.TNotebook.Tab",
                  background=[("selected", c.bg_tertiary)],
                  foreground=[("selected", c.text_primary)])
        nb = ttk.Notebook(self, style="S.TNotebook")
        nb.pack(fill="both", expand=True, padx=10, pady=10)
        tabs = {
            "Alice":  self._tab_alice,
            "API":    self._tab_api,
            "Voice":  self._tab_voice,
            "表示":   self._tab_appear,
            "Git":    self._tab_git,
        }
        for label, builder in tabs.items():
            f = tk.Frame(nb, bg=c.bg_primary)
            nb.add(f, text=label)
            builder(f, c)
        btn_row = tk.Frame(self, bg=c.bg_primary)
        btn_row.pack(fill="x", padx=10, pady=(0, 10))
        self._btn(btn_row, c, "保存", self._save, c.accent_primary).pack(side="right", padx=4)
        self._btn(btn_row, c, "キャンセル", self.destroy, c.bg_tertiary, c.text_secondary).pack(side="right")

    def _tab_alice(self, f, c):
        self._row_str(f, c, "Alice 名前", "ALICE_NAME")
        self._row_str(f, c, "AIモデル", "ALICE_MODEL")

    def _tab_api(self, f, c):
        self._row_str(f, c, "Google API Key", "GOOGLE_API_KEY", show="*")
        self._row_str(f, c, "VOICEVOX URL", "VOICEVOX_URL")
        self._row_int(f, c, "VOICEVOX Speaker ID", "VOICEVOX_SPEAKER_ID")

    def _tab_voice(self, f, c):
        self._row_flt(f, c, "速度", "VOICEVOX_SPEED")
        self._row_flt(f, c, "ピッチ", "VOICEVOX_PITCH")
        self._row_flt(f, c, "抑揚", "VOICEVOX_INTONATION")
        self._row_flt(f, c, "音量", "VOICEVOX_VOLUME")

    def _tab_appear(self, f, c):
        self._row_combo(f, c, "テーマ", "APP_THEME", ["dark", "light"])

    def _tab_git(self, f, c):
        self._row_str(f, c, "Remote URL", "GIT_URL")
        self._row_str(f, c, "Branch", "GIT_BRANCH")

    # ---- ウィジェットヘルパー ----
    def _row_str(self, f, c, label, key, show=None):
        r = tk.Frame(f, bg=c.bg_primary); r.pack(fill="x", padx=14, pady=4)
        tk.Label(r, text=label, bg=c.bg_primary, fg=c.text_secondary,
                 font=("Segoe UI", 10)).pack(anchor="w")
        var = tk.StringVar(); self._vars[key] = var
        e = tk.Entry(r, textvariable=var, bg=c.bg_tertiary, fg=c.text_primary,
                     insertbackground=c.text_primary, relief="flat",
                     font=("Segoe UI", 11), highlightthickness=1,
                     highlightbackground=c.border, highlightcolor=c.border_focus)
        if show:
            e.configure(show=show)
        e.pack(fill="x", ipady=4)

    def _row_int(self, f, c, label, key):
        r = tk.Frame(f, bg=c.bg_primary); r.pack(fill="x", padx=14, pady=4)
        tk.Label(r, text=label, bg=c.bg_primary, fg=c.text_secondary,
                 font=("Segoe UI", 10)).pack(anchor="w")
        var = tk.IntVar(); self._vars[key] = var
        tk.Entry(r, textvariable=var, bg=c.bg_tertiary, fg=c.text_primary,
                 insertbackground=c.text_primary, relief="flat",
                 font=("Segoe UI", 11), highlightthickness=1,
                 highlightbackground=c.border).pack(fill="x", ipady=4)

    def _row_flt(self, f, c, label, key):
        r = tk.Frame(f, bg=c.bg_primary); r.pack(fill="x", padx=14, pady=4)
        tk.Label(r, text=label, bg=c.bg_primary, fg=c.text_secondary,
                 font=("Segoe UI", 10)).pack(anchor="w")
        var = tk.DoubleVar(); self._vars[key] = var
        tk.Entry(r, textvariable=var, bg=c.bg_tertiary, fg=c.text_primary,
                 insertbackground=c.text_primary, relief="flat",
                 font=("Segoe UI", 11), highlightthickness=1,
                 highlightbackground=c.border).pack(fill="x", ipady=4)

    def _row_combo(self, f, c, label, key, values):
        r = tk.Frame(f, bg=c.bg_primary); r.pack(fill="x", padx=14, pady=4)
        tk.Label(r, text=label, bg=c.bg_primary, fg=c.text_secondary,
                 font=("Segoe UI", 10)).pack(anchor="w")
        var = tk.StringVar(); self._vars[key] = var
        ttk.Combobox(r, textvariable=var, values=values, state="readonly").pack(fill="x")

    def _btn(self, parent, c, text, cmd, bg=None, fg=None):
        return tk.Button(parent, text=text, command=cmd,
                         bg=bg or c.accent_primary, fg=fg or c.text_primary,
                         relief="flat", font=("Segoe UI", 10, "bold"),
                         padx=16, pady=7, cursor="hand2",
                         activebackground=c.bg_hover)

    def _load_values(self):
        if not self._env:
            return
        for key, var in self._vars.items():
            value = self._env.get(key, "")
            if isinstance(var, tk.IntVar):
                try:
                    var.set(int(value))
                except Exception:
                    var.set(0)
            elif isinstance(var, tk.DoubleVar):
                try:
                    var.set(float(value))
                except Exception:
                    var.set(0.0)
            else:
                var.set(str(value))

    def _save(self):
        if self._env:
            for key, var in self._vars.items():
                self._env.write_key(key, var.get())
        if self._on_save:
            self._on_save()
        messagebox.showinfo("保存完了", "設定を保存しました。", parent=self)
        self.destroy()


# ================================================================== #
# Git ダイアログ
# ================================================================== #

class GitDialog(tk.Toplevel):
    def __init__(self, parent, git_manager, env_binder):
        super().__init__(parent)
        self._git = git_manager
        self._env = env_binder
        theme_name = env_binder.get("APP_THEME") if env_binder else "dark"
        c = Theme.get(theme_name)
        self.title("Alice AI - Git")
        self.geometry("600x520")
        self.configure(bg=c.bg_primary)
        self.transient(parent)
        self.grab_set()
        self._build(c)
        self._refresh()

    def _build(self, c):
        def lbl(text, size=12, bold=False):
            return tk.Label(self, text=text, bg=c.bg_primary, fg=c.text_primary,
                            font=("Segoe UI", size, "bold" if bold else "normal"))

        lbl("ステータス", 13, True).pack(anchor="w", padx=14, pady=(12, 2))
        self._status_box = tk.Text(self, height=7, bg=c.bg_tertiary, fg=c.text_primary,
                                   relief="flat", font=("Consolas", 10), state="disabled")
        self._status_box.pack(fill="x", padx=14, pady=2)

        lbl("ブランチ", 12, True).pack(anchor="w", padx=14, pady=(10, 2))
        bf = tk.Frame(self, bg=c.bg_primary); bf.pack(fill="x", padx=14)
        self._branch_var = tk.StringVar()
        self._branch_cb = ttk.Combobox(bf, textvariable=self._branch_var)
        self._branch_cb.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._btn(bf, c, "切替", self._switch_branch).pack(side="left")

        lbl("コミット", 12, True).pack(anchor="w", padx=14, pady=(12, 2))
        cf = tk.Frame(self, bg=c.bg_primary); cf.pack(fill="x", padx=14)
        self._commit_entry = tk.Entry(cf, bg=c.bg_tertiary, fg=c.text_primary,
                                      insertbackground=c.text_primary, relief="flat",
                                      font=("Segoe UI", 11), highlightthickness=1,
                                      highlightbackground=c.border)
        self._commit_entry.insert(0, "Manual commit")
        self._commit_entry.pack(side="left", fill="x", expand=True, padx=(0, 8), ipady=5)
        self._btn(cf, c, "Commit", self._do_commit).pack(side="left")

        lbl("ログ", 12, True).pack(anchor="w", padx=14, pady=(12, 2))
        self._log_box = tk.Text(self, height=7, bg=c.bg_tertiary, fg=c.text_secondary,
                                relief="flat", font=("Consolas", 9), state="disabled")
        self._log_box.pack(fill="both", expand=True, padx=14, pady=2)

        br = tk.Frame(self, bg=c.bg_primary); br.pack(fill="x", padx=14, pady=(4, 12))
        self._btn(br, c, "更新", self._refresh).pack(side="left")
        self._btn(br, c, "閉じる", self.destroy, c.bg_tertiary, c.text_secondary).pack(side="right")

    def _btn(self, p, c, text, cmd, bg=None, fg=None):
        return tk.Button(p, text=text, command=cmd,
                         bg=bg or c.accent_primary, fg=fg or c.text_primary,
                         relief="flat", font=("Segoe UI", 10),
                         padx=12, pady=5, cursor="hand2",
                         activebackground=c.bg_hover)

    def _set_text(self, widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _refresh(self):
        if not self._git or not self._git.is_available:
            self._set_text(self._status_box, "Git が利用できません。")
            return
        s = self._git.get_status()
        if "error" in s:
            self._set_text(self._status_box, f"エラー: {s['error']}")
            return
        lines = [
            f"Branch : {s.get('branch', '?')}",
            f"Target : {'OK' if s.get('is_target_branch') else '不一致'}",
            f"変更   : {len(s.get('changed_files', []))} ファイル",
            f"Ahead  : {s.get('commits_ahead', 0)} コミット",
        ]
        lc = s.get("last_commit")
        if lc:
            lines.append(f"最終   : [{lc['hash']}] {lc['message'][:50]}")
        self._set_text(self._status_box, "\n".join(lines))
        branches = self._git.get_branches()
        self._branch_cb["values"] = branches
        if branches:
            self._branch_var.set(s.get("branch", branches[0]))
        log_lines = [
            f"[{e['hash']}] {e['date']} {e['message'][:50]}"
            for e in self._git.get_log(10)
        ]
        self._set_text(self._log_box, "\n".join(log_lines) or "コミットなし")

    def _switch_branch(self):
        b = self._branch_var.get().strip()
        if not b:
            return
        ok, msg = self._git.switch_branch(b)
        messagebox.showinfo("ブランチ切替", msg, parent=self)
        self._refresh()

    def _do_commit(self):
        ok, msg = self._git.auto_commit(self._commit_entry.get().strip() or None)
        messagebox.showinfo("コミット", msg, parent=self)
        self._refresh()


# ================================================================== #
# メインウィンドウ
# ================================================================== #

class AliceMainWindow:
    """
    AliceApp のメインGUIウィンドウ。
    AliceApp.py から各エンジンを受け取り、表示と操作を担当する。

    レイアウト: ttk.PanedWindow によるリサイズ可能な 左右分割
      - 左ペイン（チャット）: 初期比率 65%
      - 右ペイン（キャラクター）: 初期比率 35%
    """

    # 左右ペインの初期幅比率（チャット : キャラクター）
    _CHAT_RATIO   = 0.62
    _CHAR_RATIO   = 0.38

    def __init__(
        self,
        env_binder=None,
        alice_engine=None,
        voice_engine=None,
        git_manager=None,
        char_loader=None,
    ) -> None:
        self._env         = env_binder
        self._alice       = alice_engine
        self._voice       = voice_engine
        self._git         = git_manager
        self._char_loader = char_loader

        theme_name = env_binder.get("APP_THEME") if env_binder else "dark"
        self.colors = Theme.get(theme_name)
        self._mode = AppMode.DESKTOP

        self._msg_queue: queue.Queue = queue.Queue()
        self._streaming_started = False

        self.root = tk.Tk()
        self._setup_window()
        self._build_ui()
        self._start_services()

    def run(self) -> None:
        self.root.after(100, self._process_queue)
        self.root.mainloop()

    def _enqueue(self, fn, *args, **kwargs):
        self._msg_queue.put((fn, args, kwargs))

    def _process_queue(self):
        try:
            while True:
                fn, args, kwargs = self._msg_queue.get_nowait()
                fn(*args, **kwargs)
        except queue.Empty:
            pass
        self.root.after(50, self._process_queue)

    # ---- ウィンドウセットアップ ----

    def _setup_window(self):
        layout = get_layout(self._mode)
        c = self.colors
        self.root.title("Alice AI")
        self.root.configure(bg=c.bg_primary)
        self.root.geometry(f"{layout.default_width}x{layout.default_height}")
        self.root.minsize(layout.min_width, layout.min_height)
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        self._build_menu()
        self._build_desktop_ui()

    def _build_menu(self):
        c = self.colors

        def menu(parent):
            return tk.Menu(parent, tearoff=0, bg=c.bg_secondary, fg=c.text_primary,
                           activebackground=c.accent_primary, relief="flat")

        menubar = tk.Menu(self.root, bg=c.bg_secondary, fg=c.text_primary,
                          activebackground=c.accent_primary, relief="flat")
        self.root.configure(menu=menubar)

        # ファイル
        fm = menu(menubar)
        fm.add_command(label="設定", command=self._open_settings, accelerator="Ctrl+,")
        fm.add_separator()
        fm.add_command(label="終了", command=self._on_close)
        menubar.add_cascade(label="ファイル", menu=fm)

        # 表示
        vm = menu(menubar)
        vm.add_command(label="チャット履歴をクリア", command=self._clear_chat)
        menubar.add_cascade(label="表示", menu=vm)

        # Git
        gm = menu(menubar)
        gm.add_command(label="Git マネージャー", command=self._open_git_dialog)
        gm.add_command(label="クイックコミット",  command=self._quick_commit)
        gm.add_command(label="ブランチ切替...",   command=self._switch_branch_dialog)
        menubar.add_cascade(label="Git", menu=gm)

        # ツール
        tm = menu(menubar)
        tm.add_command(label="キャラクター再読み込み", command=self._reload_character)
        tm.add_command(label="VOICEVOX 接続確認",     command=self._check_voicevox)
        tm.add_separator()
        tm.add_command(label="ログフォルダを開く",    command=self._open_logs)
        menubar.add_cascade(label="ツール", menu=tm)

        # ヘルプ
        hm = menu(menubar)
        hm.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="ヘルプ", menu=hm)

        self.root.bind("<Control-comma>", lambda e: self._open_settings())
        self.root.bind("<Return>",        lambda e: self._on_send())

    # ---- デスクトップUI構築 ----

    def _build_desktop_ui(self):
        c = self.colors
        layout = get_layout(AppMode.DESKTOP)

        # ── PanedWindow でチャット / キャラクターを左右に分割 ──────────
        # sashrelief="flat" + sashwidth=6 でスリムな仕切り線
        self._paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self._paned.pack(fill="both", expand=True)

        # 左ペイン: チャットエリア
        chat_frame = tk.Frame(self._paned, bg=c.bg_primary)
        self._paned.add(chat_frame, weight=62)

        self._build_header(chat_frame, c)
        self._build_chat_display(chat_frame, c)
        self._build_input_area(chat_frame, c)

        # 右ペイン: キャラクターエリア
        char_frame = tk.Frame(self._paned, bg=c.bg_secondary)
        self._paned.add(char_frame, weight=38)

        self._build_character_panel(char_frame, c, layout)

        # 初期サッシ位置を遅延設定（ウィンドウ描画後に実行）
        self.root.after(50, self._set_initial_sash)

        self._build_status_bar(c)

    def _set_initial_sash(self):
        """ウィンドウ幅に応じてサッシ初期位置を設定する。"""
        try:
            total = self.root.winfo_width()
            if total > 10:
                sash_pos = int(total * self._CHAT_RATIO)
                self._paned.sashpos(0, sash_pos)
        except Exception:
            pass

    def _build_header(self, parent, c):
        h = tk.Frame(parent, bg=c.bg_secondary, height=52)
        h.pack(fill="x"); h.pack_propagate(False)
        name = self._env.get("ALICE_NAME") if self._env else "Alice"
        tk.Label(h, text=f"✦ {name} AI", bg=c.bg_secondary, fg=c.accent_primary,
                 font=("Segoe UI", 15, "bold")).pack(side="left", padx=18, pady=12)
        self._status_dot   = tk.Label(h, text="●", bg=c.bg_secondary,
                                      fg=c.accent_success, font=("Segoe UI", 12))
        self._status_dot.pack(side="right", padx=6)
        self._status_label = tk.Label(h, text="Ready", bg=c.bg_secondary,
                                      fg=c.text_secondary, font=("Segoe UI", 10))
        self._status_label.pack(side="right", padx=2)

    def _build_chat_display(self, parent, c):
        f = tk.Frame(parent, bg=c.bg_primary)
        f.pack(fill="both", expand=True)
        sb = ttk.Scrollbar(f, orient="vertical")
        sb.pack(side="right", fill="y")
        fsz = 13
        self._chat_display = AutoScrollText(
            f, state="disabled", bg=c.bg_primary, fg=c.text_primary,
            relief="flat", font=("Segoe UI", fsz), wrap="word",
            cursor="arrow", padx=18, pady=14, yscrollcommand=sb.set, spacing3=4)
        self._chat_display.pack(side="left", fill="both", expand=True)
        sb.configure(command=self._chat_display.yview)
        self._setup_chat_tags(c, fsz)

    def _setup_chat_tags(self, c, fsz):
        d = self._chat_display
        d.tag_configure("user_name",  foreground=c.accent_secondary,
                        font=("Segoe UI", fsz - 1, "bold"))
        d.tag_configure("alice_name", foreground=c.accent_primary,
                        font=("Segoe UI", fsz - 1, "bold"))
        d.tag_configure("user_text",  foreground=c.text_primary, font=("Segoe UI", fsz))
        d.tag_configure("alice_text", foreground=c.text_primary, font=("Segoe UI", fsz))
        d.tag_configure("timestamp",  foreground=c.text_muted, font=("Segoe UI", fsz - 2))
        d.tag_configure("system",     foreground=c.text_muted,
                        font=("Segoe UI", fsz - 1, "italic"))
        d.tag_configure("error",      foreground=c.accent_error, font=("Segoe UI", fsz - 1))

    def _build_input_area(self, parent, c):
        container = tk.Frame(parent, bg=c.bg_secondary, pady=10)
        container.pack(fill="x")
        inner = tk.Frame(container, bg=c.bg_secondary)
        inner.pack(fill="x", padx=12)
        self._input_box = PlaceholderEntry(
            inner,
            placeholder="メッセージを入力... (Enter=送信, Shift+Enter=改行)",
            min_height=3, max_height=8,
            bg=c.bg_tertiary, fg=c.text_primary,
            insertbackground=c.text_primary, relief="flat",
            font=("Segoe UI", 12), wrap="word", padx=12, pady=8,
            highlightthickness=1, highlightbackground=c.border,
            highlightcolor=c.border_focus,
        )
        self._input_box.pack(side="left", fill="both", expand=True, pady=2)
        self._input_box.bind("<Return>",       self._on_enter_key)
        self._input_box.bind("<Shift-Return>", lambda e: None)

        btn_col = tk.Frame(inner, bg=c.bg_secondary)
        btn_col.pack(side="right", padx=(8, 0), fill="y")
        self._send_btn = tk.Button(btn_col, text="送信", command=self._on_send,
                                   bg=c.accent_primary, fg=c.text_primary,
                                   relief="flat", font=("Segoe UI", 10, "bold"),
                                   padx=14, pady=6, cursor="hand2",
                                   activebackground=c.bg_hover)
        self._send_btn.pack(pady=2)
        self._voice_btn = tk.Button(btn_col, text="音声", command=self._toggle_voice,
                                    bg=c.bg_tertiary, fg=c.text_secondary,
                                    relief="flat", font=("Segoe UI", 10),
                                    padx=10, pady=6, cursor="hand2",
                                    activebackground=c.bg_hover)
        self._voice_btn.pack(pady=2)

    def _build_character_panel(self, parent, c, layout: LayoutConfig):
        f = tk.Frame(parent, bg=c.bg_secondary)
        f.pack(fill="both", expand=True, padx=8, pady=8)
        name = self._env.get("ALICE_NAME") if self._env else "Alice"
        tk.Label(f, text=name, bg=c.bg_secondary, fg=c.accent_primary,
                 font=("Segoe UI", 12, "bold")).pack(pady=(6, 2))
        self._char_canvas = tk.Canvas(
            f, bg=c.bg_secondary, highlightthickness=0,
        )
        self._char_canvas.pack(fill="both", expand=True)
        self._animator = CharacterAnimator(self._char_canvas)
        self._thinking_label = tk.Label(
            f, text="", bg=c.bg_secondary,
            fg=c.text_muted, font=("Segoe UI", 10, "italic")
        )
        self._thinking_label.pack(pady=(2, 6))

    def _build_status_bar(self, c):
        bar = tk.Frame(self.root, bg=c.bg_secondary, height=26)
        bar.pack(fill="x", side="bottom"); bar.pack_propagate(False)
        self._statusbar = tk.Label(bar, text="Alice AI Ready", bg=c.bg_secondary,
                                   fg=c.text_muted, font=("Segoe UI", 9), anchor="w")
        self._statusbar.pack(side="left", padx=12, pady=4)
        branch = "---"
        if self._git and self._git.is_available:
            branch = self._git.get_status().get("branch", "---")
        tk.Label(bar, text=f"Branch: {branch}", bg=c.bg_secondary,
                 fg=c.text_muted, font=("Segoe UI", 9)).pack(side="right", padx=12)

    # ---- サービス起動 ----

    def _start_services(self):
        # キャラクター読み込みは 800ms 後に開始する。
        # CharacterLoader.initialize() がバックグラウンドで preload を
        # 走らせており、200ms では競合してキャッシュが空のまま
        # get_image() が呼ばれる場合があった。
        # get_image() はキャッシュになければファイルから直接読み込むため
        # 結果は正しいが、遅延を増やすことで preload 完了後に参照できるようにする。
        self.root.after(800, self._load_character)
        self.root.after(1200, self._show_greeting)

    def _load_character(self):
        if not self._char_loader or not hasattr(self, "_animator"):
            return
        def _load():
            images = {}
            for state in ("default", "idle", "speaking", "thinking", "greeting"):
                img = self._char_loader.get_image(state)
                if img is not None:
                    images[state] = img
            self._enqueue(self._on_character_loaded, images)
        threading.Thread(target=_load, daemon=True).start()

    def _on_character_loaded(self, images: dict):
        if hasattr(self, "_animator"):
            self._animator.load_images(images)
            self._animator.start()

    def _show_greeting(self):
        self._append_system("Alice AI へようこそ。メッセージを入力して会話を始めてください。")
        if self._alice:
            def _greet():
                msg = self._alice.get_greeting()
                self._enqueue(self._append_alice, msg)
            threading.Thread(target=_greet, daemon=True).start()

    # ---- チャットロジック ----

    def _on_enter_key(self, event) -> str:
        if not (event.state & 0x1):
            self._on_send()
            return "break"
        return None

    def _on_send(self):
        text = self._input_box.get_text()
        if not text:
            return
        self._input_box.clear()
        self._append_user(text)
        self._set_thinking(True)

        def _chat():
            def on_chunk(chunk):
                self._enqueue(self._append_alice_chunk, chunk)

            def on_complete(full):
                self._enqueue(self._set_thinking, False)
                self._enqueue(self._finalize_alice_stream)
                if self._voice:
                    self._voice.speak(full)

            def on_error(err):
                self._enqueue(self._append_error, err)
                self._enqueue(self._set_thinking, False)

            if self._alice:
                self._alice.send_message(
                    text,
                    on_chunk=on_chunk,
                    on_complete=on_complete,
                    on_error=on_error,
                )
            else:
                self._enqueue(self._append_alice, "（チャットエンジンが設定されていません）")
                self._enqueue(self._set_thinking, False)

        threading.Thread(target=_chat, daemon=True).start()

    def _set_thinking(self, thinking: bool):
        if hasattr(self, "_animator"):
            self._animator.set_state(
                CharacterState.THINKING if thinking else CharacterState.IDLE
            )
        if hasattr(self, "_thinking_label"):
            self._thinking_label.configure(text="考え中..." if thinking else "")
        if hasattr(self, "_status_dot"):
            color = self.colors.accent_warning if thinking else self.colors.accent_success
            self._status_dot.configure(fg=color)
            self._status_label.configure(text="考え中..." if thinking else "Ready")

    def _toggle_voice(self):
        if self._voice and self._voice.is_speaking:
            self._voice.stop()
            self._voice_btn.configure(text="音声")
        elif self._voice:
            self._voice_btn.configure(text="停止")

    # ---- チャット表示ヘルパー ----

    def _append_user(self, text):
        ts = datetime.now().strftime("%H:%M")
        self._chat_display.append(f"\n[{ts}] あなた\n", "user_name")
        self._chat_display.append(f"{text}\n", "user_text")

    def _append_alice(self, text):
        name = self._env.get("ALICE_NAME") if self._env else "Alice"
        ts = datetime.now().strftime("%H:%M")
        self._chat_display.append(f"\n[{ts}] {name}\n", "alice_name")
        self._chat_display.append(f"{text}\n", "alice_text")

    def _append_alice_chunk(self, chunk):
        if not self._streaming_started:
            self._streaming_started = True
            name = self._env.get("ALICE_NAME") if self._env else "Alice"
            ts = datetime.now().strftime("%H:%M")
            self._chat_display.append(f"\n[{ts}] {name}\n", "alice_name")
            if hasattr(self, "_animator"):
                self._animator.set_state(CharacterState.SPEAKING)
        self._chat_display.append(chunk, "alice_text")

    def _finalize_alice_stream(self):
        self._streaming_started = False
        self._chat_display.append("\n", "alice_text")
        if hasattr(self, "_animator"):
            self._animator.set_state(CharacterState.IDLE)

    def _append_system(self, text):
        self._chat_display.append(f"\n{text}\n", "system")

    def _append_error(self, text):
        self._chat_display.append(f"\nエラー: {text}\n", "error")

    def _clear_chat(self):
        if messagebox.askyesno("クリア", "チャット履歴をクリアしますか？"):
            self._chat_display.clear()
            if self._alice:
                self._alice.clear_history()
            self._append_system("チャット履歴をクリアしました。")

    # ---- メニューコマンド ----

    def _open_settings(self):
        SettingsDialog(self.root, self._env, on_save=self._on_settings_saved)

    def _on_settings_saved(self):
        self._update_status("設定を更新しました。")

    def _open_git_dialog(self):
        GitDialog(self.root, self._git, self._env)

    def _quick_commit(self):
        if not self._git or not self._git.is_available:
            messagebox.showwarning("Git", "Git が利用できません。")
            return
        ok, msg = self._git.auto_commit()
        messagebox.showinfo("Git コミット", msg)
        self._update_status(msg)

    def _switch_branch_dialog(self):
        if not self._git:
            return
        b = simpledialog.askstring("ブランチ切替", "ブランチ名:",
                                    initialvalue="testbranch")
        if b:
            ok, msg = self._git.switch_branch(b)
            messagebox.showinfo("ブランチ", msg)

    def _reload_character(self):
        if not self._char_loader:
            messagebox.showwarning("キャラクター", "CharacterLoader が利用できません。")
            return
        self._char_loader.reload()
        self._load_character()
        self._update_status("キャラクターを再読み込みしました。")

    def _check_voicevox(self):
        if self._voice:
            ok = self._voice.check_connection()
            messagebox.showinfo("VOICEVOX",
                                "接続OK" if ok else "接続できません。VOICEVOXが起動しているか確認してください。")
        else:
            messagebox.showwarning("VOICEVOX", "VoiceEngine が初期化されていません。")

    def _open_logs(self):
        from module import result_log_module as _rl
        logs = _rl.get_logs_dir()
        logs.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(f'explorer "{logs}"', shell=True)

    def _show_about(self):
        messagebox.showinfo(
            "Alice AI について",
            "Alice AI\n\nInspired by Maid-chan from\nSakurasou no Pet na Kanojo\n\n"
            "Powered by Google Gemini × VOICEVOX"
        )

    def _update_status(self, text):
        if hasattr(self, "_statusbar"):
            self._statusbar.configure(text=text)

    def _on_close(self):
        if messagebox.askyesno("終了", "Alice AI を終了しますか？"):
            if hasattr(self, "_animator"):
                self._animator.stop()
            if self._voice:
                self._voice.stop()
            logger.info("Alice AI 終了。")
            self.root.quit()
            self.root.destroy()
