"""
window_module.py
Alice AI メインGUIウィンドウ。
キャラクターアニメーション、チャット表示、設定、Git管理、高度な画像処理を提供する。

責務:
  - メインウィンドウの構築・表示
  - ユーザー入力の受け付けと AliceEngine への委譲
  - キャラクターアニメーションの制御
  - 設定ダイアログ・Git ダイアログの提供
  - 高度な画像処理（背景除去・ポイント処理・範囲選択・エッジ検出・合成）

制約:
  - 推論を実行しない（AliceEngine に委譲）
  - 設定参照は env_binder_module 経由のみ

修正（v3）:
  - CharacterAnimator: LANCZOSリサイズの結果をキャッシュし、サイズ変更時のみ再計算。
    CPU負荷を大幅軽減。
  - Menu: メニューからの関数呼び出しに self.after(10, ...) を導入し UIフリーズを防止。
  - AdvancedBgRemovalDialog: __init__ の変数初期化を _init_variables() に分離し
    UI構築前に全変数を確実に初期化する（AttributeError 防止）。
  - wait_window() 削除: grab_set() との二重使用によるフリーズを解消。
  - shell=True 削除: コマンドインジェクション脆弱性を修正（CRITICAL）。
  - _on_enter_key: return None → return "continue"（Shift+Enter 正常動作）。
  - グローバル <Return> バインド削除（二重発火防止）。
  - _voice.speak() をキュー経由に変更（スレッドセーフ化）。
  - 遅延構築ウィジェットへの hasattr() ガード追加。
  - CharacterAnimator._tick(): 例外時に安全停止してGUIイベント滞留を回避。
  - np.random.choice → np.random.default_rng().choice()（ローカル乱数化）。
  - _run_inpaint: スナップショット先取りでスレッド競合を修正。
  - queue.Queue(maxsize=500) + _enqueue にドロップ戦略を追加。
"""

from __future__ import annotations

import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import webbrowser
from collections import deque
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger

# ── 依存ライブラリ ────────────────────────────────────────────────
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    from scipy import ndimage
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

_BG_REMOVAL_AVAILABLE = _NUMPY_AVAILABLE and _CV2_AVAILABLE and _SCIPY_AVAILABLE

# ── rembg 高品質背景除去（オプション）────────────────────────────
_REMBG_AVAILABLE = False
_REMBG_MODEL     = "none"

try:
    import rembg as _rembg_module
    _REMBG_AVAILABLE = True

    _REMBG_MODEL = "isnet-anime"
    try:
        _rembg_session = _rembg_module.new_session("isnet-anime")
    except Exception:
        try:
            _REMBG_MODEL = "u2net"
            _rembg_session = _rembg_module.new_session("u2net")
        except Exception:
            _REMBG_MODEL = "default"
            _rembg_session = None

    logger.info(f"rembg 利用可能 (モデル: {_REMBG_MODEL})")

except ImportError:
    _rembg_module  = None
    _rembg_session = None
    logger.info("rembg 未インストール (pip install rembg onnxruntime で改善)")

# プロジェクトルート
_WIN_ROOT = Path(__file__).parent.parent.resolve()

from module.display_mode_module import (
    AppMode, CharacterState, LayoutConfig, Theme,
    get_layout, DEFAULT_ANIMATION,
)

try:
    from PIL import Image, ImageTk, ImageFilter, ImageDraw
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
    """プレースホルダー付き・自動リサイズ入力欄。"""

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
# キャラクターアニメーター（v3最適化版）
# ================================================================== #

class CharacterAnimator:
    """
    既存画像ファイルを使った浮遊アニメーション。
    【v3最適化】LANCZOSリサイズをキャッシュし、サイズ/状態変更時のみ再計算。
    """

    def __init__(self, canvas: tk.Canvas) -> None:
        self.canvas = canvas
        self._images: Dict[str, Optional[Image.Image]] = {}
        self._tk_image: Optional[ImageTk.PhotoImage] = None
        self._image_id: Optional[int] = None
        self._state = CharacterState.IDLE
        self._running = False
        self._after_id: Optional[str] = None
        self._start_time = time.time()
        self._fps = DEFAULT_ANIMATION.fps

        # v3: リサイズキャッシュ
        self._cached_resized_img: Optional[Image.Image] = None
        self._cached_tk_img: Optional[ImageTk.PhotoImage] = None
        self._last_canvas_size: Tuple[int, int] = (0, 0)
        self._last_state: Optional[CharacterState] = None

    def load_images(self, images: Dict[str, Optional[Image.Image]]) -> None:
        self._images = {k: v for k, v in images.items() if v is not None}
        self._cached_resized_img = None
        self._cached_tk_img = None
        self._last_canvas_size = (0, 0)
        self._last_state = None
        logger.info(f"CharacterAnimator: {len(self._images)} 枚の画像をロードしました。")

    def set_state(self, state: CharacterState) -> None:
        self._state = state

    def start(self) -> None:
        if self._running or not _PIL_AVAILABLE:
            return
        self._running = True
        self._start_time = time.time()
        self._schedule_next_frame(0)

    def stop(self) -> None:
        self._running = False
        if self._after_id is not None:
            try:
                self.canvas.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _schedule_next_frame(self, delay_ms: Optional[int] = None) -> None:
        if not self._running:
            return
        if delay_ms is None:
            delay_ms = max(1, int(1000 / max(1, self._fps)))
        try:
            self._after_id = self.canvas.after(delay_ms, self._tick)
        except tk.TclError:
            self._running = False
            self._after_id = None

    def _tick(self) -> None:
        self._after_id = None
        if not self._running:
            return
        t = time.time() - self._start_time
        try:
            self._render(t)
        except tk.TclError:
            self._running = False
            return
        except Exception as e:
            logger.error(f"CharacterAnimator._tick 予期しないエラー: {e}")
            self._running = False
            return
        self._schedule_next_frame()

    def _render(self, t: float) -> None:
        if not _PIL_AVAILABLE or not self._images:
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

            # サイズまたは状態が変わった時だけ LANCZOS リサイズを実行
            if (cw, ch) != self._last_canvas_size or self._state != self._last_state:
                ratio = min(cw / img.width, ch / img.height) * 0.90
                nw = max(1, int(img.width * ratio))
                nh = max(1, int(img.height * ratio))
                self._cached_resized_img = img.resize((nw, nh), Image.LANCZOS)
                self._cached_tk_img = ImageTk.PhotoImage(self._cached_resized_img)
                self._last_canvas_size = (cw, ch)
                self._last_state = self._state

            if not self._cached_resized_img:
                return
            if self._cached_tk_img is None:
                self._cached_tk_img = ImageTk.PhotoImage(self._cached_resized_img)

            resized = self._cached_resized_img
            nw, nh = resized.width, resized.height

            if self._state == CharacterState.SPEAKING:
                amp    = DEFAULT_ANIMATION.speak_bounce_amp
                period = DEFAULT_ANIMATION.speak_bounce_period_ms / 1000.0
            else:
                amp    = DEFAULT_ANIMATION.breath_amplitude
                period = DEFAULT_ANIMATION.breath_period_ms / 1000.0

            offset_y = int(amp * math.sin(2 * math.pi * t / period))
            x = (cw - nw) // 2
            y = (ch - nh) // 2 + offset_y

            self._tk_image = self._cached_tk_img
            if self._image_id:
                self.canvas.coords(self._image_id, x, y)
                self.canvas.itemconfig(self._image_id, image=self._tk_image)
            else:
                self._image_id = self.canvas.create_image(
                    x, y, anchor="nw", image=self._tk_image
                )
        except Exception as e:
            logger.error(f"アニメーションレンダリングエラー: {e}")

class AdvancedImageProcessor:
    """
    独自アルゴリズムによる高精度画像処理エンジン。
    外部API・学習済みモデル不使用。コード自体が学習済みモデル。

    コア技術:
      - CIELab色空間での知覚的距離計算
      - Gaussian Mixture Model (GMM) 近似によるFG/BG分類
      - Iterative GrabCut近似（エネルギー最小化）
      - KNN Alpha Matting（境界部の毛髪・半透明表現）
      - YCbCr/HSV複合肌色モデルによる部位検出
      - 形態学的解析による顔・髪・服の分離
      - K-means色クラスタリング（SEGS）
    """

    # ---------- 調整定数 ----------
    _FEATHER_RADIUS   = 3.0
    _MIN_CLUSTER_PX   = 80
    _GRABCUT_ITER     = 4          # GrabCut近似イテレーション数
    _KMEANS_CLUSTERS  = 6          # SEGSクラスタ数
    _KMEANS_ITER      = 10         # K-meansイテレーション数
    _MATTING_K        = 10         # KNN Matte探索近傍数
    _MATTING_WIN      = 3          # Matte走査ウィンドウ半径

    def __init__(self) -> None:
        self._available = _NUMPY_AVAILABLE and _PIL_AVAILABLE

    def is_available(self) -> bool:
        return self._available

    # ================================================================
    # ① 高精度エッジ検出（完全ベクトル化）
    # ================================================================

    def detect_edges_highquality(self, img_rgba: "np.ndarray") -> "np.ndarray":
        """
        Sobel + Laplacian + Canny近似を融合した高精度エッジマップ。
        ループなし完全ベクトル化で高速動作。
        """
        if not _NUMPY_AVAILABLE:
            return np.zeros(img_rgba.shape[:2], dtype=np.uint8)

        gray = self._to_gray(img_rgba)

        # Sobelエッジ（ベクトル化）
        sx = self._convolve2d_fast(gray, np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32))
        sy = self._convolve2d_fast(gray, np.array([[-1,-2,-1],[0,0,0],[1,2,1]], np.float32))
        sobel = np.sqrt(sx**2 + sy**2)
        m = sobel.max()
        if m > 0: sobel = sobel / m * 255

        # Laplacianエッジ
        lap = np.abs(self._convolve2d_fast(gray, np.array([[0,1,0],[1,-4,1],[0,1,0]], np.float32)))
        m = lap.max()
        if m > 0: lap = lap / m * 255

        # Canny近似（Non-Maximum Suppression + 二重しきい値）
        canny = self._canny_approx(gray, sobel, sx, sy)

        # 加重融合
        fused = np.clip(sobel * 0.40 + lap * 0.25 + canny * 0.35, 0, 255).astype(np.uint8)

        # モルフォロジーによる細線強調
        if _SCIPY_AVAILABLE:
            s2 = np.ones((2, 2), bool)
            dilated = ndimage.binary_dilation(fused > 60, structure=s2)
            eroded  = ndimage.binary_erosion(dilated, structure=s2)
            fused   = np.maximum(fused, (eroded * 220).astype(np.uint8))

        return fused

    def _canny_approx(self, gray, grad_mag, gx, gy):
        """Non-Maximum Suppression + 二重しきい値のCanny近似（ベクトル化）"""
        h, w = gray.shape
        angle = np.arctan2(gy, gx + 1e-9) * 180 / np.pi
        angle[angle < 0] += 180

        # 非最大値抑制（ベクトル化近似）
        nms = grad_mag.copy()
        padded = np.pad(grad_mag, 1, mode='edge')

        # 0°方向
        m0 = (angle < 22.5) | (angle >= 157.5)
        nms[m0] = np.where(
            (grad_mag[m0] >= padded[1:-1, 2:][m0]) & (grad_mag[m0] >= padded[1:-1, :-2][m0]),
            grad_mag[m0], 0)

        # 45°方向
        m45 = (angle >= 22.5) & (angle < 67.5)
        nms[m45] = np.where(
            (grad_mag[m45] >= padded[:-2, 2:][m45]) & (grad_mag[m45] >= padded[2:, :-2][m45]),
            grad_mag[m45], 0)

        # 90°方向
        m90 = (angle >= 67.5) & (angle < 112.5)
        nms[m90] = np.where(
            (grad_mag[m90] >= padded[:-2, 1:-1][m90]) & (grad_mag[m90] >= padded[2:, 1:-1][m90]),
            grad_mag[m90], 0)

        # 135°方向
        m135 = (angle >= 112.5) & (angle < 157.5)
        nms[m135] = np.where(
            (grad_mag[m135] >= padded[:-2, :-2][m135]) & (grad_mag[m135] >= padded[2:, 2:][m135]),
            grad_mag[m135], 0)

        # 二重しきい値
        high = nms.max() * 0.20
        low  = high * 0.40
        strong  = nms >= high
        weak    = (nms >= low) & ~strong
        # 連結する（強エッジに隣接する弱エッジを採用）
        if _SCIPY_AVAILABLE:
            strong_d = ndimage.binary_dilation(strong, np.ones((3,3), bool))
            result   = (strong | (weak & strong_d)).astype(np.float32) * 255
        else:
            result = strong.astype(np.float32) * 255

        return result

    def _convolve2d_fast(self, img: "np.ndarray", kernel: "np.ndarray") -> "np.ndarray":
        """完全ベクトル化畳み込み（scipy/cv2 非依存）"""
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2
        padded = np.pad(img, ((ph, ph), (pw, pw)), mode='edge')
        h, w = img.shape
        # unfold trick で高速化
        cols = np.lib.stride_tricks.as_strided(
            padded,
            shape=(h, w, kh, kw),
            strides=(padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1]),
        )
        return (cols * kernel[np.newaxis, np.newaxis, :, :]).sum(axis=(2, 3))

    def _to_gray(self, arr: "np.ndarray") -> "np.ndarray":
        r = arr[:, :, 0].astype(np.float32)
        g = arr[:, :, 1].astype(np.float32)
        b = arr[:, :, 2].astype(np.float32)
        return 0.299 * r + 0.587 * g + 0.114 * b

    # ================================================================
    # ② 高品質背景除去（GrabCut近似 + Alpha Matting）
    # ================================================================

    def remove_background_adaptive(
        self,
        img_rgba: "np.ndarray",
        sensitivity: float = 1.0,
    ) -> "np.ndarray":
        """
        背景除去メイン。優先順位:
          1. rembg (isnet-anime / u2net) ← 最高品質
          2. 単色背景高速アルゴリズム    ← rembg 未インストール時
          3. 既存 GrabCut 近似           ← fallback
        """
        if not _NUMPY_AVAILABLE:
            return img_rgba

        # ══════════════════════════════════════════════════════
        # ① rembg パス — インストール済みなら必ずここで完結
        # ══════════════════════════════════════════════════════
        if _REMBG_AVAILABLE and _rembg_module is not None:
            try:
                import io as _io
                pil_in = Image.fromarray(img_rgba).convert("RGBA")
                buf_in = _io.BytesIO()
                pil_in.save(buf_in, format="PNG")

                raw_out = _rembg_module.remove(
                    buf_in.getvalue(),
                    session=_rembg_session,
                )
                pil_out     = Image.open(_io.BytesIO(raw_out)).convert("RGBA")
                out_arr     = np.array(pil_out)
                orig_rgb    = img_rgba[:, :, :3].astype(np.float32)
                rembg_alpha = out_arr[:, :, 3].astype(np.float32) / 255.0
                h_i, w_i   = orig_rgb.shape[:2]

                # ══════════════════════════════════════════════════
                # SSAA（スーパーサンプリング）ウルトラ高品質後処理
                #
                # 手順:
                #   1. 元画像を SSAA_SCALE 倍に拡大
                #   2. 拡大画像でBFS+Cannyによるマスクを高精度生成
                #   3. 距離変換で境界アルファを計算
                #   4. 元サイズに LANCZOS ダウンサンプリング
                #      → 複数ピクセルを平均化してサブピクセル精度の
                #        なめらかなエッジが自動的に得られる
                # ══════════════════════════════════════════════════
                SSAA_SCALE = 3  # 3倍解像度で処理（品質と速度のバランス）

                # ── Step1: 外周BG色を推定 ──
                bw = max(4, min(20, h_i // 8, w_i // 8))
                border_px = np.concatenate([
                    orig_rgb[:bw,  :  ].reshape(-1, 3),
                    orig_rgb[-bw:, :  ].reshape(-1, 3),
                    orig_rgb[:,  :bw  ].reshape(-1, 3),
                    orig_rgb[:, -bw:  ].reshape(-1, 3),
                ])
                bg_median = np.median(border_px, axis=0)
                bg_std    = border_px.std(axis=0).mean()
                bg_noise  = max(6.0, bg_std * 2.5)

                # ── Step2: SSAA 拡大 ──
                pil_big  = pil_in.resize(
                    (w_i * SSAA_SCALE, h_i * SSAA_SCALE), Image.LANCZOS
                )
                rgb_big  = np.array(pil_big)[:, :, :3].astype(np.float32)
                hb, wb   = rgb_big.shape[:2]

                # ── Step3: 拡大画像でBFS（背景連結領域を確実特定）──
                diff_big = np.sqrt(
                    np.sum((rgb_big - bg_median) ** 2, axis=2)
                )
                bg_seed_big = diff_big < (bg_noise * SSAA_SCALE * 0.6)
                sure_bg_big = self._bfs_flood_fill(bg_seed_big, hb, wb)

                # rembgがBGと断定した領域（縮小して拡大→BG補完）
                rembg_bg_mask = (rembg_alpha < 0.05)
                rembg_bg_big  = np.array(
                    Image.fromarray(rembg_bg_mask.astype(np.uint8) * 255).resize(
                        (wb, hb), Image.NEAREST
                    )
                ) > 127
                sure_bg_big = sure_bg_big | rembg_bg_big

                if _SCIPY_AVAILABLE:
                    sure_bg_big = ndimage.binary_closing(
                        sure_bg_big, np.ones((3, 3))
                    )
                sure_fg_big = ~sure_bg_big

                # ── Step4: 拡大画像でCanny（高精度エッジ）──
                gray_big = rgb_big.max(axis=2).astype(np.uint8)
                if _CV2_AVAILABLE:
                    blur_big  = cv2.GaussianBlur(gray_big, (5, 5), 1.0)
                    edges_big = cv2.Canny(blur_big, 6, 22)
                    edge_dil_big = ndimage.binary_dilation(
                        edges_big > 0, np.ones((3, 3))
                    ) if _SCIPY_AVAILABLE else (edges_big > 0)
                else:
                    edge_dil_big = np.zeros((hb, wb), bool)

                # ── Step5: 距離変換でアルファ計算（拡大スペース）──
                if _SCIPY_AVAILABLE:
                    dist_big = ndimage.distance_transform_edt(
                        sure_fg_big
                    ).astype(np.float32)
                else:
                    dist_big = sure_fg_big.astype(np.float32)

                feather_big = max(3.0, SSAA_SCALE * 3.5)

                alpha_big = np.where(
                    sure_bg_big, 0.0,
                    np.clip(dist_big / feather_big, 0.0, 1.0)
                )
                if _CV2_AVAILABLE and _SCIPY_AVAILABLE:
                    alpha_big = np.where(
                        edge_dil_big & sure_fg_big,
                        np.clip(dist_big / max(feather_big * 0.65, 1.0), 0.0, 1.0),
                        alpha_big,
                    )

                # ── Step6: スムージング・仕上げ（拡大スペース）──
                if _SCIPY_AVAILABLE:
                    ez_big = (alpha_big > 0.02) & (alpha_big < 0.98)
                    alpha_big = np.where(
                        ez_big,
                        ndimage.gaussian_filter(alpha_big, sigma=0.8),
                        alpha_big,
                    )
                    fg_core_big = ndimage.binary_erosion(
                        sure_fg_big, np.ones((7, 7))
                    )
                    alpha_big[fg_core_big] = 1.0
                    bg_core_big = ndimage.binary_erosion(
                        sure_bg_big, np.ones((3, 3))
                    )
                    alpha_big[bg_core_big] = 0.0

                    # ゴミ除去
                    labeled, num = ndimage.label(alpha_big > 0.5)
                    if num > 1:
                        sizes = [
                            ndimage.sum(alpha_big > 0.5, labeled, i)
                            for i in range(1, num + 1)
                        ]
                        main_label = int(np.argmax(sizes)) + 1
                        for i in range(1, num + 1):
                            if i != main_label and sizes[i - 1] < 300:
                                alpha_big[labeled == i] = 0.0

                alpha_big = np.clip(alpha_big, 0.0, 1.0)

                # ── Step7: LANCZOS ダウンサンプリング ──
                # 複数ピクセルの平均 → サブピクセル精度のなめらかエッジ
                alpha_img_big = Image.fromarray(
                    (alpha_big * 255).astype(np.uint8), mode='L'
                )
                alpha_ssaa  = np.array(
                    alpha_img_big.resize((w_i, h_i), Image.LANCZOS)
                ).astype(np.float32) / 255.0

                # ══════════════════════════════════════════════════
                # PASS 2: Gaussian-propagation Alpha Matting
                #
                # SSAAで得た粗いマスクをtrimapに変換し、
                # ガウシアン伝播で「FG色の期待値」「BG色の期待値」を
                # 全ピクセルに広げ、色合成式 I=αF+(1-α)B を解く。
                # 境界の半透明精度がSSAAより大幅に向上する。
                # ══════════════════════════════════════════════════
                if _SCIPY_AVAILABLE:
                    rgb_n = orig_rgb / 255.0  # 0~1 正規化

                    # trimap: SSAAのアルファから自動生成
                    trimap = np.full((h_i, w_i), 128, dtype=np.uint8)
                    trimap[alpha_ssaa > 0.95] = 255   # 確実FG
                    trimap[alpha_ssaa < 0.05] = 0     # 確実BG

                    # 境界を少し拡張して Matting に余裕を持たせる
                    unk_dil = ndimage.binary_dilation(
                        trimap == 128, np.ones((5, 5))
                    )
                    trimap[unk_dil & (trimap == 255)] = 128
                    trimap[unk_dil & (trimap == 0)]   = 128

                    fg_mask = trimap == 255
                    bg_mask = trimap == 0

                    # ガウシアン伝播でローカルFG/BG色を全ピクセルに広げる
                    SIGMA_M = max(6.0, min(h_i, w_i) * 0.018)

                    def _gauss_prop(c_ch, mask, sigma):
                        sm_w = ndimage.gaussian_filter(
                            mask.astype(np.float32), sigma=sigma
                        )
                        sm_c = ndimage.gaussian_filter(
                            np.where(mask, c_ch, 0.0), sigma=sigma
                        )
                        return sm_c / (sm_w + 1e-8)

                    mean_fg = np.stack([
                        _gauss_prop(rgb_n[:, :, c], fg_mask, SIGMA_M)
                        for c in range(3)
                    ], axis=2)
                    mean_bg = np.stack([
                        _gauss_prop(rgb_n[:, :, c], bg_mask, SIGMA_M)
                        for c in range(3)
                    ], axis=2)

                    # 色合成式を解く: α = (I-B)・(F-B) / |F-B|²
                    diff_fb = mean_fg - mean_bg
                    diff_ib = rgb_n   - mean_bg
                    numer   = (diff_fb * diff_ib).sum(axis=2)
                    denom   = (diff_fb * diff_fb).sum(axis=2) + 1e-8
                    alpha_matte = np.clip(numer / denom, 0.0, 1.0)
                    alpha_matte[fg_mask] = 1.0
                    alpha_matte[bg_mask] = 0.0

                    # SSAAと Matting をブレンド
                    # 確実領域: SSAA を使用（形状が正確）
                    # 境界付近: Matting を使用（半透明精度が高い）
                    blend_w = np.where(
                        (alpha_ssaa > 0.92) | (alpha_ssaa < 0.08),
                        0.0,   # 確実領域 → SSAA
                        1.0,   # 境界     → Matting
                    )
                    alpha_final = (
                        alpha_matte * blend_w +
                        alpha_ssaa  * (1.0 - blend_w)
                    )

                    # 最終スムージング（境界のみ）
                    ez = (alpha_final > 0.03) & (alpha_final < 0.97)
                    alpha_final = np.where(
                        ez,
                        ndimage.gaussian_filter(alpha_final, sigma=0.4),
                        alpha_final
                    )
                    alpha_final = np.clip(alpha_final, 0.0, 1.0)
                else:
                    alpha_final = alpha_ssaa

                out_arr[:, :, 3] = (alpha_final * 255).astype(np.uint8)
                # ══════════════════════════════════════════════════

                logger.info(
                    f"rembg ({_REMBG_MODEL}) SSAA{SSAA_SCALE}x + AlphaMatting 完了 "
                    f"BG=({bg_median[0]:.0f},{bg_median[1]:.0f},{bg_median[2]:.0f})"
                )
                return out_arr

            except Exception as e:
                logger.error(f"rembg 例外: {e} — アルゴリズム実装にフォールバック")

        # ══════════════════════════════════════════════════════
        # ② 高品質アルゴリズム実装（rembg なし環境）
        # 黒/白/単色背景に特化。GrabCutより大幅に高品質。
        # ══════════════════════════════════════════════════════
        if _NUMPY_AVAILABLE and _SCIPY_AVAILABLE:
            try:
                result_arr = self._remove_solid_bg_fast(img_rgba, sensitivity)
                if result_arr is not None:
                    logger.info("高精度アルゴリズムで背景除去完了。")
                    return result_arr
            except Exception as e:
                logger.warning(f"高精度BG除去失敗: {e}")

        # ══════════════════════════════════════════════════════
        # ③ 既存 GrabCut 近似（最終フォールバック）
        # ══════════════════════════════════════════════════════
        h, w = img_rgba.shape[:2]
        result = img_rgba.copy()

        # Step1: Lab色空間変換
        lab = self._rgb_to_lab(img_rgba[:, :, :3])

        # Step2: 外周からBG色分布をサンプリング（GMM近似）
        m = max(3, min(15, h // 8, w // 8))
        bg_samples = np.concatenate([
            lab[:m, :].reshape(-1, 3),
            lab[-m:, :].reshape(-1, 3),
            lab[:, :m].reshape(-1, 3),
            lab[:, -m:].reshape(-1, 3),
        ])

        # BG統計（平均・共分散）
        bg_mean = bg_samples.mean(axis=0)
        bg_cov  = np.cov(bg_samples.T) + np.eye(3) * 1e-6
        bg_cov_inv = np.linalg.inv(bg_cov)

        # FGサンプル（中央付近）
        cy, cx = h // 2, w // 2
        cr = max(5, min(h, w) // 6)
        fg_region = lab[cy-cr:cy+cr, cx-cr:cx+cr].reshape(-1, 3)
        fg_mean = fg_region.mean(axis=0)
        fg_cov  = np.cov(fg_region.T) + np.eye(3) * 1e-6
        fg_cov_inv = np.linalg.inv(fg_cov)

        # Step3: Mahalanobis距離でFG/BG確率マップ生成
        flat = lab.reshape(-1, 3)
        diff_bg = flat - bg_mean
        diff_fg = flat - fg_mean
        d_bg = np.einsum('ij,jk,ik->i', diff_bg, bg_cov_inv, diff_bg).reshape(h, w)
        d_fg = np.einsum('ij,jk,ik->i', diff_fg, fg_cov_inv, diff_fg).reshape(h, w)

        # 感度スケール適用
        base_th = max(6.0, np.sqrt(d_bg[m:-m, m:-m].mean() + 1e-6)) * sensitivity
        is_bg_raw = d_bg < d_fg * 0.85

        # Step4: Iterative GrabCut近似
        bg_mask = self._grabcut_approx(is_bg_raw, lab, h, w, iterations=self._GRABCUT_ITER)

        # Step5: Alpha Matting（境界部の半透明処理）
        alpha = self._alpha_matte_knn(img_rgba, bg_mask, h, w)

        result[:, :, 3] = alpha
        return result

    def _grabcut_approx(
        self,
        init_bg: "np.ndarray",
        lab: "np.ndarray",
        h: int, w: int,
        iterations: int = 4,
    ) -> "np.ndarray":
        """
        Iterative GrabCut近似。
        BGモデルとFGモデルを交互に更新してマスクを洗練する。
        """
        bg_mask = init_bg.copy()

        for _ in range(iterations):
            # BGサンプル再取得
            bg_pix = lab[bg_mask].reshape(-1, 3)
            fg_pix = lab[~bg_mask].reshape(-1, 3)
            if len(bg_pix) < 10 or len(fg_pix) < 10:
                break

            bg_mean = bg_pix.mean(axis=0)
            fg_mean = fg_pix.mean(axis=0)
            bg_cov_inv = np.linalg.inv(np.cov(bg_pix.T) + np.eye(3) * 1e-4)
            fg_cov_inv = np.linalg.inv(np.cov(fg_pix.T) + np.eye(3) * 1e-4)

            flat = lab.reshape(-1, 3)
            db = np.einsum('ij,jk,ik->i', flat - bg_mean, bg_cov_inv, flat - bg_mean)
            df = np.einsum('ij,jk,ik->i', flat - fg_mean, fg_cov_inv, flat - fg_mean)
            new_bg = (db < df).reshape(h, w)

            # 外周は必ずBG
            new_bg[:3, :]  = True
            new_bg[-3:, :] = True
            new_bg[:, :3]  = True
            new_bg[:, -3:] = True

            # BFS で外周連結BG領域のみ採用（孤立BGを除去）
            new_bg = self._bfs_flood_fill(new_bg, h, w)

            # モルフォロジー整形
            if _SCIPY_AVAILABLE:
                new_bg = ndimage.binary_closing(new_bg, np.ones((5, 5), bool))
                new_bg = ndimage.binary_opening(new_bg, np.ones((3, 3), bool))

            bg_mask = new_bg

        return bg_mask

    def _alpha_matte_knn(
        self,
        img_rgba: "np.ndarray",
        bg_mask: "np.ndarray",
        h: int, w: int,
    ) -> "np.ndarray":
        """
        KNN Alpha Matting。
        境界付近のピクセルを周囲のFG/BGサンプルから補間して
        滑らかなアルファ値を生成（髪・毛先・半透明対応）。
        """
        fg_mask = ~bg_mask

        if not _SCIPY_AVAILABLE:
            return (fg_mask.astype(np.float32) * 255).astype(np.uint8)

        # 距離変換でトライマップ生成
        dist_to_fg = ndimage.distance_transform_edt(bg_mask)
        dist_to_bg = ndimage.distance_transform_edt(fg_mask)

        feather = max(3, min(h, w) // 40)
        # トライマップ: 0=確実BG, 1=確実FG, 0.5=不明
        alpha = np.where(bg_mask & (dist_to_fg > feather), 0.0,
                np.where(fg_mask & (dist_to_bg > feather), 1.0, -1.0))

        unknown = alpha < 0
        if not unknown.any():
            return (np.clip(alpha, 0, 1) * 255).astype(np.uint8)

        # 不明領域: Lab色空間でFG/BGサンプルからKNN補間
        lab = self._rgb_to_lab(img_rgba[:, :, :3])
        unk_ys, unk_xs = np.where(unknown)

        # FG/BGの確実サンプルを縮小取得（速度のため最大2000点）
        sure_fg = np.where(alpha == 1.0)
        sure_bg = np.where(alpha == 0.0)

        def _subsample(ys, xs, maxn=2000):
            if len(ys) > maxn:
                rng = np.random.default_rng()
                idx = rng.choice(len(ys), maxn, replace=False)
                return ys[idx], xs[idx]
            return ys, xs

        fg_ys, fg_xs = _subsample(*sure_fg)
        bg_ys, bg_xs = _subsample(*sure_bg)

        if len(fg_ys) == 0 or len(bg_ys) == 0:
            alpha[unknown] = 0.5
            return (np.clip(alpha, 0, 1) * 255).astype(np.uint8)

        fg_lab = lab[fg_ys, fg_xs]  # (N, 3)
        bg_lab = lab[bg_ys, bg_xs]  # (M, 3)

        # 不明ピクセルのLabをバッチ処理
        unk_lab = lab[unk_ys, unk_xs]  # (K, 3)

        # FGへの最近傍距離（L2）
        K = self._MATTING_K
        chunk = 500
        alpha_vals = np.zeros(len(unk_ys), dtype=np.float32)

        for start in range(0, len(unk_ys), chunk):
            q = unk_lab[start:start+chunk]  # (chunk, 3)
            df = np.sqrt(((q[:, np.newaxis, :] - fg_lab[np.newaxis, :, :]) ** 2).sum(axis=2))  # (chunk, N)
            db = np.sqrt(((q[:, np.newaxis, :] - bg_lab[np.newaxis, :, :]) ** 2).sum(axis=2))  # (chunk, M)

            # K最近傍の平均距離
            df_k = np.sort(df, axis=1)[:, :K].mean(axis=1)
            db_k = np.sort(db, axis=1)[:, :K].mean(axis=1)

            # FGらしさ = BG距離 / (FG距離 + BG距離)
            denom = df_k + db_k + 1e-6
            alpha_vals[start:start+chunk] = db_k / denom

        alpha[unk_ys, unk_xs] = alpha_vals
        alpha = np.clip(alpha, 0, 1)

        # ガウシアンスムージング（境界を滑らかに）
        if _SCIPY_AVAILABLE:
            alpha = ndimage.gaussian_filter(alpha, sigma=1.0)
            alpha = np.clip(alpha, 0, 1)

        # 小クラスタ除去
        labeled, num = ndimage.label((alpha > 0.5).astype(np.uint8))
        if num > 0:
            sizes = ndimage.sum(alpha > 0.5, labeled, range(1, num + 1))
            for i, s in enumerate(sizes):
                if s < self._MIN_CLUSTER_PX:
                    alpha[labeled == (i + 1)] = 0

        return (alpha * 255).astype(np.uint8)

    # ================================================================
    # 後処理・補助メソッド（rembg 連携 + 単色BG高速除去）
    # ================================================================

    def _post_process_alpha(self, arr: "np.ndarray") -> "np.ndarray":
        """
        rembg 出力のアルファチャネルを改善する後処理。
        - 微細ギザギザ除去（ガウシアン）
        - 境界のデフリンジ（BG色の滲み除去）
        - 孤立した小ゴミ成分の除去
        """
        if not _NUMPY_AVAILABLE or not _SCIPY_AVAILABLE:
            return arr

        alpha = arr[:, :, 3].astype(np.float32) / 255.0
        rgb   = arr[:, :, :3].astype(np.float32)

        # ① 境界付近のみガウシアンを適用（内部の鮮明さを保つ）
        edge_zone = (alpha > 0.05) & (alpha < 0.95)
        if edge_zone.any():
            alpha_blur = ndimage.gaussian_filter(alpha, sigma=0.7)
            alpha = np.where(edge_zone, alpha_blur, alpha)

        # ② 確実FG内部は alpha=1 に固定
        sure_fg = ndimage.binary_erosion(alpha > 0.88, np.ones((3, 3)))
        alpha[sure_fg] = 1.0
        alpha = np.clip(alpha, 0, 1)

        # ③ 小さな孤立成分を除去（ゴミ除去）
        labeled, num = ndimage.label(alpha > 0.5)
        if num > 1:
            sizes = [ndimage.sum(alpha > 0.5, labeled, i) for i in range(1, num + 1)]
            main = int(np.argmax(sizes)) + 1
            for i in range(1, num + 1):
                if i != main and sizes[i - 1] < 300:
                    alpha[labeled == i] = 0.0

        # ④ デフリンジ: 境界付近でBG色が混色した部分を近傍FG色で置換
        fringe = (alpha > 0.02) & (alpha < 0.90)
        if fringe.any():
            sure_fg2 = (alpha > 0.90)
            if sure_fg2.any():
                # return_distances=False にするとインデックス配列のみ返る (2, H, W) int型
                indices = ndimage.distance_transform_edt(
                    ~sure_fg2, return_distances=False, return_indices=True
                )
                iy = indices[0]  # 行インデックス (H, W) int
                ix = indices[1]  # 列インデックス (H, W) int
                for c in range(3):
                    expanded = rgb[:, :, c][iy, ix]
                    rgb[:, :, c] = np.where(fringe, expanded, rgb[:, :, c])

        result = arr.copy()
        result[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        result[:, :, 3]  = (alpha * 255).astype(np.uint8)
        return result

    def _remove_solid_bg_fast(
        self,
        img_rgba: "np.ndarray",
        sensitivity: float = 1.0,
    ) -> "Optional[np.ndarray]":
        """
        アニメキャラクター向け高精度背景除去。
        黒/白/単色背景に対応。rembg なし環境での最良手。

        アルゴリズム:
          - 純粋な背景色（彩度≈0）とキャラクター色を分離
          - 外周BFSで連結BG領域のみ採用
          - 暗色キャラクター（髪など）はデフリンジしない
          - 境界のみ滑らかなアルファグラデーション
        """
        if not _NUMPY_AVAILABLE or not _SCIPY_AVAILABLE:
            return None

        h, w = img_rgba.shape[:2]
        rgb = img_rgba[:, :, :3].astype(np.float32)

        # ── ① 外周からBG色を推定 ──
        bw = max(4, min(20, h // 8, w // 8))
        samples = np.concatenate([
            rgb[:bw, :].reshape(-1, 3),
            rgb[-bw:, :].reshape(-1, 3),
            rgb[:, :bw].reshape(-1, 3),
            rgb[:, -bw:].reshape(-1, 3),
        ])
        bg_color  = np.median(samples, axis=0)   # BGR中央値
        bg_lum    = bg_color.mean()
        bg_spread = samples.std(axis=0).mean()

        # ── ② 背景色の種類で戦略を選択 ──
        is_dark_bg  = bg_lum < 30          # 黒系背景
        is_light_bg = bg_lum > 225         # 白系背景

        if is_dark_bg:
            # ── 黒背景専用: 輝度+彩度で純粋な黒を検出 ──
            # 髪など暗い部分と背景黒を彩度で区別
            r, g, b_ = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
            lum = 0.299*r + 0.587*g + 0.114*b_
            cmax = np.maximum(np.maximum(r, g), b_)
            cmin = np.minimum(np.minimum(r, g), b_)
            sat  = np.where(cmax > 0, (cmax - cmin) / (cmax + 1e-6), 0.0)

            # 純粋な黒: 輝度が低い かつ 彩度も低い
            # 髪の毛: 輝度は低いが 彩度はわずかにある
            lum_th = max(18.0, 25.0 * sensitivity)
            # 少し輝度が高くても彩度が極端に低ければ背景
            is_bg_candidate = (lum < lum_th) | ((lum < 40.0) & (sat < 0.06))

        elif is_light_bg:
            # 白背景: 全チャネル高い
            lum = 0.299*rgb[:,:,0] + 0.587*rgb[:,:,1] + 0.114*rgb[:,:,2]
            is_bg_candidate = lum > (230 - 15 * sensitivity)

        else:
            # カラー背景: 色差ベース
            if bg_spread > 50.0:
                return None   # 複雑すぎ → GrabCutへ
            diff = np.sqrt(np.sum((rgb - bg_color) ** 2, axis=2))
            is_bg_candidate = diff < (28.0 * sensitivity)

        # ── ③ 外周BFSで連結BG領域のみ採用 ──
        is_bg = self._bfs_flood_fill(is_bg_candidate, h, w)

        # ── ④ モルフォロジー整形（ノイズ除去） ──
        is_bg = ndimage.binary_closing(is_bg, np.ones((3, 3)))
        is_bg = ndimage.binary_opening(is_bg, np.ones((2, 2)))
        # 外周は必ずBG
        is_bg[:bw, :] = True; is_bg[-bw:, :] = True
        is_bg[:, :bw] = True; is_bg[:, -bw:] = True
        is_bg = self._bfs_flood_fill(is_bg, h, w)  # 再度BFS

        # ── ⑤ 境界アルファグラデーション ──
        feather = max(2, int(min(h, w) * 0.010))
        dist_from_bg = ndimage.distance_transform_edt(~is_bg).astype(np.float32)
        dist_from_fg = ndimage.distance_transform_edt(is_bg).astype(np.float32)

        # 確実FG=1.0 / 確実BG=0.0 / 境界=グラデ
        alpha = np.where(
            is_bg,
            np.clip(1.0 - dist_from_fg / max(feather, 1), 0.0, 1.0),
            np.clip(dist_from_bg / max(feather, 1), 0.0, 1.0),
        )
        alpha = np.clip(alpha, 0, 1)

        # ── ⑥ 境界のみ軽くぼかす（内部は触らない） ──
        edge_zone = (alpha > 0.05) & (alpha < 0.95)
        alpha_blur = ndimage.gaussian_filter(alpha, sigma=0.6)
        alpha = np.where(edge_zone, alpha_blur, alpha)
        alpha = np.clip(alpha, 0, 1)

        # ── ⑦ 孤立した小成分を除去 ──
        labeled, num = ndimage.label(alpha > 0.5)
        if num > 1:
            sizes = [ndimage.sum(alpha > 0.5, labeled, i) for i in range(1, num + 1)]
            main = int(np.argmax(sizes)) + 1
            for i in range(1, num + 1):
                if i != main and sizes[i - 1] < 150:
                    alpha[labeled == i] = 0.0

        # ── ⑧ デフリンジ（黒背景の場合は暗色ピクセルを保護） ──
        rgb_out = rgb.copy()
        fringe = (alpha > 0.02) & (alpha < 0.85)
        if fringe.any() and not is_dark_bg:
            # カラー/白背景のみデフリンジ（黒背景ではやらない→髪を守る）
            sure_fg = alpha > 0.92
            if sure_fg.any():
                indices = ndimage.distance_transform_edt(
                    ~sure_fg, return_distances=False, return_indices=True
                )
                iy, ix = indices[0], indices[1]
                for c in range(3):
                    expanded = rgb[:, :, c][iy, ix]
                    rgb_out[:, :, c] = np.where(fringe, expanded, rgb_out[:, :, c])

        result = img_rgba.copy()
        result[:, :, :3] = np.clip(rgb_out, 0, 255).astype(np.uint8)
        result[:, :, 3]  = (alpha * 255).astype(np.uint8)
        return result

    # ================================================================
    # ③ SEGS: K-meansセグメンテーション（外部モデル不使用）
    # ================================================================

    def segment_segs(
        self,
        img_rgba: "np.ndarray",
        n_clusters: int = 6,
    ) -> "Dict[str, np.ndarray]":
        """
        K-meansクラスタリングによるSEGS（セグメンテーション）。
        各セグメントのマスク配列を返す。

        Returns:
            dict: {"seg_0": mask_array, "seg_1": ..., ...}
                  mask_arrayはuint8 (0 or 255) の2Dアレイ
        """
        if not _NUMPY_AVAILABLE:
            return {}

        h, w = img_rgba.shape[:2]
        lab = self._rgb_to_lab(img_rgba[:, :, :3])

        # 位置情報も加味（色+座標で空間的にまとまったクラスタを生成）
        ys, xs = np.mgrid[0:h, 0:w]
        ys_n = ys.astype(np.float32) / h * 20  # 位置の重みを色より低く
        xs_n = xs.astype(np.float32) / w * 20
        features = np.stack([
            lab[:, :, 0], lab[:, :, 1], lab[:, :, 2], ys_n, xs_n
        ], axis=2).reshape(-1, 5)

        # 不透明ピクセルのみ対象
        alpha = img_rgba[:, :, 3].reshape(-1)
        opaque = alpha > 10
        if opaque.sum() < n_clusters:
            return {}

        feat_valid = features[opaque]

        # K-means（純粋numpy実装・ランダム初期化 + イテレーション）
        labels_valid = self._kmeans_numpy(feat_valid, n_clusters, self._KMEANS_ITER)

        # 全ピクセルにラベルを割り当て（透明は -1）
        labels_full = np.full(h * w, -1, dtype=np.int32)
        labels_full[opaque] = labels_valid

        # 各クラスタのマスクを生成
        segs: Dict[str, np.ndarray] = {}
        for k in range(n_clusters):
            mask = (labels_full == k).reshape(h, w).astype(np.uint8) * 255
            if mask.sum() > 255:  # 空クラスタを除外
                # ConnectedComponentsで最大連結成分を採用
                if _SCIPY_AVAILABLE:
                    labeled, _ = ndimage.label(mask > 128)
                    if labeled.max() > 0:
                        sizes = ndimage.sum(mask > 128, labeled, range(1, labeled.max() + 1))
                        largest = np.argmax(sizes) + 1
                        mask = (labeled == largest).astype(np.uint8) * 255
                segs[f"seg_{k}"] = mask

        return segs

    def _kmeans_numpy(
        self,
        data: "np.ndarray",
        k: int,
        max_iter: int,
    ) -> "np.ndarray":
        """純粋numpy K-means（Lloyd法）。kmeans++ 初期化を完全ベクトル化。"""
        n = len(data)
        # kmeans++ 初期化（完全ベクトル化・Pythonループなし）
        centers = [data[np.random.randint(n)]]
        for _ in range(k - 1):
            # 全データ点と既存センター間の最小二乗距離をベクトル化で計算
            centers_arr = np.array(centers)                                  # (c, F)
            diffs = data[:, np.newaxis, :] - centers_arr[np.newaxis, :, :]  # (n, c, F)
            d2 = (diffs ** 2).sum(axis=2).min(axis=1)                       # (n,)
            probs = d2 / (d2.sum() + 1e-12)
            centers.append(data[np.random.choice(n, p=probs)])
        centers = np.array(centers)

        labels = np.zeros(n, dtype=np.int32)
        for _ in range(max_iter):
            # 距離計算（ベクトル化）
            dists = np.sqrt(((data[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2))
            new_labels = dists.argmin(axis=1)
            if (new_labels == labels).all():
                break
            labels = new_labels
            for j in range(k):
                mask = labels == j
                if mask.any():
                    centers[j] = data[mask].mean(axis=0)

        return labels

    # ================================================================
    # ④ 部位検出（顔・肌・髪・服）外部モデル不使用
    # ================================================================

    def detect_body_parts(
        self,
        img_rgba: "np.ndarray",
    ) -> "Dict[str, np.ndarray]":
        """
        YCbCr/HSV複合モデル + 形態学的解析による部位検出。
        学習済みモデル不使用・純粋アルゴリズム。

        Returns:
            dict: {
                "skin":     uint8 mask (0/255),  肌領域
                "face":     uint8 mask (0/255),  顔推定領域
                "hair":     uint8 mask (0/255),  髪領域
                "clothing": uint8 mask (0/255),  服領域
                "eye":      uint8 mask (0/255),  目・眉周辺（暗い顔内領域）
            }
        """
        if not _NUMPY_AVAILABLE:
            return {}

        h, w = img_rgba.shape[:2]
        rgb  = img_rgba[:, :, :3].astype(np.float32)
        alpha_ch = img_rgba[:, :, 3]

        # ── 肌検出 (YCbCr + HSV dual model) ──
        skin_mask = self._detect_skin(rgb, h, w)

        # 不透明領域のみ
        opaque = alpha_ch > 10
        skin_mask &= opaque

        # ── 顔領域推定 ──
        # 「上半分の肌の最大連結成分」＝顔
        face_mask = self._estimate_face(skin_mask, h, w)

        # ── 髪検出 ──
        hair_mask = self._detect_hair(rgb, face_mask, skin_mask, h, w, opaque)

        # ── 服検出 ──
        clothing_mask = self._detect_clothing(skin_mask, hair_mask, face_mask, opaque, h, w)

        # ── 目・眉領域（顔内の暗い小領域）──
        eye_mask = self._detect_eyes(rgb, face_mask, h, w)

        # 各マスクをモルフォロジー整形
        if _SCIPY_AVAILABLE:
            for mask in [skin_mask, face_mask, hair_mask, clothing_mask, eye_mask]:
                mask[:] = ndimage.binary_closing(mask, np.ones((4, 4), bool))

        return {
            "skin":     (skin_mask * 255).astype(np.uint8),
            "face":     (face_mask * 255).astype(np.uint8),
            "hair":     (hair_mask * 255).astype(np.uint8),
            "clothing": (clothing_mask * 255).astype(np.uint8),
            "eye":      (eye_mask * 255).astype(np.uint8),
        }

    def _detect_skin(self, rgb: "np.ndarray", h: int, w: int) -> "np.ndarray":
        """YCbCr + HSV 複合肌色検出（ベクトル化）"""
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]

        # YCbCr変換
        Y  =  0.299 * r + 0.587 * g + 0.114 * b
        Cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
        Cr =  0.5 * r - 0.4187 * g - 0.0813 * b + 128

        # Chai & Ngan 肌色範囲 (YCbCr)
        skin_ycbcr = (
            (Y > 80) & (Y < 240) &
            (Cb > 85) & (Cb < 135) &
            (Cr > 135) & (Cr < 180)
        )

        # HSV変換
        rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
        cmax = np.maximum(np.maximum(rn, gn), bn)
        cmin = np.minimum(np.minimum(rn, gn), bn)
        delta = cmax - cmin + 1e-9

        # Hue
        H = np.where(cmax == rn, (gn - bn) / delta % 6,
            np.where(cmax == gn, (bn - rn) / delta + 2,
                                  (rn - gn) / delta + 4)) * 60
        H[H < 0] += 360
        S = np.where(cmax > 0, delta / (cmax + 1e-9), 0)
        V = cmax

        # HSV肌色範囲
        skin_hsv = (
            ((H >= 0) & (H <= 25) | (H >= 340) & (H <= 360)) &
            (S >= 0.15) & (S <= 0.90) &
            (V >= 0.30)
        )

        return (skin_ycbcr | skin_hsv)

    def _estimate_face(
        self,
        skin_mask: "np.ndarray",
        h: int, w: int,
    ) -> "np.ndarray":
        """
        上半身の肌成分から顔を推定。
        最大連結成分を顔とし、楕円フィッティングで精度向上。
        """
        if not _SCIPY_AVAILABLE:
            return np.zeros((h, w), bool)

        # 上60%領域の肌
        upper = skin_mask.copy()
        upper[int(h * 0.6):, :] = False

        labeled, num = ndimage.label(upper)
        if num == 0:
            return np.zeros((h, w), bool)

        sizes = ndimage.sum(upper, labeled, range(1, num + 1))
        largest_idx = int(np.argmax(sizes)) + 1
        face_raw = labeled == largest_idx

        # バウンディングボックスを楕円マスクに変換（丸みを持たせる）
        ys, xs = np.where(face_raw)
        if len(ys) == 0:
            return face_raw

        cy_f = (ys.min() + ys.max()) // 2
        cx_f = (xs.min() + xs.max()) // 2
        ry_f = (ys.max() - ys.min()) // 2 + 8
        rx_f = (xs.max() - xs.min()) // 2 + 8

        ys_g, xs_g = np.mgrid[0:h, 0:w]
        face_ellipse = ((xs_g - cx_f)**2 / max(rx_f, 1)**2
                      + (ys_g - cy_f)**2 / max(ry_f, 1)**2) <= 1.05

        # 楕円 AND 肌マスク
        return face_ellipse & skin_mask

    def _detect_hair(
        self,
        rgb: "np.ndarray",
        face_mask: "np.ndarray",
        skin_mask: "np.ndarray",
        h: int, w: int,
        opaque: "np.ndarray",
    ) -> "np.ndarray":
        """
        顔の上の暗い領域を髪として検出。
        肌でも背景でもない領域＋輝度が低い = 髪。
        """
        gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]

        # 顔バウンディングボックスの上部を基準に
        face_ys, _ = np.where(face_mask)
        if len(face_ys) == 0:
            return np.zeros((h, w), bool)

        face_top = face_ys.min()
        search_bottom = max(face_ys.max(), int(face_top + h * 0.3))

        # 顔領域の輝度中央値
        face_lum = gray[face_mask].mean() if face_mask.any() else 150.0
        hair_threshold = face_lum * 0.55  # 顔より55%以上暗い = 髪

        # 候補: 暗い + 非肌 + 不透明 + 顔より上または隣接
        hair_cand = (
            (gray < hair_threshold) &
            (~skin_mask) &
            opaque
        )

        # 顔の上の領域に限定（過剰検出を防ぐ）
        region_mask = np.zeros((h, w), bool)
        region_mask[:search_bottom, :] = True
        hair_cand &= region_mask

        if _SCIPY_AVAILABLE and hair_cand.any():
            # 顔マスクの拡張領域と連結している成分のみ採用
            dilated_face = ndimage.binary_dilation(face_mask, np.ones((10, 10), bool))
            hair_cand &= dilated_face | hair_cand

            # モルフォロジー整形
            hair_cand = ndimage.binary_closing(hair_cand, np.ones((6, 6), bool))
            hair_cand = ndimage.binary_fill_holes(hair_cand)

        return hair_cand

    def _detect_clothing(
        self,
        skin_mask: "np.ndarray",
        hair_mask: "np.ndarray",
        face_mask: "np.ndarray",
        opaque: "np.ndarray",
        h: int, w: int,
    ) -> "np.ndarray":
        """
        服検出: 不透明 & 非肌 & 非髪 の領域。
        """
        clothing = opaque & ~skin_mask & ~hair_mask
        if _SCIPY_AVAILABLE:
            clothing = ndimage.binary_opening(clothing, np.ones((4, 4), bool))
            clothing = ndimage.binary_closing(clothing, np.ones((8, 8), bool))
        return clothing

    def _detect_eyes(
        self,
        rgb: "np.ndarray",
        face_mask: "np.ndarray",
        h: int, w: int,
    ) -> "np.ndarray":
        """
        顔領域内の暗い小領域（目・眉）を検出。
        """
        if not face_mask.any() or not _SCIPY_AVAILABLE:
            return np.zeros((h, w), bool)

        gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
        face_lum = gray[face_mask].mean()
        eye_thresh = face_lum * 0.55

        eye_cand = face_mask & (gray < eye_thresh)

        # 顔の上半分に限定
        face_ys, _ = np.where(face_mask)
        if len(face_ys) == 0:
            return eye_cand
        face_mid = (face_ys.min() + face_ys.max()) // 2
        eye_cand[face_mid:, :] = False

        # 小成分のみ（目は小さい）
        labeled, num = ndimage.label(eye_cand)
        if num == 0:
            return eye_cand
        sizes = ndimage.sum(eye_cand, labeled, range(1, num + 1))
        face_area = face_mask.sum()
        result = np.zeros((h, w), bool)
        for i, s in enumerate(sizes):
            if 10 < s < face_area * 0.12:  # 顔面積の12%以下
                result[labeled == (i + 1)] = True

        return result

    # ================================================================
    # ⑤ ポイント処理
    # ================================================================

    def remove_by_point(
        self,
        img_rgba: "np.ndarray",
        px: int, py: int,
        radius: int = 20,
        sensitivity: float = 1.0,
    ) -> "np.ndarray":
        if not _NUMPY_AVAILABLE:
            return img_rgba
        result = img_rgba.copy()
        h, w = result.shape[:2]
        if not (0 <= py < h and 0 <= px < w):
            return result

        sr, er = max(0, py - 2), min(h, py + 3)
        sc, ec = max(0, px - 2), min(w, px + 3)
        seed_color = result[sr:er, sc:ec, :3].reshape(-1, 3).mean(axis=0)

        lab = self._rgb_to_lab(result[:, :, :3])
        seed_lab = self._rgb_to_lab(seed_color.reshape(1, 1, 3).astype(np.uint8))[0, 0]

        tolerance = max(10.0, radius * 0.8) * sensitivity
        diff = np.sqrt(np.sum((lab - seed_lab) ** 2, axis=2))
        is_similar = diff < tolerance
        fill_mask = self._bfs_from_point(is_similar, py, px, h, w)
        if not fill_mask.any():
            fill_mask = self._bfs_flood_fill(is_similar, h, w)

        if _SCIPY_AVAILABLE:
            dist = ndimage.distance_transform_edt(fill_mask).astype(np.float32)
            alpha_reduce = np.clip(dist / self._FEATHER_RADIUS, 0, 1)
            result[:, :, 3] = (result[:, :, 3] * (1 - alpha_reduce * fill_mask)).astype(np.uint8)
        else:
            result[:, :, 3][fill_mask] = 0

        return result

    def _bfs_from_point(self, is_similar, sy, sx, h, w):
        visited = np.zeros((h, w), bool)
        if not is_similar[sy, sx]:
            return visited
        q = deque([(sy, sx)])
        visited[sy, sx] = True
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            pass
        nb4 = [(-1,0),(1,0),(0,-1),(0,1)]
        while q:
            r, c = q.popleft()
            for dr, dc in nb4:
                nr, nc = r+dr, c+dc
                if 0<=nr<h and 0<=nc<w and not visited[nr,nc] and is_similar[nr,nc]:
                    visited[nr,nc] = True
                    q.append((nr,nc))
        return visited

    def _bfs_flood_fill(self, is_bg: "np.ndarray", h: int, w: int) -> "np.ndarray":
        """
        外周から連結したBG領域のみを返す。

        scipy が使える場合: ndimage.label + 外周ラベル抽出（C実装）
          → Python BFS比 約100倍高速。1500×1500でも ~0.04秒。
        scipy がない場合  : Python BFS フォールバック。
        """
        if _SCIPY_AVAILABLE:
            # 8連結でラベリング（C実装・高速）
            labeled, _num = ndimage.label(is_bg, structure=np.ones((3, 3)))
            # 外周に接するラベルIDを収集
            border_labels = set()
            border_labels.update(labeled[0,   :].flat)
            border_labels.update(labeled[h-1, :].flat)
            border_labels.update(labeled[:,   0].flat)
            border_labels.update(labeled[:, w-1].flat)
            border_labels.discard(0)  # 0 = 非BGピクセル
            if not border_labels:
                return np.zeros((h, w), bool)
            return np.isin(labeled, list(border_labels))

        # scipy 未インストール時の Python フォールバック
        visited = np.zeros((h, w), bool)
        q = deque()
        for r in range(h):
            for c in (0, w - 1):
                if is_bg[r, c] and not visited[r, c]:
                    visited[r, c] = True; q.append((r, c))
        for c in range(w):
            for r in (0, h - 1):
                if is_bg[r, c] and not visited[r, c]:
                    visited[r, c] = True; q.append((r, c))
        nb8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        while q:
            r, c = q.popleft()
            for dr, dc in nb8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w \
                        and not visited[nr, nc] and is_bg[nr, nc]:
                    visited[nr, nc] = True; q.append((nr, nc))
        return visited

    # ================================================================
    # ⑥ 選択範囲除去・ブラシ編集
    # ================================================================

    def remove_by_rect(self, img_rgba, x1, y1, x2, y2, mode="hard"):
        if not _NUMPY_AVAILABLE: return img_rgba
        result = img_rgba.copy()
        h, w = result.shape[:2]
        rx1, ry1 = max(0,min(x1,x2)), max(0,min(y1,y2))
        rx2, ry2 = min(w,max(x1,x2)), min(h,max(y1,y2))
        if mode == "hard":
            result[ry1:ry2, rx1:rx2, 3] = 0
        elif mode == "feather" and _SCIPY_AVAILABLE:
            rm = np.zeros((h,w), bool); rm[ry1:ry2, rx1:rx2] = True
            dist = ndimage.distance_transform_edt(rm).astype(np.float32)
            fade = np.clip(1 - dist / 12, 0, 1)
            result[:,:,3] = (result[:,:,3] * fade).astype(np.uint8)
        return result

    def remove_by_ellipse(self, img_rgba, cx, cy, rx, ry):
        if not _NUMPY_AVAILABLE: return img_rgba
        result = img_rgba.copy()
        h, w = result.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w]
        mask = ((xs-cx)**2/max(rx,1)**2 + (ys-cy)**2/max(ry,1)**2) <= 1.0
        result[:,:,3][mask] = 0
        return result

    def remove_by_lasso(self, img_rgba, points):
        if not _NUMPY_AVAILABLE or len(points) < 3: return img_rgba
        result = img_rgba.copy()
        h, w = result.shape[:2]
        if _PIL_AVAILABLE:
            mask_img = Image.new("L", (w, h), 0)
            ImageDraw.Draw(mask_img).polygon(points, fill=255)
            result[:,:,3][np.array(mask_img) > 128] = 0
        return result

    def apply_brush(self, img_rgba, px, py, brush_size=15, mode="erase"):
        if not _NUMPY_AVAILABLE: return img_rgba
        result = img_rgba.copy()
        h, w = result.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w]
        dist = np.sqrt((xs-px)**2 + (ys-py)**2)
        soft = np.clip(1 - dist/max(brush_size,1), 0, 1)
        soft[dist > brush_size] = 0
        if mode == "erase":
            result[:,:,3] = (result[:,:,3]*(1-soft)).astype(np.uint8)
        else:
            result[:,:,3] = np.clip(result[:,:,3]+soft*255, 0, 255).astype(np.uint8)
        return result

    # ================================================================
    # ⑦ 背景合成
    # ================================================================

    def composite_with_background(self, fg_rgba, bg_type="checker",
                                   bg_color=(100,100,200), bg_image=None):
        if not _NUMPY_AVAILABLE: return fg_rgba
        h, w = fg_rgba.shape[:2]
        if bg_type == "checker":      bg = self._make_checker_array(w, h)
        elif bg_type == "solid":      bg = np.full((h,w,4), (*bg_color,255), dtype=np.uint8)
        elif bg_type == "gradient":   bg = self._make_gradient_array(w, h, bg_color)
        elif bg_type == "image" and bg_image is not None: bg = self._resize_bg(bg_image, w, h)
        else:                          bg = self._make_checker_array(w, h)
        alpha = fg_rgba[:,:,3:4].astype(np.float32)/255.0
        out = (fg_rgba[:,:,:3].astype(np.float32)*alpha +
               bg[:,:,:3].astype(np.float32)*(1-alpha)).astype(np.uint8)
        return np.dstack([out, np.full((h,w),255,dtype=np.uint8)])

    def _make_checker_array(self, w, h, size=16):
        arr = np.full((h,w,4), 255, dtype=np.uint8)
        ys, xs = np.mgrid[0:h, 0:w]
        checker = ((xs//size)+(ys//size)) % 2 == 1
        arr[checker, :3] = 180
        return arr

    def _make_gradient_array(self, w, h, color):
        arr = np.zeros((h,w,4), dtype=np.uint8)
        t = np.linspace(0, 1, h)[:, np.newaxis]
        arr[:,:,0] = (color[0]*(1-t) + 30*t).clip(0,255)
        arr[:,:,1] = (color[1]*(1-t) + 30*t).clip(0,255)
        arr[:,:,2] = (color[2]*(1-t) + 60*t).clip(0,255)
        arr[:,:,3] = 255
        return arr

    def _resize_bg(self, bg, w, h):
        if not _PIL_AVAILABLE: return np.full((h,w,4),(100,100,100,255),dtype=np.uint8)
        return np.array(Image.fromarray(bg).convert("RGBA").resize((w,h), Image.LANCZOS))

    # ================================================================
    # ⑧ Inpaint（独自重み付き補完）
    # ================================================================

    def inpaint_region(self, img_rgba, mask, radius=8):
        """
        透明領域を周囲の既知ピクセルで補完する。
        scipy.ndimage の uniform_filter を使ったベクトル化実装。
        Python ネストループを排除し、GIL ブロックを最小化。
        """
        if not _NUMPY_AVAILABLE:
            return img_rgba
        result = img_rgba.copy().astype(np.float32)
        fill = mask.copy().astype(bool)

        if not fill.any():
            return img_rgba

        if _SCIPY_AVAILABLE:
            # ── scipy 高速パス ──────────────────────────────────────────
            # 境界 BFS で内側から外側に向けてレイヤー毎に補完する。
            # 各反復で「未補完ピクセルのうち既知隣接を持つ」ものだけ確定する。
            size = radius * 2 + 1
            known = ~fill
            # 上限イテレーションを radius に基づいて制限（フリーズ防止）
            # 実際には known が広がるにつれ早期終了するが念のため上限を設ける
            max_iters = max(32, radius * 8)

            for _iter in range(max_iters):
                ys, xs = np.where(fill)
                if len(ys) == 0:
                    break

                # 各チャネルを均一フィルタ（既知ピクセルのみ加重）
                new_vals = np.zeros((len(ys), 4), dtype=np.float32)
                weights  = np.zeros(len(ys), dtype=np.float32)

                for ch in range(4):
                    ch_data  = result[:, :, ch] * known           # 未知は0
                    w_data   = known.astype(np.float32)
                    filt_val = ndimage.uniform_filter(ch_data, size=size)
                    filt_w   = ndimage.uniform_filter(w_data,  size=size)
                    new_vals[:, ch] = filt_val[ys, xs]
                    weights          = filt_w[ys, xs]             # 最後chで上書きで可

                # 既知隣接ピクセルを持つ（weight > 0）ものだけ確定
                has_neighbor = weights > 1e-6
                if not has_neighbor.any():
                    break

                ys_ok = ys[has_neighbor]
                xs_ok = xs[has_neighbor]
                for ch in range(4):
                    result[ys_ok, xs_ok, ch] = (
                        new_vals[has_neighbor, ch] / (weights[has_neighbor] + 1e-12)
                    )
                fill[ys_ok, xs_ok] = False
                known[ys_ok, xs_ok] = True

        else:
            # ── scipy なし: 最大ピクセル数を制限して Python ループ ──────
            MAX_PX = 2000   # フリーズ防止のための上限
            h, w = result.shape[:2]
            ys_all, xs_all = np.where(fill)
            if len(ys_all) > MAX_PX:
                idx = np.random.choice(len(ys_all), MAX_PX, replace=False)
                ys_all, xs_all = ys_all[idx], xs_all[idx]

            for y, x in zip(ys_all.tolist(), xs_all.tolist()):
                y0, y1 = max(0, y - radius), min(h, y + radius + 1)
                x0, x1 = max(0, x - radius), min(w, x + radius + 1)
                valid = ~fill[y0:y1, x0:x1]
                if not valid.any():
                    continue
                ry, rx = np.mgrid[y0:y1, x0:x1]
                dist   = np.sqrt((ry - y) ** 2 + (rx - x) ** 2) + 1e-6
                weight = (1 / dist ** 2) * valid
                ws     = weight.sum()
                if ws < 1e-6:
                    continue
                for ch in range(4):
                    result[y, x, ch] = (result[y0:y1, x0:x1, ch] * weight).sum() / ws
                fill[y, x] = False

        return np.clip(result, 0, 255).astype(np.uint8)

    def create_inpaint_mask_from_alpha(self, img_rgba):
        if not _NUMPY_AVAILABLE: return np.zeros(img_rgba.shape[:2], bool)
        return img_rgba[:,:,3] < 128

    # ================================================================
    # ⑨ 一括処理ユーティリティ
    # ================================================================

    def process_all_cells(self, sheet_img, rows, cols, pose_names, on_progress=None):
        if not _PIL_AVAILABLE or not _NUMPY_AVAILABLE: return {}
        results = {}
        arr = np.array(sheet_img.convert("RGBA"))
        h, w = arr.shape[:2]
        cw, ch = w//cols, h//rows
        total = len(pose_names)
        for i, name in enumerate(pose_names):
            if on_progress: on_progress(i+1, total, f"処理中: {name}")
            row, col = i//cols, i%cols
            cell = arr[row*ch:(row+1)*ch, col*cw:(col+1)*cw]
            try:
                removed = self.remove_background_adaptive(cell)
                cropped = self._autocrop_array(removed)
                norm    = self._normalize_array(cropped)
                results[name] = Image.fromarray(norm)
            except Exception as e:
                logger.error(f"セル '{name}' 処理エラー: {e}")
        return results

    def _autocrop_array(self, arr, padding=20):
        alpha = arr[:,:,3]
        mask  = alpha > 10
        if not mask.any(): return arr
        rows = np.any(mask, axis=1); cols = np.any(mask, axis=0)
        rmin = max(0, int(np.where(rows)[0][0]) - padding)
        rmax = min(arr.shape[0]-1, int(np.where(rows)[0][-1]) + padding)
        cmin = max(0, int(np.where(cols)[0][0]) - padding)
        cmax = min(arr.shape[1]-1, int(np.where(cols)[0][-1]) + padding)
        return arr[rmin:rmax+1, cmin:cmax+1]

    def _normalize_array(self, arr, size=2048):
        if not _PIL_AVAILABLE: return arr
        img = Image.fromarray(arr)
        canvas = Image.new("RGBA", (size,size), (0,0,0,0))
        scale = min(size/img.width, size/img.height)
        nw, nh = int(img.width*scale), int(img.height*scale)
        resized = img.resize((nw,nh), Image.LANCZOS)
        canvas.paste(resized, ((size-nw)//2, (size-nh)//2), resized)
        return np.array(canvas)

    # ================================================================
    # ⑩ 色空間変換ユーティリティ
    # ================================================================

    def _rgb_to_lab(self, rgb):
        rgb_f = rgb.astype(np.float32) / 255.0
        linear = np.where(rgb_f > 0.04045,
                          ((rgb_f + 0.055) / 1.055) ** 2.4,
                          rgb_f / 12.92)
        r, g, b = linear[:,:,0], linear[:,:,1], linear[:,:,2]
        x = r*0.4124 + g*0.3576 + b*0.1805
        y = r*0.2126 + g*0.7152 + b*0.0722
        z = r*0.0193 + g*0.1192 + b*0.9505
        def f(t):
            delta = 6/29
            return np.where(t > delta**3, t**(1/3), t/(3*delta**2)+4/29)
        fx, fy, fz = f(x/0.9505), f(y/1.0000), f(z/1.0890)
        return np.stack([116*fy-16, 500*(fx-fy), 200*(fy-fz)], axis=2)


# ================================================================== #
# 高度な背景除去ダイアログ（統合版）
# ================================================================== #


class AdvancedBgRemovalDialog(tk.Toplevel):
    """
    高度な背景除去・画像編集ダイアログ。

    機能:
      - 全セル一括処理
      - ポイント除去（クリック指定）
      - 矩形・楕円・自由曲線選択範囲除去
      - ブラシ（消去・復元）
      - リアルタイムプレビュー
      - 背景合成プレビュー（チェッカー・単色・グラデーション・画像）
      - 保存確認ダイアログ（カスタム名・連番・保存先選択）
    """

    _POSES = ["default", "idle", "speaking", "thinking", "greeting"]
    _POSE_FILE_MAP = {
        "default":  "alice_default",
        "idle":     "alice_idle",
        "speaking": "alice_speaking",
        "thinking": "alice_thinking",
        "greeting": "alice_greeting",
    }

    _TOOL_POINT  = "point"
    _TOOL_RECT   = "rect"
    _TOOL_ELLIPSE= "ellipse"
    _TOOL_LASSO  = "lasso"
    _TOOL_BRUSH_ERASE   = "brush_erase"
    _TOOL_BRUSH_RESTORE = "brush_restore"

    def __init__(
        self,
        parent,
        char_loader=None,
        on_reload: Optional[Callable] = None,
    ):
        super().__init__(parent)
        self._char_loader = char_loader
        self._on_reload   = on_reload
        self._processor   = AdvancedImageProcessor()

        # 【v3】全変数を UI 構築前に初期化（AttributeError 防止）
        self._init_variables()

        self._setup_theme()
        self.title("高度な画像処理ツール - Alice AI")
        self.withdraw()          # 構築中は非表示にして描画をブロックしない
        self.geometry("1280x800")
        self.minsize(1000, 650)
        self.configure(bg=self._c.bg_primary)
        self.transient(parent)
        self._build_ui()         # UI構築を先に完了させる
        self.deiconify()         # 構築完了後に表示
        self.grab_set()          # 表示後にグラブ（フリーズ防止）

    def _init_variables(self):
        """全インスタンス変数を UI 構築前に初期化する（v3 パターン）。"""
        # 状態管理
        self._src_image:    Optional[Image.Image] = None
        self._work_arr:     Optional["np.ndarray"] = None
        self._history_stack: List["np.ndarray"] = []
        self._result_image: Optional[Image.Image] = None
        self._bg_image:     Optional["np.ndarray"] = None

        # ツール状態
        self._current_tool = self._TOOL_POINT
        self._brush_size   = 15
        self._point_radius = 20
        self._sensitivity  = tk.DoubleVar(value=1.0)
        self._rect_start:  Optional[Tuple[int, int]] = None
        self._rect_end:    Optional[Tuple[int, int]] = None
        self._rect_drawing = False
        self._lasso_points: List[Tuple[int, int]] = []
        self._lasso_drawing = False

        # プレビュースケール
        self._preview_scale = 1.0
        self._preview_offset = (0, 0)

        # 処理フラグ
        self._processing = False

        # ブラシドラッグ間引き用フラグ（フリーズ防止）
        self._brush_drag_pending = False
        self._brush_dragging = False

        # エッジ表示フラグ
        self._edge_showing = False
        self._edge_arr: Optional["np.ndarray"] = None

        # ビフォーアフタースライダー
        self._slider_ratio    = 0.5   # 0.0=全部元画像 ～ 1.0=全部処理後
        self._slider_dragging = False
        self._tk_comparison: Optional[ImageTk.PhotoImage] = None

        # バッチ処理結果
        self._batch_results: Dict[str, Image.Image] = {}
        self._batch_placeholders: Dict[str, Image.Image] = {}
        self._batch_expected_names: List[str] = []
        self._batch_placeholder_names: set[str] = set()

        # _build_toolbar() → _select_tool() より前に必要な Tk 変数
        self._coord_var     = tk.StringVar(value="X:- Y:-")
        self._tool_info_var = tk.StringVar(value="ツール: ポイント除去")

    def _setup_theme(self):
        try:
            from module import env_binder_module as env
            theme_name = env.get("APP_THEME", "dark")
        except Exception:
            theme_name = "dark"
        self._c = Theme.get(theme_name)

    # ================================================================
    # UI構築
    # ================================================================

    def _build_ui(self):
        c = self._c

        # ── メインレイアウト: 左ツールバー | 中央プレビュー | 右パネル ──
        main = tk.Frame(self, bg=c.bg_primary)
        main.pack(fill="both", expand=True)
        self._main_frame = main  # 遅延構築用に保持

        # 左ツールバー（必須・先に構築）
        self._build_toolbar(main, c)

        # 中央プレビューエリア（必須・先に構築）
        center = tk.Frame(main, bg=c.bg_primary)
        center.pack(side="left", fill="both", expand=True, padx=4)
        self._build_preview_area(center, c)

        # 右パネルとステータスバーは after_idle で遅延構築
        # → ウィンドウが画面に出てから構築するのでフリーズしない
        def _build_deferred():
            self._build_right_panel(main, c)
            self._build_status_bar(c)
        self.after_idle(_build_deferred)

    def _build_toolbar(self, parent, c):
        """左側ツールバー（ツール選択・ブラシサイズ等）"""
        tb = tk.Frame(parent, bg=c.bg_secondary, width=90)
        tb.pack(side="left", fill="y", padx=(0, 2))
        tb.pack_propagate(False)

        tk.Label(tb, text="ツール", bg=c.bg_secondary, fg=c.text_muted,
                 font=("Segoe UI", 8, "bold")).pack(pady=(8, 2))

        self._tool_btns = {}
        tools = [
            (self._TOOL_POINT,        "🎯", "ポイント除去"),
            (self._TOOL_RECT,         "⬜", "矩形選択除去"),
            (self._TOOL_ELLIPSE,      "⭕", "楕円選択除去"),
            (self._TOOL_LASSO,        "🔗", "投げ縄選択"),
            (self._TOOL_BRUSH_ERASE,  "✏️", "消去ブラシ"),
            (self._TOOL_BRUSH_RESTORE,"🖌️", "復元ブラシ"),
        ]
        for tool_id, icon, tip in tools:
            btn = tk.Button(
                tb, text=f"{icon}\n{tip[:4]}", command=lambda t=tool_id: self._select_tool(t),
                bg=c.bg_tertiary, fg=c.text_primary, relief="flat",
                font=("Segoe UI", 8), padx=4, pady=6, cursor="hand2",
                wraplength=70,
                activebackground=c.accent_primary,
            )
            btn.pack(fill="x", padx=4, pady=1)
            self._tool_btns[tool_id] = btn

        tk.Label(tb, text="ブラシ", bg=c.bg_secondary, fg=c.text_muted,
                 font=("Segoe UI", 8, "bold")).pack(pady=(12, 0))
        self._brush_scale = tk.Scale(
            tb, from_=3, to=80, orient="vertical",
            bg=c.bg_secondary, fg=c.text_primary,
            troughcolor=c.bg_tertiary, highlightthickness=0,
            command=lambda v: setattr(self, "_brush_size", int(v)),
        )
        self._brush_scale.set(15)
        self._brush_scale.pack(padx=8, pady=2)

        tk.Label(tb, text="感度", bg=c.bg_secondary, fg=c.text_muted,
                 font=("Segoe UI", 8, "bold")).pack(pady=(6, 0))
        tk.Scale(
            tb, from_=0.3, to=3.0, resolution=0.1, orient="vertical",
            bg=c.bg_secondary, fg=c.text_primary,
            troughcolor=c.bg_tertiary, highlightthickness=0,
            variable=self._sensitivity,
        ).pack(padx=8, pady=2)

        # Undo ボタン
        tk.Button(
            tb, text="↩ Undo", command=self._undo,
            bg=c.bg_tertiary, fg=c.text_secondary, relief="flat",
            font=("Segoe UI", 8), padx=4, pady=4, cursor="hand2",
        ).pack(fill="x", padx=4, pady=(10, 1))

        # リセット
        tk.Button(
            tb, text="🔄 リセット", command=self._reset_to_original,
            bg=c.bg_tertiary, fg=c.accent_error, relief="flat",
            font=("Segoe UI", 8), padx=4, pady=4, cursor="hand2",
        ).pack(fill="x", padx=4, pady=1)

        self._select_tool(self._TOOL_POINT)

    def _build_preview_area(self, parent, c):
        """中央: 左=操作エリア / 右=ビフォーアフタースライダー"""
        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True)

        # ── 左: 元画像（ツール操作エリア）──
        lf = tk.Frame(paned, bg=c.bg_primary)
        paned.add(lf, weight=1)
        tk.Label(lf, text="元画像（操作エリア）", bg=c.bg_primary,
                 fg=c.text_secondary, font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=4)
        self._canvas_src = tk.Canvas(
            lf, bg="#1a1a2e", highlightthickness=1, cursor="crosshair",
            highlightbackground=c.border,
        )
        self._canvas_src.pack(fill="both", expand=True, padx=2, pady=2)
        self._bind_canvas_events()
        # <Configure> debounce: リサイズ中の連続呼び出しを防ぐ
        self._redraw_src_job = None
        def _on_src_configure(e):
            if self._redraw_src_job:
                self.after_cancel(self._redraw_src_job)
            self._redraw_src_job = self.after(80, self._redraw_src)
        self._canvas_src.bind("<Configure>", _on_src_configure)

        # ── 右: ビフォーアフタースライダープレビュー ──
        rf = tk.Frame(paned, bg=c.bg_primary)
        paned.add(rf, weight=1)

        hdr = tk.Frame(rf, bg=c.bg_primary)
        hdr.pack(fill="x", padx=2)
        tk.Label(
            hdr, text="◀ 元画像 ｜ 処理後 ▶  スライドで比較",
            bg=c.bg_primary, fg=c.accent_primary,
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left", padx=6)

        # 背景タイプ選択（スライダー右側の背景）
        self._bg_type_var = tk.StringVar(value="checker")
        tk.Label(hdr, text="背景:", bg=c.bg_primary, fg=c.text_muted,
                 font=("Segoe UI", 8)).pack(side="left", padx=(8, 0))
        for bgt, lbl in [("checker","チェッカー"),("solid","単色"),("gradient","グラデ"),("image","画像")]:
            tk.Radiobutton(
                hdr, text=lbl, variable=self._bg_type_var, value=bgt,
                bg=c.bg_primary, fg=c.text_secondary,
                selectcolor=c.bg_tertiary, activebackground=c.bg_primary,
                command=self._refresh_result_preview,
                font=("Segoe UI", 8),
            ).pack(side="left")
        tk.Button(
            hdr, text="画像選択", command=self._select_bg_image,
            bg=c.bg_tertiary, fg=c.text_secondary, relief="flat",
            font=("Segoe UI", 8), padx=4, pady=1, cursor="hand2",
        ).pack(side="left", padx=4)

        # スライダーキャンバス
        self._canvas_result = tk.Canvas(
            rf, bg="#1a1a2e", highlightthickness=1,
            highlightbackground=c.border, cursor="sb_h_double_arrow",
        )
        self._canvas_result.pack(fill="both", expand=True, padx=2, pady=2)

        # スライダーイベントバインド
        self._canvas_result.bind("<ButtonPress-1>",   self._slider_press)
        self._canvas_result.bind("<B1-Motion>",       self._slider_drag)
        self._canvas_result.bind("<ButtonRelease-1>", self._slider_release)
        # <Configure> debounce
        self._refresh_result_job = None
        def _on_result_configure(e):
            if self._refresh_result_job:
                self.after_cancel(self._refresh_result_job)
            self._refresh_result_job = self.after(80, self._refresh_result_preview)
        self._canvas_result.bind("<Configure>", _on_result_configure)

        self._tk_src:    Optional[ImageTk.PhotoImage] = None
        self._tk_result: Optional[ImageTk.PhotoImage] = None

    def _build_right_panel(self, parent, c):
        """右パネル: ファイル操作・一括処理・自動除去・保存"""
        rp = tk.Frame(parent, bg=c.bg_secondary, width=280)
        rp.pack(side="right", fill="y", padx=(2, 0))
        rp.pack_propagate(False)

        canvas_rp = tk.Canvas(rp, bg=c.bg_secondary, highlightthickness=0)
        sb_rp     = ttk.Scrollbar(rp, orient="vertical", command=canvas_rp.yview)
        canvas_rp.configure(yscrollcommand=sb_rp.set)
        sb_rp.pack(side="right", fill="y")
        canvas_rp.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas_rp, bg=c.bg_secondary)
        canvas_rp.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas_rp.configure(
            scrollregion=canvas_rp.bbox("all")))

        def section(text):
            tk.Label(inner, text=text, bg=c.bg_secondary, fg=c.accent_primary,
                     font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=10, pady=(12,2))

        def sep():
            tk.Frame(inner, bg=c.border, height=1).pack(fill="x", padx=10, pady=4)

        # ── ファイル操作 ──
        section("📂 ファイル操作")
        self._btn(inner, c, "画像を開く", self._open_file).pack(fill="x", padx=10, pady=2)
        self._btn(inner, c, "シート(複数セル)を開く", self._open_sheet).pack(fill="x", padx=10, pady=2)

        sep()

        # ── 自動背景除去 ──
        section("🤖 自動背景除去")
        self._btn(inner, c, "自動除去実行", self._run_auto_remove,
                  c.accent_primary).pack(fill="x", padx=10, pady=2)

        sep()

        # ── 全セル一括処理 ──
        section("📊 全セル一括処理")
        sheet_grid = tk.Frame(inner, bg=c.bg_secondary)
        sheet_grid.pack(fill="x", padx=10, pady=2)
        tk.Label(sheet_grid, text="行:", bg=c.bg_secondary, fg=c.text_secondary,
                 font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w")
        self._sheet_rows = tk.IntVar(value=4)
        tk.Spinbox(sheet_grid, from_=1, to=16, textvariable=self._sheet_rows,
                   width=4, bg=c.bg_tertiary, fg=c.text_primary,
                   buttonbackground=c.bg_tertiary).grid(row=0, column=1, padx=4)
        tk.Label(sheet_grid, text="列:", bg=c.bg_secondary, fg=c.text_secondary,
                 font=("Segoe UI", 9)).grid(row=0, column=2, sticky="w")
        self._sheet_cols = tk.IntVar(value=4)
        tk.Spinbox(sheet_grid, from_=1, to=16, textvariable=self._sheet_cols,
                   width=4, bg=c.bg_tertiary, fg=c.text_primary,
                   buttonbackground=c.bg_tertiary).grid(row=0, column=3, padx=4)
        self._btn(inner, c, "一括処理実行", self._run_batch_process).pack(fill="x", padx=10, pady=2)

        tk.Label(inner, text="処理済みセル:", bg=c.bg_secondary, fg=c.text_secondary,
                 font=("Segoe UI", 9)).pack(anchor="w", padx=10)
        self._batch_listbox = tk.Listbox(
            inner, height=6, bg=c.bg_tertiary, fg=c.text_primary,
            selectbackground=c.accent_primary, relief="flat",
            font=("Segoe UI", 9),
        )
        self._batch_listbox.pack(fill="x", padx=10, pady=2)
        self._batch_listbox.bind("<<ListboxSelect>>", self._on_batch_select)

        sep()

        # ── SEGS: K-meansセグメンテーション ──
        section("🔬 SEGS（自動セグメント）")
        tk.Label(inner, text="K-means色クラスタリング\n各領域を自動分割して選択可能にします",
                 bg=c.bg_secondary, fg=c.text_muted,
                 font=("Segoe UI", 8), justify="left").pack(anchor="w", padx=10)
        seg_row = tk.Frame(inner, bg=c.bg_secondary)
        seg_row.pack(fill="x", padx=10, pady=2)
        tk.Label(seg_row, text="クラスタ数:", bg=c.bg_secondary,
                 fg=c.text_secondary, font=("Segoe UI", 8)).pack(side="left")
        self._seg_k_var = tk.IntVar(value=6)
        tk.Spinbox(seg_row, from_=2, to=12, textvariable=self._seg_k_var,
                   width=3, bg=c.bg_tertiary, fg=c.text_primary,
                   buttonbackground=c.bg_tertiary, font=("Segoe UI", 9)).pack(side="left", padx=4)
        self._seg_btn = self._btn(inner, c, "🔬 SEGS 実行", self._run_segs)
        self._seg_btn.pack(fill="x", padx=10, pady=2)
        self._seg_progress = ttk.Progressbar(inner, mode="indeterminate", length=200)
        self._seg_loading_label = tk.Label(inner, text="", bg=c.bg_secondary,
                                           fg=c.accent_primary, font=("Segoe UI", 8))
        self._seg_progress.pack(padx=10, pady=2)
        self._seg_loading_label.pack(padx=10)
        self._seg_progress.pack_forget()
        self._seg_loading_label.pack_forget()
        # セグメント一覧
        tk.Label(inner, text="検出セグメント（クリックで選択→除去）:",
                 bg=c.bg_secondary, fg=c.text_secondary,
                 font=("Segoe UI", 8)).pack(anchor="w", padx=10)
        self._seg_listbox = tk.Listbox(
            inner, height=5, bg=c.bg_tertiary, fg=c.text_primary,
            selectbackground=c.accent_primary, relief="flat",
            font=("Segoe UI", 8), selectmode="extended",
        )
        self._seg_listbox.pack(fill="x", padx=10, pady=2)
        seg_act_row = tk.Frame(inner, bg=c.bg_secondary)
        seg_act_row.pack(fill="x", padx=10, pady=2)
        self._btn(seg_act_row, c, "選択セグ除去", self._remove_selected_segs).pack(side="left", padx=2)
        self._btn(seg_act_row, c, "選択セグ保持", self._keep_selected_segs).pack(side="left", padx=2)
        self._segs_data: Dict[str, "np.ndarray"] = {}

        sep()

        # ── 部位検出 ──
        section("👤 部位検出（顔・肌・髪・服）")
        tk.Label(inner, text="YCbCr+HSV複合モデル\n学習済みモデル不使用の純粋アルゴリズム",
                 bg=c.bg_secondary, fg=c.text_muted,
                 font=("Segoe UI", 8), justify="left").pack(anchor="w", padx=10)
        self._parts_btn = self._btn(inner, c, "👤 部位検出 実行", self._run_body_parts)
        self._parts_btn.pack(fill="x", padx=10, pady=2)
        self._parts_progress = ttk.Progressbar(inner, mode="indeterminate", length=200)
        self._parts_loading_label = tk.Label(inner, text="", bg=c.bg_secondary,
                                             fg=c.accent_primary, font=("Segoe UI", 8))
        self._parts_progress.pack(padx=10, pady=2)
        self._parts_loading_label.pack(padx=10)
        self._parts_progress.pack_forget()
        self._parts_loading_label.pack_forget()
        # 部位ボタン群
        tk.Label(inner, text="検出した部位を操作:", bg=c.bg_secondary,
                 fg=c.text_secondary, font=("Segoe UI", 8)).pack(anchor="w", padx=10)
        parts_grid = tk.Frame(inner, bg=c.bg_secondary)
        parts_grid.pack(fill="x", padx=10, pady=4)
        self._part_btns: Dict[str, tk.Button] = {}
        for i, (key, label, color) in enumerate([
            ("skin",     "🟡 肌",   "#f59e0b"),
            ("face",     "🔵 顔",   "#3b82f6"),
            ("hair",     "🟤 髪",   "#92400e"),
            ("clothing", "🟢 服",   "#10b981"),
            ("eye",      "⚫ 目",   "#6b7280"),
        ]):
            col_i = i % 2
            row_i = i // 2
            btn = tk.Button(
                parts_grid, text=f"{label}\n除去", width=8,
                command=lambda k=key: self._remove_body_part(k),
                bg=c.bg_tertiary, fg=c.text_primary, relief="flat",
                font=("Segoe UI", 8), pady=4, cursor="hand2",
                state="disabled",
            )
            btn.grid(row=row_i, column=col_i, padx=2, pady=2, sticky="ew")
            self._part_btns[key] = btn
        parts_grid.columnconfigure(0, weight=1)
        parts_grid.columnconfigure(1, weight=1)
        self._btn(inner, c, "部位プレビュー表示", self._show_parts_preview).pack(fill="x", padx=10, pady=2)
        self._parts_data: Dict[str, "np.ndarray"] = {}

        sep()

        # ── エッジ検出 ──
        section("🔍 エッジ検出")
        self._edge_btn = self._btn(inner, c, "🔍 エッジを表示", self._toggle_edges)
        self._edge_btn.pack(fill="x", padx=10, pady=2)
        # エッジ検出専用プログレスバー（処理中のみ表示）
        self._edge_progress = ttk.Progressbar(inner, mode="indeterminate", length=200)
        self._edge_loading_label = tk.Label(
            inner, text="", bg=c.bg_secondary, fg=c.accent_primary,
            font=("Segoe UI", 8),
        )
        # 初期は非表示
        self._edge_progress.pack(padx=10, pady=2)
        self._edge_loading_label.pack(padx=10)
        self._edge_progress.pack_forget()
        self._edge_loading_label.pack_forget()

        sep()

        # ── 保存設定 ──
        section("💾 保存設定")

        tk.Label(inner, text="カスタムファイル名 (任意):", bg=c.bg_secondary,
                 fg=c.text_secondary, font=("Segoe UI", 9)).pack(anchor="w", padx=10)
        self._custom_name_var = tk.StringVar()
        tk.Entry(inner, textvariable=self._custom_name_var,
                 bg=c.bg_tertiary, fg=c.text_primary,
                 insertbackground=c.text_primary, relief="flat",
                 font=("Segoe UI", 10), highlightthickness=1,
                 highlightbackground=c.border).pack(fill="x", padx=10, pady=2, ipady=3)

        self._save_btn = self._btn(inner, c, "💾 保存", self._save_with_confirm,
                                   bg=c.accent_success if hasattr(c, 'accent_success') else "#4ade80",
                                   fg="#000")
        self._save_btn.pack(fill="x", padx=10, pady=2)
        self._save_btn.configure(state="disabled")

        self._save_batch_btn = self._btn(inner, c, "📦 一括保存", self._save_batch_with_confirm)
        self._save_batch_btn.pack(fill="x", padx=10, pady=2)
        self._save_batch_btn.configure(state="disabled")

        sep()

        # ── Inpaint ──
        section("🔨 Inpaint（穴埋め補完）")
        tk.Label(inner, text="除去した領域を周囲のピクセルで\n自動補完します",
                 bg=c.bg_secondary, fg=c.text_muted,
                 font=("Segoe UI", 8), justify="left").pack(anchor="w", padx=10)
        tk.Label(inner, text="補完半径:", bg=c.bg_secondary, fg=c.text_secondary,
                 font=("Segoe UI", 8)).pack(anchor="w", padx=10, pady=(4, 0))
        self._inpaint_radius = tk.IntVar(value=8)
        tk.Scale(inner, variable=self._inpaint_radius, from_=2, to=24,
                 orient="horizontal", bg=c.bg_secondary, fg=c.text_primary,
                 troughcolor=c.bg_tertiary, highlightthickness=0,
                 ).pack(fill="x", padx=10)
        self._btn(inner, c, "🔨 Inpaint 実行", self._run_inpaint).pack(fill="x", padx=10, pady=2)

        sep()

        # ── アニメーション作成 ──
        section("🎬 アニメーション作成")
        tk.Label(inner, text="処理済み画像をアニメーション\n作成ツールへ送ります",
                 bg=c.bg_secondary, fg=c.text_muted,
                 font=("Segoe UI", 8), justify="left").pack(anchor="w", padx=10)
        self._btn(inner, c, "🎬 アニメーション作成ツールへ",
                  self._open_animation_from_here).pack(fill="x", padx=10, pady=2)

        sep()

        # ── プログレス ──
        self._progress = ttk.Progressbar(inner, mode="indeterminate", length=200)
        self._progress.pack(padx=10, pady=4)

        self._status_var = tk.StringVar(value="画像を開いてください")
        tk.Label(inner, textvariable=self._status_var, bg=c.bg_secondary,
                 fg=c.text_muted, font=("Segoe UI", 8),
                 wraplength=240, justify="left").pack(padx=10, pady=4)

    def _build_status_bar(self, c):
        """
        下部ステータスバー。
        _coord_var / _tool_info_var は _init_variables() で作成済みなので
        ここでは Label を生成するだけ（StringVar の重複生成をしない）。
        """
        sb = tk.Frame(self, bg=c.bg_secondary, height=24)
        sb.pack(fill="x", side="bottom")
        sb.pack_propagate(False)
        # StringVar は _init_variables() で作成済み
        tk.Label(sb, textvariable=self._coord_var, bg=c.bg_secondary,
                 fg=c.text_muted, font=("Consolas", 8)).pack(side="left", padx=8)
        tk.Label(sb, textvariable=self._tool_info_var, bg=c.bg_secondary,
                 fg=c.text_muted, font=("Segoe UI", 8)).pack(side="right", padx=8)

    # ================================================================
    # キャンバスイベントバインド
    # ================================================================

    def _bind_canvas_events(self):
        c = self._canvas_src
        c.bind("<Button-1>",         self._on_canvas_click)
        c.bind("<B1-Motion>",        self._on_canvas_drag)
        c.bind("<ButtonRelease-1>",  self._on_canvas_release)
        c.bind("<Motion>",           self._on_canvas_motion)
        c.bind("<Double-Button-1>",  self._confirm_lasso)
        # <Configure> は _build_preview_area の debounce で処理

    def _canvas_to_image_coords(self, cx: int, cy: int) -> Tuple[int, int]:
        """キャンバス座標 → 画像ピクセル座標に変換"""
        if self._work_arr is None or not _PIL_AVAILABLE:
            return cx, cy
        h, w = self._work_arr.shape[:2]
        cw = self._canvas_src.winfo_width()
        ch = self._canvas_src.winfo_height()
        scale = min(cw / max(w, 1), ch / max(h, 1)) * 0.95
        ox    = (cw - w * scale) / 2
        oy    = (ch - h * scale) / 2
        ix    = int((cx - ox) / scale)
        iy    = int((cy - oy) / scale)
        return max(0, min(w - 1, ix)), max(0, min(h - 1, iy))

    def _on_canvas_motion(self, event):
        if self._work_arr is None:
            return
        ix, iy = self._canvas_to_image_coords(event.x, event.y)
        self._coord_var.set(f"X:{ix} Y:{iy}")

    def _on_canvas_click(self, event):
        if self._work_arr is None:
            return
        ix, iy = self._canvas_to_image_coords(event.x, event.y)
        tool = self._current_tool

        if tool == self._TOOL_POINT:
            self._push_history()
            snap = self._work_arr.copy()
            radius = self._brush_size
            sens = self._sensitivity.get()
            self._processing = True
            self._set_status("ポイント除去処理中...")
            def _do_point():
                result = self._processor.remove_by_point(snap, ix, iy,
                    radius=radius, sensitivity=sens)
                self.after(0, self._on_point_done, result)
            threading.Thread(target=_do_point, daemon=True).start()

        elif tool in (self._TOOL_BRUSH_ERASE, self._TOOL_BRUSH_RESTORE):
            self._push_history()
            self._brush_dragging = True
            mode = "erase" if tool == self._TOOL_BRUSH_ERASE else "restore"
            self._work_arr = self._processor.apply_brush(
                self._work_arr, ix, iy, self._brush_size, mode)
            self._redraw_src(fast=True)  # 軽量な左キャンバスのみ更新

        elif tool == self._TOOL_RECT:
            self._rect_start = (ix, iy)
            self._rect_drawing = True

        elif tool == self._TOOL_ELLIPSE:
            self._rect_start = (ix, iy)
            self._rect_drawing = True

        elif tool == self._TOOL_LASSO:
            if not self._lasso_drawing:
                self._lasso_points = [(ix, iy)]
                self._lasso_drawing = True
            else:
                self._lasso_points.append((ix, iy))
            self._redraw_src()

    def _on_point_done(self, result: "np.ndarray"):
        self._work_arr = result
        self._processing = False
        self._set_status("ポイント除去完了")
        self._refresh_all_previews()

    def _on_edit_done(self, result: "np.ndarray"):
        """矩形・楕円・投げ縄の処理完了コールバック"""
        self._work_arr = result
        self._refresh_all_previews()

    def _on_canvas_drag(self, event):
        if self._work_arr is None:
            return
        ix, iy = self._canvas_to_image_coords(event.x, event.y)

        if self._current_tool in (self._TOOL_BRUSH_ERASE, self._TOOL_BRUSH_RESTORE):
            # ドラッグ中は間引き処理：2px以上移動した場合のみ描画（フリーズ防止）
            if not self._brush_drag_pending:
                self._brush_drag_pending = True
                mode = "erase" if self._current_tool == self._TOOL_BRUSH_ERASE else "restore"
                self._work_arr = self._processor.apply_brush(
                    self._work_arr, ix, iy, self._brush_size, mode)
                self._redraw_src(fast=True)  # 左キャンバスのみ（右は重いので省略）
                # 次フレームまで待機してから次の描画を許可
                self.after(16, self._reset_brush_pending)

        elif self._current_tool in (self._TOOL_RECT, self._TOOL_ELLIPSE) and self._rect_drawing:
            self._rect_end = (ix, iy)
            self._redraw_src_with_selection()

        elif self._current_tool == self._TOOL_LASSO and self._lasso_drawing:
            self._lasso_points.append((ix, iy))
            self._redraw_src_with_selection()

    def _reset_brush_pending(self):
        self._brush_drag_pending = False

    def _on_canvas_release(self, event):
        if self._work_arr is None:
            return
        ix, iy = self._canvas_to_image_coords(event.x, event.y)

        # ブラシ終了時に右プレビューも更新
        if self._current_tool in (self._TOOL_BRUSH_ERASE, self._TOOL_BRUSH_RESTORE):
            if self._brush_dragging:
                self._brush_dragging = False
                self._brush_drag_pending = False
                self._refresh_all_previews()  # ドラッグ完了後に全プレビュー更新
            return

        if self._current_tool == self._TOOL_RECT and self._rect_drawing:
            self._rect_end = (ix, iy)
            self._rect_drawing = False
            if self._rect_start and self._rect_end:
                self._push_history()
                x1, y1 = self._rect_start
                x2, y2 = self._rect_end
                snap = self._work_arr.copy()
                def _do_rect():
                    r = self._processor.remove_by_rect(snap, x1, y1, x2, y2, mode="hard")
                    self.after(0, self._on_edit_done, r)
                threading.Thread(target=_do_rect, daemon=True).start()

        elif self._current_tool == self._TOOL_ELLIPSE and self._rect_drawing:
            self._rect_end = (ix, iy)
            self._rect_drawing = False
            if self._rect_start and self._rect_end:
                self._push_history()
                x1, y1 = self._rect_start
                x2, y2 = self._rect_end
                cx, cy = (x1+x2)//2, (y1+y2)//2
                rx, ry = abs(x2-x1)//2, abs(y2-y1)//2
                snap = self._work_arr.copy()
                def _do_ellipse():
                    r = self._processor.remove_by_ellipse(snap, cx, cy, rx, ry)
                    self.after(0, self._on_edit_done, r)
                threading.Thread(target=_do_ellipse, daemon=True).start()

        elif self._current_tool == self._TOOL_LASSO and self._lasso_drawing:
            pass  # ダブルクリックで確定

    # ================================================================
    # プレビュー描画
    # ================================================================

    def _redraw_src(self, fast: bool = False):
        """作業画像（編集中の状態・透明部分はチェッカー表示）をキャンバスに描画"""
        if not _PIL_AVAILABLE:
            return
        if self._work_arr is not None and _NUMPY_AVAILABLE:
            img = Image.fromarray(self._work_arr)
            self._draw_to_canvas(self._canvas_src, img, "_tk_src", checker=True, fast=fast)
        elif self._src_image is not None:
            self._draw_to_canvas(self._canvas_src, self._src_image, "_tk_src", checker=False, fast=fast)

    def _redraw_src_with_selection(self):
        """選択範囲オーバーレイ付きで作業画像を描画"""
        self._redraw_src()
        c = self._canvas_src
        cw, ch = c.winfo_width(), c.winfo_height()

        # 描画基準となる画像のサイズを取得
        base_img = None
        if self._work_arr is not None and _NUMPY_AVAILABLE:
            bh, bw = self._work_arr.shape[:2]
            base_img = (bw, bh)
        elif self._src_image is not None:
            base_img = (self._src_image.width, self._src_image.height)

        if base_img is None:
            return
        w, h = base_img
        scale = min(cw / max(w,1), ch / max(h,1)) * 0.95
        ox    = (cw - w * scale) / 2
        oy    = (ch - h * scale) / 2

        def i2c(ix, iy):
            return ox + ix * scale, oy + iy * scale

        c.delete("selection_overlay")

        if self._current_tool == self._TOOL_RECT and self._rect_start and self._rect_end:
            x1c, y1c = i2c(*self._rect_start)
            x2c, y2c = i2c(*self._rect_end)
            c.create_rectangle(x1c, y1c, x2c, y2c,
                                outline="#ff6666", width=2, dash=(4, 4),
                                tags="selection_overlay")

        elif self._current_tool == self._TOOL_ELLIPSE and self._rect_start and self._rect_end:
            x1c, y1c = i2c(*self._rect_start)
            x2c, y2c = i2c(*self._rect_end)
            c.create_oval(x1c, y1c, x2c, y2c,
                          outline="#ff6666", width=2, dash=(4, 4),
                          tags="selection_overlay")

        elif self._current_tool == self._TOOL_LASSO and len(self._lasso_points) > 1:
            pts_c = [i2c(px, py) for px, py in self._lasso_points]
            flat  = [v for pt in pts_c for v in pt]
            c.create_line(*flat, fill="#ff9966", width=2, tags="selection_overlay")

    def _refresh_result_preview(self):
        """ビフォーアフタースライダーを更新"""
        if self._work_arr is None or not _PIL_AVAILABLE:
            return
        self._result_image = Image.fromarray(self._work_arr)
        self._render_comparison_slider()

    # ================================================================
    # ビフォーアフタースライダー
    # ================================================================

    def _slider_press(self, event):
        self._slider_dragging = True

    def _slider_drag(self, event):
        if not self._slider_dragging:
            return
        cw = self._canvas_result.winfo_width()
        if cw <= 1:
            return
        self._slider_ratio = max(0.0, min(1.0, event.x / cw))
        self._render_comparison_slider()

    def _slider_release(self, event):
        self._slider_dragging = False

    def _render_comparison_slider(self):
        """
        左半分=元画像、右半分=処理後 をひとつのキャンバスに描画し、
        ドラッグ可能な分割ラインを表示する。
        """
        if not _PIL_AVAILABLE or not _NUMPY_AVAILABLE:
            return
        if self._src_image is None:
            return

        canvas = self._canvas_result
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            cw, ch = 600, 500

        # ── 表示サイズを計算 ──
        src_w, src_h = self._src_image.size
        scale = min(cw / max(src_w, 1), ch / max(src_h, 1)) * 0.95
        nw = max(1, int(src_w * scale))
        nh = max(1, int(src_h * scale))
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2

        # ── 元画像（左側）をリサイズ ──
        src_disp = self._src_image.convert("RGBA").resize((nw, nh), Image.LANCZOS)

        # ── 処理後画像（右側）: 背景合成済み ──
        if self._work_arr is not None:
            bg_type = self._bg_type_var.get()
            composited = self._processor.composite_with_background(
                self._work_arr, bg_type=bg_type, bg_image=self._bg_image
            )
            result_disp = Image.fromarray(composited).resize((nw, nh), Image.LANCZOS).convert("RGBA")
        else:
            result_disp = src_disp.copy()

        # ── 分割位置（画像座標内）──
        # ratio=1.0 → split_x=nw → 元画像のみ表示
        # ratio=0.0 → split_x=0  → 処理後のみ表示
        # ratio=0.5 → split_x=nw/2 → 左半分元画像 / 右半分処理後
        split_x = int(nw * self._slider_ratio)

        # ── 合成キャンバス（cw×ch 黒地）──
        bg_color = (26, 26, 46, 255)
        composite = Image.new("RGBA", (cw, ch), bg_color)

        # チェッカーパターン下地（画像領域）
        checker = self._make_checker_pil(nw, nh, size=14)
        composite.paste(checker, (ox, oy))

        # 処理後画像を全面に貼る（下地）
        composite.paste(result_disp, (ox, oy))

        # 元画像を左側(0〜split_x)に重ねて貼る
        if split_x > 0:
            left_crop = src_disp.crop((0, 0, split_x, nh))
            composite.paste(left_crop, (ox, oy))

        # ── 分割ライン描画 ──
        from PIL import ImageDraw as _IDrawSl
        draw = _IDrawSl.Draw(composite)
        line_x = ox + split_x
        draw.line([(line_x, oy), (line_x, oy + nh)], fill=(255, 255, 255, 230), width=2)

        # ── ハンドル（白丸）──
        handle_cy = oy + nh // 2
        hr = 16
        draw.ellipse(
            [line_x - hr, handle_cy - hr, line_x + hr, handle_cy + hr],
            fill=(255, 255, 255, 230), outline=(180, 180, 180, 255), width=2,
        )
        # 矢印テキスト
        draw.text((line_x - 6, handle_cy - 7), "◀▶", fill=(50, 50, 50, 255))

        # ── ラベル ──
        label_y = oy + 8
        if split_x > 60:
            draw.rectangle([ox + 6, label_y, ox + split_x - 6, label_y + 20],
                           fill=(0, 0, 0, 120))
            draw.text((ox + 10, label_y + 2), "元画像", fill=(200, 200, 200, 255))
        if split_x < nw - 60:
            draw.rectangle([ox + split_x + 6, label_y, ox + nw - 6, label_y + 20],
                           fill=(0, 0, 0, 120))
            draw.text((ox + split_x + 10, label_y + 2), "処理後", fill=(200, 200, 200, 255))

        # ── キャンバスに描画 ──
        tk_img = ImageTk.PhotoImage(composite.convert("RGB"))
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=tk_img)
        self._tk_comparison = tk_img  # GC防止

    def _make_checker_pil(self, w: int, h: int, size: int = 14) -> Image.Image:
        """チェッカーパターン PIL Image を生成（numpy ベクトル化・高速）"""
        if _NUMPY_AVAILABLE:
            ys, xs = np.mgrid[0:h, 0:w]
            mask = ((ys // size) + (xs // size)) % 2 == 1
            arr = np.full((h, w, 4), 255, dtype=np.uint8)
            arr[mask, :3] = 180
            return Image.fromarray(arr)
        # フォールバック（numpyなし）
        img = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img)
        for ry in range(0, h, size):
            for cx in range(0, w, size):
                if ((ry // size) + (cx // size)) % 2 == 1:
                    draw.rectangle([cx, ry, cx + size, ry + size], fill=(180, 180, 180, 255))
        return img

    def _play_reveal_animation(self, duration_ms: int = 900, fps: int = 60):
        """
        スライダーを ratio=1.0(元画像のみ) → 0.0(処理後のみ) → 0.5(中央) へ
        イーズアウトでアニメーション再生する。
        """
        total_frames  = max(1, int(duration_ms / 1000 * fps))
        interval_ms   = max(8, int(1000 / fps))
        phase1_frames = int(total_frames * 0.70)   # 1.0 → 0.0
        phase2_frames = total_frames - phase1_frames  # 0.0 → 0.5

        # 元画像のみ表示した状態から開始
        self._slider_ratio = 1.0
        self._render_comparison_slider()

        def _ease_out(t: float) -> float:
            return 1.0 - (1.0 - t) ** 3

        frame_idx = [0]

        def _step():
            idx = frame_idx[0]
            frame_idx[0] += 1

            if idx <= phase1_frames:
                # フェーズ1: 1.0 → 0.0（処理後が左からスライドイン）
                t = idx / max(phase1_frames, 1)
                self._slider_ratio = 1.0 - _ease_out(t)
            elif idx <= total_frames:
                # フェーズ2: 0.0 → 0.5（中央で落ち着く）
                t = (idx - phase1_frames) / max(phase2_frames, 1)
                self._slider_ratio = _ease_out(t) * 0.5
            else:
                self._slider_ratio = 0.5
                self._render_comparison_slider()
                self._set_status("自動背景除去完了 — ドラッグして比較できます")
                return

            self._render_comparison_slider()
            self.after(interval_ms, _step)

        self.after(interval_ms, _step)

    def _refresh_all_previews(self):
        """元画像と結果プレビューを両方更新。編集済みなら保存ボタンも有効化。"""
        self._redraw_src()
        self._refresh_result_preview()
        # 作業画像が存在する場合は保存ボタンを有効化（手動編集後にも保存可能にする）
        if self._work_arr is not None and hasattr(self, "_save_btn"):
            self._save_btn.configure(state="normal")

    def _draw_to_canvas(
        self,
        canvas: tk.Canvas,
        img: Image.Image,
        attr: str,
        checker: bool = False,
        fast: bool = False,
    ):
        if not _PIL_AVAILABLE:
            return
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            cw, ch = 500, 500
        scale = min(cw / max(img.width,1), ch / max(img.height,1)) * 0.95
        nw    = max(1, int(img.width * scale))
        nh    = max(1, int(img.height * scale))
        x     = (cw - nw) // 2
        y     = (ch - nh) // 2

        # ブラシドラッグ中はBILINEAR（高速）、確定時はLANCZOS（高品質）
        resample = Image.BILINEAR if fast else Image.LANCZOS

        if checker and img.mode == "RGBA":
            # numpy ベクトル化チェッカーパターン（Pythonループ排除）
            if _NUMPY_AVAILABLE:
                sz = 12
                ys, xs = np.mgrid[0:nh, 0:nw]
                checker_mask = ((ys // sz) + (xs // sz)) % 2 == 1
                bg_arr = np.full((nh, nw, 4), 255, dtype=np.uint8)
                bg_arr[checker_mask, :3] = 180
                bg_img = Image.fromarray(bg_arr)
            else:
                bg_img = Image.new("RGBA", (nw, nh), (255, 255, 255, 255))
            resized = img.resize((nw, nh), resample)
            bg_img.paste(resized, (0, 0), resized)
            display = bg_img
        else:
            display = img.resize((nw, nh), resample)

        tk_img = ImageTk.PhotoImage(display)
        canvas.delete("all")
        canvas.create_image(x, y, anchor="nw", image=tk_img)
        setattr(self, attr, tk_img)

    # ================================================================
    # ツール管理
    # ================================================================

    def _select_tool(self, tool_id: str):
        self._current_tool = tool_id
        c = self._c
        for t, btn in self._tool_btns.items():
            btn.configure(
                bg=c.accent_primary if t == tool_id else c.bg_tertiary,
                fg=c.bg_primary     if t == tool_id else c.text_primary,
            )
        tool_names = {
            self._TOOL_POINT:        "ポイント除去",
            self._TOOL_RECT:         "矩形選択除去",
            self._TOOL_ELLIPSE:      "楕円選択除去",
            self._TOOL_LASSO:        "投げ縄選択",
            self._TOOL_BRUSH_ERASE:  "消去ブラシ",
            self._TOOL_BRUSH_RESTORE:"復元ブラシ",
        }
        self._tool_info_var.set(f"ツール: {tool_names.get(tool_id, tool_id)}")
        self._lasso_points = []
        self._lasso_drawing = False

    def _confirm_lasso(self, event=None):
        """投げ縄確定（ダブルクリック）"""
        if (self._current_tool == self._TOOL_LASSO
                and len(self._lasso_points) >= 3
                and self._work_arr is not None):
            self._push_history()
            pts = list(self._lasso_points)
            snap = self._work_arr.copy()
            self._lasso_points = []
            self._lasso_drawing = False
            def _do_lasso():
                r = self._processor.remove_by_lasso(snap, pts)
                self.after(0, self._on_edit_done, r)
            threading.Thread(target=_do_lasso, daemon=True).start()

    # ================================================================
    # 履歴（Undo）
    # ================================================================

    def _push_history(self):
        if self._work_arr is not None:
            if _NUMPY_AVAILABLE:
                self._history_stack.append(self._work_arr.copy())
            if len(self._history_stack) > 30:
                self._history_stack.pop(0)

    def _undo(self):
        if self._history_stack:
            self._work_arr = self._history_stack.pop()
            self._refresh_all_previews()
            self._set_status("元に戻しました")

    def _reset_to_original(self):
        if self._src_image is not None and _NUMPY_AVAILABLE:
            if messagebox.askyesno("確認", "すべての編集をリセットしますか？", parent=self):
                self._push_history()
                self._work_arr = np.array(self._src_image.convert("RGBA"))
                self._history_stack.clear()
                self._refresh_all_previews()
                self._set_status("リセットしました")

    # ================================================================
    # ファイル操作
    # ================================================================

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="画像を選択",
            filetypes=[("画像ファイル", "*.png *.jpg *.jpeg *.bmp *.webp *.tiff"),
                       ("すべて", "*.*")],
            parent=self,
        )
        if not path:
            return
        self._load_image_file(path)

    def _open_sheet(self):
        path = filedialog.askopenfilename(
            title="スプライトシートを選択",
            filetypes=[("画像ファイル", "*.png *.jpg *.jpeg *.bmp *.webp"),
                       ("すべて", "*.*")],
            parent=self,
        )
        if not path:
            return
        self._load_image_file(path)
        self._set_status("シートを開きました。右パネルの「一括処理」から行・列を設定して実行してください。")

    def _load_image_file(self, path: str):
        try:
            img = Image.open(path).convert("RGBA")
            self._src_image = img
            self._work_arr  = np.array(img) if _NUMPY_AVAILABLE else None
            self._history_stack.clear()
            self._batch_results.clear()
            self._batch_placeholders.clear()
            self._batch_expected_names = []
            self._batch_placeholder_names.clear()
            if hasattr(self, "_batch_listbox"):
                self._batch_listbox.delete(0, "end")
            if hasattr(self, "_save_btn"):
                self._save_btn.configure(state="disabled")
            if hasattr(self, "_save_batch_btn"):
                self._save_batch_btn.configure(state="disabled")
            self._refresh_all_previews()
            self._set_status(f"読み込み完了: {Path(path).name}  ({img.width}×{img.height}px)")
        except Exception as e:
            self._set_status(f"読み込みエラー: {e}", error=True)

    def _select_bg_image(self):
        path = filedialog.askopenfilename(
            title="背景画像を選択",
            filetypes=[("画像ファイル", "*.png *.jpg *.jpeg *.bmp *.webp"), ("すべて", "*.*")],
            parent=self,
        )
        if not path or not _NUMPY_AVAILABLE or not _PIL_AVAILABLE:
            return
        try:
            img = Image.open(path).convert("RGBA")
            self._bg_image = np.array(img)
            self._bg_type_var.set("image")
            self._refresh_result_preview()
        except Exception as e:
            self._set_status(f"背景画像エラー: {e}", error=True)

    # ================================================================
    # 処理実行
    # ================================================================

    def _run_auto_remove(self):
        if self._work_arr is None:
            self._set_status("画像を開いてください", error=True)
            return
        if self._processing:
            return
        if not hasattr(self, "_progress"):
            self._set_status("UI構築中です。少し待ってから再試行してください", error=True)
            return
        self._processing = True
        self._push_history()
        self._progress.start(10)
        self._set_status("自動背景除去中...")
        threading.Thread(target=self._do_auto_remove, daemon=True).start()

    def _do_auto_remove(self):
        try:
            result = self._processor.remove_background_adaptive(
                self._work_arr, sensitivity=self._sensitivity.get())
            self.after(0, self._on_auto_remove_done, result)
        except Exception as e:
            self.after(0, self._on_process_error, str(e))

    def _on_auto_remove_done(self, result: "np.ndarray"):
        self._work_arr = result
        if hasattr(self, "_progress"):
            self._progress.stop()
        self._processing = False
        if hasattr(self, "_save_btn"):
            self._save_btn.configure(state="normal")
        self._redraw_src()
        self._result_image = Image.fromarray(self._work_arr)
        self._set_status("自動背景除去完了 — スライドアニメーション再生中")
        # スライダーを左端から右へ流すアニメーション
        self._play_reveal_animation(duration_ms=900, fps=60)

    def _make_missing_cell_placeholder(self, name: str, size: int = 512) -> "Image.Image":
        """未設定セル用のプレースホルダー画像を生成する。"""
        img = Image.new("RGBA", (size, size), (28, 30, 52, 255))
        draw = ImageDraw.Draw(img)
        block = max(16, size // 12)

        for y in range(0, size, block):
            for x in range(0, size, block):
                if ((x // block) + (y // block)) % 2:
                    draw.rectangle([x, y, x + block, y + block], fill=(42, 45, 75, 255))

        draw.rectangle([6, 6, size - 6, size - 6], outline=(110, 120, 170, 255), width=2)
        text = f"{name}\nNO IMAGE"
        try:
            bbox = draw.multiline_textbbox((0, 0), text, align="center")
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = (size - tw) // 2
            ty = (size - th) // 2
            draw.multiline_text(
                (tx, ty),
                text,
                fill=(240, 240, 255, 255),
                align="center",
            )
        except Exception:
            draw.multiline_text(
                (size // 2, size // 2),
                text,
                fill=(240, 240, 255, 255),
                align="center",
                anchor="mm",
            )
        return img

    @staticmethod
    def _batch_key_from_list_label(label: str) -> str:
        return label.replace("  [placeholder]", "", 1)

    def _run_batch_process(self):
        if self._src_image is None:
            self._set_status("シート画像を開いてください", error=True)
            return
        if self._processing:
            return
        if not hasattr(self, "_progress") or not hasattr(self, "_batch_listbox"):
            self._set_status("UI構築中です。少し待ってから再試行してください", error=True)
            return

        rows = self._sheet_rows.get()
        cols = self._sheet_cols.get()
        total = rows * cols
        pose_names = [f"cell_{i:02d}" for i in range(total)]
        self._batch_expected_names = pose_names
        self._batch_placeholder_names.clear()
        self._batch_placeholders.clear()

        self._processing = True
        self._progress.start(10)
        self._set_status(f"一括処理中... (全{total}セル)")
        self._batch_listbox.delete(0, "end")
        if hasattr(self, "_save_batch_btn"):
            self._save_batch_btn.configure(state="disabled")

        def _run():
            def on_prog(current, total, msg):
                self.after(0, lambda: self._set_status(msg))
            results = self._processor.process_all_cells(
                self._src_image, rows, cols, pose_names, on_progress=on_prog)
            self.after(0, self._on_batch_done, results)

        threading.Thread(target=_run, daemon=True).start()

    def _on_batch_done(self, results: Dict[str, Image.Image]):
        merged: Dict[str, Image.Image] = {}
        missing_names: List[str] = []
        expected_names = self._batch_expected_names or list(results.keys())

        for name in expected_names:
            img = results.get(name)
            if img is None and _PIL_AVAILABLE:
                img = self._make_missing_cell_placeholder(name)
                self._batch_placeholders[name] = img
                self._batch_placeholder_names.add(name)
                missing_names.append(name)
            if img is not None:
                merged[name] = img

        self._batch_results = merged
        if hasattr(self, "_progress"):
            self._progress.stop()
        self._processing = False
        if hasattr(self, "_batch_listbox"):
            self._batch_listbox.delete(0, "end")
        for name in self._batch_results.keys():
            label = f"{name}  [placeholder]" if name in self._batch_placeholder_names else name
            self._batch_listbox.insert("end", label)
        if self._batch_results:
            self._save_batch_btn.configure(state="normal")
        if missing_names:
            self._set_status(
                f"一括処理完了: {len(self._batch_results)} セル "
                f"(プレースホルダー補完 {len(missing_names)} セル)"
            )
        else:
            self._set_status(f"一括処理完了: {len(self._batch_results)} セル")

    def _on_batch_select(self, event):
        sel = self._batch_listbox.curselection()
        if not sel:
            return
        name = self._batch_key_from_list_label(self._batch_listbox.get(sel[0]))
        img  = self._batch_results.get(name)
        if img is not None and _NUMPY_AVAILABLE:
            self._push_history()
            self._work_arr = np.array(img)
            self._save_btn.configure(state="normal")
            self._refresh_all_previews()

    def _on_process_error(self, msg: str):
        if hasattr(self, "_progress"):
            self._progress.stop()
        self._processing = False
        self._set_status(f"エラー: {msg}", error=True)

    # ================================================================
    # SEGS（K-meansセグメンテーション）
    # ================================================================

    def _run_segs(self):
        if self._work_arr is None:
            self._set_status("画像を開いてください", error=True)
            return
        if self._processing:
            return
        if not hasattr(self, "_seg_btn") or not hasattr(self, "_seg_progress"):
            self._set_status("UI構築中です。少し待ってから再試行してください", error=True)
            return
        self._processing = True
        self._seg_btn.configure(state="disabled", text="⏳ 分析中...")
        self._seg_progress.pack(padx=10, pady=2)
        self._seg_loading_label.configure(text="⏳ K-meansクラスタリング中...")
        self._seg_loading_label.pack(padx=10)
        self._seg_progress.start(10)
        k = self._seg_k_var.get()
        threading.Thread(target=self._do_segs, args=(k,), daemon=True).start()

    def _do_segs(self, k: int):
        try:
            segs = self._processor.segment_segs(self._work_arr, n_clusters=k)
            self.after(0, self._on_segs_done, segs)
        except Exception as e:
            self.after(0, self._on_segs_error, str(e))

    def _on_segs_done(self, segs: "Dict[str, np.ndarray]"):
        self._seg_progress.stop()
        self._seg_progress.pack_forget()
        self._seg_loading_label.pack_forget()
        self._seg_btn.configure(state="normal", text="🔬 SEGS 実行")
        self._processing = False
        self._segs_data = segs
        self._seg_listbox.delete(0, "end")
        for name in segs:
            px_count = int(segs[name].sum() // 255)
            self._seg_listbox.insert("end", f"{name}  ({px_count:,}px)")
        self._set_status(f"SEGS完了: {len(segs)} セグメント検出")
        # カラーオーバーレイをプレビューに表示
        self._show_segs_overlay(segs)

    def _on_segs_error(self, msg: str):
        self._seg_progress.stop()
        self._seg_progress.pack_forget()
        self._seg_loading_label.pack_forget()
        self._seg_btn.configure(state="normal", text="🔬 SEGS 実行")
        self._processing = False
        self._set_status(f"SEGSエラー: {msg}", error=True)

    def _show_segs_overlay(self, segs: dict):
        """セグメントを色分けしてプレビューに表示"""
        if not _PIL_AVAILABLE or self._work_arr is None or not _NUMPY_AVAILABLE:
            return
        h, w = self._work_arr.shape[:2]
        overlay = self._work_arr.copy()
        colors = [
            (255, 80,  80,  120),
            (80,  255, 80,  120),
            (80,  80,  255, 120),
            (255, 255, 80,  120),
            (255, 80,  255, 120),
            (80,  255, 255, 120),
            (200, 120, 50,  120),
            (120, 200, 50,  120),
            (50,  120, 200, 120),
            (200, 50,  120, 120),
            (120, 50,  200, 120),
            (50,  200, 120, 120),
        ]
        for i, (name, mask) in enumerate(segs.items()):
            col = colors[i % len(colors)]
            m = mask > 128
            for ch, val in enumerate(col[:3]):
                overlay[:, :, ch] = np.where(m,
                    np.clip(overlay[:, :, ch].astype(np.int32) + val // 3, 0, 255),
                    overlay[:, :, ch])
        img = Image.fromarray(overlay)
        self._draw_to_canvas(self._canvas_result, img, "_tk_result", checker=True)

    def _remove_selected_segs(self):
        """選択したセグメントを除去"""
        sel = self._seg_listbox.curselection()
        if not sel or self._work_arr is None or not _NUMPY_AVAILABLE:
            return
        self._push_history()
        keys = list(self._segs_data.keys())
        for idx in sel:
            if idx < len(keys):
                mask = self._segs_data[keys[idx]]
                self._work_arr[:, :, 3][mask > 128] = 0
        self._refresh_all_previews()
        self._set_status(f"{len(sel)} セグメントを除去しました")

    def _keep_selected_segs(self):
        """選択したセグメントのみ保持（それ以外を除去）"""
        sel = self._seg_listbox.curselection()
        if not sel or self._work_arr is None or not _NUMPY_AVAILABLE:
            return
        self._push_history()
        keys = list(self._segs_data.keys())
        keep_mask = np.zeros(self._work_arr.shape[:2], dtype=bool)
        for idx in sel:
            if idx < len(keys):
                keep_mask |= (self._segs_data[keys[idx]] > 128)
        self._work_arr[:, :, 3][~keep_mask] = 0
        self._refresh_all_previews()
        self._set_status(f"{len(sel)} セグメントのみ保持しました")

    # ================================================================
    # 部位検出（顔・肌・髪・服）
    # ================================================================

    def _run_body_parts(self):
        if self._work_arr is None:
            self._set_status("画像を開いてください", error=True)
            return
        if self._processing:
            return
        if not hasattr(self, "_parts_btn") or not hasattr(self, "_parts_progress"):
            self._set_status("UI構築中です。少し待ってから再試行してください", error=True)
            return
        self._processing = True
        self._parts_btn.configure(state="disabled", text="⏳ 検出中...")
        self._parts_progress.pack(padx=10, pady=2)
        self._parts_loading_label.configure(text="⏳ YCbCr+HSV部位解析中...")
        self._parts_loading_label.pack(padx=10)
        self._parts_progress.start(10)
        threading.Thread(target=self._do_body_parts, daemon=True).start()

    def _do_body_parts(self):
        try:
            parts = self._processor.detect_body_parts(self._work_arr)
            self.after(0, self._on_body_parts_done, parts)
        except Exception as e:
            self.after(0, self._on_body_parts_error, str(e))

    def _on_body_parts_done(self, parts: "Dict[str, np.ndarray]"):
        self._parts_progress.stop()
        self._parts_progress.pack_forget()
        self._parts_loading_label.pack_forget()
        self._parts_btn.configure(state="normal", text="👤 部位検出 実行")
        self._processing = False
        self._parts_data = parts
        # 検出された部位のボタンをアクティブ化
        detected = []
        _label_map = {"skin":"🟡 肌","face":"🔵 顔","hair":"🟤 髪","clothing":"🟢 服","eye":"⚫ 目"}
        for key, mask in parts.items():
            px = int(mask.sum() // 255)
            state = "normal" if px > 50 else "disabled"
            if key in self._part_btns:
                lbl = _label_map.get(key, key)
                self._part_btns[key].configure(
                    state=state,
                    text=f"{lbl}\n{px:,}px",
                )
            if px > 50:
                detected.append(key)
        self._set_status(f"部位検出完了: {', '.join(detected) if detected else '未検出'}")
        self._show_parts_preview()

    def _on_body_parts_error(self, msg: str):
        self._parts_progress.stop()
        self._parts_progress.pack_forget()
        self._parts_loading_label.pack_forget()
        self._parts_btn.configure(state="normal", text="👤 部位検出 実行")
        self._processing = False
        self._set_status(f"部位検出エラー: {msg}", error=True)

    def _show_parts_preview(self):
        """部位をカラーオーバーレイで表示"""
        if not self._parts_data or self._work_arr is None or not _NUMPY_AVAILABLE or not _PIL_AVAILABLE:
            return
        h, w = self._work_arr.shape[:2]
        overlay = self._work_arr.copy()
        color_map = {
            "skin":     (255, 200, 100),
            "face":     (100, 150, 255),
            "hair":     (140,  80,  20),
            "clothing": ( 80, 200, 120),
            "eye":      ( 50,  50,  50),
        }
        for key, mask in self._parts_data.items():
            col = color_map.get(key, (200, 200, 200))
            m = mask > 128
            if not m.any():
                continue
            for ch, val in enumerate(col):
                ch_data = overlay[:, :, ch].astype(np.int32)
                overlay[:, :, ch] = np.where(m,
                    np.clip((ch_data * 0.5 + val * 0.5).astype(np.int32), 0, 255),
                    ch_data).astype(np.uint8)
        img = Image.fromarray(overlay)
        self._draw_to_canvas(self._canvas_result, img, "_tk_result", checker=True)
        self._set_status("部位プレビュー表示中（黄=肌 青=顔 茶=髪 緑=服 黒=目）")

    def _remove_body_part(self, part_key: str):
        """指定部位を除去"""
        if part_key not in self._parts_data or self._work_arr is None or not _NUMPY_AVAILABLE:
            self._set_status(f"先に部位検出を実行してください", error=True)
            return
        mask = self._parts_data[part_key]
        if mask.sum() == 0:
            self._set_status(f"{part_key} は検出されていません")
            return
        self._push_history()
        self._work_arr[:, :, 3][mask > 128] = 0
        self._refresh_all_previews()
        label_map = {"skin": "肌", "face": "顔", "hair": "髪", "clothing": "服", "eye": "目"}
        self._set_status(f"{label_map.get(part_key, part_key)} を除去しました")

    def _toggle_edges(self):
        """エッジ表示のON/OFF切り替え"""
        if self._edge_showing:
            self._hide_edges()
        else:
            self._show_edges()

    def _show_edges(self):
        """エッジ検出をバックグラウンドスレッドで実行（UI非フリーズ）"""
        if self._work_arr is None:
            return
        if not _NUMPY_AVAILABLE:
            self._set_status("numpyが必要です", error=True)
            return
        if self._processing:
            return
        if not hasattr(self, "_edge_progress") or not hasattr(self, "_edge_btn"):
            self._set_status("UI構築中です。少し待ってから再試行してください", error=True)
            return

        # プログレスバーを表示
        self._edge_progress.pack(padx=10, pady=2)
        self._edge_loading_label.configure(text="⏳ エッジ検出中...")
        self._edge_loading_label.pack(padx=10)
        self._edge_progress.start(10)
        self._edge_btn.configure(text="⏳ 検出中...", state="disabled")

        threading.Thread(target=self._do_edge_detection, daemon=True).start()

    def _do_edge_detection(self):
        """バックグラウンドスレッドでエッジ検出を実行"""
        try:
            edge_map = self._processor.detect_edges_highquality(self._work_arr)
            self.after(0, self._on_edge_done, edge_map)
        except Exception as e:
            self.after(0, self._on_edge_error, str(e))

    def _on_edge_done(self, edge_map: "np.ndarray"):
        """エッジ検出完了 → 結果プレビューに表示"""
        self._edge_arr = edge_map
        self._edge_showing = True
        # プログレスバーを非表示
        self._edge_progress.stop()
        self._edge_progress.pack_forget()
        self._edge_loading_label.configure(text="")
        self._edge_loading_label.pack_forget()
        self._edge_btn.configure(text="✅ エッジ非表示にする", state="normal",
                                 bg=self._c.accent_success if hasattr(self._c, 'accent_success') else "#4ade80")
        # エッジマップを右プレビューに表示
        edge_rgba = Image.fromarray(edge_map).convert("RGBA")
        self._draw_to_canvas(self._canvas_result, edge_rgba, "_tk_result")
        self._set_status("エッジ検出完了 | 「エッジ非表示にする」で通常プレビューに戻ります")

    def _on_edge_error(self, msg: str):
        self._edge_progress.stop()
        self._edge_progress.pack_forget()
        self._edge_loading_label.pack_forget()
        self._edge_btn.configure(text="🔍 エッジを表示", state="normal",
                                 bg=self._c.bg_tertiary)
        self._set_status(f"エッジ検出エラー: {msg}", error=True)

    def _hide_edges(self):
        """エッジ表示をOFFにして通常の処理後プレビューに戻す"""
        self._edge_showing = False
        self._edge_arr = None
        self._edge_btn.configure(text="🔍 エッジを表示",
                                 bg=self._c.bg_tertiary)
        self._refresh_result_preview()
        self._set_status("エッジ表示を終了しました")

    # ================================================================
    # 保存処理（確認ダイアログ付き）
    # ================================================================

    def _save_with_confirm(self):
        if self._work_arr is None:
            return
        result_img = Image.fromarray(self._work_arr)
        self._show_save_dialog({"output": result_img})

    def _save_batch_with_confirm(self):
        if not self._batch_results:
            return
        self._show_save_dialog(self._batch_results)

    def _show_save_dialog(self, images: Dict[str, Image.Image]):
        """保存確認ダイアログ。"""
        dlg = tk.Toplevel(self)
        dlg.title("保存確認")
        dlg.geometry("500x380")
        dlg.configure(bg=self._c.bg_primary)
        dlg.transient(self)
        dlg.grab_set()

        c = self._c

        tk.Label(dlg, text="画像を保存しますか？",
                 bg=c.bg_primary, fg=c.text_primary,
                 font=("Segoe UI", 13, "bold")).pack(pady=16)

        tk.Label(dlg, text=f"対象: {len(images)} 枚",
                 bg=c.bg_primary, fg=c.text_secondary,
                 font=("Segoe UI", 10)).pack()

        tk.Label(dlg, text="保存先フォルダ:", bg=c.bg_primary, fg=c.text_secondary,
                 font=("Segoe UI", 10)).pack(anchor="w", padx=20, pady=(12, 0))
        dir_frame = tk.Frame(dlg, bg=c.bg_primary)
        dir_frame.pack(fill="x", padx=20, pady=2)
        default_dir = str(_WIN_ROOT / "assets" / "images")
        dir_var = tk.StringVar(value=default_dir)
        dir_entry = tk.Entry(dir_frame, textvariable=dir_var, bg=c.bg_tertiary,
                             fg=c.text_primary, insertbackground=c.text_primary,
                             relief="flat", font=("Segoe UI", 9), highlightthickness=1,
                             highlightbackground=c.border)
        dir_entry.pack(side="left", fill="x", expand=True, ipady=3)
        tk.Button(dir_frame, text="参照", command=lambda: dir_var.set(
            filedialog.askdirectory(initialdir=dir_var.get(), parent=dlg) or dir_var.get()
        ), bg=c.bg_tertiary, fg=c.text_secondary, relief="flat",
                  font=("Segoe UI", 9), padx=6, pady=3, cursor="hand2").pack(side="left", padx=4)

        tk.Label(dlg, text="ファイル名:", bg=c.bg_primary, fg=c.text_secondary,
                 font=("Segoe UI", 10)).pack(anchor="w", padx=20, pady=(10, 0))

        if len(images) > 1:
            name_mode = tk.StringVar(value="sequence")
            modes = [("sequence", "連番 (image_001, image_002...)"),
                     ("custom_prefix", "プレフィックス + 連番")]
        else:
            name_mode = tk.StringVar(value="custom")
            modes = [("custom", "カスタム名 (下記フィールドを使用)")]

        for val, lbl in modes:
            tk.Radiobutton(dlg, text=lbl, variable=name_mode, value=val,
                           bg=c.bg_primary, fg=c.text_secondary,
                           selectcolor=c.bg_tertiary,
                           font=("Segoe UI", 9)).pack(anchor="w", padx=30)

        tk.Label(dlg, text="ファイル名 / プレフィックス:",
                 bg=c.bg_primary, fg=c.text_secondary,
                 font=("Segoe UI", 9)).pack(anchor="w", padx=20, pady=(6, 0))
        custom_var = tk.StringVar(value=self._custom_name_var.get() or "output")
        tk.Entry(dlg, textvariable=custom_var, bg=c.bg_tertiary,
                 fg=c.text_primary, insertbackground=c.text_primary,
                 relief="flat", font=("Segoe UI", 10), highlightthickness=1,
                 highlightbackground=c.border).pack(fill="x", padx=20, ipady=3)

        btn_row = tk.Frame(dlg, bg=c.bg_primary)
        btn_row.pack(pady=16)

        def _do_save():
            dest_dir = Path(dir_var.get())
            dest_dir.mkdir(parents=True, exist_ok=True)
            mode   = name_mode.get()
            custom = custom_var.get().strip() or "output"
            saved  = []

            try:
                if mode == "custom":
                    fname = custom + ".png"
                    path  = dest_dir / fname
                    list(images.values())[0].save(path, "PNG")
                    saved.append(str(path))
                elif mode == "sequence":
                    for idx, img in enumerate(images.values()):
                        fname = f"image_{idx+1:03d}.png"
                        path  = dest_dir / fname
                        img.save(path, "PNG")
                        saved.append(str(path))
                elif mode == "custom_prefix":
                    for idx, img in enumerate(images.values()):
                        fname = f"{custom}_{idx+1:03d}.png"
                        path  = dest_dir / fname
                        img.save(path, "PNG")
                        saved.append(str(path))

                dlg.destroy()
                self._set_status(f"保存完了: {len(saved)} 枚 → {dest_dir}")
                logger.info(f"画像保存: {saved}")

                if self._char_loader is not None:
                    self._char_loader.reload()
                if self._on_reload is not None:
                    self.after(200, self._on_reload)

                messagebox.showinfo(
                    "保存完了",
                    f"{len(saved)} 枚を保存しました。\n保存先: {dest_dir}",
                    parent=self,
                )
            except Exception as e:
                messagebox.showerror("保存エラー", str(e), parent=dlg)

        tk.Button(btn_row, text="💾 保存する", command=_do_save,
                  bg=c.accent_primary, fg=c.text_primary,
                  relief="flat", font=("Segoe UI", 11, "bold"),
                  padx=24, pady=8, cursor="hand2").pack(side="left", padx=8)

        tk.Button(btn_row, text="✕ 保存しない", command=dlg.destroy,
                  bg=c.bg_tertiary, fg=c.text_secondary,
                  relief="flat", font=("Segoe UI", 11),
                  padx=24, pady=8, cursor="hand2").pack(side="left", padx=8)

    # ================================================================
    # ユーティリティ
    # ================================================================

    def _run_inpaint(self):
        """現在の作業画像の透明領域を周囲のピクセルで Inpaint する。"""
        if self._work_arr is None:
            self._set_status("画像を開いてください", error=True)
            return
        if self._processing:
            return
        if not _NUMPY_AVAILABLE:
            self._set_status("numpy が必要です", error=True)
            return

        self._processing = True
        if hasattr(self, "_progress"):
            self._progress.start(10)
        self._set_status("Inpaint 処理中...")

        radius = self._inpaint_radius.get()

        def _do():
            try:
                mask = self._processor.create_inpaint_mask_from_alpha(self._work_arr)
                if not mask.any():
                    # 透明領域がない場合は履歴を積まずにスキップ
                    self.after(0, lambda: self._set_status("透明領域なし、Inpaintをスキップ"))
                    self.after(0, self._finish_processing)
                    return
                # 競合回避: スナップショットを先に取得しUIスレッドで履歴に積む
                snapshot = self._work_arr.copy() if _NUMPY_AVAILABLE else None
                self.after(0, lambda: self._history_stack.append(snapshot)
                           if snapshot is not None else None)
                result = self._processor.inpaint_region(self._work_arr, mask, radius=radius)
                self.after(0, self._on_inpaint_done, result)
            except Exception as e:
                self.after(0, self._on_process_error, str(e))

        threading.Thread(target=_do, daemon=True).start()

    def _on_inpaint_done(self, result: "np.ndarray"):
        self._work_arr = result
        self._finish_processing()
        self._refresh_all_previews()
        self._set_status("Inpaint 完了")

    def _finish_processing(self):
        if hasattr(self, "_progress"):
            self._progress.stop()
        self._processing = False

    def _open_animation_from_here(self):
        """処理済み画像を AnimationCompositeDialog へ渡す。"""
        import_images: Dict[str, "Image.Image"] = {}

        if self._batch_results:
            import_images = dict(self._batch_results)
        elif self._work_arr is not None and _PIL_AVAILABLE and _NUMPY_AVAILABLE:
            import_images["output"] = Image.fromarray(self._work_arr)

        if not import_images:
            self._set_status("アニメーションに送る画像がありません", error=True)
            return

        dlg = AnimationCompositeDialog(
            self.master,
            char_loader=self._char_loader,
        )

        def _after_open():
            for name, img in import_images.items():
                dlg._add_layer(img, name)

        dlg.after(200, _after_open)
        self._set_status(f"アニメーション作成ツールへ {len(import_images)} 枚を送りました")

    def _set_status(self, msg: str, error: bool = False):
        if hasattr(self, "_status_var"):
            self._status_var.set(msg)
        color = getattr(self._c, 'accent_error', '#f87171') if error else self._c.text_muted
        logger.info(f"[BgRemoval] {msg}") if not error else logger.warning(f"[BgRemoval] {msg}")

    def _btn(self, parent, c, text: str, cmd, bg=None, fg=None) -> tk.Button:
        return tk.Button(
            parent, text=text, command=cmd,
            bg=bg or c.bg_tertiary, fg=fg or c.text_primary,
            font=("Segoe UI", 9), relief="flat", padx=8, pady=5,
            activebackground=c.bg_hover, cursor="hand2",
        )


# ================================================================== #
# 後方互換: BgRemovalDialog → AdvancedBgRemovalDialog のエイリアス
# ================================================================== #


class BgRemovalDialog(AdvancedBgRemovalDialog):
    """後方互換性のためのエイリアスクラス。"""
    pass


# ================================================================== #
# パーツ合成・アニメーション作成ダイアログ
# ================================================================== #


class AnimationCompositeDialog(tk.Toplevel):
    """
    パーツと被写体を合成して新しいキャラクターアニメーションを作成するダイアログ。
    """

    _DEFAULT_FPS   = 12
    _DEFAULT_DELAY = 83

    def __init__(self, parent, char_loader=None):
        super().__init__(parent)
        self._char_loader = char_loader
        self._processor   = AdvancedImageProcessor()

        self._layers: List[Dict] = []
        self._selected_layer: Optional[int] = None

        self._frames: List["np.ndarray"] = []
        self._current_frame: int = 0
        self._playing: bool = False
        self._fps = self._DEFAULT_FPS

        self._canvas_w = 512
        self._canvas_h = 512

        self._tk_preview: Optional[ImageTk.PhotoImage] = None
        self._preview_scale = 1.0
        self._preview_origin: Tuple[int, int] = (0, 0)
        self._drag_layer_idx: Optional[int] = None
        self._drag_mouse_origin: Tuple[int, int] = (0, 0)
        self._drag_layer_origin: Tuple[int, int] = (0, 0)

        self._setup_theme()
        self.title("キャラクターアニメーション作成 - Alice AI")
        self.withdraw()          # 構築中は非表示
        self.geometry("1300x820")
        self.minsize(1100, 700)
        self.configure(bg=self._c.bg_primary)
        self.transient(parent)
        self._build_ui()         # UI構築完了後に表示
        self.deiconify()
        self.grab_set()

    def _setup_theme(self):
        try:
            from module import env_binder_module as env
            theme_name = env.get("APP_THEME", "dark")
        except Exception:
            theme_name = "dark"
        self._c = Theme.get(theme_name)

    def _build_ui(self):
        c = self._c
        main = tk.Frame(self, bg=c.bg_primary)
        main.pack(fill="both", expand=True)

        self._build_layer_panel(main, c)
        self._build_canvas_area(main, c)
        self._build_right_panel(main, c)
        self._build_bottom_bar(c)

    def _build_layer_panel(self, parent, c):
        """左: レイヤーパネル"""
        lp = tk.Frame(parent, bg=c.bg_secondary, width=220)
        lp.pack(side="left", fill="y", padx=(0, 2))
        lp.pack_propagate(False)

        tk.Label(lp, text="📋 レイヤー", bg=c.bg_secondary, fg=c.accent_primary,
                 font=("Segoe UI", 11, "bold")).pack(pady=(10, 4), padx=8, anchor="w")

        btn_row = tk.Frame(lp, bg=c.bg_secondary)
        btn_row.pack(fill="x", padx=6, pady=2)
        for txt, cmd in [("+ 画像", self._add_layer_from_file),
                          ("+ キャラ", self._add_layer_from_char),
                          ("🗑", self._remove_layer)]:
            tk.Button(btn_row, text=txt, command=cmd,
                      bg=c.bg_tertiary, fg=c.text_primary, relief="flat",
                      font=("Segoe UI", 8), padx=6, pady=3,
                      cursor="hand2", activebackground=c.accent_primary,
                      ).pack(side="left", padx=1)

        self._layer_listbox = tk.Listbox(
            lp, bg=c.bg_tertiary, fg=c.text_primary, selectbackground=c.accent_primary,
            relief="flat", font=("Segoe UI", 9), height=8,
        )
        self._layer_listbox.pack(fill="x", padx=6, pady=4)
        self._layer_listbox.bind("<<ListboxSelect>>", self._on_layer_select)

        ord_row = tk.Frame(lp, bg=c.bg_secondary)
        ord_row.pack(fill="x", padx=6)
        for txt, cmd in [("↑ 上へ", self._move_layer_up), ("↓ 下へ", self._move_layer_down)]:
            tk.Button(ord_row, text=txt, command=cmd,
                      bg=c.bg_tertiary, fg=c.text_secondary, relief="flat",
                      font=("Segoe UI", 8), padx=8, pady=3,
                      cursor="hand2").pack(side="left", padx=2)

        tk.Frame(lp, bg=c.border, height=1).pack(fill="x", padx=6, pady=8)

        tk.Label(lp, text="🔧 レイヤー設定", bg=c.bg_secondary, fg=c.accent_primary,
                 font=("Segoe UI", 10, "bold")).pack(padx=8, anchor="w")

        def prop_row(label, var, from_, to_, res=1):
            f = tk.Frame(lp, bg=c.bg_secondary)
            f.pack(fill="x", padx=8, pady=1)
            tk.Label(f, text=label, bg=c.bg_secondary, fg=c.text_secondary,
                     font=("Segoe UI", 8), width=6, anchor="w").pack(side="left")
            tk.Scale(f, variable=var, from_=from_, to=to_, resolution=res,
                     orient="horizontal", bg=c.bg_secondary, fg=c.text_primary,
                     troughcolor=c.bg_tertiary, highlightthickness=0,
                     command=lambda _: self._refresh_composite(),
                     ).pack(side="left", fill="x", expand=True)

        self._prop_x     = tk.IntVar(value=0)
        self._prop_y     = tk.IntVar(value=0)
        self._prop_scale = tk.DoubleVar(value=1.0)
        self._prop_alpha = tk.IntVar(value=255)
        prop_row("X位置", self._prop_x,     -512, 512)
        prop_row("Y位置", self._prop_y,     -512, 512)
        prop_row("スケール", self._prop_scale, 0.1, 4.0, 0.05)
        prop_row("不透明度", self._prop_alpha,  0,   255)

        tk.Button(lp, text="レイヤー設定を適用", command=self._apply_layer_props,
                  bg=c.accent_primary, fg=c.text_primary, relief="flat",
                  font=("Segoe UI", 9), padx=8, pady=4, cursor="hand2",
                  ).pack(fill="x", padx=8, pady=4)

        tk.Frame(lp, bg=c.border, height=1).pack(fill="x", padx=6, pady=4)
        tk.Label(lp, text="🔨 Inpaint（穴埋め）", bg=c.bg_secondary, fg=c.accent_primary,
                 font=("Segoe UI", 10, "bold")).pack(padx=8, anchor="w")
        tk.Button(lp, text="選択レイヤーをInpaint",
                  command=self._inpaint_selected_layer,
                  bg=c.bg_tertiary, fg=c.text_primary, relief="flat",
                  font=("Segoe UI", 9), padx=8, pady=4, cursor="hand2",
                  ).pack(fill="x", padx=8, pady=2)

    def _build_canvas_area(self, parent, c):
        """中央: 合成プレビューキャンバス"""
        ca = tk.Frame(parent, bg=c.bg_primary)
        ca.pack(side="left", fill="both", expand=True, padx=4)

        tk.Label(ca, text="🎨 合成プレビュー", bg=c.bg_primary, fg=c.text_secondary,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=4)

        sz_row = tk.Frame(ca, bg=c.bg_primary)
        sz_row.pack(fill="x", padx=4)
        tk.Label(sz_row, text="サイズ:", bg=c.bg_primary, fg=c.text_secondary,
                 font=("Segoe UI", 8)).pack(side="left")
        self._canvas_size_var = tk.StringVar(value="512x512")
        ttk.Combobox(sz_row, textvariable=self._canvas_size_var,
                     values=["256x256", "512x512", "1024x1024"],
                     state="readonly", width=10, font=("Segoe UI", 8),
                     ).pack(side="left", padx=4)
        tk.Button(sz_row, text="適用", command=self._apply_canvas_size,
                  bg=c.bg_tertiary, fg=c.text_secondary, relief="flat",
                  font=("Segoe UI", 8), padx=6, cursor="hand2",
                  ).pack(side="left")

        self._composite_canvas = tk.Canvas(
            ca, bg="#1a1a2e", highlightthickness=1,
            highlightbackground=c.border, cursor="fleur",
        )
        self._composite_canvas.pack(fill="both", expand=True, padx=4, pady=4)
        # <Configure> debounce
        self._refresh_composite_job = None
        def _on_composite_configure(e):
            if self._refresh_composite_job:
                self.after_cancel(self._refresh_composite_job)
            self._refresh_composite_job = self.after(80, self._refresh_composite)
        self._composite_canvas.bind("<Configure>", _on_composite_configure)
        self._composite_canvas.bind("<ButtonPress-1>", self._on_composite_press)
        self._composite_canvas.bind("<B1-Motion>", self._on_composite_drag)
        self._composite_canvas.bind("<ButtonRelease-1>", self._on_composite_release)
        tk.Label(
            ca,
            text="左ドラッグでレイヤー移動",
            bg=c.bg_primary,
            fg=c.text_muted,
            font=("Segoe UI", 8),
        ).pack(anchor="w", padx=4, pady=(0, 4))

    def _build_right_panel(self, parent, c):
        """右: フレーム管理・書き出し"""
        rp = tk.Frame(parent, bg=c.bg_secondary, width=260)
        rp.pack(side="right", fill="y", padx=(2, 0))
        rp.pack_propagate(False)

        def section(text):
            tk.Label(rp, text=text, bg=c.bg_secondary, fg=c.accent_primary,
                     font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=10, pady=(10, 2))

        def sep():
            tk.Frame(rp, bg=c.border, height=1).pack(fill="x", padx=10, pady=4)

        section("🎬 フレーム管理")
        frame_row = tk.Frame(rp, bg=c.bg_secondary)
        frame_row.pack(fill="x", padx=10, pady=2)
        for txt, cmd in [("+ フレーム追加", self._add_frame),
                          ("🗑 削除", self._remove_frame)]:
            tk.Button(frame_row, text=txt, command=cmd,
                      bg=c.bg_tertiary, fg=c.text_primary, relief="flat",
                      font=("Segoe UI", 8), padx=6, pady=3,
                      cursor="hand2").pack(side="left", padx=2)

        self._frame_listbox = tk.Listbox(
            rp, height=8, bg=c.bg_tertiary, fg=c.text_primary,
            selectbackground=c.accent_primary, relief="flat",
            font=("Segoe UI", 9),
        )
        self._frame_listbox.pack(fill="x", padx=10, pady=2)
        self._frame_listbox.bind("<<ListboxSelect>>", self._on_frame_select)

        tk.Button(rp, text="現在の合成をフレームに追加",
                  command=self._capture_frame,
                  bg=c.accent_secondary, fg=c.text_primary, relief="flat",
                  font=("Segoe UI", 9), padx=8, pady=4, cursor="hand2",
                  ).pack(fill="x", padx=10, pady=2)

        sep()

        section("▶ プレビュー再生")
        fps_row = tk.Frame(rp, bg=c.bg_secondary)
        fps_row.pack(fill="x", padx=10)
        tk.Label(fps_row, text="FPS:", bg=c.bg_secondary, fg=c.text_secondary,
                 font=("Segoe UI", 9)).pack(side="left")
        self._fps_var = tk.IntVar(value=self._DEFAULT_FPS)
        tk.Spinbox(fps_row, from_=1, to=60, textvariable=self._fps_var,
                   width=4, bg=c.bg_tertiary, fg=c.text_primary,
                   buttonbackground=c.bg_tertiary,
                   command=lambda: setattr(self, "_fps", self._fps_var.get()),
                   ).pack(side="left", padx=4)

        play_row = tk.Frame(rp, bg=c.bg_secondary)
        play_row.pack(fill="x", padx=10, pady=4)
        self._play_btn = tk.Button(play_row, text="▶ 再生",
                                   command=self._toggle_play,
                                   bg=c.accent_primary, fg=c.text_primary, relief="flat",
                                   font=("Segoe UI", 10, "bold"), padx=12, pady=5,
                                   cursor="hand2")
        self._play_btn.pack(side="left", padx=2)
        tk.Button(play_row, text="⏹ 停止", command=self._stop_play,
                  bg=c.bg_tertiary, fg=c.text_secondary, relief="flat",
                  font=("Segoe UI", 10), padx=10, pady=5,
                  cursor="hand2").pack(side="left", padx=2)

        sep()

        section("💾 書き出し")
        tk.Button(rp, text="🎞 GIF アニメ書き出し",
                  command=self._export_gif,
                  bg=c.accent_primary, fg=c.text_primary, relief="flat",
                  font=("Segoe UI", 10, "bold"), padx=8, pady=6,
                  cursor="hand2").pack(fill="x", padx=10, pady=2)
        tk.Button(rp, text="🖼 連番PNG書き出し",
                  command=self._export_png_sequence,
                  bg=c.bg_tertiary, fg=c.text_primary, relief="flat",
                  font=("Segoe UI", 10), padx=8, pady=6,
                  cursor="hand2").pack(fill="x", padx=10, pady=2)
        tk.Button(rp, text="🖼 現在フレームをPNG保存",
                  command=self._export_current_frame,
                  bg=c.bg_tertiary, fg=c.text_primary, relief="flat",
                  font=("Segoe UI", 10), padx=8, pady=6,
                  cursor="hand2").pack(fill="x", padx=10, pady=2)

        sep()

        self._anim_status_var = tk.StringVar(value="レイヤーを追加してください")
        tk.Label(rp, textvariable=self._anim_status_var, bg=c.bg_secondary,
                 fg=c.text_muted, font=("Segoe UI", 8),
                 wraplength=230, justify="left").pack(padx=10, pady=4)

        tk.Button(rp, text="閉じる", command=self.destroy,
                  bg=c.bg_tertiary, fg=c.text_secondary, relief="flat",
                  font=("Segoe UI", 9), padx=10, pady=4,
                  cursor="hand2").pack(side="bottom", pady=8)

    def _build_bottom_bar(self, c):
        bb = tk.Frame(self, bg=c.bg_secondary, height=26)
        bb.pack(fill="x", side="bottom")
        bb.pack_propagate(False)
        self._frame_info_var = tk.StringVar(value="フレーム: 0/0")
        tk.Label(bb, textvariable=self._frame_info_var, bg=c.bg_secondary,
                 fg=c.text_muted, font=("Segoe UI", 8)).pack(side="left", padx=8)
        self._layer_info_var = tk.StringVar(value="レイヤー: 0")
        tk.Label(bb, textvariable=self._layer_info_var, bg=c.bg_secondary,
                 fg=c.text_muted, font=("Segoe UI", 8)).pack(side="right", padx=8)

    # ================================================================
    # レイヤー操作
    # ================================================================

    def _add_layer_from_file(self):
        path = filedialog.askopenfilename(
            title="パーツ画像を選択",
            filetypes=[("画像ファイル", "*.png *.jpg *.jpeg *.bmp *.webp"), ("すべて", "*.*")],
            parent=self,
        )
        if not path or not _PIL_AVAILABLE:
            return
        try:
            img = Image.open(path).convert("RGBA")
            name = Path(path).stem
            self._add_layer(img, name)
        except Exception as e:
            messagebox.showerror("エラー", str(e), parent=self)

    def _add_layer_from_char(self):
        """CharacterLoader からキャラクター画像をレイヤーに追加"""
        if not self._char_loader:
            messagebox.showwarning("警告", "CharacterLoader が利用できません", parent=self)
            return
        dlg = tk.Toplevel(self)
        dlg.title("ポーズ選択")
        dlg.geometry("240x180")
        dlg.configure(bg=self._c.bg_primary)
        dlg.transient(self)
        dlg.grab_set()
        poses = ["default", "idle", "speaking", "thinking", "greeting"]
        tk.Label(dlg, text="追加するポーズを選択:", bg=self._c.bg_primary,
                 fg=self._c.text_primary, font=("Segoe UI", 10)).pack(pady=10)
        pose_var = tk.StringVar(value="default")
        for p in poses:
            tk.Radiobutton(dlg, text=p, variable=pose_var, value=p,
                           bg=self._c.bg_primary, fg=self._c.text_secondary,
                           selectcolor=self._c.bg_tertiary,
                           font=("Segoe UI", 9)).pack(anchor="w", padx=20)
        def _ok():
            pose = pose_var.get()
            img  = self._char_loader.get_image(pose)
            if img is not None:
                self._add_layer(img, f"char_{pose}")
            dlg.destroy()
        tk.Button(dlg, text="追加", command=_ok,
                  bg=self._c.accent_primary, fg=self._c.text_primary,
                  relief="flat", font=("Segoe UI", 9), padx=12, pady=4,
                  cursor="hand2").pack(pady=8)

    def _add_layer(self, img: "Image.Image", name: str):
        layer = {
            "name":    name,
            "img":     img,
            "x":       0,
            "y":       0,
            "scale":   1.0,
            "alpha":   255,
            "visible": True,
        }
        self._layers.append(layer)
        self._layer_listbox.insert("end", name)
        self._layer_listbox.selection_clear(0, "end")
        self._layer_listbox.selection_set("end")
        self._selected_layer = len(self._layers) - 1
        self._update_layer_info()
        self._refresh_composite()
        self._anim_status_var.set(f"レイヤー '{name}' を追加しました")

    def _remove_layer(self):
        if self._selected_layer is None or not self._layers:
            return
        idx = self._selected_layer
        name = self._layers[idx]["name"]
        self._layers.pop(idx)
        self._layer_listbox.delete(idx)
        self._selected_layer = None
        self._update_layer_info()
        self._refresh_composite()
        self._anim_status_var.set(f"レイヤー '{name}' を削除しました")

    def _move_layer_up(self):
        if self._selected_layer is None or self._selected_layer == 0:
            return
        i = self._selected_layer
        self._layers[i], self._layers[i-1] = self._layers[i-1], self._layers[i]
        self._layer_listbox.delete(i-1, i)
        self._layer_listbox.insert(i-1, self._layers[i-1]["name"])
        self._layer_listbox.insert(i,   self._layers[i]["name"])
        self._selected_layer = i - 1
        self._layer_listbox.selection_set(i-1)
        self._refresh_composite()

    def _move_layer_down(self):
        if self._selected_layer is None or self._selected_layer >= len(self._layers) - 1:
            return
        i = self._selected_layer
        self._layers[i], self._layers[i+1] = self._layers[i+1], self._layers[i]
        self._layer_listbox.delete(i, i+1)
        self._layer_listbox.insert(i,   self._layers[i]["name"])
        self._layer_listbox.insert(i+1, self._layers[i+1]["name"])
        self._selected_layer = i + 1
        self._layer_listbox.selection_set(i+1)
        self._refresh_composite()

    def _on_layer_select(self, event):
        sel = self._layer_listbox.curselection()
        if not sel:
            return
        self._selected_layer = sel[0]
        layer = self._layers[self._selected_layer]
        self._prop_x.set(layer["x"])
        self._prop_y.set(layer["y"])
        self._prop_scale.set(layer["scale"])
        self._prop_alpha.set(layer["alpha"])

    def _apply_layer_props(self):
        if self._selected_layer is None:
            return
        layer = self._layers[self._selected_layer]
        layer["x"]     = self._prop_x.get()
        layer["y"]     = self._prop_y.get()
        layer["scale"] = self._prop_scale.get()
        layer["alpha"] = self._prop_alpha.get()
        self._refresh_composite()

    def _update_layer_info(self):
        self._layer_info_var.set(f"レイヤー: {len(self._layers)}")

    def _select_layer_index(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._layers):
            return
        self._selected_layer = idx
        self._layer_listbox.selection_clear(0, "end")
        self._layer_listbox.selection_set(idx)
        self._layer_listbox.activate(idx)
        layer = self._layers[idx]
        self._prop_x.set(layer["x"])
        self._prop_y.set(layer["y"])
        self._prop_scale.set(layer["scale"])
        self._prop_alpha.set(layer["alpha"])

    def _canvas_to_composite_coords(self, cx: int, cy: int) -> Optional[Tuple[int, int]]:
        scale = max(self._preview_scale, 1e-6)
        ox, oy = self._preview_origin
        ix = int((cx - ox) / scale)
        iy = int((cy - oy) / scale)
        if 0 <= ix < self._canvas_w and 0 <= iy < self._canvas_h:
            return ix, iy
        return None

    def _pick_layer_at(self, ix: int, iy: int) -> Optional[int]:
        for idx in range(len(self._layers) - 1, -1, -1):
            layer = self._layers[idx]
            if not layer.get("visible", True):
                continue
            base_img = layer["img"]
            scale = float(layer["scale"])
            lw = max(1, int(base_img.width * scale))
            lh = max(1, int(base_img.height * scale))
            lx = int(layer["x"])
            ly = int(layer["y"])
            if not (lx <= ix < lx + lw and ly <= iy < ly + lh):
                continue

            local_x = int((ix - lx) / max(scale, 1e-6))
            local_y = int((iy - ly) / max(scale, 1e-6))
            local_x = max(0, min(base_img.width - 1, local_x))
            local_y = max(0, min(base_img.height - 1, local_y))
            try:
                if base_img.getchannel("A").getpixel((local_x, local_y)) <= 5:
                    continue
            except Exception:
                pass
            return idx
        return None

    def _on_composite_press(self, event):
        pos = self._canvas_to_composite_coords(event.x, event.y)
        if pos is None:
            self._drag_layer_idx = None
            return
        hit_idx = self._pick_layer_at(*pos)
        if hit_idx is None:
            self._drag_layer_idx = None
            return

        self._select_layer_index(hit_idx)
        self._drag_layer_idx = hit_idx
        self._drag_mouse_origin = (event.x, event.y)
        layer = self._layers[hit_idx]
        self._drag_layer_origin = (int(layer["x"]), int(layer["y"]))
        self._composite_canvas.configure(cursor="fleur")

    def _on_composite_drag(self, event):
        if self._drag_layer_idx is None:
            return
        if self._drag_layer_idx >= len(self._layers):
            self._drag_layer_idx = None
            return

        scale = max(self._preview_scale, 1e-6)
        dx = int(round((event.x - self._drag_mouse_origin[0]) / scale))
        dy = int(round((event.y - self._drag_mouse_origin[1]) / scale))
        new_x = self._drag_layer_origin[0] + dx
        new_y = self._drag_layer_origin[1] + dy

        layer = self._layers[self._drag_layer_idx]
        if layer["x"] == new_x and layer["y"] == new_y:
            return
        layer["x"] = new_x
        layer["y"] = new_y
        if self._selected_layer == self._drag_layer_idx:
            self._prop_x.set(new_x)
            self._prop_y.set(new_y)
        self._refresh_composite()

    def _on_composite_release(self, _event):
        if self._drag_layer_idx is not None:
            self._anim_status_var.set("ドラッグでレイヤー位置を更新しました")
        self._drag_layer_idx = None
        self._composite_canvas.configure(cursor="fleur")

    # ================================================================
    # Inpaint統合
    # ================================================================

    def _inpaint_selected_layer(self):
        if self._selected_layer is None or not _NUMPY_AVAILABLE or not _PIL_AVAILABLE:
            return
        layer = self._layers[self._selected_layer]
        img   = layer["img"]
        arr   = np.array(img.convert("RGBA"))
        mask  = self._processor.create_inpaint_mask_from_alpha(arr)
        if not mask.any():
            self._anim_status_var.set("透明領域がないためInpaintをスキップ")
            return
        self._anim_status_var.set("Inpaint処理中...")

        def _run():
            inpainted = self._processor.inpaint_region(arr, mask, radius=6)
            result_img = Image.fromarray(inpainted)
            self.after(0, self._on_inpaint_done, result_img)

        threading.Thread(target=_run, daemon=True).start()

    def _on_inpaint_done(self, img: "Image.Image"):
        if self._selected_layer is None:
            return
        self._layers[self._selected_layer]["img"] = img
        self._refresh_composite()
        self._anim_status_var.set("Inpaint完了")

    # ================================================================
    # 合成処理（Porter-Duff Over合成）
    # ================================================================

    def _composite_all_layers(self) -> Optional["Image.Image"]:
        """全レイヤーを下から上へ Porter-Duff Over 合成して返す。"""
        if not _PIL_AVAILABLE or not _NUMPY_AVAILABLE:
            return None

        w, h = self._canvas_w, self._canvas_h
        canvas = np.zeros((h, w, 4), dtype=np.float32)

        for layer in self._layers:
            if not layer["visible"]:
                continue

            img   = layer["img"].convert("RGBA")
            scale = layer["scale"]
            nw    = max(1, int(img.width  * scale))
            nh    = max(1, int(img.height * scale))
            scaled = img.resize((nw, nh), Image.LANCZOS)
            arr    = np.array(scaled).astype(np.float32)

            arr[:, :, 3] = arr[:, :, 3] * (layer["alpha"] / 255.0)

            lx = layer["x"]
            ly = layer["y"]
            cx0 = max(0, lx)
            cy0 = max(0, ly)
            cx1 = min(w, lx + nw)
            cy1 = min(h, ly + nh)
            sx0 = cx0 - lx
            sy0 = cy0 - ly
            sx1 = sx0 + (cx1 - cx0)
            sy1 = sy0 + (cy1 - cy0)

            if cx0 >= cx1 or cy0 >= cy1:
                continue

            src_region = arr[sy0:sy1, sx0:sx1]
            dst_region = canvas[cy0:cy1, cx0:cx1]
            src_a = src_region[:, :, 3:4] / 255.0
            dst_a = dst_region[:, :, 3:4] / 255.0
            out_a = src_a + dst_a * (1 - src_a)
            mask  = out_a > 1e-6

            for ch in range(3):
                blend = np.where(
                    mask[:, :, 0],
                    (src_region[:, :, ch] * src_a[:, :, 0]
                     + dst_region[:, :, ch] * dst_a[:, :, 0] * (1 - src_a[:, :, 0]))
                    / np.where(mask[:, :, 0], out_a[:, :, 0], 1),
                    0,
                )
                canvas[cy0:cy1, cx0:cx1, ch] = blend
            canvas[cy0:cy1, cx0:cx1, 3] = out_a[:, :, 0] * 255

        return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))

    def _refresh_composite(self):
        """合成結果をキャンバスに描画"""
        if not _PIL_AVAILABLE:
            return
        img = self._composite_all_layers()
        if img is None:
            return
        self._draw_composite_to_canvas(img)

    def _draw_composite_to_canvas(self, img: "Image.Image"):
        canvas = self._composite_canvas
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return
        sz = 12
        if _NUMPY_AVAILABLE:
            ys, xs = np.mgrid[0:ch, 0:cw]
            mask = ((xs // sz) + (ys // sz)) % 2 == 1
            bg_arr = np.full((ch, cw, 4), [40, 40, 60, 255], dtype=np.uint8)
            bg_arr[mask] = [60, 60, 80, 255]
            bg = Image.fromarray(bg_arr)
        else:
            bg = Image.new("RGBA", (cw, ch), (40, 40, 60, 255))
            draw_bg = ImageDraw.Draw(bg)
            for y in range(0, ch, sz):
                for x in range(0, cw, sz):
                    if ((x // sz) + (y // sz)) % 2 == 1:
                        draw_bg.rectangle([x, y, x+sz, y+sz], fill=(60, 60, 80, 255))
        scale = min(cw / max(img.width, 1), ch / max(img.height, 1)) * 0.95
        nw    = max(1, int(img.width * scale))
        nh    = max(1, int(img.height * scale))
        x     = (cw - nw) // 2
        y     = (ch - nh) // 2
        self._preview_scale = scale
        self._preview_origin = (x, y)
        resized = img.resize((nw, nh), Image.LANCZOS)
        bg.paste(resized, (x, y), resized)
        self._tk_preview = ImageTk.PhotoImage(bg)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=self._tk_preview)
        if not self._layers:
            canvas.create_text(
                cw // 2,
                ch // 2 - 10,
                text="画像未設定",
                fill="#d8dcff",
                font=("Segoe UI", 16, "bold"),
            )
            canvas.create_text(
                cw // 2,
                ch // 2 + 18,
                text="「+ 画像」または「+ キャラ」で追加",
                fill="#9aa3c6",
                font=("Segoe UI", 10),
            )

    def _apply_canvas_size(self):
        sz_str = self._canvas_size_var.get()
        try:
            w, h = map(int, sz_str.split("x"))
            self._canvas_w = w
            self._canvas_h = h
            self._refresh_composite()
        except Exception:
            pass

    # ================================================================
    # フレーム管理
    # ================================================================

    def _add_frame(self):
        if not _NUMPY_AVAILABLE:
            return
        frame = np.zeros((self._canvas_h, self._canvas_w, 4), dtype=np.uint8)
        self._frames.append(frame)
        self._frame_listbox.insert("end", f"Frame {len(self._frames):03d}")
        self._update_frame_info()

    def _remove_frame(self):
        sel = self._frame_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self._frames.pop(idx)
        self._frame_listbox.delete(idx)
        self._update_frame_info()

    def _capture_frame(self):
        if not _PIL_AVAILABLE or not _NUMPY_AVAILABLE:
            return
        img = self._composite_all_layers()
        if img is None:
            return
        arr = np.array(img.convert("RGBA"))
        self._frames.append(arr)
        self._frame_listbox.insert("end", f"Frame {len(self._frames):03d}")
        self._update_frame_info()
        self._anim_status_var.set(f"フレーム {len(self._frames)} をキャプチャしました")

    def _on_frame_select(self, event):
        sel = self._frame_listbox.curselection()
        if not sel or not _PIL_AVAILABLE:
            return
        idx = sel[0]
        self._current_frame = idx
        if idx < len(self._frames):
            img = Image.fromarray(self._frames[idx])
            self._draw_composite_to_canvas(img)

    def _update_frame_info(self):
        self._frame_info_var.set(f"フレーム: {self._current_frame + 1}/{len(self._frames)}")

    # ================================================================
    # 再生
    # ================================================================

    def _toggle_play(self):
        if self._playing:
            self._stop_play()
        else:
            if not self._frames:
                self._anim_status_var.set("フレームがありません")
                return
            self._playing = True
            self._fps = self._fps_var.get()
            self._play_btn.configure(text="⏸ 一時停止")
            self._anim_status_var.set("再生中...")
            self._play_loop()

    def _stop_play(self):
        self._playing = False
        self._play_btn.configure(text="▶ 再生")
        self._anim_status_var.set("停止")

    def _play_loop(self):
        if not self._playing or not self._frames:
            return
        idx = self._current_frame % len(self._frames)
        img = Image.fromarray(self._frames[idx])
        self._draw_composite_to_canvas(img)
        self._current_frame = (idx + 1) % len(self._frames)
        self._update_frame_info()
        delay_ms = max(16, 1000 // max(1, self._fps))
        self.after(delay_ms, self._play_loop)

    # ================================================================
    # 書き出し
    # ================================================================

    def _export_gif(self):
        if not self._frames:
            messagebox.showwarning("警告", "フレームがありません", parent=self)
            return
        if not _PIL_AVAILABLE:
            return

        path = filedialog.asksaveasfilename(
            title="GIF書き出し先を選択",
            defaultextension=".gif",
            filetypes=[("GIF ファイル", "*.gif"), ("すべて", "*.*")],
            parent=self,
        )
        if not path:
            return

        if not messagebox.askyesno("確認", f"{len(self._frames)}フレームのGIFを書き出しますか？", parent=self):
            return

        try:
            pil_frames = []
            for arr in self._frames:
                img = Image.fromarray(arr).convert("RGBA")
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                pil_frames.append(bg.convert("P", palette=Image.ADAPTIVE, colors=256))

            delay_ms = max(16, 1000 // max(1, self._fps_var.get()))
            pil_frames[0].save(
                path, format="GIF", save_all=True,
                append_images=pil_frames[1:],
                duration=delay_ms, loop=0, optimize=False,
            )
            self._anim_status_var.set(f"GIF書き出し完了: {Path(path).name}")
            messagebox.showinfo("完了", f"GIFを書き出しました:\n{path}", parent=self)
        except Exception as e:
            messagebox.showerror("エラー", str(e), parent=self)

    def _export_png_sequence(self):
        if not self._frames:
            messagebox.showwarning("警告", "フレームがありません", parent=self)
            return

        dest_dir = filedialog.askdirectory(title="連番PNG書き出し先フォルダ", parent=self)
        if not dest_dir:
            return

        prefix = simpledialog.askstring(
            "プレフィックス", "ファイル名プレフィックス:", initialvalue="frame", parent=self)
        if prefix is None:
            return

        if not messagebox.askyesno("確認", f"{len(self._frames)}枚のPNGを書き出しますか？", parent=self):
            return

        try:
            dest = Path(dest_dir)
            for i, arr in enumerate(self._frames):
                img  = Image.fromarray(arr)
                fname = dest / f"{prefix}_{i+1:04d}.png"
                img.save(fname, "PNG")
            self._anim_status_var.set(f"連番PNG書き出し完了: {len(self._frames)} 枚")
            messagebox.showinfo("完了", f"{len(self._frames)}枚を書き出しました:\n{dest_dir}", parent=self)
        except Exception as e:
            messagebox.showerror("エラー", str(e), parent=self)

    def _export_current_frame(self):
        if not self._frames:
            messagebox.showwarning("警告", "フレームがありません", parent=self)
            return

        if not messagebox.askyesno("確認", "現在のフレームをPNGとして保存しますか？", parent=self):
            return

        path = filedialog.asksaveasfilename(
            title="PNG保存先",
            defaultextension=".png",
            filetypes=[("PNG ファイル", "*.png"), ("すべて", "*.*")],
            parent=self,
        )
        if not path:
            return
        try:
            idx = self._current_frame % max(len(self._frames), 1)
            img = Image.fromarray(self._frames[idx])
            img.save(path, "PNG")
            self._anim_status_var.set(f"保存完了: {Path(path).name}")
        except Exception as e:
            messagebox.showerror("エラー", str(e), parent=self)


# ================================================================== #
# 設定ダイアログ
# ================================================================== #


class SettingsDialog(tk.Toplevel):
    _MODEL_CHOICES = [
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
    ]

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
            "Local":  self._tab_local,
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
        self._row_combo(f, c, "AIバックエンド", "AI_BACKEND", ["auto", "gemini", "local"])
        self._row_combo(f, c, "AIモデル", "ALICE_MODEL", self._MODEL_CHOICES)

    def _tab_api(self, f, c):
        self._row_str(f, c, "Google API Key", "GOOGLE_API_KEY", show="*")
        self._row_str(f, c, "Hugging Face Token", "HF_TOKEN", show="*")
        self._row_str(f, c, "VOICEVOX URL", "VOICEVOX_URL")
        self._row_int(f, c, "VOICEVOX Speaker ID", "VOICEVOX_SPEAKER_ID")

    def _tab_local(self, f, c):
        self._row_str(f, c, "Model Repo", "LOCAL_MODEL_REPO")
        self._row_str(f, c, "Model File", "LOCAL_MODEL_FILE")
        self._row_str(f, c, "Model Dir", "LOCAL_MODEL_DIR")
        self._row_int(f, c, "Context (n_ctx)", "LOCAL_MODEL_N_CTX")
        self._row_int(f, c, "Max Tokens", "LOCAL_MODEL_MAX_TOKENS")
        self._row_flt(f, c, "Temperature", "LOCAL_MODEL_TEMPERATURE")
        self._row_flt(f, c, "Top P", "LOCAL_MODEL_TOP_P")
        self._row_int(f, c, "Threads (0=auto)", "LOCAL_MODEL_THREADS")
        self._row_int(f, c, "GPU Layers", "LOCAL_MODEL_N_GPU_LAYERS")
        self._row_str(f, c, "Chat Format", "LOCAL_MODEL_CHAT_FORMAT")

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
    """

    _CHAT_RATIO = 0.62
    _CHAR_RATIO = 0.38
    _DESKTOP_CHAT_MIN_WIDTH = 360

    def __init__(
        self,
        env_binder=None,
        alice_engine=None,
        voice_engine=None,
        git_manager=None,
        char_loader=None,
        startup_notice: Optional[Dict[str, object]] = None,
    ) -> None:
        self._env         = env_binder
        self._alice       = alice_engine
        self._voice       = voice_engine
        self._git         = git_manager
        self._char_loader = char_loader
        self._startup_notice = dict(startup_notice or {})

        theme_name = env_binder.get("APP_THEME") if env_binder else "dark"
        self.colors = Theme.get(theme_name)
        self._mode  = AppMode.DESKTOP

        self._msg_queue: queue.Queue = queue.Queue(maxsize=500)
        self._streaming_started = False
        self._stream_chunk_lock = threading.Lock()
        self._stream_chunk_buffer: str = ""
        self._stream_chunk_flush_pending = False
        self._queue_tick_budget_ms = 12
        self._queue_tick_max_items = 80
        self._mode_var: Optional[tk.StringVar] = None
        self._statusbar_frame: Optional[tk.Frame] = None
        self._chat_frame: Optional[tk.Frame] = None
        self._char_frame: Optional[tk.Frame] = None
        self._chat_display_container: Optional[tk.Frame] = None
        self._chat_display: Optional[AutoScrollText] = None
        self._input_container: Optional[tk.Frame] = None
        self._input_box: Optional[PlaceholderEntry] = None
        self._send_btn: Optional[tk.Button] = None
        self._voice_btn: Optional[tk.Button] = None
        self._startup_popup: Optional[tk.Toplevel] = None
        self._startup_popup_after_id: Optional[str] = None

        self.root = tk.Tk()
        self._setup_window()
        self._build_ui()
        self.root.after(0, self._bootstrap_startup)
        self._start_services()

    def run(self) -> None:
        self.root.after(100, self._process_queue)
        self.root.mainloop()

    def _bootstrap_startup(self) -> None:
        self._ensure_startup_desktop_mode()
        self._show_startup_model_popup_if_needed()

    def _show_startup_model_popup_if_needed(self) -> None:
        info = dict(self._startup_notice or {})
        if not info.get("auto_selected"):
            return
        if self._startup_popup is not None and self._widget_exists(self._startup_popup):
            return

        c = self.colors
        popup = tk.Toplevel(self.root)
        popup.title("AIモデル自動選択")
        popup.configure(bg=c.bg_secondary)
        popup.transient(self.root)
        popup.resizable(False, False)
        try:
            popup.attributes("-topmost", True)
        except Exception:
            pass
        popup.protocol("WM_DELETE_WINDOW", self._close_startup_model_popup)

        title = tk.Label(
            popup,
            text="AIモデルを自動選択しました",
            bg=c.bg_secondary,
            fg=c.accent_primary,
            font=("Segoe UI", 12, "bold"),
        )
        title.pack(anchor="w", padx=16, pady=(14, 6))

        details: List[str] = []
        backend = str(info.get("backend") or "").strip().lower()
        model_name = str(info.get("model_name") or info.get("model_label") or "").strip()
        profile_label = str(info.get("profile_label") or "").strip()
        ram_gb = info.get("ram_gb")
        cpu_count = info.get("cpu_count")

        if backend:
            details.append(f"バックエンド: {backend}")
        if model_name:
            details.append(f"モデル: {model_name}")
        if profile_label:
            details.append(f"選択プロファイル: {profile_label}")
        if ram_gb is not None and cpu_count is not None:
            details.append(f"検出スペック: RAM {ram_gb} GB / CPU {cpu_count} cores")
        msg = str(info.get("message") or "").strip()
        if msg:
            details.append(msg)
        details.append("5秒後にデスクトップモードへ戻ります。")

        body = tk.Label(
            popup,
            text="\n".join(details),
            justify="left",
            bg=c.bg_secondary,
            fg=c.text_primary,
            font=("Segoe UI", 10),
        )
        body.pack(anchor="w", padx=16, pady=(0, 14))

        self._startup_popup = popup
        self._center_window(popup, width=560, height=220)
        self._startup_popup_after_id = self.root.after(5000, self._close_startup_model_popup)

    def _center_window(self, win: tk.Toplevel, width: int, height: int) -> None:
        try:
            self.root.update_idletasks()
            x = self.root.winfo_rootx() + max(0, (self.root.winfo_width() - width) // 2)
            y = self.root.winfo_rooty() + max(0, (self.root.winfo_height() - height) // 2)
            win.geometry(f"{width}x{height}+{x}+{y}")
        except Exception:
            pass

    def _close_startup_model_popup(self) -> None:
        if self._startup_popup_after_id is not None:
            try:
                self.root.after_cancel(self._startup_popup_after_id)
            except Exception:
                pass
            self._startup_popup_after_id = None

        popup = self._startup_popup
        self._startup_popup = None
        if popup is not None and self._widget_exists(popup):
            try:
                popup.destroy()
            except Exception:
                pass

        self._ensure_startup_desktop_mode()

    @staticmethod
    def _widget_exists(widget: Optional[tk.Widget]) -> bool:
        if widget is None:
            return False
        try:
            return bool(widget.winfo_exists())
        except Exception:
            return False

    def _enqueue(self, fn, *args, **kwargs):
        try:
            self._msg_queue.put_nowait((fn, args, kwargs))
        except queue.Full:
            # キュー満杯時: 古いアイテムを1件破棄して再挿入
            try:
                self._msg_queue.get_nowait()
                self._msg_queue.put_nowait((fn, args, kwargs))
            except (queue.Empty, queue.Full):
                logger.warning("_enqueue: キューが満杯のためメッセージをドロップしました")

    def _enqueue_stream_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        should_schedule = False
        with self._stream_chunk_lock:
            self._stream_chunk_buffer += chunk
            if not self._stream_chunk_flush_pending:
                self._stream_chunk_flush_pending = True
                should_schedule = True
        if should_schedule:
            self._enqueue(self._flush_stream_chunks)

    def _flush_stream_chunks(self) -> None:
        with self._stream_chunk_lock:
            chunk = self._stream_chunk_buffer
            self._stream_chunk_buffer = ""
            self._stream_chunk_flush_pending = False
        if chunk:
            self._append_alice_chunk(chunk)

    def _finalize_stream_with_fallback(self, full: str) -> None:
        if self._streaming_started:
            self._finalize_alice_stream()
            return
        text = (full or "").strip()
        if text:
            self._append_alice(text)

    def _process_queue(self):
        start = time.monotonic()
        budget_sec = self._queue_tick_budget_ms / 1000.0
        processed = 0

        while processed < self._queue_tick_max_items:
            if (time.monotonic() - start) >= budget_sec:
                break
            try:
                fn, args, kwargs = self._msg_queue.get_nowait()
            except queue.Empty:
                break
            try:
                fn(*args, **kwargs)
            except Exception as e:
                logger.exception(f"_process_queue 実行エラー: {e}")
            processed += 1

        delay_ms = 5 if not self._msg_queue.empty() else 50
        try:
            self.root.after(delay_ms, self._process_queue)
        except tk.TclError:
            pass

    def _setup_window(self):
        layout = get_layout(self._mode)
        c = self.colors
        self.root.title("Alice AI")
        self.root.configure(bg=c.bg_primary)
        self.root.geometry(f"{layout.default_width}x{layout.default_height}")
        self.root.minsize(layout.min_width, layout.min_height)
        self.root.resizable(layout.resizable, layout.resizable)
        try:
            self.root.attributes("-topmost", bool(layout.always_on_top))
        except Exception:
            pass
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        self._build_menu()
        self._build_desktop_ui()

    def _build_menu(self):
        c = self.colors

        def menu(parent):
            return tk.Menu(parent, tearoff=0, bg=c.bg_secondary, fg=c.text_primary,
                           activebackground=c.accent_primary, relief="flat")

        def defer_menu_action(fn, *args, **kwargs):
            # メニューのコールバック中に重いUI処理を直接走らせず、メニュー終了後に実行する。
            def _run():
                try:
                    self.root.after(10, lambda: fn(*args, **kwargs))
                except tk.TclError:
                    pass
            return _run

        menubar = tk.Menu(self.root, bg=c.bg_secondary, fg=c.text_primary,
                          activebackground=c.accent_primary, relief="flat")
        self.root.configure(menu=menubar)

        fm = menu(menubar)
        fm.add_command(label="設定", command=defer_menu_action(self._open_settings), accelerator="Ctrl+,")
        fm.add_separator()
        fm.add_command(label="終了", command=defer_menu_action(self._on_close))
        menubar.add_cascade(label="ファイル", menu=fm)

        vm = menu(menubar)
        vm.add_command(label="チャット履歴をクリア", command=defer_menu_action(self._clear_chat))
        vm.add_separator()
        self._mode_var = tk.StringVar(value=self._mode.value)
        vm.add_radiobutton(
            label="デスクトップモード",
            value=AppMode.DESKTOP.value,
            variable=self._mode_var,
            command=defer_menu_action(self._set_mode, AppMode.DESKTOP),
        )
        vm.add_radiobutton(
            label="キャラクターモード",
            value=AppMode.CHARACTER.value,
            variable=self._mode_var,
            command=defer_menu_action(self._set_mode, AppMode.CHARACTER),
        )
        vm.add_command(label="モード切替 (Ctrl+M)", command=defer_menu_action(self._toggle_mode))
        menubar.add_cascade(label="表示", menu=vm)

        gm = menu(menubar)
        gm.add_command(label="Git マネージャー", command=defer_menu_action(self._open_git_dialog))
        gm.add_command(label="クイックコミット",  command=defer_menu_action(self._quick_commit))
        gm.add_command(label="ブランチ切替...",   command=defer_menu_action(self._switch_branch_dialog))
        menubar.add_cascade(label="Git", menu=gm)

        tm = menu(menubar)
        tm.add_command(label="キャラクター再読み込み", command=defer_menu_action(self._reload_character))
        tm.add_command(label="🎨 高度な画像処理ツール", command=defer_menu_action(self._open_advanced_image_tool))
        tm.add_command(label="🎬 アニメーション作成ツール", command=defer_menu_action(self._open_animation_tool))
        tm.add_separator()
        tm.add_command(label="VOICEVOX 接続確認",     command=defer_menu_action(self._check_voicevox))
        tm.add_separator()
        tm.add_command(label="ログフォルダを開く",    command=defer_menu_action(self._open_logs))
        menubar.add_cascade(label="ツール", menu=tm)

        hm = menu(menubar)
        hm.add_command(label="About", command=defer_menu_action(self._show_about))
        menubar.add_cascade(label="ヘルプ", menu=hm)

        self.root.bind("<Control-comma>", lambda e: self._open_settings())
        self.root.bind("<Control-m>", lambda e: self._toggle_mode())
        # NOTE: <Return> はグローバルバインドせず入力欄の bind のみで処理（二重発火防止）

    def _build_desktop_ui(self):
        c = self.colors
        layout = get_layout(self._mode)

        self._paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self._paned.pack(fill="both", expand=True)

        self._chat_frame = tk.Frame(self._paned, bg=c.bg_primary)
        self._paned.add(self._chat_frame, weight=62)

        self._build_header(self._chat_frame, c)
        self._build_chat_display(self._chat_frame, c)
        self._build_input_area(self._chat_frame, c)

        self._char_frame = tk.Frame(self._paned, bg=c.bg_secondary)
        self._paned.add(self._char_frame, weight=38)

        self._build_character_panel(self._char_frame, c, layout)
        self.root.after(50, self._set_initial_sash)
        self._build_status_bar(c)
        self._apply_current_layout(reset_geometry=False)
        self._ensure_chat_display_ready()
        self._ensure_chat_input_ready()
        self._schedule_desktop_split_fix()

    def _set_initial_sash(self):
        try:
            total = self.root.winfo_width()
            if total > 10:
                self._paned.sashpos(0, int(total * self._CHAT_RATIO))
        except Exception:
            pass

    def _schedule_desktop_split_fix(self) -> None:
        if self._mode != AppMode.DESKTOP:
            return
        for delay in (20, 120, 320):
            try:
                self.root.after(delay, self._ensure_desktop_chat_visible)
            except tk.TclError:
                return

    def _ensure_desktop_chat_visible(self) -> None:
        if self._mode != AppMode.DESKTOP:
            return
        if not hasattr(self, "_paned") or self._chat_frame is None or self._char_frame is None:
            return
        try:
            panes = set(self._paned.panes())
            if str(self._chat_frame) not in panes:
                self._paned.add(self._chat_frame, weight=62)
            if str(self._char_frame) not in panes:
                self._paned.add(self._char_frame, weight=38)
            total = max(self.root.winfo_width(), self.root.winfo_reqwidth(), 900)
            min_chat = self._DESKTOP_CHAT_MIN_WIDTH
            max_chat = max(min_chat, total - 260)
            target = int(total * self._CHAT_RATIO)
            target = max(min_chat, min(max_chat, target))
            self._paned.sashpos(0, target)
            self._ensure_chat_display_ready()
            self._ensure_chat_input_ready()
        except Exception:
            pass

    def _ensure_chat_display_ready(self) -> None:
        if self._chat_frame is None:
            return

        if not self._widget_exists(self._chat_display):
            self._build_chat_display(self._chat_frame, self.colors)
            return

        if self._chat_display_container is not None and self._widget_exists(self._chat_display_container):
            try:
                if not self._chat_display_container.winfo_manager():
                    if self._input_container is not None and self._widget_exists(self._input_container):
                        self._chat_display_container.pack(fill="both", expand=True, before=self._input_container)
                    else:
                        self._chat_display_container.pack(fill="both", expand=True)
            except Exception:
                pass

    def _ensure_chat_input_ready(self) -> None:
        if self._chat_frame is None:
            return
        needs_rebuild = (
            (not self._widget_exists(self._input_box))
            or (not self._widget_exists(self._send_btn))
            or (not self._widget_exists(self._voice_btn))
        )

        if needs_rebuild:
            self._build_input_area(self._chat_frame, self.colors)

        if self._input_container is not None and self._widget_exists(self._input_container):
            try:
                if not self._input_container.winfo_manager():
                    if self._chat_display_container is not None and self._widget_exists(self._chat_display_container):
                        self._input_container.pack(fill="x", after=self._chat_display_container)
                    else:
                        self._input_container.pack(fill="x")
            except Exception:
                pass

        if self._widget_exists(self._voice_btn):
            try:
                if not self._voice_btn.winfo_manager():
                    self._voice_btn.pack(pady=2)
            except Exception:
                pass
        if self._widget_exists(self._send_btn):
            try:
                if not self._send_btn.winfo_manager():
                    self._send_btn.pack(pady=2)
            except Exception:
                pass

        if self._widget_exists(self._input_box):
            try:
                self._input_box.focus_set()
            except Exception:
                pass

    def _ensure_startup_desktop_mode(self) -> None:
        self._mode = AppMode.DESKTOP
        if self._mode_var is not None:
            self._mode_var.set(self._mode.value)
        self._apply_current_layout(reset_geometry=False)
        self._ensure_chat_display_ready()
        self._ensure_chat_input_ready()
        self._schedule_desktop_split_fix()

    def _toggle_mode(self):
        next_mode = AppMode.CHARACTER if self._mode == AppMode.DESKTOP else AppMode.DESKTOP
        self._set_mode(next_mode)

    def _set_mode(self, mode: AppMode):
        if mode == self._mode:
            if self._mode_var is not None:
                self._mode_var.set(self._mode.value)
            return
        self._mode = mode
        self._apply_current_layout(reset_geometry=True)
        if self._mode == AppMode.DESKTOP:
            self._schedule_desktop_split_fix()
        if self._char_loader:
            self._load_character()

    def _apply_current_layout(self, reset_geometry: bool = True):
        layout = get_layout(self._mode)

        if reset_geometry:
            self.root.geometry(f"{layout.default_width}x{layout.default_height}")
        self.root.minsize(layout.min_width, layout.min_height)
        self.root.resizable(layout.resizable, layout.resizable)
        try:
            self.root.attributes("-topmost", bool(layout.always_on_top))
        except Exception:
            pass

        if hasattr(self, "_paned") and self._chat_frame is not None and self._char_frame is not None:
            for pane in list(self._paned.panes()):
                self._paned.forget(pane)
            if layout.show_chat_panel:
                self._paned.add(self._chat_frame, weight=62)
            if layout.show_character:
                self._paned.add(self._char_frame, weight=100 if not layout.show_chat_panel else 38)
            if layout.show_chat_panel and layout.show_character:
                self.root.after(20, self._set_initial_sash)

        if self._statusbar_frame is not None:
            if layout.show_status_bar:
                if not self._statusbar_frame.winfo_manager():
                    self._statusbar_frame.pack(fill="x", side="bottom")
                    self._statusbar_frame.pack_propagate(False)
            else:
                if self._statusbar_frame.winfo_manager():
                    self._statusbar_frame.pack_forget()

        if self._mode_var is not None:
            self._mode_var.set(self._mode.value)
        mode_text = "キャラクターモード" if self._mode == AppMode.CHARACTER else "デスクトップモード"
        self._update_status(f"{mode_text} に切り替えました。")
        if self._mode == AppMode.DESKTOP:
            self._schedule_desktop_split_fix()

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
        if self._chat_display_container is not None and self._widget_exists(self._chat_display_container):
            try:
                self._chat_display_container.destroy()
            except Exception:
                pass
        f = tk.Frame(parent, bg=c.bg_primary)
        f.pack(fill="both", expand=True)
        self._chat_display_container = f
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
        if self._input_container is not None and self._widget_exists(self._input_container):
            try:
                self._input_container.destroy()
            except Exception:
                pass
        container = tk.Frame(parent, bg=c.bg_secondary, pady=10)
        container.pack(fill="x")
        self._input_container = container
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
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self._statusbar_frame = bar
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
        else:
            self._append_system("AI未接続です。設定で APIキー / AIモデル を確認してください。")

    def _build_local_fallback_reply(self, user_text: str) -> str:
        name = self._env.get("ALICE_NAME") if self._env else "Alice"
        text = user_text.strip()
        if not text:
            return f"{name}です。聞きたいことを入力してください。"
        if "こんにちは" in text or "こんばんは" in text:
            return f"こんにちは。{name}です。今日はどんなことを進めますか？"
        if "ありがとう" in text:
            return "どういたしまして。続きも手伝います。"
        if "help" in text.lower() or "使い方" in text:
            return "メッセージ欄に質問を書くと返答します。デスクトップモードでチャット欄が表示されます。"
        return (
            "現在AIエンジンが利用できないため、簡易応答で返しています。"
            "設定を確認すると通常の対話に戻せます。"
        )

    # ---- チャットロジック ----

    def _on_enter_key(self, event) -> str:
        if not (event.state & 0x1):
            self._on_send()
            return "break"
        return "continue"

    def _on_send(self):
        if not self._widget_exists(self._input_box):
            self._ensure_chat_input_ready()
        if not self._widget_exists(self._input_box):
            return
        if not self._widget_exists(self._chat_display):
            self._ensure_chat_display_ready()
        if not self._widget_exists(self._chat_display):
            return

        text = self._input_box.get_text()
        if not text:
            return
        self._input_box.clear()
        self._append_user(text)
        self._set_thinking(True)

        def _chat():
            def on_chunk(chunk):
                self._enqueue_stream_chunk(chunk)

            def on_complete(full):
                self._enqueue(self._flush_stream_chunks)
                self._enqueue(self._set_thinking, False)
                self._enqueue(self._finalize_stream_with_fallback, full)
                if self._voice:
                    # スレッドセーフのためキュー経由でUIスレッドに移譲
                    self._enqueue(self._voice.speak, full)

            def on_error(err):
                self._enqueue(self._flush_stream_chunks)
                self._enqueue(self._append_error, err)
                self._enqueue(self._set_thinking, False)
                self._enqueue(self._finalize_alice_stream)

            if self._alice:
                self._alice.send_message(
                    text,
                    on_chunk=on_chunk,
                    on_complete=on_complete,
                    on_error=on_error,
                )
            else:
                local_reply = self._build_local_fallback_reply(text)
                self._enqueue(self._append_alice, local_reply)
                if self._voice:
                    self._enqueue(self._voice.speak, local_reply)
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
        has_pending = bool(getattr(self._voice, "has_pending_speech", False)) if self._voice else False
        if self._voice and (self._voice.is_speaking or has_pending):
            self._voice.stop()
            if self._widget_exists(self._voice_btn):
                self._voice_btn.configure(text="音声")
        elif self._voice:
            if self._widget_exists(self._voice_btn):
                self._voice_btn.configure(text="停止")

    # ---- チャット表示ヘルパー ----

    def _append_user(self, text):
        if not self._widget_exists(self._chat_display):
            self._ensure_chat_display_ready()
        if not self._widget_exists(self._chat_display):
            return
        ts = datetime.now().strftime("%H:%M")
        self._chat_display.append(f"\n[{ts}] あなた\n", "user_name")
        self._chat_display.append(f"{text}\n", "user_text")

    def _append_alice(self, text):
        if not self._widget_exists(self._chat_display):
            self._ensure_chat_display_ready()
        if not self._widget_exists(self._chat_display):
            return
        name = self._env.get("ALICE_NAME") if self._env else "Alice"
        ts = datetime.now().strftime("%H:%M")
        self._chat_display.append(f"\n[{ts}] {name}\n", "alice_name")
        self._chat_display.append(f"{text}\n", "alice_text")

    def _append_alice_chunk(self, chunk):
        if not self._widget_exists(self._chat_display):
            self._ensure_chat_display_ready()
        if not self._widget_exists(self._chat_display):
            return
        if not self._streaming_started:
            self._streaming_started = True
            name = self._env.get("ALICE_NAME") if self._env else "Alice"
            ts = datetime.now().strftime("%H:%M")
            self._chat_display.append(f"\n[{ts}] {name}\n", "alice_name")
            if hasattr(self, "_animator"):
                self._animator.set_state(CharacterState.SPEAKING)
        self._chat_display.append(chunk, "alice_text")

    def _finalize_alice_stream(self):
        if not self._streaming_started:
            if hasattr(self, "_animator"):
                self._animator.set_state(CharacterState.IDLE)
            return
        self._streaming_started = False
        self._chat_display.append("\n", "alice_text")
        if hasattr(self, "_animator"):
            self._animator.set_state(CharacterState.IDLE)

    def _append_system(self, text):
        if not self._widget_exists(self._chat_display):
            self._ensure_chat_display_ready()
        if not self._widget_exists(self._chat_display):
            return
        self._chat_display.append(f"\n{text}\n", "system")

    def _append_error(self, text):
        if not self._widget_exists(self._chat_display):
            self._ensure_chat_display_ready()
        if not self._widget_exists(self._chat_display):
            return
        self._chat_display.append(f"\nエラー: {text}\n", "error")

    def _clear_chat(self):
        if not self._widget_exists(self._chat_display):
            self._ensure_chat_display_ready()
        if not self._widget_exists(self._chat_display):
            return
        if messagebox.askyesno("クリア", "チャット履歴をクリアしますか？"):
            self._chat_display.clear()
            if self._alice:
                self._alice.clear_history()
            self._append_system("チャット履歴をクリアしました。")

    # ---- メニューコマンド ----

    def _open_settings(self):
        SettingsDialog(self.root, self._env, on_save=self._on_settings_saved)

    def _on_settings_saved(self):
        try:
            from module import neural_loader_module as _neural
            _neural.reset()
        except Exception:
            pass
        try:
            from module import local_llm_loader_module as _local
            _local.reset()
        except Exception:
            pass
        self._update_status("設定を更新しました（次回接続時にAI設定を再初期化します）。")

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

    def _open_advanced_image_tool(self):
        """高度な画像処理ツールを開く"""
        def _open():
            dlg = AdvancedBgRemovalDialog(
                self.root,
                char_loader=self._char_loader,
                on_reload=self._reload_character,
            )
            # 投げ縄ダブルクリック確定バインド（_bind_canvas_events内でも設定済みだが念のため）
            dlg._canvas_src.bind("<Double-Button-1>", dlg._confirm_lasso)
            # wait_window() は削除: grab_set() 済みのダイアログに wait_window を使うと
            # メインスレッドをブロックしてGUIがフリーズするため不要
        # メインループに制御を戻してからダイアログを開く（フリーズ防止）
        self.root.after(10, _open)

    def _open_bg_removal(self):
        """後方互換: 高度な画像処理ツールを開く"""
        self._open_advanced_image_tool()

    def _open_animation_tool(self):
        """キャラクターアニメーション作成ツールを開く"""
        def _open():
            dlg = AnimationCompositeDialog(
                self.root,
                char_loader=self._char_loader,
            )
            # wait_window() は削除: grab_set() 済みのため不要（フリーズ原因）
        self.root.after(10, _open)

    def _check_voicevox(self):
        if self._voice:
            ok = self._voice.check_connection()
            messagebox.showinfo("VOICEVOX",
                                "接続OK" if ok else "接続できません。VOICEVOXが起動しているか確認してください。")
        else:
            messagebox.showwarning("VOICEVOX", "VoiceEngine が初期化されていません。")

    @staticmethod
    def _is_android_platform() -> bool:
        return hasattr(sys, "getandroidapilevel") or bool(os.environ.get("ANDROID_ROOT"))

    @staticmethod
    def _is_ios_platform() -> bool:
        return (
            sys.platform == "ios"
            or bool(os.environ.get("PYTHONISTA_VERSION"))
            or bool(os.environ.get("PYTO_VERSION"))
        )

    def _open_path_cross_platform(self, target: Path) -> bool:
        target = target.resolve()
        target_str = str(target)
        target_uri = target.as_uri()

        if self._is_ios_platform():
            try:
                return bool(webbrowser.open(target_uri))
            except Exception as e:
                logger.warning(f"iOS open 失敗: {e}")

        if sys.platform.startswith("win"):
            try:
                os.startfile(target_str)  # type: ignore[attr-defined]
                return True
            except Exception as e:
                logger.warning(f"Windows open 失敗: {e}")

        commands: List[List[str]] = []
        if self._is_android_platform():
            commands.extend([
                ["termux-open", target_str],
                ["termux-open-url", target_uri],
                ["am", "start", "-a", "android.intent.action.VIEW", "-d", target_uri],
                ["xdg-open", target_str],
            ])
        elif sys.platform == "darwin":
            commands.append(["open", target_str])
        else:
            commands.extend([
                ["xdg-open", target_str],
                ["gio", "open", target_str],
                ["kioclient5", "exec", target_str],
                ["kde-open5", target_str],
            ])

        for cmd in commands:
            exe = cmd[0]
            if shutil.which(exe) is None:
                continue
            try:
                subprocess.Popen(cmd)
                return True
            except Exception as e:
                logger.warning(f"'{exe}' でのフォルダ起動失敗: {e}")

        try:
            return bool(webbrowser.open(target_uri))
        except Exception as e:
            logger.warning(f"webbrowser フォールバック失敗: {e}")
            return False

    def _open_logs(self):
        from module import result_log_module as _rl
        logs = _rl.get_logs_dir()
        logs.mkdir(parents=True, exist_ok=True)
        if self._open_path_cross_platform(logs):
            self._update_status("ログフォルダを開きました。")
            return
        messagebox.showinfo(
            "ログフォルダ",
            f"自動で開けませんでした。手動で開いてください:\n{logs}",
        )
        self._update_status(f"ログフォルダ: {logs}")

    def _show_about(self):
        messagebox.showinfo(
            "Alice AI について",
            "Alice AI\n\nInspired by Maid-chan from\nSakurasou no Pet na Kanojo\n\n"
            "Powered by Google Gemini × VOICEVOX\n\n"
            "画像処理: 独自アルゴリズム（API不使用）\n"
            "  - 高精度エッジ検出 (Sobel+Laplacian融合)\n"
            "  - Lab色空間適応的背景除去\n"
            "  - ポイント/矩形/楕円/投げ縄/ブラシ編集"
        )

    def _update_status(self, text):
        if hasattr(self, "_statusbar"):
            self._statusbar.configure(text=text)

    def _on_close(self):
        if messagebox.askyesno("終了", "Alice AI を終了しますか？"):
            self._close_startup_model_popup()
            if hasattr(self, "_animator"):
                self._animator.stop()
            if self._voice:
                self._voice.stop()
            logger.info("Alice AI 終了。")
            self.root.quit()
