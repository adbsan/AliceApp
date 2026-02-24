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
"""

from __future__ import annotations

import math
import queue
import subprocess
import threading
import time
import tkinter as tk
from collections import deque
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger

# 背景除去・画像処理に使用するライブラリ
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
# 高度な画像処理エンジン（独自アルゴリズム・API不使用）
# ================================================================== #

class AdvancedImageProcessor:
    """
    独自アルゴリズムによる高精度画像処理エンジン。
    外部API・学習済みモデル不使用。すべてアルゴリズムで実装。

    機能:
      1. 高精度エッジ検出（Canny + Laplacian + Sobel の融合）
      2. 適応的背景除去（Lab色空間 + グラフカット近似 + BFS）
      3. 精細マスク精錬（形態学的処理 + ガウシアンフェザリング）
      4. ポイント処理（ユーザー指定ピクセルからの範囲除去）
      5. 選択範囲処理（矩形・楕円・自由曲線領域の除去）
      6. 新背景合成（チェッカー・単色・グラデーション・画像）
    """

    # ----- 定数 -----
    _FEATHER_RADIUS   = 2.5    # エッジのフェザリング半径(px)
    _HAIR_DETAIL_ITER = 3      # 髪の毛詳細処理イテレーション数
    _MIN_CLUSTER_PX   = 50     # 小クラスタ除去しきい値
    _EDGE_BLEND_ALPHA = 0.35   # エッジ検出融合比率

    def __init__(self) -> None:
        self._available = _NUMPY_AVAILABLE and _PIL_AVAILABLE

    def is_available(self) -> bool:
        return self._available

    # ================================================================
    # ① 高精度エッジ検出（Sobel + Laplacian + 適応的閾値）
    # ================================================================

    def detect_edges_highquality(self, img_rgba: "np.ndarray") -> "np.ndarray":
        """
        複数のエッジ検出手法を融合した高精度エッジマップを返す。
        髪の毛・細かいディテールも検出できるよう設計。

        Returns:
            uint8 グレースケール配列 (0=背景, 255=エッジ)
        """
        if not _NUMPY_AVAILABLE:
            return np.zeros(img_rgba.shape[:2], dtype=np.uint8)

        gray = self._to_gray(img_rgba)

        # Sobelフィルタ（x, y方向の勾配を合成）
        sobel_x = self._sobel_x(gray)
        sobel_y = self._sobel_y(gray)
        sobel   = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel   = np.clip(sobel / sobel.max() * 255, 0, 255).astype(np.uint8) if sobel.max() > 0 else sobel.astype(np.uint8)

        # Laplacianフィルタ（二次微分：細かい輪郭強調）
        lap = self._laplacian(gray)
        lap = np.abs(lap)
        lap = np.clip(lap / lap.max() * 255, 0, 255).astype(np.uint8) if lap.max() > 0 else lap.astype(np.uint8)

        # 適応的しきい値によるCannyライク処理
        canny_like = self._adaptive_threshold_edge(gray)

        # 3種のエッジマップを加重融合
        fused = (
            sobel.astype(np.float32)     * 0.40 +
            lap.astype(np.float32)       * 0.25 +
            canny_like.astype(np.float32)* 0.35
        )
        fused = np.clip(fused, 0, 255).astype(np.uint8)

        # 髪の毛などの細線強調（細線化ノイズ除去）
        fused = self._enhance_thin_lines(fused)

        return fused

    def _to_gray(self, arr: "np.ndarray") -> "np.ndarray":
        """RGBA → グレースケール（知覚的重みづけ）"""
        r = arr[:, :, 0].astype(np.float32)
        g = arr[:, :, 1].astype(np.float32)
        b = arr[:, :, 2].astype(np.float32)
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)

    def _sobel_x(self, gray: "np.ndarray") -> "np.ndarray":
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        return self._convolve2d(gray, kernel)

    def _sobel_y(self, gray: "np.ndarray") -> "np.ndarray":
        kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        return self._convolve2d(gray, kernel)

    def _laplacian(self, gray: "np.ndarray") -> "np.ndarray":
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        return self._convolve2d(gray, kernel)

    def _convolve2d(self, img: "np.ndarray", kernel: "np.ndarray") -> "np.ndarray":
        """手動畳み込み（scipy/cv2 非依存の純粋numpy実装）"""
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2
        h, w = img.shape
        padded = np.pad(img, ((ph, ph), (pw, pw)), mode='edge')
        result = np.zeros_like(img)
        for i in range(kh):
            for j in range(kw):
                result += kernel[i, j] * padded[i:i+h, j:j+w]
        return result

    def _adaptive_threshold_edge(self, gray: "np.ndarray") -> "np.ndarray":
        """局所適応的しきい値によるエッジ検出"""
        h, w = gray.shape
        block = 15
        result = np.zeros((h, w), dtype=np.uint8)
        ph, pw = block // 2, block // 2
        padded = np.pad(gray, ((ph, ph), (pw, pw)), mode='edge')
        for y in range(h):
            for x in range(w):
                local = padded[y:y+block, x:x+block]
                mean  = local.mean()
                std   = local.std()
                thr   = mean - 0.5 * std
                result[y, x] = 255 if gray[y, x] < thr else 0
        return result

    def _enhance_thin_lines(self, edge_map: "np.ndarray") -> "np.ndarray":
        """細線（髪の毛など）の強調処理"""
        if not _SCIPY_AVAILABLE:
            return edge_map
        # 細いエッジを膨張させてから元に戻す（ノイズ除去しつつ細線保持）
        struct = np.ones((2, 2), dtype=bool)
        dilated  = ndimage.binary_dilation(edge_map > 128, structure=struct).astype(np.uint8) * 255
        eroded   = ndimage.binary_erosion(dilated > 128,  structure=struct).astype(np.uint8) * 255
        return np.maximum(edge_map, eroded)

    # ================================================================
    # ② 適応的背景除去
    # ================================================================

    def remove_background_adaptive(
        self,
        img_rgba: "np.ndarray",
        sensitivity: float = 1.0,
    ) -> "np.ndarray":
        """
        Lab色空間 + 適応的クラスタリング + BFS によって背景を除去する。
        人物・動物・製品・車・グラフィックスなど多様な被写体に対応。

        Args:
            img_rgba:    RGBA numpy配列
            sensitivity: 除去感度 (0.5=少なく, 1.0=標準, 2.0=多く)

        Returns:
            背景が透明になった RGBA numpy配列
        """
        if not _NUMPY_AVAILABLE:
            return img_rgba

        h, w = img_rgba.shape[:2]
        result = img_rgba.copy()

        # Lab色空間に変換（人間の知覚に近い色差計算のため）
        lab = self._rgb_to_lab(img_rgba[:, :, :3])

        # 四隅から背景色サンプリング
        m = max(3, min(12, h // 10, w // 10))
        corner_pixels = np.concatenate([
            lab[:m, :m].reshape(-1, 3),
            lab[:m, -m:].reshape(-1, 3),
            lab[-m:, :m].reshape(-1, 3),
            lab[-m:, -m:].reshape(-1, 3),
        ])
        bg_lab = corner_pixels.mean(axis=0)
        bg_std = corner_pixels.std(axis=0).mean()

        # Lab色差によるマスク生成
        diff = np.sqrt(np.sum((lab - bg_lab) ** 2, axis=2))
        base_threshold = max(8.0, bg_std * 2.5) * sensitivity
        is_bg_raw = diff < base_threshold

        # BFSで外周から連結背景領域を特定
        bg_mask = self._bfs_flood_fill(is_bg_raw, h, w)

        # 半透明領域（エッジ付近）の精細処理
        alpha_mask = self._refine_mask_with_edges(bg_mask, img_rgba, h, w)

        result[:, :, 3] = np.where(bg_mask, 0, alpha_mask)
        return result

    def _rgb_to_lab(self, rgb: "np.ndarray") -> "np.ndarray":
        """RGB → CIELab 変換（近似実装）"""
        rgb_f = rgb.astype(np.float32) / 255.0

        # sRGB → Linear RGB（ガンマ補正除去）
        mask = rgb_f > 0.04045
        linear = np.where(mask, ((rgb_f + 0.055) / 1.055) ** 2.4, rgb_f / 12.92)

        # Linear RGB → XYZ (D65白色点)
        r, g, b = linear[:, :, 0], linear[:, :, 1], linear[:, :, 2]
        x = r * 0.4124 + g * 0.3576 + b * 0.1805
        y = r * 0.2126 + g * 0.7152 + b * 0.0722
        z = r * 0.0193 + g * 0.1192 + b * 0.9505

        # XYZ → Lab
        xn, yn, zn = 0.9505, 1.0000, 1.0890
        fx = self._lab_f(x / xn)
        fy = self._lab_f(y / yn)
        fz = self._lab_f(z / zn)

        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b_c = 200 * (fy - fz)
        return np.stack([L, a, b_c], axis=2)

    def _lab_f(self, t: "np.ndarray") -> "np.ndarray":
        delta = 6 / 29
        return np.where(t > delta**3, t ** (1/3), t / (3 * delta**2) + 4/29)

    def _bfs_flood_fill(self, is_bg: "np.ndarray", h: int, w: int) -> "np.ndarray":
        """外周から連結背景領域をBFS探索"""
        visited = np.zeros((h, w), dtype=bool)
        q = deque()

        def seed(r, c):
            if not visited[r, c] and is_bg[r, c]:
                visited[r, c] = True
                q.append((r, c))

        for r in range(h):
            seed(r, 0); seed(r, w - 1)
        for c in range(w):
            seed(0, c); seed(h - 1, c)

        nb8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        while q:
            r, c = q.popleft()
            for dr, dc in nb8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and is_bg[nr, nc]:
                    visited[nr, nc] = True
                    q.append((nr, nc))
        return visited

    def _refine_mask_with_edges(
        self,
        bg_mask: "np.ndarray",
        img_rgba: "np.ndarray",
        h: int, w: int,
    ) -> "np.ndarray":
        """エッジ情報を使ってマスクの境界を精細化し、フェザリング処理を施す"""
        fg_mask = ~bg_mask

        if _SCIPY_AVAILABLE:
            # 距離変換でフェザリング
            dist_in  = ndimage.distance_transform_edt(fg_mask).astype(np.float32)
            dist_out = ndimage.distance_transform_edt(bg_mask).astype(np.float32)
            feather  = self._FEATHER_RADIUS
            alpha    = np.clip(dist_in / feather, 0.0, 1.0)
            alpha[fg_mask & (dist_out > feather * 3)] = 1.0

            # 小クラスタ（ノイズ）除去
            labeled, num = ndimage.label(fg_mask.astype(np.uint8))
            if num > 0:
                sizes = ndimage.sum(fg_mask, labeled, range(1, num + 1))
                for i, s in enumerate(sizes):
                    if s < self._MIN_CLUSTER_PX:
                        alpha[labeled == (i + 1)] = 0

            # 穴埋め（被写体内部の孤立した背景ピクセル）
            filled = ndimage.binary_fill_holes(alpha > 0.5)
            alpha  = np.where(filled & ~fg_mask, alpha.max() * 0.8, alpha)
        else:
            alpha = fg_mask.astype(np.float32)

        return (alpha * 255).astype(np.uint8)

    # ================================================================
    # ③ ポイント処理（ユーザー指定点からの除去）
    # ================================================================

    def remove_by_point(
        self,
        img_rgba: "np.ndarray",
        px: int, py: int,
        radius: int = 20,
        sensitivity: float = 1.0,
    ) -> "np.ndarray":
        """
        指定ピクセル座標を起点に、色が近いピクセルを除去する。
        フラッドフィル（塗りつぶし除去）方式。

        Args:
            img_rgba:    RGBA numpy配列
            px, py:      除去起点のピクセル座標 (display座標 → 変換済み)
            radius:      除去半径ヒント（色許容差に影響）
            sensitivity: 除去感度
        """
        if not _NUMPY_AVAILABLE:
            return img_rgba

        result = img_rgba.copy()
        h, w   = result.shape[:2]

        if not (0 <= py < h and 0 <= px < w):
            return result

        # 基準色をサンプリング（指定点の周辺平均）
        sr = max(0, py - 2)
        er = min(h, py + 3)
        sc = max(0, px - 2)
        ec = min(w, px + 3)
        seed_color = result[sr:er, sc:ec, :3].reshape(-1, 3).mean(axis=0)

        # Lab色空間で色差計算
        lab = self._rgb_to_lab(result[:, :, :3])
        seed_lab = self._rgb_to_lab(
            seed_color.reshape(1, 1, 3).astype(np.uint8)
        )[0, 0]

        # 許容差を radius と sensitivity から決定
        tolerance = max(10.0, radius * 0.8) * sensitivity

        # BFSフラッドフィル
        diff   = np.sqrt(np.sum((lab - seed_lab) ** 2, axis=2))
        is_similar = diff < tolerance
        fill_mask  = self._bfs_flood_fill(is_similar, h, w)

        # 指定点が外周に隣接していない場合、指定点を起点にローカルBFS
        if not fill_mask[py, px]:
            fill_mask = self._bfs_from_point(is_similar, py, px, h, w)

        # フェザリング付きで透明化
        if _SCIPY_AVAILABLE:
            dist = ndimage.distance_transform_edt(fill_mask).astype(np.float32)
            alpha_reduce = np.clip(dist / self._FEATHER_RADIUS, 0, 1)
            result[:, :, 3] = (result[:, :, 3] * (1 - alpha_reduce * fill_mask)).astype(np.uint8)
        else:
            result[:, :, 3][fill_mask] = 0

        return result

    def _bfs_from_point(
        self,
        is_similar: "np.ndarray",
        start_y: int, start_x: int,
        h: int, w: int,
    ) -> "np.ndarray":
        """指定点を起点としたBFSフラッドフィル"""
        visited = np.zeros((h, w), dtype=bool)
        if not is_similar[start_y, start_x]:
            return visited

        q = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        nb4 = [(-1,0),(1,0),(0,-1),(0,1)]
        while q:
            r, c = q.popleft()
            for dr, dc in nb4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and is_similar[nr, nc]:
                    visited[nr, nc] = True
                    q.append((nr, nc))
        return visited

    # ================================================================
    # ④ 選択範囲処理
    # ================================================================

    def remove_by_rect(
        self,
        img_rgba: "np.ndarray",
        x1: int, y1: int, x2: int, y2: int,
        mode: str = "hard",
    ) -> "np.ndarray":
        """
        矩形選択範囲内のピクセルを除去する。

        Args:
            mode: "hard"=即時除去, "color"=色マッチング除去, "feather"=フェザリング除去
        """
        if not _NUMPY_AVAILABLE:
            return img_rgba

        result = img_rgba.copy()
        h, w   = result.shape[:2]
        rx1, ry1 = max(0, min(x1, x2)), max(0, min(y1, y2))
        rx2, ry2 = min(w, max(x1, x2)), min(h, max(y1, y2))

        if mode == "hard":
            result[ry1:ry2, rx1:rx2, 3] = 0
        elif mode == "color":
            region = result[ry1:ry2, rx1:rx2]
            lab_r  = self._rgb_to_lab(region[:, :, :3])
            lab_m  = lab_r.reshape(-1, 3).mean(axis=0)
            diff   = np.sqrt(np.sum((lab_r - lab_m) ** 2, axis=2))
            mask   = diff < 20
            region[:, :, 3][mask] = 0
            result[ry1:ry2, rx1:rx2] = region
        elif mode == "feather":
            if _SCIPY_AVAILABLE:
                rect_mask = np.zeros((h, w), dtype=bool)
                rect_mask[ry1:ry2, rx1:rx2] = True
                dist = ndimage.distance_transform_edt(rect_mask).astype(np.float32)
                alpha_fade = np.clip(1 - dist / 10, 0, 1)
                result[:, :, 3] = (result[:, :, 3] * alpha_fade).astype(np.uint8)
            else:
                result[ry1:ry2, rx1:rx2, 3] = 0

        return result

    def remove_by_ellipse(
        self,
        img_rgba: "np.ndarray",
        cx: int, cy: int, rx: int, ry: int,
    ) -> "np.ndarray":
        """楕円選択範囲内のピクセルを除去する。"""
        if not _NUMPY_AVAILABLE:
            return img_rgba

        result = img_rgba.copy()
        h, w   = result.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w]
        ellipse_mask = ((xs - cx)**2 / max(rx, 1)**2 + (ys - cy)**2 / max(ry, 1)**2) <= 1.0
        result[:, :, 3][ellipse_mask] = 0
        return result

    def remove_by_lasso(
        self,
        img_rgba: "np.ndarray",
        points: List[Tuple[int, int]],
    ) -> "np.ndarray":
        """
        自由曲線（投げ縄）選択領域のピクセルを除去する。
        点列を内外判定（Ray Casting）でマスク生成。
        """
        if not _NUMPY_AVAILABLE or len(points) < 3:
            return img_rgba

        result = img_rgba.copy()
        h, w   = result.shape[:2]

        # PIL DrawでポリゴンマスクをRasterize
        if _PIL_AVAILABLE:
            mask_img = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask_img)
            draw.polygon(points, fill=255)
            lasso_mask = np.array(mask_img) > 128
            result[:, :, 3][lasso_mask] = 0

        return result

    # ================================================================
    # ⑤ マスク手動調整（ブラシ追加・消去）
    # ================================================================

    def apply_brush(
        self,
        img_rgba: "np.ndarray",
        px: int, py: int,
        brush_size: int = 15,
        mode: str = "erase",
    ) -> "np.ndarray":
        """
        ブラシで手動編集（消去または復元）。

        Args:
            mode: "erase"=透明化, "restore"=不透明化
        """
        if not _NUMPY_AVAILABLE:
            return img_rgba

        result = img_rgba.copy()
        h, w   = result.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w]
        dist   = np.sqrt((xs - px)**2 + (ys - py)**2)
        brush  = dist <= brush_size

        # ソフトブラシ（距離に応じてフェード）
        soft_alpha = np.clip(1 - dist / brush_size, 0, 1)
        soft_alpha[~brush] = 0

        if mode == "erase":
            result[:, :, 3] = (result[:, :, 3] * (1 - soft_alpha)).astype(np.uint8)
        else:  # restore
            result[:, :, 3] = np.clip(
                result[:, :, 3] + (soft_alpha * 255), 0, 255
            ).astype(np.uint8)

        return result

    # ================================================================
    # ⑥ 背景合成
    # ================================================================

    def composite_with_background(
        self,
        fg_rgba: "np.ndarray",
        bg_type: str = "checker",
        bg_color: Tuple[int, int, int] = (100, 100, 200),
        bg_image: Optional["np.ndarray"] = None,
    ) -> "np.ndarray":
        """
        前景（透過済み）と背景を合成する。

        Args:
            bg_type: "checker"|"solid"|"gradient"|"image"
        """
        if not _NUMPY_AVAILABLE:
            return fg_rgba

        h, w = fg_rgba.shape[:2]

        if bg_type == "checker":
            bg = self._make_checker_array(w, h)
        elif bg_type == "solid":
            bg = np.full((h, w, 4), (*bg_color, 255), dtype=np.uint8)
        elif bg_type == "gradient":
            bg = self._make_gradient_array(w, h, bg_color)
        elif bg_type == "image" and bg_image is not None:
            bg = self._resize_bg(bg_image, w, h)
        else:
            bg = self._make_checker_array(w, h)

        # アルファブレンディング
        alpha = fg_rgba[:, :, 3:4].astype(np.float32) / 255.0
        out   = (fg_rgba[:, :, :3].astype(np.float32) * alpha +
                 bg[:, :, :3].astype(np.float32) * (1 - alpha)).astype(np.uint8)
        return np.dstack([out, np.full((h, w), 255, dtype=np.uint8)])

    def _make_checker_array(self, w: int, h: int, size: int = 16) -> "np.ndarray":
        arr = np.full((h, w, 4), 255, dtype=np.uint8)
        for y in range(0, h, size):
            for x in range(0, w, size):
                if ((x // size) + (y // size)) % 2 == 1:
                    arr[y:y+size, x:x+size, :3] = 180
        return arr

    def _make_gradient_array(
        self,
        w: int, h: int,
        color: Tuple[int, int, int],
    ) -> "np.ndarray":
        arr = np.zeros((h, w, 4), dtype=np.uint8)
        for y in range(h):
            t = y / max(h - 1, 1)
            r = int(color[0] * (1 - t) + 30 * t)
            g = int(color[1] * (1 - t) + 30 * t)
            b = int(color[2] * (1 - t) + 60 * t)
            arr[y, :, :3] = [r, g, b]
        arr[:, :, 3] = 255
        return arr

    def _resize_bg(self, bg: "np.ndarray", w: int, h: int) -> "np.ndarray":
        if not _PIL_AVAILABLE:
            return np.full((h, w, 4), (100, 100, 100, 255), dtype=np.uint8)
        img = Image.fromarray(bg).convert("RGBA").resize((w, h), Image.LANCZOS)
        return np.array(img)

    # ================================================================
    # ⑥-B Inpaint（マスク領域の穴埋め補完）独自実装
    # ================================================================

    def inpaint_region(
        self,
        img_rgba: "np.ndarray",
        mask: "np.ndarray",
        radius: int = 8,
    ) -> "np.ndarray":
        """
        マスク領域を周囲のピクセルで補完する（Inpaint）。
        ComfyUI/Impact Pack的な「マスク→穴埋め」機能を独自実装。

        アルゴリズム:
          1. マスク境界を外側から内側へ同心円状に走査
          2. 各ピクセルを有効な近傍ピクセルの加重平均で補完
          3. 距離に応じた重み付け（近い画素を優先）
          4. 複数回イタレーションで品質向上

        Args:
            img_rgba: RGBA numpy配列
            mask:     補完対象マスク (True=補完する領域)
            radius:   補完参照半径

        Returns:
            補完済み RGBA numpy配列
        """
        if not _NUMPY_AVAILABLE:
            return img_rgba

        result = img_rgba.copy().astype(np.float32)
        h, w   = result.shape[:2]
        fill   = mask.copy()

        # 境界ピクセルから内側へ反復補完（Telea法近似）
        max_iter = max(h, w) // 2
        for iteration in range(max_iter):
            changed = False
            # 補完すべきピクセルのうち、有効な隣接ピクセルがあるものを処理
            ys, xs = np.where(fill)
            if len(ys) == 0:
                break

            for y, x in zip(ys, xs):
                # 参照半径内の有効ピクセルを収集
                y0 = max(0, y - radius)
                y1 = min(h, y + radius + 1)
                x0 = max(0, x - radius)
                x1 = min(w, x + radius + 1)

                region_valid = ~fill[y0:y1, x0:x1]
                if not region_valid.any():
                    continue

                # 距離加重平均で補完
                ry, rx = np.mgrid[y0:y1, x0:x1]
                dist   = np.sqrt((ry - y)**2 + (rx - x)**2) + 1e-6
                weight = (1.0 / dist**2) * region_valid.astype(np.float32)
                w_sum  = weight.sum()

                if w_sum < 1e-6:
                    continue

                for ch in range(4):
                    val = (result[y0:y1, x0:x1, ch] * weight).sum() / w_sum
                    result[y, x, ch] = val

                fill[y, x] = False
                changed = True

            if not changed:
                break

        return np.clip(result, 0, 255).astype(np.uint8)

    def create_inpaint_mask_from_alpha(self, img_rgba: "np.ndarray") -> "np.ndarray":
        """
        アルファチャンネルから Inpaint マスクを生成する。
        透明領域（除去済み領域）をInpaint対象として返す。
        """
        if not _NUMPY_AVAILABLE:
            return np.zeros(img_rgba.shape[:2], dtype=bool)
        return img_rgba[:, :, 3] < 128

    # ================================================================
    # ⑦ 全セル一括処理ユーティリティ
    # ================================================================

    def process_all_cells(
        self,
        sheet_img: "Image.Image",
        rows: int,
        cols: int,
        pose_names: List[str],
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, "Image.Image"]:
        """
        スプライトシートの全セルを一括処理して返す。
        背景除去・クロップ・正規化を自動適用。
        """
        if not _PIL_AVAILABLE or not _NUMPY_AVAILABLE:
            return {}

        results = {}
        total = len(pose_names)
        arr = np.array(sheet_img.convert("RGBA"))
        h, w = arr.shape[:2]
        cw, ch = w // cols, h // rows

        for i, name in enumerate(pose_names):
            if on_progress:
                on_progress(i + 1, total, f"処理中: {name}")
            row = i // cols
            col = i % cols
            cell_arr = arr[row*ch:(row+1)*ch, col*cw:(col+1)*cw]
            try:
                removed = self.remove_background_adaptive(cell_arr)
                cropped = self._autocrop_array(removed)
                norm    = self._normalize_array(cropped)
                results[name] = Image.fromarray(norm)
            except Exception as e:
                logger.error(f"セル '{name}' 処理エラー: {e}")

        return results

    def _autocrop_array(self, arr: "np.ndarray", padding: int = 20) -> "np.ndarray":
        """透明余白を自動クロップ"""
        alpha = arr[:, :, 3]
        mask  = alpha > 10
        if not mask.any():
            return arr
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin = max(0, int(np.where(rows)[0][0])  - padding)
        rmax = min(arr.shape[0] - 1, int(np.where(rows)[0][-1]) + padding)
        cmin = max(0, int(np.where(cols)[0][0])  - padding)
        cmax = min(arr.shape[1] - 1, int(np.where(cols)[0][-1]) + padding)
        return arr[rmin:rmax+1, cmin:cmax+1]

    def _normalize_array(self, arr: "np.ndarray", size: int = 2048) -> "np.ndarray":
        """正方形キャンバスに正規化（upscale対応）"""
        if not _PIL_AVAILABLE:
            return arr
        img = Image.fromarray(arr)
        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        scale  = min(size / img.width, size / img.height)
        nw, nh = int(img.width * scale), int(img.height * scale)
        resized = img.resize((nw, nh), Image.LANCZOS)
        canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2), resized)
        return np.array(canvas)


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

        # 状態管理
        self._src_image:    Optional[Image.Image] = None   # 元画像
        self._work_arr:     Optional["np.ndarray"] = None  # 現在の編集配列
        self._history_stack: List["np.ndarray"] = []       # Undo履歴
        self._result_image: Optional[Image.Image] = None   # 最終結果
        self._bg_image:     Optional["np.ndarray"] = None  # 合成用背景

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

        # バッチ処理結果
        self._batch_results: Dict[str, Image.Image] = {}

        self._setup_theme()
        self.title("高度な画像処理ツール - Alice AI")
        self.geometry("1280x800")
        self.minsize(1000, 650)
        self.configure(bg=self._c.bg_primary)
        self.transient(parent)
        self.grab_set()
        self._build_ui()

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

        # 左ツールバー
        self._build_toolbar(main, c)

        # 中央プレビューエリア（PanedWindow）
        center = tk.Frame(main, bg=c.bg_primary)
        center.pack(side="left", fill="both", expand=True, padx=4)

        self._build_preview_area(center, c)

        # 右パネル（設定・一括処理）
        self._build_right_panel(main, c)

        # 下部ステータスバー
        self._build_status_bar(c)

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
        """中央: 元画像 / 処理後 の左右プレビュー"""
        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True)

        # 左: 元画像（クリック操作受付）
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

        # 右: 処理後プレビュー（背景合成表示）
        rf = tk.Frame(paned, bg=c.bg_primary)
        paned.add(rf, weight=1)

        # 背景選択ヘッダー
        hdr = tk.Frame(rf, bg=c.bg_primary)
        hdr.pack(fill="x", padx=2)
        tk.Label(hdr, text="処理後プレビュー  背景:", bg=c.bg_primary,
                 fg=c.text_secondary, font=("Segoe UI", 9, "bold")).pack(side="left", padx=4)
        self._bg_type_var = tk.StringVar(value="checker")
        for bgt, lbl in [("checker","チェッカー"),("solid","単色"),
                          ("gradient","グラデ"),("image","画像")]:
            tk.Radiobutton(
                hdr, text=lbl, variable=self._bg_type_var, value=bgt,
                bg=c.bg_primary, fg=c.text_secondary,
                selectcolor=c.bg_tertiary, activebackground=c.bg_primary,
                command=self._refresh_result_preview,
                font=("Segoe UI", 8),
            ).pack(side="left")

        tk.Button(
            hdr, text="背景画像選択", command=self._select_bg_image,
            bg=c.bg_tertiary, fg=c.text_secondary, relief="flat",
            font=("Segoe UI", 8), padx=6, pady=2, cursor="hand2",
        ).pack(side="left", padx=4)

        self._canvas_result = tk.Canvas(
            rf, bg="#1a1a2e", highlightthickness=1,
            highlightbackground=c.border,
        )
        self._canvas_result.pack(fill="both", expand=True, padx=2, pady=2)

        # TkImage保持用
        self._tk_src:    Optional[ImageTk.PhotoImage] = None
        self._tk_result: Optional[ImageTk.PhotoImage] = None

    def _build_right_panel(self, parent, c):
        """右パネル: ファイル操作・一括処理・自動除去・保存"""
        rp = tk.Frame(parent, bg=c.bg_secondary, width=280)
        rp.pack(side="right", fill="y", padx=(2, 0))
        rp.pack_propagate(False)

        # スクロール可能エリア
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

        # バッチ結果リスト
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

        # ── エッジ検出 ──
        section("🔍 エッジ検出")
        self._btn(inner, c, "エッジを表示", self._show_edges).pack(fill="x", padx=10, pady=2)

        sep()

        # ── 保存先ポーズ ──
        section("💾 保存設定")
        tk.Label(inner, text="ポーズ名:", bg=c.bg_secondary, fg=c.text_secondary,
                 font=("Segoe UI", 9)).pack(anchor="w", padx=10)
        self._pose_var = tk.StringVar(value="default")
        ttk.Combobox(inner, textvariable=self._pose_var,
                     values=self._POSES, state="readonly",
                     font=("Segoe UI", 10)).pack(fill="x", padx=10, pady=2)

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

        # ── Inpaint（穴埋め補完）──
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

        # ── アニメーション作成へ連携 ──
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
        sb = tk.Frame(self, bg=c.bg_secondary, height=24)
        sb.pack(fill="x", side="bottom")
        sb.pack_propagate(False)
        self._coord_var = tk.StringVar(value="X:- Y:-")
        tk.Label(sb, textvariable=self._coord_var, bg=c.bg_secondary,
                 fg=c.text_muted, font=("Consolas", 8)).pack(side="left", padx=8)
        self._tool_info_var = tk.StringVar(value="ツール: ポイント除去")
        tk.Label(sb, textvariable=self._tool_info_var, bg=c.bg_secondary,
                 fg=c.text_muted, font=("Segoe UI", 8)).pack(side="right", padx=8)

    # ================================================================
    # キャンバスイベントバインド
    # ================================================================

    def _bind_canvas_events(self):
        c = self._canvas_src
        c.bind("<Button-1>",       self._on_canvas_click)
        c.bind("<B1-Motion>",      self._on_canvas_drag)
        c.bind("<ButtonRelease-1>",self._on_canvas_release)
        c.bind("<Motion>",         self._on_canvas_motion)
        c.bind("<Configure>",      lambda e: self._redraw_src())

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
            self._work_arr = self._processor.remove_by_point(
                self._work_arr, ix, iy,
                radius=self._brush_size,
                sensitivity=self._sensitivity.get(),
            )
            self._refresh_all_previews()

        elif tool in (self._TOOL_BRUSH_ERASE, self._TOOL_BRUSH_RESTORE):
            self._push_history()
            mode = "erase" if tool == self._TOOL_BRUSH_ERASE else "restore"
            self._work_arr = self._processor.apply_brush(
                self._work_arr, ix, iy, self._brush_size, mode)
            self._refresh_all_previews()

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

    def _on_canvas_drag(self, event):
        if self._work_arr is None:
            return
        ix, iy = self._canvas_to_image_coords(event.x, event.y)

        if self._current_tool in (self._TOOL_BRUSH_ERASE, self._TOOL_BRUSH_RESTORE):
            mode = "erase" if self._current_tool == self._TOOL_BRUSH_ERASE else "restore"
            self._work_arr = self._processor.apply_brush(
                self._work_arr, ix, iy, self._brush_size, mode)
            self._refresh_all_previews()

        elif self._current_tool in (self._TOOL_RECT, self._TOOL_ELLIPSE) and self._rect_drawing:
            self._rect_end = (ix, iy)
            self._redraw_src_with_selection()

        elif self._current_tool == self._TOOL_LASSO and self._lasso_drawing:
            self._lasso_points.append((ix, iy))
            self._redraw_src_with_selection()

    def _on_canvas_release(self, event):
        if self._work_arr is None:
            return
        ix, iy = self._canvas_to_image_coords(event.x, event.y)

        if self._current_tool == self._TOOL_RECT and self._rect_drawing:
            self._rect_end = (ix, iy)
            self._rect_drawing = False
            if self._rect_start and self._rect_end:
                self._push_history()
                x1, y1 = self._rect_start
                x2, y2 = self._rect_end
                self._work_arr = self._processor.remove_by_rect(
                    self._work_arr, x1, y1, x2, y2, mode="hard")
                self._refresh_all_previews()

        elif self._current_tool == self._TOOL_ELLIPSE and self._rect_drawing:
            self._rect_end = (ix, iy)
            self._rect_drawing = False
            if self._rect_start and self._rect_end:
                self._push_history()
                x1, y1 = self._rect_start
                x2, y2 = self._rect_end
                cx, cy = (x1+x2)//2, (y1+y2)//2
                rx, ry = abs(x2-x1)//2, abs(y2-y1)//2
                self._work_arr = self._processor.remove_by_ellipse(
                    self._work_arr, cx, cy, rx, ry)
                self._refresh_all_previews()

        elif self._current_tool == self._TOOL_LASSO and self._lasso_drawing:
            # ダブルクリック相当: release で確定
            pass  # ダブルクリックで確定（別バインド）

    # ================================================================
    # プレビュー描画
    # ================================================================

    def _redraw_src(self):
        """元画像（+ 操作ガイド）をキャンバスに描画"""
        if not _PIL_AVAILABLE or self._src_image is None:
            return
        self._draw_to_canvas(self._canvas_src, self._src_image, "_tk_src",
                             checker=False)

    def _redraw_src_with_selection(self):
        """選択範囲オーバーレイ付きで元画像を描画"""
        self._redraw_src()
        c = self._canvas_src
        cw, ch = c.winfo_width(), c.winfo_height()

        if self._src_image is None:
            return
        h, w = self._src_image.height, self._src_image.width
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
        """処理後プレビューを更新"""
        if self._work_arr is None or not _PIL_AVAILABLE:
            return
        bg_type = self._bg_type_var.get()
        composited = self._processor.composite_with_background(
            self._work_arr, bg_type=bg_type, bg_image=self._bg_image)
        img = Image.fromarray(composited)
        self._draw_to_canvas(self._canvas_result, img, "_tk_result", checker=False)
        self._result_image = Image.fromarray(self._work_arr)

    def _refresh_all_previews(self):
        """元画像と結果プレビューを両方更新"""
        self._redraw_src()
        self._refresh_result_preview()

    def _draw_to_canvas(
        self,
        canvas: tk.Canvas,
        img: Image.Image,
        attr: str,
        checker: bool = False,
    ):
        if not _PIL_AVAILABLE:
            return
        canvas.update_idletasks()
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            cw, ch = 500, 500
        scale = min(cw / max(img.width,1), ch / max(img.height,1)) * 0.95
        nw    = max(1, int(img.width * scale))
        nh    = max(1, int(img.height * scale))
        x     = (cw - nw) // 2
        y     = (ch - nh) // 2

        if checker and img.mode == "RGBA":
            bg_img = Image.new("RGBA", (nw, nh), (255,255,255,255))
            draw   = ImageDraw.Draw(bg_img)
            sz = 12
            for r in range(0, nh, sz):
                for col in range(0, nw, sz):
                    if ((r // sz) + (col // sz)) % 2 == 1:
                        draw.rectangle([col, r, col+sz, r+sz], fill=(180,180,180,255))
            resized = img.resize((nw, nh), Image.LANCZOS)
            bg_img.paste(resized, (0,0), resized)
            display = bg_img
        else:
            display = img.resize((nw, nh), Image.LANCZOS)

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
        # 投げ縄をリセット
        self._lasso_points = []
        self._lasso_drawing = False

    def _confirm_lasso(self, event=None):
        """投げ縄確定（ダブルクリック）"""
        if (self._current_tool == self._TOOL_LASSO
                and len(self._lasso_points) >= 3
                and self._work_arr is not None):
            self._push_history()
            self._work_arr = self._processor.remove_by_lasso(
                self._work_arr, self._lasso_points)
            self._lasso_points = []
            self._lasso_drawing = False
            self._refresh_all_previews()

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
            self._batch_listbox.delete(0, "end")
            self._save_btn.configure(state="disabled")
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
        self._progress.stop()
        self._processing = False
        self._save_btn.configure(state="normal")
        self._refresh_all_previews()
        self._set_status("自動背景除去完了")

    def _run_batch_process(self):
        if self._src_image is None:
            self._set_status("シート画像を開いてください", error=True)
            return
        if self._processing:
            return

        rows = self._sheet_rows.get()
        cols = self._sheet_cols.get()
        total = rows * cols
        pose_names = [f"cell_{i:02d}" for i in range(total)]

        self._processing = True
        self._progress.start(10)
        self._set_status(f"一括処理中... (全{total}セル)")
        self._batch_listbox.delete(0, "end")

        def _run():
            def on_prog(current, total, msg):
                self.after(0, lambda: self._set_status(msg))
            results = self._processor.process_all_cells(
                self._src_image, rows, cols, pose_names, on_progress=on_prog)
            self.after(0, self._on_batch_done, results)

        threading.Thread(target=_run, daemon=True).start()

    def _on_batch_done(self, results: Dict[str, Image.Image]):
        self._batch_results = results
        self._progress.stop()
        self._processing = False
        self._batch_listbox.delete(0, "end")
        for name in results.keys():
            self._batch_listbox.insert("end", name)
        if results:
            self._save_batch_btn.configure(state="normal")
        self._set_status(f"一括処理完了: {len(results)} セル")

    def _on_batch_select(self, event):
        sel = self._batch_listbox.curselection()
        if not sel:
            return
        name = self._batch_listbox.get(sel[0])
        img  = self._batch_results.get(name)
        if img is not None and _NUMPY_AVAILABLE:
            self._push_history()
            self._work_arr = np.array(img)
            self._save_btn.configure(state="normal")
            self._refresh_all_previews()

    def _on_process_error(self, msg: str):
        self._progress.stop()
        self._processing = False
        self._set_status(f"エラー: {msg}", error=True)

    def _show_edges(self):
        if self._work_arr is None:
            return
        if not _NUMPY_AVAILABLE:
            self._set_status("numpyが必要です", error=True)
            return
        edge_map = self._processor.detect_edges_highquality(self._work_arr)
        edge_img = Image.fromarray(edge_map).convert("RGBA")
        self._draw_to_canvas(self._canvas_result, edge_img, "_tk_result")
        self._set_status("エッジ検出マップを表示中")

    # ================================================================
    # 保存処理（確認ダイアログ付き）
    # ================================================================

    def _save_with_confirm(self):
        if self._work_arr is None:
            return
        result_img = Image.fromarray(self._work_arr)
        self._show_save_dialog({"single": result_img})

    def _save_batch_with_confirm(self):
        if not self._batch_results:
            return
        self._show_save_dialog(self._batch_results)

    def _show_save_dialog(self, images: Dict[str, Image.Image]):
        """
        保存確認ダイアログ。
        - 保存する / しない の選択
        - 保存先フォルダ選択
        - カスタム名 / 連番名の選択
        - ポーズ名マッピング（単体の場合）
        """
        dlg = tk.Toplevel(self)
        dlg.title("保存確認")
        dlg.geometry("500x420")
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

        # 保存先フォルダ
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

        # 命名モード
        tk.Label(dlg, text="ファイル命名:", bg=c.bg_primary, fg=c.text_secondary,
                 font=("Segoe UI", 10)).pack(anchor="w", padx=20, pady=(10, 0))
        name_mode = tk.StringVar(value="pose")
        modes = [
            ("pose",     "ポーズ名 (alice_default 等)"),
            ("custom",   "カスタム名"),
            ("sequence", "連番 (image_001, image_002...)"),
        ]
        if len(images) > 1:
            modes = [("sequence", "連番 (image_001, image_002...)"),
                     ("custom_prefix", "プレフィックス + 連番")]
        for val, lbl in modes:
            tk.Radiobutton(dlg, text=lbl, variable=name_mode, value=val,
                           bg=c.bg_primary, fg=c.text_secondary,
                           selectcolor=c.bg_tertiary,
                           font=("Segoe UI", 9)).pack(anchor="w", padx=30)

        # カスタム名入力
        tk.Label(dlg, text="カスタム名 / プレフィックス:",
                 bg=c.bg_primary, fg=c.text_secondary,
                 font=("Segoe UI", 9)).pack(anchor="w", padx=20, pady=(6, 0))
        custom_var = tk.StringVar(value=self._custom_name_var.get()
                                  or self._POSE_FILE_MAP.get(self._pose_var.get(), "output"))
        tk.Entry(dlg, textvariable=custom_var, bg=c.bg_tertiary,
                 fg=c.text_primary, insertbackground=c.text_primary,
                 relief="flat", font=("Segoe UI", 10), highlightthickness=1,
                 highlightbackground=c.border).pack(fill="x", padx=20, ipady=3)

        # ボタン行
        btn_row = tk.Frame(dlg, bg=c.bg_primary)
        btn_row.pack(pady=16)

        def _do_save():
            dest_dir = Path(dir_var.get())
            dest_dir.mkdir(parents=True, exist_ok=True)
            mode     = name_mode.get()
            custom   = custom_var.get().strip() or "output"
            pose_key = self._pose_var.get()
            saved = []

            try:
                if len(images) == 1 and mode == "pose":
                    # ポーズ名で保存
                    fname = self._POSE_FILE_MAP.get(pose_key, custom) + ".png"
                    path  = dest_dir / fname
                    list(images.values())[0].save(path, "PNG")
                    saved.append(str(path))
                elif mode in ("custom", "pose"):
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

                # CharacterLoader リロード
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
        """
        現在の作業画像の透明領域（除去済み部分）を
        周囲のピクセルで Inpaint（穴埋め補完）する。
        """
        if self._work_arr is None:
            self._set_status("画像を開いてください", error=True)
            return
        if self._processing:
            return
        if not _NUMPY_AVAILABLE:
            self._set_status("numpy が必要です", error=True)
            return

        self._processing = True
        self._push_history()
        self._progress.start(10)
        self._set_status("Inpaint 処理中...")

        radius = self._inpaint_radius.get()

        def _do():
            try:
                mask   = self._processor.create_inpaint_mask_from_alpha(self._work_arr)
                if not mask.any():
                    self.after(0, lambda: self._set_status("透明領域なし、Inpaintをスキップ"))
                    self.after(0, self._finish_processing)
                    return
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
        self._progress.stop()
        self._processing = False

    def _open_animation_from_here(self):
        """
        現在の処理済み画像（または一括処理結果）を
        AnimationCompositeDialog に渡してアニメーション作成へ移行する。
        """
        # 現在の work_arr から PIL Image を作成
        import_images: Dict[str, "Image.Image"] = {}

        if self._batch_results:
            import_images = dict(self._batch_results)
        elif self._work_arr is not None and _PIL_AVAILABLE and _NUMPY_AVAILABLE:
            pose = self._pose_var.get() if hasattr(self, "_pose_var") else "default"
            import_images[pose] = Image.fromarray(self._work_arr)

        if not import_images:
            self._set_status("アニメーションに送る画像がありません", error=True)
            return

        # AnimationCompositeDialog を開く
        dlg = AnimationCompositeDialog(
            self.master,
            char_loader=self._char_loader,
        )

        # 処理済み画像を自動レイヤーとして追加
        def _after_open():
            for name, img in import_images.items():
                dlg._add_layer(img, name)

        dlg.after(200, _after_open)
        self._set_status(f"アニメーション作成ツールへ {len(import_images)} 枚を送りました")

    def _set_status(self, msg: str, error: bool = False):
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

    機能:
      1. レイヤー管理（背景・被写体・前景パーツの重ね合わせ）
      2. パーツ位置・スケール・不透明度の調整
      3. アニメーションフレーム管理（複数フレーム構成）
      4. フレームプレビュー（コマ送り再生）
      5. GIF / 連番PNG 書き出し（外部ライブラリ不要）
      6. Inpaint統合（除去した穴を補完してから合成）

    独自アルゴリズム:
      - アルファブレンディング（Porter-Duff Over 合成）
      - 双線形補間リサイズ（PIL LANCZOS）
      - フレーム差分圧縮（GIF Palette量子化）
    """

    # フレームのデフォルト設定
    _DEFAULT_FPS   = 12
    _DEFAULT_DELAY = 83   # ms (≒12fps)

    def __init__(self, parent, char_loader=None):
        super().__init__(parent)
        self._char_loader = char_loader
        self._processor   = AdvancedImageProcessor()

        # レイヤー管理
        self._layers: List[Dict] = []          # 各レイヤー: {name, img, x, y, scale, alpha, visible}
        self._selected_layer: Optional[int] = None

        # フレーム管理
        self._frames: List["np.ndarray"] = []  # 合成済みフレーム一覧
        self._current_frame: int = 0
        self._playing: bool = False
        self._fps = self._DEFAULT_FPS

        # キャンバスサイズ
        self._canvas_w = 512
        self._canvas_h = 512

        # TkImage保持
        self._tk_preview: Optional[ImageTk.PhotoImage] = None

        self._setup_theme()
        self.title("キャラクターアニメーション作成 - Alice AI")
        self.geometry("1300x820")
        self.minsize(1100, 700)
        self.configure(bg=self._c.bg_primary)
        self.transient(parent)
        self.grab_set()
        self._build_ui()

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

        # レイヤー追加ボタン群
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

        # レイヤーリスト
        self._layer_listbox = tk.Listbox(
            lp, bg=c.bg_tertiary, fg=c.text_primary, selectbackground=c.accent_primary,
            relief="flat", font=("Segoe UI", 9), height=8,
        )
        self._layer_listbox.pack(fill="x", padx=6, pady=4)
        self._layer_listbox.bind("<<ListboxSelect>>", self._on_layer_select)

        # レイヤー順序変更
        ord_row = tk.Frame(lp, bg=c.bg_secondary)
        ord_row.pack(fill="x", padx=6)
        for txt, cmd in [("↑ 上へ", self._move_layer_up), ("↓ 下へ", self._move_layer_down)]:
            tk.Button(ord_row, text=txt, command=cmd,
                      bg=c.bg_tertiary, fg=c.text_secondary, relief="flat",
                      font=("Segoe UI", 8), padx=8, pady=3,
                      cursor="hand2").pack(side="left", padx=2)

        tk.Frame(lp, bg=c.border, height=1).pack(fill="x", padx=6, pady=8)

        # レイヤープロパティ
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

        # Inpaint ボタン
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

        # キャンバスサイズ選択
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
        self._composite_canvas.bind("<Configure>", lambda e: self._refresh_composite())

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

        # ── フレーム操作 ──
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

        # フレームコピー
        tk.Button(rp, text="現在の合成をフレームに追加",
                  command=self._capture_frame,
                  bg=c.accent_secondary, fg=c.text_primary, relief="flat",
                  font=("Segoe UI", 9), padx=8, pady=4, cursor="hand2",
                  ).pack(fill="x", padx=10, pady=2)

        sep()

        # ── 再生 ──
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

        # ── 書き出し ──
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

        # ステータス
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
        name = self._layers[i-1]["name"]
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

    # ================================================================
    # Inpaint統合
    # ================================================================

    def _inpaint_selected_layer(self):
        """選択レイヤーの透明領域をInpaintで補完する"""
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
        self.update_idletasks()

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
        """
        全レイヤーを下から上へ Porter-Duff Over 合成して返す。
        独自アルファブレンディング実装。
        """
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

            # レイヤーアルファを適用
            arr[:, :, 3] = arr[:, :, 3] * (layer["alpha"] / 255.0)

            # キャンバスへの貼り付け座標
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

            # Porter-Duff Over: dst = src + dst * (1 - src_alpha)
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
        canvas.update_idletasks()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return
        # チェッカー背景で透明度を可視化
        bg = Image.new("RGBA", (cw, ch), (40, 40, 60, 255))
        sz = 12
        draw = ImageDraw.Draw(bg)
        for y in range(0, ch, sz):
            for x in range(0, cw, sz):
                if ((x // sz) + (y // sz)) % 2 == 1:
                    draw.rectangle([x, y, x+sz, y+sz], fill=(60, 60, 80, 255))
        scale = min(cw / max(img.width, 1), ch / max(img.height, 1)) * 0.95
        nw    = max(1, int(img.width * scale))
        nh    = max(1, int(img.height * scale))
        x     = (cw - nw) // 2
        y     = (ch - nh) // 2
        resized = img.resize((nw, nh), Image.LANCZOS)
        bg.paste(resized, (x, y), resized)
        self._tk_preview = ImageTk.PhotoImage(bg)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=self._tk_preview)

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
        """空フレームを追加"""
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
        """現在の合成結果をフレームとして追加"""
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
        """フレームをGIFアニメとして書き出す"""
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
                # GIF は RGB+パレット → RGBA → P変換
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
        """フレームを連番PNGとして書き出す"""
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
        """現在フレームを単体PNGで保存"""
        if not self._frames:
            messagebox.showwarning("警告", "フレームがありません", parent=self)
            return

        # 保存確認
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
        self._mode  = AppMode.DESKTOP

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

        fm = menu(menubar)
        fm.add_command(label="設定", command=self._open_settings, accelerator="Ctrl+,")
        fm.add_separator()
        fm.add_command(label="終了", command=self._on_close)
        menubar.add_cascade(label="ファイル", menu=fm)

        vm = menu(menubar)
        vm.add_command(label="チャット履歴をクリア", command=self._clear_chat)
        menubar.add_cascade(label="表示", menu=vm)

        gm = menu(menubar)
        gm.add_command(label="Git マネージャー", command=self._open_git_dialog)
        gm.add_command(label="クイックコミット",  command=self._quick_commit)
        gm.add_command(label="ブランチ切替...",   command=self._switch_branch_dialog)
        menubar.add_cascade(label="Git", menu=gm)

        tm = menu(menubar)
        tm.add_command(label="キャラクター再読み込み", command=self._reload_character)
        tm.add_command(label="🎨 高度な画像処理ツール", command=self._open_advanced_image_tool)
        tm.add_command(label="🎬 アニメーション作成ツール", command=self._open_animation_tool)
        tm.add_separator()
        tm.add_command(label="VOICEVOX 接続確認",     command=self._check_voicevox)
        tm.add_separator()
        tm.add_command(label="ログフォルダを開く",    command=self._open_logs)
        menubar.add_cascade(label="ツール", menu=tm)

        hm = menu(menubar)
        hm.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="ヘルプ", menu=hm)

        self.root.bind("<Control-comma>", lambda e: self._open_settings())
        self.root.bind("<Return>",        lambda e: self._on_send())

    def _build_desktop_ui(self):
        c = self.colors
        layout = get_layout(AppMode.DESKTOP)

        self._paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self._paned.pack(fill="both", expand=True)

        chat_frame = tk.Frame(self._paned, bg=c.bg_primary)
        self._paned.add(chat_frame, weight=62)

        self._build_header(chat_frame, c)
        self._build_chat_display(chat_frame, c)
        self._build_input_area(chat_frame, c)

        char_frame = tk.Frame(self._paned, bg=c.bg_secondary)
        self._paned.add(char_frame, weight=38)

        self._build_character_panel(char_frame, c, layout)
        self.root.after(50, self._set_initial_sash)
        self._build_status_bar(c)

    def _set_initial_sash(self):
        try:
            total = self.root.winfo_width()
            if total > 10:
                self._paned.sashpos(0, int(total * self._CHAT_RATIO))
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

    def _open_advanced_image_tool(self):
        """高度な画像処理ツールを開く（旧 BgRemovalDialog を置き換え）"""
        dlg = AdvancedBgRemovalDialog(
            self.root,
            char_loader=self._char_loader,
            on_reload=self._reload_character,
        )
        # 投げ縄ダブルクリック確定バインド
        dlg._canvas_src.bind("<Double-Button-1>", dlg._confirm_lasso)
        self.root.wait_window(dlg)

    def _open_bg_removal(self):
        """後方互換: 高度な画像処理ツールを開く"""
        self._open_advanced_image_tool()

    def _open_animation_tool(self):
        """キャラクターアニメーション作成ツールを開く"""
        dlg = AnimationCompositeDialog(
            self.root,
            char_loader=self._char_loader,
        )
        self.root.wait_window(dlg)

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
            if hasattr(self, "_animator"):
                self._animator.stop()
            if self._voice:
                self._voice.stop()
            logger.info("Alice AI 終了。")
            self.root.quit()
            self.root.destroy()
