"""
image_builder_module.py
スプライトシートからキャラクター画像を生成する外装モジュール。

処理パイプライン（直列・逆流禁止）:
  1. Load Sheet    - assets/parts/sheet.png を読み込む
  2. Trim Parts    - グリッド指定でパーツをトリミング
  3. Remove BG     - 背景色を自動検出してBFSフラッドフィルで除去
  4. Normalize     - 2048x2048 に正規化（1:1・透明パディング）
  5. Save          - assets/images/ に alice_*.png として保存

背景除去アルゴリズム（改訂版）:
  四隅サンプルから背景タイプを自動判定する:
    - 暗色(黒)背景: max(R,G,B) < 50 の連続領域を除去
    - チェッカー柄: グレー系(R≈G≈B)かつ輝度>120 の連続領域を除去
  JPEGアーティファクト耐性のため判定閾値を緩和(±30)。
"""

from __future__ import annotations

import threading
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger

# ============================================================
# 定数
# ============================================================
OUTPUT_SIZE   = 2048
DISPLAY_SIZE  = 512

DEFAULT_POSE_MAP: Dict[int, str] = {
    0: "alice_default",
    1: "alice_idle",
    2: "alice_speaking",
    3: "alice_thinking",
    4: "alice_greeting",
}

# ============================================================
# 依存チェック
# ============================================================
try:
    from PIL import Image
    import numpy as np
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    logger.error("Pillow / numpy が未インストールです。pip install Pillow numpy を実行してください。")


# ============================================================
# メイン処理クラス
# ============================================================

class ImageBuilder:
    """スプライトシートからキャラクター画像を生成するクラス。"""

    def __init__(
        self,
        sheet_path: str,
        rows: int,
        cols: int,
        output_dir: str = "assets/images",
    ) -> None:
        self._sheet_path = Path(sheet_path)
        self._rows = rows
        self._cols = cols
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def build(
        self,
        pose_map: Optional[Dict[int, str]] = None,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, bool]:
        if not _PIL_AVAILABLE:
            logger.error("Pillow / numpy が利用できないため処理を中止します。")
            return {}

        _pose_map = pose_map or DEFAULT_POSE_MAP
        results: Dict[str, bool] = {}

        sheet = self._load_sheet()
        if sheet is None:
            return {}

        total = len(_pose_map)

        for i, (cell_idx, name) in enumerate(_pose_map.items()):
            msg = f"処理中: {name} ({i+1}/{total})"
            logger.info(msg)
            if on_progress:
                on_progress(i + 1, total, msg)

            try:
                cell = self._trim_cell(sheet, cell_idx)
                if cell is None:
                    logger.warning(f"セル {cell_idx} が範囲外です。スキップします。")
                    results[name] = False
                    continue

                cell_nobg = remove_bg(cell)
                normalized = self._normalize(cell_nobg)
                out_path = self._output_dir / f"{name}.png"
                normalized.save(out_path, "PNG")
                logger.info(f"保存完了: {out_path}")
                results[name] = True

            except Exception as e:
                logger.error(f"{name} の処理中にエラー: {e}")
                results[name] = False

        return results

    def build_single(self, image_path: str, output_name: str) -> bool:
        if not _PIL_AVAILABLE:
            return False
        try:
            img = Image.open(image_path).convert("RGBA")
            logger.info(f"単体画像読み込み: {image_path} ({img.width}x{img.height})")
            nobg = remove_bg(img)
            normalized = self._normalize(nobg)
            out_path = self._output_dir / f"{output_name}.png"
            normalized.save(out_path, "PNG")
            logger.info(f"単体画像保存完了: {out_path}")
            return True
        except Exception as e:
            logger.error(f"単体画像処理エラー: {e}")
            return False

    def build_async(
        self,
        pose_map: Optional[Dict[int, str]] = None,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        on_complete: Optional[Callable[[Dict[str, bool]], None]] = None,
    ) -> None:
        def _run():
            results = self.build(pose_map=pose_map, on_progress=on_progress)
            if on_complete:
                on_complete(results)
        threading.Thread(target=_run, daemon=True).start()

    def get_cell_count(self) -> int:
        return self._rows * self._cols

    def preview_cells(self) -> List[Tuple[int, "Image.Image"]]:
        if not _PIL_AVAILABLE:
            return []
        sheet = self._load_sheet()
        if sheet is None:
            return []
        previews = []
        for idx in range(self._rows * self._cols):
            cell = self._trim_cell(sheet, idx)
            if cell:
                thumb = cell.copy()
                thumb.thumbnail((256, 256), Image.LANCZOS)
                previews.append((idx, thumb))
        return previews

    def _load_sheet(self) -> Optional["Image.Image"]:
        if not self._sheet_path.exists():
            logger.error(f"シート画像が見つかりません: {self._sheet_path}")
            return None
        try:
            img = Image.open(self._sheet_path).convert("RGBA")
            logger.info(
                f"シート読み込み完了: {self._sheet_path} "
                f"({img.width}x{img.height}) → {self._rows}行 x {self._cols}列"
            )
            return img
        except Exception as e:
            logger.error(f"シート読み込みエラー: {e}")
            return None

    def _trim_cell(self, sheet: "Image.Image", cell_idx: int) -> Optional["Image.Image"]:
        total = self._rows * self._cols
        if cell_idx >= total:
            return None
        row = cell_idx // self._cols
        col = cell_idx % self._cols
        cell_w = sheet.width  // self._cols
        cell_h = sheet.height // self._rows
        left  = col * cell_w
        upper = row * cell_h
        right = left + cell_w
        lower = upper + cell_h
        return sheet.crop((left, upper, right, lower))

    def _normalize(self, img: "Image.Image") -> "Image.Image":
        canvas = Image.new("RGBA", (OUTPUT_SIZE, OUTPUT_SIZE), (0, 0, 0, 0))
        img.thumbnail((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)
        x = (OUTPUT_SIZE - img.width)  // 2
        y = (OUTPUT_SIZE - img.height) // 2
        canvas.paste(img, (x, y), img if img.mode == "RGBA" else None)
        return canvas

    # 後方互換性のためにメソッドとしても保持
    def _remove_checker_bg(self, img: "Image.Image") -> "Image.Image":
        return remove_bg(img)


# ============================================================
# 背景除去（モジュール公開関数）
# ============================================================

def remove_bg(img: "Image.Image") -> "Image.Image":
    """
    背景を自動検出して除去する。

    対応する背景タイプ:
      - 純黒 / 暗色背景  : max(R,G,B) < 50 の連続領域を除去
      - チェッカー / 白系 : グレー系(R≈G≈B ±30) かつ輝度>120 の連続領域を除去

    JPEGアーティファクトに対応するため判定を緩和（従来 ±20 → ±30）。

    既に有意な透過ピクセルを持つ画像（アルファ最小値 < 200）はスキップする。

    Returns:
        背景透明のRGBA画像
    """
    if not _PIL_AVAILABLE:
        return img

    # 既存アルファが有効な場合はスキップ
    if img.mode == "RGBA":
        import numpy as np_
        alpha_arr = np_.array(img.getchannel("A"))
        if alpha_arr.min() < 200:
            logger.debug("既存アルファチャンネルを使用します。")
            return img

    import numpy as np
    arr = np.array(img.convert("RGBA")).copy()
    rgb = arr[:, :, :3].astype(np.int32)
    h, w = rgb.shape[:2]

    # ── 四隅サンプルで背景タイプを判定 ──────────────────────
    margin = min(10, h // 8, w // 8)
    corner_samples = np.concatenate([
        arr[:margin, :margin, :3].reshape(-1, 3),
        arr[:margin, -margin:, :3].reshape(-1, 3),
        arr[-margin:, :margin, :3].reshape(-1, 3),
        arr[-margin:, -margin:, :3].reshape(-1, 3),
    ], axis=0).astype(np.float32)
    avg_brightness = corner_samples.mean()

    if avg_brightness < 40:
        # ────── 暗色(黒)背景モード ──────
        # チャンネル最大値が 50 未満のピクセルを背景候補とする
        max_channel = np.max(rgb, axis=2)
        is_bg_candidate = max_channel < 50
        logger.debug(f"背景タイプ: 暗色(黒) avg_brightness={avg_brightness:.1f}")
    else:
        # ────── チェッカー / グレー / 白系背景モード ──────
        # R≈G≈B (彩度低) かつ輝度 > 120
        diff_rg = np.abs(rgb[:, :, 0] - rgb[:, :, 1])
        diff_gb = np.abs(rgb[:, :, 1] - rgb[:, :, 2])
        is_gray = (diff_rg < 30) & (diff_gb < 30)   # ±30: JPEG対応
        brightness = (rgb[:, :, 0] + rgb[:, :, 1] + rgb[:, :, 2]) // 3
        is_bg_candidate = is_gray & (brightness > 120)
        logger.debug(f"背景タイプ: チェッカー/グレー avg_brightness={avg_brightness:.1f}")

    # ── 四隅・端辺からBFSで連続背景領域を探索 ──────────────
    visited = np.zeros((h, w), dtype=bool)
    queue: deque = deque()

    def seed(r: int, c: int) -> None:
        if not visited[r, c] and is_bg_candidate[r, c]:
            visited[r, c] = True
            queue.append((r, c))

    for r in range(h):
        seed(r, 0)
        seed(r, w - 1)
    for c in range(w):
        seed(0, c)
        seed(h - 1, c)

    # 8近傍BFS
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while queue:
        r, c = queue.popleft()
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                if is_bg_candidate[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))

    arr[:, :, 3][visited] = 0

    removed = visited.sum()
    total = h * w
    logger.debug(f"背景除去完了: {removed:,}px ({removed/total*100:.1f}%)")

    return Image.fromarray(arr)


# 後方互換性エイリアス
def _remove_checker_bg_compat(img: "Image.Image") -> "Image.Image":
    return remove_bg(img)


# ============================================================
# モジュールレベル関数
# ============================================================

def is_available() -> bool:
    return _PIL_AVAILABLE

def is_rembg_available() -> bool:
    """後方互換性のため残す。本モジュールは rembg を使用しない。"""
    return False

def get_output_size() -> int:
    return OUTPUT_SIZE

def get_display_size() -> int:
    return DISPLAY_SIZE
