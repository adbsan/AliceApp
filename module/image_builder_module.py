"""
image_builder_module.py
スプライトシートからキャラクター画像を生成する外装モジュール。

修正点（v3）:
  - autocrop() 追加: 背景除去後の透明余白を自動トリミング
  - normalize() 修正: thumbnail()（縮小専用）→ resize()（拡大対応）
  - remove_bg() : 黒背景・チェッカー柄の両対応（四隅自動判定）
"""

from __future__ import annotations

import threading
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger

OUTPUT_SIZE   = 2048
DISPLAY_SIZE  = 512

DEFAULT_POSE_MAP: Dict[int, str] = {
    0: "alice_default",
    1: "alice_idle",
    2: "alice_speaking",
    3: "alice_thinking",
    4: "alice_greeting",
}

try:
    from PIL import Image
    import numpy as np
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    logger.error("Pillow / numpy が未インストールです。pip install Pillow numpy を実行してください。")


class ImageBuilder:
    """スプライトシートからキャラクター画像を生成するクラス。"""

    def __init__(self, sheet_path: str, rows: int, cols: int,
                 output_dir: str = "assets/images") -> None:
        self._sheet_path = Path(sheet_path)
        self._rows = rows
        self._cols = cols
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def build(self, pose_map: Optional[Dict[int, str]] = None,
              on_progress: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, bool]:
        if not _PIL_AVAILABLE:
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
                    results[name] = False
                    continue
                nobg    = remove_bg(cell)
                cropped = autocrop(nobg)
                norm    = _normalize(cropped)
                out_path = self._output_dir / f"{name}.png"
                norm.save(out_path, "PNG")
                results[name] = True
            except Exception as e:
                logger.error(f"{name} の処理中にエラー: {e}")
                results[name] = False

        return results

    def build_single(self, image_path: str, output_name: str) -> bool:
        if not _PIL_AVAILABLE:
            return False
        try:
            img  = Image.open(image_path).convert("RGBA")
            nobg = remove_bg(img)
            cropped = autocrop(nobg)
            norm = _normalize(cropped)
            out_path = self._output_dir / f"{output_name}.png"
            norm.save(out_path, "PNG")
            return True
        except Exception as e:
            logger.error(f"単体画像処理エラー: {e}")
            return False

    def build_async(self, pose_map=None, on_progress=None, on_complete=None) -> None:
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

    def _load_sheet(self):
        if not self._sheet_path.exists():
            logger.error(f"シート画像が見つかりません: {self._sheet_path}")
            return None
        try:
            img = Image.open(self._sheet_path).convert("RGBA")
            logger.info(f"シート読み込み完了: {img.width}x{img.height}")
            return img
        except Exception as e:
            logger.error(f"シート読み込みエラー: {e}")
            return None

    def _trim_cell(self, sheet, cell_idx):
        if cell_idx >= self._rows * self._cols:
            return None
        row = cell_idx // self._cols
        col = cell_idx % self._cols
        cw = sheet.width  // self._cols
        ch = sheet.height // self._rows
        return sheet.crop((col*cw, row*ch, (col+1)*cw, (row+1)*ch))

    # 後方互換性エイリアス
    def _remove_checker_bg(self, img):
        return remove_bg(img)


# ============================================================
# モジュール公開関数
# ============================================================

def remove_bg(img: "Image.Image") -> "Image.Image":
    """
    背景タイプを四隅サンプルから自動判定し、BFSで透過化する。
    対応: 暗色(黒)背景 / チェッカー・グレー・白系背景
    """
    if not _PIL_AVAILABLE:
        return img

    if img.mode == "RGBA":
        import numpy as np_
        if np_.array(img.getchannel("A")).min() < 200:
            return img

    import numpy as np
    arr = np.array(img.convert("RGBA")).copy()
    rgb = arr[:, :, :3].astype(np.int32)
    h, w = rgb.shape[:2]

    m = max(3, min(10, h // 8, w // 8))
    corners = np.concatenate([
        arr[:m, :m, :3].reshape(-1, 3),
        arr[:m, -m:, :3].reshape(-1, 3),
        arr[-m:, :m, :3].reshape(-1, 3),
        arr[-m:, -m:, :3].reshape(-1, 3),
    ]).astype(np.float32)
    avg_bright = corners.mean()

    if avg_bright < 40:
        is_bg = np.max(rgb, axis=2) < 50
        logger.debug(f"背景タイプ: 黒(avg={avg_bright:.0f})")
    else:
        drg = np.abs(rgb[:, :, 0] - rgb[:, :, 1])
        dgb = np.abs(rgb[:, :, 1] - rgb[:, :, 2])
        is_gray = (drg < 30) & (dgb < 30)
        lum = (rgb[:, :, 0] + rgb[:, :, 1] + rgb[:, :, 2]) // 3
        is_bg = is_gray & (lum > 120)
        logger.debug(f"背景タイプ: チェッカー/グレー(avg={avg_bright:.0f})")

    visited = np.zeros((h, w), dtype=bool)
    queue: deque = deque()

    def seed(r, c):
        if not visited[r, c] and is_bg[r, c]:
            visited[r, c] = True; queue.append((r, c))

    for r in range(h):
        seed(r, 0); seed(r, w - 1)
    for c in range(w):
        seed(0, c); seed(h - 1, c)

    nb = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    while queue:
        r, c = queue.popleft()
        for dr, dc in nb:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and is_bg[nr, nc]:
                visited[nr, nc] = True; queue.append((nr, nc))

    arr[:, :, 3][visited] = 0
    logger.debug(f"背景除去: {visited.sum():,}px")
    return Image.fromarray(arr)


def autocrop(img: "Image.Image", padding: int = 20) -> "Image.Image":
    """
    透明ピクセルを除いた最小バウンディングボックスでクロップする。

    ★ これが v2 で欠けていた重要な処理。
    背景除去後の余白を詰めることで normalize() での拡大率が最大化され、
    小さいセル（顔アイコン等）がキャンバス全体に正しく広がる。
    """
    if not _PIL_AVAILABLE:
        return img

    import numpy as np
    arr = np.array(img.convert("RGBA"))
    alpha = arr[:, :, 3]
    mask = alpha > 10

    if not mask.any():
        return img

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin = int(np.where(rows)[0][0])
    rmax = int(np.where(rows)[0][-1])
    cmin = int(np.where(cols)[0][0])
    cmax = int(np.where(cols)[0][-1])

    h, w = arr.shape[:2]
    rmin = max(0, rmin - padding)
    rmax = min(h - 1, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(w - 1, cmax + padding)

    return Image.fromarray(arr[rmin:rmax + 1, cmin:cmax + 1])


def _normalize(img: "Image.Image", size: int = OUTPUT_SIZE) -> "Image.Image":
    """
    size x size に正規化（アスペクト比保持・upscale対応・透明パディング）。

    ★ thumbnail()（縮小専用）→ resize()（拡大対応）に変更。
    """
    if not _PIL_AVAILABLE:
        return img
    from PIL import Image as _Image
    import numpy as np
    canvas = _Image.new("RGBA", (size, size), (0, 0, 0, 0))
    iw, ih = img.width, img.height
    if iw == 0 or ih == 0:
        return canvas
    scale = min(size / iw, size / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    resized = img.resize((nw, nh), _Image.LANCZOS)
    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas.paste(resized, (x, y), resized)
    return canvas


def is_available() -> bool:
    return _PIL_AVAILABLE

def is_rembg_available() -> bool:
    """後方互換性のため残す。本モジュールは rembg を使用しない。"""
    return False

def get_output_size() -> int:
    return OUTPUT_SIZE

def get_display_size() -> int:
    return DISPLAY_SIZE
