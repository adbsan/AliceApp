"""
image_builder_module.py
スプライトシートからキャラクター画像を生成する外装モジュール。

変更点（v4）:
  - AdvancedImageProcessor（window_module.py）と連携
  - 独自アルゴリズムによる高精度背景除去・エッジ検出を使用
  - autocrop / normalize は processor 内部メソッドに委譲
  - 後方互換性のため remove_bg() / autocrop() / _normalize() は保持
"""

from __future__ import annotations

import threading
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


def _get_processor():
    """AdvancedImageProcessor を遅延インポートして返す（循環インポート防止）。"""
    try:
        from module.window_module import AdvancedImageProcessor
        return AdvancedImageProcessor()
    except ImportError:
        return None


class ImageBuilder:
    """スプライトシートからキャラクター画像を生成するクラス。"""

    def __init__(self, sheet_path: str, rows: int, cols: int,
                 output_dir: str = "assets/images") -> None:
        self._sheet_path = Path(sheet_path)
        self._rows = rows
        self._cols = cols
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._processor = _get_processor()

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

                if self._processor is not None and self._processor.is_available():
                    # AdvancedImageProcessor を使用
                    arr = np.array(cell.convert("RGBA"))
                    removed_arr = self._processor.remove_background_adaptive(arr)
                    cropped_arr = self._processor._autocrop_array(removed_arr)
                    norm_arr    = self._processor._normalize_array(cropped_arr, OUTPUT_SIZE)
                    norm        = Image.fromarray(norm_arr)
                else:
                    # フォールバック: 既存アルゴリズム
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
            img = Image.open(image_path).convert("RGBA")

            if self._processor is not None and self._processor.is_available():
                arr         = np.array(img)
                removed_arr = self._processor.remove_background_adaptive(arr)
                cropped_arr = self._processor._autocrop_array(removed_arr)
                norm_arr    = self._processor._normalize_array(cropped_arr, OUTPUT_SIZE)
                norm        = Image.fromarray(norm_arr)
            else:
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

    def _remove_checker_bg(self, img):
        """後方互換エイリアス"""
        return remove_bg(img)


# ============================================================
# モジュール公開関数（後方互換性のため保持）
# ============================================================

def remove_bg(img: "Image.Image") -> "Image.Image":
    """
    背景除去（後方互換）。
    AdvancedImageProcessor が利用可能な場合はそちらを優先使用。
    """
    if not _PIL_AVAILABLE:
        return img

    processor = _get_processor()
    if processor is not None and processor.is_available():
        try:
            arr = np.array(img.convert("RGBA"))
            result_arr = processor.remove_background_adaptive(arr)
            return Image.fromarray(result_arr)
        except Exception as e:
            logger.warning(f"AdvancedImageProcessor 失敗、フォールバック使用: {e}")

    # フォールバック: 旧BFSアルゴリズム
    return _remove_bg_legacy(img)


def _remove_bg_legacy(img: "Image.Image") -> "Image.Image":
    """旧来のBFS背景除去（フォールバック用）。"""
    if not _PIL_AVAILABLE:
        return img

    from collections import deque

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
    else:
        drg = np.abs(rgb[:, :, 0] - rgb[:, :, 1])
        dgb = np.abs(rgb[:, :, 1] - rgb[:, :, 2])
        is_gray = (drg < 30) & (dgb < 30)
        lum = (rgb[:, :, 0] + rgb[:, :, 1] + rgb[:, :, 2]) // 3
        is_bg = is_gray & (lum > 120)

    visited = np.zeros((h, w), dtype=bool)
    q: deque = deque()

    def seed(r, c):
        if not visited[r, c] and is_bg[r, c]:
            visited[r, c] = True; q.append((r, c))

    for r in range(h):
        seed(r, 0); seed(r, w - 1)
    for c in range(w):
        seed(0, c); seed(h - 1, c)

    nb = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    while q:
        r, c = q.popleft()
        for dr, dc in nb:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and is_bg[nr, nc]:
                visited[nr, nc] = True; q.append((nr, nc))

    arr[:, :, 3][visited] = 0
    logger.debug(f"背景除去(legacy): {visited.sum():,}px")
    return Image.fromarray(arr)


def autocrop(img: "Image.Image", padding: int = 20) -> "Image.Image":
    """透明余白を自動クロップ（後方互換）。"""
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
    """正規化（後方互換）。"""
    if not _PIL_AVAILABLE:
        return img
    from PIL import Image as _Image
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
    """後方互換性のため残す。"""
    return False

def get_output_size() -> int:
    return OUTPUT_SIZE

def get_display_size() -> int:
    return DISPLAY_SIZE
