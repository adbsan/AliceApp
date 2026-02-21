"""
image_builder_module.py
スプライトシートからキャラクター画像を生成する外装モジュール。

処理パイプライン（直列・逆流禁止）:
  1. Load Sheet    - assets/parts/sheet.png を読み込む
  2. Trim Parts    - グリッド指定でパーツをトリミング
  3. Remove BG     - チェッカー柄背景をフラッドフィルで除去（rembg不要）
  4. Normalize     - 2048x2048 に正規化（1:1・透明パディング）
  5. Save          - assets/images/ に alice_*.png として保存

責務:
  - スプライトシートの分割
  - チェッカー柄背景の除去（アルゴリズム的アプローチ・AI依存なし）
  - 2048x2048 正規化・保存

制約:
  - 推論を実行しない
  - 設定参照は env_binder_module 経由のみ
  - 出力サイズは常に 2048x2048（表示リサイズは window_module が担当）
  - rembg / onnxruntime に依存しない

背景除去アルゴリズム:
  チェッカー柄は「明色（白系 ~255）」と「暗色（グレー ~197）」の2色からなる。
  暗色タイルは衣装の白エプロン（RGB ~255）と明確に区別可能。
  四隅から BFS フラッドフィルでグレー系連続領域を背景として検出し、
  alpha=0 に設定することで白エプロン・フリルを損なわずに除去する。
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
OUTPUT_SIZE   = 2048          # 保存する正規化サイズ（px）
DISPLAY_SIZE  = 512           # window_module で表示するサイズ（px）

# パーツインデックス → ファイル名のマッピング（デフォルト）
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
    """
    スプライトシートからキャラクター画像を生成するクラス。

    Usage:
        builder = ImageBuilder(sheet_path, rows=4, cols=4)
        results = builder.build(pose_map={0: "alice_default", ...})
    """

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
        """
        スプライトシートを分割し、背景除去・正規化・保存を行う。

        Args:
            pose_map:    {セルインデックス: ファイル名(拡張子なし)} の辞書
            on_progress: 進捗コールバック (current, total, message)

        Returns:
            {ファイル名: 成功フラグ} の辞書
        """
        if not _PIL_AVAILABLE:
            logger.error("Pillow / numpy が利用できないため処理を中止します。")
            return {}

        _pose_map = pose_map or DEFAULT_POSE_MAP
        results: Dict[str, bool] = {}

        # Step 1: Load Sheet
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
                # Step 2: Trim
                cell = self._trim_cell(sheet, cell_idx)
                if cell is None:
                    logger.warning(f"セル {cell_idx} が範囲外です。スキップします。")
                    results[name] = False
                    continue

                # Step 3: Remove BG（チェッカー柄フラッドフィル除去）
                cell_nobg = self._remove_checker_bg(cell)

                # Step 4: Normalize → 2048x2048
                normalized = self._normalize(cell_nobg)

                # Step 5: Save
                out_path = self._output_dir / f"{name}.png"
                normalized.save(out_path, "PNG")
                logger.info(f"保存完了: {out_path}")
                results[name] = True

            except Exception as e:
                logger.error(f"{name} の処理中にエラー: {e}")
                results[name] = False

        return results

    def build_single(self, image_path: str, output_name: str) -> bool:
        """
        単体画像（スプライトシートでない立ち絵）を処理して保存する。
        alice_default.jpg などの単独ファイルに使用する。

        Args:
            image_path:  入力画像パス
            output_name: 出力ファイル名（拡張子なし）

        Returns:
            成功フラグ
        """
        if not _PIL_AVAILABLE:
            return False
        try:
            img = Image.open(image_path).convert("RGBA")
            logger.info(f"単体画像読み込み: {image_path} ({img.width}x{img.height})")
            nobg = self._remove_checker_bg(img)
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
        """build() を別スレッドで実行する（GUI用）。"""
        def _run():
            results = self.build(pose_map=pose_map, on_progress=on_progress)
            if on_complete:
                on_complete(results)
        threading.Thread(target=_run, daemon=True).start()

    def get_cell_count(self) -> int:
        return self._rows * self._cols

    def preview_cells(self) -> List[Tuple[int, "Image.Image"]]:
        """全セルのサムネイルリストを返す（GUI プレビュー用）。"""
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

    # ============================================================
    # 内部処理
    # ============================================================

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
        """
        セルインデックスからトリミング座標を計算して切り出す。
        セルは左上から右→下の順で 0 始まり。
        """
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

    def _remove_checker_bg(self, img: "Image.Image") -> "Image.Image":
        """
        チェッカー柄背景をフラッドフィルで除去する（rembg不要）。

        アルゴリズム:
          1. 画像をRGBA配列に変換
          2. グレー系ピクセル（R≈G≈B、輝度>150）を背景候補とする
             ※ チェッカーの明色(~255)・暗色(~197)は両方グレー系
             ※ キャラクターの白エプロンも白だが、外部からの連続性で区別
          3. 四隅+端辺からBFSで連続したグレー系領域を背景として検出
          4. 検出領域のアルファを0（透明）に設定
          5. 白エプロン・フリルはキャラクター内部にあるため連続しない → 保持

        注意: 既にアルファチャンネルが有効な画像（RGBA・実透明あり）の場合は
              既存のアルファを優先し、チェッカー処理はスキップする。

        Returns:
            背景透明のRGBA画像
        """
        if not _PIL_AVAILABLE:
            return img

        # 既存アルファが有効かチェック（非JPEG源でRGBAなら再処理不要の可能性）
        if img.mode == "RGBA":
            alpha_arr = np.array(img.getchannel("A"))
            if alpha_arr.min() < 200:
                # 意味のある透明ピクセルが既に存在する → そのまま使用
                logger.debug("既存アルファチャンネルを使用します。")
                return img

        arr = np.array(img.convert("RGBA")).copy()
        rgb = arr[:,:,:3].astype(np.int32)
        h, w = rgb.shape[:2]

        # グレー系判定: R≈G≈B（彩度が低い）かつ輝度 > 150
        diff_rg = np.abs(rgb[:,:,0] - rgb[:,:,1])
        diff_gb = np.abs(rgb[:,:,1] - rgb[:,:,2])
        is_gray = (diff_rg < 20) & (diff_gb < 20)
        brightness = rgb[:,:,0]
        is_bg_candidate = is_gray & (brightness > 150)

        # BFS: 四隅・端辺から連続する背景候補領域を探索
        visited = np.zeros((h, w), dtype=bool)
        queue = deque()

        def seed(r, c):
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
        neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        while queue:
            r, c = queue.popleft()
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                    if is_bg_candidate[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

        # 背景領域をアルファ0（透明）に設定
        arr[:, :, 3][visited] = 0

        return Image.fromarray(arr)

    def _normalize(self, img: "Image.Image") -> "Image.Image":
        """
        画像を OUTPUT_SIZE x OUTPUT_SIZE（2048x2048）の正方形に正規化する。
        アスペクト比を保持し、余白は透明パディング。
        """
        canvas = Image.new("RGBA", (OUTPUT_SIZE, OUTPUT_SIZE), (0, 0, 0, 0))
        img.thumbnail((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)
        x = (OUTPUT_SIZE - img.width)  // 2
        y = (OUTPUT_SIZE - img.height) // 2
        canvas.paste(img, (x, y), img if img.mode == "RGBA" else None)
        return canvas


# ============================================================
# モジュールレベル関数
# ============================================================

def is_available() -> bool:
    """画像生成に必要なライブラリが揃っているか確認する。"""
    return _PIL_AVAILABLE

def is_rembg_available() -> bool:
    """
    後方互換性のために残す。
    本モジュールは rembg を使用しないため常に False を返す。
    rembg なしで正しく動作する。
    """
    return False

def get_output_size() -> int:
    return OUTPUT_SIZE

def get_display_size() -> int:
    return DISPLAY_SIZE
