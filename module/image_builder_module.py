"""
image_builder_module.py
スプライトシートからキャラクター画像を生成する外装モジュール。

処理パイプライン（直列・逆流禁止）:
  1. Load Sheet    - assets/parts/sheet.png を読み込む
  2. Trim Parts    - グリッド指定でパーツをトリミング
  3. Remove BG     - rembg で背景を自動除去（RGBA）
  4. Normalize     - 2048x2048 に正規化（1:1・透明パディング）
  5. Save          - assets/images/ に alice_*.png として保存

責務:
  - スプライトシートの分割
  - rembg による背景除去
  - 2048x2048 正規化・保存

制約:
  - 推論を実行しない
  - 設定参照は env_binder_module 経由のみ
  - 出力サイズは常に 2048x2048（表示リサイズは window_module が担当）
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger

# ============================================================
# 定数
# ============================================================
OUTPUT_SIZE   = 2048          # 保存する正規化サイズ（px）
DISPLAY_SIZE  = 512           # window_module で表示するサイズ（px）

# パーツインデックス → ファイル名のマッピング（デフォルト）
# GUI でユーザーが自由に割り当て可能
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
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    logger.error("Pillow が未インストールです。pip install Pillow を実行してください。")

try:
    from rembg import remove as rembg_remove
    _REMBG_AVAILABLE = True
except ImportError:
    _REMBG_AVAILABLE = False
    logger.warning(
        "rembg が未インストールです。背景除去はスキップされます。"
        " pip install rembg でインストールしてください。"
    )


# ============================================================
# メイン処理クラス
# ============================================================

class ImageBuilder:
    """
    スプライトシートからキャラクター画像を生成するクラス。

    Usage:
        builder = ImageBuilder(sheet_path, rows=2, cols=3)
        results = builder.build(pose_map={0: "alice_default", ...})
    """

    def __init__(
        self,
        sheet_path: str,
        rows: int,
        cols: int,
        output_dir: str = "assets/images",
    ) -> None:
        """
        Args:
            sheet_path: スプライトシート画像のパス
            rows:       シートの行数
            cols:       シートの列数
            output_dir: 出力ディレクトリ
        """
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
                         例: {0: "alice_default", 1: "alice_idle"}
            on_progress: 進捗コールバック (current, total, message)

        Returns:
            {ファイル名: 成功フラグ} の辞書
        """
        if not _PIL_AVAILABLE:
            logger.error("Pillow が利用できないため処理を中止します。")
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

                # Step 3: Remove BG
                cell_nobg = self._remove_bg(cell)

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
        """シートの総セル数を返す。"""
        return self._rows * self._cols

    def preview_cells(self) -> List[Tuple[int, "Image.Image"]]:
        """
        全セルのサムネイルリストを返す（GUI プレビュー用）。

        Returns:
            [(cell_index, PIL.Image), ...]
        """
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
        """シート画像を読み込む。"""
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

        left   = col * cell_w
        upper  = row * cell_h
        right  = left + cell_w
        lower  = upper + cell_h

        return sheet.crop((left, upper, right, lower))

    def _remove_bg(self, img: "Image.Image") -> "Image.Image":
        """
        rembg で背景を除去する。
        rembg 未インストールの場合はそのまま返す。
        """
        if not _REMBG_AVAILABLE:
            logger.debug("rembg 未利用。背景除去をスキップします。")
            return img
        try:
            import io
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            result_bytes = rembg_remove(buf.read())
            result = Image.open(io.BytesIO(result_bytes)).convert("RGBA")
            logger.debug("rembg 背景除去完了。")
            return result
        except Exception as e:
            logger.warning(f"rembg 処理エラー（元画像を使用）: {e}")
            return img

    def _normalize(self, img: "Image.Image") -> "Image.Image":
        """
        画像を OUTPUT_SIZE x OUTPUT_SIZE（2048x2048）の正方形に正規化する。

        - アスペクト比を保持してリサイズ
        - 余白は透明（RGBA Alpha=0）でパディング
        """
        canvas = Image.new("RGBA", (OUTPUT_SIZE, OUTPUT_SIZE), (0, 0, 0, 0))

        # アスペクト比保持でリサイズ
        img.thumbnail((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)

        # 中央配置
        x = (OUTPUT_SIZE - img.width)  // 2
        y = (OUTPUT_SIZE - img.height) // 2
        canvas.paste(img, (x, y), img if img.mode == "RGBA" else None)

        return canvas


# ============================================================
# モジュールレベル関数（AliceApp.py から呼び出し用）
# ============================================================

def is_available() -> bool:
    """画像生成に必要なライブラリが揃っているか確認する。"""
    return _PIL_AVAILABLE

def is_rembg_available() -> bool:
    """rembg が利用可能か確認する。"""
    return _REMBG_AVAILABLE

def get_output_size() -> int:
    """正規化後の出力サイズ（px）を返す。"""
    return OUTPUT_SIZE

def get_display_size() -> int:
    """表示サイズ（px）を返す。"""
    return DISPLAY_SIZE