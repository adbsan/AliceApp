"""
setup_images.py
キャラクター画像の自動セットアップスクリプト。

使用方法:
  1. 以下のファイルを assets/parts/ に配置する:
       assets/parts/alice_default.jpg  (全身立ち絵)
       assets/parts/sheet.jpg          (4x4スプライトシート - 表情・全身)
       assets/parts/sheet2.png         (4x4スプライトシート - 表情・パーツ)
       assets/parts/sheet3.png         (4x4スプライトシート - 追加表情) ※任意

  2. 仮想環境を有効化した状態で実行:
       venvAlice\\Scripts\\activate
       python setup_images.py

  3. assets/images/ に以下のPNGが生成される:
       alice_default.png   ... 全身立ち絵（IDLE/DEFAULT フォールバック）
       alice_idle.png      ... 待機ポーズ（顔・微笑み）
       alice_speaking.png  ... 発話ポーズ（全身・斜め）
       alice_thinking.png  ... 思考ポーズ（全身・正面）
       alice_greeting.png  ... 挨拶ポーズ（顔・正面）

注意:
  - sheet3.png が存在しない場合は sheet.jpg で代替します
  - 画像の cell 番号は assets/parts/ 配置のシートに合わせて変更してください
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import deque

ROOT_DIR = Path(__file__).parent.resolve()
PARTS_DIR  = ROOT_DIR / "assets" / "parts"
IMAGES_DIR = ROOT_DIR / "assets" / "images"

# ============================================================
# 依存チェック
# ============================================================
try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("[ERROR] Pillow / numpy が未インストールです。")
    print("  pip install Pillow numpy")
    sys.exit(1)


# ============================================================
# 背景除去アルゴリズム（image_builder_module と同一）
# ============================================================

def remove_checker_bg(img: "Image.Image") -> "Image.Image":
    """チェッカー柄背景を四隅BFSフラッドフィルで除去する。"""
    arr = np.array(img.convert("RGBA")).copy()
    rgb = arr[:, :, :3].astype(np.int32)
    h, w = rgb.shape[:2]

    # グレー系判定（R≈G≈B かつ輝度>150）
    is_gray = (np.abs(rgb[:,:,0]-rgb[:,:,1]) < 20) & (np.abs(rgb[:,:,1]-rgb[:,:,2]) < 20)
    is_bg   = is_gray & (rgb[:,:,0] > 150)

    visited = np.zeros((h, w), dtype=bool)
    q = deque()

    for r in range(h):
        for c in [0, w-1]:
            if not visited[r,c] and is_bg[r,c]:
                visited[r,c] = True; q.append((r,c))
    for c in range(w):
        for r in [0, h-1]:
            if not visited[r,c] and is_bg[r,c]:
                visited[r,c] = True; q.append((r,c))

    nb = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    while q:
        r,c = q.popleft()
        for dr,dc in nb:
            nr,nc = r+dr,c+dc
            if 0<=nr<h and 0<=nc<w and not visited[nr,nc] and is_bg[nr,nc]:
                visited[nr,nc] = True; q.append((nr,nc))

    arr[:,:,3][visited] = 0
    return Image.fromarray(arr)


def normalize(img: "Image.Image", size: int = 2048) -> "Image.Image":
    """2048x2048の正方形に正規化（アスペクト比保持・透明パディング）。"""
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    img.thumbnail((size, size), Image.LANCZOS)
    x = (size - img.width)  // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y), img)
    return canvas


def trim_cell(sheet: "Image.Image", cell_idx: int,
              rows: int, cols: int) -> "Image.Image":
    """スプライトシートからセルを切り出す（左上から右→下、0始まり）。"""
    cw, ch = sheet.width // cols, sheet.height // rows
    r, c   = cell_idx // cols, cell_idx % cols
    return sheet.crop((c*cw, r*ch, (c+1)*cw, (r+1)*ch))


# ============================================================
# ポーズマッピング設定
# ============================================================
# ★ ここを変更して自分のシートに合わせる ★
#
# 利用可能なシート:
#   default_img : 全身立ち絵（単体ファイル）
#   sheet_img   : sheet.jpg  (4x4, 16セル)
#   sheet2_img  : sheet2.png (4x4, 16セル)
#   sheet3_img  : sheet3.png (4x4, 16セル) ※存在すれば
#
# (ソースキー, cell_idx, rows, cols, 出力ファイル名)
POSE_CONFIG = [
    ("default", 0, 1, 1, "alice_default"),   # 全身立ち絵
    ("sheet2",  0, 4, 4, "alice_idle"),       # 顔・微笑み (sheet2 cell#0)
    ("sheet2",  8, 4, 4, "alice_speaking"),   # 全身・斜め (sheet2 cell#8)
    ("sheet",   9, 4, 4, "alice_thinking"),   # 全身・白エプロン (sheet cell#9)
    ("sheet",   0, 4, 4, "alice_greeting"),   # 顔・正面 (sheet cell#0)
]
# ★ sheet3.png がある場合の例:
# ("sheet3", 0, 4, 4, "alice_idle"),     # sheet3 cell#0 を使う場合


# ============================================================
# メイン処理
# ============================================================

def main() -> None:
    print("=" * 55)
    print("Alice AI キャラクター画像セットアップ")
    print("=" * 55)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # ── ソース画像ロード ──────────────────────────────────
    def load(fname: str) -> "Image.Image | None":
        for ext in ["", ".png", ".jpg", ".jpeg", ".webp"]:
            p = PARTS_DIR / (fname + ext)
            if p.exists():
                print(f"  読み込み: {p}")
                return Image.open(p).convert("RGBA")
        return None

    sources = {}
    sources["default"] = load("alice_default")
    sources["sheet"]   = load("sheet")
    sources["sheet2"]  = load("sheet2")
    sources["sheet3"]  = load("sheet3")

    # sheet3 が無い場合は sheet で代替
    if sources["sheet3"] is None:
        print("  ⚠ sheet3 未検出 → sheet で代替します")
        sources["sheet3"] = sources["sheet"]

    missing = [k for k, v in sources.items()
               if v is None and k not in ("sheet3",)]
    if missing:
        print(f"\n[ERROR] 必須ファイルが見つかりません: {missing}")
        print(f"  assets/parts/ に配置してください。")
        sys.exit(1)

    print()

    # ── ポーズ生成 ──────────────────────────────────────
    ok_count = 0
    for src_key, cell_idx, rows, cols, out_name in POSE_CONFIG:
        src = sources.get(src_key)
        if src is None:
            print(f"  ⚠ スキップ: {out_name} (ソース '{src_key}' 未検出)")
            continue

        try:
            if rows == 1 and cols == 1:
                part = src
            else:
                part = trim_cell(src, cell_idx, rows, cols)

            print(f"  処理中: {out_name}  [{src_key} cell#{cell_idx}]", end=" ... ", flush=True)
            nobg = remove_checker_bg(part)
            norm = normalize(nobg, 2048)

            out_path = IMAGES_DIR / f"{out_name}.png"
            norm.save(out_path, "PNG")

            arr = np.array(norm)
            opaque = (arr[:,:,3] > 0).sum()
            print(f"✅ ({opaque:,}px)")
            ok_count += 1

        except Exception as e:
            print(f"❌ エラー: {e}")

    print()
    print(f"完了: {ok_count}/{len(POSE_CONFIG)} 件")
    print(f"出力先: {IMAGES_DIR}")
    print()
    print("次のステップ: python AliceApp.py")


if __name__ == "__main__":
    main()
