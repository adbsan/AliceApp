"""
setup_images.py
キャラクター画像の自動セットアップスクリプト。

使用方法:
  1. 以下のファイルを assets/parts/ に配置する:
       assets/parts/alice_default.jpg  (全身立ち絵)
       assets/parts/sheet.jpg          (4x4スプライトシート)
       assets/parts/sheet2.png         (4x4スプライトシート)
       assets/parts/sheet3.png         (4x4スプライトシート) ※任意

  2. 仮想環境を有効化した状態で実行:
       venvAlice\\Scripts\\activate
       python setup_images.py

  3. assets/images/ に以下のPNGが生成される:
       alice_default.png   ... 全身立ち絵
       alice_idle.png      ... 待機（通常顔）
       alice_speaking.png  ... 発話（口開き驚き顔）
       alice_thinking.png  ... 思考（赤面困り顔）
       alice_greeting.png  ... 挨拶（目閉じ笑顔）

背景除去アルゴリズム:
  四隅サンプルで背景タイプを自動判定:
    - 暗色(黒)背景: max(R,G,B) < 50 の連続領域をBFSで除去
    - チェッカー/グレー系: グレー(R≈G≈B ±30) かつ輝度>120 の連続領域をBFSで除去
  JPEGアーティファクト対応のため閾値を緩和（±30）。
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

ROOT_DIR   = Path(__file__).parent.resolve()
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
# 背景除去（黒背景・チェッカー柄の両対応）
# ============================================================

def remove_bg(img: "Image.Image") -> "Image.Image":
    """
    背景タイプを四隅サンプルから自動判定し、BFSで透過化する。

    対応背景:
      暗色(黒)背景: max(R,G,B) < 50 の連続領域
      チェッカー柄: グレー(R≈G≈B ±30) かつ輝度>120 の連続領域
    """
    # 既存アルファが有効なら再処理しない
    if img.mode == "RGBA":
        a = np.array(img.getchannel("A"))
        if a.min() < 200:
            return img

    arr = np.array(img.convert("RGBA")).copy()
    rgb = arr[:, :, :3].astype(np.int32)
    h, w = rgb.shape[:2]

    # 四隅10×10pxの平均輝度で背景タイプを判定
    m = max(3, min(10, h // 8, w // 8))
    corners = np.concatenate([
        arr[:m, :m, :3].reshape(-1, 3),
        arr[:m, -m:, :3].reshape(-1, 3),
        arr[-m:, :m, :3].reshape(-1, 3),
        arr[-m:, -m:, :3].reshape(-1, 3),
    ]).astype(np.float32)
    avg_bright = corners.mean()

    if avg_bright < 40:
        # ─── 暗色(黒)背景 ───
        is_bg = np.max(rgb, axis=2) < 50
        print(f"    → 背景タイプ: 黒(avg={avg_bright:.0f})")
    else:
        # ─── チェッカー/グレー/白系 ───
        drg = np.abs(rgb[:, :, 0] - rgb[:, :, 1])
        dgb = np.abs(rgb[:, :, 1] - rgb[:, :, 2])
        is_gray = (drg < 30) & (dgb < 30)          # JPEG対応: ±30
        lum = (rgb[:, :, 0] + rgb[:, :, 1] + rgb[:, :, 2]) // 3
        is_bg = is_gray & (lum > 120)
        print(f"    → 背景タイプ: チェッカー/グレー(avg={avg_bright:.0f})")

    # BFS: 四隅・端辺から連続背景領域を探索
    visited = np.zeros((h, w), dtype=bool)
    q: deque = deque()

    def seed(r: int, c: int) -> None:
        if not visited[r, c] and is_bg[r, c]:
            visited[r, c] = True
            q.append((r, c))

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
                visited[nr, nc] = True
                q.append((nr, nc))

    arr[:, :, 3][visited] = 0
    removed = int(visited.sum())
    print(f"    → 除去: {removed:,}px ({removed / (h * w) * 100:.1f}%)")
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
    return sheet.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch))


# ============================================================
# ポーズマッピング設定
# ============================================================
#
# セルレイアウト（4×4 グリッド、左上=cell#0、右→下方向）:
#   sheet.jpg / sheet2.png:
#     row0(cell 0-3) : 顔表情
#     row1(cell 4-7) : 手・小物
#     row2(cell 8-11): 胴体ポーズ（ほぼ頭なし）
#     row3(cell12-15): パーツ（髪・スカート等）
#
# ─── 確認済みセル内容 ───────────────────────────────────
#   alice_default.jpg          : 全身立ち絵（チェッカー/白系背景）
#   sheet2.png  cell#0         : 通常顔・微笑み  → idle
#   sheet2.png  cell#1         : 驚き口開き顔    → speaking
#   sheet2.png  cell#3         : 赤面困り顔      → thinking
#   sheet.jpg   cell#0         : 目閉じ笑顔      → greeting
#
# (ソースキー, cell_idx, rows, cols, 出力ファイル名)
POSE_CONFIG = [
    ("default", 0, 1, 1, "alice_default"),   # 全身立ち絵
    ("sheet2",  0, 4, 4, "alice_idle"),       # 通常顔・微笑み
    ("sheet2",  1, 4, 4, "alice_speaking"),   # 驚き口開き顔
    ("sheet2",  3, 4, 4, "alice_thinking"),   # 赤面困り顔
    ("sheet",   0, 4, 4, "alice_greeting"),   # 目閉じ笑顔
]


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
        for ext in ["", ".jpg", ".jpeg", ".png", ".webp"]:
            p = PARTS_DIR / (fname + ext)
            if p.exists():
                print(f"  読み込み: {p.name}")
                img = Image.open(p).convert("RGBA")
                print(f"    サイズ: {img.width}x{img.height}")
                return img
        return None

    sources = {
        "default": load("alice_default"),
        "sheet":   load("sheet"),
        "sheet2":  load("sheet2"),
        "sheet3":  load("sheet3"),
    }

    if sources["sheet3"] is None:
        print("  ⚠ sheet3 未検出 → sheet で代替します")
        sources["sheet3"] = sources["sheet"]

    missing = [k for k, v in sources.items() if v is None and k != "sheet3"]
    if missing:
        print(f"\n[ERROR] 必須ファイルが見つかりません: {missing}")
        print("  assets/parts/ に配置してください。")
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
                part = src.copy()
            else:
                part = trim_cell(src, cell_idx, rows, cols)

            print(f"  [{out_name}]  ← {src_key} cell#{cell_idx}  ({part.width}x{part.height}px)")
            nobg = remove_bg(part)

            # 透過確認
            a_arr = np.array(nobg.getchannel("A"))
            opaque = int((a_arr > 200).sum())
            if opaque == 0:
                print(f"    ⚠ 警告: 透過後に描画ピクセルが0件です。閾値を確認してください。")

            norm = normalize(nobg, 2048)
            out_path = IMAGES_DIR / f"{out_name}.png"
            norm.save(out_path, "PNG")

            # 最終確認
            final_arr = np.array(norm)
            final_opaque = int((final_arr[:, :, 3] > 200).sum())
            print(f"    ✅ 保存: {out_path.name}  描画px={final_opaque:,}")
            ok_count += 1

        except Exception as e:
            print(f"    ❌ エラー: {e}")
            import traceback
            traceback.print_exc()

    print()
    print(f"完了: {ok_count}/{len(POSE_CONFIG)} 件")
    print(f"出力先: {IMAGES_DIR}")
    print()
    print("次のステップ: python AliceApp.py")


if __name__ == "__main__":
    main()
