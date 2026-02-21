"""
setup_images.py
キャラクター画像の自動セットアップスクリプト。

使用方法:
  1. 以下のファイルを assets/parts/ に配置する:
       assets/parts/alice_default.jpg  (全身立ち絵)
       assets/parts/sheet.jpg          (4x4スプライトシート)
       assets/parts/sheet2.png         (4x4スプライトシート)

  2. 仮想環境を有効化した状態で実行:
       venvAlice\\Scripts\\activate
       python setup_images.py

  3. assets/images/ に以下のPNGが生成される:
       alice_default.png   ... 全身立ち絵
       alice_idle.png      ... 待機（通常顔）
       alice_speaking.png  ... 発話（口開き驚き顔）
       alice_thinking.png  ... 思考（赤面困り顔）
       alice_greeting.png  ... 挨拶（目閉じ笑顔）

修正点（v3）:
  - autocrop() 追加: 背景除去後の透明余白を自動トリミング
  - normalize() 修正: thumbnail()（縮小専用）→ resize()（拡大対応）
    これにより小さいセル画像（256x256等）が適切にフルサイズに拡大される
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

ROOT_DIR   = Path(__file__).parent.resolve()
PARTS_DIR  = ROOT_DIR / "assets" / "parts"
IMAGES_DIR = ROOT_DIR / "assets" / "images"

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("[ERROR] Pillow / numpy が未インストールです。")
    print("  pip install Pillow numpy")
    sys.exit(1)


def remove_bg(img: "Image.Image") -> "Image.Image":
    """
    背景タイプを四隅サンプルから自動判定し、BFSで透過化する。
    対応: 暗色(黒)背景 / チェッカー・グレー・白系背景
    """
    if img.mode == "RGBA":
        a = np.array(img.getchannel("A"))
        if a.min() < 200:
            return img

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
        print(f"    → 背景: 黒(avg={avg_bright:.0f})")
    else:
        drg = np.abs(rgb[:, :, 0] - rgb[:, :, 1])
        dgb = np.abs(rgb[:, :, 1] - rgb[:, :, 2])
        is_gray = (drg < 30) & (dgb < 30)
        lum = (rgb[:, :, 0] + rgb[:, :, 1] + rgb[:, :, 2]) // 3
        is_bg = is_gray & (lum > 120)
        print(f"    → 背景: チェッカー/グレー(avg={avg_bright:.0f})")

    visited = np.zeros((h, w), dtype=bool)
    q: deque = deque()

    def seed(r: int, c: int) -> None:
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
    removed = int(visited.sum())
    print(f"    → BFS除去: {removed:,}px ({removed / (h * w) * 100:.1f}%)")
    return Image.fromarray(arr)


def autocrop(img: "Image.Image", padding: int = 20) -> "Image.Image":
    """
    透明ピクセルを除いた最小バウンディングボックスでクロップする。

    ★ これが v2 で欠けていた重要な処理。
    背景除去後の余白を詰めることで normalize() の拡大率が最大化され、
    小さいセル（顔アイコン等）がキャンバス全体に正しく広がる。
    """
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

    result = Image.fromarray(arr[rmin:rmax + 1, cmin:cmax + 1])
    print(f"    → クロップ: {img.width}x{img.height} → {result.width}x{result.height}")
    return result


def normalize(img: "Image.Image", size: int = 2048) -> "Image.Image":
    """
    size x size の正方形に正規化する（アスペクト比保持・透明パディング）。

    ★ v2 の thumbnail()（縮小専用）を resize()（拡大対応）に変更。
    小さい画像が size に拡大されるようになった。
    """
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    iw, ih = img.width, img.height
    if iw == 0 or ih == 0:
        return canvas

    scale = min(size / iw, size / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    resized = img.resize((nw, nh), Image.LANCZOS)

    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas.paste(resized, (x, y), resized)
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
#   alice_default.jpg    : 全身立ち絵（白/チェッカー背景）
#   sheet2.png  cell#0   : 通常顔・微笑み  → idle
#   sheet2.png  cell#1   : 驚き口開き顔    → speaking
#   sheet2.png  cell#3   : 赤面困り顔      → thinking
#   sheet.jpg   cell#0   : 目閉じ笑顔      → greeting
#
# (ソースキー, cell_idx or None, rows, cols, 出力ファイル名)
POSE_CONFIG = [
    ("default", None, 1, 1, "alice_default"),
    ("sheet2",  0,    4, 4, "alice_idle"),
    ("sheet2",  1,    4, 4, "alice_speaking"),
    ("sheet2",  3,    4, 4, "alice_thinking"),
    ("sheet",   0,    4, 4, "alice_greeting"),
]


def main() -> None:
    print("=" * 55)
    print("Alice AI キャラクター画像セットアップ v3")
    print("=" * 55)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    def load(fname: str):
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

    ok_count = 0
    for src_key, cell_idx, rows, cols, out_name in POSE_CONFIG:
        src = sources.get(src_key)
        if src is None:
            print(f"  ⚠ スキップ: {out_name}")
            continue

        try:
            part = src.copy() if cell_idx is None else trim_cell(src, cell_idx, rows, cols)
            label = f"cell#{cell_idx}" if cell_idx is not None else "全体"
            print(f"  [{out_name}]  ← {src_key} {label}  ({part.width}x{part.height}px)")

            # 1. 背景除去
            nobg = remove_bg(part)
            # 2. 透明余白オートクロップ（★必須: 小セルをキャンバス全体に拡大するために必要）
            cropped = autocrop(nobg, padding=20)
            # 3. 2048x2048 正規化（upscale対応）
            norm = normalize(cropped, 2048)

            out_path = IMAGES_DIR / f"{out_name}.png"
            norm.save(out_path, "PNG")

            final_opaque = int((np.array(norm)[:, :, 3] > 200).sum())
            print(f"    ✅ 保存: {out_path.name}  描画px={final_opaque:,}")
            ok_count += 1

        except Exception as e:
            print(f"    ❌ エラー: {e}")
            import traceback; traceback.print_exc()

    print()
    print(f"完了: {ok_count}/{len(POSE_CONFIG)} 件")
    print(f"出力先: {IMAGES_DIR}")
    print()
    print("次のステップ: python AliceApp.py")


if __name__ == "__main__":
    main()
