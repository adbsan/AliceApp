# Alice AI

Inspired by Maid-chan from Sakurasou no Pet na Kanojo

## セットアップ

### 1. 仮想環境の作成と有効化
```bash
python -m venv venvAlice
venvAlice\Scripts\activate
pip install -r requirements.txt
```

### 2. 環境設定
```bash
copy .env.example .env
# .env を編集してAPIキーを設定
```

### 3. キャラクター画像の配置
```
assets/images/alice_default.png   ... 全身立ち絵（必須）
```

> **注意**: `assets/images/` に PNG ファイルを直接配置してください。  
> 画像は `1:1` の正方形（推奨: 512×512 以上）、背景透過（RGBA）が最適です。

### 4. 起動
```bash
python AliceApp.py
```

## ディレクトリ構成

```
AliceApp/
├─ AliceApp.py               # エントリポイント（起動制御・実行シーケンス管理）
├─ .env                      # 環境設定（Git管理外）
├─ .env.example              # 設定テンプレート
├─ requirements.txt
│
├─ module/                   # 外装モジュール（物理操作担当）
│   ├─ env_binder_module.py         # .env 読み込み・設定値提供
│   ├─ neural_loader_module.py      # Gemini クライアントロード・検証
│   ├─ prompt_shaper_module.py      # ペイロード構築・Message 型定義
│   ├─ result_log_module.py         # 履歴永続化（JSON）・コンソール出力
│   ├─ display_mode_module.py       # テーマ・レイアウト・アニメーション定義
│   └─ window_module.py             # メインウィンドウ・各ダイアログ
│
└─ src/
    └─ AI/
        └─ heart.py          # AI推論の心臓部（中層構造・外部依存なし）
```

## 設計原則

- AIロジックの完全隔離（`src/AI/heart.py`）
- 単方向データフロー（逆流禁止）
- モジュール責務の厳格分離
- 環境変数は `env_binder_module` 経由でのみ取得
- 外装モジュールは末尾に `_module` を必ず付与

## 責務分離マトリクス

| モジュール               | 物理処理 | 設定参照 | 推論 | 永続化 |
|--------------------------|----------|----------|------|--------|
| module/                  | ○        | ○        | ×    | ○      |
| src/AI/heart.py          | ×        | ×        | ○    | ×      |
| AliceApp.py              | 制御のみ | ×        | ×    | ×      |

## Git運用

- ブランチは常に `testbranch` を使用
- commit は GUI メニュー → Git から実行可能
- push（sync）はユーザーが手動で実行

## 未実装
## クロスプラットフォーム対応未実装
- 各プラットフォーム事のアプリ化

## AIキャラクターのアニメーション化
- GUI上またはアプリ上でキャラクター動き、話、ユーザとコミニュケーションを取る機能が不完全。
- 現状ではキャラクターではなく画像に対してアニメーションが付与されている。

## 更新履歴
- 2026-03-01 01:54:48 JST
  - キャラクターモード切替を実装（表示メニューから Desktop / Character を切替可能）
  - ログフォルダを開く処理をクロスプラットフォーム化（Windows / macOS / Linux / Android / iOS フォールバック）
  - レイヤーのドラッグ移動操作を実装（合成プレビュー上で左ドラッグ）
  - 音声キューイングを実装（VOICEVOX 発話を順番再生、stopで未再生キューもクリア）
  - ブロック方式（全セル一括処理）で未設定セルにプレースホルダー画像を自動補完
