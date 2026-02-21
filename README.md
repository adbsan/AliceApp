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
assets/images/alice_idle.png      ... 待機ポーズ（任意）
assets/images/alice_speaking.png  ... 話し中ポーズ（任意）
assets/images/alice_thinking.png  ... 考え中ポーズ（任意）
assets/images/alice_greeting.png  ... 挨拶ポーズ（任意）
assets/parts/sheet.png            ... 4x4スプライトシート（任意）
```

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
