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

#### ローカルモデルを使う場合（任意）
- `.env` で `AI_BACKEND=local` または `AI_BACKEND=auto` を設定
- `LOCAL_MODEL_REPO` / `LOCAL_MODEL_FILE` を指定
- 起動時に Hugging Face から GGUF をダウンロードして利用
- 既定値は TinyLlama GGUF（`TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`）

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
- 2026-03-01 02:55:01 JST
  - ローカル学習済みモデル（GGUF）をWeb（Hugging Face）から自動DLして利用する機能を追加
  - `AI_BACKEND`（auto/gemini/local）を追加し、Gemini失敗時のローカルLLMフォールバックを実装
  - Local LLM 推論クラス (`src/AI/local_heart.py`) を追加し、既存チャットUIからそのまま対話可能に拡張
  - 設定画面に `AIバックエンド` と `Local` タブを追加（Repo/File/Dir/Context/Token等を編集可能）
  - 設定保存時に Gemini/Local ローダーのキャッシュをリセットする処理を追加
- 2026-03-01 02:46:30 JST
  - 起動時にデスクトップモードを強制適用し、初回表示からチャット欄が見えるように修正
  - デスクトップモード復帰時にチャットペイン分割を再補正する処理を追加（入力欄欠落の再発防止）
  - 入力欄の存在チェックと再構築ガードを追加し、ユーザー入力UIを常に利用可能に修正
  - 設定画面の AIモデル をコンボ選択化（主要 Gemini Flash 系モデルを選択可能）
  - AI接続失敗時に利用可能モデルを自動選択して再接続を試行する起動処理を追加
  - AI未接続時にチャット上へ設定案内メッセージを表示するよう改善
- 2026-03-01 01:54:48 JST
  - キャラクターモード切替を実装（表示メニューから Desktop / Character を切替可能）
  - ログフォルダを開く処理をクロスプラットフォーム化（Windows / macOS / Linux / Android / iOS フォールバック）
  - レイヤーのドラッグ移動操作を実装（合成プレビュー上で左ドラッグ）
  - 音声キューイングを実装（VOICEVOX 発話を順番再生、stopで未再生キューもクリア）
  - ブロック方式（全セル一括処理）で未設定セルにプレースホルダー画像を自動補完
