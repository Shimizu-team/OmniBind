# CLAUDE.md — OmniBind

## Project Overview

OmniBind: Unified, 3D-structure-informed framework for pan-pharmacological prediction of compound-protein interactions (CPIs). Predicts four affinity metrics (Ki, Kd, IC50, EC50) simultaneously using protein 3Di sequences via adaptive gated fusion.

**Publication-quality repository** for the paper:
> "Unified and 3D-Structure-Informed Pan-Pharmacological Prediction of Drug-Target Interactions by OmniBind"

## Repository Structure (Target)

```
OmniBind/
├── README.md
├── CLAUDE.md
├── LICENSE                 # MIT
├── .gitignore
├── requirements.txt
├── setup.py or pyproject.toml
├── omnibind/
│   ├── __init__.py
│   ├── model/              # Architecture (gated fusion, prediction heads)
│   ├── data/               # Data loading, preprocessing, 3Di encoding
│   ├── train.py
│   ├── predict.py
│   └── utils/
├── scripts/                # Paper experiment reproduction scripts
├── configs/                # Training/evaluation configs (YAML)
├── notebooks/              # Demo notebooks (optional)
├── data/
│   └── README.md           # Download instructions (NOT raw data)
├── checkpoints/
│   └── README.md           # Download links (NOT weight files)
├── assets/                 # Figures for README
└── tests/
```

## README Structure (Nature系列準拠 + Ubicon参照)

README.mdは以下の順序・内容を必ず含めること:

### 1. Header
- プロジェクトタイトル + 論文タイトル
- アーキテクチャ図 (`assets/`から参照)
- バッジ (License, Python version, etc.)

### 2. System Requirements (Nature必須)
- ソフトウェア依存関係とバージョン番号
- テスト済みOS・バージョン
- 必要な非標準ハードウェア (GPU要件等)
- Python バージョン

### 3. Installation Guide (Nature必須)
- 番号付きステップバイステップ手順
- conda/pip環境構築コマンド
- **標準的なデスクトップでの一般的なインストール時間**を明記

### 4. Demo (Nature必須)
- サンプルデータでの実行手順
- **期待される出力**を具体的に記載
- **標準的なデスクトップでのデモ実行時間**を明記

### 5. Usage
- 独自データでのソフトウェア実行方法
- 入力ファイル形式の説明
- コマンドライン引数の説明

### 6. Reproduction (オプション, 推奨)
- 論文の実験を再現する手順
- random seedの指定方法

### 7. Directory Structure
- ツリー形式 (plain text)

### 8. Citation
- BibTeX形式

### 9. License
- MIT License

## Development Guidelines

### Code Style
- Python 3.9+
- PEP 8 準拠
- 型ヒント: 関数シグネチャに必須
- Docstrings: Google style
- Import順序: stdlib → third-party → local

### Commit Conventions
- Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`
- Atomic commits

### Key Technical Details
- Protein structures → 3Di sequences (Foldseek format)
- Adaptive gated fusion: amino acid + 3Di representations
- Training data: BindingDB
- 4 prediction heads: Ki, Kd, IC50, EC50
- Inference: millisecond-scale (proteome-wide screening対応)

## Security & Quality Checklist

### 秘密情報の除去 (公開前に必ず確認)
- [ ] APIキー、トークンが含まれていないこと
- [ ] 個人パス (e.g., `/home/username/...`) がハードコードされていないこと
- [ ] サーバーアドレス、内部URLが含まれていないこと
- [ ] `.env`ファイルが.gitignoreに含まれていること
- **公開前コマンド**: `claude "リポジトリ内にAPIキー、個人パス、サーバーアドレスなどが含まれていないかチェックしてください"`

### 再現性
- [ ] Random seedの固定 (コード内 + READMEに記載)
- [ ] データ前処理パイプラインが完全に明文化されている
- [ ] 環境構築手順で同一環境を再構築可能
- [ ] requirements.txtにバージョン番号を固定 (`package==x.y.z`)

### .gitignore必須項目
- `__pycache__/`, `*.pyc`
- `*.pt`, `*.pth`, `*.ckpt` (大規模モデルファイル)
- `data/*.csv`, `data/*.sdf` 等の大規模データ
- `.env`, `*.log`
- `.DS_Store`, `Thumbs.db`
- `wandb/`, `mlruns/` 等の実験トラッキング

### 大規模ファイル
- モデル重みやデータセットは直接コミットしない
- Git LFS or ダウンロードスクリプト/リンクで提供
- `checkpoints/README.md` と `data/README.md` にDL手順を記載
