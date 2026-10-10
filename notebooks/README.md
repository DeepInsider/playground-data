# 記事ノートブックの実行・検証

記事の原本は `docs/articles/` と `docs/sample_notebooks/` に残しています。
`validated/` は現在のライブラリで実行するためのコピーです。
修正は `patches.json` に原本のセル番号・理由とともに記録しています。
コピーは `scripts/validate_notebooks.py --prepare` で再生成できます。
原本を変更した場合はハッシュ検査で検出します。

## 環境

Python は **3.12.15** に固定し、依存関係は環境ごとの `uv.lock` に記録します。
パッケージ本体の開発環境（Python 3.15）とは独立しています。
各環境はローカルの playground-data 1.2.0 を使用し、PyPI の旧版へ戻しません。

| 環境 | 対象 |
|---|---|
| `environments/base` | 基本サンプル、NumPy・Pandas・データ生成の記事4冊 |
| `environments/tensorflow` | TensorFlow 2.21.0 / Keras 3 の記事5冊 |
| `environments/pytorch` | PyTorch 2.14.1 の記事1冊 |

## 再実行

リポジトリのルートで実行します。

```powershell
uv python install 3.12.15
uv sync --project notebooks/environments/base --locked
uv sync --project notebooks/environments/tensorflow --locked
uv sync --project notebooks/environments/pytorch --locked
powershell -File scripts/setup_graphviz.ps1
uv run --project notebooks/environments/base --locked python scripts/validate_notebooks.py base --prepare
uv run --project notebooks/environments/tensorflow --locked python scripts/validate_notebooks.py tensorflow --prepare
uv run --project notebooks/environments/pytorch --locked python scripts/validate_notebooks.py pytorch --prepare
```

Windows の Graphviz は公式 ZIP と固定 SHA256 を使い `.artifacts/graphviz/` に配置します。
グローバルなインストールは不要です。Linux/macOS では別途 `dot` を導入してください。
TensorFlow は CPU で検証します。MNIST は初回にインターネットから取得し、
`.artifacts/notebooks/keras/` に保存します。
学習回数やデータ件数は省略せず、記事の指定を使います。
Estimator の旧章は廃止APIのため明示した Keras 版への移植です。

実行済み `.ipynb`・HTML・ログ・モデル図は `.artifacts/notebooks/` に保存されます。
原本や検証コピーへ実行出力を上書きしません。
通常の実行失敗は終了コード1、教材で意図した例外は種類を照合して別記します。
`deeplearningpyt` の `data.csv` がない場合の例外2件が該当します。
初回に作った `work/` 内に `data.csv` を置くと、この例外の再現検査は失敗します。

## JupyterLab で開く

上記の検証を実行した後、対応する環境から実行済みコピーを開きます。

```powershell
uv run --project notebooks/environments/base --locked jupyter lab .artifacts/notebooks/executed/basic.ipynb
uv run --project notebooks/environments/tensorflow --locked jupyter lab .artifacts/notebooks/executed/tf2_keras_neuralnetwork.ipynb
uv run --project notebooks/environments/pytorch --locked jupyter lab .artifacts/notebooks/executed/pytorch_neuralnetwork.ipynb
```

JupyterLab のカーネルには、その環境の Python 3 を選んでください。
冒頭のセルで Python、plygdata のバージョンとインポート先を確認できます。
最初から実行するには「カーネルを再起動してすべてのセルを実行」を選択します。
ノートブック内の無指定 `!pip install` や `%tensorflow_version` は無効化してあります。
変更する依存は専用環境の `pyproject.toml` で管理し、ロック更新後に再検証してください。
Colab のダウンロードセルはローカルのファイルリンクへ置き換えています。

実行結果と移植内容は [検証結果](../docs/notebook-validation.md) を参照してください。
