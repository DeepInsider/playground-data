# 記事ノートブック検証結果

検証日: 2026-10-10。Windows、CPU、Python 3.12.15。

10冊、セットアップを含む395コードセルを先頭から実行しました。
予期しない実行エラーは0件です。教材が意図した FileNotFoundError 2件は別記しました。
学習回数・データ件数を減らすスモーク実行ではありません。

| ノートブック | 環境 | 実行コードセル | 予期しないエラー | 意図した例外 |
|---|---|---:|---:|---:|
| `basic` | base | 14 | 0 | 0 |
| `deeplearning1g1t_lesson08` | base | 36 | 0 | 0 |
| `deeplearning1g2t_lesson01` | base | 20 | 0 | 0 |
| `deeplearningdat` | base | 24 | 0 | 0 |
| `deeplearning1g1t_lesson07` | tensorflow | 48 | 0 | 0 |
| `deeplearningpyt` | tensorflow | 102 | 0 | 2 |
| `tf2_keras_neuralnetwork` | tensorflow | 37 | 0 | 0 |
| `tf2_keras_regression` | tensorflow | 20 | 0 | 0 |
| `tf2_keras_howtowrite` | tensorflow | 62 | 0 | 0 |
| `pytorch_neuralnetwork` | pytorch | 32 | 0 | 0 |

## 固定環境

| 環境 | 主なバージョン |
|---|---|
| base | playground-data 1.2.0, numpy 2.5.3, matplotlib 3.11.2, pandas 3.0.6, jupyterlab 4.6.4 |
| tensorflow | playground-data 1.2.0, numpy 2.5.3, matplotlib 3.11.2, pandas 3.0.6, tensorflow 2.21.0, keras 3.15.1, jupyterlab 4.6.4 |
| pytorch | playground-data 1.2.0, numpy 2.5.3, matplotlib 3.11.2, pandas 3.0.6, torch 2.14.1, torchvision 0.29.1, jupyterlab 4.6.4 |

Python のパッチ版は各 `.python-version`、全依存は環境別 `uv.lock` で固定しています。
Graphviz 16.1.0 は公式 ZIP の SHA256 を照合してプロジェクト内に展開しました。

## 修正と原本の扱い

- 原本の `.ipynb` は変更せず、ハッシュが一致することを確認しました。
- 基本サンプルの旧関数名を `generate_data` / `split_data` / `plot_points_with_playground_style` に更新しました。
- Windows の `os.uname()` を `platform.uname()` に置き換えました。
- 無指定 pip install、古い Colab マジック、シェルの環境操作は専用 uv 環境の手順へ移しました。
- Keras 3 の重み初期化、モデル入力、compile 引数、配列入力、指標の状態リセット、optimizer 拡張を修正しました。
- TensorFlow 内部モジュールの参照を公開 API に置き換えました。
- 廃止された Estimator の章は **Keras に移植した例**です。Estimator 原文そのものの動作保証ではありません。元の構成と1600学習ステップを保っています。
- 連続値の決定境界描画で見つかった NumPy の不具合はパッケージ本体でも修正し、離散値・連続値の回帰テストを追加しました。
- `deeplearningpyt` の原本セル262・264は存在しない data.csv の例外を示す教材です。FileNotFoundError を確認し、他の例外を成功扱いしません。
- 乱数シード42を設定しました。CPU演算・ライブラリの差により数値や図が記事掲載時と完全一致する保証はありません。

## 保存結果と再現

実行済みノートブック・HTMLは `.artifacts/notebooks/executed/`、学習ログ・モデル図は
`.artifacts/notebooks/work/`、実行結果JSONは `.artifacts/notebooks/*-results.json` にあります。
これらは Git 管理対象外です。検証コピーには実行出力を混入させていません。
実行出力のグラフとHTMLの代表画面を確認しました。長い学習ログはHTMLのみ省略し、
実行済みノートブックには保持しています。

パッケージ本体の23件のテストと、更新したwheelの隔離環境での23件のテストも成功しました。
配布wheel・sdistのtwine check、秘密情報パターン検査、アーカイブへの環境非混入も確認しました。
GitHub の手動 Article notebooks ワークフローは用意しましたが、リモート実行はしていません。
Gitコミット・プッシュ・PyPI公開は未実施です。

JupyterLabの起動と再実行は [ノートブック実行手順](../notebooks/README.md) を参照してください。
