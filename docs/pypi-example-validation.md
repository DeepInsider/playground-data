# PyPI使用例の検証

2026-10-10に https://pypi.org/project/playground-data/ と公開JSONの説明文を取得しました。
公開版は1.1.1です。以下は、その掲載コードを更新予定の1.2.0環境で検証した結果です。
公開版1.1.1のインストール検証ではありません。

- Pythonコード5ブロックのうち、インポート・バージョン表示・サンプル描画・データ生成と決定境界描画の4ブロックは変更せず成功しました。
- `!pip install funcsigs` はJupyter/IPython向けの記述であり、想定環境では問題ありません。関数名を修正した公開版の署名表示ブロックをIPythonのセル変換で実行し、成功を確認しました。Python 3ではPython 2向けのインストール分岐は実行されません。
- 同ブロックの `pg.generate` と `pg.data` は存在しないため、個別にAttributeErrorを確認しました。正しくは `pg.generate_data` と `pg.split_data` です。
- 次回配布に使うREADMEの関数名は `pg.generate_data` と `pg.split_data` に修正済みです。全5ブロックを順番に実行し、グラフの出力を確認しました。
- コメントに掲載した代替データセットを含む全6種で、生成・分割・描画を検証しました。

`tests/test_readme_usage.py` はREADMEからコードを直接抽出するため、例の更新時にも実行確認します。
本体環境Python 3.15と配布wheelの隔離環境Python 3.14で、それぞれ全30テストが成功しました。
取得した説明文・結果JSON・描画PNGは `.artifacts/pypi/` に保存しています（Git管理対象外）。

PyPI掲載文自体はまだ更新していません。1.2.0を公開すると、新しいREADMEがプロジェクト説明として反映されます。
