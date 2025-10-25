# ライティングコンテストで作成したサンプルコード

このリポジトリは、[InterSystems 開発者ウェビナー「ベクトル検索のご紹介」（2024年5月30日開催）](https://community.intersystems.com/)  及び「IRISのベクトル検索を使ってテキストから画像を検索してみよう」で紹介された内容を、実際に **IRIS Community Edition** と **Python** を組み合わせて再現した実装例です。

Python初心者が、動画をもとに「まずやってみよう！」の精神で構築した記録兼サンプルになります。  
同じように試してみたい方の参考になれば幸いです。

---

##  プロジェクト概要

本リポジトリでは以下の2種類のベクトル検索を実装しています。

| 種類 | 内容 | 使用モデル |
|------|------|-------------|
| **テキスト検索** | テキスト同士の類似検索 | `stsb-xlm-r-multilingual` |
| **テキストから画像検索** | テキスト入力に近い画像を検索 | `clip-ViT-B-32`（画像） / `clip-ViT-B-32-multilingual-v1`（テキスト） |

---

##  動作環境

| 項目 | バージョン例 |
|------|---------------|
| OS | Windows Server 2019 |
| IRIS | IRIS Community 2025.2.0.227.0 |
| Python | 3.12.10 |
| 開発環境 | VS Code（推奨） |

> IRIS 2025.2 では Studio および Apache Web Server が廃止されているため、  
> IIS と VS Code の環境構築を行ってください。

---

##  Pythonライブラリのインストール

```bash
pip install datasets==2.19.0
pip install tensorflow
pip install tensorflow-datasets==4.8.3
pip install pyarrow
pip install pandas
pip install sentence_transformers
pip install tf-keras
pip install requests
pip install sqlalchemy-iris
````

> IRISとの接続には [`sqlalchemy-iris`](https://pypi.org/project/sqlalchemy-iris/) を使用します。

---

## 実行内容

### ① テキストベクトル検索

* Hugging Faceの `cc100` データセットを使用し、Parquet形式で保存
* SentenceTransformer によるテキスト埋め込み
* IRIS にベクトル型カラムとして保存し、`VECTOR_DOT_PRODUCT` によりドット積検索を実行

```bash
python vectorsearch.py 大都市での生活は便利な反面、混雑や環境の悪さなどの問題もある。
```

### ② テキストから画像検索

* `recruit-jp/japanese-image-classification-evaluation-dataset` を使用
* URL画像を取得 → `clip-ViT-B-32` によるベクトル化
* IRISで `VECTOR_COSINE` によるコサイン類似度検索を実行

```bash
python imagesearch.py 黄色い花
```

IRIS 側のクラスメソッドや REST API による検索にも対応しています。

---

## 📁 主なファイル構成

```
.
├ source
│   ├── html
│   │   └── index.html        # 画像検索用htmlファイル
│   │
│   ├── ObjectScript
│   │   ├── ImageData.cls     # 画像検索用データクラス
│   │   ├── Rest.cls          # RESTSサービス用クラス
│   │   └── Vector.cls        # Vector検索用クラス
│   │
│   └── Python
│        ├── download_cc100.py # cc100データセット作成スクリプト
│        ├── download_image.py # 画像データセット作成スクリプト
│        ├── imagesearch.py    # 画像検索要スクリプト
│        ├── imagetest.py      # 画像検索データ作成スクリプト
│        ├── vectorsearch.py   # 文字検索用スクリプト
│        └── vectortest.py     # 文字検索用データ作成スクリプト
│
└── README.md
```


---

## ⚙️ 補足情報

* cc100データセットは非常に大きいため、**約200GBの空き容量**と**10時間以上の処理時間**が必要です。
* IRIS側でPythonライブラリを別途インストールする必要はないとの事ですが、もし動作しない場合は以下のように手動インストールしてください：

  ```bash
  python -m pip --target <iris_dir>\mgr\python <module>
  ```
