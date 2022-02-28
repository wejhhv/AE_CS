# 実験手順

- ファイルの実行等はシェル上で行います
- 必要なライブラリは随時インストールしてください
- [tensorflowのドキュメント](https://www.tensorflow.org/tutorials/generative/adversarial_fgsm?hl=ja)より参照
## 1. 開発環境

|  種類  |  バージョン・型  |
| ---- | ---- |
|  OS  |  Windows  |
|  Python  |  3.9.1  |
|  pip  |  21.2.4  |

### Pythonのインストール
[公式サイト](https://www.python.org/downloads/)からPythonをローカル環境にインストール

### 仮想環境の導入
`pip install virtualenv`

### 仮想環境の作成

`virtualenv env`

### 仮想化を有効にする

`env\Scripts\activate`

### 必要なライブラリのインストール

```python 
pip install tensorflow
pip install matplotlib
pip install glob
pip install icrawler
pip install cv2
```

## 2. 画像（データセット）収集

- [flickr](https://www.flickr.com/photos/tags/imagenet/)等の写真共有サイトから探すか
**image_collect.py**ファイルを編集して実行

  `python image_collect.py`

- 直接写真共有サイトで探すときは`animal`フォルダーを作成し、そこに画像を入れて置く

- 画像サイズ比率やサイズの調整


## 3. Adversarial Examplesの作成
- **ad.py**をファイル実行をすると各画像ごとにフォルダーが作成され、その中にAE画像が入る

  `python ad.py`

- ファイル名は`<摂動の大きさ><CNN入力時の出力ラベル>.jpg`で生成される

## 4. 圧縮センシング
- [MATLAB](https://jp.mathworks.com/?s_tid=gn_logo)上で**ori.m**を実行
*MATLABでは**Image Processing Toolbox**を別途インストールする必要がある
- ファイル名`はCS_<摂動の大きさ><CNN入力時の出力ラベル>.jpg`で生成される
## 5. 4のファイルをCNNに入力して比較
  `python check_cs.py`で**check_cs.py**を実行
  <br>
フォルダーの画像ごとに正解数が表示されるようにしている
## 6. その他
|  ファイル・フォルダ名  |  用途  |
| ---- | ---- |
|  t-test.py  |  テンソルの動作確認  |
|  check_label.py  |  単体画像をCNNに入力した際のラベル確認用  |
|  subfuntions  |  ori.mの参照ファイル群  |
|gaussian.py | ガウシアンフィルタ |