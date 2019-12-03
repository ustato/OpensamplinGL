# samplinGL
OpenGLでサンプリングを3Dでやってみる


# Installation

```shell
open /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg
brew install glfw3
pip install -r requirements.txt
```

# Description
## システム概要と特徴
このシステムは多次元空間内の点を確率分布に基づいてサンプリングする方法であるマルコフ連鎖モンテカルロ法(Markov Chain Monte Carlo method, MCMC)と，
それを拡張した効率的なサンプリング法のギブスサンプリング(Gibbs sampling)をリアルタイムで可視化する機能を提供する．
ギブスサンプリングを可視化するにあたって，コンピュータグラフィックスライブラリのOpenGLと，
OpenGLをプログラミング言語Pythonで利用できるパッケージPyOpenGLを用いている．
システムの大きな特徴として，3次元ディリクレ分布のサンプリング結果を2次元上に射影し，
1描写におけるサンプル点数を1から9で自由に変更できるほか，
総サンプル点の分散に従って3次元ディリクレ分布の確率密度を簡易的にヒートマップ化している．

## システム動作説明
PyOpneGLを利用するにあたって，OpenGL2.1以降，Python3.6以降の環境導入を推奨する．
また，Pythonのパッケージ管理ツールpipを利用し，ソースコード`requirements.txt`内の各パッケージを導入する必要がある．
その後，ソースコード`source/jupyter/gibbs_sampling.py`を実行する．
実行画面に説明が表示されるが，1から9の数字キーを押すとその数に応じてサンプル点数が変化して毎回描写され，qキーを押すとプログラムが終了する．
実行画面には，総サンプル点が黄緑色の点，平均点が黒色の点，
分散1シグマ範囲が赤色の三角形，分散2シグマ範囲が紫色の三角形，分散3シグマ範囲が青色の三角形で表示される．

## 数理モデルの説明
ギブンサンプリングは3次元ディリクレ分布
<img src="https://latex.codecogs.com/gif.latex?{\rm&space;Dir}&space;\left(&space;\Theta&space;\middle|&space;{\bf&space;a}&space;\right)&space;=&space;\frac{&space;\Gamma&space;\left(&space;\sum_{k=1}^{3}&space;a_{k}&space;\right&space;)&space;}&space;{&space;\Pi_{k=1}^{3}&space;\Gamma&space;\left(&space;a_{k}&space;\right&space;)&space;}&space;\Pi_{k=1}^{3}&space;\theta_{k}^{a_{k}-1},&space;\quad&space;\theta_{k}&space;\ge&space;0,&space;\quad&space;\sum_{k=1}^{3}&space;\theta_{k}&space;=&space;1">
に従う変数
<img src="https://latex.codecogs.com/gif.latex?\Theta&space;&&space;\sim&space;&&space;p&space;\left(&space;\Theta&space;)&space;=&space;p&space;\left(&space;\theta_{1},&space;\theta_{2},&space;\theta_{3}&space;)">
を目標として，
<img src="https://latex.codecogs.com/gif.latex?\theta_{1}^{t&plus;1}&space;&&space;\sim&space;&&space;p&space;\left(&space;\theta_{1}&space;\middle|&space;\theta_{2}^{t},&space;\theta_{3}^{t}&space;\right&space;)">
<img src="https://latex.codecogs.com/gif.latex?\theta_{2}^{t&plus;1}&space;&&space;\sim&space;&&space;p&space;\left(&space;\theta_{2}&space;\middle|&space;\theta_{1}^{t&plus;1},&space;\theta_{3}^{t}&space;\right&space;)">
<img src="https://latex.codecogs.com/gif.latex?\theta_{3}^{t&plus;1}&space;&&space;\sim&space;&&space;p&space;\left(&space;\theta_{3}&space;\middle|&space;\theta_{1}^{t&plus;1},&space;\theta_{2}^{t&plus;1}&space;\right&space;)">
に従って
<img src="https://latex.codecogs.com/gif.latex?{\bf&space;\theta}^{t}">
を得る方法である．
これには，アルゴリズムで利用するマルコフ連鎖が不変分布に収束する十分条件の詳細釣り合い条件(detailed balance)と，そのマルコフモデルがエルゴード性を持つことが要求されている．


## ラフスケッチ


