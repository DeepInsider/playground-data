# playground-data
Data Generation for Neural Network Playground of Deep Insider.

This project that exists as an aid to the [Nerural Network Playground - Deep Insider][playground page] which was forked from [tensorflow/playground: Deep playground][original page].

[The python module "playground-data" on PyPI for this project is available here][pypi].

## Install the module using `pip`

```bash
pip install playground-data
```

## Usage

```python
print('サンプルのグラフをプロットするためのコード')

from plygdata.datahelper import DataHelper, DatasetType
#dir(DataHelper)
dir(DatasetType) # ClassifyCircleData など指定できるものを探せます
DataHelper.plot_sample(DatasetType.ClassifyCircleData)
```

```python
print('データを生成してグラフ化するための基本コード')

from plygdata.dataset import DataGenerator
from plygdata.datahelper import DataHelper
#dir(DataGenerator)

data_noise=0.0
test_data_ratio = 0.5

# データを生成する
data_array = DataGenerator.classify_two_gauss(noise=data_noise)

# データをトレーニング用とテスト用に指定の割合で分割する（さらに、それぞれのデータをデータと教師ラベルに分離する）
X_train, y_train, X_test, y_test = DataHelper.split_train_test_x_data_label(data_array, test_size=test_data_ratio)

# データをPlayground用標準グラフにプロットする
fig, ax = DataHelper.plot_with_playground_style(X_train, y_train, X_test, y_test)
```

```python
print('主要スタティックメソッドのシグネチャ')

import inspect
inspect.signature(DataHelper.plot_sample)
# <Signature (X_train:list, y_train:list, X_test:list=None, y_test:list=None, figsize:tuple=(5, 5), dpi:int=100)>

inspect.signature(DataGenerator.classify_two_gauss)
# <Signature (noise:float=0.0, numSamples:int=500) -> list>

inspect.signature(DataHelper.split_train_test_x_data_label)
# <Signature (data:list, test_size:float=0.5) -> (<class 'list'>, <class 'list'>, <class 'list'>, <class 'list'>)>

inspect.signature(DataHelper.plot_with_playground_style)
# <Signature (X_train:list, y_train:list, X_test:list=None, y_test:list=None, figsize:tuple=(5, 5), dpi:int=100)>

```

## License

Copyright 2018 Digital Advantage Inc. All Rights Reserved.
Licensed under the Apache License, Version 2.0.

### The licenses of using open-source code

This project uses the JavaScript-to-Python-translation of the following open-source code:

 [Deep playground/src/dataset.ts][dataset.py origin]  
Copyright 2016 Google Inc. All Rights Reserved.  
Licensed under the Apache License, Version 2.0.

 [d3/d3-scale/linear.js][scalelinear.py origin]  
Copyright 2010-2015 Mike Bostock. All rights reserved.  
Licensed under the BSD 3-Clause "New" or "Revised" License.

[playground page]: https://re.deepinsider.jp/playground/index.html
[original page]: https://github.com/tensorflow/playground
[pypi]: https://pypi.org/project/playground-data/
[dataset.py origin]: https://github.com/tensorflow/playground/blob/master/src/dataset.ts
[scalelinear.py origin]: https://github.com/d3/d3-scale/blob/master/src/linear.js
