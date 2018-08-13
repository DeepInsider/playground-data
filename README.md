playground-data
====================================

Data Generation for Neural Network Playground of Deep Insider.

This project/package that exists as an aid to the [Nerural Network Playground - Deep Insider][playground page] which was forked from [tensorflow/playground: Deep playground][original page].

Official pages
-------------------------------------------------------------------

- [The python package "playground-data" on PyPI for this project is available here][pypi].
- [The source for this package is available here][src]. [![Build Status](https://travis-ci.org/DeepInsider/playground-data.svg?branch=master)](https://travis-ci.org/DeepInsider/playground-data)

Requirements
-------------------------------------------------------------------

- Python 2: 2.7+ | Python 3: 3.4, 3.5, 3.6+
- numpy
- matplotlib

Install this package using `pip`
-------------------------------------------------------------------

```bash
pip install playground-data
```

Usage
-------------------------------------------------------------------

```python
from __future__ import print_function

print('Import plygdata package as pg')

import plygdata as pg

# Or, you can 'import' class directly like this:
# from plygdata.datahelper import DataHelper, DatasetType
# from plygdata.dataset import DataGenerator
```

```python
print('Code for plotting sample graph')

#dir(pg.DataHelper)    # How to find class members
#dir(pg.DataGenerator)

dir(pg.DatasetType)
#['ClassifyCircleData',
# 'ClassifySpiralData',
# 'ClassifyTwoGaussData',
# 'ClassifyXORData',
# 'RegressGaussian',
# 'RegressPlane',
# ...]

pg.DataHelper.plot_sample(pg.DatasetType.ClassifyCircleData)
```

```python
print('Basic code for generating and graphing data')

data_noise=0.0
test_data_ratio = 0.5

# Generate data
data_array = pg.DataGenerator.classify_two_gauss(noise=data_noise)
#data_array = pg.DataGenerator.classify_circle(noise=data_noise)
#data_array = pg.DataGenerator.classify_spiral(noise=data_noise)
#data_array = pg.DataGenerator.classify_xor(noise=data_noise)
#data_array = pg.DataGenerator.regress_gaussian(noise=data_noise)
#data_array = pg.DataGenerator.regress_plane(noise=data_noise)

# Divide the data for training and testing at a specified ratio (further, separate each data into Coordinate point data part and teacher label part)
X_train, y_train, X_test, y_test = pg.DataHelper.split_train_test_x_data_label(data_array, test_size=test_data_ratio)

# Plot the data on the standard graph for Playground
fig, ax = pg.DataHelper.plot_with_playground_style(X_train, y_train, X_test, y_test)
```

```python
print('Signature of main @staticmethod')

import sys
if sys.version_info[0] < 3.3: # inspect.signature was introduced at version Python 3.3
  !pip install funcsigs

try:
    from inspect import signature
except ImportError:
    from funcsigs import signature

print('pg.DataHelper.plot_sample', str(signature(pg.DataHelper.plot_sample)))
# pg.DataHelper.plot_sample (data_type, visualize_test_data=False, noise=0.0, test_size=0.5, figsize=(5, 5), dpi=100)

print('pg.DataGenerator.classify_two_gauss', str(signature(pg.DataGenerator.classify_two_gauss)))
# pg.DataGenerator.classify_two_gauss (noise=0.0, numSamples=500)

print('pg.DataHelper.split_train_test_x_data_label', str(signature(pg.DataHelper.split_train_test_x_data_label)))
# pg.DataHelper.split_train_test_x_data_label (data, test_size=0.5, label_num=1)

print('pg.DataHelper.plot_with_playground_style', str(signature(pg.DataHelper.plot_with_playground_style)))
# pg.DataHelper.plot_with_playground_style (X_train, y_train, X_test=None, y_test=None, figsize=(5, 5), dpi=100)
```

```python
print('Imported "playground-data" package version is ...')

print(pg.__version__)
```

License
-------------------------------------------------------------------

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

[playground page]: https://deepinsider.github.io/playground/
[original page]: https://github.com/tensorflow/playground
[src]: https://github.com/DeepInsider/playground-data
[pypi]: https://pypi.org/project/playground-data/
[dataset.py origin]: https://github.com/tensorflow/playground/blob/master/src/dataset.ts
[scalelinear.py origin]: https://github.com/d3/d3-scale/blob/master/src/linear.js
