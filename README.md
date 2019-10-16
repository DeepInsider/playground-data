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

- Python 2: 2.7+ | Python 3: 3.4, 3.5, 3.6, 3.7+
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

# Or, you can 'import' classes and functions directly like this:
# from plygdata.datahelper import DatasetType
# from plygdata.dataset import generate
```

```python
print('Imported "playground-data" package version is ...')

print(pg.__version__)
```

```python
print('Code for plotting sample graph')

import pprint
pprint.pprint(dir(pg))    # How to find class members

pprint.pprint(dir(pg.DatasetType))
#['ClassifyCircleData',
# 'ClassifySpiralData',
# 'ClassifyTwoGaussData',
# 'ClassifyXORData',
# 'RegressGaussian',
# 'RegressPlane',
# ...]

fig, ax = pg.plot_sample(pg.DatasetType.ClassifyCircleData)
# # uncomment if a graph is not shown
# import matplotlib.pyplot as plt
# plt.show()
```

```python
print('Basic code for generating and graphing data')

data_noise=0.0
validation_data_ratio = 0.5

# Generate data
data_array = pg.generate_data(pg.DatasetType.ClassifyCircleData, data_noise)
#data_array = pg.generate_data(pg.DatasetType.ClassifyXORData, data_noise)
#data_array = pg.generate_data(pg.DatasetType.ClassifyTwoGaussData, data_noise)
#data_array = pg.generate_data(pg.DatasetType.ClassifySpiralData, data_noise)
#data_array = pg.generate_data(pg.DatasetType.RegressPlane, data_noise)
#data_array = pg.generate_data(pg.DatasetType.RegressGaussian, data_noise)

# Divide the data for training and validating at a specified ratio (further, separate each data into Coordinate point data part and teacher label part)
X_train, y_train, X_valid, y_valid = pg.split_data(data_array, validation_size=validation_data_ratio)
# You can use training_size instead of validation_size. training_size takes precedence over validation_size.

# Plot the data on the standard graph for Playground
fig, ax = pg.plot_points_with_playground_style(X_train, y_train, X_valid, y_valid, figsize = (6, 6), dpi = 100)

# # get figure + axes of matplotlib graph and plot the data points
# fig = pg.get_playground_figure(enable_colorbar=True)
# ax = pg.get_playground_axes(fig)
# pg.plot_points(ax, X_train, y_train, X_valid, y_valid)
# # These 3 lines equal to `plot_points_with_playground_style` function

# draw the decision boundary of X1 input (feature)
pg.draw_decision_boundary(fig, ax, node_id=pg.InputType.X1, discretize=False)

# # uncomment if a graph is not shown
# import matplotlib.pyplot as plt
# plt.show()
```

```python
print('Signature of main @staticmethod')

import sys
if sys.version_info[0] < 3: # inspect.signature was introduced at version Python 3.3
  !pip install funcsigs

try:
    from inspect import signature
except ImportError:
    from funcsigs import signature

print('pg.plot_sample', str(signature(pg.plot_sample)))
# pg.plot_sample (data_type, noise=0.0, validation_size=0.5, visualize_validation_data=False, figsize=(5, 5), dpi=100, node_id=None, discretize=False)

print('pg.generate', str(signature(pg.generate)))
# pg.generate (data_type, noise=0.0)

print('pg.split_data', str(signature(pg.data)))
# pg.split_data (data, validation_size=0.5, label_num=1)

print('pg.plot_points_with_playground_style', str(signature(pg.plot_points_with_playground_style)))
# pg.plot_points_with_playground_style (X_train, y_train, X_valid=None, y_valid=None, figsize=(5, 5), dpi=100)

print('pg.get_playground_figure', str(signature(pg.get_playground_figure)))
# pg.get_playground_figure (enable_colorbar=False)

print('pg.get_playground_axes', str(signature(pg.get_playground_axes)))
# pg.get_playground_axes (fig)

print('pg.plot_points', str(signature(pg.plot_points)))
# pg.plot_points (ax, X_train, y_train, X_valid=None, y_valid=None)

print('pg.draw_decision_boundary', str(signature(pg.draw_decision_boundary)))
# pg.draw_decision_boundary (fig, ax, node_id='x', discretize=False, enable_colorbar=True)

```

Sample Web app
-------------------------------------------------------------------

- [Sample Web app of Neural Network Playground-DATA using TensorFlow.js](https://deepinsider.github.io/playground-data/)

License
-------------------------------------------------------------------

Copyright 2018-2019 Digital Advantage Co., Ltd. All Rights Reserved.
Licensed under the Apache License, Version 2.0.

### The licenses of using open-source code

This project uses the JavaScript-to-Python-translation of the following open-source code:

[tensorflow / playground (Deep playground) / dataset.ts][dataset.py origin], [heatmap.ts][heatmap.py origin], [playground.ts][playground.py origin], [state.ts][state.py origin]  
Copyright 2016 Google Inc. All Rights Reserved.  
Licensed under the Apache License, Version 2.0.

 [d3 / d3-scale / linear.js][scalelinear.py origin]  
Copyright 2010-2015 Mike Bostock. All rights reserved.  
Licensed under the BSD 3-Clause "New" or "Revised" License.

[playground page]: https://deepinsider.github.io/playground/
[original page]: https://github.com/tensorflow/playground
[src]: https://github.com/DeepInsider/playground-data
[pypi]: https://pypi.org/project/playground-data/
[dataset.py origin]: https://github.com/tensorflow/playground/blob/master/src/dataset.ts
[heatmap.py origin]: https://github.com/tensorflow/playground/blob/master/src/heatmap.ts
[playground.py origin]: https://github.com/tensorflow/playground/blob/master/src/playground.ts
[scalelinear.py origin]: https://github.com/d3/d3-scale/blob/master/src/linear.js
[state.py origin]: https://github.com/tensorflow/playground/blob/master/src/state.ts