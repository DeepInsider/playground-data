playground-data
====================================

Data Generation for Neural Network Playground of Deep Insider.

This project/package that exists as an aid to the [Nerural Network Playground - Deep Insider][playground page] which was forked from [tensorflow/playground: Deep playground][original page].

Official pages
-------------------------------------------------------------------

- [The python package "playground-data" on PyPI for this project is available here][pypi].
- [The source for this package is available here][src].

Requirements
-------------------------------------------------------------------

- Python 3.11+
- NumPy 2.0+
- Matplotlib 3.9+

Install this package using `pip`
-------------------------------------------------------------------

```bash
pip install playground-data
```

Usage
-------------------------------------------------------------------

```python
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
from inspect import signature

print('pg.plot_sample', str(signature(pg.plot_sample)))
# pg.plot_sample (data_type, noise=0.0, validation_size=0.5, visualize_validation_data=False, figsize=(5, 5), dpi=100, node_id=None, discretize=False, training_size=-1.0)

print('pg.generate_data', str(signature(pg.generate_data)))
# pg.generate_data (data_type, noise=0.0)

print('pg.split_data', str(signature(pg.split_data)))
# pg.split_data (data, validation_size=0.5, label_num=1, training_size=-1.0)

print('pg.plot_points_with_playground_style', str(signature(pg.plot_points_with_playground_style)))
# pg.plot_points_with_playground_style (X_train, y_train, X_valid=None, y_valid=None, figsize=(5, 5), dpi=100)

print('pg.get_playground_figure', str(signature(pg.get_playground_figure)))
# pg.get_playground_figure (enable_colorbar=False)

print('pg.get_playground_axes', str(signature(pg.get_playground_axes)))
# pg.get_playground_axes (fig)

print('pg.plot_points', str(signature(pg.plot_points)))
# pg.plot_points (ax, X_train, y_train, X_valid=None, y_valid=None)

print('pg.draw_decision_boundary', str(signature(pg.draw_decision_boundary)))
# pg.draw_decision_boundary (fig, ax, node_id='x', trained_model=None, discretize=False, enable_colorbar=True)

```

Development with uv
-------------------------------------------------------------------

The Usage examples and all six dataset alternatives are covered by automated tests.
See [PyPI example validation](docs/pypi-example-validation.md) for the published 1.1.1 example issues.

Version 1.2.0 uses uv and a project-local `.venv`; Conda is not required.

```bash
uv python install 3.15
uv sync --locked
uv run pytest -q
uv build --no-sources --out-dir dist/release
uv run twine check --strict dist/release/*
```

Commit `uv.lock` and `.python-version`. Python 2 and Python 3.4–3.10 are no
longer supported. Consumers can continue using `pip install playground-data`.
For article notebooks, see [locked Jupyter environments and execution instructions](notebooks/README.md).

See [development and release instructions](docs/development.md) and
[release notes](CHANGELOG.md). The Web demo uses TensorFlow.js 4.22.0 and D3 7.9.0.

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