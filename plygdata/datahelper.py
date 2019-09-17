# ==============================================================================
# Copyright 2018 Digital Advantage Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import division

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from plygdata.dataset import generate_data
from plygdata.state import InputType
from plygdata.heatmap import HeatMap


POINT_DOMAIN = [-6.0, 6.0]
TICKS_POINT = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
TICKS_MIDDLE = 6
TICKS_VALUE = [-1, 0, 1]


def split_data(data, validation_size = 0.5, label_num = 1, training_size = -1.0):
    '''
    Split data int training and validation data. And Each data will be split into input data and teacher labels.
    :param data:
    :param validation_size:
    :param label_num:
    :param training_size:
    :return: (X_train, y_train, X_valid, y_valid)
    '''

    len_data = len(data)
    if len_data == 0:
        raise ValueError("No array error.")

    mat = np.array(data)

    # It is easy to use scikit-learn.
    # X = mat[:,:-(label_num)] # data
    # y = mat[:,-(label_num)]  # label
    # from sklearn.model_selection import train_test_split
    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=validation_size)

    # But, because this class doesn't use scikit-learn.
    np.random.shuffle(mat)

    X = mat[:, :-(label_num)]  # data
    y = mat[:, -(label_num)]   # label

    if training_size >= 0.0 and training_size <= 1.0:
        validation_size = 1.0 - training_size

    split_point = int(len(mat) * (1.0 - validation_size))
    X_train = X[0:split_point, :] # train data
    y_train = y[0:split_point]    # train label
    X_valid = X[split_point:, :]   # validation data
    y_valid = y[split_point:]      # validation label

    if label_num == 1:
        y_train = np.reshape(y_train, [len(y_train), 1])
        y_valid = np.reshape(y_valid, [len(y_valid), 1])

    # To Pandas example:
    # import numpy as np
    # import pandas as pd
    # df_X_train = pd.DataFrame(X_train)
    # df_X_train.columns = ['x1', 'x2']
    # print(df_X_train)
    # df_y_train = pd.DataFrame(y_train)
    # df_y_train.columns = ['label']
    # print(df_y_train)

    return X_train, y_train, X_valid, y_valid


def get_playground_figure(enable_colorbar=False):
    if enable_colorbar:
        fig = plt.figure(figsize=(6, 6), dpi=100)
    else:
        fig = plt.figure(figsize=(5, 5), dpi=100)
    return fig


def get_playground_axes(fig):
    ax = fig.add_subplot(111)  # Row: 1, Column: 1, Place: 1
    ax.set_facecolor("#e8eaeb")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.tick_right()
    ax.tick_params(axis='x', colors='#bbbbbb')
    ax.tick_params(axis='y', colors='#bbbbbb')
    ax.set_xlim(POINT_DOMAIN)
    ax.set_ylim(POINT_DOMAIN)
    ax.set_xticks(TICKS_POINT)
    ax.set_yticks(TICKS_POINT)
    ax.get_xticklabels()[TICKS_MIDDLE].set_color("#333333")
    ax.get_yticklabels()[TICKS_MIDDLE].set_color("#333333")
    #ax.grid(color='white', linestyle='-')
    ax.grid(b=False)
    return ax


def plot_points(ax, X_train, y_train, X_valid=None, y_valid=None):

    HeatMap.updateTrainPoints(ax, X_train, y_train)

    if all([X_valid is not None, y_valid is not None]):
        HeatMap.updateValidationPoints(ax, X_valid, y_valid)


def plot_points_with_playground_style(X_train, y_train, X_valid=None, y_valid=None, figsize=(5, 5), dpi=100):
    """
    if width 5 inches x height 5 inches x dpi 100, = 500 x 500 dots figure will be created.

    :param X_train: train data
    :param y_train: grain label
    :param X_valid: validation data
    :param y_valid: validation label
    :param figsize: width inches (default: 5) x height (default: 5) inches
    :param dpi: dpi (default: 100)
    :return: a Figure object of matplotlib,  and the axes (Coordinate axis)
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # get the axes (Coordinate axis) designed for NeuralNetwork Playground of Deep Insider.
    ax = get_playground_axes(fig)

    plot_points(ax, X_train, y_train, X_valid, y_valid)

    return fig, ax


def _add_colorbar_on_bottom(fig, ax, im):
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes("bottom", size="5%", pad=0.3)
    ax_cb.tick_params(axis='x', colors='#777777')
    ax_cb.tick_params(axis='y', colors='#777777')
    cb = fig.colorbar(im, cax=ax_cb, ticks=TICKS_VALUE, orientation='horizontal')
    cb.outline.set_edgecolor('white')
    cb.outline.set_linewidth(0)
    cb.solids.set_edgecolor("face")
    ax_cb.xaxis.set_ticks_position("bottom")


def draw_decision_boundary(fig, ax, node_id=InputType.X1, discretize=False, enable_colorbar=True):

    im = HeatMap.updateBackground(ax, None, node_id, discretize)

    if enable_colorbar:
        _add_colorbar_on_bottom(fig, ax, im)

    return im


def plot_sample(data_type, noise=0.0, validation_size=0.5, visualize_validation_data=False, figsize=(5, 5), dpi=100, node_id=None, discretize=False):

    data_array = generate_data(data_type, noise)
    if data_array is None:
        return None, None

    mat = np.array(data_array)
    X_train, y_train, X_valid, y_valid = split_data(mat, validation_size=validation_size)

    fig, ax = plot_points_with_playground_style(X_train, y_train, X_valid if (visualize_validation_data) else None, y_valid if (visualize_validation_data) else None, figsize=figsize, dpi=dpi)

    if node_id is not None:
        draw_decision_boundary(fig, ax, node_id=node_id, discretize=discretize)

    return fig, ax

def predict_classes(model, x, batch_size=32, verbose=0):
    """Generate class predictions for the input samples.
    The input samples are processed batch by batch.
    # Arguments
        x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
        batch_size: integer.
        verbose: verbosity mode, 0 or 1.
    # Returns:
        A numpy array of class predictions.
    """
    preds = model.predict(x, batch_size=batch_size, verbose=verbose)
    discretized = np.frompyfunc(lambda x: 1 if (x >= 0.0) else -1, 1, 1)(preds)
    return discretized

def predict_proba(model, x, batch_size=32, verbose=0):
    """Generates class probability predictions for the input samples.
    The input samples are processed batch by batch.
    # Arguments
        x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
        batch_size: integer.
        verbose: verbosity mode, 0 or 1.
    # Returns
        A Numpy array of probability predictions.
    """
    probability = model.predict(x, batch_size=batch_size, verbose=verbose)
    return probability

def predict_classes_proba(model, x, batch_size=32, verbose=0):
    """Generate class and probability predictions for the input samples.
    The input samples are processed batch by batch.
    # Arguments
        x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
        batch_size: integer.
        verbose: verbosity mode, 0 or 1.
    # Returns:
        A numpy array of class and probability predictions.
    """
    probability = model.predict(x, batch_size=batch_size, verbose=verbose)
    discretized = np.frompyfunc(lambda x: 1 if (x >= 0.0) else -1, 1, 1)(probability)
    return discretized, probability
