# ==============================================================================
# Copyright 2018 Digital Advantage Inc. All Rights Reserved.
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
from plygdata.datacolor import DataColor
from plygdata.dataset import DataGenerator
from plygdata.playground import Player


RECT_DOMAIN = [-6.0, 6.0, -6.0, 6.0]
POINT_DOMAIN = [-6.0, 6.0]
TICKS_POINT = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
TICKS_MIDDLE = 6
TICKS_VALUE = [-1, 0, 1]

class DatasetType:
    ClassifyCircleData = "circle"
    ClassifyXORData = "xor"
    ClassifyTwoGaussData = "gauss"
    ClassifySpiralData = "spiral"
    RegressPlane = "reg-plane"
    RegressGaussian = "reg-gauss"

class InputType:
    X1 = "x"
    X2 = "y"
    X1Squared = "xSquared"
    X2Squared = "ySquared"
    X1TimesX2 = "xTimesY"
    SinX1 = "sinX"
    SinX2 = "sinY"


class DataHelper:

    @staticmethod
    def split_train_test_x_data_label(data, test_size = 0.5, label_num = 1): #  -> (list, list, list, list)

        len_data = len(data)
        if len_data == 0:
            raise ValueError("No array error.")

        mat = np.array(data)

        # It is easy to use scikit-learn.
        # X = mat[:,:-(label_num)] # data
        # y = mat[:,-(label_num)]  # label
        # from sklearn.model_selection import train_test_split
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        # But, because this class doesn't use scikit-learn.
        np.random.shuffle(mat)

        X = mat[:, :-(label_num)]  # data
        y = mat[:, -(label_num)]   # label

        split_point = int(len(mat) * (1.0 - test_size))
        X_train = X[0:split_point, :] # train data
        y_train = y[0:split_point]    # train label
        X_test = X[split_point:, :]   # test data
        y_test = y[split_point:]      # test label

        if label_num == 1:
            y_train = np.reshape(y_train, [len(y_train), 1])
            y_test = np.reshape(y_test, [len(y_test), 1])

        # To Pandas example:
        # import numpy as np
        # import pandas as pd
        # df_X_train = pd.DataFrame(X_train)
        # df_X_train.columns = ['x1', 'x2']
        # print(df_X_train)
        # df_y_train = pd.DataFrame(y_train)
        # df_y_train.columns = ['label']
        # print(df_y_train)

        return X_train, y_train, X_test, y_test


    @staticmethod
    def get_playground_axes(fig): # : fg.Figure
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


    @staticmethod
    def get_all_boundaries(boundaries=None, discretize=False):
        boundaries = Player.update_decision_boundary(boundaries, discretize)
        return boundaries


    @staticmethod
    def draw_decision_boundary_of_node(fig, ax, boundary_of_node): # : fg.Figure
        cmap, norm = DataColor.get_colormap()
        alpha = (160 / 255)
        im = ax.imshow(boundary_of_node, extent=RECT_DOMAIN, cmap=cmap, norm=norm, alpha=alpha, interpolation='Bicubic')
        divider = make_axes_locatable(ax)
        ax_cb = divider.append_axes("bottom", size="5%", pad=0.3)
        ax_cb.tick_params(axis='x', colors='#777777')
        ax_cb.tick_params(axis='y', colors='#777777')
        cb = fig.colorbar(im, cax=ax_cb, ticks=TICKS_VALUE, orientation='horizontal')
        cb.outline.set_edgecolor('white')
        cb.outline.set_linewidth(0)
        cb.solids.set_edgecolor("face")
        ax_cb.xaxis.set_ticks_position("bottom")


    @staticmethod
    def draw_decision_boundary(fig, ax, node_id=InputType.X1, discretize=False): # : fg.Figure
        boundaries = DataHelper.get_all_boundaries(None, discretize)
        DataHelper.draw_decision_boundary_of_node(fig, ax, boundaries[node_id])


    @staticmethod
    def plot_with_playground_style(X_train, y_train, X_test = None, y_test = None, figsize = (5, 5), dpi = 100):
        """
        if width 5 inches x height 5 inches x dpi 100, = 500 x 500 dots figure will be created.

        :param X_train: train data
        :param y_train: grain label
        :param X_test: test data
        :param y_test: test label
        :param figsize: width inches (default: 5) x height (default: 5) inches
        :param dpi: dpi (default: 100)
        :return: a Figure object of matplotlib,  and the axes (Coordinate axis)
        """
        fig = plt.figure(figsize=figsize, dpi=dpi)

        # get the axes (Coordinate axis) designed for NeuralNetwork Playground of Deep Insider.
        ax = DataHelper.get_playground_axes(fig)

        for i in range(len(X_train)):
            ax.scatter(X_train[i, 0], X_train[i, 1], s=20, c=DataColor.get_rgb(y_train[i]), alpha=0.9, linewidths=0.7,
                       edgecolors="#ffffff")

        if all([X_test is not None, y_test is not None]):
            for n in range(len(X_test)):
                ax.scatter(X_test[n, 0], X_test[n, 1], s=20, c=DataColor.get_rgb(y_test[n]), alpha=0.9, linewidths=0.7,
                       edgecolors="#555555")

        return fig, ax


    @staticmethod
    def plot_sample(data_type, visualize_test_data = False, noise = 0.0, test_size = 0.5, figsize = (5, 5), dpi = 100):

        if data_type == DatasetType.ClassifyTwoGaussData:
            data_array = DataGenerator.classify_two_gauss(noise=noise)
        elif data_type == DatasetType.ClassifyXORData:
            data_array = DataGenerator.classify_xor(noise=noise)
        elif data_type == DatasetType.ClassifyCircleData:
            data_array = DataGenerator.classify_circle(noise=noise)
        elif data_type == DatasetType.ClassifySpiralData:
            data_array = DataGenerator.classify_spiral(noise=noise)
        elif data_type == DatasetType.RegressPlane:
            data_array = DataGenerator.regress_plane(noise=noise)
        elif data_type == DatasetType.RegressGaussian:
            data_array = DataGenerator.regress_gaussian(noise=noise)
        else:
            return None, None

        mat = np.array(data_array)
        X_train, y_train, X_test, y_test = DataHelper.split_train_test_x_data_label(mat, test_size=test_size)
        fig, ax = DataHelper.plot_with_playground_style(X_train, y_train, X_test if (visualize_test_data) else None, y_test if (visualize_test_data) else None)
        return fig, ax
