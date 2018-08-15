# ==============================================================================
# Copyright 2018 Digital Advantage Inc. All Rights Reserved.
#
# This is a Python implementation of [tensorflow / playground (Deep playground) / heatmap.ts](https://github.com/tensorflow/playground/blob/master/src/heatmap.ts).
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

from plygdata.datacolor import DataColor
from plygdata.playground import Player


RECT_DOMAIN = [-6.0, 6.0, -6.0, 6.0]


class HeatMap:
    '''
     Draws a heatmap using matplotlib.
     Used for showing the learned decision boundary of the classification algorithm.
     Can also draw data points on top of the heatmap figure using matplotlib.
    '''

    @staticmethod
    def draw_decision_boundary_of_node(ax, boundary_of_node):
        cmap, norm = DataColor.get_colormap()
        alpha = (160 / 255)
        im = ax.imshow(boundary_of_node, extent=RECT_DOMAIN, cmap=cmap, norm=norm, alpha=alpha, interpolation='Bicubic')
        return im


    @staticmethod
    def updateBackground(ax, boundaries, node_id, discretize):
        boundaries = Player.update_decision_boundary(boundaries, discretize)
        boundary_of_node = boundaries[node_id]
        im = HeatMap.draw_decision_boundary_of_node(ax, boundary_of_node)
        return im


    @staticmethod
    def updateCircles(ax, X_points, y_label, edgecolors):
        for i in range(len(X_points)):
            ax.scatter(X_points[i, 0], X_points[i, 1], s=20, c=DataColor.get_rgb(y_label[i]), alpha=0.9, linewidths=0.7, edgecolors=edgecolors)


    @staticmethod
    def updateTrainPoints(ax, X_train, y_train):
        HeatMap.updateCircles(ax, X_train, y_train, "#ffffff")


    @staticmethod
    def updateValidationPoints(ax, X_valid, y_valid):
        HeatMap.updateCircles(ax, X_valid, y_valid, "#FF5555")


    @staticmethod
    def updateTestPoints(ax, X_test, y_test):
        HeatMap.updateCircles(ax, X_test, y_test, "#555555")

    # export function reduceMatrix(matrix: number[][], factor: number): number[][] {
    #   if (matrix.length !== matrix[0].length) {
    #     throw new Error("The provided matrix must be a square matrix");
    #   }
    #   if (matrix.length % factor !== 0) {
    #     throw new Error("The width/height of the matrix must be divisible by " +
    #         "the reduction factor");
    #   }
    #   let result: number[][] = new Array(matrix.length / factor);
    #   for (let i = 0; i < matrix.length; i += factor) {
    #     result[i / factor] = new Array(matrix.length / factor);
    #     for (let j = 0; j < matrix.length; j += factor) {
    #       let avg = 0;
    #       // Sum all the values in the neighborhood.
    #       for (let k = 0; k < factor; k++) {
    #         for (let l = 0; l < factor; l++) {
    #           avg += matrix[i + k][j + l];
    #         }
    #       }
    #       avg /= (factor * factor);
    #       result[i / factor][j / factor] = avg;
    #     }
    #   }
    #   return result;
    # }

