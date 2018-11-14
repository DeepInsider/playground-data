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

from matplotlib.colors import LinearSegmentedColormap, Normalize
import math


_colors = [
    "#f59322", # orange=negative
    "#f4992f",
    "#f39f3d",
    "#f2a44a",
    "#f2aa58",
    "#f1b065",
    "#f0b672",
    "#efbc80",
    "#eec18d",
    "#edc79b",
    "#eccda8",
    "#ebd3b5",
    "#ebd9c3",
    "#eaded0",
    "#e9e4de",
    "#e8eaeb", # white=middle
    "#d9e2e8",
    "#cadbe5",
    "#bbd3e2",
    "#accbdf",
    "#9dc4dc",
    "#8ebcd9",
    "#7fb4d6",
    "#71add2",
    "#62a5cf",
    "#539dcc",
    "#4496c9",
    "#358ec6",
    "#2686c3",
    "#177fc0",
    "#0877bd"  # blue=positive
]


class DataColor:

    @staticmethod
    def get_all():
        return _colors


    @staticmethod
    def get_positive():
        return _colors[0]


    @staticmethod
    def get_neutral():
        return _colors[15]


    @staticmethod
    def get_negative():
        return _colors[30]


    @staticmethod
    def get_rgb(label):
        """
         0 = index_first
         30 = index_last
         15.5 = point_half
        """
        int_label = int(math.floor(15.5 * (label + 1)))
        index = int(max(0, min(30, int_label)))
        return _colors[index]

    @staticmethod
    def get_colormap(x_domain=[-1.0, 1.0]):
        vmin = x_domain[0]
        vmax = x_domain[1]
        cmap = LinearSegmentedColormap.from_list('plygdata_cmap', _colors)
        norm = Normalize(vmin=vmin, vmax=vmax)
        return cmap, norm

