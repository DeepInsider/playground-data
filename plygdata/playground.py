# ==============================================================================
# Copyright 2018 Digital Advantage Inc. All Rights Reserved.
#
# This is a Python implementation of [tensorflow / playground (Deep playground) / dataset.ts](https://github.com/tensorflow/playground/blob/master/src/playground.ts).
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
from plygdata.scalelinear import ScaleLinear
from plygdata.state import DatasetType
from plygdata.dataset import shuffle
import math
import numpy as np


DENSITY = 100

POINT_DOMAIN = [-6.0, 6.0]
VALUE_DOMAIN = [-1.0, 1.0]


INPUTS = {
    'x': {'f': lambda x, y: x, 'label': 'X_1'},
    'y': {'f': lambda x, y: y, 'label': 'X_2'},
    'xSquared': {'f': lambda x, y: x * x, 'label': 'X_1^2'},
    'ySquared': {'f': lambda x, y: y * y,  'label': 'X_2^2'},
    'xTimesY': {'f': lambda x, y: x * y, 'label': 'X_1X_2'},
    'sinX': {'f': lambda x, y: math.sin(x), 'label': 'sin(X_1)'},
    'sinY': {'f': lambda x, y: math.sin(y), 'label': 'sin(X_2)'},
}


class Player:

    # ...
    ''' Plays/pauses the player. '''
    # playOrPause()
    # onPlayPause(callback: (isPlaying: boolean) => void)
    # play()
    # pause()
    # private start(localTimerIndex: number)
    # ...
    # function makeGUI()
    # function updateBiasesUI(network: nn.Node[][])
    # function updateWeightsUI(network: nn.Node[][], container: d3.Selection<any>)
    # function drawNode(cx: number, cy: number, nodeId: string, isInput: boolean, container: d3.Selection<any>, node?: nn.Node)
    # # Draw network
    # function drawNetwork(network: nn.Node[][]): void
    # function getRelativeHeight(selection: d3.Selection<any>)
    # function addPlusMinusControl(x: number, layerIdx: number)
    # function updateHoverCard(type: HoverType, nodeOrLink?: nn.Node | nn.Link, coordinates?: [number, number])
    # function drawLink(input: nn.Link, node2coord: {[id: string]: {cx: number, cy: number}}, network: nn.Node[][], container: d3.Selection<any>, isFirst: boolean, index: number, length: number)


    @staticmethod
    def update_decision_boundary(boundary = None, discretize=False):
        '''
         Given a neural network, it asks the network for the output (prediction)
         of every node in the network using inputs sampled on a square grid.
         It returns a map where each key is the node ID and the value is a square
         matrix of the outputs of the network for each input in the grid respectively.

        ###:param network:
        :param boundary:
        :param discretize:
        :return:
        '''

        first_time = True if boundary is None else False
        if first_time:
            boundary = {}
            # Go through all predefined inputs.
            for nodeId in INPUTS:
                boundary[nodeId] = np.empty([DENSITY, DENSITY])
            # nn.forEachNode(network, true, node => {
            #     boundary[node.id] = np.empty([DENSITY, DENSITY])
            # });

        xScale = ScaleLinear(domain=[0, DENSITY - 1], slrange=POINT_DOMAIN)
        yScale = ScaleLinear(domain=[DENSITY - 1, 0], slrange=POINT_DOMAIN)

        for row in range(DENSITY):
            for col in range(DENSITY):
                # 1 for points inside the circle, and 0 for points outside the circle.
                x = xScale(col)
                y = yScale(row)
                input = (x, y)
                if first_time:
                    # Go through all predefined inputs.
                    for nodeId in INPUTS:
                        value = INPUTS[nodeId]['f'](*input)
                        if discretize:
                            value = 1 if value >= 0 else -1
                        boundary[nodeId][row][col] = value
                # nn.forwardProp(network, *input);
                # nn.forEachNode(network, true, node => {
                #     value = node.output
                #     if discretize:
                #        value = 1 if value >= 0 else -1
                #     boundary[nodeId][row][col] = value
                # });

        return boundary


    # function getLoss(network: nn.Node[][], dataPoints: Example2D[]): number
    # function updateUI(firstStep = false)
    # function zeroPad(n: number): string
    # function addCommas(s: string): string
    # function humanReadable(n: number): string
    # function constructInputIds(): string[]
    # function constructInput(x: number, y: number): number[]
    # function oneStep(): void
    # export function getOutputWeights(network: nn.Node[][]): number[]
    # function reset(onStartup=false)
    # function initTutorial()
    # function drawDatasetThumbnails()
    # function hideControls()
    # generate_data(true) -> dateset.py
    # function userHasInteracted()
    # function simulationStarted()


# drawDatasetThumbnails()
# initTutorial()
# makeGUI()
# generate_data(true)
# reset(true)
# hideControls()