# ==============================================================================
# Copyright 2018-2019 Digital Advantage Co., Ltd. All Rights Reserved.
#
# This is a Python implementation of [tensorflow / playground (Deep playground) / state.ts](https://github.com/tensorflow/playground/blob/master/src/state.ts).
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


''' A map between names and activation functions. '''
# export let activations: {[key: string]: nn.ActivationFunction} = {
#   "relu": nn.Activations.RELU,
#   "tanh": nn.Activations.TANH,
#   "sigmoid": nn.Activations.SIGMOID,
#   "linear": nn.Activations.LINEAR
# };

''' A map between names and regularization functions. '''
# export let regularizations: {[key: string]: nn.RegularizationFunction} = {
#   "none": null,
#   "L1": nn.RegularizationFunction.L1,
#   "L2": nn.RegularizationFunction.L2
# };

''' The names of classification and regression dataset-type. '''
class DatasetType:
    ClassifyCircleData = "circle"
    ClassifyXORData = "xor"
    ClassifyTwoGaussData = "gauss"
    ClassifySpiralData = "spiral"
    RegressPlane = "reg-plane"
    RegressGaussian = "reg-gauss"

''' The names of input data-type. '''
class InputType:
    X1 = "x"
    X2 = "y"
    X1Squared = "xSquared"
    X2Squared = "ySquared"
    X1TimesX2 = "xTimesY"
    SinX1 = "sinX"
    SinX2 = "sinY"

''' The names of Layer data-type. '''
class LayerType:
    L1 = "1"
    L2 = "2"
    L3 = "3"
    L4 = "4"
    L5 = "5"
    L6 = "6"

''' The names of Neuron data-type. '''
class NeuronType:
    N1 = "1"
    N2 = "2"
    N3 = "3"
    N4 = "4"
    N5 = "5"
    N6 = "6"
    N7 = "7"
    N8 = "8"
