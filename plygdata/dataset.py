# ==============================================================================
# Copyright 2018-2019 Digital Advantage Co., Ltd. All Rights Reserved.
#
# This is a Python implementation of [tensorflow / playground (Deep playground) / dataset.ts](https://github.com/tensorflow/playground/blob/master/src/dataset.ts).
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

import random
import math
from plygdata.scalelinear import ScaleLinear
from plygdata.state import DatasetType


NUM_SAMPLES_CLASSIFY = 500
NUM_SAMPLES_REGRESS = 1200


#class Example2D:
#    """
#    A two dimensional example: x and y coordinates with the label.
#    """
#    def __init__(self, x=0, y=0, label=0) :
#        self.x = x
#        self.y = y
#        self.label = label

class Point:
    def __init__(self, x = 0.0, y = 0.0):
        self.x = x
        self.y = y


def rand_uniform(a, b):
    """
    Returns a sample from a uniform [a, b] distribution.
    """
    return random.random() * (b - a) + a


def normal_random(mean = 0.0, variance = 1.0):
    """
        Samples from a normal distribution.

        :param mean
            The mean. Default is 0.0.
        :param variance
            The variance. Default is 1.0.
    """
    while True:
        v1 = 2.0 * random.random() - 1.0
        v2 = 2.0 * random.random() - 1.0
        s = v1 * v1 + v2 * v2
        if s <= 1.0:
            break
    result = math.sqrt(-2.0 * math.log(s) / s) * v1
    return mean + math.sqrt(variance) * result


def dist(a, b):
    """
    Returns the euclidean distance between two points in space.
    """
    dx = a.x - b.x
    dy = a.y - b.y
    return math.sqrt(dx * dx + dy * dy)


def shuffle(array):
    """
    Shuffles the array using Fisher-Yates algorithm. Uses the seedrandom
    library as the random generator.
    """
    counter = len(array)
    temp = 0
    index = 0
    # While there are elements in the array
    while counter > 0:
        # Pick a random index
        index = int(math.floor(random.random() * counter))
        # Decrease counter by 1
        counter -= 1
        # And swap the last element with it
        temp = array[counter]
        array[counter] = array[index]
        array[index] = temp


def generate_data(data_type, noise = 0.0):

    if data_type == DatasetType.ClassifyTwoGaussData:
        data_array = DataGenerator.classify_two_gauss(NUM_SAMPLES_CLASSIFY, noise=noise)
    elif data_type == DatasetType.ClassifyXORData:
        data_array = DataGenerator.classify_xor(NUM_SAMPLES_CLASSIFY, noise=noise)
    elif data_type == DatasetType.ClassifyCircleData:
        data_array = DataGenerator.classify_circle(NUM_SAMPLES_CLASSIFY, noise=noise)
    elif data_type == DatasetType.ClassifySpiralData:
        data_array = DataGenerator.classify_spiral(NUM_SAMPLES_CLASSIFY, noise=noise)
    elif data_type == DatasetType.RegressPlane:
        data_array = DataGenerator.regress_plane(NUM_SAMPLES_REGRESS, noise=noise)
    elif data_type == DatasetType.RegressGaussian:
        data_array = DataGenerator.regress_gaussian(NUM_SAMPLES_REGRESS, noise=noise)
    else:
        return None

    return data_array


class DataGenerator:

    @staticmethod
    def classify_two_gauss(numSamples, noise = 0.0):
        points = []

        varianceScale = ScaleLinear(domain=[0.0, 0.5], slrange=[0.5, 4.0])
        variance = varianceScale(noise)

        def genGauss(cx, cy, label):
            for _ in range(numSamples // 2):
                x = normal_random(cx, variance)
                y = normal_random(cy, variance)
                points.append([x, y, label])

        genGauss(2.0, 2.0, 1) # Gaussian with positive examples.
        genGauss(-2.0, -2.0, -1) # Gaussian with negative examples.
        return points


    @staticmethod
    def classify_xor(numSamples, noise = 0.0):

        def getXORLabel(p):
            return 1 if (p.x * p.y >= 0) else -1

        points = []
        for _ in range(numSamples):
            x = rand_uniform(-5.0, 5.0)
            padding = 0.3
            x = x + padding if (x > 0) else x - padding  # Padding.
            y = rand_uniform(-5.0, 5.0)
            y = y + padding if (y > 0) else y - padding
            noiseX = rand_uniform(-5.0, 5.0) * noise
            noiseY = rand_uniform(-5.0, 5.0) * noise
            label = getXORLabel(Point(x + noiseX, y + noiseY))
            points.append([x, y, label])
        return points


    @staticmethod
    def classify_circle(numSamples, noise = 0.0):
        points = []
        radius = 5.0

        def getCircleLabel(p, center):
            return 1 if (dist(p, center) < (radius * 0.5)) else -1

        # Generate positive points inside the circle.
        for _ in range(numSamples // 2):
            r = rand_uniform(0.0, radius * 0.5)
            angle = rand_uniform(0.0, 2.0 * math.pi)
            x = r * math.sin(angle)
            y = r * math.cos(angle)
            noiseX = rand_uniform(-radius, radius) * noise
            noiseY = rand_uniform(-radius, radius) * noise
            label = getCircleLabel(Point(x + noiseX, y + noiseY), Point(0.0, 0.0))
            points.append([x, y, label])

        # Generate negative points outside the circle.
        for _ in range(numSamples // 2):
            r = rand_uniform(radius * 0.7, radius)
            angle = rand_uniform(0.0, 2.0 * math.pi)
            x = r * math.sin(angle)
            y = r * math.cos(angle)
            noiseX = rand_uniform(-radius, radius) * noise
            noiseY = rand_uniform(-radius, radius) * noise
            label = getCircleLabel(Point(x + noiseX, y + noiseY), Point(0.0, 0.0))
            points.append([x, y, label])

        return points


    @staticmethod
    def classify_spiral(numSamples, noise = 0.0):
        points = []
        n = numSamples // 2

        def genSpiral(deltaT, label):
            for i in range(n):
                r = i / n * 5
                t = 1.75 * i / n * 2 * math.pi + deltaT
                x = r * math.sin(t) + rand_uniform(-1.0, 1.0) * noise
                y = r * math.cos(t) + rand_uniform(-1.0, 1.0) * noise
                points.append([x, y, label])

        genSpiral(0.0, 1)  # Positive examples.
        genSpiral(math.pi, -1)  # Negative examples.

        return points


    @staticmethod
    def regress_plane(numSamples, noise = 0.0):
        radius = 6
        labelScale = ScaleLinear(domain=[-10, 10], slrange=[-1, 1])
        getLabel = lambda x, y: labelScale(x + y)

        points = []
        for _ in range(numSamples):
            x = rand_uniform(-radius, radius)
            y = rand_uniform(-radius, radius)
            noiseX = rand_uniform(-radius, radius) * noise
            noiseY = rand_uniform(-radius, radius) * noise
            label = getLabel(x + noiseX, y + noiseY)
            points.append([x, y, label])

        return points


    @staticmethod
    def regress_gaussian(numSamples, noise = 0.0):
        points = []
        
        labelScale = ScaleLinear(domain=[0.0, 2.0], slrange=[1.0, 0.0], clamp=True)
        
        gaussians = [
            [-4.0,  2.5,  1.0],
            [ 0.0,  2.5, -1.0],
            [ 4.0,  2.5,  1.0],
            [-4.0, -2.5, -1.0],
            [ 0.0, -2.5,  1.0],
            [ 4.0, -2.5, -1.0]
        ]
        
        def getLabel(x, y):
        # Choose the one that is maximum in abs value.
            curlabel = 0
            for c in gaussians:
                cx = c[0]
                cy = c[1]
                sign = c[2]
                newLabel = sign * labelScale(dist(Point(x, y), Point(cx, cy)))
                if abs(newLabel) > abs(curlabel):
                    curlabel = newLabel
            return curlabel
        
        radius = 6.0
        for _ in range(numSamples):
            x = rand_uniform(-radius, radius)
            y = rand_uniform(-radius, radius)
            noiseX = rand_uniform(-radius, radius) * noise
            noiseY = rand_uniform(-radius, radius) * noise
            label = getLabel(x + noiseX, y + noiseY)
            points.append([x, y, label])
        
        return points

