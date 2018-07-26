# ==============================================================================
# Copyright 2018 Digital Advantage Inc. All Rights Reserved.
#
# This is a Python implementation of [Deep playground](https://github.com/tensorflow/playground).
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

import random
import math
from .scalelinear import ScaleLinear

#class Example2D:
#    """
#    A two dimensional example: x and y coordinates with the label.
#    """
#    def __init__(self, x=0, y=0, label=0) :
#        self.x = x
#        self.y = y
#        self.label = label

class Point:
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y


def rand_uniform(a: float, b: float) -> float:
    """
    Returns a sample from a uniform [a, b] distribution.
    """
    return random.random() * (b - a) + a


def normal_random(mean: float = 0.0, variance: float = 1.0) -> float:
    """
        Samples from a normal distribution.

        :param mean: int
            The mean. Default is 0.
        :param variance: int
            The variance. Default is 1.
    """
    while True:
        v1 = 2.0 * random.random() - 1.0
        v2 = 2.0 * random.random() - 1.0
        s = v1 * v1 + v2 * v2
        if s <= 1.0:
            break
    result = math.sqrt(-2.0 * math.log(s) / s) * v1
    return mean + math.sqrt(variance) * result


def dist(a: Point, b: Point) -> float:
    """
    Returns the euclidean distance between two points in space.
    """
    dx: float = a.x - b.x
    dy: float = a.y - b.y
    return math.sqrt(dx * dx + dy * dy)


def shuffle(array: list) -> None:
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


class DataGenerator:

    @staticmethod
    def classify_two_gauss(noise: float = 0.0, numSamples: int = 500) -> list:
        points = []

        varianceScale = ScaleLinear(domain=[0.0, 0.5], slrange=[0.5, 4.0])
        variance = varianceScale(noise)

        def genGauss(cx: float, cy: float, label: int):
            for _ in range(numSamples // 2):
                x = normal_random(cx, variance)
                y = normal_random(cy, variance)
                points.append([x, y, label])

        genGauss(2.0, 2.0, 1) # Gaussian with positive examples.
        genGauss(-2.0, -2.0, -1) # Gaussian with negative examples.
        return points


    @staticmethod
    def classify_xor(noise: float = 0.0, numSamples: int = 500) -> list:

        def getXORLabel(p: Point) -> int:
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
    def classify_circle(noise: float = 0.0, numSamples: int = 500) -> list:
        points = []
        radius = 5.0

        def getCircleLabel(p: Point, center: Point):
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
    def classify_spiral(noise: float = 0.0, numSamples: int = 500) -> list:
        points = []
        n = numSamples // 2

        def genSpiral(deltaT: float, label: int):
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
    def regress_plane(noise: float = 0.0, numSamples: int = 1200) -> list:
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
    def regress_gaussian(noise: float = 0.0, numSamples: int = 1200) -> list:
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
        
        def getLabel(x: float, y: float) -> float:
        # Choose the one that is maximum in abs value.
            curlabel = 0
            for c in gaussians:
                cx: float = c[0]
                cy: float = c[1]
                sign: int = c[2]
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

