# ==============================================================================
# Copyright 2018 Digital Advantage Inc. All Rights Reserved.
#
# This is a Python translagtion of [d3/d3-scale/linear.js](https://github.com/d3/d3-scale/blob/master/src/linear.js).
#
# Licensed under the Apache License, Version 2.0 (the "License")
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


def deinterpolate_linear(a: float, b: float):
    b = b - a
    if b == 0:
        return lambda x: b
    else:
        return lambda x: (x - a) / b


def deinterpolate_clamp(a: float, b: float):
    d = deinterpolate_linear(a, b)

    def _deinterpolate(x: float):
        if x <= a:
            return 0.0
        else:
            if x >= b:
                return 1.0
            else:
                return d(x)

    return _deinterpolate


def interpolate_number(a: float, b: float):
    b = b - a
    return lambda t: a + b * t


def bimap(values, domain: list, slrange: list, clamp: bool):
    d0 = domain[0]
    d1 = domain[1]
    r0 = slrange[0]
    r1 = slrange[1]
    if d1 < d0:
        if clamp:
            calcDomain = deinterpolate_clamp(d1, d0)
        else:
            calcDomain = deinterpolate_linear(d1, d0)
        calcRange = interpolate_number(r1, r0)
    else:
        if clamp:
            calcDomain = deinterpolate_clamp(d0, d1)
        else:
            calcDomain = deinterpolate_linear(d0, d1)
        calcRange = interpolate_number(r0, r1)
    return calcRange(calcDomain(values))


class ScaleLinear:

    _INIT_VALUE = [0, 1]

    def __init__(self, domain=_INIT_VALUE, slrange=_INIT_VALUE, clamp=False):
        self._domain = domain
        self._range = slrange
        self._clamp = bool(clamp)
        self._piecewise = bimap
        self._output = None

    def __call__(self, values):
        return self._piecewise(values, self._domain, self._range, self._clamp)
