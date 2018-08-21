// ==============================================================================
// Copyright 2018 Digital Advantage Inc. All Rights Reserved.
//
// This is a JavaScript implementation of the graph part of [tensorflow / playground (Deep playground)](https://github.com/tensorflow/playground).
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

"use strict";

const NUM_SHADES = 30;
const DENSITY = 100;
const MAX_DOMAIN = 6;
const DECIMAL_PLACE_LIB = 2; // 小数点の桁数

const width = 300;
const height = width;
const numSamples = DENSITY;
const padding = 20;

const xDomain = [-MAX_DOMAIN, MAX_DOMAIN];
const yDomain = [-MAX_DOMAIN, MAX_DOMAIN];

let onCoordsClickCallback;
let onButtonClickCallback;

let xScalePoint = d3.scaleLinear().domain(xDomain).range([0, width - 2 * padding]);
let yScalePoint = d3.scaleLinear().domain(yDomain).range([height - 2 * padding, 0]);

let xScaleReverse = d3.scaleLinear().domain([0, width - 2 * padding]).range(xDomain);
let yScaleReverse = d3.scaleLinear().domain([height - 2 * padding, 0]).range(yDomain);

let xScaleBoundary = d3.scaleLinear().domain([0, DENSITY - 1]).range(xDomain);
let yScaleBoundary = d3.scaleLinear().domain([DENSITY - 1, 0]).range(yDomain);

let labelScaleColor = d3.scaleLinear().domain([0, .5, 1]).range(["#f59322", "#e8eaeb", "#0877bd"]).clamp(true);
let colors = d3.range(0, 1 + 1E-9, 1 / NUM_SHADES).map(function (a) { return labelScaleColor(a); });
let labelScaleDiscretizedColor = d3.scaleQuantize().domain([-1, 1]).range(colors);

let container = d3.select("#heatmap")
    .append("div")
        .style("width", width + "px")
        .style("height", height + "px")
        .style("position", "relative")
        .style("top", -padding + "px")
        .style("left", -padding + "px");

let canvas = container
    .append("canvas")
        .attr("width", numSamples)
        .attr("height", numSamples)
        .style("width", (width - 2 * padding) + "px")
        .style("height", (height - 2 * padding) + "px")
        .style("position", "absolute")
        .style("top", padding + "px")
        .style("left", padding + "px");

let getCords = function (_this) {
    var coords =d3.mouse(_this);
    var x = xScaleReverse(coords[0] - padding);
    var y = yScaleReverse(coords[1] - padding);
    return [x, y];
}

let onCoordsMouseMove = function (d, i) {
    var coords =getCords(this);
    var gCoords = svg.select("g.coords");
    gCoords.selectAll("text").remove();
    gCoords.append("text")
        .attr("x", 0)
        .attr("y", -4)
        .style("font-size", "12px")
        .text("(" + coords[0].toFixed(DECIMAL_PLACE_LIB) + ", " + coords[1].toFixed(DECIMAL_PLACE_LIB) +")");
}

let onCoordsMouseLeave = function (d, i) {
    var gCoords = svg.select("g.coords");
    gCoords.selectAll("text").remove();
}

let onCoordsClick = function (d, i) {
    var coords =getCords(this);
    if (onCoordsClickCallback) onCoordsClickCallback(coords);
}

let svg = container
    .append("svg")
        .attr("width", width)
        .attr("height", height)
        .style("position", "absolute")
        .style("top", "0")
        .style("left", "0")
        .on("click", onCoordsClick)
        .on("mouseleave", onCoordsMouseLeave)
        .on("mousemove", onCoordsMouseMove)
    .append("g")
        .attr("transform", "translate(" + padding + "," + padding + ")");
svg.append("g")
        .attr("class", "test");
svg.append("g")
        .attr("class", "coords");

let xAxis = d3.axisBottom().scale(xScalePoint);
let yAxis = d3.axisRight().scale(yScalePoint);
svg
    .append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + (height - 2 * padding) + ")")
            .call(xAxis);
svg
    .append("g")
        .attr("class", "y axis")
        .attr("transform", "translate(" + (width - 2 * padding) + ",0)")
            .call(yAxis);

let xScaleColorBar = d3.scaleLinear().domain([-1, 1]).range([0, 254]);
let xAxisColorBar = d3.axisBottom().scale(xScaleColorBar).tickValues([-1, 0, 1]).tickFormat(d3.format("d"));
d3.select("#color-map g.core")
    .append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0,10)")
            .call(xAxisColorBar);


let showSpinner = function () {
    d3.select(".mdl-spinner").classed("is-active", true)
}

let hideSpinner = function () {
    d3.select(".mdl-spinner").classed("is-active", false)
}

let getDecisionBoundary = async function (callback) {
    var boundary = new Array(DENSITY);
    for (var i = 0; i < DENSITY; i++) {
        boundary[i] = new Array(DENSITY);

        var arrayInputPoints = new Array(DENSITY);
        for (var j = 0; j < DENSITY; j++) {
            var x = xScaleBoundary(i);
            var y = yScaleBoundary(j);
            arrayInputPoints[j] = [x, y];
        }
        var arrayOutputPoints = callback(arrayInputPoints); // 10000件まとめて処理できなかったので、100件ずつにした
        
        for (var k = 0; k < DENSITY; k++) {
            boundary[i][k] = arrayOutputPoints[k];
        }
    }
    return boundary;
};

let updateBackground = function (data, discretize) {
    var dx = data[0].length;
    var dy = data.length;
    if (dx !== numSamples || dy !== numSamples) {
        throw new Error("The provided data matrix must be of size " +
            "numSamples X numSamples");
    }

    var context = canvas.node().getContext("2d");
    var image = context.createImageData(dx, dy);
    for (var y = 0, p = -1; y < dy; ++y) {
        for (var x = 0; x < dx; ++x) {
            var value = data[x][y];
            if (discretize) {
                value = (value >= 0 ? 1 : -1);
            }
            var c = d3.rgb(labelScaleDiscretizedColor(value));
            image.data[++p] = c.r;
            image.data[++p] = c.g;
            image.data[++p] = c.b;
            image.data[++p] = 160;
        }
    }
    context.putImageData(image, 0, 0);
};

let updateCircles = function (container, points, discretize) {
    var xScaleDomain = xScalePoint.domain();
    var yScaleDomain = yScalePoint.domain();
    points = points.filter(function (p) {
        return p.x >= xScaleDomain[0] && p.x <= xScaleDomain[1]
            && p.y >= yScaleDomain[0] && p.y <= yScaleDomain[1];
    });

    container.selectAll('circle').remove();
    var selection = container.selectAll("circle").data(points);

    selection.enter()
        .append('circle')
            .attr("r", 3)
            .attr('cx', function (d) { return xScalePoint(d.x) })
            .attr('cy', function (d) { return yScalePoint(d.y) })
            .attr('fill', function (d) { return discretize ? labelScaleDiscretizedColor(d.label) : labelScaleColor(d.label) })
    selection.exit().remove();
};

let updatePoints = function (points, discretize) {
    //console.log(points);
    updateCircles(svg.select("g.test"), points, discretize);
};

let updateTableHead = function (isClassificaion) {
    var names = [];
    names.push(isClassificaion
        ? {x: "X座標", y: "Y座標", label: "分類", pred: "確度" }
        : {x: "X座標", y: "Y座標", label: "回帰", pred: "予測" }
    );
    //console.log(names);

    var thead = d3.select("#data-table-head");

    thead.selectAll('tr').remove();
    var tr = thead.selectAll('tr').data(names);
    var th = tr.enter().append("tr").selectAll('th').data(function(row) { return d3.entries(row); });
    tr.exit().remove();

    th.enter().append("th").text(function (name) { return name.value; });
    th.exit().remove();
};

let updateDataTable = function (dataset, isClassificaion) {
    //console.log(dataset);

    var tbody = d3.select("#data-table-body");

    tbody.selectAll("tr").remove();
    var tr = tbody.selectAll("tr").data(dataset);
    var td = tr.enter().append("tr").selectAll("td").data(function(row) { return d3.entries(row); });
    tr.exit().remove();

    td.enter().append("td").text(function(d) { return d.key == 'label' ? (isClassificaion ? d.value : '') : d.value.toFixed(DECIMAL_PLACE); });
    td.exit().remove();
};

let randUniform = function (a, b) {
    return Math.random() * (b - a) + a;
}

let generateTestPoints = function (numSamples) {
    var points = [];
    for (var i = 0; i < numSamples; i++) {
        var x = randUniform(-MAX_DOMAIN, MAX_DOMAIN);
        var y = randUniform(-MAX_DOMAIN, MAX_DOMAIN);
        points.push([x, y]);
    }
    return points;
};

d3.select("#generate-test-data")
  .on("click", function(d) {

    if (onButtonClickCallback) onButtonClickCallback();

  });
