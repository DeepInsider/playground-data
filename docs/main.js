// ==============================================================================
// Copyright 2018-2019 Digital Advantage Co., Ltd. All Rights Reserved.
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

const MODEL_FILE_NAME = './models/dnn-classify_two_gauss/model.json';
let isClassificaion = true;                             // 分類の場合は true 、回帰の場合は false 。
let discretizeBoundaryColor = false;             // 決定協会の色を離散化する（＝1か-1にする）かどうか
let discretizePointsColor = isClassificaion;     // 点の色を離散化する（＝1か-1にする）かどうか

const NUM_TEST_DATA = 10; // テストデータの数
const DECIMAL_PLACE = 2; // 小数点の桁数

let savedModel;  // 学習済みモデルを保存するための変数


/*
【注意】tf.tidy もしくは tf.dipose について
TensorFlow.jsのテンソル（tf.Tensor ）オブジェクトはGPUメモリを確保します。
メモリリークを防ぐために、tf.Tensorを生成して計算終了後に、dispose する必要があります。

個々のテンソルオブジェクトを dispose するのが面倒な場合は、tf.tidy({ ... }) の中でテンソルの生成と計算（ops）を行ってください。
tf.tidy のスコープ内の処理は、dispose を呼び出す必要がありません。
機械学習モデルのライフサイクル全体の最後に、テンソルオブジェクトが削除されます。明示的に早く削除したい場合は、dipsose を呼んでください。
*/

let getTensor2dData = function (arrayData) {

  var tensorData = tf.tidy(() => {
    return tf.tensor2d(arrayData);
  });
  //tensorData.print();

  return tensorData;
}


let generateTestData = function (numSamples) {
  console.log('テストデータの生成:');

  var arrayPoints = generateTestPoints(numSamples);

  var tensorPoints = getTensor2dData(arrayPoints);

  return tensorPoints;
}

let loadLearnedModel = async function (filepath) {
  console.log('学習済みモデルの読み込み：');

  try {
    showSpinner()

    var loadedModel = await tf.loadModel(filepath);

    loadedModel.summary();

    return loadedModel;

  } catch (err) {
    console.error(err);

  } finally {
    hideSpinner()
  }
}

let drawDecisionBoundary = async function (model, discretize) {
  console.log("学習済みモデルの決定境界を取得して、背景として描画：");

  try {
    //console.log('start...')
    showSpinner()

    var boundary = await getDecisionBoundary(function (arrayInput) {
      //console.log(arrayInput);

      var tensorInput = getTensor2dData(arrayInput);
      //tensorInput.print();

      var tensorOutput = model.predict(tensorInput);
      //tensorOutput.print();

      var arrayOutput = tensorOutput.dataSync();
      //console.log(arrayOutpuPoints);

      return arrayOutput;
    });

    //console.log('finish.')
  } catch (err) {
    console.error(err);

  } finally {
    hideSpinner()
  }

  updateBackground(boundary, discretize);
}

let predictFromModel = async function (model, tensorInput) {
  console.log('学習済みモデルで、テストデータから予測：');

  // データをTensor2d型からJavaScriptのArray型に変換
  var arrayInput = tensorInput.dataSync(); // なぜか1次元配列にしてしまう...
  //console.log(arrayInput)

  // 学習済みモデルで予測
  var tensorOutput = model.predict(tensorInput, {
    batchSize: 32,
    verbose: false
  });

  // データをTensor2d型からJavaScriptのArray型に変換
  var arrayOutput = tensorOutput.dataSync();
  //console.log(arrayOutput);

  // テストデータの座標点と予測結果を配列にまとめる
  var dictArrayOutput = [];
  for (var i = 0; i < arrayInput.length; i += 2) {
    dictArrayOutput.push({
      x: arrayInput[i],
      y: arrayInput[i + 1],
      label: (arrayOutput[i / 2] >= 0.0) ? 1 : -1,
      pred: arrayOutput[i / 2]
    });
  }
  //console.log(dictArrayOutput)

  return dictArrayOutput;
}

let renderTableHead = async function (enableClass) {
  console.log("表の見出しを描画更新：");

  updateTableHead(enableClass);
}

let renderDataTable = async function (dictArrayOutput, enableClass) {
  console.log("表のデータを描画更新：");

  updateDataTable(dictArrayOutput, enableClass);
}

let plotCoordinatePoints = async function (dictArrayOutput, discretize) {
  console.log("予測結果を色付きの点としてプロット：");

  updatePoints(dictArrayOutput, discretize);
}


// ----------------------------------------------------
// ボタンクリック時の処理

onButtonClickCallback = function() {

  var tensorInput = generateTestData(NUM_TEST_DATA);

  updateUI(tensorInput);
}


// ----------------------------------------------------
// グラフクリック時の処理

onCoordsClickCallback = function (coords) {

  var tensorInput = getTensor2dData([coords]);

  updateUI(tensorInput);
}


// ----------------------------------------------------
// メイン処理

let updateUI = async function (tensorInput) {

  predictFromModel(savedModel, tensorInput).then(dictArrayOutput => {

    renderTableHead(isClassificaion);

    renderDataTable(dictArrayOutput, isClassificaion);

    plotCoordinatePoints(dictArrayOutput, discretizePointsColor);

  });

}

let mainInit = async function () {

  var tensorInput = generateTestData(NUM_TEST_DATA);

  loadLearnedModel(MODEL_FILE_NAME).then(model => {

    savedModel = model;

    drawDecisionBoundary(model, discretizeBoundaryColor).then(() => {

      updateUI(tensorInput);

    });
  });

}

mainInit();
