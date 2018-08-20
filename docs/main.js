// ==============================================================================
// Copyright 2018 Digital Advantage Inc. All Rights Reserved.
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

const MODEL_FILE_NAME = '/model_dnn.json';
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


let getTensor2dData = function (testPoints) {

  var testTensor2d = tf.tidy(() => {
    return tf.tensor2d(testPoints);
  });
  testTensor2d.print();

  return testTensor2d;
}


let generateTestData = function (numSamples) {
  console.log('テストデータの生成:');

  var testPoints = generateTestPoints(numSamples);

  var testData = getTensor2dData(testPoints);

  return testData;
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

    var boundary = await getDecisionBoundary(function (temp2d) {
      //console.log(temp2d);

      var boundaryPoint2d = tf.tidy(() => {
        return tf.tensor2d(temp2d);
      });
      //boundaryPoint2d.print();

      var boundaryPredict2d = model.predict(boundaryPoint2d);
      //boundaryPredict2d.print();

      var predictData2d = boundaryPredict2d.dataSync();
      //console.log(predictData2d);

      return predictData2d;
    });

    //console.log('finish.')
  } catch (err) {
    console.error(err);

  } finally {
    hideSpinner()
  }

  updateBackground(boundary, discretize);
}

let predictFromModel = async function (model, testData) {
  console.log('学習済みモデルで、テストデータの分類と確度を予測：');

  var tensorProbas = model.predict(testData, {
    batchSize: 32,
    verbose: false
  });

  // データをTensor2d型からJavaScriptのArray型に変換
  var dataProbas = tensorProbas.dataSync();

  // 全データの分類と確度をコンソール出力
  for (var i = 0; i < dataProbas.length; i++) {
    var dataClass = (dataProbas[i] >= 0.0) ? 1 : -1;
    console.log('[', i, '] 分類=', dataClass);
    console.log('[', i, '] 確度=', dataProbas[i]);
  }

  // 学習済みモデルで予測
  var predictTest = model.predict(testData);

  // データをTensor2d型からJavaScriptのArray型に変換
  var arrayPredictTest = predictTest.dataSync();
  //console.log(arrayPredictTest);

  // データをTensor2d型からJavaScriptのArray型に変換
  var arrayPointTemp = testData.dataSync(); // なぜか1次元配列にしてしまう...
  //console.log(arrayPointTemp)

  // テストデータの座標点と予測結果を配列にまとめる
  var arrayTestData = [];
  for (var i = 0; i < arrayPointTemp.length; i += 2) {
    arrayTestData.push({
      x: arrayPointTemp[i],
      y: arrayPointTemp[i + 1],
      label: (arrayPredictTest[i / 2] >= 0.0) ? 1 : -1,
      pred: arrayPredictTest[i / 2]
    });
  }
  //console.log(arrayTestData)

  return arrayTestData;
}

let renderTableHead = async function (enableClass) {
  console.log("表の見出しを描画更新：");

  updateTableHead( enableClass);
}

let renderDataTable = async function (dataset, enableClass) {
  console.log("表のデータを描画更新：");

  updateDataTable(dataset, enableClass);
}

let plotCoordinatePoints = async function (arrayTestData, discretize) {
  console.log("予測結果を色付きの点としてプロット：");

  updatePoints(arrayTestData, discretize);
}


// ----------------------------------------------------
// メイン処理

let updateUI = async function (testData) {

    predictFromModel(savedModel, testData).then(arrayTestData => {

      renderTableHead(isClassificaion);

      renderDataTable(arrayTestData, isClassificaion);

      plotCoordinatePoints(arrayTestData, discretizePointsColor);

    });
}

let mainInit = async function () {

  var testData = generateTestData(NUM_TEST_DATA);

  loadLearnedModel(MODEL_FILE_NAME).then(model => {

    savedModel = model;

    drawDecisionBoundary(model, discretizeBoundaryColor).then(() => {

      updateUI(testData);

    });
  });

}

mainInit();

// ----------------------------------------------------
// ボタンクリック時の処理

onButtonClickCallback = function() {

    var testData = generateTestData(NUM_TEST_DATA);

    updateUI(testData);
}


// ----------------------------------------------------
// グラフクリック時の処理

onCoordsClickCallback = function (coords) {

  var testData = getTensor2dData([coords]);

  updateUI(testData);
}
