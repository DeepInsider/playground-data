// Copyright 2018-2026 Digital Advantage Co., Ltd. Licensed under Apache-2.0.
"use strict";
const MODEL_FILE_NAME = './models/dnn-classify_two_gauss/model.json';
const DECIMAL_PLACE = 2;
let savedModel;
let cachedBoundary;
let ready = false;
const statusElement = document.getElementById('model-status');
const controls = document.querySelectorAll('#generate-test-data, #coordinate-form input, #coordinate-form button, #discrete-boundary');
const retryButton = document.getElementById('retry-model');
let statusMessage = 'モデルを読み込んでいます…';
let statusValues = {};
function setStatus(message, failed = false, values = {}) {
  statusMessage = message;
  statusValues = values;
  statusElement.textContent = translate(message, values);
  statusElement.classList.toggle('error', failed);
}
function setReady(value) {
  ready = value;
  controls.forEach(control => { control.disabled = !value; });
  statusElement.setAttribute('aria-busy', String(!value && retryButton.hidden));
}
function updateUI(points) {
  if (!ready) return;
  try {
    const rows = tf.tidy(() => {
      const predictions = savedModel.predict(tf.tensor2d(points)).dataSync();
      return points.map(([x, y], index) => ({x, y, label: predictions[index] >= 0 ? 1 : -1, pred: predictions[index]}));
    });
    updateTableHead(true);
    updateDataTable(rows, true);
    updatePoints(rows, true);
    setStatus('{count}点の予測を表示しています。', false, {count: rows.length});
  } catch (error) {
    setStatus('予測できませんでした。もう一度読み込んでください。', true);
    retryButton.hidden = false;
    setReady(false);
  }
}
async function mainInit() {
  retryButton.hidden = true;
  setReady(false);
  setStatus('モデルを読み込んでいます…');
  try {
    if (!window.tf || !window.d3) throw new Error('Dependencies unavailable');
    if (savedModel) { savedModel.dispose(); savedModel = undefined; }
    savedModel = await tf.loadLayersModel(MODEL_FILE_NAME);
    setStatus('予測の地図を描いています…');
    cachedBoundary = await getDecisionBoundary(points => tf.tidy(() => savedModel.predict(tf.tensor2d(points)).dataSync()));
    updateBackground(cachedBoundary, document.getElementById('discrete-boundary').checked);
    setReady(true);
    updateUI(generateTestPoints(10));
  } catch (error) {
    if (savedModel) { savedModel.dispose(); savedModel = undefined; }
    setStatus('モデルを読み込めませんでした。ネットワーク接続を確認して再試行してください。', true);
    retryButton.hidden = false;
    setReady(false);
  }
}
if (window.d3) {
  onButtonClickCallback = () => updateUI(generateTestPoints(10));
  onCoordsClickCallback = coords => {
    if (!ready) return;
    document.getElementById('coordinate-x').value = coords[0].toFixed(2);
    document.getElementById('coordinate-y').value = coords[1].toFixed(2);
    updateUI([coords]);
  };
}
document.getElementById('coordinate-form').addEventListener('submit', event => {
  event.preventDefault();
  const points = ['coordinate-x', 'coordinate-y'].map(id => Number(document.getElementById(id).value));
  if (points.every(value => Number.isFinite(value) && value >= -6 && value <= 6)) updateUI([points]);
});
document.getElementById('discrete-boundary').addEventListener('change', event => {
  if (ready) updateBackground(cachedBoundary, event.target.checked);
});
retryButton.addEventListener('click', () => {
  if (!window.tf || !window.d3) window.location.reload();
  else mainInit();
});
document.addEventListener('languagechange', () => {
  statusElement.textContent = translate(statusMessage, statusValues);
  if (window.d3) updateTableHead(true);
});
mainInit();
