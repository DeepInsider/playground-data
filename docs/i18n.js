// English and Japanese UI. No external translation service is used.
"use strict";
const englishMessages = {
  '点を置いて、': 'Place a point.',
  '予測を見てみよう。': 'Explore the prediction.',
  '2つのガウス分布で学習したニューラルネットワークが、座標をオレンジ（−1）か青（+1）に分類します。グラフをクリック・タップして試してください。': 'A neural network trained on two Gaussian clusters classifies coordinates as orange (−1) or blue (+1). Click or tap the plot to try it.',
  '予測の地図': 'Prediction map',
  '学習済み DNN': 'Pretrained DNN',
  '横軸はX、縦軸はY。背景の色は各座標でのモデルの出力です。': 'X is the horizontal axis and Y is the vertical axis. The background shows the model output at each coordinate.',
  '−1 オレンジ': '−1 Orange',
  '+1 青': '+1 Blue',
  '白に近いほど出力が0に近く、分類の境界にあります。': 'White marks outputs near zero, along the classification boundary.',
  '背景を2色で表示する': 'Show a two-color background',
  '座標を試す': 'Try a coordinate',
  'モデルを読み込んでいます…': 'Loading the model…',
  'もう一度読み込む': 'Retry loading',
  'ランダムな10点を予測': 'Predict 10 random points',
  'または座標を入力': 'Or enter a coordinate',
  '予測する': 'Predict',
  '入力範囲：−6〜6。操作するたびに点を置き換えます。': 'Range: −6 to 6. Each action replaces the plotted points.',
  '現在の点の予測結果': 'Predictions for the current points',
  'X座標': 'X coordinate',
  'Y座標': 'Y coordinate',
  '予測分類': 'Class',
  '出力値': 'Output',
  '出力値が0以上なら+1、0未満なら−1。出力値は確率ではありません。': 'Outputs of zero or above give +1; negative outputs give −1. The output is not a probability.',
  'このデモについて': 'About this demo',
  '入力はX・Yの2値、隠れ層は各5ユニットの2層、出力は1値（tanh）。Pythonで学習したモデルをブラウザー内で動かしています。ここでは追加学習は行いません。': 'The model has two inputs (X and Y), two hidden layers with five units each, and one tanh output. It was trained in Python and runs in your browser. This demo performs inference only.',
  'Pythonパッケージを使う ↗': 'Get the Python package ↗',
  '記事ノートブックを見る ↗': 'Explore the article notebooks ↗',
  'このデモを使うにはJavaScriptを有効にしてください。': 'Enable JavaScript to use this demo.',
  'ソースコード': 'Source code',
  '予測の地図を描いています…': 'Drawing the prediction map…',
  '{count}点の予測を表示しています。': 'Showing predictions for {count} points.',
  '予測できませんでした。もう一度読み込んでください。': 'Prediction failed. Please reload the model.',
  'モデルを読み込めませんでした。ネットワーク接続を確認して再試行してください。': 'Could not load the model. Check your connection and retry.'
};
let currentLanguage = navigator.language.toLowerCase().startsWith('ja') ? 'ja' : 'en';
try {
  const stored = localStorage.getItem('playground-data-language');
  if (['en', 'ja'].includes(stored)) currentLanguage = stored;
} catch (_) { /* Language switching also works without browser storage. */ }
function translate(message, values = {}) {
  let text = currentLanguage === 'en' ? (englishMessages[message] || message) : message;
  if (currentLanguage === 'en' && message === '{count}点の予測を表示しています。' && values.count === 1) text = 'Showing the prediction for 1 point.';
  for (const [key, value] of Object.entries(values)) text = text.replaceAll(`{${key}}`, String(value));
  return text;
}
const translatedNodes = [];
const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
let textNode;
while ((textNode = walker.nextNode())) {
  const key = textNode.textContent.trim();
  if (englishMessages[key] && !['SCRIPT', 'STYLE'].includes(textNode.parentElement.tagName) && textNode.parentElement.id !== 'model-status') {
    translatedNodes.push({node: textNode, key, original: textNode.textContent});
  }
}
function setLanguage(language) {
  currentLanguage = language;
  document.documentElement.lang = language;
  document.title = language === 'en' ? 'Playground-DATA | Explore neural network predictions' : 'Playground-DATA | 座標で試すニューラルネットワーク';
  for (const {node, key, original} of translatedNodes) node.textContent = original.replace(key, translate(key));
  document.querySelectorAll('[data-language]').forEach(button => {
    button.setAttribute('aria-pressed', String(button.dataset.language === language));
  });
  try { localStorage.setItem('playground-data-language', language); } catch (_) { /* Optional persistence. */ }
  document.dispatchEvent(new Event('languagechange'));
}
document.querySelectorAll('[data-language]').forEach(button => {
  button.addEventListener('click', () => setLanguage(button.dataset.language));
});
setLanguage(currentLanguage);
