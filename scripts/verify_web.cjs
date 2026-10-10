// npm install --prefix .artifacts/browser playwright
// node scripts/verify_web.cjs [https://deepinsider.github.io/playground-data/]
const { chromium } = require('../.artifacts/browser/node_modules/playwright');
const fs = require('fs');
const path = require('path');
const http = require('http');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '../docs');
const artifacts = path.resolve(__dirname, '../.artifacts/demo');
fs.mkdirSync(artifacts, {recursive: true});
const server = http.createServer((req, res) => {
  const pathname = decodeURIComponent(req.url.split('?')[0]);
  const file = path.join(root, pathname === '/' ? '/index.html' : pathname);
  if (!file.startsWith(root + path.sep)) { res.writeHead(403); return res.end(); }
  try {
    res.setHeader('Content-Type', {'.html': 'text/html', '.js': 'application/javascript', '.css': 'text/css', '.json': 'application/json'}[path.extname(file)] || 'application/octet-stream');
    res.end(fs.readFileSync(file));
  } catch { res.writeHead(404); res.end(); }
});
(async () => {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const url = process.argv[2] || `http://127.0.0.1:${server.address().port}/`;
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  try {
    const context = await browser.newContext({locale: 'en-US', viewport: {width: 1280, height: 1000}});
    const page = await context.newPage();
    const errors = [];
    const resources = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('response', response => { if (response.status() >= 400) resources.push(response.url()); });
    await page.goto(url, {waitUntil: 'networkidle'});
    await page.waitForFunction(() => !document.querySelector('#generate-test-data').disabled);
    assert.equal(await page.locator('html').getAttribute('lang'), 'en');
    assert.equal(await page.locator('#data-table-body tr').count(), 10);
    assert.equal(await page.locator('h1').innerText(), 'Place a point.\nExplore the prediction.');
    assert.equal((await page.locator('#data-table-head').innerText()).trim(), 'X coordinate\tY coordinate\tClass\tOutput');
    const before = await page.evaluate(() => tf.memory().numTensors);
    for (let i = 0; i < 25; i++) await page.locator('#generate-test-data').click();
    await page.locator('#coordinate-x').fill('3');
    await page.locator('#coordinate-y').fill('-3');
    await page.locator('#coordinate-form button').click();
    assert.equal(await page.locator('#data-table-body tr').count(), 1);
    assert.equal(await page.locator('#data-table-body td').nth(0).innerText(), '3.00');
    assert.equal(await page.locator('#data-table-body td').nth(1).innerText(), '-3.00');
    const expected = await page.evaluate(() => tf.tidy(() => savedModel.predict(tf.tensor2d([[3, -3]])).dataSync()[0]));
    assert.equal(await page.locator('#data-table-body td').nth(3).innerText(), expected.toFixed(2));
    await page.locator('#coordinate-x').fill('7');
    await page.locator('#coordinate-form button').click();
    assert.equal(await page.locator('#data-table-body td').nth(0).innerText(), '3.00');
    await page.locator('#coordinate-x').fill('3');
    const continuous = await page.locator('#heatmap canvas').evaluate(canvas => canvas.toDataURL());
    await page.locator('#discrete-boundary').check();
    assert.notEqual(await page.locator('#heatmap canvas').evaluate(canvas => canvas.toDataURL()), continuous);
    await page.locator('#discrete-boundary').uncheck();
    assert.equal(await page.locator('#heatmap canvas').evaluate(canvas => canvas.toDataURL()), continuous);
    for (const width of [1280, 800, 390, 320]) {
      await page.setViewportSize({width, height: 1000});
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `No overflow at ${width}px`);
      const box = await page.locator('#heatmap svg').boundingBox();
      await page.locator('#heatmap svg').click({position: {x: box.width / 2, y: box.height / 2}});
      assert(Math.abs(Number(await page.locator('#data-table-body td').nth(0).innerText())) < 0.04);
      assert(Math.abs(Number(await page.locator('#data-table-body td').nth(1).innerText())) < 0.04);
      await page.screenshot({path: path.join(artifacts, `english-${width}.png`), fullPage: true});
    }
    await page.locator('[data-language=ja]').click();
    assert.equal(await page.locator('html').getAttribute('lang'), 'ja');
    assert((await page.locator('#model-status').innerText()).includes('1点'));
    assert((await page.locator('#data-table-head').innerText()).includes('予測分類'));
    await page.screenshot({path: path.join(artifacts, 'japanese-320.png'), fullPage: true});
    await page.reload({waitUntil: 'networkidle'});
    await page.waitForFunction(() => !document.querySelector('#generate-test-data').disabled);
    assert.equal(await page.locator('html').getAttribute('lang'), 'ja');
    await page.locator('[data-language=en]').click();
    const after = await page.evaluate(() => tf.memory().numTensors);
    assert.equal(after, before, 'Prediction tensors must be disposed');
    assert.deepEqual(errors, []);
    assert.deepEqual(resources, []);
    await page.route('**/models/**/model.json', route => route.fulfill({status: 503, body: 'Test outage'}));
    await page.reload({waitUntil: 'networkidle'});
    await page.waitForFunction(() => !document.querySelector('#retry-model').hidden);
    assert(await page.locator('#generate-test-data').isDisabled());
    assert((await page.locator('#model-status').innerText()).includes('Could not load'));
    assert.equal(await page.locator('#model-status').getAttribute('aria-busy'), 'false');
    await page.locator('[data-language=ja]').click();
    assert((await page.locator('#model-status').innerText()).includes('読み込めません'));
    await page.screenshot({path: path.join(artifacts, 'load-error.png'), fullPage: true});
    await page.unroute('**/models/**/model.json');
    await page.locator('#retry-model').click();
    await page.waitForFunction(() => !document.querySelector('#generate-test-data').disabled);
    assert.equal(await page.locator('#data-table-body tr').count(), 10);
    const report = {url, responsiveWidths: [1280, 800, 390, 320], languages: ['en', 'ja'], predictions: true, graphCoordinates: true, discreteBackground: true, invalidCoordinates: true, languagePersistence: true, failureAndRetry: true, tensorsBefore: before, tensorsAfter: after, initialErrors: [], initialResourceErrors: []};
    fs.writeFileSync(path.join(artifacts, process.argv[2] ? 'public-results.json' : 'local-results.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report));
  } finally { await browser.close(); server.close(); }
})().catch(error => { console.error(error); server.close(); process.exitCode = 1; });
