# Web prediction demo

The [public demo](https://deepinsider.github.io/playground-data/) is served by GitHub Pages from `master:/docs`.
It needs no application server. TensorFlow.js 4.22.0 loads the model JSON and binary weights
from `docs/models/dnn-classify_two_gauss/`; D3 7.9.0 draws the axes, points, and prediction map.
Dependencies are version-pinned on jsDelivr. An internet connection is needed to load them.

The existing model was trained on two Gaussian clusters. Its topology is 2 inputs,
5-unit tanh hidden layer, 5-unit tanh hidden layer, and 1 tanh output. Model weights
are preserved. Outputs at or above zero are class +1; negative outputs are class −1.
The output is not a probability. The browser performs inference, without retraining.

## Interface

- Click/tap a coordinate or submit X/Y in the −6 to 6 range.
- Generate 10 random points; each action replaces the displayed points.
- Switch between a continuous background and a two-color boundary.
- Select English or Japanese. Initial language follows the browser; a selection
  is saved locally when storage is available. Status and errors switch languages too.
- Controls are disabled until loading completes. Loading failures show a retry button.
- Use the coordinate form with a keyboard as an alternative to plot clicks.

The historical CNN tab only contained a placeholder and had no model or implementation.
It has been removed. The completed demo uses the working DNN model. The old Material Design Lite
layout and Universal Analytics snippet have also been removed.

`index.html` defines the page; `bundle.css` controls layout; `i18n.js` supplies UI translations;
`bundle.js` draws the graph/table; `main.js` loads the model and handles predictions and state.
The filenames are historical: there is no bundler or compilation step.

## Verify locally and after deployment

```powershell
npm install --prefix .artifacts/browser playwright
node scripts/verify_web.cjs
node scripts/verify_web.cjs https://deepinsider.github.io/playground-data/
```

The script uses locally installed Microsoft Edge in headless mode. It serves `docs/` itself for
local checks. Node.js and Edge must be available. Generated screenshots and JSON reports are
saved under ignored `.artifacts/demo/`.

Checks cover prediction values, coordinate input and validation, click coordinates at widths
1280/800/390/320, horizontal overflow, language changes and persistence, background toggling,
model-load failure and retry, and repeated prediction tensor disposal. Screenshots are reviewed
for English/Japanese desktop/mobile layouts. Initial page and resource errors must be zero.
Network failures are injected only for the explicit recovery test.

Validation on 2026-10-10: local browser checks passed; tensor count stayed at 6.
Push changes to `master` to trigger GitHub Pages deployment, then rerun the public check.
Web-only changes do not require a new PyPI package version.
