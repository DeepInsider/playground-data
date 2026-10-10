# Changelog

## 1.2.0 (2026-10-10)

- Adopt uv, uv.lock, Python 3.11+, and modern setuptools metadata.
- Fix Matplotlib grid/interpolation and NumPy scalar conversion.
- Repair matrix reduction and a zero-radius normal random sample.
- Preserve multiple label columns; prediction helpers now return integer arrays.
- Add generation, split, rendering, prediction, and reduction regression tests.
- Replace Travis with GitHub Actions and prepare manual PyPI trusted publishing.
- Replace the blocked Material Design Lite CDN with its version-pinned jsDelivr URL.
- Upgrade the Web demo to TensorFlow.js 4.22.0 and D3 7.9.0, migrate model loading,
  pointer events and table entries, and dispose prediction tensors.
- Remove training-only metadata from the inference model; weights are unchanged.

Python 2 and Python 3.4–3.10 are no longer supported.
Historical article notebooks are preserved. Maintained copies and independent locked
uv environments are available under notebooks/, with explicit compatibility patches
and a manual notebook execution workflow.
- Fix continuous prediction heatmaps with current NumPy arrays, found by executing
  the regression article; cover both discrete and continuous model boundaries.
