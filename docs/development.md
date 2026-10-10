# Development and release

Install current uv, then run `uv python install 3.15` and `uv sync --locked`
from the repository root. Conda is not required. `.venv` is project-local.
The default Python is 3.15; supported versions start at 3.11.
`uv.lock` records development dependencies. PyPI consumers resolve compatible
NumPy and Matplotlib versions from package metadata.

Run `uv run pytest -q`. Tests use Agg and treat warnings as errors.
CI covers Linux and Windows on Python 3.11–3.15.
Refresh dependencies with `uv lock --upgrade`, sync, and rerun tests.
Do not reuse historical `.tox` or Conda environments. Generated output belongs
in ignored `.artifacts/` or `dist/`.

## PyPI release

Distribution: `playground-data`. Import: `plygdata`.
The version source is `plygdata/_version.py` (1.2.0).
Metadata lives in `pyproject.toml`; `setup.py` is a compatibility shim.

1. Check that the version is not already published on PyPI.
2. Run `uv sync --locked`, `uv run pytest -q`,
   `uv build --no-sources --out-dir dist/release`, and
   `uv run twine check --strict dist/release/*`.
3. Install the exact wheel in an isolated environment and exercise import,
   data generation and rendering. Inspect both archives and metadata.
4. Review and commit, update CHANGELOG's release date, and tag the reviewed
   commit `v1.2.0` when ready to release.
5. Configure PyPI's Trusted Publisher: owner `DeepInsider`, repository
   `playground-data`, workflow `publish.yml`, environment `pypi`.
   Configure GitHub's `pypi` environment with required reviewers as appropriate.
6. Manually run Publish to PyPI on the reviewed release tag.
   The workflow builds and uploads only fresh `dist/release/` artifacts.

Alternatively run the following with a project-scoped token provided through
`UV_PUBLISH_TOKEN` (never commit credentials):

```bash
uv publish dist/release/playground_data-1.2.0-py3-none-any.whl dist/release/playground_data-1.2.0.tar.gz
```

Preparation does not publish. Do not upload historical files already in `dist/`.

## Web demo and notebooks

Serve `docs/` over HTTP, for example:
`uv run python -m http.server 8000 --directory docs`.
The demo pins TensorFlow.js 4.22.0 and D3 7.9.0.
Verify model loading, decision boundary, generated points, graph clicks,
and repeated predictions for tensor leaks. Model weights are unchanged;
legacy optimizer metadata is omitted for inference.
Article originals remain historical teaching material. Maintained execution copies and
independent, locked TensorFlow/PyTorch environments are in `notebooks/`. See
[notebook instructions](../notebooks/README.md) and [validation results](notebook-validation.md).
Before release, run all three notebook groups or the manual Article notebooks workflow;
check that unexpected errors are zero and review expected teaching exceptions.
