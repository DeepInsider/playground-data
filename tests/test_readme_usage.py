"""Execute the Python Usage examples shipped as the PyPI description."""
import contextlib
import io
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def usage_blocks():
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    usage = readme.split('\nUsage\n', 1)[1].split('\nDevelopment with uv\n', 1)[0]
    return re.findall(r'```python\s*\n(.*?)```', usage, re.S)


def test_readme_usage_all_blocks():
    blocks = usage_blocks()
    assert len(blocks) == 5
    scope = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for index, code in enumerate(blocks):
                exec(compile(code, f'README Usage block {index + 1}', 'exec'), scope)
        assert len(scope['X_train']) == len(scope['X_valid']) == 250
        assert scope['y_train'].shape == (250, 1)
        assert len(scope['ax'].images) == 1
        for number in plt.get_fignums():
            fig = plt.figure(number)
            fig.canvas.draw()
            assert np.asarray(fig.canvas.buffer_rgba()).std() > 0
    finally:
        plt.close('all')


@pytest.mark.parametrize('dataset', [
    'ClassifyCircleData', 'ClassifyXORData', 'ClassifyTwoGaussData',
    'ClassifySpiralData', 'RegressPlane', 'RegressGaussian',
])
def test_readme_usage_dataset_alternatives(dataset):
    scope = {}
    blocks = usage_blocks()
    code = blocks[3].replace('pg.DatasetType.ClassifyCircleData, data_noise',
                             f'pg.DatasetType.{dataset}, data_noise')
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(blocks[0], scope)
            exec(code, scope)
        total = 1200 if dataset.startswith('Regress') else 500
        assert len(scope['X_train']) + len(scope['X_valid']) == total
        assert np.isfinite(scope['X_train']).all()
        assert np.isfinite(scope['ax'].images[0].get_array()).all()
        scope['fig'].canvas.draw()
    finally:
        plt.close('all')
