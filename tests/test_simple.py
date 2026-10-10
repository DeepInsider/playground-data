import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import plygdata as pg
from plygdata.heatmap import HeatMap

DATASETS = ["circle", "xor", "gauss", "spiral", "reg-plane", "reg-gauss"]

@pytest.mark.parametrize("dataset", DATASETS)
def test_generation_and_split(dataset):
    data = np.asarray(pg.generate_data(dataset))
    assert data.shape == (1200 if dataset.startswith("reg-") else 500, 3)
    assert np.isfinite(data).all()
    x, y, xv, yv = pg.split_data(data, validation_size=0.25)
    assert len(x) + len(xv) == len(data)
    assert len(xv) == len(data) // 4
    assert x.shape[1] == 2 and y.shape == (len(x), 1)
    assert yv.shape == (len(xv), 1)
    actual = np.concatenate((np.column_stack((x, y)), np.column_stack((xv, yv))))
    assert sorted(map(tuple, actual)) == sorted(map(tuple, data))

@pytest.mark.parametrize("dataset", DATASETS)
def test_plot_and_boundary(dataset):
    fig, ax = pg.plot_sample(dataset, visualize_validation_data=True, node_id=pg.InputType.X1)
    try:
        fig.canvas.draw()
        assert len(ax.images) == 1
        assert len(ax.collections) == len(pg.generate_data(dataset))
    finally:
        plt.close(fig)

def test_training_size_and_multiple_labels():
    x, y, xv, yv = pg.split_data(np.arange(40).reshape(10, 4), label_num=2, training_size=0.7)
    assert x.shape == (7, 2) and y.shape == (7, 2)
    assert xv.shape == (3, 2) and yv.shape == (3, 2)

class Model:
    def predict(self, x, batch_size=32, verbose=0):
        return np.asarray(x)[:, :1]

@pytest.mark.parametrize("discretize", [False, True])
def test_prediction_and_model_boundary(discretize):
    x = np.array([[-1., 2.], [0., 3.], [1., 4.]])
    labels, probabilities = pg.predict_classes_proba(Model(), x)
    np.testing.assert_array_equal(labels.ravel(), [-1, 1, 1])
    assert labels.dtype.kind == "i"
    np.testing.assert_array_equal(pg.predict_classes(Model(), x), labels)
    np.testing.assert_array_equal(pg.predict_proba(Model(), x), probabilities)
    fig = pg.get_playground_figure()
    try:
        ax = pg.get_playground_axes(fig)
        pg.draw_decision_boundary(fig, ax, trained_model=Model(), discretize=discretize)
        fig.canvas.draw()
    finally:
        plt.close(fig)

def test_matrix_reduction():
    assert HeatMap.reduceMatrix([[1, 2], [3, 4]], 2) == [[2.5]]
    np.testing.assert_array_equal(HeatMap.reduceMatrix(np.arange(16).reshape(4, 4), 2), [[2.5, 4.5], [10.5, 12.5]])

@pytest.mark.parametrize("matrix,factor", [([], 1), ([[1, 2]], 1), ([[1]], 0), ([[1]], 2), ([[1]], 1.5)])
def test_invalid_reduction(matrix, factor):
    with pytest.raises(ValueError):
        HeatMap.reduceMatrix(matrix, factor)


def test_normal_random_retries_zero_radius(monkeypatch):
    from plygdata.dataset import normal_random
    values = iter([0.5, 0.5, 0.75, 0.5])
    monkeypatch.setattr("plygdata.dataset.random.random", lambda: next(values))
    assert np.isfinite(normal_random())

def test_empty_split():
    with pytest.raises(ValueError):
        pg.split_data([])
