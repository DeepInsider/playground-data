
""" This folder is processed as 'plygdata package'. Initialization modules for this package. """

from plygdata import _version
__version__ = _version.__version__

# For importing main classes
from plygdata.scalelinear import ScaleLinear
from plygdata.datacolor import DataColor
from plygdata.datahelper import \
    split_data,\
    get_playground_figure,\
    get_playground_axes,\
    plot_points, \
    plot_points_with_playground_style,\
    draw_decision_boundary,\
    plot_sample, \
    predict_classes, \
    predict_proba, \
    predict_classes_proba
from plygdata.dataset import  \
    generate_data
from plygdata.playground import Player
from plygdata.state import DatasetType, InputType

