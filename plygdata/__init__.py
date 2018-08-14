
""" This folder is processed as 'plygdata package'. Initialization modules for this package. """

from plygdata import _version
__version__ = _version.__version__

# For importing main classes
from plygdata.scalelinear import ScaleLinear
from plygdata.datacolor import DataColor
from plygdata.datahelper import split_train_test_x_data_label, get_playground_axes, get_all_boundaries, draw_decision_boundary_of_node, draw_decision_boundary, plot_with_playground_style, plot_sample
from plygdata.dataset import generate
from plygdata.playground import Player
from plygdata.state import DatasetType, InputType

