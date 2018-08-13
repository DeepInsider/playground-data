
""" This folder is processed as 'plygdata package'. Initialization modules for this package. """

from plygdata import _version
__version__ = _version.__version__

# For importing main classes
from plygdata.datacolor import DataColor
##from .scalelinear import ScaleLinear   # This is not for user.
from plygdata.datahelper import DatasetType, DataHelper
from plygdata.dataset import DataGenerator

# For importing all modules
##from plygdata import datacolor
##from plygdata import scalelinear
##from plygdata import dataset
##from plygdata import datahelper
##__all__ = ["datacolor", "scalelinear", "dataset", "datahelper"]
