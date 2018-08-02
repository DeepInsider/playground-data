
""" This folder is processed as 'plygdata package'. Initialization modules for this package. """

from . import _version
__version__ = _version.__version__

# For importing main classes
from .datacolor import DataColor
##from .scalelinear import ScaleLinear   # This is not for user.
from .datahelper import DatasetType, DataHelper
from .dataset import DataGenerator

# For importing all modules
##from . import datacolor
##from . import scalelinear
##from . import dataset
##from . import datahelper
##__all__ = ["datacolor", "scalelinear", "dataset", "datahelper"]
