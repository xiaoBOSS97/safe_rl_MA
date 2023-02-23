"""Logger module.

This module instantiates a global logger singleton.
"""


from logger.histogram import Histogram
from logger.logger import Logger, LogOutput
from logger.simple_outputs import StdOutput, TextOutput
from logger.tabular_input import TabularInput
from logger.csv_output import CsvOutput
from logger.snapshotter import Snapshotter
from logger.tensor_board_output import TensorBoardOutput

logger = Logger()
tabular = TabularInput()
snapshotter = Snapshotter()

__all__ = [
    'Histogram', 'Logger', 'CsvOutput', 'StdOutput', 'TextOutput', 'LogOutput',
    'Snapshotter', 'TabularInput', 'TensorBoardOutput', 'logger', 'tabular',
    'snapshotter'
]
