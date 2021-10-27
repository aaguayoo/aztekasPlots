"""Aztekasplot Model."""
# Standar modules
from typing import Dict

# Third party modules
from pydantic.dataclasses import dataclass

# Local modules
from Aztekasplot.utils.miscellaneous import get_data_dict, get_plot_dim


@dataclass
class Plotter:
    """aztekasPlotter Class."""

    source: str = None

    data_dict: Dict = None

    def __post_init_post_parse__(self):
        """Post init section."""
        # Set number of dimensions
        file = self.source
        plot_dim = get_plot_dim(file)

        self.data_dict = get_data_dict(file, plot_dim)

    def __str__(self):
        """Model docstring."""
        pass

    pass
