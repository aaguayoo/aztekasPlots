"""Aztekasplot Model."""
# Standar modules
from typing import Dict

# Third party modules
from pydantic.dataclasses import dataclass

# Local modules
from Aztekasplot.utils.coordinates import convert_to_plot_coordinates
from Aztekasplot.utils.miscellaneous import get_data_dict, get_plot_dim


@dataclass
class Plotter:
    """aztekasPlotter Class."""

    source: str = None

    data_dict: Dict = None

    metric: str = "Minkowski"

    a_spin: float = None

    def __post_init_post_parse__(self):
        """Post init section."""
        # Set number of dimensions
        file = self.source
        plot_dim = get_plot_dim(file)

        self.data_dict = get_data_dict(file, plot_dim)
        self.data_dict["metric"] = self.metric

        if self.metric == "Kerr-Schild" and self.a_spin is None:
            raise ValueError("a_spin must be set for Kerr-Schild metric")

        self.data_dict["a_spin"] = self.a_spin
        if plot_dim == 2:
            self.data_dict = convert_to_plot_coordinates(self.data_dict)

    def __str__(self):
        """Model docstring."""
        pass

    pass


if __name__ == "__main__":
    filename = "./notebooks/data.dat"
    obj = Plotter(filename)
