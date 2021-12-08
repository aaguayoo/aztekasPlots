"""Aztekasplot Model."""
# Standar modules
from typing import Dict, Tuple

# Third party modules
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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

    def set_axis(
        self, x1min=None, x1max=None, x2min=None, x2max=None, X2_reflect=False
    ):
        """Set axis."""
        self.X2_reflect = X2_reflect

        if self.X2_reflect:
            self.x1min = -self.data_dict["X1"].max()

        self.x1min = self.data_dict["X1"].min() if x1min is None else x1min
        self.x1max = self.data_dict["X1"].max() if x1max is None else x1max
        self.x2min = self.data_dict["X2"].min() if x2min is None else x2min
        self.x2max = self.data_dict["X2"].max() if x2max is None else x2max

        plt.xlim(self.x1min, self.x1max)
        plt.ylim(self.x2min, self.x2max)

    def set_axis_ticks(
        self,
        x1_ticks=None,
        x2_ticks=None,
        x1_nticks=5,
        x2_nticks=5,
        ticks_decimals=0,
    ):
        """Set axis label."""
        if x1_nticks or ticks_decimals:
            if not x1_ticks:
                x1_ticks = np.linspace(
                    self.x1min, self.x1max, num=x1_nticks, endpoint=True
                )
            plt.xticks(
                x1_ticks, [f"{x1_tick:.{ticks_decimals}f}" for x1_tick in x1_ticks]
            )

        if x2_nticks or ticks_decimals:
            if not x2_ticks:
                x2_ticks = np.linspace(
                    self.x2min, self.x2max, num=x2_nticks, endpoint=True
                )
            plt.yticks(
                x2_ticks, [f"{x2_tick:.{ticks_decimals}f}" for x2_tick in x2_ticks]
            )

    def set_axis_labels(
        self,
        x1_label: str = "X",
        x2_label: str = "Y",
        x1_units: str = "",
        x2_units: str = "",
    ):
        """Set axis label."""
        if x1_label:
            plt.xlabel(f"{x1_label} {x1_units}")

        if x2_label:
            plt.ylabel(f"{x2_label} {x2_units}")

    def initialize_plot(
        self,
        fig: object = None,
        ax: object = None,
        set_aspect: str = "equal",
        LaTeX: bool = True,
        figsize: Tuple = (10, 8),
    ) -> None:
        """Initialize plot.

        Parameters:
        -----------
            fig [object]:
                Figure object.
            ax [object]:
                Axis object.
        """
        if fig or ax:
            self.fig, self.ax = fig, ax
        else:
            self.fig, self.ax = plt.subplots(figsize=figsize)

        self.ax.set_aspect(set_aspect)

        if LaTeX:
            matplotlib.rcParams["text.usetex"] = True
            matplotlib.rcParams["font.family"] = "serif"
            matplotlib.rcParams["font.serif"] = ["Computer Modern Roman"]
            matplotlib.rcParams["font.size"] = 12
            matplotlib.rcParams["axes.labelsize"] = 12
            matplotlib.rcParams["axes.titlesize"] = 12
            matplotlib.rcParams["xtick.labelsize"] = 12
            matplotlib.rcParams["ytick.labelsize"] = 12
            matplotlib.rcParams["legend.fontsize"] = 12

        self.set_axis()
        self.set_axis_ticks()
        self.set_axis_labels()

    def get_contour_plot(
        self,
        plot_var: str = None,
        scale: str = "linear",
        plot_var_0: float = None,
        cmap: str = "viridis",
        cbar_min: float = None,
        cbar_max: float = None,
        cbar_levels: int = 400,
        set_contour: bool = True,
        contour_levels: int = 20,
        contour_color: str = "black",
        contour_linewidth: float = 0.5,
        contour_style: str = "dashed",
    ) -> None:
        """Get contour plot.

        Parameters:
        -----------
            plot_var [str]:
                Variable to plot.

            contour_plot_dict [dict]:
                Contour plot dictionary.
        """
        if not plot_var_0:
            plot_var_0 = 1.0
        elif plot_var_0 == "min":
            plot_var_0 = self.plot_var.min()
        elif plot_var_0 == "max":
            plot_var_0 = self.plot_var.max()

        # Check if plot_var is set
        if plot_var in self.data_dict.keys():
            self.plot_var = self.data_dict[plot_var] / plot_var_0
        else:
            raise ValueError(f"plot_var {plot_var} is not in data_dict")

        # Check scale
        if scale == "log":
            if self.plot_var.min() <= 0 or self.plot_var.max() <= 0:
                raise ValueError(f"plot_var {plot_var} has negative values. ")
            self.plot_var = np.log10(self.plot_var / plot_var_0)

        # Get contour plot parameters
        if not cbar_min:
            cbar_min = self.plot_var.min()
        if not cbar_max:
            cbar_max = self.plot_var.max()
        cmap_levels = np.linspace(cbar_min, cbar_max, cbar_levels)

        if set_contour:
            self.contour = self.ax.contour(
                self.data_dict["X1"],
                self.data_dict["X2"],
                self.plot_var,
                colors=contour_color,
                linewidths=contour_linewidth,
                levels=contour_levels,
                linestyles=contour_style,
            )
            if self.X2_reflect:
                self.contour = self.ax.contour(
                    -self.data_dict["X1"],
                    self.data_dict["X2"],
                    self.plot_var,
                    colors=contour_color,
                    linewidths=contour_linewidth,
                    levels=contour_levels,
                    linestyles=contour_style,
                )

        self.contour = self.ax.contourf(
            self.data_dict["X1"],
            self.data_dict["X2"],
            self.plot_var,
            cmap=cmap,
            levels=cmap_levels,
        )
        if self.X2_reflect:
            self.contour = self.ax.contourf(
                -self.data_dict["X1"],
                self.data_dict["X2"],
                self.plot_var,
                cmap=cmap,
                levels=cmap_levels,
            )

        self.contour_plot_dict = {
            "plot_var": self.plot_var,
            "cmap": cmap,
            "cbar_min": cbar_min,
            "cbar_max": cbar_max,
            "cbar_levels": cbar_levels,
            "contour_levels": contour_levels,
            "contour_color": contour_color,
            "contour_linewidth": contour_linewidth,
            "X2_reflect": self.X2_reflect,
            "plot_var_0": plot_var_0,
        }

    def set_streamlines():
        """TODO"""
        pass

    def get_colorbar(self, cbor="vertical", cbar_decimals=2, n_ticks=5) -> None:
        """Get colorbar.

        Parameters:
        -----------
            kwargs:
                Keyword arguments:
                    - cmap
                    - cmap_levels
                    - X2_reflect
        """
        self.cax = inset_axes(
            self.ax,
            width="3%",
            height="100%",
            loc="lower right",
            bbox_to_anchor=(0.07, 0.0, 1, 1),
            bbox_transform=self.ax.transAxes,
            borderpad=0,
        )
        self.cbar = self.fig.colorbar(self.contour, orientation=cbor, cax=self.cax)

        cbar_ticks = np.linspace(
            self.contour_plot_dict["cbar_min"],
            self.contour_plot_dict["cbar_max"],
            num=n_ticks,
        )
        self.cbar.set_ticks(cbar_ticks)
        self.cbar.ax.set_yticklabels(
            [f"{cbar_tick:.{cbar_decimals}f}" for cbar_tick in cbar_ticks]
        )


if __name__ == "__main__":
    filename = "./notebooks/data.dat"
    obj = Plotter(filename)
