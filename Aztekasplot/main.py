"""Aztekasplot Model."""
# Standar modules
from typing import Dict, List, Tuple

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

    def __post_init_post_parse__(self) -> None:
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

    def __call_contour_methods(
        self,
        cmap: str,
        cbar_extend: str,
        set_contour: bool,
        contour_levels: int,
        contour_color: str,
        contour_linewidth: float,
        contour_style: str,
    ) -> None:
        """Call contour methods.

        Parameters:
        -----------
            cmap [str]:
                Color map.

            cbar_extend [str]:
                Colorbar extend.

            set_contour [bool]:
                Set contour.

            contour_levels [int]:
                Contour levels.

            contour_color [str]:
                Contour color.

            contour_linewidth [float]:
                Contour linewidth.

            contour_style [str]:
                Contour style.
        """
        if set_contour:
            self.__set_contour(
                contour_levels, contour_color, contour_linewidth, contour_style
            )
        self.__set_contourf(cmap, cbar_extend)

    def __call_axis_methods(self) -> None:
        """Call axis methods."""
        self.set_axis()
        self.set_axis_labels()
        self.set_axis_ticks()

    def __set_colorbar_label(
        self, cbar_label: str, labelpad: float, rotation: float
    ) -> None:
        """Set colorbar.

        Parameters:
        -----------
            cbar_label [str]:
                Colorbar label.

            labelpad [float]:
                Colorbar labelpad.

            rotation [float]:
                Colorbar label rotation.
        """
        self.cbar.set_label(
            cbar_label, labelpad=labelpad, rotation=rotation, fontsize=self.fontsize
        )

    def __set_contour(
        self,
        contour_levels: int,
        contour_color: str,
        contour_linewidth: float,
        contour_style: str,
    ) -> None:
        """Set contour plot.

        Parameters:
        -----------
            contour_levels [int]:
                Contour levels.

            contour_color [str]:
                Contour color.

            contour_linewidth [float]:
                Contour linewidth.

            contour_style [str]:
                Contour style.
        """
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

    def __set_contourf(self, cmap: str, cbar_extend: str) -> None:
        """Set contourf plot.

        Parameters:
        -----------
            cmap [str]:
                Color map.

            cbar_extend [str]:
                Colorbar extend.
        """
        self.contour = self.ax.contourf(
            self.data_dict["X1"],
            self.data_dict["X2"],
            self.plot_var,
            cmap=cmap,
            levels=self.cmap_levels,
            extend=cbar_extend,
        )
        if self.X2_reflect:
            self.contour = self.ax.contourf(
                -self.data_dict["X1"],
                self.data_dict["X2"],
                self.plot_var,
                cmap=cmap,
                levels=self.cmap_levels,
                extend=cbar_extend,
            )

    def __set_LaTeX(self):
        """Set LaTeX."""
        matplotlib.rcParams["text.usetex"] = True
        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["font.serif"] = ["Computer Modern Roman"]
        matplotlib.rcParams["font.size"] = self.fontsize
        matplotlib.rcParams["axes.labelsize"] = 12
        matplotlib.rcParams["axes.titlesize"] = 12
        matplotlib.rcParams["xtick.labelsize"] = 12
        matplotlib.rcParams["ytick.labelsize"] = 12
        matplotlib.rcParams["legend.fontsize"] = 12

    def __set_plot_var(self, plot_var: str, scale: str, plot_var_0: float) -> None:
        """Set plot variable.

        Parameters:
        -----------
            plot_var [str]:
                Plot variable.

            scale [str]:
                Plot variable scale.

            plot_var_0 [float]:
                Unit plot variable.
        """
        self.__set_plot_var_0(plot_var_0)

        # Check if plot_var is set
        if plot_var in self.data_dict.keys():
            self.plot_var = self.data_dict[plot_var] / self.plot_var_0
        else:
            raise ValueError(f"plot_var {plot_var} is not in data_dict")

        # Check scale
        if scale == "log":
            if self.plot_var.min() <= 0 or self.plot_var.max() <= 0:
                raise ValueError(f"plot_var {plot_var} has negative values. ")
            self.plot_var = np.log10(self.plot_var / self.plot_var_0)

    def __set_plot_var_0(self, plot_var_0: float) -> None:
        """Set unit plot variable.

        Parameters:
        -----------
            plot_var_0 [float]:
                Unit plot variable.
        """
        if not plot_var_0:
            self.plot_var_0 = 1.0
        elif plot_var_0 == "min":
            self.plot_var_0 = self.plot_var.min()
        elif plot_var_0 == "max":
            self.plot_var_0 = self.plot_var.max()
        else:
            self.plot_var_0 = plot_var_0

    def __set_xmin_and_xmax(
        self, x1min: float, x1max: float, x2min: float, x2max: float
    ) -> None:
        """Set xmin and xmax.

        Parameters:
        -----------
            x1min [float]:
                Minimum value for x1.

            x1max [float]:
                Maximum value for x1.

            x2min [float]:
                Minimum value for x2.

            x2max [float]:
                Maximum value for x2.
        """
        self.x1min = self.data_dict["X1"].min() if x1min is None else x1min
        self.x1max = self.data_dict["X1"].max() if x1max is None else x1max
        self.x2min = self.data_dict["X2"].min() if x2min is None else x2min
        self.x2max = self.data_dict["X2"].max() if x2max is None else x2max

    def initialize_plot(
        self,
        fig: object = None,
        ax: object = None,
        set_aspect: str = "equal",
        LaTeX: bool = True,
        fontsize: int = 12,
        figsize: Tuple = (10, 8),
    ) -> None:
        """Initialize plot.

        Parameters:
        -----------
            fig [object]:
                Figure object.

            ax [object]:
                Axis object.

            set_aspect [str]:
                Set aspect ratio.

            LaTeX [bool]:
                Use LaTeX.

            figsize [Tuple]:
                Figure size.
        """
        if fig or ax:
            self.fig, self.ax = fig, ax
        else:
            self.fig, self.ax = plt.subplots(figsize=figsize)

        # Set aspect ratio
        self.ax.set_aspect(set_aspect)

        # Set LaTeX
        if LaTeX:
            self.fontsize = fontsize
            self.__set_LaTeX()

        # Set axis
        self.__call_axis_methods()

    def get_contour_plot(
        self,
        plot_var: str = None,
        plot_var_0: float = None,
        scale: str = "linear",
        cmap: str = "viridis",
        cbar_extend: str = "neither",
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
        self.__set_plot_var(plot_var, scale, plot_var_0)

        # Get contour plot parameters
        if cbar_min is None:
            cbar_min = self.plot_var.min()
        if cbar_max is None:
            cbar_max = self.plot_var.max()

        self.cmap_levels = np.linspace(cbar_min, cbar_max, cbar_levels)
        self.cbar_min = cbar_min
        self.cbar_max = cbar_max

        self.__call_contour_methods(
            cmap,
            cbar_extend,
            set_contour,
            contour_levels,
            contour_color,
            contour_linewidth,
            contour_style,
        )

    def get_colorbar(
        self,
        cbor: str = "vertical",
        cbar_pad: float = 0.07,
        cbar_decimals: int = 2,
        n_ticks: int = 5,
        bar_width: float = 3,
        cbar_label: str = "variable",
        labelpad: float = 20,
        rotation: float = 270,
    ) -> None:
        """Get colorbar.

        Parameters:
        -----------
            cbor [str]:
                Colorbar orientation.

            cbar_pad [float]:
                Colorbar pad.

            cbar_decimals [int]:
                Colorbar decimals.

            n_ticks [int]:
                Number of ticks.

            bar_width [float]:
                Colorbar width.

            cbar_label [str]:
                Colorbar label.

            labelpad [float]:
                Colorbar label pad.

            rotation [float]:
                Colorbar label rotation.
        """
        self.cax = inset_axes(
            self.ax,
            width=f"{bar_width}%",
            height="100%",
            loc="lower right",
            bbox_to_anchor=(cbar_pad, 0.0, 1, 1),
            bbox_transform=self.ax.transAxes,
            borderpad=0,
        )
        self.cbar = self.fig.colorbar(self.contour, orientation=cbor, cax=self.cax)

        cbar_ticks = np.linspace(
            self.cbar_min,
            self.cbar_max,
            num=n_ticks,
        )
        self.cbar.set_ticks(cbar_ticks)
        self.cbar.ax.set_yticklabels(
            [f"{cbar_tick:.{cbar_decimals}f}" for cbar_tick in cbar_ticks],
            fontsize=self.fontsize,
        )

        self.__set_colorbar_label(cbar_label, labelpad, rotation)

    def get_streamlines():
        """TODO"""
        pass

    def save_figure(self, filename: str, dpi=300) -> None:
        """Save figure.

        Parameters:
        -----------
            filename [str]:
                Filename.

            dpi [int]:
                DPI.
        """
        self.fig.savefig(filename, dpi=dpi, bbox_inches="tight")

    def set_axis(
        self, x1min=None, x1max=None, x2min=None, x2max=None, X2_reflect=False
    ) -> None:
        """Set axis.

        Parameters:
        -----------
            x1min [float]:
                Minimum value for x1.

            x1max [float]:
                Maximum value for x1.

            x2min [float]:
                Minimum value for x2.

            x2max [float]:
                Maximum value for x2.

            X2_reflect [bool]:
                Reflect x2.
        """
        self.X2_reflect = X2_reflect

        if self.X2_reflect and not x1min:
            x1min = -self.data_dict["X1"].max()

        self.__set_xmin_and_xmax(x1min, x1max, x2min, x2max)

        plt.xlim(self.x1min, self.x1max)
        plt.ylim(self.x2min, self.x2max)

    def set_axis_ticks(
        self,
        x1_ticks: List = None,
        x2_ticks: List = None,
        x1_nticks: int = 5,
        x2_nticks: int = 5,
        x1_ticks_decimals: int = 0,
        x2_ticks_decimals: int = 0,
    ) -> None:
        """Set axis label.

        Parameters:
        -----------
            x1_ticks [List]:
                List of x1 ticks.

            x2_ticks [List]:
                List of x2 ticks.

            x1_nticks [int]:
                Number of x1 ticks.

            x2_nticks [int]:
                Number of x2 ticks.

            x1_ticks_decimals [int]:
                Number of x1 ticks decimals.

            x2_ticks_decimals [int]:
                Number of x2 ticks decimals.
        """
        if x1_nticks or x1_ticks_decimals:
            if not x1_ticks:
                x1_ticks = np.linspace(
                    self.x1min, self.x1max, num=x1_nticks, endpoint=True
                )
                plt.xticks(
                    x1_ticks,
                    [f"{x1_tick:.{x2_ticks_decimals}f}" for x1_tick in x1_ticks],
                    fontsize=self.fontsize,
                )
            else:
                plt.xticks(
                    x1_ticks,
                    x1_ticks,
                    fontsize=self.fontsize,
                )

        if x2_nticks or x1_ticks_decimals:
            if not x2_ticks:
                x2_ticks = np.linspace(
                    self.x2min, self.x2max, num=x2_nticks, endpoint=True
                )
                plt.yticks(
                    x2_ticks,
                    [f"{x2_tick:.{x2_ticks_decimals}f}" for x2_tick in x2_ticks],
                    fontsize=self.fontsize,
                )
            else:
                plt.yticks(
                    x2_ticks,
                    x2_ticks,
                    fontsize=self.fontsize,
                )

    def set_axis_labels(
        self,
        x1_label: str = "X",
        x2_label: str = "Y",
        x1_units: str = "",
        x2_units: str = "",
    ) -> None:
        """Set axis label.

        Parameters:
        -----------
            x1_label [str]:
                Label for x1.

            x2_label [str]:
                Label for x2.

            x1_units [str]:
                Units for x1.

            x2_units [str]:
                Units for x2.
        """
        if x1_label:
            plt.xlabel(f"{x1_label} {x1_units}", fontsize=self.fontsize)

        if x2_label:
            plt.ylabel(f"{x2_label} {x2_units}", fontsize=self.fontsize)


if __name__ == "__main__":
    filename = "./notebooks/data.dat"
    obj = Plotter(filename)
