"""Miscellaneous functions."""
import linecache
import re
from typing import Dict

import numpy as np

PATTERN = "^#"


def get_plot_dim(source: str = None) -> int:
    """Get dimension of the plot.

    Parameters:
    -----------
        source [str]:
            Source file

    Returns:
    --------
        plot_dim [int]:
            Dimension of the plot.

    """
    with open(source) as file:
        match = False

        for index, line in enumerate(file):
            if (re.match(PATTERN, line)) and (index == 0):
                match = True
                count = 0
                continue
            elif re.match(PATTERN, line):
                match = False
                break
            elif match:
                count += 1

        if count == 3:
            plot_dim = 1
        elif count == 4:
            plot_dim = 2

        return plot_dim


def get_data_dict(source: str, plot_dim: int) -> Dict:
    """Get data dict."""
    time = float(linecache.getline(source, 2))
    Nx1 = int(linecache.getline(source, 3))

    data_dict = {"time": time, "Nx1": Nx1}

    if plot_dim == 1:
        raw_data = np.loadtxt(source, skiprows=5, unpack=True)
        data_dict["COORD"] = str(linecache.getline(source, 4)).rstrip("\n")
        data_dict["raw_data"] = raw_data
        data_dict["x1"] = raw_data[0]
        data_dict["rho"] = raw_data[1]
        data_dict["pre"] = raw_data[2]
        data_dict["vx1"] = raw_data[3]
    elif plot_dim == 2:
        raw_data = np.loadtxt(source, skiprows=5, unpack=True)
        data_dict["Nx2"] = int(linecache.getline(source, 4))
        data_dict["COORD"] = str(linecache.getline(source, 4)).rstrip("\n")

        Nx1 = data_dict["Nx1"]
        Nx2 = data_dict["Nx2"]

        data_dict["x1"] = raw_data[0].reshape(Nx1, Nx2).T
        data_dict["x2"] = raw_data[1].reshape(Nx1, Nx2).T
        data_dict["rho"] = raw_data[2].reshape(Nx1, Nx2).T
        data_dict["pre"] = raw_data[3].reshape(Nx1, Nx2).T
        data_dict["vx1"] = raw_data[4].reshape(Nx1, Nx2).T
        data_dict["vx2"] = raw_data[5].reshape(Nx1, Nx2).T

        if len(raw_data) == 7:
            data_dict["vx3"] = raw_data[5].reshape(Nx1, Nx2).T

    return data_dict
