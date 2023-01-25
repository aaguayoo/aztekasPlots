"""Coordinates conversion functions."""
from typing import Dict

import numpy as np
from aztekasPlot.utils.miscellaneous import remove_nan


def convert_to_plot_coordinates(data_dict: Dict) -> dict:
    """Convert data to plot coordinates.

    Parameters
    ----------
        data_dict [dict]:
            Data dictionary.

    Returns:
    --------
        data_dict [dict]:
            Data dictionary with converted coordinates.
    """
    x1 = data_dict["x1"]
    x2 = data_dict["x2"]
    Nx1 = data_dict["Nx1"]
    Nx2 = data_dict["Nx2"]
    COORD = data_dict["COORD"]
    metric = data_dict["metric"]

    # Mesh grid
    nx1 = x1[::Nx2, :]
    nx2 = x2[::, ::Nx1]

    X1, X2 = np.meshgrid(nx1, nx2)

    if COORD in ["CARTESIAN", "CYLINDRICAL"] and metric == "Non-rel":
        data_dict = convert_from_cartesian_minkowski(data_dict, X1, X2)
    elif COORD in ["CARTESIAN", "CYLINDRICAL"] and metric == "Minkowski":
        data_dict = convert_from_cartesian_minkowski(data_dict, X1, X2)
    elif COORD in ["SPHERICAL"]:
        r = X1
        th = X2

        if metric == "Non-rel":
            data_dict = convert_from_spherical(data_dict, r, th)
        elif metric == "Minkowski":
            data_dict = convert_from_spherical_minkowski(data_dict, r, th)
        elif metric == "Kerr-Schild":
            data_dict = convert_from_spherical_kerr_schild(data_dict, r, th)
    elif COORD in ["POLAR"]:
        r = X1
        _ = X2  # phi
        # if metric == "Kerr-Schild":
        # data_dict = convert_from_polar_kerr_schild(data_dict, r, phi)

    return data_dict


def convert_from_cartesian(data_dict: Dict, x: np.ndarray, y: np.ndarray) -> dict:
    """Convert data from cartesian Minkowski coordinates to cartesian.

    Parameters
    ----------
        data_dict [dict]:
            Data dictionary.

        x [np.ndarray]:
            X coordinate.

        y [np.ndarray]:
            Y coordinate.

    Returns:
    --------
        data_dict [dict]:
            Data dictionary with converted coordinates.
    """
    X1 = x
    X2 = y

    vx = data_dict["vx1"]
    vy = data_dict["vx2"]

    data_dict["X1"] = X1
    data_dict["X2"] = X2
    data_dict["vX1"] = vx
    data_dict["vX2"] = vy
    data_dict["vv"] = np.sqrt(vx ** 2 + vy ** 2)

    return data_dict


def convert_from_cartesian_minkowski(
    data_dict: Dict, x: np.ndarray, y: np.ndarray
) -> dict:
    """Convert data from cartesian Minkowski coordinates to cartesian.

    Parameters
    ----------
        data_dict [dict]:
            Data dictionary.

        x [np.ndarray]:
            X coordinate.

        y [np.ndarray]:
            Y coordinate.

    Returns:
    --------
        data_dict [dict]:
            Data dictionary with converted coordinates.
    """
    X1 = x
    X2 = y

    vx = data_dict["vx1"]
    vy = data_dict["vx2"]

    data_dict["X1"] = X1
    data_dict["X2"] = X2
    data_dict["vX1"] = vx
    data_dict["vX2"] = vy
    data_dict["vv"] = np.sqrt(vx ** 2 + vy ** 2)
    data_dict["W"] = 1.0 / np.sqrt(1.0 - data_dict["vv"] ** 2)

    return data_dict


def convert_from_spherical_minkowski(
    data_dict: Dict, r: np.ndarray, th: np.ndarray
) -> dict:
    """Convert data from spherical Minkowski coordinates to cartesian.

    Parameters
    ----------
        data_dict [dict]:
            Data dictionary.

        r [np.ndarray]:
            Spherical radius.

        th [np.ndarray]:
            Longitudinal angle.

    Returns:
    --------
        data_dict [dict]:
            Data dictionary with converted coordinates.
    """
    X1 = r * np.sin(th)
    X2 = r * np.cos(th)

    vr = data_dict["vx1"]
    vth = data_dict["vx2"]

    data_dict["X1"] = X1
    data_dict["X2"] = X2
    data_dict["vX1"] = vr * X1 + vth * X2
    data_dict["vX2"] = vr * X2 / r - vth * X1
    data_dict["vv"] = np.sqrt(vr ** 2 + r ** 2 * vth ** 2)
    data_dict["W"] = 1.0 / np.sqrt(1.0 - data_dict["vv"] ** 2)

    return data_dict


def convert_from_spherical(data_dict: Dict, r: np.ndarray, th: np.ndarray) -> dict:
    """Convert data from spherical non-relativistic coordinates to cartesian.

    Parameters
    ----------
        data_dict [dict]:
            Data dictionary.

        r [np.ndarray]:
            Spherical radius.

        th [np.ndarray]:
            Longitudinal angle.

    Returns:
    --------
        data_dict [dict]:
            Data dictionary with converted coordinates.
    """
    X1 = r * np.sin(th)
    X2 = r * np.cos(th)

    vr = data_dict["vx1"]
    vth = data_dict["vx2"]

    data_dict["X1"] = X1
    data_dict["X2"] = X2
    data_dict["vX1"] = vr * X1 / r + vth * X2 / r
    data_dict["vX2"] = vr * X2 / r - vth * X1 / r
    data_dict["vv"] = np.sqrt(vr ** 2 + vth ** 2)

    return data_dict


def convert_from_spherical_kerr_schild(
    data_dict: Dict, r: np.ndarray, th: np.ndarray
) -> dict:
    """Convert data from spherical Kerr-Schild coordinates to cartesian.

    Parameters
    ----------
        data_dict [dict]:
            Data dictionary.

        r [np.ndarray]:
            Spherical radius.

        th [np.ndarray]:
            Longitudinal angle.

    Returns:
    --------
        data_dict [dict]:
            Data dictionary with converted coordinates.
    """
    a_spin = data_dict["a_spin"]
    metric_dict = get_metric_dict(a_spin, r, th)

    # Define X1 and X2
    X1 = np.sqrt(r ** 2 + a_spin ** 2) * np.sin(th)
    X2 = r * np.cos(th)

    # Define v_i
    v_r = data_dict["vx1"]
    v_th = data_dict["vx2"]
    if data_dict["rotation"]:
        v_phi = data_dict["vx3"]

    # Compute v^i = gamma^{ij} v_j
    vr = metric_dict["gammarr"] * v_r + metric_dict["gammarth"] * v_th
    vth = metric_dict["gammarth"] * v_r + metric_dict["gammathth"] * v_th
    if data_dict["rotation"]:
        vr += metric_dict["gammarphi"] * v_phi
        vth += metric_dict["gammathphi"] * v_phi
        vphi = (
            metric_dict["gammarphi"] * v_r
            + metric_dict["gammathphi"] * v_th
            + metric_dict["gammaphiphi"] * v_phi
        )

    # Compute Horizon Penetrating (HP) velocity magnitude square VV_HP = v^i * v_i
    vv_HP = vr * v_r + vth * v_th
    if data_dict["rotation"]:
        vv_HP += vphi * v_phi

    # Compute Lorentz factor W_HP = 1/sqrt(1 - VV_HP)
    W_HP = 1 / np.sqrt(1 - vv_HP)
    vv_HP = np.sqrt(vv_HP)

    # Compute teh four velocity vector U^\mu
    UT = W_HP / metric_dict["alpha"]
    Ur = W_HP * (vr - metric_dict["betar"] / metric_dict["alpha"])
    Uth = W_HP * (vth - metric_dict["betath"] / metric_dict["alpha"])
    if data_dict["rotation"]:
        Uphi = W_HP * (vphi - metric_dict["betaphi"] / metric_dict["alpha"])

    # Compute HP velocity vector
    vr_HP = Ur / UT
    vth_HP = Uth / UT
    vphi_HP = Uphi / UT

    # Compute stream velocities
    vX1 = Ur * X1 * r / (r ** 2 + a_spin ** 2) + Uth * X2
    vX2 = Ur * X2 / r - Uth * X1

    # Compute the Non-Horizon Penetrating (NH) velocity vector
    alpha_NH = np.sqrt(
        np.abs(metric_dict["rho2"] * metric_dict["Delta"] / metric_dict["Sigma"])
    )
    gamma_rr = metric_dict["rho2"] / metric_dict["Delta"]
    gamma_thth = metric_dict["rho2"]

    UT_NH = UT - 2 * r * Ur / metric_dict["Delta"]
    Ur_NH = Ur
    Uth_NH = Uth
    if data_dict["rotation"]:
        Uphi_NH = Uphi - a_spin * Ur / metric_dict["Delta"]

    # Compute NH velocity vector
    W_NH = alpha_NH * UT_NH
    vr_NH = Ur_NH / W_NH
    vth_NH = Uth_NH / W_NH
    vphi_NH = Uphi_NH / W_NH

    vv_NH = gamma_rr * vr_NH ** 2 + gamma_thth * vth_NH ** 2
    W_NH = 1.0 / np.sqrt(1.0 - vv_NH)

    UhatT = UT / np.sqrt(1.0 + 2.0 * r / metric_dict["rho2"])
    Uhatr = np.sqrt(1.0 + 2.0 * r / metric_dict["rho2"]) * (
        Ur
        + (2.0 * r / (metric_dict["rho2"] + 2.0 * r)) * UT
        - Uphi * a_spin * np.sin(th) ** 2.0
    )
    Uhatth = np.sqrt(metric_dict["rho2"]) * Uth
    Uhatphi = np.sqrt(metric_dict["rho2"]) * np.sin(th) * Uphi

    data_dict["metric_dict"] = metric_dict
    data_dict["X1"] = remove_nan(X1)
    data_dict["X2"] = remove_nan(X2)
    data_dict["vX1"] = remove_nan(vX1)
    data_dict["vX2"] = remove_nan(vX2)
    data_dict["vv"] = remove_nan(vv_HP)
    data_dict["W"] = remove_nan(W_HP)
    data_dict["vv_NH"] = remove_nan(vv_NH)
    data_dict["W_NH"] = remove_nan(W_NH)
    data_dict["vr_NH"] = remove_nan(vr_NH)
    data_dict["vth_NH"] = remove_nan(vth_NH)
    data_dict["vphi_NH"] = remove_nan(vphi_NH)
    data_dict["vr_HP"] = remove_nan(vr_HP)
    data_dict["vth_HP"] = remove_nan(vth_HP)
    data_dict["vphi_HP"] = remove_nan(vphi_HP)
    data_dict["UhatT"] = remove_nan(UhatT)
    data_dict["Uhatr"] = remove_nan(Uhatr)
    data_dict["Uhatth"] = remove_nan(Uhatth)
    data_dict["Uhatphi"] = remove_nan(Uhatphi)

    return data_dict


def get_metric_dict(a: float, r: np.ndarray, th: np.ndarray) -> dict:
    """Get metric dictionary.

    Parameters
    ----------
        a [float]:
            Spin parameter.

        r [np.ndarray]:
            First coordinate.

        th [np.ndarray]:
            Second coordinate.

    Returns:
    --------
        metric_dict [dict]:
            Metric dictionary.
    """
    rho2 = r ** 2.0 + (a ** 2.0) * (np.cos(th) ** 2.0)
    Delta = r ** 2.0 - 2.0 * r + a ** 2.0
    Sigma = (r ** 2.0 + a ** 2.0) ** 2.0 - (a ** 2.0) * Delta * (np.sin(th) ** 2.0)

    alpha = 1.0 / np.sqrt(1.0 + 2.0 * r / rho2)

    beta_r = 2.0 * r / rho2
    beta_th = 0.0
    beta_phi = 0.0

    betar = (2.0 * r / rho2) * (1.0 / (1.0 + 2.0 * r / rho2))
    betath = 0.0
    betaphi = 0.0

    gamma_rr = 1.0 + 2.0 * r / rho2
    gamma_rth = 0
    gamma_rphi = -a * (1.0 + 2.0 * r / rho2) * np.sin(th) ** 2
    gamma_thth = rho2
    gamma_thphi = 0
    gamma_phiphi = (np.sin(th) ** 2.0) * (
        (a ** 2.0) * (1.0 + 2.0 * r / rho2) * (np.sin(th) ** 2.0) + rho2
    )

    gammarr = ((a ** 2) * (rho2 + 2.0 * r) * (np.sin(th) ** 2.0) + rho2 ** 2.0) / (
        rho2 * (rho2 + 2.0 * r)
    )
    gammarth = 0.0
    gammarphi = a / rho2
    gammathth = 1.0 / rho2
    gammathphi = 0
    gammaphiphi = 1.0 / (rho2 * np.sin(th + 0.000001) ** 2.0)

    return {
        "rho2": rho2,
        "Delta": Delta,
        "Sigma": Sigma,
        "alpha": alpha,
        "beta_r": beta_r,
        "betha_th": beta_th,
        "betha_phi": beta_phi,
        "betar": betar,
        "betath": betath,
        "betaphi": betaphi,
        "gamma_rr": gamma_rr,
        "gamma_rth": gamma_rth,
        "gamma_rphi": gamma_rphi,
        "gamma_thth": gamma_thth,
        "gamma_thphi": gamma_thphi,
        "gamma_phiphi": gamma_phiphi,
        "gammarr": gammarr,
        "gammarth": gammarth,
        "gammarphi": gammarphi,
        "gammathth": gammathth,
        "gammathphi": gammathphi,
        "gammaphiphi": gammaphiphi,
    }
