import numpy as np


def calc_lmtd(T_h_in: float, T_h_out: float, T_c_in: float, T_c_out: float,
              flow: str = "counter") -> dict:
    """
    Log Mean Temperature Difference.

    Args:
        T_h_in: Hot fluid inlet temperature [K or °C]
        T_h_out: Hot fluid outlet temperature [K or °C]
        T_c_in: Cold fluid inlet temperature [K or °C]
        T_c_out: Cold fluid outlet temperature [K or °C]
        flow: Flow arrangement — 'counter' or 'parallel'

    Returns:
        dict with keys:
            - LMTD_K (float): Log mean temperature difference [K or °C]
            - delta_T1_K (float): Temperature difference at end 1
            - delta_T2_K (float): Temperature difference at end 2
            - flow_arrangement (str): The flow type used
        Or dict with 'error' key if temperature cross detected.
    """
    if flow == "counter":
        dT1 = T_h_in - T_c_out
        dT2 = T_h_out - T_c_in
    else:  # parallel
        dT1 = T_h_in - T_c_in
        dT2 = T_h_out - T_c_out

    if dT1 <= 0 or dT2 <= 0:
        return {"error": "Temperature cross detected — check inlet/outlet values"}

    if abs(dT1 - dT2) < 1e-6:
        lmtd = dT1
    else:
        lmtd = (dT1 - dT2) / np.log(dT1 / dT2)

    return {
        "LMTD_K": round(lmtd, 3),
        "delta_T1_K": round(dT1, 3),
        "delta_T2_K": round(dT2, 3),
        "flow_arrangement": flow
    }


def calc_heat_exchanger_area(Q_W: float, U_Wm2K: float, lmtd_K: float, F: float = 1.0) -> dict:
    """
    Required heat transfer area. Q = U·A·F·LMTD

    Args:
        Q_W: Heat duty [W]
        U_Wm2K: Overall heat transfer coefficient [W/m²·K]
        lmtd_K: Log mean temperature difference [K]
        F: LMTD correction factor for multi-pass (default 1.0 for pure counter-current)

    Returns:
        dict with keys:
            - area_m2 (float): Required heat transfer area [m²]
            - heat_duty_W (float): Input heat duty [W]
            - U_Wm2K (float): Input U value
            - LMTD_K (float): Input LMTD
            - F_factor (float): Correction factor used
            - reference (str): Governing equation reference
    """
    A = Q_W / (U_Wm2K * F * lmtd_K)

    return {
        "area_m2": round(A, 3),
        "heat_duty_W": Q_W,
        "U_Wm2K": U_Wm2K,
        "LMTD_K": lmtd_K,
        "F_factor": F,
        "reference": "Q = U·A·F·ΔT_lm"
    }
