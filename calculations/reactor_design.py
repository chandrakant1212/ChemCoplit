import numpy as np
from scipy.integrate import odeint


def cstr_volume(F_A0: float, X: float, neg_r_A: float) -> dict:
    """
    CSTR design equation: V = F_A0 * X / (-r_A)

    Args:
        F_A0: Molar feed rate of A [mol/s]
        X: Desired conversion [-] (0 to 1)
        neg_r_A: Rate of reaction at exit conditions [mol/(m³·s)]

    Returns:
        dict with keys:
            - volume_m3 (float): Required CSTR volume [m³]
            - conversion (float): Target conversion
            - space_time_s (float or None): Space time [s]
            - reference (str): Design equation reference
    """
    V = F_A0 * X / neg_r_A
    tau = V * neg_r_A / (F_A0 * (1 - X)) if F_A0 > 0 else None

    return {
        "volume_m3": round(V, 4),
        "conversion": X,
        "space_time_s": round(tau, 4) if tau else None,
        "reference": "Fogler Eq. 2-13 (CSTR Design Equation)"
    }


def pfr_volume_nth_order(F_A0: float, C_A0: float, X: float,
                         k: float, n: float, v0: float) -> dict:
    """
    PFR volume for nth-order reaction via numerical integration.
    -r_A = k * C_A^n, C_A = C_A0*(1-X)

    Args:
        F_A0: Inlet molar flow [mol/s]
        C_A0: Inlet concentration [mol/m³]
        X: Desired conversion [-] (0 to 1)
        k: Rate constant [appropriate units for order n]
        n: Reaction order [-]
        v0: Volumetric flow rate [m³/s]

    Returns:
        dict with keys:
            - volume_m3 (float): Required PFR volume [m³]
            - conversion (float): Target conversion
            - Damkohler_number (float): Damköhler number
            - reaction_order (float): Order used in calculation
            - reference (str): Design equation reference
    """
    X_values = np.linspace(0, X, 1000)
    C_A = C_A0 * (1 - X_values)
    C_A = np.maximum(C_A, 1e-10)  # Avoid division by zero

    neg_r_A = k * C_A**n
    integrand = F_A0 / neg_r_A

    V = np.trapz(integrand, X_values)
    Da = k * (C_A0**(n - 1)) * (V / v0)  # Damköhler number

    return {
        "volume_m3": round(V, 4),
        "conversion": X,
        "Damkohler_number": round(Da, 4),
        "reaction_order": n,
        "reference": "Fogler Eq. 2-15 (PFR Design Equation)"
    }
