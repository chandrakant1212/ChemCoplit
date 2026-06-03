import numpy as np


def calc_umf_wen_yu(dp_m: float, rho_p: float, rho_g: float, mu: float, emf: float = 0.45) -> dict:
    """
    Minimum fluidization velocity via Wen & Yu correlation.
    Reference: Kunii & Levenspiel Eq. 3.1

    Args:
        dp_m: Particle diameter [m]
        rho_p: Particle density [kg/m³]
        rho_g: Gas density [kg/m³]
        mu: Gas viscosity [Pa·s]
        emf: Void fraction at minimum fluidization [-]

    Returns:
        dict with keys:
            - U_mf_ms (float): Minimum fluidization velocity [m/s]
            - Archimedes_number (float): Dimensionless Ar number
            - Re_mf (float): Reynolds number at minimum fluidization
            - correlation (str): Source reference
            - regime (str): Regime classification string
    """
    g = 9.81
    Ar = dp_m**3 * rho_g * (rho_p - rho_g) * g / mu**2
    Re_mf = np.sqrt(1135.7 + 0.0408 * Ar) - 33.7
    U_mf = Re_mf * mu / (dp_m * rho_g)

    return {
        "U_mf_ms": round(U_mf, 5),
        "Archimedes_number": round(Ar, 2),
        "Re_mf": round(Re_mf, 4),
        "correlation": "Wen & Yu (K&L Eq. 3.1)",
        "regime": classify_regime(dp_m, rho_p, rho_g, mu)
    }


def calc_terminal_velocity(dp_m: float, rho_p: float, rho_g: float, mu: float) -> dict:
    """
    Terminal velocity via iterative drag coefficient method.
    Reference: K&L Chapter 2

    Args:
        dp_m: Particle diameter [m]
        rho_p: Particle density [kg/m³]
        rho_g: Gas density [kg/m³]
        mu: Gas viscosity [Pa·s]

    Returns:
        dict with keys:
            - U_t_ms (float): Terminal velocity [m/s]
            - Archimedes_number (float): Dimensionless Ar number
            - drag_regime (str): Stokes / Intermediate / Newton
    """
    g = 9.81
    Ar = dp_m**3 * rho_g * (rho_p - rho_g) * g / mu**2

    if Ar < 36:          # Stokes regime
        Ut = dp_m**2 * (rho_p - rho_g) * g / (18 * mu)
        regime = "Stokes"
    elif Ar < 83000:     # Intermediate regime
        Re_t = 0.153 * Ar**0.714
        Ut = Re_t * mu / (dp_m * rho_g)
        regime = "Intermediate"
    else:                # Newton regime
        Re_t = 1.74 * Ar**0.5
        Ut = Re_t * mu / (dp_m * rho_g)
        regime = "Newton"

    return {
        "U_t_ms": round(Ut, 5),
        "Archimedes_number": round(Ar, 2),
        "drag_regime": regime
    }


def classify_regime(dp_m: float, rho_p: float, rho_g: float, mu: float) -> str:
    """
    Classify the expected fluidization regime based on the Archimedes number.

    Args:
        dp_m: Particle diameter [m]
        rho_p: Particle density [kg/m³]
        rho_g: Gas density [kg/m³]
        mu: Gas viscosity [Pa·s]

    Returns:
        str: Human-readable regime classification.
    """
    g = 9.81
    Ar = dp_m**3 * rho_g * (rho_p - rho_g) * g / mu**2
    if Ar < 100:
        return "Fine particles — homogeneous fluidization likely"
    elif Ar < 10000:
        return "Intermediate — bubbling bed expected"
    else:
        return "Coarse particles — slugging or turbulent regime"


def calc_bubble_velocity(U: float, U_mf: float, db_m: float) -> dict:
    """
    Absolute bubble rise velocity. K&L Eq. 6.2

    Args:
        U: Superficial gas velocity [m/s]
        U_mf: Minimum fluidization velocity [m/s]
        db_m: Bubble diameter [m]

    Returns:
        dict with keys:
            - u_br_ms (float): Isolated bubble rise velocity [m/s]
            - u_b_ms (float): Absolute bubble velocity [m/s]
            - bubble_diameter_m (float): Bubble diameter [m]
            - reference (str): Source reference
    """
    g = 9.81
    u_br = 0.711 * np.sqrt(g * db_m)        # Isolated bubble rise
    u_b = U - U_mf + u_br                    # Absolute bubble velocity

    return {
        "u_br_ms": round(u_br, 4),
        "u_b_ms": round(u_b, 4),
        "bubble_diameter_m": db_m,
        "reference": "K&L Eq. 6.2"
    }
