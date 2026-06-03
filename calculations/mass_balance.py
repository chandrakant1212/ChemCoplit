import numpy as np


def simple_mass_balance(inputs: dict, outputs: dict = None) -> dict:
    """
    Perform a simple steady-state mass balance: Σ(inputs) = Σ(outputs).
    If outputs are partially known, calculates the unknown stream.

    Args:
        inputs: dict mapping stream names to mass flow rates [kg/s].
                Example: {"Feed": 100, "Recycle": 20}
        outputs: dict mapping stream names to mass flow rates [kg/s].
                 Use None for the unknown stream.
                 Example: {"Product": 80, "Waste": None}

    Returns:
        dict with keys:
            - total_input_kgs (float): Sum of all input streams
            - total_output_kgs (float): Sum of all output streams (calculated)
            - outputs (dict): Output streams with unknowns resolved
            - balanced (bool): Whether the balance closes
            - reference (str): Governing principle
    """
    total_in = sum(inputs.values())

    if outputs is None:
        return {
            "total_input_kgs": round(total_in, 4),
            "total_output_kgs": round(total_in, 4),
            "outputs": {"Total Output": round(total_in, 4)},
            "balanced": True,
            "reference": "Steady-state mass balance: Σṁ_in = Σṁ_out"
        }

    known_out = sum(v for v in outputs.values() if v is not None)
    unknown_keys = [k for k, v in outputs.items() if v is None]

    resolved_outputs = dict(outputs)
    if unknown_keys:
        unknown_total = total_in - known_out
        per_unknown = unknown_total / len(unknown_keys)
        for key in unknown_keys:
            resolved_outputs[key] = round(per_unknown, 4)

    total_out = sum(resolved_outputs.values())

    return {
        "total_input_kgs": round(total_in, 4),
        "total_output_kgs": round(total_out, 4),
        "outputs": resolved_outputs,
        "balanced": abs(total_in - total_out) < 1e-6,
        "reference": "Steady-state mass balance: Σṁ_in = Σṁ_out"
    }


def energy_balance_heater(m_dot: float, Cp: float, T_in: float, T_out: float,
                          Q_loss: float = 0.0) -> dict:
    """
    Steady-state energy balance for a heater/cooler.
    Q = ṁ·Cp·(T_out - T_in) + Q_loss

    Args:
        m_dot: Mass flow rate [kg/s]
        Cp: Specific heat capacity [J/(kg·K)]
        T_in: Inlet temperature [K or °C]
        T_out: Outlet temperature [K or °C]
        Q_loss: Heat losses to surroundings [W] (default 0)

    Returns:
        dict with keys:
            - Q_required_W (float): Required heat duty [W]
            - Q_required_kW (float): Required heat duty [kW]
            - delta_T_K (float): Temperature change
            - m_dot_kgs (float): Mass flow rate used
            - reference (str): Governing equation
    """
    delta_T = T_out - T_in
    Q = m_dot * Cp * delta_T + Q_loss

    return {
        "Q_required_W": round(Q, 2),
        "Q_required_kW": round(Q / 1000, 4),
        "delta_T_K": round(delta_T, 2),
        "m_dot_kgs": m_dot,
        "reference": "Q = ṁ·Cp·ΔT + Q_loss"
    }


def stoichiometry_check(species: dict, stoich_coeffs: dict) -> dict:
    """
    Verify stoichiometric consistency for a given reaction.

    Args:
        species: dict mapping species name to molar mass [g/mol].
                 Example: {"CH4": 16, "O2": 32, "CO2": 44, "H2O": 18}
        stoich_coeffs: dict mapping species name to stoichiometric coefficient.
                       Negative for reactants, positive for products.
                       Example: {"CH4": -1, "O2": -2, "CO2": 1, "H2O": 2}

    Returns:
        dict with keys:
            - mass_balance_check (float): Net mass (should be ~0 for balanced)
            - reactant_mass (float): Total reactant mass per basis
            - product_mass (float): Total product mass per basis
            - balanced (bool): Whether the reaction is mass-balanced
            - reference (str): Conservation law reference
    """
    reactant_mass = 0.0
    product_mass = 0.0

    for sp, coeff in stoich_coeffs.items():
        mw = species.get(sp, 0)
        mass_contrib = abs(coeff) * mw
        if coeff < 0:
            reactant_mass += mass_contrib
        else:
            product_mass += mass_contrib

    net = product_mass - reactant_mass

    return {
        "mass_balance_check": round(net, 4),
        "reactant_mass": round(reactant_mass, 4),
        "product_mass": round(product_mass, 4),
        "balanced": abs(net) < 0.01,
        "reference": "Law of Conservation of Mass"
    }
