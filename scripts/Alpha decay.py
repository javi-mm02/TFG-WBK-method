import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

# Control de salida gráfica y de tabla de sensibilidad
MAKE_PLOTS = True
MAKE_SENSITIVITY_TABLE = True

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Configuración gráfica
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "axes.linewidth": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


# ============================================================
# PARÁMETROS FÍSICOS
# ============================================================

# Parámetros del decaimiento alfa:
# - A_d: número másico del núcleo hijo.
# - Z_d: número atómico del núcleo hijo.
# - Q_alpha: energía liberada en el decaimiento alfa.
# - l: momento angular orbital del canal de salida.
# - V0, r0, a: profundidad, radio reducido y difusividad del Woods-Saxon.
# - r0_coulomb: radio reducido usado en el término de Coulomb.
# - n_radial: número radial usado para calibrar la condición interna.
# - experimental_half_life_seconds: semivida experimental de referencia.
@dataclass
class AlphaDecayParameters:
    A_d: int
    Z_d: int
    Q_alpha: float
    l: int = 0
    V0: float = 150.0
    r0: float = 1.20
    a: float = 0.65
    r0_coulomb: float = 1.20
    n_radial: int = 10
    experimental_half_life_seconds: float = 295.1e-9


# ============================================================
# CONSTANTES
# ============================================================

HBAR_C = 197.3269804
E2_OVER_4PI_EPS0 = 1.439964
AMU_C2 = 931.49410242
C_LIGHT = 299792458.0
FM_TO_M = 1.0e-15
HBAR_MEV_S = 6.582119569e-22


# ============================================================
# MASA REDUCIDA Y RADIO NUCLEAR
# ============================================================

# Masa reducida del sistema alfa-núcleo hijo.
# Se devuelve mu*c^2 en megaelectronvoltios.
def reduced_mass_c2(A_d: int) -> float:
    A_alpha = 4.0
    m_alpha_c2 = A_alpha * AMU_C2
    m_d_c2 = float(A_d) * AMU_C2

    return (m_alpha_c2 * m_d_c2) / (m_alpha_c2 + m_d_c2)


# Radio nuclear parametrizado como R = r0 A^(1/3).
def nuclear_radius(A_d: int, r0: float) -> float:
    return r0 * float(A_d) ** (1.0 / 3.0)


# ============================================================
# POTENCIAL EFECTIVO RADIAL
# ============================================================

def woods_saxon_potential(r, params: AlphaDecayParameters):
    R = nuclear_radius(params.A_d, params.r0)
    r_array = np.asarray(r)

    value = -params.V0 / (1.0 + np.exp((r_array - R) / params.a))

    if np.isscalar(r):
        return float(value)

    return value


# Potencial de Coulomb.
def coulomb_potential(r, params: AlphaDecayParameters):
    Z_alpha = 2
    R_C = nuclear_radius(params.A_d, params.r0_coulomb)
    prefactor = params.Z_d * Z_alpha * E2_OVER_4PI_EPS0

    r_array = np.asarray(r)

    value = np.where(
        r_array < R_C,
        prefactor * (3.0 - (r_array / R_C) ** 2) / (2.0 * R_C),
        prefactor / r_array,
    )

    if np.isscalar(r):
        return float(value)

    return value


# Término centrífugo con corrección de Langer.
def centrifugal_potential(r, params: AlphaDecayParameters, mu_c2: float):
    r_array = np.asarray(r)
    l_eff = params.l + 0.5

    value = HBAR_C**2 * l_eff**2 / (2.0 * mu_c2 * r_array**2)

    if np.isscalar(r):
        return float(value)

    return value


# Potencial efectivo radial completo
def effective_potential(r, params: AlphaDecayParameters, mu_c2: float):
    return (
        woods_saxon_potential(r, params)
        + coulomb_potential(r, params)
        + centrifugal_potential(r, params, mu_c2)
    )


# Puntos de giro
def turning_function(r, params: AlphaDecayParameters, mu_c2: float):
    return effective_potential(r, params, mu_c2) - params.Q_alpha


# ============================================================
# RESOLUCIÓN NUMÉRICAS
# ============================================================

# Método de bisección
def bisection_root(
    function,
    left: float,
    right: float,
    tol: float = 1.0e-12,
    max_iter: int = 250,
) -> float:
    f_left = function(left)
    f_right = function(right)

    if not np.isfinite(f_left) or not np.isfinite(f_right):
        raise ValueError("Function values must be finite at the interval endpoints.")

    if f_left * f_right > 0.0:
        raise ValueError("The interval does not bracket a root.")

    for _ in range(max_iter):
        midpoint = 0.5 * (left + right)
        f_midpoint = function(midpoint)

        if abs(f_midpoint) < tol or 0.5 * (right - left) < tol:
            return midpoint

        if f_left * f_midpoint < 0.0:
            right = midpoint
            f_right = f_midpoint
        else:
            left = midpoint
            f_left = f_midpoint

    return 0.5 * (left + right)


# Integración con cambio de variable
def integrate_with_turning_points(
    function,
    left: float,
    right: float,
    n_points: int = 25000,
    epsilon: float = 1.0e-8,
) -> tuple[float, float]:
    theta = np.linspace(epsilon, 0.5 * np.pi - epsilon, n_points)

    r_values = left + (right - left) * np.sin(theta) ** 2
    dr_dtheta = 2.0 * (right - left) * np.sin(theta) * np.cos(theta)

    y_values = function(r_values) * dr_dtheta
    integral_fine = np.trapezoid(y_values, theta)

    theta_coarse = np.linspace(epsilon, 0.5 * np.pi - epsilon, n_points // 2)

    r_coarse = left + (right - left) * np.sin(theta_coarse) ** 2
    dr_dtheta_coarse = (
        2.0
        * (right - left)
        * np.sin(theta_coarse)
        * np.cos(theta_coarse)
    )

    y_coarse = function(r_coarse) * dr_dtheta_coarse
    integral_coarse = np.trapezoid(y_coarse, theta_coarse)

    stability = abs(integral_fine - integral_coarse)

    return float(integral_fine), float(stability)


# ============================================================
# PUNTOS DE GIRO
# ============================================================

def find_turning_points(
    params: AlphaDecayParameters,
    mu_c2: float,
    r_min: float = 1.0e-4,
    r_max: float = 300.0,
    n_grid: int = 80000,
) -> list[float]:
    r_grid = np.linspace(r_min, r_max, n_grid)
    f_grid = turning_function(r_grid, params, mu_c2)

    roots = []

    for i in range(len(r_grid) - 1):
        f_left = f_grid[i]
        f_right = f_grid[i + 1]

        if not np.isfinite(f_left) or not np.isfinite(f_right):
            continue

        if f_left == 0.0:
            roots.append(r_grid[i])

        if f_left * f_right < 0.0:
            root = bisection_root(
                lambda r: turning_function(r, params, mu_c2),
                r_grid[i],
                r_grid[i + 1],
                tol=1.0e-12,
                max_iter=250,
            )
            roots.append(root)

    cleaned_roots = []

    for root in roots:
        if not cleaned_roots or abs(root - cleaned_roots[-1]) > 1.0e-5:
            cleaned_roots.append(root)

    if len(cleaned_roots) < 3:
        raise RuntimeError("Three turning points could not be found.")

    return cleaned_roots[:3]


# ============================================================
# MAGNITUDES SEMICLÁSICAS
# ============================================================

# Número de onda
#
#     k(r) = sqrt(2 mu c^2 [Q_alpha - V_eff(r)]) / (hbar c).
def allowed_wave_number(r, params: AlphaDecayParameters, mu_c2: float):
    value = params.Q_alpha - effective_potential(r, params, mu_c2)
    return np.sqrt(2.0 * mu_c2 * np.maximum(value, 0.0)) / HBAR_C


# Acción bajo la barrera
def barrier_integrand(r, params: AlphaDecayParameters, mu_c2: float):
    value = effective_potential(r, params, mu_c2) - params.Q_alpha
    return np.sqrt(2.0 * mu_c2 * np.maximum(value, 0.0)) / HBAR_C


# Velocidad radial
def allowed_velocity_over_c(r, params: AlphaDecayParameters, mu_c2: float):
    value = params.Q_alpha - effective_potential(r, params, mu_c2)
    return np.sqrt(2.0 * np.maximum(value, 0.0) / mu_c2)


# Acción interna
def compute_internal_action(
    r1: float,
    r2: float,
    params: AlphaDecayParameters,
    mu_c2: float,
) -> tuple[float, float]:
    return integrate_with_turning_points(
        lambda r: allowed_wave_number(r, params, mu_c2),
        r1,
        r2,
        n_points=25000,
    )


# Acción de penetración en la región prohibida.
def compute_barrier_action(
    r2: float,
    r3: float,
    params: AlphaDecayParameters,
    mu_c2: float,
) -> tuple[float, float]:
    return integrate_with_turning_points(
        lambda r: barrier_integrand(r, params, mu_c2),
        r2,
        r3,
        n_points=25000,
    )


# Frecuencia de asalto
def compute_assault_frequency(
    r1: float,
    r2: float,
    params: AlphaDecayParameters,
    mu_c2: float,
) -> tuple[float, float]:
    def transformed_time_integrand(theta_array):
        r_values = r1 + (r2 - r1) * np.sin(theta_array) ** 2
        dr_dtheta = (
            2.0
            * (r2 - r1)
            * np.sin(theta_array)
            * np.cos(theta_array)
        )

        value = params.Q_alpha - effective_potential(r_values, params, mu_c2)
        value = np.maximum(value, 1.0e-14)
        beta = np.sqrt(2.0 * value / mu_c2)

        return dr_dtheta / beta

    epsilon = 1.0e-5

    theta = np.linspace(epsilon, 0.5 * np.pi - epsilon, 25000)
    y_values = transformed_time_integrand(theta)
    integral_fm = np.trapezoid(y_values, theta)

    theta_coarse = np.linspace(epsilon, 0.5 * np.pi - epsilon, 12500)
    y_coarse = transformed_time_integrand(theta_coarse)
    integral_fm_coarse = np.trapezoid(y_coarse, theta_coarse)

    stability = abs(integral_fm - integral_fm_coarse)

    tau_seconds = 2.0 * integral_fm * FM_TO_M / C_LIGHT
    nu = 1.0 / tau_seconds

    return nu, stability


# ============================================================
# CALIBRACIÓN DE LA PROFUNDIDAD DEL WOODS-SAXON
# ============================================================

def bohr_sommerfeld_target(params: AlphaDecayParameters) -> float:
    return (params.n_radial + 0.5) * np.pi


# Acción interna para un valor dado de V0.
def internal_action_for_depth(
    V0: float,
    params: AlphaDecayParameters,
    mu_c2: float,
) -> float:
    trial_params = AlphaDecayParameters(
        A_d=params.A_d,
        Z_d=params.Z_d,
        Q_alpha=params.Q_alpha,
        l=params.l,
        V0=V0,
        r0=params.r0,
        a=params.a,
        r0_coulomb=params.r0_coulomb,
        n_radial=params.n_radial,
        experimental_half_life_seconds=params.experimental_half_life_seconds,
    )

    roots = find_turning_points(trial_params, mu_c2)
    r1, r2, _ = roots

    internal_action, _ = compute_internal_action(r1, r2, trial_params, mu_c2)

    return internal_action


# Calibración de V0 mediante Bohr-Sommerfeld.
# Se busca la profundidad nuclear que hace que la acción interna coincida
# con el valor semiclasico impuesto.
def calibrate_woods_saxon_depth(
    params: AlphaDecayParameters,
    mu_c2: float,
    V0_min: float = 20.0,
    V0_max: float = 300.0,
    n_scan: int = 120,
    tol: float = 1.0e-9,
) -> tuple[AlphaDecayParameters, float, float]:
    target = bohr_sommerfeld_target(params)

    scan_values = np.linspace(V0_min, V0_max, n_scan)

    previous_V0 = None
    previous_diff = None
    bracket_left = None
    bracket_right = None

    for V0 in scan_values:
        try:
            action = internal_action_for_depth(V0, params, mu_c2)
            diff = action - target
        except Exception:
            continue

        if previous_diff is not None and previous_diff * diff < 0.0:
            bracket_left = previous_V0
            bracket_right = V0
            break

        if abs(diff) < tol:
            calibrated_params = AlphaDecayParameters(
                A_d=params.A_d,
                Z_d=params.Z_d,
                Q_alpha=params.Q_alpha,
                l=params.l,
                V0=V0,
                r0=params.r0,
                a=params.a,
                r0_coulomb=params.r0_coulomb,
                n_radial=params.n_radial,
                experimental_half_life_seconds=params.experimental_half_life_seconds,
            )
            return calibrated_params, action, target

        previous_V0 = V0
        previous_diff = diff

    if bracket_left is None or bracket_right is None:
        raise RuntimeError("The calibrated value of V0 could not be bracketed.")

    def depth_residual(V0):
        return internal_action_for_depth(V0, params, mu_c2) - target

    V0_calibrated = bisection_root(
        depth_residual,
        bracket_left,
        bracket_right,
        tol=tol,
        max_iter=250,
    )

    calibrated_params = AlphaDecayParameters(
        A_d=params.A_d,
        Z_d=params.Z_d,
        Q_alpha=params.Q_alpha,
        l=params.l,
        V0=V0_calibrated,
        r0=params.r0,
        a=params.a,
        r0_coulomb=params.r0_coulomb,
        n_radial=params.n_radial,
        experimental_half_life_seconds=params.experimental_half_life_seconds,
    )

    roots = find_turning_points(calibrated_params, mu_c2)
    r1, r2, _ = roots
    final_action, _ = compute_internal_action(r1, r2, calibrated_params, mu_c2)

    return calibrated_params, final_action, target


# ============================================================
# CÁLCULO PRINCIPAL
# ============================================================

def run_case(params: AlphaDecayParameters) -> dict:
    mu_c2 = reduced_mass_c2(params.A_d)

    r1, r2, r3 = find_turning_points(params, mu_c2)

    internal_action, internal_action_stability = compute_internal_action(
        r1,
        r2,
        params,
        mu_c2,
    )

    barrier_action, barrier_action_stability = compute_barrier_action(
        r2,
        r3,
        params,
        mu_c2,
    )

    log_P = -2.0 * barrier_action
    log10_P = log_P / np.log(10.0)

    nu, nu_integral_stability = compute_assault_frequency(
        r1,
        r2,
        params,
        mu_c2,
    )

    log_lambda_no_preformation = np.log(nu) + log_P
    log_t_half_no_preformation = (
        np.log(np.log(2.0))
        - log_lambda_no_preformation
    )
    log10_t_half_no_preformation = log_t_half_no_preformation / np.log(10.0)

    t_half_no_preformation = np.exp(log_t_half_no_preformation)

    experimental_half_life = params.experimental_half_life_seconds
    preformation_factor_required = t_half_no_preformation / experimental_half_life
    preformation_factor_used = min(preformation_factor_required, 1.0)

    log_lambda_with_preformation = (
        log_lambda_no_preformation
        + np.log(preformation_factor_used)
    )

    log_t_half_with_preformation = (
        np.log(np.log(2.0))
        - log_lambda_with_preformation
    )
    log10_t_half_with_preformation = log_t_half_with_preformation / np.log(10.0)

    gamma_no_preformation_MeV = HBAR_MEV_S * np.exp(log_lambda_no_preformation)
    gamma_with_preformation_MeV = HBAR_MEV_S * np.exp(log_lambda_with_preformation)

    return {
        "mu_c2": mu_c2,
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "internal_action": internal_action,
        "internal_action_stability": internal_action_stability,
        "barrier_action": barrier_action,
        "barrier_action_stability": barrier_action_stability,
        "log10_P": log10_P,
        "nu": nu,
        "nu_integral_stability": nu_integral_stability,
        "gamma_no_preformation_MeV": gamma_no_preformation_MeV,
        "gamma_with_preformation_MeV": gamma_with_preformation_MeV,
        "log10_t_half_no_preformation_seconds": log10_t_half_no_preformation,
        "log10_t_half_with_preformation_seconds": log10_t_half_with_preformation,
        "t_half_no_preformation_seconds": t_half_no_preformation,
        "t_half_with_preformation_seconds": np.exp(log_t_half_with_preformation),
        "preformation_factor_required": preformation_factor_required,
        "preformation_factor_used": preformation_factor_used,
        "experimental_half_life_seconds": experimental_half_life,
    }


# Calibración de V0
def calibrate_and_run_case(
    params: AlphaDecayParameters,
) -> tuple[AlphaDecayParameters, float, float, dict]:
    mu_c2 = reduced_mass_c2(params.A_d)

    calibrated_params, calibration_action, calibration_target = (
        calibrate_woods_saxon_depth(
            params,
            mu_c2,
            V0_min=20.0,
            V0_max=300.0,
            n_scan=120,
            tol=1.0e-9,
        )
    )

    results = run_case(calibrated_params)

    return calibrated_params, calibration_action, calibration_target, results


# ============================================================
# RESULTADOS
# ============================================================

# Resultados principales
def print_results(
    params_initial: AlphaDecayParameters,
    params_calibrated: AlphaDecayParameters,
    calibration_action: float,
    calibration_target: float,
    results: dict,
) -> None:
    print("Alpha decay with calibrated Woods-Saxon, Coulomb and Langer potential")
    print("====================================================================")
    print(f"A_d = {params_calibrated.A_d}")
    print(f"Z_d = {params_calibrated.Z_d}")
    print(f"Q_alpha = {params_calibrated.Q_alpha:.6f} MeV")
    print(f"l = {params_calibrated.l}")
    print(f"r0 = {params_calibrated.r0:.6f} fm")
    print(f"a = {params_calibrated.a:.6f} fm")
    print(f"r0_coulomb = {params_calibrated.r0_coulomb:.6f} fm")
    print()
    print(f"Initial V0 = {params_initial.V0:.6f} MeV")
    print(f"Calibrated V0 = {params_calibrated.V0:.6f} MeV")
    print(f"n_radial = {params_calibrated.n_radial}")
    print(f"Internal action = {calibration_action:.8f}")
    print(f"Bohr-Sommerfeld target = {calibration_target:.8f}")
    print()
    print(f"mu c^2 = {results['mu_c2']:.6f} MeV")
    print()
    print(f"r1 = {results['r1']:.6f} fm")
    print(f"r2 = {results['r2']:.6f} fm")
    print(f"r3 = {results['r3']:.6f} fm")
    print()
    print(f"Internal action = {results['internal_action']:.8f}")
    print(f"Internal action stability = {results['internal_action_stability']:.3e}")
    print(f"Barrier action G = {results['barrier_action']:.8f}")
    print(f"Barrier action stability = {results['barrier_action_stability']:.3e}")
    print(f"log10(P_alpha) = {results['log10_P']:.8f}")
    print()
    print(f"nu = {results['nu']:.6e} s^(-1)")
    print(f"Inner time integral stability = {results['nu_integral_stability']:.3e} fm")
    print(f"Gamma without preformation = {results['gamma_no_preformation_MeV']:.6e} MeV")
    print(
        "log10(t_1/2 / s) without preformation = "
        f"{results['log10_t_half_no_preformation_seconds']:.8f}"
    )
    print(f"t_1/2 without preformation = {results['t_half_no_preformation_seconds']:.6e} s")
    print()
    print(f"Experimental t_1/2 = {results['experimental_half_life_seconds']:.6e} s")
    print(f"Required P0 = {results['preformation_factor_required']:.8f}")
    print(f"Used P0 = {results['preformation_factor_used']:.8f}")
    print()
    print(f"Gamma with preformation = {results['gamma_with_preformation_MeV']:.6e} MeV")
    print(
        "log10(t_1/2 / s) with preformation = "
        f"{results['log10_t_half_with_preformation_seconds']:.8f}"
    )
    print(f"t_1/2 with preformation = {results['t_half_with_preformation_seconds']:.6e} s")


# ============================================================
# ANÁLISIS DE SENSIBILIDAD
# ============================================================

def make_modified_params(
    base_params: AlphaDecayParameters,
    *,
    r0: float | None = None,
    a: float | None = None,
    r0_coulomb: float | None = None,
    n_radial: int | None = None,
) -> AlphaDecayParameters:
    return AlphaDecayParameters(
        A_d=base_params.A_d,
        Z_d=base_params.Z_d,
        Q_alpha=base_params.Q_alpha,
        l=base_params.l,
        V0=base_params.V0,
        r0=base_params.r0 if r0 is None else r0,
        a=base_params.a if a is None else a,
        r0_coulomb=base_params.r0_coulomb if r0_coulomb is None else r0_coulomb,
        n_radial=base_params.n_radial if n_radial is None else n_radial,
        experimental_half_life_seconds=base_params.experimental_half_life_seconds,
    )


# Tabla de sensibilidad frente a los parámetros principales.
def make_alpha_sensitivity_table(base_params: AlphaDecayParameters) -> list[dict]:
    cases = [
        ("Reference", base_params),
        (r"$r_0=1.15\,\mathrm{fm}$", make_modified_params(base_params, r0=1.15)),
        (r"$r_0=1.25\,\mathrm{fm}$", make_modified_params(base_params, r0=1.25)),
        (r"$a=0.55\,\mathrm{fm}$", make_modified_params(base_params, a=0.55)),
        (r"$a=0.75\,\mathrm{fm}$", make_modified_params(base_params, a=0.75)),
        (
            r"$r_{0,\mathrm{C}}=1.15\,\mathrm{fm}$",
            make_modified_params(base_params, r0_coulomb=1.15),
        ),
        (
            r"$r_{0,\mathrm{C}}=1.25\,\mathrm{fm}$",
            make_modified_params(base_params, r0_coulomb=1.25),
        ),
        (
            r"$n_{\mathrm{radial}}=9$",
            make_modified_params(base_params, n_radial=9),
        ),
        (
            r"$n_{\mathrm{radial}}=11$",
            make_modified_params(base_params, n_radial=11),
        ),
    ]

    rows = []

    for case_label, case_params in cases:
        calibrated_params, calibration_action, calibration_target, results = (
            calibrate_and_run_case(case_params)
        )

        rows.append(
            {
                "case": case_label,
                "r0": calibrated_params.r0,
                "a": calibrated_params.a,
                "r0_coulomb": calibrated_params.r0_coulomb,
                "n_radial": calibrated_params.n_radial,
                "V0": calibrated_params.V0,
                "r1": results["r1"],
                "r2": results["r2"],
                "r3": results["r3"],
                "barrier_action": results["barrier_action"],
                "log10_P": results["log10_P"],
                "log10_t_half_no_preformation_seconds": results[
                    "log10_t_half_no_preformation_seconds"
                ],
                "preformation_factor_required": results[
                    "preformation_factor_required"
                ],
                "calibration_action": calibration_action,
                "calibration_target": calibration_target,
            }
        )

    return rows


# Exportación de la tabla de sensibilidad en formato CSV y LaTeX.
def save_alpha_sensitivity_table(table_rows: list[dict]) -> None:
    csv_path = RESULTS_DIR / "alpha_model_sensitivity.csv"
    latex_path = RESULTS_DIR / "alpha_model_sensitivity_table.tex"

    csv_columns = [
        "case",
        "r0",
        "a",
        "r0_coulomb",
        "n_radial",
        "V0",
        "r1",
        "r2",
        "r3",
        "barrier_action",
        "log10_P",
        "log10_t_half_no_preformation_seconds",
        "preformation_factor_required",
        "calibration_action",
        "calibration_target",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        writer.writeheader()

        for row in table_rows:
            writer.writerow(row)

    with latex_path.open("w", encoding="utf-8") as file:
        file.write("\\begin{table}[h]\n")
        file.write("\\centering\n")
        file.write("\\small\n")
        file.write("\\begin{tabular}{lcccc}\n")
        file.write("\\toprule\n")
        file.write(
            "Case & "
            "$V_0$ (MeV) & "
            "$G$ & "
            "$\\log_{10}(t_{1/2}^{(P_0=1)}/\\mathrm{s})$ & "
            "$P_0^{\\mathrm{eff}}$ \\\\\n"
        )
        file.write("\\midrule\n")

        for row in table_rows:
            file.write(
                f"{row['case']} & "
                f"{row['V0']:.3f} & "
                f"{row['barrier_action']:.3f} & "
                f"{row['log10_t_half_no_preformation_seconds']:.3f} & "
                f"{row['preformation_factor_required']:.3f} \\\\\n"
            )

        file.write("\\bottomrule\n")
        file.write("\\end{tabular}\n")
        file.write(
            "\\caption{Sensitivity of the semiclassical alpha-decay estimate "
            "to the main model parameters. In each row, the Woods-Saxon depth "
            "$V_0$ is recalibrated through the internal Bohr-Sommerfeld "
            "condition.}\n"
        )
        file.write("\\label{tab:alpha_model_sensitivity}\n")
        file.write("\\end{table}\n")


# ============================================================
# FIGURA POTENCIAL EFECTIVO
# ============================================================

def make_potential_plot(params: AlphaDecayParameters, results: dict) -> None:
    mu_c2 = results["mu_c2"]

    r_values = np.linspace(0.05, 40.0, 4000)
    potential_values = effective_potential(r_values, params, mu_c2)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    ax.plot(
        r_values,
        potential_values,
        linewidth=2.0,
        label=r"$V_{\mathrm{eff}}(r)$",
    )

    ax.axhline(
        params.Q_alpha,
        linestyle="--",
        linewidth=1.4,
        color="black",
        label=r"$Q_\alpha$",
    )

    for label, root in [
        (r"$r_1$", results["r1"]),
        (r"$r_2$", results["r2"]),
        (r"$r_3$", results["r3"]),
    ]:
        ax.axvline(root, linestyle=":", linewidth=1.1, color="black")
        ax.scatter(root, params.Q_alpha, s=25, zorder=5)

        ax.text(
            root - 1.0,
            params.Q_alpha + 2.0,
            label,
            fontsize=15,
            ha="center",
            va="bottom",
        )

    ax.set_xlim(-5.0, 40.0)
    ax.set_ylim(-110.0, 100.0)

    ax.set_xlabel(r"$r$ (fm)")
    ax.set_ylabel(r"$V_{\mathrm{eff}}(r)$ (MeV)")
    ax.set_title("Effective radial potential for alpha decay")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "alpha_effective_potential.pdf")
    fig.savefig(FIGURES_DIR / "alpha_effective_potential.png", dpi=300)
    plt.close(fig)


# ============================================================
# FIGURA SENSIBILIDAD Q_ALPHA
# ============================================================

def make_q_sensitivity_plot(params: AlphaDecayParameters) -> None:
    Q_values = np.linspace(params.Q_alpha - 0.5, params.Q_alpha + 0.5, 41)
    log10_half_lives = []

    for Q_value in Q_values:
        trial_params = AlphaDecayParameters(
            A_d=params.A_d,
            Z_d=params.Z_d,
            Q_alpha=float(Q_value),
            l=params.l,
            V0=params.V0,
            r0=params.r0,
            a=params.a,
            r0_coulomb=params.r0_coulomb,
            n_radial=params.n_radial,
            experimental_half_life_seconds=params.experimental_half_life_seconds,
        )

        try:
            trial_results = run_case(trial_params)
            log10_half_lives.append(
                trial_results["log10_t_half_no_preformation_seconds"]
            )
        except Exception:
            log10_half_lives.append(np.nan)

    log10_half_lives = np.array(log10_half_lives)

    valid_mask = np.isfinite(log10_half_lives)
    Q_fit = Q_values[valid_mask]
    y_fit = log10_half_lives[valid_mask]

    if len(Q_fit) < 2:
        raise RuntimeError("Not enough valid points for the Q_alpha sensitivity fit.")

    linear_coefficients = np.polyfit(Q_fit, y_fit, 1)
    slope, intercept = linear_coefficients
    y_linear_fit = np.polyval(linear_coefficients, Q_fit)

    residuals = y_fit - y_linear_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)

    if ss_tot == 0.0:
        r_squared = np.nan
    else:
        r_squared = 1.0 - ss_res / ss_tot

    max_abs_residual = np.max(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    ax.plot(
        Q_values,
        log10_half_lives,
        marker="o",
        linestyle="None",
        markersize=4,
        label=r"WBK results",
    )

    ax.plot(
        Q_fit,
        y_linear_fit,
        linestyle="-",
        linewidth=2.0,
        label=r"Linear fit",
    )

    ax.axvline(
        params.Q_alpha,
        linestyle="--",
        linewidth=1.4,
        color="black",
        label=r"Reference $Q_\alpha$",
    )

    fit_text = "\n".join([
        rf"$\log_{{10}}(t_{{1/2}}/\mathrm{{s}})= {slope:.3f}Q_\alpha {intercept:+.3f}$",
        rf"$R^2={r_squared:.6f}$",
        rf"$\max |r_i|={max_abs_residual:.2e}$",
        rf"$\mathrm{{RMSE}}={rmse:.2e}$",
    ])

    ax.text(
        0.53,
        0.97,
        fit_text,
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    ax.set_xlabel(r"$Q_\alpha$ (MeV)")
    ax.set_ylabel(r"$\log_{10}(t_{1/2}/\mathrm{s})$")
    ax.set_title(r"Sensitivity of the WBK half-life estimate to $Q_\alpha$")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "alpha_q_sensitivity.pdf")
    fig.savefig(FIGURES_DIR / "alpha_q_sensitivity.png", dpi=300)
    plt.close(fig)

    fit_path = RESULTS_DIR / "alpha_q_sensitivity_fit.csv"

    with fit_path.open("w", encoding="utf-8") as file:
        file.write("slope,intercept,r_squared,max_abs_residual,rmse\n")
        file.write(
            f"{slope:.12e},"
            f"{intercept:.12e},"
            f"{r_squared:.12e},"
            f"{max_abs_residual:.12e},"
            f"{rmse:.12e}\n"
        )


# ============================================================
# CASO PRINCIPAL
# ============================================================

if __name__ == "__main__":
    initial_params = AlphaDecayParameters(
        A_d=208,
        Z_d=82,
        Q_alpha=8.95412,
        l=0,
        V0=180.0,
        r0=1.20,
        a=0.65,
        r0_coulomb=1.20,
        n_radial=10,
        experimental_half_life_seconds=295.1e-9,
    )

    calibrated_params, calibration_action, calibration_target, final_results = (
        calibrate_and_run_case(initial_params)
    )

    print_results(
        initial_params,
        calibrated_params,
        calibration_action,
        calibration_target,
        final_results,
    )

    if MAKE_SENSITIVITY_TABLE:
        sensitivity_rows = make_alpha_sensitivity_table(initial_params)
        save_alpha_sensitivity_table(sensitivity_rows)

    if MAKE_PLOTS:
        make_potential_plot(calibrated_params, final_results)
        make_q_sensitivity_plot(calibrated_params)

    print()
    print(f"Results saved in: {RESULTS_DIR.resolve()}")
    print(f"Figures saved in: {FIGURES_DIR.resolve()}")