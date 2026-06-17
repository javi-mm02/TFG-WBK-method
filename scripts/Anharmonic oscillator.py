from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# Opciones generales de ejecución
MAKE_PLOTS = True


# Carpetas del repositorio
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


# Parámetros físicos y numéricos del oscilador anarmónico
@dataclass
class AnharmonicOscillatorParameters:
    mass: float = 1.0
    omega: float = 1.0
    hbar: float = 1.0
    lambda_quartic: float = 0.10
    n_max: int = 8
    q_box: float = 8.0
    n_grid_schrodinger: int = 900


# Potencial y momento clásico
def anharmonic_potential(q, params: AnharmonicOscillatorParameters):
    q_array = np.asarray(q)

    value = (
        0.5 * params.mass * params.omega**2 * q_array**2
        + params.lambda_quartic * q_array**4
    )

    if np.isscalar(q):
        return float(value)

    return value


def classical_momentum(q, energy: float, params: AnharmonicOscillatorParameters):
    kinetic_energy = energy - anharmonic_potential(q, params)
    return np.sqrt(2.0 * params.mass * np.maximum(kinetic_energy, 0.0))


# Punto de giro positivo obtenido de V(q) = E
def positive_turning_point(
    energy: float,
    params: AnharmonicOscillatorParameters,
) -> float:
    if energy <= 0.0:
        raise ValueError("Energy must be positive.")

    lam = params.lambda_quartic
    harmonic_coeff = 0.5 * params.mass * params.omega**2

    if lam <= 0.0:
        return np.sqrt(energy / harmonic_coeff)

    discriminant = harmonic_coeff**2 + 4.0 * lam * energy
    y_value = (-harmonic_coeff + np.sqrt(discriminant)) / (2.0 * lam)

    if y_value <= 0.0:
        raise RuntimeError("Positive turning point could not be found.")

    return float(np.sqrt(y_value))


# Método de bisección para resolver las condiciones de cuantización
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


# Integración con cambio de variable adaptado al punto de giro
def integrate_with_turning_point(
    function,
    left: float,
    right: float,
    n_points: int = 25000,
    epsilon: float = 1.0e-8,
) -> tuple[float, float]:
    theta = np.linspace(epsilon, 0.5 * np.pi - epsilon, n_points)

    q_values = left + (right - left) * np.sin(theta) ** 2
    dq_dtheta = 2.0 * (right - left) * np.sin(theta) * np.cos(theta)

    y_values = function(q_values) * dq_dtheta
    integral_fine = np.trapezoid(y_values, theta)

    theta_coarse = np.linspace(epsilon, 0.5 * np.pi - epsilon, n_points // 2)

    q_coarse = left + (right - left) * np.sin(theta_coarse) ** 2
    dq_dtheta_coarse = (
        2.0 * (right - left)
        * np.sin(theta_coarse)
        * np.cos(theta_coarse)
    )

    y_coarse = function(q_coarse) * dq_dtheta_coarse
    integral_coarse = np.trapezoid(y_coarse, theta_coarse)

    stability = abs(integral_fine - integral_coarse)

    return float(integral_fine), float(stability)


# Integral de acción cerrada usada en la cuantización WBK
def action_integral(
    energy: float,
    params: AnharmonicOscillatorParameters,
    n_points: int = 25000,
) -> tuple[float, float]:
    q_t = positive_turning_point(energy, params)

    quarter_action, stability = integrate_with_turning_point(
        lambda q: classical_momentum(q, energy, params),
        0.0,
        q_t,
        n_points=n_points,
    )

    return 4.0 * quarter_action, 4.0 * stability

# ============================================================
# COMPROBACIÓN DEL CRITERIO DE VALIDEZ WBK
# ============================================================

def compute_validity_ratio_from_wave_number(
    coordinate: np.ndarray,
    wave_number: np.ndarray,
) -> np.ndarray:
    """
    Calcula el cociente adimensional asociado al criterio de validez WBK,

        R(q) =
        | 3/4 (k'(q)/k(q))^2 - 1/2 k''(q)/k(q) | / |k(q)^2|.

    El criterio local requiere R(q) << 1. La función debe aplicarse lejos
    de los puntos de giro, donde la forma WBK estándar es válida.
    """
    dk_dq = np.gradient(wave_number, coordinate, edge_order=2)
    d2k_dq2 = np.gradient(dk_dq, coordinate, edge_order=2)

    with np.errstate(divide="ignore", invalid="ignore"):
        correction = np.abs(
            0.75 * (dk_dq / wave_number) ** 2
            - 0.5 * d2k_dq2 / wave_number
        )
        leading = np.abs(wave_number**2)
        ratio = correction / leading

    return ratio


def summarize_validity_ratio(
    coordinate: np.ndarray,
    wave_number: np.ndarray,
    left: float,
    right: float,
    exclusion_fraction: float = 0.05,
) -> dict:
    """
    Resume el cociente de validez excluyendo una vecindad de los puntos de giro.

    Se excluye el mismo porcentaje a izquierda y derecha del intervalo permitido,
    porque en los puntos de giro k(q) = 0 y el criterio local diverge por
    construcción. Esa región queda descrita por el análisis de Airy.
    """
    interval_length = right - left
    inner_left = left + exclusion_fraction * interval_length
    inner_right = right - exclusion_fraction * interval_length

    ratio = compute_validity_ratio_from_wave_number(coordinate, wave_number)

    mask = (
        (coordinate > inner_left)
        & (coordinate < inner_right)
        & np.isfinite(wave_number)
        & (wave_number > 0.0)
        & np.isfinite(ratio)
    )

    selected_ratio = ratio[mask]
    selected_wave_number = wave_number[mask]

    if selected_ratio.size == 0:
        raise RuntimeError("No valid points were found for the validity check.")

    return {
        "median_validity_ratio": float(np.median(selected_ratio)),
        "percentile_95_validity_ratio": float(np.percentile(selected_ratio, 95)),
        "max_validity_ratio": float(np.max(selected_ratio)),
        "min_wave_number": float(np.min(selected_wave_number)),
        "max_wave_number": float(np.max(selected_wave_number)),
        "n_points_used": int(selected_ratio.size),
        "exclusion_fraction": float(exclusion_fraction),
    }


def compute_validity_checks(
    params: AnharmonicOscillatorParameters,
    wbk_levels: list[dict],
    n_points: int = 20000,
    exclusion_fraction: float = 0.05,
) -> list[dict]:
    """
    Evalúa el criterio de validez WBK en la región permitida de cada nivel.
    """
    rows = []

    for level in wbk_levels:
        n = level["n"]
        energy = level["energy_wbk"]
        q_t = level["turning_point"]

        q_values = np.linspace(-q_t, q_t, n_points)
        momentum = classical_momentum(q_values, energy, params)
        wave_number = momentum / params.hbar

        summary = summarize_validity_ratio(
            coordinate=q_values,
            wave_number=wave_number,
            left=-q_t,
            right=q_t,
            exclusion_fraction=exclusion_fraction,
        )

        rows.append(
            {
                "n": n,
                "energy_wbk": energy,
                "q_left": -q_t,
                "q_right": q_t,
                **summary,
            }
        )

    return rows


def save_validity_table(validity_rows: list[dict]) -> None:
    output_path = RESULTS_DIR / "anharmonic_validity_check.csv"

    columns = [
        "n",
        "energy_wbk",
        "q_left",
        "q_right",
        "median_validity_ratio",
        "percentile_95_validity_ratio",
        "max_validity_ratio",
        "min_wave_number",
        "max_wave_number",
        "n_points_used",
        "exclusion_fraction",
    ]

    with output_path.open("w", encoding="utf-8") as file:
        file.write(",".join(columns) + "\n")

        for row in validity_rows:
            file.write(",".join(str(row[column]) for column in columns) + "\n")


def print_validity_results(validity_rows: list[dict]) -> None:
    print()
    print("WBK validity criterion check")
    print("============================")
    print(
        f"{'n':>3s} "
        f"{'median R':>14s} "
        f"{'P95 R':>14s} "
        f"{'max R':>14s} "
        f"{'q_left':>12s} "
        f"{'q_right':>12s}"
    )

    for row in validity_rows:
        print(
            f"{row['n']:3d} "
            f"{row['median_validity_ratio']:14.6e} "
            f"{row['percentile_95_validity_ratio']:14.6e} "
            f"{row['max_validity_ratio']:14.6e} "
            f"{row['q_left']:12.6f} "
            f"{row['q_right']:12.6f}"
        )

    print()
    print(
        "The criterion is evaluated away from the turning points, "
        "excluding the Airy neighbourhoods."
    )



# Término derecho de la condición de Bohr-Sommerfeld
def bohr_sommerfeld_target(
    n: int,
    params: AnharmonicOscillatorParameters,
) -> float:
    return 2.0 * np.pi * params.hbar * (n + 0.5)


def quantization_residual(
    energy: float,
    n: int,
    params: AnharmonicOscillatorParameters,
) -> float:
    action, _ = action_integral(energy, params)
    target = bohr_sommerfeld_target(n, params)

    return action - target


# Cota superior inicial para encerrar cada raíz WBK
def estimate_energy_upper_bound(
    n: int,
    params: AnharmonicOscillatorParameters,
) -> float:
    harmonic_energy = params.hbar * params.omega * (n + 0.5)
    quartic_scale = params.lambda_quartic * (2.0 * n + 1.0) ** 2

    return 5.0 * (harmonic_energy + quartic_scale + 1.0)


# Cálculo de un nivel mediante la condición de cuantización WBK
def find_wbk_energy(
    n: int,
    params: AnharmonicOscillatorParameters,
) -> tuple[float, float]:
    left = 1.0e-12
    right = estimate_energy_upper_bound(n, params)

    def residual(energy):
        return quantization_residual(energy, n, params)

    f_left = residual(left)
    f_right = residual(right)

    expansion_counter = 0

    while f_left * f_right > 0.0 and expansion_counter < 30:
        right *= 2.0
        f_right = residual(right)
        expansion_counter += 1

    if f_left * f_right > 0.0:
        raise RuntimeError(f"Could not bracket the WBK root for n = {n}.")

    energy = bisection_root(
        residual,
        left,
        right,
        tol=1.0e-11,
        max_iter=250,
    )

    _, action_stability = action_integral(energy, params)

    return energy, action_stability


# Espectro semiclasico WBK
def compute_wbk_spectrum(
    params: AnharmonicOscillatorParameters,
) -> list[dict]:
    levels = []

    for n in range(params.n_max + 1):
        energy, action_stability = find_wbk_energy(n, params)
        action, _ = action_integral(energy, params)
        target = bohr_sommerfeld_target(n, params)
        q_t = positive_turning_point(energy, params)

        levels.append(
            {
                "n": n,
                "energy_wbk": energy,
                "turning_point": q_t,
                "action": action,
                "target": target,
                "action_stability": action_stability,
            }
        )

    return levels


# Hamiltoniano de Schrödinger por diferencias finitas
def build_schrodinger_hamiltonian(
    params: AnharmonicOscillatorParameters,
) -> tuple[np.ndarray, np.ndarray]:
    n_total = params.n_grid_schrodinger

    if n_total < 50:
        raise ValueError("n_grid_schrodinger is too small.")

    q_full = np.linspace(-params.q_box, params.q_box, n_total)
    dq = q_full[1] - q_full[0]

    q_inner = q_full[1:-1]
    n_inner = len(q_inner)

    kinetic_prefactor = params.hbar**2 / (2.0 * params.mass * dq**2)

    diagonal = (
        2.0 * kinetic_prefactor
        + anharmonic_potential(q_inner, params)
    )

    off_diagonal = -kinetic_prefactor * np.ones(n_inner - 1)

    hamiltonian = (
        np.diag(diagonal)
        + np.diag(off_diagonal, k=1)
        + np.diag(off_diagonal, k=-1)
    )

    return q_inner, hamiltonian


# Solución numérica directa de la ecuación de Schrödinger estacionaria
def solve_schrodinger_finite_difference(
    params: AnharmonicOscillatorParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_inner, hamiltonian = build_schrodinger_hamiltonian(params)

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

    n_levels = params.n_max + 1

    selected_values = eigenvalues[:n_levels]
    selected_vectors = eigenvectors[:, :n_levels]

    for i in range(n_levels):
        norm = np.sqrt(np.trapezoid(selected_vectors[:, i] ** 2, q_inner))
        selected_vectors[:, i] = selected_vectors[:, i] / norm

        max_index = np.argmax(np.abs(selected_vectors[:, i]))

        if selected_vectors[max_index, i] < 0.0:
            selected_vectors[:, i] *= -1.0

    return q_inner, selected_values, selected_vectors


# Tabla comparativa entre niveles WBK y niveles numéricos
def build_comparison_table(
    wbk_levels: list[dict],
    numerical_energies: np.ndarray,
) -> list[dict]:
    table = []

    for level in wbk_levels:
        n = level["n"]

        energy_wbk = level["energy_wbk"]
        energy_numerical = float(numerical_energies[n])

        absolute_error = abs(energy_wbk - energy_numerical)
        relative_error = absolute_error / abs(energy_numerical)

        table.append(
            {
                "n": n,
                "energy_wbk": energy_wbk,
                "energy_numerical": energy_numerical,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "turning_point": level["turning_point"],
                "action_stability": level["action_stability"],
            }
        )

    return table


# Exportación de resultados numéricos
def save_comparison_table(table: list[dict]) -> None:
    output_path = RESULTS_DIR / "anharmonic_spectrum_comparison.csv"

    header = (
        "n,"
        "energy_wbk,"
        "energy_numerical,"
        "absolute_error,"
        "relative_error,"
        "turning_point,"
        "action_stability\n"
    )

    with output_path.open("w", encoding="utf-8") as file:
        file.write(header)

        for row in table:
            file.write(
                f"{row['n']},"
                f"{row['energy_wbk']:.12e},"
                f"{row['energy_numerical']:.12e},"
                f"{row['absolute_error']:.12e},"
                f"{row['relative_error']:.12e},"
                f"{row['turning_point']:.12e},"
                f"{row['action_stability']:.12e}\n"
            )


# Impresión de resultados principales en consola
def print_results(
    params: AnharmonicOscillatorParameters,
    table: list[dict],
) -> None:
    print("Quartic anharmonic oscillator")
    print("=============================")
    print(f"mass = {params.mass:.8f}")
    print(f"omega = {params.omega:.8f}")
    print(f"hbar = {params.hbar:.8f}")
    print(f"lambda_quartic = {params.lambda_quartic:.8f}")
    print(f"q_box = {params.q_box:.8f}")
    print(f"n_grid_schrodinger = {params.n_grid_schrodinger}")
    print(f"n_max = {params.n_max}")
    print()

    print(
        f"{'n':>3s} "
        f"{'E_WBK':>16s} "
        f"{'E_num':>16s} "
        f"{'abs_error':>16s} "
        f"{'rel_error':>16s} "
        f"{'q_t':>12s} "
        f"{'action_stab':>14s}"
    )

    for row in table:
        print(
            f"{row['n']:3d} "
            f"{row['energy_wbk']:16.8f} "
            f"{row['energy_numerical']:16.8f} "
            f"{row['absolute_error']:16.8e} "
            f"{row['relative_error']:16.8e} "
            f"{row['turning_point']:12.6f} "
            f"{row['action_stability']:14.3e}"
        )


# Figura del potencial con niveles WBK y niveles numéricos
def make_potential_plot(
    params: AnharmonicOscillatorParameters,
    table: list[dict],
) -> None:
    q = np.linspace(-params.q_box, params.q_box, 4000)
    potential_values = anharmonic_potential(q, params)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    ax.plot(q, potential_values, linewidth=2.0, label=r"$V(q)$")

    for row in table:
        n = row["n"]

        if n <= 5:
            energy_wbk = row["energy_wbk"]
            energy_numerical = row["energy_numerical"]

            ax.axhline(
                energy_wbk,
                linestyle="--",
                linewidth=1.1,
                alpha=0.75,
            )

            ax.axhline(
                energy_numerical,
                linestyle=":",
                linewidth=1.1,
                alpha=0.75,
            )

            ax.text(
                params.q_box * 0.72,
                energy_numerical,
                rf"$n={n}$",
                fontsize=12,
                va="center",
            )

    max_energy = table[min(5, len(table) - 1)]["energy_numerical"]

    ax.plot([], [], linestyle="--", color="black", label="WBK levels")
    ax.plot([], [], linestyle=":", color="black", label="Numerical levels")

    ax.set_xlim(-params.q_box, params.q_box)
    ax.set_ylim(0.0, 1.25 * max_energy)

    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$V(q)$, $E_n$")
    ax.set_title("Quartic anharmonic oscillator")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "anharmonic_potential_levels.pdf")
    fig.savefig(FIGURES_DIR / "anharmonic_potential_levels.png", dpi=300)
    plt.close(fig)


# Figura del error relativo entre el resultado semiclasico y el numérico
def make_error_plot(table: list[dict]) -> None:
    n_values = np.array([row["n"] for row in table])
    relative_errors = np.array([row["relative_error"] for row in table])

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    ax.plot(
        n_values,
        relative_errors,
        marker="o",
        linestyle="-",
        linewidth=1.8,
        label="Relative error",
    )

    ax.set_xlabel(r"$n$")
    ax.set_ylabel(
        r"$|E_n^{\mathrm{WBK}}-E_n^{\mathrm{num}}|/E_n^{\mathrm{num}}$"
    )
    ax.set_title("Relative error of the semiclassical spectrum")

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "anharmonic_relative_error.pdf")
    fig.savefig(FIGURES_DIR / "anharmonic_relative_error.png", dpi=300)
    plt.close(fig)


# Figura de funciones propias numéricas desplazadas por su energía
def make_wavefunction_plot(
    params: AnharmonicOscillatorParameters,
    q_inner: np.ndarray,
    numerical_energies: np.ndarray,
    numerical_wavefunctions: np.ndarray,
) -> None:
    q = np.linspace(-params.q_box, params.q_box, 4000)
    potential_values = anharmonic_potential(q, params)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    ax.plot(q, potential_values, linewidth=2.0, label=r"$V(q)$")

    n_to_plot = min(5, params.n_max)

    for n in range(n_to_plot + 1):
        energy = numerical_energies[n]
        wavefunction = numerical_wavefunctions[:, n]

        scale = 0.25 * params.omega
        shifted_wavefunction = (
            energy
            + scale * wavefunction / np.max(np.abs(wavefunction))
        )

        ax.plot(q_inner, shifted_wavefunction, linewidth=1.2)
        ax.axhline(energy, linestyle=":", linewidth=0.8, alpha=0.5)

        ax.text(
            params.q_box * 0.75,
            energy,
            rf"$n={n}$",
            fontsize=12,
            va="center",
        )

    max_energy = numerical_energies[n_to_plot]

    ax.set_xlim(-params.q_box, params.q_box)
    ax.set_ylim(0.0, 1.35 * max_energy)

    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$V(q)$ and shifted $\psi_n(q)$")
    ax.set_title("Numerical eigenfunctions")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "anharmonic_numerical_wavefunctions.pdf")
    fig.savefig(FIGURES_DIR / "anharmonic_numerical_wavefunctions.png", dpi=300)
    plt.close(fig)


# Ejecución completa del caso: WBK, solución numérica y comparación
def run_case(params: AnharmonicOscillatorParameters) -> dict:
    wbk_levels = compute_wbk_spectrum(params)

    q_inner, numerical_energies, numerical_wavefunctions = (
        solve_schrodinger_finite_difference(params)
    )

    comparison_table = build_comparison_table(wbk_levels, numerical_energies)
    validity_table = compute_validity_checks(params, wbk_levels)

    return {
        "wbk_levels": wbk_levels,
        "q_inner": q_inner,
        "numerical_energies": numerical_energies,
        "numerical_wavefunctions": numerical_wavefunctions,
        "comparison_table": comparison_table,
        "validity_table": validity_table,
    }


# Caso principal
if __name__ == "__main__":
    params = AnharmonicOscillatorParameters(
        mass=1.0,
        omega=1.0,
        hbar=1.0,
        lambda_quartic=0.10,
        n_max=8,
        q_box=8.0,
        n_grid_schrodinger=900,
    )

    results = run_case(params)

    print_results(
        params,
        results["comparison_table"],
    )

    save_comparison_table(results["comparison_table"])
    print_validity_results(results["validity_table"])
    save_validity_table(results["validity_table"])

    if MAKE_PLOTS:
        make_potential_plot(
            params,
            results["comparison_table"],
        )

        make_error_plot(
            results["comparison_table"],
        )

        make_wavefunction_plot(
            params,
            results["q_inner"],
            results["numerical_energies"],
            results["numerical_wavefunctions"],
        )

        print()
        print(f"Results saved in: {RESULTS_DIR.resolve()}")
        print(f"Figures saved in: {FIGURES_DIR.resolve()}")