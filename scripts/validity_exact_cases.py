import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ============================================================
# COMPROBACIÓN NUMÉRICA DEL CRITERIO DE VALIDEZ WBK
# Casos exactos:
#   1) Oscilador armónico unidimensional
#   2) Oscilador armónico radial tridimensional
#   3) Potencial de Coulomb radial
#
# Unidades reducidas:
#   hbar = 1
#   m = 1
#   omega = 1
#   kappa = 1
#
# El criterio local se evalúa como:
#
#   R(x) =
#   | 3/4 (k'(x)/k(x))^2 - 1/2 k''(x)/k(x) | / |k(x)^2|
#
# donde
#
#   k(x) = sqrt(2m(E - V_eff(x))) / hbar
#
# en la región clásicamente permitida.
#
# Cerca de los puntos de giro, k(x) -> 0 y el criterio local
# diverge. Esa región no debe evaluarse con la forma WBK estándar,
# sino mediante el análisis local de Airy. Por eso se excluye una
# fracción pequeña de cada extremo del intervalo permitido.
# ============================================================


@dataclass
class ValidityResult:
    case_name: str
    quantum_numbers: str
    x_left: float
    x_right: float
    median_ratio: float
    percentile_95_ratio: float
    max_ratio: float
    min_k: float
    max_k: float
    n_points_used: int
    exclusion_fraction: float


# ============================================================
# CARPETAS DEL REPOSITORIO
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PARÁMETROS GLOBALES
# ============================================================

HBAR = 1.0
MASS = 1.0
OMEGA = 1.0
KAPPA = 1.0

N_GRID = 20000

# Fracción excluida cerca de cada punto de giro.
# Por ejemplo, 0.05 elimina el 5 % inicial y final del intervalo permitido.
EXCLUSION_FRACTION = 0.05


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def compute_validity_ratio(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Calcula el cociente adimensional del criterio de validez WBK.

    Parámetros
    ----------
    x:
        Malla de coordenadas.
    k:
        Número de onda local k(x).

    Devuelve
    --------
    ratio:
        Cociente adimensional R(x). El criterio WBK requiere R(x) << 1.
    """

    dk_dx = np.gradient(k, x, edge_order=2)
    d2k_dx2 = np.gradient(dk_dx, x, edge_order=2)

    with np.errstate(divide="ignore", invalid="ignore"):
        correction = np.abs(
            0.75 * (dk_dx / k) ** 2
            - 0.5 * d2k_dx2 / k
        )

        leading = np.abs(k ** 2)
        ratio = correction / leading

    return ratio


def summarize_ratio(
    case_name: str,
    quantum_numbers: str,
    x: np.ndarray,
    k: np.ndarray,
    x_left: float,
    x_right: float,
    exclusion_fraction: float = EXCLUSION_FRACTION,
) -> ValidityResult:
    """
    Resume el cociente de validez excluyendo las vecindades de los puntos
    de giro.

    La exclusión se hace en coordenada, no en valor de k, para evitar
    mezclar el fallo físico del criterio en los puntos de giro con
    problemas puramente numéricos.
    """

    interval_length = x_right - x_left
    inner_left = x_left + exclusion_fraction * interval_length
    inner_right = x_right - exclusion_fraction * interval_length

    mask = (
        (x > inner_left)
        & (x < inner_right)
        & np.isfinite(k)
        & (k > 0.0)
    )

    ratio = compute_validity_ratio(x, k)
    ratio_masked = ratio[mask]
    k_masked = k[mask]

    ratio_masked = ratio_masked[np.isfinite(ratio_masked)]

    if ratio_masked.size == 0:
        raise RuntimeError(
            f"No hay puntos válidos para evaluar el criterio en {case_name}."
        )

    return ValidityResult(
        case_name=case_name,
        quantum_numbers=quantum_numbers,
        x_left=x_left,
        x_right=x_right,
        median_ratio=float(np.median(ratio_masked)),
        percentile_95_ratio=float(np.percentile(ratio_masked, 95)),
        max_ratio=float(np.max(ratio_masked)),
        min_k=float(np.min(k_masked)),
        max_k=float(np.max(k_masked)),
        n_points_used=int(ratio_masked.size),
        exclusion_fraction=float(exclusion_fraction),
    )


def print_result(result: ValidityResult) -> None:
    """
    Imprime el resultado en formato legible.
    """

    print("=" * 78)
    print(f"Case: {result.case_name}")
    print(f"Quantum numbers: {result.quantum_numbers}")
    print(f"Allowed interval: [{result.x_left:.8f}, {result.x_right:.8f}]")
    print(f"Points used after excluding turning-point regions: {result.n_points_used}")
    print(f"min(k) in evaluated region: {result.min_k:.6e}")
    print(f"max(k) in evaluated region: {result.max_k:.6e}")
    print(f"median validity ratio: {result.median_ratio:.6e}")
    print(f"95th percentile validity ratio: {result.percentile_95_ratio:.6e}")
    print(f"max validity ratio: {result.max_ratio:.6e}")
    print("=" * 78)
    print()


def save_validity_csv(results: list[ValidityResult]) -> None:
    """
    Guarda la comprobación del criterio de validez en formato CSV.
    """

    csv_path = RESULTS_DIR / "exact_cases_validity_check.csv"

    columns = [
        "case_name",
        "quantum_numbers",
        "x_left",
        "x_right",
        "median_ratio",
        "percentile_95_ratio",
        "max_ratio",
        "min_k",
        "max_k",
        "n_points_used",
        "exclusion_fraction",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for result in results:
            writer.writerow({column: getattr(result, column) for column in columns})

    print(f"CSV table saved in: {csv_path.resolve()}")


def save_validity_latex_table(results: list[ValidityResult]) -> None:
    """
    Guarda una tabla LaTeX compacta con los resultados principales.
    """

    latex_path = RESULTS_DIR / "exact_cases_validity_check_table.tex"

    with latex_path.open("w", encoding="utf-8") as file:
        file.write("\\begin{table}[h]\n")
        file.write("\\centering\n")
        file.write("\\small\n")
        file.write("\\begin{tabular}{llccc}\n")
        file.write("\\toprule\n")
        file.write(
            "Case & Quantum numbers & "
            "$\\mathrm{median}(R)$ & "
            "$P_{95}(R)$ & "
            "$\\max(R)$ \\\\\n"
        )
        file.write("\\midrule\n")

        for result in results:
            file.write(
                f"{result.case_name} & "
                f"{result.quantum_numbers} & "
                f"{result.median_ratio:.3e} & "
                f"{result.percentile_95_ratio:.3e} & "
                f"{result.max_ratio:.3e} \\\\\n"
            )

        file.write("\\bottomrule\n")
        file.write("\\end{tabular}\n")
        file.write(
            "\\caption{Numerical check of the local WBK validity ratio "
            "for the exactly solvable benchmark cases. The Airy neighbourhoods "
            "of the turning points are excluded.}\n"
        )
        file.write("\\label{tab:exact_cases_validity_check}\n")
        file.write("\\end{table}\n")

    print(f"LaTeX table saved in: {latex_path.resolve()}")


# ============================================================
# 1) OSCILADOR ARMÓNICO UNIDIMENSIONAL
# ============================================================

def harmonic_1d_energy(n: int) -> float:
    """
    Energía exacta del oscilador armónico unidimensional.
    """
    return HBAR * OMEGA * (n + 0.5)


def harmonic_1d_potential(q: np.ndarray) -> np.ndarray:
    """
    Potencial del oscilador armónico unidimensional.
    """
    return 0.5 * MASS * OMEGA ** 2 * q ** 2


def check_harmonic_1d(n_values: list[int]) -> list[ValidityResult]:
    """
    Comprueba el criterio de validez en el oscilador armónico 1D.
    """

    results = []

    for n in n_values:
        energy = harmonic_1d_energy(n)

        q_turn = np.sqrt(2.0 * energy / (MASS * OMEGA ** 2))
        q_left = -q_turn
        q_right = q_turn

        q = np.linspace(q_left, q_right, N_GRID)
        potential = harmonic_1d_potential(q)

        k_squared = 2.0 * MASS * (energy - potential) / HBAR ** 2
        k_squared = np.maximum(k_squared, 0.0)
        k = np.sqrt(k_squared)

        result = summarize_ratio(
            case_name="1D harmonic oscillator",
            quantum_numbers=f"n = {n}",
            x=q,
            k=k,
            x_left=q_left,
            x_right=q_right,
        )

        results.append(result)

    return results


# ============================================================
# 2) OSCILADOR ARMÓNICO RADIAL TRIDIMENSIONAL
# ============================================================

def harmonic_3d_energy(n_radial: int, ell: int) -> float:
    """
    Energía exacta del oscilador armónico tridimensional isotrópico.
    """
    return HBAR * OMEGA * (2 * n_radial + ell + 1.5)


def harmonic_3d_effective_potential(r: np.ndarray, ell: int) -> np.ndarray:
    """
    Potencial efectivo radial con corrección de Langer.
    """
    langer = (ell + 0.5) ** 2
    return (
        0.5 * MASS * OMEGA ** 2 * r ** 2
        + HBAR ** 2 * langer / (2.0 * MASS * r ** 2)
    )


def harmonic_3d_turning_points(energy: float, ell: int) -> tuple[float, float]:
    """
    Puntos de giro radiales del oscilador armónico 3D con corrección de Langer.
    """

    langer = ell + 0.5
    discriminant = energy ** 2 - (HBAR * OMEGA * langer) ** 2

    if discriminant <= 0.0:
        raise ValueError(
            f"No hay dos puntos de giro reales para E={energy}, ell={ell}."
        )

    root = np.sqrt(discriminant)

    r1_squared = (energy - root) / (MASS * OMEGA ** 2)
    r2_squared = (energy + root) / (MASS * OMEGA ** 2)

    return np.sqrt(r1_squared), np.sqrt(r2_squared)


def check_harmonic_3d(states: list[tuple[int, int]]) -> list[ValidityResult]:
    """
    Comprueba el criterio de validez en el oscilador armónico radial 3D.
    Cada estado se da como (n_radial, ell).
    """

    results = []

    for n_radial, ell in states:
        energy = harmonic_3d_energy(n_radial, ell)
        r_left, r_right = harmonic_3d_turning_points(energy, ell)

        r = np.linspace(r_left, r_right, N_GRID)
        potential_eff = harmonic_3d_effective_potential(r, ell)

        k_squared = 2.0 * MASS * (energy - potential_eff) / HBAR ** 2
        k_squared = np.maximum(k_squared, 0.0)
        k = np.sqrt(k_squared)

        result = summarize_ratio(
            case_name="3D radial harmonic oscillator",
            quantum_numbers=f"n_r = {n_radial}, ell = {ell}",
            x=r,
            k=k,
            x_left=r_left,
            x_right=r_right,
        )

        results.append(result)

    return results


# ============================================================
# 3) POTENCIAL DE COULOMB RADIAL
# ============================================================

def coulomb_energy(n_principal: int) -> float:
    """
    Energía exacta del problema de Coulomb en unidades reducidas.
    """
    return -MASS * KAPPA ** 2 / (2.0 * HBAR ** 2 * n_principal ** 2)


def coulomb_effective_potential(r: np.ndarray, ell: int) -> np.ndarray:
    """
    Potencial efectivo radial de Coulomb con corrección de Langer.
    """
    langer = (ell + 0.5) ** 2
    return (
        -KAPPA / r
        + HBAR ** 2 * langer / (2.0 * MASS * r ** 2)
    )


def coulomb_turning_points(n_radial: int, ell: int) -> tuple[float, float, int, float]:
    """
    Puntos de giro radiales del potencial de Coulomb con corrección de Langer.

    Se usa:
        n = n_radial + ell + 1
    """

    n_principal = n_radial + ell + 1
    energy = coulomb_energy(n_principal)
    abs_energy = abs(energy)

    discriminant = KAPPA ** 2 - (
        2.0 * abs_energy * HBAR ** 2 * (ell + 0.5) ** 2 / MASS
    )

    if discriminant <= 0.0:
        raise ValueError(
            f"No hay dos puntos de giro reales para n_r={n_radial}, ell={ell}."
        )

    root = np.sqrt(discriminant)

    r1 = (KAPPA - root) / (2.0 * abs_energy)
    r2 = (KAPPA + root) / (2.0 * abs_energy)

    return r1, r2, n_principal, energy


def check_coulomb_radial(states: list[tuple[int, int]]) -> list[ValidityResult]:
    """
    Comprueba el criterio de validez en el problema radial de Coulomb.
    Cada estado se da como (n_radial, ell).
    """

    results = []

    for n_radial, ell in states:
        r_left, r_right, n_principal, energy = coulomb_turning_points(
            n_radial=n_radial,
            ell=ell,
        )

        r = np.linspace(r_left, r_right, N_GRID)
        potential_eff = coulomb_effective_potential(r, ell)

        k_squared = 2.0 * MASS * (energy - potential_eff) / HBAR ** 2
        k_squared = np.maximum(k_squared, 0.0)
        k = np.sqrt(k_squared)

        result = summarize_ratio(
            case_name="3D radial Coulomb potential",
            quantum_numbers=(
                f"n = {n_principal}, n_r = {n_radial}, ell = {ell}"
            ),
            x=r,
            k=k,
            x_left=r_left,
            x_right=r_right,
        )

        results.append(result)

    return results


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

def main() -> None:
    """
    Ejecuta todas las comprobaciones.
    """

    print()
    print("WBK validity criterion check for exact benchmark cases")
    print("Units: hbar = 1, m = 1, omega = 1, kappa = 1")
    print(f"Grid points per interval: {N_GRID}")
    print(f"Turning-point exclusion fraction: {EXCLUSION_FRACTION}")
    print()

    all_results: list[ValidityResult] = []

    # Estados del oscilador armónico 1D.
    harmonic_1d_states = [0, 1, 2, 5, 10, 20]
    all_results.extend(check_harmonic_1d(harmonic_1d_states))

    # Estados del oscilador armónico radial 3D.
    # Cada par es (n_radial, ell).
    harmonic_3d_states = [
        (0, 0),
        (1, 0),
        (0, 2),
        (3, 1),
        (5, 0),
    ]
    all_results.extend(check_harmonic_3d(harmonic_3d_states))

    # Estados del problema radial de Coulomb.
    # Cada par es (n_radial, ell), con n = n_radial + ell + 1.
    coulomb_states = [
        (0, 0),
        (1, 0),
        (0, 1),
        (3, 0),
        (2, 2),
    ]
    all_results.extend(check_coulomb_radial(coulomb_states))

    for result in all_results:
        print_result(result)

    save_validity_csv(all_results)
    save_validity_latex_table(all_results)

    print()
    print(f"Results saved in: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
