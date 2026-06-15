from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.ticker import FormatStrFormatter


# =============================================================================
# CARPETAS DEL REPOSITORIO
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CÁLCULO DEL DESDOBLAMIENTO DE NIVELES EN EL DOBLE POZO
# =============================================================================

MAKE_PLOT = True


@dataclass
class DoubleWellParameters:
    hbar: float = 1.0
    mass: float = 1.0
    lam: float = 0.10
    a: float = 3.0
    q_max: float = 10.0
    n_grid: int = 1800
    n_doublets: int = 3


# =============================================================================
# POTENCIAL Y MAGNITUDES BÁSICAS
# =============================================================================

def potential(q, params: DoubleWellParameters):
    return params.lam * (np.asarray(q) ** 2 - params.a**2) ** 2


def barrier_height(params: DoubleWellParameters) -> float:
    return params.lam * params.a**4


def harmonic_frequency(params: DoubleWellParameters) -> float:
    return np.sqrt(8.0 * params.lam * params.a**2 / params.mass)


def local_harmonic_energy(n: int, params: DoubleWellParameters) -> float:
    omega_0 = harmonic_frequency(params)
    return params.hbar * omega_0 * (n + 0.5)


def inner_turning_points_for_energy(
    energy: float,
    params: DoubleWellParameters,
) -> tuple[float, float]:
    v_barrier = barrier_height(params)

    if energy <= 0.0:
        raise ValueError("La energía debe ser positiva.")

    if energy >= v_barrier:
        raise ValueError(
            "La energía local está por encima de la barrera. "
            f"E = {energy:.6f}, V(0) = {v_barrier:.6f}. "
            "Aumenta lambda o a, o reduce n_doublets."
        )

    inner_argument = params.a**2 - np.sqrt(energy / params.lam)

    if inner_argument <= 0.0:
        raise ValueError("No hay puntos de giro internos reales para esta energía.")

    q_inner = np.sqrt(inner_argument)

    return -q_inner, q_inner


def outer_turning_points_for_energy(
    energy: float,
    params: DoubleWellParameters,
) -> tuple[float, float]:
    """
    Devuelve los puntos de giro externos q1 y q4 para una energía por debajo
    de la barrera:
        q1 = -sqrt(a^2 + sqrt(E/lambda))
        q4 = +sqrt(a^2 + sqrt(E/lambda))
    """
    v_barrier = barrier_height(params)

    if energy <= 0.0:
        raise ValueError("La energía debe ser positiva.")

    if energy >= v_barrier:
        raise ValueError(
            "La energía debe estar por debajo de la barrera para usar "
            "los puntos de giro externos en esta figura."
        )

    outer_argument = params.a**2 + np.sqrt(energy / params.lam)
    q_outer = np.sqrt(outer_argument)

    return -q_outer, q_outer


# =============================================================================
# INTEGRACIÓN NUMÉRICA BAJO LA BARRERA
# =============================================================================

def integrate_with_turning_points(
    function,
    left: float,
    right: float,
    n_points: int = 40000,
    epsilon: float = 1.0e-8,
) -> tuple[float, float]:
    theta = np.linspace(epsilon, 0.5 * np.pi - epsilon, n_points)

    q = left + (right - left) * np.sin(theta) ** 2
    dq_dtheta = 2.0 * (right - left) * np.sin(theta) * np.cos(theta)

    y = function(q) * dq_dtheta
    integral_fine = np.trapezoid(y, theta)

    theta_coarse = np.linspace(epsilon, 0.5 * np.pi - epsilon, n_points // 2)

    q_coarse = left + (right - left) * np.sin(theta_coarse) ** 2
    dq_dtheta_coarse = (
        2.0
        * (right - left)
        * np.sin(theta_coarse)
        * np.cos(theta_coarse)
    )

    y_coarse = function(q_coarse) * dq_dtheta_coarse
    integral_coarse = np.trapezoid(y_coarse, theta_coarse)

    stability = abs(integral_fine - integral_coarse)

    return float(integral_fine), float(stability)


def barrier_action(
    energy: float,
    params: DoubleWellParameters,
) -> tuple[float, float, float, float]:
    q2, q3 = inner_turning_points_for_energy(energy, params)

    def integrand(q):
        value = potential(q, params) - energy
        return np.sqrt(2.0 * params.mass * np.maximum(value, 0.0))

    action, stability = integrate_with_turning_points(
        integrand,
        q2,
        q3,
        n_points=40000,
    )

    return action, stability, q2, q3


def wbk_splitting(
    n: int,
    params: DoubleWellParameters,
) -> dict:
    energy_0 = local_harmonic_energy(n, params)
    action, stability, q2, q3 = barrier_action(energy_0, params)
    omega_0 = harmonic_frequency(params)

    delta = (
        params.hbar
        * omega_0
        / np.pi
        * np.exp(-action / params.hbar)
    )

    return {
        "n": n,
        "local_energy": energy_0,
        "q2": q2,
        "q3": q3,
        "barrier_action": action,
        "barrier_action_stability": stability,
        "wbk_splitting": delta,
    }


# =============================================================================
# DIAGONALIZACIÓN NUMÉRICA DE SCHRÖDINGER
# =============================================================================

def build_hamiltonian_matrix(params: DoubleWellParameters) -> tuple[np.ndarray, np.ndarray]:
    q = np.linspace(-params.q_max, params.q_max, params.n_grid)
    dq = q[1] - q[0]

    q_inner = q[1:-1]
    n_inner = len(q_inner)

    kinetic_diagonal = params.hbar**2 / (params.mass * dq**2)
    kinetic_off_diagonal = -params.hbar**2 / (2.0 * params.mass * dq**2)

    diagonal = kinetic_diagonal + potential(q_inner, params)
    off_diagonal = kinetic_off_diagonal * np.ones(n_inner - 1)

    hamiltonian = (
        np.diag(diagonal)
        + np.diag(off_diagonal, k=1)
        + np.diag(off_diagonal, k=-1)
    )

    return q_inner, hamiltonian


def numerical_doublets(params: DoubleWellParameters) -> dict:
    _, hamiltonian = build_hamiltonian_matrix(params)
    eigenvalues, _ = np.linalg.eigh(hamiltonian)

    n_needed = 2 * params.n_doublets
    selected = eigenvalues[:n_needed]

    doublets = []

    for n in range(params.n_doublets):
        e_minus = selected[2 * n]
        e_plus = selected[2 * n + 1]
        delta = e_plus - e_minus

        doublets.append(
            {
                "n": n,
                "e_minus": float(e_minus),
                "e_plus": float(e_plus),
                "numerical_splitting": float(delta),
            }
        )

    return {
        "eigenvalues": eigenvalues,
        "doublets": doublets,
    }


# =============================================================================
# RESULTADOS Y TABLAS
# =============================================================================

def compute_comparison_table(params: DoubleWellParameters) -> list[dict]:
    v_barrier = barrier_height(params)

    for n in range(params.n_doublets):
        energy_0 = local_harmonic_energy(n, params)

        if energy_0 >= v_barrier:
            raise ValueError(
                "Con los parámetros actuales no todos los dobletes pedidos "
                "están por debajo de la barrera. "
                f"Para n = {n}, E_n^(0) = {energy_0:.6f}, "
                f"mientras que V(0) = {v_barrier:.6f}. "
                "Aumenta lambda o a, o reduce n_doublets."
            )

    numerical = numerical_doublets(params)
    rows = []

    for doublet in numerical["doublets"]:
        n = doublet["n"]
        semiclassical = wbk_splitting(n, params)

        row = {
            "n": n,
            "E_minus_num": doublet["e_minus"],
            "E_plus_num": doublet["e_plus"],
            "Delta_E_num": doublet["numerical_splitting"],
            "E_local": semiclassical["local_energy"],
            "q2": semiclassical["q2"],
            "q3": semiclassical["q3"],
            "S_B": semiclassical["barrier_action"],
            "S_B_stability": semiclassical["barrier_action_stability"],
            "Delta_E_WBK": semiclassical["wbk_splitting"],
        }

        row["ratio_WBK_to_num"] = row["Delta_E_WBK"] / row["Delta_E_num"]

        rows.append(row)

    return rows


def print_summary(params: DoubleWellParameters, rows: list[dict]) -> None:
    omega_0 = harmonic_frequency(params)
    v_barrier = barrier_height(params)

    print("Double-well tunnelling splitting")
    print("================================")
    print(f"hbar = {params.hbar:.6f}")
    print(f"m = {params.mass:.6f}")
    print(f"lambda = {params.lam:.6f}")
    print(f"a = {params.a:.6f}")
    print(f"V(0) = {v_barrier:.6f}")
    print(f"omega_0 = {omega_0:.6f}")
    print(f"q_max = {params.q_max:.6f}")
    print(f"n_grid = {params.n_grid}")
    print()
    print("Splitting comparison")
    print("--------------------")
    print(
        "n  "
        "E_minus_num       "
        "E_plus_num        "
        "Delta_E_num       "
        "E_local           "
        "S_B               "
        "Delta_E_WBK       "
        "WBK/num"
    )

    for row in rows:
        print(
            f"{row['n']:d}  "
            f"{row['E_minus_num']:.10e}  "
            f"{row['E_plus_num']:.10e}  "
            f"{row['Delta_E_num']:.10e}  "
            f"{row['E_local']:.10e}  "
            f"{row['S_B']:.10e}  "
            f"{row['Delta_E_WBK']:.10e}  "
            f"{row['ratio_WBK_to_num']:.6e}"
        )


def save_csv(rows: list[dict]) -> None:
    csv_path = RESULTS_DIR / "double_well_splitting.csv"

    columns = [
        "n",
        "E_minus_num",
        "E_plus_num",
        "Delta_E_num",
        "E_local",
        "q2",
        "q3",
        "S_B",
        "S_B_stability",
        "Delta_E_WBK",
        "ratio_WBK_to_num",
    ]

    with csv_path.open("w", encoding="utf-8") as file:
        file.write(",".join(columns) + "\n")

        for row in rows:
            values = [str(row[column]) for column in columns]
            file.write(",".join(values) + "\n")

    print()
    print(f"CSV table saved in: {csv_path.resolve()}")


def save_latex_table(rows: list[dict]) -> None:
    latex_path = RESULTS_DIR / "double_well_splitting_table.tex"

    with latex_path.open("w", encoding="utf-8") as file:
        file.write("\\begin{table}[h]\n")
        file.write("\\centering\n")
        file.write("\\small\n")
        file.write("\\begin{tabular}{ccccc}\n")
        file.write("\\toprule\n")
        file.write(
            "$n$ & "
            "$E_n^{-}$ & "
            "$E_n^{+}$ & "
            "$\\Delta E_n^{\\mathrm{num}}$ & "
            "$\\Delta E_n^{\\mathrm{WBK}}$ \\\\\n"
        )
        file.write("\\midrule\n")

        for row in rows:
            file.write(
                f"{row['n']} & "
                f"{row['E_minus_num']:.6f} & "
                f"{row['E_plus_num']:.6f} & "
                f"{row['Delta_E_num']:.3e} & "
                f"{row['Delta_E_WBK']:.3e} \\\\\n"
            )

        file.write("\\bottomrule\n")
        file.write("\\end{tabular}\n")
        file.write(
            "\\caption{Numerical and semiclassical tunnelling splittings "
            "for the symmetric double-well potential.}\n"
        )
        file.write("\\label{tab:double_well_splitting}\n")
        file.write("\\end{table}\n")

    print(f"LaTeX table saved in: {latex_path.resolve()}")


# =============================================================================
# FIGURA ÚNICA DEL DOBLE POZO CON VENTANAS DE ZOOM
# =============================================================================

def make_splitting_plot(params: DoubleWellParameters, rows: list[dict]) -> None:
    """
    Genera una única figura:
    - potencial de doble pozo,
    - niveles sin desdoblar, definidos como el centro de cada doblete,
    - niveles desdoblados numéricos,
    - ventanas de zoom a la derecha, con escala vertical reescalada.

    Las ventanas no usan la misma escala vertical. El reescalado se elige para
    que los splittings pequeños sean visibles, pero manteniendo una apertura
    relativa creciente con n.
    """
    q_plot_min = -5.0
    q_plot_max = 5.0
    q = np.linspace(q_plot_min, q_plot_max, 4000)
    v = potential(q, params)

    potential_color = "tab:blue"
    split_color = "tab:orange"
    unsplit_color = "tab:green"

    fig = plt.figure(figsize=(12.0, 6.4))

    # Eje principal. Se deja espacio a la derecha para las ventanas externas.
    ax = fig.add_axes([0.075, 0.115, 0.65, 0.75])

    ax.plot(
        q,
        v,
        color=potential_color,
        linewidth=2.2,
        label=r"$V(q)$",
        zorder=2,
    )

    level_data = []

    for row in rows:
        e_minus = row["E_minus_num"]
        e_plus = row["E_plus_num"]
        e_center = 0.5 * (e_minus + e_plus)
        delta_e = e_plus - e_minus

        ql_center, qr_center = outer_turning_points_for_energy(e_center, params)
        ql_minus, qr_minus = outer_turning_points_for_energy(e_minus, params)
        ql_plus, qr_plus = outer_turning_points_for_energy(e_plus, params)

        level_data.append(
            {
                "n": row["n"],
                "e_minus": e_minus,
                "e_plus": e_plus,
                "e_center": e_center,
                "delta_e": delta_e,
                "ql_center": ql_center,
                "qr_center": qr_center,
                "ql_minus": ql_minus,
                "qr_minus": qr_minus,
                "ql_plus": ql_plus,
                "qr_plus": qr_plus,
            }
        )

        ax.hlines(
            e_center,
            ql_center,
            qr_center,
            color=unsplit_color,
            linewidth=1.8,
            linestyle="--",
            zorder=3,
            label="W/o tunnelling" if row["n"] == 0 else None,
        )

        ax.hlines(
            e_minus,
            ql_minus,
            qr_minus,
            color=split_color,
            linewidth=1.55,
            zorder=4,
            label="W tunelling" if row["n"] == 0 else None,
        )

        ax.hlines(
            e_plus,
            ql_plus,
            qr_plus,
            color=split_color,
            linewidth=1.55,
            zorder=4,
        )

    ax.set_xlim(q_plot_min, q_plot_max)
    ax.set_ylim(-0.05, barrier_height(params) * 1.12)
    ax.set_xlabel(r"$q$", fontsize=18)
    ax.set_ylabel(r"$V(q),\ E_n$", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper center", fontsize=16, ncol=3)

    fig.suptitle(
        "Tunnelling-induced doublet splitting in a symmetric double well",
        fontsize=20,
        y=0.95,
    )

    # Reescalado vertical progresivo.
    min_delta = min(level["delta_e"] for level in level_data)
    max_delta = max(level["delta_e"] for level in level_data)

    if max_delta > min_delta:
        for level in level_data:
            log_position = (
                np.log(level["delta_e"] / min_delta)
                / np.log(max_delta / min_delta)
            )
            level["target_gap_fraction"] = 0.22 + 0.42 * log_position
    else:
        for level in level_data:
            level["target_gap_fraction"] = 0.45

    # Ventanas externas: más estrechas.
    inset_positions = [
        [0.75, 0.15, 0.15, 0.15],
        [0.75, 0.45, 0.15, 0.15],
        [0.75, 0.70, 0.15, 0.15],
    ]

    for level, inset_pos in zip(level_data, inset_positions):
        inset = fig.add_axes(inset_pos)

        e_minus = level["e_minus"]
        e_plus = level["e_plus"]
        e_center = level["e_center"]
        delta_e = level["delta_e"]

        target_gap_fraction = level["target_gap_fraction"]
        y_half_width = 0.5 * delta_e / target_gap_fraction

        y_min = e_center - y_half_width
        y_max = e_center + y_half_width

        # Las líneas ocupan toda la ventana horizontalmente.
        inset.hlines(
            e_center,
            0.0,
            1.0,
            color=unsplit_color,
            linewidth=2.2,
            linestyle="--",
            zorder=2,
        )
        inset.hlines(
            e_minus,
            0.0,
            1.0,
            color=split_color,
            linewidth=2.2,
            zorder=3,
        )
        inset.hlines(
            e_plus,
            0.0,
            1.0,
            color=split_color,
            linewidth=2.2,
            zorder=3,
        )

        inset.set_xlim(0.0, 1.0)
        inset.set_ylim(y_min, y_max)
        inset.set_xticks([])

        # Escala vertical propia de cada ventana, situada a la derecha.
        inset.set_yticks([y_min, e_center, y_max])
        inset.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))

        inset.yaxis.tick_right()
        inset.yaxis.set_label_position("right")

        inset.tick_params(
            axis="y",
            labelsize=14,
            labelright=True,
            labelleft=False,
            right=True,
            left=False,
        )

        # Etiqueta n centrada encima de cada ventana.
        inset.text(
            0.5,
            1.0,
            rf"$n={level['n']}$",
            transform=inset.transAxes,
            ha="center",
            va="bottom",
            fontsize=15,
            clip_on=False,
        )

        # Valor del splitting centrado debajo de cada ventana.
        inset.text(
            0.5,
            -0.15,
            rf"$\Delta E={delta_e:.2e}$",
            transform=inset.transAxes,
            ha="center",
            va="top",
            fontsize=15,
            clip_on=False,
        )

        for spine in inset.spines.values():
            spine.set_linewidth(0.95)

        # Región ampliada en el eje principal, cerca del punto de giro derecho.
        q_right_ref = level["qr_center"]
        x_box_left = q_right_ref - 0.5
        x_box_right = q_right_ref -0.1

        box_height_factor = 2

        # Altura mínima del recuadro en la escala vertical del potencial.
        # Esto solo afecta al recuadro del gráfico principal, no a la escala
        # vertical de la ventana de zoom.
        min_box_half_height = 0.05

        y_box_half_height = max(
            box_height_factor * y_half_width,
            min_box_half_height,
        )

        y_box_min = e_center - y_box_half_height
        y_box_max = e_center + y_box_half_height


        rect = Rectangle(
            (x_box_left, y_box_min),
            x_box_right - x_box_left,
            y_box_max - y_box_min,
            fill=False,
            edgecolor="0.35",
            linewidth=0.95,
            zorder=5,
        )
        ax.add_patch(rect)

        # Conectores hacia la ventana.
        con_top = ConnectionPatch(
            xyA=(x_box_right, y_max),
            coordsA=ax.transData,
            xyB=(0.0, y_max),
            coordsB=inset.transData,
            color="0.35",
            linewidth=0.9,
        )
        con_bottom = ConnectionPatch(
            xyA=(x_box_right, y_min),
            coordsA=ax.transData,
            xyB=(0.0, y_min),
            coordsB=inset.transData,
            color="0.35",
            linewidth=0.9,
        )

        fig.add_artist(con_top)
        fig.add_artist(con_bottom)

    # Guardar figuras.
    fig.savefig(FIGURES_DIR / "DW_Desdoblamiento.pdf")
    fig.savefig(FIGURES_DIR / "DW_Desdoblamiento.png", dpi=300)
    plt.close(fig)

    print(f"Figure saved in: {(FIGURES_DIR / 'DW_Desdoblamiento.png').resolve()}")


# =============================================================================
# CASO PRINCIPAL DEL CÁLCULO
# =============================================================================

if __name__ == "__main__":
    params = DoubleWellParameters(
        hbar=1.0,
        mass=1.0,
        lam=0.10,
        a=3.0,
        q_max=10.0,
        n_grid=1800,
        n_doublets=3,
    )

    rows = compute_comparison_table(params)

    print_summary(params, rows)
    save_csv(rows)
    save_latex_table(rows)

    if MAKE_PLOT:
        make_splitting_plot(params, rows)

    print()
    print(f"Figures saved in: {FIGURES_DIR.resolve()}")
    print(f"Results saved in: {RESULTS_DIR.resolve()}")