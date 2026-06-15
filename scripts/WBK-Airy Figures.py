import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# Configuración gráfica
plt.rcParams.update({
    "font.size": 20,
    "axes.labelsize": 22,
    "axes.titlesize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})


# Parámetros del problema modelo
HBAR = 1.0
MASS = 1.0
ENERGY = 0.0
TURNING_POINT = 0.0

Q_MIN = -12.0
Q_MAX = 4.0
N_POINTS = 7000


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def potential(q):
    return q + 0.06 * q**2


def k_allowed(q):
    value = 2.0 * MASS * (ENERGY - potential(q))
    return np.sqrt(np.maximum(value, 0.0)) / HBAR


def kappa_forbidden(q):
    value = 2.0 * MASS * (potential(q) - ENERGY)
    return np.sqrt(np.maximum(value, 0.0)) / HBAR


# Valores iniciales de Ai(0) y Ai'(0)
AI0 = 1.0 / (3.0 ** (2.0 / 3.0) * math.gamma(2.0 / 3.0))
AIP0 = -1.0 / (3.0 ** (1.0 / 3.0) * math.gamma(1.0 / 3.0))


def airy_ai(x_values):
    """
    Evalúa numéricamente la función de Airy Ai(x) resolviendo y'' - x y = 0.

    La integración se realiza desde x = 0 con un método de Runge-Kutta de
    cuarto orden, usando los valores exactos Ai(0) y Ai'(0).
    """
    x_values = np.asarray(x_values, dtype=float)
    ai_values = np.empty_like(x_values)

    order = np.argsort(x_values)
    x_sorted = x_values[order]

    x_positive = x_sorted[x_sorted >= 0.0]
    x_negative = x_sorted[x_sorted < 0.0]

    def rk4_step(x, y, dy, h):
        k1_y = h * dy
        k1_dy = h * x * y

        k2_y = h * (dy + 0.5 * k1_dy)
        k2_dy = h * (x + 0.5 * h) * (y + 0.5 * k1_y)

        k3_y = h * (dy + 0.5 * k2_dy)
        k3_dy = h * (x + 0.5 * h) * (y + 0.5 * k2_y)

        k4_y = h * (dy + k3_dy)
        k4_dy = h * (x + h) * (y + k3_y)

        y_new = y + (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y) / 6.0
        dy_new = dy + (k1_dy + 2.0 * k2_dy + 2.0 * k3_dy + k4_dy) / 6.0

        return y_new, dy_new

    def integrate_from_zero(targets):
        y = AI0
        dy = AIP0
        x_current = 0.0
        values = []

        for x_target in targets:
            dx = x_target - x_current
            n_steps = max(1, int(abs(dx) / 0.002))
            h = dx / n_steps

            for _ in range(n_steps):
                y, dy = rk4_step(x_current, y, dy, h)
                x_current += h

            values.append(y)

        return np.array(values)

    if len(x_positive) > 0:
        positive_values = integrate_from_zero(x_positive)
    else:
        positive_values = np.array([])

    if len(x_negative) > 0:
        negative_values = integrate_from_zero(x_negative[::-1])[::-1]
    else:
        negative_values = np.array([])

    values_sorted = np.empty_like(x_sorted)
    values_sorted[x_sorted < 0.0] = negative_values
    values_sorted[x_sorted >= 0.0] = positive_values

    ai_values[order] = values_sorted

    return ai_values


def action_allowed(q_values, n_steps=400):
    """
    Calcula la acción clásica en la región permitida:
    J(q) = integral desde q hasta q0 de k(t) dt.
    """
    q_values = np.asarray(q_values, dtype=float)
    action = np.empty_like(q_values)

    for i, q_value in enumerate(q_values):
        grid = np.linspace(q_value, TURNING_POINT, n_steps)
        action[i] = np.trapezoid(k_allowed(grid), grid)

    return action


def action_forbidden(q_values, n_steps=400):
    """
    Calcula la acción bajo la barrera:
    K(q) = integral desde q0 hasta q de kappa(t) dt.
    """
    q_values = np.asarray(q_values, dtype=float)
    action = np.empty_like(q_values)

    for i, q_value in enumerate(q_values):
        grid = np.linspace(TURNING_POINT, q_value, n_steps)
        action[i] = np.trapezoid(kappa_forbidden(grid), grid)

    return action


# Malla y regiones separadas del punto de giro
q = np.linspace(Q_MIN, Q_MAX, N_POINTS)

eps = 0.008
allowed_mask = q < TURNING_POINT - eps
forbidden_mask = q > TURNING_POINT + eps

q_allowed = q[allowed_mask]
q_forbidden = q[forbidden_mask]

k_values = k_allowed(q_allowed)
kappa_values = kappa_forbidden(q_forbidden)

J_values = action_allowed(q_allowed)
K_values = action_forbidden(q_forbidden)


# Ramas WBK estándar a ambos lados del punto de giro
normalization = 0.70

allowed_prefactor = normalization / np.sqrt(np.pi)
forbidden_prefactor = normalization / (2.0 * np.sqrt(np.pi))

psi_allowed_wbk = (
    allowed_prefactor
    * np.sin(J_values + np.pi / 4.0)
    / np.sqrt(k_values)
)

psi_forbidden_wbk = (
    forbidden_prefactor
    * np.exp(-K_values)
    / np.sqrt(kappa_values)
)

psi_wbk = np.full_like(q, np.nan, dtype=float)
psi_wbk[allowed_mask] = psi_allowed_wbk
psi_wbk[forbidden_mask] = psi_forbidden_wbk

envelope_positive = allowed_prefactor / np.sqrt(k_values)
envelope_negative = -allowed_prefactor / np.sqrt(k_values)


# Variable uniforme xi(q)
left_mask = q < TURNING_POINT
right_mask = q > TURNING_POINT
center_mask = np.isclose(q, TURNING_POINT, atol=1e-14)

xi = np.empty_like(q)

if np.any(left_mask):
    J_left = action_allowed(q[left_mask])
    xi[left_mask] = -((3.0 / 2.0) * J_left) ** (2.0 / 3.0)

if np.any(right_mask):
    K_right = action_forbidden(q[right_mask])
    xi[right_mask] = ((3.0 / 2.0) * K_right) ** (2.0 / 3.0)

if np.any(center_mask):
    xi[center_mask] = 0.0


# Derivada xi'(q) obtenida de las relaciones del cambio uniforme
xi_prime = np.empty_like(q)

if np.any(left_mask):
    xi_prime[left_mask] = k_allowed(q[left_mask]) / np.sqrt(-xi[left_mask])

if np.any(right_mask):
    xi_prime[right_mask] = kappa_forbidden(q[right_mask]) / np.sqrt(xi[right_mask])

if np.any(center_mask):
    center_indices = np.where(center_mask)[0]

    for index in center_indices:
        if 0 < index < len(q) - 1:
            xi_prime[index] = 0.5 * (xi_prime[index - 1] + xi_prime[index + 1])
        elif index == 0:
            xi_prime[index] = xi_prime[index + 1]
        else:
            xi_prime[index] = xi_prime[index - 1]


# Aproximación uniforme construida con la función de Airy
psi_global = normalization * airy_ai(xi) / np.sqrt(xi_prime)


# Magnitudes auxiliares para la representación gráfica
airy_halfwidth = 0.55

potential_scale = 0.18
potential_plot = potential_scale * potential(q)
energy_plot = np.zeros_like(q)


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")


def draw_variant(
    ax,
    show_potential=True,
    show_energy=True,
    show_turning_point=True,
    show_turning_line=True,
    show_wbk=False,
    show_envelope=False,
    show_airy_region=False,
    show_global=False,
    show_region_labels=True,
    title="",
    legend=True,
):
    """
    Dibuja distintas combinaciones de ramas WBK, envolventes y aproximación uniforme.
    """
    handles = []

    ax.axvspan(-8.0, TURNING_POINT, color="green", alpha=0.08, zorder=0)
    ax.axvspan(TURNING_POINT, 4.0, color="red", alpha=0.08, zorder=0)

    if show_potential:
        line_potential, = ax.plot(
            q,
            potential_plot,
            linewidth=3.0,
            color="tab:blue",
            label=r"$V(q)-E$",
        )
        handles.append(line_potential)

    if show_energy:
        ax.plot(q, energy_plot, linewidth=2.0, color="black")

    if show_turning_point:
        ax.plot(
            [TURNING_POINT],
            [potential_scale * potential(TURNING_POINT)],
            marker="o",
            markersize=10,
            color="tab:blue",
            linestyle="None",
        )

    if show_turning_line:
        ax.axvline(TURNING_POINT, linewidth=2.0, color="black")

    if show_envelope:
        line_envelope, = ax.plot(
            q_allowed,
            envelope_positive,
            linestyle=":",
            linewidth=1.8,
            color="black",
            label=r"Envelopes",
        )
        ax.plot(
            q_allowed,
            envelope_negative,
            linestyle=":",
            linewidth=1.8,
            color="black",
        )
        handles.append(line_envelope)

    if show_wbk:
        line_wbk, = ax.plot(
            q,
            psi_wbk,
            color="red",
            linewidth=2.0,
            alpha=0.9,
            label=r"WBK wavefunction",
        )
        handles.append(line_wbk)

    if show_airy_region:
        airy_patch = ax.axvspan(
            TURNING_POINT - airy_halfwidth,
            TURNING_POINT + airy_halfwidth,
            color="yellow",
            alpha=0.2,
            label=r"Airy region",
        )
        handles.append(airy_patch)

    if show_global:
        line_global, = ax.plot(
            q,
            psi_global,
            color="green",
            linewidth=3.0,
            label=r"Global wavefunction",
        )
        handles.append(line_global)

    if show_region_labels:
        ax.text(-6.0, -0.4, "Allowed region", fontsize=20)
        ax.text(1.0, -0.4, "Forbidden region", fontsize=20)

    ax.set_xlim(-8.0, 4.0)
    ax.set_ylim(-0.6, 0.8)

    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$\psi(q)$")
    ax.set_title(title)

    ax.grid(alpha=0.25)

    if legend and len(handles) > 0:
        labels = [handle.get_label() for handle in handles]

        ax.legend(
            handles,
            labels,
            loc="upper left",
            ncol=1,
            frameon=True,
            fontsize=15,
            columnspacing=1.0,
            handlelength=2.0,
            handletextpad=0.5,
            borderpad=0.8,
            labelspacing=0.9,
        )


# Figuras exportadas
fig, ax = plt.subplots(figsize=(12.0, 7.0))
draw_variant(
    ax,
    show_wbk=True,
    show_envelope=True,
    show_airy_region=True,
    show_global=True,
    title="Global wavefunction",
)
save_figure(fig, "fig_complete_global_method.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(12.0, 7.0))
draw_variant(
    ax,
    show_wbk=True,
    title="WBK wavefunction",
)
save_figure(fig, "fig_wbk.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(12.0, 7.0))
draw_variant(
    ax,
    show_wbk=True,
    show_envelope=True,
    title="WBK wavefunction",
)
save_figure(fig, "fig_envelopes.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(12.0, 7.0))
draw_variant(
    ax,
    show_airy_region=True,
    show_global=True,
    title="Global wavefunction",
)
save_figure(fig, "fig_global_method.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(12.0, 7.0))
draw_variant(
    ax,
    show_potential=False,
    show_turning_point=False,
    show_turning_line=False,
    show_wbk=True,
    show_airy_region=True,
    show_global=True,
    show_region_labels=False,
    title="Comparison",
)
save_figure(fig, "fig_branch_comparison.png")
plt.close(fig)


# Vista previa
fig, ax = plt.subplots(figsize=(12.0, 7.0))
draw_variant(
    ax,
    show_wbk=True,
    show_envelope=True,
    show_airy_region=True,
    show_global=True,
    title="Global wavefunction compared with standard WBK",
)
plt.tight_layout()
plt.show()
plt.close(fig)