from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# Configuración gráfica
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 15,
    "axes.labelsize": 17,
    "axes.titlesize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 17,
    "axes.linewidth": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Puntos de giro y energía de referencia
q1 = -3.0
q2 = 0.0
q3 = 0.5 * q2 - q1

energy = 0.0


# Parámetros del potencial esquemático por tramos
amplitude = 3.0
left_steepness = 3.0
right_asymptote = -1.5


# Dominio de representación
q_min = -5.0
q_max = 5.5
q = np.linspace(q_min, q_max, 2500)


# Magnitudes derivadas que fijan la forma de cada tramo
omega = np.pi / (q2 - q1)
left_curvature = amplitude * np.pi / (2.0 * (q2 - q1) ** 2)
right_width = -right_asymptote / (amplitude * omega)


# Construcción del potencial a trozos
potential = np.empty_like(q)

left_mask = q <= q1
middle_mask = (q > q1) & (q <= q3)
right_mask = q > q3

potential[left_mask] = (
    left_curvature * ((q[left_mask] - q2) ** 2 - (q1 - q2) ** 2)
    + left_steepness * (q1 - q[left_mask]) ** 4
)

potential[middle_mask] = (
    -amplitude * np.sin(omega * (q[middle_mask] - q1))
)

potential[right_mask] = (
    right_asymptote
    * (1.0 - np.exp(-(q[right_mask] - q3) / right_width))
)


# Figura
fig, ax = plt.subplots(figsize=(10.0, 6.0))

ax.plot(q, potential, linewidth=2.2, label=r"$V(q)$")
ax.axhline(energy, linestyle="--", linewidth=2.0, label=r"$E$")


# Sombreado de regiones permitidas y prohibidas
ax.fill_between(
    q,
    potential,
    energy,
    where=q <= q1,
    color="tab:red",
    alpha=0.18,
)

ax.fill_between(
    q,
    potential,
    energy,
    where=(q >= q1) & (q <= q2),
    color="tab:green",
    alpha=0.18,
)

ax.fill_between(
    q,
    potential,
    energy,
    where=(q >= q2) & (q <= q3),
    color="tab:red",
    alpha=0.18,
)

ax.fill_between(
    q,
    potential,
    energy,
    where=q >= q3,
    color="tab:green",
    alpha=0.18,
)


# Marcado de los puntos de giro
for turning_point, label in zip(
    [q1, q2, q3],
    [r"$q_1$", r"$q_2$", r"$q_3$"],
):
    ax.axvline(turning_point, linestyle=":", linewidth=1.4)
    ax.text(
        turning_point,
        -3.75,
        label,
        ha="center",
        va="bottom",
        fontsize=18,
    )


# Etiquetas de las regiones
ax.text(-2.75, 2.0, "Classically allowed", fontsize=15)
ax.text(0.25, -2.0, "Classically forbidden", fontsize=15)
ax.text(3.1, 2.0, "Classically allowed", fontsize=15)


# Ajustes finales de la figura
ax.set_xlim(q_min, q_max)
ax.set_ylim(-4.0, 5.0)

ax.set_xlabel(r"$q$")
ax.set_ylabel(r"$V(q)$")
ax.set_title("Schematic potential with three turning points")

ax.legend(frameon=True)
ax.grid(True, alpha=0.28)

ax.xaxis.set_major_locator(MultipleLocator(1.0))
ax.yaxis.set_major_locator(MultipleLocator(1.0))

fig.tight_layout()

fig.savefig(OUTPUT_DIR / "three_turning_points_potential.pdf")
fig.savefig(OUTPUT_DIR / "three_turning_points_potential.png", dpi=300)

plt.show()