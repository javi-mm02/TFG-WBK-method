from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.linewidth": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Parámetros
mass = 1.0
omega = 1.0
energy = 4.0


# Dominio
q_min = -4.5
q_max = 4.5
q = np.linspace(q_min, q_max, 1000)


# Potencial
def potential(q_values):
    return 0.5 * mass * omega**2 * q_values**2


# Puntos de giro
q_turn = np.sqrt(2.0 * energy / (mass * omega**2))
q1 = -q_turn
q2 = q_turn

potential_values = potential(q)
allowed_mask = (q >= q1) & (q <= q2)


# Figura
fig, ax = plt.subplots(figsize=(8.0, 5.0))

ax.plot(
    q,
    potential_values,
    linewidth=2.0,
    label=r"$V(q)=\frac{1}{2}m\omega^2 q^2$",
)

ax.axhline(
    energy,
    linestyle="--",
    linewidth=1.5,
    label=r"$E$",
)

ax.vlines(
    [q1, q2],
    ymin=-2.0,
    ymax=energy,
    linestyles=":",
    linewidth=1.2,
)

ax.scatter(
    [q1, q2],
    [energy, energy],
    s=25,
    color="orange",
    zorder=5,
)

ax.fill_between(
    q[allowed_mask],
    potential_values[allowed_mask],
    energy,
    alpha=0.25,
)

ax.text(q1, energy + 0.2, r"$q_1$", ha="center", va="bottom", fontsize=15)
ax.text(q2, energy + 0.2, r"$q_2$", ha="center", va="bottom", fontsize=15)

ax.text(0.0, -1.0, "Classically allowed", ha="center", fontsize=14)
ax.text(q_min + 0.8, -1.0, "Forbidden", ha="center", fontsize=14)
ax.text(q_max - 0.8, -1.0, "Forbidden", ha="center", fontsize=14)

ax.set_xlim(q_min, q_max)
ax.set_ylim(-2.0, max(potential_values.max(), energy) + 1.0)

ax.set_xlabel(r"$q$")
ax.set_ylabel(r"$V(q)$")
ax.set_title("One-dimensional harmonic oscillator")

ax.legend(loc="upper center")
ax.grid(True, alpha=0.3)

fig.tight_layout()

fig.savefig(OUTPUT_DIR / "harmonic_oscillator_regions.pdf")
fig.savefig(OUTPUT_DIR / "harmonic_oscillator_regions.png", dpi=300)

plt.show()
