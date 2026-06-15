from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 15,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Potencial esquemático y energía de referencia
q_old = np.linspace(0.0, 8.0, 1600)

V_old = (
    0.18 * (q_old - 4.1) ** 2
    + 0.03 * (q_old - 4.1) ** 4
    + 0.8
)

energy = 2.2


# Puntos de giro
diff = V_old - energy
sign_change_indices = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]

turning_points_old = []

for index in sign_change_indices:
    q_left = q_old[index]
    q_right = q_old[index + 1]

    v_left = diff[index]
    v_right = diff[index + 1]

    q_turn = q_left - v_left * (q_right - q_left) / (v_right - v_left)
    turning_points_old.append(q_turn)

q1_old, q2_old = turning_points_old

q_center = 0.5 * (q1_old + q2_old)
x_scale = 4.0 / (q2_old - q1_old)

q = x_scale * (q_old - q_center)
potential_shifted = V_old - energy

q1 = x_scale * (q1_old - q_center)
q2 = x_scale * (q2_old - q_center)

psi_color = "C1"

amplitude_turning = 0.70
amplitude_center = 0.78
n_oscillations = 3

q_left_old = q_old[q_old <= q1_old]
q_left_region = x_scale * (q_left_old - q_center)
psi_left = amplitude_turning * np.exp((q_left_old - q1_old) / 0.55)

q_mid_old = q_old[(q_old >= q1_old) & (q_old <= q2_old)]
q_mid_region = x_scale * (q_mid_old - q_center)

s = (q_mid_old - q1_old) / (q2_old - q1_old)
envelope = amplitude_turning + (
    amplitude_center - amplitude_turning
) * np.sin(np.pi * s) ** 2

phase = 2.0 * np.pi * n_oscillations * s
psi_mid = envelope * np.cos(phase)

q_right_old = q_old[q_old >= q2_old]
q_right_region = x_scale * (q_right_old - q_center)
psi_right = amplitude_turning * np.exp(-(q_right_old - q2_old) / 0.55)


# Figura
fig, ax = plt.subplots(figsize=(8.2, 5.1))

line_potential, = ax.plot(
    q,
    potential_shifted,
    linewidth=2.0,
    label=r"$V(q)-E$",
)

line_energy = ax.axhline(
    0.0,
    linestyle="--",
    linewidth=1.4,
    label=r"$E=0$",
)

ax.plot(q_left_region, psi_left, color=psi_color, linewidth=2.0)
ax.plot(q_mid_region, psi_mid, color=psi_color, linewidth=2.0)
ax.plot(q_right_region, psi_right, color=psi_color, linewidth=2.0)

for turning_point in (q1, q2):
    ax.vlines(
        turning_point,
        -2.25,
        2.25,
        linestyles="dotted",
        linewidth=1.1,
    )

ax.plot(
    [q1, q2],
    [0.0, 0.0],
    marker="o",
    linestyle="None",
    markersize=5.5,
    markerfacecolor="black",
    markeredgewidth=0,
)

ax.annotate(
    r"$q_1$",
    xy=(q1 - 0.1, 0.0),
    xytext=(6, 6),
    textcoords="offset points",
    ha="left",
    va="bottom",
)

ax.annotate(
    r"$q_2$",
    xy=(q2, 0.0),
    xytext=(-15, 6),
    textcoords="offset points",
    ha="left",
    va="bottom",
)

ax.text(-3.0, -0.45, r"$E<V(q)$", ha="center", va="center")
ax.text(0.0, -0.95, r"$E>V(q)$", ha="center", va="center")
ax.text(3.0, -0.45, r"$E<V(q)$", ha="center", va="center")

ax.text(-3.0, -1.75, "Classically\nforbidden", ha="center", va="center")
ax.text(0.0, -1.75, "Classically allowed", ha="center", va="center")
ax.text(3.0, -1.75, "Classically\nforbidden", ha="center", va="center")

ax.set_xlim(-4.2, 4.2)
ax.set_ylim(-2.25, 2.25)

ax.set_xlabel(r"$q$")
ax.set_ylabel(r"$V(q)-E$")

ax.xaxis.set_major_locator(MultipleLocator(1.0))
ax.yaxis.set_major_locator(MultipleLocator(0.5))

wave_proxy = Line2D([0], [0], color=psi_color, linewidth=2.0)

ax.legend(
    handles=[line_potential, line_energy, wave_proxy],
    labels=[r"$V(q)-E$", r"$E=0$", r"$\psi(q)$"],
    frameon=True,
    fancybox=False,
    framealpha=1.0,
    edgecolor="black",
    facecolor="white",
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
)

ax.grid(False)

for side in ["top", "right", "bottom", "left"]:
    ax.spines[side].set_linewidth(0.8)

fig.tight_layout()

fig.savefig(OUTPUT_DIR / "turning_points_regions.pdf")
fig.savefig(OUTPUT_DIR / "turning_points_regions.png", dpi=300)

plt.show()
