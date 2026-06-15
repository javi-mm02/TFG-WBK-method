# TFG-WBK-method

Python scripts used to generate the numerical results and figures discussed in the thesis.

## Repository structure

- `scripts/`: source Python scripts.
- `figures/`: output directory for generated figures.
- `results/`: output directory for generated numerical data and tables.

## Requirements

Python 3 is required. Dependencies are listed in `requirements.txt`.

Installation:

    pip install -r requirements.txt

## Execution

Scripts are executed from the repository root. Quotation marks are used because some filenames contain spaces.

    python "scripts/Alpha decay.py"
    python "scripts/Anharmonic oscillator.py"
    python "scripts/Double Well.py"
    python "scripts/Harmonic Oscillator.py"
    python "scripts/Quasi-stationary states.py"
    python "scripts/Schematic Wavefunction.py"
    python "scripts/WBK-Airy Figures.py"

Generated figures are written to `figures/`. Numerical outputs are written to `results/`.

## Contents

The scripts cover the numerical and graphical material associated with the semiclassical WBK treatment used in the thesis: turning-point structure, Airy-type approximations, bound-state examples, tunnelling in a symmetric double well, and alpha-decay barrier penetration.
