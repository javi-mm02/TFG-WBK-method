# TFG-WBK-method

Python scripts used to generate the numerical results and figures discussed in the thesis.

## Repository structure

- `scripts/`: source Python scripts.
- `figures/`: directory used by the scripts to save generated figures.
- `results/`: directory used by the scripts to save numerical outputs and tables.

## Requirements

The scripts require Python 3. The required packages are listed in `requirements.txt`.

To install them, run:

    pip install -r requirements.txt

## Execution

The scripts can be executed from the root directory of the repository. Since some filenames contain spaces, quotation marks should be used.

    python "scripts/Alpha decay.py"
    python "scripts/Anharmonic oscillator.py"
    python "scripts/Double Well.py"
    python "scripts/Harmonic Oscillator.py"
    python "scripts/Quasi-stationary states.py"
    python "scripts/Schematic Wavefunction.py"
    python "scripts/WBK-Airy Figures.py"

Generated figures are saved in `figures/`. Numerical outputs are saved in `results/`.

## Contents

The repository contains scripts related to the semiclassical WBK method, including turning-point analysis, Airy-type uniform approximations, bound-state examples, double-well tunnelling, and alpha-decay barrier penetration.
