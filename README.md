# WBK method: numerical scripts

This repository contains the Python scripts used to generate the numerical results and figures discussed in the thesis.

## Repository structure

* `scripts/`: Python scripts used for the numerical calculations and graphical representations.
* `figures/`: output directory used by the scripts for generated figures.
* `results/`: output directory used by the scripts for generated numerical tables and data files.

## Requirements

The scripts require Python 3 and the packages listed in `requirements.txt`.

The dependencies can be installed with:

```bash
pip install -r requirements.txt
```

## Execution

From the root directory of the repository, each script can be run with:

```bash
python "scripts/Alpha decay.py"
python "scripts/Anharmonic oscillator.py"
python "scripts/Double Well.py"
python "scripts/Harmonic Oscillator.py"
python "scripts/Quasi-stationary states.py"
python "scripts/Schematic Wavefunction.py"
python "scripts/WBK-Airy Figures.py"
```

The quotation marks are required because some filenames contain spaces.

The scripts save generated figures in `figures/` and numerical outputs in `results/`.

## Contents

The repository includes scripts for the one-dimensional WBK construction, turning-point analysis, bound-state examples, tunnelling in a symmetric double well, and alpha-decay barrier penetration.
