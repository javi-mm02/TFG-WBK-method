# TFG-WBK-method

Python scripts associated with the numerical calculations, validity checks and graphical material used in the thesis.

## Contents

* `scripts/`: source Python scripts.
* `requirements.txt`: Python package dependencies.

The figures and numerical results are generated when the scripts are executed. The scripts automatically create the `figures/` and `results/` directories in the root folder of the local copy of the repository.

## Requirements

Python 3 is required. Dependencies are listed in `requirements.txt`.

Installation:

```
pip install -r requirements.txt
```

## Execution

The scripts should be kept inside the `scripts/` directory. They are executed from the repository root. Quotation marks are used because some filenames contain spaces.

Main numerical and graphical scripts:

```
python "scripts/Alpha decay.py"
python "scripts/Anharmonic oscillator.py"
python "scripts/Double Well.py"
python "scripts/Harmonic Oscillator.py"
python "scripts/Quasi-stationary states.py"
python "scripts/Schematic Wavefunction.py"
python "scripts/WBK-Airy Figures.py"
python "scripts/validity_exact_cases.py"
```

The generated files are stored in:

```
figures/
results/
```

These paths are relative to the local repository. Therefore, the absolute location depends on the computer where the repository is downloaded, but the internal structure is the same.

## Scope

The scripts cover the numerical and graphical material associated with the semiclassical WBK treatment used in the thesis: turning-point structure, Airy-type approximations, bound-state examples, tunnelling in a symmetric double well, and alpha-decay barrier penetration.

The validity-check scripts evaluate the local WBK validity criterion for the benchmark systems and numerical examples discussed in the thesis. They are intended to support the interpretation of the semiclassical results and the reproducibility of the numerical analysis.

The alpha-decay scripts perform the most expensive calculations because they include turning-point detection, numerical quadrature, recalibration of the effective potential and sensitivity checks. Their execution may therefore take longer than the other scripts.
