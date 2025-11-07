# MMA-QENS

MMA-QENS (Minimal Model Analysis for QENS) is a Python framework for analyzing quasielastic neutron scattering (QENS) data in the time domain. This package can be used to obtain the intermediate scattering function `F(Q, t)` from QENS spectra through inverse Fourier transformation and fit it to extract dynamic parameters, the stretching exponent, relaxation time, and elastic incoherent structure factor (EISF).

This includes the following four methods, performing a key step in the analysis process:

- **`Sym_Norm()`** — Symmetrizes and normalizes the measured spectra `S(Q, ω)`.  
- **`Deconvolve()`** — Fourier transforms and deconvolves the data to obtain the sample’s `F(Q, t)`.  
- **`Fitting()`** — Fits `F(Q, t)` to extract dynamic parameters.  
- **`Resample()`** — Reconstructs the fitted `S(Q, ω)` from the fitted `F(Q, t)` for direct comparison with experimental data.

An example of this is shown in 

To install MMA-QENS use the commando, when in the repository;
```bash
pip install .
```

When using the code cite: 
