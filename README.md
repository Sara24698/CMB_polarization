# 🌀 CENN: CMB Extraction Neural Network

**CENN (CMB Extraction Neural Network)** is a deep learning framework designed to generate and clean **polarization maps of the Cosmic Microwave Background (CMB)**.  
It builds synthetic sky maps from theoretical power spectra and removes astrophysical contaminants through a fully connected neural network (FCNN) trained on multi-frequency data.

---

## 🌌 Overview

This software generates simulated maps of the CMB temperature and polarization fields — **T, Q, U**, or their derived **E and B modes** — from theoretical angular power spectra (**Cℓs**) computed with [**CAMB**](https://camb.info/) (Lewis et al., 2000).

The workflow is composed of several key stages:

1. **CMB Map Generation**  
   Theoretical CMB power spectra obtained from CAMB are used to create full-sky maps in the HEALPix format for T, Q, and U components.

2. **Patch Extraction and Contamination**  
   The CMB maps are divided into smaller **sky patches**, to which various **foreground contaminants** are added:
   - Galactic **dust emission**  
   - **Synchrotron radiation**  
   - **Extragalactic point sources**  
   - **Instrumental white noise**

   Foreground templates are derived from **Planck Legacy Archive (PLA)** data, while white noise is generated synthetically.

3. **Multi-Frequency Data Preparation**  
   The contaminated patches are produced for several observing frequencies — **100, 143, and 217 GHz** — corresponding to the frequency range where instrumental noise is minimized.

4. **Neural Network Training (CENN)**  
   The fully connected neural network (**CENN**) is trained to **recover clean CMB patches at 143 GHz** from the contaminated multi-frequency inputs.  
   The network learns to disentangle CMB emission from astrophysical foregrounds and noise.

5. **Power Spectrum Estimation with NaMaster**  
   Once the clean patches are reconstructed, the **E- and B-mode angular power spectra** are computed using [**NaMaster**](https://namaster.readthedocs.io/) (Alonso et al., 2019).  
   These recovered spectra are then compared against:
   - Theoretical input spectra (labels from CAMB)
   - Residuals (`Output - Label`)
   - Analytical theoretical models

---

## 🧠 Purpose

The goal of **CENN** is to provide a realistic framework for **CMB component separation using neural networks**, allowing controlled tests of performance on synthetic yet physically motivated datasets.  
It bridges cosmological simulations, foreground modeling, and modern deep learning approaches for precision CMB analysis.

---

## 📁 Repository Structure (to be expanded)

