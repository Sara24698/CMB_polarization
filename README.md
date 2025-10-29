# 🌀 CENN: CMB Extraction Neural Network

**CENN (CMB Extraction Neural Network)** is a deep learning framework designed to generate and clean **polarization maps of the Cosmic Microwave Background (CMB)**.  
It builds synthetic sky maps from theoretical power spectra and removes astrophysical contaminants through a fully connected neural network (FCNN) trained on multi-frequency data.

Through this software, the authors demonstrate that a neural‑network‑based approach (similar in concept to our CENN) is capable of **recovering Q/U polarization sky maps** of the CMB from realistic multi‑frequency simulations (including foregrounds and noise). They show that:
- The mean absolute error for the E‑mode and B‑mode power spectra remains small (on the order of µK²) in ideal and realistic (Planck‑level noise)
- The method can generalize to real observational data (i.e., the processed maps agree with established component‐separation outputs like those from the Planck mission) within ~5 % at intermediate and small angular scales.
- Although promising, the approach still faces limitations at large angular scales and in low signal‐to‐noise regimes; hence, further improvements (e.g., larger patches, more realistic noise modelling) are required for high‑precision cosmology.

Our pipeline — generating maps with CAMB, adding contaminants, training CENN, and estimating spectra with NaMaster — is grounded on state‑of‑the‑art methodology and aligns with recent advances in CMB polarization analysis.

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

## ⚙️ Requirements

The following Python packages are required to run the code:

- `tensorflow`
- `numpy`
- `matplotlib`
- `healpy`
- `pandas`
---

## 📁 Repository Structure (to be expanded)

### 🗺️ `Create_maps.py`

This script generates **synthetic full-sky CMB maps** in temperature and polarization (**T, Q, U** or **E, B**) from theoretical power spectra (**Cℓs**) computed with [CAMB](https://camb.info/).

#### 🔧 What it does
- Reads CAMB-generated Cℓ files (`Cl_files/r_<r>_prueba.csv`).
- Synthesizes full-sky maps with **HEALPix** at a chosen frequency (100, 143, or 217 GHz).  
- Optionally converts **Q/U maps** into **E/B maps** using spherical harmonic decomposition.
- Saves all results as **FITS files** (e.g. `Mapa_E_143_GHz_0.1.fits`).

#### ▶️ How to run
```bash
python Create_maps.py
```

### 🧩 `simulate4CMB_pol.py`

This script generates realistic **polarization sky patches** (Q/U or E/B) combining cosmological, astrophysical, and instrumental components.  
It is used to create the **training and validation datasets** for the neural network (CENN).

---

#### 🔧 Functionality

- Creates random sky positions outside a Galactic mask.  
- Generates simulated patches including:
  - **CMB** signal from precomputed HEALPix maps.  
  - **Foregrounds**: synchrotron, thermal dust, radio and infrared point sources (IR, IRLT, etc.) using `corrsky` models.  
  - **Instrumental white noise** (optional).  
- Simulates three observing frequencies (**low, central, high**) and scales components using spectral indices.  
- Applies beam smoothing to match the desired resolution.  
- Combines all components into final total **Q/U** (or **E/B**) maps.  
- Stores outputs in `.h5` format for machine learning or spectral analysis.

---

#### ▶️ How to Run

From Python:
```bash
make_parallel_simu4CMB('./sim_cfreq143.par', './Validation', noise=False, QU=False)
```

#### 📥 Inputs

- **Parameter file (`.par`)** — JSON-like configuration file containing:
  - `frequency_low`, `frequency_central`, `frequency_high`
  - `beam_fwhm` — beam full-width at half-maximum (arcmin)
  - `pixel_size` — patch resolution
  - `n_patches` — number of simulated patches
  - `contaminants` — list of components to include (e.g. `CMB`, `dust`, `sync`, `radio`, `IR`)
  - Paths to input **HEALPix maps** for CMB, dust, and synchrotron templates  

  Example: `sim_cfreq143.par`

- **Input maps** — FITS files of CMB, dust, and synchrotron components specified in the parameter file.

---

#### 📤 Outputs

- `.h5` files containing simulated sky patches and corresponding CMB-only labels:
  - `Validation_E.h5`, `Validation_B.h5` (if `QU=False`)  
  - `Validation_Q.h5`, `Validation_U.h5` (if `QU=True`)

Each file contains:
- **`M`** → array of total simulated patches (low, central, high frequencies)  
- **`M0`** → array of CMB-only patches (ground truth labels)

---

#### 💡 Notes

- Simulations are **parallelized** using Python’s `multiprocessing` for efficient patch generation.  
- **Noise levels**, **beam properties**, and **number of patches** can be adjusted through the `.par` configuration file.  
- Relies on helper routines:  
  - `corrsky_TQU_hp` — generates correlated foreground maps  
  - `cutting_patches_with_ForSE_code` — extracts sky patches from full-sky maps  
- Outputs are used directly for **CENN network training** and **validation**.

---

## 🧠 CENN Neural Network

This repository implements the **Cosmic Microwave Background Extraction Neural Network (CENN)**,  
based on Casas et al. (2022) — [arXiv:2205.05623](https://arxiv.org/abs/2205.05623).

---

### 🚀 Main Script: `Parametros.py`

This script controls the full **CENN workflow** — training, execution, and plotting — through a single call.

---

#### ⚙️ Description

`Parametros.py` defines hyperparameters and file paths, then runs the complete process:

1. **Training** — trains the CNN with CMB + foreground patches.  
2. **Execution** — applies the trained model to validation maps.  
3. **Plotting** — generates visual and statistical comparisons.

---

#### ▶️ How to Run

```bash
python Parametros.py
```

#### 📥 Inputs

- **.h5 training datasets** (generated by `Create_maps.py`):
  - `Train_<Filtro>.h5`
  - `Test_<Filtro>.h5`
  - `Validation_<Filtro>.h5`

- **Parameters defined in the script:**
  - `learning_rate`, `batch_size`, `num_epochs`, etc.
  - `Patch_Size = 256`
  - `Filtro = 'E'` or `'B'` (polarization component)
  - `Num_imagenes` — number of validation examples for plotting

---

#### 📤 Outputs

- **Trained model files:**
  - `Models_<Filtro>/*.h5`

- **Training plots:**
  - `training_history.png`

- **Validation predictions and plots** (from `execute()` and `plots()` functions)

### 🧠 CENN Neural Network Properties

The Cosmic Microwave Background Extraction Neural Network (CENN) is a convolutional neural network (CNN) designed to extract the CMB signal from multi-frequency sky patches. Below are its main properties:

#### Architecture
- **Input:** 256×256 pixel patches with 3 frequency channels.
- **Convolutional Layers:**
  - Conv1: 8 filters, 9×9 kernel, stride 2, LeakyReLU activation
  - Conv2: 16 filters, 9×9 kernel, stride 2, LeakyReLU activation
  - Conv3: 64 filters, 7×7 kernel, stride 2, LeakyReLU activation
  - Conv4: 128 filters, 7×7 kernel, stride 2, LeakyReLU activation
- **Deconvolutional (Upsampling) Layers:**
  - Deconv3 → Deconv6 with LeakyReLU activation, incorporating skip connections from convolutional layers for residual learning.
- **Output:** Single-channel reconstructed CMB patch (256×256).

#### Hyperparameters
- Learning rate: 0.005 (adjustable)
- Optimizer: Adagrad
- Loss: Mean Squared Error (MSE)
- Regularization: L2 (`1e-6`)
- Activation function: LeakyReLU (α=0.2)
- Batch size: 32
- Epochs: 500 (can be changed)
- Early stopping applied with patience = 20 epochs

#### Training Features
- Uses skip connections (concatenation of convolutional and deconvolutional layers) to preserve spatial information.
- Training includes checkpoints to save best models based on validation loss.
- History of training and validation losses is plotted for performance monitoring.

#### Implementation
- Built in TensorFlow/Keras.
- Parallelized training using GPU (configurable via `CENN_Execute.py <GPU>`).
- Inputs/outputs handled via `.h5` datasets.
- Flexible for polarization components: `'E'` or `'B'`.

#### Evaluation
- Model predictions are compared to ground truth CMB patches using MSE.
- Output can be directly used for plotting and validation.

## NaMaster Analysis of E/B Power Spectra

The estimation of the E- and B-mode power spectra from CMB maps is performed using **NaMaster** (`pymaster`), a Python library designed for pseudo-$C_\ell$ estimation in flat-sky and full-sky maps, including proper handling of masks and mode-coupling corrections. In this work, NaMaster is applied to both **simulated input maps** and **CENN-reconstructed output maps**.

### Purpose

The main goals of the NaMaster analysis are:

1. Compute the E- and B-mode power spectra from patches of the CMB polarization maps (Q and U).
2. Correct for the effects of partial sky coverage via mask apodization and decoupling.
3. Compare input (true) CMB spectra with the reconstructed output from the CENN network and quantify residuals.

### Inputs

The main inputs for the NaMaster analysis are:

- **Q and U maps**:
  - Input simulations: `data_Q[i]["sim"]` and `data_U[i]["sim"]`
  - CENN outputs: `data_Q[i]["net"]` and `data_U[i]["net"]`
- **Patch size**: 240×240 pixels after cutting 8 pixels from each edge (from original 256×256 patches).
- **Masking parameters**:
  - Flat-sky masks with apodization (apodization factor varies per resolution or patch type).
  - Holes in the mask for specific dust simulations.
- **Binning**: multipole bins defined via `NmtBinFlat`, e.g., `bins = np.arange(0, 2525, 35)`.

### Processing Steps

1. **Mask Creation**: Define a mask for each patch, apply apodization, and flatten for NaMaster.
2. **NaMaster Fields**: Create `NmtFieldFlat` objects for input, output, and residual maps, enabling pure-B mode estimation.
3. **Coupling Matrix**: Compute the mode-coupling matrix (`NmtWorkspaceFlat`) for each field.
4. **Pseudo-$C_\ell$ Computation**: Calculate coupled power spectra using `compute_coupled_cell_flat`.
5. **Decoupling**: Remove mode-coupling effects using `decouple_cell`.
6. **Averaging**: Combine spectra from all patches to obtain mean and standard deviation for EE, BB, and residual spectra.

### Outputs

The analysis produces:

- **Mean E-mode power spectrum** (`mean_ee`)
- **Mean B-mode power spectrum** (`mean_bb`)
- **Standard deviations** (`std_ee`, `std_bb`) representing uncertainties across patches.
- **Multipole values** (`ells_uncoupled`) corresponding to the binned spectra.

These results are later used for plotting and validation of the CENN network's reconstruction performance, comparing input, output, and residual power spectra.

---

## 📚 References

- Casas, J. M., Bonavera, L., González‑Nuevo, J., Puglisi, G., Baccigalupi, C., Cabo, S. R., Cueli, M. M., Crespo, D., González‑Gutiérrez, C., & de Cos, F. J. (2025). *Recovering CMB polarization maps with neural networks: Performance in realistic simulations*. *Journal of Cosmology and Astroparticle Physics*, 2025(10), 063. DOI: 10.1088/1475‑7516/2025/10/063.













