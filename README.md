Here is a clean, professional English version of your documentation, rewritten concisely while keeping it technically sharp and structured.

---

# 📋 Overview

An interactive framework for analyzing qubit errors in quantum processors using the **Category-Based Error Budgeting (CBEB)** methodology. The program provides a comprehensive graphical interface for processing qubit calibration data, classifying it under multiple criteria, and computing key quantitative metrics.

---

# ✨ Key Features

| Feature                             | Description                                                                  |
| ----------------------------------- | ---------------------------------------------------------------------------- |
| 📋 Data Collection & Classification | Analyzes single- and two-qubit data with three classification schemes        |
| 📊 Category-Based Budget Analysis   | Computes ( E_A ), ( R_A ), ( D_A ) and identifies dominant error categories  |
| 🔄 Correlation Analysis             | Builds covariance matrices from two-qubit gate errors                        |
| 🎯 Decoder Integration              | Computes weights for three decoding models and estimates logical error rates |
| 📈 Visual Dashboard                 | 6 interactive plots for result visualization                                 |
| 💾 Export Results                   | Saves all outputs as PNG figures and TXT summary reports                     |

---

# 🚀 Quick Start

## Requirements

* Python 3.7+
* pip

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/QubitErrorAnalyzer.git
cd QubitErrorAnalyzer
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn openpyxl
```

> Note: `tkinter` comes preinstalled with Python.

Prepare your calibration file:

* Place `IBM-data.xlsx` in the project directory.

Run:

```bash
python qubit_analyzer.py
```

---

# 📁 Project Structure

```
QubitErrorAnalyzer/
│
├── qubit_analyzer.py
├── IBM-data.xlsx
├── README.md
│
└── Results/
    └── analysis_results_timestamp/
```

---

# 📊 Input Data Structure (Excel)

### Single-Qubit Data

Required columns:

* Qubit
* T1 (us)
* T2 (us)
* Readout assignment error
* Prob meas0 prep1
* Prob meas1 prep0
* ID error
* RX error
* √x (sx) error
* Pauli-X error
* MEASURE error

### Two-Qubit Data

* CZ error ("0_1: 0.000954")
* RZZ error ("0_1: 0.000865")
* Gate length (ns) ("0_1: 68")

---

# 🖱️ Usage

Main window includes:

* 📊 Data summary
* 🔘 8 interactive analysis buttons

### Main Actions

| Button            | Function                                   |
| ----------------- | ------------------------------------------ |
| Step 7.1          | Calibration data & category distribution   |
| Step 7.2          | Category metrics ((R_A, D_A))              |
| Correlation-Aware | Covariance analysis                        |
| Step 7.3          | Decoder weights & logical error comparison |
| Visualization     | 6-plot dashboard                           |
| Summary Report    | Full analysis summary                      |
| Save All Results  | Export all figures & report                |
| Exit              | Close application                          |

---

# 📈 Outputs

### Saved Figures (7)

* Readout error distribution
* Spatial category metrics
* Error-rate category metrics
* Coherence category metrics
* Decoder weights comparison
* Correlation matrix
* Logical error rates comparison

### Text Summary

Includes:

* Analysis date
* Total qubits and total error budget
* Logical error rates (3 models)
* Most dominant categories with (R_A) and (D_A)

---

# 📐 Theoretical Framework

### Core Metrics

| Symbol      | Definition             | Meaning                    |                  |                           |
| ----------- | ---------------------- | -------------------------- | ---------------- | ------------------------- |
| (E_{total}) | ( \sum e_i )           | Total readout error budget |                  |                           |
| (E_A)       | ( \sum_{i \in A} e_i ) | Error budget of category A |                  |                           |
| (R_A)       | ( E_A / E_{total} )    | Relative contribution rate |                  |                           |
| (D_A)       | ( \frac{E_A/           | A                          | }{E_{total}/N} ) | Disproportionality factor |

### Interpretation of (D_A)

* (D_A > 1): Category performs worse than average
* (D_A = 1): Category matches average
* (D_A < 1): Category performs better than average

---

# 🎯 Decoding Models

* **Uniform Model**
  ( w = -\log(\text{avg_error}) )

* **Individual Model**
  ( w_i = -\log(e_i) )

* **Category-Correlation Model**
  ( w_i = -\log(e_i + \sum \rho_{ij}\Sigma_{ij}) )

---

If you want, I can now:

* Refine this into a **publication-ready README**
* Convert it into **LaTeX (Overleaf-ready) documentation**
* Rewrite it in a more academic research tone**
* Or critique the scientific positioning of your framework**
