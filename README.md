# KASA-ST: Multiscale Temporal and Hybrid Spatial Prediction via Kolmogorov-Arnold Spectral Anchoring

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

This repository contains the official PyTorch implementation of the paper: **"KASA-ST: Multiscale Temporal and Hybrid Spatial Prediction via Kolmogorov-Arnold Spectral Anchoring"**.

## 📖 Abstract

Non-stationarity remains a primary bottleneck in spatio-temporal traffic forecasting, where deterministic physical regularities are often obscured by high-frequency stochastic volatility. To address these issues, we introduce **KASA-ST**, a unified framework that elevates physical consistency to a primary modeling objective. 

The framework operates in three phases:
1.  **Multi-scale Temporal Backbone**: Employing patching and hierarchical downsampling to pre-stabilize representations.
2.  **Hybrid Spatial Tuning**: Using a **Physical Inertia Gate** to dynamically balance node-wise temporal momentum against neighbor diffusion.
3.  **Spectral Anchoring**: Utilizing RBF-parameterized **FastKAN** to establish a global input-to-output residual connection, preventing spectral drift.

Extensive experiments on **PeMS04, PeMS07, PeMS08, and PEMS-BAY** demonstrate that KASA-ST achieves state-of-the-art performance, notably reducing MAPE by **8.0%** on the challenging PeMS07 dataset.

---

## 🧩 Model Architecture

The overall framework of KASA-ST works in three cooperative phases to handle non-stationarity and spectral drift.

<p align="center">
  <img src="your_architecture_diagram.png" alt="KASA-ST Overall Architecture" width="800"/>
</p>
<p align="center">
  <em>Figure 1: The proposed KASA-ST framework illustrating the Multi-scale Temporal Backbone, Hybrid Spatial Tuning, and Spectral Anchoring phases.</em>
</p>

---

## 🏗️ Project Structure

The project is built upon the **BasicTS** framework.

```text
KASA-ST/
├── basicts/                  # Core Framework
│   ├── archs/arch_zoo/       # Model Architectures
│   │   └── KASA_arch_v2/     # KASA-ST Implementation
│   │       ├── KASA_arch.py             # Main Model
│   │       ├── KASA_arch_w_bspline.py   # Ablation: Standard KAN (B-Spline)
│   │       ├── KASA_arch_wo_GCN.py      # Ablation: w/o Hybrid Spatial Tuning
│   │       ├── KASA_arch_wo_spectral.py # Ablation: w/o Spectral Anchoring
│   │       └── ...
│   ├── data/                 # Data loading & transformation
│   ├── metrics/              # MAE, RMSE, MAPE
│   ├── runners/              # Training & Inference Runners
│   └── ...
├── examples/                 # Configuration & Entry Points
│   ├── KASAST_v2/            # Configs for each dataset
│   │   ├── KASAST_PEMS04.py
│   │   ├── KASAST_PEMS07.py
│   │   ├── KASAST_PEMS08.py
│   │   └── KASAST_PEMS-BAY.py
│   └── run.py                # Main execution script
├── scripts/                  # Data Preparation Scripts
│   └── data_preparation/
├── figs/                     # Figures and Visualizations
├── requirements.txt
└── README.md

```

---

## ⚡ Installation

1. **Clone the repository:**
```bash
git clone [https://github.com/anonymous/KASA-ST.git](https://github.com/anonymous/KASA-ST.git)
cd KASA-ST

```


2. **Create a virtual environment (Recommended):**
```bash
conda create -n kasast python=3.8
conda activate kasast

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```



---

## 📂 Data Preparation

We use standard traffic benchmarks: **PeMS04, PeMS07, PeMS08, and PEMS-BAY**.

1. Navigate to the data preparation script directory:
```bash
cd scripts/data_preparation

```


2. Run the generation scripts for the target datasets. For example, to prepare PeMS04 datasets:
```bash
python scripts/data_preparation/PeMS04/generate_holost_data.py

```


---

## 🚀 Usage

To train and evaluate the model, use the `examples/run.py` script along with the specific configuration file for the dataset.

### Training

To train KASA-ST on **PeMS04**:

```bash
python examples/run.py --cfg examples/KASAST_v2/KASAST_PEMS04.py

```

* **PeMS07**: `examples/KASAST_v2/KASAST_PEMS07.py`
* **PeMS08**: `examples/KASAST_v2/KASAST_PEMS08.py`
* **PEMS-BAY**: `examples/KASAST_v2/KASAST_PEMS-BAY.py`

### Ablation Studies

We provide architecture variants for ablation studies (as discussed in the paper's Component Analysis). You can switch the model architecture in the config file or point to a modified config that uses:

* `KASA_arch_wo_spectral.py` (w/o Spectral Anchor)
* `KASA_arch_wo_GCN.py` (w/o Hybrid Spatial Tuning)
* `KASA_arch_w_bspline.py` (Replace RBF-KAN with B-Spline KAN)

---

## 📊 Main Results

Comparison of KASA-ST with state-of-the-art baselines (Horizon = 12 steps).

| Dataset | Metric | DCRNN | GWNet | AGCRN | PDFormer | D2STGNN | **KASA-ST (Ours)** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **PeMS04** | MAE | 21.22 | 19.85 | 19.83 | 19.55 | 19.30 | **18.10** |
|  | RMSE | 33.44 | 32.26 | 32.26 | 31.99 | 31.46 | **29.55** |
|  | MAPE | 14.17% | 12.97% | 12.97% | 12.82% | 12.70% | **12.63%** |
| **PeMS07** | MAE | 25.22 | 22.37 | 22.37 | 21.55 | 21.42 | **19.08** |
|  | RMSE | 38.61 | 36.55 | 36.55 | 34.83 | 34.51 | **32.24** |
|  | MAPE | 11.82% | 9.12% | 9.12% | 9.39% | 9.01% | **8.00%** |

*Note: For the full comparison table including SCINet, PatchTST, and STWave, please refer to Table 1 in the paper.*

---

## 📈 Analysis & Visualization

### Spectral Drift Mitigation

KASA-ST effectively anchors long-horizon predictions to invariant frequency patterns.

*(Figure: Power Spectral Density comparison showing KASA-ST prevents high-frequency attenuation compared to MLP variants)*

### Sensitivity Analysis

Hyperparameter sensitivity results for Patch Length () and Frequency Components () can be found in `figs/sensitivity_final.pdf`.

---

## 📜 Citation

If you find this repository useful for your research, please cite our paper:

```bibtex
@inproceedings{anonymous2026kasast,
  title={KASA-ST: Multiscale Temporal and Hybrid Spatial Prediction via Kolmogorov-Arnold Spectral Anchoring},
  author={Anonymous Author(s)},
  booktitle={KDD '26: Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2026}
}

```
