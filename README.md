![Cover](Figures/cover.png)

# AttackGenerationViaSMARTDS.ipynb
Generation of Labeled Voltage Phasors under Physical and FDI Attacks Using SMART-DS and OpenDSS

## Overview

This notebook provides a comprehensive workflow to generate a high-fidelity dataset of voltage phasors in three-phase unbalanced distribution networks under both physical and stealthy False Data Injection (FDI) attacks. The data generation leverages:

- **[SMART-DS](https://data.openei.org/submissions/2790)**: a large-scale, realistic dataset of U.S. distribution networks with three-year historical load and solar data across multiple voltage levels.
- **[OpenDSS](https://sourceforge.net/projects/electricdss/)**: an industry-grade, open-source simulation tool for power flow analysis in distribution systems.

The end result is a time-series dataset of complex voltage phasors labeled with attack types, suitable for use in machine learning tasks such as anomaly detection, graph signal processing, and distribution network state estimation under adversarial conditions.

## Objectives

- Simulate physical control manipulation at PV inverters (e.g., tampering with Volt-Var/Volt-Watt curves).
- Inject stealthy FDI attacks on PMU voltage measurements to evade traditional bad-data detection.
- Generate labeled voltage phasors at all nodes over time.
- Store data in NumPy-compatible format for downstream use in GNN/GCN training.

## Attack Types Considered

### 1. Physical Attacks
- Direct manipulation of distributed energy resources (DERs), specifically by altering the Volt-Var and Volt-Watt control curves of PV inverters.
- Attack randomly targets phase-2 connected PV inverters during each timestep.
- Labeled attack vectors indicate which PV nodes are compromised.

### 2. False Data Injection (FDI) Attacks
- Sophisticated stealthy corruption of voltage measurements from mu-PMU sensors.
- Noise is added to sensor data.
- FDI is injected in a way that mimics realistic attacker knowledge, bypassing traditional bad-data detectors.
- MMSE-based estimation is then used to reconstruct the system's true voltage state from corrupted measurements.

## SMART-DS Dataset Structure

SMART-DS includes 3-year (2016–2018) time-series data at multiple voltage levels:

- **230 kV** – Sub-transmission level  
- **69 kV** – Substation level  
- **4–25 kV** – Feeder levels

```
SMART-DS/
├── GIS/
├── PLACEMENTS/
└── YEARS/
    └── <YEAR>/
        └── <DATASETS>/
            ├── full_dataset_analysis/
            └── <SUB-REGIONS>/
                ├── load_data/
                ├── solar_data/
                ├── cyme_profiles/
                └── scenarios/
                    └── opendss/
                        └── <SUBSTATIONS>/
                            └── <FEEDERS>/
```

## Sensor Placement Strategy

To ensure observability and data fidelity:
- The top-`k` singular vectors from the normalized admittance matrix (`Y`) are used.
- A greedy algorithm optimally places 120 mu-PMU sensors based on maximizing the smallest singular value of the sensor matrix.
- A seed set of 30 known PV-connected sensor locations are pre-selected.
- The remaining 90 sensors are strategically placed across the network for maximum coverage and robustness to stealthy FDI.

## Requirements

```bash
# Python 3.11 environment setup
pip install .
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install cplxmodule numpy==1.24.4 scikit-learn pyarrow
```

## Outputs

The following artifacts are saved:

- `Vphasor_<attack_mode>.npy`: complex voltage phasors (after MMSE estimation)
- `AttackLabel_<attack_mode>.npy`: labels indicating attack presence
- `metadata_run_config.npy`: run configuration metadata

## Example Usage

```python
from data_loader import data_loader_FDIPhy

data = data_loader_FDIPhy(
    'Vphasor_FDI_Physical_WithVoltageNoise.npy',
    'AttackLabel_FDI_Physical_WithVoltageNoise.npy'
)
X = data.data_recover   # shape: (time, nodes, samples)
Y = data.label_truth    # shape: (time, labels)
```


# FDIPhyDet_Final.ipynb

## Overview

This repository implements a deep learning pipeline for the joint detection of:

- **Physical attacks** at PV inverters (voltage control curve manipulation)
- **False Data Injection (FDI) attacks** on measurement sensors (e.g., mu-PMUs)

It integrates:

- Time-series simulation of unbalanced 3-phase distribution systems (via OpenDSS)
- Complex-valued GCN (Graph Convolutional Network) for spatiotemporal attack detection
- ROC-AUC metrics for model evaluation

---

## Features

- Uses **OpenDSS** to simulate voltage behavior of PV-rich feeders
- Greedy optimization for **sensor placement**
- **Synthetic attack injection** for physical and FDI scenarios
- Dataset construction for **GCN training and testing**
- Accurate **binary classification** of attack locations over time

---

## Workflow Summary

1. **Import 3-phase unbalanced distribution network** from OpenDSS (.dss format)
2. **Place sensors optimally** using greedy observability maximization
3. **Run time-series simulations** for 35,040 timepoints (1 year of 15-min data)
4. **Inject physical attacks** by modifying PV control parameters (VV/VW)
5. **Inject FDI attacks** on voltage measurements (with optional voltage noise)
6. **Create labeled dataset** (voltage phasors, attack labels)
7. **Train/test GCN** to classify physical and FDI attacks
8. **Evaluate via ROC curves and accuracy**

---

## File Structure

```
├── scenarios/                    # OpenDSS networks and timeseries
├── *.npy                         # Saved numpy files: voltages, labels, metadata
├── *.pth                         # Trained PyTorch models
├── *.py                          # Core scripts: training, testing, plotting
├── metadata_run_config.npy       # Metadata for indexing, sensor info, nodes
```

---

## Metadata Implementation

The file `metadata_run_config.npy` stores a dictionary for all necessary experiment info:

### Contents

```python
metadata = {
  'total_timepoints': 35040,
  'sampling_rate': 20,
  'pv_feeders': [...],         # List of PV node names
  'sensor_nodes': [...],       # List of mu-PMU sensor node names
  'Y_node_order': [...],       # Canonical node ordering used in Ybus
}
```

### Usage

This file is loaded in both the training and testing phases to ensure consistent:

- Mapping between node indices and names
- Sensor attack indexing
- Y-bus matrix ordering

```python
metadata = np.load("metadata_run_config.npy", allow_pickle=True).item()
Y_NodeOrder = metadata["Y_node_order"]
sensor_location_indices = [Y_NodeOrder.index(s) for s in metadata["sensor_nodes"]]
```

---


## Results Summary

| Metric             | Score   |
| ------------------ | ------- |
| ROC-AUC (Physical) | \~0.986 |
| ROC-AUC (FDI)      | \~0.809 |
| Combined ROC-AUC   | \~0.891 |


### ROC Curve Visualizations

<p align="center">
  <img src="Figures/Physical_and_FDI.png" alt="Combined ROC" width="30%" />
  <img src="Figures/Physical.png" alt="Physical ROC" width="30%" />
  <img src="Figures/FDI.png" alt="FDI ROC" width="30%" />
</p>

---
