# AegisNav — Supervised Anomaly Detection on NASA SMAP/MSL Telemetry

**Author**: Roman Di Domizio  
**Course**: CSCA 5622 - Supervised Learning (Fall 2025)  
**Project**: Final Project - Supervised Learning Problem

---

## 🎯 Project Overview

This project delivers the **first production ML artifact** for [AegisNav](https://github.com/romandidomizio/AegisNav), an agentic deep-space autonomy sandbox: a supervised anomaly detection model trained on expert-labeled NASA spacecraft telemetry.

**Problem**: Binary classification over time windows to predict whether telemetry contains an anomaly.  
**Data**: NASA SMAP (Soil Moisture Active Passive) and MSL (Mars Science Laboratory) labeled telemetry.  
**Methods**: ISLP Ch 1-9 supervised learning baselines (Logistic Regression, LDA/QDA, SVM, Random Forest, Gradient Boosting).  
**Metric**: PR-AUC (primary), ROC-AUC, segment-level recall.

---

## 📁 Repository Structure

```
AegisNav-Anomaly-Detection/
├── anomaly_detection_final.ipynb    # Main notebook with EDA, modeling, evaluation
├── aegisnav_detector.py             # Deployment utilities for Monitor Agent
├── models/
│   ├── anomaly_detector_pipeline.joblib  # Trained pipeline (scaler + model)
│   └── model_config.json                 # Configuration and hyperparameters
├── data/                            # Telemetry data (gitignored, download separately)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/romandidomizio/AegisNav-Anomaly-Detection.git
cd AegisNav-Anomaly-Detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Notebook

```bash
jupyter notebook anomaly_detection_final.ipynb
```

### 4. Use Trained Model

```python
from aegisnav_detector import AnomalyDetector
import pandas as pd

# Load detector
detector = AnomalyDetector(
    model_path='models/anomaly_detector_pipeline.joblib',
    config_path='models/model_config.json'
)

# Score telemetry (DataFrame with columns: channel_0, channel_1, ...)
telemetry_df = pd.read_csv('your_telemetry.csv')
results = detector.score_windows(telemetry_df)

print(f"Detected {sum(results['predictions'])} anomalies")
```

---

## 📊 Key Results

| Metric      | Validation | Test |
|-------------|-----------|------|
| **PR-AUC**  | [Value]   | [Value] |
| **ROC-AUC** | [Value]   | [Value] |
| **Recall**  | [Value]   | [Value] |
| **Precision** | [Value] | [Value] |

**Best Model**: Random Forest / Gradient Boosting (selected based on PR-AUC)

---

## 🔬 Methods Summary

### Feature Engineering
- **Window Size**: 120 timesteps
- **Stride**: 10 timesteps (90% overlap)
- **Feature Types**: Statistical, temporal, spectral (FFT), derivative
- **Total Features**: ~75 per multi-channel window

### Models Evaluated (ISLP Ch 1-9)
1. Dummy Classifier (baseline)
2. Logistic Regression (Ch 4)
3. Linear/Quadratic Discriminant Analysis (Ch 4)
4. Support Vector Machines - Linear & RBF (Ch 9)
5. Random Forest (Ch 8)
6. Gradient Boosting (Ch 8)

### Evaluation Strategy
- **Time-based splits**: 60% train, 20% validation, 20% test
- **Primary metric**: PR-AUC (handles class imbalance)
- **Threshold tuning**: Target 80% recall on validation set
- **Segment-level detection**: Count true positive if any window triggers during anomaly segment

---

## 🔗 Integration with AegisNav

This model plugs into the **Monitor Agent** in the AegisNav LangGraph workflow:

```
Simulation Engine → Monitor Agent → Anomaly Detector (This Model)
                         ↓
                 Anomaly Detection Agent
                         ↓
                 Replanning Agent (Trajectory Correction)
```

See `aegisnav_detector.py` for the `AnomalyDetector` class ready for integration.

---

## 📚 References

**Primary Data Source**:
- Hundman, K., et al. (2018). "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding." *KDD 2018*. [arXiv:1802.04431](https://arxiv.org/abs/1802.04431)
- Telemanom Repository: [khundman/telemanom](https://github.com/khundman/telemanom)

**Textbook**:
- James, G., et al. (2023). *An Introduction to Statistical Learning with Applications in Python* (ISLP). Springer.

**NASA Missions**:
- [SMAP Mission](https://smap.jpl.nasa.gov/)
- [MSL (Curiosity Rover)](https://mars.nasa.gov/msl/)

---

## 📝 Data Acquisition

**Option 1: Real NASA Data**

```bash
# Clone Telemanom repository
git clone https://github.com/khundman/telemanom.git
mv telemanom/data ./data/telemanom

# Update notebook: Set USE_REAL_DATA = True
```

**Option 2: Kaggle Mirror**

Search for "NASA SMAP MSL anomaly detection" on Kaggle and download labeled datasets.

**Option 3: Synthetic Data (Default)**

The notebook generates synthetic spacecraft telemetry for demonstration if real data is unavailable.

---

## 🛠️ Development

### Running Tests

```bash
# Validate model loading
python -c "from aegisnav_detector import AnomalyDetector; print('✓ Utilities OK')"
```

### Model Retraining

Edit hyperparameters in notebook cells and re-run model training section. Save new pipeline:

```python
joblib.dump(new_pipeline, 'models/anomaly_detector_pipeline_v2.joblib')
```

---

## 🙏 Acknowledgments

- NASA for SMAP/MSL telemetry datasets
- Hundman et al. for the Telemanom benchmark
- ISLP authors for foundational supervised learning methods
- CSCA 5622 course staff for guidance

---

## 📧 Contact

Roman Di Domizio - [GitHub](https://github.com/romandidomizio)

**AegisNav Project**: [GitHub Repository](https://github.com/romandidomizio/AegisNav)
