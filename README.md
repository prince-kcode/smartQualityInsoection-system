# Smart Quality Inspection System (IndustrialGuard AI)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12%2BCUDA-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-brightgreen.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Computer Vision & Deep Learning inspection system for manufacturing quality assurance. The platform features automated industrial product category detection (cables, bottles, capsules, screws, etc.), normal-only anomaly detection via PatchCore, supervised defect detection via Faster R-CNN/YOLO, and an interactive inspection workspace with instant **APPROVED TO SELL / NOT APPROVED TO SELL** quality verdicts.

---

## 🌟 Key Features

1. **3-Model Industrial AI Architecture**
   - **🧠 Model 1: Category Recognition Engine (Wide-ResNet-50 Prototypes)**: Instantly classifies any uploaded or webcam product image across 15 industrial classes with top-1 and top-5 confidence scores.
   - **🔬 Model 2: Surface Quality & Anomaly Analyzer (PatchCore)**: Extracts multi-scale neighborhood patch features, calculates calibrated anomaly scores against normal references, and generates high-resolution pixel defect heatmaps.
   - **🎯 Model 3: Defect Localizer (Ultralytics YOLOv11)**: Detects and draws precise defect bounding boxes with bounding box coordinates, defect classes, and instance confidence ratings.

2. **Actionable Commercial Quality Verdicts**
   - **✅ APPROVED TO SELL**: Product passes all quality tolerance thresholds; confirmed zero defect bounding boxes and low anomaly score. Ready for commercial release.
   - **❌ NOT APPROVED TO SELL**: Product fails inspection; highlights exact defect regions on both the anomaly heatmap and YOLO bounding box overlay.

3. **1-Click Live Demo Samples**
   - Quick testing with 9 preloaded industrial samples: Defective & Normal Cables, Bottles, Capsules, Metal Nuts, Screws, and Hazelnuts.

4. **Multi-View Visual Inspection Gallery**
   - Side-by-side view of Original Product, Model 2 Surface Anomaly Heatmap (JET colormap), and Model 3 YOLO Defect Bounding Boxes.

5. **Compliance & Reporting**
   - Automatically generates formal PDF Inspection Certificates with original, heatmap, and YOLO detection panels.
   - SQLite persistent history tracking every inspection run with timestamps, scores, and processing times.

---

## 🏗️ Project Layout

```text
├── app.py                     # Main Streamlit Dashboard & AI Inspection App
├── src/industrialguard/       # Core Package
│   ├── inference/             # Predictors, Product Classifier, Bottle Detector
│   ├── preprocessing/         # Edge detection, CLAHE, filters
│   ├── database/              # SQLite inspection history persistence
│   ├── visualization/         # Anomaly heatmap generation and overlays
│   └── reports/               # Formal PDF report generator
├── models/                    # Classifier models and category centroids
├── deliverables/              # Project Report (PDF/DOCX), Presentation (PPTX), Guides
├── reports/                   # Model performance comparisons and benchmark CSVs
├── results/retrained/         # Evaluation curves, test metrics, and detector weights
├── scripts/                   # Training pipelines and dataset audit scripts
└── requirements.txt           # Project dependencies
```

---

## 🚀 Getting Started

### 1. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/prince-kcode/smartQualityInsoection-system.git
cd smartQualityInsoection-system

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### 2. Run the Inspection Dashboard
```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\streamlit.exe run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🔬 Supported Product Categories

| Category | Description | Model AUROC |
| :--- | :--- | :--- |
| **Cable** | Electrical wiring & harness defect detection | 0.996 |
| **Bottle** | Glass/translucent container inspection | 1.000 |
| **Capsule** | Pharmaceutical capsule cracks & faults | 0.998 |
| **Screw** | Thread damage, bending & head defects | 0.985 |
| **Metal Nut** | Machined surface scratches & structural cracks | 0.998 |
| **Hazelnut** | Agricultural sorting & shell defects | 1.000 |
| **Transistor**| Electronic component pin & body damage | 0.989 |
| **Wood / Tile / Carpet** | Texture surface anomalies and stains | >0.96 |

---

## 📊 Evaluation & Deliverables
Detailed project reports, presentation slides, and viva preparation guides are stored in [`deliverables/`](deliverables/):
- **Project Report**: [`IndustrialGuard_AI_Project_Report.pdf`](deliverables/IndustrialGuard_AI_Project_Report.pdf)
- **Presentation**: [`IndustrialGuard_AI_Presentation.pptx`](deliverables/IndustrialGuard_AI_Presentation.pptx)
- **Deployment Guide**: [`Deployment_Guide.md`](deliverables/Deployment_Guide.md)
