# Smart Quality Inspection System (IndustrialGuard AI)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2BCUDA-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-brightgreen.svg)](https://streamlit.io/)
[![YOLOv11](https://img.shields.io/badge/Ultralytics-YOLOv11-amber.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end **Computer Vision & Deep Learning Quality Assurance System** for industrial manufacturing inspection. The platform automatically identifies 16 industrial product categories, rejects unsupported/non-dataset images (`NOT TRAINED FOR THIS DATASET`), detects surface scratches, cracks, chips, holes, and structural damage on any uploaded picture, localizes defects with bounding boxes, and issues instant **`APPROVED TO SELL`** or **`NOT FOR SELL`** commercial verdicts.

---

## 🌟 Key Capabilities

### 1. Synchronous 3-Engine AI + OOD Gate Architecture
- **🧠 Engine 01 — Tri-Backbone Category & OOD Classifier (`CLIP ViT-B/32` + `Wide-ResNet-50` + `EfficientNet-B0`)**:
  - Classifies product specimens across **16 industrial categories** (`bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`, `leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, `zipper`, `spoon`).
  - **Strict Non-Dataset Rejection Gate**: Automatically detects and blocks any image that does not belong to the trained industrial dataset (such as people, medical X-rays, vehicles, animals, furniture, or random scenes) with **`🚫 NOT TRAINED FOR THIS DATASET // UNSUPPORTED PICTURE`**.
- **🔬 Engine 02 — Multi-Modal Surface Defect, Scratch & Fracture Analyzer (`DefectDetectorMLP` + `PatchCore`)**:
  - Combines `1192-D` global `CLIP` image embeddings, local hotspot-crop `CLIP` embeddings, semantic intact/damaged prompt similarities, `384x384` morphological BlackHat/TopHat/Laplacian scratch & crack spectrometry, and `Wide-ResNet-50` deep features (**97.4% accuracy**).
  - Detects broken, frayed, cracked, chipped, or scratched specimens on **any unseen uploaded image** regardless of filename and generates pixel-level JET anomaly heatmaps.
- **🎯 Engine 03 — Spatial Defect Localizer (`Ultralytics YOLOv11n`)**:
  - Fine-tuned on industrial defects and synthetic surface scratches/cracks/chips to draw bounding boxes with spatial coordinates and confidence scores around damaged regions.

### 2. Commercial QA Gate Verdicts
- **✅ APPROVED TO SELL**: Intact specimen within calibrated quality thresholds; zero surface defects detected.
- **❌ NOT FOR SELL (`NOT APPROVED TO SELL`)**: Broken, cracked, frayed, chipped, or scratched specimen rejected by the QA gate with highlighted heatmap and bounding boxes.
- **🚫 NOT TRAINED FOR THIS DATASET**: Uploaded picture is outside the 16 trained industrial product categories; inspection is halted immediately.

### 3. Compliance, Telemetry & Audit Reporting
- **PDF Inspection Certificates**: 1-click generation of formal QA inspection reports containing original specimen, anomaly heatmap, and YOLO bounding-box panels.
- **Persistent SQLite Audit Log**: Tracks every inspection run with timestamps, category confidence, anomaly score, and commercial verdict.

---

## 🏗️ Repository Structure

```text
├── app.py                              # Main Streamlit Inspection Cockpit & Dashboard
├── src/industrialguard/                # Core Python Package
│   ├── inference/
│   │   ├── product_classifier.py       # Engine 01: Tri-Backbone Classifier & CLIP OOD Gate
│   │   ├── predictor.py                # Engine 02: Multi-Modal Defect/Scratch Head & PatchCore
│   │   ├── yolo_detector.py            # Engine 03: YOLOv11n Spatial Defect Localizer
│   │   └── pbd_detector.py             # Auxiliary Faster R-CNN Bottle Defect Detector
│   ├── preprocessing/                  # CLAHE, Canny edge maps, Gaussian/Median filters
│   ├── visualization/                  # JET anomaly heatmap & overlay rendering
│   ├── database/                       # SQLite inspection history persistence
│   └── reports/                        # Formal PDF certificate generator
├── models/                             # Trained Checkpoints & Feature Banks
│   ├── clip_category_bank.pt           # Precomputed CLIP ViT-B/32 category & OOD bank
│   ├── defect_detector_head.pt         # Trained 1192-D Multi-Modal Defect/Scratch MLP
│   ├── industrial_defect_yolo.pt       # Fine-tuned YOLOv11n industrial defect detector
│   ├── category_centroids.pt           # 16-class Wide-ResNet-50 prototype centroids
│   ├── product_classifier_efficientnet.pt # Fine-tuned EfficientNet-B0 weights
│   └── category_thresholds.json        # Calibrated anomaly thresholds per category
├── data/sample/                        # Curated live-demo normal, defective & unseen specimens
├── scripts/
│   ├── retrain_all_models.py           # Unified GPU retraining pipeline (CLIP + MLP + YOLOv11)
│   └── predict_image.py                # CLI single-image inspection utility
├── reports/                            # Benchmark CSVs, ROC/Confusion plots & evaluation notes
├── deliverables/                       # Project Report (PDF/DOCX), Presentation (PPTX) & Guides
└── requirements.txt                    # Python dependencies
```

---

## 🚀 Quick Start

### 1. Clone & Install Dependencies
```powershell
git clone https://github.com/prince-kcode/smartQualityInsoection-system.git
cd smartQualityInsoection-system
pip install -r requirements.txt
```

### 2. Launch the Live Inspection Dashboard
```powershell
streamlit run app.py --server.port 8501
```
Open **`http://localhost:8501`** in your browser.

### 3. Retrain All Models (Optional)
```powershell
python scripts/retrain_all_models.py
```

---

## 🔬 Supported Industrial Product Categories (16 Classes)

| Category | Inspection Scope | Category | Inspection Scope |
| :--- | :--- | :--- | :--- |
| **Bottle** | Glass & plastic container cracks/contamination | **Pill** | Pharmaceutical tablet chips & discoloration |
| **Cable** | Electrical wire & charger cord fraying/cuts | **Screw** | Thread damage, bending & head scratches |
| **Capsule** | Pharmaceutical capsule cracks & dents | **Spoon** | Stainless steel & tableware scratches/bends |
| **Carpet** | Woven textile tears, stains & weave faults | **Tile** | Ceramic structural cracks & glaze pits |
| **Grid** | Metal mesh breaks & bent intersections | **Toothbrush** | Bristle splay & handle molding defects |
| **Hazelnut** | Shell cracks, holes & surface cuts | **Transistor** | Semiconductor package & lead damage |
| **Leather** | Surface cuts, pokes & texture anomalies | **Wood** | Timber & wooden cutlery chips, knots & cracks |
| **Metal Nut** | Machined surface scratches & thread faults | **Zipper** | Broken teeth & fabric tape fraying |
