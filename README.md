# IndustrialGuard AI

An academic mini-project for industrial visual inspection that combines classical image processing, normal-only anomaly detection, a compact CNN autoencoder comparison, and a local inspection dashboard.

## First milestone

- Dataset: complete MVTec AD benchmark with all 15 object and texture categories.
- Training protocol: fit on defect-free training images; use validation data for threshold selection; reserve test images and labels for final evaluation.
- Learned detector: one PatchCore model per category, trained and evaluated on the local CUDA environment.
- Model comparison: PatchCore across all 15 categories; the compact CNN autoencoder is trained as a comparison for selected categories.
- Additional supervised model: a separate plastic-bottle detector trained on the PBD body's and shoulder's labeled defects.
- Interface and persistence: Streamlit and SQLite for local development, with downloadable PDF inspection reports.
- Image processing: compare original images with optional filtering, contrast enhancement, and edge views. Never silently replace the model input with a transformation that may erase defects.

## Project layout

```text
data/                 Dataset folders (kept out of Git)
src/industrialguard/  Reusable data, model, inference, and app modules
scripts/              Audit, training, inference, and evaluation commands
results/              Local model checkpoints (kept out of Git)
artifacts/            Local inspection outputs and SQLite history (kept out of Git)
reports/              Metrics, evaluation charts, and notes
deliverables/         Academic report, presentation, viva, and deployment guide
```

Expected MVTec AD category layout:

```text
data/raw/mvtec_ad/
  bottle/   # plus cable, capsule, carpet, grid, hazelnut, leather, metal_nut,
            # pill, screw, tile, toothbrush, transistor, wood, and zipper
    train/good/
    test/good/
    test/<defect-type>/
    ground_truth/<defect-type>/
  metal_nut/
    ...
```

## Current setup status

- Workspace scaffold, dataset audit, image-processing utilities, PatchCore training and inference entry points, and Streamlit pages created.
- Python 3.12.10 is installed and isolated in `.venv`.
- NVIDIA GeForce RTX 4060 Laptop GPU detected with 8 GB VRAM; PyTorch CUDA tensor allocation verified.
- Installed versions: PyTorch 2.12.0+cu130, TorchVision 0.27.0+cu130, Anomalib 2.7.0, Streamlit 1.66.1.
- `requirements.txt` records direct dependencies; `requirements-lock.txt` records the resolved environment.
- The complete 5,354-image MVTec AD archive is extracted; all 15 category image and mask audits pass.
- PatchCore was retrained for all 15 categories on 3,629 defect-free training images; held-out metrics are in `reports/retrained_patchcore_metrics.csv`.
- CNN autoencoders were trained for representative categories; completed per-category results are listed in `reports/model_comparison.csv`.
- The Streamlit inference page supports the 15 category-specific PatchCore checkpoints.
- AI inspection uses a supervised 15-category Wide-ResNet feature classifier before PatchCore. A nearest-neighbor feature-distance gate rejects unsupported images; an operator may choose a category for an in-distribution but uncertain image, while unsupported images are stopped before PatchCore. Category scores and PatchCore anomaly scores are displayed separately.
- The product classifier is trained from the MVTec `train/good` folders and calibrated on held-out MVTec validation images plus a separate unsupported-image set. Its label order is stored and checked against `MVTEC_CATEGORIES` when loaded. The PBD detector remains a separate plastic-bottle workflow.
- A separate detector supports seven PBD plastic-bottle defect labels; it is not a universal product or packaging model.
- AI Defect Detection accepts image uploads or camera snapshots; each new camera photo is inspected automatically.
- The Streamlit inspection page saves input and visualization images under `artifacts/inspections`, records metadata in `artifacts/industrialguard.sqlite3`, and creates a one-page PDF for each completed inspection.
- Dashboard, Inspection History, Reports, and Model Analytics pages read the local inspection and evaluation outputs.

## Development stages

1. Configure Python and a project virtual environment; verify PyTorch CUDA access.
2. Obtain the full MVTec AD archive and audit all 15 categories.
3. Build and compare image-processing views.
4. Train and evaluate separate PatchCore models, reserving the MVTec test split for evaluation.
5. Prepare the academic report, presentation, deployment notes, and viva questions under `deliverables/`.
6. Verify image upload with a local browser file picker before demonstration.

Do not commit dataset files, model weights, local databases, generated reports, or secrets.

## First commands

Run these from the project root in PowerShell:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\streamlit.exe run app.py
```

For dataset inspection and model training, run:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\inspect_dataset.py
.\.venv\Scripts\python.exe scripts\train_product_classifier.py
.\scripts\train_all_categories.ps1
```

The training script audits the full dataset, trains PatchCore for all 15 categories, refreshes the bottle and metal nut autoencoder comparison, and exports charts. New checkpoints are written to `results/retrained/`; summaries and charts are written to `reports/`.

The downloaded academic PBD set can be used to retrain the separate plastic-bottle detector:

```powershell
.\.venv\Scripts\python.exe scripts\train_pbd_detector.py --epochs 3 --batch-size 2
```

This writes its model and validation metrics under `results/retrained/pbd_bottle_detector/`. The set is for academic purposes; its validation split is not an independent production test.

Inspect a single image with its category checkpoint:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\predict_image.py --category bottle --image "data/raw/mvtec_ad/bottle/test/broken_large/000.png"
```

The command writes the anomaly heatmap and overlay under `artifacts/predictions`. The Streamlit **AI Defect Detection** page accepts an uploaded image, shows the checkpoint-backed score and visualizations, saves an inspection record, and provides a PDF report. **Inspection History**, **Reports**, and **Model Analytics** let you review inspections and measured model results.

## Expanded 15-category training results

All 15 PatchCore category models were retrained on 3,629 defect-free training images. Image AUROC ranged from 0.643 (grid) to 1.000 (bottle, hazelnut, leather); pixel AUROC ranged from 0.931 (wood) to 0.993 (leather). Pixel F1 was low for some categories, including screw (0.008) and grid (0.034), so the model still needs per-category localization threshold work.

Full per-category metrics: [`reports/retrained_patchcore_metrics.csv`](reports/retrained_patchcore_metrics.csv). Autoencoder comparisons for seven categories and charts: [`reports/model_comparison.csv`](reports/model_comparison.csv).

## Initial measured results

| Category | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 |
| --- | ---: | ---: | ---: | ---: |
| Bottle | 1.000000 | 0.992000 | 0.985560 | 0.692171 |
| Metal nut | 0.998045 | 0.994595 | 0.987006 | 0.810754 |

These are single-seed baseline results. F1 depends on the validation-calibrated threshold; the bottle pixel F1 shows localization needs additional work. See `reports/evaluation-notes.md` for run details and limitations. Checkpoints and generated images stay out of Git.

The compact CNN autoencoder comparison is weaker at its normal-validation percentile thresholds: bottle image AUROC/F1 are 0.709524/0.602151, and metal nut values are 0.268328/0.039604. This is a measured result, not a placeholder. See `reports/model_comparison.csv` and `deliverables/IndustrialGuard_AI_Project_Report.md`.

### Plastic bottle detector

The additional Faster R-CNN MobileNetV3-FPN model was trained for three epochs on all 3,170 PBD training images (body and shoulder) and measured on its 793-image validation split. Validation mAP@0.50 was 0.355 across seven labels: body stain, concave, score and squeeze; shoulder pipe, no-burr and stain. Performance varied sharply by defect: shoulder pipe/no-burr AP@0.50 were 1.000/0.967, while body stain/concave/score/squeeze were 0.042/0.065/0.080/0.202 and shoulder stain was 0.129. This is an early baseline, and the provided split is not an independent factory test. Use the separate **Plastic bottle defect detector** option under AI Defect Detection. The model does not cover metal bottles, every cold-drink bottle design, snack packet packaging, or cables. No such product-specific labeled datasets are included in this run.
